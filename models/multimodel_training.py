import os
import json
import sys
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from transformers import BertModel, BertTokenizer,DistilBertModel, DistilBertTokenizer, GPT2LMHeadModel, GPT2Config

import torch
from panderm.builder import get_encoder
from multimodels.multimodel_CrossAtt import MultimodalModel as cross_att_model
from multimodels.multimodel_UNITER import MultimodalModel as uniter_model
from multimodels.multimodel_ViLT import MultimodalModel as vilt_model

from data.derm_data import DermDatasetQnA

from torch.utils.data import Subset, DataLoader
import evaluate

# Set the GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1 only

# Metrics
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")

# Read parameters for language model and fusion mechanism
if len(sys.argv) != 3:
    print("Usage: python multimodal_training.py <LanguageModel> <FusionModel>")
    print(f"Langauge Models: 'BERT' or 'DistilBERT'")
    print(f"Fusion Models: 'UNITER', 'CrossAttention', 'ViLT'")
    sys.exit(1)

param_languagemodel = sys.argv[1]  # 'BERT' or 'DistilBERT'
param_fusionmodel = sys.argv[2]  # 'UNITER', 'CrossAttention', 'ViLT'

def evaluate_model(model, dataloader, tokenizer, device, max_length=128):
    model.eval()
    predictions, references = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            image, question, target, filename = batch
            image = image.to(device)

            question_tokens = tokenizer(
                question,
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(device)

            input_ids = question_tokens["input_ids"].to(device)
            attention_mask = question_tokens["attention_mask"].to(device)
            
            outputs = model(
                image, 
                {"input_ids": input_ids, "attention_mask": attention_mask}, 
                target=None, 
                tokenizer=tokenizer, 
                max_length=max_length
            )

            predicted_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            predictions.extend(predicted_texts)
            references.extend([[ref] for ref in target])

        # To monitor the preditions of the model:
        output_path = f"./results/outputDetails_{param_fusionmodel}.txt"
        with open(output_path, 'a') as file:
            file.write(f"Predictions: {predictions[0]}\n")
            file.write(f"References: {references[0]}\n")
            file.write(f"Questions: {question[0]}\n")
            file.write(f"Filename: {filename[0]}\n")
            file.write("-------------------------------\n")
    
    bleu_score = bleu_metric.compute(predictions=predictions, references=references)
    rouge_score = rouge_metric.compute(predictions=predictions, references=references)
    return bleu_score, rouge_score


def train(model, dataloader, tokenizer, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc="Training"):
        image, question, target, filename = batch
        image = image.to(device)

        seq_length = 128  # Maximum sequence length
        # Tokenize questions and targets
        question_tokens = tokenizer(
            question,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=seq_length
        ).to(device)

        target_tokens = tokenizer(
            target,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=seq_length
        ).to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(image, question_tokens, target_tokens.input_ids)

        # Compute loss
        loss = criterion(
            outputs.view(-1, outputs.size(-1)),
            target_tokens.input_ids[:, 1:].contiguous().view(-1)
        )
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(dataloader)


def main():

    # Reading the vision model PanDerm encode:
    vision_model, eval_transform = get_encoder(model_name="PanDerm")

    # The language model based on parameter
    if param_languagemodel == "BERT":
        language_model = BertModel.from_pretrained("bert-base-uncased")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    elif param_languagemodel == "DistilBERT":
        language_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    else:
        print(f"Invalid language model: {param_languagemodel}")
        sys.exit(1)

    gpt_config = GPT2Config.from_pretrained("gpt2")
    gpt_config.add_cross_attention = True
    gpt_decoder = GPT2LMHeadModel.from_pretrained("gpt2", config=gpt_config)

    # Fusion model based on parameter
    if param_fusionmodel == "UNITER":
        model = uniter_model(vision_model, language_model, gpt_decoder)
    elif param_fusionmodel == "CrossAttention":
        model = cross_att_model(vision_model, language_model, gpt_decoder)  
    elif param_fusionmodel == "ViLT":
        model = vilt_model(vision_model, language_model, gpt_decoder)
    else:
        print(f"Invalid fusion model: {param_fusionmodel}")
        sys.exit(1)

    ham_clean_df = pd.read_csv("/path/to/images/groundtruth/data/HAM_clean.csv")
    with open('data/questions_diag_mapping.json') as f:
        questions_diag_mapping = json.load(f)
        questions_diag_mapping.pop("SCC")
        questions_diag_mapping.pop("UNK")

    root_path = "/path/to/images/folder/"

    dataset_train = DermDatasetQnA(df_im=ham_clean_df, qna_mapping=questions_diag_mapping, root=root_path, train=True, transforms=eval_transform)
    dataset_val = DermDatasetQnA(df_im=ham_clean_df, qna_mapping=questions_diag_mapping, root=root_path, val=True, transforms=eval_transform)

    batch_size = 32
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

    # Use the specified device (GPU 0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    optimizer = optim.AdamW(
        [
            {"params": model.language_model.parameters(), "lr": 1e-4},
            {"params": model.gpt_decoder.parameters(), "lr": 5e-5},
        ]
    )
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    training_losses = {}
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train(model, train_dataloader, tokenizer, optimizer, criterion, device)
        training_losses[epoch] = train_loss
        print(f"Training Loss: {train_loss:.4f}")
        
        bleu_score, rouge_score = evaluate_model(model, val_dataloader, tokenizer, device)
        print(f"BLEU Score: {bleu_score['bleu']:.4f}")
        print(f"ROUGE Score: {rouge_score}")

    try:
        model_save_path = f"./models/model_{param_fusionmodel}_{param_languagemodel}.pth"
        torch.save(model, model_save_path)
    except Exception as error:
        print(f"Error saving the model: {error}")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
