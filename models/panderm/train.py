import json
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from transformers import BertModel
from transformers import BertTokenizer
# from datasets import load_metric
import evaluate

from builder import get_encoder
from multimodal import MultimodalModel
from data.derm_data import DermDatasetQnA

# def train_model(model, train_loader, val_loader, tokenizer, device, num_epochs=5, lr=1e-4):
#     """
#     Train and evaluate the multimodal model.
#     Args:
#         model: The multimodal model.
#         train_loader: DataLoader for training.
#         val_loader: DataLoader for validation.
#         tokenizer: Tokenizer for processing text.
#         device: CUDA or CPU device.
#         num_epochs: Number of epochs for training.
#         lr: Learning rate for the optimizer.

#     Returns:
#         Trained model and loss history.
#     """
#     # Define optimizer and loss function
#     optimizer = optim.AdamW(model.parameters(), lr=lr)
#     criterion = nn.CrossEntropyLoss()

#     # Move model to device
#     model = model.to(device)

#     # Training and validation history
#     train_loss_history = []
#     val_loss_history = []

#     for epoch in range(num_epochs):
#         print(f"Epoch {epoch + 1}/{num_epochs}")

#         # Training phase
#         model.train()
#         train_loss = 0.0
#         for batch in tqdm(train_loader, desc="Training"):
#             images, questions, answers = batch
#             images = images.to(device)
#             answers = answers.to(device)

#             # Tokenize questions
#             questions = tokenizer(questions, return_tensors="pt", padding=True, truncation=True).to(device)

#             # Forward pass
#             outputs = model(images, questions)
            
#             # Compute loss
#             loss = criterion(outputs, answers)
#             train_loss += loss.item()

#             # Backpropagation and optimization
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         # Calculate average training loss
#         train_loss /= len(train_loader)
#         train_loss_history.append(train_loss)

#         # Validation phase
#         model.eval()
#         val_loss = 0.0
#         with torch.no_grad():
#             for batch in tqdm(val_loader, desc="Validation"):
#                 images, questions, answers = batch
#                 images = images.to(device)
#                 answers = answers.to(device)

#                 # Tokenize questions
#                 questions = tokenizer(questions, return_tensors="pt", padding=True, truncation=True).to(device)

#                 # Forward pass
#                 outputs = model(images, questions)
                
#                 # Compute loss
#                 loss = criterion(outputs, answers)
#                 val_loss += loss.item()

#         # Calculate average validation loss
#         val_loss /= len(val_loader)
#         val_loss_history.append(val_loss)

#         print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

#     return model, {"train_loss": train_loss_history, "val_loss": val_loss_history}


## Prepping models
vision_model, eval_transform = get_encoder(model_name="PanDerm")
language_model = BertModel.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

multimodal_model = MultimodalModel(vision_model=vision_model, language_model=language_model)

## Prepping data loaders
# code for dl's here
ham_clean_df = pd.read_csv("../../HAM_clean.csv")

with open('data/questions_diag_mapping.json') as f:
    questions_diag_mapping = json.load(f)
    questions_diag_mapping.pop("SCC")
    questions_diag_mapping.pop("UNK")
    print(questions_diag_mapping.keys())

root_path = "/mount/studenten-temp1/users/yassir/datasets/HAM10000_clean/ISIC2018/"

percent_data = 1.0
dataset_train = DermDatasetQnA(df_im=ham_clean_df,
                            qna_mapping=questions_diag_mapping,
                            root=root_path,
                            train=True,
                            transforms=eval_transform,
                            data_percent=percent_data)
dataset_val = DermDatasetQnA(df_im=ham_clean_df,
                            qna_mapping=questions_diag_mapping,
                            root=root_path,
                            val=True,
                            transforms=eval_transform,)
dataset_test = DermDatasetQnA(df_im=ham_clean_df,
                            qna_mapping=questions_diag_mapping,
                            root=root_path,
                            test=True,
                            transforms=eval_transform,)
print('train size:', len(dataset_train), ',val size:', len(dataset_val), ',test size:', len(dataset_test))

batch_size = 1000
num_workers = 4

train_dataloader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)

val_dataloader = torch.utils.data.DataLoader(
    dataset_val,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)
test_dataloader = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True
)


## Metrics
# bleu_metric = load_metric("bleu")
# rouge_metric = load_metric("rouge")
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from datasets import load_metric  # For BLEU and ROUGE
from tqdm import tqdm


# Training loop
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        image, question, target = batch
        image, target = image.to(device), target.to(device)

        # Tokenize the question and target
        question_tokens = tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(device)
        target_tokens = tokenizer(target, return_tensors="pt", padding=True, truncation=True).to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(image, question_tokens)
        
        # Compute loss (use cross-entropy for generation tasks)
        loss = criterion(outputs, target_tokens.input_ids[:, :-1].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    return total_loss / len(dataloader)


# Evaluation loop
def evaluate(model, dataloader, tokenizer, device):
    model.eval()
    predictions, references = [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            image, question, target, filename = batch
            image = image.to(device)

            # Tokenize the question
            question_tokens = tokenizer(question, return_tensors="pt", padding=True, truncation=True).to(device)

            # Forward pass
            outputs = model(image, question_tokens)
            
            # Decode predictions and collect references
            predicted_texts = tokenizer.batch_decode(torch.argmax(outputs, dim=-1), skip_special_tokens=True)
            predictions.extend(predicted_texts)
            references.extend([[ref] for ref in target])  # Wrap references in a list for BLEU compatibility

    # Compute BLEU and ROUGE scores
    bleu_score = bleu_metric.compute(predictions=predictions, references=references)
    rouge_score = rouge_metric.compute(predictions=predictions, references=references)
    
    return bleu_score, rouge_score


# Main training and evaluation script
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize your multimodal model
    model = MultimodalModel(vision_model, language_model).to(device)

    # Prepare optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Training and evaluation
    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train(model, train_dataloader, optimizer, criterion, device)
        print(f"Training Loss: {train_loss:.4f}")
        
        bleu_score, rouge_score = evaluate(model, val_dataloader, tokenizer, device)
        print(f"BLEU Score: {bleu_score['bleu']:.4f}")
        print(f"ROUGE Score: {rouge_score}")

if __name__ == "__main__":
    main()



