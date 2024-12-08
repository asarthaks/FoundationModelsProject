import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from transformers import BertModel, BertTokenizer, BertConfig, GPT2LMHeadModel


from builder import get_encoder


# Consider using advanced cross-modal transformers such as UNITER, VilBERT, or BLIP for enhanced fusion.
class CrossModalFusion(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)

    def forward(self, img_features, text_features):
        # Cross-attention: text queries over image keys/values
        return self.cross_attention(text_features, img_features, img_features)
    

# class MultimodalModel(nn.Module):
#     def __init__(self, vision_model, language_model, freeze_vision=True, fine_tune_layers=0):
#         super().__init__()
#         self.vision_model = vision_model
#         self.language_model = language_model

#         # Freeze PanDerm layers
#         if freeze_vision:
#             for param in self.vision_model.parameters():
#                 param.requires_grad = False  # Freeze all layers initially

#             # Optionally fine-tune the last few transformer blocks
#             for param in self.vision_model.blocks[-fine_tune_layers:].parameters():
#                 param.requires_grad = True

#         self.cross_modal_fusion = CrossModalFusion(dim=512, num_heads=8)  # Example dimensions
#         self.fc = nn.Linear(512, 256)  # Example output layer size

#     def forward(self, image, question):
#         # Extract vision features
#         image_features = self.vision_model.forward_features(image)
        
#         # Extract language features
#         question_embedding = self.language_model(**question).last_hidden_state[:, 0, :]  # CLS token

#         # Apply cross-modal fusion
#         fused_features, _ = self.cross_modal_fusion(image_features, question_embedding.unsqueeze(1))

#         # Flatten and classify
#         return self.fc(fused_features.squeeze(1))


class MultimodalModel(nn.Module):
    def __init__(self, vision_model, language_model, gpt_decoder, freeze_vision=True, fine_tune_layers=0):
        super().__init__()
        self.vision_model = vision_model
        self.language_model = language_model
        self.gpt_decoder = gpt_decoder

        # Freeze PanDerm layers
        if freeze_vision:
            for param in self.vision_model.parameters():
                param.requires_grad = False  # Freeze all layers initially

            # Optionally fine-tune the last few transformer blocks
            for param in self.vision_model.blocks[-fine_tune_layers:].parameters():
                param.requires_grad = True

        # Cross-modal fusion mechanism
        self.cross_modal_fusion = CrossModalFusion(dim=512, num_heads=8)  # Example dimensions

    def forward(self, image, question, max_length=30):
        # Extract vision features
        image_features = self.vision_model.forward_features(image)
        
        # Extract language features (CLS token from BERT)
        question_embedding = self.language_model(**question).last_hidden_state[:, 0, :]  # CLS token

        # Apply cross-modal fusion
        fused_features, _ = self.cross_modal_fusion(image_features, question_embedding.unsqueeze(1))

        # Use the decoder to generate text
        outputs = self.gpt_decoder.generate(
            input_ids=torch.zeros((fused_features.size(0), 1), dtype=torch.long).to(fused_features.device),
            max_length=max_length,
            encoder_hidden_states=fused_features,
            encoder_attention_mask=None
        )
        return outputs


    


# vision_model, eval_transform = get_encoder(model_name="PanDerm")
# language_model = BertModel

# question = "What is this disease?"

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# question_tokens = tokenizer(question, return_tensors="pt", padding=True, truncation=True)


# ## Training: Loss Function and Training Objectives
# from transformers import BertTokenizer

# tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# target_tokens = tokenizer(target_answers, return_tensors="pt")
# loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), target_tokens.view(-1))

# Evaluation: Add metrics like BLEU, F1-score, or ROUGE for Q&A tasks.



