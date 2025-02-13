from transformers import AutoModel
import torch
import torch.nn as nn
from transformers import BertModel
import torch
import torch.nn as nn
from transformers import AutoModel


class UNITERFusion(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        #Since we couldn't have access to the UNITER weights/ checkpoints, we used BERT and constructed the same architecture
        #following the original
        self.uniter_model = BertModel.from_pretrained("bert-base-uncased")
        self.hidden_size = hidden_size

    def forward(self, img_features, text_features, attention_mask=None):
        """
        img_features: (batch_size, num_patches, hidden_size)
        text_features: (batch_size, seq_len, hidden_size)
        attention_mask: Combined attention mask for both modalities
        """
        batch_size = img_features.size(0)
        img_seq_len = img_features.size(1)
        text_seq_len = text_features.size(1)

        # Generate attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.cat(
                [
                    torch.ones((batch_size, img_seq_len), device=img_features.device),
                    torch.ones((batch_size, text_seq_len), device=text_features.device),
                ],
                dim=1
            )

        # Concatenate image and text features along sequence dimension
        combined_features = torch.cat([img_features, text_features], dim=1)

        # Pass to UNITER model
        outputs = self.uniter_model(
            inputs_embeds=combined_features,
            attention_mask=attention_mask,
            return_dict=True
        )

        return outputs.last_hidden_state  # Extract fused embeddings

class MultimodalModel(nn.Module):
    def __init__(self, vision_model, language_model, gpt_decoder):
        super(MultimodalModel, self).__init__()
        self.vision_model = vision_model
        self.language_model = language_model
        self.gpt_decoder = gpt_decoder
        for param in self.vision_model.parameters():
            param.requires_grad = False

        # Project text features to match the dimension of image features
        self.text_projection = nn.Linear(768, 1024)
        # Project fused features back to the dimension expected by the GPT decoder
        self.fusion_projection = nn.Linear(1024, 768)
        # Use UNITERFusion instead of CrossModalFusion
        self.uniter_fusion = UNITERFusion(dim=1024, num_heads=8, num_layers=6)

    def forward(self, image, question, target=None, tokenizer=None, max_length=30):
        device = next(self.parameters()).device
        image = image.to(device)
        question = {k: v.to(device) for k, v in question.items()}

        # Extract image features
        image_features = self.vision_model.forward_features(image)
        # Extract text features
        question_embedding = self.language_model(**question).last_hidden_state[:, 0, :]
        question_embedding = self.text_projection(question_embedding)

        # Reshape image features to match the sequence length expected by UNITERFusion
        batch_size, seq_len, img_dim = image_features.shape
        image_features = image_features.view(batch_size, seq_len, img_dim)
        question_embedding = question_embedding.unsqueeze(1)  # Add sequence dimension

        # Fuse features using UNITERFusion
        fused_features = self.uniter_fusion(image_features, question_embedding)
        fused_features = self.fusion_projection(fused_features)

        # Create encoder attention mask
        encoder_attention_mask = torch.ones(fused_features.size()[:-1], dtype=torch.long, device=device)

        if target is None:
            if tokenizer is None:
                raise ValueError("Tokenizer must be provided during evaluation.")

            start_token = tokenizer.bos_token_id or tokenizer.cls_token_id or tokenizer.pad_token_id
            if start_token is None:
                raise ValueError("Tokenizer does not have a valid start token.")

            # Generate output using GPT decoder
            outputs = self.gpt_decoder.generate(
                input_ids=torch.full((fused_features.size(0), 1), start_token, dtype=torch.long).to(device),
                max_length=max_length,
                encoder_hidden_states=fused_features,
                encoder_attention_mask=encoder_attention_mask,
                pad_token_id=tokenizer.pad_token_id  # Explicitly pass pad_token_id
            )
        else:
            decoder_input_ids = target[:, :-1].to(device)
            outputs = self.gpt_decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=fused_features,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=True
            ).logits

        return outputs