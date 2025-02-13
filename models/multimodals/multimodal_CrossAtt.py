import torch
import torch.nn as nn


class CrossModalFusion(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)

    def forward(self, img_features, text_features):
        text_features = text_features.unsqueeze(1).expand(-1, img_features.size(1), -1)
        img_features = img_features.transpose(0, 1)
        text_features = text_features.transpose(0, 1)
        attn_output, _ = self.cross_attention(text_features, img_features, img_features)
        return attn_output.transpose(0, 1)
    

class MultimodalModel(nn.Module):
    def __init__(self, vision_model, language_model, gpt_decoder):
        super(MultimodalModel, self).__init__()
        self.vision_model = vision_model
        self.language_model = language_model
        self.gpt_decoder = gpt_decoder

        # Define projections and fusion layers
        self.text_projection = nn.Linear(768, 1024)
        self.fusion_projection = nn.Linear(1024, 768)
        self.cross_modal_fusion = CrossModalFusion(dim=1024, num_heads=8)
    
    def forward(self, image, question, target=None, tokenizer=None, max_length=30):
        # Ensure inputs are on the same device as the model
        device = next(self.parameters()).device
        image = image.to(device)
        question = {k: v.to(device) for k, v in question.items()}

        # Extract vision features
        image_features = self.vision_model.forward_features(image)

        # Extract language features (CLS token from BERT)
        question_embedding = self.language_model(**question).last_hidden_state[:, 0, :]
        question_embedding = self.text_projection(question_embedding)

        # Apply cross-modal fusion
        fused_features = self.cross_modal_fusion(image_features, question_embedding)

        # Project fused features to match GPT-2 input size
        fused_features = self.fusion_projection(fused_features)

        # Generate attention mask for fused features
        encoder_attention_mask = torch.ones(fused_features.size()[:-1], dtype=torch.long, device=device)

        if target is None:  # Evaluation Mode
            if tokenizer is None:
                raise ValueError("Tokenizer must be provided during evaluation.")

            # Ensure the tokenizer has a pad_token_id
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                if tokenizer.pad_token_id is None:
                    raise ValueError("The tokenizer does not have a valid pad_token_id or eos_token_id.")

            # Ensure a valid start token
            start_token = tokenizer.bos_token_id or tokenizer.cls_token_id or tokenizer.pad_token_id
            if start_token is None:
                raise ValueError("The tokenizer does not have a valid bos_token_id, cls_token_id, or pad_token_id.")

            # Create input_ids and attention_mask for generation
            input_ids = torch.full((fused_features.size(0), 1), start_token, dtype=torch.long).to(device)
            attention_mask = torch.ones_like(input_ids)  # All tokens are actual input (no padding)

            # Evaluation: Use the `generate` method for autoregressive text generation
            outputs = self.gpt_decoder.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,  # Pass attention mask for decoder input
                max_length=max_length,
                encoder_hidden_states=fused_features,
                encoder_attention_mask=encoder_attention_mask,  # Pass attention mask for encoder input
                pad_token_id=tokenizer.pad_token_id  # Explicitly pass pad_token_id
            )
        else:  # Training Mode
            if tokenizer is None:
                raise ValueError("Tokenizer must be provided during training.")

            # Ensure the tokenizer has a pad_token_id
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                if tokenizer.pad_token_id is None:
                    raise ValueError("The tokenizer does not have a valid pad_token_id or eos_token_id.")

            # Prepare decoder input IDs for training
            decoder_input_ids = target[:, :-1].to(device)  # Exclude the last token for teacher forcing
            decoder_attention_mask = (decoder_input_ids != tokenizer.pad_token_id).long()  # Create attention mask for decoder input

            outputs = self.gpt_decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,  # Pass attention mask for decoder input
                encoder_hidden_states=fused_features,
                encoder_attention_mask=encoder_attention_mask,  # Pass attention mask for encoder input
                return_dict=True
            ).logits

        return outputs