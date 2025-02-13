import torch
import torch.nn as nn
from transformers import ViltModel, ViltConfig
import torch
import torch.nn as nn
from transformers import ViltModel, ViltConfig

class ViLTFusion(nn.Module):
    def __init__(self, model_name="dandelin/vilt-b32-mlm", num_image_tokens=196):
        super().__init__()

        # Load ViLT model configuration and override default behavior
        config = ViltConfig.from_pretrained(model_name)
        config.image_size = 14  # **Fix: Use a single integer, NOT a tuple (14,14)**
        config.patch_size = 1  # Ensure ViLT does not expect different patching behavior

        # Load model with `ignore_mismatched_sizes=True` to avoid shape mismatch issues
        self.vilt_model = ViltModel.from_pretrained(model_name, config=config, ignore_mismatched_sizes=True)
        self.num_image_tokens = num_image_tokens

        # Ensure image embeddings match ViLT's expected hidden size
        self.image_projection = nn.Linear(1024, config.hidden_size)

    def forward(self, img_features, text_inputs, text_embeddings):
        """
        img_features: Pre-extracted image embeddings from PanDerm (batch_size, num_image_tokens, 1024)
        text_inputs: Tokenized text inputs (input_ids, attention_mask, etc.)
        text_embeddings: Precomputed text embeddings (batch_size, text_seq_len, 768)
        """

        # Project image embeddings to match ViLT's hidden size
        img_features = self.image_projection(img_features)  # Shape: (batch, 196, 768)

        # Ensure image token count is as expected
        if img_features.shape[1] != self.num_image_tokens:
            raise ValueError(f"Expected {self.num_image_tokens} image tokens, but got {img_features.shape[1]}")

        # Explicitly pass `image_embeds` and `inputs_embeds` while setting `pixel_values=None`
        outputs = self.vilt_model(
            attention_mask=text_inputs["attention_mask"],
            pixel_values=None,  # Ensure ViLT does NOT process raw images
            image_embeds=img_features,  # Correct image embeddings
            inputs_embeds=text_embeddings,  # Correct text embeddings
            return_dict=True
        )

        return outputs.last_hidden_state


class MultimodalModel(nn.Module):
    def __init__(self, vision_model, language_model, gpt_decoder):
        super(MultimodalModel, self).__init__()
        self.vision_model = vision_model  # PanDerm for vision
        self.language_model = language_model  # DistilBERT for text
        self.gpt_decoder = gpt_decoder
        self.vilt_fusion = ViLTFusion()

    def forward(self, image, question, target=None, tokenizer=None, max_length=30):
        device = next(self.parameters()).device
        image = image.to(device)
        question = {k: v.to(device) for k, v in question.items()}

        # Step 1: Extract image features from PanDerm
        image_features = self.vision_model.forward_features(image)  # Shape: (batch, 196, 1024)
        
        # Step 2: Extract text embeddings
        text_embeddings = self.language_model(**question).last_hidden_state  # Shape: (batch, text_seq_len, 768)
        
        # Step 3: Use ViLT for fusion (after correcting image embeddings)
        fused_features = self.vilt_fusion(
            img_features=image_features,
            text_inputs=question,
            text_embeddings=text_embeddings
        )

        # Step 4: Create attention mask
        encoder_attention_mask = torch.ones(fused_features.size()[:-1], dtype=torch.long, device=device)

        if target is None:
            if tokenizer is None:
                raise ValueError("Tokenizer must be provided during evaluation.")

            start_token = tokenizer.bos_token_id or tokenizer.cls_token_id or tokenizer.pad_token_id
            if start_token is None:
                raise ValueError("Tokenizer does not have a valid start token.")

            outputs = self.gpt_decoder.generate(
                input_ids=torch.full((fused_features.size(0), 1), start_token, dtype=torch.long).to(device),
                max_length=max_length,
                encoder_hidden_states=fused_features,
                encoder_attention_mask=encoder_attention_mask,
                pad_token_id=tokenizer.pad_token_id
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
