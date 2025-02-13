from .multimodal_ViLT import MultimodalModel as ViLTModel
from .multimodal_CrossAtt import MultimodalModel as CrossAttentionModel
from .multimodal_UNITER import MultimodalModel as UNITERModel

FUSION_MODELS = {
    "UNITER": UNITERModel,
    "CrossAttention": CrossAttentionModel,
    "ViLT": ViLTModel
}
