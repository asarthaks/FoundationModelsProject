from .multimodel_ViLT import MultimodalModel as ViLTModel
from .multimodel_CrossAtt import MultimodalModel as CrossAttentionModel
from .multimodel_UNITER import MultimodalModel as UNITERModel

FUSION_MODELS = {
    "UNITER": UNITERModel,
    "CrossAttention": CrossAttentionModel,
    "ViLT": ViLTModel
}
