import torch
import torch.nn as nn
from transformers import AutoModel
from torchvision.models import resnet18

class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name, local_files_only=False)
        hidden = self.backbone.config.hidden_size
        self.out_dim = hidden

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # use CLS token (first hidden state) or pooled output if available
        if hasattr(out, "last_hidden_state"):
            cls = out.last_hidden_state[:,0]  # (B, H)
        else:
            cls = out.pooler_output
        return cls

class ImageEncoder(nn.Module):
    def __init__(self, model_name: str = "resnet18", pretrained: bool = True):
        super().__init__()
        # Use a small ResNet18 for broad compatibility
        self.backbone = resnet18(weights=None if not pretrained else None)  # avoid internet downloads here
        # replace final layer and expose features
        in_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.out_dim = in_dim

    def forward(self, images):
        feats = self.backbone(images)
        return feats
