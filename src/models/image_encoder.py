import torch
import torch.nn as nn
import torchvision.models as tv
import timm

class ImageEncoder(nn.Module):
    def __init__(self, backbone="resnet18", out_dim=256):
        super().__init__()
        if backbone == "resnet18":
            base = tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1)
            base.fc = nn.Identity()
            self.feature_dim = 512
            self.encoder = base
        elif backbone == "vit_tiny":
            base = timm.create_model("vit_tiny_patch16_224", pretrained=True, num_classes=0)
            self.feature_dim = base.num_features  # usually 192
            self.encoder = base
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.proj = nn.Linear(self.feature_dim, out_dim)

    def forward(self, x):
        z = self.encoder(x)
        z = self.proj(z)
        return nn.functional.normalize(z, dim=-1)