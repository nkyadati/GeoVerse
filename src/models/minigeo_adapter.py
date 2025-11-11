import torch, torch.nn as nn
import timm

class MiniGeoBackbone(nn.Module):
    """Adapter around a tiny ViT (or your custom GeoTransformer) with a projection head.
    If ckpt is provided (state_dict), loads it. Otherwise initializes randomly.
    """
    def __init__(self, out_dim: int = 256, ckpt: str | None = None, backbone: str = "vit_tiny_patch16_224"):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
        feat_dim = self.backbone.num_features
        self.proj = nn.Linear(feat_dim, out_dim)
        if ckpt is not None:
            sd = torch.load(ckpt, map_location="cpu")
            # support both raw state_dict and {'model': state_dict}
            if isinstance(sd, dict) and "model" in sd: sd = sd["model"]
            missing, unexpected = self.load_state_dict(sd, strict=False)
            print(f"[MiniGeoBackbone] loaded ckpt with missing={len(missing)} unexpected={len(unexpected)}")
    def forward(self, x):
        feats = self.backbone(x)
        return self.proj(feats)
