import torch
import torch.nn as nn
from .image_encoder import ImageEncoder
from .text_encoder import TextEncoder  # or wherever it’s defined
from .coord_encoder import CoordEncoder  # if applicable

class GeoVerse(nn.Module):
    def __init__(self, dim=256, text_model="sentence-transformers/all-MiniLM-L6-v2",
                 use_coords=False, minigeo_ckpt=None, backbone="resnet18"):
        super().__init__()
        self.dim = dim
        self.text_model_name = text_model
        self.use_coords = use_coords
        self.minigeo_ckpt = minigeo_ckpt
        self.backbone = backbone

        # === Vision Encoder ===
        self.image_enc = ImageEncoder(backbone=backbone, out_dim=dim)

        # === Text Encoder ===
        self.text_enc = TextEncoder(model_name=text_model, out_dim=dim)

        # === Coordinate Encoder (optional) ===
        if use_coords:
            self.coord_enc = CoordEncoder(out_dim=dim)
        else:
            self.coord_enc = None

    def forward(self, image, input_ids, attention_mask, coord=None):
        z_img = self.image_enc(image)
        z_txt = self.text_enc(input_ids, attention_mask)
        if self.coord_enc is not None and coord is not None:
            z_geo = self.coord_enc(coord)
            return z_img, z_txt, z_geo
        return z_img, z_txt