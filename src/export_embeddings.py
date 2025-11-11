# src/export_embeddings.py
import os, csv, json, argparse
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from .models.geoverse import GeoVerse  # adjust import if your tree differs


def load_state_dict_flex(path, map_location="cpu"):
    obj = torch.load(path, map_location=map_location)
    sd = None
    ckpt_args = {}
    if isinstance(obj, dict):
        # try common wrappers
        for k in ("model", "state_dict", "ema", "model_ema", "net", "weights"):
            if k in obj and isinstance(obj[k], dict):
                sd = obj[k]; break
        if sd is None and all(isinstance(v, torch.Tensor) for v in obj.values()):
            sd = obj
        # harvest args if present
        if "args" in obj and isinstance(obj["args"], (dict,)):
            ckpt_args = obj["args"]
    if sd is None:
        # fallback: treat object itself as state_dict
        sd = obj if isinstance(obj, dict) else {}
    # strip 'module.' if DDP
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    return sd, ckpt_args


class ManifestDS(Dataset):
    """Expects CSV header: filepath,label_text,lat,lon"""
    def __init__(self, csv_path: str, image_size: int, text_model: str, use_coords: bool):
        self.rows: List[Dict[str, str]] = []
        with open(csv_path, newline="") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                self.rows.append(r)
        self.use_coords = use_coords
        self.tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.tx = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225]),
        ])

    def __len__(self): return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        fp = r["filepath"]
        lbl = r.get("label_text", "") or "unknown"
        img = Image.open(fp).convert("RGB")
        img_t = self.tx(img)
        tok = self.tokenizer(lbl, padding="max_length", truncation=True,
                             max_length=24, return_tensors="pt")
        input_ids = tok["input_ids"].squeeze(0)
        attn = tok["attention_mask"].squeeze(0)
        if self.use_coords:
            lat = r.get("lat", "")
            lon = r.get("lon", "")
            try:
                coord = torch.tensor([float(lat), float(lon)], dtype=torch.float32)
            except Exception:
                coord = torch.tensor([0.0, 0.0], dtype=torch.float32)
        else:
            coord = torch.tensor([0.0, 0.0], dtype=torch.float32)

        return {
            "image": img_t, "input_ids": input_ids, "attention_mask": attn,
            "coord": coord, "filepath": fp, "label_text": lbl
        }


@torch.no_grad()
def export_embeddings(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load state dict + training args from ckpt (if present)
    sd, ckpt_args = load_state_dict_flex(args.ckpt, map_location=device)

    # Defaults from ckpt, allow CLI override
    dim = int(args.dim if args.dim is not None else ckpt_args.get("dim", 256))
    text_model = args.text_model or ckpt_args.get("text_model", "sentence-transformers/all-MiniLM-L6-v2")
    use_coords = (args.use_coords if args.use_coords is not None
                  else bool(ckpt_args.get("use_coords", False)))
    backbone = args.backbone or ckpt_args.get("backbone", "resnet18")

    # If image_size not provided, pick sensible default per backbone
    image_size = int(args.image_size or (224 if backbone.startswith("vit") else 128))

    # Dataset / loader
    ds = ManifestDS(args.manifest, image_size=image_size, text_model=text_model, use_coords=use_coords)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = GeoVerse(dim=dim, text_model=text_model, use_coords=use_coords, backbone=backbone).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[export] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

    model.eval()

    img_embs = []
    txt_embs = []
    coord_embs = [] if use_coords else None
    filepaths = []
    labels = []

    for batch in dl:
        imgs = batch["image"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attn = batch["attention_mask"].to(device, non_blocking=True)
        coords = batch["coord"].to(device, non_blocking=True)

        # Encode each modality separately; do NOT call model(...)
        zi = model.image_enc(imgs)
        zt = model.text_enc(input_ids, attn)
        img_embs.append(zi.cpu().numpy())
        txt_embs.append(zt.cpu().numpy())

        if use_coords and hasattr(model, "coord_enc") and model.coord_enc is not None:
            zg = model.coord_enc(coords)
            coord_embs.append(zg.cpu().numpy())

        filepaths.extend(batch["filepath"])
        labels.extend(batch["label_text"])

    img_embs = np.concatenate(img_embs, axis=0)
    txt_embs = np.concatenate(txt_embs, axis=0)
    if coord_embs is not None:
        coord_embs = np.concatenate(coord_embs, axis=0)

    os.makedirs(args.out_npy, exist_ok=True)
    np.save(os.path.join(args.out_npy, "img_embeddings.npy"), img_embs)
    np.save(os.path.join(args.out_npy, "txt_embeddings.npy"), txt_embs)
    if coord_embs is not None:
        np.save(os.path.join(args.out_npy, "coords.npy"), coord_embs)

    meta = {
        "filepaths": filepaths,
        "labels": labels,
        "backbone": backbone,
        "dim": dim,
        "text_model": text_model,
        "use_coords": use_coords,
        "image_size": image_size,
        "count": len(filepaths),
    }
    with open(os.path.join(args.out_npy, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Export complete: {args.out_npy}")
    print(f"   img_embeddings: {img_embs.shape} | txt_embeddings: {txt_embs.shape}"
          + (f" | coords: {coord_embs.shape}" if coord_embs is not None else ""))
    print(f"   meta.json written with {len(filepaths)} entries.")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV with filepath,label_text,lat,lon")
    ap.add_argument("--ckpt", required=True, help="Path to trained checkpoint")
    ap.add_argument("--out_npy", required=True, help="Output dir for .npy and meta.json")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=None, help="Override input size (default per backbone)")
    ap.add_argument("--dim", type=int, default=None, help="Override embedding dim (fallback to ckpt)")
    ap.add_argument("--text_model", type=str, default=None, help="Override text model (fallback to ckpt)")
    ap.add_argument("--use_coords", type=lambda x: str(x).lower()=='true', default=None,
                    help="Override coord usage (fallback to ckpt)")
    ap.add_argument("--backbone", type=str, default=None, choices=["resnet18","vit_tiny"],
                    help="Override backbone (fallback to ckpt)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_embeddings(args)