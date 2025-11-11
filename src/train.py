# src/train.py
import argparse, os, csv, random, math, time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from transformers import AutoTokenizer

from .models.geoverse import GeoVerse  # adapt import if your path differs


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_sim(a: torch.Tensor, b: torch.Tensor, temp: float = 0.07):
    # a: [N, D], b: [N, D] => return [N, N]
    a = nn.functional.normalize(a, dim=-1)
    b = nn.functional.normalize(b, dim=-1)
    return (a @ b.t()) / temp


def clip_contrastive_loss(a: torch.Tensor, b: torch.Tensor, temp: float = 0.07):
    # symmetric cross-entropy over cosine sims
    logits_ab = cosine_sim(a, b, temp=temp)
    logits_ba = logits_ab.t()
    labels = torch.arange(a.size(0), device=a.device)
    loss_ab = nn.functional.cross_entropy(logits_ab, labels)
    loss_ba = nn.functional.cross_entropy(logits_ba, labels)
    return 0.5 * (loss_ab + loss_ba)


def save_checkpoint(path: str, model: nn.Module, epoch: int, args, best_val: float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val,
            "args": vars(args),
        },
        path,
    )


# ----------------------------
# Dataset
# ----------------------------
@dataclass
class Sample:
    filepath: str
    label_text: str
    lat: Optional[float]
    lon: Optional[float]


class ManifestDataset(Dataset):
    """
    Expects a CSV with header: filepath,label_text,lat,lon
    Images are loaded from 'filepath'.
    """
    def __init__(
        self,
        csv_path: str,
        image_transform: transforms.Compose,
        tokenizer: AutoTokenizer,
        max_txt_len: int = 24,
        use_coords: bool = False,
    ):
        super().__init__()
        self.items: List[Sample] = []
        self.transform = image_transform
        self.tokenizer = tokenizer
        self.max_txt_len = max_txt_len
        self.use_coords = use_coords

        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fp = row["filepath"]
                lbl = row.get("label_text", "") or ""
                lat = row.get("lat", "")
                lon = row.get("lon", "")
                lat = float(lat) if (lat is not None and str(lat).strip() != "") else None
                lon = float(lon) if (lon is not None and str(lon).strip() != "") else None
                self.items.append(Sample(fp, lbl, lat, lon))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        s = self.items[idx]
        # Image
        img = Image.open(s.filepath).convert("RGB")
        img_t = self.transform(img)

        # Text tokens
        enc = self.tokenizer(
            s.label_text if s.label_text else "unknown",
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)  # [L]
        attn_mask = enc["attention_mask"].squeeze(0)  # [L]

        # Coords
        if self.use_coords:
            lat = s.lat if s.lat is not None else 0.0
            lon = s.lon if s.lon is not None else 0.0
            coord = torch.tensor([lat, lon], dtype=torch.float32)
        else:
            coord = torch.tensor([0.0, 0.0], dtype=torch.float32)

        return {
            "image": img_t,
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "coord": coord,
        }


# ----------------------------
# Transforms
# ----------------------------
def make_transforms(image_size: int, aug: str):
    if aug == "none":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    elif aug == "light":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:  # 'strong'
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


# ----------------------------
# Train / Val Step
# ----------------------------
def forward_step(model: GeoVerse, batch, amp: bool, temp: float):
    with torch.cuda.amp.autocast(enabled=amp):
        z_img = model.image_enc(batch["image"])                    # [B, D]
        z_txt = model.text_enc(batch["input_ids"], batch["attention_mask"])  # [B, D]

        loss = clip_contrastive_loss(z_img, z_txt, temp=temp)

        if hasattr(model, "coord_enc") and model.coord_enc is not None:
            z_geo = model.coord_enc(batch["coord"])                # [B, D]
            loss += 0.5 * clip_contrastive_loss(z_img, z_geo, temp=temp)
            loss += 0.5 * clip_contrastive_loss(z_geo, z_txt, temp=temp)

    return loss


def evaluate(model: GeoVerse, loader: DataLoader, device: torch.device, amp: bool, temp: float):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            for k in ("image", "input_ids", "attention_mask", "coord"):
                batch[k] = batch[k].to(device, non_blocking=True)
            loss = forward_step(model, batch, amp=amp, temp=temp)
            total += loss.item()
            count += 1
    return total / max(count, 1)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True, help='CSV: filepath,label_text,lat,lon')
    ap.add_argument('--out_dir', required=True)

    ap.add_argument('--image_size', type=int, default=128)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--dim', type=int, default=256)
    ap.add_argument('--text_model', type=str, default='sentence-transformers/all-MiniLM-L6-v2')
    ap.add_argument('--use_coords', type=lambda x: str(x).lower() == 'true', default=False)
    ap.add_argument('--temp', type=float, default=0.07)
    ap.add_argument('--minigeo_ckpt', type=str, default=None)

    # New flags
    ap.add_argument('--backbone', default='resnet18', choices=['resnet18', 'vit_tiny'],
                    help='Vision backbone (default: resnet18)')
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--augment', default='none', choices=['none', 'light', 'strong'])
    ap.add_argument('--amp', type=lambda x: str(x).lower() == 'true', default=False)
    ap.add_argument('--save_best', type=lambda x: str(x).lower() == 'true', default=True)
    ap.add_argument('--val_split', type=float, default=0.1, help='Fraction for validation split')
    ap.add_argument('--seed', type=int, default=42)

    args = ap.parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    # Tokenizer & transforms
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    t_train = make_transforms(args.image_size, args.augment)
    t_val = make_transforms(args.image_size, "none")  # keep val deterministic

    # Split manifest into train/val temporary CSVs
    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    random.shuffle(rows)
    n_total = len(rows)
    n_val = max(1, int(n_total * args.val_split))
    val_rows = rows[:n_val]
    train_rows = rows[n_val:]

    tmp_train = os.path.join(args.out_dir, "_train_manifest.csv")
    tmp_val = os.path.join(args.out_dir, "_val_manifest.csv")

    def write_manifest(path, dict_rows):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["filepath", "label_text", "lat", "lon"])
            for r in dict_rows:
                w.writerow([r["filepath"], r.get("label_text", ""), r.get("lat", ""), r.get("lon", "")])

    write_manifest(tmp_train, train_rows)
    write_manifest(tmp_val, val_rows)

    # Datasets / loaders
    ds_train = ManifestDataset(tmp_train, t_train, tokenizer, use_coords=args.use_coords)
    ds_val = ManifestDataset(tmp_val, t_val, tokenizer, use_coords=args.use_coords)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)

    # Model
    model = GeoVerse(
        dim=args.dim,
        text_model=args.text_model,
        use_coords=args.use_coords,
        minigeo_ckpt=args.minigeo_ckpt,
        backbone=args.backbone,
    ).to(device)

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_val = float("inf")
    last_ckpt = os.path.join(args.out_dir, "last.pt")
    best_ckpt = os.path.join(args.out_dir, "best.pt")

    # Training loop
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}/{args.epochs}")
        running = 0.0
        for batch in pbar:
            for k in ("image", "input_ids", "attention_mask", "coord"):
                batch[k] = batch[k].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=args.amp):
                loss = forward_step(model, batch, amp=args.amp, temp=args.temp)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            pbar.set_postfix(loss=f"{(running / len(pbar)):0.4f}")

        # Validation
        val_loss = evaluate(model, dl_val, device, amp=args.amp, temp=args.temp)
        print(f"\nEpoch {epoch} — val_loss: {val_loss:.4f}")

        # Save last
        save_checkpoint(last_ckpt, model, epoch, args, best_val)

        # Save best
        if args.save_best and val_loss < best_val:
            best_val = val_loss
            save_checkpoint(best_ckpt, model, epoch, args, best_val)
            print(f"✅ Saved best to {best_ckpt}")

    print("Training complete.")
    print(f"Last checkpoint: {last_ckpt}")
    if args.save_best:
        print(f"Best checkpoint: {best_ckpt} (val_loss={best_val:.4f})")


if __name__ == "__main__":
    main()