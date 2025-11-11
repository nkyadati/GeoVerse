import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

class GeoVerseDataset(Dataset):
    def __init__(self, manifest_csv: str, image_size: int = 128, text_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.df = pd.read_csv(manifest_csv)
        assert {"filepath","label_text"}.issubset(self.df.columns), "manifest.csv must have filepath,label_text[,lat,lon]"
        self.has_coords = ("lat" in self.df.columns) and ("lon" in self.df.columns)
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        self.tok = AutoTokenizer.from_pretrained(text_model)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.filepath).convert("RGB")
        img = self.transform(img)

        txt = str(row.label_text)
        enc = self.tok(txt, padding="max_length", truncation=True, max_length=24, return_tensors="pt")
        item = {
            "image": img,
            "text_ids": enc["input_ids"].squeeze(0),
            "text_mask": enc["attention_mask"].squeeze(0),
            "label_text": txt,
            "filepath": row.filepath,
        }
        if self.has_coords and (not pd.isna(row.get("lat", None))) and (not pd.isna(row.get("lon", None))):
            item["latlon"] = torch.tensor([float(row.lat), float(row.lon)], dtype=torch.float32)
        return item

def collate(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        vals = [b[k] for b in batch if k in b]
        if k in ("image","text_ids","text_mask"):
            out[k] = torch.stack(vals, dim=0)
        elif k == "latlon":
            out[k] = torch.stack(vals, dim=0) if len(vals) > 0 else None
        else:
            out[k] = vals
    return out
