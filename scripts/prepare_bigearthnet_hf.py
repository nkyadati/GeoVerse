# scripts/prepare_bigearthnet_hf.py
import os, csv
from datasets import load_dataset
from PIL import Image

def main():
    out_dir = "data/bigearthnet_images"
    manifest = "data/bigearthnet_manifest.csv"
    os.makedirs(out_dir, exist_ok=True)

    # Load BigEarthNet RGB subset (HF)
    ds = load_dataset("Sentinel-2/BigEarthNet", "RGB", split="train")

    rows = [("filepath","label_text","lat","lon")]
    max_samples = 2000  # adjust as needed
    count = 0

    for ex in ds:
        if count >= max_samples:
            break
        img = ex["image"]  # PIL Image
        labels = ex["labels"] or ["unknown"]
        label_text = labels[0]
        lat, lon = ex.get("lat", ""), ex.get("lon", "")

        fname = f"{label_text}_{count:05d}.jpg"
        fpath = os.path.join(out_dir, fname)
        img.save(fpath, quality=95)
        rows.append((fpath, label_text, lat, lon))
        count += 1

    os.makedirs(os.path.dirname(manifest), exist_ok=True)
    with open(manifest, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"✅ Saved {count} images to {out_dir}")
    print(f"✅ Manifest written to {manifest}")

if __name__ == "__main__":
    main()