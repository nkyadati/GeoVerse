# scripts/prepare_eurosat.py
import argparse
import os
import csv
from PIL import Image
from torchvision import datasets, transforms

# Creates data/eurosat_images/* and data/manifest.csv
# Columns: filepath,label_text,lat,lon  (lat/lon blank for EuroSAT)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='data/eurosat_images')
    ap.add_argument('--manifest', default='data/manifest.csv')
    ap.add_argument('--image_size', type=int, default=128)
    ap.add_argument('--max_per_class', type=int, default=1200)  # cap for quick runs
    ap.add_argument('--download_root', default='data/_eurosat_cache')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)

    # Download/load EuroSAT via torchvision
    # Split is not needed; EuroSAT is a single dataset (label folders).
    base = datasets.EuroSAT(root=args.download_root, download=True)

    # Build a mapping label_index -> label_name
    # torchvision EuroSAT folders are label names
    classes = base.classes  # e.g., ['AnnualCrop','Forest',...]
    counts = {c: 0 for c in classes}

    # Simple transform for resizing before saving
    resize = transforms.Resize((args.image_size, args.image_size), interpolation=Image.BICUBIC)

    rows = [("filepath", "label_text", "lat", "lon")]

    for idx in range(len(base)):
        img, label_idx = base[idx]
        label_name = classes[label_idx]

        # enforce per-class cap
        if counts[label_name] >= args.max_per_class:
            continue

        # resize & save
        img = resize(img)
        fname = f"{label_name}_{counts[label_name]+1:05d}.jpg"
        fpath = os.path.join(args.out_dir, fname)
        img.save(fpath, quality=95)

        rows.append((fpath, label_name, "", ""))
        counts[label_name] += 1

    with open(args.manifest, 'w', newline='') as f:
        csv.writer(f).writerows(rows)

    total = sum(counts.values())
    print(f"✅ Wrote {total} images to {args.out_dir}")
    print(f"✅ Manifest: {args.manifest}")
    print("Class counts:", {k: v for k, v in counts.items() if v > 0})

if __name__ == '__main__':
    main()