import argparse, os, csv, zipfile
from PIL import Image

def extract_if_zip(zip_path: str, dest_dir: str):
    if not zip_path: return None
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip not found: {zip_path}")
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    return dest_dir

def find_ucm_root(root_dir: str):
    # UC Merced unzips to <root>/UCMerced_LandUse/Images/<Class>/*.tif
    candidates = [
        os.path.join(root_dir, "UCMerced_LandUse", "Images"),
        os.path.join(root_dir, "Images"),
        root_dir
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError("Could not locate 'Images' folder under given root.")

def build_manifest(src_root: str, out_dir: str, manifest_csv: str, image_size: int):
    dst_root = os.path.join(out_dir, "ucm_resized")
    os.makedirs(dst_root, exist_ok=True)

    rows = [("filepath","label_text","lat","lon")]
    total = 0

    # class folders inside Images/
    classes = [d for d in sorted(os.listdir(src_root)) if os.path.isdir(os.path.join(src_root, d))]
    for cls in classes:
        in_cls = os.path.join(src_root, cls)
        out_cls = os.path.join(dst_root, cls)
        os.makedirs(out_cls, exist_ok=True)

        for fn in sorted(os.listdir(in_cls)):
            if not fn.lower().endswith((".tif", ".tiff", ".jpg", ".jpeg", ".png")):
                continue
            src_path = os.path.join(in_cls, fn)
            try:
                img = Image.open(src_path).convert("RGB").resize((image_size, image_size), Image.BICUBIC)
                out_name = os.path.splitext(fn)[0] + ".jpg"
                out_path = os.path.join(out_cls, out_name)
                img.save(out_path, quality=95)
            except Exception:
                continue
            rows.append((out_path, cls, "", ""))
            total += 1

    os.makedirs(os.path.dirname(manifest_csv), exist_ok=True)
    with open(manifest_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"✅ Wrote {total} images")
    print(f"✅ Manifest: {manifest_csv}")
    print(f"✅ Resized dataset at: {dst_root}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip_path", type=str, default=None, help="Local UC Merced zip")
    ap.add_argument("--root_dir", type=str, default=None, help="Already-extracted UC Merced root")
    ap.add_argument("--out_dir", type=str, default="data/public_ucm")
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--manifest", type=str, default="data/public_ucm_manifest.csv")
    args = ap.parse_args()

    if not args.zip_path and not args.root_dir:
        raise SystemExit("Provide either --zip_path (local zip) OR --root_dir (extracted). No network downloads are attempted.")

    if args.zip_path:
        extracted = extract_if_zip(args.zip_path, args.out_dir)
        src_root = find_ucm_root(extracted or args.out_dir)
    else:
        src_root = find_ucm_root(args.root_dir)

    build_manifest(src_root, args.out_dir, args.manifest, args.image_size)

if __name__ == "__main__":
    main()