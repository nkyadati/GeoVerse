import argparse, os, csv, zipfile, urllib.request, shutil
from PIL import Image
from tqdm import tqdm

UCM_URL = "https://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip"

def download(url: str, out_zip: str):
    os.makedirs(os.path.dirname(out_zip), exist_ok=True)
    if os.path.exists(out_zip):
        print(f"➡️  Reusing existing file: {out_zip}")
        return
    print(f"⬇️  Downloading UC Merced Land-Use…")
    with urllib.request.urlopen(url) as r, open(out_zip, "wb") as f:
        total = int(r.info().get("Content-Length", -1))
        block = 8192
        with tqdm(total=total if total>0 else None, unit="B", unit_scale=True) as pbar:
            while True:
                chunk = r.read(block)
                if not chunk: break
                f.write(chunk)
                pbar.update(len(chunk))
    print(f"✅ Downloaded: {out_zip}")

def unzip(zip_path: str, dest_dir: str):
    if os.path.exists(dest_dir) and any(os.scandir(dest_dir)):
        print(f"➡️  Data already extracted: {dest_dir}")
        return
    print(f"📦 Extracting to {dest_dir} …")
    os.makedirs(dest_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print("✅ Extraction done")

def build_manifest(root_in: str, out_dir: str, manifest_csv: str, image_size: int):
    """
    UC Merced unzips to: <out_dir>/UCMerced_LandUse/Images/<ClassName>/*.tif
    We'll convert to RGB JPG, resize, and store under <out_dir>/ucm_resized/<ClassName>/xxx.jpg
    """
    src_root = os.path.join(root_in, "UCMerced_LandUse", "Images")
    dst_root = os.path.join(root_in, "ucm_resized")
    os.makedirs(dst_root, exist_ok=True)

    rows = [("filepath","label_text","lat","lon")]
    classes = [d for d in sorted(os.listdir(src_root)) if os.path.isdir(os.path.join(src_root, d))]
    count = 0

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
            count += 1

    os.makedirs(os.path.dirname(manifest_csv), exist_ok=True)
    with open(manifest_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    print(f"✅ Wrote {count} images")
    print(f"✅ Manifest: {manifest_csv}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/public_ucm")
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--manifest", default="data/public_ucm_manifest.csv")
    args = ap.parse_args()

    zip_path = os.path.join(args.out_dir, "UCMerced_LandUse.zip")
    download(UCM_URL, zip_path)
    unzip(zip_path, args.out_dir)
    build_manifest(args.out_dir, args.out_dir, args.manifest, args.image_size)

if __name__ == "__main__":
    main()