import argparse, os, csv, json, glob
from PIL import Image

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Path to BigEarthNet-S2 (pre-extracted RGB patches)')
    ap.add_argument('--manifest', default='data/manifest.csv')
    ap.add_argument('--image_size', type=int, default=128)
    args = ap.parse_args()

    rows = [("filepath","label_text","lat","lon")]
    jpgs = glob.glob(os.path.join(args.root, '**', '*.jpg'), recursive=True)
    for jp in jpgs:
        meta_path = jp.replace('.jpg', '.json')
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            lat, lon = meta.get('lat'), meta.get('lon')
            labels = meta.get('labels', ['unknown'])
            label_text = labels[0] if isinstance(labels, list) else str(labels)
        except Exception:
            continue
        img = Image.open(jp).convert('RGB').resize((args.image_size, args.image_size))
        img.save(jp, quality=95)
        rows.append((jp, label_text, lat, lon))

    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)
    with open(args.manifest, 'w', newline='') as f:
        csv.writer(f).writerows(rows)
    print(f"Wrote {len(rows)-1} rows → {args.manifest}")

if __name__ == '__main__':
    main()
