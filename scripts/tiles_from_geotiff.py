import argparse, os, csv
import rasterio
from rasterio.windows import Window
from PIL import Image
import numpy as np
from rasterio.warp import transform as crs_transform

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tiff', required=True)
    ap.add_argument('--out_dir', default='data/tiles')
    ap.add_argument('--manifest', default='data/manifest.csv')
    ap.add_argument('--tile', type=int, default=256)
    ap.add_argument('--stride', type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = [("filepath","label_text","lat","lon")]

    with rasterio.open(args.tiff) as src:
        H, W = src.height, src.width
        for r in range(0, H - args.tile + 1, args.stride):
            for c in range(0, W - args.tile + 1, args.stride):
                win = Window(c, r, args.tile, args.tile)
                arr = src.read(out_shape=(3, args.tile, args.tile), window=win)
                arr = np.transpose(arr, (1,2,0))
                img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
                fname = f"tile_r{r}_c{c}.png"
                fpath = os.path.join(args.out_dir, fname)
                img.save(fpath)
                rr, cc = r + args.tile//2, c + args.tile//2
                x, y = src.transform * (cc, rr)
                lat = lon = ''
                try:
                    lon_, lat_ = crs_transform(src.crs, 'EPSG:4326', [x], [y])
                    lon, lat = float(lon_[0]), float(lat_[0])
                except Exception:
                    pass
                rows.append((fpath, 'unknown', lat, lon))

    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)
    with open(args.manifest, 'w', newline='') as f:
        csv.writer(f).writerows(rows)
    print(f"Wrote {len(rows)-1} tiles → {args.manifest}")

if __name__ == '__main__':
    main()
