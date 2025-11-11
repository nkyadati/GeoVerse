# 🌍 GeoVerse — Unified Geospatial Embedding Space

> **GeoVerse** is a lightweight planetary foundation model prototype that learns a **shared embedding space** for **satellite imagery**, **natural-language descriptions**, and optionally **geographic coordinates**.  
> It enables **text-to-image**, **image-to-image**, and **coordinate-to-image** retrieval — a step toward universal geospatial understanding.

---

## 🚀 Highlights
- 🔗 **Unified latent space** for imagery, language, and coordinates  
- 🧠 **CLIP-style contrastive training** with ResNet-18 or ViT-Tiny backbones  
- 🌐 **Cross-dataset generalization** (trained on EuroSAT, tested zero-shot on UC Merced)  
- ⚙️ **Modular OOP design** (Extractor / Encoder / Reorderer / Evaluator structure)  
- 💡 **Interactive demo**
  - **Semantic Earth Explorer** — text → image retrieval  
  
---

## 🧩 Project structure
```
GeoVerse/
│
├── src/                        # Core codebase
│   ├── models/                 # Encoders (image, text, coord)
│   ├── train.py                # Training loop (CLIP-style)
│   ├── export_embeddings.py    # Embedding export utility
│   └── utils.py                # Helpers
│
├── demos/                      # Streamlit demos
│   ├── 01_semantic_earth_explorer.py
│   ├── 02_geo_similarity_explorer.py
│   └── 03_explain_location.py
│
├── scripts/                    # Dataset prep utilities
│   ├── prepare_eurosat.py
│   ├── prepare_ucmerced.py
│   ├── tiles_from_geotiff.py
│   └── add_coords_random.py
│
├── data/                       # Manifest CSVs
│   ├── eurosat_manifest.csv
│   ├── public_ucm_manifest.csv
│   └── ...
│
└── runs/                       # Model checkpoints + embeddings
    └── geoverse_vit_tiny_v1/
```

---

## ⚙️ Setup

### 1. Clone and create environment
```bash
git clone https://github.com/<your-username>/GeoVerse.git
cd GeoVerse
python3 -m venv GeoVerse_venv
source GeoVerse_venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> 🧩 Python ≥3.10 is recommended.  
> Conda is **not** required — this project uses `venv`.

### 2. Prepare datasets
**EuroSAT** (training):
```bash
python scripts/prepare_eurosat.py
```

**UC Merced** (retrieval testing):
```bash
python scripts/prepare_ucmerced.py
```

> Each script produces a `manifest.csv` listing  
> `filepath,label_text,lat,lon` for your dataset.

(Optional) add synthetic coordinates:
```bash
python scripts/add_coords_random.py \
  --in_manifest data/eurosat_manifest.csv \
  --out_manifest data/eurosat_manifest_coords.csv \
  --preset eurosat
```

---

## 🧠 Train a model

### Example: ViT-Tiny backbone
```bash
python -m src.train \
  --manifest data/eurosat_manifest_coords.csv \
  --out_dir runs/geoverse_vit_tiny_v1 \
  --backbone vit_tiny \
  --image_size 224 \
  --batch_size 32 \
  --epochs 15 \
  --lr 2e-4 \
  --weight_decay 1e-4 \
  --augment strong \
  --amp true \
  --use_coords true
```

> The model learns joint embeddings for images, text, and coordinates using contrastive loss.

---

## 📤 Export embeddings for retrieval

```bash
python -m src.export_embeddings \
  --manifest data/public_ucm_manifest.csv \
  --ckpt runs/geoverse_vit_tiny_v1/best.pt \
  --out_npy runs/geoverse_vit_tiny_ucm/embeddings \
  --backbone vit_tiny \
  --image_size 224 \
  --use_coords false
```

This creates:
```
embeddings_images.npy
embeddings_texts.npy
meta.json
```

---

## 🌎 Run the demo

### Semantic Earth Explorer (Text → Image)
```bash
streamlit run demos/01_semantic_earth_explorer.py -- \
  --emb runs/geoverse_vit_tiny_ucm/embeddings \
  --ckpt runs/geoverse_vit_tiny_v1/best.pt \
  --backbone vit_tiny \
  --use_coords false
```

**Try queries like:**
- “dense urban area with grid-like streets”  
- “airport with visible runways”  
- “agricultural fields with different crop colors”  
- “coastal region with beach and water”  
- “forest near mountain slopes”

---

## 🧬 Architecture Overview

```
           ┌────────────────────────┐
           │  Image Encoder         │
           │  (ResNet-18 / ViT-Tiny)│
           └──────────┬─────────────┘
                      │
           ┌──────────┴───────────┐
           │ Text Encoder         │
           │ (MiniLM)             │
           └──────────┬───────────┘
                      │
           ┌──────────┴───────────┐
           │ Coord Encoder (opt.) │
           │ (MLP + sinusoid)     │
           └──────────┬───────────┘
                      │
                      ▼
              Shared Embedding (D=256)
                Contrastive CLIP loss
           (image↔text↔coord alignment)
```

---

## 🧭 Roadmap
- [ ] Integrate **BigEarthNet / So2Sat** with true coordinates  
- [ ] Add objective evaluation metrics like Recall@K for the retrieval demo
- [ ] Add **Faiss-HNSW** for fast vector search  
- [ ] Add **caption enrichment** (LLM-based class text expansion)  
- [ ] Extend to **multispectral & SAR** imagery  

---

## 💬 Citation
If you use this project, please cite or mention it as:
```
Yadati, Karthik. *GeoVerse: A Unified Geospatial Embedding Space.*
(2025) Globeholder.ai Prototype
```

---

## 👨‍💻 Author
**Karthik Yadati** — [karthik.yadati@gmail.com](mailto:karthik.yadati@gmail.com)  
Computer Vision & Applied AI Engineer  
📍 Toulouse, France  

---

### 🏁 License
MIT License © 2025 Karthik Yadati  
Use freely for research and educational purposes.
