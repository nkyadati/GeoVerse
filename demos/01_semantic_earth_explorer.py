# demos/01_semantic_earth_explorer.py
import os, sys, json, argparse, math
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import streamlit as st
from PIL import Image
import torch
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer

# ---- Helpers ----------------------------------------------------------------

def _load_npy_with_fallbacks(base_dir, candidates):
    for name in candidates:
        p = os.path.join(base_dir, name)
        if os.path.isfile(p):
            return np.load(p)
    return None

def load_embeddings(emb_dir):
    """
    Returns: imgs (NxD), txts (NxD or None), coords (NxD or None), meta (dict)
    Tries multiple filename conventions for compatibility.
    """
    imgs = _load_npy_with_fallbacks(emb_dir, [
        "embeddings_images.npy",
        "img_embeddings.npy",
        "embeddings_image.npy",
    ])
    txts = _load_npy_with_fallbacks(emb_dir, [
        "embeddings_texts.npy",
        "embeddings_text.npy",
        "txt_embeddings.npy",
    ])
    coords = _load_npy_with_fallbacks(emb_dir, [
        "embeddings_coords.npy",
        "coords.npy",
    ])

    meta_path = os.path.join(emb_dir, "meta.json")
    meta = {}
    if os.path.isfile(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)

    if imgs is None:
        raise FileNotFoundError(
            f"Could not find image embeddings in {emb_dir}. "
            "Looked for: embeddings_images.npy / img_embeddings.npy."
        )

    return imgs, txts, coords, meta

def build_knn(mat, k=20):
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(mat)
    return nn

@st.cache_resource
def load_model(ckpt_path, dim, text_model, use_coords, backbone):
    from src.models.geoverse import GeoVerse
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GeoVerse(dim=dim, text_model=text_model, use_coords=use_coords, backbone=backbone).to(device)

    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    # non-strict to survive small naming/shape diffs
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(text_model)
    return model, tokenizer, device

def encode_text(model, tokenizer, device, text, max_len=32):
    enc = tokenizer(text, padding="max_length", truncation=True,
                    max_length=max_len, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    with torch.no_grad():
        zt = model.text_enc(input_ids, attn)  # (1, D), already L2-normalized
    return zt.cpu().numpy()

# ---- App --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb", required=True, help="Path to folder containing exported embeddings (*.npy) and meta.json")
    parser.add_argument("--ckpt", required=True, help="Path to trained checkpoint")
    parser.add_argument("--use_coords", type=lambda x: str(x).lower()=='true', default=False)
    parser.add_argument("--backbone", default="resnet18", choices=["resnet18","vit_tiny"])
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--text_model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = parser.parse_args()

    st.set_page_config(page_title="Semantic Earth Explorer", layout="wide")
    st.title("🌍 Semantic Earth Explorer — Text → Image Retrieval")

    # Load data
    imgs, txts, coords, meta = load_embeddings(args.emb)
    D = imgs.shape[1]
    knn = build_knn(imgs, k=50)  # build once

    # Load model + tokenizer
    model, tokenizer, device = load_model(args.ckpt, args.dim, args.text_model, args.use_coords, args.backbone)

    # Sidebar info
    with st.sidebar:
        st.header("Run Info")
        st.write(f"Embeddings dir: `{args.emb}`")
        st.write(f"Checkpoint: `{os.path.basename(args.ckpt)}`")
        st.write(f"Backbone: `{args.backbone}`")
        st.write(f"Embedding dim (images): **{D}**")
        if meta.get("count"):
            st.write(f"Items indexed: **{meta['count']}**")
        if txts is None:
            st.info("No dataset text embeddings saved — that's fine; we only need image embeddings for retrieval.")

        # Optional quick queries
        st.markdown("---")
        st.subheader("Quick queries")
        presets = [
            "dense urban area with grid-like streets",
            "coastline with beach and waves",
            "agricultural fields with different crop colors",
            "forest region near a mountain slope",
            "airport with visible runways",
        ]
        chosen = st.radio("Presets", presets, index=0)

    st.markdown("Enter a **natural-language** query describing the landscape you’re looking for:")
    q = st.text_input("Query", value=chosen, label_visibility="collapsed")
    topk = st.slider("Top-K results", min_value=3, max_value=30, value=12, step=3)

    if st.button("Search", type="primary"):
        # Encode query text → vector
        q_vec = encode_text(model, tokenizer, device, q)  # shape (1, D)
        # kNN (cosine distance)
        dist, idx = knn.kneighbors(q_vec, n_neighbors=topk)
        dist, idx = dist[0], idx[0]  # flatten

        # Render results as a grid with similarity scores
        st.markdown("### 🔎 Results")
        ncols = 4
        cols = st.columns(ncols)
        filepaths = meta.get("filepaths")
        labels = meta.get("labels")

        for i in range(topk):
            c = cols[i % ncols]
            fp = filepaths[idx[i]] if (filepaths and idx[i] < len(filepaths)) else None
            caption = []
            if labels and idx[i] < len(labels) and labels[idx[i]]:
                caption.append(labels[idx[i]])
            # cosine similarity = 1 - dist
            sim = 1.0 - float(dist[i])
            caption.append(f"cos sim: {sim:.3f}")
            cap = " • ".join(caption) if caption else f"cos sim: {sim:.3f}"

            try:
                if fp and os.path.isfile(fp):
                    c.image(Image.open(fp), caption=cap, use_column_width=True)
                else:
                    c.write(f"Missing file for index {idx[i]}"); c.code(fp or "<none>")
            except Exception as e:
                c.write(f"Error loading image: {e}")

        st.caption("Note: cosine similarity shown; higher is more similar. Retrieval is zero-shot if your model was trained on a different dataset than these embeddings.")

    # Footer diagnostics
    with st.expander("Diagnostics", expanded=False):
        st.write(f"Images embedding shape: {imgs.shape}")
        if txts is not None:
            st.write(f"Texts embedding shape: {txts.shape}")
        if coords is not None:
            st.write(f"Coords embedding shape: {coords.shape}")
        st.json({
            "meta_keys": list(meta.keys()),
            "backbone_in_meta": meta.get("backbone"),
            "dim_in_meta": meta.get("dim"),
            "use_coords_in_meta": meta.get("use_coords"),
        })

if __name__ == "__main__":
    main()