"""Streamlit UI: `streamlit run app.py`."""

from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps

from localsearch import documents, images

st.set_page_config(page_title="Local Search", layout="wide")
st.title("Local Search")
st.caption("Semantic search over your own photos and documents. Nothing leaves this machine.")

with st.sidebar:
    st.header("Index")
    kind = st.radio("Library", ["Photos", "Documents"], horizontal=True)
    directory = st.text_input("Directory", value=str(Path.home() / ("Pictures" if kind == "Photos" else "Documents")))
    force = st.checkbox("Re-embed everything", help="Ignore what's already indexed.")
    if st.button("Build / update index"):
        root = Path(directory).expanduser()
        if not root.is_dir():
            st.error(f"Not a directory: {root}")
        else:
            bar = st.progress(0.0)
            module = images if kind == "Photos" else documents
            n = module.index(root, force=force, progress=lambda d, t: bar.progress(d / t if t else 1.0))
            bar.progress(1.0)
            st.success(f"Indexed {n} new {'images' if kind == 'Photos' else 'chunks'}.")

photos_tab, docs_tab = st.tabs(["Photos", "Documents"])

with photos_tab:
    q = st.text_input("Describe a photo", placeholder="kids asleep on a couch, a red car in snow")
    limit = st.slider("Results", 1, 24, 8)
    if q:
        hits = images.search(q, limit)
        cols = st.columns(4)
        for i, hit in enumerate(hits):
            path = Path(hit["payload"]["path"])
            with cols[i % 4]:
                if path.exists():
                    st.image(ImageOps.exif_transpose(Image.open(path)), use_container_width=True)
                    st.caption(f"{hit['score']:.3f} · {path.name}")
                else:
                    st.warning(f"Missing: {path.name}")

with docs_tab:
    q = st.text_input("Ask about your documents", placeholder="what caused the flood narrative to differ")
    if q:
        for hit in documents.search(q, 5):
            p = hit["payload"]
            with st.expander(f"{hit['score']:.3f} · {p['filename']} (chunk {p['chunk']}/{p['chunks']})", expanded=True):
                st.write(p["text"])
                st.caption(p["path"])
