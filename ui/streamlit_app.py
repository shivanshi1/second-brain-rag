"""Minimal Streamlit UI placeholder (Phase 5). Run: streamlit run ui/streamlit_app.py"""

import httpx
import streamlit as st

API = "http://localhost:8000"

st.title("Second Brain")
query = st.text_input("Search your notes")
tags = st.text_input("Filter by tags (comma-separated)")

if query:
    filters = {"tags": [t.strip() for t in tags.split(",") if t.strip()]}
    hits = httpx.post(f"{API}/search", json={"query": query, "filters": filters}, timeout=60).json()
    for hit in hits:
        with st.expander(f"{hit['document_title']}  {hit['heading_path']}"):
            st.write(hit["text"])
