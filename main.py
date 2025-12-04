import os
from pathlib import Path
from typing import List, Optional

import streamlit as st

# try to import langchain Document, fallback to simple local Document if not installed
try:
    from langchain_core.documents import Document
except Exception:
    class Document:
        def __init__(self, page_content: str, metadata: Optional[dict] = None):
            self.page_content = page_content
            self.metadata = metadata or {}

from langchain_helper import create_vectordb, get_response

ROOT = Path(__file__).parent
CSV_PATH = ROOT / "Ecommerce_FAQs.csv"
PERSIST_PATH = str(ROOT / "faiss_index")

def ensure_sample_csv(path: Path) -> bool:
    if path.exists():
        return False
    sample = [
        {"Question": "How do I track my order?", "Answer": "You can track your order via the Orders page."},
        {"Question": "What is the return policy for wrong items?", "Answer": "Contact support within 7 days for return."},
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("Question,Answer\n")
        for r in sample:
            # simple CSV escape for commas/newlines
            q = r["Question"].replace('"', '""')
            a = r["Answer"].replace('"', '""')
            fh.write(f'"{q}","{a}"\n')
    return True

created = ensure_sample_csv(CSV_PATH)

st.set_page_config(page_title="E-commerce FAQ (local)", layout="centered")
st.title("E-commerce FAQ Chat (local-only)")

st.sidebar.header("Diagnostics")
st.sidebar.write(f"CSV: {CSV_PATH}")
st.sidebar.write(f"FAISS persist path: {PERSIST_PATH}")
if created:
    st.sidebar.info("Sample CSV created because none was present.")

@st.cache_data(ttl=60 * 60)
def load_documents_from_csv(path: Path) -> List[Document]:
    docs: List[Document] = []
    if not path.exists():
        return docs
    import csv
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        headers = {h.lower(): h for h in (reader.fieldnames or [])}
        q_col = headers.get("question")
        a_col = headers.get("answer")
        if not q_col or not a_col:
            flds = reader.fieldnames or []
            if len(flds) >= 2:
                q_col, a_col = flds[0], flds[1]
        for row in reader:
            q = row.get(q_col, "").strip()
            a = row.get(a_col, "").strip()
            if q and a:
                docs.append(Document(page_content=f"Q: {q}\nA: {a}", metadata={"source": str(path)}))
    return docs

@st.cache_resource
def get_or_build_vectordb():
    docs = load_documents_from_csv(CSV_PATH)
    if not docs:
        return None
    # create_vectordb in langchain_helper returns a dict with info (or saves)
    try:
        info = create_vectordb(docs, persist_path=PERSIST_PATH)
        return info
    except Exception as e:
        st.sidebar.error(f"Failed to build vector DB: {e}")
        return None

vectordb = get_or_build_vectordb()

if vectordb is None:
    st.error("No documents loaded or vector DB failed to build. Check terminal logs.")
else:
    st.success("Dataset loaded. Vector DB ready.")

query = st.text_input("Ask a question about the e-commerce site", "")
col1, col2 = st.columns([1, 4])
with col1:
    k = st.number_input("Top-k documents", min_value=1, max_value=10, value=3)
with col2:
    show_docs = st.checkbox("Show retrieved documents", value=False)

if st.button("Get answer") and query.strip():
    if vectordb is None:
        st.error("Vector DB not available.")
    else:
        with st.spinner("Retrieving answer..."):
            try:
                resp = get_response(query, vectordb_path=PERSIST_PATH, k=int(k))
                answer = resp.get("answer", "")
                docs = resp.get("docs", [])
            except Exception as e:
                st.error(f"Error retrieving response: {e}")
                answer = ""
                docs = []

        if answer:
            st.subheader("Answer")
            st.write(answer)

        if show_docs and docs:
            st.subheader("Retrieved documents (top-k)")
            for i, p in enumerate(docs[: int(k)], start=1):
                st.markdown(f"**Doc {i}:**")
                st.write(p)

st.markdown("---")
st.caption("If the UI is blank, check the terminal where you ran `streamlit run main.py` for errors.")

st.markdown("---")
st.caption("If the UI is blank, check the terminal where you ran `streamlit run main.py` for errors.")
