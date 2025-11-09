import os
from pathlib import Path
import csv
import sys
import traceback

import streamlit as st
from langchain.schema import Document

from langchain_helper import create_vectordb, get_response

ROOT = Path(__file__).parent
CSV_PATH = ROOT / "Ecommerce_FAQs.csv"
PERSIST_PATH = str(ROOT / "faiss_index")

def ensure_sample_csv(path: Path):
    if path.exists():
        return False
    sample = [
        {"Question": "How do I track my order?", "Answer": "You can track your order via the Orders page."},
        {"Question": "What is the return policy for wrong items?", "Answer": "Contact support within 7 days for return."},
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["Question", "Answer"])
        writer.writeheader()
        writer.writerows(sample)
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
def load_documents_from_csv(path: Path) -> list[Document]:
    docs = []
    if not path.exists():
        return docs
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            headers = {h.lower(): h for h in reader.fieldnames or []}
            q_col = headers.get("question")
            a_col = headers.get("answer")
            if not q_col or not a_col:
                fieldnames = reader.fieldnames or []
                if len(fieldnames) >= 2:
                    q_col, a_col = fieldnames[0], fieldnames[1]
                else:
                    return []
            for row in reader:
                q = (row.get(q_col) or "").strip()
                a = (row.get(a_col) or "").strip()
                if q and a:
                    content = f"Question: {q}\nAnswer: {a}"
                    docs.append(Document(page_content=content))
    except Exception:
        traceback.print_exc(file=sys.stderr)
    return docs

@st.cache_resource
def get_or_build_vectordb():
    documents = load_documents_from_csv(CSV_PATH)
    if not documents:
        return None
    try:
        vectordb = create_vectordb(documents, persist_path=PERSIST_PATH)
        return vectordb
    except Exception:
        traceback.print_exc(file=sys.stderr)
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
        st.error("Vector DB not available. See logs in terminal.")
    else:
        with st.spinner("Retrieving and generating answer (local models)..."):
            try:
                if show_docs:
                    docs = vectordb.similarity_search(query, k=int(k))
                    with st.expander("Retrieved documents"):
                        for i, d in enumerate(docs, 1):
                            st.markdown(f"**Doc {i}:**")
                            st.write(d.page_content)

                answer = get_response(query, vectordb=vectordb, k=int(k))
                st.markdown("**Answer:**")
                st.write(answer)
            except Exception as e:
                st.error(f"Error generating answer: {e}")
                traceback.print_exc(file=sys.stderr)

st.markdown("---")
st.caption("If the UI is blank, check the terminal where you ran `streamlit run main.py` for errors.")