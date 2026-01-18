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

st.set_page_config(
    page_title="E-commerce FAQ Assistant",
    page_icon="🛒",
    layout="centered"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
.main {
    background-color: #f9fafb;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 800px;
}
.card {
    background: white;
    padding: 1.25rem;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    margin-bottom: 1rem;
}
.answer {
    font-size: 1.05rem;
    line-height: 1.6;
}
.footer {
    text-align: center;
    color: #6b7280;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.title("🛒 E-commerce FAQ Assistant")
st.caption("Ask anything about orders, returns, payments & support")

# ---------- Sidebar ----------
st.sidebar.header("⚙️ System Info")
st.sidebar.write(f"📄 **CSV:** `{CSV_PATH.name}`")
st.sidebar.write(f"🧠 **Vector DB:** FAISS")
st.sidebar.write(f"📁 **Index Path:** `{PERSIST_PATH}`")

if created:
    st.sidebar.info("Sample FAQ dataset was auto-created.")

# ---------- Status ----------
if vectordb is None:
    st.error("❌ Vector database not ready. Check logs.")
else:
    st.success("✅ Knowledge base loaded successfully")

# ---------- Query Input ----------
st.markdown("### ❓ Ask your question")
query = st.text_input(
    "",
    placeholder="e.g. How can I track my order?"
)

col1, col2 = st.columns([1, 3])
with col1:
    k = st.number_input("Top-K", 1, 10, 3)
with col2:
    show_docs = st.checkbox("Show retrieved documents")

# ---------- Action ----------
if st.button("🔍 Get Answer", use_container_width=True) and query.strip():
    with st.spinner("Thinking..."):
        try:
            resp = get_response(query, vectordb_path=PERSIST_PATH, k=int(k))
            answer = resp.get("answer", "")
            docs = resp.get("docs", [])
        except Exception as e:
            st.error(f"Error: {e}")
            answer, docs = "", []

    if answer:
        st.markdown("### ✅ Answer")
        st.markdown(f"""
        <div class="card answer">
            {answer}
        </div>
        """, unsafe_allow_html=True)

    if show_docs and docs:
        st.markdown("### 📄 Retrieved Documents")
        for i, d in enumerate(docs[:k], 1):
            st.markdown(f"""
            <div class="card">
                <strong>Doc {i}</strong><br>
                {d}
            </div>
            """, unsafe_allow_html=True)

# ---------- Footer ----------
st.markdown("---")
st.markdown(
    "<div class='footer'><strong>Created by Shubham Pandey</strong></div>",
    unsafe_allow_html=True
)
