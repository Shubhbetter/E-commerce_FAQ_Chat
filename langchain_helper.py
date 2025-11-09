import os
import re
import csv
from pathlib import Path
from typing import List, Optional

# optionally load .env (harmless)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Embeddings: prefer langchain wrapper, fallback to sentence-transformers direct
EMBEDDING_IMPL = None
try:
    from langchain.embeddings import SentenceTransformerEmbeddings
    EMBEDDING_IMPL = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
except Exception:
    try:
        from sentence_transformers import SentenceTransformer
        _s_model = SentenceTransformer("all-MiniLM-L6-v2")
        class _ManualEmbeddings:
            def embed_documents(self, texts: List[str]):
                return _s_model.encode(texts, convert_to_numpy=True).tolist()
            def embed_query(self, text: str):
                return _s_model.encode([text], convert_to_numpy=True)[0].tolist()
        EMBEDDING_IMPL = _ManualEmbeddings()
    except Exception as e:
        raise RuntimeError(
            "Missing embedding backend. Install sentence-transformers and/or langchain:\n"
            "  pip3 install sentence-transformers langchain"
        ) from e

# Vectorstore (FAISS) and Document type
try:
    from langchain.vectorstores import FAISS
    from langchain.schema import Document
except Exception:
    raise RuntimeError(
        "Missing vectorstore backend. Install faiss-cpu and langchain:\n"
        "  pip3 install faiss-cpu langchain"
    )

def create_vectordb(documents: List[Document], persist_path: Optional[str] = None):
    """
    Create or load a FAISS vectorstore from a list of langchain Documents.
    If persist_path exists it will be loaded; otherwise saved to that path if provided.
    """
    if persist_path:
        try:
            return FAISS.load_local(persist_path, EMBEDDING_IMPL)
        except Exception:
            pass

    vectordb = FAISS.from_documents(documents, EMBEDDING_IMPL)
    if persist_path:
        try:
            vectordb.save_local(persist_path)
        except Exception:
            pass
    return vectordb

def _generate_with_local_llm(prompt: str, max_new_tokens: int = 200, temperature: float = 0.2) -> str:
    """
    Generate text using a local Hugging Face transformers model.
    Default model: distilgpt2 (small, CPU-friendly).
    """
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    except Exception:
        raise RuntimeError("Missing transformers. Install: pip3 install transformers torch")

    model_name = os.getenv("LOCAL_LLM_MODEL", "distilgpt2")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
    except Exception:
        raise RuntimeError(f"Failed to load local model '{model_name}'. Try 'distilgpt2' or 'gpt2'.")

    out = generator(
        prompt,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        top_k=50,
        num_return_sequences=1,
        return_full_text=False,
    )[0]["generated_text"]

    if "Answer:" in prompt or "Answer:" in out:
        return out.split("Answer:")[-1].strip()
    return out.strip()

def get_response(query: str, vectordb, k: int = 3, debug: bool = False) -> str:
    """
    Retrieve top-k documents from vectordb and generate an answer with a local LLM.
    """
    if vectordb is None:
        raise RuntimeError("vectordb is not provided. Call create_vectordb(...) and pass it in.")

    docs = vectordb.similarity_search(query, k=k)
    context = "\n\n".join(d.page_content for d in docs)

    if debug:
        return "DEBUG: Retrieved documents:\n\n" + "\n\n---\n\n".join(d.page_content for d in docs)

    prompt = (
        "You are a concise, helpful assistant. Use ONLY the provided context to answer the question.\n"
        "If the answer is not contained in the context, reply: \"I don't know. Please contact customer support.\"\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )

    answer = _generate_with_local_llm(
        prompt,
        max_new_tokens=int(os.getenv("LOCAL_LLM_MAX_NEW_TOKENS", "220")),
        temperature=float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.15")),
    )

    # cleanup repetitions and length
    answer = re.sub(r"(\bdefective products;?\b)(\s*\1){2,}", r"\1", answer, flags=re.IGNORECASE)
    answer = re.sub(r"([?.!,-])\1{3,}", r"\1", answer)
    answer = answer.strip()
    if len(answer) > 2000:
        answer = answer[:2000].rsplit(".", 1)[0] + "."

    if not answer or re.match(r"^(I don't know|No information|Unknown)", answer, flags=re.IGNORECASE):
        return "I don't know. Please contact customer support."

    return answer

if __name__ == "__main__":
    # quick CLI: build vectordb from CSV and run a sample query
    ROOT = Path(__file__).parent
    CSV_PATH = ROOT / "Ecommerce_FAQs.csv"
    PERSIST_PATH = str(ROOT / "faiss_index")

    docs: List[Document] = []
    if CSV_PATH.exists():
        with CSV_PATH.open(newline="", encoding="utf-8") as fh:
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
                    docs.append(Document(page_content=f"Question: {q}\nAnswer: {a}"))

    if not docs:
        print("No documents loaded. Populate Ecommerce_FAQs.csv with Question and Answer columns.")
    else:
        vectordb = create_vectordb(docs, persist_path=PERSIST_PATH)
        print("Sample response:")
        print(get_response("What is the return policy for wrong items?", vectordb=vectordb, k=3))