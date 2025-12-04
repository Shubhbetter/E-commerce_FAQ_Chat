import os
import csv
import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Any
import hashlib
import requests
import math
from collections import Counter, defaultdict

# load .env if present (do NOT commit .env to source control)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


class Document:
    def __init__(self, page_content: str, metadata: Optional[dict] = None):
        self.page_content = page_content
        self.metadata = metadata or {}


# Try to import FAISS-based vectorstore; if unavailable we'll use a simple fallback.
try:
    from langchain_community.vectorstores import FAISS  # type: ignore
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False


class HFInferenceEmbeddings:
    """Embeddings using Hugging Face router endpoint (no local torch).
    Uses direct HTTPS POST to https://router.huggingface.co/api/models/{model}.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", token: Optional[str] = None):
        self.model = model_name
        self.token = token or os.environ.get("HF_TOKEN")
        if not self.token:
            raise ValueError("HF_TOKEN not set. Export it as HF_TOKEN or pass token argument.")
        self.url = f"https://router.huggingface.co/api/models/{self.model}"
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _is_num_list(self, obj: Any) -> bool:
        if not isinstance(obj, list) or len(obj) == 0:
            return False
        return all(isinstance(x, (int, float)) for x in obj)

    def _find_embedding(self, obj: Any) -> Optional[List[float]]:
        """Recursively find the first list of numbers in obj and return as floats."""
        if obj is None:
            return None
        if self._is_num_list(obj):
            return [float(x) for x in obj]
        if isinstance(obj, dict):
            if "error" in obj:
                raise RuntimeError(f"Hugging Face API error: {obj.get('error')}")
            for key in ("embedding", "embeddings", "data", "result", "outputs", "vector"):
                if key in obj:
                    val = obj[key]
                    if self._is_num_list(val):
                        return [float(x) for x in val]
                    if isinstance(val, list) and len(val) > 0:
                        for itm in val:
                            emb = self._find_embedding(itm)
                            if emb is not None:
                                return emb
                    emb = self._find_embedding(val)
                    if emb is not None:
                        return emb
            for v in obj.values():
                emb = self._find_embedding(v)
                if emb is not None:
                    return emb
            return None
        if isinstance(obj, list):
            for item in obj:
                emb = self._find_embedding(item)
                if emb is not None:
                    return emb
        return None

    def _call(self, texts: List[str]) -> List[List[float]]:
        outputs: List[List[float]] = []
        for t in texts:
            payload = {"inputs": t}
            try:
                resp = requests.post(self.url, headers=self.headers, json=payload, timeout=60)
            except Exception as e:
                raise RuntimeError(f"Hugging Face request failed: {e}")
            if resp.status_code != 200:
                # include body for debugging
                raise RuntimeError(f"Hugging Face API error: {resp.status_code} {resp.text}")
            try:
                parsed = resp.json()
            except Exception as e:
                raise RuntimeError(f"Failed to parse HF API response JSON: {e} - {resp.text}")
            emb = self._find_embedding(parsed)
            if emb is None:
                raise RuntimeError(f"Unexpected HF API output shape: {type(parsed)}; content: {parsed}")
            outputs.append(emb)
        return outputs

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._call(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._call([text])[0]


class SimpleHashEmbeddings:
    """Deterministic lightweight fallback embeddings (no external service)."""
    def __init__(self, dim: int = 32):
        self.dim = dim

    def _hash_to_vec(self, text: str):
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # produce fixed-length float vector
        arr = np.frombuffer(h, dtype=np.uint8).astype(np.float32)
        # reduce/expand to desired dim
        if arr.size >= self.dim:
            vec = arr[: self.dim]
        else:
            # repeat if needed
            reps = int(np.ceil(self.dim / arr.size))
            vec = np.tile(arr, reps)[: self.dim]
        # normalize
        vec = vec / (np.linalg.norm(vec) + 1e-12)
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_to_vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._hash_to_vec(text)


class SimpleTFIDFEmbeddings:
    """Lightweight on-the-fly TF-IDF embeddings (no external libs)."""
    def __init__(self, docs: Optional[List[str]] = None):
        self.vocab = {}
        self.idf = {}
        if docs:
            self.fit(docs)

    def _tokenize(self, text: str):
        return [t.lower() for t in "".join(c if c.isalnum() else " " for c in text).split() if len(t) > 2]

    def fit(self, docs: List[str]):
        # build vocab and idf
        df = defaultdict(int)
        for d in docs:
            tokens = set(self._tokenize(d))
            for t in tokens:
                df[t] += 1
        self.vocab = {t: i for i, t in enumerate(sorted(df.keys()))}
        n = max(1, len(docs))
        self.idf = {t: math.log((n + 1) / (df[t] + 1)) + 1.0 for t in self.vocab}

    def _tfidf_vector(self, text: str):
        tf = Counter(self._tokenize(text))
        vec = [0.0] * len(self.vocab)
        for t, cnt in tf.items():
            if t in self.vocab:
                vec[self.vocab[t]] = cnt * self.idf.get(t, 0.0)
        # normalize
        norm = math.sqrt(sum(x * x for x in vec)) + 1e-12
        return [x / norm for x in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self.vocab:
            self.fit(texts)
        return [self._tfidf_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        if not self.vocab:
            return [0.0] * 1  # won't match — ensure fit() called when building DB
        return self._tfidf_vector(text)


def create_vectordb(documents: List[Document], persist_path: str):
    """Create and persist a vector DB. Uses HF router if available, otherwise a simple numpy store."""
    persist_dir = Path(persist_path)
    persist_dir.mkdir(parents=True, exist_ok=True)

    texts = [doc.page_content for doc in documents]

    # Prefer HF, fall back to TF-IDF embeddings (better than hash)
    try:
        embedder = HFInferenceEmbeddings()
        _ = embedder.embed_query("test")
        embeddings = embedder.embed_documents(texts)
    except Exception:
        embedder = SimpleTFIDFEmbeddings(docs=texts)
        embeddings = embedder.embed_documents(texts)

    if HAVE_FAISS:
        try:
            vectordb = FAISS.from_documents(documents, embedder)  # type: ignore
            vectordb.save_local(str(persist_dir))
            return {"type": "faiss", "path": str(persist_dir)}
        except Exception:
            pass

    vecs = np.array(embeddings, dtype=np.float32)
    np.save(persist_dir / "embeddings.npy", vecs)
    with open(persist_dir / "docs.json", "w", encoding="utf-8") as fh:
        json.dump(texts, fh, ensure_ascii=False)
    return {"type": "simple", "path": str(persist_dir)}


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return a_norm @ b_norm


def get_response(query: str, vectordb_path: str, k: int = 3) -> dict:
    """Return dict with concise 'answer' (best A:) and list of retrieved 'docs' (top-k)."""
    persist = Path(vectordb_path)
    emb_path = persist / "embeddings.npy"
    docs_path = persist / "docs.json"
    if not docs_path.exists():
        return {"answer": "Vector DB not found. Build it first.", "docs": []}

    with open(docs_path, "r", encoding="utf-8") as fh:
        texts_all = json.load(fh)

    # try HF embedder first, else TF-IDF
    try:
        embedder = HFInferenceEmbeddings()
        q_emb = np.array(embedder.embed_query(query), dtype=np.float32)
        # if embeddings.npy exists and dims match, use it; else compute on the fly
        if emb_path.exists():
            vecs = np.load(emb_path)
        else:
            vecs = np.array(embedder.embed_documents(texts_all), dtype=np.float32)
    except Exception:
        # TF-IDF fallback: build vectorizer from docs and compute all vectors
        tf = SimpleTFIDFEmbeddings(docs=texts_all)
        vecs = np.array(tf.embed_documents(texts_all), dtype=np.float32)
        q_emb = np.array(tf.embed_query(query), dtype=np.float32)

    # cosine similarity
    def cosine_sim_matrix(mat: np.ndarray, q: np.ndarray):
        if mat.size == 0 or q.size == 0:
            return np.array([])
        mat_norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
        q_norm = q / (np.linalg.norm(q) + 1e-12)
        return mat_norm @ q_norm

    sims = cosine_sim_matrix(vecs, q_emb)
    if sims.size == 0:
        return {"answer": "No relevant documents found.", "docs": []}

    idxs = list(np.argsort(-sims)[:k])
    best_docs = [texts_all[i] for i in idxs]
    best_raw = best_docs[0] if best_docs else ""

    # extract A: portion
    answer = ""
    if best_raw:
        if "\nA:" in best_raw:
            answer = best_raw.split("\nA:", 1)[1].strip()
        elif " A: " in best_raw and best_raw.startswith("Q:"):
            answer = best_raw.split(" A: ", 1)[1].strip()
        else:
            lines = best_raw.splitlines()
            answer = "\n".join(lines[1:]).strip() if len(lines) > 1 else best_raw.strip()

    return {"answer": answer or "No relevant answer found.", "docs": best_docs}


if __name__ == "__main__":
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
                    docs.append(Document(page_content=f"Q: {q}\nA: {a}"))

    if not docs:
        print("No documents loaded. Populate Ecommerce_FAQs.csv with Question and Answer columns.")
    else:
        info = create_vectordb(docs, persist_path=PERSIST_PATH)
        print("Vector DB built:", info)
        print("Sample response:")
        print(get_response("What is the return policy?", vectordb_path=PERSIST_PATH, k=3))
