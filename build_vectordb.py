import csv
from pathlib import Path
import sys

from langchain.schema import Document
from langchain_helper import create_vectordb

ROOT = Path(__file__).parent
CSV_PATH = ROOT / "Ecommerce_FAQs.csv"
PERSIST_PATH = str(ROOT / "faiss_index")


def load_documents_from_csv(path: Path) -> list[Document]:
    docs = []
    if not path.exists():
        print(f"CSV dataset not found at {path}. Create a CSV with 'Question' and 'Answer' columns.", file=sys.stderr)
        return docs

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
                print("CSV must contain Question and Answer columns.", file=sys.stderr)
                return []

        for row in reader:
            q = row.get(q_col, "").strip()
            a = row.get(a_col, "").strip()
            if q and a:
                content = f"Question: {q}\nAnswer: {a}"
                docs.append(Document(page_content=content))
    return docs


def main():
    print(f"Loading CSV from: {CSV_PATH}")
    docs = load_documents_from_csv(CSV_PATH)
    if not docs:
        print("No documents loaded. Exiting.", file=sys.stderr)
        sys.exit(1)

    print(f"Creating/loading vector DB at: {PERSIST_PATH}")
    try:
        vectordb = create_vectordb(docs, persist_path=PERSIST_PATH)
    except Exception as e:
        print(f"Failed to create/load vectordb: {e}", file=sys.stderr)
        sys.exit(1)

    # quick sanity check
    try:
        sample = vectordb.similarity_search("What is the return policy?", k=1)
        print(f"Vector DB built. Sample document retrieved: {len(sample)} item(s).")
    except Exception as e:
        print(f"Vector DB built but similarity search failed: {e}", file=sys.stderr)

    print("Done. Vector DB persisted to:", PERSIST_PATH)


if __name__ == "__main__":
    main()