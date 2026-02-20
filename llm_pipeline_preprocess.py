import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd


def clean_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def to_instruction_row(question: str, answer: str) -> dict:
    return {
        "instruction": "Answer the e-commerce customer question factually and concisely.",
        "input": question,
        "output": answer,
        "text": (
            "### Instruction:\n"
            "Answer the e-commerce customer question factually and concisely.\n\n"
            f"### Input:\n{question}\n\n"
            f"### Response:\n{answer}"
        ),
    }


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess Ecommerce FAQ data for LoRA/QLoRA fine-tuning.")
    parser.add_argument("--input_csv", default="Ecommerce_FAQs.csv")
    parser.add_argument("--output_dir", default="training_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args()

    random.seed(args.seed)

    df = pd.read_csv(args.input_csv)
    lower_cols = {c.lower(): c for c in df.columns}
    q_col = lower_cols.get("question", df.columns[0])
    a_col = lower_cols.get("answer", df.columns[1])

    df = df[[q_col, a_col]].rename(columns={q_col: "question", a_col: "answer"})
    df["question"] = df["question"].map(clean_text)
    df["answer"] = df["answer"].map(clean_text)
    df = df[(df["question"] != "") & (df["answer"] != "")]
    df = df.drop_duplicates(subset=["question"]).reset_index(drop=True)

    rows = [to_instruction_row(q, a) for q, a in zip(df["question"], df["answer"]) ]
    random.shuffle(rows)

    n = len(rows)
    train_end = int(n * args.train_ratio)
    val_end = train_end + int(n * args.val_ratio)

    train_rows = rows[:train_end]
    val_rows = rows[train_end:val_end]
    test_rows = rows[val_end:]

    out = Path(args.output_dir)
    write_jsonl(train_rows, out / "train.jsonl")
    write_jsonl(val_rows, out / "val.jsonl")
    write_jsonl(test_rows, out / "test.jsonl")

    meta = {
        "source": args.input_csv,
        "total_rows": n,
        "train": len(train_rows),
        "val": len(val_rows),
        "test": len(test_rows),
        "notes": "Dataset sourced from public Kaggle Ecommerce FAQ dataset represented by Ecommerce_FAQs.csv.",
    }
    with (out / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved processed dataset to: {out.resolve()}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
