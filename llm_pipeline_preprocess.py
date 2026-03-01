"""Preprocess dataset into instruction‑format for LLM fine-tuning.

This module is designed to take a simple Q/A CSV (originally sourced from a
public Kaggle/UCL dataset) and clean it up for supervised fine-tuning with
LoRA/QLoRA. The output is a train/val/test split in JSONL where each record
contains both a structured ``instruction``/``input``/``output`` tuple and a
concatenated ``text`` field suitable for feeding directly to ``trl``.

Example usage::

    python llm_pipeline_preprocess.py --input_csv Ecommerce_FAQs.csv \
        --output_dir training_data

The default CSV is assumed to have columns named "Question" and "Answer" but
other column names are detected case-insensitively.
"""

import argparse  # parse CLI arguments
import json  # read/write JSON
import random  # randomness for shuffling
import re  # regular expressions for cleaning
from pathlib import Path  # filesystem path utilities

import pandas as pd  # dataframes for CSV processing


def clean_text(text: str) -> str:  # normalize and collapse whitespace
    text = str(text).strip()  # ensure string and trim surrounding whitespace
    text = re.sub(r"\s+", " ", text)  # collapse all whitespace runs to single space
    return text  # return cleaned string


def to_instruction_row(question: str, answer: str) -> dict:  # format a single example
    return {
        "instruction": "Answer the e-commerce customer question factually and concisely.",  # fixed task instruction
        "input": question,  # the user's question
        "output": answer,  # the ground-truth answer
        "text": (  # concatenated prompt+response text for TRL/processing
            "### Instruction:\n"
            "Answer the e-commerce customer question factually and concisely.\n\n"
            f"### Input:\n{question}\n\n"
            f"### Response:\n{answer}"
        ),
    }


def write_jsonl(rows: list[dict], path: Path) -> None:  # persist rows as JSONL
    path.parent.mkdir(parents=True, exist_ok=True)  # ensure output directory exists
    with path.open("w", encoding="utf-8") as f:  # open file for writing
        for row in rows:  # iterate examples
            f.write(json.dumps(row, ensure_ascii=False) + "\n")  # write JSON per line


def main() -> None:  # CLI entrypoint
    parser = argparse.ArgumentParser(description="Preprocess Ecommerce FAQ data for LoRA/QLoRA fine-tuning.")  # build argparser
    parser.add_argument("--input_csv", default="Ecommerce_FAQs.csv",
                        help="CSV file path containing question/answer pairs.")  # local CSV path
    parser.add_argument("--hf_dataset", default=None,
                        help="Optional Hugging Face dataset name to download instead of reading a CSV.")  # HF dataset id
    parser.add_argument("--hf_split", default="train",
                        help="Split name to use when loading an HF dataset (default: train).")  # HF split
    parser.add_argument("--output_dir", default="training_data")  # where to save JSONL
    parser.add_argument("--seed", type=int, default=42)  # RNG seed for reproducibility
    parser.add_argument("--train_ratio", type=float, default=0.8)  # fraction for train
    parser.add_argument("--val_ratio", type=float, default=0.1)  # fraction for validation
    args = parser.parse_args()  # parse CLI args

    random.seed(args.seed)  # seed python RNG

    if args.hf_dataset:  # optionally load directly from HF hub
        # load directly from Hugging Face hub if requested
        from datasets import load_dataset  # local import to avoid top-level dependency

        ds = load_dataset(args.hf_dataset)  # fetch dataset dict of splits
        if args.hf_split not in ds:  # validate split exists
            raise ValueError(f"split '{args.hf_split}' not found in dataset {args.hf_dataset}")
        df = pd.DataFrame(ds[args.hf_split])  # convert HF split to pandas DataFrame
    else:
        df = pd.read_csv(args.input_csv)  # read local CSV into DataFrame

    # choose column names depending on dataset format; the original
    # e-commerce FAQ CSV uses "Question"/"Answer", whereas the newer
    # HuggingFace "Bitext" dataset exposes "instruction"/"response" along
    # with extra metadata fields.  we detect either format automatically and
    # fall back to the first two columns if nothing matches.
    lower_cols = {c.lower(): c for c in df.columns}  # map lowercase name -> original

    if "instruction" in lower_cols and "response" in lower_cols:  # HF Bitext format
        q_col = lower_cols["instruction"]  # input column name
        a_col = lower_cols["response"]  # output column name
    else:
        q_col = lower_cols.get("question", df.columns[0])  # fallback to first col
        a_col = lower_cols.get("answer", df.columns[1])  # fallback to second col

    # Normalize column names and strip whitespace/duplicates
    df = df[[q_col, a_col]].rename(columns={q_col: "question", a_col: "answer"})  # keep only two cols
    df["question"] = df["question"].map(clean_text)  # clean questions
    df["answer"] = df["answer"].map(clean_text)  # clean answers
    # remove empty entries and repeated questions which could bias training
    df = df[(df["question"] != "") & (df["answer"] != "")]  # drop empty rows
    df = df.drop_duplicates(subset=["question"]).reset_index(drop=True)  # deduplicate

    # convert to instruction format expected by transformers/trl
    rows = [to_instruction_row(q, a) for q, a in zip(df["question"], df["answer"]) ]  # build examples
    random.shuffle(rows)  # shuffle before splitting to avoid ordering artifacts

    n = len(rows)  # total examples
    train_end = int(n * args.train_ratio)  # index where train ends
    val_end = train_end + int(n * args.val_ratio)  # index where val ends

    train_rows = rows[:train_end]  # slice train
    val_rows = rows[train_end:val_end]  # slice val
    test_rows = rows[val_end:]  # slice test

    out = Path(args.output_dir)  # output directory Path
    write_jsonl(train_rows, out / "train.jsonl")  # write train
    write_jsonl(val_rows, out / "val.jsonl")  # write val
    write_jsonl(test_rows, out / "test.jsonl")  # write test

    # metadata file contains provenance info for reproducibility
    # record dataset provenance in metadata
    notes = ""  # notes placeholder
    if args.hf_dataset:
        notes = f"Dataset sourced from Hugging Face dataset {args.hf_dataset}."  # HF provenance
    else:
        notes = "Dataset sourced from public Kaggle Ecommerce FAQ dataset (saadmakhdoom/ecommerce-faq-chatbot-dataset)."  # Kaggle provenance

    meta = {
        "source": args.hf_dataset if args.hf_dataset else args.input_csv,  # source identifier
        "total_rows": n,  # total rows processed
        "train": len(train_rows),  # count train
        "val": len(val_rows),  # count val
        "test": len(test_rows),  # count test
        "notes": notes,  # provenance notes
    }
    with (out / "metadata.json").open("w", encoding="utf-8") as f:  # write metadata
        json.dump(meta, f, indent=2)

    print(f"Saved processed dataset to: {out.resolve()}")  # human-friendly path
    print(json.dumps(meta, indent=2))  # pretty-print metadata


if __name__ == "__main__":
    main()
