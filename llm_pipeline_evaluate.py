import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_prompt(question: str) -> str:
    return (
        "### Instruction:\n"
        "Answer the e-commerce customer question factually and concisely.\n\n"
        f"### Input:\n{question}\n\n"
        "### Response:\n"
    )


def hallucination_proxy(pred: str, ref: str) -> float:
    pred_tokens = set(pred.lower().split())
    ref_tokens = set(ref.lower().split())
    if not pred_tokens:
        return 1.0
    unsupported = [t for t in pred_tokens if t not in ref_tokens]
    return len(unsupported) / len(pred_tokens)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate factuality/relevance of fine-tuned model.")
    parser.add_argument("--base_model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--adapter_path", default="artifacts/tinyllama-ecom-qlora")
    parser.add_argument("--test_file", default="training_data/test.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    args = parser.parse_args()

    ds = load_dataset("json", data_files={"test": args.test_file})["test"]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto")
    model = PeftModel.from_pretrained(base, args.adapter_path)

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_ls, hallucinations = [], []

    for row in ds:
        prompt = build_prompt(row["input"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        pred = text.split("### Response:\n")[-1].strip()

        ref = row["output"].strip()
        rouge_ls.append(scorer.score(ref, pred)["rougeL"].fmeasure)
        hallucinations.append(hallucination_proxy(pred, ref))

    report = {
        "samples": len(ds),
        "rougeL_f1": float(np.mean(rouge_ls)) if rouge_ls else 0.0,
        "hallucination_proxy": float(np.mean(hallucinations)) if hallucinations else 1.0,
    }
    Path("artifacts").mkdir(exist_ok=True)
    with open("artifacts/eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
