"""Simple inference helper showing how to load a fine-tuned adapter.

After training and saving an adapter directory (see ``llm_pipeline_train.py``),
this script can be used to generate answers from the resulting model.  It uses
PEFT's ``PeftModel`` wrapper to merge the adapter with the base network at
runtime.

Example::

    python run_inference.py \
        --adapter artifacts/tinyllama-ecom-qlora \
        --base_model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --question "How long does shipping take?"

"""

import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def build_prompt(question: str) -> str:
    return (
        "### Instruction:\n"
        "Answer the e-commerce customer question factually and concisely.\n\n"
        f"### Input:\n{question}\n\n"
        "### Response:\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Load adapter and answer a question")
    parser.add_argument("--adapter", required=True, help="Directory with saved adapter")
    parser.add_argument("--base_model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    # load tokenizer and base model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(args.base_model, device_map="auto")
    model = PeftModel.from_pretrained(base, args.adapter)

    prompt = build_prompt(args.question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = text.split("### Response:\n")[-1].strip()
    print("Question:", args.question)
    print("Answer:", answer)


if __name__ == "__main__":
    main()
