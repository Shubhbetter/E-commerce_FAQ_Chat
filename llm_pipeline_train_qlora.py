"""Deprecated training entrypoint; use ``llm_pipeline_train.py`` instead.

This module previously implemented QLoRA-specific logic but has been
superseded by ``llm_pipeline_train.py`` which supports both LoRA and QLoRA and
is more fully documented.  The file remains here for backward compatibility
and will print a helpful message when executed.
"""

import argparse
import sys

print("WARNING: llm_pipeline_train_qlora.py is deprecated."
      " Please switch to llm_pipeline_train.py which supports both lora and qlora.")

# The original implementation is available in git history if you really need it.
sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for e-commerce FAQ model.")
    parser.add_argument("--model_name", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--data_dir", default="training_data")
    parser.add_argument("--output_dir", default="artifacts/tinyllama-ecom-qlora")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "val.jsonl"),
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_seq_length=512,
        packing=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        processing_class=tokenizer,
        formatting_func=lambda x: x["text"],
    )

    trainer.train()
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Saved QLoRA adapter + tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
