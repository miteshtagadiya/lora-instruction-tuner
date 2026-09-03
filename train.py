#!/usr/bin/env python3
"""Minimal LoRA instruction fine-tune smoke pipeline.

Uses a tiny HF model so the script is runnable without a GPU farm.
For real training, swap model_name to a larger instruct checkpoint.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_example(row: dict) -> str:
    instruction = row.get("instruction", "").strip()
    inp = row.get("input", "").strip()
    output = row.get("output", "").strip()
    if inp:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
    return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA instruction tuner (smoke / demo)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip torch/PEFT; only validate data + write a stub metrics file",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    train = load_jsonl(Path(cfg["train_file"]))
    eval_rows = load_jsonl(Path(cfg["eval_file"]))
    random.seed(cfg.get("seed", 42))

    print(f"train examples: {len(train)}")
    print(f"eval examples: {len(eval_rows)}")
    print(f"sample prompt:\n{format_example(train[0])[:400]}\n...")

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        metrics = {
            "mode": "dry-run",
            "train_size": len(train),
            "eval_size": len(eval_rows),
            "model_name": cfg["model_name"],
            "note": "Install requirements and re-run without --dry-run for a real tiny LoRA step.",
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"wrote {out_dir / 'metrics.json'}")
        return

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(batch):
        texts = [
            format_example({"instruction": i, "input": x, "output": o})
            for i, x, o in zip(batch["instruction"], batch["input"], batch["output"])
        ]
        return tokenizer(
            texts,
            truncation=True,
            max_length=cfg["max_length"],
            padding="max_length",
        )

    ds = Dataset.from_list(train).map(tokenize, batched=True, remove_columns=list(train[0].keys()))
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"])
    lora = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        max_steps=cfg["max_steps"],
        learning_rate=cfg["learning_rate"],
        per_device_train_batch_size=2,
        logging_steps=5,
        save_steps=cfg["max_steps"],
        report_to=[],
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    result = trainer.train()
    trainer.save_model(str(out_dir / "adapter"))
    metrics = {
        "mode": "train",
        "train_loss": float(result.training_loss),
        "train_size": len(train),
        "eval_size": len(eval_rows),
        "model_name": cfg["model_name"],
        "max_steps": cfg["max_steps"],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
