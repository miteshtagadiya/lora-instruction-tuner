# LoRA Instruction Tuner

Small, honest demo of an instruction fine-tune loop with **LoRA/PEFT**.

Built for portfolio / smoke testing: tiny model, tiny JSONL dataset, train + eval scripts, sample metrics artifact.

## Why

Hiring managers for AI training / LLM product roles usually want to see:

1. Dataset format
2. Train config
3. Adapter training path
4. Offline eval after train

This repo is that loop, kept small so anyone can run it.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Validate data + write stub metrics (no GPU needed)
python train.py --dry-run

# Optional: real tiny LoRA steps (downloads tiny-gpt2)
python train.py

# Score eval set (gold-echo demo if no predictions file yet)
python eval.py
```

## Layout

```
configs/default.yaml   # model, LoRA ranks, steps
data/train.jsonl       # instruction examples
data/eval.jsonl
train.py               # PEFT LoRA trainer (+ --dry-run)
eval.py                # exact match + token overlap report
outputs/               # metrics / adapters
```

## Sample metrics (dry-run)

See `outputs/tiny-lora/metrics.json`.

## Notes

- Default model is `sshleifer/tiny-gpt2` on purpose (CI-friendly). Swap in config for real instruct models.
- Not a fork of any training framework. Original glue code around HF Transformers + PEFT.

## License

MIT
