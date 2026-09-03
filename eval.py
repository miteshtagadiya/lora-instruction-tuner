#!/usr/bin/env python3
"""Simple exact-match / token-overlap eval against held-out JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def tokenize(s: str) -> set[str]:
    return set(s.lower().split())


def overlap(a: str, b: str) -> float:
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", default="data/eval.jsonl")
    parser.add_argument(
        "--predictions",
        default="outputs/tiny-lora/predictions.jsonl",
        help="JSONL with fields: instruction, prediction (optional; falls back to gold echo demo)",
    )
    parser.add_argument("--out", default="outputs/tiny-lora/eval_report.json")
    args = parser.parse_args()

    gold = [json.loads(l) for l in Path(args.eval_file).read_text().splitlines() if l.strip()]
    pred_path = Path(args.predictions)
    if pred_path.exists():
        preds = {json.loads(l)["instruction"]: json.loads(l)["prediction"] for l in pred_path.read_text().splitlines() if l.strip()}
    else:
        # Demo mode: score gold against itself so the pipeline is runnable offline
        preds = {row["instruction"]: row["output"] for row in gold}
        print("no predictions file; running gold-echo demo scoring")

    exact = 0
    overlaps = []
    rows = []
    for row in gold:
        p = preds.get(row["instruction"], "")
        em = int(p.strip().lower() == row["output"].strip().lower())
        ov = overlap(p, row["output"])
        exact += em
        overlaps.append(ov)
        rows.append({"instruction": row["instruction"], "exact_match": em, "token_overlap": ov})

    report = {
        "n": len(gold),
        "exact_match": exact / max(len(gold), 1),
        "mean_token_overlap": sum(overlaps) / max(len(overlaps), 1),
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("n", "exact_match", "mean_token_overlap")}, indent=2))


if __name__ == "__main__":
    main()
