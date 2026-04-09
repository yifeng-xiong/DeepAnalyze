#!/usr/bin/env python3
"""
Preview how Agent JSONL rows are transformed before / as they enter the LM during
ms-swift SFT: same path as training uses (StdTemplateInputs.from_dict → preprocess
tool_call → encode with loss labels).

Requires a local model directory (tokenizer + swift model_type metadata) or HF hub id.

Examples:
  # Match scripts/sft_swift_agent_full.sh defaults where possible
  python scripts/preview_swift_agent_dataset.py \\
    --jsonl data/swift_agent_biomedical-easy-6.jsonl \\
    --model /path/to/DeepAnalyze-8B \\
    --model-type deepseek_r1_distill \\
    --agent-template react_en \\
    --loss-scale react \\
    --output data/swift_agent_biomedical-easy-6.swift_preview.txt

  # First row only, truncate long message bodies in the report
  python scripts/preview_swift_agent_dataset.py \\
    --jsonl data/swift_agent_biomedical-easy-6.jsonl \\
    --model Qwen/Qwen2.5-7B-Instruct --use-hf \\
    --max-rows 1 --max-content-chars 1500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SWIFT_ROOT = Path(os.environ.get("SWIFT_ROOT", ROOT / "deepanalyze" / "ms-swift"))


def _truncate(s: str, max_chars: int) -> str:
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    return s[:max_chars] + f"\n... [{len(s) - max_chars} more chars truncated]"


def _messages_for_print(messages: list[dict[str, Any]], max_content_chars: int) -> str:
    out: list[dict[str, Any]] = []
    for m in messages:
        row = {"role": m.get("role"), "content": m.get("content")}
        c = row["content"]
        if isinstance(c, str) and max_content_chars:
            row["content"] = _truncate(c, max_content_chars)
        out.append(row)
    return json.dumps(out, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show ms-swift Template.encode output for Agent JSONL (training view)."
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        required=True,
        help="Path to Agent JSONL (tools string + messages).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model dir or hub id (e.g. DeepAnalyze-8B path or Qwen/Qwen2.5-7B-Instruct).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        help="swift model_type if not inferred from config (e.g. deepseek_r1_distill).",
    )
    parser.add_argument(
        "--use-hf",
        action="store_true",
        help="Load --model from Hugging Face Hub instead of local path.",
    )
    parser.add_argument(
        "--agent-template",
        type=str,
        default="react_en",
        help="Same as swift sft --agent_template (default: react_en).",
    )
    parser.add_argument(
        "--loss-scale",
        type=str,
        default="react",
        help="Same as swift sft --loss_scale (default: react).",
    )
    parser.add_argument(
        "--response-prefix",
        type=str,
        default="",
        help="Same as swift --response_prefix (default: empty).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=262144,
        help="Template max_length for preview (avoid truncation on long traces).",
    )
    parser.add_argument(
        "--truncation-strategy",
        choices=("raise", "left", "right"),
        default="right",
        help="If sequence exceeds max_length (default: right).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Only process first N rows (0 = all).",
    )
    parser.add_argument(
        "--indices",
        type=str,
        default="",
        help="Comma-separated 0-based row indices to print (overrides --max-rows if set).",
    )
    parser.add_argument(
        "--max-content-chars",
        type=int,
        default=4000,
        help="Truncate each message content in the 'After preprocess' JSON dump (0 = no limit).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Write report to this file (default: stdout).",
    )
    args = parser.parse_args()

    if not SWIFT_ROOT.is_dir():
        sys.exit(f"ms-swift not found at {SWIFT_ROOT}. Set SWIFT_ROOT or clone ms-swift there.")

    sys.path.insert(0, str(SWIFT_ROOT))

    from swift.llm import get_model_tokenizer, get_template

    if not args.jsonl.is_file():
        sys.exit(f"JSONL not found: {args.jsonl}")

    rows: list[dict[str, Any]] = []
    with args.jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if args.indices.strip():
        want = {int(x.strip()) for x in args.indices.split(",") if x.strip() != ""}
        selected = [(i, rows[i]) for i in sorted(want) if 0 <= i < len(rows)]
    elif args.max_rows > 0:
        selected = list(enumerate(rows[: args.max_rows]))
    else:
        selected = list(enumerate(rows))

    if not selected:
        sys.exit("No rows selected.")

    _, processor = get_model_tokenizer(
        args.model,
        load_model=False,
        model_type=args.model_type,
        use_hf=args.use_hf,
    )
    tokenizer = processor
    template = get_template(
        tokenizer.model_meta.template,
        tokenizer,
        max_length=args.max_length,
        truncation_strategy=args.truncation_strategy,
        agent_template=args.agent_template,
        loss_scale=args.loss_scale,
        response_prefix=args.response_prefix or None,
    )
    template.set_mode("train")

    lines_out: list[str] = []
    lines_out.append("swift Agent dataset preview (Template.encode train mode)")
    lines_out.append(f"jsonl: {args.jsonl}")
    lines_out.append(f"model: {args.model}  model_type: {args.model_type!r}  use_hf: {args.use_hf}")
    lines_out.append(
        f"swift template: {tokenizer.model_meta.template}  "
        f"agent_template: {args.agent_template}  loss_scale: {args.loss_scale}"
    )
    lines_out.append(
        "Note: tool_call rows become assistant text (e.g. Action / Action Input / Observation); "
        "tool_response becomes role tool, then merged per agent_template."
    )
    lines_out.append("")

    for idx, row in selected:
        lines_out.append("=" * 80)
        lines_out.append(f"ROW {idx}")
        lines_out.append("=" * 80)
        try:
            encoded = template.encode(dict(row), return_template_inputs=True)
        except Exception as e:
            lines_out.append(f"ENCODE ERROR: {e!r}")
            lines_out.append("")
            continue

        ti = encoded.pop("template_inputs", None)
        if ti is not None:
            lines_out.append("--- After preprocess (messages fed to template encoder) ---")
            lines_out.append(f"system (separate field): {_truncate(repr(ti.system), args.max_content_chars)}")
            lines_out.append(
                _messages_for_print(ti.messages, args.max_content_chars)
            )
            lines_out.append("")

        input_ids = encoded.get("input_ids")
        labels = encoded.get("labels")
        if input_ids is not None:
            lines_out.append("--- Decoded input_ids (full training sequence) ---")
            lines_out.append(template.safe_decode(list(input_ids)))
            lines_out.append("")
        if labels is not None:
            lines_out.append("--- Decoded labels (masked spans use token id -100 in swift) ---")
            lines_out.append(template.safe_decode(list(labels)))
            lines_out.append("")

        lines_out.append(f"length: {encoded.get('length')!r}  keys: {sorted(encoded.keys())}")
        lines_out.append("")

    text = "\n".join(lines_out)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
