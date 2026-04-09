#!/usr/bin/env python3
"""
Convert task-style JSON (messages with structured tool-call / tool-result parts)
into ms-swift Agent JSONL: each line is {"tools": "<json string>", "messages": [...]}
with roles user | assistant | tool_call | tool_response, plus optional system from
system_prompt.md.

Example:
  python scripts/convert_task_to_swift_jsonl.py \\
    --input task3.json \\
    --system-prompt system_prompt.md \\
    --output data/swift_agent.jsonl

  # Same run, plus a human-readable JSON file (indented; tools expanded from string):
  python scripts/convert_task_to_swift_jsonl.py \\
    --input task3.json \\
    --system-prompt system_prompt.md \\
    --output data/swift_agent.jsonl \\
    --pretty-output data/swift_agent.preview.json

  python scripts/convert_task_to_swift_jsonl.py \\
    --input-dir ./raw_samples/ \\
    --system-prompt system_prompt.md \\
    --output data/swift_agent.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator


# OpenAI-style tool list; ms-swift expects `tools` as a JSON *string* on each row.
DEFAULT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "addOperator",
            "description": (
                "Add a node (operator) to the dataflow DAG. Each operator performs "
                "one step; connect operators with addLink."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operatorId": {
                        "type": "string",
                        "description": "Unique id for this operator in the workflow.",
                    },
                    "operatorType": {
                        "type": "string",
                        "description": (
                            "Operator kind, e.g. CSVFileScan, Sort, Limit, Projection, "
                            "Aggregate, Filter, etc."
                        ),
                    },
                    "properties": {
                        "type": "object",
                        "description": "Operator-specific configuration (see system prompt).",
                    },
                    "summary": {
                        "type": "string",
                        "description": "Short human-readable description of this step.",
                    },
                },
                "required": ["operatorId", "operatorType", "properties"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "addLink",
            "description": "Add a directed edge between two operators (data dependency).",
            "parameters": {
                "type": "object",
                "properties": {
                    "sourceOperatorId": {"type": "string"},
                    "sourcePortIndex": {"type": "integer"},
                    "targetOperatorId": {"type": "string"},
                    "targetPortIndex": {"type": "integer"},
                },
                "required": [
                    "sourceOperatorId",
                    "sourcePortIndex",
                    "targetOperatorId",
                    "targetPortIndex",
                ],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "executeOperator",
            "description": (
                "Execute an operator and return its output table (text preview / stats)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operatorId": {
                        "type": "string",
                        "description": "Id of the operator to run.",
                    },
                },
                "required": ["operatorId"],
            },
        },
    },
]


def _normalize_user_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text", "")))
            else:
                parts.append(json.dumps(p, ensure_ascii=False))
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False)


def _iter_parts(content: Any) -> Iterator[dict[str, Any]]:
    if isinstance(content, str):
        if content.strip():
            yield {"type": "text", "text": content}
        return
    if not isinstance(content, list):
        yield {"type": "text", "text": json.dumps(content, ensure_ascii=False)}
        return
    for p in content:
        if isinstance(p, dict):
            yield p


def convert_messages(
    raw_messages: list[dict[str, Any]],
    *,
    system_text: str | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system_text is not None and system_text.strip():
        out.append({"role": "system", "content": system_text.strip()})

    i = 0
    while i < len(raw_messages):
        msg = raw_messages[i]
        role = msg.get("role")
        content = msg.get("content")

        if role == "user":
            out.append({"role": "user", "content": _normalize_user_content(content)})
            i += 1
            continue

        if role == "assistant":
            text_buf: list[str] = []
            for part in _iter_parts(content):
                if part.get("type") == "text":
                    t = part.get("text")
                    if t:
                        text_buf.append(str(t))
                elif part.get("type") == "tool-call":
                    if text_buf:
                        out.append({"role": "assistant", "content": "\n".join(text_buf)})
                        text_buf = []
                    name = part.get("toolName") or part.get("name")
                    args = part.get("input")
                    if args is None:
                        args = part.get("arguments", {})
                    tc = json.dumps(
                        {"name": name, "arguments": args},
                        ensure_ascii=False,
                    )
                    out.append({"role": "tool_call", "content": tc})
                else:
                    text_buf.append(json.dumps(part, ensure_ascii=False))
            if text_buf:
                out.append({"role": "assistant", "content": "\n".join(text_buf)})
            i += 1
            continue

        if role == "tool":
            for part in _iter_parts(content):
                if isinstance(part, dict) and part.get("type") == "tool-result":
                    payload = part.get("output", part)
                    out.append(
                        {
                            "role": "tool_response",
                            "content": json.dumps(payload, ensure_ascii=False),
                        }
                    )
                else:
                    out.append(
                        {
                            "role": "tool_response",
                            "content": json.dumps(part, ensure_ascii=False),
                        }
                    )
            i += 1
            continue

        # Fallback: pass through as user text (unknown role)
        out.append(
            {
                "role": "user",
                "content": f"[unknown role={role}] "
                + _normalize_user_content(content),
            }
        )
        i += 1

    return out


def load_task_record(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        if len(data) != 1:
            raise ValueError(
                f"{path}: JSON array must contain exactly one sample, got {len(data)}"
            )
        data = data[0]
    if not isinstance(data, dict):
        raise ValueError(f"{path}: root must be object or single-element array")
    if "messages" not in data:
        raise ValueError(f"{path}: missing 'messages'")
    return data


def record_to_swift_line(
    record: dict[str, Any],
    *,
    system_text: str | None,
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    messages = convert_messages(record["messages"], system_text=system_text)
    return {
        "tools": json.dumps(tools, ensure_ascii=False),
        "messages": messages,
    }


def record_for_preview(record: dict[str, Any]) -> dict[str, Any]:
    """Same sample as training record, but tools is a list (not a string) for readability."""
    out = dict(record)
    raw = out.get("tools")
    if isinstance(raw, str):
        out["tools"] = json.loads(raw)
    return out


def collect_input_files(
    input_path: Path | None,
    input_dir: Path | None,
) -> list[Path]:
    files: list[Path] = []
    if input_path:
        files.append(input_path)
    if input_dir:
        for p in sorted(input_dir.glob("*.json")):
            if p.is_file():
                files.append(p)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert task JSON to ms-swift Agent JSONL (tools + tool_call/tool_response)."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        help="Single task JSON file (one sample per file).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Directory of *.json task files (one sample per file).",
    )
    parser.add_argument(
        "--system-prompt",
        type=Path,
        default=Path("system_prompt.md"),
        help="Markdown/text prepended as role=system (default: ./system_prompt.md).",
    )
    parser.add_argument(
        "--tools-json",
        type=Path,
        help="Optional JSON file: list of OpenAI-style tools; overrides built-in addOperator/addLink/executeOperator.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output path: compact JSONL (one JSON object per line, for training).",
    )
    parser.add_argument(
        "--pretty-output",
        type=Path,
        help=(
            "Optional second file: JSON array of samples with indent=2 for reading in an "
            "editor; `tools` is expanded to a real JSON array (preview only, not for swift)."
        ),
    )
    parser.add_argument(
        "--no-system",
        action="store_true",
        help="Do not add system message (ignore --system-prompt).",
    )
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.error("Provide --input and/or --input-dir")
    if args.input and not args.input.is_file():
        parser.error(f"--input not a file: {args.input}")

    tools = DEFAULT_TOOLS
    if args.tools_json:
        tools = json.loads(args.tools_json.read_text(encoding="utf-8"))
        if not isinstance(tools, list):
            sys.exit("--tools-json must contain a JSON array")

    system_text: str | None = None
    if not args.no_system:
        sp = args.system_prompt
        if not sp.is_file():
            sys.exit(f"system prompt file not found: {sp}")
        system_text = sp.read_text(encoding="utf-8")

    in_files = collect_input_files(args.input, args.input_dir)
    if not in_files:
        sys.exit("No input JSON files found")

    rows: list[dict[str, Any]] = []
    for fp in in_files:
        rec = load_task_record(fp)
        rows.append(record_to_swift_line(rec, system_text=system_text, tools=tools))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fout:
        for line in rows:
            fout.write(json.dumps(line, ensure_ascii=False) + "\n")

    n = len(rows)
    print(f"Wrote {n} sample(s) to {args.output}", file=sys.stderr)

    if args.pretty_output:
        args.pretty_output.parent.mkdir(parents=True, exist_ok=True)
        preview = [record_for_preview(r) for r in rows]
        args.pretty_output.write_text(
            json.dumps(preview, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote pretty preview ({n} sample(s)) to {args.pretty_output}", file=sys.stderr)


if __name__ == "__main__":
    main()
