#!/usr/bin/env python3
"""
Convert LocalLLM `react_steps.json` (or legacy task JSON) into ms-swift Agent JSONL:
each line is {"tools": "<json string>", "messages": [...]} with roles
system | user | assistant | tool_call. Tool / observation turns are omitted (no tool_response).

Default: scan data/LocalLLM/**/react_steps.json — one row per agent step with non-empty
toolCalls, plus (by default) agent steps with only a text reply (e.g. **Final Answer:**).

Examples:
  python scripts/convert_task_to_swift_jsonl.py \\
    --output data/swift_agent.jsonl

  python scripts/convert_task_to_swift_jsonl.py \\
    --input data/LocalLLM/biomedical-easy-6/react_steps.json \\
    --output data/swift_agent.jsonl \\
    --pretty-output data/swift_agent.preview.json

  # Legacy: single task file with messages + tool-call parts (one sample per file):
  python scripts/convert_task_to_swift_jsonl.py \\
    --format legacy-task \\
    --input task3.json \\
    --output data/swift_agent.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent.parent
THOUGHT_PREFIX_RE = re.compile(r"(?i)^thought:\s*")
TURN_HEADING_RE = re.compile(r"^### Turn \d+\s*$")
DUPLICATE_THOUGHT_PREFIX_RE = re.compile(
    r"(?im)^(\s*Thought:\s*)(?:Thought:\s*)+"
)


def load_tools_from_file(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and "tools" in raw:
        tools = raw["tools"]
        if isinstance(tools, list):
            return tools
    sys.exit(f"{path}: expected JSON array or object with \"tools\" array")


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


def _collapse_blank_lines(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _dedupe_duplicate_thought_prefixes(text: str) -> str:
    """Collapse accidental `Thought: Thought: ...` into one prefix."""
    return DUPLICATE_THOUGHT_PREFIX_RE.sub(r"\1", text)


def _strip_leading_thought_from_assistant_text(text: str) -> str:
    cleaned = _dedupe_duplicate_thought_prefixes(text).strip()
    while cleaned:
        if not THOUGHT_PREFIX_RE.match(cleaned):
            break
        cleaned = THOUGHT_PREFIX_RE.sub("", cleaned, count=1).lstrip()
        parts = re.split(r"\n\s*\n", cleaned, maxsplit=1)
        if len(parts) == 2 and parts[1].strip():
            cleaned = parts[1].lstrip()
        else:
            break
    return _collapse_blank_lines(cleaned)


def _clean_embedded_history_text(text: str) -> str:
    lines = _dedupe_duplicate_thought_prefixes(text).splitlines()
    cleaned_lines: list[str] = []
    skipping_thought_block = False

    for line in lines:
        stripped = line.lstrip()

        if stripped.startswith("Above is user's request and the steps you already took."):
            continue
        if stripped.startswith(
            "You as an assistant please keep working on solving user's request"
        ):
            continue

        if THOUGHT_PREFIX_RE.match(stripped):
            skipping_thought_block = True
            continue

        if skipping_thought_block:
            if (
                stripped.startswith("- ")
                or stripped.startswith("#")
                or stripped.startswith("Result:")
                or stripped.startswith("Properties:")
                or stripped.startswith("Summary:")
                or stripped.startswith("code:")
            ):
                skipping_thought_block = False
            elif stripped == "":
                continue
            else:
                continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{2,}(?=### Turn \d+\s*(?:\n|$))", "\n", text)
    text = re.sub(
        r"(?ms)^### Turn \d+\s*\n(?=(?:### Turn \d+|# Current Dataflow|$))",
        "",
        text,
    )
    return _collapse_blank_lines(text)


def _clean_user_content(content: Any, *, strip_thought: bool) -> str:
    text = _dedupe_duplicate_thought_prefixes(_normalize_user_content(content))
    if not strip_thought:
        return _finalize_user_message_content(text)
    return _finalize_user_message_content(_clean_embedded_history_text(text))


def _clean_assistant_content(content: Any, *, strip_thought: bool) -> str | None:
    text = _dedupe_duplicate_thought_prefixes(_normalize_user_content(content)).strip()
    if not text:
        return None
    if strip_thought:
        text = _strip_leading_thought_from_assistant_text(text)
    return text or None


def _finalize_user_message_content(text: str) -> str:
    """Append a trailing newline to user text when missing (spacing before assistant / tool_call)."""
    if text and not text.endswith("\n"):
        return text + "\n"
    return text


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
    strip_thought: bool,
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
            out.append(
                {
                    "role": "user",
                    "content": _clean_user_content(content, strip_thought=strip_thought),
                }
            )
            i += 1
            continue

        if role == "assistant":
            text_buf: list[str] = []
            for part in _iter_parts(content):
                if part.get("type") == "text":
                    t = part.get("text")
                    if t:
                        cleaned = _clean_assistant_content(
                            str(t), strip_thought=strip_thought
                        )
                        if cleaned:
                            text_buf.append(cleaned)
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
            # Omit tool results from training JSONL (tool_call-only traces).
            i += 1
            continue

        out.append(
            {
                "role": "user",
                "content": _finalize_user_message_content(
                    f"[unknown role={role}] " + _normalize_user_content(content)
                ),
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
    strip_thought: bool,
) -> dict[str, Any]:
    messages = convert_messages(
        record["messages"],
        system_text=system_text,
        strip_thought=strip_thought,
    )
    return {
        "tools": json.dumps(tools, ensure_ascii=False),
        "messages": messages,
    }


def record_for_preview(record: dict[str, Any]) -> dict[str, Any]:
    out = dict(record)
    raw = out.get("tools")
    if isinstance(raw, str):
        out["tools"] = json.loads(raw)
    return out


def collect_react_steps_files(
    *,
    input_path: Path | None,
    local_llm_dir: Path,
) -> list[Path]:
    files: list[Path] = []
    if input_path:
        files.append(input_path)
    else:
        if not local_llm_dir.is_dir():
            return []
        for p in sorted(local_llm_dir.rglob("react_steps.json")):
            if p.is_file():
                files.append(p)
    return files


def _user_content_from_input_messages(
    input_messages: Any, *, strip_thought: bool
) -> str | None:
    if not isinstance(input_messages, list):
        return None
    user_parts: list[str] = []
    for m in input_messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if c is None:
            continue
        user_parts.append(_clean_user_content(c, strip_thought=strip_thought).rstrip())
    if not user_parts:
        return None
    return _finalize_user_message_content("\n\n".join(user_parts))


def react_step_to_swift_line(
    step: dict[str, Any],
    *,
    system_text: str | None,
    tools: list[dict[str, Any]],
    source_path: Path,
    step_index: int,
    strip_thought: bool,
) -> dict[str, Any] | None:
    tool_calls = step.get("toolCalls") or []
    if not isinstance(tool_calls, list) or not tool_calls:
        return None

    user_text = _user_content_from_input_messages(
        step.get("inputMessages"), strip_thought=strip_thought
    )
    if user_text is None:
        print(
            f"skip {source_path} step[{step_index}]: no user content in inputMessages",
            file=sys.stderr,
        )
        return None

    messages: list[dict[str, Any]] = []
    if system_text is not None and system_text.strip():
        messages.append({"role": "system", "content": system_text.strip()})
    messages.append({"role": "user", "content": user_text})

    assistant_text = _clean_assistant_content(
        step.get("content"), strip_thought=strip_thought
    )
    if assistant_text:
        messages.append({"role": "assistant", "content": assistant_text})

    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        name = tc.get("toolName") or tc.get("name")
        args = tc.get("input")
        if args is None:
            args = tc.get("arguments", {})
        if not name:
            print(
                f"skip {source_path} step[{step_index}]: tool call missing toolName",
                file=sys.stderr,
            )
            return None
        messages.append(
            {
                "role": "tool_call",
                "content": json.dumps(
                    {"name": name, "arguments": args},
                    ensure_ascii=False,
                ),
            }
        )

    return {
        "tools": json.dumps(tools, ensure_ascii=False),
        "messages": messages,
    }


def react_final_assistant_step_to_swift_line(
    step: dict[str, Any],
    *,
    system_text: str | None,
    tools: list[dict[str, Any]],
    source_path: Path,
    step_index: int,
    strip_thought: bool,
) -> dict[str, Any] | None:
    """Agent turn with no tool calls but non-empty assistant text (e.g. final answer)."""
    tool_calls = step.get("toolCalls") or []
    if isinstance(tool_calls, list) and tool_calls:
        return None

    assistant_text = _clean_assistant_content(
        step.get("content"), strip_thought=strip_thought
    )
    if not assistant_text:
        return None

    user_text = _user_content_from_input_messages(
        step.get("inputMessages"), strip_thought=strip_thought
    )
    if user_text is None:
        print(
            f"skip {source_path} step[{step_index}]: final-assistant turn has no user "
            "content in inputMessages",
            file=sys.stderr,
        )
        return None

    messages: list[dict[str, Any]] = []
    if system_text is not None and system_text.strip():
        messages.append({"role": "system", "content": system_text.strip()})
    messages.append({"role": "user", "content": user_text})
    messages.append({"role": "assistant", "content": assistant_text})

    return {
        "tools": json.dumps(tools, ensure_ascii=False),
        "messages": messages,
    }


def iter_react_steps_rows(
    path: Path,
    *,
    system_text: str | None,
    tools: list[dict[str, Any]],
    include_final_assistant: bool,
    strip_thought: bool,
) -> Iterator[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        print(f"skip {path}: root must be object", file=sys.stderr)
        return
    steps = data.get("steps")
    if not isinstance(steps, list):
        print(f"skip {path}: missing steps array", file=sys.stderr)
        return
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        if step.get("role") != "agent":
            continue
        line = react_step_to_swift_line(
            step,
            system_text=system_text,
            tools=tools,
            source_path=path,
            step_index=i,
            strip_thought=strip_thought,
        )
        if line is not None:
            yield line
        elif include_final_assistant:
            final_line = react_final_assistant_step_to_swift_line(
                step,
                system_text=system_text,
                tools=tools,
                source_path=path,
                step_index=i,
                strip_thought=strip_thought,
            )
            if final_line is not None:
                yield final_line


def collect_legacy_input_files(
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
        description=(
            "Convert react_steps.json (default) or legacy task JSON to ms-swift Agent JSONL."
        )
    )
    parser.add_argument(
        "--format",
        choices=("react_steps", "legacy-task"),
        default="react_steps",
        help="Input format (default: react_steps).",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        help=(
            "react_steps: single react_steps.json. "
            "legacy-task: single task JSON (one sample per file)."
        ),
    )
    parser.add_argument(
        "--local-llm-dir",
        type=Path,
        default=ROOT / "data" / "LocalLLM",
        help="Directory to scan for **/react_steps.json (default: <repo>/data/LocalLLM).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="legacy-task only: directory of *.json task files (one sample per file).",
    )
    parser.add_argument(
        "--system-prompt",
        type=Path,
        default=ROOT / "system_prompt.md",
        help="Prepended as role=system (default: <repo>/system_prompt.md).",
    )
    parser.add_argument(
        "--tools-json",
        type=Path,
        default=ROOT / "code-tools-schema.json",
        help="OpenAI-style tools: array or {\"tools\": [...]} (default: <repo>/code-tools-schema.json).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output JSONL path (one JSON object per line).",
    )
    parser.add_argument(
        "--pretty-output",
        type=Path,
        help="Optional JSON array preview (tools expanded; not for swift training).",
    )
    parser.add_argument(
        "--no-system",
        action="store_true",
        help="Do not add system message.",
    )
    parser.add_argument(
        "--include-final-assistant",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Include agent steps with non-empty content and empty toolCalls "
            "(text-only reply, e.g. final answer). Use --no-include-final-assistant to skip."
        ),
    )
    parser.add_argument(
        "--strip-thought",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Remove leading assistant Thought: text and embedded Thought: history from "
            "react_steps / legacy messages (default: disabled; preserve raw training data)."
        ),
    )
    args = parser.parse_args()

    if not args.tools_json.is_file():
        sys.exit(f"tools file not found: {args.tools_json}")
    tools = load_tools_from_file(args.tools_json)

    system_text: str | None = None
    if not args.no_system:
        sp = args.system_prompt
        if not sp.is_file():
            sys.exit(f"system prompt file not found: {sp}")
        system_text = sp.read_text(encoding="utf-8")

    rows: list[dict[str, Any]] = []

    if args.format == "react_steps":
        in_files = collect_react_steps_files(
            input_path=args.input,
            local_llm_dir=args.local_llm_dir,
        )
        if not in_files:
            sys.exit(
                "No react_steps.json found: set --input or ensure --local-llm-dir exists "
                "and contains react_steps.json files"
            )
        for fp in in_files:
            if not fp.is_file():
                sys.exit(f"not a file: {fp}")
            n_before = len(rows)
            for line in iter_react_steps_rows(
                fp,
                system_text=system_text,
                tools=tools,
                include_final_assistant=args.include_final_assistant,
                strip_thought=args.strip_thought,
            ):
                rows.append(line)
            n_file = len(rows) - n_before
            print(f"{fp}: {n_file} sample(s)", file=sys.stderr)
    else:
        if not args.input and not args.input_dir:
            parser.error("legacy-task: provide --input and/or --input-dir")
        if args.input and not args.input.is_file():
            parser.error(f"--input not a file: {args.input}")
        in_files = collect_legacy_input_files(args.input, args.input_dir)
        if not in_files:
            sys.exit("No input JSON files found")
        for fp in in_files:
            rec = load_task_record(fp)
            rows.append(
                record_to_swift_line(
                    rec,
                    system_text=system_text,
                    tools=tools,
                    strip_thought=args.strip_thought,
                )
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fout:
        for line in rows:
            fout.write(json.dumps(line, ensure_ascii=False) + "\n")

    n = len(rows)
    print(f"Wrote {n} sample(s) total to {args.output}", file=sys.stderr)

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
