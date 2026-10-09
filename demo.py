"""Guided demo for screen recording: runs a fixed set of queries one by one,
pausing between them so the presenter can talk over each step.

Run:  .venv\\Scripts\\python.exe demo.py            (Windows)
      python demo.py --no-pause                    (run straight through)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import llm_file_assistant as lfa

ROOT = Path(__file__).resolve().parent

# (title, query, what to point out on screen)
SCENES = [
    ("1. Read all resumes",
     "Read all resumes in the resumes folder and give me a one-line overview of each candidate.",
     "list_files first, then one read_file per resume — PDF, DOCX and TXT all parsed."),
    ("2. Keyword search across files",
     "Find resumes mentioning Python experience.",
     "search_in_file runs on every resume; only John Doe and Jane Smith match, with context."),
    ("3. Create a summary file",
     "Create a summary file for resume_john_doe.pdf",
     "read_file then write_file — a new Markdown file appears in summaries/."),
    ("4. Error handling",
     "Read resumes/resume_unknown_person.pdf",
     "The tool returns a structured error and the model explains it instead of crashing."),
]


def banner(text: str) -> str:
    line = "═" * max(60, len(text) + 4)
    return f"\n{line}\n  {text}\n{line}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Guided demo of LLM tool calling.")
    parser.add_argument("--no-pause", action="store_true", help="Don't wait for Enter between scenes")
    parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: set OPENROUTER_API_KEY in .env first.", file=sys.stderr)
        return 1

    assistant = lfa.LLMFileAssistant(model=args.model, base_dir=ROOT, verbose=True)
    print(banner(f"LLM File Assistant — tool-calling demo  (model: {assistant.model})"))
    print("Tools available to the model: " + ", ".join(t["function"]["name"] for t in lfa.TOOL_SCHEMAS))

    for title, query, note in SCENES:
        if not args.no_pause:
            input(f"\nPress Enter for scene {title} …")
        print(banner(title))
        print(f"USER> {query}")
        print(f"(watch for: {note})")
        sys.stdout.flush()
        assistant.reset()  # each scene starts fresh so the tool calls are easy to follow
        answer = assistant.ask(query)
        sys.stderr.flush()
        print(f"\nASSISTANT>\n{answer}")

    summary = ROOT / "summaries" / "resume_john_doe_summary.md"
    if summary.exists():
        if not args.no_pause:
            input("\nPress Enter to show the file the model wrote …")
        print(banner(f"File written by the model: {summary.relative_to(ROOT)}"))
        print(summary.read_text(encoding="utf-8"))

    print(banner("Demo complete"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
