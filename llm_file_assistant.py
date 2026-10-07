"""LLM File Assistant — lets an LLM (via OpenRouter) answer questions about
resume files by calling the tools in ``fs_tools``.

Usage:
    python llm_file_assistant.py "Find resumes mentioning Python experience"
    python llm_file_assistant.py            # interactive mode

Environment:
    OPENROUTER_API_KEY  required  (https://openrouter.ai/keys)
    OPENROUTER_MODEL    optional  (default: openai/gpt-4o-mini; any OpenRouter
                                   model that supports tool calling works)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

import fs_tools

DEFAULT_MODEL = "openai/gpt-4o-mini"
MAX_CONTENT_CHARS = 30_000  # cap file text sent back to the model per call
MAX_TOKENS = 4096
APP_TITLE = "LLM File Assistant"

SYSTEM_PROMPT = """You are a helpful file assistant for a recruiting team.
You can work with resume files (PDF, DOCX, TXT, MD) through these tools:
read_file, list_files, write_file, search_in_file.

Guidelines:
- Paths are relative to the project folder; resumes are usually in "resumes/".
- To answer questions about "all resumes", first call list_files, then read or
  search each file. Call several tools in one turn when they are independent.
- To find resumes mentioning a skill, call search_in_file on every resume and
  report which files matched, quoting the context you found. Try sensible
  variants of a keyword (e.g. "ML" and "machine learning") if useful.
- When asked to create a summary file, read the resume first, then write a
  concise Markdown summary (name, title, contact, skills, experience highlights,
  education) to "summaries/<resume_name>_summary.md" unless told otherwise.
- Never invent resume content. If a tool returns an error, explain it plainly.
- Finish with a short, clear answer for the user and list any files you wrote.
"""


def _function_tool(name: str, description: str, properties: dict, required: list[str]) -> dict:
    """Build an OpenAI-style function tool definition (the format OpenRouter uses)."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOL_SCHEMAS: list[dict] = [
    _function_tool(
        "read_file",
        "Read a resume file (PDF, TXT, DOCX, MD) and return its full text plus metadata "
        "(name, size, modified date, word count, pages).",
        {"filepath": {"type": "string", "description": "Path to the file, e.g. resumes/john_doe.pdf"}},
        ["filepath"],
    ),
    _function_tool(
        "list_files",
        "List files in a directory with metadata (name, path, size, modified date). "
        "Optionally filter by extension such as '.pdf'.",
        {
            "directory": {"type": "string", "description": "Directory to list, e.g. resumes"},
            "extension": {"type": "string", "description": "Optional extension filter, e.g. .pdf"},
        },
        ["directory"],
    ),
    _function_tool(
        "write_file",
        "Write text content to a file, creating parent directories if needed. Overwrites existing files.",
        {
            "filepath": {"type": "string", "description": "Destination path, e.g. summaries/x.md"},
            "content": {"type": "string", "description": "Text to write"},
        },
        ["filepath", "content"],
    ),
    _function_tool(
        "search_in_file",
        "Case-insensitive keyword search inside a file (PDF, DOCX, TXT, MD). "
        "Returns each match with line number and surrounding context.",
        {
            "filepath": {"type": "string", "description": "File to search"},
            "keyword": {"type": "string", "description": "Keyword or phrase to find"},
        },
        ["filepath", "keyword"],
    ),
]

_TOOLS: dict[str, Callable[..., Any]] = {
    "read_file": fs_tools.read_file,
    "list_files": fs_tools.list_files,
    "write_file": fs_tools.write_file,
    "search_in_file": fs_tools.search_in_file,
}
_PATH_ARGS = {"filepath", "directory"}


# --------------------------------------------------------------------------- #
# Tool dispatch
# --------------------------------------------------------------------------- #

def _resolve_path(raw: str, base_dir: Optional[Path]) -> Path:
    """Resolve ``raw`` against ``base_dir`` and refuse paths that escape it."""
    p = Path(raw).expanduser()
    if base_dir is None:
        return p.resolve()
    base = base_dir.resolve()
    full = (p if p.is_absolute() else base / p).resolve()
    if full != base and base not in full.parents:
        raise PermissionError(f"Path '{raw}' is outside the allowed folder {base}")
    return full


def dispatch_tool(name: str, args: dict | str | None, base_dir: Optional[Path | str] = None) -> str:
    """Run a tool by name and return a JSON string for the model.

    ``args`` may be a dict or the raw JSON string the model produced.
    """
    base = Path(base_dir) if base_dir is not None else None
    func = _TOOLS.get(name)
    if func is None:
        return json.dumps({"success": False, "error": f"Unknown tool: {name}"})

    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError as exc:
            return json.dumps({"success": False, "error": f"Invalid JSON arguments for {name}: {exc}"})
    if not isinstance(args, dict):
        return json.dumps({"success": False, "error": f"Invalid arguments for {name}: expected an object"})

    try:
        kwargs = dict(args)
        for key in _PATH_ARGS & kwargs.keys():
            kwargs[key] = str(_resolve_path(kwargs[key], base))
        result = func(**kwargs)
    except PermissionError as exc:
        return json.dumps({"success": False, "error": str(exc)})
    except TypeError as exc:
        return json.dumps({"success": False, "error": f"Invalid arguments for {name}: {exc}"})
    except Exception as exc:  # last-resort safety net
        return json.dumps({"success": False, "error": f"{type(exc).__name__}: {exc}"})

    if name == "list_files":
        result = {"success": True, "directory": kwargs["directory"], "count": len(result), "files": result}
    elif name == "read_file" and result.get("success"):
        content = result["content"]
        result["truncated"] = len(content) > MAX_CONTENT_CHARS
        if result["truncated"]:
            result["content"] = content[:MAX_CONTENT_CHARS]
    return json.dumps(result, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Assistant
# --------------------------------------------------------------------------- #

def make_openrouter_client(api_key: Optional[str] = None):
    """Create an OpenRouter SDK client (imported lazily so tests need no key)."""
    from openrouter import OpenRouter

    return OpenRouter(
        api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
        x_open_router_title=APP_TITLE,
    )


class LLMFileAssistant:
    """A small agent loop: model ⇄ file tools, until the model stops calling tools."""

    def __init__(
        self,
        client: Any = None,
        model: Optional[str] = None,
        base_dir: Optional[Path | str] = ".",
        max_iterations: int = 15,
        verbose: bool = False,
    ):
        self.client = client if client is not None else make_openrouter_client()
        self.model = model or os.getenv("OPENROUTER_MODEL", DEFAULT_MODEL)
        self.base_dir = Path(base_dir).resolve() if base_dir is not None else None
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.messages: list[dict] = []
        self.tool_log: list[dict] = []
        self.reset()

    def reset(self) -> None:
        self.messages[:] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.tool_log.clear()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=sys.stderr)

    def ask(self, query: str) -> str:
        """Send a user query, let the model call tools, return its final text."""
        checkpoint = len(self.messages)
        self.messages.append({"role": "user", "content": query})
        try:
            return self._run_loop()
        except Exception:
            # Roll back so a failed API call doesn't leave a broken history.
            del self.messages[checkpoint:]
            raise

    def _run_loop(self) -> str:
        for _ in range(self.max_iterations):
            response = self.client.chat.send(
                model=self.model,
                messages=self.messages,
                tools=TOOL_SCHEMAS,
                max_tokens=MAX_TOKENS,
            )
            message = response.choices[0].message
            text = message.content if isinstance(message.content, str) else ""
            tool_calls = list(message.tool_calls or [])

            assistant_msg: dict = {"role": "assistant", "content": text or ""}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"},
                    }
                    for tc in tool_calls
                ]
            self.messages.append(assistant_msg)

            if not tool_calls:
                return (text or "").strip()

            for tc in tool_calls:
                name, raw_args = tc.function.name, tc.function.arguments or "{}"
                self._log(f"→ {name}({raw_args[:200]})")
                output = dispatch_tool(name, raw_args, self.base_dir)
                try:
                    parsed_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    parsed_args = raw_args
                self.tool_log.append({"name": name, "input": parsed_args})
                self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": output})

        return (f"Stopped after {self.max_iterations} tool rounds without a final answer. "
                "Try a more specific question or raise --max-iterations.")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

EXAMPLES = [
    "Read all resumes in the resumes folder",
    "Find resumes mentioning Python experience",
    "Create a summary file for resume_john_doe.pdf",
]


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Ask an LLM (via OpenRouter) about your resume files.")
    parser.add_argument("query", nargs="*", help="Question to ask (omit for interactive mode)")
    parser.add_argument("--base-dir", default=str(Path(__file__).resolve().parent),
                        help="Folder the assistant may access (default: this project folder)")
    parser.add_argument("--model", default=None,
                        help=f"OpenRouter model id (default: $OPENROUTER_MODEL or {DEFAULT_MODEL})")
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("-v", "--verbose", action="store_true", help="Print tool calls")
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv(Path(args.base_dir) / ".env")
    except ImportError:
        pass

    if not os.getenv("OPENROUTER_API_KEY"):
        print("Error: set OPENROUTER_API_KEY (or put it in a .env file).", file=sys.stderr)
        return 1

    assistant = LLMFileAssistant(model=args.model, base_dir=args.base_dir,
                                 max_iterations=args.max_iterations, verbose=args.verbose)

    if args.query:
        print(assistant.ask(" ".join(args.query)))
        return 0

    print(f"{APP_TITLE} ({assistant.model}) — type a question, 'reset' to clear history, 'quit' to exit.")
    print("Examples:\n  " + "\n  ".join(EXAMPLES))
    while True:
        try:
            query = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not query:
            continue
        if query.lower() in {"quit", "exit"}:
            return 0
        if query.lower() == "reset":
            assistant.reset()
            print("(history cleared)")
            continue
        try:
            print(f"\nassistant> {assistant.ask(query)}")
        except Exception as exc:
            print(f"\n[error] {type(exc).__name__}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
