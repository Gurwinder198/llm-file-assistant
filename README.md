# LLM File Assistant

An LLM (any tool-capable model via the [OpenRouter](https://openrouter.ai) Python SDK) that answers questions about resume
files by calling four file-system tools.

```
llm-file-assistant/
├── fs_tools.py               # Part A: read_file, list_files, write_file, search_in_file
├── llm_file_assistant.py     # Part B: tool schemas, dispatcher, agent loop, CLI
├── resumes/                  # fictional sample resumes (PDF, DOCX, TXT)
├── scripts/make_sample_resumes.py
├── setup.bat                 # one-click Windows setup
├── tests/                    # pytest suite (no API key needed)
└── docs/plans/               # design + implementation plan
```

## Setup (Windows)

**Easiest:** double-click `setup.bat` (or run `.\setup.bat`). It finds Python,
creates `.venv`, installs the requirements and creates `.env`. Then open `.env`
and paste your `OPENROUTER_API_KEY`.

Manual equivalent — note it uses `py -m pip` / `python -m pip`, so it works even
when `pip` is "not recognized":

```powershell
cd $HOME\Documents\llm-file-assistant
py -3 -m venv .venv                       # or: python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env                    # then put your OPENROUTER_API_KEY in .env
```

If neither `py` nor `python` is recognized, Python isn't installed (or not on
PATH): install it from python.org with **"Add python.exe to PATH"** ticked, or
run `winget install Python.Python.3.12`, then open a **new** terminal.

## Run

Using the venv's Python directly means you don't need to activate anything:

```powershell
.\.venv\Scripts\python.exe llm_file_assistant.py -v "Read all resumes in the resumes folder"
.\.venv\Scripts\python.exe llm_file_assistant.py -v "Find resumes mentioning Python experience"
.\.venv\Scripts\python.exe llm_file_assistant.py -v "Create a summary file for resume_john_doe.pdf"
.\.venv\Scripts\python.exe llm_file_assistant.py      # interactive mode (keeps history)
```

(Or activate first with `.\.venv\Scripts\Activate.ps1` and just use `python`.
If PowerShell blocks that script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.)

`-v` prints each tool call the model makes. Summaries are written to
`summaries/`. Options: `--model`, `--base-dir`, `--max-iterations`.

Pick any OpenRouter model that supports tool calling, e.g.
`--model anthropic/claude-sonnet-4.5` or `--model google/gemini-2.5-flash`
(or set `OPENROUTER_MODEL` in `.env`). Default: `openai/gpt-4o-mini`.

## Part A — `fs_tools.py`

| Tool | Returns | Notes |
|---|---|---|
| `read_file(filepath)` | `{success, content, metadata}` | PDF (pypdf), DOCX incl. tables (python-docx), TXT/MD (UTF-8 → latin-1 fallback). Metadata: name, path, extension, size, modified, char/word count, pages. |
| `list_files(directory, extension=None)` | `list[dict]` | name, path, extension, size_bytes, modified. `pdf`, `.pdf`, `.PDF` all work. Missing dir → `[]`. |
| `write_file(filepath, content)` | `{success, path, bytes_written}` | Creates parent dirs, UTF-8. |
| `search_in_file(filepath, keyword)` | `{success, match_count, matches[]}` | Case-insensitive; each match has line, position, matched text, ±80 chars of context. Works on PDF/DOCX too. |

All errors come back as `{"success": false, "error": "..."}` — no exceptions
escape to the LLM loop.

## Part B — `llm_file_assistant.py`

1. `TOOL_SCHEMAS` describe the tools as OpenAI-style function tools (the
   format OpenRouter uses).
2. `LLMFileAssistant.ask()` calls `OpenRouter().chat.send(model, messages, tools)`.
   When the model replies with `tool_calls`, each is executed by
   `dispatch_tool()` and the result is appended as a `role: "tool"` message.
   This repeats until the model gives a final answer (capped by
   `--max-iterations`).
3. **Sandboxing:** every path is resolved against `--base-dir` (the project
   folder by default); paths that escape it (`../..`) are refused.
4. Long file contents are truncated before being sent to the model.

Because OpenRouter fronts many providers, switching between OpenAI,
Anthropic, Google, Mistral, etc. is just a different `--model` id.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The assistant tests run offline: most use a scripted fake client, and one
drives the real `openrouter` SDK against a mocked HTTP server to check the
request/response wiring.
