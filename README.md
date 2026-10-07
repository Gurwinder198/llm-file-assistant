# LLM File Assistant

An LLM (any tool-capable model via the [OpenRouter](https://openrouter.ai) Python SDK) that answers questions about resume
files by calling four file-system tools.

```
llm-file-assistant/
├── fs_tools.py               # Part A: read_file, list_files, write_file, search_in_file
├── llm_file_assistant.py     # Part B: tool schemas, dispatcher, agent loop, CLI
├── resumes/                  # fictional sample resumes (PDF, DOCX, TXT)
├── scripts/make_sample_resumes.py
├── tests/                    # pytest suite (no API key needed)
└── docs/plans/               # design + implementation plan
```

## Setup (Windows PowerShell)

```powershell
cd $HOME\Documents\llm-file-assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then put your OPENROUTER_API_KEY in .env
```

(macOS/Linux: `source .venv/bin/activate`, `cp .env.example .env`.)

## Run

```powershell
python llm_file_assistant.py -v "Read all resumes in the resumes folder"
python llm_file_assistant.py -v "Find resumes mentioning Python experience"
python llm_file_assistant.py -v "Create a summary file for resume_john_doe.pdf"
python llm_file_assistant.py          # interactive mode (keeps conversation history)
```

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
pytest -q
```

The assistant tests run offline: most use a scripted fake client, and one
drives the real `openrouter` SDK against a mocked HTTP server to check the
request/response wiring.
