import json
from types import SimpleNamespace

import pytest

import llm_file_assistant as lfa


# ---------- fake Anthropic client ----------

def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_block(id_, name, input_):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=input_)


def response(blocks, stop="end_turn"):
    return SimpleNamespace(content=blocks, stop_reason=stop)


class FakeClient:
    """Replays scripted responses and records every request."""

    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.requests = []
        self.messages = self

    def create(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs["messages"], default=str)))
        return self.scripted.pop(0)


# ---------- dispatcher ----------

def test_tool_schemas_cover_all_tools():
    names = {t["name"] for t in lfa.TOOL_SCHEMAS}
    assert names == {"read_file", "list_files", "write_file", "search_in_file"}
    for t in lfa.TOOL_SCHEMAS:
        assert t["input_schema"]["type"] == "object"


def test_dispatch_unknown_tool(resume_dir):
    out = json.loads(lfa.dispatch_tool("delete_everything", {}, base_dir=resume_dir.parent))
    assert out["success"] is False
    assert "unknown tool" in out["error"].lower()


def test_dispatch_relative_path_resolves_against_base_dir(resume_dir):
    out = json.loads(lfa.dispatch_tool("list_files", {"directory": "resumes", "extension": "pdf"},
                                       base_dir=resume_dir.parent))
    assert out["success"] is True
    assert [f["name"] for f in out["files"]] == ["john_doe.pdf"]


def test_dispatch_blocks_path_escape(resume_dir):
    out = json.loads(lfa.dispatch_tool("read_file", {"filepath": "../../etc/passwd"},
                                       base_dir=resume_dir.parent))
    assert out["success"] is False
    assert "outside" in out["error"].lower()


def test_dispatch_bad_arguments(resume_dir):
    out = json.loads(lfa.dispatch_tool("read_file", {"wrong": "x"}, base_dir=resume_dir.parent))
    assert out["success"] is False
    assert "argument" in out["error"].lower()


def test_dispatch_truncates_long_content(tmp_path):
    (tmp_path / "big.txt").write_text("a" * (lfa.MAX_CONTENT_CHARS + 500))
    out = json.loads(lfa.dispatch_tool("read_file", {"filepath": "big.txt"}, base_dir=tmp_path))
    assert out["success"] is True
    assert out["truncated"] is True
    assert len(out["content"]) == lfa.MAX_CONTENT_CHARS


# ---------- agent loop ----------

def test_single_tool_round_trip(resume_dir):
    client = FakeClient([
        response([text_block("Let me look."),
                  tool_block("t1", "list_files", {"directory": "resumes"})], stop="tool_use"),
        response([text_block("There are 3 resumes.")]),
    ])
    a = lfa.LLMFileAssistant(client=client, base_dir=resume_dir.parent)
    answer = a.ask("How many resumes are there?")

    assert answer == "There are 3 resumes."
    # second request must contain the tool_result for t1
    last_user = client.requests[1][-1]
    assert last_user["role"] == "user"
    result = last_user["content"][0]
    assert result["type"] == "tool_result" and result["tool_use_id"] == "t1"
    assert "jane_smith.txt" in result["content"]


def test_parallel_tool_calls_and_write(resume_dir):
    client = FakeClient([
        response([tool_block("a", "search_in_file", {"filepath": "resumes/jane_smith.txt", "keyword": "python"}),
                  tool_block("b", "search_in_file", {"filepath": "resumes/raj_patel.docx", "keyword": "python"})],
                 stop="tool_use"),
        response([tool_block("c", "write_file", {"filepath": "summaries/out.md", "content": "# Python: Jane"})],
                 stop="tool_use"),
        response([text_block("Done.")]),
    ])
    a = lfa.LLMFileAssistant(client=client, base_dir=resume_dir.parent)
    assert a.ask("Find python resumes and save a summary") == "Done."

    results = client.requests[1][-1]["content"]
    assert [r["tool_use_id"] for r in results] == ["a", "b"]
    assert json.loads(results[0]["content"])["match_count"] == 2
    assert json.loads(results[1]["content"])["match_count"] == 0
    assert (resume_dir.parent / "summaries" / "out.md").read_text() == "# Python: Jane"
    assert [c["name"] for c in a.tool_log] == ["search_in_file", "search_in_file", "write_file"]


def test_max_iterations_guard(resume_dir):
    loop = [response([tool_block(f"t{i}", "list_files", {"directory": "resumes"})], stop="tool_use")
            for i in range(5)]
    a = lfa.LLMFileAssistant(client=FakeClient(loop), base_dir=resume_dir.parent, max_iterations=3)
    answer = a.ask("loop forever")
    assert "stopped" in answer.lower()


def test_api_error_rolls_back_history(resume_dir):
    class Boom:
        def __init__(self):
            self.messages = self

        def create(self, **_):
            raise RuntimeError("network down")

    a = lfa.LLMFileAssistant(client=Boom(), base_dir=resume_dir.parent)
    with pytest.raises(RuntimeError):
        a.ask("hello")
    assert a.messages == []


def test_history_persists_between_questions(resume_dir):
    client = FakeClient([response([text_block("Hi")]), response([text_block("Again")])])
    a = lfa.LLMFileAssistant(client=client, base_dir=resume_dir.parent)
    a.ask("one")
    a.ask("two")
    assert len(client.requests[1]) == 3  # user, assistant, user
    a.reset()
    assert a.messages == []
