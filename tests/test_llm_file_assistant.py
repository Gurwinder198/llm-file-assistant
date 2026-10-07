import json
from types import SimpleNamespace

import pytest

import llm_file_assistant as lfa


# ---------- fake OpenRouter client (same shape as openrouter.OpenRouter) ----------

def tool_call(id_, name, args):
    return SimpleNamespace(id=id_, type="function",
                           function=SimpleNamespace(name=name, arguments=json.dumps(args)))


def response(content=None, tool_calls=None):
    finish = "tool_calls" if tool_calls else "stop"
    msg = SimpleNamespace(role="assistant", content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish, index=0)])


class FakeClient:
    """Replays scripted responses and records every request's messages."""

    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.requests = []
        self.chat = self

    def send(self, **kwargs):
        self.requests.append(json.loads(json.dumps(kwargs["messages"])))
        self.last_kwargs = kwargs
        return self.scripted.pop(0)


# ---------- schemas + dispatcher ----------

def test_tool_schemas_are_openai_function_format():
    names = {t["function"]["name"] for t in lfa.TOOL_SCHEMAS}
    assert names == {"read_file", "list_files", "write_file", "search_in_file"}
    for t in lfa.TOOL_SCHEMAS:
        assert t["type"] == "function"
        assert t["function"]["parameters"]["type"] == "object"


def test_dispatch_unknown_tool(resume_dir):
    out = json.loads(lfa.dispatch_tool("delete_everything", {}, base_dir=resume_dir.parent))
    assert out["success"] is False
    assert "unknown tool" in out["error"].lower()


def test_dispatch_accepts_json_string_args(resume_dir):
    out = json.loads(lfa.dispatch_tool("list_files", '{"directory": "resumes", "extension": "pdf"}',
                                       base_dir=resume_dir.parent))
    assert out["success"] is True
    assert [f["name"] for f in out["files"]] == ["john_doe.pdf"]


def test_dispatch_invalid_json_args(resume_dir):
    out = json.loads(lfa.dispatch_tool("read_file", "{not json", base_dir=resume_dir.parent))
    assert out["success"] is False
    assert "invalid json" in out["error"].lower()


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
        response("Let me look.", [tool_call("t1", "list_files", {"directory": "resumes"})]),
        response("There are 3 resumes."),
    ])
    a = lfa.LLMFileAssistant(client=client, model="test/model", base_dir=resume_dir.parent)
    answer = a.ask("How many resumes are there?")

    assert answer == "There are 3 resumes."
    assert client.last_kwargs["model"] == "test/model"
    assert client.last_kwargs["tools"] is lfa.TOOL_SCHEMAS
    first = client.requests[0]
    assert first[0]["role"] == "system" and first[1] == {"role": "user", "content": "How many resumes are there?"}
    # second request: ..., assistant(tool_calls), tool(result)
    second = client.requests[1]
    assert second[-2]["role"] == "assistant"
    assert second[-2]["tool_calls"][0]["function"]["name"] == "list_files"
    tool_msg = second[-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "t1"
    assert "jane_smith.txt" in tool_msg["content"]


def test_parallel_tool_calls_and_write(resume_dir):
    client = FakeClient([
        response(None, [tool_call("a", "search_in_file", {"filepath": "resumes/jane_smith.txt", "keyword": "python"}),
                        tool_call("b", "search_in_file", {"filepath": "resumes/raj_patel.docx", "keyword": "python"})]),
        response(None, [tool_call("c", "write_file", {"filepath": "summaries/out.md", "content": "# Python: Jane"})]),
        response("Done."),
    ])
    a = lfa.LLMFileAssistant(client=client, base_dir=resume_dir.parent)
    assert a.ask("Find python resumes and save a summary") == "Done."

    tool_msgs = [m for m in client.requests[1] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["a", "b"]
    assert json.loads(tool_msgs[0]["content"])["match_count"] == 2
    assert json.loads(tool_msgs[1]["content"])["match_count"] == 0
    assert (resume_dir.parent / "summaries" / "out.md").read_text() == "# Python: Jane"
    assert [c["name"] for c in a.tool_log] == ["search_in_file", "search_in_file", "write_file"]


def test_max_iterations_guard(resume_dir):
    loop = [response(None, [tool_call(f"t{i}", "list_files", {"directory": "resumes"})]) for i in range(5)]
    a = lfa.LLMFileAssistant(client=FakeClient(loop), base_dir=resume_dir.parent, max_iterations=3)
    assert "stopped" in a.ask("loop forever").lower()


def test_api_error_rolls_back_history(resume_dir):
    class Boom:
        def __init__(self):
            self.chat = self

        def send(self, **_):
            raise RuntimeError("network down")

    a = lfa.LLMFileAssistant(client=Boom(), base_dir=resume_dir.parent)
    with pytest.raises(RuntimeError):
        a.ask("hello")
    assert [m["role"] for m in a.messages] == ["system"]


def test_history_persists_and_reset(resume_dir):
    client = FakeClient([response("Hi"), response("Again")])
    a = lfa.LLMFileAssistant(client=client, base_dir=resume_dir.parent)
    a.ask("one")
    a.ask("two")
    assert [m["role"] for m in client.requests[1]] == ["system", "user", "assistant", "user"]
    a.reset()
    assert [m["role"] for m in a.messages] == ["system"]


# ---------- real OpenRouter SDK, mocked HTTP ----------

def test_real_openrouter_sdk_wiring(resume_dir):
    """Drive the actual `openrouter` SDK against a fake HTTP server to verify
    request serialisation and response parsing end to end."""
    httpx = pytest.importorskip("httpx")
    openrouter = pytest.importorskip("openrouter")

    bodies = []

    def chat_result(message, finish):
        return {
            "id": "gen-1", "object": "chat.completion", "created": 1, "model": "openai/gpt-4o-mini",
            "system_fingerprint": None,
            "choices": [{"index": 0, "finish_reason": finish, "message": message}],
        }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        assert request.headers["authorization"] == "Bearer test-key"
        body = json.loads(request.content)
        bodies.append(body)
        if len(bodies) == 1:
            return httpx.Response(200, json=chat_result({
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "call_1", "type": "function",
                                "function": {"name": "search_in_file",
                                             "arguments": '{"filepath": "resumes/john_doe.pdf", "keyword": "pytorch"}'}}],
            }, "tool_calls"))
        return httpx.Response(200, json=chat_result(
            {"role": "assistant", "content": "John Doe mentions PyTorch."}, "stop"))

    client = openrouter.OpenRouter(api_key="test-key",
                                   client=httpx.Client(transport=httpx.MockTransport(handler)))
    a = lfa.LLMFileAssistant(client=client, model="openai/gpt-4o-mini", base_dir=resume_dir.parent)

    assert a.ask("Who knows PyTorch?") == "John Doe mentions PyTorch."
    assert bodies[0]["model"] == "openai/gpt-4o-mini"
    assert bodies[0]["tools"][0]["function"]["name"] == "read_file"
    sent = bodies[1]["messages"]
    assert sent[-2]["tool_calls"][0]["id"] == "call_1"
    assert sent[-1]["role"] == "tool" and sent[-1]["tool_call_id"] == "call_1"
    assert json.loads(sent[-1]["content"])["match_count"] == 1
