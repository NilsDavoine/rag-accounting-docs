"""The Claude path, tested with a fake client: no key, no network, no cost."""
from types import SimpleNamespace

import numpy as np
import pytest

from eval import run_generation as rg
from rag import answer as answer_mod
from rag import llm


class FakeMessages:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        text = self.reply(kwargs["messages"][-1]["content"]) if callable(self.reply) else self.reply
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)],
                               usage=SimpleNamespace(input_tokens=1000, output_tokens=50))


@pytest.fixture
def fake_claude(monkeypatch):
    def install(reply):
        msgs = FakeMessages(reply)
        monkeypatch.setattr(llm, "_client", lambda: SimpleNamespace(messages=msgs))
        return msgs
    return install


@pytest.fixture(autouse=True)
def stub_reranker(monkeypatch):
    def fake(query, texts):
        q = set(query.lower().split())
        return np.array([len(q & set(t.lower().split())) / 10 for t in texts])
    monkeypatch.setattr(answer_mod, "rerank_scores", fake)


def hit(doc, heading, text, ce):
    return {"id": hash((doc, heading)) % 10_000, "doc_id": doc, "heading": heading, "title": doc, "text": text,
            "prefixed": f"[{doc} | 2025]\n{text}", "ce": ce}


RENT = [hit("lease", "Rent", "The monthly rent is EUR 900.00 for the offices.", 0.9)]


def test_request_has_no_sampling_parameters_and_no_prefill(fake_claude):
    msgs = fake_claude("The monthly rent is EUR 900.00 [1].")
    a = answer_mod.answer("What is the monthly rent for the offices?", [dict(h) for h in RENT], 0.2, 0.1, "anthropic")
    call = msgs.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert not {"temperature", "top_p", "top_k"} & set(call)
    assert [m["role"] for m in call["messages"]] == ["user"]          # last turn is the user's: no prefill
    assert call["max_tokens"] <= 100_000 and call["output_config"] == {"effort": "low"}
    assert "EUR 900.00" in a.text and not a.refused


def test_model_sentinel_becomes_a_refusal(fake_claude):
    fake_claude("NO_ANSWER")
    a = answer_mod.answer("What is the monthly rent for the offices?", [dict(h) for h in RENT], 0.2, 0.1, "anthropic")
    assert a.refused


def test_invented_number_falls_back_to_the_quote(fake_claude):
    fake_claude("The rent is EUR 950.00.")
    a = answer_mod.answer("What is the monthly rent for the offices?", [dict(h) for h in RENT], 0.2, 0.1, "anthropic")
    assert "llm_number_not_in_sources" in a.flags and "900.00" in a.text and "950" not in a.text


def test_guardrail_refusals_never_reach_the_model(fake_claude):
    msgs = fake_claude("should not be called")
    a = answer_mod.answer("What is the landlord's IBAN?", [dict(h) for h in RENT], 0.2, 0.1, "anthropic")
    assert a.refused and msgs.calls == []


def test_cost_estimate_uses_haiku_prices():
    est = rg.estimate(["x" * 3000, "y" * 3000], "claude-haiku-5-5")
    assert est["requests"] == 2 and est["input_tokens"] == 2000
    assert est["cost_expected"] == pytest.approx((2000 * 0.10 + 2 * rg.EXPECTED_OUTPUT_TOKENS * 0.50) / 1e6)
    assert est["cost_max"] == pytest.approx((2000 * 0.10 + 2 * llm.MAX_TOKENS * 0.50) / 1e6)


def test_no_key_stops_before_any_work(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    def must_not_run(_):
        raise AssertionError("retrieval started without a key")
    with pytest.raises(SystemExit, match="not set"):
        rg.main(["--generator", "anthropic", "--yes"], collect_fn=must_not_run)


def test_bench_command_end_to_end_with_fake_client(fake_claude, monkeypatch, tmp_path, capsys):
    """`make eval-claude` path: estimate printed first, then calls, then the Claude row lands in results.md."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-placeholder")
    results, store = tmp_path / "results.md", tmp_path / "generation.json"
    results.write_text("# r\n\n<!-- generation:start -->\n<!-- generation:end -->\n", encoding="utf-8")
    monkeypatch.setattr(rg, "RESULTS", results)
    monkeypatch.setattr(rg, "STORE", store)
    monkeypatch.setattr(rg, "load_settings", lambda: {"tau": 0.2, "margin": 0.1, "nature_weight": 0.0})
    msgs = fake_claude(lambda prompt: "The monthly rent is EUR 900.00 [1]." if "rent" in prompt else "NO_ANSWER")
    qs = [({"question": "What is the monthly rent for the offices?", "answerable": True, "kind": "lease_rent",
            "type": "amount", "expected": ["900.00"]}, RENT),
          ({"question": "What is the landlord's IBAN?", "answerable": False, "kind": "unanswerable:absent_value",
            "type": None, "expected": []}, RENT)]
    rg.main(["--generator", "anthropic", "--yes"], collect_fn=lambda s: {"main (test split)": qs})
    out = capsys.readouterr().out
    assert out.index("Estimated cost") < out.index("main (test split) {")
    assert len(msgs.calls) == 1                         # the IBAN question is refused before the model
    text = results.read_text(encoding="utf-8")
    assert "| anthropic | `claude-haiku-5-5` | main (test split): 2 | 100.0 %" in text
    assert "$0.0001 (1,000 in / 50 out)" in text
