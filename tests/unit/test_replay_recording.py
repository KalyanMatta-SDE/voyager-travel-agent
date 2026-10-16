"""Replay must reuse LLM-generated research outputs, not regenerate them.

The experience list and the parsed intent are LLM output. Before this change a
replay regenerated both on every run, so location_mismatch and
weather_activity_mismatch (both computed from the experience list) could
differ between runs even with identical API fixtures. These tests pin the
record-once / replay-exactly behaviour. No test calls an LLM or the network.
"""

import json
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents import inventory
from agents.experience_agent import ExperienceAgent
from agents.intent_parser import IntentParserAgent
from config.settings import reload_settings
from graph.travel_graph import _merge_results


@pytest.fixture()
def replay_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("VOYAGER_INVENTORY_DIR", str(tmp_path))
    monkeypatch.setenv("VOYAGER_INVENTORY_MODE", "replay")
    monkeypatch.delenv("VOYAGER_REPLAY_RECORD_MISSING", raising=False)
    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
    reload_settings()
    yield tmp_path
    reload_settings()


def _record_missing(monkeypatch):
    monkeypatch.setenv("VOYAGER_REPLAY_RECORD_MISSING", "true")
    reload_settings()


# ── replay_or_record ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_replay_uses_fixture_and_never_calls_producer(replay_dir):
    qid = inventory.inventory_query_id("experiences", destination="Greece")
    inventory.capture("experiences", {"experiences": [{"name": "A"}]}, qid, run_label="t")
    produce = AsyncMock()
    out = await inventory.replay_or_record("experiences", qid, produce, run_label="t")
    assert out == {"experiences": [{"name": "A"}]}
    produce.assert_not_called()


@pytest.mark.asyncio
async def test_replay_missing_fixture_fails_without_record_flag(replay_dir):
    produce = AsyncMock()
    with pytest.raises(FileNotFoundError):
        await inventory.replay_or_record("experiences", "missing", produce, run_label="t")
    produce.assert_not_called()


@pytest.mark.asyncio
async def test_record_missing_produces_once_then_replays(replay_dir, monkeypatch):
    _record_missing(monkeypatch)
    produce = AsyncMock(return_value={"experiences": [{"name": "B"}]})
    first = await inventory.replay_or_record("experiences", "q1", produce, run_label="t")
    second = await inventory.replay_or_record("experiences", "q1", produce, run_label="t")
    assert first == second == {"experiences": [{"name": "B"}]}
    produce.assert_awaited_once()
    manifest = inventory.load_manifest()
    assert "q1_experiences" in manifest["fixtures"]


# ── ExperienceAgent ──────────────────────────────────────────────────────────

INTENT = {"destination": "Greece", "interests": ["local food", "beaches"],
          "budget_usd": 2000, "duration_days": 7}


def _experience_agent(content: list[dict]):
    with patch("agents.experience_agent.ChatAnthropic"):
        agent = ExperienceAgent()
    response = MagicMock()
    response.content = json.dumps(content)
    chain = AsyncMock()
    chain.ainvoke = AsyncMock(return_value=response)
    prompt = MagicMock()
    prompt.__or__ = MagicMock(return_value=chain)
    agent.prompt = prompt
    agent._test_chain = chain
    return agent


@pytest.mark.asyncio
async def test_experience_replay_is_stable_even_if_llm_would_change(replay_dir, monkeypatch):
    _record_missing(monkeypatch)
    first_llm = [{"name": "Beach day", "category": "beach", "location": "Naxos"}]
    agent = _experience_agent(first_llm)
    r1 = await agent._execute({"intent": INTENT})

    # A second run whose LLM would answer differently must still replay run 1.
    monkeypatch.delenv("VOYAGER_REPLAY_RECORD_MISSING")
    reload_settings()
    agent2 = _experience_agent([{"name": "Museum", "category": "culture", "location": "Athens"}])
    r2 = await agent2._execute({"intent": INTENT})

    assert r1["experiences"] == r2["experiences"] == first_llm
    agent2._test_chain.ainvoke.assert_not_called()


@pytest.mark.asyncio
async def test_experience_key_ignores_interest_order(replay_dir, monkeypatch):
    _record_missing(monkeypatch)
    await _experience_agent([{"name": "X"}])._execute({"intent": INTENT})
    monkeypatch.delenv("VOYAGER_REPLAY_RECORD_MISSING")
    reload_settings()
    reordered = {**INTENT, "interests": ["beaches", "local food"]}
    agent = _experience_agent([{"name": "Y"}])
    out = await agent._execute({"intent": reordered})
    assert out["experiences"] == [{"name": "X"}]


@pytest.mark.asyncio
async def test_mock_mode_records_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("VOYAGER_INVENTORY_DIR", str(tmp_path))
    monkeypatch.setenv("VOYAGER_INVENTORY_MODE", "mock")
    reload_settings()
    try:
        agent = _experience_agent([{"name": "Z"}])
        await agent._execute({"intent": INTENT})
        agent._test_chain.ainvoke.assert_awaited_once()
        assert list(tmp_path.iterdir()) == []
    finally:
        reload_settings()


# ── IntentParserAgent ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_intent_replay_skips_llm(replay_dir, monkeypatch):
    with patch("agents.intent_parser.ChatAnthropic"):
        agent = IntentParserAgent()
    parsed = {"destination": "Greece", "origin": "NYC", "budget_usd": 2000}
    query = "Greece under $2000, flying from NYC"

    _record_missing(monkeypatch)
    with patch.object(agent, "chain") as chain:
        chain.ainvoke = AsyncMock(return_value=type("R", (), {"model_dump": lambda self: parsed})())
        r1 = await agent._execute({"user_query": query})

    monkeypatch.delenv("VOYAGER_REPLAY_RECORD_MISSING")
    reload_settings()
    with patch.object(agent, "chain") as chain:
        chain.ainvoke = AsyncMock(side_effect=AssertionError("LLM must not be called in replay"))
        r2 = await agent._execute({"user_query": query})

    assert r1["intent"] == r2["intent"] == parsed


# ── replay anchor date ───────────────────────────────────────────────────────

def test_manifest_anchor_date_wins_over_local_timezone(replay_dir):
    (replay_dir / "manifest.json").write_text(json.dumps({
        "replay_anchor_date": "2026-09-04",
        # 00:58 UTC on the 5th is still the 4th in Los Angeles but the 5th
        # anywhere east of UTC-01; the explicit anchor removes that dependence.
        "fixtures": {"x_serpapi": {"captured_at": "2026-09-05T00:58:27+00:00"}},
    }))
    with patch("agents.experience_agent.ChatAnthropic"):
        agent = ExperienceAgent()
    assert agent._get_fixture_capture_date() == date(2026, 9, 4)


# ── research-round merge ─────────────────────────────────────────────────────

def test_merge_results_records_exceptions_instead_of_dropping_them():
    merged: dict = {}
    _merge_results(merged, [
        {"flights": []},
        FileNotFoundError("Replay fixture not found: abc_serpapi.json"),
        {"errors": {"hotel_agent": "No intent in state"}},
    ])
    assert merged["flights"] == []
    assert merged["errors"]["hotel_agent"] == "No intent in state"
    assert any("abc_serpapi" in v for v in merged["errors"].values())
