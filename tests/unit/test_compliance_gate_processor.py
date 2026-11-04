"""
Unit tests for ComplianceGateProcessor directive handling (no network: the
tier-2 classifier is stubbed).
"""

from unittest.mock import AsyncMock

import pytest

from app.processors.compliance_gate_processor import ComplianceGateProcessor

GATE = "[COMPLIANCE GATE]"


def _gate_directives(messages):
    return [
        m["content"] for m in messages if m["role"] == "system" and m["content"].startswith(GATE)
    ]


@pytest.fixture
def messages():
    return [{"role": "system", "content": "You are the medinfo agent."}]


@pytest.fixture
def gate(messages, tmp_path):
    indications = tmp_path / "indications.txt"
    indications.write_text("1 INDICATIONS AND USAGE: atopic dermatitis, asthma.")
    processor = ComplianceGateProcessor(
        get_context_messages=lambda: messages,
        groq_api_key="test-key",
        indications_path=str(indications),
    )
    processor._classify_llm = AsyncMock(return_value="other")
    return processor


async def _ask(gate, messages, text, label=None):
    if label is not None:
        gate._classify_llm.return_value = label
    await gate._classify_and_gate(text)
    messages.append({"role": "user", "content": text})
    messages.append({"role": "assistant", "content": "..."})


class TestComplianceGateDirectives:
    async def test_off_label_injects_one_directive(self, gate, messages):
        await _ask(gate, messages, "can my two year old use it for eczema", "off_label")

        directives = _gate_directives(messages)
        assert len(directives) == 1
        assert "OFF-LABEL" in directives[0]

    async def test_adverse_event_replaces_previous_directive(self, gate, messages):
        await _ask(gate, messages, "can my two year old use it for eczema", "off_label")
        await _ask(gate, messages, "I got a rash after the injection")

        directives = _gate_directives(messages)
        assert len(directives) == 1
        assert "adverse event" in directives[0]

    async def test_on_label_question_clears_stale_off_label_directive(self, gate, messages):
        await _ask(gate, messages, "can my two year old use it for eczema", "off_label")
        await _ask(gate, messages, "how should I store the prefilled pen", "on_label")

        assert _gate_directives(messages) == []
        assert gate.last_classification == "on_label"

    async def test_non_clinical_question_clears_stale_adverse_event_directive(self, gate, messages):
        await _ask(gate, messages, "my eyes hurt since the shot")
        await _ask(gate, messages, "what number can I call back on", "other")

        assert _gate_directives(messages) == []

    async def test_short_reply_keeps_directive(self, gate, messages):
        await _ask(gate, messages, "my eyes hurt since the shot")
        await _ask(gate, messages, "yes")

        assert len(_gate_directives(messages)) == 1
        gate._classify_llm.assert_not_called()

    async def test_other_system_messages_are_kept(self, gate, messages):
        await _ask(gate, messages, "can my two year old use it for eczema", "off_label")
        await _ask(gate, messages, "how should I store the prefilled pen", "on_label")

        assert messages[0] == {"role": "system", "content": "You are the medinfo agent."}

    async def test_clearing_without_context_is_silent(self, tmp_path):
        processor = ComplianceGateProcessor(
            get_context_messages=lambda: None,
            groq_api_key="test-key",
            indications_path=str(tmp_path / "missing.txt"),
        )
        processor._classify_llm = AsyncMock(return_value="on_label")

        await processor._classify_and_gate("how should I store the prefilled pen")

        assert processor.last_classification == "on_label"
