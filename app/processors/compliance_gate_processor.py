"""Compliance Gate Processor — pharma guardrail between STT and the LLM.

Classifies every final user utterance into:
    on_label | off_label | adverse_event | other
and, for the two regulated classes, injects a [COMPLIANCE GATE] system
directive into the LLM context BEFORE the response is generated, so the
model produces a natural, compliant reply (or transfer) instead of a
canned string.

Design (deliberate, interview-worthy):
- Tier 1: deterministic trigger screen (regex, zero latency, high recall
  for adverse events). A tier-1 AE hit skips the LLM call entirely.
- Tier 2: one fast LLM classification (Groq, small model, ~200-300ms).
  This tier is BLOCKING by design — unlike ToneAwareProcessor, a
  compliance gate that runs "eventually" is not a gate.
- Enforcement lives in the pipeline, not in persona prompts. Prompts ask;
  the gate makes sure.

The approved-indications text fed to the classifier is generated from the
ingested label (Section 1) by scripts/ingest_dupixent.py — never hardcoded,
because labels get revised (Dupixent's indication list has grown almost
yearly; hardcoding it is how you misclassify asthma as off-label).
"""

import asyncio
import json
import pathlib
import re
from typing import Awaitable, Callable, Optional

import httpx
from loguru import logger
from pipecat.frames.frames import Frame, StartFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

# ---------------------------------------------------------------------------
# Tier 1: deterministic adverse-event triggers.
# High recall on purpose: a false positive costs one extra classifier call
# or a gentle Sam handoff; a false negative is a missed safety report.
# ---------------------------------------------------------------------------
_AE_TRIGGERS = re.compile(
    r"\b("
    r"side effects?|adverse|reaction|allerg\w+|rash|hives|swell\w+|"
    r"hospitali[sz]\w+|emergency|anaphyla\w+|"
    r"(got|became|getting|becoming)\s+(worse|sick|ill)|"
    r"(developed|experienc\w+|suffer\w+)\s+\w+|"
    r"(stopped|discontinu\w+|quit)\s+(taking|using|the)|"
    r"(after|since)\s+(starting|taking|the\s+injection|the\s+shot|the\s+dose)|"
    r"eyes?\s+(hurt|red|painful|itch\w*)|conjunctivitis|keratitis|"
    r"joint\s+(pain|ache)|arthralgia"
    r")\b",
    re.IGNORECASE,
)

# Utterances this short carry no classifiable clinical content.
_MIN_WORDS_FOR_LLM = 4

_CLASSIFIER_PROMPT = """You are a compliance classifier for a pharmaceutical medical information line for DUPIXENT (dupilumab).

The APPROVED indications (from Section 1 of the current FDA label) are:
{indications}

Classify the caller's utterance into exactly one class:
- "adverse_event": the caller describes a REAL event happening to a real person (themselves or a patient) after/while using the product: a symptom, reaction, worsening, hospitalization, or stopping the drug because of a problem. Asking what side effects are LISTED is NOT an adverse event.
- "off_label": the caller asks about using the product outside the approved indications above: an unapproved condition, an age/population below the approved range, unapproved dosing (doubling doses, shortening intervals), or combinations/comparisons the label does not cover.
- "on_label": a question answerable from the approved labeling (indications, dosing, warnings, listed adverse reaction rates, storage, administration).
- "other": greetings, small talk, logistics, anything non-clinical.

Reply with JSON only: {{"class": "...", "why": "<one short sentence>"}}"""


class ComplianceGateProcessor(FrameProcessor):
    """Blocking compliance classifier for final user transcriptions.

    Args:
        get_context_messages: callable returning the live LLM context message
            list (ConversationManager.context.messages) so directives land
            before the response is generated.
        groq_api_key: key for the tier-2 classifier.
        model: fast classifier model id.
        indications_path: text file with the label's Section 1 content
            (written by scripts/ingest_dupixent.py --fetch).
        enabled: master switch.
    """

    def __init__(
        self,
        get_context_messages: Callable[[], Optional[list]],
        groq_api_key: str = "",
        model: str = "llama-3.1-8b-instant",
        indications_path: str = "data/dupixent/hcp__indications_and_usage.txt",
        enabled: bool = True,
        timeout_s: float = 2.5,
    ):
        super().__init__()
        self._get_context_messages = get_context_messages
        self._api_key = groq_api_key
        self._model = model
        self._enabled = enabled and bool(groq_api_key)
        self._timeout_s = timeout_s
        self._client: Optional[httpx.AsyncClient] = None
        self.last_classification: str = "other"  # exposed for evals/telemetry
        self._on_a2ui: Optional[Callable] = None  # set to push the compliance badge

        p = pathlib.Path(indications_path)
        self._indications = (
            p.read_text()[:2500] if p.exists()
            else "(indications file missing — run scripts/ingest_dupixent.py --fetch)"
        )
        if not p.exists():
            logger.warning(f"ComplianceGate: indications file not found at {indications_path}")
        logger.info(f"ComplianceGateProcessor initialized (enabled={self._enabled}, model={model})")

    # ------------------------------------------------------------------
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if (
            self._enabled
            and isinstance(frame, TranscriptionFrame)
            and getattr(frame, "text", "").strip()
        ):
            try:
                await self._classify_and_gate(frame.text.strip())
            except Exception as e:  # never let the gate kill the call
                logger.error(f"ComplianceGate error (failing open): {e}")

        await self.push_frame(frame, direction)

    # ------------------------------------------------------------------
    async def _classify_and_gate(self, text: str) -> None:
        label = None

        # Tier 1: deterministic AE screen — instant, no LLM.
        if _AE_TRIGGERS.search(text):
            label = "adverse_event"
            logger.info(f"🛡️ ComplianceGate tier-1 AE trigger: {text!r}")
        elif len(text.split()) < _MIN_WORDS_FOR_LLM:
            self.last_classification = "other"
            return
        else:
            label = await self._classify_llm(text)

        self.last_classification = label or "other"

        # Make the gate's decision visible for the two REGULATED classes. On-label
        # answers already show a content card (which would replace a badge anyway),
        # so we badge only off-label (where nothing else renders) and AE (badge shows
        # now; the full AE report card comes later after Sam's intake).
        if label in ("off_label", "adverse_event") and self._on_a2ui:
            try:
                from app.services.a2ui.pharma_cards import compliance_badge_card
                await self._on_a2ui(compliance_badge_card(label))
            except Exception as e:  # noqa: BLE001 - visual is best-effort
                logger.error(f"compliance badge failed (non-fatal): {e}")

        if label == "adverse_event":
            self._inject(
                "[COMPLIANCE GATE] The caller's last message likely describes a "
                "real-world adverse event. Follow your side effect protocol NOW: "
                "one empathetic sentence, then transfer_to_agent('drug_safety'). "
                "If you ARE the drug-safety agent, continue the four-element intake. "
                "Do not answer any other question first."
            )
        elif label == "off_label":
            self._inject(
                "[COMPLIANCE GATE] The caller's last message was classified as an "
                "OFF-LABEL inquiry. Respond with the compliant redirect: acknowledge "
                "without judgment, decline to discuss beyond the approved labeling, "
                "state what the label DOES cover, and offer a medical affairs "
                "follow-up. If you are not the medical information agent, transfer "
                "to 'medinfo' instead of answering."
            )

    async def _classify_llm(self, text: str) -> str:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url="https://api.groq.com/openai/v1",
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout_s,
            )
        try:
            r = await self._client.post("/chat/completions", json={
                "model": self._model,
                "temperature": 0,
                "max_tokens": 80,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system",
                     "content": _CLASSIFIER_PROMPT.format(indications=self._indications)},
                    {"role": "user", "content": text},
                ],
            })
            r.raise_for_status()
            verdict = json.loads(r.json()["choices"][0]["message"]["content"])
            label = verdict.get("class", "other")
            logger.info(f"🛡️ ComplianceGate tier-2: {label} — {verdict.get('why', '')!r}")
            return label if label in ("on_label", "off_label", "adverse_event", "other") else "other"
        except (httpx.HTTPError, asyncio.TimeoutError, KeyError, json.JSONDecodeError) as e:
            logger.warning(f"ComplianceGate classifier failed open: {e}")
            return "other"

    def set_a2ui_callback(self, callback: Callable) -> None:
        """Register the callback that pushes the compliance-badge A2UI doc."""
        self._on_a2ui = callback

    def _inject(self, directive: str) -> None:
        messages = self._get_context_messages()
        if messages is None:
            logger.warning("ComplianceGate: no context available, directive dropped")
            return
        # Replace any previous gate directive so they never accumulate.
        messages[:] = [
            m for m in messages
            if not (m.get("role") == "system" and str(m.get("content", "")).startswith("[COMPLIANCE GATE]"))
        ]
        messages.append({"role": "system", "content": directive})
        logger.info("🛡️ ComplianceGate directive injected")
