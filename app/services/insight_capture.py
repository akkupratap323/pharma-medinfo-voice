"""Insight capture — Ather's fourth pillar (real-time HCP insight capture).

After a call, turn the transcript into structured intelligence: what was
asked, which questions the label could NOT answer, competitor mentions,
access/logistics concerns, and adverse-event flags. Appended to
data/insights.jsonl and aggregatable into the report pharma clients
actually buy: "unanswered questions per label section."

This is deliberately POST-call (zero added conversational latency), the
same non-blocking discipline as the emotion detector. Uses the existing
Groq LLM; no new dependency.
"""

import json
import pathlib
import time
from typing import Optional

import httpx
from loguru import logger

INSIGHTS_PATH = pathlib.Path("data/insights.jsonl")

# Medical-information intent taxonomy — mirrors how pharma MedInfo teams
# categorize inquiries. The classifier picks from THIS list only, so the
# dashboard's intent distribution is stable across calls.
MEDICAL_INTENTS = [
    "dosing_administration",   # doses, schedules, loading/maintenance, how to inject
    "safety_side_effects",     # AE rates in the label, warnings, precautions
    "adverse_event_report",    # a REAL patient event being reported
    "off_label_use",           # unapproved condition / population / dose
    "drug_interactions",       # co-administration, vaccines, other drugs
    "efficacy_evidence",       # trial results, response rates, how well it works
    "eligibility_trials",      # clinical trial or program screening
    "access_cost_insurance",   # price, copay, coverage, logistics
    "device_usage_training",   # pen vs syringe, storage, handling
    "competitor_comparison",   # how it stacks up vs other products
    "general_product_info",    # indications, what it is, who makes it
    "other",
]

_EXTRACT_PROMPT = """You extract structured commercial intelligence from a single medical-information call transcript about a pharmaceutical product. Output JSON only.

TRANSCRIPT:
{transcript}

Return JSON with:
- "primary_intent": ONE of {intents} — the caller's main reason for calling
- "secondary_intents": array of OTHER intents from the same list that came up (empty if none)
- "themes": array of short topic tags the caller asked about (e.g. "dosing", "conjunctivitis safety", "pediatric use", "access/cost")
- "unanswered": array of {{"question": "...", "reason": "off-label" | "not-in-label" | "escalated" | "other"}} — questions the agent could NOT fully answer from the label
- "competitor_mentions": array of competitor product names the caller raised (empty if none)
- "access_barriers": array of cost/insurance/logistics concerns raised (empty if none)
- "adverse_event_flag": true if any real-world adverse event was described
- "emotion_start": caller's emotional state at the START: "calm" | "anxious" | "frustrated" | "confused" | "urgent" | "upbeat"
- "emotion_end": caller's emotional state at the END: same options plus "reassured"
- "resolved": true if the caller's need was fully handled on this call (answered, properly transferred, or report completed)
- "sentiment": "positive" | "neutral" | "frustrated"
- "caller_type": "hcp" | "patient" | "field_rep" | "unknown"

JSON only, no prose."""


async def capture_insight(
    transcript_text: str,
    call_id: str,
    groq_api_key: str,
    # 70b default: the 8b extractor misread agent intake questions as caller
    # questions (38 spurious "unanswered") and missed an AE flag. Post-call,
    # so the extra ~1s costs nothing.
    model: str = "llama-3.3-70b-versatile",
    persona_id: str = "",
) -> Optional[dict]:
    """Extract + persist one call's insights. Returns the record or None.
    Fail-safe: never raises into the caller (post-call, best-effort)."""
    if not groq_api_key or not transcript_text.strip():
        return None
    try:
        async with httpx.AsyncClient(
            base_url="https://api.groq.com/openai/v1",
            headers={"Authorization": f"Bearer {groq_api_key}"},
            timeout=30,
        ) as cli:
            r = await cli.post("/chat/completions", json={
                "model": model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "user",
                     "content": _EXTRACT_PROMPT.format(
                         transcript=transcript_text[:8000],
                         intents=", ".join(f'"{i}"' for i in MEDICAL_INTENTS),
                     )},
                ],
            })
            r.raise_for_status()
            insight = json.loads(r.json()["choices"][0]["message"]["content"])
    except Exception as e:
        logger.warning(f"insight capture failed for call {call_id}: {e}")
        return None

    record = {
        "call_id": call_id,
        "persona_id": persona_id,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        # Truncated transcript kept for the dashboard drill-down view (demo system;
        # a production deployment would gate this behind retention/PII policy).
        "transcript": transcript_text[:4000],
        **insight,
    }
    INSIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with INSIGHTS_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")
    logger.info(f"📊 insight captured for call {call_id}: "
                f"{len(insight.get('unanswered', []))} unanswered, "
                f"AE={insight.get('adverse_event_flag')}")
    return record


def aggregate_report() -> str:
    """Roll up all captured insights into the buyable report. CLI:
    python -m app.services.insight_capture"""
    if not INSIGHTS_PATH.exists():
        return "No insights captured yet."
    records = [json.loads(l) for l in INSIGHTS_PATH.read_text().splitlines() if l.strip()]

    unanswered, competitors, barriers, ae = {}, {}, {}, 0
    for rec in records:
        for u in rec.get("unanswered", []):
            key = u.get("question", "")[:80]
            unanswered[key] = unanswered.get(key, 0) + 1
        for c in rec.get("competitor_mentions", []):
            competitors[c] = competitors.get(c, 0) + 1
        for b in rec.get("access_barriers", []):
            barriers[b[:60]] = barriers.get(b[:60], 0) + 1
        ae += 1 if rec.get("adverse_event_flag") else 0

    def top(d, n=10):
        return sorted(d.items(), key=lambda kv: -kv[1])[:n]

    lines = [
        "# HCP Insight Report",
        f"\n{len(records)} calls analyzed | {ae} flagged an adverse event\n",
        "## Top unanswered questions (label gaps — the roadmap for content)",
    ]
    lines += [f"- ({n}x) {q}" for q, n in top(unanswered)] or ["- none"]
    lines += ["\n## Competitor mentions"]
    lines += [f"- {c}: {n}" for c, n in top(competitors)] or ["- none"]
    lines += ["\n## Access / cost barriers raised"]
    lines += [f"- ({n}x) {b}" for b, n in top(barriers)] or ["- none"]
    return "\n".join(lines)


if __name__ == "__main__":
    report = aggregate_report()
    pathlib.Path("data/insight_report.md").write_text(report)
    print(report)
    print("\n-> data/insight_report.md")
