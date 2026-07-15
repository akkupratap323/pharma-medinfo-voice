"""Deterministic A2UI documents for the pharma demo.

Unlike the generic A2UI flow (LightRAG fills a template from retrieved text),
these build the document from data we ALREADY hold as structured values:
- the verbatim label section actually retrieved (label-citation)
- the ICH four-element adverse-event report captured by the safety agent (ae-report-card)

Deterministic = the on-screen evidence always matches what was said/recorded. That
"voice + visual proof" pairing is the SynthioLabs differentiator; here it can't drift.
"""

import re
from typing import Any, Dict, Optional

# "Section 5, Warnings and Precautions" / "Section 1, Indications..." / "Patient Information"
_SECTION_RE = re.compile(
    r"(Section\s+\d+[A-Za-z0-9.\s,&/-]*?)(?:\n|$)", re.IGNORECASE
)


def extract_section_label(retrieved_text: str) -> str:
    """Pull the first 'Section N, Title' header from retrieved label text.
    We embed this header at ingest, so it is present in raw chunks. Falls back
    to 'Prescribing Information' if no header is found."""
    if not retrieved_text:
        return "Prescribing Information"
    m = _SECTION_RE.search(retrieved_text)
    if m:
        return " ".join(m.group(1).split())[:80]
    if "patient information" in retrieved_text.lower():
        return "Patient Information"
    if "instructions for use" in retrieved_text.lower():
        return "Instructions for Use"
    return "Prescribing Information"


def _clean_snippet(text: str, limit: int = 600) -> str:
    """Trim boilerplate header lines and cap length for on-screen display."""
    if not text:
        return ""
    lines = [ln for ln in text.splitlines()
             if ln.strip() and "PRESCRIBING INFORMATION" not in ln.upper()]
    body = " ".join(lines).strip()
    return (body[:limit] + "…") if len(body) > limit else body


def label_citation_card(
    retrieved_text: str,
    drug: str = "Dupixent (dupilumab)",
    scope: str = "hcp",
) -> Dict[str, Any]:
    """A2UI doc showing the cited label section beside the spoken answer."""
    section = extract_section_label(retrieved_text)
    return {
        "version": "1.0",
        "root": {
            "type": "label-citation",
            "props": {
                "drug": drug,
                "section": section,
                "text": _clean_snippet(retrieved_text),
                "scope": "Healthcare Professional" if scope == "hcp" else "Patient",
                "sourceNote": "Verbatim from FDA-approved labeling",
            },
        },
        "_metadata": {"tier": "pharma", "tier_name": "deterministic_label_citation"},
    }


_DOSE_ROW_RE = re.compile(
    r"(adult|adolescent|p(a?ediatric)|child|infant|\b\d+\s*(?:to|-|–)\s*\d+\s*years?|"
    r"\b\d+\s*kg\b|weigh\w*)", re.IGNORECASE
)
_MG_RE = re.compile(r"\b\d[\d,]*\s*mg\b", re.IGNORECASE)


def dosing_table_card(retrieved_text: str, drug: str = "Dupixent (dupilumab)") -> Optional[Dict[str, Any]]:
    """Best-effort structured dosing table from retrieved Section 2 text.

    Splits the text into sentences, keeps those that mention BOTH a population
    cue and a mg dose, and renders each as a row. Returns None if fewer than one
    usable row is found, so the caller can fall back to the plain citation card
    (never show an empty table).
    """
    if not retrieved_text:
        return None
    # sentence-ish split; label dosing text is dense so this is heuristic
    chunks = re.split(r"(?<=[.;])\s+", retrieved_text)
    rows = []
    for c in chunks:
        if _MG_RE.search(c) and _DOSE_ROW_RE.search(c):
            doses = _MG_RE.findall(c)
            pop_m = _DOSE_ROW_RE.search(c)
            population = pop_m.group(0).strip().title() if pop_m else "See label"
            # heuristic: first mg = loading if "initial/loading" nearby, else maintenance
            loading = doses[0] if len(doses) > 1 or re.search(r"initial|loading|first", c, re.I) else "—"
            maint = doses[-1] if doses else "—"
            detail = " ".join(c.split())[:160]
            rows.append({"population": population, "loading": loading.upper(),
                         "maintenance": maint.upper(), "detail": detail})
        if len(rows) >= 6:
            break
    if not rows:
        return None
    return {
        "version": "1.0",
        "root": {
            "type": "dosing-table",
            "props": {
                "title": "Dosing",
                "drug": drug,
                "section": extract_section_label(retrieved_text),
                "rows": rows,
                "sourceNote": "Verbatim from FDA-approved labeling · Section 2",
            },
        },
        "_metadata": {"tier": "pharma", "tier_name": "deterministic_dosing_table"},
    }


def compliance_badge_card(classification: str) -> Dict[str, Any]:
    """Live badge making the compliance gate's decision on the last utterance visible."""
    mapping = {
        "on_label": ("On-label", "ok", "Answered from approved labeling"),
        "off_label": ("Off-label — deflected", "warn", "Redirected; not discussed beyond the label"),
        "adverse_event": ("Adverse event — routed", "alert", "Escalated to drug safety"),
        "other": ("General", "neutral", "Non-clinical"),
    }
    label, level, note = mapping.get(classification, mapping["other"])
    return {
        "version": "1.0",
        "root": {
            "type": "compliance-badge",
            "props": {"label": label, "level": level, "note": note},
        },
        "_metadata": {"tier": "pharma", "tier_name": "compliance_badge"},
    }


def handoff_timeline_card(trail: list) -> Dict[str, Any]:
    """Timeline of the call's agent routing: Grace -> Claire -> Sam, with reasons.
    `trail` is a list of {agent, role, reason} dicts (reason optional for the first)."""
    return {
        "version": "1.0",
        "root": {
            "type": "handoff-timeline",
            "props": {"title": "Call routing", "steps": trail},
        },
        "_metadata": {"tier": "pharma", "tier_name": "handoff_timeline"},
    }


def insight_panel_card(insight: Dict[str, Any]) -> Dict[str, Any]:
    """Post-call intelligence: themes, unanswered questions (label gaps), competitor
    mentions, sentiment — Ather's fourth pillar rendered live."""
    return {
        "version": "1.0",
        "root": {
            "type": "insight-panel",
            "props": {
                "title": "Call insights",
                "themes": insight.get("themes", []) or [],
                "unanswered": [u.get("question", "") for u in insight.get("unanswered", []) if u.get("question")],
                "competitors": insight.get("competitor_mentions", []) or [],
                "sentiment": insight.get("sentiment", "neutral"),
                "callerType": insight.get("caller_type", "unknown"),
                "adverseEvent": bool(insight.get("adverse_event_flag")),
            },
        },
        "_metadata": {"tier": "pharma", "tier_name": "insight_panel"},
    }


def ae_report_card(args: Dict[str, Any], report_id: Optional[str] = None) -> Dict[str, Any]:
    """A2UI doc showing the captured adverse-event report for on-screen confirmation.
    Fields map to the ICH four minimum elements plus onset/status/outcome."""
    def g(*keys: str) -> str:
        for k in keys:
            v = args.get(k)
            if v:
                return str(v)
        return "—"

    ongoing = args.get("ongoing")
    status = "Ongoing" if ongoing is True else ("Resolved/Improving" if ongoing is False else "—")
    return {
        "version": "1.0",
        "root": {
            "type": "ae-report-card",
            "props": {
                "title": "Adverse Event Report — Captured",
                "product": g("product"),
                "reporter": g("reporter_name"),
                "reporterContact": g("reporter_contact"),
                "patient": g("patient_descriptor"),
                "doseDuration": g("dose_and_duration"),
                "event": g("event_description"),
                "onset": g("onset"),
                "status": status,
                "outcome": g("outcome"),
                "reportId": report_id or "—",
                "footer": "Logged to safety team · audit trail recorded",
            },
        },
        "_metadata": {"tier": "pharma", "tier_name": "deterministic_ae_report"},
    }
