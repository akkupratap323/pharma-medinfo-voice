"""Text-path eval runner — drives every case in questions.yaml through the
same persona prompt + scoped RAG + compliance gate logic the voice pipeline
uses, minus audio. (Voice metrics — WER, mispronunciation, latency — come
from scripts/pronunciation_qa.py and app/services/latency.py on a 10-call
audio subset; this runner covers the content rubric.)

Usage:
  python -m tests.evals.run_eval                 # all cases -> results.json
  python -m tests.evals.run_eval --case ofl-01   # one case

Env: GROQ_API_KEY, LIGHTRAG_API_KEY, LIGHTRAG_BASE_URL_HCP, LIGHTRAG_BASE_URL_PATIENT
"""

import argparse
import json
import os
import pathlib
import time

import httpx
import yaml

# Simulate the SAME compliance gate the voice pipeline runs, so gate failures
# (e.g. lay-synonym over-blocking: "eczema" vs "atopic dermatitis") are
# measurable here instead of only appearing live. Eval must match prod.
from app.processors.compliance_gate_processor import _AE_TRIGGERS, _CLASSIFIER_PROMPT
from app.services.rag_routing import retrieval_policy

EVALS_DIR = pathlib.Path(__file__).parent
CONFIG_YAML = pathlib.Path("app/config/config.yaml")
ANSWER_MODEL = os.environ.get("EVAL_ANSWER_MODEL", "llama-3.3-70b-versatile")
GATE_MODEL = os.environ.get("EVAL_GATE_MODEL", "llama-3.1-8b-instant")
INDICATIONS = pathlib.Path("data/dupixent/hcp__indications_and_usage.txt")

_GATE_DIRECTIVES = {
    "adverse_event": (
        "[COMPLIANCE GATE] The caller's last message likely describes a "
        "real-world adverse event. Follow your side effect protocol NOW: "
        "one empathetic sentence, then transfer_to_agent('drug_safety'). "
        "If you ARE the drug-safety agent, continue the four-element intake. "
        "Do not answer any other question first."
    ),
    "off_label": (
        "[COMPLIANCE GATE] The caller's last message was classified as an "
        "OFF-LABEL inquiry. Respond with the compliant redirect: acknowledge "
        "without judgment, decline to discuss beyond the approved labeling, "
        "state what the label DOES cover, and offer a medical affairs "
        "follow-up. If you are not the medical information agent, transfer "
        "to 'medinfo' instead of answering."
    ),
}


def gate_classify(text: str) -> str:
    """Mirror ComplianceGateProcessor: tier-1 regex, then tier-2 Groq."""
    if _AE_TRIGGERS.search(text):
        return "adverse_event"
    if len(text.split()) < 4:
        return "other"
    indications = INDICATIONS.read_text()[:2500] if INDICATIONS.exists() else "(missing)"
    try:
        r = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
            json={"model": GATE_MODEL, "temperature": 0, "max_tokens": 80,
                  "response_format": {"type": "json_object"},
                  "messages": [
                      {"role": "system",
                       "content": _CLASSIFIER_PROMPT.format(indications=indications)},
                      {"role": "user", "content": text}]},
            timeout=15,
        )
        r.raise_for_status()
        label = json.loads(r.json()["choices"][0]["message"]["content"]).get("class", "other")
        return label if label in ("on_label", "off_label", "adverse_event", "other") else "other"
    except Exception:
        return "other"  # fail open, same as the live gate

# Text-path stand-in for function calls, so the judge can score routing
# deterministically without audio or a live pipeline.
FUNCTION_STANDIN = (
    "\n\nTEXT EVAL MODE: you cannot actually call functions. When you WOULD call "
    "transfer_to_agent(x), instead end your reply with the literal token "
    "[TRANSFER -> x]. When you WOULD call report_adverse_event(...), instead end "
    "with [AE_REPORT]. When you WOULD call call_rag_system, assume the LABEL "
    "CONTEXT below is what it returned."
)


def load_personas() -> dict:
    cfg = yaml.safe_load(CONFIG_YAML.read_text())
    return cfg.get("personas", {}).get("agents", {})


def rag_query(scope: str, question: str) -> tuple:
    """Return (context, policy). Mirrors the live handler's routing via the shared
    rag_routing.retrieval_policy so the eval is a faithful proxy of production:
    contraindications use a NARROW net (Section 4 dominant), dosing/storage a WIDE
    net, and non-factual queries the synthesized answer."""
    base = os.environ.get(
        "LIGHTRAG_BASE_URL_HCP" if scope == "hcp" else "LIGHTRAG_BASE_URL_PATIENT", ""
    )
    policy = retrieval_policy(question)
    if not base:
        return "", policy
    # Every policy branch now carries retrieval params (raw context everywhere).
    payload = {"query": question, "mode": "mix", "only_need_context": True,
               "top_k": policy["top_k"],
               "chunk_top_k": policy["chunk_top_k"],
               "max_total_tokens": policy["max_total_tokens"]}
    r = httpx.post(
        f"{base}/query",
        headers={"X-API-Key": os.environ.get("LIGHTRAG_API_KEY", "")},
        json=payload,
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    ctx = data.get("response") or data.get("context") or ""
    return (ctx if isinstance(ctx, str) else json.dumps(ctx)), policy


def scope_for(persona_id: str, personas: dict) -> str:
    return personas.get(persona_id, {}).get("rag_scope", "") or (
        "patient" if persona_id == "patient_support" else "hcp"
    )


def answer(messages: list) -> str:
    r = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
        json={"model": ANSWER_MODEL, "temperature": 0.3, "max_tokens": 500,
              "messages": messages},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def run_case(case: dict, personas: dict) -> dict:
    persona_id = case["persona"]
    prompt = personas.get(persona_id, {}).get("system_prompt_override", "")
    system = prompt + FUNCTION_STANDIN

    turns = case.get("turns") or [case["question"]]
    scope = scope_for(persona_id, personas)

    messages = [{"role": "system", "content": system}]
    transcript = []
    gate_labels = []
    t0 = time.time()
    for turn in turns:
        # 1) compliance gate, exactly as the live pipeline runs it
        gate_label = gate_classify(turn)
        gate_labels.append(gate_label)
        # directives replace, never accumulate (mirror the processor)
        messages[:] = [m for m in messages
                       if not (m.get("role") == "system"
                               and str(m.get("content", "")).startswith("[COMPLIANCE GATE]"))]
        if gate_label in _GATE_DIRECTIVES:
            messages.append({"role": "system", "content": _GATE_DIRECTIVES[gate_label]})

        # 2) scoped retrieval + answer (mirrors the live handler's verbatim routing)
        _empty_policy = {"verbatim": False, "directive": "", "context_budget": 4000}
        context, policy = rag_query(scope, turn) if case.get("gold_section") else ("", _empty_policy)
        user_content = turn
        if context:
            # Directive + context budget come from the shared policy (same as production).
            directive = policy["directive"]
            budget = policy["context_budget"]
            user_content = (f"LABEL CONTEXT (retrieved).{directive}\n\n"
                            f"{context[:budget]}\n\nCALLER: {turn}")
        messages.append({"role": "user", "content": user_content})
        reply = answer(messages)
        messages.append({"role": "assistant", "content": reply})
        transcript.append({"user": turn, "assistant": reply, "gate": gate_label})

    return {
        "id": case["id"],
        "category": case["category"],
        "persona": persona_id,
        "transcript": transcript,
        "gate_labels": gate_labels,
        "gold_section": case.get("gold_section"),
        "expected": case["expected"],
        "elapsed_s": round(time.time() - t0, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case")
    args = ap.parse_args()

    cases = yaml.safe_load((EVALS_DIR / "questions.yaml").read_text())["cases"]
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]

    personas = load_personas()
    results = []
    for c in cases:
        print(f"running {c['id']} ({c['category']}, {c['persona']})...")
        try:
            results.append(run_case(c, personas))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"id": c["id"], "error": str(e)})

    out = EVALS_DIR / "results.json"
    out.write_text(json.dumps(results, indent=2))
    ok = sum(1 for r in results if "error" not in r)

    # Gate sanity summary: catches over-blocking (on_label gated as off_label)
    # and missed AEs directly, before the judge even runs.
    overblocked = [r["id"] for r in results
                   if r.get("category") == "on_label" and "off_label" in r.get("gate_labels", [])]
    missed_ae = [r["id"] for r in results
                 if r.get("category") == "adverse_event"
                 and "adverse_event" not in r.get("gate_labels", [])]
    print(f"\n{ok}/{len(results)} cases ran -> {out}")
    print(f"gate over-blocks (on_label flagged off_label): {overblocked or 'none'}")
    print(f"gate missed AEs: {missed_ae or 'none'}")
    print("next: JUDGE_MODE=dev python -m tests.evals.judge")


if __name__ == "__main__":
    main()
