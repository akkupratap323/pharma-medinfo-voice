"""Shared RAG-routing policy — single source of truth for the live pipeline
(app/services/conversation.py) AND the eval harness (tests/evals/run_eval.py).

High-stakes factual lookups (dosing numbers, contraindications, storage,
administration) must be answered from VERBATIM label text with an explicit
"quote exact figures, don't paraphrase" directive, because a summarized
dosing table silently corrupts the numbers (600 mg -> 100 mg). Everything
else can use LightRAG's synthesized answer.

Keeping the regex and the directive HERE means the eval can never quietly
diverge from production behavior — the exact bug this module exists to prevent.
"""

import re

# Questions that must read raw label text, never a paraphrase.
FACTUAL_LOOKUP_RE = re.compile(
    r"\b(dos(?:e|es|ing|age)|mg|milligram|loading dose|maintenance|"
    r"how (?:much|often)|frequency|interval|every \w+ weeks|q2w|q4w|"
    r"contraindicat\w+|administ\w+|inject\w+|stor(?:e|es|ed|age)|refrigerat\w+|"
    r"how (?:is it )?supplied|concentration|strength)\b",
    re.IGNORECASE,
)

# Contraindication questions are a SPECIAL factual case that needs the opposite
# of dosing: NARROW retrieval, not wide. Section 4 (Contraindications) is short and
# self-contained; a wide net drags in Section 5 warnings (live vaccines, helminth
# infection) and the answer model then over-lists them as contraindications — the
# exact onl-05 failure. Narrow retrieval keeps Section 4 dominant in context.
CONTRAINDICATION_RE = re.compile(r"\bcontraindicat\w+\b", re.IGNORECASE)

# Retrieval params for the verbatim path (raw chunks, wider net) — good for dosing,
# where breadth is needed to surface the right population block with exact figures.
VERBATIM_TOP_K = 12
VERBATIM_CHUNK_TOP_K = 20
VERBATIM_MAX_TOTAL_TOKENS = 16000

# Narrow params for contraindication lookups — few chunks so Section 4 wins.
NARROW_TOP_K = 3
NARROW_CHUNK_TOP_K = 4
NARROW_MAX_TOTAL_TOKENS = 4000

# Directive appended to raw-context results so the answer model quotes exact
# figures and never conflates a Warning (Sec 5) with a Contraindication (Sec 4).
VERBATIM_DIRECTIVE = (
    " This is RAW label text. Quote the exact figures (doses, mg, intervals) as "
    "written; do NOT round, paraphrase, or infer numbers not present. Attribute "
    "facts to the correct section: a Contraindication (Section 4) is NOT the same "
    "as a Warning or Precaution (Section 5) — do not present a warning as a "
    "contraindication. If a specific fact is not in this text, say you'll have "
    "medical affairs confirm it rather than estimating."
)

# Stronger, contraindication-specific attribution guidance. Grounded in the label:
# names which real label items are Warnings (Sec 5), NOT contraindications — this is
# guidance on attribution, not a hardcoded answer (Section 4 text still comes from
# retrieval), and it's what stops the model over-listing warnings as contraindications.
CONTRAINDICATION_DIRECTIVE = (
    " This is RAW label text. A CONTRAINDICATION is ONLY what appears under the "
    "'Contraindications' heading (Section 4) — for DUPIXENT that is hypersensitivity "
    "to dupilumab or its excipients. Do NOT list Warnings and Precautions (Section 5 — "
    "e.g. conjunctivitis, keratitis, live vaccines, helminth/parasitic infection, "
    "eosinophilic conditions) as contraindications. If the Section 4 text is not "
    "present here, say you'll have medical affairs confirm rather than guessing."
)


# General (non-factual) clinical questions: moderate retrieval width.
GENERAL_TOP_K = 8
GENERAL_CHUNK_TOP_K = 10
GENERAL_MAX_TOTAL_TOKENS = 8000

# Directive for general questions answered from raw label context.
GENERAL_DIRECTIVE = (
    " This is RAW label text. Answer ONLY from it and name the section naturally in "
    "speech. If the answer is not in this text, say you can't address it from the "
    "prescribing information and offer a medical affairs follow-up — never answer "
    "from general knowledge."
)

# Character budgets for the context handed to the ANSWER model. Retrieval can
# return 50K+ chars; shipping it all inflates the voice LLM's prefill and TTS
# start. Wide enough to keep exact figures, tight enough to stay fast.
VERBATIM_CONTEXT_BUDGET = 12000
NARROW_CONTEXT_BUDGET = 4000
GENERAL_CONTEXT_BUDGET = 6000


def is_factual_lookup(question: str) -> bool:
    return bool(FACTUAL_LOOKUP_RE.search(question or ""))


def is_contraindication_lookup(question: str) -> bool:
    return bool(CONTRAINDICATION_RE.search(question or ""))


def retrieval_policy(question: str) -> dict:
    """Single source of truth for how a question is retrieved — used by BOTH the
    live handler and the eval harness so they can never diverge.

    EVERY query now uses RAW retrieved context (only_need_context), never
    LightRAG's server-side generation: the voice LLM re-generates the spoken
    answer anyway, so a server-side generation pass is pure duplicated latency
    (measured 21s at production token budgets vs ~3s retrieval-only).

    Returns {verbatim, top_k, chunk_top_k, max_total_tokens, directive, context_budget}:
    - contraindication -> NARROW net (keep Section 4 dominant)
    - other factual (dosing/storage/admin) -> WIDE net (find the right block)
    - everything else -> moderate net + general grounding directive
    """
    q = question or ""
    if CONTRAINDICATION_RE.search(q):
        return {
            "verbatim": True,
            "top_k": NARROW_TOP_K,
            "chunk_top_k": NARROW_CHUNK_TOP_K,
            "max_total_tokens": NARROW_MAX_TOTAL_TOKENS,
            "directive": CONTRAINDICATION_DIRECTIVE,
            "context_budget": NARROW_CONTEXT_BUDGET,
        }
    if FACTUAL_LOOKUP_RE.search(q):
        return {
            "verbatim": True,
            "top_k": VERBATIM_TOP_K,
            "chunk_top_k": VERBATIM_CHUNK_TOP_K,
            "max_total_tokens": VERBATIM_MAX_TOTAL_TOKENS,
            "directive": VERBATIM_DIRECTIVE,
            "context_budget": VERBATIM_CONTEXT_BUDGET,
        }
    return {
        "verbatim": False,
        "top_k": GENERAL_TOP_K,
        "chunk_top_k": GENERAL_CHUNK_TOP_K,
        "max_total_tokens": GENERAL_MAX_TOTAL_TOKENS,
        "directive": GENERAL_DIRECTIVE,
        "context_budget": GENERAL_CONTEXT_BUDGET,
    }
