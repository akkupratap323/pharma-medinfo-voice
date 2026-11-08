"""Tests for the shared RAG-routing policy (app/services/rag_routing.py)."""

import pytest

from app.services.rag_routing import (
    CONTRAINDICATION_DIRECTIVE,
    GENERAL_DIRECTIVE,
    NARROW_TOP_K,
    VERBATIM_DIRECTIVE,
    VERBATIM_TOP_K,
    is_dosing_lookup,
    is_factual_lookup,
    retrieval_policy,
)

# Dosing questions in the word forms callers actually use. Each one must get
# the verbatim path, or the answer model may paraphrase the numbers.
DOSING_QUESTIONS = [
    "What dosages are approved for children?",
    "How is it dosed for kids under six?",
    "How many milligrams for a teenager?",
    "Is 300mg right for my son?",
    "Is it 200 mg for adults?",
    "What is the recommended dose for adults with atopic dermatitis?",
    "Is the dosing weight-based?",
    "Is it given every other week?",
]

# Other label facts that need exact text: storage, administration, strengths.
FACTUAL_QUESTIONS = DOSING_QUESTIONS + [
    "Can I inject it myself?",
    "What's the right way of storing the pens?",
    "What strengths does it come in?",
    "Which concentrations are available?",
    "What are the dosing intervals?",
    "How should it be stored?",
]

GENERAL_QUESTIONS = [
    "What are the approved indications for Dupixent?",
    "What were the most common adverse reactions in trials?",
    "What do you have on conjunctivitis and keratitis with this drug?",
]


@pytest.mark.parametrize("question", FACTUAL_QUESTIONS)
def test_factual_questions_get_the_verbatim_path(question):
    policy = retrieval_policy(question)

    assert is_factual_lookup(question)
    assert policy["verbatim"] is True
    assert policy["top_k"] == VERBATIM_TOP_K
    assert policy["directive"] == VERBATIM_DIRECTIVE


@pytest.mark.parametrize("question", DOSING_QUESTIONS)
def test_dosing_questions_are_dosing_and_factual(question):
    # The dosing card must never render for an answer that took the general path.
    assert is_dosing_lookup(question)
    assert is_factual_lookup(question)


@pytest.mark.parametrize("question", GENERAL_QUESTIONS)
def test_general_questions_keep_the_general_path(question):
    policy = retrieval_policy(question)

    assert not is_factual_lookup(question)
    assert not is_dosing_lookup(question)
    assert policy["verbatim"] is False
    assert policy["directive"] == GENERAL_DIRECTIVE


@pytest.mark.parametrize(
    "question",
    [
        "What are the contraindications?",
        "Is it contraindicated at 300mg dosages?",
    ],
)
def test_contraindication_questions_get_the_narrow_path(question):
    policy = retrieval_policy(question)

    assert not is_dosing_lookup(question)
    assert policy["top_k"] == NARROW_TOP_K
    assert policy["directive"] == CONTRAINDICATION_DIRECTIVE


@pytest.mark.parametrize(
    "question",
    ["Is the dosage form a pen?", "Tell me about the dose", "storage please"],
)
def test_existing_word_forms_still_match(question):
    assert is_factual_lookup(question)


def test_empty_question_is_general():
    assert retrieval_policy("")["verbatim"] is False
    assert retrieval_policy(None)["verbatim"] is False
