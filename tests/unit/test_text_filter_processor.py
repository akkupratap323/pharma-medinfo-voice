"""
Unit tests for TextFilterProcessor.clean_text_for_speech.
"""

import pytest

from app.processors.text_filter_processor import TextFilterProcessor


@pytest.fixture
def text_filter() -> TextFilterProcessor:
    return TextFilterProcessor()


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("Reduce the dose if CrCl <30 mL/min.", "Reduce the dose if CrCl less than 30 mL/min."),
        ("Do not exceed >400 mg per day.", "Do not exceed greater than 400 mg per day."),
        ("Approved in adults >=18 years.", "Approved in adults greater than or equal to 18 years."),
        ("Use if eGFR <= 15 mL/min.", "Use if eGFR less than or equal to 15 mL/min."),
        ("Stop if ALT > 3 x ULN.", "Stop if ALT greater than 3 x ULN."),
        ("Doses <.5 mg were not studied.", "Doses less than .5 mg were not studied."),
    ],
    ids=["lt", "gt", "ge", "le-spaced", "gt-spaced", "decimal"],
)
def test_comparison_before_a_number_is_spoken(text_filter, text, spoken) -> None:
    assert text_filter.clean_text_for_speech(text) == spoken


def test_comparison_split_across_streamed_chunks_is_spoken(text_filter) -> None:
    chunks = [" Reduce", " the", " dose", " if", " CrCl", " <", " 30", " mL/min."]
    spoken = "".join(text_filter.clean_text_for_speech(chunk) for chunk in chunks)
    assert spoken == " Reduce the dose if CrCl less than 30 mL/min."


@pytest.mark.parametrize(
    ("chunk", "spoken"),
    [(" <=", " less than or equal to"), (">=", " greater than or equal to")],
    ids=["le", "ge-at-start"],
)
def test_two_character_comparison_alone_in_a_chunk_is_spoken(text_filter, chunk, spoken) -> None:
    assert text_filter.clean_text_for_speech(chunk) == spoken


def test_tag_end_at_the_end_of_a_chunk_is_not_spoken(text_filter) -> None:
    spoken = text_filter.clean_text_for_speech('"dose"}</function>')
    assert "than" not in spoken


@pytest.mark.parametrize(
    ("text", "spoken"),
    [
        ("a < b", "a b"),
        ("> Quoted label text", " Quoted label text"),
        ('<function=call_rag_system>{"question": "dose"}</function> Sure.', " Sure."),
        ("Take with food & water.", "Take with food and water."),
        ("**Boxed warning**: serious infections", "Boxed warning: serious infections"),
    ],
    ids=["comparison-without-number", "blockquote", "function-tag", "ampersand", "bold"],
)
def test_other_brackets_and_markdown_are_cleaned_as_before(text_filter, text, spoken) -> None:
    assert text_filter.clean_text_for_speech(text) == spoken
