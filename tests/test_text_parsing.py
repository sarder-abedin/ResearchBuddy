"""
tests/test_text_parsing.py
───────────────────────────
Unit tests for tools/text_parsing.py:

  - extract_suggested_questions(): pulls a trailing "suggested_questions"
    JSON array off LLM output, tolerating markdown-bolded/italicized keys,
    smart quotes, trailing commas, and several JSON-shape variants.

  - extract_references_section(): isolates the bibliography at the end of
    an academic paper, for the Notebook Citation Timeline feature.

Pure stdlib — no network access or heavy deps required.
"""

from __future__ import annotations

from tools.text_parsing import (
    extract_references_section,
    extract_suggested_questions,
    join_chunks_with_headings,
    prepend_heading,
)


def test_full_json_object():
    """A trailing well-formed JSON object with a "suggested_questions" key is parsed and stripped from the body."""
    raw = (
        "Here is the explanation.\n\n"
        '{"suggested_questions": ["Q1?", "Q2?", "Q3?"]}'
    )
    body, questions = extract_suggested_questions(raw)
    assert body == "Here is the explanation."
    assert questions == ["Q1?", "Q2?", "Q3?"]


def test_quoted_key_without_braces():
    """A quoted "suggested_questions": [...] block with no enclosing {} is still recognized."""
    raw = (
        "Here is the explanation.\n\n"
        '"suggested_questions": ["Q1?", "Q2?", "Q3?"]'
    )
    body, questions = extract_suggested_questions(raw)
    assert body == "Here is the explanation."
    assert questions == ["Q1?", "Q2?", "Q3?"]


def test_bare_key_with_colon():
    """An unquoted suggested_questions: [...] key (colon separator) is recognized."""
    raw = (
        "Here is the explanation.\n\n"
        'suggested_questions: ["Q1?", "Q2?", "Q3?"]'
    )
    body, questions = extract_suggested_questions(raw)
    assert body == "Here is the explanation."
    assert questions == ["Q1?", "Q2?", "Q3?"]


def test_bare_key_with_equals():
    """An unquoted suggested_questions = [...] key (equals separator) is recognized."""
    raw = (
        "Here is the explanation.\n\n"
        'suggested_questions = ["Q1?", "Q2?", "Q3?"]'
    )
    body, questions = extract_suggested_questions(raw)
    assert body == "Here is the explanation."
    assert questions == ["Q1?", "Q2?", "Q3?"]


def test_markdown_bold_key_reported_bug():
    """Reproduces the reported bug: the LLM bolds the key with **...**.

    Streamlit renders the asterisks away, so the user sees a plain
    `suggested_questions: [...]` block with no clickable buttons. The
    fix normalizes the bolded key before pattern-matching.
    """
    raw = (
        "A UAV base station balances coverage and energy use.\n\n"
        '**suggested_questions**: [ "How might a UAV-BS detect an unexpected '
        'energy drain?", "What strategies could be implemented to optimize '
        'UAV-BS trajectories when facing sudden energy drains?", "Could '
        "machine learning algorithms improve a UAV-BS's ability to adapt to "
        'unexpected obstacles?" ]'
    )
    body, questions = extract_suggested_questions(raw)
    assert body == "A UAV base station balances coverage and energy use."
    assert questions == [
        "How might a UAV-BS detect an unexpected energy drain?",
        "What strategies could be implemented to optimize UAV-BS trajectories when facing sudden energy drains?",
        "Could machine learning algorithms improve a UAV-BS's ability to adapt to unexpected obstacles?",
    ]


def test_markdown_italic_and_underscore_keys():
    """Single-asterisk, single-underscore, and double-underscore markdown emphasis around the key are all normalized away."""
    for opener, closer in [("*", "*"), ("_", "_"), ("__", "__")]:
        raw = (
            "Body text.\n\n"
            f'{opener}suggested_questions{closer}: ["Q1?", "Q2?"]'
        )
        body, questions = extract_suggested_questions(raw)
        assert body == "Body text.", (opener, closer)
        assert questions == ["Q1?", "Q2?"], (opener, closer)


def test_smart_quotes_are_normalized():
    """Curly/smart quotes (“ ”) inside the question strings are normalized to plain ASCII quotes."""
    raw = (
        "Body text.\n\n"
        "suggested_questions: [“What is X?”, “What is Y?”]"
    )
    body, questions = extract_suggested_questions(raw)
    assert body == "Body text."
    assert questions == ["What is X?", "What is Y?"]


def test_trailing_comma_is_tolerated():
    """A trailing comma before the closing bracket (invalid strict JSON) doesn't break parsing."""
    raw = (
        "Body text.\n\n"
        'suggested_questions: ["Q1?", "Q2?", "Q3?",]'
    )
    body, questions = extract_suggested_questions(raw)
    assert body == "Body text."
    assert questions == ["Q1?", "Q2?", "Q3?"]


def test_max_questions_caps_results():
    """max_questions truncates the parsed list even when more questions are present in the source text."""
    raw = 'suggested_questions: ["Q1?", "Q2?", "Q3?", "Q4?", "Q5?"]'
    _, questions = extract_suggested_questions(raw, max_questions=2)
    assert questions == ["Q1?", "Q2?"]


def test_no_match_returns_raw_unchanged():
    """With no recognizable suggested_questions block, the raw text passes through unchanged and questions is empty."""
    raw = "Just a plain explanation with no trailing JSON block."
    body, questions = extract_suggested_questions(raw)
    assert body == raw
    assert questions == []


# ── extract_references_section() ─────────────────────────────────────────

def test_references_heading_extracts_trailing_section():
    """A "References" heading near the end of the document isolates the bibliography, excluding earlier body text."""
    text = (
        "Title of the Paper\n\n"
        "Abstract\nThis paper studies X.\n\n"
        "1. Introduction\nSome introduction text. " + ("filler " * 50) + "\n\n"
        "References\n"
        "[1] Smith, J. (2020). A great paper. Journal of Things.\n"
        "[2] Doe, A. (2019). Another paper. Conference of Stuff."
    )
    section = extract_references_section(text)
    assert section.startswith("[1] Smith, J. (2020)")
    assert "[2] Doe, A. (2019)" in section
    assert "Introduction" not in section


def test_bibliography_and_works_cited_headings_recognized():
    """Alternate bibliography headings ("Bibliography", "Works Cited", "Literature Cited", all-caps "REFERENCES") are all recognized."""
    for heading in ["Bibliography", "Works Cited", "Literature Cited", "REFERENCES"]:
        text = ("Body text. " * 50) + f"\n\n{heading}\n[1] Author (2021). Paper title."
        section = extract_references_section(text)
        assert section == "[1] Author (2021). Paper title.", heading


def test_early_mention_is_ignored_in_favor_of_later_heading():
    """An early "References" mention in a table of contents is ignored in favor of the real, later heading."""
    text = (
        "Table of Contents\n"
        "1. Introduction\n"
        "2. References\n"
        "3. Appendix\n\n"
        + ("Body text discussing prior work. " * 30) + "\n\n"
        "References\n"
        "[1] Real Citation (2022). The actual bibliography entry."
    )
    section = extract_references_section(text)
    assert section == "[1] Real Citation (2022). The actual bibliography entry."


def test_no_heading_returns_empty_string():
    """With no recognizable bibliography heading anywhere, the function returns an empty string."""
    text = "Just a regular document body with no bibliography heading at all. " * 10
    assert extract_references_section(text) == ""


def test_heading_with_trailing_colon():
    """A "References:" heading (trailing colon) is recognized the same as a bare "References" heading."""
    text = ("Body text. " * 50) + "\n\nReferences:\n[1] Author (2023). Title."
    section = extract_references_section(text)
    assert section == "[1] Author (2023). Title."


def test_heading_with_trailing_page_number():
    """A heading line with a trailing running-footer page number ("References  12") still matches."""
    # PDF extraction sometimes leaves a running-footer page number on the
    # same line as the heading: "References  12".
    text = ("Body text. " * 50) + "\n\nReferences  12\n[1] Author (2024). Title."
    section = extract_references_section(text)
    assert section == "[1] Author (2024). Title."


def test_heading_at_chunk_boundary_recovered_via_paragraph_join():
    """Re-joining two notebook chunks with "\\n\\n" restores the line break the heading regex needs when a chunk boundary fell right after "References"."""
    # Simulates two notebook chunks: the first ends right at the "References"
    # heading (chunk text is .strip()'d, so the trailing newline is gone),
    # and the second begins with the first bibliography entry. Joining with
    # "\n\n" (as agents.notebook_advanced.extract_citation_timeline does)
    # restores the line break the heading regex needs.
    chunk_a = ("Body discussing prior work. " * 30) + "\n\nReferences"
    chunk_b = "[1] Author A (2018). Title One.\n[2] Author B (2019). Title Two."
    combined = "\n\n".join([chunk_a, chunk_b])
    section = extract_references_section(combined)
    assert section == "[1] Author A (2018). Title One.\n[2] Author B (2019). Title Two."


def test_numbered_bibliography_fallback_when_heading_unrecognizable():
    """When the heading is mangled beyond regex recognition (e.g. letter-spaced "R E F E R E N C E S"), a numbered [1][2][3]... run is used as a fallback signal."""
    # PDF extraction sometimes mangles a letter-spaced "REFERENCES" heading
    # into something that no longer contains "references" as a word. Fall
    # back to recognizing the standard [1] [2] [3]... numbered list.
    text = (
        ("Body text discussing the topic in depth. " * 40) + "\n\n"
        "R E F E R E N C E S\n"
        "[1] A. Author (2018). First paper.\n"
        "[2] B. Author (2019). Second paper.\n"
        "[3] C. Author (2020). Third paper.\n"
        "[4] D. Author (2021). Fourth paper.\n"
        "[5] E. Author (2022). Fifth paper."
    )
    section = extract_references_section(text)
    assert section.startswith("[1] A. Author (2018)")
    assert "[5] E. Author (2022)" in section
    assert "REFERENCES" not in section


def test_numbered_fallback_requires_minimum_run_length():
    """A short numbered list unrelated to citations (e.g. "our contributions") is NOT mistaken for a bibliography by the fallback heuristic."""
    # A short numbered list (e.g. "our contributions") with no References/
    # Bibliography heading should NOT be mistaken for a bibliography.
    text = (
        ("Body text. " * 50) + "\n\n"
        "Our contributions are:\n"
        "[1] We propose a new method.\n"
        "[2] We evaluate it extensively."
    )
    assert extract_references_section(text) == ""


# ── prepend_heading() ────────────────────────────────────────────────────


def test_prepend_heading_adds_heading_on_its_own_line():
    """A heading held only in chunk metadata is restored as a leading line."""
    assert prepend_heading("Smith, J. (2020).", "References") == "References\nSmith, J. (2020)."


def test_prepend_heading_is_idempotent():
    """Text that already opens with the heading is returned unchanged, so chunks ingested after the fix aren't double-prefixed."""
    already = "References\nSmith, J. (2020)."
    assert prepend_heading(already, "References") == already
    # Same line rather than its own line still counts as already present.
    assert prepend_heading("References Smith, J.", "References") == "References Smith, J."
    # Case differences in the stored heading shouldn't cause a duplicate.
    assert prepend_heading("REFERENCES\nSmith, J.", "References") == "REFERENCES\nSmith, J."


def test_prepend_heading_empty_heading_is_a_noop():
    """Chunks with no heading metadata pass through untouched."""
    assert prepend_heading("body text", "") == "body text"
    assert prepend_heading("body text", "   ") == "body text"


# ── join_chunks_with_headings() ──────────────────────────────────────────


def _chunk(text: str, heading: str = "") -> dict:
    """Build a stored-chunk dict of the shape NotebookMemory.load() returns."""
    meta = {"source": "paper.pdf"}
    if heading:
        meta["heading"] = heading
    return {"text": text, "metadata": meta}


def test_join_chunks_emits_each_heading_once_per_section():
    """Docling repeats a section's heading on every chunk; the rejoin emits it only where it changes."""
    chunks = [
        _chunk("First part of the methods.", "Methods"),
        _chunk("Second part of the methods.", "Methods"),
        _chunk("What we found.", "Results"),
    ]
    joined = join_chunks_with_headings(chunks)
    assert joined.count("Methods") == 1
    assert joined.count("Results") == 1
    assert joined.index("Methods") < joined.index("Results")


def test_join_chunks_without_heading_metadata_matches_plain_join():
    """Chunks carrying no heading metadata rejoin exactly as the old plain join did."""
    chunks = [_chunk("alpha"), _chunk("beta")]
    assert join_chunks_with_headings(chunks) == "alpha\n\nbeta"


def test_join_chunks_tolerates_missing_and_malformed_metadata():
    """A chunk with absent or non-dict metadata doesn't break the rejoin."""
    chunks = [{"text": "alpha"}, {"text": "beta", "metadata": None}, {"text": "gamma", "metadata": []}]
    assert join_chunks_with_headings(chunks) == "alpha\n\nbeta\n\ngamma"


def test_author_year_bibliography_recovered_from_heading_metadata():
    """The reported bug: an author-year bibliography whose "References" heading lived only in chunk metadata was invisible to extract_references_section().

    Author-year styles (APA, Harvard, Nature) have no "[1] [2] [3]" markers, so
    the numbered-bibliography fallback cannot rescue them -- without the heading
    the whole source was silently skipped by extract_citation_timeline().
    """
    chunks = [
        _chunk("Body text discussing prior work. " * 30, "Introduction"),
        _chunk("Smith, J. (2020). A great paper. Journal of Things.", "References"),
        _chunk("Doe, A. (2019). Another paper. Conference of Stuff.", "References"),
    ]

    # Old behaviour: heading dropped -> nothing found.
    plain = "\n\n".join(c["text"] for c in chunks)
    assert extract_references_section(plain) == ""

    # With the heading restored, the bibliography is isolated correctly.
    section = extract_references_section(join_chunks_with_headings(chunks))
    assert section.startswith("Smith, J. (2020)")
    assert "Doe, A. (2019)" in section
    assert "prior work" not in section
