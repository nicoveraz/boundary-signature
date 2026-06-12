"""Tests for the Decomposer."""
from __future__ import annotations

import pytest

from bsig.medqa.conditions import (
    Decomposer,
    DecomposerConfig,
    DecomposerError,
    DecomposerResult,
)


# ---- DecomposerConfig validation ----


def test_config_defaults() -> None:
    c = DecomposerConfig()
    assert c.min_steps == 3
    assert c.max_steps == 10
    assert c.failure_mode == "graceful"
    assert c.paragraph_fallback is True


def test_config_rejects_zero_min_steps() -> None:
    with pytest.raises(ValueError, match="min_steps"):
        DecomposerConfig(min_steps=0)


def test_config_rejects_min_greater_than_max() -> None:
    with pytest.raises(ValueError, match="min_steps"):
        DecomposerConfig(min_steps=5, max_steps=3)


def test_config_validates_step_pattern_at_construction() -> None:
    with pytest.raises(ValueError, match="invalid step_pattern"):
        DecomposerConfig(step_pattern=r"[unclosed")


def test_config_validates_answer_pattern_at_construction() -> None:
    with pytest.raises(ValueError, match="invalid answer_pattern"):
        DecomposerConfig(answer_pattern=r"(unclosed")


# ---- Happy path ----


def _canonical_output() -> str:
    return """Reasoning step 1: First, consider the patient's symptoms.
Reasoning step 2: The lab values suggest acute presentation.
Reasoning step 3: Differential includes A, B, and C.
Reasoning step 4: B is most consistent with the clinical picture.

Final answer: B
"""


def test_canonical_output_extracts_steps_and_answer() -> None:
    result = Decomposer().decompose(_canonical_output())
    assert isinstance(result, DecomposerResult)
    assert len(result.reasoning_steps) == 4
    assert result.answer_letter == "B"
    assert result.n_steps_raw == 4
    assert result.used_fallback is False
    assert result.warnings == ()


def test_canonical_output_step_text_extracted_correctly() -> None:
    result = Decomposer().decompose(_canonical_output())
    assert "patient's symptoms" in result.reasoning_steps[0]
    assert "lab values" in result.reasoning_steps[1]


def test_answer_letter_uppercased() -> None:
    output = "Reasoning step 1: x\nReasoning step 2: y\nReasoning step 3: z\nFinal answer: c"
    result = Decomposer().decompose(output)
    assert result.answer_letter == "C"


# ---- D6 refinement: answer line stripped from body before step extraction ----


def test_paragraph_fallback_excludes_final_answer_line() -> None:
    """When the canonical step regex misses but paragraph fallback engages,
    the 'Final answer: X' line should NOT appear as a step."""
    output = """The patient presents with chest pain.

The differential is broad including MI, PE, and dissection.

The lab values point toward MI.

Final answer: A
"""
    result = Decomposer().decompose(output)
    assert result.used_fallback is True
    assert result.answer_letter == "A"
    # None of the reasoning steps should be the final-answer line
    for step in result.reasoning_steps:
        assert "Final answer" not in step


# ---- Q4: case-insensitive default ----


def test_canonical_regex_is_case_insensitive_by_default() -> None:
    output = """reasoning step 1: lowercase prefix.
REASONING STEP 2: uppercase prefix.
Reasoning Step 3: mixed case.
Final answer: B
"""
    result = Decomposer().decompose(output)
    assert len(result.reasoning_steps) == 3


# ---- Paragraph fallback ----


def test_paragraph_fallback_engages_when_canonical_fails() -> None:
    output = """The patient is a 50-year-old with chest pain radiating to the left arm.

ECG shows ST elevations in leads II, III, aVF.

This is consistent with inferior wall MI.

Final answer: A
"""
    result = Decomposer().decompose(output)
    assert result.used_fallback is True
    assert len(result.reasoning_steps) == 3
    assert result.answer_letter == "A"
    assert any("paragraph split" in w for w in result.warnings)


def test_paragraph_fallback_disabled_via_config() -> None:
    cfg = DecomposerConfig(paragraph_fallback=False)
    output = "Just some prose, no canonical format.\n\nFinal answer: B"
    result = Decomposer(cfg).decompose(output)
    assert result.used_fallback is False
    assert len(result.reasoning_steps) == 0


# ---- Step-count clamping ----


def test_above_max_steps_downsampled() -> None:
    steps_text = "\n".join(
        f"Reasoning step {i + 1}: step {i + 1} content."
        for i in range(15)
    )
    output = f"{steps_text}\n\nFinal answer: A"
    cfg = DecomposerConfig(max_steps=5)
    result = Decomposer(cfg).decompose(output)
    assert len(result.reasoning_steps) == 5
    assert result.n_steps_raw == 15
    assert any("downsampled" in w for w in result.warnings)


def test_below_min_steps_graceful_keeps_with_warning() -> None:
    output = "Reasoning step 1: one step only.\nFinal answer: A"
    cfg = DecomposerConfig(min_steps=3)
    result = Decomposer(cfg).decompose(output)
    assert len(result.reasoning_steps) == 1
    assert any("min_steps" in w for w in result.warnings)


def test_below_min_steps_strict_raises() -> None:
    output = "Reasoning step 1: one step only.\nFinal answer: A"
    cfg = DecomposerConfig(min_steps=3, failure_mode="strict")
    with pytest.raises(DecomposerError, match="only 1 step"):
        Decomposer(cfg).decompose(output)


def test_zero_steps_strict_raises() -> None:
    cfg = DecomposerConfig(failure_mode="strict")
    with pytest.raises(DecomposerError, match="No reasoning steps"):
        Decomposer(cfg).decompose("Final answer: A")


def test_zero_steps_graceful_keeps_empty() -> None:
    cfg = DecomposerConfig(failure_mode="graceful", paragraph_fallback=False)
    result = Decomposer(cfg).decompose("Final answer: A")
    assert result.reasoning_steps == ()
    assert result.answer_letter == "A"


# ---- Answer extraction ----


def test_no_final_answer_returns_none() -> None:
    output = "Reasoning step 1: a.\nReasoning step 2: b.\nReasoning step 3: c."
    result = Decomposer().decompose(output)
    assert result.answer_letter is None
    assert any("no final answer" in w for w in result.warnings)


def test_answer_extracted_from_middle_of_output() -> None:
    """The answer regex uses MULTILINE so it can match anywhere on a line."""
    output = """Reasoning step 1: a.
Reasoning step 2: b.
Final answer: D
Reasoning step 3: trailing reasoning that doesn't matter.
"""
    result = Decomposer().decompose(output)
    assert result.answer_letter == "D"


# ---- DecomposerResult is frozen ----


def test_result_is_frozen() -> None:
    import dataclasses
    result = Decomposer().decompose(_canonical_output())
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.answer_letter = "X"  # type: ignore[misc]


# ---- Real-world sample (qwen2.5:7b output from grounded exploration) ----


# ---- Stage-4b decomposer relaxation (2026-05-05) ----
#
# The original strict regex r"^Final answer:\s*([A-D])\s*$" rejected
# 35-60% of qwen2.5:7b outputs on MMLU prompts. These tests pin down
# the relaxed pattern's behavior on the deviation classes observed in
# the stage-4b smoke run.


def test_trailing_period_after_letter() -> None:
    output = (
        "Reasoning step 1: a.\nReasoning step 2: b.\n"
        "Reasoning step 3: c.\nFinal answer: A.\n"
    )
    result = Decomposer().decompose(output)
    assert result.answer_letter == "A"
    assert result.warnings == ()


def test_trailing_explanation_after_letter() -> None:
    """Model continues talking after the answer."""
    output = (
        "Reasoning step 1: a.\nReasoning step 2: b.\n"
        "Reasoning step 3: c.\n"
        "Final answer: B. The reason is that option B aligns "
        "with the clinical guidelines.\n"
    )
    result = Decomposer().decompose(output)
    assert result.answer_letter == "B"


def test_markdown_bold_around_prefix() -> None:
    output = (
        "Reasoning step 1: a.\nReasoning step 2: b.\n"
        "Reasoning step 3: c.\n**Final answer:** C\n"
    )
    result = Decomposer().decompose(output)
    assert result.answer_letter == "C"


def test_markdown_bold_around_whole_line() -> None:
    output = (
        "Reasoning step 1: a.\nReasoning step 2: b.\n"
        "Reasoning step 3: c.\n**Final answer: D**\n"
    )
    result = Decomposer().decompose(output)
    assert result.answer_letter == "D"


def test_markdown_italic_around_prefix() -> None:
    output = (
        "Reasoning step 1: a.\nReasoning step 2: b.\n"
        "Reasoning step 3: c.\n*Final answer:* B\n"
    )
    result = Decomposer().decompose(output)
    assert result.answer_letter == "B"


def test_parenthesized_letter() -> None:
    output = (
        "Reasoning step 1: a.\nReasoning step 2: b.\n"
        "Reasoning step 3: c.\nFinal answer: (C)\n"
    )
    result = Decomposer().decompose(output)
    assert result.answer_letter == "C"


def test_letter_with_closing_paren() -> None:
    """'A)' style — model parenthesizes the letter as if it were an
    option label."""
    output = (
        "Reasoning step 1: a.\nReasoning step 2: b.\n"
        "Reasoning step 3: c.\nFinal answer: A)\n"
    )
    result = Decomposer().decompose(output)
    assert result.answer_letter == "A"


def test_no_space_between_colon_and_letter() -> None:
    output = (
        "Reasoning step 1: a.\nReasoning step 2: b.\n"
        "Reasoning step 3: c.\nFinal answer:D\n"
    )
    result = Decomposer().decompose(output)
    assert result.answer_letter == "D"


def test_takes_last_match_on_self_correction() -> None:
    """If the model writes 'Final answer:' multiple times (rare but
    possible during self-correction), the LAST mention wins."""
    output = (
        "Reasoning step 1: a.\nReasoning step 2: b.\n"
        "Reasoning step 3: c.\n"
        "Final answer: A. Wait, on reflection, "
        "the better choice is B.\nFinal answer: B\n"
    )
    result = Decomposer().decompose(output)
    assert result.answer_letter == "B"


def test_letter_with_markdown_emphasis() -> None:
    """The answer letter itself is bolded."""
    output = (
        "Reasoning step 1: a.\nReasoning step 2: b.\n"
        "Reasoning step 3: c.\nFinal answer: **A**\n"
    )
    result = Decomposer().decompose(output)
    assert result.answer_letter == "A"


def test_does_not_match_in_reasoning_text() -> None:
    """A reasoning step containing 'final answer' as descriptive text
    should NOT match — only the literal 'Final answer: X' form."""
    output = (
        "Reasoning step 1: We must determine the final answer carefully.\n"
        "Reasoning step 2: Let's eliminate options one by one.\n"
        "Reasoning step 3: c.\n"
    )
    result = Decomposer().decompose(output)
    # 'final answer carefully' has no letter immediately after, so no
    # match. Letter-extraction returns None.
    assert result.answer_letter is None


def test_does_not_match_letter_in_word_boundary() -> None:
    """'Final answer: ANSWER' should not match A from ANSWER (word
    boundary requirement)."""
    output = (
        "Reasoning step 1: a.\nReasoning step 2: b.\n"
        "Reasoning step 3: c.\nFinal answer: ANSWER\n"
    )
    result = Decomposer().decompose(output)
    # 'A' followed by 'NSWER' is inside a word — \b after [A-D] fails.
    assert result.answer_letter is None


def test_strips_answer_line_from_paragraph_fallback() -> None:
    """When canonical step regex finds nothing, paragraph fallback
    kicks in. The matched 'Final answer' span (and any markdown
    emphasis) must be stripped from the body so paragraph fallback
    doesn't pick it up as a step."""
    output = (
        "First the patient presents with chest pain.\n\n"
        "Then troponin is elevated, suggesting cardiac etiology.\n\n"
        "Likely MI given the constellation of findings.\n\n"
        "**Final answer:** A. Most consistent with STEMI.\n"
    )
    result = Decomposer().decompose(output)
    assert result.answer_letter == "A"
    # Paragraph fallback should NOT include 'Final answer' as a step.
    assert all(
        "final answer" not in step.lower() for step in result.reasoning_steps
    )


def test_qwen25_grounded_exploration_output() -> None:
    """The actual output from the 2026-05-03 grounded exploration."""
    output = """Reasoning step 1: The resident is required to maintain accurate and complete medical records for the patient's care. This includes documenting all surgical complications, even minor ones, as they are part of the patient's medical history.

Reasoning step 2: Ethical standards in medicine mandate transparency with patients regarding their health information. Withholding this information could be considered unethical and potentially harmful to the patient if the complication were to recur or affect future care.

Reasoning step 3: The attending physician's suggestion to omit the complication from the operative report is not aligned with medical ethics, legal requirements, and best practices in patient care.

Final answer: A
"""
    result = Decomposer().decompose(output)
    assert len(result.reasoning_steps) == 3
    assert result.answer_letter == "A"
    assert result.used_fallback is False
    assert result.warnings == ()
