"""Tests for prompt loading and version parsing."""
from __future__ import annotations

import pytest

from bsig.medqa import (
    PROMPT_NAMES,
    load_all_versions,
    load_prompt,
    load_prompt_version,
)


def test_all_known_prompts_loadable() -> None:
    """Every name in PROMPT_NAMES has a corresponding template file."""
    for name in PROMPT_NAMES:
        text = load_prompt(name)
        assert isinstance(text, str)
        assert len(text) > 0


def test_all_versions_returns_dict_keyed_by_name() -> None:
    versions = load_all_versions()
    assert set(versions) == set(PROMPT_NAMES)
    for name in PROMPT_NAMES:
        assert isinstance(versions[name], int)


def test_current_prompt_versions() -> None:
    """Track which prompts have real (v1+) versus placeholder (v0)
    content. As prompts get filled in across stages, update this
    expected map.

    Stage 3.3a: condition_a, condition_b, repair at v1; condition_c_*
    still at v0 (filled in stage 3.3b).
    """
    expected = {
        "condition_a": 1,
        "condition_b": 1,
        "condition_c_initial": 2,  # bumped 2026-05-04 per ADR-0008 (minimal CoT prompt; no "Final answer:" line)
        "condition_c_measurement": 1,  # ADR-0008 unified-measurement protocol
        "repair": 1,
    }
    versions = load_all_versions()
    assert versions == expected, (
        f"prompt versions don't match expected. got={versions}, "
        f"expected={expected}. Update this test if prompts intentionally "
        f"changed."
    )


def test_load_prompt_strips_version_header() -> None:
    """The returned text does NOT contain the # version: line."""
    text = load_prompt("condition_a")
    assert "# version:" not in text


def test_unknown_prompt_raises_with_hint() -> None:
    with pytest.raises(FileNotFoundError, match="src/bsig/medqa/prompts"):
        load_prompt("nonexistent_template")


def test_load_prompt_version_returns_integer() -> None:
    v = load_prompt_version("condition_a")
    assert isinstance(v, int)
    assert v >= 0


def test_caching_returns_same_object() -> None:
    """The @cache decorator means repeated calls return the identical
    string (not just equal)."""
    t1 = load_prompt("condition_a")
    t2 = load_prompt("condition_a")
    assert t1 is t2
