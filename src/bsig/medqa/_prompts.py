"""Prompt template loading for the MedQA domain pack.

Templates live in ``bsig/medqa/prompts/<name>.txt``. Each template's
first non-blank line is a version marker:

    # version: N

where ``N`` is a non-negative integer. Version 0 indicates a placeholder
template (e.g., shipped during stage 3.1 before real prompts were
filled in); runs using version-0 prompts should NOT be trusted for
headline metrics, and ``signature_metadata.json`` records the version
explicitly so downstream analysis can detect this.

Real prompts start at version 1. Bumping a prompt's content means
bumping its version comment; ``signature_metadata.json`` records which
versions were used so reproducibility is preserved across prompt
revisions.

API:
- ``load_prompt(name)``: returns the template text with the version
  comment stripped. Cached.
- ``load_prompt_version(name)``: returns the integer version. Cached.

Both raise ``FileNotFoundError`` for unknown names with a hint about
which templates exist.
"""
from __future__ import annotations

import importlib.resources
import re
from functools import cache

_VERSION_LINE_RE = re.compile(r"^\s*#\s*version\s*:\s*(\d+)\s*$")


def _read_raw(name: str) -> str:
    try:
        return (
            importlib.resources.files("bsig.medqa.prompts")
            .joinpath(f"{name}.txt")
            .read_text(encoding="utf-8")
        )
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise FileNotFoundError(
            f"Prompt template not found: {name}.txt. "
            f"Available templates live in src/bsig/medqa/prompts/."
        ) from exc


def _split_version_and_body(raw: str, name: str) -> tuple[int, str]:
    lines = raw.splitlines(keepends=True)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        m = _VERSION_LINE_RE.match(line)
        if not m:
            raise ValueError(
                f"Prompt template {name!r} is missing a "
                f"'# version: N' header on its first non-blank line. "
                f"First non-blank line was: {line!r}"
            )
        version = int(m.group(1))
        body = "".join(lines[i + 1 :])
        return version, body
    raise ValueError(
        f"Prompt template {name!r} is empty or contains no header"
    )


@cache
def load_prompt(name: str) -> str:
    """Return the prompt template text (with the version comment stripped)."""
    raw = _read_raw(name)
    _, body = _split_version_and_body(raw, name)
    return body


@cache
def load_prompt_version(name: str) -> int:
    """Return the integer version of a prompt template."""
    raw = _read_raw(name)
    version, _ = _split_version_and_body(raw, name)
    return version


PROMPT_NAMES: tuple[str, ...] = (
    "condition_a",
    "condition_b",
    "condition_c_initial",
    "condition_c_measurement",  # ADR-0008 unified-measurement protocol
    "repair",
)


def load_all_versions() -> dict[str, int]:
    """Return ``{name: version}`` for every known prompt. Useful for
    stuffing into ``signature_metadata.json``'s ``prompt_versions``
    field at run time.
    """
    return {name: load_prompt_version(name) for name in PROMPT_NAMES}
