# Resumable-campaign property

**Date:** 2026-05-07
**Status:** ACCEPTED
**Category:** project convention (not an ADR; doesn't change framework
architecture).
**Cross-references:** ``corruption_registry.md`` entry for port-8080
contention; ``project_corruption_registry.md`` memory.

## Statement

Any framework script with expected wall-time > 30 minutes MUST satisfy
the resumable-campaign property:

1. **Per-unit artifacts.** Output is written incrementally as each
   unit (question, encounter, trajectory) completes — not buffered
   until end-of-run.
2. **Skip-existing on resume.** Re-invoking the script with the same
   configuration scans existing per-unit artifacts and skips units
   already complete.
3. **Atomic per-unit writes.** Each unit's output is written via a
   write-temp + rename pattern (or equivalent) so a crash mid-write
   leaves the unit either fully complete or absent — never partial.
4. **Configuration captured.** The script writes its full
   configuration (input paths, model identifier, seed, prompt
   template hash, etc.) alongside the artifacts. Resume detects
   configuration drift and refuses to mix.

A script that satisfies all four can be killed at any point and
resumed with no data loss and no double-counted units.

## Why

Stage-4b's full N=1534 MMLU professional_law run took ~19 hours on
M1 Pro. The cross-adapter agreement test crashed at q17/50 on
2026-05-07 due to port-8080 contention with the local clinical-app
UI. Both cases would have lost substantial work without per-unit
streaming.

Stage-4b survived its 19-hour run because the script wrote per-question
parquet rows incrementally — accidentally resumable, not by design.
Cross-adapter test survived its q17 crash because per-question
results were streamed via stdout (with ``flush=True``) — accidentally
resumable in a different sense (logs preserved partial run for
diagnosis, not for resumption).

These were two separate accidents. Making resumability a *named
property* future scripts must satisfy converts the accidental into
the deliberate — and prevents the next long-running script from
losing work to an avoidable interruption.

## How to apply

For new long-running scripts (>30 min expected wall-time):

- Write outputs per-unit to a directory (one file or one parquet
  row per question/encounter/trajectory). Avoid buffering into
  memory and dumping at end.
- At script start, scan the output directory for completed units
  and build a skip-set. Iterate input units; skip those in the
  skip-set; process the rest.
- Use ``tempfile.NamedTemporaryFile`` + ``os.replace`` for atomic
  writes when the per-unit artifact is a single file. For parquet,
  write each unit to a uniquely-named file under the directory;
  appending to a single parquet is *not* atomic and is forbidden.
- Write a ``config.json`` (or equivalent) on first invocation
  capturing the script's configuration. On resume, compare; refuse
  to resume if configuration changed without an explicit
  ``--force-fresh`` flag.

For existing long-running scripts that don't yet satisfy this:

- Treat as registered corruption-mode candidates. When the next
  long-running incident reveals a missing piece (config drift on
  resume, crash mid-write of an aggregated parquet, etc.), apply
  the fix and update this document with the specific failure mode
  encountered.

## Canonical example

``experiments/medqa_generalization/scripts/03c_run_condition_c.py``
(stage-4b version) is the canonical resumable script. It:

- Writes one parquet row per question to a directory.
- On startup, lists completed question_ids and skips them.
- Captures model + seed + prompt-hash in a sidecar config.

Future long-running scripts should follow this shape unless there's
a documented reason not to.

## Non-goals

- This is not a framework feature. There's no shared
  ``ResumableCampaign`` base class to inherit from. Each script
  implements the property in whatever shape fits its data flow.
  Premature abstraction across scripts is exactly the speculative-
  framework antipattern documented in ``project_corruption_registry.md``.
- This is not retroactive. Stage-1 through stage-4 scripts that
  pre-date this convention are not required to be retrofitted
  unless they're going to run again.

## Promotion criterion

If 3+ scripts independently arrive at substantially-similar
resumability shape, that's evidence for shared infrastructure
(a small ``persistence`` helper module or similar). Until then,
each script implements its own resumability and the convention
itself is the discipline.
