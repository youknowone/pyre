---
name: codex-review
description: Run a Codex-CLI static-analysis parity review of the current diff against the local RPython/PyPy sources, then act on it — fix the regressions and new mismatches in-session, and file the rest as follow-up tasks. Invoked via `/codex-review`. Use this skill at the END of a work cycle, or whenever the user asks to "run codex review", "codex 리뷰 돌려줘", "리뷰 사이클 돌려줘", "parity review with codex", "check this diff against RPython with codex", or any request to have Codex audit the current changes for RPython/PyPy porting fidelity. Reach for it any time the user wants a Codex-driven parity pass over the working diff, even if they don't say the exact word "codex-review".
---

# Codex parity review cycle

Each development cycle in this repo closes with a Codex-CLI review: Codex
statically compares the working diff (`git diff upstream/main`) against the local
RPython/PyPy sources and reports porting divergences in four sections. This
skill runs that review and then **acts on the report** so the cycle actually
converges instead of just producing a wall of text.

The split is deliberate:

- **Sections 1 & 2** are problems *this patch* introduced (parity regressions,
  and other new mismatches). They are the cost of the work just done, so they
  are fixed **now**, in this session, before the cycle is considered closed.
- **Sections 3 & 4** are not this patch's fault — section 3 is pre-existing
  debt and section 4 is intended structural adaptation. Stopping to fix those
  would balloon the current diff and blur what this cycle changed, so they
  become **follow-up tasks** picked up after the current work is wrapped up.

## Step 1 — Run the Codex review

The exact review prompt is checked into the repo so the skill and the CI
workflow (`.github/workflows/codex-review.yml`) stay in sync. Read it from
`.github/codex-review-prompt.md` and pass it to Codex. Run from the repo root
so Codex sees the diff and the `rpython/`/`pypy/` trees:

```bash
PROMPT="$(cat .github/codex-review-prompt.md)"
codex exec --dangerously-bypass-approvals-and-sandbox -m gpt-5.5 \
  -C "$(git rev-parse --show-toplevel)" \
  --output-last-message /tmp/codex-review-report.md \
  "$PROMPT" </dev/null
```

`--output-last-message` writes just the final four-section report to the file
(stdout still streams Codex's progress). Read `/tmp/codex-review-report.md` for
the report to parse.

Notes:

- `--dangerously-bypass-approvals-and-sandbox` and `</dev/null` are required —
  without them `codex exec` blocks on an interactive approval prompt and hangs.
  See the `/codex` command for the same pattern.
- The prompt is read-only by construction (it asks Codex to *report*, not edit).
  Do not let Codex modify files in this step.
- `-m gpt-5.5` is the default; honor a `--model <name>` the user passes in their
  invocation.
- The diff base is `upstream/main` (the remote base, NOT local `main` or
  `origin/main`). This skill does **not** auto-fetch — `upstream/main` is
  whatever the user last fetched (they sync the `upstream` remote manually). If
  `upstream/main` is missing, stop and ask the user to add the remote
  (`git remote add upstream <URL>`) and fetch it; do not fall back to local `main`.

If `codex` is missing, unauthenticated, or exits non-zero, stop and report that
plainly — do not fabricate a review.

## Step 2 — Parse the four sections

The report uses these verbatim headings (guaranteed by the shared prompt):

```
## 1. Regressions to PyPy parity introduced by this patch
## 2. Other mismatches introduced by this patch
## 3. Pre-existing mismatches (already present before this patch)
## 4. Structural adaptations
```

Split the report on those headings. A section whose body is just `None.` is
empty — skip it. Every finding should carry an
`our_file.rs:line ↔ upstream.py:line` citation; if one is vague, open the cited
files and pin it down before acting.

## Step 3 — Fix sections 1 & 2 in-session (always)

These are regressions and new mismatches this patch introduced, so fix them
now. This is exactly the job `/parity` exists for, so run it under parity
discipline:

- For each finding, open the cited `our_*.rs` and the upstream
  `rpython/`/`pypy/` counterpart and confirm the divergence is real (Codex can
  be wrong — verify against the source, don't fix on faith).
- Port back to the upstream structure rather than inventing a new adaptation.
  Section 1 (parity regressions) is highest priority; section 2 next.
- A finding that turns out to be a false positive, or a genuine structural
  adaptation Codex mis-sorted into 1/2, gets reclassified — drop it from the
  fix list and note why. Don't fix something that isn't actually wrong.
- After the fixes, verify the way the repo expects: `cargo check`/`cargo test`
  with `--features dynasm`, and `python ./pyre/check.py` for end-to-end
  correctness. Don't claim the section is closed until the checks pass.

If a section-1/2 fix is genuinely too large to land in this session (it needs an
unported upstream dependency, or cascades across many files), say so explicitly
and move that single item to a follow-up task (Step 4) with the blocker named —
but the default is to fix it here.

## Step 4 — Defer sections 3 & 4 to follow-up tasks

Sections 3 (pre-existing mismatches) and 4 (structural adaptations) are out of
scope for the current diff. Don't widen this cycle's changes to chase them.
Instead register each as a follow-up task in the harness task system with
`TaskCreate`, so they surface after the current work is wrapped up:

- One task per distinct finding (or per tightly-related cluster). Title it with
  the file and the gist; put the full Codex citation and reasoning in the body.
- Tag section-4 items clearly as *candidate structural adaptations* — many are
  legitimate and intended (Python 3.11↔3.14, CPython-compatible-compiler opcode
  differences, GIL/free-threading, RPython↔Rust language gaps). The follow-up
  task is to *verify and document* the adaptation (cite the upstream decision
  point), not necessarily to "fix" it.
- Section-3 items are pre-existing parity debt: the task is to port them back to
  upstream shape later, per `/parity` Principle 7 (PRE-EXISTING-ADAPTATION is a
  fix queue, not an absolution).

Do not start working these tasks now — they are explicitly for after the
current cycle closes.

## Step 5 — Report

Close with a short summary:

- Counts per section (e.g. `1: 2 fixed, 2: 1 fixed + 1 reclassified, 3: 3
  deferred, 4: 2 deferred`).
- What was changed in-session and the verification result.
- The follow-up tasks that were filed (ids/titles).
- The raw report path (`/tmp/codex-review-report.md`) for reference.

## Relationship to CI

`.github/workflows/codex-review.yml` runs the **same** prompt on every PR push
and posts the four-section report as a sticky PR comment. CI only *reports*;
this skill is the side that *acts*. Keep the prompt change in lockstep: edit
`.github/codex-review-prompt.md` once and both consumers pick it up.
