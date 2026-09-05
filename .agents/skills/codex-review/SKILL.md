---
name: codex-review
description: Run an independent Codex CLI parity review of the majit/pyre diff against local RPython/PyPy, verify findings, fix new regressions, and record justified follow-ups. Use for $codex-review, legacy /codex-review, or requests for a Codex parity review cycle in this repository.
---

# Codex parity review cycle

When this review cycle is requested, run an independent Codex-CLI review: Codex
statically compares the working diff (`git diff upstream/main`) against the local
RPython/PyPy sources and reports porting divergences in four sections. This
skill runs that review and then **acts on the report** so the cycle actually
converges instead of just producing a wall of text.

The split is deliberate:

- **Sections 1 & 2** are problems *this patch* introduced (parity regressions,
  and other new mismatches). They are the cost of the work just done, so they
  are fixed **now**, in this session, before the cycle is considered closed.
- **Sections 3 & 4** are not this patch's fault — section 3 is pre-existing
  debt and section 4 is intended structural adaptation. But "pre-existing" and
  "adaptation" are *classifications, not verdicts*: the reason to surface them
  is to **decide** — per finding — whether to fix now, fix later, or
  consciously leave in place with a cited justification. Each one gets a
  reasoned disposition (Step 4), never an automatic skip. The common outcome is
  a follow-up task, because chasing unrelated old debt would balloon the
  current diff and blur what this cycle changed — but that is the *result of a
  judgment*, not a reflex, and it is overridden when the fix is small, adjacent
  to the work just done, or removes a latent bug this patch can now reach.

## Codex scope and prerequisites

Use `$codex-review`; recognize `/codex-review` as legacy wording. Read the
current worktree's `AGENTS.md` and the sibling [parity skill](../parity/SKILL.md)
before reviewing. A request to import or edit these skills does not run the
review cycle. Respect explicit review-only/no-edit instructions throughout
the session: in that mode verify and report findings without implementing fixes.
The child review is always read-only; only the parent performs authorized fixes.

Confirm `git rev-parse --show-toplevel` and `git status --short` before work.
Keep all paths rooted in that worktree, preserve unrelated user changes, and
confirm the root again before staging. This cycle does not itself request a
commit, push, PR comment, or external issue creation.

## Step 1 — Run the Codex review

The exact review prompt is checked into the repo so the skill and the CI
workflow (`.github/workflows/codex-review.yml`) stay in sync. Read it from
`.github/codex-review-prompt.md` and pass it to Codex. Run from the repo root
so Codex sees the diff and the `rpython/`/`pypy/` trees:

Check `codex` availability and `git rev-parse --verify upstream/main` first.
Do not auto-fetch or substitute `main`/`origin/main`. If the base is absent,
report the missing prerequisite and request the upstream remote information
only if needed to resolve it. Do not fabricate a review if the CLI is missing,
unauthenticated, or fails.

Use a saved brief so retries preserve the exact CI prompt. The added execution
context prevents automatic skill selection in the child from recursively
starting this cycle or applying fixes. Run from the verified repository root:

```bash
review_root="$(git rev-parse --show-toplevel)"
mkdir -p "$review_root/scratchpad/codex-review"
review_dir="$(mktemp -d "$review_root/scratchpad/codex-review/run.XXXXXX")"
review_model='gpt-5.6-terra'
cat > "$review_dir/brief.md" <<'BRIEF'
You are the read-only reviewer for a parent Codex review cycle.
Read AGENTS.md and .agents/skills/parity/SKILL.md as review criteria.
Read parity/references/spec-deviation.md relative to .agents/skills in full
before adjudicating a spec exception. Run applicable read-only pypy3 probes
before an orthodoxy diagnosis. Do not invoke the codex-review cycle, spawn
reviewers, fix findings, or write project files. Report only. Add upstream
symbol names to the current file:line citations required by the prompt.
The following is the shared CI review prompt; keep its four-section output.
BRIEF
cat .github/codex-review-prompt.md >> "$review_dir/brief.md"
codex exec --dangerously-bypass-approvals-and-sandbox -m "$review_model" \
  -C "$review_root" \
  --output-last-message "$review_dir/report.md" \
  "Read $review_dir/brief.md and carry out that read-only review exactly." </dev/null
```

The run directory is under the repository's ignored scratch directory. Use a fresh run directory so an older successful
report cannot be mistaken for a failed run's result. `--output-last-message`
writes only the child's final report; stdout carries progress. Save the exit
status and only consume the report when the command succeeds and all four
headings are present. Missing/truncated output is an incomplete review.

`gpt-5.6-terra` preserves the source skill's reviewer default; honor a user
`--model`/`-m` override. The repository requires both
`--dangerously-bypass-approvals-and-sandbox` and `</dev/null` for noninteractive
CLI delegation. This flag does not enforce read-only access: the brief defines
the child's scope. Inspect the worktree after it finishes; any unexpected
mutation must be inspected before proceeding, without discarding user edits.

Read the exact current `.github/codex-review-prompt.md`; do not maintain a
second copy of its criteria in this skill. Its `git diff upstream/main` file
list, excluding `*.jitstats`, defines the review scope. Include task-related
untracked source only through an explicit authoritative list and tell the
reviewer which files are new; exclude unrelated scratch work. A modified file
does not prove introduction: verify each section-1/2 finding against the base.

If the CLI fails, report the failure and retained brief path. Do not repeatedly
retry unchanged failures or silently replace an independent review with a
self-review. Run another review only after fixes or a resolved failure justify it.

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
`our_file.rs:line ↔ upstream.py:line` citation **with upstream symbol names**; if one is vague, open the cited
files and pin it down before acting.

## Step 3 — Fix sections 1 & 2 in-session by default

These are regressions and new mismatches this patch introduced, so fix them
now unless the user requested review-only operation. Apply the already-loaded [parity skill](../parity/SKILL.md) to these fixes;
keep this review's base and changed-file scope rather than starting another audit cycle:

- For each finding, open the cited `our_*.rs` and the upstream
  `rpython/`/`pypy/` counterpart and confirm the divergence is real (Codex can
  be wrong — verify against the source, don't fix on faith).
- Port back to the upstream structure rather than inventing a new adaptation.
  Section 1 (parity regressions) is highest priority; section 2 next.
- A finding that turns out to be a false positive, or a genuine structural
  adaptation Codex mis-sorted into 1/2, gets reclassified — drop it from the
  fix list and note why. Don't fix something that isn't actually wrong.
- After the fixes, verify the way the repo expects: `cargo check`/`cargo test`
  with `--no-default-features --features dynasm` (the full required test is
  `cargo test --all --no-default-features --features dynasm`), and
  `python3 pyre/check.py` across every host-buildable backend. Follow the parity
  skill and `AGENTS.md` for LLBC refresh, Cranelift checks, and wasm coverage.
  For skill/documentation-only edits, validate those artifacts instead. Don't claim the section is closed until the checks pass.

If a section-1/2 fix is genuinely too large to land in this session (it needs an
unported upstream dependency, or cascades across many files), say so explicitly
and move that single item to a follow-up task (Step 4) with the blocker named —
but the default is to fix it here.

## Step 4 — Adjudicate sections 3 & 4 (decide; never reflex-defer)

Sections 3 (pre-existing mismatches) and 4 (structural adaptations) are not
this patch's fault, but that classification is **not a verdict**. The point of
surfacing them is to reach a reasoned *disposition* on each one. "It's
pre-existing" and "it's an adaptation" are never, by themselves, a reason to
skip — that is exactly the reflex `$parity` Principle 7 forbids
("PRE-EXISTING-ADAPTATION is a fix queue, not an absolution; 'it works today'
is never a sufficient reason").

For **each** finding (verify the citation against the upstream source first —
Codex can misclassify, and a real regression sometimes lands in section 3/4),
assign one disposition:

- **fix-now** — port it back in this session. This is the *default* under
  `$parity` Principle 7 whenever the fix is self-contained and the original
  blocker is gone. Choose it especially when the fix is small, sits in code
  this patch already touches/depends on, or closes a latent bug the current
  work can now reach. A fixed section-3/4 item moves into this cycle's diff
  exactly like a section-1/2 fix; verify the same way (Step 3).
- **won't-fix (documented)** — the divergence is *correct as-is*: a deliberate,
  still-valid structural adaptation (RPython↔Rust language gap, 3.11↔3.14
  opcode difference, GIL/free-threading, no-filesystem/no-libc on wasm, …).
  A CPython-observable exemption must pass all six tests in
  [spec-deviation.md](../parity/references/spec-deviation.md), including the
  free-threaded 3.14t requirement and PyPy module boundary. Confirm the upstream decision point it encodes and ensure an in-code comment
  cites it; if the comment is missing, that documentation *is* the fix and is
  made now when edits are authorized; otherwise record the needed comment.
  Record the justification in the dispositions file.
- **defer (blocked)** — fixing it is right but out of safe session scope: a
  specific, cited, still-real blocker (an unported upstream dependency, a layout
  change cascading across more files than this session can touch, a regression
  rooted in another unported optimization), or it is unrelated old debt large
  enough that pulling it in would balloon and blur this cycle's diff. Only then
  record a follow-up entry in `<review_dir>/follow-ups.md` — one per finding or
  tight cluster, with a stable local ID (`CR-001`, etc.), file and gist, full
  citation including upstream symbols, named blocker, convergence path, and
  acceptance checks. These are local records, not external issue IDs. Preserve
  existing entries when updating a run; do not assume task-management tools
  exist. These deferred items are not worked now.

The bias for unrelated section-3/4 debt is still toward a follow-up task rather
than ballooning the diff — but reaching that outcome requires the judgment
above, not a blanket "section 3/4 ⇒ defer". Stating only "it's pre-existing"
as the reason is a process error.

## Step 5 — Report

Keep the raw `<review_dir>/report.md` unchanged and write the verified
dispositions and check results to `<review_dir>/dispositions.md`. Close with a
short summary:

- Counts per section, each broken down by disposition — fixed-now /
  won't-fix(documented) / deferred(blocked) / reclassified (e.g. `1: 2 fixed,
  2: 1 fixed + 1 reclassified, 3: 1 fixed + 1 won't-fix, 4: 1 won't-fix + 1
  deferred`). A bare "3: 2 deferred" with no per-item reasoning is a process
  error (Step 4).
- What was changed in-session and the verification result.
- The local follow-up entries that were recorded (IDs/titles and path).
- The raw report path (`<review_dir>/report.md`) for reference.

## Relationship to CI

`.github/workflows/codex-review.yml` runs the **same** prompt on every PR push
and posts the four-section report as a sticky PR comment. CI only *reports*;
this skill is the side that *acts*. Keep the prompt change in lockstep: edit
`.github/codex-review-prompt.md` once and both consumers pick it up.
