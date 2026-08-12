# majit environment gate triage

The `MAJIT_*` half of the gate brake. `pyre/gate-triage.md` is the `PYRE_*`
half; the two are separate documents on purpose. majit is versioned separately
and consumed by cel-jit *by rev*, so a majit gate's record has to live where
the gate does — a row kept in a pyre document goes stale on every majit gate
addition, and only pyre's test would ever notice.

`pyre/pyrex/tests/gate_triage_complete.rs` enforces both directions against
this file: every `MAJIT_*` name read from a workspace member needs a row here,
and every name here needs a reader. A row outliving its reader is the quiet
half — `PYRE_FBW_REC_UNROLL` sat in the pyre list from 2026-07-06, when PR#374
deleted `fbw_unroll_bound()`, until that test went looking.

## ⛔ What this file does NOT yet record

**Almost every row below is missing its retirement condition.** The charter
(§3.6) is that a gate is a staging area, not a home, and this list is meant to
say what goes and when. A row's retirement condition reads **UNRECORDED**
unless that gate's own doc comment already states one. When this file was
written exactly one gate did — `MAJIT_OPREF_VARIANT_AUDIT`, whose module
header names the structural change that ends it. `rg -c 'UNRECORDED' ` gives
the live count; do not trust a number written here.

The rest are empty rather than filled because the author of this file could
not answer it for those gates without inventing the answer, and an invented
retirement condition would satisfy the brake permanently while describing
nothing — strictly worse than the visible hole, because the hole is legible as
an absence and a plausible sentence is not. Each gate's owner owes one line.

⚠ And note what the brake does and does not check: it compares NAMES. A row
with an empty description satisfies it exactly as well as a filled one. Do not
read this file being green as evidence that it says anything.

Descriptions below are quoted or condensed from the accessor's own doc comment
where one exists, and are the accessor's name alone where one does not. None
of them were composed for this file.

⚠ A `What it does` entry may be the **accessor's** doc comment rather than the
**gate's**, and the two are not the same thing — an accessor's doc describes what
the function computes. Where the enclosing function's doc turned out to say
nothing about the gate, the row now reads `UNRECORDED` with a labelled statement
of what the read site literally does. A row that is honestly blank is worth more
than one that is plausibly wrong.

## Conventions this file is itself subject to

⛔ **Any prose added here is parsed as data.** `gates_documented_in` collects
names from **every line**, not only headings, so a commit subject, an example, or
a cautionary sentence all mint gate names. A name with no reader fails
`every_live_triage_entry_still_has_a_reader`, and a name that is a real gate
silently satisfies a row it was never meant to. Before committing an edit, sweep
the file for `MAJIT_` tokens that are not headings; the expected count is zero.
Mutation-proven: re-adding the one name elided below turns the brake red.

⛔ **Any name-keyed search over a prefixed namespace needs a terminator.** This
bit both tools used to build this file. `git log -S MAJIT_DUMP` is satisfied by
`MAJIT_DUMP_BYTECODE`; `contains("PYRE_A")` is satisfied by
`PYRE_ANCHOR_STRICT`. The brake already guards its side — see
`a_prefix_appearing_inside_a_longer_name_is_not_a_gate_of_that_namespace`, which
exists because `PYRE_MAJIT_STATS_*` would otherwise read as a majit gate — and
the derivation side uses `-S '<name>[^_A-Z]' --pickaxe-regex`. Neither guard
helps the other; a new name that is a prefix of an existing one needs both.

⛔ **A gate's namespace is not its directory, and the split is not marginal.**
The brake routes a name to a document by its prefix, never by where it is read,
and the tree does not line up with the prefixes in either direction. Measured at
HEAD: **32 distinct `PYRE_*` names are read from 50 sites under `majit/`**,
spanning seven majit crates and four of its test files — `majit-rlib`, the crate
this catalog gained most recently, carries **no** `MAJIT_*` gate at all and does
read a `PYRE_*` one. Seven names in this file have read sites under `pyre/`, five
of them exclusively. ⇒ Reading the tree layout to decide which document owns a
crate's gates gets it wrong for both namespaces; the prefix is the only answer,
and the two documents are not a partition of the two directories.

⛔ **A gate's default is not visible at the line that names it.** Most gates here
read as `env::var_os("NAME").is_some()`, where the default is legible in place —
which makes it tempting to classify a whole list by pattern-matching the read
line. That misreads exactly the gates whose semantics were factored into a helper,
which is to say the carefully built ones. Demonstrated on the `PYRE_*` side: a
scan for inverted-default patterns across the names listed in
`pyre/gate-triage.md` §6c returned **no hits at all**, including for
`PYRE_LLBC_STRICT`, whose unset-means-enforce default is decided by
`classify_strict()` in `llbc_fingerprint.rs` and never appears beside the name.
The scan could not see the one gate it had been written to find, and a unanimous
zero reads as a clean result rather than a dead instrument. ⇒ Take a gate's
default from the enclosing accessor's return, not from its `env::var` line — and
when a census over a name list comes back unanimous, put a member whose answer
you already know into the same run.

⛔ **A `Read sites:` row is a count and a file list; the symbol lives one field
below.** These rows used to carry line numbers, and the line numbers went stale
silently: a rebase moved every one of them while every cited file still existed
and every cited line stayed inside its file, so nothing about a dead citation
looked wrong from here. A file path is not perishable that way, and neither is a
symbol. ⇒ Re-location is `rg <accessor> <file>` — the `Read sites:` row supplies
the files, the `Accessor:` field supplies the symbol. Do not write the accessor
into the `Read sites:` row: that is one fact stated twice, one line apart, and
the two copies are then free to disagree. Where a gate's reads sit in more than
one enclosing function, name every one of them in `Accessor:` and pair none of
them with a file — pairing puts the file list in a second place and re-opens the
same drift. Where a gate has no single accessor, say so in that field
explicitly: a stated absence is a third value, and an empty field reads as
unexamined rather than as none.

## The `Introduced:` line, and what it is not

Every row carries the commit that first put the gate's name in the tree, from
`git log -S<name> --reverse --no-merges` over the full history with no pathspec.
It is a derivation with a citable source, so a row's owner can go read what the
commit said the gate was for instead of inferring it from the code around the
read site.

⚠ **A sha is not a retirement condition.** It supplies the evidence an owner
needs to write one; it does not write one, and no `Retirement condition` line
below was filled from it. Where an introducing commit does state the condition,
quote it and cite the sha — where it does not, the row stays `UNRECORDED`.

⛔ `git log -S` matches **substrings**, so a gate whose name is a strict prefix
of another (`MAJIT_DUMP` inside `MAJIT_DUMP_BYTECODE`, `MAJIT_LOG` inside
`MAJIT_LOG_OPT`) is also matched by its sibling's commits, and the first hit can
belong to the sibling. Those two were re-derived with
`-S '<name>[^_A-Z]' --pickaxe-regex`; `MAJIT_DUMP`'s answer changed and
`MAJIT_LOG`'s did not. Any later addition that is a prefix of an existing name
needs the same treatment.

⚠ This file is itself part of the history it cites: `1d670fa7c9c` (re-pointed
2026-08-11, DOOMED; subject: "gate-triage: catalog the MAJIT_* gates and arm the
brake's second namespace") put all 48
names into the tree, so an unrestricted `git log -S` returns one more commit per
gate than it did before the catalog landed — and one more again for every commit
that has edited a row since. Every introduced-and-never-revisited mark below is
therefore taken with this file excluded:

    git log HEAD -S '<name>' -- . ':!majit/gate-triage.md'

with the prefix rule above applied to `MAJIT_DUMP` and `MAJIT_LOG`. Re-run over
all 48 rows on 2026-08-10: every `Introduced:` sha is the earliest such commit,
and all 16 marked rows still return exactly one. Omitting the pathspec makes
every marked row read as revisited; omitting the terminator makes `MAJIT_DUMP`
return ten commits instead of three.

## ⛔ How to read an `Introduced:` sha — most of them are mortal

A sha in this file is an **annotation, not the citation**. The subject beside it
is the citation, because a rebase rewrites shas and leaves subjects alone. Find a
row's commit by its subject:

    git log --format=%h -1 --fixed-strings --grep='<the subject on the row>'

Shas here fall in three classes, and only the first is durable:

| class | test | lifetime |
|---|---|---|
| DURABLE | `git merge-base --is-ancestor <sha> origin/main` RC=0 | permanent — merged upstream |
| **DOOMED** | on `HEAD`, not on `origin/main` | **dies at the next rebase** |
| DEAD | neither | already rewritten; *usually* still `git show`s, which proves nothing |

⚠ **A dead sha resolves.** `git cat-file -e` and `git show` both succeed for
commits reachable only from the reflog — that was true for 32 of 32 measured on
2026-08-11 — so "the sha still works" is not a check. Use `--is-ancestor`.

### ⛔ …but `--is-ancestor` alone answers ONE question, and a citation asks three

Run that table on its own and healthy shas come back DEAD. Censused 2026-08-11
over every hex token in 9979 tracked files: 114 resolve to commits, 33 classify
DEAD, and **3 of the 33 are not rot at all** — each for a different reason:

| reads DEAD | actually carried by | why it is healthy |
|---|---|---|
| `1f81807bcfde` (`majit-rlib/src/rbigint.rs:5`) | `pypy/main` +15 refs | **another project.** Our mainline was never the applicable history |
| `3ccbd1f5f4d` (`cel-jit/cel/Cargo.toml`) | `origin/cel` | **our repo, another branch.** A cargo `rev =` pin is *supposed* to name the feature branch |
| `a4e191f71b5` (the harvest) | `upstream/wasmi` | **ours, pre-squash.** A stale remote pins the object so it is never gc'd — while the harvest's squash row for it stays correct |

⚠ The `path:line` coordinates in that first column are a **frozen census record**
and are deliberately **not** converted to the file-plus-accessor form the
`Read sites:` rows below use. They state where each sha stood on the census date
beside them; rewriting them into a present-tense locator would restate a dated
measurement as a standing fact, and the date is the whole reason the counts are
readable. Re-locate any of these by searching for the sha itself — a hex token is
its own search key, which is exactly what a line number is not.

The last one is why this is a wrong shape rather than a missing case: it is alive
on a remote **and** correctly filed as rot, both at once. Three questions had
been riding on one call:

| question | oracle | decides |
|---|---|---|
| will the next reader resolve it? | `git branch -r --contains <sha>` | whether the citation can be followed at all |
| does it name a tree on our mainline? | `--is-ancestor <sha> origin/main` | whether the claim is about code we ship |
| is it about *this* repository? | which remote carries it | whether the ancestry test applies |

⛔ Do not use object absence (`git cat-file -e`) to spot a foreign sha. A foreign
object that has been **fetched** is present — the pypy one above resolves here —
so over the same 33 that test answers "off-branch rot" 33 times, including for
the one genuinely foreign sha. Its real job is the **precondition**, because
`branch -r --contains` errors on an object that is gone. The harvest's rule
section carries the full five-valued form; this table is only about
`Introduced:` shas.

⚠ **UNRESOLVABLE is a fourth state, not a strong form of DEAD.** `git gc` can
take a commit outright — `flatten-path4-closure-roadmap.md` cites two, verified
five ways against a live positive control. For those, **subject recovery, the
repair this whole section rests on, is unavailable.** Harvest subjects while the
objects still exist; that preventative's failure mode is invisible until it is
already too late.

⚠ Count hygiene: the 33 here and the 33 rows in `citation-rot-harvest.md` are
**different sets** — `1f81807bcfde` is in this one and not that one. Equal
totals are the easiest way for two populations to look reconciled without having
been intersected.

The 2026-08-11 rebase rewrote every branch-local sha in this file. The rows below
were re-pointed on that date by matching content (`git patch-id --stable`), not
by trusting a subject alone; each annotated sha was DOOMED when written, so
expect it to be stale again after the next rebase and re-derive from the subject.

### ⛔ Three things a sha can be doing, and only one of them is repairable

Re-pointing a dead sha to its successor is correct for exactly one of these.
Decide which kind you have **before** substituting, by asking what the sha was
*for* rather than what it resolves to:

| the sha is… | when it rots | repair |
|---|---|---|
| an **attribution** — "this commit made that change" | the change survives under a new sha | **RE-POINT.** Substitution is sound: the successor carries the same patch |
| a **provenance** — "this reading was taken on that tree" | the tree is gone and a squash-merge does not restore it | **ANNOTATE.** Name the run, say the figure cannot be re-taken. A successor sha here relocates a measurement onto a tree it never ran on — a false statement that reads as a repair |
| a **verification** — "the suite was green at that tree" | same, but the claim is about a *result* | **RE-RUN.** Nothing can be substituted: the successor is a different tree and the measurement never happened there |

⚠ The third is the dangerous one, because a rotted verification still reads as
evidence. On 2026-08-11 the three check.py runs this branch cited as proof the
rebase was verified were all measured on trees the branch no longer has —
`--is-ancestor` rc=1 against both `HEAD` and `origin/main`, with a resolvable
object, so genuinely DEAD rather than merely unresolvable. A green measured on a
tree nobody holds is not a weaker green; it is not a green about this tree at all.

#### ⛔ The rotted verifications are not in this file — they are in the TASK LIST

This file can be re-read. A closed task is not. Re-measured 2026-08-11, unpiped,
three-way, with `git branch -r --contains` to rule out the foreign class:

| sha | cited by | verdict |
|---|---|---|
| `ade128fd387` | #83 "REBASE VERIFIED — 1244 PASS / 1 FAIL" | rc=1, off-HEAD, no remote ref |
| `ea3d5a1ea74` | #87 "cranelift 416/416, wasm 412/412" | rc=1, off-HEAD, no remote ref |
| `76de01a6eaf` | #87 (the pyre-jit re-extract) | rc=1, off-HEAD, no remote ref |
| `553045da4f6` | #89 "6/6 green, CARGO_EXIT=0" | rc=1, off-HEAD, no remote ref |
| `c7752760030` | #17, #27, #64 | rc=1, off-HEAD, no remote ref |

**5 of 5 dead**, and the search was not exhaustive — those are the ones a scan of
`[completed] ✅` rows turned up, so treat 5 as a floor.

⭐⭐⭐ **A `[completed] ✅` carrying a sha is a claim about a TREE, and a rebase
voids it without editing one character of the task.** That is what makes this
class worse than ordinary citation rot: rot in a source comment is found by
anyone who reads the comment, but a closed task is *read once, by the person who
closed it*. Nothing about #83 looks stale — it is precise, it names a real run,
its numbers were true — and it is now unfalsifiable. **None of these tasks is
lying; all of them are unverifiable.**

⇒ Two consequences worth acting on rather than noting:

- **A verdict that cites a branch-local sha has a shelf life measured in hours
  here** — two rebases landed within 6.5 hours on 2026-08-11. Prefer the durable
  citation forms: the subject for a change, byte-identity for a measurement
  (`these N files are unchanged base→HEAD` survives both rebase and gc), and a
  sha only as a dated annotation that says so.
- **Re-running is the only repair**, and it is tracked as its own task rather
  than folded into the tasks it invalidates — a closed task that quietly acquires
  a re-run obligation is how the obligation gets lost. See the branch
  re-verification task.

⚠ Do not "fix" these by re-pointing them at HEAD. The successor tree is a
different tree and the suite never ran there; substituting relocates a result
onto a tree that never produced it, which is the provenance error one row up
wearing a verification's clothes.

⭐ A fourth disposition exists and is not a repair: **keep, labelled.** A sha
cited *as an example of rot* must stay dead — `scripts/check-backend-edge.py`
holds one deliberately, because it is the evidence for the paragraph that
predicted it would die. Re-pointing it would delete the finding.

⚠ A gate marked as introduced-and-untouched-since is a **fact about the history**,
not a verdict. It says the name has never been revisited, which is where a
retirement question is most likely to be answerable — not that the gate is dead.
One of them, `MAJIT_PORTAL_INLINE`, was proposed for deletion during this work
and is live: `portal_inline_experiment_enabled()` selects a real-pointer-vs-null
on the deopt path. `MAJIT_LOG_OPT` was proposed on the same grounds and is not
even in this group — its printer gained three callers in `780c5bee112`
(re-pointed 2026-08-11, DOOMED; subject: "majit: make MAJIT_LOG_OPT dump the
optimized body at all three compile paths"). Both
proposals cited a task that recorded the gate as dead, and both were true when
filed and false by the time they were cited. **Check callers at HEAD** before
reading any row as a deletion candidate; a task asserting a property of the code
is evidence about the day it was written.

## §1 Live gates

### `MAJIT_BH_DEBUG`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `bh_debug_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Introduced: 2026-06-17 in `dffea4d86b0` — majit: typed state-field JIT — ref fields, virtualized arrays, green-pc dispatch (#195)
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_BH_NULL_ARG`

- Read sites: 1 — `majit/majit-metainterp/src/blackhole.rs`
- Accessor: `bh_null_arg_report()`
- What it does: `MAJIT_BH_NULL_ARG`: report a null ref argument about to be handed to a residual call, with the jitcode coordinate, before the callee can dereference it.  Some ABIs pass a legitimate null sentinel (e.g. the CallFn `null_or_self` slot), so this reports rather than aborts.
- Introduced: 2026-08-05 in `eaad8f9dfe5` — jit: fix symbolic-fnaddr misclassification on aarch64 Linux; blackhole null-arg diagnostic (#1030)
  ⚠ Introduced here and never revisited: as of 2026-08-10 no other commit's diff changes an occurrence of the name, this file excluded — see the conventions above.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_BRIDGE_DEBUG`

- Read sites: 5 — `majit/majit-macros/src/jit_interp/codegen_state.rs`, `majit/majit-metainterp/src/lib.rs`
- Accessor: `ref_identity_slots_end()`; also read inline in `setup_bridge_sym()` and `bridge_debug_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Introduced: 2026-06-26 in `769911b5c57` — majit: trace-compile pipeline fixes — O(1) compile, is_gc_managed guard gating, pool-array reads, macro bridges (#263)
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_BRIDGE_DIAG`

- Read sites: 2 — `majit/majit-macros/src/jit_interp/codegen_state.rs`, `majit/majit-metainterp/src/resume_box_reader.rs`
- Accessor: `setup_bridge_sym()`; also read inline in `replay_pending_fields()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Introduced: 2026-07-31 in `5a9f4e7c1f1` — jit: run a tracing abort through convert_and_run_from_pyjitpl; gate write barriers on the collector's descriptor (#895)
  ⚠ Introduced here and never revisited: as of 2026-08-10 no other commit's diff changes an occurrence of the name, this file excluded — see the conventions above.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_BRIDGE_FUEL_LOG`

- Read sites: 1 — `majit/majit-metainterp/src/jitdriver.rs`
- Accessor: read inline in `bridge_fuel_take()`
- What it does: "reports each one taken" — the one clause `MAJIT_MAX_BRIDGES`'s doc comment spends on it; it has none of its own. Read off the site, not quoted: set, each bridge that takes fuel prints `@@@FUEL bridge #<n>` to stderr, so the log line and the bisection index are the same number.
- Introduced: 2026-08-10 in `36666ef933c` — jit: a dead-var link-arg trim scoped on the wrong reachability, a shared-receiver locals_w_mut!, bridge opt-fuel, and 91 stale CPython-suite baseline entries (#1138)
  ⚠ Introduced here and never revisited: as of 2026-08-10 no other commit's diff changes an occurrence of the name, this file excluded — see the conventions above.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_CLOSEDBG`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `closedbg_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Introduced: 2026-07-03 in `2cc8179f2a7` — single-pass / walker-as-tracer tracing scaffold (aheui logo --jit) (#311)
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_CL_GCSTORE_LOG`

- Read sites: 1 — `majit/majit-backend-cranelift/src/compiler.rs`
- Accessor: read inline in `do_compile()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Introduced: 2026-05-01 in `27e96e3d93e` — Activate PyPyJitDriver extra_reds=[ec], bridge resume + cranelift fib_recursive
  ⚠ Introduced here and never revisited: as of 2026-08-10 no other commit's diff changes an occurrence of the name, this file excluded — see the conventions above.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_COVERAGE_AUDIT`

- Read sites: 1 — `majit/majit-translate/src/codewriter/assembler.rs`
- Accessor: `assemble_with_callcontrol()`
- What it does: "Pyre-only diagnostic: under `MAJIT_COVERAGE_AUDIT=1` enumerate every Variable referenced in `ssarepr.insns` that has no regalloc coloring in any class. Complements the `MAJIT_COVERAGE_PANIC=1` path (which panics at the first gap hit during `write_insn`) by surfacing the full per-graph gap catalogue in one build." — quoted from the comment directly above the read at `assembler.rs:437-444`.
  ⚠ That range is kept as a line span deliberately and is **not** a relocation
  recipe: it states how far the quoted comment runs, which is what lets a reader
  tell a complete quotation from a truncated one. The `Accessor:` field above is
  what locates the comment; the span only says how much of it is reproduced here.
- Introduced: 2026-04-24 in `e6eb6cadecf` — jit_codewriter/assembler + build.rs: MAJIT_COVERAGE_AUDIT walker + deterministic pipeline.insns serialization
  ⚠ Introduced here and never revisited: as of 2026-08-10 no other commit's diff changes an occurrence of the name, this file excluded — see the conventions above.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_COVERAGE_PANIC`

- Read sites: 1 — `majit/majit-translate/src/codewriter/assembler.rs`
- Accessor: read inline in `assemble_with_callcontrol()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Introduced: 2026-04-23 in `bd3f98d09e2` — eval-restack: canonical op-shape pass + liveness must-definedness
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_DEBUG_DECLARES`

- Read sites: 1 — `majit/majit-backend-cranelift/src/compiler.rs`
- Accessor: read inline in `do_compile()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Introduced: 2026-03-21 in `46dd371dacc` — Debug: detect undeclared variables in resolve_opref
  ⚠ Introduced here and never revisited: as of 2026-08-10 no other commit's diff changes an occurrence of the name, this file excluded — see the conventions above.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_DIAG`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `diag_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
  ⚠ A tally surfaced by this gate can name an outcome while counting only the
  attempts that reached it. `bridge_declined_close` (slot 50) is bumped at three
  sites in `jitdriver.rs` — locate them with `rg -n 'mc_diag_bump\(50\)'`, they
  are ~400 lines apart in a file that moves. All three count the same event: a
  close whose compile attempt returned `Declined`. Two of them read a `result`
  bound directly from `close_bridge(...)`, so reaching the arm already implies an
  attempt. The third reads a `result` **initialized** to `Declined` and only
  conditionally overwritten, so it needs its `if attempted` guard to avoid
  counting the initializer; that guard is deliberate and its reason is at the
  site.
  ⇒ The predicate to hold onto: **the slot counts declined ATTEMPTS, not refused
  CLOSES.** A close the gate declines to attempt runs the state reset beside the
  bump. Widening the bump to match the reset would not fix that — it would make
  the slot count the initializer — so the missing population needed its own slot,
  not a wider one. `bridge_unattempted_close` (slot 67) is that slot, added on
  the `else` of the same guard by `cd5875a6edc` (re-pointed 2026-08-11, DOOMED;
  subject: "majit-metainterp: count a close the gate never attempted under its
  own slot"); before it, that close was counted
  at no site under no name.
  It moves under `cargo test -p majit-metainterp`; it does not move under any
  subject in `majit/examples/**`. (ex-gates, #65 — supersedes this note's earlier
  claim that the three sites admit on different conditions; they do not, and the
  earlier text was written from the one site then read.)
  ⚠ **Read either slot against its denominator, which is tiny.** Instrumenting
  the `match result` these two slots hang off shows the whole block is reached
  **twice** in the ~1500-test `majit-metainterp` suite, both times with
  `attempted` true. So a zero on 50 or 66 is nearly uninformative about the JIT's
  behaviour: it mostly reports that the corpus does not drive this path, not that
  the event does not occur. The one fixture that reaches slot 67 exists because
  it was written for it (`interp_origin_close_into_an_uncompiled_target_is_not_a_declined_attempt`);
  nothing else in the tree does.
- Introduced: 2026-06-22 in `b7c3d792210` — box-identity: BoxRef re-key + Operand::Box drain to zero; GC-manage object wrappers (#222)
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_DUMP`

- Read sites: 1 — `majit/majit-backend-dynasm/src/lib.rs`
- Accessor: `majit_dump_enabled()`
- What it does: Whether `MAJIT_DUMP` is set, cached at first access.
- Introduced: 2026-04-08 in `e963c88b42d` — dynasm: fix aarch64 FP register save, trampoline bridge logic
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_DUMP_BYTECODE`

- Read sites: 1 — `pyre/pyre-jit/src/eval.rs`
- Accessor: `dump_bytecode_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Introduced: 2026-03-27 in `730ffbc9766` — Squashed commit of the following:
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_DUMP_CLIF`

- Read sites: 2 — `majit/majit-backend-cranelift/src/compiler.rs`
- Accessor: read inline in `do_compile()`, at both sites
- What it does: **UNRECORDED** — no doc comment at the read site.
- Introduced: 2026-04-06 in `b0783026f7e` — Emit SameAs for label args without preamble definitions
  ⚠ Introduced here and never revisited: as of 2026-08-10 no other commit's diff changes an occurrence of the name, this file excluded — see the conventions above.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_DUMP_LIVENESS`

- Read sites: 1 — `majit/majit-macros/src/jit_interp/jitcode_lower/liveness.rs`
- Accessor: `maybe_dump_liveness()`
- What it does: Print per-marker live sets to stderr when `MAJIT_DUMP_LIVENESS` is set in the proc-macro build environment. `label` is the lowerer scope being dumped (e.g. helper name) so concurrent expansions are distinguishable.
- Introduced: 2026-05-01 in `b9f47181e8c` — state-field JIT snapshot + observer + Phase 4 Epic B per-pc liveness infra
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_DUMP_SSAREPR`

- Read sites: 1 — `pyre/pyre-jit/src/jit/assembler.rs`
- Accessor: `dump_assembled_ssarepr()`
- What it does: Print the assembled instruction stream, byte position first, for graphs whose name matches `MAJIT_DUMP_SSAREPR`. A blackhole failure reports a raw `(jitcode, position)` pair; without the stream there is no way back from that byte offset to the op that wrote it or to the register operands it reads.  The env lookup is cached because `try_assemble` runs per graph on the tracing path.
- Introduced: 2026-08-01 in `93f455ce984` — jit: close the Context.run finally-skip (five divergences) and harden the exception-channel boundaries (#937)
  ⚠ Introduced here and never revisited: as of 2026-08-10 no other commit's diff changes an occurrence of the name, this file excluded — see the conventions above.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_FAILVALS`

- Read sites: 1 — `majit/majit-metainterp/src/jitdriver.rs`
- Accessor: `failvals_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Introduced: 2026-07-03 in `2cc8179f2a7` — single-pass / walker-as-tracer tracing scaffold (aheui logo --jit) (#311)
  ⚠ Introduced here and never revisited: as of 2026-08-10 no other commit's diff changes an occurrence of the name, this file excluded — see the conventions above.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_FIELD_POS_UNRESOLVED`

- Read sites: 1 — `majit/majit-ir/src/descr.rs`
- Accessor: `field_position_unresolved_limit()`
- What it does: How many rows the knob asks for: `MAJIT_FIELD_POS_UNRESOLVED=<n>`, or the whole table for `1` / any non-numeric value. `None` when unset. A cap is a parameter, not a constant. `size_shell_owner_sample`'s sibling diagnostic hardcodes 24 and prints it against a count of 155 — so the rows it shows are the alphabetically-first sixth of a `BTreeSet`, which is the one thing a reader must not compute a proportion from. Whoever asks a census for names is asking a question about the whole population; let them say how much of it they want.
- Introduced: 2026-08-08 in `8c57bb2b206` (re-pointed 2026-08-11, DOOMED) — majit-ir/pyre: name the field_pos_unresolved mints
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_GC_BH_PROBE`

- Read sites: 1 — `majit/majit-gc/src/lib.rs`
- Accessor: `bh_probe_enabled()`
- What it does: Whether the blackhole-object probe is enabled.
- Introduced: 2026-08-06 in `d1fef848351` — jit: exception-path and iteration inlining, with an object-strategy args_w (#1033)
  ⚠ Introduced here and never revisited: as of 2026-08-10 no other commit's diff changes an occurrence of the name, this file excluded — see the conventions above.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_GC_DRAIN_CENSUS`

- Read sites: 1 — `majit/majit-gc/src/lib.rs`
- Accessor: `drain_census_dump_interval()`
- What it does: Set `MAJIT_GC_DRAIN_CENSUS` to a positive integer to also dump the running summary every that many collections. The end-of-run summary is unreachable for the runs this census is most needed on — a collection storm that has to be killed rather than waited out — so those need the periodic line.
- Introduced: 2026-08-04 in `7a7d9174088` — jit: resume past the residual at a forced-vable escape; narrow the nested-break gate (#945)
  ⚠ Introduced here and never revisited: as of 2026-08-10 no other commit's diff changes an occurrence of the name, this file excluded — see the conventions above.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_GC_LIFETIME_LOG`

- Read sites: 1 — `majit/majit-gc/src/lib.rs`
- Accessor: `gc_lifetime_log_enabled()`
- What it does: `MAJIT_GC_LIFETIME_LOG` — trace remembered-set adds and old-gen frees. Read once.  The gate sits in the write barrier and the old-gen sweep, and `std::env::var_os` takes the environment lock and scans it linearly on every call, so asking per event costs whether or not the variable is set.  Same shape as `majit_metainterp::majit_log_enabled`.
- Introduced: 2026-07-30 in `e5546b2ed36` — jit: enforce the recursion-unroll bound once; frame and tuple owner roots (#887)
  ⚠ Introduced here and never revisited: as of 2026-08-10 no other commit's diff changes an occurrence of the name, this file excluded — see the conventions above.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_GC_NURSERY_POISON`

- Read sites: 2 — `majit/majit-gc/src/nursery.rs`, `majit/majit-gc/src/oldgen.rs`
- Accessor: `new()`
- What it does: **UNRECORDED** — no doc comment describes the gate. Read off the sites, not quoted: the two reads initialise `poison_on_reset` (`nursery.rs`) and `poison_on_alloc` (`oldgen.rs`).
- Introduced: 2026-07-14 in `13c2cdf41b2` — gc: incminimark parity — nursery zero-fill removal, ArenaCollection port, incremental sweep (#516)
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_GC_STRESS`

- Read sites: 1 — `majit/majit-gc/src/collector.rs`
- Accessor: read inline in `with_config()`, behind `#[cfg(feature = "gc_stress")]`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Introduced: 2026-06-12 in `ca1c642f18b` — mapdict instance storage, method cache, JIT attr/call ops
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_GUARDLOG`

- Read sites: 2 — `majit/majit-metainterp/src/jitdriver.rs`, `majit/majit-metainterp/src/pyjitpl.rs`
- Accessor: `guardlog_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Introduced: 2026-07-03 in `2cc8179f2a7` — single-pass / walker-as-tracer tracing scaffold (aheui logo --jit) (#311)
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_GUARD_CENSUS`

- Read sites: 2 — `majit/majit-metainterp/src/lib.rs`, `pyre/pyrex/src/lib.rs`
- Accessor: `guard_census_enabled()`; also read inline in `maybe_print_jit_stats()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Introduced: 2026-08-04 in `7a7d9174088` — jit: resume past the residual at a forced-vable escape; narrow the nested-break gate (#945)
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_HEAPDBG`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `heapdbg_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Introduced: 2026-07-03 in `2cc8179f2a7` — single-pass / walker-as-tracer tracing scaffold (aheui logo --jit) (#311)
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_J2PLAN_LOG`

- Read sites: 2 — `majit/majit-backend-dynasm/src/aarch64/assembler.rs`, `majit/majit-backend-dynasm/src/lib.rs`
- Accessor: `majit_j2plan_log_enabled()`; also read inline in `_assemble()`
- What it does: Whether `MAJIT_J2PLAN_LOG` is set, cached at first access.
- Introduced: 2026-04-29 in `2a416afb259` — Add j2 planning path to dynasm regalloc
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_LEAF3_PROV`

- Read sites: 1 — `majit/majit-metainterp/src/resume.rs`
- Accessor: `leaf3_prov_enabled()` — the whole of it: a five-line function whose only statement is a `LazyLock` holding the sole `std::env::var` read. ⚠ Relocate by the accessor, not by the gate name: the name occurs a second time in the same file, as a comment at the census site, and a bare name search returns both.
- What it does: `=1` emits one `[leaf3-prov]` line per `consume_vable_info` call naming the vable identity slot's resume tag, its value, and whether it is `NULLREF`. Census, not a check: the assert at the same site refuses one value on the override arm, while this reports the whole distribution on every call. Off, it is one cached bool read. Count with `rg -c '\[leaf3-prov\]'`; the numerator is `rg -c 'nullref=yes'`.
- Introduced: 2026-08-10 — find by symbol: `git log -S 'MAJIT_LEAF3_PROV' -- majit/majit-metainterp/src/resume.rs`.
  ⚠ Cited by symbol rather than sha deliberately: this landed on a branch whose shas do not survive its next rebase, and a dead sha in this column reads exactly like a live one.
- Retirement condition: Retires when leaf 3 closes — when the unseeded-snapshot route into `_number_boxes` is either refused outright or shown unreachable. Until then this is the only instrument that separates those two, because a silent run is consistent with both "`NULLREF` cannot occur here" and "`NULLREF` was not observed here", and they have opposite consequences.

### `MAJIT_LLBC_EXTRACTION`

- Read sites: 2 — `pyre/pyre-jit-trace/build.rs`
- Accessor: `main()` — one `cargo::rerun-if-env-changed=` declaration and one `env::var_os` read, on adjacent lines
- What it does: **UNRECORDED** — no doc comment describes the gate; the one above `main()` describes the build script. Read off the site, not quoted: `=1` calls `emit_llbc_extraction_placeholders()` and returns early, so the extraction does not run.
- Introduced: 2026-07-16 in `6564959c41f` — jit: make MaJIT portal translation driver-generic (#573)
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_LOG`

- Read sites: 19 — `majit/majit-backend-cranelift/src/compiler.rs`, `majit/majit-backend-dynasm/src/lib.rs`, `majit/majit-gc/src/lib.rs`, `majit/majit-gc/src/rewrite.rs`, `majit/majit-ir/src/debug.rs`, `majit/majit-metainterp/src/lib.rs`, `majit/majit-trace/src/logger.rs`
- Accessor: no single accessor — `majit_log_enabled()` is defined once per crate and several sites read the environment inline; relocate with `rg -w MAJIT_LOG <file>`.
- What it does: Whether `MAJIT_LOG` is set, cached at first access.  Mirrors PyPy's `PYPYLOG` env-var check (`rpython/rlib/debug.py:31-38`).
- Introduced: 2026-03-11 in `10142a8fdc5` — add trace dump via MAJIT_LOG env var and Display for Op
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_LOG_JTET`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `log_jtet_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Introduced: 2026-04-07 in `4b6f1d54d7d` — unroll/virtualstate: opt-in MAJIT_LOG_JTET for jump_to_existing_trace failures
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_LOG_OPT`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `log_opt_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Introduced: 2026-03-18 in `472bbc9912d` — RPython parity: unicode force, chain following, GC methods
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_MACRO_DEBUG`

- Read sites: 8 — `majit/majit-macros/src/jit_interp/jitcode_lower/dispatch.rs`, `majit/majit-macros/src/jit_interp/jitcode_lower/lower_stmt.rs`
- Accessor: `try_inline_dispatch_arm()`; also read inline in `lower_dispatch_chain()`, `lower_return_stmt()` and `lower_stmt_fallback()`
- What it does: **UNRECORDED** — no doc comment describes the gate. Read off the sites, not quoted: every read guards an `eprintln!` and nothing else. The prose around them documents the lowering decisions being printed, not what the gate is for.
- Introduced: 2026-06-17 in `dffea4d86b0` — majit: typed state-field JIT — ref fields, virtualized arrays, green-pc dispatch (#195)
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_MAX_BRIDGES`

- Read sites: 1 — `majit/majit-metainterp/src/jitdriver.rs`
- Accessor: `bridge_fuel_take()`
- What it does: `=N` (diagnostic): allow the first N bridge compilations and behave as `MAJIT_NO_BRIDGE` from then on. Bisecting N names the bridge whose compilation first produces a wrong value, at seconds per run rather than a rebuild per arm. Consumes fuel only when the rest of `should_bridge` already held, so the count is bridges actually taken — place it last in the `&&` chain.
- Introduced: 2026-08-10 in `36666ef933c` — jit: a dead-var link-arg trim scoped on the wrong reachability, a shared-receiver locals_w_mut!, bridge opt-fuel, and 91 stale CPython-suite baseline entries (#1138)
  ⚠ Introduced here and never revisited: as of 2026-08-10 no other commit's diff changes an occurrence of the name, this file excluded — see the conventions above.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_MINT_INDEX_CENSUS`

- Read sites: 1 — `pyre/pyre-jit-trace/build.rs`
- Accessor: read inline in `real_main()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Introduced: 2026-08-09 in `05ed5528648` (re-pointed 2026-08-11, DOOMED) — majit: record whether a fielddescrof mint resolved its index_in_parent
  ⚠ Introduced here and never revisited: as of 2026-08-10 no other commit's diff changes an occurrence of the name, this file excluded — see the conventions above.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_MPTRACE`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `mptrace_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Introduced: 2026-07-03 in `2cc8179f2a7` — single-pass / walker-as-tracer tracing scaffold (aheui logo --jit) (#311)
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_NO_BRIDGE`

- Read sites: 1 — `majit/majit-metainterp/src/jitdriver.rs`
- Accessor: `no_bridge_enabled()`
- What it does: `MAJIT_NO_BRIDGE`: suppress bridge recording so every guard failure resumes through the blackhole.  Public because a frontend that owns its own guard-failure entry point has to honour it there too — gating only the jitdriver-internal paths leaves the variable set but inert, which reads as "bridges are off" while they keep recording.
- Introduced: 2026-07-03 in `2cc8179f2a7` — single-pass / walker-as-tracer tracing scaffold (aheui logo --jit) (#311)
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_OPREF_VARIANT_AUDIT`

- Read sites: 1 — `majit/majit-ir/src/opref_audit.rs`
- Accessor: read inline in `resolve_mode()`
- What it does: `=1` reports, `=abort` panics on the first collision of two `OpRef` variants on one `raw()` key. Off, it is one thread-local read and a return. State and mode are thread-local so two tests in parallel cannot read each other's collisions or silence one another.
- Introduced: 2026-08-10 in `d922882072c` (re-pointed 2026-08-11, DOOMED) — majit-ir: add a debug-gated detector for two OpRef variants on one raw() key
  ⚠ Introduced here and never revisited: as of 2026-08-10 no other commit's diff changes an occurrence of the name, this file excluded — see the conventions above.
- Retirement condition: "Retires when the two namespaces are made structurally uncollidable: at that point a collision is unrepresentable rather than merely unobserved, and this instrument has nothing left to detect." — quoted from the module's own doc header, the only gate in this file whose author recorded one.

### `MAJIT_OPTRACE`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `optrace_enabled()`
- What it does: Per-op trace of `run_to_end`'s dispatch loop (frame depth, pc, raw opcode). Diagnostic for pinpointing the op that faults a hardware-signal crash (SIGBUS/SIGSEGV) which `catch_unwind` cannot capture.
- Introduced: 2026-07-21 in `b159a2fd9eb` — jit: genericize the trace walker over WalkSym; unify interpreter/GC subclass ranges (#205 C2) (#646)
  ⚠ Introduced here and never revisited: as of 2026-08-10 no other commit's diff changes an occurrence of the name, this file excluded — see the conventions above.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_PCSEQ`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `pcseq_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Introduced: 2026-07-03 in `2cc8179f2a7` — single-pass / walker-as-tracer tracing scaffold (aheui logo --jit) (#311)
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_PORTAL_INLINE`

- Read sites: 1 — `majit/majit-metainterp/src/pyjitpl/dispatch.rs`
- Accessor: `portal_inline_experiment_enabled()`
- What it does: [FR] WIP gate for the state-field recursive-portal Inline re-entry rework. OFF by default: the state-field `portal_jitcode`-None path keeps its clean-abort fallback so existing consumers are unaffected. Set `MAJIT_PORTAL_INLINE=1` to exercise the experimental inline path.
- Introduced: 2026-07-03 in `5fef244feb1` — majit: switch dispatch, split_dispatch routing, and back-edge builder pooling for the wasmi JIT tier (#307)
  ⚠ Introduced here and never revisited: as of 2026-08-10 no other commit's diff changes an occurrence of the name, this file excluded — see the conventions above.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_PROBE_LIVENESS`

- Read sites: 1 — `pyre/pyre-jit/src/call_jit.rs`
- Accessor: `majit_probe_liveness_enabled()`
- What it does: Whether `MAJIT_PROBE_LIVENESS` is set, cached at first access.
- Introduced: 2026-04-27 in `d0fd3c227a3` — eval2: autogenintrules port + Phase 0/1/2 epic groundwork
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_REG_WRITE_AUDIT`

- Read sites: 1 — `majit/majit-ir/src/reg_write_audit.rs`
- Accessor: read inline in `resolve()`, cached per thread by `enabled()`
- What it does: debug-gated attribution for writes into a frame's `int_regs` — which code last wrote the slot a reader is about to read. The register file is written from ten production sites across two files and a value read out of it carries no record of which of them put it there; other instruments in the area report what a slot *holds*, not where it *came from*. Writers are not named by hand: `#[track_caller]` makes each note carry its own call site, so a site cannot be mislabelled and a site that moves re-reports itself at the new position. Off, every entry point is one thread-local read and a return. All state is thread-local, matching `opref_audit`, because a trace is recorded on one thread and a process-global table would let two tests in parallel attribute each other's writes. — condensed from the module's own doc header.
- Default polarity: **OFF**. The read is `Ok(v) if v != "0" && !v.is_empty()`, so unset, empty and `=0` are all off and any other value turns it on; the doc header spells the intended form `=1`. It is not a default-ON experiment, so no epic's close disposes of it.
- Introduced: 2026-08-11 in `965e5c3d88c` — majit-ir: add a debug-gated writer-attribution audit for int_regs
  ⚠ The introduced-and-untouched-since marker is **withheld as uninformative, not omitted by oversight**: `git log -S` over the name returns exactly one commit, but that commit is hours older than this row, so "never revisited" is entailed by the gate's age rather than being a fact about the gate. The marker becomes a real reading once some other commit has had the opportunity to touch the name.
- Retirement condition: **UNRECORDED** — owed by this gate's owner. The module's doc header states none, so there is no sentence here to quote.

### `MAJIT_SIBLING_TARGET_DIR`

- Read sites: 1 — `scripts/check-sibling-consumers.py`
- Accessor: read inline in `check()`, via `os.environ.get`
- What it does: a value, not a switch: the target directory the sibling-consumer check builds into, defaulting to `<tmpdir>/majit-sibling`. Sourced to the comment above the read: the sibling trees are read-only to this workspace and the shared `target/` is contended, so the check needs a third location; setting this to a persistent path is what keeps those builds incremental. Unset it and every run is a cold build into a temp dir.
- Introduced: 2026-08-10 in `4a587477004` (re-pointed 2026-08-11, DOOMED) — scripts: add a local check that compiles the path-dep sibling consumer trees
  ⚠ Introduced here and never revisited: as of 2026-08-10 no other commit's diff changes an occurrence of the name, this file excluded — see the conventions above.
- Retirement condition: **RECORDED 2026-08-11 by the introducing author**, who is
  the owner this line was owed by — the condition is derivable here because the
  author is present to state it, which is the case the `UNRECORDED` convention
  above leaves open rather than the case it describes.
  The gate exists only because the check has nowhere good to build: the sibling
  trees are read-only to this workspace and the shared `target/` is contended, so
  a third location is needed and only an override makes it incremental. It
  retires when that premise ends, whichever comes first:
  (a) `scripts/check-sibling-consumers.py` is removed — the gate has exactly one
  read site and dies with it; or
  (b) the path-dep consumer trees stop being out-of-workspace checkouts, i.e.
  `git ls-files wasmi/` becomes non-empty, at which point the check builds them
  in place under the workspace's own `target/` and the third location is
  redundant. ⚠ (b) is the same event that empties axis 1 of the consumer-coupling
  table, so it retires this gate and that blind spot together.
  ⛔ A permanent `RC=2` from the check is **not** the condition: that reports the
  trees are absent, which is when the gate is unused, not when it is unneeded.

### `MAJIT_SMALLIR`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `smallir_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Introduced: 2026-07-03 in `2cc8179f2a7` — single-pass / walker-as-tracer tracing scaffold (aheui logo --jit) (#311)
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_SPDIAG`

- Read sites: 1 — `majit/majit-metainterp/src/jitdriver.rs`
- Accessor: `spdiag_enabled()`
- What it does: Diagnostic env gates read once and cached — these are checked on the hot back-edge / guard-failure paths that run every loop iteration, so re-reading the environment per call would add a syscall to each iteration.
- Introduced: 2026-07-03 in `2cc8179f2a7` — single-pass / walker-as-tracer tracing scaffold (aheui logo --jit) (#311)
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_STALL_WINDOW`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `stall_window()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Introduced: 2026-06-26 in `769911b5c57` — majit: trace-compile pipeline fixes — O(1) compile, is_gc_managed guard gating, pool-array reads, macro bridges (#263)
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_STATS`

- Read sites: 3 — `majit/majit-trace/src/logger.rs`, `pyre/pyre-wasm-runner/src/main.rs`, `pyre/pyrex/src/lib.rs`
- Accessor: `stats_enabled()`; also read inline in `run()` and `maybe_print_jit_stats()`
- What it does: Whether JIT statistics collection is enabled. Checks MAJIT_STATS=1 or MAJIT_LOG=1.
- Introduced: 2026-03-11 in `349bde9c5df` — add IntFloorDiv opcode, guard optimization pass, jitlog profiling, bridge infrastructure
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_STEP_LIMIT`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `step_limit()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Introduced: 2026-06-26 in `769911b5c57` — majit: trace-compile pipeline fixes — O(1) compile, is_gc_managed guard gating, pool-array reads, macro bridges (#263)
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_STRICT`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `jit_strict_mode()`
- What it does: Strict JIT mode: a non-`InvalidLoop` panic during compilation is a bug and must fail loudly rather than silently degrade to the interpreter and mask the bug behind correct output. Enabled in debug builds (`cargo test`) and whenever `MAJIT_STRICT` is set (release benches / CI); off in plain release so production keeps graceful degradation. Cached like `majit_log_enabled`.
- Introduced: 2026-04-20 in `161f03843a2` — virtualstate: strict leaf-store type check
  ⚠ Subject elided where it spells a second, longer gate name in this family. That
  name has **no reader in the tree today**, and this document collects gate names
  from every line, not only headings — spelling it here would register it as a
  live documented gate with no reader and fail the brake. Read `161f03843a2`'s
  subject for it. Whether it was folded into this row's gate or dropped outright
  is unrecorded, and it is the one candidate this pass turned up for the
  reader-outlived-by-its-row case the preamble describes.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_TLDBG`

- Read sites: 1 — `majit/majit-metainterp/src/lib.rs`
- Accessor: `tldbg_enabled()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Introduced: 2026-07-03 in `2cc8179f2a7` — single-pass / walker-as-tracer tracing scaffold (aheui logo --jit) (#311)
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_VERIFY`

- Read sites: 1 — `majit/majit-backend-cranelift/src/compiler.rs`
- Accessor: `majit_verify_enabled()`
- What it does: Whether `MAJIT_VERIFY` is set, cached at first access.
- Introduced: 2026-04-03 in `fd8f25c0f1d` — RPython parity: call_assembler bridge dispatch, inputarg types, exit_types
- Retirement condition: **UNRECORDED** — owed by this gate's owner.

### `MAJIT_X2_PROBE`

- Read sites: 1 — `majit/majit-backend-cranelift/src/compiler.rs`
- Accessor: `drop()`
- What it does: **UNRECORDED** — no doc comment at the read site.
- Introduced: 2026-05-19 in `92d6da40c0e` — cranelift: unify backend descr identity + in-code closing-jump dispatch (#68)
  ⚠ Introduced here and never revisited: as of 2026-08-10 no other commit's diff changes an occurrence of the name, this file excluded — see the conventions above.
- Retirement condition: **UNRECORDED** — owed by this gate's owner.
