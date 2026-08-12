# Citation harvest — the rewritten commit hashes cited in this tree

⛔ **This file is deliberately full of dead commit hashes, and it is not citation rot.**
Every hash carries its status. A hash *with its status stated* is an annotation; a bare
hash presented as a live reference is the defect. Do not "repair" this file by
re-pointing its hashes — the dead ones are the subject matter.

## Why it exists, and why it expires

The rebase rewrote every branch-only commit. A rewritten hash does **not** fail loudly:
the object survives in the reflog, so `git show` prints a real commit with a plausible
diff and nothing in the output says "this is not in your history". A dead citation is
byte-for-byte indistinguishable from a live one at the point of use — strictly worse
than a pruned object, which at least fails with `unknown revision`.

Those objects are reachable from **no ref**. They survive only until
`gc.reflogExpireUnreachable` (30 days by default) or any `git gc --prune`, and the
subjects below go with them. That is why this is written down instead of looked up on
demand.

## The core rule is three-valued, and the axis is `origin/main` — not `HEAD`

⚠ Three values are the *mainline* verdict, and they are not the whole rule: two more
states (FOREIGN, UNRECOVERABLE) sit outside this test entirely. See the subsection
below before classifying anything — an earlier revision of this file stated three and
was used to score shas the test does not apply to.

```
DURABLE   git merge-base --is-ancestor <sha> origin/main   RC=0   a rebase never rewrites it
DOOMED    ancestor of HEAD but not of origin/main                 live now, dies at the next rebase
DEAD      neither                                                 already rewritten
```

⚠ Testing against `HEAD` collapses DOOMED into DURABLE — the false-safety direction.
Everything we land on this branch is DOOMED until it merges upstream, so "it resolves
and it is on my branch" is not durability.

### ⛔ Those three are verdicts about OUR mainline, and they do not partition the world

Two further states exist, and a sha in either is **not rot** — the predicate above was
never applicable to it. Both were found the same day, from opposite directions.

```
FOREIGN        the sha belongs to another repository or another remote's history
UNRECOVERABLE  no object anywhere, and no subject to recover it by
```

⇒ **The rule is five-valued.** Recording a FOREIGN sha as DEAD is a false accusation,
and "repair" would replace a correct citation with a wrong one.

### The exit code is three-valued — do not collapse it, and never pipe it

```
git merge-base --is-ancestor <sha> origin/main     # ⛔ NEVER pipe this

rc=0     ancestor                       -> DURABLE
rc=1     resolves, NOT an ancestor      -> real rot, OR foreign-ref (see below)
rc=128   does not resolve here          -> foreign repo, never fetched, or gc'd
```

⛔ `if ! git merge-base --is-ancestor …` merges 1 and 128 and reports every
unresolvable sha as rot. The instrument HAS the third value; the caller throws it away.

⛔ NEVER pipe it. zsh has no `PIPESTATUS` (it is `pipestatus`), so
`… | head; rc=${PIPESTATUS[0]:-$?}` silently falls back to `head`'s status. Measured
here, not imagined — a FOREIGN sha reported **`rc=0`, i.e. DURABLE**, with
`fatal: Not a valid commit name` printed on the line directly above it. That is the
most dangerous direction the instrument can fail in.

### ⛔ Absence does NOT detect foreign — it detects only UNFETCHED

A first version of this section proposed `git cat-file -e <sha>^{commit}` as the
FOREIGN splitter: absent ⇒ foreign, resolves ⇒ rot. **Measured over all 33 rows
below: 33 resolve, 0 absent — it detects nothing.** A foreign object that has been
*fetched* is present. `1f81807bcfde` (`majit/majit-rlib/src/rbigint.rs:5`, labelled
`pypy/main` two lines above it) resolves here and would have been scored as rot.

The two kinds of foreign need two different instruments:

| kind | example | `cat-file -e` | `branch -r --contains` |
|---|---|---|---|
| foreign REPOSITORY | a `cel-jit/` sha | **absent** — caught | cannot resolve |
| foreign REF, fetched | `pypy/main`, `upstream/wasmi` | resolves — **missed** | **caught** |

```sh
# rc=1 (resolves, not ours): is any remote ref carrying it?
git branch -r --contains <sha>        # non-empty => durable SOMEWHERE, not rot here

# rc=128 (does not resolve): ask every repo in the tree, WITH A CONTROL
for r in . cel-jit wasmi wasmi-majit-pr; do
  printf '%-16s control=%s ' "$r" "$(git -C "$r" rev-parse --short HEAD)"
  git -C "$r" cat-file -e "$SHA^{commit}" 2>/dev/null && echo FOUND || echo -
done
```

The `control=` column is mandatory: without a resolvable HEAD printed per repo,
"found nowhere" cannot be told from "the loop never ran" or "that path is not a repo".
Found in a nested repo ⇒ FOREIGN, cite it against that repo (`cel-jit`'s default branch
is `origin/master`; it has **no `origin/main`**, so this file's test cannot even be
spelled there). Found nowhere with all controls live ⇒ UNRECOVERABLE.

Measured on the 33 rows below: **31 carried by no remote ref** (genuine local-only
rot), **2 carried by one** — `3ccbd1f5f4d` on `origin/cel` (our repo, another branch;
pinned deliberately by a `rev =` in `cel-jit/cel/Cargo.toml`) and `a4e191f71b5` on
`upstream/wasmi`.

⭐⭐⭐ **`a4e191f71b5` is why this is two questions and not one.** It is simultaneously
alive on `upstream/wasmi` *and* correctly listed below as squashed into `2c23e8b00de`.
Both are true because they answer different things:

- *"will the next reader be able to resolve this?"* — yes, a stale remote pins it
- *"does it name a tree on our mainline?"* — no

`--is-ancestor origin/main` only ever answered the second. Everything above is about
**resolvability**; the repair tables below are about **mainline identity**. Do not use
one to overrule the other.

## Two repair classes, decided by how the commit left the branch

Measured over all 33; the split is exact and mechanical.

| class | n | what happened | what to cite |
|---|---|---|---|
| REBASED | 24 | rewritten in place; the subject is preserved on the twin | the **subject** — the twin is itself DOOMED, so citing its hash buys exactly one rebase |
| SQUASHED | 9 | merged upstream inside a squash commit; the original subject survives only in the squash **body** | the **upstream squash hash** — DURABLE, verified 9/9 |

⭐ The SQUASHED class is why an exact-subject lookup reports "no twin" and is wrong to
believe: `git log --format=%s` cannot match a subject that lives in another commit's
message body. Searching only `origin/main..HEAD` misses them a second way — they are
upstream, outside that range. Both mistakes were made and corrected while taking this
harvest.

```sh
# REBASED — the twin carries the same subject
git log --format='%h %s' origin/main..HEAD | rg -F "$(git log -1 --format='%s' <dead>)"
# SQUASHED — the subject is in the squash BODY, and it is upstream
git log origin/main --fixed-strings --grep="$(git log -1 --format='%s' <dead>)" --format='%h %s'
```

⚠ `git log -S` is blind to an add+revert inside a single commit, so a zero is not proof
of absence.

## ⛔ This harvest was 5 rows short, and the gap was a SELECTION defect

The first cut of this file had 28 rows. ex-asserts re-ran the census independently,
got 32 off-branch against team-lead's 32, and intersected: **27 in common, 5 present
in theirs and absent here, 1 present here and absent from theirs** (`f905ce6a997`).
27 + 1 = the 28 published. So the two populations did not nest in either direction —
neither was a subset — and a bare count of 28 vs 32 would have looked like a near-miss
rather than two different sets.

⚠ The tell that it was a selection defect and not a judgement call: one of the 5 was
`68db82dfc74`, #85's landing sha, which team-lead had **named explicitly** as a
citation to preserve. A specifically-flagged row does not fall out of a list by
judgement.

⭐⭐⭐ **A missing row here is the only defect in this file that expires.** An
unrepaired citation elsewhere stays repairable for as long as the object resolves;
a row absent from the harvest becomes **unrecoverable** the moment `gc --prune` runs,
because the harvest is the artefact built to outlive exactly that. All 5 were still
resolvable when added, and each twin was verified DOOMED with an **exact** subject
equality — not a `--grep` containment, which matches anywhere in a message and would
accept a commit that merely quotes the subject.

⚠ **The dead→twin mapping is many-to-one.** `fbdd06e5bc7` and `d71b211b2d6` are two
distinct dead shas carrying one identical subject and resolving to the same twin
`1d670fa7c9c`. The table's rows are therefore not in bijection with commits, and a
reader counting twins will get a smaller number than a reader counting dead shas.
Both rows are kept: each dead hash is separately citable somewhere in the tree.

## The harvest

### REBASED — 24 rows. The target is DOOMED; cite the SUBJECT.

| dead sha | live twin (DOOMED) | subject — this is the durable identity |
|---|---|---|
| `0ed7150e440` | `492ac152a8d` | skills: correct the cel design's back-edge premise, which is refuted |
| `2f0e44cde70` | `f84bac81367` | llbc_extract: hash out-of-root fingerprint inputs into a new external= stamp field |
| `3b68292ce5c` | `05ed5528648` | majit: record whether a fielddescrof mint resolved its index_in_parent |
| `3ccbd1f5f4d` | `70d4d8cec7b` | cel example: convert all seven merge points to the single-executor form |
| `54aa829125b` | `178776c81a9` | majit: keep a storing function-final expression out of the walk |
| `553045da4f6` | `ecf55a993ad` | gate-triage: move PYRE_LLBC_STRICT to its own section and state 6c's criterion |
| `6116063585e` | `8c57bb2b206` | majit-ir/pyre: name the field_pos_unresolved mints |
| `63e387c1a04` | `d45ea243ef8` | majit-metainterp: borrow the virtualizable name in sync_before instead of cloning it |
| `65ce9196d26` | `3f9b12db219` | majit-metainterp: consult the hotness counter before building the tracing arm's arguments |
| `68db82dfc74` | `4a587477004` | scripts: add a local check that compiles the path-dep sibling consumer trees |
| `6bc4893e459` | `f3d502f0852` | majit-metainterp: add typed creator forms for attach_procedure_to_interp and mark_force_finish_tracing |
| `8dd18ba2bf4` | `4956df0352f` | pyrex gate-triage test: scan gates per namespace instead of per PYRE_ prefix |
| `9bad7d19d0a` | `cd5875a6edc` | majit-metainterp: count a close the gate never attempted under its own slot |
| `a2dc018601b` | `824681e0a82` | majit: resume a degraded-stub abort at the aborting opcode's own boundary |
| `c590763140a` | `6eca81d5cef` | majit-translate: type a borrow of a fieldless enum as Int, like the enum itself |
| `c6737bbeece` | `8d5e266eab1` | tlc: land ex-tlc's operand-less degraded-arm fixture, with its red arm measured |
| `c8502ef76ae` | `780c5bee112` | majit: make MAJIT_LOG_OPT dump the optimized body at all three compile paths |
| `d71b211b2d6` | `1d670fa7c9c` | gate-triage: catalog the MAJIT_* gates and arm the brake's second namespace |
| `e69f4388a3c` | `ec87b55696c` | majit: assert BC_ARRAYBASE_VABLE has no emitter |
| `e939e25b713` | `d922882072c` | majit-ir: add a debug-gated detector for two OpRef variants on one raw() key |
| `eba7c58ad35` | `76d4cda71b4` | majit,pyre: put a backend in the default of every crate that needs one to build |
| `f740ffa5882` | `ff091caba8d` | majit-metainterp: count chained cells in get_stats, not bucket heads |
| `f905ce6a997` | `2678cb00b30` | wasm: export the four field_position_census counters and ask for them |
| `fbdd06e5bc7` | `1d670fa7c9c` | gate-triage: catalog the MAJIT_* gates and arm the brake's second namespace |

### SQUASHED — 9 rows. The target is DURABLE; cite the HASH.

| dead sha | upstream squash (DURABLE) | squash subject | original subject |
|---|---|---|---|
| `0f9c371b63` | `0135f56fbc0` | jit(fbw): flip and then retire the two blackhole-resume gates (#901) | jit(fbw): retire the PYRE_FBW_BLACKHOLE_RESUME reader |
| `111bdb4eeb8` | `802b79ff8db` | majit: field-level IR infrastructure + aarch64 large trace support + observer replay fix (#390) | majit: full same_greenkey close gate for primary traces, drop PYRE_SAME_GREENKEY |
| `4d3d6e290f6` | `7944966b4dd` | jit: retire foreign-lib cluster walls (gh#346 Slice A + B1a + B2 + B3a) (#606) | jit: fold RangeInclusive::contains into native integer comparisons |
| `654df9dd46` | `3662d1f05b6` | wasm JIT: bring nbody/fannkuch/fib to a few× of pypy (bridge chaining + CALL_ASSEMBLER default-on) (#347) | backend-wasm: bridge chaining and CALL_ASSEMBLER fast paths, on by default |
| `a4e191f71b5` | `2c23e8b00de` | majit: trace-abort hook, literal-range unroll, cranelift non-ref demotion, sub-word raw-load (#694) | majit-macros: lower float comparisons to the value-form float_*/ff>i ops |
| `bb6ee8d179c` | `7944966b4dd` | jit: retire foreign-lib cluster walls (gh#346 Slice A + B1a + B2 + B3a) (#606) | jit: lower malachite i64::try_from(&BigInt) narrowing to runtime-discriminant Result aggregate |
| `c6cfcb758c2` | `e18ec90cac1` | jit: fix FOR_ITER polymorphic deopt crash + body-guard liveness (#342) (#387) | jit: retire PYRE_57_INLINE_NEXT kill-switch |
| `ca2640e797b` | `e1c43d3ff08` | jit: P2 bridge compile leg default-ON + guard-thrash fixes + vstack-mirror classification (#607) | Remove PYRE_P2_COMPILE gate |
| `f41cb0496dc` | `2ae46cd8687` | rtyper: #131 follow-up — PR#306 review findings, to_vec copy, W_List/W_Tuple constructor residualize (#310) | majit: lower vec![..] to newlist via OpKind::NewList + rtype_newlist |

## Where these are cited

- `pyre/gate-triage.md` — 7
- `pyre/pyrex/tests/jit_trace_shape.rs` — 3
- `scripts/llbc_extract.py` — 2
- `majit/majit-translate/docs/design-346-foreign-lib-cluster-epic.md` — 2
- `majit/examples/tlc/src/jit_interp.rs` — 2
- `.claude/skills/cel-unboxed-values/SKILL.md` — 2
- `scripts/check-sibling-consumers.py` — 1
- `scripts/check-backend-edge.py` — 1
- `pyre/pyrex/tests/gate_triage_complete.rs` — 1
- `pyre/pyre-jit-trace/tests/llbc_fingerprint_format_test.rs` — 1
- `pyre/pyre-jit-trace/build.rs` — 1
- `pyre/check.py` — 1
- `majit/majit-translate/src/front/mir.rs` — 1
- `majit/majit-metainterp/tests/jit_interp_dispatch_ir_shape.rs` — 1
- `majit/majit-metainterp/src/warmstate.rs` — 1
- `majit/majit-macros/src/jit_interp/mod.rs` — 1
- `majit/majit-macros/src/jit_interp/jitcode_lower/lower_control.rs` — 1
- `majit/majit-ir/src/opref_audit.rs` — 1
- `majit/examples/dualtape/src/jit_interp.rs` — 1

## Denominator, stated

Tokens of 9–12 hex characters, word-bounded, over 2969 tracked files matching
`majit/*`, `pyre/*`, `scripts/*.py`, `*.md`, `.claude/*`. **This is a lower bound**:
7–8 character hashes are excluded, untracked files are excluded, and any hash written
in a form the pattern does not match is invisible to it. Of 174 tokens, 46 are DURABLE
(leave them alone), 7 DOOMED, 28 DEAD, and 93 do not resolve — the great majority of
those being decimal constants the hex pattern also matches, not citations.
