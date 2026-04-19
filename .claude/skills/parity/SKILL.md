---
name: parity
description: Enforce strict line-by-line RPython/PyPy structural parity for changes in the majit/pyre codebase. Invoked via `/parity`, usually combined with a follow-up task (e.g. `/parity continue #task 15`, `/parity fix the optimizer`, `/parity why is nested_loop slow`). Use this skill whenever the user types `/parity`, or when they ask for RPython-parity verification, line-by-line checking against upstream, structural-equivalence review, or pypy-source comparison. The skill puts Claude into "parity mode" — the rest of the user's message is executed under the strict parity principles below, and Claude must read the local PyPy/RPython source at `/Users/al03219714/Projects/pypy/{rpython,pypy}/` before making non-trivial changes.
---

# RPython/PyPy Line-by-line Parity

## First principles

Apply these to every decision made while this skill is active, even if the user's follow-up task seems to point elsewhere.

1. **Structural equivalence, not functional equivalence.** Match module paths, module names, class/type names, function names, variable names, even down to `_` prefixes and `self`-passing conventions. Functional-only equivalence silently accumulates divergence and causes bugs that never occur in RPython.

2. **Upstream source is the spec.** The local PyPy tree at `/Users/al03219714/Projects/pypy/rpython/` and `/Users/al03219714/Projects/pypy/pypy/` is the authoritative reference. Before writing any non-trivial change, open the RPython counterpart and read it. Line-by-line porting usually has the answer; trust upstream over clever adaptations.

3. **Every deviation must be classified and justified.** Three classes:
   - **NEW-DEVIATION** — introduced by us, in this diff or recent history, with no RPython backing. These must be removed. Replace with the correct RPython code, not with another adaptation.
   - **PRE-EXISTING-ADAPTATION** — documented in code comments as an unavoidable Rust/architecture adaptation (borrow checker, enum vs class hierarchy, stack-machine Python bytecode vs register-machine jitcode, etc.). These stay but their scope must remain minimal and they must reference the RPython line they diverge from.
   - **PARITY** — matches RPython structure line-by-line. No action.

4. **Performance can temporarily regress.** If a benchmark slows down because parity-correct code replaced a clever local shortcut, accept it. Performance is recovered by further line-by-line porting of the upstream optimization, not by reintroducing the shortcut. Neither the user nor the benchmark suite overrides the parity principle.

5. **Don't stop at the first dependency.** If line-by-line porting is impossible because a dependency (helper, pass, opcode, descriptor) has not been ported yet, port that dependency first in the same RPython-parity style. Then come back.

6. **Removing new deviations takes priority over adding features.** When auditing a diff, new deviations found must be addressed before the follow-up task proceeds, unless the user has explicitly directed otherwise in the same invocation.

## Behavior under `/parity`

When this skill is invoked — either by the user typing `/parity ...` or by Claude recognizing a parity-related ask — do the following in order:

### Step 1: Parse invocation

Separate the invocation into:
- The `/parity` flag itself (already consumed).
- The **follow-up task** — everything the user wrote after `/parity`. This is the actual work to be done.

If there is no follow-up task (just bare `/parity`), default to "run a full parity audit of the current diff and report findings without writing code".

### Step 2: Parity audit

Before touching any code, run a quick audit of the current state:

```bash
git diff main...HEAD                 # committed changes on current branch
git diff HEAD                         # working tree (unstaged + staged)
git status --short                    # untracked files worth noting
```

For each changed file, find the RPython/PyPy counterpart **mechanically** (see next section). Read the upstream file and compare.

Report findings as a short **Parity audit** section before executing the follow-up task:

- Files modified with their mechanically-derived RPython counterpart (one line each).
- For each hunk that contains a candidate NEW-DEVIATION: cite the file:line in the diff, cite the RPython file:line it should have mirrored, quote the deviation concisely.
- Files modified where the mechanical rule does not produce a valid counterpart — treat these as structural deviations in themselves and flag to the user.

If the audit turns up NEW-DEVIATIONs that touch the follow-up task's scope, surface them before continuing.

### Step 3: Execute the follow-up task under parity

Run the follow-up task with these constraints:

- Read the relevant RPython source before writing code. Cite the file:line being ported in the code comment or in the summary to the user.
- When removing pre-existing code, confirm with the RPython upstream that the removal agrees with the upstream. If RPython has the code in a different form, **replace** rather than **delete**.
- When introducing anything new, check whether RPython has a counterpart. If so, port it. If not, either (a) port the RPython dependency that provides it, or (b) mark it clearly as a documented PRE-EXISTING-ADAPTATION with a comment citing the RPython decision point.
- If the follow-up task's natural implementation would introduce a NEW-DEVIATION, push back: state the tradeoff, propose the RPython-aligned alternative, and let the user choose. Do not silently introduce new deviations.

### Step 4: Verify

After making changes, run the checks that would catch regressions:

- `cargo test --all --features dynasm` — parity regressions in the metainterp layer often surface as unrelated test failures.
- `cargo test --all --features cranelift` if cranelift paths are touched.
- `./pyre/check.sh` for end-to-end correctness.

Accept temporary performance regressions. Do NOT re-introduce shortcuts to recover perf. Record the regression in MEMORY and move on.

## Finding the counterpart: mechanical rule first

The goal is that **any** file in majit/pyre has its RPython/PyPy counterpart discoverable from the path alone. When this works, no mapping table is needed. When it doesn't work, the failure is itself a parity signal — the path structure diverged from upstream and that divergence should be on the audit list.

### Crate-level roots

Only the crate-level roots need memorizing. Everything below is mechanical.

| majit/pyre crate root | Upstream root |
|---|---|
| `majit/majit-metainterp/src/` | `rpython/jit/metainterp/` |
| `majit/majit-translate/src/jit_codewriter/` | `rpython/jit/codewriter/` |
| `majit/majit-translate/src/flowspace/` | `rpython/flowspace/` |
| `majit/majit-translate/src/annotator/` (future) | `rpython/annotator/` |
| `majit/majit-translate/src/rtyper/` (future) | `rpython/rtyper/` |
| `majit/majit-translate/src/translator/` (future) | `rpython/translator/` |
| `majit/majit-translate/src/translate_legacy/` | pre-roadmap ad-hoc — deleted at P8.11, no upstream |
| `majit/majit-backend-dynasm/src/x86/` | `rpython/jit/backend/x86/` |
| `majit/majit-backend-dynasm/src/aarch64/` | `rpython/jit/backend/aarch64/` |
| `majit/majit-backend-cranelift/src/` | `rpython/jit/backend/llsupport/` (Cranelift plays the role of LLSupport) |
| `pyre/pyre-interpreter/src/` | `pypy/interpreter/` + `pypy/objspace/std/` + `pypy/module/` |
| `pyre/pyre-object/src/` | `pypy/objspace/std/` (object layouts) |

The `majit-` / `pyre-` prefix is a Cargo workspace namespace, not a claim
that the crate lives under `rpython/jit/` or `pypy/<anything>/`. Each row
is an independent mapping; add new rows rather than deriving from the
prefix. `rpython/` ↔ `majit/`, `pypy/` ↔ `pyre/` at the package-root
level — crates under `majit/` can correspond to any `rpython/<package>/`,
not only `rpython/jit/`.

**Crate boundary invariant**: `majit/*` crates MUST NOT depend on any
`pyre/*` crate, mirroring upstream's `rpython/` ⊥ `pypy/` separation.
External third-party crates (e.g. `rustpython-compiler-core` for CPython
3.14 bytecode tables) are allowed as they play the role of RPython's
host-stdlib imports (e.g. `from opcode import ...`).

The following crates carry architectural divergences from upstream and their roots are PRE-EXISTING-ADAPTATIONs by design. Audit individual files against RPython as if the root were `rpython/jit/metainterp/`, and classify mismatches per the rules below.

- `majit/majit-ir/` — extracted-out IR / OpCode / Descr layer. In RPython these live inside `rpython/jit/metainterp/resoperation.py` + `history.py` + scattered descr files in `rpython/jit/backend/`. The crate split itself is a Rust adaptation; the file-level names inside still need to line up.
- `pyre/pyre-jit-trace/` — pyre-specific layer for tracing Python bytecode. RPython's register-machine jitcode path lives in `rpython/jit/metainterp/pyjitpl.py` (opimpl_*), `blackhole.py`, and `codewriter/`. Auditors: the *logic* here must match `pyjitpl.py` opimpls file-by-file even though the directory is different.
- `pyre/pyre-jit/` — pyre's warm-entry / portal-runner / resume glue. RPython's counterparts are `rpython/jit/metainterp/warmstate.py`, `warmspot.py`, `compile.py` (loop/bridge creation).

### Within-root mechanical transform

For any file under a known crate root, compute the counterpart by:

1. Strip the crate root prefix.
2. Replace the trailing `.rs` with `.py`. For `mod.rs`, treat it as `__init__.py` (usually empty; the real content is in siblings).
3. Prepend the upstream root.

Examples (all should resolve to an **existing** file on disk):

| pyre/majit path | Expected upstream path |
|---|---|
| `majit/majit-metainterp/src/pyjitpl.rs` | `rpython/jit/metainterp/pyjitpl.py` |
| `majit/majit-metainterp/src/optimizeopt/unroll.rs` | `rpython/jit/metainterp/optimizeopt/unroll.py` |
| `majit/majit-metainterp/src/optimizeopt/heap.rs` | `rpython/jit/metainterp/optimizeopt/heap.py` |
| `majit/majit-metainterp/src/blackhole.rs` | `rpython/jit/metainterp/blackhole.py` |
| `majit/majit-codewriter/src/jtransform.rs` | `rpython/jit/codewriter/jtransform.py` |
| `majit/majit-backend-dynasm/src/x86/regalloc.rs` | `rpython/jit/backend/x86/regalloc.py` |
| `pyre/pyre-interpreter/src/baseobjspace.rs` | `pypy/interpreter/baseobjspace.py` |

Verify the counterpart exists before proceeding:

```bash
test -f /Users/al03219714/Projects/pypy/rpython/jit/metainterp/optimizeopt/unroll.py && echo OK
```

### When the mechanical transform fails

If the derived path does not exist, **that is itself a parity finding**. Do not silently fall back to a multi-file "whichever upstream file has similar content" search without naming the problem. Classify the mismatch:

1. **File renamed in majit/pyre vs upstream.** e.g. `majit-metainterp/src/heap.rs` vs `rpython/jit/metainterp/heapcache.py`. The rename is a structural deviation; rename back unless there is a cited reason. If the majit file genuinely combines multiple upstream files, see #2.

2. **One majit/pyre file fuses multiple upstream files.** e.g. `pyre-interpreter/src/baseobjspace.rs` containing both `baseobjspace.py` and `abstractinst.py` content. Fused files are structural deviations: upstream's module boundaries carry semantic meaning (separate import graph, separate test units). Split as two files unless merging was explicit user intent. Even without splitting, annotate each function with the upstream file:line it was copied from.

3. **majit/pyre file at a different directory than upstream.** e.g. `majit-metainterp/src/jitcode/` whose upstream lives in `rpython/jit/codewriter/`. The crate boundary is wrong. Flag as structural deviation. When porting, keep the function-level parity to `rpython/jit/codewriter/` even if the directory cannot be moved in this change.

4. **Upstream has no counterpart.** Could be (a) a Rust-specific adaptation that deserves PRE-EXISTING-ADAPTATION status (document which upstream decision it encodes), or (b) a NEW-DEVIATION that should never have been created. Determine which. A file named with pyre-specific domain vocabulary (e.g. `pyre_sym.rs`, `jit_state.rs`, `constant_pool.rs`) that has no upstream parallel is high-risk — those are often where NEW-DEVIATIONs live.

5. **Auto-generated files.** `target/release/build/*/out/*.rs`, proc-macro expansions, `.template.rs` files. Skip these in the audit; follow the template source instead.

Report every mechanical-transform failure in the audit, even if you then successfully locate the logical counterpart manually.

### Within a file: mechanical name match

Within a matched file, the structural expectation continues:

- **Type names** should match upstream class names 1:1 (Rust CamelCase ↔ Python CamelCase, trivial).
- **Function / method names** should match upstream, modulo Rust snake_case (`optimize_INT_ADD` → `optimize_int_add`).
- **Field / variable names** should match upstream identifiers literally (`box`, `opnum`, `orgpc`, `resumepc`, `postponed_op`, `truthy_values`, …).
- **Control flow order** should match upstream — don't reorder if/elif chains or loop bodies for "clarity".

Any local name that doesn't appear anywhere in upstream (grep `rpython/` and `pypy/` for the string) is a candidate NEW-DEVIATION. Examples seen in this repo:

- `pending_branch_other_target` — no upstream equivalent
- `last_comparison_*` cache — no upstream equivalent
- `pre_opcode_*` stack snapshot — no upstream equivalent (RPython uses per-PC liveness, not per-opcode state capture)
- `other_target` resume adaptation — no upstream equivalent

## Signals of NEW-DEVIATION

Treat these patterns as high-likelihood NEW-DEVIATION. Verify against upstream before removing, but flag in every audit.

- **Undocumented side tables / HashMap caches** in structs that RPython does not have. RPython's Box-identity model (Python object `is`) removes the need for most side tables; if pyre has one, it's often a flat-OpRef compensation that should either be lifted or removed.
- **`take()` / `Option::take` / `std::mem::replace` used to temporarily swap state for a snapshot**, where RPython would write and read a single `frame.pc` field. If the save/restore is over fields with no RPython counterpart (`pre_opcode_*`, `pending_branch_*`, `last_comparison_*`), the fields themselves may be the deviation.
- **Function names that don't appear in RPython**: grep `rpython/jit/` and `pypy/` for the name. If nothing matches, the function is a pyre-only helper. Check whether an RPython counterpart exists under a different name before declaring it NEW.
- **Comments like** `// pyre-only`, `// TODO remove`, `// workaround`, `// adaptation`, `// fallback`, `// temporary`, `// hack` without a cited RPython file:line. Honest self-reports of deviation.
- **`is_*_classlike` / `is_*_like` / `_maybe_*` helpers that reinterpret a raw pointer** as a different struct and peek at offsets. Type-confusion bombs — RPython has proper `isinstance` checks.
- **Feature flags / environment-variable switches** that gate behavior differences from RPython. RPython doesn't use env vars to switch semantics; if pyre does, both code paths are deviations in different directions.
- **"Simplified" resume / guard paths** (other_target resume, post-pop snapshot, …). RPython's resume is one path, not several.

None of these are automatically NEW-DEVIATION — check upstream first. But they warrant scrutiny.

## Output shape

When the skill is active and a response is being written:

1. **Parity audit** (short section, file:line references):
   - 3–10 lines max.
   - List each modified file with its mechanically-derived counterpart (or a "❌ mechanical transform failed: <reason>" note).
   - Cite each candidate NEW-DEVIATION as `<majit path>:<line> ↔ <rpython path>:<line> — <deviation summary>`.
   - If there is nothing to report, still include the section with `Clean — no new deviations in current diff`.

2. **Follow-up task** (the bulk of the response):
   - Executed under the principles above.
   - Reading RPython source is expected. Cite it.
   - New deviations proposed by the natural solution must be flagged to the user with an RPython-aligned alternative before committing to code.

3. **Verification** (if code was changed):
   - Which test commands were run and their outcomes.
   - Any temporary performance regression noted.

4. **Summary** (one or two sentences):
   - What changed, what was left for follow-up.

Keep the tone direct and the citations concrete. No vague "I'll make sure this matches PyPy" language — always point at a file:line.

## Interaction with other mindsets

- The user's `CLAUDE.md` already carries "majit ↔ RPython Parity Rules" (section 1–5). This skill is the stronger form of that: under `/parity`, there is no wiggle room for "Rust language adaptations" except where the RPython line is explicitly cited. If CLAUDE.md and this skill conflict, this skill wins for the duration of the `/parity` invocation.
- `/commit` (the commit skill) is compatible — parity auditing should happen before committing, not after.
- If the user invokes `/parity` inside a larger plan document (e.g. `jtransform_optimize_goto_if_not_port.md`), the plan's existing PRE-EXISTING-ADAPTATION annotations are respected; the audit focuses only on changes introduced since the plan was written.
