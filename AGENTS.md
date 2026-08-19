# AGENTS.md

## The JIT is generated from the interpreter source

pyre is structured like PyPy: `pyre-interpreter` is the RPython-interpreter
analog, and **the JIT is not hand-written**. `majit-translate` reads the
interpreter's Rust source and generates it — `front/ast.rs` (parse) →
`flowspace/` (`flowcontext.py`/`framestate.py`) → `annotator/` (`annrpython.py`)
→ `rtyper/` → `codewriter/` (`jtransform.py`/`codewriter.py`, emits JitCode) —
the same pipeline RPython's translator runs over PyPy.

**So "Rust can't be meta-traced" is never a valid excuse for a deviation.**
Whatever the interpreter source expresses is what the generated JIT must
preserve; a JIT that diverges from the interpreter has a generation defect to
fix. Never justify a mismatch by appeal to the implementation language.

### One red frame per frame

PyPy keeps one frame object per inlined Python call (`MIFrame` tracing,
`BlackholeInterpreter` resuming), each with its own
`jitcode → pycode → w_globals → locals`. `LOAD_GLOBAL` reads it off the *live*
frame (`pyframe.py` `get_w_globals`); resume rebuilds one frame per encoded
jitcode header (`resume.py` `rebuild_from_resumedata`). No shared frame slot
exists, so namespace confusion is impossible.

The frame is the loop's single **red** input, `pycode` the **green**. Thread
that red frame through **every** frame, inlined non-portal callees included.
Collapsing callees onto one anchor (a single `portal_frame_reg`, a single
bridge-resume root frame) drops the callee's own pycode/globals/locals — the one
root cause behind the pycode-`names` miscompile, the LOAD_GLOBAL namespace
mismatch, bridge-resume inline-frame globals and vable-resident root locals. Fix
it by restoring the per-frame red frame, never by baking an anchor's value as a
constant.

## Charon LLBC extraction — the prepass input

The annotator/rtyper prepass and the `PYRE_RTYPER_VERBOSE` census read
**pre-extracted `.ullbc` under `build/llbc/`, not the Rust source**. A change to
`pyre-interpreter` / `pyre-object` / `pyre-jit` is invisible until re-extraction;
`majit-translate` changes take effect immediately, because the translator runs
live over the frozen bodies.

Charon is shared and pre-installed at `../.pyre-build/charon/<platform>/charon`
(pinned `nightly-2026.05.29`), **not on `PATH`** — `which charon` finding nothing
does not mean it is missing.

```bash
python3 scripts/install-charon.py    # idempotent, usually a no-op
python3 scripts/extract-llbc.py      # pyre-object pyre-interpreter pyre-jit
touch pyre/pyre-jit-trace/build.rs
PYRE_RTYPER_VERBOSE=1 cargo build --release -p pyre-jit-trace   # runs the prepass
# census: target/release/build/pyre-jit-trace-*/stderr, rg -c 'PREPASS phaseA fail'
```

- `extract-llbc.py` **skips** a crate whose source fingerprint is unchanged;
  `--force` / `LLBC_FORCE_REEXTRACT=1` overrides. `pyre-interpreter.ullbc` is
  ~300 MB and takes minutes.
- It internally sets `MAJIT_LLBC_EXTRACTION=1` to break the
  `pyre-jit → pyre-jit-trace → pyre-jit.ullbc` bootstrap cycle, which makes
  `build.rs` emit placeholder artifacts. Never set it for an ordinary build.
- Interpreter-source work ⇒ re-extract, then rebuild the prepass.
  Translator-only work ⇒ prepass rebuild alone.
- `rg` note: a stray `--replace` in user config mangles long lines; pass
  `rg --no-config` when bucketing.

## The wasm gate runs on every OS

Its only prerequisite beyond the native backends is the
`wasm32-unknown-unknown` target — wasmtime is linked into `pyre-wasm-runner`,
and the guest builds on stable with no `-Z build-std`. Once the target is
installed `DEFAULT_BACKENDS` **adds wasm by itself**, so a bare
`python3 pyre/check.py` runs three backends and `--backend dynasm,cranelift`
*narrows* it.

CI installs that target on the ubuntu leg alone because wasm output is
platform-independent — **a cost decision, not a capability limit.** Never defer
a wasm-only failure to CI on the grounds that the host cannot reproduce it.

## Data structure parity with RPython/PyPy

majit and pyre are line-by-line ports, so the container choice is part of the
port. **Treat every `HashMap` and every `thread_local!` as suspicious**: find the
upstream owner and storage shape first.

1. **Look up the upstream attribute before choosing a container.** A Rust
   `HashMap` is not the default translation of a Python `dict` — over a
   small/dense key space, or where insertion order or index lookup suffices,
   `VecMap` / `IndexMap` / `Vec` is closer. Prove the semantics from upstream.
2. **Side-tables are usually wrong.** RPython stores per-box information *on the
   box* (`box._forwarded`, `PtrInfo`, `IntBound`, descr attributes). Reaching for
   `HashMap<OpRef, _>` means you skipped that machinery; route through
   `OptContext::with_intbound_mut` / `set_ptr_info` instead.
3. **A borrow-checker workaround** is acceptable only when every alternative was
   tried, the deviation is minimal, and a comment cites the RPython original.
4. **Do not delete an RPython method to "simplify".** If `optimizer.py` has
   `ensure_ptr_info_arg0`, the port has it — the shortcut diverges, and the next
   porter's `heap.py` port stops compiling for no visible reason.
5. **TLS is almost never the right owner.** Type objects, module state,
   registries and semantic caches are process-global or interpreter-owned in
   PyPy and stay shared here; never duplicate them per thread to satisfy `Sync`.
   TLS is right only where upstream makes the state itself thread-specific
   (current thread/execution context, errno) or for a disposable cache that
   cannot affect identity, semantics, lifetime or GC reachability. Anything else
   needs an upstream citation in the code.

The measured cascade this prevents: deleting `ensure_ptr_info_arg0` for a
side-table `OptHeap.array_min_lengths` left `postprocess_arraylen_gc` unable to
read it, so it was crippled to a hardcoded `IntBound::nonnegative()`, which then
forced a parallel `ExportedValueInfo::int_lower_bound`. One non-orthodox
`HashMap`, four files of divergence.

## The PyPy oracle: run it before arguing about orthodoxy

`rpython/` says what upstream *claims*; a real `pypy3` shows what it *does*. For
"is this orthodox or our deviation?", run the oracle **first** — before reading
source, forming a theory, or recording a verdict.

```bash
PYPYLOG=jit-summary:- pypy3 pyre/bench/synth/<fixture>.py    # vs MAJIT_STATS=1
```

Most `pyre/bench/synth` fixtures need no stdlib, so this costs one command. Read
`Total # of loops`/`bridges`, `forcings`, `virtualizables forced`, every
`abort: *`, `nvirtuals`. **A differing counter is a pointer, not the answer** —
find the upstream line behind it, usually a single JIT hint
(`@jit.look_inside_iff`, `dont_look_inside`, `elidable`, `unroll_safe`), and cite
it. `PYPYLOG=jit-log-opt:FILE` dumps the optimized trace when the summary is too
coarse. `pypy3` is 3.11, so trim newer syntax out of the fixture.

This overturned a weeks-old "unfixable by construction" verdict on
`getframe_inline_subwalk_multiframe` in one command: the oracle reported
`forcings: 0`, and the `@jit.look_inside_iff` on `getframe`
(`pypy/module/sys/vm.py`) named the reason.

## Spec follows CPython 3.14t; engineering follows PyPy

**Pursue PyPy parity to the extreme as engineering, and CPython 3.14t as spec;
where they collide, take the 3.14t behaviour and engineer it the way PyPy
would.** Neither goal yields wholesale — the axis decides which one governs the
line in front of you.

The `t` is the **free-threaded** build: "CPython does X" is an answer only once X
holds without the GIL. Correct-because-a-global-lock-serialises-it is not
on-spec.

**The spec governs only what a caller can observe** — return value, exception
type/message/attributes, identity, encoding-and-errors contract, accepted
argument shapes. Everything else follows PyPy **unconditionally**: names, module
paths, control-flow order, data structures, storage owner, JIT hints. A
structural divergence does not become a spec fix by sitting next to one.

**It is not a 3.11-vs-3.14 question** — six of seven adjudicated cases had no
version delta at all (`sched_setscheduler` since 3.3, `PyUnicode_FSConverter`
since 3.3, PEP 529, PEP 471). "3.14" pins *which* CPython you read
(`lib-python/stdlib-version.txt`); a missing delta is not grounds to refuse the
exception, and a real one earns nothing by itself.

**Six tests, in order; stop at the first leaf.** The full procedure is in
`/parity` under "SPEC-DEVIATION" — do not invoke this ruling without reading it.

1. Can a Python snippet print a difference? No → ordinary parity finding.
2. Do you hold an admissible 3.14 artefact — an in-tree `lib-python/3/…`
   assertion, a measured run at the pinned version, or C source read at that tag?
   Prose is not admissible; a comment in pyre's own source is never the artefact.
3. Do the two upstreams actually disagree? If they agree and pyre differs from
   both, it is a plain regression that no spec reasoning rescues.
4. Is PyPy's shape load-bearing for a mechanism pyre also has? Search the whole
   definition, decorators included, in `rpython/` and `pypy/` for `@jit.*`,
   `_immutable_*`, `_attrs_`, `unrolling_iterable`, `make_sure_not_resized`,
   `rgc.*`. A hint governing the value you are changing → **STOP, follow PyPy**.
   Record the negative search too.
5. Per-site artefact plus a blast-radius census: every departing site needs the
   artefact forcing *that* site ("consistency with a sibling" is not one), then
   `rg` pyre's own readers, `pyre-jit*` and `majit*` included.
6. Does pyre land on 3.14t across the whole decision? Landing where **neither**
   upstream sits is a defect however well one axis matched.

File the result under the review's `## 4. Structural adaptations` as
`[3.14-spec] <ours> ↔ <pypy> — <observable>; evidence: <route + cite>`, and
comment at the site citing both sides.

## Porting discipline

- Strict line-by-line structural parity. No shortcuts, no reimplementation from
  scratch, no declaring a phase complete without the literal refactor.
- If a parity fix regresses, find the root cause before reverting. Structural
  alignment skipped is not success, even with green benchmarks.
- Cite upstream by **symbol**, not `file:line`. Numbers rot silently and a
  rotted citation still reads as authoritative; a symbol stays checkable with
  `rg`. Use a line number only where no symbol pins the claim, and name the
  enclosing symbol beside it.
- Confirm the worktree (`git rev-parse --show-toplevel`) before editing and
  before staging — dozens of sibling worktrees share one `.git`.

## Before committing

- `cargo test --all --features dynasm`. The feature flag is mandatory: without it
  `majit-metainterp` emits `compile_error!` and every error after it is noise.
- `python3 pyre/check.py` — every backend the host can build. A perf regression
  is a finding to explain, not an automatic veto: if the slower code is the
  line-by-line port and the faster was a shortcut, **the port stands** — record
  it and name the upstream optimization that would recover it (`/parity`
  Principle 4). Revert only when the regression has no such explanation. This
  file is loaded every session and Principle 4 is not, so do not restate that
  rule here in a form that contradicts it.
- Re-record a `.jitstats` baseline only when the new number is the one that
  should hold. "The recorded number no longer matches" is never on its own a
  reason: a gate is a target to reach, not a figure to refit.
- When rebasing or cherry-picking, check the fix isn't already on main
  (`git log main --grep=…`).

## Debugging discipline

- Before running the test, verify the traced path is actually reached — check
  gating and feature flags.
- Fix the interpreter/JIT root cause. Do not build workarounds (fallback
  modules, special cases) around it.
