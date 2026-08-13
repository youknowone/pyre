Assess by static analysis whether our changes in git diff upstream/main are equivalent
to the corresponding RPython/PyPy source code. The RPython and PyPy sources
are available locally (under `rpython/` and `pypy/` in this repository).

If anything was ported incorrectly, report every instance in detail. After
collecting all differences, organize the report into separate sections:

1. Cases where our patch regressed PyPy parity compared to main
2. Other mismatches introduced by our patch
3. Mismatches that already existed before this patch
4. Structural adaptations

Exceptions — mark these under "Structural adaptations", not under 1 or 2:
opcode mismatches caused by using a CPython-compatible compiler,
GIL/free-threading differences, and fundamental implementation-language
differences between RPython and Rust.

Also section 4: a deviation from PyPy that makes pyre match the observable
behaviour of the CPython in `lib-python/3` (version pinned in
`lib-python/stdlib-version.txt`), where PyPy and that CPython genuinely differ.
This is usually NOT a 3.11-vs-3.14 delta but a standing PyPy-vs-CPython
divergence; "CPython did not change in 3.14" is not grounds to refile it under
1 or 2. It qualifies only when the finding carries all four of:
(a) an observable difference — return value, exception type/message/attributes,
    identity, encoding-and-errors contract, or accepted argument shapes;
(b) a cited CPython artefact — a `lib-python/3/...:line` assertion, a measured
    run at the pinned version, or C read at that tag in a named checkout. Not
    docs, not a PEP, not a comment in pyre's own source;
(c) the PyPy `file:line` that decides, showing the two upstreams actually
    differ (if PyPy contradicts itself, pyre following PyPy's own declaration
    is section 4 as ordinary parity);
(d) no PyPy-side JIT/GC/annotator hint governing the value being changed —
    `@jit.*`, `_immutable_*`, `_attrs_`, `make_sure_not_resized`,
    `unrolling_iterable`, `rgc.*`, on the function, its helpers, or the class-
    and module-level bindings they read.
Missing any of (a)-(d), or leaving pyre matching NEITHER upstream on an
adjacent observable of the same decision, keep it in section 1 or 2 and say
which test it failed. Full rule: AGENTS.md "Spec follows CPython 3.14;
implementation follows PyPy".

Scope discipline: before writing the report, run
`git diff upstream/main --name-only -- . ':(exclude)*.jitstats'` and treat that
file list as the authoritative definition of "this patch" (when an authoritative
changed-file list is appended below, use that instead of re-deriving it). The
excluded `*.jitstats` files are `pyre/check.py`'s recorded jit-stats baselines —
generated golden data with no RPython/PyPy counterpart, so no parity finding can
cite one, and a bulk re-record of them is not a change to review. Findings under
sections 1 and 2 MUST cite our-side files from that list; a divergence in any
file NOT in the list is by definition not introduced by this patch — report
it under section 3 instead, or omit it. Verify every section-1/2 citation
against the list before finalizing the report.

---

Output format requirements (so the report can be parsed mechanically and
posted/triaged automatically). Use these four headings VERBATIM, in this
order, and nothing else at heading level 2:

## 1. Regressions to PyPy parity introduced by this patch
## 2. Other mismatches introduced by this patch
## 3. Pre-existing mismatches (already present before this patch)
## 4. Structural adaptations

Under each heading, list every finding as a bullet. For each finding cite the
concrete `our_file.rs:line ↔ rpython_or_pypy_file.py:line` pair and quote the
divergence concisely. If a section has no findings, still emit the heading
followed by a single line `None.` so all four sections are always present.
Do not modify any files; produce the report only.
