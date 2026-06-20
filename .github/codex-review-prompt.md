Assess by static analysis whether our changes in git diff upstream/main are equivalent
to the corresponding RPython/PyPy source code. The RPython and PyPy sources
are available locally (under `rpython/` and `pypy/` in this repository).

If anything was ported incorrectly, report every instance in detail. After
collecting all differences, organize the report into separate sections:

1. Cases where our patch regressed PyPy parity compared to main
2. Other mismatches introduced by our patch
3. Mismatches that already existed before this patch
4. Structural adaptations

Exceptions: some differences cannot be ported 1:1 because of Python 3.11 vs
3.14 differences, opcode mismatches caused by using a CPython-compatible
compiler, GIL/free-threading differences, and fundamental implementation-
language differences between RPython and Rust. Mark those separately under
"Structural adaptations".

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
