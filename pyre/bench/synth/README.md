# Pyre Synthetic Benchmark Suite

This directory contains small deterministic benchmarks grouped by common Python
language features.  They are meant to expose Pyre/PyPy parity gaps by comparing
stdout and runtime across interpreters.

Run all cases with CPython only:

```sh
python3 pyre/check_synthetic.py
```

Compare against PyPy and a Pyre binary:

```sh
python3 pyre/check_synthetic.py --pypy pypy3 --pyre ./target/release/pyre-dynasm
```

Each benchmark prints a stable checksum.  A Pyre failure is useful signal: it
marks a feature category that needs trace, optimizer, backend, or frontend
parity work.

Fixtures whose invariant cannot be expressed as stable CPython/PyPy output may
opt into the same directory's self-checking mode:

```python
# pyre-check: selfcheck
```

Such a fixture must exit successfully, print `PASS`, and name what it needs
compiled:

```python
# pyre-check: selfcheck-compiles=hot,inner
```

Each entry is a code object name, optionally prefixed by the compile arm that
has to mint it; a bare name means `loop`. This is what stops an interpreted run
from satisfying a guard written about compiled code: the assertion holds either
way, so without it the fixture would keep passing if the shapes it guards
stopped reaching the JIT.

Measure it by running the fixture with `PYRE_LOOP_CENSUS=1` and reading the
`[loop-census] <arm> <name>` lines, one per compiled trace. Declare the shapes
the guard is *about* — a bookkeeping loop that happens to compile is neither
required nor forbidden, so it does not belong in the list. The check runs per
backend, so declare what compiles on all of them; a shape admitted on dynasm
and declined on wasm is a wasm failure, which is the point.

The arms are `loop`, `retrace`, `root` and `entry-bridge`, one per place
`pyjitpl.rs` bumps `loops_compiled`. Only three of them close a loop:
`finish_and_compile` attaches a root trace that ends in FINISH with no LABEL,
so a fixture whose subject reaches the JIT that way declares it as such:

```python
# pyre-check: selfcheck-compiles=root:callee_d1
```

That is a different claim, not a weaker one — a loop that degrades into a root
trace is still reported.

This used to be a count, `selfcheck-loops=8`, read off `loops_compiled`. That
number answered a different question twice over: it counted the non-loop arms
alike, and it was whole-process, so any hot loop in the file cleared the floor
— a fixture's own bookkeeping loop included.

A fixture whose invariant is not about compiled code says so with

```python
# pyre-check: selfcheck-interpreted
```

followed by a comment saying why. It declares the empty set: the fixture asks
for nothing compiled, and is graded on the exit status and the marker alone.

A selfcheck fixture is still discovered by both runners; `check_synthetic.py`
runs it only when `--pyre` is supplied. A backend lacking a required mechanism
can be scoped out in `check.py` with `# pyre-check: skip-backends=wasm`
followed by a comment explaining the missing mechanism.
