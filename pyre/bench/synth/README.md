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

Such a fixture must exit successfully, print `PASS`, and have compiled at
least one loop. The last of those is what stops an interpreted run from
satisfying a guard written about compiled code: the assertion holds either way,
so without it the fixture would keep passing if the shape it guards stopped
reaching the JIT. A fixture whose invariant is not about compiled code says so
with

```python
# pyre-check: selfcheck-interpreted
```

followed by a comment saying why, and is then graded on the exit status and the
marker alone.

A selfcheck fixture is still discovered by both runners; `check_synthetic.py`
runs it only when `--pyre` is supplied. A backend lacking a required mechanism
can be scoped out in `check.py` with `# pyre-check: skip-backends=wasm`
followed by a comment explaining the missing mechanism.
