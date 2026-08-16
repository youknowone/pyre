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

Such a fixture must exit successfully and print `PASS`. It is still discovered
by both runners; `check_synthetic.py` runs it only when `--pyre` is supplied.
A backend lacking a required mechanism can be scoped out in `check.py` with
`# pyre-check: skip-backends=wasm` followed by a comment explaining the missing
mechanism.
