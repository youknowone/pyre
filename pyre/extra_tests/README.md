# pyre/extra_tests

Pure-Python snippet tests imported from
[RustPython's `extra_tests/snippets`](https://github.com/RustPython/RustPython/tree/main/extra_tests/snippets),
plus pyre-authored generic CPython-compatibility gaps that do not belong to a
JIT, GC, or PyPy-internal parity suite.

Each `snippets/*.py` is a self-contained script that asserts a small
piece of CPython semantics.  A snippet "passes" when the interpreter
exits with code 0.

## Running

```sh
python3 pyre/extra_tests/run.py            # all three backends
python3 pyre/extra_tests/run.py --cpython-only
python3 pyre/extra_tests/run.py --gated-only   # the set CI blocks on
python3 pyre/extra_tests/run.py --filter builtin_dict
python3 pyre/extra_tests/run.py -v         # show every (script, backend)
```

Parts of the imported corpus fail today — a few only under CPython,
because they assert behaviour the original suite never ran there — so a
bare run is survey material rather than a gate.  A snippet that is green
under CPython **and** both backends can declare `# pyre-check: gate=1` on
its first line; CI runs exactly that subset (`--gated-only`) and a red
run there blocks the merge.  Add the marker only after checking all three.

The runner sets `cwd` to `snippets/` so `from testutils import ...`
works.  `testutils.py` is the helper module shipped with the snippets
(`assert_raises`, `TestFailingBool`, etc.).

## Layout

- `snippets/` — imported RustPython tests and generic pyre-authored gaps,
  breadth-first surface coverage.  The pyre-authored ones carry
  `# pyre-check: gate=1`, which is what CI runs.  Runner:
  `pyre/extra_tests/run.py`.
- `parity_tests/` — pyre-authored scripts reserved for specific PyPy/JIT/GC
  invariants not already covered by `snippets/` or the vendored CPython suite
  under `lib-python/3/test/`.  Do not mirror general language, builtin, or
  stdlib behavior here. Each script must state both the missing CPython-suite
  coverage and why JIT/GC/PyPy internals require placement here; the runner
  rejects missing header fields. Each script cites the upstream file:line it guards;
  passing requires `exit 0` AND the final stdout line being `OK`.  Runner:
  `pyre/extra_tests/parity_tests/run.py`.
- `upstream/` — no tests of its own: a runner plus a driver for the
  vendored PyPy tree at the repository **root** `extra_tests/`.  Those
  files stay where upstream put them and run in place, so anything they
  already cover does not get rewritten under `parity_tests/`.
  Runner: `pyre/extra_tests/upstream/run.py`.

All three runners share the same backend discovery (cpython +
pyre-dynasm + pyre-cranelift) and exit code semantics.

## The vendored root `extra_tests/`

Those files are pytest modules: module-level `test_*` functions,
`pytest.raises`, `pytest.skip`.  The `pytest.py` / `_pytest/` copy
vendored beside them predates 3.12 and imports `imp`, so it loads under
neither interpreter, and no pytest is installed.  `upstream/driver.py`
registers a `pytest` module in `sys.modules` carrying `raises`, `skip`
and `fail`, then executes one file's tests in place:

```sh
python3 pyre/extra_tests/upstream/run.py              # the enabled set
python3 pyre/extra_tests/upstream/run.py --all -v     # survey the whole tree
./target/release/pyre-dynasm \
    pyre/extra_tests/upstream/driver.py extra_tests/test_os.py
```

`run.py`'s `ENABLED` list is what the gate runs; `--all` is a survey
switch.  Surveyed 2026-08-09 with `--all --cpython-only`: **11 of 59**
files run under CPython at all.  The other 48 need pytest surface the
shim does not carry (`mark`, `fixture`, `importorskip`), a third-party
package (`hypothesis`, `cffi`, `greenlet`), or a module that is PyPy's
(`_structseq`, `_immutables_map`).  Widening `ENABLED` means checking a
file against CPython and every pyre backend first.

## Source

The base corpus came from `RustPython/extra_tests/snippets/`. Future updates
should pull upstream changes while preserving generic pyre-authored gap tests;
only JIT/GC/PyPy-internal additions belong in `parity_tests/`.
