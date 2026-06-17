# CPython regression suite for pyre

Runs the vendored CPython 3.14 regression suite (`lib-python/3/test/`) against
a built pyre interpreter and gates on regressions. This is the analog of
PyPy's `lib-python/conftest.py` and RustPython's `-m test` CI, adapted to
pyre's plain-Python runner convention (no pytest / RPython dependency).

## Design

- **Per-module subprocess.** Each test module runs in its own subprocess of
  the built interpreter, exactly like PyPy launches `pypy3-c -m test <module>`.
  Test pollution can't leak between modules.
- **Pristine vendored tests.** `lib-python/3/test/` stays an unmodified copy
  of CPython's `Lib/test`. Every skip / expected-status decision lives in the
  external `baseline.json` — we never edit test files (so stdlib upgrades stay
  a clean drop-in; see `lib-python/stdlib-upgrade.txt`).
- **External baseline.** `baseline.json` records the expected status of every
  module per backend. A module recorded `PASS` that stops passing is a
  regression and fails CI; a newly-passing module is reported (run
  `--update-baseline` to record it) but does not fail the run.

## Usage

```sh
# build the interpreter first
cargo build --release -p pyrex --bin pyre-dynasm --no-default-features --features dynasm

# gate against the baseline (default: dynasm, JIT on, script mode)
python3 pyre/cpython_tests/run.py

# explore the whole suite without gating, write a JSON report
python3 pyre/cpython_tests/run.py --full --report /tmp/report.json

# focus on one module / list selection / regenerate the baseline
python3 pyre/cpython_tests/run.py --filter test_json
python3 pyre/cpython_tests/run.py --list
python3 pyre/cpython_tests/run.py --full --update-baseline
```

Key flags: `--backend dynasm|cranelift`, `--no-jit` (`PYRE_NO_JIT=1`),
`--mode script|module|regrtest`, `--jobs N`, `--timeout S`, `--filter SUB`,
`--baseline PATH`, `--update-baseline`, `--strict-baseline`, `--full`,
`--report PATH`.

### Run modes

- `script` (default): `pyre <path>/test_xxx.py` — runs the file directly as
  `__main__`, firing its `unittest.main()`. The most robust mode today; it
  bypasses `runpy` / `importlib.util.find_spec`.
- `module`: `pyre -m test.<module>` — same entry via `runpy`. Blocked until
  `importlib.util.find_spec` is wired to pyre's importer (see backlog).
- `regrtest`: `pyre -m test -v <module>` — CPython's libregrtest, the
  PyPy/RustPython form, for once libregrtest imports cleanly.

## Result classes

`PASS` (rc 0) · `FAIL` (unittest ran but failed) · `IMPORTERROR` (module could
not even be imported/run — an interpreter or stdlib-compat gap) · `CRASH`
(rust panic / nonzero `internal_compile_panics` / signal) · `TIMEOUT` ·
`SKIP` (curated out in the baseline / `KNOWN_SKIPS`).

## CI

- `.github/workflows/pyre-ci.yml` job `cpython-tests` — gates PRs on the
  baseline-`PASS` subset, dynasm with **JIT on** (`MAJIT_STRICT=1`), on
  `macos-latest` (aarch64). The baseline is recorded on darwin-aarch64 and the
  JIT codegen is architecture-specific, so the gate runs on the same arch the
  baseline was observed on (x86_64 JIT-on is a separate, unstable surface).
- `.github/workflows/pyre-cpython-nightly.yml` — non-gating nightly `--full`
  across three lanes (dynasm JIT-on, dynasm JIT-off, cranelift) with reports
  uploaded as artifacts. A module that passes JIT-off but not JIT-on is a JIT
  correctness divergence.

## Current state and backlog (Phase 0)

The baseline currently records **32 `PASS`**, 381 `IMPORTERROR`, 20 `SKIP`,
1 `CRASH` (434 modules, stdlib 3.14.6). The `PASS` set grows as the gaps
below are closed; the rest still hit an interpreter or stdlib-compat gap
before their tests can run. (It was 0 `PASS` / 414 `IMPORTERROR` before the
`-m`/`STORE_SLICE`/submodule-binding/`_ast`/PEP-709-hidden-locals fixes
below.)

Building this infra already surfaced and fixed five interpreter gaps:

- **`-m module`** support (`pyrex/src/lib.rs` — `runpy._run_module_as_main`).
- **`STORE_SLICE`** (`obj[i:j] = v`) — `eval.rs` / `runtime_ops.rs`.
- **submodule attribute binding** — `import a.b` now binds `b` on `a`
  (`importing.rs::absolute_import`, the `_find_and_load` step). Fixes
  `import xml.etree.ElementTree`, `importlib.util`, and a broad class of
  stdlib imports.
- **`_ast` node types are now heap types** (`module/_ast/interp_ast.rs`,
  built via `type(name, bases, {})` along the ASDL hierarchy). `ast.py` can
  monkeypatch / subclass them. Fixes `import ast` and, transitively,
  `collections`, `json`, `functools`, `enum`, `textwrap`, `argparse`, …
- **PEP 709 hidden comprehension locals at module scope** (`eval.rs`
  `store_name_value` + `pyframe.rs` fast↔locals sync). A module-level inlined
  comprehension makes its iteration variable a `CO_FAST_HIDDEN` fast local
  whose binding lives in the module dict (via `STORE_NAME`), not the NULL
  fast slot. Two parity fixes: **(a)** `STORE_NAME` now writes straight to
  `w_locals` (PyPy `pyopcode.py:855` `space.setitem_str(getorcreatedebug().
  w_locals, …)`) instead of via `getdictscope()` — the old path ran
  `fast2locals` on every store, which deleted the hidden name from the dict
  (the module's `w_locals` *is* its globals). **(b)** `fast2locals` /
  `locals2fast` skip `CO_FAST_HIDDEN` slots (CPython `frameobject.c`; PyPy
  has no PEP 709) so an explicit `locals()`/`vars()` can't erase them either.
  This was **the gateway**: it unblocks `import dis` → `inspect` →
  `unittest`. Repro: `y=[op for op in range(2)]; op=5; i=6; print(op)`
  (byte-identical to CPython 3.14 after the fix).

Known remaining blockers (the backlog the suite tracks), highest-leverage
first:

1. **`from package import submodule` fallback.** `from test import support`
   fails when `support` isn't already an attribute; IMPORT_FROM must fall back
   to importing the submodule. Also `from unittest import mock`
   (`unittest.mock` submodule).
2. **`importlib.util.find_spec` returns `None`.** pyre's native importer is
   not wired into importlib's `meta_path`, so `runpy` (`-m`) can't locate
   modules. Blocks `--mode module`/`regrtest`.
3. **Compiler gap (external `rustpython-codegen`): `class_decorator` symbol
   missing from the symbol table** — `compile error: the symbol
   'class_decorator' must be present in the symbol table` (hit by
   `test_grammar`). Lives in the git-pinned compiler dependency, not pyre's
   tree.
4. **Assorted stdlib-compat gaps** surfaced per module (e.g. `datetime.date`,
   `genericpath._splitext`, `typing`'s `_idfunc`, complex `re` patterns). Run
   `--full --report` to enumerate the current set.
