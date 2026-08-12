# cpyext in pyre

Pyre follows PyPy's `pypy/module/cpyext` architecture.  It does not expose an
internal pyre object as a CPython `PyObject *`: the two layouts are unrelated.
Instead, a C extension sees a fixed-address, reference-counted mirror carrying
an `ob_pyre_link` to the interpreter's GC object.  The interpreter's global GC
root walker forwards that link while an external C reference owns it.

## Build option

Extension loading is behind the non-default `cpyext` cargo feature. The switch
itself is upstream's: `has_so_extension` (`pypy/module/imp/importing.py:60`)
reads `config.objspace.usemodules.cpyext` and gates `create_dynamic`
(`pypy/module/imp/interp_imp.py:49-51`). Build it with

```sh
cargo build --release -p pyrex --bin pyre-dynasm \
  --no-default-features --features dynasm,cpyext
```

The feature also decides whether `_imp.extension_suffixes()` answers with a
suffix. **That part is a deliberate divergence**: upstream
`extension_suffixes` (`pypy/module/imp/interp_imp.py:11-15`) answers
unconditionally, so a `--withoutmod-cpyext` PyPy still names a suffix whose
`create_dynamic` then raises. A suffix names a loader, so pyre keeps the two
together — without the feature the import path advertises nothing it cannot
load, which is also what pyre answered before extension loading existed.

The consequence is why the feature is off rather than on:
`importlib._bootstrap_external.EXTENSION_SUFFIXES` is that call's result, and
`test_importlib/extension` skips its whole suite while the list is empty. A
non-empty list un-skips 39 tests that load CPython's `_testsinglephase` and
`_testmultiphase`; the ABI header below builds neither yet. Turn the feature on
by default once slices 1–5 below land and both modules build (slice 6 is
Windows packaging, which the os gates make irrelevant to the default).

The initial macOS/Linux supported slice is deliberately narrow but end-to-end:

- a pyre-specific Python 3.14 ABI header and extension suffix;
- `_imp.extension_suffixes()`, `_imp.create_dynamic()` and
  `_imp.exec_dynamic()`;
- `dlopen`/`dlsym` of `PyInit_<name>` with the library handle retained;
- `PyModuleDef_Init` and single-phase `PyModule_Create2`;
- PyPy-style path-to-module-dictionary extension caching;
- GC forwarding for C mirror links and cached dictionaries.

The loader serializes extension initialization in the current import path;
the complete port must move that ownership into interpreter/execution-context
state before subinterpreters or parallel no-GIL imports are enabled.

This imports a single-phase, method-less, stateless `PyModuleDef` extension. A
non-empty `PyMethodDef` table, module state/finalizer fields, and PEP 489 slots
are rejected rather than silently ignored. The following slices remain, in
upstream dependency order:

1. execution-context-owned C exception indicator and `PyErr_*`;
2. a dedicated `PyCFunction` carrier with `METH_NOARGS`, `METH_O`, then
   `METH_VARARGS`/keywords;
3. primitive object mirrors and APIs (`long`, Unicode, bytes, tuple/list);
4. PEP 489 create/exec slots and module state;
5. C-defined types, slots, buffers, capsules and the remaining generated API;
6. Windows API DLL/import-library packaging.

The public suffix uses `pyre314`, not `cpython-314`: accepting a CPython-tagged
binary would falsely claim that CPython's private object layouts and symbols
are ABI-compatible with pyre.

## PyPy reference map

- `pypy/module/cpyext/api.py`: API declaration/generation, C-call boundary,
  extension loader and initialization checks.
- `pypy/module/cpyext/pyobject.py`: raw mirror descriptors and W_Root ↔
  `PyObject *` conversion.
- `rpython/rlib/rawrefcount.py` and `pypy/doc/discussion/rawrefcount.rst`:
  collector ownership and delayed deallocation.
- `pypy/module/cpyext/state.py`: exception state, extension cache and GC dead
  queue.
- `pypy/module/cpyext/modsupport.py`: module definitions, method conversion and
  PEP 489 slots.
- `pypy/module/imp/interp_imp.py`: `_imp` entry points.
