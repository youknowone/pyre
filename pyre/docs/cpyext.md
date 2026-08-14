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
by default once slice 5 below lands and both modules build (slice 6 is Windows
packaging, which the os gates make irrelevant to the default).

## What is implemented

The macOS/Linux slice is end-to-end: `pyrex/tests/cpyext_smoke.rs` imports a
single-phase extension, `pyrex/tests/cpyext_methods.rs` a multi-phase one and
`pyrex/tests/cpyext_types.rs` one defining types, all compiled from C against
the header below.

- a pyre-specific Python 3.14 ABI header and extension suffix;
- `_imp.extension_suffixes()`, `_imp.create_dynamic()` and
  `_imp.exec_dynamic()`;
- `dlopen`/`dlsym` of `PyInit_<name>` with the library handle retained;
- the raw mirror layer (`cpyext/pyobject.rs`): the identity table, borrowed
  references owned by their container, the per-mirror byte cache behind
  `PyUnicode_AsUTF8` / `PyBytes_AsString`, and the immortal singletons
  `Py_None` / `Py_True` / `Py_False` / `Py_Ellipsis` / `Py_NotImplemented`;
- the C exception indicator (`cpyext/pyerrors.rs`): 37 `PyExc_*` class mirrors
  and `PyErr_SetString` / `SetObject` / `SetNone` / `Occurred` / `Clear` /
  `Fetch` / `Restore` / `ExceptionMatches` / `NoMemory` / `BadArgument` /
  `BadInternalCall`, plus the `PyErr_Format` half the header cannot do;
- the `PyCFunction` carrier (`cpyext/methodobject.rs`) and the call bridge for
  `METH_NOARGS`, `METH_O`, `METH_VARARGS`, `METH_VARARGS | METH_KEYWORDS`,
  `METH_FASTCALL` and `METH_FASTCALL | METH_KEYWORDS`;
- single-phase `PyModule_Create2`, PEP 489 `PyModuleDef_Init` with
  `Py_mod_create` / `Py_mod_exec`, module state (`m_size`) and the
  `PyModule_Add*` family;
- the object protocol (`cpyext/object.rs`) and the primitive constructors and
  accessors for `int`, `bool`, `float`, `str`, `bytes`, `tuple`, `list` and
  `dict`;
- `PyArg_ParseTuple`, `PyArg_ParseTupleAndKeywords`, `PyArg_UnpackTuple`,
  `Py_BuildValue` and `PyErr_Format`, which are `static inline` C in the header
  because pyre ships no companion library and rustc's `c_variadic` is unstable;
- PyPy-style path-to-module-dictionary extension caching for single-phase
  modules — a multi-phase module is rebuilt from its definition on each import,
  as upstream's `create_cpyext_module` does;
- C-defined types (`cpyext/typeobject.rs`): `PyType_Ready` with `tp_base`
  inheritance and `inherit_slots`, `PyType_GenericAlloc` / `PyType_GenericNew` /
  `PyObject_Init`, `PyType_IsSubtype` / `PyType_GetFlags` / `PyObject_TypeCheck`,
  the `tp_methods` / `tp_members` / `tp_getset` descriptors with `__objclass__`,
  and wrappers for `tp_new`, `tp_init`, `tp_repr`, `tp_str`, `tp_hash`,
  `tp_call`, `tp_iter`, `tp_iternext` and `tp_richcompare`;
- `PyErr_NewException` and `PyErr_NewExceptionWithDoc`, which build the class
  through the interpreter's own `type` so it gets the exception layout;
- GC forwarding for C mirror links and cached dictionaries.

The loader serializes extension initialization in the current import path;
the complete port must move that ownership into interpreter/execution-context
state before subinterpreters or parallel no-GIL imports are enabled.

Known divergences, each documented at its definition:

- a mirror is freed as soon as C releases its last reference, so a reference
  cycle running through C leaks; upstream's `rawrefcount` dead queue is what
  removes that (`cpyext/pyobject.rs`);
- `md_def` and `md_state` ride reserved module-dictionary keys, and the
  `PyMethodDef` pointer rides a reserved carrier-dictionary key, because pyre
  has no typed payload for either yet;
- the module state block is never released, pyre having no module deallocation
  path;
- `PyBytes_FromStringAndSize(NULL, n)` is rejected: pyre's `bytes` is immutable
  from construction and its storage is not the address `PyBytes_AsString`
  returns;
- `PyList_New(n)` fills the slots with `None` rather than NULL, `PyTuple_New(n)`
  leaving them NULL as CPython does;
- an instance of a C-defined type is immortal, because its mirror block *is* its
  storage: freeing the block when C drops its last reference would destroy
  fields the interpreter object still exposes. Such an instance is therefore
  never reclaimed. Removing that needs the same `rawrefcount` dead queue the
  first divergence names, so that a block is released only once the collector
  has proved the interpreter object dead as well.

## What remains

5. `tp_dealloc`, `tp_traverse` and `tp_clear` on top of a `rawrefcount` dead
   queue; the protocol tables (`tp_as_number`, `tp_as_sequence`, `tp_as_mapping`,
   `tp_as_buffer`), which are declared but not read; heap types
   (`PyType_FromSpec`); capsules; and the remaining generated API;
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
- `pypy/module/cpyext/methodobject.py`: the `PyCFunction` carrier and its
  calling conventions.
- `pypy/module/cpyext/pyerrors.py`: the `PyErr_*` entry points.
- `pypy/module/cpyext/typeobject.py`: `PyType_Ready`, slot inheritance and the
  descriptors built from `tp_methods` / `tp_members` / `tp_getset`.
- `pypy/module/cpyext/slotdefs.py`: the wrappers that turn a C slot into an
  app-level method.
- `pypy/module/cpyext/structmemberdefs.py`: the `T_*` member type codes.
- `pypy/module/cpyext/src/getargs.c`: `PyArg_ParseTuple` and `Py_BuildValue`,
  which are C there too.
- `pypy/module/imp/interp_imp.py`: `_imp` entry points.
