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

## The header

`include/pyre3.14t/` is what an extension compiles against.
An extension includes `Python.h`; that file is only the include list, and the
content is split by topic the way `pypy/module/cpyext/include` and CPython's
own `Include` are -- `object.h`, `longobject.h`, `modsupport.h`, and so on, one
per `cpyext` module. `pytypedefs.h` names the shared types once so no
declaration waits on a definition.

`pyre_decl.h` holds every exported entry point and **is generated** by
`scripts/cpyext-abi.py` from the `#[unsafe(no_mangle)] extern "C"` functions
themselves, so a declaration cannot drift from its implementation. It is
included after the headers that name the types it uses and before the ones
defining `static inline` functions that call it. `refcount.h` is the first of
those: `Py_INCREF` expands to the `Py_IncRef` it declares, and `Py_NewRef` is an
inline function calling that macro.

The generator does not spell the declaration from the Rust signature where
CPython has one of its own. It emits **CPython's** declaration -- `PyLongObject
*`, `Py_hash_t`, `PyCapsule_Destructor` -- having first checked that the two
describe the same call. That check is `scripts/cpyext-abi.py check`, and it is
a CI gate:

```sh
python3 pyre/scripts/cpyext-abi.py check          # every entry point vs. CPython
python3 pyre/scripts/cpyext-abi.py generate --check
```

An *entry point* is either an export or a `static inline` function in the
header. Both reach the extension as a call it compiles against, so both are
checked; a header inline is the one an export cannot catch, because it is
written by hand rather than generated.

It compares ABI slots rather than spellings, resolving CPython's typedefs, so
`Py_hash_t` and `Py_ssize_t` agree and `long` and `Py_ssize_t` do not. The
recorded declarations live in `scripts/cpython-abi.txt` because CI has no
CPython checkout; `snapshot` rewrites them from one, and that diff is the ABI
change a version bump makes.

This matters because nothing else here can see the failure. Every fixture in
this tree is compiled against pyre's own header, so a parameter declared at the
wrong width agrees with itself and only disagrees with the real world -- and
only on a platform where the widths differ. `PyModule_AddIntConstant` took a
`Py_ssize_t` where the reference declaration says `long`, which is the same on
LP64 and 32 bits against 64 on Windows.

The exports CPython does not declare are the macros it spells over struct
fields -- `PyLong_Check`, `PySequence_Fast_GET_ITEM`, `PyBytes_AS_STRING` --
which have to be calls here, plus the two `_PyPyre_*` helpers the header's own
`static inline` functions use.

`patchlevel.h` carries the version in parts -- `PY_MAJOR_VERSION`,
`PY_RELEASE_LEVEL`, `PY_RELEASE_SERIAL` -- and computes `PY_VERSION_HEX` from
them, so an extension testing any one of them reads the same number the runtime
reports. `PYRE_VERSION` is the interpreter's own version beside it, the slot
`PYPY_VERSION` fills upstream. `cpyext_methods` compares the whole set against
`sys.version_info`, `sys.hexversion` and `sys.pyre_version_info`.

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
- the type mirror `Py_TYPE(x)` answers with. A type an extension defined is its
  own `PyTypeObject` static, which `PyType_Ready` links in place and which is
  immortal because that storage is the library's; a type pyre defines gets a
  synthesized block carrying the ordinary link share, so a class the extension
  merely observed dies with the interpreter's own last reference to it. Its
  `tp_flags` carry `Py_TPFLAGS_HEAPTYPE` for a heap type, which is what decides
  whether a block holds a reference to its own type (`pyobject.py:91-93`,
  `object.py:72-73`) and whose storage the mirror is
  (`typeobject.py:716-722`);
- the two forms a mirror is handed out in before its interpreter object exists
  (`cpyext/unicodeobject.rs`, `cpyext/bytesobject.rs`): `PyUnicode_New(size,
  maxchar)` and `PyBytes_FromStringAndSize(NULL, size)` return an unlinked
  mirror over a buffer the caller fills, and `pyobject::realize` builds the
  `str` or `bytes` the first time it is read as a value, which is where
  `pyobject.py:330-337` reaches the type descriptor's `realize`. The type tests
  and the length are answered from the buffer, so asking them mid-fill does not
  decide the contents early;
- the C exception indicator (`cpyext/pyerrors.rs`): 37 `PyExc_*` class mirrors
  and `PyErr_SetString` / `SetObject` / `SetNone` / `Occurred` / `Clear` /
  `Fetch` / `Restore` / `ExceptionMatches` / `NoMemory` / `BadArgument` /
  `BadInternalCall`, plus the `PyErr_Format` half the header cannot do;
- the `PyCFunction` carrier (`cpyext/methodobject.rs`) and the call bridge for
  `METH_NOARGS`, `METH_O`, `METH_VARARGS`, `METH_VARARGS | METH_KEYWORDS`,
  `METH_FASTCALL` and `METH_FASTCALL | METH_KEYWORDS`;
- the call entry points in the other direction (`cpyext/object.rs`):
  `PyObject_Call` / `CallObject` / `CallNoArgs` / `CallOneArg`, and the
  vectorcall spellings `PyObject_Vectorcall`, `PyObject_VectorcallMethod` and
  `PyVectorcall_Call`, which unpack the argument vector and go through the
  interpreter's own call path rather than a type's `tp_vectorcall`. The
  variadic ones -- `PyObject_CallFunction` / `CallMethod`, the two `ObjArgs`
  spellings, `CallMethodNoArgs` / `CallMethodOneArg` and `PyVectorcall_NARGS`
  -- are `static inline` in the header, like `PyArg_ParseTuple`;
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
  `tp_call`, `tp_iter`, `tp_iternext`, `tp_richcompare`, `tp_getattro`,
  `tp_setattro`, `tp_descr_get` and `tp_descr_set`;
- heap types: `PyType_FromSpec`, `PyType_FromSpecWithBases`,
  `PyType_FromModuleAndSpec`, `PyType_GetSlot`, `PyType_GetName` and
  `PyType_GetQualName`, with every `typeslots.h` identifier except
  `Py_tp_vectorcall` and `Py_tp_token`, which name a type field pyre never
  reads;
- the `tp_as_number`, `tp_as_sequence` and `tp_as_mapping` tables, each slot
  becoming the dunder `slotdefs.py` names for it, and the `PyNumber_*`,
  `PySequence_*` and `PyMapping_*` entry points (`cpyext/number.rs`,
  `cpyext/sequence.rs`, `cpyext/mapping.rs`), which go through the
  interpreter's own operators so either operand may be a pyre object;
- the `tp_as_async` table as `__await__` / `__aiter__` / `__anext__`, with
  `am_send` reached through `PyIter_Send`, and the iterator entry points
  `PyObject_GetIter`, `PyObject_SelfIter`, `PyIter_Check`, `PyIter_Next`,
  `PyIter_NextItem`, `PyObject_GetAIter` and `PyAiter_Check`
  (`cpyext/iterator.rs`);
- the concrete `list`, `tuple` and `slice` protocols (`cpyext/listobject.rs`,
  `cpyext/tupleobject.rs`, `cpyext/sliceobject.rs`): all of `listobject.h` and
  `tupleobject.h`, and `PySlice_New` / `Check` / `Unpack` / `AdjustIndices` /
  `GetIndices` / `GetIndicesEx`.  Three of these are not exports:
  `PyTuple_Pack` is variadic, so it is a `static inline` in `Python.h` built
  from `PyTuple_New` and `PyTuple_SetItem`; `PySequence_Fast_GET_ITEM` and
  `PySequence_Fast_GET_SIZE` are functions rather than the macros the
  reference header spells, because a mirror carries no item array of its own
  and the length and items have to come from the interpreter object.
  `PySlice_GetIndicesEx` is both, matching the reference header: a macro over
  `PySlice_Unpack` and `PySlice_AdjustIndices` plus an export that is the same
  composition;
- the `tp_as_buffer` table (`cpyext/buffer.rs`): a C exporter's `bf_getbuffer`
  becomes a `memoryview` over the exported memory and its `bf_releasebuffer`
  runs when that view is released, so `memoryview(x)`, `bytes(x)` and the
  element writes all reach the exporter's own storage; in the other direction
  `PyObject_GetBuffer` / `PyBuffer_Release` / `PyBuffer_FillInfo` /
  `PyBuffer_IsContiguous` / `PyBuffer_SizeFromFormat` / `PyBuffer_GetPointer` /
  `PyBuffer_ToContiguous` / `PyBuffer_FromContiguous` / `PyObject_CopyData`, the
  legacy `PyObject_As*Buffer` family, and `PyMemoryView_FromObject` /
  `FromMemory` / `FromBuffer`;
- `PyErr_NewException` and `PyErr_NewExceptionWithDoc`, which build the class
  through the interpreter's own `type` so it gets the exception layout;
- capsules (`cpyext/capsule.rs`) and the strong-reference import entry points
  `PyImport_ImportModule`, `PyImport_Import`, `PyImport_AddModuleRef` and
  `PyImport_GetModule` (`cpyext/import_.rs`);
- GC forwarding for cached dictionaries, and `rawrefcount` in the collector
  itself (`majit-gc/src/rawrefcount.rs`, `majit-gc/src/collector.rs`): a mirror
  and its interpreter object are joined by a P-link, the collector traces and
  frees the two together in its own minor and major passes, and a mirror whose
  object died is handed back through a dead queue that `tp_dealloc` drains from
  an execution-context action. `tp_dealloc` and `tp_free` therefore run, and an
  instance of a C-defined type is reclaimed.

The loader serializes extension initialization in the current import path;
the complete port must move that ownership into interpreter/execution-context
state before subinterpreters or parallel no-GIL imports are enabled.

Known divergences, each documented at its definition:

- a reference cycle running through C leaks: a C reference marks its object as
  surviving along with everything the object reaches, and no `rawrefcount` pass
  consults `tp_traverse` / `tp_clear` to break such a cycle. That limit is
  upstream's as well (`pypy/doc/discussion/rawrefcount.rst`);
- `md_def` and `md_state` ride reserved module-dictionary keys, and the
  `PyMethodDef` pointer, the descriptor definitions and a capsule's four C
  values ride reserved carrier-dictionary keys, because pyre has no typed
  payload for any of them yet;
- a capsule's destructor is recorded and never called, and the borrowed-
  reference `PyImport_AddModule` / `PyImport_GetModuleDict` are absent: pyre has
  no container for the borrow to belong to;
- the module state block is never released, pyre having no module deallocation
  path;
- `PyObject_GC_IsTracked` is the constant 1 and `PyObject_GC_IsFinalized` the
  constant 0: every object pyre holds is reachable by its collector, and
  nothing runs `tp_finalize`;
- `PyObject_Malloc` / `Calloc` / `Realloc` hand out blocks from the same census
  the mirror allocator uses, so `PyObject_Free` releases either kind; a block
  from the `PyMem_*` family is a different allocator's and must go back to
  `PyMem_Free`. A request smaller than a `PyObject` header is rounded up to
  one, because freeing a block clears that header;
- `PyList_New(n)` fills the slots with `None` rather than NULL, `PyTuple_New(n)`
  leaving them NULL as CPython does;
- a type is built on a single base, so a `PyType_Spec` naming more than one is
  rejected rather than silently losing the rest, and `PyType_FromMetaclass`
  accepts only `type` for its metaclass: pyre builds a type through its own
  constructor, which has no place to put another one;
- a frozen type is one whose `flag_heaptype` is false, the state every builtin
  type is already in and the one `type.__setattr__` consults. CPython keeps
  the two bits apart, so a type frozen here also stops answering
  `__annotations__`;
- `PyType_ClearCache` empties the method cache and reports 0: upstream's return
  value is a per-interpreter version counter, and pyre mints one version
  identity per type, so no single number describes the entries dropped;
- `PyObject_GetBuffer` of an *interpreter* object hands C a read-only snapshot,
  the collector being free to move the storage a `Py_buffer` would otherwise
  name; a `PyBUF_WRITABLE` request for one is refused rather than answered with
  a copy whose writes go nowhere. The `w*` argument format therefore reaches
  only a C exporter's own `bf_getbuffer`, and raises `BufferError` for a
  `bytearray`;
- `PyBytes_AS_STRING` and `PyBytes_GET_SIZE` are the calls their checked
  spellings make, and `PyUnicode_GET_LENGTH` is `PyUnicode_GetLength`: neither
  a `bytes` nor a `str` mirror carries the field the reference header reads,
  the storage being a cached copy rather than a tail allocated with the
  object;
- a C exporter's view must be C-contiguous and free of suboffsets: a strided or
  indirect export is refused. The `Py_buffer` a `bf_getbuffer` filled in is kept
  and handed back to `bf_releasebuffer` unchanged -- `internal` is the
  exporter's own state -- keyed by the exported address, since the interpreter
  object it belongs to moves and the foreign memory does not;
- a `memoryview` that is never released never ends its export, pyre having no
  reference counting to end it at the last drop;
- three `rawrefcount` passes diverge from `incminimark.py` because the
  collector they sit in is not the one upstream wrote them for, each documented
  at its definition in `majit-gc/src/collector.rs`: the non-moving major
  additionally traces the *young* lists, that collection running no leading
  minor; a young mirror whose object was pinned stays young rather than being
  freed, a pinned survivor carrying no forwarding pointer to prove it alive;
  and a mirror linked to an object outside the managed heap is treated as
  alive, such an object having no header to read a mark from.

## What remains

5. `tp_traverse` and `tp_clear`, without which a cycle through C is not
   collectable; the *dispatch* half of vectorcall -- a type declaring
   `Py_TPFLAGS_HAVE_VECTORCALL` and `tp_vectorcall_offset` is called through
   `tp_call`, which such a type is required to have
   (`cpython/Objects/typeobject.c:8455-8459`), so the slot is an optimisation
   pyre does not take; and the remaining generated API. Of the 749 public
   `PyAPI_FUNC` entry points CPython 3.14 declares in its top-level
   `Include/*.h` -- public meaning the declared name does not begin with an
   underscore -- 342 are present, counting every form `Python.h` offers one
   in: an export, a `static inline`, or a macro of either kind. (The
   previously recorded 763/292 came from a pattern that missed the
   declarations annotated `_Py_NO_RETURN` on one side and the object-like
   aliases on the other; on the same header the corrected count is 293, so
   this figure moved by 49.) The figure counts only that population, so the
   private `_PyLong_*ByteArray` pair and the unchecked accessor macros the
   extensions below need do not appear in it;
6. Windows API DLL/import-library packaging.

A type built from a spec carries the module it was created from and the
`Py_tp_token` it declared, both in a side table keyed by the type's own
address: pyre has no heap-type struct, and a spec type is leaked deliberately,
so the key is stable for the life of the process. The module is held there as
an owned mirror reference, and it is that reference — a count above the link
share — that roots it, not the table.

`PySequence_Fast_ITEMS` is absent rather than pending: it hands out a
`PyObject **` into the sequence's own storage, which requires a list whose
items are a contiguous array of mirrors. Upstream converts the list to a
CPython-backed strategy for exactly this (`cpyext/listobject.py`
`get_list_storage`); pyre has no such strategy, so there is no array to point
at.

The public suffix uses `pyre314`, not `cpython-314`: accepting a CPython-tagged
binary would falsely claim that CPython's private object layouts and symbols
are ABI-compatible with pyre.

## A published extension

`mmh3` 5.2.1 -- MurmurHash3 bindings, hand-written C, no code generator -- was
compiled against this header and run. It builds three static types through
`PyType_Ready`, publishes them with `PyModule_AddObject` from a single-phase
`PyInit_mmh3`, and reaches the interpreter through `METH_FASTCALL`,
`METH_NOARGS`, `METH_O` and `METH_VARARGS | METH_KEYWORDS` methods, `tp_new` /
`tp_init` / `tp_dealloc` / `tp_getset`, the `s*` argument format, and
`_PyLong_FromByteArray`. Its module-level functions and all three hashers --
one-shot, streaming across several `update` calls, and `copy` -- answer what
CPython 3.14.6 answers for the same inputs.

Nothing in the tree builds it: the sources are not vendored, and the fixtures
under `pyrex/tests/fixtures` cover the entry points it needed rather than the
package itself.

## PyPy reference map

- `pypy/module/cpyext/api.py`: API declaration/generation, C-call boundary,
  extension loader and initialization checks.
- `pypy/module/cpyext/pyobject.py`: raw mirror descriptors and W_Root ↔
  `PyObject *` conversion.
- `rpython/rlib/rawrefcount.py` and `pypy/doc/discussion/rawrefcount.rst`: the
  vocabulary -- the P/O link kinds, `REFCNT_FROM_PYPY`, the dead queue -- and
  what a C reference does and does not keep alive.
- `rpython/memory/gc/incminimark.py`, the `rrc_*` methods: the algorithm
  itself. The front end declares; the collector owns the lists, the traces and
  the frees.
- `pypy/module/cpyext/state.py`: exception state, extension cache and GC dead
  queue.
- `pypy/module/cpyext/modsupport.py`: module definitions, method conversion,
  PEP 489 slots and the `PyModule_*` entry points. `PyModule_GetName` /
  `GetNameObject` read `__name__` from the module dictionary where
  `modsupport.py:240-276` reads `w_mod.w_name`, so a module renamed after
  import reports the new name; `PyModule_GetFilenameObject` reports a missing
  or non-`str` `__file__` as the `SystemError` it is documented to raise
  rather than as the `KeyError` a plain `getitem` produces.
- `pypy/module/cpyext/methodobject.py`: the `PyCFunction` carrier and its
  calling conventions.
- `pypy/module/cpyext/pyerrors.py`: the `PyErr_*` entry points.
- `pypy/module/cpyext/typeobject.py`: `PyType_Ready`, slot inheritance and the
  descriptors built from `tp_methods` / `tp_members` / `tp_getset`.
- `pypy/module/cpyext/slotdefs.py`: the wrappers that turn a C slot into an
  app-level method.
- `pypy/module/cpyext/structmemberdefs.py`: the `T_*` member type codes.
- `pypy/module/cpyext/number.py`, `sequence.py` and `mapping.py`: the
  `PyNumber_*`, `PySequence_*` and `PyMapping_*` entry points.
- `pypy/module/cpyext/object.py`: the object protocol, the object allocator and
  the `object.__getattribute__` / `__setattr__` / `__dict__` terminals.
  `PyObject_Bytes` runs a `__bytes__` override and then falls back to
  `PyBytes_FromObject`, which does not consult it -- so the two answer
  differently for an object that defines one. `PyObject_Hash` reports a hash of
  -1 as -2, -1 being the failure answer.
- `pypy/module/cpyext/longobject.py`: the `int` conversions. `PyLong_AsInt32` /
  `AsUInt32` / `AsInt64` / `AsUInt64` and their `From` counterparts are 3.14
  additions upstream predates, as are `PyLong_AsNativeBytes`,
  `PyLong_FromNativeBytes` and `PyLong_FromUnsignedNativeBytes`;
  `PyLong_GetInfo` reads `sys.int_info`, which is where the same numbers are
  already published.
- `pypy/module/cpyext/pycapsule.py` and `import_.py`: capsules and the import
  entry points.
- `pypy/module/cpyext/buffer.py` and `memoryobject.py`: `Py_buffer`, the
  `PyBuffer_*` entry points and `PyMemoryView_*`.
- `pypy/module/cpyext/iterator.py`: the iterator entry points.
- `pypy/module/cpyext/listobject.py`, `tupleobject.py` and `sliceobject.py`:
  the concrete list, tuple and slice entry points. Two divergences, both
  following the 3.14 specification where upstream predates it:
  `PySequence_Fast` answers a list where `sequence.py:52-66` answers a tuple,
  and `PyNumber_ToBase(n, 10)` is the plain decimal spelling rather than a
  radix format, `format_index_radix` covering only the power-of-two radices.
- `pypy/module/cpyext/src/getargs.c`: `PyArg_ParseTuple` and `Py_BuildValue`,
  which are C there too.
- `pypy/module/imp/interp_imp.py`: `_imp` entry points.
