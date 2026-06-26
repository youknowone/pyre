# #264 — memoryview rewrite around a real BufferView struct

## Stage 0 — prerequisites (do these FIRST, before any memoryview code)

1. **Save this plan into the repo** as `memoryview.plan.md` (repo root) so it is
   tracked alongside the work.
2. **Fix `scripts/extract-llbc.py`.** The memoryview rewrite edits
   `pyre/pyre-object/src/` (new `memoryview.rs` + gutted stub), and pyre-object is
   charon-extracted, so the LLBC extraction pipeline must be working before the
   build can pick up the change. The read-only paths already pass
   (`--list-inputs` / `--fingerprint` for pyre-object → exit 0), so the failure is
   in the real charon-extraction path. Diagnose by running the full extraction
   (`python3 scripts/extract-llbc.py pyre-object pyre-interpreter pyre-jit`),
   identify the failure, fix it, and re-run until extraction succeeds and the
   `.ullbc` artefacts + `.fingerprint` stamps are written under `build/llbc/`.
   Land this as its own commit (`Assisted-by: Claude`, no push) before starting
   Stage 1.

   _Root cause found (2026-06-26):_ `BhDescr::Size` (majit-translate
   `codewriter/jitcode.rs:1216`) gained an `is_gc_managed: bool` field; the
   `bh_new_with_vtable` construction site at `codewriter/assembler.rs:1495` was
   not updated (`error[E0063]`). The runtime build did not surface it (it reused a
   cached `majit-translate`); only the extraction build (which recompiles
   `pyre-jit` → `majit-translate`) hits it. Fix = set `is_gc_managed: true`
   (vtable-bearing GC allocation; matches the serde default and sibling sites).

## Context

pyre's `memoryview` is a **dict-backed stub**, not a real object. The type is
created with `make_builtin_type("memoryview")` and each instance is a generic
`w_instance_new` carrying three magic attribute slots — `__pyre_buf__`,
`__pyre_fmt__`, `__pyre_itemsize__` (`pyre-interpreter/src/builtins.rs:417-648`).
Every "buffer" access **copies** the backing bytes via `memoryview_data`
(`builtins.rs:14-28`, `.to_vec()`), so the view is detached from its source.

This produces real defects vs. CPython/PyPy semantics:
- **Slice `__getitem__` returns a COPY, not a live sub-view** (`builtins.rs:72-95`):
  `m[1:][0] = …` cannot write through, and the slice is always read-only.
- **`ndim` hardcoded to 1** (`builtins.rs:215-217`); **`shape`/`strides`/`obj`
  absent entirely**; no `release`/`__enter__`/`__exit__`/`hex`/`__hash__`/
  `__delitem__`/`c_contiguous`/`f_contiguous`/`contiguous`/`suboffsets`.
- **Element unpack is always unsigned little-endian** (`builtins.rs:45-52`), so
  signed/float/big-endian formats (`i`,`f`,`d`,`>I`) decode wrong; `cast` only
  changes itemsize.
- **No buffer-protocol abstraction anywhere** — no `Buffer`/`BufferView`,
  `memoryview(obj)` accepts any object without a `TypeError` for non-buffers.

The acceptance spec is `pyre/extra_tests/snippets/builtin_memoryview.py` (93 lines),
which is far ahead of the stub (it exercises `obj`, live slice-views, `release()`,
`BufferError` on exported-mutation, `__delitem__`→TypeError).

CLAUDE.md forbids "find another way" (i.e. adding more magic attribute slots).
The faithful target is PyPy's design: a byte-level `Buffer` + an item-level
`BufferView` (format/itemsize/ndim/shape/strides), with `W_MemoryView` wrapping a
`BufferView` (`view = None` ⟺ released). This is a multi-stage architectural epic.

## Faithfulness target (PyPy)

- `pypy/objspace/std/memoryobject.py` — `W_MemoryView` (thin; delegates to `self.view`).
- `pypy/interpreter/buffer.py` — `BufferView` (item layer: format/itemsize/ndim/
  shape/strides + `get_offset`/`w_getitem`/`new_slice`/`value_from_bytes`/
  `bytes_from_value`); concrete `SimpleView` (bytes/bytearray, `'B'`/1/strides`[1]`),
  `RawBufferView` (array), `BufferSlice` (strided sub-view), `ReadonlyWrapper`,
  `BufferView1D`/`BufferViewND` (cast results).
- `rpython/rlib/buffer.py` — byte layer `Buffer`/`SubBuffer`.

## Design decision — flatten the two PyPy layers into one Rust struct

PyPy's `W_MemoryView`→`BufferView`→`Buffer` split exists for RPython's type
system; at runtime the behavior is a single view over a byte backing. The port
flattens it into one `#[pyre_class]` struct (blessed structural adaptation, same
class as the #37-audited collapses). **No `Vec`/`Box`/`String` fields** — every
existing `#[pyre_class]` struct holds only `PyObjectRef` + scalar fields, and the
GC tracer / charon extraction expect that, so shape/strides are stored as Python
tuple `PyObjectRef`s.

New `pyre/pyre-object/src/memoryview.rs` (precedent: `W_Range`,
`functional.rs:647`, the #263 stub→real-type conversion):

```
#[pyre_class("memoryview", type_id = 58, static_name = "MEMORYVIEW")]
pub struct W_MemoryView {
    pub w_obj:     PyObjectRef,  // exporter (bytes/bytearray/array); `obj` property
    pub w_backing: PyObjectRef,  // byte-storage object actually read/written
    pub w_format:  PyObjectRef,  // format str
    pub w_shape:   PyObjectRef,  // tuple[int]
    pub w_strides: PyObjectRef,  // tuple[int]
    pub itemsize:  i64,
    pub ndim:      i64,
    pub offset:    i64,          // byte offset into backing
    pub length:    i64,          // total bytes in the view
    pub readonly:  bool,
    pub released:  bool,
}
```

Byte access reads/writes the **live** backing through existing accessors — no
copy: `bytes_like_data` (`pyre-object/src/bytesobject.rs:133`),
`w_bytearray_data`/`w_bytearray_data_mut` (`bytearrayobject.rs:103/112`),
`w_array_bytes` (`interp_array.rs:122`). A slice builds a new `W_MemoryView`
sharing `w_obj`/`w_backing` with `offset += start*stride0`, `shape=[slicelength]`,
`strides=[stride0*step]` — fixing the copy bug structurally.

Method registration follows the native-type pattern (type via `#[pyre_class]` +
`MEMORYVIEW_TYPE`; methods/properties registered as today but reading struct
fields, not dict slots). `buffer_as_bytes_like` (`typedef.rs:9449`) and
`interp_buffer.rs` PickleBuffer extraction get unified onto the new accessor.

## Staged implementation (each stage: `cargo check -p pyre-interpreter --features dynasm` + check.py dynasm & cranelift 153/153, then extend `builtin_memoryview.py`)

**Stage 1 — foundational struct + live 1-D access (the headline fix).**
Replace the dict-backed stub with `W_MemoryView` (`pyre-object/src/memoryview.rs`)
+ `w_memoryview_new(w_obj)` doing buffer acquisition: bytes/bytearray→`('B',1)`
readonly-or-not; array→typecode/itemsize; another memoryview→share view;
**`TypeError` for non-buffer args** (list/tuple/dict/str/instance). Port every
currently-working method onto the struct with **live** byte access:
`__getitem__` (scalar 'B' + **slice→live sub-view**), `__setitem__`, `__len__`,
`__iter__`, `__contains__`, `tobytes`, `tolist`, `__eq__`/`__ne__`, `__repr__`,
`toreadonly`, `cast` (1-D itemsize only), and properties `format`/`itemsize`/
`nbytes`/`readonly` **plus new `ndim` (real)**, **`shape`**, **`strides`**,
**`obj`**. Keep the `'B'`/unsigned-LE element path for now (general formats →
Stage 3). Files: new `memoryview.rs`; `builtins.rs:13-648` (gut stub, rewire
`__new__` + dispatch); `typedef.rs:9449` (unify resolver).

**Stage 2 — released state machine.** `release`/`__enter__`/`__exit__`/
`__release_buffer__`; `released` flag + a `_check_released` helper at the top of
every method (ValueError "operation forbidden on released memoryview object");
`__repr__`→"released memory"; `obj`→`None` after release. Extend
`builtin_memoryview.py` `test_resizable` release path.

**Stage 3 — format-aware scalar pack/unpack.** Signed/float/endian for
`__getitem__`/`__setitem__`/`cast` formats, mirroring PyPy `value_from_bytes`/
`bytes_from_value` (reuse pyre's `struct` module machinery if present, else a
small format decoder). `SimpleView` `'B'` fast path stays.

**Stage 4 — remaining typedef members.** `hex` (reuse `_array_to_hexstring`
path), `__hash__` (readonly-gated, cached), `__delitem__`→TypeError,
`c_contiguous`/`f_contiguous`/`contiguous` flags, `suboffsets` (always `()`).

**Stage 5 — cast-to-ND + multi-dim.** `BufferViewND` semantics: tuple-index
`__getitem__`/`__setitem__`, C-contiguous `_strides_from_shape`; multi-dim
*slicing* raises `NotImplementedError` (parity stub).

**Stage 6 — BufferError export-lock.** Add an export-count to `bytearray`
(`bytearrayobject.rs`); `w_memoryview_new` increments, `release` decrements;
`bytearray` mutators (append/extend/insert/resize) raise `BufferError` while
exported. Satisfies `builtin_memoryview.py` `test_resizable` lines 57-76.

## Risks / notes
- The Stage-1 dict-slots→struct switch is **atomic** (storage changes for all
  methods at once); it must keep check.py 153/153 (memoryview is light in that
  corpus) and not regress the parts of `builtin_memoryview.py` the stub already
  passed. `builtin_memoryview.py` is in `extra_tests/snippets`, **not** the
  check.py 153 gate — confirm and use it as the acceptance target, run via
  `./target/release/pyre-dynasm pyre/extra_tests/snippets/builtin_memoryview.py`.
- `type_id = 58` (max existing is 57); pick the next free id and a `MEMORYVIEW`
  static name.
- GC: `w_obj`/`w_backing`/`w_format`/`w_shape`/`w_strides` are `PyObjectRef`
  fields (traced, write-barrier on mutation); the rest are inline scalars.
- memoryview is a runtime builtin; it is not JIT-traced (dual-gate Skip), so the
  new struct does not enter the JIT walker — but it IS charon-extracted with
  pyre-object, hence the no-`Vec`/no-`Box` field rule above.

## Verification (per stage)
- `cargo check -p pyre-interpreter --features dynasm` (build), `cargo test -p
  pyre-interpreter --features dynasm` (unit).
- `python pyre/check.py --backend dynasm` and `--backend cranelift` → 153/153 each
  (re-run cranelift once if `raise_catch` wall-clock flakes).
- `./target/release/pyre-dynasm pyre/extra_tests/snippets/builtin_memoryview.py`
  diffed against `python3` for the slice/release/obj cases the stage covers.
- Commit per stage, `Assisted-by: Claude` trailer, **no push**.
