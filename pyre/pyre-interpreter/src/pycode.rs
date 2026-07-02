//! PyCode — Python `code` object wrapper.
//!
//! Wraps an opaque pointer to the compiler's CodeObject, allowing it to
//! be placed on the value stack as a PyObjectRef during `LoadConst`.
//! MakeFunction then extracts this pointer to build a function object.

use pyre_object::pyobject::*;

/// Compatibility marker for malformed bytecode.
#[derive(Debug, Clone)]
pub struct BytecodeCorruption;

impl From<BytecodeCorruption> for crate::PyError {
    fn from(_: BytecodeCorruption) -> Self {
        crate::PyError::new(
            crate::PyErrorKind::BytecodeCorruption,
            "bytecode corruption",
        )
    }
}

/// pycode.py:683-695 — decode one CPython-3.11 varint at `i`.
///
/// Returns `(value, new_i)`. Reads 6 bits per byte, MSB first. Bit 6
/// (0x40) is the continuation flag; bit 7 (0x80) is the start-of-entry
/// marker, ignored here and masked off along with the continuation bit
/// via `& 63`.
#[inline]
pub fn decode_varint(table: &[u8], mut i: usize) -> (u32, usize) {
    let mut b = table[i] as u32;
    i += 1;
    let mut value = b & 63;
    while b & 64 != 0 {
        b = table[i] as u32;
        i += 1;
        value = (value << 6) | (b & 63);
    }
    (value, i)
}

/// Decoded exception-table entry. Byte offsets throughout.
///
/// Field shape mirrors PyPy's `(start, length, target, depth, lasti)`
/// per-entry varint sequence; `end = start + length` is precomputed for
/// callers that want a half-open `start..end` range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExceptionTableEntry {
    pub start: u32,
    pub end: u32,
    pub target: u32,
    pub depth: u32,
    pub lasti: bool,
}

/// pycode.py:229-254 `lookup_exceptiontable`.
///
/// Search `table` for a handler covering `instr_offset` (byte offset
/// into `co_code`). Returns `Some((target, depth, lasti))` when found,
/// `None` otherwise.
///
/// **Last matching wins**: entries are scanned in encoding order; if
/// multiple entries cover `instr_offset`, the later one (innermost in
/// CPython's emission order) is returned. Scanning short-circuits when
/// `start > instr_offset`, since entries are emitted in ascending
/// `start` order.
pub fn lookup_exceptiontable(table: &[u8], instr_offset: u32) -> Option<(u32, u32, bool)> {
    let n = table.len();
    if n == 0 {
        return None;
    }
    let mut best: Option<(u32, u32, bool)> = None;
    let mut i = 0;
    while i < n {
        let (start_raw, ni) = decode_varint(table, i);
        let start = start_raw * 2;
        let (length_raw, ni) = decode_varint(table, ni);
        let length = length_raw * 2;
        let (target_raw, ni) = decode_varint(table, ni);
        let target = target_raw * 2;
        let (dl, ni) = decode_varint(table, ni);
        let depth = dl >> 1;
        let lasti = (dl & 1) != 0;
        i = ni;
        if start <= instr_offset && instr_offset < start + length {
            best = Some((target, depth, lasti));
        } else if start > instr_offset {
            break;
        }
    }
    best
}

/// Iterator over all decoded entries in `table`.
///
/// Convenience for callers that want a structural view (JIT codewriter,
/// liveness, the PyPy-style `mark_stacks` handler-shape seeder). The
/// runtime `handle_operation_error` dispatch uses [`lookup_exceptiontable`]
/// directly.
pub fn decode_exceptiontable(table: &[u8]) -> ExceptionTableIter<'_> {
    ExceptionTableIter { table, i: 0 }
}

pub struct ExceptionTableIter<'a> {
    table: &'a [u8],
    i: usize,
}

impl Iterator for ExceptionTableIter<'_> {
    type Item = ExceptionTableEntry;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.table.len() {
            return None;
        }
        let (start_raw, i) = decode_varint(self.table, self.i);
        let start = start_raw * 2;
        let (length_raw, i) = decode_varint(self.table, i);
        let length = length_raw * 2;
        let (target_raw, i) = decode_varint(self.table, i);
        let target = target_raw * 2;
        let (dl, i) = decode_varint(self.table, i);
        self.i = i;
        Some(ExceptionTableEntry {
            start,
            end: start + length,
            target,
            depth: dl >> 1,
            lasti: (dl & 1) != 0,
        })
    }
}

/// Compatibility container for code-hook caching state.
#[derive(Debug, Default)]
pub struct CodeHookCache {
    _code_hook: Option<PyObjectRef>,
}

/// Type descriptor for code objects.
pub static CODE_TYPE: PyType = pyre_object::pyobject::new_pytype("code");

/// Python code object wrapper.
///
/// Stores an opaque pointer to the bytecode CodeObject. The pointer is
/// `Box::into_raw`'d from a cloned CodeObject, so we own the allocation.
#[repr(C)]
pub struct PyCode {
    pub ob_header: PyObject,
    /// Opaque pointer to a `CodeObject` (owned via Box::into_raw).
    pub code_ptr: *const (),
    /// PyPy: `PyCode.w_globals` — the globals dict OBJECT (`W_DictMultiObject`,
    /// `pycode.py:105 "w_globals?"`).  Module globals are `malloc_typed`-
    /// immortal, but `exec`/custom-globals dicts are `try_gc_alloc` movable.
    /// The code object is Box-immortal, so the collector never reaches this
    /// slot by tracing into it; `eval::walk_raw_code_roots` forwards it as a
    /// root (via `walk_raw_function_roots` for `func.code` and the frame walk
    /// for `frame.pycode`).  Null until first stamped by `frame_stores_global`.
    /// The off-GC `DictStorage` storage is recovered on demand via
    /// `w_globals_storage`.
    pub w_globals: PyObjectRef,
    /// PyPy: `PyCode.hidden_applevel` (`pycode.py:111, 147`). Set by
    /// `pycompiler.compile(hidden_applevel=True)` for PyPy gateway/
    /// app_main bridge code.  Pyre has no such call site yet, so this
    /// is always `false` on currently constructed instances; the
    /// field exists so that `frame.hide()` can read the canonical
    /// `pyframe.py:521-522 return self.pycode.hidden_applevel`.
    pub hidden_applevel: bool,
    /// pycode.py:226-238 `_compute_flatcall`. Cached arity descriptor:
    /// - 0-4: impossible (builtins only)
    /// - FLATPYCALL | co_argcount: simple user function
    /// - HOPELESS: has *args/**kwargs/kwonly/too many params
    pub fast_natural_arity: u16,
    /// `pycode.py:198 self._globals_caches = [None] * len(self.co_names_w)`.
    ///
    /// Per-name slot for `LOAD_GLOBAL_cached` / `STORE_GLOBAL_cached`
    /// (`celldict.py:292,321,335,353`).  Stores a weak reference to
    /// the `GlobalCache` resolved on the first miss, so subsequent
    /// hits bypass the `mstrategy.get_global_cache(varname)` string
    /// lookup.
    ///
    /// Owned via `Box::into_raw`; allocated once at construction sized
    /// to `code.names.len()`, never resized.  `null` when `code_ptr`
    /// is null or unaligned (test fixtures, gateway builtins).
    pub globals_caches:
        *mut Vec<Option<std::rc::Weak<std::cell::RefCell<pyre_object::celldict::GlobalCache>>>>,
    /// `mapdict.py:1457-1458 self._mapdict_caches = [INVALID_CACHE_ENTRY] *
    /// len(co_names_w)`.
    ///
    /// Per-name slot for the `LOAD_ATTR_caching` / `STORE_ATTR_caching` inline
    /// attribute cache (`mapdict.py:1480/1574`).  A `None` slot is PyPy's
    /// `INVALID_CACHE_ENTRY` (mapdict.py:1452); a `Some` holds the immortal map
    /// node + attribute node + `version_tag` last resolved for this slot, so a
    /// monomorphic re-read skips the type lookup + map walk.  The
    /// LOAD_METHOD fill additionally stores a movable `w_method`
    /// reference (mapdict.py:1418), forwarded during collection by
    /// `walk_mapdict_method_cache_gc`; the other fields are immortal
    /// node pointers and need no walking.
    ///
    /// Owned via `Box::into_raw`, sized to `code.names.len()` at construction,
    /// never resized; `null` when `code_ptr` is null or unaligned.
    pub mapdict_caches: *mut Vec<Option<crate::objspace::std::mapdict::MapdictCacheEntry>>,
    /// `pycode.py:126 self.co_consts_w = consts` (`_immutable_fields_
    /// co_consts_w[*]`, pycode.py:97).  The realized constant objects indexed by
    /// constant index.  `getconstant_w(index)` (`pyopcode.py:498-499`) returns
    /// `co_consts_w[index]`, so a `LOAD_CONST` of a code constant yields the one
    /// shared `PyCode` stored here — repeated loads (and the blackhole
    /// resume reading the same `pycode` off the virtualizable) get identical
    /// `__code__` identity and a stable JIT green key, with no side table.
    ///
    /// Only the reference-typed code-constant slots are realized into this list
    /// (lazily, once per index); value constants realize through
    /// `load_const_value` as before.  The stored code wrappers are `Box`-immortal
    /// (`w_code_new` → `Box::into_raw`), so the slots never dangle and need no GC
    /// walking (the rationale that retired `CODE_CONSTANT_INTERN`).
    ///
    /// Owned via `Box::into_raw`, sized to `code.constants.len()` at construction,
    /// never resized; a `null` slot is unrealized.  The whole pointer is `null`
    /// when `code_ptr` is null or unaligned (test fixtures, gateway builtins).
    pub co_consts_w: *mut Vec<PyObjectRef>,
}

/// Field offset of `code_ptr` within `PyCode`.
pub const CODE_PTR_OFFSET: usize = std::mem::offset_of!(PyCode, code_ptr);
/// Field offset of `w_globals` within `PyCode`.
pub const CODE_W_GLOBALS_OFFSET: usize = std::mem::offset_of!(PyCode, w_globals);

/// GC type id assigned to `PyCode`.
///
/// `PyCode` is a normal interpreter-level code object in PyPy
/// (`pycode.py:93 class PyCode(eval.Code)`).  This tid is pinned by
/// a `debug_assert_eq!` in the pyre-jit type-registration sequence: the
/// `PyCode` `TypeInfo` is registered explicitly just before the
/// foreign-pytype loop, taking the slot directly after
/// `GC_FLOAT_ARRAY_GC_TYPE_ID = 42`.  Pre-registering it there (and
/// inserting `CODE_TYPE` into `pytype_to_tid`) makes the foreign loop
/// skip `CODE_TYPE`, so the net register-call count up to
/// `W_MODULE_DICT_GC_TYPE_ID = 48` is unchanged and no downstream tid
/// shifts.  The numeric value coincides with the dormant
/// `pytraceback::PYTRACEBACK_GC_TYPE_ID` constant, but `PyTraceback`
/// is still host-allocated and is never GC-registered, so tid 43 only
/// ever tags a `PyCode` at runtime and the two do not collide.
pub const W_CODE_GC_TYPE_ID: u32 = 43;

impl pyre_object::lltype::GcType for PyCode {
    fn type_id() -> u32 {
        W_CODE_GC_TYPE_ID
    }
    const SIZE: usize = std::mem::size_of::<PyCode>();
}

/// Compatibility helper for unpacking a tuple of strings.
pub fn unpack_text_tuple(_space: PyObjectRef, w_str_tuple: PyObjectRef) -> Vec<String> {
    let _ = (_space, w_str_tuple);
    Vec::new()
}

/// Compatibility API for building a signature-like object.
pub fn make_signature(_code: &PyCode) -> PyObjectRef {
    let _ = _code;
    pyre_object::w_none()
}

/// pycode.py:637-659 _compute_args_as_cellvars
pub fn _compute_args_as_cellvars(
    varnames: &[String],
    cellvars: &[String],
    argcount: usize,
) -> Vec<isize> {
    let mut args_as_cellvars = Vec::new();
    for i in 0..cellvars.len() {
        let cellname = &cellvars[i];
        for j in 0..argcount {
            if *cellname == varnames[j] {
                while args_as_cellvars.len() < i {
                    args_as_cellvars.push(-1isize);
                }
                args_as_cellvars.push(j as isize);
            }
        }
    }
    args_as_cellvars
}

#[inline]
pub fn _code_const_eq(_space: PyObjectRef, w_a: PyObjectRef, w_b: PyObjectRef) -> bool {
    let _ = _space;
    std::ptr::eq(w_a, w_b)
}

#[inline]
pub fn _convert_const(_space: PyObjectRef, w_a: PyObjectRef) -> PyObjectRef {
    let _ = _space;
    w_a
}

/// pypy/interpreter/pycode.py:107-147 `PyCode.__init__`
/// (`hidden_applevel` field assignment, line 147).
///
/// ```python
/// def __init__(self, space, ..., hidden_applevel=False, magic=default_magic):
///     ...
///     self.hidden_applevel = hidden_applevel
/// ```
///
/// `w_code_new(code_ptr)` is the `hidden_applevel=False` default
/// shorthand; callers who need the flag set (mirroring PyPy's
/// `BuiltinCode` (gateway.py:743) / `ApplevelClass`
/// (gateway.py:1355) / `_continuation` entrypoint dummy
/// (interp_continuation.py:195)) construct via this entry point.
///
/// # Safety
/// `code_ptr` must be a valid pointer to a `CodeObject` obtained
/// via `Box::into_raw`.
pub fn w_code_new_with_hidden_applevel(code_ptr: *const (), hidden_applevel: bool) -> PyObjectRef {
    // RPython pointer alignment idiom (`rpython/memory/gc/minimarkpage.py:159
    // ll_assert((nsize & (WORD-1)) == 0, "malloc: size is not aligned")`):
    // bitwise AND of `cast_ptr_to_int(p)` against `(power_of_two_align - 1)`
    // gives the misalignment residual.  `front::mir` lowers a `(Ref, Int)`
    // cast to the `cast_ptr_to_int` op, so casting through `i64` (not
    // `usize`) routes the pointer through the proper LL conversion.
    // `align_of::<T>()` is always a power of two — `& (align - 1)` is
    // equivalent to `% align` for power-of-two alignments and matches the
    // RPython pattern bit-for-bit.
    let align_mask = std::mem::align_of::<crate::CodeObject>() as i64 - 1;
    let fast_natural_arity = if code_ptr.is_null() || (code_ptr as i64) & align_mask != 0 {
        crate::gateway::HOPELESS
    } else {
        compute_flatcall(unsafe { &*(code_ptr as *const crate::CodeObject) })
    };
    // `pycode.py:198 self._globals_caches = [None] * len(self.co_names_w)`.
    let globals_caches = if code_ptr.is_null() || (code_ptr as i64) & align_mask != 0 {
        std::ptr::null_mut()
    } else {
        let code_ref = unsafe { &*(code_ptr as *const crate::CodeObject) };
        let names_len = code_ref.names.len();
        let mut v: Vec<
            Option<std::rc::Weak<std::cell::RefCell<pyre_object::celldict::GlobalCache>>>,
        > = Vec::with_capacity(names_len);
        v.resize_with(names_len, || None);
        Box::into_raw(Box::new(v))
    };
    // `mapdict.py:1457-1458 self._mapdict_caches = [INVALID_CACHE_ENTRY] *
    // len(co_names_w)` — `None` is `INVALID_CACHE_ENTRY`.
    let mapdict_caches = if code_ptr.is_null() || (code_ptr as i64) & align_mask != 0 {
        std::ptr::null_mut()
    } else {
        let code_ref = unsafe { &*(code_ptr as *const crate::CodeObject) };
        let names_len = code_ref.names.len();
        let mut v: Vec<Option<crate::objspace::std::mapdict::MapdictCacheEntry>> =
            Vec::with_capacity(names_len);
        v.resize_with(names_len, || None);
        Box::into_raw(Box::new(v))
    };
    // `pycode.py:126 self.co_consts_w = consts` — the realized-constant table
    // sized to the constant count, with code-constant slots filled lazily by
    // `w_code_co_const`.
    let co_consts_w = if code_ptr.is_null() || (code_ptr as i64) & align_mask != 0 {
        std::ptr::null_mut()
    } else {
        let code_ref = unsafe { &*(code_ptr as *const crate::CodeObject) };
        let consts_len = code_ref.constants.len();
        let mut v: Vec<PyObjectRef> = Vec::with_capacity(consts_len);
        v.resize(consts_len, std::ptr::null_mut());
        Box::into_raw(Box::new(v))
    };
    let obj = Box::new(PyCode {
        ob_header: PyObject {
            ob_type: &CODE_TYPE as *const PyType,
            w_class: pyre_object::pyobject::get_instantiate(&CODE_TYPE),
        },
        code_ptr,
        w_globals: pyre_object::PY_NULL,
        hidden_applevel,
        fast_natural_arity,
        globals_caches,
        mapdict_caches,
        co_consts_w,
    });
    Box::into_raw(obj) as PyObjectRef
}

/// pypy/interpreter/pycode.py:107-147 `PyCode.__init__` shorthand —
/// equivalent to PyPy `hidden_applevel=False` default
/// (pycode.py:111).  Most user-level pycode constructions take this
/// path; only the gateway / continuation / `__pypy__.hidden_applevel`
/// surfaces flip the flag to `True`.
///
/// # Safety
/// `code_ptr` must be a valid pointer to a `CodeObject` obtained
/// via `Box::into_raw`.
pub fn w_code_new(code_ptr: *const ()) -> PyObjectRef {
    w_code_new_with_hidden_applevel(code_ptr, false)
}

/// Box a cloned compiler code object into a heap Python code wrapper.
pub fn box_code_constant(code: &crate::CodeObject) -> PyObjectRef {
    let code_ptr = Box::into_raw(Box::new(code.clone())) as *const ();
    w_code_new(code_ptr)
}

/// The keyword-only fields `code.replace` accepts, in the order
/// `pypy/interpreter/pycode.py:77-81` reconstructs the code object.
const REPLACE_KWARGS: [&str; 18] = [
    "co_argcount",
    "co_posonlyargcount",
    "co_kwonlyargcount",
    "co_nlocals",
    "co_stacksize",
    "co_flags",
    "co_firstlineno",
    "co_code",
    "co_consts",
    "co_names",
    "co_varnames",
    "co_freevars",
    "co_cellvars",
    "co_filename",
    "co_name",
    "co_qualname",
    "co_linetable",
    "co_exceptiontable",
];

/// `code.replace(**kwds)` — `pypy/interpreter/pycode.py:74-91` applevel
/// `replace`, which gathers every `co_*` attribute (taking the keyword
/// override where present) and reconstructs the code object through the
/// `CodeType` constructor.  pyre stores a compiler `CodeObject`, so the
/// equivalent is to clone it, override each supplied field, and re-box it.
///
/// # Safety
/// `args[0]` must be the receiver `code` object (verified).
pub unsafe fn code_replace(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let w_self = pos.first().copied().unwrap_or(PY_NULL);
    if w_self.is_null() || !unsafe { is_code(w_self) } {
        return Err(crate::PyError::type_error(
            "descriptor 'replace' requires a 'code' object",
        ));
    }
    // `replace` is keyword-only (`__args__.topacked()` asserts no positional
    // args at pycode.py:548-549).
    if pos.len() > 1 {
        return Err(crate::PyError::type_error(
            "replace() takes no positional arguments",
        ));
    }
    // pycode.py:86-87 `raise TypeError(f"{kwds.popitem()[0]!r} is an invalid
    // keyword argument for replace()")`.
    if let Some(dict) = kwargs {
        for (key, _) in unsafe { pyre_object::w_dict_str_entries(dict) } {
            if key == "__pyre_kw__" {
                continue;
            }
            if !REPLACE_KWARGS.contains(&key.as_str()) {
                return Err(crate::PyError::type_error(format!(
                    "'{key}' is an invalid keyword argument for replace()"
                )));
            }
        }
    }

    let code_ptr = unsafe { w_code_get_ptr(w_self) } as *const crate::CodeObject;
    if code_ptr.is_null() {
        return Err(crate::PyError::type_error(
            "cannot replace fields of a code object with no code body",
        ));
    }
    let mut code = unsafe { (*code_ptr).clone() };
    let get = |name: &str| crate::builtins::kwarg_get(kwargs, name);

    if let Some(v) = get("co_argcount") {
        code.arg_count = unsafe { read_code_u32(v, "co_argcount")? };
    }
    if let Some(v) = get("co_posonlyargcount") {
        code.posonlyarg_count = unsafe { read_code_u32(v, "co_posonlyargcount")? };
    }
    if let Some(v) = get("co_kwonlyargcount") {
        code.kwonlyarg_count = unsafe { read_code_u32(v, "co_kwonlyargcount")? };
    }
    // co_nlocals is derived from the varnames length; it is accepted for
    // constructor-signature compatibility but carries no independent field.
    if let Some(v) = get("co_nlocals") {
        let _ = unsafe { read_code_u32(v, "co_nlocals")? };
    }
    if let Some(v) = get("co_stacksize") {
        code.max_stackdepth = unsafe { read_code_u32(v, "co_stacksize")? };
    }
    if let Some(v) = get("co_flags") {
        let bits = unsafe { crate::builtins::space_index_w(v)? } as u32;
        code.flags = crate::bytecode::CodeFlags::from_bits_retain(bits);
    }
    if let Some(v) = get("co_firstlineno") {
        let n = unsafe { crate::builtins::space_index_w(v)? };
        code.first_line_number = if n <= 0 {
            None
        } else {
            rustpython_compiler_core::OneIndexed::new(n as usize)
        };
    }
    if let Some(v) = get("co_name") {
        code.obj_name = unsafe { read_code_str(v, "co_name")? };
    }
    if let Some(v) = get("co_qualname") {
        code.qualname = unsafe { read_code_str(v, "co_qualname")? };
    }
    if let Some(v) = get("co_filename") {
        code.source_path = unsafe { read_code_str(v, "co_filename")? };
    }
    if let Some(v) = get("co_names") {
        code.names = unsafe { read_code_names(v, "co_names")? };
    }
    if let Some(v) = get("co_varnames") {
        code.varnames = unsafe { read_code_names(v, "co_varnames")? };
    }
    if let Some(v) = get("co_freevars") {
        code.freevars = unsafe { read_code_names(v, "co_freevars")? };
    }
    if let Some(v) = get("co_cellvars") {
        code.cellvars = unsafe { read_code_names(v, "co_cellvars")? };
    }
    if let Some(v) = get("co_linetable") {
        code.linetable = unsafe { read_code_bytes(v, "co_linetable")? };
    }
    if let Some(v) = get("co_exceptiontable") {
        code.exceptiontable = unsafe { read_code_bytes(v, "co_exceptiontable")? };
    }
    if let Some(v) = get("co_consts") {
        code.constants = unsafe { read_code_consts(v)? };
    }
    if let Some(v) = get("co_code") {
        code.instructions = unsafe { read_code_units(v)? };
    }

    Ok(box_code_constant(&code))
}

/// A non-negative `co_*` count argument as `u32`.
unsafe fn read_code_u32(v: PyObjectRef, field: &str) -> Result<u32, crate::PyError> {
    let n = unsafe { crate::builtins::space_index_w(v)? };
    if n < 0 {
        return Err(crate::PyError::value_error(format!(
            "{field} must be a non-negative integer"
        )));
    }
    Ok(n as u32)
}

/// A `str` `co_*` field as an owned `String` (the compiler `Name` type).
unsafe fn read_code_str(v: PyObjectRef, field: &str) -> Result<String, crate::PyError> {
    if !unsafe { pyre_object::is_str(v) } {
        return Err(crate::PyError::type_error(format!("{field} must be a str")));
    }
    Ok(unsafe { pyre_object::w_str_get_value(v) }.to_string())
}

/// A `tuple[str]` `co_*` field (names / varnames / freevars / cellvars).
unsafe fn read_code_names(v: PyObjectRef, field: &str) -> Result<Box<[String]>, crate::PyError> {
    if !unsafe { is_tuple(v) } {
        return Err(crate::PyError::type_error(format!(
            "{field} must be a tuple of strings"
        )));
    }
    let n = pyre_object::w_tuple_len(v);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let e = pyre_object::w_tuple_getitem(v, i as i64).unwrap_or_else(pyre_object::w_none);
        if !unsafe { pyre_object::is_str(e) } {
            return Err(crate::PyError::type_error(format!(
                "{field} must be a tuple of strings"
            )));
        }
        out.push(unsafe { pyre_object::w_str_get_value(e) }.to_string());
    }
    Ok(out.into_boxed_slice())
}

/// A `bytes` `co_*` field (linetable / exceptiontable) as raw bytes.
unsafe fn read_code_bytes(v: PyObjectRef, field: &str) -> Result<Box<[u8]>, crate::PyError> {
    if !unsafe { pyre_object::bytesobject::is_bytes_like(v) } {
        return Err(crate::PyError::type_error(format!(
            "{field} must be a bytes object"
        )));
    }
    Ok(unsafe { pyre_object::bytesobject::bytes_like_data(v) }
        .to_vec()
        .into_boxed_slice())
}

/// `co_code` bytes → the decoded `CodeUnits` instruction stream.  The byte
/// form is the `original_bytes` layout: one `(opcode, arg)` pair per unit.
unsafe fn read_code_units(v: PyObjectRef) -> Result<crate::bytecode::CodeUnits, crate::PyError> {
    if !unsafe { pyre_object::bytesobject::is_bytes_like(v) } {
        return Err(crate::PyError::type_error("co_code must be a bytes object"));
    }
    let bytes = unsafe { pyre_object::bytesobject::bytes_like_data(v) };
    if bytes.len() % 2 != 0 {
        return Err(crate::PyError::value_error(
            "co_code length must be a multiple of 2",
        ));
    }
    let mut units = Vec::with_capacity(bytes.len() / 2);
    for pair in bytes.chunks_exact(2) {
        let op = crate::bytecode::Instruction::try_from(pair[0]).map_err(|_| {
            crate::PyError::value_error(format!("co_code contains unknown opcode {}", pair[0]))
        })?;
        units.push(crate::bytecode::CodeUnit::new(
            op,
            crate::bytecode::OpArgByte::from(pair[1]),
        ));
    }
    Ok(crate::bytecode::CodeUnits::from(units))
}

/// A `tuple` `co_consts` field → the compiler `Constants` table.
unsafe fn read_code_consts(
    v: PyObjectRef,
) -> Result<crate::bytecode::Constants<crate::bytecode::ConstantData>, crate::PyError> {
    if !unsafe { is_tuple(v) } {
        return Err(crate::PyError::type_error("co_consts must be a tuple"));
    }
    let n = pyre_object::w_tuple_len(v);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let e = pyre_object::w_tuple_getitem(v, i as i64).unwrap_or_else(pyre_object::w_none);
        out.push(unsafe { obj_to_constant_data(e)? });
    }
    Ok(out.into_iter().collect())
}

/// Convert a Python object into the compiler `ConstantData` a code object
/// stores.  Covers None/Ellipsis/bool/int/float/str/bytes/tuple and nested
/// code; `complex` and `frozenset` constants (rare in a replaced
/// `co_consts`) and any object that is not a valid code constant raise
/// `ValueError` (pyre's constant table cannot hold an arbitrary object the
/// way a CPython `co_consts` tuple can).
unsafe fn obj_to_constant_data(
    obj: PyObjectRef,
) -> Result<crate::bytecode::ConstantData, crate::PyError> {
    use crate::bytecode::ConstantData;
    unsafe {
        if is_none(obj) {
            return Ok(ConstantData::None);
        }
        if is_ellipsis(obj) {
            return Ok(ConstantData::Ellipsis);
        }
        // bool is a subclass of int, so test it first.
        if is_bool(obj) {
            let value = crate::builtins::space_index_w(obj)? != 0;
            return Ok(ConstantData::Boolean { value });
        }
        if is_int_or_long(obj) {
            return Ok(ConstantData::Integer {
                value: crate::builtins::obj_to_bigint(obj),
            });
        }
        if is_float(obj) {
            return Ok(ConstantData::Float {
                value: pyre_object::w_float_get_value(obj),
            });
        }
        if pyre_object::is_str(obj) {
            return Ok(ConstantData::Str {
                value: pyre_object::w_str_get_wtf8(obj).to_owned(),
            });
        }
        if pyre_object::bytesobject::is_bytes_like(obj) {
            return Ok(ConstantData::Bytes {
                value: pyre_object::bytesobject::bytes_like_data(obj).to_vec(),
            });
        }
        if is_tuple(obj) {
            let n = pyre_object::w_tuple_len(obj);
            let mut elements = Vec::with_capacity(n);
            for i in 0..n {
                let e =
                    pyre_object::w_tuple_getitem(obj, i as i64).unwrap_or_else(pyre_object::w_none);
                elements.push(obj_to_constant_data(e)?);
            }
            return Ok(ConstantData::Tuple { elements });
        }
        if is_code(obj) {
            let ptr = w_code_get_ptr(obj) as *const crate::CodeObject;
            if !ptr.is_null() {
                return Ok(ConstantData::Code {
                    code: Box::new((*ptr).clone()),
                });
            }
        }
        Err(crate::PyError::value_error(
            "co_consts contains a value that is not a valid code constant",
        ))
    }
}

/// `pyopcode.py:498-499 getconstant_w(index) -> co_consts_w[index]` for a code
/// constant: return the one shared `PyCode` the enclosing code holds at
/// `index`, realizing it into the slot on first access (`pycode.py:126` builds
/// `co_consts_w` eagerly; pyre fills the reference-typed slots lazily).
///
/// `w_code_obj` is the enclosing `PyCode` (`frame.pycode` for the
/// interpreter, the virtualizable `pycode` field for the blackhole — the same
/// object for a given running code), and `idx` is the constant index.  Repeated
/// `LOAD_CONST` of the same code constant returns the same wrapper, giving stable
/// `__code__` identity and a stable JIT green key (`greens = [..., 'pycode']`);
/// without it a closure defined in a hot-loop-called function gets a fresh wrapper
/// each iteration and the trace give-up counter never accumulates.
///
/// The stored wrapper is `Box`-immortal, so the slot never dangles and needs no
/// GC walking.  An absent slot table (null `co_consts_w`) still realizes a fresh
/// wrapper from the nested code.
///
/// Returns `PY_NULL` (the empty pointer) when the constant cannot be resolved as
/// a code wrapper — a null/misaligned `code_ptr` (the nested code is
/// unreadable), `idx` out of range, or `constants[idx]` not a code constant — so
/// callers fall back to their value-constant realization path.
///
/// # Safety
/// `w_code_obj` must point to a valid `PyCode`.
pub unsafe fn w_code_co_const(w_code_obj: PyObjectRef, idx: usize) -> PyObjectRef {
    let w_code = unsafe { &*(w_code_obj as *const PyCode) };
    // Guard `code_ptr` before dereferencing it — the same null/alignment check
    // the lazy-cache initializers use. A null/misaligned pointer means the
    // nested code is unreadable, so return PY_NULL and let the caller realize
    // the constant from its own code object.
    let align_mask = std::mem::align_of::<crate::CodeObject>() as i64 - 1;
    if w_code.code_ptr.is_null() || (w_code.code_ptr as i64) & align_mask != 0 {
        return pyre_object::pyobject::PY_NULL;
    }
    let code = unsafe { &*(w_code.code_ptr as *const crate::CodeObject) };
    let constants = crate::pyframe::code_constants(code);
    let Some(crate::bytecode::ConstantData::Code { code: nested }) = constants.get(idx) else {
        return pyre_object::pyobject::PY_NULL;
    };
    if w_code.co_consts_w.is_null() {
        return box_code_constant(nested);
    }
    let slot_table = unsafe { &mut *w_code.co_consts_w };
    match slot_table.get_mut(idx) {
        Some(slot) if !slot.is_null() => *slot,
        Some(slot) => {
            let realized = box_code_constant(nested);
            *slot = realized;
            realized
        }
        None => box_code_constant(nested),
    }
}

/// pypy/module/__pypy__/interp_magic.py:79
/// `func.getcode().hidden_applevel = True` — explicit setter for the
/// `__pypy__.hidden_applevel(func)` builtin marker, plus the
/// `_continuation.entrypoint_pycode.hidden_applevel = True`
/// hand-edit (interp_continuation.py:195).  PyPy mutates the field
/// directly; pyre wraps the raw write because the field is private
/// to this module.
///
/// # Safety
/// `obj` must point to a valid `PyCode`.
#[inline]
pub unsafe fn w_code_set_hidden_applevel(obj: PyObjectRef, hidden_applevel: bool) {
    if obj.is_null() {
        return;
    }
    unsafe {
        (*(obj as *mut PyCode)).hidden_applevel = hidden_applevel;
    }
}

/// Extract the opaque code pointer from a known PyCode.
///
/// # Safety
/// `obj` must point to a valid `PyCode`.
#[inline]
pub unsafe fn w_code_get_ptr(obj: PyObjectRef) -> *const () {
    unsafe { (*(obj as *const PyCode)).code_ptr }
}

/// PyPy: `PyCode.hidden_applevel` (`pycode.py:147`). Reads the field
/// initialised by `w_code_new`.  `pyframe.py:521-522
/// hide(self): return self.pycode.hidden_applevel` is the sole caller
/// in the canonical interpreter; pyre routes through this accessor
/// from `pyframe.rs::PyFrame::hide`.
///
/// # Safety
/// `obj` must point to a valid `PyCode`.
#[inline]
pub unsafe fn w_code_hidden_applevel(obj: PyObjectRef) -> bool {
    if obj.is_null() {
        return false;
    }
    unsafe { (*(obj as *const PyCode)).hidden_applevel }
}

/// PyPy: `PyCode.w_globals` — the globals dict OBJECT. The JIT
/// codewriter/bridge read this to fold globals lookups without an off-GC
/// proxy.
#[inline]
pub unsafe fn w_code_get_w_globals(obj: PyObjectRef) -> PyObjectRef {
    if obj.is_null() {
        return pyre_object::PY_NULL;
    }
    unsafe { (*(obj as *const PyCode)).w_globals }
}

/// PyPy: `PyCode.w_globals = w_globals`.
#[inline]
pub unsafe fn w_code_set_w_globals(obj: PyObjectRef, w_globals: PyObjectRef) {
    if obj.is_null() {
        return;
    }
    unsafe {
        (*(obj as *mut PyCode)).w_globals = w_globals;
    }
    // Box-immortal code slot reached only by `walk_raw_code_roots`,
    // skipped on clean minor collections; record the store.
    pyre_object::gc_roots::mark_prebuilt_roots_dirty();
    if !w_globals.is_null() {
        let code_ptr = unsafe { (*(obj as *const PyCode)).code_ptr };
        register_live_code_wrapper(code_ptr, obj);
    }
}

/// PyPy: `PyCode.frame_stores_global(w_globals)`.
#[inline]
pub unsafe fn w_code_frame_stores_global(obj: PyObjectRef, w_globals: PyObjectRef) -> bool {
    if obj.is_null() {
        return false;
    }
    let code = unsafe { &mut *(obj as *mut PyCode) };
    if code.w_globals.is_null() {
        code.w_globals = w_globals;
        // Prebuilt-family store (see `w_code_set_w_globals`).
        pyre_object::gc_roots::mark_prebuilt_roots_dirty();
        register_live_code_wrapper(code.code_ptr, obj);
        return false;
    }
    !std::ptr::eq(code.w_globals, w_globals)
}

thread_local! {
    /// Registry mapping a raw CodeObject pointer (`PyCode.code_ptr`) to the
    /// live, globals-stamped `PyCode` wrapper. Populated where a frame stamps
    /// the wrapper's `w_globals` — the only point both the raw pointer and the
    /// live wrapper are in hand — and consumed by the JIT to recover the live
    /// wrapper (and hence its `w_globals`) from a raw code pointer it already
    /// holds, so the JIT need not carry the wrapper identity as a separate
    /// `w_code` courier. First-write-wins, mirroring the first-store-wins
    /// `PyCode.w_globals` semantics in `w_code_frame_stores_global`. Wrappers
    /// are `Box::into_raw`-immortal and non-moving, so stored pointers never
    /// dangle and need no GC rooting.
    static LIVE_CODE_WRAPPERS: std::cell::RefCell<std::collections::HashMap<*const (), PyObjectRef>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Record `wrapper` as the live wrapper for `code_ptr`, keeping the first one
/// stamped (later stores are ignored). No-op on null inputs.
pub fn register_live_code_wrapper(code_ptr: *const (), wrapper: PyObjectRef) {
    if code_ptr.is_null() || wrapper.is_null() {
        return;
    }
    LIVE_CODE_WRAPPERS.with(|m| {
        m.borrow_mut().entry(code_ptr).or_insert(wrapper);
    });
}

/// Recover the live wrapper previously registered for `code_ptr`, or `PY_NULL`
/// if none has been stamped.
pub fn live_code_wrapper(code_ptr: *const ()) -> PyObjectRef {
    if code_ptr.is_null() {
        return PY_NULL;
    }
    LIVE_CODE_WRAPPERS.with(|m| m.borrow().get(&code_ptr).copied().unwrap_or(PY_NULL))
}

/// pycode.py:226-238 `_compute_flatcall`.
///
/// Returns FLATPYCALL | co_argcount for simple user functions (no *args,
/// **kwargs, keyword-only args). Returns HOPELESS otherwise.
fn compute_flatcall(code: &crate::CodeObject) -> u16 {
    use crate::CodeFlags;
    use crate::gateway::{FLATPYCALL, HOPELESS};
    if code
        .flags
        .intersects(CodeFlags::VARARGS | CodeFlags::VARKEYWORDS)
    {
        return HOPELESS;
    }
    if code.kwonlyarg_count > 0 {
        return HOPELESS;
    }
    if code.arg_count > 0xff {
        return HOPELESS;
    }
    // pycode.py:234 — disqualify if any arg is also a cellvar.
    // Pyre's CodeObject exposes cellvars; check for overlap.
    let argcount = code.arg_count as usize;
    if !code.cellvars.is_empty() && argcount > 0 {
        for cellname in &code.cellvars {
            for j in 0..argcount {
                if j < code.varnames.len() && *cellname == code.varnames[j] {
                    return HOPELESS;
                }
            }
        }
    }
    FLATPYCALL | (code.arg_count as u16)
}

/// eval.py:16-23 — read `fast_natural_arity` from a PyCode.
///
/// # Safety
/// `obj` must point to a valid `PyCode`.
#[inline]
pub unsafe fn w_code_get_fast_natural_arity(obj: PyObjectRef) -> u16 {
    if obj.is_null() {
        return crate::gateway::HOPELESS;
    }
    unsafe { (*(obj as *const PyCode)).fast_natural_arity }
}

/// Unified accessor: read `fast_natural_arity` from any code object
/// (BuiltinCode or PyCode).
///
/// # Safety
/// `obj` must point to a valid code object (either type).
#[inline]
pub unsafe fn code_get_fast_natural_arity(obj: PyObjectRef) -> u16 {
    if obj.is_null() {
        return crate::gateway::HOPELESS;
    }
    unsafe {
        if crate::gateway::is_builtin_code(obj) {
            crate::gateway::builtin_code_get_fast_natural_arity(obj)
        } else {
            w_code_get_fast_natural_arity(obj)
        }
    }
}

/// pycode.py:229-254 `PyCode.lookup_exceptiontable`.
///
/// Search the wrapped code object's exception table for a handler
/// covering `instr_offset` (byte offset into `co_code`).  Returns
/// `Some((target, depth, lasti))` with byte-offset `target` when found.
///
/// # Safety
/// `obj` must point to a valid `PyCode`.
#[inline]
pub unsafe fn w_code_lookup_exceptiontable(
    obj: PyObjectRef,
    instr_offset: u32,
) -> Option<(u32, u32, bool)> {
    if obj.is_null() {
        return None;
    }
    let code_ptr = unsafe { (*(obj as *const PyCode)).code_ptr };
    if code_ptr.is_null() {
        return None;
    }
    let code = unsafe { &*(code_ptr as *const crate::CodeObject) };
    crate::pycode::lookup_exceptiontable(&code.exceptiontable, instr_offset)
}

/// pycode.py:145 `self.co_exceptiontable = exceptiontable` — copy the
/// varint-packed table bytes out of the wrapped `CodeObject`.
///
/// The bytes are owned by the inner `CodeObject` (`Box<[u8]>` field), so
/// returning a reference would tie the lifetime to the obj's heap
/// allocation.  Callers that need to hand the bytes to Python (where
/// they get copied into a `W_BytesObject`) take the owned `Vec<u8>`.
///
/// # Safety
/// `obj` must point to a valid `PyCode`.
#[inline]
pub unsafe fn w_code_exceptiontable(obj: PyObjectRef) -> Vec<u8> {
    if obj.is_null() {
        return Vec::new();
    }
    let code_ptr = unsafe { (*(obj as *const PyCode)).code_ptr };
    if code_ptr.is_null() {
        return Vec::new();
    }
    let code = unsafe { &*(code_ptr as *const crate::CodeObject) };
    code.exceptiontable.to_vec()
}

/// `celldict.py:292 cache_wref = pycode._globals_caches[nameindex]` —
/// read slot `nameindex` and upgrade the weakref to a strong
/// `Rc<RefCell<GlobalCache>>` (returning `None` when the slot is
/// unset, the weak target is gone, or `code_ptr` is invalid).
///
/// # Safety
/// `obj` must point to a valid `PyCode` (or be null).
#[inline]
pub unsafe fn w_code_globals_caches_get(
    obj: PyObjectRef,
    nameindex: usize,
) -> Option<std::rc::Rc<std::cell::RefCell<pyre_object::celldict::GlobalCache>>> {
    if obj.is_null() {
        return None;
    }
    let code = unsafe { &*(obj as *const PyCode) };
    if code.globals_caches.is_null() {
        return None;
    }
    let vec = unsafe { &*code.globals_caches };
    vec.get(nameindex)
        .and_then(|slot| slot.as_ref())
        .and_then(|w| w.upgrade())
}

/// `celldict.py:321/353 pycode._globals_caches[nameindex] = cache.ref`
/// — store `Rc::downgrade(cache)` in slot `nameindex`.  No-op when
/// `code_ptr` is invalid or `nameindex` is out of range.
///
/// # Safety
/// `obj` must point to a valid `PyCode` (or be null).
#[inline]
pub unsafe fn w_code_globals_caches_set(
    obj: PyObjectRef,
    nameindex: usize,
    cache: &std::rc::Rc<std::cell::RefCell<pyre_object::celldict::GlobalCache>>,
) {
    if obj.is_null() {
        return;
    }
    let code = unsafe { &*(obj as *const PyCode) };
    if code.globals_caches.is_null() {
        return;
    }
    let vec = unsafe { &mut *code.globals_caches };
    if let Some(slot) = vec.get_mut(nameindex) {
        *slot = Some(std::rc::Rc::downgrade(cache));
    }
}

/// Number of `_globals_caches` slots — equals `len(co_names_w)` at
/// construction time.  Returns 0 for code objects built from null
/// or unaligned `code_ptr`.
///
/// # Safety
/// `obj` must point to a valid `PyCode` (or be null).
#[inline]
pub unsafe fn w_code_globals_caches_len(obj: PyObjectRef) -> usize {
    if obj.is_null() {
        return 0;
    }
    let code = unsafe { &*(obj as *const PyCode) };
    if code.globals_caches.is_null() {
        return 0;
    }
    unsafe { (*code.globals_caches).len() }
}

/// `mapdict.py:1483/1546/1575 entry = pycode._mapdict_caches[nameindex]` — read
/// slot `nameindex`, returning `None` (PyPy `INVALID_CACHE_ENTRY`) when the slot
/// is unset, out of range, or `code_ptr` is invalid.  The entry is `Copy`, so a
/// value is returned (no aliasing of the slot).
///
/// # Safety
/// `obj` must point to a valid `PyCode` (or be null).
#[inline]
pub unsafe fn w_code_mapdict_caches_get(
    obj: PyObjectRef,
    nameindex: usize,
) -> Option<crate::objspace::std::mapdict::MapdictCacheEntry> {
    if obj.is_null() {
        return None;
    }
    let code = unsafe { &*(obj as *const PyCode) };
    if code.mapdict_caches.is_null() {
        return None;
    }
    let vec = unsafe { &*code.mapdict_caches };
    vec.get(nameindex).copied().flatten()
}

/// `mapdict.py:1467-1475 pycode._mapdict_caches[nameindex] = entry` — store the
/// filled entry in slot `nameindex`.  No-op when `code_ptr` is invalid or
/// `nameindex` is out of range.
///
/// # Safety
/// `obj` must point to a valid `PyCode` (or be null).
#[inline]
pub unsafe fn w_code_mapdict_caches_set(
    obj: PyObjectRef,
    nameindex: usize,
    entry: crate::objspace::std::mapdict::MapdictCacheEntry,
) {
    if obj.is_null() {
        return;
    }
    let code = unsafe { &*(obj as *const PyCode) };
    if code.mapdict_caches.is_null() {
        return;
    }
    let vec = unsafe { &mut *code.mapdict_caches };
    if let Some(slot) = vec.get_mut(nameindex) {
        *slot = Some(entry);
        // The LOAD_METHOD fill (mapdict.py:1474) stores a movable
        // `w_method` reference; register this code object so
        // `walk_mapdict_method_cache_gc` forwards the slot.
        if !entry.w_method.is_null() {
            MAPDICT_METHOD_CACHE_CODES.with(|s| {
                s.borrow_mut().insert(obj as usize);
            });
            // Prebuilt-family store: the slot is reached only by
            // `walk_mapdict_method_cache_gc`, skipped on clean minors.
            pyre_object::gc_roots::mark_prebuilt_roots_dirty();
        }
    }
}

thread_local! {
    /// Code objects whose `_mapdict_caches` hold (or once held) a filled
    /// `w_method` slot.  In PyPy `CacheEntry.w_method` (mapdict.py:1418)
    /// is traced through the GC-managed `PyCode`; pyre code objects are
    /// Box-immortal (`w_code_new` → `Box::into_raw`), so no trace
    /// reaches the slot — the extra-root walker forwards it through
    /// this registry instead (same family as `walk_method_cache_gc`).
    /// Entries are immortal code pointers, so they never dangle; the
    /// registry retires when code objects become GC-managed.
    static MAPDICT_METHOD_CACHE_CODES: std::cell::RefCell<std::collections::HashSet<usize>> =
        std::cell::RefCell::new(std::collections::HashSet::new());
}

/// Forward every filled `entry.w_method` slot during collection — the
/// faithful equivalent of the GC tracing PyPy's `CacheEntry.w_method`
/// (mapdict.py:1418) gets through its GC-managed holder.  The cached
/// map/attr node pointers are immortal interned nodes and the
/// `version_tag` is a `u64`, so `w_method` is the entry's only movable
/// reference.
pub(crate) unsafe fn walk_mapdict_method_cache_gc(forward: &mut dyn FnMut(&mut PyObjectRef)) {
    MAPDICT_METHOD_CACHE_CODES.with(|s| {
        for &code in s.borrow().iter() {
            let code = unsafe { &*(code as *const PyCode) };
            if code.mapdict_caches.is_null() {
                continue;
            }
            let vec = unsafe { &mut *code.mapdict_caches };
            for slot in vec.iter_mut() {
                if let Some(entry) = slot.as_mut() {
                    if !entry.w_method.is_null() {
                        forward(&mut entry.w_method);
                    }
                }
            }
        }
    });
}

/// Number of `_mapdict_caches` slots — equals `len(co_names_w)` at construction
/// time.  Returns 0 for code objects built from null or unaligned `code_ptr`.
///
/// # Safety
/// `obj` must point to a valid `PyCode` (or be null).
#[inline]
pub unsafe fn w_code_mapdict_caches_len(obj: PyObjectRef) -> usize {
    if obj.is_null() {
        return 0;
    }
    let code = unsafe { &*(obj as *const PyCode) };
    if code.mapdict_caches.is_null() {
        return 0;
    }
    unsafe { (*code.mapdict_caches).len() }
}

/// Check if an object is a code object.
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn is_code(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &CODE_TYPE) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal varint-encoded exception table from `(start, length,
    /// target, depth, lasti)` tuples, mirroring the encoding produced by
    /// `assemble.py::_encode_varint`. Values are passed as word offsets
    /// (the on-disk unit), not byte offsets.
    fn encode_table(entries: &[(u32, u32, u32, u32, bool)]) -> Vec<u8> {
        let mut out = Vec::new();
        for (start, length, target, depth, lasti) in entries.iter().copied() {
            push_varint(&mut out, start, true);
            push_varint(&mut out, length, false);
            push_varint(&mut out, target, false);
            push_varint(&mut out, (depth << 1) | (lasti as u32), false);
        }
        out
    }

    fn push_varint(out: &mut Vec<u8>, mut value: u32, entry_start: bool) {
        let mut chunks = [0u8; 6];
        let mut n = 0;
        loop {
            chunks[n] = (value & 63) as u8;
            n += 1;
            value >>= 6;
            if value == 0 {
                break;
            }
        }
        for j in (0..n).rev() {
            let mut byte = chunks[j];
            if j != 0 {
                byte |= 0x40;
            }
            if j == n - 1 && entry_start {
                byte |= 0x80;
            }
            out.push(byte);
        }
    }

    #[test]
    fn empty_table_returns_none() {
        assert_eq!(lookup_exceptiontable(&[], 0), None);
    }

    #[test]
    fn lookup_returns_byte_offsets() {
        // entry: word offsets start=4 (byte 8), length=10 (byte 20),
        // target=20 (byte 40), depth=2, lasti=false
        let table = encode_table(&[(4, 10, 20, 2, false)]);
        assert_eq!(lookup_exceptiontable(&table, 8), Some((40, 2, false)));
        assert_eq!(lookup_exceptiontable(&table, 27), Some((40, 2, false)));
        assert_eq!(lookup_exceptiontable(&table, 28), None);
        assert_eq!(lookup_exceptiontable(&table, 7), None);
    }

    #[test]
    fn last_matching_wins() {
        // Two overlapping ranges; outer first, inner second
        // (CPython emission order).
        let table = encode_table(&[(0, 10, 20, 1, false), (3, 4, 30, 3, true)]);
        assert_eq!(lookup_exceptiontable(&table, 2), Some((40, 1, false)));
        // PC 8 (byte) is covered by both. PyPy returns the later entry.
        assert_eq!(lookup_exceptiontable(&table, 8), Some((60, 3, true)));
        assert_eq!(lookup_exceptiontable(&table, 14), Some((40, 1, false)));
    }

    #[test]
    fn lasti_low_bit() {
        let table = encode_table(&[(0, 2, 10, 5, true)]);
        let (target, depth, lasti) = lookup_exceptiontable(&table, 0).unwrap();
        assert_eq!((target, depth, lasti), (20, 5, true));
    }

    #[test]
    fn iter_matches_lookup_count() {
        let table = encode_table(&[
            (0, 4, 8, 1, false),
            (10, 6, 20, 2, true),
            (30, 2, 40, 0, false),
        ]);
        let entries: Vec<_> = decode_exceptiontable(&table).collect();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[1].start, 20);
        assert_eq!(entries[1].end, 32);
        assert_eq!(entries[1].target, 40);
        assert_eq!(entries[1].depth, 2);
        assert!(entries[1].lasti);
    }

    #[test]
    fn early_break_when_start_past_offset() {
        let table = encode_table(&[(0, 2, 10, 1, false), (100, 2, 200, 2, false)]);
        assert_eq!(lookup_exceptiontable(&table, 50), None);
    }

    #[test]
    fn live_code_wrapper_round_trips_first_write() {
        let code = 0x1000usize as *const ();
        let w1 = 0x2000usize as PyObjectRef;
        let w2 = 0x3000usize as PyObjectRef;
        register_live_code_wrapper(code, w1);
        // First-write-wins: a later store for the same code is ignored.
        register_live_code_wrapper(code, w2);
        assert_eq!(live_code_wrapper(code), w1);
        // An unregistered code pointer recovers to PY_NULL.
        assert!(live_code_wrapper(0x9999usize as *const ()).is_null());
        // Null inputs are no-ops / recover to PY_NULL.
        register_live_code_wrapper(std::ptr::null(), w1);
        assert!(live_code_wrapper(std::ptr::null()).is_null());
    }

    #[test]
    fn w_code_co_const_null_code_ptr_returns_py_null() {
        // A `PyCode` built from a null `code_ptr` must not be
        // dereferenced; the guard returns PY_NULL so the caller falls back to
        // its own constant realization.
        let w_code = w_code_new(std::ptr::null());
        let result = unsafe { w_code_co_const(w_code, 0) };
        assert_eq!(result, pyre_object::pyobject::PY_NULL);
    }
}
