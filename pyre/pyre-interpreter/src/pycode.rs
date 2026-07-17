//! PyCode — Python `code` object wrapper.
//!
//! Wraps an opaque pointer to the compiler's CodeObject, allowing it to
//! be placed on the value stack as a PyObjectRef during `LoadConst`.
//! MakeFunction then extracts this pointer to build a function object.

use pyre_object::pyobject::*;
use pyre_object::{
    w_bool_from, w_bool_get_value, w_int_new, w_list_new, w_seq_iter_new, w_str_new, w_tuple_new,
};

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
    /// `pycode.py:143 self.co_firstlineno = firstlineno`. RustPython's
    /// `CodeObject.first_line_number: Option<OneIndexed>` cannot represent
    /// the zero/negative values accepted by Python 3.14's CodeType
    /// constructor, so preserve the exact Python integer on the PyCode itself.
    pub co_firstlineno_raw: i32,
    /// PyPy: `PyCode.w_globals` — the globals dict OBJECT (`W_DictMultiObject`,
    /// `pycode.py:105 "w_globals?"`).  Module globals are `malloc_typed`-
    /// immortal, but `exec`/custom-globals dicts are `try_gc_alloc` movable.
    /// The code object is Box-immortal, so the collector never reaches this
    /// slot by tracing into it; `eval::walk_raw_code_roots` forwards it as a
    /// root (via `walk_raw_function_roots` for `func.code` and the frame walk
    /// for `frame.pycode`). Null until first stamped by `frame_stores_global`.
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
    /// Cached [`crate::pyframe::npure_cellvars`] — the count of cellvars that
    /// are not also varnames.  Code-invariant, so computed once here instead
    /// of re-walking the O(cellvars × varnames) overlap check on every
    /// `PyFrame::ncells()` / stack-base query (a per-`pop_value` hot path).
    /// `u32::MAX` sentinel when `code_ptr` is null/unaligned (test stubs).
    pub npure_cellvars: u32,
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
    // RPython pattern bit-for-bit.  The residual is computed once and shared
    // by every field initializer below.
    let align_mask = std::mem::align_of::<crate::CodeObject>() as i64 - 1;
    let code_ptr_aligned = !code_ptr.is_null() && (code_ptr as i64) & align_mask == 0;
    let fast_natural_arity = if !code_ptr_aligned {
        crate::gateway::HOPELESS
    } else {
        compute_flatcall(unsafe { &*(code_ptr as *const crate::CodeObject) })
    };
    // `pycode.py:198 self._globals_caches = [None] * len(self.co_names_w)`.
    let globals_caches = if !code_ptr_aligned {
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
    let mapdict_caches = if !code_ptr_aligned {
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
    let co_consts_w = if !code_ptr_aligned {
        std::ptr::null_mut()
    } else {
        let code_ref = unsafe { &*(code_ptr as *const crate::CodeObject) };
        let consts_len = code_ref.constants.len();
        let mut v: Vec<PyObjectRef> = Vec::with_capacity(consts_len);
        v.resize(consts_len, std::ptr::null_mut());
        Box::into_raw(Box::new(v))
    };
    let npure_cellvars = if !code_ptr_aligned {
        u32::MAX
    } else {
        let code_ref = unsafe { &*(code_ptr as *const crate::CodeObject) };
        crate::pyframe::npure_cellvars(code_ref) as u32
    };
    let co_firstlineno_raw = if code_ptr.is_null() || (code_ptr as i64) & align_mask != 0 {
        1
    } else {
        unsafe { &*(code_ptr as *const crate::CodeObject) }
            .first_line_number
            .map_or(1, |line| line.get() as i32)
    };
    let obj = Box::new(PyCode {
        ob_header: PyObject {
            ob_type: &CODE_TYPE as *const PyType,
            w_class: pyre_object::pyobject::get_instantiate(&CODE_TYPE),
        },
        code_ptr,
        co_firstlineno_raw,
        w_globals: pyre_object::PY_NULL,
        hidden_applevel,
        fast_natural_arity,
        npure_cellvars,
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

fn box_code_constant_with_firstlineno(code: &crate::CodeObject, firstlineno: i32) -> PyObjectRef {
    let obj = box_code_constant(code);
    unsafe {
        (*(obj as *mut PyCode)).co_firstlineno_raw = firstlineno;
    }
    obj
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

#[inline]
unsafe fn require_code(
    obj: PyObjectRef,
    descriptor: &str,
) -> Result<&'static crate::CodeObject, crate::PyError> {
    if obj.is_null() || !unsafe { is_code(obj) } {
        return Err(crate::PyError::type_error(format!(
            "descriptor '{descriptor}' requires a 'code' object"
        )));
    }
    let ptr = unsafe { w_code_get_ptr(obj) } as *const crate::CodeObject;
    if ptr.is_null() || !(ptr as usize).is_multiple_of(std::mem::align_of::<crate::CodeObject>()) {
        return Err(crate::PyError::type_error("code object has no code body"));
    }
    Ok(unsafe { &*ptr })
}

fn names_tuple(names: &[String]) -> PyObjectRef {
    w_tuple_new(names.iter().map(|name| w_str_new(name)).collect())
}

fn constants_tuple(code: &crate::CodeObject) -> PyObjectRef {
    w_tuple_new(
        crate::pyframe::code_constants(code)
            .iter()
            .map(crate::pyframe::pyobject_from_constant)
            .collect(),
    )
}

fn legacy_lnotab(code: &crate::CodeObject, firstlineno: i64) -> Vec<u8> {
    fn encode_pair(mut address: usize, mut line: i64, out: &mut Vec<u8>) {
        while address > 255 {
            out.extend_from_slice(&[255, 0]);
            address -= 255;
        }
        while line < -128 {
            out.extend_from_slice(&[address as u8, 128]);
            line += 128;
            address = 0;
        }
        while line > 127 {
            out.extend_from_slice(&[address as u8, 127]);
            line -= 127;
            address = 0;
        }
        out.extend_from_slice(&[address as u8, line as i8 as u8]);
    }

    let mut out = Vec::new();
    let mut line = firstlineno;
    let mut start_offset = 0usize;
    for (index, (start, _)) in code.locations.iter().enumerate() {
        let next_line = start.line.get() as i64;
        if next_line != line {
            let offset = index * 2;
            encode_pair(offset - start_offset, next_line - line, &mut out);
            line = next_line;
            start_offset = offset;
        }
    }
    out
}

/// `PyCode.typedef` field getters. Each type-dict descriptor delegates here so
/// the object carries one authoritative compiler `CodeObject`, matching
/// `pycode.py`'s direct `co_*` attributes rather than a parallel side table.
pub unsafe fn code_get_field(obj: PyObjectRef, name: &str) -> Result<PyObjectRef, crate::PyError> {
    let code = unsafe { require_code(obj, name)? };
    Ok(match name {
        "co_argcount" => w_int_new(code.arg_count as i64),
        "co_posonlyargcount" => w_int_new(code.posonlyarg_count as i64),
        "co_kwonlyargcount" => w_int_new(code.kwonlyarg_count as i64),
        "co_nlocals" => w_int_new(code.varnames.len() as i64),
        "co_stacksize" => w_int_new(code.max_stackdepth as i64),
        "co_flags" => w_int_new(code.flags.bits() as i64),
        "co_code" | "_co_code_adaptive" => {
            pyre_object::bytesobject::w_bytes_from_bytes(&code.instructions.original_bytes())
        }
        "co_consts" => constants_tuple(code),
        "co_names" => names_tuple(&code.names),
        "co_varnames" => names_tuple(&code.varnames),
        "co_freevars" => names_tuple(&code.freevars),
        "co_cellvars" => names_tuple(&code.cellvars),
        "co_filename" => w_str_new(&code.source_path),
        "co_name" => w_str_new(&code.obj_name),
        "co_qualname" => w_str_new(&code.qualname),
        "co_firstlineno" => w_int_new((*(obj as *const PyCode)).co_firstlineno_raw as i64),
        "co_linetable" => pyre_object::bytesobject::w_bytes_from_bytes(&code.linetable),
        "co_exceptiontable" => pyre_object::bytesobject::w_bytes_from_bytes(&code.exceptiontable),
        // location.py:163-182 `linetable2lnotab`, reconstructed from the
        // canonical decoded positions kept on CodeObject.
        "co_lnotab" => pyre_object::bytesobject::w_bytes_from_bytes(&legacy_lnotab(
            code,
            (*(obj as *const PyCode)).co_firstlineno_raw as i64,
        )),
        _ => {
            return Err(crate::PyError::attribute_error(format!(
                "'code' object has no attribute '{name}'"
            )));
        }
    })
}

/// CPython 3.14 `code.__new__` positional-only constructor, with the PyPy
/// `descr_code__new__` validations and field order adjusted to 3.14 (the
/// exception table precedes freevars/cellvars).
pub unsafe fn code_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if !(17..=19).contains(&args.len()) {
        return Err(crate::PyError::type_error(format!(
            "code expected at least 16 arguments, got {}",
            args.len().saturating_sub(1),
        )));
    }
    let argcount = unsafe { read_code_u32(args[1], "argcount")? };
    let posonly = unsafe { read_code_u32(args[2], "posonlyargcount")? };
    let kwonly = unsafe { read_code_u32(args[3], "kwonlyargcount")? };
    let nlocals = unsafe { read_code_u32(args[4], "nlocals")? };
    let stacksize_value = unsafe { read_code_c_int(args[5])? };
    let flags_value = unsafe { read_code_c_int(args[6])? };
    if stacksize_value < 0 || flags_value < 0 {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "Objects/codeobject.c: bad argument to internal function",
        ));
    }
    let stacksize = stacksize_value as u32;
    let flags_bits = flags_value as u32;
    let instructions = unsafe { read_code_units(args[7])? };
    let constants = unsafe { read_code_consts(args[8])? };
    let names = unsafe { read_code_names(args[9], "names")? };
    let varnames = unsafe { read_code_names(args[10], "varnames")? };
    if varnames.len() != nlocals as usize {
        return Err(crate::PyError::value_error(format!(
            "code: co_nlocals != len(co_varnames)"
        )));
    }
    let source_path = unsafe { read_code_str(args[11], "filename")? };
    let obj_name = unsafe { read_code_str(args[12], "name")? };
    let qualname = unsafe { read_code_str(args[13], "qualname")? };
    let first_line = unsafe { read_code_c_int(args[14])? } as i64;
    let first_line_number = if first_line <= 0 {
        None
    } else {
        rustpython_compiler_core::OneIndexed::new(first_line as usize)
    };
    let linetable = unsafe { read_code_bytes(args[15], "linetable")? };
    let exceptiontable = unsafe { read_code_bytes(args[16], "exceptiontable")? };
    let freevars = if args.len() >= 18 {
        unsafe { read_code_names(args[17], "freevars")? }
    } else {
        Vec::<String>::new().into_boxed_slice()
    };
    let cellvars = if args.len() >= 19 {
        unsafe { read_code_names(args[18], "cellvars")? }
    } else {
        Vec::<String>::new().into_boxed_slice()
    };
    if argcount + kwonly > nlocals || posonly > argcount {
        return Err(crate::PyError::value_error("code: invalid argument count"));
    }

    // CPython's localsplus table stores cell aliases on the local slot and
    // appends only pure cells, followed by free variables.
    let mut localspluskinds = vec![crate::bytecode::CO_FAST_LOCAL; varnames.len()];
    for cell in cellvars.iter() {
        if let Some(index) = varnames.iter().position(|name| name == cell) {
            localspluskinds[index] |= crate::bytecode::CO_FAST_CELL;
        } else {
            localspluskinds.push(crate::bytecode::CO_FAST_CELL);
        }
    }
    localspluskinds.extend(std::iter::repeat_n(
        crate::bytecode::CO_FAST_FREE,
        freevars.len(),
    ));

    let locations = rustpython_compiler_core::marshal::linetable_to_locations(
        &linetable,
        first_line.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
        instructions.len(),
    );
    let code = crate::CodeObject {
        instructions,
        locations,
        flags: crate::bytecode::CodeFlags::from_bits_retain(flags_bits),
        posonlyarg_count: posonly,
        arg_count: argcount,
        kwonlyarg_count: kwonly,
        source_path,
        first_line_number,
        max_stackdepth: stacksize,
        obj_name,
        qualname,
        constants,
        names,
        varnames,
        cellvars,
        freevars,
        localspluskinds: localspluskinds.into_boxed_slice(),
        linetable,
        exceptiontable,
    };
    Ok(box_code_constant_with_firstlineno(
        &code,
        first_line.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
    ))
}

fn code_data_equal(a: &crate::CodeObject, b: &crate::CodeObject) -> bool {
    a.obj_name == b.obj_name
        && a.qualname == b.qualname
        && a.arg_count == b.arg_count
        && a.posonlyarg_count == b.posonlyarg_count
        && a.kwonlyarg_count == b.kwonlyarg_count
        && a.varnames.len() == b.varnames.len()
        && a.flags == b.flags
        && a.first_line_number == b.first_line_number
        && a.instructions.original_bytes() == b.instructions.original_bytes()
        && a.names.len() == b.names.len()
        && a.constants.len() == b.constants.len()
        && a.varnames == b.varnames
        && a.freevars == b.freevars
        && a.cellvars == b.cellvars
        && a.names == b.names
        && a.linetable == b.linetable
        && a.exceptiontable == b.exceptiontable
        && crate::pyframe::code_constants(a)
            .iter()
            .zip(crate::pyframe::code_constants(b).iter())
            .all(|(left, right)| constant_strong_equal(left, right))
}

fn constant_strong_equal(
    left: &crate::bytecode::ConstantData,
    right: &crate::bytecode::ConstantData,
) -> bool {
    use crate::bytecode::ConstantData;
    match (left, right) {
        (ConstantData::Code { code: a }, ConstantData::Code { code: b }) => code_data_equal(a, b),
        (ConstantData::Tuple { elements: a }, ConstantData::Tuple { elements: b }) => {
            a.len() == b.len()
                && a.iter()
                    .zip(b.iter())
                    .all(|(x, y)| constant_strong_equal(x, y))
        }
        (ConstantData::Slice { elements: a }, ConstantData::Slice { elements: b }) => a
            .iter()
            .zip(b.iter())
            .all(|(x, y)| constant_strong_equal(x, y)),
        (ConstantData::Frozenset { elements: a }, ConstantData::Frozenset { elements: b }) => {
            if a.len() != b.len() {
                return false;
            }
            let mut matched = vec![false; b.len()];
            a.iter().all(|item| {
                b.iter()
                    .enumerate()
                    .find(|(index, candidate)| {
                        !matched[*index] && constant_strong_equal(item, candidate)
                    })
                    .map(|(index, _)| {
                        matched[index] = true;
                    })
                    .is_some()
            })
        }
        _ => left == right,
    }
}

pub unsafe fn code_eq(
    this: PyObjectRef,
    other: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    if !unsafe { is_code(other) } {
        return Ok(pyre_object::special::w_not_implemented());
    }
    let a = unsafe { require_code(this, "__eq__")? };
    let b = unsafe { require_code(other, "__eq__")? };
    if (*(this as *const PyCode)).co_firstlineno_raw
        != (*(other as *const PyCode)).co_firstlineno_raw
    {
        return Ok(w_bool_from(false));
    }
    Ok(w_bool_from(code_data_equal(a, b)))
}

pub unsafe fn code_ne(
    this: PyObjectRef,
    other: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let equal = unsafe { code_eq(this, other)? };
    if unsafe { pyre_object::is_not_implemented(equal) } {
        Ok(equal)
    } else {
        Ok(w_bool_from(!unsafe { w_bool_get_value(equal) }))
    }
}

pub unsafe fn code_hash(obj: PyObjectRef) -> Result<i64, crate::PyError> {
    let code = unsafe { require_code(obj, "__hash__")? };
    #[inline]
    fn scramble(result: i64, value: i64) -> i64 {
        ((result as u64 ^ value as u64).wrapping_mul(1_000_003)) as i64
    }
    #[inline]
    fn add_obj(result: &mut i64, value: PyObjectRef) -> Result<(), crate::PyError> {
        *result = scramble(*result, crate::baseobjspace::hash_w_strict(value)?);
        Ok(())
    }
    let mut result = 20_250_211i64;
    add_obj(&mut result, w_str_new(&code.obj_name))?;
    add_obj(&mut result, w_str_new(&code.qualname))?;
    for value in [
        code.arg_count as i64,
        code.posonlyarg_count as i64,
        code.kwonlyarg_count as i64,
        code.varnames.len() as i64,
        code.flags.bits() as i64,
        (*(obj as *const PyCode)).co_firstlineno_raw as i64,
    ] {
        result = scramble(result, value);
    }
    add_obj(
        &mut result,
        pyre_object::bytesobject::w_bytes_from_bytes(&code.instructions.original_bytes()),
    )?;
    add_obj(
        &mut result,
        pyre_object::bytesobject::w_bytes_from_bytes(&code.linetable),
    )?;
    add_obj(
        &mut result,
        pyre_object::bytesobject::w_bytes_from_bytes(&code.exceptiontable),
    )?;
    for names in [&code.varnames, &code.freevars, &code.cellvars, &code.names] {
        for name in names.iter() {
            add_obj(&mut result, w_str_new(name))?;
        }
    }
    for constant in crate::pyframe::code_constants(code) {
        add_obj(
            &mut result,
            crate::pyframe::pyobject_from_constant(constant),
        )?;
    }
    Ok(if result == -1 { -2 } else { result })
}

pub unsafe fn code_repr(obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let code = unsafe { require_code(obj, "__repr__")? };
    let line = (*(obj as *const PyCode)).co_firstlineno_raw as i64;
    Ok(w_str_new(&format!(
        "<code object {} at {obj:p}, file \"{}\", line {line}>",
        code.obj_name, code.source_path,
    )))
}

pub unsafe fn code_sizeof(obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let code = unsafe { require_code(obj, "__sizeof__")? };
    let size = std::mem::size_of::<PyCode>()
        + std::mem::size_of::<crate::CodeObject>()
        + code.instructions.len() * 2
        + code.locations.len()
            * std::mem::size_of::<(
                rustpython_compiler_core::SourceLocation,
                rustpython_compiler_core::SourceLocation,
            )>()
        + code.linetable.len()
        + code.exceptiontable.len();
    Ok(w_int_new(size as i64))
}

pub unsafe fn code_varname_from_oparg(
    obj: PyObjectRef,
    index: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    let code = unsafe { require_code(obj, "_varname_from_oparg")? };
    let mut index = unsafe { crate::builtins::space_index_w(index)? };
    if index >= 0 {
        if let Some(name) = code.varnames.get(index as usize) {
            return Ok(w_str_new(name));
        }
        index -= code.varnames.len() as i64;
        let pure_cellvars = code
            .cellvars
            .iter()
            .filter(|cell| !code.varnames.iter().any(|var| var == *cell));
        let pure_cellvar_count = pure_cellvars.clone().count();
        if let Some(name) = pure_cellvars.skip(index as usize).next() {
            return Ok(w_str_new(name));
        }
        index -= pure_cellvar_count as i64;
        if let Some(name) = code.freevars.get(index as usize) {
            return Ok(w_str_new(name));
        }
    }
    Err(crate::PyError::new(
        crate::PyErrorKind::IndexError,
        "tuple index out of range",
    ))
}

pub unsafe fn code_positions(obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let code = unsafe { require_code(obj, "co_positions")? };
    let rows = code
        .locations
        .iter()
        .map(|(start, end)| {
            w_tuple_new(vec![
                w_int_new(start.line.get() as i64),
                w_int_new(end.line.get() as i64),
                w_int_new(start.character_offset.get().saturating_sub(1) as i64),
                w_int_new(end.character_offset.get().saturating_sub(1) as i64),
            ])
        })
        .collect::<Vec<_>>();
    let n = rows.len();
    Ok(w_seq_iter_new(w_list_new(rows), n))
}

pub unsafe fn code_lines(obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let code = unsafe { require_code(obj, "co_lines")? };
    let mut rows = Vec::new();
    let mut start = 0usize;
    while start < code.locations.len() {
        let line = code.locations[start].0.line.get();
        let mut end = start + 1;
        while end < code.locations.len() && code.locations[end].0.line.get() == line {
            end += 1;
        }
        rows.push(w_tuple_new(vec![
            w_int_new((start * 2) as i64),
            w_int_new((end * 2) as i64),
            w_int_new(line as i64),
        ]));
        start = end;
    }
    let n = rows.len();
    Ok(w_seq_iter_new(w_list_new(rows), n))
}

pub unsafe fn code_branches(obj: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
    let code = unsafe { require_code(obj, "co_branches")? };
    let mut rows = Vec::new();
    let mut index = 0usize;
    let mut op_arg = 0usize;
    while index < code.instructions.len() {
        let op = code.instructions.read_op(index).deoptimize();
        let next = index + 1 + op.cache_entries();
        let arg = u8::from(code.instructions.read_arg(index)) as usize;
        match op {
            crate::bytecode::Instruction::ExtendedArg => {
                op_arg = (op_arg << 8) | arg;
            }
            crate::bytecode::Instruction::ForIter { .. } => {
                op_arg = (op_arg << 8) | arg;
                rows.push(w_tuple_new(vec![
                    w_int_new((index * 2) as i64),
                    w_int_new((next * 2) as i64),
                    w_int_new(((next + op_arg + 2) * 2) as i64),
                ]));
                op_arg = 0;
            }
            crate::bytecode::Instruction::PopJumpIfFalse { .. }
            | crate::bytecode::Instruction::PopJumpIfTrue { .. }
            | crate::bytecode::Instruction::PopJumpIfNone { .. }
            | crate::bytecode::Instruction::PopJumpIfNotNone { .. } => {
                op_arg = (op_arg << 8) | arg;
                // Python 3.14 inserts NOT_TAKEN at the fallthrough edge so
                // branch instrumentation can distinguish the untaken path.
                let not_taken = next + 1;
                rows.push(w_tuple_new(vec![
                    w_int_new((index * 2) as i64),
                    w_int_new((not_taken * 2) as i64),
                    w_int_new(((next + op_arg) * 2) as i64),
                ]));
                op_arg = 0;
            }
            crate::bytecode::Instruction::EndAsyncFor => {
                op_arg = (op_arg << 8) | arg;
                let source = next - op_arg;
                debug_assert!(matches!(
                    code.instructions.read_op(source).deoptimize(),
                    crate::bytecode::Instruction::EndSend
                ));
                debug_assert!(matches!(
                    code.instructions.read_op(source + 1).deoptimize(),
                    crate::bytecode::Instruction::NotTaken
                ));
                rows.push(w_tuple_new(vec![
                    w_int_new((source * 2) as i64),
                    w_int_new(((source + 2) * 2) as i64),
                    w_int_new((next * 2) as i64),
                ]));
                op_arg = 0;
            }
            _ => op_arg = 0,
        }
        index = next.max(index + 1);
    }
    let n = rows.len();
    Ok(w_seq_iter_new(w_list_new(rows), n))
}

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
                    "replace() got an unexpected keyword argument '{key}'"
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
    let mut firstlineno_raw = unsafe { (*(w_self as *const PyCode)).co_firstlineno_raw };
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
    let requested_nlocals = get("co_nlocals")
        .map(|v| unsafe { read_code_u32(v, "co_nlocals") })
        .transpose()?;
    if let Some(v) = get("co_stacksize") {
        code.max_stackdepth = unsafe { read_code_u32(v, "co_stacksize")? };
    }
    if let Some(v) = get("co_flags") {
        let value = unsafe { read_code_c_int(v)? };
        if value < 0 {
            return Err(crate::PyError::value_error(
                "co_flags must be a positive integer",
            ));
        }
        let bits = value as u32;
        code.flags = crate::bytecode::CodeFlags::from_bits_retain(bits);
    }
    if let Some(v) = get("co_firstlineno") {
        let n = unsafe { read_code_c_int(v)? } as i64;
        if n < 0 {
            return Err(crate::PyError::value_error(
                "co_firstlineno must be a positive integer",
            ));
        }
        firstlineno_raw = n.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
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

    if requested_nlocals.is_some_and(|n| n as usize != code.varnames.len()) {
        return Err(crate::PyError::value_error(
            "code: co_nlocals != len(co_varnames)",
        ));
    }
    if code.posonlyarg_count > code.arg_count {
        return Err(crate::PyError::new(
            crate::PyErrorKind::SystemError,
            "Objects/codeobject.c: bad argument to internal function",
        ));
    }
    if code.arg_count as usize + code.kwonlyarg_count as usize > code.varnames.len() {
        return Err(crate::PyError::value_error(
            "code: co_varnames is too small",
        ));
    }

    // CPython's locals-plus table marks local/cell aliases in their existing
    // local slot, then appends pure cells and free variables.
    let mut localspluskinds = vec![crate::bytecode::CO_FAST_LOCAL; code.varnames.len()];
    for cell in code.cellvars.iter() {
        if let Some(index) = code.varnames.iter().position(|name| name == cell) {
            localspluskinds[index] |= crate::bytecode::CO_FAST_CELL;
        } else {
            localspluskinds.push(crate::bytecode::CO_FAST_CELL);
        }
    }
    localspluskinds.extend(std::iter::repeat_n(
        crate::bytecode::CO_FAST_FREE,
        code.freevars.len(),
    ));
    code.localspluskinds = localspluskinds.into_boxed_slice();
    code.locations = rustpython_compiler_core::marshal::linetable_to_locations(
        &code.linetable,
        firstlineno_raw,
        code.instructions.len(),
    );

    Ok(box_code_constant_with_firstlineno(&code, firstlineno_raw))
}

/// A non-negative `co_*` count argument as `u32`.
unsafe fn read_code_u32(v: PyObjectRef, field: &str) -> Result<u32, crate::PyError> {
    let n = unsafe { read_code_c_int(v)? } as i64;
    if n < 0 {
        let message = if field.starts_with("co_") {
            format!("{field} must be a positive integer")
        } else {
            format!("code: {field} must not be negative")
        };
        return Err(crate::PyError::value_error(message));
    }
    Ok(n as u32)
}

/// Argument Clinic converts every public code-object integer through C `int`.
unsafe fn read_code_c_int(v: PyObjectRef) -> Result<i32, crate::PyError> {
    let n = unsafe { crate::builtins::space_index_w(v)? };
    i32::try_from(n).map_err(|_| {
        crate::PyError::new(
            crate::PyErrorKind::OverflowError,
            "Python int too large to convert to C int",
        )
    })
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
        if pyre_object::is_complex(obj) {
            return Ok(ConstantData::Complex {
                value: num_complex::Complex64::new(
                    pyre_object::w_complex_get_real(obj),
                    pyre_object::w_complex_get_imag(obj),
                ),
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
        if pyre_object::setobject::is_frozenset(obj) {
            let elements = pyre_object::w_set_items(obj)
                .into_iter()
                .map(|item| obj_to_constant_data(item))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(ConstantData::Frozenset { elements });
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

/// Cached [`crate::pyframe::npure_cellvars`] for the code wrapper `obj`,
/// or `None` for a null/stub wrapper (sentinel `u32::MAX`) so the caller
/// falls back to recomputation.
///
/// # Safety
/// `obj` must be null or point to a valid `PyCode`.
#[inline]
pub unsafe fn w_code_npure_cellvars(obj: PyObjectRef) -> Option<usize> {
    if obj.is_null() {
        return None;
    }
    match unsafe { (*(obj as *const PyCode)).npure_cellvars } {
        u32::MAX => None,
        n => Some(n as usize),
    }
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

pub(crate) fn capture_mapdict_method_cache_root_area() -> *const () {
    MAPDICT_METHOD_CACHE_CODES.with(|codes| codes as *const _ as *const ())
}

/// # Safety
/// `data` must come from [`capture_mapdict_method_cache_root_area`], and the
/// owning thread must be quiesced.
pub(crate) unsafe fn walk_mapdict_method_cache_root_area(
    data: *const (),
    forward: &mut dyn FnMut(&mut PyObjectRef),
) {
    let codes = unsafe {
        &*(*(data as *const std::cell::RefCell<std::collections::HashSet<usize>>)).as_ptr()
    };
    for &code in codes.iter() {
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
