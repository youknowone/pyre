//! W_CodeObject — Python `code` object wrapper.
//!
//! Wraps an opaque pointer to the compiler's CodeObject, allowing it to
//! be placed on the value stack as a PyObjectRef during `LoadConst`.
//! MakeFunction then extracts this pointer to build a function object.

use pyre_object::pyobject::*;

/// Compatibility alias for PyPy's `PyCode` type.
pub type PyCode = W_CodeObject;

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
pub struct W_CodeObject {
    pub ob_header: PyObject,
    /// Opaque pointer to a `CodeObject` (owned via Box::into_raw).
    pub code_ptr: *const (),
    /// PyPy: `PyCode.w_globals`.
    pub w_globals: *mut crate::DictStorage,
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
}

/// Field offset of `code_ptr` within `W_CodeObject`.
pub const CODE_PTR_OFFSET: usize = std::mem::offset_of!(W_CodeObject, code_ptr);
/// Field offset of `w_globals` within `W_CodeObject`.
pub const CODE_W_GLOBALS_OFFSET: usize = std::mem::offset_of!(W_CodeObject, w_globals);

/// GC type id assigned to `W_CodeObject`.
///
/// `PyCode` is a normal `W_Root` subclass in PyPy (`pycode.py:52 class
/// PyCode(W_Root)`), so it lives in the GC heap.  This tid is pinned by
/// a `debug_assert_eq!` in the pyre-jit type-registration sequence: the
/// `W_CodeObject` `TypeInfo` is registered explicitly just before the
/// foreign-pytype loop, taking the slot directly after
/// `GC_FLOAT_ARRAY_GC_TYPE_ID = 42`.  Pre-registering it there (and
/// inserting `CODE_TYPE` into `pytype_to_tid`) makes the foreign loop
/// skip `CODE_TYPE`, so the net register-call count up to
/// `W_MODULE_DICT_GC_TYPE_ID = 48` is unchanged and no downstream tid
/// shifts.  The numeric value coincides with the dormant
/// `pytraceback::PYTRACEBACK_GC_TYPE_ID` constant, but `W_PyTraceback`
/// is still host-allocated and is never GC-registered, so tid 43 only
/// ever tags a `W_CodeObject` at runtime and the two do not collide.
pub const W_CODE_GC_TYPE_ID: u32 = 43;

impl pyre_object::lltype::GcType for W_CodeObject {
    fn type_id() -> u32 {
        W_CODE_GC_TYPE_ID
    }
    const SIZE: usize = std::mem::size_of::<W_CodeObject>();
}

/// Compatibility helper for unpacking a tuple of strings.
pub fn unpack_text_tuple(_space: PyObjectRef, w_str_tuple: PyObjectRef) -> Vec<String> {
    let _ = (_space, w_str_tuple);
    Vec::new()
}

/// Compatibility API for building a signature-like object.
pub fn make_signature(_code: &W_CodeObject) -> PyObjectRef {
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
    let obj = Box::new(W_CodeObject {
        ob_header: PyObject {
            ob_type: &CODE_TYPE as *const PyType,
            w_class: pyre_object::pyobject::get_instantiate(&CODE_TYPE),
        },
        code_ptr,
        w_globals: std::ptr::null_mut(),
        hidden_applevel,
        fast_natural_arity,
        globals_caches,
        mapdict_caches,
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

thread_local! {
    /// Realized `W_CodeObject` for each nested code constant, keyed by
    /// the frozen `CodeObject`'s address.  `pycode.py` realizes a nested
    /// code constant into one `PyCode` at enclosing-code construction and
    /// shares it through `co_consts`; pyre realizes lazily on `LOAD_CONST`,
    /// so this table reproduces the sharing.  The frozen `CodeObject` lives
    /// in the enclosing code's `constants` array, and that enclosing code is
    /// Box-immortal (`w_code_new` → `Box::into_raw`), so the key address is
    /// stable for the process and never reused.  The cached wrappers are
    /// likewise Box-immortal, so the raw pointers never dangle and need no
    /// GC tracing (same rationale as `MAPDICT_METHOD_CACHE_CODES`).
    static CODE_CONSTANT_INTERN: std::cell::RefCell<std::collections::HashMap<usize, PyObjectRef>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// Realize a nested code constant into its shared `W_CodeObject`,
/// reproducing `co_consts` holding one realized code object per nested
/// code.  `code` must be the frozen constant embedded in an enclosing
/// (Box-immortal) `CodeObject`'s `constants` array — its address is the
/// stable identity key.  Repeated `LOAD_CONST` of the same nested code
/// returns the same wrapper, giving stable `__code__` identity and a
/// stable JIT green key (`greens = [..., 'pycode']`).  Without this the
/// per-`LOAD_CONST` wrapper differs every call, so a closure defined in a
/// hot-loop-called function gets a fresh green key each iteration and the
/// trace give-up counter never accumulates.
pub fn intern_code_constant(code: &crate::CodeObject) -> PyObjectRef {
    let key = code as *const crate::CodeObject as usize;
    if let Some(w) = CODE_CONSTANT_INTERN.with(|c| c.borrow().get(&key).copied()) {
        return w;
    }
    let w = box_code_constant(code);
    CODE_CONSTANT_INTERN.with(|c| c.borrow_mut().insert(key, w));
    w
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
/// `obj` must point to a valid `W_CodeObject`.
#[inline]
pub unsafe fn w_code_set_hidden_applevel(obj: PyObjectRef, hidden_applevel: bool) {
    if obj.is_null() {
        return;
    }
    unsafe {
        (*(obj as *mut W_CodeObject)).hidden_applevel = hidden_applevel;
    }
}

/// Extract the opaque code pointer from a known W_CodeObject.
///
/// # Safety
/// `obj` must point to a valid `W_CodeObject`.
#[inline]
pub unsafe fn w_code_get_ptr(obj: PyObjectRef) -> *const () {
    unsafe { (*(obj as *const W_CodeObject)).code_ptr }
}

/// PyPy: `PyCode.hidden_applevel` (`pycode.py:147`). Reads the field
/// initialised by `w_code_new`.  `pyframe.py:521-522
/// hide(self): return self.pycode.hidden_applevel` is the sole caller
/// in the canonical interpreter; pyre routes through this accessor
/// from `pyframe.rs::PyFrame::hide`.
///
/// # Safety
/// `obj` must point to a valid `W_CodeObject`.
#[inline]
pub unsafe fn w_code_hidden_applevel(obj: PyObjectRef) -> bool {
    if obj.is_null() {
        return false;
    }
    unsafe { (*(obj as *const W_CodeObject)).hidden_applevel }
}

/// PyPy: `PyCode.w_globals`.
#[inline]
pub unsafe fn w_code_get_w_globals(obj: PyObjectRef) -> *mut crate::DictStorage {
    if obj.is_null() {
        return std::ptr::null_mut();
    }
    unsafe { (*(obj as *const W_CodeObject)).w_globals }
}

/// PyPy: `PyCode.w_globals = w_globals`.
#[inline]
pub unsafe fn w_code_set_w_globals(obj: PyObjectRef, w_globals: *mut crate::DictStorage) {
    if obj.is_null() {
        return;
    }
    unsafe {
        (*(obj as *mut W_CodeObject)).w_globals = w_globals;
    }
}

/// PyPy: `PyCode.frame_stores_global(w_globals)`.
#[inline]
pub unsafe fn w_code_frame_stores_global(
    obj: PyObjectRef,
    w_globals: *mut crate::DictStorage,
) -> bool {
    if obj.is_null() {
        return false;
    }
    let code = unsafe { &mut *(obj as *mut W_CodeObject) };
    if code.w_globals.is_null() {
        code.w_globals = w_globals;
        return false;
    }
    !std::ptr::eq(code.w_globals, w_globals)
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

/// eval.py:16-23 — read `fast_natural_arity` from a W_CodeObject.
///
/// # Safety
/// `obj` must point to a valid `W_CodeObject`.
#[inline]
pub unsafe fn w_code_get_fast_natural_arity(obj: PyObjectRef) -> u16 {
    if obj.is_null() {
        return crate::gateway::HOPELESS;
    }
    unsafe { (*(obj as *const W_CodeObject)).fast_natural_arity }
}

/// Unified accessor: read `fast_natural_arity` from any code object
/// (BuiltinCode or W_CodeObject).
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
/// `obj` must point to a valid `W_CodeObject`.
#[inline]
pub unsafe fn w_code_lookup_exceptiontable(
    obj: PyObjectRef,
    instr_offset: u32,
) -> Option<(u32, u32, bool)> {
    if obj.is_null() {
        return None;
    }
    let code_ptr = unsafe { (*(obj as *const W_CodeObject)).code_ptr };
    if code_ptr.is_null() {
        return None;
    }
    let code = unsafe { &*(code_ptr as *const crate::CodeObject) };
    crate::exception_table::lookup_exceptiontable(&code.exceptiontable, instr_offset)
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
/// `obj` must point to a valid `W_CodeObject`.
#[inline]
pub unsafe fn w_code_exceptiontable(obj: PyObjectRef) -> Vec<u8> {
    if obj.is_null() {
        return Vec::new();
    }
    let code_ptr = unsafe { (*(obj as *const W_CodeObject)).code_ptr };
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
/// `obj` must point to a valid `W_CodeObject` (or be null).
#[inline]
pub unsafe fn w_code_globals_caches_get(
    obj: PyObjectRef,
    nameindex: usize,
) -> Option<std::rc::Rc<std::cell::RefCell<pyre_object::celldict::GlobalCache>>> {
    if obj.is_null() {
        return None;
    }
    let code = unsafe { &*(obj as *const W_CodeObject) };
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
/// `obj` must point to a valid `W_CodeObject` (or be null).
#[inline]
pub unsafe fn w_code_globals_caches_set(
    obj: PyObjectRef,
    nameindex: usize,
    cache: &std::rc::Rc<std::cell::RefCell<pyre_object::celldict::GlobalCache>>,
) {
    if obj.is_null() {
        return;
    }
    let code = unsafe { &*(obj as *const W_CodeObject) };
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
/// `obj` must point to a valid `W_CodeObject` (or be null).
#[inline]
pub unsafe fn w_code_globals_caches_len(obj: PyObjectRef) -> usize {
    if obj.is_null() {
        return 0;
    }
    let code = unsafe { &*(obj as *const W_CodeObject) };
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
/// `obj` must point to a valid `W_CodeObject` (or be null).
#[inline]
pub unsafe fn w_code_mapdict_caches_get(
    obj: PyObjectRef,
    nameindex: usize,
) -> Option<crate::objspace::std::mapdict::MapdictCacheEntry> {
    if obj.is_null() {
        return None;
    }
    let code = unsafe { &*(obj as *const W_CodeObject) };
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
/// `obj` must point to a valid `W_CodeObject` (or be null).
#[inline]
pub unsafe fn w_code_mapdict_caches_set(
    obj: PyObjectRef,
    nameindex: usize,
    entry: crate::objspace::std::mapdict::MapdictCacheEntry,
) {
    if obj.is_null() {
        return;
    }
    let code = unsafe { &*(obj as *const W_CodeObject) };
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
            let code = unsafe { &*(code as *const W_CodeObject) };
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
/// `obj` must point to a valid `W_CodeObject` (or be null).
#[inline]
pub unsafe fn w_code_mapdict_caches_len(obj: PyObjectRef) -> usize {
    if obj.is_null() {
        return 0;
    }
    let code = unsafe { &*(obj as *const W_CodeObject) };
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
