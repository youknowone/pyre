//! Core Python object model with `#[repr(C)]` layout for JIT compatibility.
//!
//! Every Python object starts with a `PyObject` header containing a type pointer.
//! Concrete types (W_IntObject, W_BoolObject, etc.) embed this header as their
//! first field, enabling safe pointer casts between `*mut PyObject` and typed pointers.

use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicI64, AtomicPtr, AtomicU64, Ordering};

/// Type descriptor for Python objects — corresponds to RPython's OBJECT_VTABLE
/// (rclass.py:167-174).
///
/// Each built-in type has a single static `PyType` instance.
/// The JIT uses `GuardClass` on the `ob_type` pointer to specialize code paths,
/// and `GuardSubclass` via `int_between(cls.min, subcls.min, cls.max)`
/// (rclass.py:1133-1137 `ll_issubclass`).
///
/// Fields match OBJECT_VTABLE layout order:
///   subclassrange_min, subclassrange_max, (rtti omitted), name, (instantiate omitted)
///
/// `AtomicI64`/`AtomicPtr` provide interior mutability for static instances:
/// ranges and instantiate are assigned once at init time,
/// mirroring `assign_inheritance_ids` (normalizecalls.py:373-389).
/// The JIT backend reads them at raw offsets — atomics are layout-
/// compatible with their inner types (same size and alignment).
#[repr(C)]
pub struct PyType {
    pub subclassrange_min: AtomicI64,
    pub subclassrange_max: AtomicI64,
    pub name: &'static str,
    /// rclass.py:172 `('instantiate', Ptr(FuncType([], OBJECTPTR)))`.
    ///
    /// RPython stores an instantiate function pointer; pyre caches
    /// the W_TypeObject pointer here instead. rclass.py:739-743
    /// `new_instance` sets `__class__` at allocation — pyre reads
    /// this cached pointer to set `w_class` at allocation time.
    /// Null until `init_typeobjects()` runs.
    pub instantiate: AtomicPtr<PyObject>,
}

/// Common header for all Python objects.
///
/// RPython rclass.py: OBJECT = GcStruct('object', ('typeptr', CLASSTYPE))
///
/// - `ob_type`: static dispatch tag (like RPython's typeptr for guard_class)
/// - `w_class`: Python class pointer (like RPython's gettypefor(typeptr) result)
///
/// `w_class` is set at allocation time when the type registry is available,
/// or populated lazily by `init_typeobjects()` for static singletons.
#[repr(C)]
pub struct PyObject {
    pub ob_type: *const PyType,
    pub w_class: *mut PyObject,
}

impl Default for PyObject {
    /// Null header — `Self::allocate` rewrites both fields at malloc time.
    #[inline]
    fn default() -> Self {
        Self {
            ob_type: std::ptr::null(),
            w_class: std::ptr::null_mut(),
        }
    }
}

/// The universal Python object reference — a raw pointer to `PyObject`.
///
/// `pyre` currently passes this through the JIT as an integer-sized raw pointer.
/// Uses leaked Box allocations; a proper GC will replace this later.
// Safety: PyType instances are read-only static data, safe to share across threads.
unsafe impl Sync for PyType {}
unsafe impl Send for PyType {}

// Safety: PyObject's ob_type points to immutable static PyType instances.
unsafe impl Sync for PyObject {}
unsafe impl Send for PyObject {}

pub type PyObjectRef = *mut PyObject;

/// Null object reference, used as a sentinel for "no value".
pub const PY_NULL: PyObjectRef = std::ptr::null_mut();

// ── Type identity ─────────────────────────────────────────────────────

/// Construct a PyType with zeroed subclass ranges.
/// Ranges are assigned at init time by `assign_subclass_range()`.
pub const fn new_pytype(name: &'static str) -> PyType {
    PyType {
        subclassrange_min: AtomicI64::new(0),
        subclassrange_max: AtomicI64::new(0),
        name,
        instantiate: AtomicPtr::new(std::ptr::null_mut()),
    }
}

/// rclass.py:739-743 parity — cache the W_TypeObject on the PyType
/// so allocators can set `w_class` at allocation time.
///
/// Called by `init_typeobjects()` for each built-in type.
pub fn set_instantiate(tp: &PyType, w_typeobject: PyObjectRef) {
    tp.instantiate.store(w_typeobject, Ordering::Release);
}

/// Read the cached W_TypeObject from a PyType.
///
/// Returns the W_TypeObject (for `w_class`), or null if not yet initialized
/// (bootstrap phase before `init_typeobjects()`).
#[inline]
pub fn get_instantiate(tp: &PyType) -> PyObjectRef {
    tp.instantiate.load(Ordering::Acquire)
}

/// True when `obj`'s Python class is exactly the builtin type for its
/// layout — i.e. NOT a user subclass.
///
/// A user subclass of a builtin keeps the builtin `ob_type` (and therefore
/// the builtin struct layout and the `is_int` / `is_list` / … layout
/// predicates) while `w_class` is retagged to the subclass type object
/// (`typedef::subclass_to_tag`).  The type-specific fast paths in
/// `space.is_true` / `eq_w` / `len` / `getitem` / … assume the receiver's
/// Python class IS the builtin (no overridable special method); for a
/// subclass instance they would bypass an overridden `__bool__` / `__len__`
/// / `__eq__` / `__getitem__` / … .  Gate each fast path on this predicate
/// and let a subclass fall through to the MRO `lookup` path.
///
/// A fresh builtin carries `w_class == get_instantiate(ob_type)` (see
/// `w_int_new` etc.); the read-only singletons (`True` / `False` / `None` /
/// `Ellipsis` / `NotImplemented`) leave `w_class` null and are always exact.
///
/// # Safety
/// `obj` must be null or a valid `PyObjectRef`.
#[inline]
pub unsafe fn is_exact_builtin_instance(obj: PyObjectRef) -> bool {
    // A tagged immediate is an exact builtin `int` (subclasses stay boxed),
    // so it is always an exact builtin instance. Gated on `CAN_BE_TAGGED`
    // (default false), synthesized before the `w_class`/`ob_type` derefs.
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        return true;
    }
    if obj.is_null() {
        return false;
    }
    unsafe {
        let w_class = (*obj).w_class;
        w_class.is_null() || std::ptr::eq(w_class, get_instantiate(&*(*obj).ob_type))
    }
}

/// `type(obj) is <the builtin type object for `tp`>` — the exact-type test
/// used for pickle dispatch and the `tuple`/`str`/`float` constructors.
///
/// Unlike [`is_exact_builtin_instance`] this is correct for the specialised
/// arity-2 tuples: they carry a distinct `ob_type`
/// (`SPECIALISED_TUPLE_*_TYPE`) but a `w_class` of the canonical `tuple` type
/// object, so `is_exact_type(t, &TUPLE_TYPE)` is `true` for them while
/// `is_exact_builtin_instance` (which keys off `ob_type`) is not.  A user
/// subclass retags `w_class` to its own type object and so is rejected.
///
/// # Safety
/// `obj` must be null or a valid `PyObjectRef`; `tp` must be a canonical
/// builtin layout type with `get_instantiate(tp)` initialized.
#[inline]
pub unsafe fn is_exact_type(obj: PyObjectRef, tp: &PyType) -> bool {
    // A tagged immediate is always an exact builtin `int` (never a
    // subclass — those stay boxed via `w_int_new_unique`), so it is the
    // exact `tp` iff `tp` is the `int` vtable. Gated on `CAN_BE_TAGGED`
    // (default false), synthesized before the `w_class`/`ob_type` derefs.
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        return std::ptr::eq(tp as *const PyType, &INT_TYPE as *const PyType);
    }
    if obj.is_null() {
        return false;
    }
    unsafe {
        let w_class = (*obj).w_class;
        if w_class.is_null() {
            std::ptr::eq((*obj).ob_type, tp as *const PyType)
        } else {
            std::ptr::eq(w_class, get_instantiate(tp))
        }
    }
}

// Compile-time verification: AtomicI64/AtomicPtr are layout-compatible
// with i64/*mut T so the JIT can read PyType fields at raw offsets.
// Also verify OBJECT_VTABLE field order: subclassrange_min @ 0, max @ 8.
const _: () = {
    assert!(std::mem::size_of::<AtomicI64>() == std::mem::size_of::<i64>());
    assert!(std::mem::align_of::<AtomicI64>() == std::mem::align_of::<i64>());
    assert!(std::mem::size_of::<AtomicPtr<PyObject>>() == std::mem::size_of::<*mut PyObject>());
    assert!(std::mem::offset_of!(PyType, subclassrange_min) == 0);
    assert!(std::mem::offset_of!(PyType, subclassrange_max) == 8);
    // `instantiate` must sit immediately after `name` with no padding, so
    // its offset is exactly `offset_of(name) + size_of::<&str>()`: 32 on a
    // 64-bit JIT host (`&str` is a 16-byte fat pointer), 24 on 32-bit
    // targets such as wasm32 (`&str` is 8 bytes). The JIT reads this slot
    // at a raw offset via the exact Charon struct layout on the 64-bit
    // host; this assert pins the no-padding invariant on every target.
    assert!(
        std::mem::offset_of!(PyType, instantiate)
            == std::mem::offset_of!(PyType, name) + std::mem::size_of::<&'static str>()
    );
};

pub static INT_TYPE: PyType = new_pytype("int");
pub static BOOL_TYPE: PyType = new_pytype("bool");
pub static FLOAT_TYPE: PyType = new_pytype("float");
pub static COMPLEX_TYPE: PyType = new_pytype("complex");
pub static STR_TYPE: PyType = new_pytype("str");
pub static LIST_TYPE: PyType = new_pytype("list");
pub static TUPLE_TYPE: PyType = new_pytype("tuple");
pub static DICT_TYPE: PyType = new_pytype("dict");
pub static LONG_TYPE: PyType = new_pytype("int");
pub static NONE_TYPE: PyType = new_pytype("NoneType");
pub static NOTIMPLEMENTED_TYPE: PyType = new_pytype("NotImplementedType");
pub static ELLIPSIS_TYPE: PyType = new_pytype("ellipsis");
pub static MODULE_TYPE: PyType = new_pytype("module");
pub static MAPPING_PROXY_TYPE: PyType = new_pytype("mappingproxy");
pub static TYPE_TYPE: PyType = new_pytype("type");
pub static INSTANCE_TYPE: PyType = new_pytype("object");

/// Field offset of `ob_type` within PyObject, for JIT field access.
pub const OB_TYPE_OFFSET: usize = std::mem::offset_of!(PyObject, ob_type);

/// Field offset of `w_class` within PyObject, for JIT field access.
/// RPython: this corresponds to reading typeptr + gettypefor (fused into one field).
pub const W_CLASS_OFFSET: usize = std::mem::offset_of!(PyObject, w_class);

/// Field offset of `subclassrange_min` within PyType (OBJECT_VTABLE).
/// rclass.py:168 — first field in OBJECT_VTABLE.
pub const SUBCLASSRANGE_MIN_OFFSET: usize = std::mem::offset_of!(PyType, subclassrange_min);

/// Field offset of `subclassrange_max` within PyType (OBJECT_VTABLE).
/// rclass.py:169 — second field in OBJECT_VTABLE.
pub const SUBCLASSRANGE_MAX_OFFSET: usize = std::mem::offset_of!(PyType, subclassrange_max);

/// Field offset of `instantiate` within PyType (OBJECT_VTABLE).
/// rclass.py:172 — `('instantiate', Ptr(FuncType([], OBJECTPTR)))`.
/// 32 on a 64-bit host (`name` is a 16-byte fat pointer); 24 on 32-bit
/// targets where `&str` is 8 bytes.
pub const INSTANTIATE_OFFSET: usize = std::mem::offset_of!(PyType, instantiate);

/// rclass.py:1126-1127 `ll_cast_to_object(obj)`.
///
/// In RPython this casts a typed pointer to `OBJECTPTR`. In pyre all
/// objects are already `PyObjectRef`, so this is an identity function
/// kept for structural parity.
#[inline]
pub fn ll_cast_to_object(obj: PyObjectRef) -> PyObjectRef {
    obj
}

/// rclass.py:1130-1131 `ll_type(obj)`.
///
/// Extract the type pointer (CLASSTYPE) from an object.
///
/// # Safety
/// `obj` must be a valid non-null `PyObject`.
#[inline]
pub unsafe fn ll_type(obj: PyObjectRef) -> *const PyType {
    // `ll_unboxed_getclass`: a tagged immediate's class is the `int`
    // vtable, synthesized before the `ob_type` deref. Gated on the
    // `CAN_BE_TAGGED` static (default false), so the deref is the only
    // live path until enablement.
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        return &INT_TYPE as *const PyType;
    }
    unsafe { (*obj).ob_type }
}

/// rclass.py:1133-1137 `ll_issubclass(subcls, cls)`.
///
/// O(1) subclass check via preorder numbering:
///   `int_between(cls.subclassrange_min, subcls.subclassrange_min, cls.subclassrange_max)`
#[inline]
pub fn ll_issubclass(subcls: &PyType, cls: &PyType) -> bool {
    // Seqlock read: a concurrent one-time batch re-stamp must not be observed
    // half-applied, or `cls`/`subcls` could temporarily carry ranges from
    // different completed batches.
    subclass_range_read(|| {
        let cls_min = cls.subclassrange_min.load(Ordering::Relaxed);
        let subcls_min = subcls.subclassrange_min.load(Ordering::Relaxed);
        let cls_max = cls.subclassrange_max.load(Ordering::Relaxed);
        // int_between(a, b, c) ≡ a <= b < c
        cls_min <= subcls_min && subcls_min < cls_max
    })
}

/// rclass.py:1139-1140 `ll_issubclass_const(subcls, minid, maxid)`.
///
/// Variant of `ll_issubclass` where the class bounds are already known
/// constants. Used by the JIT when the target class is constant-folded.
#[inline]
pub fn ll_issubclass_const(subcls: &PyType, minid: i64, maxid: i64) -> bool {
    // Seqlock read: `minid`/`maxid` are baked from one numbering, so
    // `subcls_min` must be read from a matching (fully-published) batch.
    subclass_range_read(|| {
        let subcls_min = subcls.subclassrange_min.load(Ordering::Relaxed);
        // int_between(a, b, c) ≡ a <= b < c
        minid <= subcls_min && subcls_min < maxid
    })
}

/// rclass.py:1143-1147 `ll_isinstance(obj, cls)`.
///
/// RPython-level type check: reads `obj.typeptr` (= `ob_type`) and checks
/// subclass ranges. This checks the **RPython class** (W_IntObject,
/// W_ListObject, etc.), NOT the Python-level class. All user-defined
/// instances share `INSTANCE_TYPE` as their RPython class, just as
/// RPython groups them under W_ObjectObject's vtable.
///
/// For Python-level `isinstance()`, use `issubtype_w` (MRO walk on
/// `w_class`), not this function.
///
/// # Safety
/// `obj` must be a valid non-null `PyObject`.
#[inline]
pub unsafe fn ll_isinstance(obj: PyObjectRef, cls: &PyType) -> bool {
    // `ll_unboxed_isinstance`: a tagged immediate's RPython class is the
    // `int` vtable, checked against `cls`'s subclass range without the
    // `ob_type` deref. Gated on `CAN_BE_TAGGED` (default false).
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        return ll_issubclass(&INT_TYPE, cls);
    }
    if obj.is_null() {
        return false;
    }
    let obj_cls = unsafe { &*(*obj).ob_type };
    ll_issubclass(obj_cls, cls)
}

/// rclass.py:1173-1178 `ll_inst_type(obj)`.
///
/// Return the typeptr if obj is non-null, null otherwise.
///
/// # Safety
/// If non-null, `obj` must be a valid `PyObject`.
#[inline]
pub unsafe fn ll_inst_type(obj: PyObjectRef) -> *const PyType {
    // `ll_unboxed_getclass_canbenone`: a tagged immediate has the low
    // bit set and is therefore non-null, so the `int`-vtable synth
    // precedes the null check. Gated on `CAN_BE_TAGGED` (default false).
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        return &INT_TYPE as *const PyType;
    }
    unsafe {
        if !obj.is_null() {
            (*obj).ob_type
        } else {
            std::ptr::null()
        }
    }
}

/// Write subclass ranges to a `PyType` instance.
///
/// Mirrors `assign_inheritance_ids` (normalizecalls.py:373-389) which
/// assigns `classdef.minid` / `classdef.maxid` to each vtable entry.
///
/// Uses `Relaxed` ordering: ranges are written once at init time
/// before any concurrent reads.
pub fn assign_subclass_range(tp: &PyType, min: i64, max: i64) {
    tp.subclassrange_min.store(min, Ordering::Relaxed);
    tp.subclassrange_max.store(max, Ordering::Relaxed);
}

/// Sequence lock (seqlock) guarding the batch (re)stamping of the static
/// `subclassrange_{min,max}` fields. The interpreter and GC initializers use
/// the same registration-ordered `TotalOrderSymbolic` numbering, but a reader
/// must still not observe a partially written batch while startup writers run
/// concurrently.
///
/// A seqlock, not a mutex/rwlock, because this is free-threaded (`nogil`):
/// the writes happen once at startup while `ll_issubclass` is a hot,
/// concurrently-read path.  Optimistic readers touch only `SUBCLASS_RANGE_SEQ`
/// with plain loads (no read-side atomic RMW), so once both one-time inits
/// settle and the sequence stops changing, concurrent readers share that
/// cache line read-only with no cross-core contention.  Even = stable,
/// odd = a batch write is in flight.
static SUBCLASS_RANGE_SEQ: AtomicU64 = AtomicU64::new(0);

/// Serializes the (rare, one-time) writers against each other so the seqlock
/// parity stays well-formed; readers never touch it.
static SUBCLASS_RANGE_WRITER_LOCK: Mutex<()> = Mutex::new(());

/// RAII write section for a batch subclass-range update.  Held by
/// `compute_subclass_ranges_from` and the JIT GC-tid writeback in `eval.rs`
/// for the whole batch: entering makes the sequence odd (optimistic readers
/// retry), dropping publishes the writes and makes it even again.
pub struct SubclassRangeWriteGuard {
    _writers: std::sync::MutexGuard<'static, ()>,
    seq: u64,
}

/// Enter a subclass-range write section (see [`SubclassRangeWriteGuard`]).
pub fn subclass_range_write_guard() -> SubclassRangeWriteGuard {
    let writers = SUBCLASS_RANGE_WRITER_LOCK
        .lock()
        .unwrap_or_else(|poison| poison.into_inner());
    // Serialized by the writer lock, so this load/store pair is race-free.
    let seq = SUBCLASS_RANGE_SEQ.load(Ordering::Relaxed).wrapping_add(1);
    SUBCLASS_RANGE_SEQ.store(seq, Ordering::Relaxed);
    std::sync::atomic::fence(Ordering::Release);
    SubclassRangeWriteGuard {
        _writers: writers,
        seq,
    }
}

impl Drop for SubclassRangeWriteGuard {
    fn drop(&mut self) {
        // Publish the batch, then leave the write section (sequence even).
        std::sync::atomic::fence(Ordering::Release);
        SUBCLASS_RANGE_SEQ.store(self.seq.wrapping_add(1), Ordering::Release);
    }
}

/// Optimistic seqlock read: run `read` (which loads the relevant
/// `subclassrange_*` atomics) and retry until it lands in a window with no
/// concurrent batch write, so the returned value reflects one coherent
/// numbering.  In steady state (sequence stable) this is two plain loads of
/// `SUBCLASS_RANGE_SEQ` plus one acquire fence — no read-side RMW.
#[inline]
fn subclass_range_read<T>(read: impl Fn() -> T) -> T {
    loop {
        let seq1 = SUBCLASS_RANGE_SEQ.load(Ordering::Acquire);
        if seq1 & 1 != 0 {
            std::hint::spin_loop();
            continue;
        }
        let value = read();
        std::sync::atomic::fence(Ordering::Acquire);
        let seq2 = SUBCLASS_RANGE_SEQ.load(Ordering::Relaxed);
        if seq1 == seq2 {
            return value;
        }
    }
}

/// One static `PyType` alias for an `rclass.OBJECT` typeid.
///
/// Several PyTypes can share one GC typeid (for example `set` and
/// `frozenset`), while some GC typeids contribute an inheritance peer but
/// have no vtable alias. Keeping aliases separate from the hierarchy
/// preserves that exact GC registration shape.
#[derive(Clone, Copy)]
pub struct SubclassRangeAlias {
    pub type_id: u32,
    pub pytype: &'static PyType,
}

pub const fn subclass_range_alias(type_id: u32, pytype: &'static PyType) -> SubclassRangeAlias {
    SubclassRangeAlias { type_id, pytype }
}

/// Canonical `rclass.OBJECT` inheritance census in GC registration order.
///
/// Each entry is `(typeid, parent_typeid)`. This is the shared input for the
/// interpreter-side fallback and is checked against the GC's `TypeRegistry`
/// before `freeze_types`. Sparse non-object typeids are intentionally absent.
/// The order and parent links mirror the `TypeInfo::object{,_subclass}` calls
/// in `pyre-jit/src/eval.rs`.
pub const SUBCLASS_RANGE_HIERARCHY: &[(u32, Option<u32>)] = &[
    (0, None),
    (1, Some(0)),
    (2, Some(0)),
    (5, Some(1)),
    (6, Some(0)),
    (7, Some(0)),
    (8, Some(0)),
    (10, Some(0)),
    (11, Some(0)),
    (12, Some(0)),
    (13, Some(0)),
    (14, Some(0)),
    (15, Some(0)),
    (16, Some(0)),
    (17, Some(0)),
    (18, Some(0)),
    (19, Some(0)),
    (20, Some(0)),
    (21, Some(0)),
    (22, Some(0)),
    (23, Some(0)),
    (24, Some(0)),
    (25, Some(0)),
    (26, Some(0)),
    (27, Some(0)),
    (28, Some(0)),
    (29, Some(0)),
    (30, Some(0)),
    (31, Some(0)),
    (32, Some(0)),
    (33, Some(0)),
    (34, Some(0)),
    (35, Some(0)),
    (36, Some(0)),
    (38, Some(0)),
    (39, Some(0)),
    (40, Some(0)),
    (43, Some(0)),
    (44, Some(0)),
    (45, Some(0)),
    (46, Some(0)),
    (47, Some(0)),
    (48, Some(0)),
    (49, Some(0)),
    (50, Some(0)),
    (52, Some(0)),
    (53, Some(0)),
    (54, Some(0)),
    (56, Some(0)),
    (57, Some(31)),
    (58, Some(31)),
    (59, Some(31)),
    (60, Some(57)),
    (61, Some(60)),
    (62, Some(60)),
    (63, Some(57)),
    (64, Some(57)),
    (65, Some(64)),
    (66, Some(65)),
    (67, Some(65)),
    (68, Some(65)),
    (69, Some(57)),
    (70, Some(57)),
    (71, Some(70)),
    (72, Some(70)),
    (73, Some(57)),
    (74, Some(57)),
    (75, Some(74)),
    (76, Some(74)),
    (77, Some(57)),
    (78, Some(57)),
    (79, Some(57)),
    (80, Some(57)),
    (81, Some(57)),
    (82, Some(81)),
    (83, Some(57)),
    (84, Some(57)),
    (85, Some(0)),
    (86, Some(0)),
    (87, Some(0)),
    (88, Some(0)),
    (89, Some(0)),
    (90, Some(0)),
    (91, Some(0)),
    (92, Some(0)),
    (93, Some(0)),
    (94, Some(0)),
    (95, Some(0)),
    (96, Some(0)),
    (97, Some(0)),
    (98, Some(0)),
    (99, Some(0)),
    (100, Some(0)),
    (101, Some(0)),
    (105, Some(0)),
    (106, Some(0)),
    (107, Some(0)),
    (108, Some(0)),
    (109, Some(0)),
    (110, Some(0)),
    (111, Some(0)),
    (112, Some(0)),
    (113, Some(0)),
    (114, Some(0)),
    (115, Some(0)),
    (116, Some(0)),
];

/// Compute subclass IDs from [`SUBCLASS_RANGE_HIERARCHY`] and write every
/// supplied PyType alias via `assign_subclass_range`.
///
/// This mirrors RPython `TotalOrderSymbolic.compute_fn`
/// (`normalizecalls.py:302-354`): build each reversed-MRO witness, add its
/// Min and `witness + [MAX]` peers, lexicographically sort all peers, then
/// assign their 0-based `enumerate()` positions. The root Min peer is 0.
///
/// Pyre's interpreter-only paths (tests + `run_exec_frame`) skip the
/// JIT init that normally seeds ranges via `gc.subclass_range`, so
/// without this helper `ll_isinstance(obj, &EXCEPTION_TYPE)` returns
/// false (every range stays at the static `0` default). Callers must
/// invoke this once at startup before any `is_exception` /
/// `ll_isinstance` call (typically from `init_typeobjects` on the
/// interpreter side). The later GC writeback consumes the same hierarchy,
/// so either writer leaves byte-identical ranges.
pub fn compute_subclass_ranges_from(alias_chains: &[&[SubclassRangeAlias]]) {
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
    enum WitnessElement {
        Cdef(u32),
        Max,
    }

    let slots = SUBCLASS_RANGE_HIERARCHY
        .last()
        .map_or(0, |(type_id, _)| *type_id as usize + 1);
    let mut witnesses: Vec<Option<Vec<WitnessElement>>> = vec![None; slots];
    for &(type_id, parent) in SUBCLASS_RANGE_HIERARCHY {
        let mut witness = match parent {
            Some(parent_id) => witnesses[parent_id as usize]
                .clone()
                .expect("subclass-range parent must precede its child"),
            None => Vec::new(),
        };
        witness.push(WitnessElement::Cdef(type_id));
        witnesses[type_id as usize] = Some(witness);
    }

    #[derive(Clone)]
    struct Peer {
        witness: Vec<WitnessElement>,
        owner: u32,
        is_max: bool,
    }

    let mut peers = Vec::with_capacity(SUBCLASS_RANGE_HIERARCHY.len() * 2);
    for &(type_id, _) in SUBCLASS_RANGE_HIERARCHY {
        let witness = witnesses[type_id as usize]
            .as_ref()
            .expect("every subclass-range typeid must have a witness");
        peers.push(Peer {
            witness: witness.clone(),
            owner: type_id,
            is_max: false,
        });
        let mut max_witness = witness.clone();
        max_witness.push(WitnessElement::Max);
        peers.push(Peer {
            witness: max_witness,
            owner: type_id,
            is_max: true,
        });
    }
    peers.sort_by(|a, b| a.witness.cmp(&b.witness));

    let mut ranges = vec![(0, 0); slots];
    for (value, peer) in peers.iter().enumerate() {
        let range = &mut ranges[peer.owner as usize];
        if peer.is_max {
            range.1 = value as i64;
        } else {
            range.0 = value as i64;
        }
    }

    // Serialize against the JIT GC-tid writeback and publish the batch
    // atomically w.r.t. seqlock readers so none observes a half-renumbered
    // hierarchy.
    let _range_guard = subclass_range_write_guard();
    for aliases in alias_chains {
        for alias in *aliases {
            let range = ranges
                .get(alias.type_id as usize)
                .copied()
                .expect("subclass-range alias typeid must be in the hierarchy");
            assert!(
                witnesses[alias.type_id as usize].is_some(),
                "subclass-range alias typeid must name an rclass.OBJECT node"
            );
            assign_subclass_range(alias.pytype, range.0, range.1);
        }
    }
}

/// Lazy first-caller-wins gate around `compute_subclass_ranges_from`.
/// Pyre's interpreter-side `init_typeobjects` passes both object and
/// interpreter alias slices so cross-crate types (e.g. `CODE_TYPE`,
/// `PYTRACEBACK_TYPE`) are written. Pyre-object's own tests can reach
/// `is_exception` without calling `init_typeobjects`, so this `OnceLock`
/// triggers a fallback write of the object-owned aliases. Both paths compute
/// from the complete shared hierarchy; only the set of PyType aliases written
/// differs. A later GC writeback is byte-identical.
static SUBCLASS_RANGES_INIT: OnceLock<()> = OnceLock::new();

// `dont_look_inside`: one-time host initialization (`OnceLock` +
// global type-table walk) stays opaque to the JIT — production
// entry points have run the full init before any trace executes,
// so the residual call is a no-op there.
#[majit_macros::dont_look_inside]
pub extern "C" fn ensure_object_subclass_ranges_initialized() {
    SUBCLASS_RANGES_INIT.get_or_init(|| {
        let aliases = all_subclass_range_aliases();
        compute_subclass_ranges_from(&[&aliases]);
    });
}

/// Marker called by full-init paths (interpreter `init_typeobjects`,
/// JIT init) after they've populated subclass ranges across the
/// complete pair set, so the lazy `ensure_object_subclass_ranges_
/// initialized` no-ops on subsequent calls instead of overwriting
/// with the object-only subset.
pub fn mark_subclass_ranges_initialized() {
    let _ = SUBCLASS_RANGES_INIT.set(());
}

/// Every built-in `PyType` static that represents a full `PyObject`
/// subtype (i.e. instances carry `ob_type` at offset 0, matching
/// `rclass.OBJECT` layout), paired with its parent class.
///
/// Modelled on RPython's `assign_inheritance_ids`
/// (normalizecalls.py:373-389) which walks `classdef.getmro()` to build
/// the reversed-MRO witness for each class. The JIT registers each
/// `(type, parent)` pair with the GC via `register_vtable_for_type`,
/// using the parent typeid as `TypeInfo::object_subclass`'s `parent`
/// argument so the resulting `subclassrange_{min,max}` faithfully
/// represents the `rclass.OBJECT` hierarchy. `GUARD_SUBCLASS` then
/// resolves to `int_between(cls.min, subcls.min, cls.max)` per
/// rclass.py:1133-1137 `ll_issubclass`.
///
/// `INSTANCE_TYPE` (the `name = "object"` root) is intentionally
/// absent: it is registered separately as the `rclass.OBJECT` root
/// with no parent. `INT_TYPE` and `FLOAT_TYPE` are also absent: they
/// get their own ids (`W_INT_GC_TYPE_ID` / `W_FLOAT_GC_TYPE_ID`)
/// because the JIT backend allocates W_IntObject / W_FloatObject
/// through NewWithVtable and needs the correct payload size.
pub fn all_foreign_pytypes() -> &'static [(&'static PyType, &'static PyType)] {
    static PYTYPES: &[(&PyType, &PyType)] = &[
        // bool inherits from int (objectobject.py W_BoolObject.typedef).
        (&BOOL_TYPE, &INT_TYPE),
        (&STR_TYPE, &INSTANCE_TYPE),
        (&LIST_TYPE, &INSTANCE_TYPE),
        (&TUPLE_TYPE, &INSTANCE_TYPE),
        (&DICT_TYPE, &INSTANCE_TYPE),
        // longobject.py W_LongObject — Python 3 unifies long under int,
        // but pyre carries a separate static for the BigInt-backed flavour.
        (&LONG_TYPE, &INSTANCE_TYPE),
        (&NONE_TYPE, &INSTANCE_TYPE),
        (&NOTIMPLEMENTED_TYPE, &INSTANCE_TYPE),
        (&ELLIPSIS_TYPE, &INSTANCE_TYPE),
        (&MODULE_TYPE, &INSTANCE_TYPE),
        (&MAPPING_PROXY_TYPE, &INSTANCE_TYPE),
        (&TYPE_TYPE, &INSTANCE_TYPE),
        (&crate::descriptor::SUPER_TYPE, &INSTANCE_TYPE),
        (&crate::bytearrayobject::BYTEARRAY_TYPE, &INSTANCE_TYPE),
        (&crate::bytesobject::BYTES_TYPE, &INSTANCE_TYPE),
        (&crate::generator::GENERATOR_TYPE, &INSTANCE_TYPE),
        (&crate::_pypy_generic_alias::UNION_TYPE, &INSTANCE_TYPE),
        (&crate::functional::RANGE_ITER_TYPE, &INSTANCE_TYPE),
        (&crate::iterobject::SEQ_ITER_TYPE, &INSTANCE_TYPE),
        (&crate::nestedscope::CELL_TYPE, &INSTANCE_TYPE),
        (&crate::function::METHOD_TYPE, &INSTANCE_TYPE),
        (&crate::descriptor::PROPERTY_TYPE, &INSTANCE_TYPE),
        (&crate::function::STATICMETHOD_TYPE, &INSTANCE_TYPE),
        (&crate::function::CLASSMETHOD_TYPE, &INSTANCE_TYPE),
        // Exception hierarchy: per-kind PyType statics chain to
        // `EXCEPTION_TYPE` (the BaseException root) so backend
        // `GuardClass` at `OB_TYPE_OFFSET` discriminates subclasses.
        // Order is topological — parent must register before child for
        // the `all_foreign_pytypes` loop in `pyre-jit/src/eval.rs` that
        // looks up `parent_tid` via `pytype_to_tid`.
        (&crate::interp_exceptions::EXCEPTION_TYPE, &INSTANCE_TYPE),
        (
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
            &crate::interp_exceptions::EXCEPTION_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_ARITHMETIC_ERROR_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_OVERFLOW_ERROR_TYPE,
            &crate::interp_exceptions::EXC_ARITHMETIC_ERROR_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_ZERO_DIVISION_ERROR_TYPE,
            &crate::interp_exceptions::EXC_ARITHMETIC_ERROR_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_TYPE_ERROR_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_VALUE_ERROR_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        // `W_SyntaxError` — direct subclass of Exception
        // (`compile`/`exec`/`eval`/`ast.parse` raise it).
        (
            &crate::interp_exceptions::EXC_SYNTAX_ERROR_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        // UnicodeError is the intermediate parent of UnicodeDecodeError
        // and UnicodeEncodeError per `pypy/module/exceptions/
        // interp_exceptions.py:418 W_UnicodeError = _new_exception(
        // 'UnicodeError', W_ValueError, ...)`.  Register before its
        // subclasses so the topological-order constraint of the
        // foreign-pytype loop in pyre-jit's eval init holds.
        (
            &crate::interp_exceptions::EXC_UNICODE_ERROR_TYPE,
            &crate::interp_exceptions::EXC_VALUE_ERROR_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_UNICODE_DECODE_ERROR_TYPE,
            &crate::interp_exceptions::EXC_UNICODE_ERROR_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_UNICODE_ENCODE_ERROR_TYPE,
            &crate::interp_exceptions::EXC_UNICODE_ERROR_TYPE,
        ),
        // `pypy/module/exceptions/interp_exceptions.py:426
        // W_UnicodeTranslateError = _new_exception('UnicodeTranslateError',
        // W_UnicodeError, ...)`.
        (
            &crate::interp_exceptions::EXC_UNICODE_TRANSLATE_ERROR_TYPE,
            &crate::interp_exceptions::EXC_UNICODE_ERROR_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_NAME_ERROR_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        // UnboundLocalError subclasses NameError; listed after it so the
        // topological-order constraint of the foreign-pytype loop holds.
        // Its GC tid is pre-registered to the shared `W_BaseException`
        // tid by the per-ExcKind loop in `pyre-jit/src/eval.rs`, so that
        // loop skips this entry; without the pre-registration it would
        // assign an undersized standalone `sizeof(PyObject)` tid and shift
        // every hardcoded post-loop GC tid.
        (
            &crate::interp_exceptions::EXC_UNBOUND_LOCAL_ERROR_TYPE,
            &crate::interp_exceptions::EXC_NAME_ERROR_TYPE,
        ),
        // LookupError is the intermediate parent of IndexError and
        // KeyError per `pypy/module/exceptions/interp_exceptions.py:474
        // W_LookupError = _new_exception('LookupError', W_Exception,
        // ...)`.  Register before its subclasses.
        (
            &crate::interp_exceptions::EXC_LOOKUP_ERROR_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_INDEX_ERROR_TYPE,
            &crate::interp_exceptions::EXC_LOOKUP_ERROR_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_KEY_ERROR_TYPE,
            &crate::interp_exceptions::EXC_LOOKUP_ERROR_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_ATTRIBUTE_ERROR_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_RUNTIME_ERROR_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_NOT_IMPLEMENTED_ERROR_TYPE,
            &crate::interp_exceptions::EXC_RUNTIME_ERROR_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_RECURSION_ERROR_TYPE,
            &crate::interp_exceptions::EXC_RUNTIME_ERROR_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_STOP_ITERATION_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_IMPORT_ERROR_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_MODULE_NOT_FOUND_ERROR_TYPE,
            &crate::interp_exceptions::EXC_IMPORT_ERROR_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_ASSERTION_ERROR_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_REFERENCE_ERROR_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_OS_ERROR_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_FILE_NOT_FOUND_ERROR_TYPE,
            &crate::interp_exceptions::EXC_OS_ERROR_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_MEMORY_ERROR_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_SYSTEM_ERROR_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_GENERATOR_EXIT_TYPE,
            &crate::interp_exceptions::EXCEPTION_TYPE,
        ),
        (
            &crate::interp_exceptions::EXC_SYSTEM_EXIT_TYPE,
            &crate::interp_exceptions::EXCEPTION_TYPE,
        ),
        (&crate::sliceobject::SLICE_TYPE, &INSTANCE_TYPE),
        (&crate::setobject::SET_TYPE, &INSTANCE_TYPE),
        (&crate::setobject::FROZENSET_TYPE, &INSTANCE_TYPE),
        (&crate::typedef::MEMBER_TYPE, &INSTANCE_TYPE),
        // `pypy/objspace/std/dictmultiobject.py:449/459/469` —
        // dict_keys / dict_values / dict_items.  The three Python
        // visible types share the `W_DictViewObject` payload but each
        // gets a distinct W_TypeObject so `type(d.keys()) is
        // dict_keys` parity holds.
        (&crate::dictmultiobject::DICT_KEYS_TYPE, &INSTANCE_TYPE),
        (&crate::dictmultiobject::DICT_VALUES_TYPE, &INSTANCE_TYPE),
        (&crate::dictmultiobject::DICT_ITEMS_TYPE, &INSTANCE_TYPE),
        // `pypy/interpreter/typedef.py:444 GetSetProperty.typedef`.
        // Registered in the foreign-pytype loop so the `instantiate`
        // back-pointer is set before the first GetSetProperty
        // allocation runs (typedef.rs::getset_descriptor_type forces
        // it for the W_TypeObject side, but the static PyType also
        // needs the foreign-loop entry to seed pytype_to_tid for the
        // GC vtable lookup).
        (&crate::typedef::GETSET_DESCRIPTOR_TYPE, &INSTANCE_TYPE),
        // Appended at the TAIL: inserting mid-list would shift the
        // positionally-assigned type ids of every following entry,
        // silently breaking GuardClass / pytype_to_tid lookups.  The
        // parent `EXC_EXCEPTION_TYPE` is registered far earlier, so the
        // topological constraint still holds at the end.
        (
            &crate::interp_exceptions::EXC_BUFFER_ERROR_TYPE,
            &crate::interp_exceptions::EXC_EXCEPTION_TYPE,
        ),
    ];
    PYTYPES
}

/// PyType aliases owned by `pyre-object`, keyed by the GC typeid whose
/// inheritance peer supplies their range. Interpreter-owned aliases are
/// appended by `pyre-interpreter::all_subclass_range_aliases`.
pub fn all_subclass_range_aliases() -> Vec<SubclassRangeAlias> {
    use crate::lltype::PyreClassPyTypeOf;

    fn typed<T: PyreClassPyTypeOf>() -> &'static PyType {
        // Every `#[pyre_class]` descriptor points at its macro-emitted static
        // PyType for the program lifetime.
        unsafe { &*T::PYTYPE }
    }

    vec![
        subclass_range_alias(0, &INSTANCE_TYPE),
        subclass_range_alias(1, &INT_TYPE),
        subclass_range_alias(2, &FLOAT_TYPE),
        subclass_range_alias(5, &BOOL_TYPE),
        subclass_range_alias(6, &crate::functional::RANGE_ITER_TYPE),
        subclass_range_alias(7, &LIST_TYPE),
        subclass_range_alias(8, &TUPLE_TYPE),
        subclass_range_alias(15, &crate::nestedscope::CELL_TYPE),
        subclass_range_alias(16, &crate::function::METHOD_TYPE),
        subclass_range_alias(17, &crate::sliceobject::SLICE_TYPE),
        subclass_range_alias(18, &crate::descriptor::SUPER_TYPE),
        subclass_range_alias(19, &crate::descriptor::PROPERTY_TYPE),
        subclass_range_alias(20, &crate::function::STATICMETHOD_TYPE),
        subclass_range_alias(21, &crate::function::CLASSMETHOD_TYPE),
        subclass_range_alias(22, &crate::_pypy_generic_alias::UNION_TYPE),
        subclass_range_alias(23, &crate::iterobject::SEQ_ITER_TYPE),
        subclass_range_alias(24, typed::<crate::interp_itertools::W_Count>()),
        subclass_range_alias(25, typed::<crate::interp_itertools::W_Repeat>()),
        subclass_range_alias(26, &crate::typedef::MEMBER_TYPE),
        subclass_range_alias(27, &crate::bytesobject::BYTES_TYPE),
        subclass_range_alias(28, &crate::bytearrayobject::BYTEARRAY_TYPE),
        subclass_range_alias(29, &DICT_TYPE),
        subclass_range_alias(30, &crate::setobject::SET_TYPE),
        subclass_range_alias(30, &crate::setobject::FROZENSET_TYPE),
        subclass_range_alias(31, &crate::interp_exceptions::EXCEPTION_TYPE),
        subclass_range_alias(31, &crate::interp_exceptions::EXC_SYNTAX_ERROR_TYPE),
        subclass_range_alias(
            31,
            &crate::interp_exceptions::EXC_MODULE_NOT_FOUND_ERROR_TYPE,
        ),
        subclass_range_alias(31, &crate::interp_exceptions::EXC_UNBOUND_LOCAL_ERROR_TYPE),
        subclass_range_alias(31, &crate::interp_exceptions::EXC_BUFFER_ERROR_TYPE),
        subclass_range_alias(32, &crate::generator::GENERATOR_TYPE),
        subclass_range_alias(33, &TYPE_TYPE),
        subclass_range_alias(34, &STR_TYPE),
        subclass_range_alias(35, &LONG_TYPE),
        subclass_range_alias(36, &MODULE_TYPE),
        subclass_range_alias(38, &MAPPING_PROXY_TYPE),
        subclass_range_alias(39, &crate::dictmultiobject::DICT_KEYS_TYPE),
        subclass_range_alias(39, &crate::dictmultiobject::DICT_VALUES_TYPE),
        subclass_range_alias(39, &crate::dictmultiobject::DICT_ITEMS_TYPE),
        subclass_range_alias(40, &crate::typedef::GETSET_DESCRIPTOR_TYPE),
        subclass_range_alias(45, &NONE_TYPE),
        subclass_range_alias(46, &NOTIMPLEMENTED_TYPE),
        subclass_range_alias(47, &ELLIPSIS_TYPE),
        subclass_range_alias(48, &crate::dictmultiobject::MODULE_DICT_TYPE),
        subclass_range_alias(49, &crate::celldict::OBJECT_MUTABLE_CELL_TYPE),
        subclass_range_alias(50, &crate::celldict::INT_MUTABLE_CELL_TYPE),
        subclass_range_alias(52, &crate::weakref::GC_WEAKREF_BOX_TYPE),
        subclass_range_alias(54, &COMPLEX_TYPE),
        subclass_range_alias(57, &crate::interp_exceptions::EXC_EXCEPTION_TYPE),
        subclass_range_alias(58, &crate::interp_exceptions::EXC_SYSTEM_EXIT_TYPE),
        subclass_range_alias(59, &crate::interp_exceptions::EXC_GENERATOR_EXIT_TYPE),
        subclass_range_alias(60, &crate::interp_exceptions::EXC_ARITHMETIC_ERROR_TYPE),
        subclass_range_alias(61, &crate::interp_exceptions::EXC_OVERFLOW_ERROR_TYPE),
        subclass_range_alias(62, &crate::interp_exceptions::EXC_ZERO_DIVISION_ERROR_TYPE),
        subclass_range_alias(63, &crate::interp_exceptions::EXC_TYPE_ERROR_TYPE),
        subclass_range_alias(64, &crate::interp_exceptions::EXC_VALUE_ERROR_TYPE),
        subclass_range_alias(65, &crate::interp_exceptions::EXC_UNICODE_ERROR_TYPE),
        subclass_range_alias(66, &crate::interp_exceptions::EXC_UNICODE_DECODE_ERROR_TYPE),
        subclass_range_alias(67, &crate::interp_exceptions::EXC_UNICODE_ENCODE_ERROR_TYPE),
        subclass_range_alias(
            68,
            &crate::interp_exceptions::EXC_UNICODE_TRANSLATE_ERROR_TYPE,
        ),
        subclass_range_alias(69, &crate::interp_exceptions::EXC_NAME_ERROR_TYPE),
        subclass_range_alias(70, &crate::interp_exceptions::EXC_LOOKUP_ERROR_TYPE),
        subclass_range_alias(71, &crate::interp_exceptions::EXC_INDEX_ERROR_TYPE),
        subclass_range_alias(72, &crate::interp_exceptions::EXC_KEY_ERROR_TYPE),
        subclass_range_alias(73, &crate::interp_exceptions::EXC_ATTRIBUTE_ERROR_TYPE),
        subclass_range_alias(74, &crate::interp_exceptions::EXC_RUNTIME_ERROR_TYPE),
        subclass_range_alias(
            75,
            &crate::interp_exceptions::EXC_NOT_IMPLEMENTED_ERROR_TYPE,
        ),
        subclass_range_alias(76, &crate::interp_exceptions::EXC_RECURSION_ERROR_TYPE),
        subclass_range_alias(77, &crate::interp_exceptions::EXC_STOP_ITERATION_TYPE),
        subclass_range_alias(78, &crate::interp_exceptions::EXC_IMPORT_ERROR_TYPE),
        subclass_range_alias(79, &crate::interp_exceptions::EXC_ASSERTION_ERROR_TYPE),
        subclass_range_alias(80, &crate::interp_exceptions::EXC_REFERENCE_ERROR_TYPE),
        subclass_range_alias(81, &crate::interp_exceptions::EXC_OS_ERROR_TYPE),
        subclass_range_alias(82, &crate::interp_exceptions::EXC_FILE_NOT_FOUND_ERROR_TYPE),
        subclass_range_alias(83, &crate::interp_exceptions::EXC_MEMORY_ERROR_TYPE),
        subclass_range_alias(84, &crate::interp_exceptions::EXC_SYSTEM_ERROR_TYPE),
        subclass_range_alias(85, typed::<crate::interp_sre::W_SRE_Pattern>()),
        subclass_range_alias(86, typed::<crate::interp_sre::W_SRE_Match>()),
        subclass_range_alias(87, typed::<crate::interp_sre::W_SRE_Scanner>()),
        subclass_range_alias(88, typed::<crate::_pypy_generic_alias::GenericAlias>()),
        subclass_range_alias(94, typed::<crate::functional::W_ReversedIterator>()),
        subclass_range_alias(95, typed::<crate::functional::W_Filter>()),
        subclass_range_alias(96, typed::<crate::functional::W_Map>()),
        subclass_range_alias(97, typed::<crate::functional::W_Zip>()),
        subclass_range_alias(98, typed::<crate::interp_itertools::W_Cycle>()),
        subclass_range_alias(99, typed::<crate::interp_array::W_Array>()),
        subclass_range_alias(100, typed::<crate::interp_itertools::W_Chain>()),
        subclass_range_alias(101, typed::<crate::memoryview::W_MemoryView>()),
        subclass_range_alias(105, typed::<crate::setobject::W_SetIterObject>()),
        subclass_range_alias(106, typed::<crate::iterobject::W_ListIterObject>()),
        subclass_range_alias(107, typed::<crate::iterobject::W_ListReverseIterObject>()),
        subclass_range_alias(108, typed::<crate::iterobject::W_TupleIterObject>()),
        subclass_range_alias(109, typed::<crate::interp_itertools::W_Compress>()),
        subclass_range_alias(110, typed::<crate::interp_itertools::W_StarMap>()),
        subclass_range_alias(111, typed::<crate::interp_itertools::W_Accumulate>()),
        subclass_range_alias(112, typed::<crate::interp_itertools::W_ZipLongest>()),
        subclass_range_alias(113, &crate::generator::COROUTINE_TYPE),
        subclass_range_alias(114, typed::<crate::generator::CoroutineWrapper>()),
        subclass_range_alias(115, &crate::dictmultiobject::DICT_KEYITERATOR_TYPE),
        subclass_range_alias(115, &crate::dictmultiobject::DICT_VALUEITERATOR_TYPE),
        subclass_range_alias(115, &crate::dictmultiobject::DICT_ITEMITERATOR_TYPE),
        subclass_range_alias(115, &crate::dictmultiobject::DICT_REVERSEKEYITERATOR_TYPE),
        subclass_range_alias(115, &crate::dictmultiobject::DICT_REVERSEVALUEITERATOR_TYPE),
        subclass_range_alias(115, &crate::dictmultiobject::DICT_REVERSEITEMITERATOR_TYPE),
    ]
}

// ── Type checks ───────────────────────────────────────────────────────

/// Type name of any object, tag-safe. A tagged immediate is an exact `int`;
/// name it without derefing its (non-pointer) tagged bits as `ob_type`.
/// Gated on `CAN_BE_TAGGED` (folds to the raw `ob_type` deref at flag-false →
/// byte-identical). The chokepoint for the "must be X, not <name>" error
/// messages that a tagged int reaches after the tag-safe type probes reject
/// it. The else arm keeps the RAW `(*(*obj).ob_type).name` (NOT `r#type`,
/// which returns the `w_class` subclass name).
///
/// # Safety
/// `obj` must be a valid pointer to a `PyObject` unless it is a tagged
/// immediate.
#[inline]
pub unsafe fn type_name_of(obj: PyObjectRef) -> &'static str {
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        "int"
    } else {
        unsafe { (*(*obj).ob_type).name }
    }
}

/// Check if an object is of a given type (pointer identity comparison).
///
/// # Safety
/// `obj` must be a valid, non-null pointer to a `PyObject`.
#[inline]
pub unsafe fn py_type_check(obj: PyObjectRef, tp: &PyType) -> bool {
    // A tagged immediate's type is `int`, synthesized before the `ob_type`
    // deref: it matches iff `tp` is the `int` vtable. Gated on
    // `CAN_BE_TAGGED` (default false), so the deref below is the only live
    // path until enablement. This is the shared chokepoint for
    // `is_bool`/`is_float`/`is_long`/`is_complex`, which inherit the guard.
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        return std::ptr::eq(tp as *const PyType, &INT_TYPE as *const PyType);
    }
    !obj.is_null() && unsafe { std::ptr::eq((*obj).ob_type, tp as *const PyType) }
}

#[inline]
pub unsafe fn is_int(obj: PyObjectRef) -> bool {
    // A tagged immediate is a plain `int` (never a `bool`: bools are
    // even-aligned singletons). `is_int` reaches `ob_type` via
    // `py_type_check`, which derefs directly — so it carries its own
    // tag short-circuit rather than routing through `ll_type`. Gated on
    // the `CAN_BE_TAGGED` static (default false), inspecting only the
    // pointer bits, so the deref path below is the only live one.
    if crate::tagged_int::CAN_BE_TAGGED && crate::tagged_int::is_tagged_int(obj) {
        return true;
    }
    unsafe { py_type_check(obj, &INT_TYPE) || py_type_check(obj, &BOOL_TYPE) }
}

#[inline]
pub unsafe fn is_bool(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &BOOL_TYPE) }
}

#[inline]
pub unsafe fn is_float(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &FLOAT_TYPE) }
}

#[inline]
pub unsafe fn is_complex(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &COMPLEX_TYPE) }
}

#[inline]
pub unsafe fn is_long(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &LONG_TYPE) }
}

#[inline]
pub unsafe fn is_int_or_long(obj: PyObjectRef) -> bool {
    unsafe { is_int(obj) || is_long(obj) }
}

#[inline]
pub unsafe fn is_list(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &LIST_TYPE) }
}

/// Recognise any of the four tuple variants —
/// canonical `W_TupleObject` plus the three `W_SpecialisedTupleObject_*`
/// arity-2 specialisations from
/// `pypy/objspace/std/specialisedtupleobject.py`. All four share the
/// same Python `tuple` typedef in pypy; pyre encodes that by giving
/// each variant a distinct `ob_type` (RPython-vtable equivalent) while
/// `w_class` always resolves to the canonical `tuple` class object.
#[inline]
pub unsafe fn is_tuple(obj: PyObjectRef) -> bool {
    use crate::specialisedtupleobject::{
        SPECIALISED_TUPLE_FF_TYPE, SPECIALISED_TUPLE_II_TYPE, SPECIALISED_TUPLE_OO_TYPE,
    };
    unsafe {
        py_type_check(obj, &TUPLE_TYPE)
            || py_type_check(obj, &SPECIALISED_TUPLE_II_TYPE)
            || py_type_check(obj, &SPECIALISED_TUPLE_FF_TYPE)
            || py_type_check(obj, &SPECIALISED_TUPLE_OO_TYPE)
    }
}

/// `PyTuple_CheckExact` — an exact `tuple`, excluding tuple subclasses.
/// Covers the specialised arity-2 variants too: they all carry
/// `w_class == get_instantiate(&TUPLE_TYPE)`, so comparing the user-visible
/// class object (not `get_instantiate(ob_type)`) keeps them exact while a
/// subclass instance (retagged `w_class`) reads as non-exact.
#[inline]
pub unsafe fn is_exact_tuple(obj: PyObjectRef) -> bool {
    unsafe { is_tuple(obj) && std::ptr::eq((*obj).w_class, get_instantiate(&TUPLE_TYPE)) }
}

/// `PyList_CheckExact` — an exact `list`, excluding list subclasses.
#[inline]
pub unsafe fn is_exact_list(obj: PyObjectRef) -> bool {
    unsafe { is_list(obj) && std::ptr::eq((*obj).w_class, get_instantiate(&LIST_TYPE)) }
}

/// `pypy/objspace/std/dictmultiobject.py` makes both `W_DictObject` and
/// `W_ModuleDictObject` subclasses of `W_DictMultiObject`, so user-level
/// `isinstance(obj, dict)` is true for both.  Pyre exposes each layout
/// behind a distinct static `PyType` tag (so the Rust runtime can pick
/// the right cast), but `is_dict` reports the user-visible answer and
/// returns true for either.
#[inline]
pub unsafe fn is_dict(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &DICT_TYPE) || crate::dictmultiobject::is_module_dict(obj) }
}

#[inline]
pub unsafe fn is_none(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &NONE_TYPE) }
}

#[inline]
pub unsafe fn is_not_implemented(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &NOTIMPLEMENTED_TYPE) }
}

#[inline]
pub unsafe fn is_ellipsis(obj: PyObjectRef) -> bool {
    unsafe { py_type_check(obj, &ELLIPSIS_TYPE) }
}
