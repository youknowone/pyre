//! Raw `PyObject *` mirrors and their links -- PyPy `cpyext/pyobject.py`.
//!
//! A C extension never sees a [`pyre_object::PyObject`].  It sees a mirror at a
//! fixed address carrying a reference count and an `ob_pyre_link` to the moving
//! interpreter object, which is the `rawrefcount` P-link
//! (`rpython/rlib/rawrefcount.py`).
//!
//! # Ownership, and where it differs from upstream
//!
//! `rawrefcount` lets a mirror outlive the interpreter object: a mirror whose
//! count is exactly the link share is *not* a root, so the collector may free
//! the linked object and queue the mirror for deallocation.  Pyre's collector
//! has no such queue yet, so a mirror lives exactly as long as the C side holds
//! at least one reference and the link is a strong root for that whole time.
//! The consequence is that a reference cycle running through C leaks; the
//! upstream dead-queue is what the later slice has to add.

use super::ForkMutex;
use pyre_object::{PY_NULL, PyObjectRef};
use std::collections::HashMap;
use std::ffi::c_char;
use std::hash::BuildHasherDefault;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};

/// The large ownership contribution held by the linked pyre object.
///
/// PyPy calls this `rawrefcount.REFCNT_FROM_PYPY`.  Ordinary C references are
/// added above it by the public `Py_INCREF`/`Py_DECREF` inline functions.
pub const REFCNT_FROM_PYRE: isize = 1 << (isize::BITS - 3);

/// The count an immortal mirror starts at.
///
/// Type mirrors and the singletons are borrowed by every consumer and never
/// handed out with an owning reference, so their count must never fall back to
/// [`REFCNT_FROM_PYRE`] and free them.  The second large constant leaves the
/// same headroom in both directions, so no realistic incref/decref imbalance
/// can reach the deallocation threshold.
pub const REFCNT_IMMORTAL: isize = REFCNT_FROM_PYRE + (1 << (isize::BITS - 4));

/// C-visible `PyObject`, matching PyPy's `parse/cpyext_object.h` shape.
#[repr(C)]
pub struct CPyObject {
    pub ob_refcnt: isize,
    pub ob_pyre_link: PyObjectRef,
    pub ob_type: *mut CPyTypeObject,
}

/// A type mirror.
///
/// Opaque in the public header, but its first field is an ordinary mirror
/// header so `Py_TYPE(x)` is usable as a `PyObject *` — the same relationship
/// `PyTypeObject`'s leading `PyObject_HEAD` gives upstream.  The named
/// `PyXxx_Type` statics that let C compare against a specific builtin type
/// arrive with C-defined types; until then a type mirror is reachable only
/// through `Py_TYPE`, which is enough to compare two objects' types.
#[repr(C)]
pub struct CPyTypeObject {
    pub ob_base: CPyObject,
}

/// What a mirror actually is.
///
/// C only ever sees the leading [`CPyObject`], which is why the two are
/// separate types: everything past it is pyre-side bookkeeping.  The byte cache
/// is the counterpart of the `c_utf8` field PyPy keeps on its `PyUnicodeObject`
/// mirror — `PyUnicode_AsUTF8` and `PyBytes_AsString` must hand out a stable,
/// NUL-terminated address, and the interpreter's own storage is neither
/// NUL-terminated nor at a fixed address.
#[repr(C)]
pub struct Mirror {
    pub base: CPyObject,
    cached_bytes: AtomicPtr<u8>,
    cached_len: AtomicUsize,
}

impl Mirror {
    const fn immortal() -> Self {
        Self {
            base: CPyObject {
                ob_refcnt: REFCNT_IMMORTAL,
                ob_pyre_link: PY_NULL,
                ob_type: std::ptr::null_mut(),
            },
            cached_bytes: AtomicPtr::new(std::ptr::null_mut()),
            cached_len: AtomicUsize::new(0),
        }
    }
}

/// Sentinel type for a `PyModuleDef`, which is C static storage rather than a
/// mirror of an interpreter object: its `ob_pyre_link` stays null and it is
/// never entered in the census below.
pub static mut CPY_MODULE_DEF_TYPE: CPyTypeObject = CPyTypeObject {
    ob_base: CPyObject {
        ob_refcnt: REFCNT_IMMORTAL,
        ob_pyre_link: PY_NULL,
        ob_type: std::ptr::null_mut(),
    },
};

/// PyPy/rawrefcount's P-list analogue.  The relationship itself lives on the
/// raw object (`ob_pyre_link`); this Vec is only the collector's census of
/// mirrors whose inline link must be visited.
static RAW_OBJECTS: ForkMutex<Vec<usize>> = ForkMutex::new(Vec::new());
static CPYEXT_GC_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Interpreter object address -> mirror address, `rawrefcount`'s `w2r`
/// direction (`pyobject.py:make_ref` looks the mirror up before building one,
/// so `make_ref(w)` twice is the same pointer twice).
///
/// Keys are addresses of moving objects, so every collection invalidates the
/// whole table; [`LINKED_STALE`] records that and the table is rebuilt from the
/// census — whose links the collector has just forwarded — before the next
/// lookup.
type LinkMap = HashMap<usize, usize, BuildHasherDefault<std::hash::DefaultHasher>>;
static LINKED: ForkMutex<LinkMap> = ForkMutex::new(HashMap::with_hasher(BuildHasherDefault::new()));
static LINKED_STALE: AtomicBool = AtomicBool::new(false);

/// Re-key the identity table against the addresses the collector left behind.
///
/// Locking order is `LINKED` before `RAW_OBJECTS` everywhere, so the rebuild
/// cannot deadlock against a concurrent [`attach`] or [`dealloc`].
fn linked() -> parking_lot::MutexGuard<'static, LinkMap> {
    let mut map = LINKED.lock();
    if LINKED_STALE.swap(false, Ordering::Acquire) {
        map.clear();
        for &address in RAW_OBJECTS.lock().iter() {
            let raw = address as *mut CPyObject;
            let link = unsafe { (*raw).ob_pyre_link };
            if !link.is_null() {
                map.insert(link as usize, address);
            }
        }
    }
    map
}

pub(super) unsafe fn after_fork_child() {
    unsafe {
        RAW_OBJECTS.reinit_after_fork();
        LINKED.reinit_after_fork();
        BORROWED.reinit_after_fork();
    }
}

/// The mirror of an interpreter object's type, allocated once and immortal.
fn type_mirror(w_obj: PyObjectRef) -> *mut CPyTypeObject {
    let Some(w_type) = crate::typedef::r#type(w_obj) else {
        return std::ptr::null_mut();
    };
    let w_type = w_type.as_ptr();
    if let Some(&existing) = linked().get(&(w_type as usize)) {
        return existing as *mut CPyTypeObject;
    }
    // A type is an instance of its own metatype, so resolving `ob_type` before
    // the mirror exists would recurse forever on `type`; the second call finds
    // this mirror in the table and terminates.
    let mirror = attach(w_type, REFCNT_IMMORTAL, std::ptr::null_mut());
    let of_type = type_mirror(w_type);
    unsafe { (*mirror).ob_type = of_type };
    mirror as *mut CPyTypeObject
}

/// Enter a fresh mirror for `w_obj` into the census and the identity table.
fn attach(w_obj: PyObjectRef, refcnt: isize, ob_type: *mut CPyTypeObject) -> *mut CPyObject {
    let raw = Box::into_raw(Box::new(Mirror {
        base: CPyObject {
            ob_refcnt: refcnt,
            ob_pyre_link: w_obj,
            ob_type,
        },
        cached_bytes: AtomicPtr::new(std::ptr::null_mut()),
        cached_len: AtomicUsize::new(0),
    })) as *mut CPyObject;
    let mut map = linked();
    RAW_OBJECTS.lock().push(raw as usize);
    map.insert(w_obj as usize, raw as usize);
    CPYEXT_GC_ACTIVE.store(true, Ordering::Release);
    raw
}

/// `pyobject.py:make_ref` — a new reference to `w_obj`'s mirror.
pub fn make_ref(w_obj: PyObjectRef) -> *mut CPyObject {
    if w_obj.is_null() {
        return std::ptr::null_mut();
    }
    if let Some(&existing) = linked().get(&(w_obj as usize)) {
        let raw = existing as *mut CPyObject;
        unsafe { (*raw).ob_refcnt += 1 };
        return raw;
    }
    // The type mirror is resolved before the object's own mirror is entered so
    // the two `linked()` acquisitions never nest.
    let ob_type = type_mirror(w_obj);
    attach(w_obj, REFCNT_FROM_PYRE + 1, ob_type)
}

/// `pyobject.py:as_pyobj` — the mirror without taking a reference.
///
/// The result is only valid while some other owner keeps the mirror alive, so
/// this is for reading a mirror the caller already owns; use [`make_ref`] for
/// anything handed out to C.
pub fn as_pyobj(w_obj: PyObjectRef) -> *mut CPyObject {
    if w_obj.is_null() {
        return std::ptr::null_mut();
    }
    linked()
        .get(&(w_obj as usize))
        .map(|&address| address as *mut CPyObject)
        .unwrap_or(std::ptr::null_mut())
}

/// `pyobject.py:from_ref` — the interpreter object a mirror links to.
///
/// # Safety
/// `raw` must be null or a live mirror.  The result is only valid until the
/// next operation that can collect; re-read it through the mirror (or pin it
/// on the shadow stack) across anything that allocates.
pub unsafe fn from_ref(raw: *mut CPyObject) -> PyObjectRef {
    if raw.is_null() {
        return PY_NULL;
    }
    unsafe { (*raw).ob_pyre_link }
}

/// # Safety
/// `raw` must be null or a live mirror.
pub unsafe fn incref(raw: *mut CPyObject) {
    if !raw.is_null() {
        unsafe { (*raw).ob_refcnt += 1 };
    }
}

/// # Safety
/// `raw` must be null or a live mirror, and the caller must own the reference
/// being released.
pub unsafe fn decref(raw: *mut CPyObject) {
    if raw.is_null() {
        return;
    }
    unsafe { (*raw).ob_refcnt -= 1 };
    if unsafe { (*raw).ob_refcnt } <= REFCNT_FROM_PYRE {
        unsafe { dealloc(raw) };
    }
}

/// Drop a mirror the C side no longer references.
///
/// This is `_Py_Dealloc`'s job upstream.  There is no type-specific
/// deallocator to dispatch to yet: every mirror this slice hands out owns
/// nothing but its link and its byte cache.
///
/// # Safety
/// `raw` must be a live mirror whose count has fallen to the link share.
unsafe fn dealloc(raw: *mut CPyObject) {
    let address = raw as usize;
    // Before the mirror goes: a borrow it owns may be the last reference to
    // its own container, and that recursion has to run outside the locks below.
    release_borrowed(raw);
    let link = unsafe { (*raw).ob_pyre_link };
    let mut map = linked();
    RAW_OBJECTS.lock().retain(|&candidate| candidate != address);
    if map.get(&(link as usize)) == Some(&address) {
        map.remove(&(link as usize));
    }
    drop(map);
    unsafe {
        (*raw).ob_pyre_link = PY_NULL;
        (*raw).ob_refcnt = 0;
        let mirror = Box::from_raw(raw as *mut Mirror);
        let cached = mirror.cached_bytes.load(Ordering::Acquire);
        if !cached.is_null() {
            let len = mirror.cached_len.load(Ordering::Acquire);
            drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                cached, len,
            )));
        }
    }
}

/// References a container mirror owns on behalf of C.
///
/// `PyTuple_GetItem` and friends return a *borrowed* reference, which needs an
/// owner that outlives the call.  Upstream gives the container mirror an
/// `ob_item` array of owned references (`tupleobject.py:tuple_attach`); pyre
/// records the same ownership here, keyed by the container's mirror, and
/// releases it when that mirror is deallocated.
type BorrowMap = HashMap<usize, Vec<usize>, BuildHasherDefault<std::hash::DefaultHasher>>;
static BORROWED: ForkMutex<BorrowMap> =
    ForkMutex::new(HashMap::with_hasher(BuildHasherDefault::new()));

/// A borrowed reference to `w_item`, owned by `container`'s mirror.
///
/// The reference taken here is dropped again when the item already has an
/// entry, so repeatedly reading the same slot cannot grow the list.
pub(super) fn borrow_from(container: *mut CPyObject, w_item: PyObjectRef) -> *mut CPyObject {
    if w_item.is_null() {
        return std::ptr::null_mut();
    }
    let item = make_ref(w_item);
    if container.is_null() {
        // No owner to attach to: the reference stays outstanding, which is the
        // conservative choice — the alternative hands C a freed pointer.
        return item;
    }
    let mut borrowed = BORROWED.lock();
    let owned = borrowed.entry(container as usize).or_default();
    if owned.contains(&(item as usize)) {
        drop(borrowed);
        unsafe { decref(item) };
    } else {
        owned.push(item as usize);
    }
    item
}

/// Release everything a dying container mirror borrowed on C's behalf.
fn release_borrowed(container: *mut CPyObject) {
    let owned = BORROWED.lock().remove(&(container as usize));
    for item in owned.into_iter().flatten() {
        unsafe { decref(item as *mut CPyObject) };
    }
}

/// The mirror's NUL-terminated byte view, filled on first use.
///
/// The `bytes` the producer returns is the payload; the terminator is appended
/// here, so the address is usable as a C string whenever the payload has no
/// interior NUL, and `PyBytes_AsStringAndSize`-style callers get the length
/// separately either way.
///
/// # Safety
/// `raw` must be a live mirror.
pub(super) unsafe fn cached_bytes(
    raw: *mut CPyObject,
    produce: impl FnOnce() -> Vec<u8>,
) -> (*const c_char, usize) {
    let mirror = unsafe { &*(raw as *mut Mirror) };
    let existing = mirror.cached_bytes.load(Ordering::Acquire);
    if !existing.is_null() {
        return (
            existing as *const c_char,
            mirror.cached_len.load(Ordering::Acquire) - 1,
        );
    }
    let mut bytes = produce();
    let payload = bytes.len();
    bytes.push(0);
    let buffer = bytes.into_boxed_slice();
    let len = buffer.len();
    let pointer = Box::into_raw(buffer) as *mut u8;
    // The length is published before the pointer, so a reader that observes a
    // non-null pointer observes the matching length.
    mirror.cached_len.store(len, Ordering::Release);
    match mirror.cached_bytes.compare_exchange(
        std::ptr::null_mut(),
        pointer,
        Ordering::AcqRel,
        Ordering::Acquire,
    ) {
        Ok(_) => (pointer as *const c_char, payload),
        Err(winner) => {
            // Another thread filled the cache first; its buffer is the one
            // every caller must share, so this one is dropped again.
            drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(pointer, len)) });
            (
                winner as *const c_char,
                mirror.cached_len.load(Ordering::Acquire) - 1,
            )
        }
    }
}

/// Forward every mirror link.
///
/// External C ownership keeps the linked object alive, which is the P-link
/// rule from `rawrefcount.rst` — with the ownership divergence documented at
/// the top of this module: pyre roots the link for as long as the mirror
/// exists, and a mirror exists for exactly as long as C holds a reference.
pub fn walk_gc_roots(visitor: &mut dyn FnMut(&mut PyObjectRef)) {
    if !CPYEXT_GC_ACTIVE.load(Ordering::Acquire) {
        return;
    }
    for &address in RAW_OBJECTS.lock().iter() {
        let raw = address as *mut CPyObject;
        unsafe {
            if !(*raw).ob_pyre_link.is_null() {
                visitor(&mut (*raw).ob_pyre_link);
            }
        }
    }
    // Every key in the identity table is an address the visitor above was free
    // to relocate, so the table is rebuilt from the links themselves before the
    // next lookup rather than here — a collection must not allocate.
    LINKED_STALE.store(true, Ordering::Release);
}

// ── the immortal singletons ─────────────────────────────────────────────

/// `Py_None`. C compares against this pointer, so the mirror has to be the one
/// `make_ref(w_none())` returns; [`init_singletons`] enters it in the table.
#[unsafe(no_mangle)]
pub static mut _Py_NoneStruct: Mirror = Mirror::immortal();

/// `Py_True`.
#[unsafe(no_mangle)]
pub static mut _Py_TrueStruct: Mirror = Mirror::immortal();

/// `Py_False`.
#[unsafe(no_mangle)]
pub static mut _Py_FalseStruct: Mirror = Mirror::immortal();

/// `Py_NotImplemented`.
#[unsafe(no_mangle)]
pub static mut _Py_NotImplementedStruct: Mirror = Mirror::immortal();

/// `Py_Ellipsis`.
#[unsafe(no_mangle)]
pub static mut _Py_EllipsisObject: Mirror = Mirror::immortal();

/// Bind each preallocated singleton mirror to its interpreter object.
///
/// Called once before the first `PyInit_*`, so a C function that compares a
/// result against `Py_None` sees the same pointer `make_ref` hands out.
pub fn init_singletons() {
    let bound: [(*mut CPyObject, PyObjectRef); 5] = [
        (
            &raw mut _Py_NoneStruct as *mut CPyObject,
            pyre_object::w_none(),
        ),
        (
            &raw mut _Py_TrueStruct as *mut CPyObject,
            pyre_object::boolobject::w_bool_from(true),
        ),
        (
            &raw mut _Py_FalseStruct as *mut CPyObject,
            pyre_object::boolobject::w_bool_from(false),
        ),
        (
            &raw mut _Py_NotImplementedStruct as *mut CPyObject,
            pyre_object::w_not_implemented(),
        ),
        (
            &raw mut _Py_EllipsisObject as *mut CPyObject,
            pyre_object::w_ellipsis(),
        ),
    ];
    let mut map = linked();
    for (raw, w_obj) in bound {
        if unsafe { !(*raw).ob_pyre_link.is_null() } {
            continue;
        }
        unsafe { (*raw).ob_pyre_link = w_obj };
        RAW_OBJECTS.lock().push(raw as usize);
        map.insert(w_obj as usize, raw as usize);
    }
    CPYEXT_GC_ACTIVE.store(true, Ordering::Release);
    drop(map);
    // Deferred until the links are in the table: `type_mirror` takes the same
    // lock, and a singleton's type mirror is an ordinary immortal mirror.
    for (raw, _) in bound {
        if unsafe { (*raw).ob_type.is_null() } {
            let of_type = type_mirror(unsafe { (*raw).ob_pyre_link });
            unsafe { (*raw).ob_type = of_type };
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Py_IncRef(object: *mut CPyObject) {
    unsafe { incref(object) };
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn Py_DecRef(object: *mut CPyObject) {
    unsafe { decref(object) };
}

/// The C-visible reference count, with the interpreter's own share removed —
/// `Py_REFCNT` as a test can read it without depending on the link share.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn _PyPyre_RefCount(object: *mut CPyObject) -> isize {
    if object.is_null() {
        return 0;
    }
    unsafe { (*object).ob_refcnt - REFCNT_FROM_PYRE }
}

pub(super) fn ensure_linked() {
    std::hint::black_box(Py_IncRef as *const ());
    std::hint::black_box(Py_DecRef as *const ());
    std::hint::black_box(_PyPyre_RefCount as *const ());
}
