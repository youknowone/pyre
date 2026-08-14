//! Raw `PyObject *` mirrors and their links -- PyPy `cpyext/pyobject.py`.
//!
//! A C extension never sees a [`pyre_object::PyObject`].  It sees a mirror at a
//! fixed address carrying a reference count and an `ob_pyre_link` to the moving
//! interpreter object, which is the `rawrefcount` P-link
//! (`rpython/rlib/rawrefcount.py`).
//!
//! A mirror's block is as large as its type asks for: `sizeof(PyObject)` for an
//! ordinary object, `sizeof(PyTypeObject)` for a type, `tp_basicsize` for an
//! instance of a C-defined type.  That is upstream's `make_typedescr(basestruct
//! =...)`, and it is why the allocation is a byte block rather than a Rust
//! struct.
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
use super::typeobject::CPyTypeObject;
use pyre_object::{PY_NULL, PyObjectRef};
use std::alloc::Layout;
use std::collections::HashMap;
use std::ffi::c_char;
use std::hash::BuildHasherDefault;
use std::sync::atomic::{AtomicBool, Ordering};

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

/// The alignment every mirror block is allocated at.
///
/// A C-defined type's `tp_basicsize` covers fields the extension declared, so
/// the block has to satisfy the strictest alignment a C compiler would give a
/// `malloc` result — `max_align_t`, which is 16 on every target this feature
/// builds for.
const BLOCK_ALIGN: usize = 16;

fn block_layout(size: usize) -> Layout {
    Layout::from_size_align(size, BLOCK_ALIGN).expect("a mirror block size is a small constant")
}

/// PyPy/rawrefcount's P-list analogue.  The relationship itself lives on the
/// raw object (`ob_pyre_link`); this table is only the collector's census of
/// mirrors whose inline link must be visited, and it records how each block was
/// allocated: the byte size for one this layer owns, and 0 for a block it does
/// not own — a `static` in this crate or in the loaded extension.
type Census = HashMap<usize, usize, BuildHasherDefault<std::hash::DefaultHasher>>;
static RAW_OBJECTS: ForkMutex<Census> =
    ForkMutex::new(HashMap::with_hasher(BuildHasherDefault::new()));
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
        for &address in RAW_OBJECTS.lock().keys() {
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
        BYTE_CACHE.reinit_after_fork();
    }
}

/// The mirror of an interpreter object's type.
///
/// A type a C extension defined already has one — its own `PyTypeObject`
/// static, entered by `PyType_Ready` — and every other type gets an immortal
/// block of the same shape so that `Py_TYPE(x)->tp_name` reads something.
pub fn type_mirror(w_obj: PyObjectRef) -> *mut CPyTypeObject {
    match crate::typedef::r#type(w_obj) {
        Some(w_type) => ensure_mirror(w_type.as_ptr()) as *mut CPyTypeObject,
        None => std::ptr::null_mut(),
    }
}

/// `w_obj`'s mirror, allocated on first demand.
///
/// A type gets a `PyTypeObject`-shaped block by whichever route reaches it
/// first: `Py_TYPE(x)->tp_basicsize` is read off any type mirror, so a type may
/// never receive the plain `PyObject`-sized block a non-type receives — a
/// `PyModule_AddObject` of a class would otherwise decide the shape.
fn ensure_mirror(w_obj: PyObjectRef) -> *mut CPyObject {
    if let Some(&existing) = linked().get(&(w_obj as usize)) {
        return existing as *mut CPyObject;
    }
    if unsafe { pyre_object::is_type(w_obj) } {
        // A type is an instance of its own metatype, so resolving `ob_type`
        // before the mirror is entered would recurse forever on `type`; the
        // second call finds this mirror in the table and terminates.
        let mirror = attach(
            w_obj,
            REFCNT_IMMORTAL,
            std::ptr::null_mut(),
            size_of::<CPyTypeObject>(),
        ) as *mut CPyTypeObject;
        super::typeobject::describe_interpreter_type(mirror, w_obj);
        let of_type = type_mirror(w_obj);
        unsafe { (*mirror).ob_base.ob_base.ob_type = of_type };
        return mirror as *mut CPyObject;
    }
    // The type mirror is resolved before the object's own mirror is entered so
    // the two `linked()` acquisitions never nest.
    let ob_type = type_mirror(w_obj);
    attach(w_obj, REFCNT_FROM_PYRE, ob_type, mirror_size(ob_type))
}

/// Allocate a zero-filled mirror block of `size` bytes and enter it into the
/// census and the identity table.
pub(super) fn attach(
    w_obj: PyObjectRef,
    refcnt: isize,
    ob_type: *mut CPyTypeObject,
    size: usize,
) -> *mut CPyObject {
    let size = size.max(size_of::<CPyObject>());
    let raw = unsafe { std::alloc::alloc_zeroed(block_layout(size)) } as *mut CPyObject;
    if raw.is_null() {
        std::alloc::handle_alloc_error(block_layout(size));
    }
    unsafe {
        (*raw).ob_refcnt = refcnt;
        (*raw).ob_pyre_link = w_obj;
        (*raw).ob_type = ob_type;
    }
    enter(w_obj, raw, size);
    raw
}

/// Enter a block this layer did not allocate — a `static` here or in the loaded
/// extension — as `w_obj`'s mirror.  Its header must already be filled.
pub(super) fn attach_foreign(w_obj: PyObjectRef, raw: *mut CPyObject) {
    unsafe { (*raw).ob_pyre_link = w_obj };
    enter(w_obj, raw, 0);
}

fn enter(w_obj: PyObjectRef, raw: *mut CPyObject, size: usize) {
    let mut map = linked();
    RAW_OBJECTS.lock().insert(raw as usize, size);
    map.insert(w_obj as usize, raw as usize);
    CPYEXT_GC_ACTIVE.store(true, Ordering::Release);
}

/// How large a mirror of `ob_type` has to be.
///
/// Everything pyre itself defines is exactly a `PyObject`, which is what
/// [`super::typeobject::describe_interpreter_type`] leaves `tp_basicsize` as;
/// an instance of a C-defined type carries that type's declared fields.
fn mirror_size(ob_type: *mut CPyTypeObject) -> usize {
    if ob_type.is_null() {
        return size_of::<CPyObject>();
    }
    let declared = unsafe { (*ob_type).tp_basicsize };
    if declared <= 0 {
        return size_of::<CPyObject>();
    }
    (declared as usize).max(size_of::<CPyObject>())
}

/// `pyobject.py:make_ref` — a new reference to `w_obj`'s mirror.
pub fn make_ref(w_obj: PyObjectRef) -> *mut CPyObject {
    if w_obj.is_null() {
        return std::ptr::null_mut();
    }
    let raw = ensure_mirror(w_obj);
    unsafe { (*raw).ob_refcnt += 1 };
    raw
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
    let size = RAW_OBJECTS.lock().remove(&address).unwrap_or(0);
    if map.get(&(link as usize)) == Some(&address) {
        map.remove(&(link as usize));
    }
    drop(map);
    BYTE_CACHE.lock().remove(&address);
    unsafe {
        (*raw).ob_pyre_link = PY_NULL;
        (*raw).ob_refcnt = 0;
    }
    // A block this layer did not allocate is a `static` that outlives it.
    if size != 0 {
        unsafe { std::alloc::dealloc(raw as *mut u8, block_layout(size)) };
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

/// The NUL-terminated byte view of a mirror, filled on first use.
///
/// This is the counterpart of the `c_utf8` field PyPy fills on its
/// `PyUnicodeObject` mirror: `PyUnicode_AsUTF8` and `PyBytes_AsString` must
/// hand out a stable, NUL-terminated address, and the interpreter's own storage
/// is neither NUL-terminated nor at a fixed address.  It is a side table rather
/// than a field so that a mirror block is exactly what its type declares.
type ByteCache = HashMap<usize, Box<[u8]>, BuildHasherDefault<std::hash::DefaultHasher>>;
static BYTE_CACHE: ForkMutex<ByteCache> =
    ForkMutex::new(HashMap::with_hasher(BuildHasherDefault::new()));

/// The mirror's cached bytes and their length, without the terminator.
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
    let mut cache = BYTE_CACHE.lock();
    let entry = cache.entry(raw as usize).or_insert_with(|| {
        let mut bytes = produce();
        bytes.push(0);
        bytes.into_boxed_slice()
    });
    // The box owns its bytes, so the address stays put as the map rehashes.
    (entry.as_ptr() as *const c_char, entry.len() - 1)
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
    for &address in RAW_OBJECTS.lock().keys() {
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

const fn immortal() -> CPyObject {
    CPyObject {
        ob_refcnt: REFCNT_IMMORTAL,
        ob_pyre_link: PY_NULL,
        ob_type: std::ptr::null_mut(),
    }
}

/// `Py_None`. C compares against this pointer, so the mirror has to be the one
/// `make_ref(w_none())` returns; [`init_singletons`] enters it in the table.
#[unsafe(no_mangle)]
pub static mut _Py_NoneStruct: CPyObject = immortal();

/// `Py_True`.
#[unsafe(no_mangle)]
pub static mut _Py_TrueStruct: CPyObject = immortal();

/// `Py_False`.
#[unsafe(no_mangle)]
pub static mut _Py_FalseStruct: CPyObject = immortal();

/// `Py_NotImplemented`.
#[unsafe(no_mangle)]
pub static mut _Py_NotImplementedStruct: CPyObject = immortal();

/// `Py_Ellipsis`.
#[unsafe(no_mangle)]
pub static mut _Py_EllipsisObject: CPyObject = immortal();

/// Bind each preallocated singleton mirror to its interpreter object.
///
/// Called once before the first `PyInit_*`, so a C function that compares a
/// result against `Py_None` sees the same pointer `make_ref` hands out.
pub fn init_singletons() {
    let bound: [(*mut CPyObject, PyObjectRef); 5] = [
        (&raw mut _Py_NoneStruct, pyre_object::w_none()),
        (
            &raw mut _Py_TrueStruct,
            pyre_object::boolobject::w_bool_from(true),
        ),
        (
            &raw mut _Py_FalseStruct,
            pyre_object::boolobject::w_bool_from(false),
        ),
        (
            &raw mut _Py_NotImplementedStruct,
            pyre_object::w_not_implemented(),
        ),
        (&raw mut _Py_EllipsisObject, pyre_object::w_ellipsis()),
    ];
    for (raw, w_obj) in bound {
        if unsafe { !(*raw).ob_pyre_link.is_null() } {
            continue;
        }
        attach_foreign(w_obj, raw);
    }
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
