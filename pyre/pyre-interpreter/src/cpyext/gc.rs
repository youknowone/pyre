//! The cyclic-collection protocol -- `tp_traverse`, `tp_clear`, and the set of
//! blocks that declare them.
//!
//! A mirror's count above [`REFCNT_FROM_PYRE`] is what tells the collector that
//! C still holds the object, and that rule alone cannot see a cycle: a block
//! reachable only from another dead block still carries its reference, so both
//! stay. Breaking one needs to know which references a block holds, which is
//! what `tp_traverse` reports. Nothing upstream does this -- neither pypy's
//! `rawrefcount` nor its `cpyext` ever reads the slot -- so the shape here
//! follows CPython's `gcmodule.c` rather than a pypy file.
//!
//! This module is the reporting half: which blocks participate, and what each
//! one references. Deciding who dies is the collector's.

use super::pyobject::CPyObject;
use super::typeobject::{CPyTypeObject, PY_TPFLAGS_HAVE_GC};
use std::ffi::{c_int, c_void};

/// The blocks the collector may ask about.
///
/// CPython keeps its tracked objects in generation lists; the equivalent
/// question here is only ever "is this block one of them", so a set of
/// addresses is enough. An address is stable for a block's life, and
/// [`forget`] is called from `dealloc` before the block is released.
type TrackedSet =
    std::collections::HashSet<usize, std::hash::BuildHasherDefault<std::hash::DefaultHasher>>;
static TRACKED: super::ForkMutex<TrackedSet> =
    super::ForkMutex::new(TrackedSet::with_hasher(std::hash::BuildHasherDefault::new()));

pub(super) unsafe fn after_fork_child() {
    unsafe { TRACKED.reinit_after_fork() };
}

/// `true` when `tp` declares the cyclic-collection protocol.
pub(super) fn has_gc(tp: *mut CPyTypeObject) -> bool {
    !tp.is_null() && unsafe { (*tp).tp_flags } & PY_TPFLAGS_HAVE_GC != 0
}

/// Enter a block in the tracked set, if its type asked to be collected.
///
/// `PyType_GenericAlloc` calls this for the same reason CPython's does: an
/// instance of a `Py_TPFLAGS_HAVE_GC` type is tracked from the moment
/// `tp_alloc` returns it, and only a type allocating its own storage has to
/// call `PyObject_GC_Track` by hand.
pub(super) fn track(raw: *mut CPyObject) {
    if raw.is_null() {
        return;
    }
    if !has_gc(unsafe { (*raw).ob_type }) {
        return;
    }
    TRACKED.lock().insert(raw as usize);
}

/// Drop a block from the tracked set -- `PyObject_GC_UnTrack`, and the release
/// `dealloc` performs for a block that never untracked itself.
pub(super) fn forget(raw: usize) {
    TRACKED.lock().remove(&raw);
}

pub(super) fn is_tracked(raw: *mut CPyObject) -> bool {
    !raw.is_null() && TRACKED.lock().contains(&(raw as usize))
}

/// Every tracked block, as addresses.
///
/// A copy rather than a guard: the collector calls `tp_traverse` while walking
/// this, and an extension's traverse may reach code that tracks or untracks.
pub(super) fn tracked_blocks() -> Vec<usize> {
    TRACKED.lock().iter().copied().collect()
}

/// `tp_traverse`, read straight off the block's type.
///
/// The same single read `tp_dealloc_of` documents: `PyType_Ready`'s
/// `inherit_slots` has already copied a base's slot onto the subtype.
fn tp_traverse_of(raw: *mut CPyObject) -> Option<*const c_void> {
    let tp = unsafe { (*raw).ob_type };
    if !has_gc(tp) {
        return None;
    }
    let slot = unsafe { (*tp).tp_traverse };
    if slot.is_null() { None } else { Some(slot) }
}

fn tp_clear_of(raw: *mut CPyObject) -> Option<*const c_void> {
    let tp = unsafe { (*raw).ob_type };
    if !has_gc(tp) {
        return None;
    }
    let slot = unsafe { (*tp).tp_clear };
    if slot.is_null() { None } else { Some(slot) }
}

/// What a `visitproc` is handed as its opaque argument: the Rust side's
/// receiver for each reported reference.
struct Visit<'a> {
    report: &'a mut dyn FnMut(*mut CPyObject),
}

/// The `visitproc` an extension's `tp_traverse` calls once per reference it
/// holds. Answering non-zero would stop the walk, and this one never wants to.
unsafe extern "C" fn visit(object: *mut CPyObject, arg: *mut c_void) -> c_int {
    if object.is_null() || arg.is_null() {
        return 0;
    }
    let visit = unsafe { &mut *(arg as *mut Visit) };
    (visit.report)(object);
    0
}

/// Report every reference `raw`'s block holds, through its `tp_traverse`.
///
/// Returns `false` when the block declares no traverse, which is the case that
/// leaves the collector unable to judge it.
///
/// # Safety
/// `raw` must be a live block, and `report` must not allocate: an extension's
/// `tp_traverse` runs in between, and it is written on the assumption that a
/// collection cannot start underneath it.
pub(super) unsafe fn references(
    raw: *mut CPyObject,
    report: &mut dyn FnMut(*mut CPyObject),
) -> bool {
    let Some(slot) = tp_traverse_of(raw) else {
        return false;
    };
    let call: unsafe extern "C" fn(
        *mut CPyObject,
        unsafe extern "C" fn(*mut CPyObject, *mut c_void) -> c_int,
        *mut c_void,
    ) -> c_int = unsafe { std::mem::transmute(slot) };
    let mut receiver = Visit { report };
    unsafe { call(raw, visit, &raw mut receiver as *mut c_void) };
    true
}

/// Run `raw`'s `tp_clear`, which drops the references it reported.
///
/// # Safety
/// `raw` must be a live block.
pub(super) unsafe fn clear(raw: *mut CPyObject) -> bool {
    let Some(slot) = tp_clear_of(raw) else {
        return false;
    };
    let call: unsafe extern "C" fn(*mut CPyObject) -> c_int = unsafe { std::mem::transmute(slot) };
    unsafe { call(raw) };
    true
}

// ── the entry points an extension calls ─────────────────────────────────

/// `PyObject_GC_Track(object)` — start collecting `object`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_GC_Track(object: *mut CPyObject) {
    track(object);
}

/// `PyObject_GC_UnTrack(object)` — stop collecting `object`.
///
/// A `tp_dealloc` opens with this, so it has to tolerate a block that was never
/// tracked and one that is being torn down.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyObject_GC_UnTrack(object: *mut CPyObject) {
    if !object.is_null() {
        forget(object as usize);
    }
}

// `PyObject_GC_IsTracked` deliberately stays where it is, answering the
// constant 1 (`object.py:495-498`): this set holds the blocks of C-defined
// `Py_TPFLAGS_HAVE_GC` types, and every other object pyre holds is reachable by
// its collector too. Answering from this set would report 0 for a module or a
// list, which are tracked.

pub(super) fn ensure_linked() {
    std::hint::black_box(PyObject_GC_Track as *const ());
    std::hint::black_box(PyObject_GC_UnTrack as *const ());
}
