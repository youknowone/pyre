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
//! one references. Deciding who dies is the collector's -- and once it has,
//! [`clear_garbage`] runs the `tp_clear` that breaks the cycle apart, no other
//! layer being able to drop a reference that lives in a C field.
//!
//! [`c_edges`] is where every reference of that kind is collected, `tp_traverse`
//! being one of two sources; the other is
//! [`super::pyobject::borrowed_edges`].

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

/// The blocks whose `tp_finalize` has already run.
///
/// A collected type's finalizer runs at most once over the object's life, and
/// nothing in the block records that: `ob_pyre_link` is the interpreter
/// object and the header carries no spare bit here, so the answer is kept
/// beside the tracked set. An entry is cleared where the block is released
/// rather than in `dealloc`, which runs before `tp_dealloc` -- that is the
/// moment a second deallocation still has to read it -- and clearing at
/// release is also what keeps a recycled address from inheriting a stale one.
static FINALIZED: super::ForkMutex<TrackedSet> =
    super::ForkMutex::new(TrackedSet::with_hasher(std::hash::BuildHasherDefault::new()));

pub(super) unsafe fn after_fork_child() {
    unsafe { TRACKED.reinit_after_fork() };
    unsafe { FINALIZED.reinit_after_fork() };
}

/// Whether `tp_finalize` has already run for the block at `raw`.
pub(super) fn is_finalized(raw: usize) -> bool {
    FINALIZED.lock().contains(&raw)
}

/// Record that `tp_finalize` has run for the block at `raw`.
pub(super) fn mark_finalized(raw: usize) {
    FINALIZED.lock().insert(raw);
}

/// Drop the finalized flag for a block that is being released.
pub(super) fn forget_finalized(raw: usize) {
    FINALIZED.lock().remove(&raw);
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
    // A block with no interpreter object behind it is one the collector knows
    // nothing about, so there is nothing for it to be asked. Both routes that
    // reach here -- `PyType_GenericAlloc` and `PyObject_Init` -- link the block
    // first; the test is what keeps [`clear_garbage`] able to read a cleared
    // link as "the collector freed this one's object".
    if unsafe { (*raw).ob_pyre_link }.is_null() {
        return;
    }
    TRACKED.lock().insert(raw as usize);
}

/// Drop a block from the tracked set -- `PyObject_GC_UnTrack`, and the release
/// `dealloc` performs for a block that never untracked itself.
pub(super) fn forget(raw: usize) {
    TRACKED.lock().remove(&raw);
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

/// Every tracked block and what its `tp_traverse` reports — the collector's
/// [`majit_gc::rawrefcount::CEdgeCensusFn`].
///
/// The collector reads a count above the link share as "C still holds this
/// object". That is what a cycle defeats: two blocks holding each other each
/// read as externally held. Handing over the edges is what lets it tell a
/// reference from inside this graph from one from outside, and it decides
/// nothing here — a block whose object turns out to live still roots what its
/// fields name, which only its trace can establish.
///
/// A referent is usually not itself tracked: an instance of a C-defined type
/// holding a plain Python object is the ordinary case, and that edge is exactly
/// the one the interpreter cannot see. A block that reports nothing goes on
/// rooting, which is why a type declaring no traverse is still a cycle nobody
/// can break -- the position CPython takes for an untracked type.
///
/// A `tp_traverse` is not the only place a reference from one mirror to another
/// lives, so [`super::pyobject::borrowed_edges`] adds the ones this layer holds
/// on C's behalf.
///
/// Runs with the collector borrowed, so nothing it reaches may allocate. Only
/// mirror blocks are read, and those never move.
pub(super) fn c_edges() -> Vec<(usize, Vec<usize>)> {
    let tracked = tracked_blocks();
    let mut edges = Vec::with_capacity(tracked.len());
    for block in tracked {
        let mut referents = Vec::new();
        let reported = unsafe {
            references(block as *mut CPyObject, &mut |referent| {
                referents.push(referent as usize)
            })
        };
        if reported && !referents.is_empty() {
            edges.push((block, referents));
        }
    }
    super::pyobject::borrowed_edges(&mut edges);
    edges
}

/// `gcmodule.c:2000 delete_garbage` — break apart the blocks the collector
/// found to be a cycle.
///
/// A block whose link the collector cleared has lost its interpreter object,
/// and one still holding references after that is held by another block in the
/// same state: the ends of a cycle. Neither can reach a count of zero while the
/// other stands, and nothing this layer owns can drop the references — they
/// live in C fields, and dropping one is the extension's `tp_clear`.
///
/// The reference taken over the whole pass is what makes it safe to run
/// arbitrary `tp_clear` code: a peer's clear can decref any of these, and
/// without it the first clear could free a block this pass has yet to reach.
/// `delete_garbage` takes the same one for the same reason.
pub(super) fn clear_garbage() {
    let garbage: Vec<*mut CPyObject> = tracked_blocks()
        .into_iter()
        .map(|block| block as *mut CPyObject)
        .filter(|&raw| unsafe { (*raw).ob_pyre_link }.is_null() && tp_clear_of(raw).is_some())
        .collect();
    if garbage.is_empty() {
        return;
    }
    for &raw in &garbage {
        unsafe { super::pyobject::incref(raw) };
    }
    for &raw in &garbage {
        unsafe { clear(raw) };
    }
    for raw in garbage {
        unsafe { super::pyobject::decref(raw) };
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
