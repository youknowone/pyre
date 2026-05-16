//! `compile.py:855` `AbstractResumeGuardDescr._attrs_ = ('rd_numb',
//! 'rd_consts', 'rd_virtuals', 'rd_pendingfields', 'status')`.
//!
//! Resume-payload backing store shared between metainterp and backend
//! crates as part of the Unified-Descr Port (Phase C-1).  Moving the
//! storage to `majit-backend` lets backends instantiate the unified
//! `ResumeGuardDescr` directly without depending on `majit-metainterp`.
use std::cell::UnsafeCell;
use std::rc::Rc;
use std::sync::Arc;

use majit_ir::{Const, GuardPendingFieldEntry, RdVirtualInfo};

/// Resume-payload backing store for ResumeGuardDescr (`compile.py:855`).
///
/// Mutated through `set_rd_*` setters — the optimizer's
/// `store_final_boxes_in_guard` (compile.py:869) replaces the whole
/// slot post-numbering.  In-place modification of the inner slice
/// never happens; sharing is observably identical to RPython's
/// `self.rd_consts = other.rd_consts` reference-share semantics
/// (compile.py:861-867 `copy_all_attributes_from`).
///
/// Each field stores `Arc<[T]>` so the donor-share path in
/// `copy_all_attributes_from` can mirror RPython's reference-share
/// semantics with a single `Arc::clone()` rather than a `Vec::clone()`
/// that would deep-copy the bytes.  External setters still accept
/// `Option<Vec<T>>`; the conversion to `Arc<[T]>` is one move per
/// (rare) write.
#[derive(Debug)]
pub struct RdPayload {
    rd_numb: UnsafeCell<Option<Arc<[u8]>>>,
    rd_consts: UnsafeCell<Option<Arc<[Const]>>>,
    rd_virtuals: UnsafeCell<Option<Arc<[Rc<RdVirtualInfo>]>>>,
    rd_pendingfields: UnsafeCell<Option<Arc<[GuardPendingFieldEntry]>>>,
}

// Safety: single-threaded JIT (RPython GIL parity).  Rc<RdVirtualInfo>
// is non-Send/Sync in general, but JIT readers never cross threads;
// the unsafe impls here are part of the same RPython GIL contract that
// every other UnsafeCell-bearing descr field relies on.
unsafe impl Send for RdPayload {}
unsafe impl Sync for RdPayload {}

impl RdPayload {
    pub fn empty() -> Self {
        Self {
            rd_numb: UnsafeCell::new(None),
            rd_consts: UnsafeCell::new(None),
            rd_virtuals: UnsafeCell::new(None),
            rd_pendingfields: UnsafeCell::new(None),
        }
    }

    /// `compile.py:861-872` `copy_all_attributes_from` — construct an
    /// `RdPayload` whose four slots share `Arc<[T]>` refcounts with a
    /// donor descr, matching RPython's reference-share semantics.
    pub fn from_arcs(
        rd_numb: Option<Arc<[u8]>>,
        rd_consts: Option<Arc<[Const]>>,
        rd_virtuals: Option<Arc<[Rc<RdVirtualInfo>]>>,
        rd_pendingfields: Option<Arc<[GuardPendingFieldEntry]>>,
    ) -> Self {
        Self {
            rd_numb: UnsafeCell::new(rd_numb),
            rd_consts: UnsafeCell::new(rd_consts),
            rd_virtuals: UnsafeCell::new(rd_virtuals),
            rd_pendingfields: UnsafeCell::new(rd_pendingfields),
        }
    }

    /// `clone()` shares every field — used by `clone_descr()`, which
    /// mirrors RPython's `ResumeGuardDescr.clone()` (compile.py:844-846).
    ///
    /// RPython `copy_all_attributes_from` (compile.py:861-867) does
    /// `self.rd_consts = other.rd_consts` etc. — list reference share.
    /// `Arc<[T]>` provides the equivalent: `Arc::clone()` only bumps a
    /// refcount.  In-place mutation never happens (the only writer is
    /// `set_rd_*` which swap-replaces the whole slot), so sharing is
    /// safe and observably identical to RPython.
    pub fn deep_clone(&self) -> Self {
        Self {
            rd_numb: UnsafeCell::new(unsafe { (*self.rd_numb.get()).clone() }),
            rd_consts: UnsafeCell::new(unsafe { (*self.rd_consts.get()).clone() }),
            rd_virtuals: UnsafeCell::new(unsafe { (*self.rd_virtuals.get()).clone() }),
            rd_pendingfields: UnsafeCell::new(unsafe { (*self.rd_pendingfields.get()).clone() }),
        }
    }

    pub fn rd_numb(&self) -> Option<&[u8]> {
        unsafe { (*self.rd_numb.get()).as_deref() }
    }
    pub fn rd_numb_arc(&self) -> Option<Arc<[u8]>> {
        unsafe { (*self.rd_numb.get()).clone() }
    }
    pub fn set_rd_numb(&self, value: Option<Vec<u8>>) {
        unsafe { *self.rd_numb.get() = value.map(Arc::from) }
    }
    pub fn set_rd_numb_arc(&self, value: Option<Arc<[u8]>>) {
        unsafe { *self.rd_numb.get() = value }
    }

    pub fn rd_consts(&self) -> Option<&[Const]> {
        unsafe { (*self.rd_consts.get()).as_deref() }
    }
    pub fn rd_consts_arc(&self) -> Option<Arc<[Const]>> {
        unsafe { (*self.rd_consts.get()).clone() }
    }
    pub fn set_rd_consts(&self, value: Option<Vec<Const>>) {
        unsafe { *self.rd_consts.get() = value.map(Arc::from) }
    }
    pub fn set_rd_consts_arc(&self, value: Option<Arc<[Const]>>) {
        unsafe { *self.rd_consts.get() = value }
    }

    pub fn rd_virtuals(&self) -> Option<&[Rc<RdVirtualInfo>]> {
        unsafe { (*self.rd_virtuals.get()).as_deref() }
    }
    pub fn rd_virtuals_arc(&self) -> Option<Arc<[Rc<RdVirtualInfo>]>> {
        unsafe { (*self.rd_virtuals.get()).clone() }
    }
    pub fn set_rd_virtuals(&self, value: Option<Vec<Rc<RdVirtualInfo>>>) {
        unsafe { *self.rd_virtuals.get() = value.map(Arc::from) }
    }
    pub fn set_rd_virtuals_arc(&self, value: Option<Arc<[Rc<RdVirtualInfo>]>>) {
        unsafe { *self.rd_virtuals.get() = value }
    }

    pub fn rd_pendingfields(&self) -> Option<&[GuardPendingFieldEntry]> {
        unsafe { (*self.rd_pendingfields.get()).as_deref() }
    }
    pub fn rd_pendingfields_arc(&self) -> Option<Arc<[GuardPendingFieldEntry]>> {
        unsafe { (*self.rd_pendingfields.get()).clone() }
    }
    pub fn set_rd_pendingfields(&self, value: Option<Vec<GuardPendingFieldEntry>>) {
        unsafe { *self.rd_pendingfields.get() = value.map(Arc::from) }
    }
    pub fn set_rd_pendingfields_arc(&self, value: Option<Arc<[GuardPendingFieldEntry]>>) {
        unsafe { *self.rd_pendingfields.get() = value }
    }
}
