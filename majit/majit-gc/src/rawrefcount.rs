//! `rpython/rlib/rawrefcount.py` — the raw-refcount surface over the GC.
//!
//! A C extension never sees a moving interpreter object.  It sees a mirror
//! block at a fixed address whose first two words are the collector's
//! business: an ownership count, and a link to the interpreter object.  The
//! algorithm over those two words belongs to the collector — the section in
//! `collector.rs` ported from `incminimark.py:3157-3409` — and what lives here
//! is the vocabulary both sides share: the constants, the block prefix, and
//! the state the collector keeps.
//!
//! `rawrefcount.py` is itself only a front end.  Its `ExtRegistryEntry`s
//! (`:237-322`) lower every public name to a `gc_rawrefcount_*` operation, and
//! `rpython/memory/gctransform/framework.py:491-516` binds those to the
//! collector's own methods — but only for a collector that has them.  The one
//! build that implements the algorithm outside the collector is Boehm
//! (`rpython/rlib/src/boehm-rawrefcount.c`), which can afford to because Boehm
//! never moves an object; a link it hands out stays valid, so a disappearing
//! link plus a periodic poll is enough.  A collector that moves has to own the
//! forwarding, and this one moves.

use crate::address_dict::AddressMap;

/// The ownership share the linked interpreter object contributes to a mirror's
/// count — `rawrefcount.py:15 REFCNT_FROM_PYPY`, spelled `sys.maxint // 4 + 1`.
///
/// A mirror whose count is exactly this is referenced by nothing except the
/// link, so the collector is free to let the linked object die.  Every count
/// above it is a reference the C side holds.
pub const REFCNT_FROM_PYRE: isize = (isize::MAX >> 2) + 1;

/// The count a mirror that must never be freed starts at.
///
/// `rawrefcount.py:16-20` has two constants above [`REFCNT_FROM_PYRE`]:
/// `REFCNT_FROM_PYPY_LIGHT`, for a mirror whose deallocation is a plain free
/// with no deallocator to run, and `_Py_IMMORTAL_REFCNT`, for one that is never
/// deallocated at all.  Only the second has a port — nothing here creates a
/// light mirror.
///
/// The value is chosen for headroom rather than to match either upstream
/// constant: the asserts below hold it at least `1 << 60` clear of the
/// threshold at which a mirror is freed, and the same clear of overflow, so no
/// incref/decref imbalance a running process can produce reaches either end.
pub const REFCNT_IMMORTAL: isize = REFCNT_FROM_PYRE + (1 << (isize::BITS - 4));

const IMMORTAL_HEADROOM: isize = 1 << (isize::BITS - 4);
const _: () = assert!(REFCNT_IMMORTAL - REFCNT_FROM_PYRE >= IMMORTAL_HEADROOM);
const _: () = assert!(isize::MAX - REFCNT_IMMORTAL >= IMMORTAL_HEADROOM);

/// Scheduled when the dead queue becomes non-empty.
///
/// `incminimark.py:3181 rrc_dealloc_trigger_callback`.  It runs at a public
/// collection entry point with the collector borrowed, so it may only schedule
/// work; the drain itself happens later, from the embedder.
pub type DeallocTriggerFn = fn();

/// `incminimark.py:3163-3166 PYOBJ_HDR`: the prefix of a mirror block the
/// collector reads.  The real block is longer — a type pointer and whatever
/// the C type's declared size adds follow it — and none of that is the
/// collector's business.
#[repr(C)]
pub struct PyObjHeader {
    pub ob_refcnt: isize,
    pub ob_link: usize,
}

/// `incminimark.py:3169-3170 _pyobj`.
#[inline]
pub(crate) fn pyobj(address: usize) -> *mut PyObjHeader {
    address as *mut PyObjHeader
}

/// Which of the two link directions a shared pass is walking.
///
/// The free passes are written once and run over both lists;
/// `incminimark.py:3276-3282` and `:3383-3395` express the same thing by
/// passing the surviving list and either the dictionary or a null one.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum RrcList {
    /// `rrc_p_list_*` — the interpreter object owns the mirror.
    P,
    /// `rrc_o_list_*` — the mirror owns the interpreter object.
    O,
}

/// The collector's rawrefcount section — `incminimark.py:3172-3183
/// rawrefcount_init` allocates exactly these.
///
/// `enabled` is `rrc_enabled`: every phase call site is guarded on it, so a
/// process that has loaded no extension pays one boolean test per collection.
#[derive(Default)]
pub(crate) struct RawRefCount {
    pub(crate) enabled: bool,
    /// `rrc_p_list_young` / `rrc_p_list_old` — mirrors of objects the
    /// interpreter owns, split by whether the link is a nursery address.
    pub(crate) p_list_young: Vec<usize>,
    pub(crate) p_list_old: Vec<usize>,
    /// `rrc_o_list_young` / `rrc_o_list_old` — the other direction: a mirror
    /// the C side owns, whose interpreter object must not outlive it.
    pub(crate) o_list_young: Vec<usize>,
    pub(crate) o_list_old: Vec<usize>,
    /// `rrc_p_dict` (non-nursery keys) / `rrc_p_dict_nurs` (nursery keys):
    /// interpreter object address -> mirror address.  Two tables because every
    /// nursery key moves on the next minor, so that half is emptied and
    /// refilled there while the old half is only re-keyed by a major.
    pub(crate) p_dict: AddressMap<usize>,
    pub(crate) p_dict_nurs: AddressMap<usize>,
    /// `rrc_dealloc_pending` — mirrors whose linked object has died and whose
    /// deallocator the embedder still has to run.
    pub(crate) dealloc_pending: Vec<usize>,
    pub(crate) dealloc_trigger: Option<DeallocTriggerFn>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refcnt_from_pyre_is_maxint_over_four_plus_one() {
        // rawrefcount.py:15, on a 64-bit host.
        assert_eq!(REFCNT_FROM_PYRE, (isize::MAX / 4) + 1);
        assert_eq!(REFCNT_FROM_PYRE, 1 << (isize::BITS - 3));
    }

    #[test]
    fn pyobj_header_is_the_two_words_the_collector_reads() {
        assert_eq!(
            std::mem::size_of::<PyObjHeader>(),
            2 * std::mem::size_of::<usize>()
        );
        assert_eq!(
            std::mem::align_of::<PyObjHeader>(),
            std::mem::align_of::<usize>()
        );
    }
}
