//! Building blocks for `PtrInfo` — the pointer-analysis info type
//! attached to each `_forwarded` slot. Hosted in `majit-ir` so the
//! `Forwarded` move that follows can reference these types without
//! a `majit-metainterp → majit-ir` circular dep.
//!
//! At present only the variant-shared `AbstractVirtualPtrInfo` lives
//! here; the rest of the PtrInfo hierarchy is moved in follow-up
//! slices.

use crate::RdVirtualInfo;

/// info.py: `AbstractVirtualPtrInfo` (RPython base class hint). Pyre
/// hoists only the fields shared by every Virtual* variant so each
/// `PtrInfo::Virtual*` carries a single embedded slot instead of N
/// independent copies of the same field set.
///
/// `descr` and `_is_virtual` are NOT lifted here:
///   - `descr` is variant-specific (SizeDescr for Virtual, ArrayDescr
///     for VirtualArray, etc.) — RPython's `_attrs_` is a hint to the
///     translator's slot allocator, not a parity constraint on the
///     storage *type*. Each pyre variant keeps its own typed `descr`.
///   - `_is_virtual` collapses into the pyre enum tag itself
///     (`PtrInfo::Virtual(_)` IS the truthy carrier of `_is_virtual`);
///     no separate slot is needed.
///
/// `make_virtual_info` (resume.py:307-315) reads `cached_vinfo` to
/// dedup RdVirtualInfo allocations across multiple finish() calls
/// referencing the same virtual. `RefCell` provides interior
/// mutability so the immutable-receiver accessor can populate the
/// cache on first miss.
#[derive(Clone, Debug, Default)]
pub struct AbstractVirtualPtrInfo {
    pub cached_vinfo: std::cell::RefCell<Option<std::rc::Rc<RdVirtualInfo>>>,
}

impl AbstractVirtualPtrInfo {
    pub fn new() -> Self {
        Self {
            cached_vinfo: std::cell::RefCell::new(None),
        }
    }
}
