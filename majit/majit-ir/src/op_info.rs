//! `OpInfo` — abstract information attached to operations during optimization.
//!
//! info.py: `AbstractInfo` hierarchy. Each operation can have associated
//! analysis info (e.g. known integer bounds, pointer info, virtual object
//! state). Hosted in `majit-ir` so the `Forwarded` move that follows can
//! reference it without a `majit-metainterp → majit-ir` circular dep.
//!
//! Pure data only. The `Rc<RefCell<…>>` wrappers around `IntBound` /
//! `PtrInfo` provide shared-identity semantics matching PyPy's
//! `_forwarded` slot — see `info.py:865-894 get*ptrinfo` "return fw".

use crate::intbound::IntBound;
use crate::ptr_info::PtrInfo;

/// `info.py` `AbstractInfo` hierarchy collapsed into a Rust enum.
///
/// `Ptr` carries `Rc<RefCell<PtrInfo>>` so the underlying info object has
/// the same object-identity semantics as RPython's `_forwarded` slot:
/// when two `_forwarded` slots are set to the same `Ptr(rc.clone())`,
/// in-place mutations through one handle are observable through the
/// other. `IntBound` is similarly wrapped so `optimizer.py:99-113
/// getintbound` mutations propagate.
#[derive(Clone)]
pub enum OpInfo {
    /// No information known.
    Unknown,
    /// Known integer bounds. info.py:1264 IntBound.
    /// `IntBound::from_constant(v)` is the canonical Int constant carrier.
    IntBound(std::rc::Rc<std::cell::RefCell<IntBound>>),
    /// Pointer info (non-null, known class, virtual, etc.).
    /// `PtrInfo::Constant(GcRef)` is the Ref constant carrier
    /// (info.py:706 ConstPtrInfo).
    Ptr(std::rc::Rc<std::cell::RefCell<PtrInfo>>),
    /// Known constant float value.
    /// info.py:851 FloatConstInfo — Float constant carrier.
    FloatConst(f64),
}

impl std::fmt::Debug for OpInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OpInfo::Unknown => f.write_str("OpInfo::Unknown"),
            OpInfo::IntBound(ib) => f
                .debug_tuple("OpInfo::IntBound")
                .field(&*ib.borrow())
                .finish(),
            OpInfo::Ptr(p) => f.debug_tuple("OpInfo::Ptr").field(&*p.borrow()).finish(),
            OpInfo::FloatConst(v) => f.debug_tuple("OpInfo::FloatConst").field(v).finish(),
        }
    }
}

impl OpInfo {
    /// Helper for constructing `OpInfo::Ptr` from owned `PtrInfo` —
    /// wraps in a fresh `Rc<RefCell<>>` for the shared-identity storage.
    pub fn ptr(info: PtrInfo) -> Self {
        OpInfo::Ptr(std::rc::Rc::new(std::cell::RefCell::new(info)))
    }

    /// Helper for constructing `OpInfo::IntBound` from owned `IntBound`.
    pub fn int_bound(b: IntBound) -> Self {
        OpInfo::IntBound(std::rc::Rc::new(std::cell::RefCell::new(b)))
    }

    pub fn is_constant(&self) -> bool {
        match self {
            OpInfo::FloatConst(_) => true,
            OpInfo::Ptr(p) => matches!(&*p.borrow(), PtrInfo::Constant(_)),
            OpInfo::IntBound(b) => b.borrow().is_constant(),
            OpInfo::Unknown => false,
        }
    }

    /// Get the constant float value if this is a FloatConst.
    pub fn get_constant_float(&self) -> Option<f64> {
        match self {
            OpInfo::FloatConst(f) => Some(*f),
            _ => None,
        }
    }

    /// Returns the live `Rc` handle to the `IntBound` for the `IntBound`
    /// variant. Mirrors RPython object identity: callers that retain the
    /// handle observe in-place mutations through other holders.
    pub fn get_int_bound(&self) -> Option<&std::rc::Rc<std::cell::RefCell<IntBound>>> {
        match self {
            OpInfo::IntBound(b) => Some(b),
            _ => None,
        }
    }

    /// Whether this info is known non-null.
    /// info.py: is_nonnull()
    pub fn is_nonnull(&self) -> bool {
        match self {
            OpInfo::Ptr(p) => p.borrow().is_nonnull(),
            _ => false,
        }
    }

    /// Whether this info represents a virtual (allocation-removed) object.
    /// info.py: is_virtual()
    pub fn is_virtual(&self) -> bool {
        matches!(self, OpInfo::Ptr(p) if p.borrow().is_virtual())
    }

    /// Returns the live `Rc` handle to the `PtrInfo` for the `Ptr`
    /// variant. Mirrors RPython object identity: callers that retain the
    /// handle observe in-place mutations through other holders. Borrow
    /// the returned handle (`handle.borrow()` / `handle.borrow_mut()`)
    /// to access the inner `PtrInfo`.
    pub fn get_ptr_info(&self) -> Option<&std::rc::Rc<std::cell::RefCell<PtrInfo>>> {
        match self {
            OpInfo::Ptr(p) => Some(p),
            _ => None,
        }
    }
}
