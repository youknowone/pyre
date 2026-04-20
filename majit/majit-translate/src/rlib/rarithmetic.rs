//! RPython `rpython/rlib/rarithmetic.py` — integer arithmetic helpers.
//!
//! Only the subset consumed by the annotator port lands here. The
//! full RPython rarithmetic module covers ~700 LOC of overflow-safe
//! operators, r_uint / r_longlong / r_singlefloat types, ovfcheck
//! decorators, etc. Those land as their call sites in downstream
//! ports come online.

use crate::annotator::model::KnownType;

/// RPython `compute_restype(self_type, other_type)` (rarithmetic.py:211-226).
///
/// ```python
/// def compute_restype(self_type, other_type):
///     if self_type is other_type:
///         if self_type is bool:
///             return int
///         return self_type
///     if other_type in (bool, int, long):
///         if self_type is bool:
///             return int
///         return self_type
///     if self_type in (bool, int, long):
///         return other_type
///     if self_type is float or other_type is float:
///         return float
///     if self_type.SIGNED == other_type.SIGNED:
///         return build_int(None, self_type.SIGNED,
///                          max(self_type.BITS, other_type.BITS))
///     raise AssertionError("Merging these types (%s, %s) is not supported" % (self_type, other_type))
/// ```
///
/// Rust port scopes the domain to [`KnownType`] variants the annotator
/// actually exercises. Upstream `long` is Python 2's arbitrary-precision
/// int, folded into [`KnownType::Int`] for Rust. `r_uint` maps to
/// [`KnownType::Ruint`]. The wider-bit fallback (rarithmetic.py:224-225)
/// lands with the `r_longlong` / `r_ulonglong` KnownType variants when
/// those are added to the lattice.
pub fn compute_restype(self_type: KnownType, other_type: KnownType) -> KnownType {
    // Upstream: `if self_type is other_type:`.
    if self_type == other_type {
        if self_type == KnownType::Bool {
            return KnownType::Int;
        }
        return self_type;
    }
    // Upstream: `if other_type in (bool, int, long):`.
    if matches!(other_type, KnownType::Bool | KnownType::Int) {
        if self_type == KnownType::Bool {
            return KnownType::Int;
        }
        return self_type;
    }
    // Upstream: `if self_type in (bool, int, long):`.
    if matches!(self_type, KnownType::Bool | KnownType::Int) {
        return other_type;
    }
    // Upstream: `if self_type is float or other_type is float:`.
    if self_type == KnownType::Float || other_type == KnownType::Float {
        return KnownType::Float;
    }
    // Upstream: `if self_type.SIGNED == other_type.SIGNED: build_int(...)`.
    // Only r_uint is present in the Rust lattice currently; signedness
    // comparison collapses to "both r_uint → r_uint" which the
    // `self_type == other_type` branch above already handles.
    panic!(
        "compute_restype: unsupported merge of {:?} and {:?}",
        self_type, other_type
    );
}

/// RPython `signedtype(t)` (rarithmetic.py:228-233).
///
/// ```python
/// @specialize.memo()
/// def signedtype(t):
///     if t in (bool, int, long):
///         return True
///     else:
///         return t.SIGNED
/// ```
pub fn signedtype(t: KnownType) -> bool {
    match t {
        // Upstream: `if t in (bool, int, long): return True`.
        KnownType::Bool | KnownType::Int => true,
        // Upstream: `return t.SIGNED`. `r_uint` is unsigned; currently
        // the lattice carries no other `_longlong` / `r_singlefloat`
        // numeric types that would be SIGNED=True — if they land later
        // their signedness match goes here.
        KnownType::Ruint => false,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bool_bool_widens_to_int() {
        assert_eq!(
            compute_restype(KnownType::Bool, KnownType::Bool),
            KnownType::Int
        );
    }

    #[test]
    fn int_int_stays_int() {
        assert_eq!(
            compute_restype(KnownType::Int, KnownType::Int),
            KnownType::Int
        );
    }

    #[test]
    fn int_ruint_picks_ruint() {
        assert_eq!(
            compute_restype(KnownType::Int, KnownType::Ruint),
            KnownType::Ruint
        );
        assert_eq!(
            compute_restype(KnownType::Ruint, KnownType::Int),
            KnownType::Ruint
        );
    }

    #[test]
    fn bool_int_widens_to_int() {
        assert_eq!(
            compute_restype(KnownType::Bool, KnownType::Int),
            KnownType::Int
        );
    }

    #[test]
    fn int_float_widens_to_float() {
        assert_eq!(
            compute_restype(KnownType::Int, KnownType::Float),
            KnownType::Float
        );
    }
}
