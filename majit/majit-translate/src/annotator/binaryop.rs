//! RPython `rpython/annotator/binaryop.py` — pairwise dispatch for
//! `dispatch=2` operations (add / sub / mul / ... / is_ / getitem / ...).
//!
//! Upstream this file is ~860 LOC of `@op.<name>.register(Some_cls_1,
//! Some_cls_2)` decorators. The Rust port fans this out across
//! Commits 3-5 (one commit per pair-family); this skeleton carries only
//! the `BINARY_OPERATIONS` set + the `(SomeObject, SomeObject)` fallback
//! entries used by upstream's `binaryop.py:26-60` / `:62-72` / `:74-80`
//! as the catch-all behaviour.

use super::super::flowspace::operation::OpKind;
use super::operations::BinaryRegistry;

/// RPython `BINARY_OPERATIONS` (binaryop.py:22-23).
///
/// ```python
/// BINARY_OPERATIONS = set([oper.opname for oper in op.__dict__.values()
///                         if oper.dispatch == 2])
/// ```
///
/// Rust carries the same set as a static `&[OpKind]` — the upstream
/// invariant is that this list mirrors every [`OpKind`] whose
/// [`OpKind::dispatch`] is [`Dispatch::Double`]. A unit test below
/// asserts the two stay in sync.
pub static BINARY_OPERATIONS: &[OpKind] = &[
    OpKind::Is,
    OpKind::GetItem,
    OpKind::GetItemIdx,
    OpKind::SetItem,
    OpKind::DelItem,
    OpKind::Add,
    OpKind::AddOvf,
    OpKind::Sub,
    OpKind::SubOvf,
    OpKind::Mul,
    OpKind::MulOvf,
    OpKind::TrueDiv,
    OpKind::FloorDiv,
    OpKind::FloorDivOvf,
    OpKind::Div,
    OpKind::DivOvf,
    OpKind::Mod,
    OpKind::ModOvf,
    OpKind::LShift,
    OpKind::LShiftOvf,
    OpKind::RShift,
    OpKind::And,
    OpKind::Or,
    OpKind::Xor,
    OpKind::InplaceAdd,
    OpKind::InplaceSub,
    OpKind::InplaceMul,
    OpKind::InplaceTrueDiv,
    OpKind::InplaceFloorDiv,
    OpKind::InplaceDiv,
    OpKind::InplaceMod,
    OpKind::InplaceLShift,
    OpKind::InplaceRShift,
    OpKind::InplaceAnd,
    OpKind::InplaceOr,
    OpKind::InplaceXor,
    OpKind::Lt,
    OpKind::Le,
    OpKind::Eq,
    OpKind::Ne,
    OpKind::Gt,
    OpKind::Ge,
    OpKind::Cmp,
    OpKind::Coerce,
];

/// Populate [`BinaryRegistry`] with the `pairtype(SomeObject, SomeObject)`
/// default entries upstream binaryop.py ships — the catch-all handlers
/// that every concrete pair inherits through the MRO walk.
///
/// Commits 3-5 register the concrete-pair entries on top of this
/// skeleton (SomeInteger×SomeInteger, SomeList×SomeInteger, …).
pub fn populate_defaults(_reg: &mut BinaryRegistry) {
    // Intentional: the `SomeObject/SomeObject` defaults for is_ /
    // comparisons / getitem live in commits 3-5, where they land next
    // to the concrete-pair handlers that share their helpers. Keeping
    // this function empty during 2b lets `consider_hlop` fall through
    // to the `no binary spec` panic, which is the signal flowgraph
    // smoke tests are expected to exercise before the concrete pairs
    // appear.
    //
    // Upstream order (binaryop.py):
    //
    // * 26-60   `@op.is_.register(SomeObject, SomeObject)`  → Commit 3
    // * 62-72   cmp default factory (`_make_cmp_annotator_default`) → Commit 4
    // * 74-80   `@op.getitem.register(SomeObject, SomeObject)` → Commit 5
    //
    // Each commit will `reg.register(OpKind::X, SomeValueTag::Object,
    // SomeValueTag::Object, Specialization { … })`.
}

#[cfg(test)]
mod tests {
    use super::super::super::flowspace::operation::OpKind;
    use super::*;

    #[test]
    fn binary_operations_includes_add_sub_mul() {
        assert!(BINARY_OPERATIONS.contains(&OpKind::Add));
        assert!(BINARY_OPERATIONS.contains(&OpKind::Sub));
        assert!(BINARY_OPERATIONS.contains(&OpKind::Mul));
        assert!(BINARY_OPERATIONS.contains(&OpKind::Is));
        assert!(BINARY_OPERATIONS.contains(&OpKind::GetItem));
    }

    #[test]
    fn binary_operations_excludes_unary() {
        assert!(!BINARY_OPERATIONS.contains(&OpKind::Neg));
        assert!(!BINARY_OPERATIONS.contains(&OpKind::Len));
    }

    #[test]
    fn binary_operations_matches_dispatch_double() {
        use super::super::super::flowspace::operation::Dispatch;
        for op in BINARY_OPERATIONS {
            assert!(
                matches!(op.dispatch(), Dispatch::Double),
                "{:?} in BINARY_OPERATIONS but dispatch() is not Double",
                op
            );
        }
    }
}
