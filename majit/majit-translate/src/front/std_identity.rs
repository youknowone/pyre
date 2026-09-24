//! Transparent one-field std wrappers and scalar Copy combinators as
//! identity / zero, not residual calls.
//!
//! RPython has no `Cell` / `Ref` / `MutexGuard` / `Box` / `AtomicUsize`
//! types: the field *is* the value.  Charon still emits residual
//! `FunctionPath` / `Method` calls to those bodies (Opaque in the LLBC),
//! and every caller then dies at `translate_op` with an unregistered
//! `CallRegistry` miss.  The call is not a graph and not a foreign
//! function — it is `same_as` (or a typed zero for `Default` on a
//! machine-word integer / bool).
//!
//! `core::mem::replace` is recognised but *not* rewritten here.  The
//! front models `&mut T` as the referent value (`Rvalue::Ref` aliases
//! the place's Variable), so the residual call's first argument is the
//! old `T`, not the slot address.  Emitting `__deref_write(old, new)`
//! would store through the old value; yielding only `same_as(old)`
//! would drop the store.  The write needs the borrowed Place, the same
//! fact `Atomic*::store` records in `atomic_ref_place` — and that map
//! is atomic-only.

use crate::flowspace::model::Variable;
use crate::model::{CallTarget, LinkArg, OpKind};

fn path_has(segments: &[String], needle: &str) -> bool {
    segments.iter().any(|s| s.as_str() == needle)
}

fn path_leaf(path: Option<&str>) -> Option<&str> {
    path.map(|p| p.rsplit("::").next().unwrap_or(p))
}

fn function_leaf(segments: &[String]) -> Option<&str> {
    segments.last().map(String::as_str)
}

/// `Cell::{new,get}`, `Atomic*::new`, `Box::as_ref`,
/// `Ref::deref`, `MutexGuard::{deref,deref_mut}` — one-arg identity.
pub(crate) fn identity_operand(
    target: &CallTarget,
    args: &[LinkArg],
    receiver_path: Option<&str>,
    dest_path: Option<&str>,
) -> Option<Variable> {
    if args.len() != 1 {
        return None;
    }
    if !is_identity_wrapper_target(target, receiver_path, dest_path) {
        return None;
    }
    args[0].as_variable().cloned()
}

fn is_identity_wrapper_target(
    target: &CallTarget,
    receiver_path: Option<&str>,
    dest_path: Option<&str>,
) -> bool {
    match target {
        CallTarget::Method {
            name,
            receiver_root,
            ..
        } => match (name.as_str(), path_leaf(receiver_root.as_deref())) {
            ("get", Some("Cell")) => true,
            ("as_ref", Some("Box")) => true,
            ("deref" | "deref_mut", Some("Ref") | Some("MutexGuard")) => true,
            _ => false,
        },
        CallTarget::FunctionPath { segments, .. } => {
            let leaf = function_leaf(segments);
            let recv = path_leaf(receiver_path);
            let dest = path_leaf(dest_path);
            match leaf {
                Some("new")
                    if dest == Some("Cell")
                        || path_has(segments, "Cell") && !path_has(segments, "RefCell") =>
                {
                    true
                }
                Some("new") if path_has(segments, "atomic") => true,
                Some("get") if recv == Some("Cell") || path_has(segments, "Cell") => true,
                Some("as_ref")
                    if recv == Some("Box")
                        || path_has(segments, "Box")
                        || path_has(segments, "boxed") =>
                {
                    true
                }
                Some("deref" | "deref_mut")
                    if matches!(recv, Some("Ref") | Some("MutexGuard"))
                        || path_has(segments, "MutexGuard")
                        || (path_has(segments, "Ref") && !path_has(segments, "RefCell")) =>
                {
                    true
                }
                _ => false,
            }
        }
        _ => false,
    }
}

/// The concrete `Copy` impls, spelled in full — the same items
/// [`crate::front::mir::is_core_clone_impls_clone_path`] /
/// [`crate::front::mir::is_core_default_path`] recognise on the
/// `RegularCall` route. Both routes call those predicates.
///
/// A leaf-name match would also take `core::clone::<Impl>::clone` and
/// `core::default::Default::default`, which are the blanket impl and the
/// trait item. Charon does not monomorphise them, so the destination
/// width these folds read describes the declaration and not the
/// instantiation, and folding either one answers for every `T` a caller
/// might have instantiated. A method spelling carries no module path at
/// all, so it can never be shown to be one of these items.
pub(crate) fn is_clone_target(target: &CallTarget) -> bool {
    match target {
        CallTarget::FunctionPath { segments, .. } => {
            crate::front::mir::is_core_clone_impls_clone_path(&segments.join("::"))
        }
        _ => false,
    }
}

pub(crate) fn is_default_target(target: &CallTarget) -> bool {
    match target {
        CallTarget::FunctionPath { segments, .. } => {
            crate::front::mir::is_core_default_path(&segments.join("::"))
        }
        _ => false,
    }
}

/// Machine-word integer / bool `Copy` — the only `Clone::clone` /
/// `Default::default` instantiations with one uniform answer.
pub(crate) fn is_scalar_copy_width(
    dest_int_atom: Option<&str>,
    dest_uint_atom: Option<&str>,
    dest_is_bool: bool,
) -> bool {
    dest_is_bool
        || dest_int_atom.is_some_and(crate::front::checked_arith::is_ovf_width_int_atom)
        || dest_uint_atom.is_some_and(crate::front::checked_arith_uint::is_word_sized_uint_atom)
}

/// Rewrite a residual std primitive call in place when the destination
/// width / wrapper shape is already in hand.  Unknown shapes are
/// returned unchanged.
///
/// `banks_agree` is the caller's verdict that the receiver and the
/// destination occupy the same register bank (`flatten.py getkind`).
/// It gates the identity arm only: aliasing across banks spells
/// `int_same_as/r>i`, and its consumer `int_add/ir>i` -- opnames
/// `blackhole.rs` leaves unwired because they are kind-flow bugs.  A
/// wrapper whose payload the LLBC does not expose (an Opaque foreign
/// `Cell`) reads as a `Ref`, so `Cell::get` is a load across banks and
/// declines here; it becomes an identity again once the layout is
/// visible and the payload's bank is the wrapper's.
pub(crate) fn lower_std_primitive_op(
    op_kind: OpKind,
    receiver_path: Option<&str>,
    dest_path: Option<&str>,
    dest_int_atom: Option<&str>,
    dest_uint_atom: Option<&str>,
    dest_is_bool: bool,
    banks_agree: bool,
) -> OpKind {
    let OpKind::Call {
        target,
        args,
        result_ty,
    } = &op_kind
    else {
        return op_kind;
    };
    if banks_agree && let Some(operand) = identity_operand(target, args, receiver_path, dest_path) {
        return OpKind::UnaryOp {
            op: "same_as".to_string(),
            operand,
            result_ty: result_ty.clone(),
        };
    }
    if args.len() == 1
        && is_clone_target(target)
        && is_scalar_copy_width(dest_int_atom, dest_uint_atom, dest_is_bool)
    {
        let Some(operand) = args[0].as_variable().cloned() else {
            return op_kind;
        };
        return OpKind::UnaryOp {
            op: "same_as".to_string(),
            operand,
            result_ty: result_ty.clone(),
        };
    }
    if args.is_empty() && is_default_target(target) {
        if dest_int_atom.is_some_and(crate::front::checked_arith::is_ovf_width_int_atom) {
            return OpKind::ConstInt(0);
        }
        if dest_uint_atom.is_some_and(crate::front::checked_arith_uint::is_word_sized_uint_atom) {
            return OpKind::ConstUInt(0);
        }
        if dest_is_bool {
            return OpKind::ConstBool(false);
        }
    }
    op_kind
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ValueType, call_args};

    fn path(segments: &[&str]) -> CallTarget {
        CallTarget::FunctionPath {
            segments: segments.iter().map(|s| s.to_string()).collect(),
            fun_decl_id: None,
        }
    }

    fn call(target: CallTarget, args: Vec<Variable>, result_ty: ValueType) -> OpKind {
        OpKind::Call {
            target,
            args: call_args(args),
            result_ty,
        }
    }

    fn dummy_var() -> Variable {
        let mut g = crate::model::FunctionGraph::new("std_identity_dummy");
        g.push_op_var(g.startblock, OpKind::ConstInt(1), true)
            .unwrap()
    }

    #[test]
    fn a_cross_bank_wrapper_member_stays_residual() {
        // `Cell::get(&Cell<T>) -> T` is a load: the receiver is in the
        // ref bank and the result in the int bank.  Aliasing it spells
        // `int_same_as/r>i`, which the blackhole leaves unwired.
        let v = dummy_var();
        let get = lower_std_primitive_op(
            call(
                path(&["core", "cell", "Cell", "get"]),
                vec![v.clone()],
                ValueType::Int,
            ),
            Some("core::cell::Cell"),
            None,
            Some("I64"),
            None,
            false,
            false,
        );
        assert!(
            matches!(get, OpKind::Call { .. }),
            "a cross-bank wrapper member must keep its residual call"
        );
    }

    #[test]
    fn cell_get_and_atomic_new_are_identity() {
        let v = dummy_var();
        let get = lower_std_primitive_op(
            call(
                path(&["core", "cell", "Cell", "get"]),
                vec![v.clone()],
                ValueType::Int,
            ),
            Some("core::cell::Cell"),
            None,
            Some("I64"),
            None,
            false,
            true,
        );
        match get {
            OpKind::UnaryOp { op, operand, .. } => {
                assert_eq!(op, "same_as");
                assert_eq!(operand, v);
            }
            other => panic!("expected same_as, got {other:?}"),
        }

        let new = lower_std_primitive_op(
            call(
                path(&["core", "sync", "atomic", "<Impl>", "new"]),
                vec![v.clone()],
                ValueType::Unsigned,
            ),
            None,
            Some("core::sync::atomic::AtomicUsize"),
            None,
            Some("Usize"),
            false,
            true,
        );
        assert!(
            matches!(new, OpKind::UnaryOp { ref op, .. } if op == "same_as"),
            "AtomicUsize::new is identity on the inner usize"
        );
    }

    #[test]
    fn cell_new_uses_dest_leaf_when_impl_is_erased() {
        let v = dummy_var();
        let lowered = lower_std_primitive_op(
            call(
                path(&["core", "cell", "<Impl>", "new"]),
                vec![v.clone()],
                ValueType::Int,
            ),
            None,
            Some("core::cell::Cell"),
            None,
            None,
            false,
            true,
        );
        assert!(matches!(
            lowered,
            OpKind::UnaryOp { ref op, .. } if op == "same_as"
        ));

        let refcell = lower_std_primitive_op(
            call(
                path(&["core", "cell", "<Impl>", "new"]),
                vec![v],
                ValueType::Ref(None),
            ),
            None,
            Some("core::cell::RefCell"),
            None,
            None,
            false,
            true,
        );
        assert!(
            matches!(refcell, OpKind::Call { .. }),
            "RefCell::new is not a one-field identity"
        );
    }

    #[test]
    fn box_as_ref_and_mutex_guard_deref_are_identity() {
        let v = dummy_var();
        let as_ref = lower_std_primitive_op(
            call(
                CallTarget::method("as_ref", Some("Box".into())),
                vec![v.clone()],
                ValueType::Ref(None),
            ),
            Some("alloc::boxed::Box"),
            None,
            None,
            None,
            false,
            true,
        );
        assert!(matches!(
            as_ref,
            OpKind::UnaryOp { ref op, .. } if op == "same_as"
        ));

        let deref = lower_std_primitive_op(
            call(
                CallTarget::method("deref", Some("MutexGuard".into())),
                vec![v],
                ValueType::Ref(None),
            ),
            Some("sync::poison::mutex::MutexGuard"),
            None,
            None,
            None,
            false,
            true,
        );
        assert!(matches!(
            deref,
            OpKind::UnaryOp { ref op, .. } if op == "same_as"
        ));
    }

    #[test]
    fn word_sized_clone_and_default_lower_to_identity_and_zero() {
        let v = dummy_var();
        let clone = lower_std_primitive_op(
            call(
                path(&["core", "clone", "impls", "<Impl>", "clone"]),
                vec![v.clone()],
                ValueType::Int,
            ),
            None,
            None,
            Some("I64"),
            None,
            false,
            true,
        );
        assert!(matches!(
            clone,
            OpKind::UnaryOp { ref op, .. } if op == "same_as"
        ));

        let default_i = lower_std_primitive_op(
            call(
                path(&["core", "default", "<Impl>", "default"]),
                vec![],
                ValueType::Int,
            ),
            None,
            None,
            Some("I64"),
            None,
            false,
            true,
        );
        assert!(matches!(default_i, OpKind::ConstInt(0)));

        let default_u = lower_std_primitive_op(
            call(
                path(&["core", "default", "<Impl>", "default"]),
                vec![],
                ValueType::Unsigned,
            ),
            None,
            None,
            None,
            Some("Usize"),
            false,
            true,
        );
        assert!(matches!(default_u, OpKind::ConstUInt(0)));

        let default_bool = lower_std_primitive_op(
            call(
                path(&["core", "default", "<Impl>", "default"]),
                vec![],
                ValueType::Bool,
            ),
            None,
            None,
            None,
            None,
            true,
            true,
        );
        assert!(matches!(default_bool, OpKind::ConstBool(false)));
    }

    #[test]
    fn adt_clone_default_and_mem_replace_stay_residual() {
        let v = dummy_var();
        let clone_adt = lower_std_primitive_op(
            call(
                CallTarget::method("clone", Some("W_ListObject".into())),
                vec![v.clone()],
                ValueType::Ref(Some("W_ListObject".into())),
            ),
            Some("pyre_object::listobject::W_ListObject"),
            Some("pyre_object::listobject::W_ListObject"),
            None,
            None,
            false,
            true,
        );
        assert!(matches!(clone_adt, OpKind::Call { .. }));

        let tuple_default = lower_std_primitive_op(
            call(
                path(&["core", "tuple", "<Impl>", "default"]),
                vec![],
                ValueType::Ref(None),
            ),
            None,
            Some("Tuple"),
            None,
            None,
            false,
            true,
        );
        assert!(matches!(tuple_default, OpKind::Call { .. }));

        let replace = call(
            path(&["core", "mem", "replace"]),
            vec![v.clone(), dummy_var()],
            ValueType::Int,
        );
        let replace = lower_std_primitive_op(replace, None, None, Some("I64"), None, false, true);
        assert!(
            matches!(replace, OpKind::Call { .. }),
            "replace cannot lower without the borrowed Place"
        );
    }

    #[test]
    fn i32_clone_is_not_word_sized_copy() {
        let v = dummy_var();
        let clone = lower_std_primitive_op(
            call(
                path(&["core", "clone", "impls", "<Impl>", "clone"]),
                vec![v],
                ValueType::Int,
            ),
            None,
            None,
            Some("I32"),
            None,
            false,
            true,
        );
        assert!(
            matches!(clone, OpKind::Call { .. }),
            "i32 is not the machine-word Copy width"
        );
    }

    /// The blanket impl and the trait item share the leaf name with the
    /// concrete `Copy` impls and arrive at the machine word width, so a
    /// leaf-name match folds them too. `mir.rs` guards the same pair
    /// through `lower_function`; this pins the predicate itself.
    #[test]
    fn the_blanket_clone_and_the_default_trait_item_stay_residual() {
        let v = dummy_var();
        let blanket = lower_std_primitive_op(
            call(
                path(&["core", "clone", "<Impl>", "clone"]),
                vec![v],
                ValueType::Int,
            ),
            None,
            None,
            Some("I64"),
            None,
            false,
            true,
        );
        assert!(
            matches!(blanket, OpKind::Call { .. }),
            "core::clone::<Impl>::clone is not the impls identity"
        );

        let trait_item = lower_std_primitive_op(
            call(
                path(&["core", "default", "Default", "default"]),
                vec![],
                ValueType::Int,
            ),
            None,
            None,
            Some("I64"),
            None,
            false,
            true,
        );
        assert!(
            matches!(trait_item, OpKind::Call { .. }),
            "Default::default names no impl to read a zero from"
        );
    }

    #[test]
    fn clone_and_default_targets_use_the_regular_call_predicates() {
        let clone_impl = path(&["core", "clone", "impls", "<Impl>", "clone"]);
        let blanket = path(&["core", "clone", "<Impl>", "clone"]);
        let default_impl = path(&["core", "default", "<Impl>", "default"]);
        let ptr_default = path(&["core", "ptr", "mut_ptr", "<Impl>", "default"]);
        let trait_item = path(&["core", "default", "Default", "default"]);
        assert!(is_clone_target(&clone_impl));
        assert!(!is_clone_target(&blanket));
        assert!(is_default_target(&default_impl));
        assert!(is_default_target(&ptr_default));
        assert!(!is_default_target(&trait_item));
        assert_eq!(
            is_clone_target(&clone_impl),
            crate::front::mir::is_core_clone_impls_clone_path("core::clone::impls::<Impl>::clone")
        );
        assert_eq!(
            is_default_target(&ptr_default),
            crate::front::mir::is_core_default_path("core::ptr::mut_ptr::<Impl>::default")
        );
    }
}
