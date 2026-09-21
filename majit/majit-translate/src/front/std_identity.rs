//! Transparent one-field std wrappers as identity, not residual calls.
//!
//! RPython has no `Cell` / `Ref` / `MutexGuard` / `Box` / `AtomicUsize`
//! types: the field *is* the value.  Charon still emits residual
//! `FunctionPath` / `Method` calls to those bodies (Opaque in the LLBC),
//! and every caller then dies at `translate_op` with an unregistered
//! `CallRegistry` miss.  The call is not a graph and not a foreign
//! function — it is `same_as`.
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

/// Rewrite a residual std wrapper call in place when the wrapper shape
/// is already in hand.  Unknown shapes are returned unchanged.
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
            true,
        );
        assert!(matches!(
            deref,
            OpKind::UnaryOp { ref op, .. } if op == "same_as"
        ));
    }

    #[test]
    fn mem_replace_stays_residual() {
        let v = dummy_var();
        let replace = call(
            path(&["core", "mem", "replace"]),
            vec![v, dummy_var()],
            ValueType::Int,
        );
        let replace = lower_std_primitive_op(replace, None, None, true);
        assert!(
            matches!(replace, OpKind::Call { .. }),
            "replace cannot lower without the borrowed Place"
        );
    }
}
