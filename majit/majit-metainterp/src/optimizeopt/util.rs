//! RPython parity module for `rpython/jit/metainterp/optimizeopt/util.py`.
//!
//! PyPy's dispatcher builders are compile-time `make_dispatcher_method`
//! helpers. In Rust those dispatch tables are static `match` / method calls in
//! each optimization pass, so this module exposes the data-structure helpers
//! that remain meaningful at the Rust type level.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use majit_ir::box_ref::BoxRef;

/// util.py:93-96 `get_box_replacement`.
pub fn get_box_replacement(op: Option<&BoxRef>) -> Option<BoxRef> {
    op.map(|op| op.get_box_replacement(false))
}

/// util.py:100-111 `args_eq`.
///
/// Uses `same_box`, so constants compare by value while regular boxes compare
/// by object identity.
pub fn args_eq(args1: &[Option<BoxRef>], args2: &[Option<BoxRef>]) -> bool {
    args1.len() == args2.len()
        && args1
            .iter()
            .zip(args2)
            .all(|(arg1, arg2)| match (arg1, arg2) {
                (None, None) => true,
                (Some(arg1), Some(arg2)) => arg1.same_box(arg2),
                _ => false,
            })
}

/// util.py:113-122 `args_hash`.
///
/// The exact Python integer hash width is not part of Rust's hash API; this
/// preserves the load-bearing contract with `args_eq`: equal argument lists
/// produce equal hashes, with constants hashed by value and other boxes by
/// identity.
pub fn args_hash(args: &[Option<BoxRef>]) -> u64 {
    let mut state = DefaultHasher::new();
    0x345678_u64.hash(&mut state);
    for arg in args {
        match arg {
            None => 17_u8.hash(&mut state),
            Some(arg) => hash_arg(arg, &mut state),
        }
    }
    state.finish()
}

fn hash_arg<H: Hasher>(arg: &BoxRef, state: &mut H) {
    if let Some(value) = arg.const_value() {
        value.hash(state);
    } else {
        arg.hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::{args_eq, args_hash, get_box_replacement};
    use majit_ir::Value;
    use majit_ir::box_ref::BoxRef;

    #[test]
    fn args_eq_uses_same_box_const_value_semantics() {
        let a = Some(BoxRef::new_const(Value::Int(5)));
        let b = Some(BoxRef::new_const(Value::Int(5)));
        assert!(args_eq(&[a.clone()], &[b.clone()]));
        assert_eq!(args_hash(&[a]), args_hash(&[b]));
    }

    #[test]
    fn args_eq_distinguishes_non_const_box_identity() {
        let a = Some(BoxRef::new_resop(majit_ir::Type::Int, 0));
        let b = Some(BoxRef::new_resop(majit_ir::Type::Int, 0));
        assert!(!args_eq(&[a], &[b]));
    }

    #[test]
    fn get_box_replacement_preserves_none() {
        assert!(get_box_replacement(None).is_none());
    }
}
