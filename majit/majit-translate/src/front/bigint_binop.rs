//! `<BigInt as {BitAnd,BitOr,BitXor,Sub,Mul}>::op()` → `jit_bigint_*` residual.
//!
//! ## Positioning
//!
//! malachite's operator impls (`impl BitAnd for &BigInt`, …) have Opaque bodies
//! in the LLBC, so a traced-into caller (`bigint_truediv`, which is `#[elidable]`
//! and therefore walked) emits a residual `<Impl>` FunctionPath call — an
//! unregistered callee the rtyper census Skips.  Unlike a method-syntax call
//! (`num.div_rem(&den)` → `["bigint","BigInt","div_rem"]`, owner resolved), an
//! operator desugars to `<BigInt as Trait>::op(a, b)` whose path keeps the
//! unresolved `<Impl>` segment, so the harvester's resolved-owner key never
//! matches and the call stays unregistered.
//!
//! Each such operator returns a single owned `BigInt`, which the front models as
//! the classdef-less `*mut BigInt` GcRef — ABI-identical to the
//! `#[dont_look_inside] jit_bigint_*(a: i64, b: i64) -> i64` residuals.  So the
//! fix is a pure **call-target retarget** (no aggregate, no control flow): swap
//! the `<Impl>::op` target for the residual path, keep the operand args and the
//! result var.  `front::mir` performs the swap in place while lowering the Call
//! op, guarded on **both operands resolving to the opaque `BigInt` ADT**, so a
//! same-named operator on any other type is never mis-retargeted.  This module
//! owns the (leaf → residual path) mapping the guard consults.
//!
//! Sibling of [`crate::front::bigint_div_rem`]; the retarget is simpler because
//! the result is a single Ref rather than a `(BigInt, BigInt)` tuple.

/// The `#[dont_look_inside]` residual's module path (in `pyre-interpreter`),
/// matching its `jit_fnaddr` binding.
const RESIDUAL_MODULE: [&str; 3] = ["pyre_interpreter", "objspace", "descroperation"];

/// If `segments` is a foreign BigInt binary-operator impl-method path
/// (`[…, "<Impl>", op]` for one of the retargetable operators), return the
/// fully-qualified `jit_bigint_*` residual path to retarget it to; otherwise
/// `None`.  The caller separately confirms the operands are `BigInt` before
/// applying the retarget — this only classifies the operator leaf.
pub(crate) fn bigint_binop_residual_path(segments: &[String]) -> Option<Vec<String>> {
    // The impl-method form is `[…, "<Impl>", <op>]`: the operator leaf preceded
    // by the unresolved `<Impl>` owner segment.
    let [.., impl_seg, leaf] = segments else {
        return None;
    };
    if impl_seg != "<Impl>" {
        return None;
    }
    let residual_leaf = match leaf.as_str() {
        "bitand" => "jit_bigint_and",
        "bitor" => "jit_bigint_or",
        "bitxor" => "jit_bigint_xor",
        "sub" => "jit_bigint_sub",
        "mul" => "jit_bigint_mul",
        "add" => "jit_bigint_add",
        _ => return None,
    };
    Some(residual_path(residual_leaf))
}

/// If `segments` is a foreign BigInt **shift** impl-method path (`[…,
/// "<Impl>", {"shl"|"shr"}]`), return the `jit_bigint_{shl,shr}` residual
/// path; otherwise `None`.  Shifts are split from [`bigint_binop_residual_path`]
/// because the shift amount is a machine integer (`usize`), not a `BigInt`: the
/// caller confirms the first operand is `BigInt` and the second is an integer
/// (so the residual reads `b` as the count, not a pointer).
pub(crate) fn bigint_shift_residual_path(segments: &[String]) -> Option<Vec<String>> {
    let [.., impl_seg, leaf] = segments else {
        return None;
    };
    if impl_seg != "<Impl>" {
        return None;
    }
    let residual_leaf = match leaf.as_str() {
        "shl" => "jit_bigint_shl",
        "shr" => "jit_bigint_shr",
        _ => return None,
    };
    Some(residual_path(residual_leaf))
}

/// Build the fully-qualified residual path for a `jit_bigint_*` leaf.
fn residual_path(leaf: &str) -> Vec<String> {
    let mut path: Vec<String> = RESIDUAL_MODULE.iter().map(|s| s.to_string()).collect();
    path.push(leaf.to_string());
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segs(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|s| s.to_string()).collect()
    }

    fn desc(residual: &str) -> Vec<String> {
        segs(&["pyre_interpreter", "objspace", "descroperation", residual])
    }

    #[test]
    fn maps_each_retargetable_operator_to_its_residual() {
        for (op, residual) in [
            ("bitand", "jit_bigint_and"),
            ("bitor", "jit_bigint_or"),
            ("bitxor", "jit_bigint_xor"),
            ("sub", "jit_bigint_sub"),
            ("mul", "jit_bigint_mul"),
            ("add", "jit_bigint_add"),
        ] {
            let path =
                bigint_binop_residual_path(&segs(&["malachite_bigint", "bigint", "<Impl>", op]))
                    .unwrap_or_else(|| panic!("{op} must map"));
            assert_eq!(path, desc(residual));
        }
    }

    #[test]
    fn maps_each_shift_operator_to_its_residual() {
        for (op, residual) in [("shl", "jit_bigint_shl"), ("shr", "jit_bigint_shr")] {
            let path =
                bigint_shift_residual_path(&segs(&["malachite_bigint", "bigint", "<Impl>", op]))
                    .unwrap_or_else(|| panic!("{op} must map"));
            assert_eq!(path, desc(residual));
        }
        // Shifts are not in the two-BigInt-operand map, and vice versa.
        assert!(
            bigint_binop_residual_path(&segs(&["malachite_bigint", "bigint", "<Impl>", "shr"]))
                .is_none()
        );
        assert!(
            bigint_shift_residual_path(&segs(&["malachite_bigint", "bigint", "<Impl>", "sub"]))
                .is_none()
        );
    }

    #[test]
    fn declines_non_impl_and_unlisted_operators() {
        // Method-syntax (owner resolved, no `<Impl>`) is handled elsewhere.
        assert!(bigint_binop_residual_path(&segs(&["bigint", "BigInt", "sub"])).is_none());
        // An operator not in either retarget set (e.g. `rem`).
        assert!(
            bigint_binop_residual_path(&segs(&["malachite_bigint", "bigint", "<Impl>", "rem"]))
                .is_none()
        );
        // Too short / no `<Impl>` segment.
        assert!(bigint_binop_residual_path(&segs(&["sub"])).is_none());
        assert!(bigint_shift_residual_path(&segs(&["shr"])).is_none());
    }
}
