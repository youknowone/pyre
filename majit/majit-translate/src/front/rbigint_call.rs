//! RPython `rbigint.fromint`/`from_u128` call → GC-reference residual.
//!
//! RPython represents every `rbigint` value as one GC reference.  Rust's
//! `RBigInt` source struct is intentionally available to the translator, but
//! its native by-value return ABI is not the JIT ABI.  Calls that construct an
//! `RBigInt` from a machine word therefore target a `dont_look_inside` wrapper
//! returning `*mut RBigInt`, exactly like the arithmetic residuals in
//! [`crate::front::bigint_binop`].

const RESIDUAL_MODULE: [&str; 2] = ["pyre_object", "longobject"];
const RAISING_SCALAR_RESIDUAL_MODULE: [&str; 3] =
    ["pyre_interpreter", "objspace", "descroperation"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ScalarResult {
    Int,
    Bool,
}

/// Return the pointer-ABI constructor residual for an RBigInt constructor call.
///
/// The caller proves the destination is the exact local `RBigInt` ADT and that
/// the source is a supported machine-word integer.  `i128`/`u128` deliberately
/// decline: RPython's JIT has no longlonglong register kind, so those paths
/// must not be silently narrowed to a word.
pub(crate) fn constructor_residual_path(
    segments: &[String],
    unsigned: bool,
) -> Option<Vec<String>> {
    let leaf = segments.last()?.as_str();
    let matches_constructor = match leaf {
        "from" => true,
        "fromint" => !unsigned,
        // RPython's JIT has no longlonglong register kind. Keep the decline
        // local to this mapper instead of relying on the caller's width gate.
        "from_u128" => return None,
        _ => false,
    };
    if !matches_constructor
        || !segments
            .windows(2)
            .any(|pair| pair == ["rbigint", "RBigInt"])
    {
        return None;
    }
    let mut path: Vec<String> = RESIDUAL_MODULE
        .iter()
        .map(|part| part.to_string())
        .collect();
    path.push(
        if unsigned {
            "jit_bigint_from_u64"
        } else {
            "jit_bigint_from_i64"
        }
        .to_string(),
    );
    Some(path)
}

/// Translate Rust's shallow handle clone to a fresh one-GC-reference payload.
///
/// The caller proves both the receiver and destination are the exact local
/// RBigInt ADT. The residual preserves `Clone`'s shared immutable digit array
/// while giving the returned by-value handle its own translated GC object.
pub(crate) fn clone_residual_for_method(leaf: &str) -> Option<Vec<String>> {
    if leaf != "clone" {
        return None;
    }
    Some(
        ["pyre_object", "longobject", "jit_bigint_clone"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
}

/// Bare-payload residual for scalar RBigInt queries. The caller has already
/// proved exact RBigInt receiver identity.
pub(crate) fn scalar_residual_for_method(leaf: &str) -> Option<(Vec<String>, ScalarResult)> {
    let (module, residual, result) = match leaf {
        "get_sign" => (
            RESIDUAL_MODULE.as_slice(),
            "jit_bigint_sign_i64",
            ScalarResult::Int,
        ),
        "bits" => (
            RESIDUAL_MODULE.as_slice(),
            "jit_bigint_bits",
            ScalarResult::Int,
        ),
        "is_zero" => (
            RESIDUAL_MODULE.as_slice(),
            "jit_bigint_is_zero",
            ScalarResult::Bool,
        ),
        "is_one" => (
            RESIDUAL_MODULE.as_slice(),
            "jit_bigint_is_one",
            ScalarResult::Bool,
        ),
        "fits_int" => (
            RESIDUAL_MODULE.as_slice(),
            "jit_bigint_to_i64_fits",
            ScalarResult::Bool,
        ),
        "tobool" => (
            RESIDUAL_MODULE.as_slice(),
            "jit_bigint_tobool",
            ScalarResult::Bool,
        ),
        "hash" => (
            RESIDUAL_MODULE.as_slice(),
            "jit_bigint_hash",
            ScalarResult::Int,
        ),
        "bit_length" => (
            RAISING_SCALAR_RESIDUAL_MODULE.as_slice(),
            "jit_bigint_bit_length",
            ScalarResult::Int,
        ),
        "bit_count" => (
            RAISING_SCALAR_RESIDUAL_MODULE.as_slice(),
            "jit_bigint_bit_count",
            ScalarResult::Int,
        ),
        _ => return None,
    };
    let mut path: Vec<String> = module.iter().map(|part| part.to_string()).collect();
    path.push(residual.to_string());
    Some((path, result))
}

/// Bare-payload residual for an RBigInt operation specialized for one signed
/// machine-word operand.  PyPy's W_LongObject descriptors call these exact
/// `rbigint.int_*` methods instead of first allocating an rbigint for the
/// W_IntObject.  The MIR caller proves the receiver is RBigInt and the second
/// argument is exactly i64 before applying this mapping.
pub(crate) fn int_binop_residual_for_method(leaf: &str) -> Option<Vec<String>> {
    let residual = match leaf {
        "int_add" => "jit_bigint_int_add",
        "int_sub" => "jit_bigint_int_sub",
        "int_mul" => "jit_bigint_int_mul",
        "int_and_" => "jit_bigint_int_and",
        "int_or_" => "jit_bigint_int_or",
        "int_xor" => "jit_bigint_int_xor",
        _ => return None,
    };
    Some(
        ["pyre_interpreter", "objspace", "descroperation", residual]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
}

/// Scalar-bool counterpart of [`int_binop_residual_for_method`] for PyPy's
/// mixed W_LongObject/W_IntObject rich comparisons.
pub(crate) fn int_comparison_residual_for_method(leaf: &str) -> Option<Vec<String>> {
    let residual = match leaf {
        "int_eq" => "jit_bigint_int_eq",
        "int_ne" => "jit_bigint_int_ne",
        "int_lt" => "jit_bigint_int_lt",
        "int_le" => "jit_bigint_int_le",
        "int_gt" => "jit_bigint_int_gt",
        "int_ge" => "jit_bigint_int_ge",
        _ => return None,
    };
    Some(
        ["pyre_interpreter", "objspace", "descroperation", residual]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
}

/// `descroperation::bigint_pow_nomod(&RBigInt, &RBigInt) ->
/// Result<RBigInt, PyError>` is Rust's source-level spelling of RPython's
/// elidable `rbigint.pow` call with an implicit MemoryError edge.  The caller
/// rule in `result_exc` removes the `Result` diamond; this target swap supplies
/// the matching one-GC-reference result ABI.
pub(crate) fn pow_nomod_residual_path(segments: &[String]) -> Option<Vec<String>> {
    if !segments.ends_with(&[
        "objspace".to_string(),
        "descroperation".to_string(),
        "bigint_pow_nomod".to_string(),
    ]) {
        return None;
    }
    Some(
        [
            "pyre_interpreter",
            "objspace",
            "descroperation",
            "jit_bigint_pow_nomod",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
    )
}

/// `descroperation::bigint_lshift_count(&RBigInt, i64) ->
/// Result<RBigInt, PyError>` carries RPython `rbigint.lshift`'s implicit
/// MemoryError through Rust source. `result_exc` removes that carrier after
/// this target swap supplies the one-GC-reference residual result.
pub(crate) fn lshift_count_residual_path(segments: &[String]) -> Option<Vec<String>> {
    if !segments.ends_with(&[
        "objspace".to_string(),
        "descroperation".to_string(),
        "bigint_lshift_count".to_string(),
    ]) {
        return None;
    }
    Some(
        [
            "pyre_interpreter",
            "objspace",
            "descroperation",
            "jit_bigint_lshift_count",
        ]
        .into_iter()
        .map(str::to_string)
        .collect(),
    )
}

/// RustPython's compiler exposes integer constants as an opaque foreign
/// BigInt.  Retarget the one sanctioned conversion seam to a pointer-returning
/// pure residual so Malachite never enters the translated arithmetic graph.
pub(crate) fn compiler_bigint_residual_path(segments: &[String]) -> Option<Vec<String>> {
    if !segments.ends_with(&[
        "pyre_interpreter".to_string(),
        "compiler_bigint_to_rbigint".to_string(),
    ]) {
        return None;
    }
    Some(
        ["pyre_interpreter", "jit_compiler_bigint_to_rbigint"]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
}

/// Zero-checked interpreter seams for the two projections of
/// `rbigint.divmod`. Each returns a direct RBigInt handle at source level, so
/// the target swap is ABI-identical to the other pointer residual mappings.
pub(crate) fn divmod_projection_residual_path(segments: &[String]) -> Option<Vec<String>> {
    let residual = match segments.last().map(String::as_str) {
        Some("bigint_floordiv_nonzero") => "jit_bigint_div_floor",
        Some("bigint_modulo_nonzero") => "jit_bigint_mod_floor",
        _ => return None,
    };
    if !segments
        .iter()
        .rev()
        .skip(1)
        .take(2)
        .eq(["descroperation", "objspace"])
    {
        return None;
    }
    Some(
        ["pyre_interpreter", "objspace", "descroperation", residual]
            .into_iter()
            .map(str::to_string)
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segs(parts: &[&str]) -> Vec<String> {
        parts.iter().map(|part| part.to_string()).collect()
    }

    #[test]
    fn maps_signed_and_unsigned_word_constructors() {
        assert_eq!(
            constructor_residual_path(&segs(&["rbigint", "RBigInt", "fromint"]), false),
            Some(segs(&["pyre_object", "longobject", "jit_bigint_from_i64"]))
        );
        assert_eq!(
            constructor_residual_path(&segs(&["rbigint", "RBigInt", "from"]), true),
            Some(segs(&["pyre_object", "longobject", "jit_bigint_from_u64"]))
        );
    }

    #[test]
    fn declines_other_types_and_wrong_signedness() {
        assert!(constructor_residual_path(&segs(&["other", "RBigInt", "from"]), false).is_none());
        assert!(
            constructor_residual_path(&segs(&["rbigint", "RBigInt", "fromint"]), true).is_none()
        );
        assert!(
            constructor_residual_path(&segs(&["rbigint", "RBigInt", "from_u128"]), true).is_none()
        );
        assert!(constructor_residual_path(&segs(&["rbigint", "RBigInt", "add"]), false).is_none());
    }

    #[test]
    fn maps_only_the_exact_clone_method() {
        assert_eq!(
            clone_residual_for_method("clone"),
            Some(segs(&["pyre_object", "longobject", "jit_bigint_clone"]))
        );
        assert!(clone_residual_for_method("clone_from").is_none());
    }

    #[test]
    fn maps_scalar_queries_with_their_result_bank() {
        assert_eq!(
            scalar_residual_for_method("get_sign"),
            Some((
                segs(&["pyre_object", "longobject", "jit_bigint_sign_i64"]),
                ScalarResult::Int
            ))
        );
        assert_eq!(
            scalar_residual_for_method("is_zero"),
            Some((
                segs(&["pyre_object", "longobject", "jit_bigint_is_zero"]),
                ScalarResult::Bool
            ))
        );
        assert_eq!(
            scalar_residual_for_method("bit_length"),
            Some((
                segs(&[
                    "pyre_interpreter",
                    "objspace",
                    "descroperation",
                    "jit_bigint_bit_length",
                ]),
                ScalarResult::Int
            ))
        );
        assert!(scalar_residual_for_method("add").is_none());
    }

    #[test]
    fn maps_each_machine_int_specialized_operation() {
        for (method, residual) in [
            ("int_add", "jit_bigint_int_add"),
            ("int_sub", "jit_bigint_int_sub"),
            ("int_mul", "jit_bigint_int_mul"),
            ("int_and_", "jit_bigint_int_and"),
            ("int_or_", "jit_bigint_int_or"),
            ("int_xor", "jit_bigint_int_xor"),
        ] {
            assert_eq!(
                int_binop_residual_for_method(method),
                Some(segs(&[
                    "pyre_interpreter",
                    "objspace",
                    "descroperation",
                    residual,
                ]))
            );
        }
        assert!(int_binop_residual_for_method("add").is_none());
        assert!(int_binop_residual_for_method("int_floordiv").is_none());
    }

    #[test]
    fn maps_each_machine_int_specialized_comparison() {
        for (method, residual) in [
            ("int_eq", "jit_bigint_int_eq"),
            ("int_ne", "jit_bigint_int_ne"),
            ("int_lt", "jit_bigint_int_lt"),
            ("int_le", "jit_bigint_int_le"),
            ("int_gt", "jit_bigint_int_gt"),
            ("int_ge", "jit_bigint_int_ge"),
        ] {
            assert_eq!(
                int_comparison_residual_for_method(method),
                Some(segs(&[
                    "pyre_interpreter",
                    "objspace",
                    "descroperation",
                    residual,
                ]))
            );
        }
        assert!(int_comparison_residual_for_method("lt").is_none());
        assert!(int_comparison_residual_for_method("int_add").is_none());
    }

    #[test]
    fn maps_nomod_pow_result_seam_to_pointer_abi() {
        assert_eq!(
            pow_nomod_residual_path(&segs(&[
                "pyre_interpreter",
                "objspace",
                "descroperation",
                "bigint_pow_nomod",
            ])),
            Some(segs(&[
                "pyre_interpreter",
                "objspace",
                "descroperation",
                "jit_bigint_pow_nomod",
            ]))
        );
        assert!(
            pow_nomod_residual_path(&segs(&["pyre_object", "rbigint", "RBigInt", "pow",]))
                .is_none()
        );
    }

    #[test]
    fn maps_lshift_result_seam_to_pointer_abi() {
        assert_eq!(
            lshift_count_residual_path(&segs(&[
                "pyre_interpreter",
                "objspace",
                "descroperation",
                "bigint_lshift_count",
            ])),
            Some(segs(&[
                "pyre_interpreter",
                "objspace",
                "descroperation",
                "jit_bigint_lshift_count",
            ]))
        );
        assert!(
            lshift_count_residual_path(&segs(&["pyre_object", "rbigint", "RBigInt", "lshift",]))
                .is_none()
        );
    }

    #[test]
    fn maps_only_the_compiler_bigint_conversion_seam() {
        assert_eq!(
            compiler_bigint_residual_path(&segs(&[
                "pyre_interpreter",
                "compiler_bigint_to_rbigint",
            ])),
            Some(segs(&[
                "pyre_interpreter",
                "jit_compiler_bigint_to_rbigint",
            ]))
        );
        assert!(
            compiler_bigint_residual_path(&segs(&["malachite_bigint", "BigInt", "to_bytes_le",]))
                .is_none()
        );
    }

    #[test]
    fn maps_zero_checked_divmod_projections() {
        for (source, residual) in [
            ("bigint_floordiv_nonzero", "jit_bigint_div_floor"),
            ("bigint_modulo_nonzero", "jit_bigint_mod_floor"),
        ] {
            assert_eq!(
                divmod_projection_residual_path(&segs(&[
                    "pyre_interpreter",
                    "objspace",
                    "descroperation",
                    source,
                ])),
                Some(segs(&[
                    "pyre_interpreter",
                    "objspace",
                    "descroperation",
                    residual,
                ]))
            );
        }
    }
}
