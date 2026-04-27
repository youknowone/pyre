//! Codewriter helpers shared by `call.py`, `jtransform.py`, etc.
//!
//! Translated from `rpython/jit/codewriter/support.py`. Only the items
//! actually used by the rest of the parity port live here; the bulk of
//! `support.py` (`LLtypeHelpers`, `OOtypeHelpers`, RPython annotator
//! glue, `MixLevelHelperAnnotator` integration, `parse_oopspec` /
//! `normalize_opargs` / `get_send_oopspec`) is RPython-internal and
//! has no Rust counterpart.
//!
//! **Stub status of the items below** (preserved for parity reviewers):
//!
//! - [`INLINE_CALLS_TO`] is intentionally empty.  RPython's
//!   `find_all_graphs` seeds the BFS from this list so the inliner can
//!   always reach `int_abs` / `int_floordiv` / `int_mod` /
//!   `ll_math.ll_math_sqrt`.  Pyre's `find_all_graphs`
//!   ([`crate::jit_codewriter::call::CallControl::find_all_graphs_bfs`])
//!   already iterates this slice (see the seed loop at `call.rs:1422`);
//!   re-enabling the four entries above only requires landing the
//!   matching `register_function_graph` entries upstream of the BFS.
//! - [`decode_builtin_call`] returns the last segment of the call
//!   target as the oopspec name and an empty arg list.  The upstream
//!   resolution that walks `op.args[0].value._obj` to recover the full
//!   `(oopspec_name, opargs)` pair lives behind RPython's lltype
//!   reflection and is not reproduced here.  Callers needing the real
//!   tuple should use `CallControl::oopspec_for_target` instead.
//! - [`builtin_func_for_spec`] returns a descriptor only; it does NOT
//!   materialize a function pointer or a graph.  Pyre's analogue lives
//!   in `majit-ir::effectinfo::OopSpecIndex` — this helper exists to
//!   keep the upstream call shape compilable.

use crate::model::{CallTarget, OpKind, SpaceOperation};

use majit_ir::value::Type;

/// support.py:444-449 `inline_calls_to`.
///
/// List of `(oopspec_name, ll_args, ll_res)` triples.  RPython's
/// `CallControl.find_all_graphs` seeds the BFS with these so the
/// optimizer can always look inside them.  Pyre lacks distinct
/// graphs for these helpers (they are either inlined as Rust intrinsics
/// or routed through `OopSpecIndex`) so the table is empty in our
/// build, but the named binding stays so `call.py:60-64` can be ported
/// line-by-line when needed.
pub static INLINE_CALLS_TO: &[(&str, &[Type], Type)] = &[
    // ("int_abs",              &[Type::Int], Type::Int),
    // ("int_floordiv",         &[Type::Int, Type::Int], Type::Int),
    // ("int_mod",              &[Type::Int, Type::Int], Type::Int),
    // ("ll_math.ll_math_sqrt", &[Type::Float], Type::Float),
];

/// support.py:755-765 `decode_builtin_call(op)`.
///
/// In RPython this resolves an op to its `(oopspec_name, opargs)` pair.
/// Pyre routes this through `crate::call::CallControl::oopspec_targets`
/// during `jtransform`, so this function is here only as a structural
/// stub: callers that need it should be ported alongside `jtransform`'s
/// builtin family in Phase 5C.
pub fn decode_builtin_call(op: &SpaceOperation) -> Option<(String, Vec<usize>)> {
    match &op.kind {
        OpKind::Call { target, .. } => {
            let segments = match target {
                CallTarget::FunctionPath { segments } => segments,
                _ => return None,
            };
            // The actual oopspec lookup table lives on `CallControl`;
            // a free function like this one cannot reach it without a
            // borrow.  Pyre's call sites use
            // `CallControl::oopspec_for_target` directly; if a future
            // phase needs `decode_builtin_call` itself it can plumb the
            // lookup table through as an argument, mirroring RPython's
            // `op.args[0].value._obj` accessor.
            Some((segments.last().cloned().unwrap_or_default(), Vec::new()))
        }
        _ => None,
    }
}

/// support.py:767-808 `builtin_func_for_spec(rtyper, oopspec_name, ll_args, ll_res, …)`.
///
/// In RPython this takes an oopspec name and signature, materializes a
/// helper function via `MixLevelHelperAnnotator`, and caches the result
/// on the rtyper.  Pyre's analogue is the static [`OopSpecIndex`] table
/// in `majit-ir/src/effectinfo.rs`; we expose `BuiltinFuncSpec` here so
/// the upstream call shape can be reproduced when porting `call.py:60`.
#[derive(Debug, Clone)]
pub struct BuiltinFuncSpec {
    pub oopspec_name: String,
    pub ll_args: Vec<Type>,
    pub ll_res: Type,
}

pub fn builtin_func_for_spec(
    oopspec_name: &str,
    ll_args: &[Type],
    ll_res: Type,
) -> Option<BuiltinFuncSpec> {
    // Pyre does not maintain a function-pointer cache like RPython's
    // `rtyper._builtin_func_for_spec_cache`; instead we return the
    // descriptor unconditionally so callers can build the synthetic
    // graph on demand.  The optional return mirrors RPython, where
    // unknown oopspec names raise.
    if oopspec_name.is_empty() {
        return None;
    }
    Some(BuiltinFuncSpec {
        oopspec_name: oopspec_name.to_string(),
        ll_args: ll_args.to_vec(),
        ll_res,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inline_calls_to_is_empty_in_pyre() {
        // PRE-EXISTING-ADAPTATION: pyre keeps the upstream binding but
        // the table is empty until an oopspec actually needs seeding.
        assert!(INLINE_CALLS_TO.is_empty());
    }

    #[test]
    fn builtin_func_for_spec_round_trip() {
        let spec = builtin_func_for_spec("int_abs", &[Type::Int], Type::Int).unwrap();
        assert_eq!(spec.oopspec_name, "int_abs");
        assert_eq!(spec.ll_args, &[Type::Int]);
        assert_eq!(spec.ll_res, Type::Int);
    }

    #[test]
    fn builtin_func_for_spec_rejects_empty_name() {
        assert!(builtin_func_for_spec("", &[], Type::Void).is_none());
    }
}
