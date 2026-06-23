//! Harvest JIT-hint markers from the ullbc surrogate consts the
//! `majit_macros` proc-macros emit (`_elidable_function_<NAME>`,
//! `_jit_look_inside_<NAME>`, `_jit_loop_invariant_<NAME>`,
//! `_jit_unroll_safe_<NAME>`, `oopspec_<NAME>`).
//!
//! The source attribute (`#[elidable]` / `#[dont_look_inside]` / …) is
//! consumed by the proc-macro at expansion time and does NOT survive in
//! Charon's `attr_info`.  The macro instead leaves these `#[doc(hidden)]`
//! marker consts next to the user function, and Charon extracts them
//! into `global_decls`.  Reading them back is the analog of RPython's
//! translator reading `func._elidable_function_` off the function
//! object.
//!
//! The harvested map is keyed and ordered so that `merge_hints_from_llbcs`
//! can apply the hints to each function order- and multiplicity-exact.

use majit_charon_reader::{
    Llbc,
    ullbc::{GlobalDecl, Operand, PlaceKind, Rvalue, StmtKind},
};
use std::collections::HashMap;

/// Marker-const name prefix → the JIT hint strings it implies.  The
/// user function's leaf name is the const leaf with the prefix stripped.
///
/// This is the inverse of `majit_macros::rpython_attribute_const_for`.
/// `_jit_look_inside_` is handled separately because the same marker
/// prefix carries a bool value (`true` = `jit_look_inside`, `false` =
/// `dont_look_inside`).
const CONST_PREFIX_HINTS: &[(&str, &[&str])] = &[
    ("_elidable_function_", &["elidable"]),
    ("_jit_elidable_cannot_raise_", &["elidable_cannot_raise"]),
    ("_jit_elidable_or_memerror_", &["elidable_or_memerror"]),
    ("_jit_loop_invariant_", &["loopinvariant"]),
    ("_jit_unroll_safe_", &["unroll_safe"]),
];

/// Build a `{crate_stripped_fn_path → sorted-deduped hints}` map from
/// the marker consts present in `llbcs`.
pub fn harvest_hints_from_llbcs(llbcs: &[Llbc]) -> HashMap<String, Vec<String>> {
    let mut out: HashMap<String, Vec<String>> = HashMap::new();
    for llbc in llbcs {
        for gd in llbc.iter_global_decls() {
            let path = gd.item_meta.name_path();
            let leaf = path.rsplit("::").next().unwrap_or(path.as_str());
            if let Some(fn_leaf) = leaf.strip_prefix("_jit_look_inside_") {
                if should_skip_generated_elidable_helper(fn_leaf) {
                    continue;
                }
                // The marker const exists (its name routed control here),
                // so its bool initializer must decode.  Mirror policy.py:56
                // honouring the exact `_jit_look_inside_` value rather than
                // silently coercing an undecodable marker to
                // `dont_look_inside`: a `None` here means the marker is
                // present but its literal bool could not be read, which is a
                // decoder/encoding fault, not a `False`.
                let look_inside = global_marker_bool(llbc, gd).unwrap_or_else(|| {
                    panic!(
                        "_jit_look_inside_ marker `{path}` has an undecodable bool \
                         initializer; the macro emits a literal bool, so this signals \
                         a Charon encoding change"
                    )
                });
                let hint = if look_inside {
                    "jit_look_inside"
                } else {
                    "dont_look_inside"
                };
                push_hint(
                    &mut out,
                    marker_path_to_fn_path(&path, "_jit_look_inside_"),
                    hint,
                );
                continue;
            }
            // `#[oopspec("spec")]` emits `oopspec_<NAME>: &'static str =
            // "spec"` (majit_macros::oopspec, rlib/jit.py:255 `func.oopspec
            // = spec`).  Unlike the fixed-hint markers, the payload is the
            // const's *value*, so decode the string literal and emit a
            // companion `oopspec:<spec>` hint that `lib.rs` consumes via
            // `CallControl::mark_oopspec` — which `guess_call_kind` then
            // classifies as `CallKind::Builtin` (call.py:135-136).
            if leaf.starts_with("oopspec_") {
                // Fail fast like the `_jit_look_inside_` arm above: the macro
                // emits a literal string, so an undecodable initializer is a
                // marker encoding/decoder drift, not an absent spec. Silently
                // dropping the hint would quietly disable oopspec lowering for
                // the function path.
                let spec = global_marker_str(llbc, gd).unwrap_or_else(|| {
                    panic!(
                        "oopspec marker `{path}` has an undecodable string \
                         initializer; this signals a marker encoding/decoder drift"
                    )
                });
                // Only the spec string is harvested, not a companion
                // `oopspec_argnames`.  The support layer's `parse_oopspec`
                // splits the spec's argument list positionally, which is exact
                // for the current specs (every arg is a bare parameter name in
                // declaration order).  A spec that referenced its parameters
                // out of order or by expression would need the argnames map to
                // resolve names→positions; emit it here when such a spec lands.
                push_hint(
                    &mut out,
                    marker_path_to_fn_path(&path, "oopspec_"),
                    &format!("oopspec:{spec}"),
                );
                continue;
            }
            for (prefix, hints) in CONST_PREFIX_HINTS {
                if let Some(fn_name) = leaf.strip_prefix(prefix) {
                    // `elidable_promote` emits a synthetic `_orig_<name>_unlikely_name`
                    // helper carrying `_elidable_function_`.  This generated fn is not
                    // a user function, so skip it and only harvest hints for the
                    // user-written functions.
                    if should_skip_generated_elidable_helper(fn_name) {
                        continue;
                    }
                    let key = marker_path_to_fn_path(&path, prefix);
                    for hint in *hints {
                        push_hint(&mut out, key.clone(), hint);
                    }
                }
            }
        }
    }
    for v in out.values_mut() {
        v.sort();
        v.dedup();
    }
    out
}

fn should_skip_generated_elidable_helper(fn_name: &str) -> bool {
    fn_name.starts_with("_orig_") && fn_name.ends_with("_unlikely_name")
}

fn push_hint(out: &mut HashMap<String, Vec<String>>, key: String, hint: &str) {
    out.entry(key).or_default().push(hint.to_string());
}

fn marker_path_to_fn_path(marker_path: &str, prefix: &str) -> String {
    let stripped = strip_crate_prefix(marker_path);
    match stripped.rsplit_once("::") {
        Some((module, leaf)) => format!("{module}::{}", leaf.strip_prefix(prefix).unwrap_or(leaf)),
        None => stripped
            .strip_prefix(prefix)
            .unwrap_or(&stripped)
            .to_string(),
    }
}

fn strip_crate_prefix(path: &str) -> String {
    let mut parts = path.split("::");
    match (parts.next(), parts.next()) {
        (Some(_crate), Some(second)) => {
            let mut out = String::from(second);
            for part in parts {
                out.push_str("::");
                out.push_str(part);
            }
            out
        }
        _ => path.to_string(),
    }
}

fn global_marker_bool(llbc: &Llbc, gd: &GlobalDecl) -> Option<bool> {
    let init_id = gd.rest.get("init")?.as_u64()?;
    let body = llbc.fn_by_id(init_id)?.unstructured()?;
    for block in &body.body {
        for stmt in &block.statements {
            let StmtKind::Assign(place, Rvalue::Use(Operand::Const(value))) =
                stmt.stmt_kind().ok()?
            else {
                continue;
            };
            if !matches!(place.kind, PlaceKind::Local(0)) {
                continue;
            }
            if let Some(b) = decode_bool_const(&value) {
                return Some(b);
            }
        }
    }
    None
}

/// Read the `&'static str` value of an oopspec marker const. The init
/// body assigns `Local(0) = Const(ConstantExpr)` whose `kind` nests the
/// string literal (`{"kind": {"Literal": {"Str": spec}}, "ty": …}`),
/// mirroring [`global_marker_bool`]'s bool path.
fn global_marker_str(llbc: &Llbc, gd: &GlobalDecl) -> Option<String> {
    let init_id = gd.rest.get("init")?.as_u64()?;
    let body = llbc.fn_by_id(init_id)?.unstructured()?;
    for block in &body.body {
        for stmt in &block.statements {
            let StmtKind::Assign(place, Rvalue::Use(Operand::Const(value))) =
                stmt.stmt_kind().ok()?
            else {
                continue;
            };
            if !matches!(place.kind, PlaceKind::Local(0)) {
                continue;
            }
            if let Some(s) = decode_str_const(&value) {
                return Some(s);
            }
        }
    }
    None
}

fn decode_str_const(value: &serde_json::Value) -> Option<String> {
    if let Some(s) = value.as_str() {
        return Some(s.to_string());
    }
    let obj = value.as_object()?;
    for key in ["Str", "str"] {
        if let Some(v) = obj.get(key) {
            if let Some(s) = v.as_str() {
                return Some(s.to_string());
            }
        }
    }
    // `ConstantExpr` nests the literal under `kind` →
    // `{"Literal": {"Str": spec}}`; descend through both wrappers.
    for key in ["kind", "Literal"] {
        if let Some(nested) = obj.get(key) {
            if let Some(s) = decode_str_const(nested) {
                return Some(s);
            }
        }
    }
    None
}

fn decode_bool_const(value: &serde_json::Value) -> Option<bool> {
    if let Some(b) = value.as_bool() {
        return Some(b);
    }
    let obj = value.as_object()?;
    for key in ["Bool", "bool"] {
        if let Some(v) = obj.get(key) {
            return v.as_bool();
        }
    }
    if let Some(lit) = obj.get("Literal") {
        if let Some(b) = lit.as_bool() {
            return Some(b);
        }
        if let Some(lit_obj) = lit.as_object() {
            for key in ["Bool", "bool"] {
                if let Some(v) = lit_obj.get(key) {
                    return v.as_bool();
                }
            }
        }
    }
    if let Some(scalar) = obj.get("Scalar").or_else(|| obj.get("scalar")) {
        return decode_bool_const(scalar);
    }
    // A `ConstantExpr` nests the literal under `kind`
    // (`{"kind": {"Literal": {"Bool": b}}, "ty": …}`).  The marker init
    // assigns exactly this shape (`Local(0) = Const(ConstantExpr)`), so
    // descend through `kind` to reach the literal instead of missing it.
    if let Some(kind) = obj.get("kind") {
        return decode_bool_const(kind);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::decode_str_const;

    #[test]
    fn decode_str_const_reads_constant_expr_literal() {
        // The exact shape Charon emits for `const X: &str = "spec"`:
        // `Local(0) = Const(ConstantExpr{kind:{Literal:{Str:spec}}})`.
        let value = serde_json::json!({
            "kind": {"Literal": {"Str": "list.int_capacity(l)"}},
            "ty": {"Deduplicated": 7911}
        });
        assert_eq!(
            decode_str_const(&value).as_deref(),
            Some("list.int_capacity(l)")
        );
    }

    #[test]
    fn decode_str_const_reads_bare_and_wrapped_strings() {
        assert_eq!(
            decode_str_const(&serde_json::json!("plain")).as_deref(),
            Some("plain")
        );
        assert_eq!(
            decode_str_const(&serde_json::json!({"Str": "wrapped"})).as_deref(),
            Some("wrapped")
        );
    }

    #[test]
    fn decode_str_const_rejects_non_string() {
        assert_eq!(decode_str_const(&serde_json::json!(42)), None);
        assert_eq!(
            decode_str_const(&serde_json::json!({"kind": {"Literal": {"Bool": true}}})),
            None
        );
    }
}
