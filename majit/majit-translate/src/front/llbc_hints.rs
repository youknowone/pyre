//! Harvest JIT-hint markers from the ullbc surrogate consts the
//! `majit_macros` proc-macros emit (`_elidable_function_<NAME>`,
//! `_jit_look_inside_<NAME>`, `_jit_loop_invariant_<NAME>`,
//! `_jit_unroll_safe_<NAME>`).
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
                let hint = if global_marker_bool(llbc, gd).unwrap_or(false) {
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
    None
}
