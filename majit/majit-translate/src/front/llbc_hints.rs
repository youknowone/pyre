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

use majit_charon_reader::Llbc;
use std::collections::HashMap;

/// Marker-const name prefix → the JIT hint strings it implies.  The
/// user function's leaf name is the const leaf with the prefix stripped.
///
/// This is the inverse of `majit_macros::rpython_attribute_const_for`.
/// It is intentionally PRESENCE-based (the const value is not read): a
/// marker whose mere existence identifies the source attribute keeps the
/// harvester independent of the ullbc's const-initializer encoding.
const CONST_PREFIX_HINTS: &[(&str, &[&str])] = &[
    ("_elidable_function_", &["elidable"]),
    ("_jit_elidable_cannot_raise_", &["elidable_cannot_raise"]),
    ("_jit_elidable_or_memerror_", &["elidable_or_memerror"]),
    ("_jit_look_inside_", &["dont_look_inside"]),
    ("_jit_loop_invariant_", &["loopinvariant"]),
    ("_jit_unroll_safe_", &["unroll_safe"]),
];

/// Build a `{fn_leaf_name → sorted-deduped hints}` map from the marker
/// consts present in `llbcs`.  Keyed by leaf name so `merge_hints_from_llbcs`
/// can match each `SemanticFunction` by the trailing segment of its path.
pub fn harvest_hints_from_llbcs(llbcs: &[Llbc]) -> HashMap<String, Vec<String>> {
    let mut out: HashMap<String, Vec<String>> = HashMap::new();
    for llbc in llbcs {
        for gd in llbc.iter_global_decls() {
            let path = gd.item_meta.name_path();
            let leaf = path.rsplit("::").next().unwrap_or(path.as_str());
            for (prefix, hints) in CONST_PREFIX_HINTS {
                if let Some(fn_name) = leaf.strip_prefix(prefix) {
                    // `elidable_promote` emits a synthetic `_orig_<name>_unlikely_name`
                    // helper carrying `_elidable_function_`.  This generated fn is not
                    // a user function, so skip it and only harvest hints for the
                    // user-written functions.
                    if fn_name.starts_with("_orig_") && fn_name.ends_with("_unlikely_name") {
                        continue;
                    }
                    let bucket = out.entry(fn_name.to_string()).or_default();
                    bucket.extend(hints.iter().map(|h| (*h).to_string()));
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
