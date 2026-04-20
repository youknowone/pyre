//! Flow-graph building for Python generators.
//!
//! RPython upstream: `rpython/flowspace/generator.py` (177 LOC).
//!
//! Phase 3 F3.7 scope: port the structural scaffolding so `objspace.rs`
//! can link against a concrete target, plus the pure helpers that do
//! not depend on `rpython/translator/`. The meaty transformation
//! functions — `make_generator_entry_graph`, `tweak_generator_graph`,
//! `tweak_generator_body_graph` — call out to
//! `rpython/translator/unsimplify.py` (`insert_empty_startblock`,
//! `split_block`) and `rpython/translator/simplify.py`
//! (`simplify_graph`, `eliminate_empty_blocks`) which are NOT part of
//! flowspace and will be ported in Phase 5 alongside the annotator /
//! translator pipeline.
//!
//! Until then the transformation entry points panic with a
//! parity-preserving message so that any caller that needs generator
//! flowspace surfaces the gap loudly instead of silently producing a
//! broken graph.
//!
//! Parity-critical pieces already landed:
//! * [`AbstractPosition`] marker trait (upstream `class AbstractPosition`).
//! * [`get_variable_names`] (upstream `get_variable_names`, a pure
//!   string helper with no flowspace coupling).

use super::model::FunctionGraph;

/// RPython `rpython/flowspace/generator.py:14-16` — `class
/// AbstractPosition`. Marker used by generator graphs to tag
/// `Entry` / `Resume<n>` subclasses. The Rust port models it as an
/// empty sealed trait; concrete `Entry` / `Resume` types are synthesised
/// when the translator/simplify port lands.
pub trait AbstractPosition: core::fmt::Debug {}

/// RPython `generator.py:86-95` — `get_variable_names(variables)`.
///
/// Rename every `_name.strip('_')` to a `g_`-prefixed unique name so
/// the subsequent `_insert_reads` call can safely stash each variable
/// on the generator's `Entry` / `Resume` object via `setattr`.
///
/// Direct translation from upstream; unused until the transformation
/// entry points below land.
pub fn get_variable_names(variables: &[&str]) -> Vec<String> {
    use std::collections::HashSet;

    let mut seen: HashSet<String> = HashSet::new();
    let mut result: Vec<String> = Vec::new();
    for v in variables {
        let mut name: String = v.trim_matches('_').to_string();
        while seen.contains(&name) {
            name.push('_');
        }
        result.push(format!("g_{name}"));
        seen.insert(name);
    }
    result
}

/// RPython `generator.py:18-34` — `make_generator_entry_graph(func)`.
///
/// **Phase 3 F3.7 gap**: the implementation requires
/// `replace_graph_with_bootstrap` / `attach_next_method`, both of which
/// synthesise Python classes at runtime (`GeneratorIterator`, `Entry`)
/// and register them against `func._generator_next_method_of_`. The
/// Rust port has no runtime class-construction facility; the real port
/// lands together with the annotator (Phase 5) which also needs to
/// reason about the synthesised `Entry` / `Resume` classes.
pub fn make_generator_entry_graph(_func: super::model::GraphFunc) -> FunctionGraph {
    unimplemented!(
        "generator flowspace requires rpython/translator/simplify.py + \
         runtime class synthesis (Phase 5). See rpython/flowspace/generator.py:18-34."
    )
}

/// RPython `generator.py:36-39` — `tweak_generator_graph(graph)`.
///
/// **Phase 3 F3.7 gap**: delegates to `tweak_generator_body_graph`
/// whose port is blocked on `translator/simplify` (see module doc).
pub fn tweak_generator_graph(_graph: &mut FunctionGraph) {
    unimplemented!(
        "generator flowspace requires rpython/translator/simplify.py (Phase 5). \
         See rpython/flowspace/generator.py:36-39."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_variable_names_prefixes_and_deduplicates() {
        // upstream `generator.py:86-95` strips leading/trailing `_`
        // and then runs a dedup suffix loop.
        let out = get_variable_names(&["x", "y", "_x_"]);
        // `x` → `g_x`, `y` → `g_y`, `_x_` → strip to `x` → dedup collides
        // with `x` already in `seen`; append `_` → `x_` → `g_x_`.
        assert_eq!(out, vec!["g_x", "g_y", "g_x_"]);
    }

    #[test]
    fn get_variable_names_empty_input() {
        assert!(get_variable_names(&[]).is_empty());
    }
}
