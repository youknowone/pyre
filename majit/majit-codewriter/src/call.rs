//! Call control — inline vs residual decision for function calls.
//!
//! RPython equivalent: `rpython/jit/codewriter/call.py` class `CallControl`.
//!
//! Decides which functions should be inlined into JitCode ("regular") and
//! which should remain as opaque calls ("residual").  Also handles builtin
//! (oopspec) and recursive (portal) call classification.

use std::collections::{HashMap, HashSet};

use majit_ir::descr::EffectInfo;
use serde::{Deserialize, Serialize};

use crate::graph::{CallTarget, MajitGraph, OpKind};
use crate::parse::CallPath;

/// Call descriptor — associates a call target with its effect info.
///
/// RPython equivalent: the combination of `CallDescr` + `EffectInfo`
/// stored on call operations by `CallControl.getcalldescr()`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CallDescriptor {
    pub target: CallTarget,
    pub effect_info: EffectInfo,
}

impl CallDescriptor {
    pub fn known(target: CallTarget, effect_info: EffectInfo) -> Self {
        Self {
            target,
            effect_info,
        }
    }

    pub fn override_effect(target: CallTarget, effect_info: EffectInfo) -> Self {
        Self {
            target,
            effect_info,
        }
    }

    pub fn effect_info(&self) -> EffectInfo {
        self.effect_info.clone()
    }
}

/// Call classification — RPython `guess_call_kind()` return values.
///
/// RPython: the string literals `'regular'`, `'residual'`, `'builtin'`,
/// `'recursive'` returned by `CallControl.guess_call_kind()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallKind {
    /// Inline this call — callee graph is available and is a candidate.
    /// RPython: `'regular'` → produces `inline_call_*` jitcode instruction.
    Regular,
    /// Leave as a residual call in the trace.
    /// RPython: `'residual'` → produces `residual_call_*` jitcode instruction.
    Residual,
    /// Built-in operation with oopspec semantics (list ops, string ops, etc.)
    /// RPython: `'builtin'` → special handling per oopspec name.
    Builtin,
    /// Recursive call back to the portal (JIT entry point).
    /// RPython: `'recursive'` → produces `recursive_call_*` jitcode instruction.
    Recursive,
}

/// Call control — decides inline vs residual for each call target.
///
/// RPython: `call.py::CallControl`.
///
/// In RPython, `CallControl` discovers all candidate graphs by traversing
/// from the portal graph, then for each `direct_call` operation it classifies
/// the call as regular/residual/builtin/recursive.
///
/// In majit-codewriter, we don't have RPython's function pointer linkage.
/// Instead, callee graphs are collected from parsed Rust source files
/// (free functions via `collect_function_graphs` and trait impl methods
/// via `extract_trait_impls`).
pub struct CallControl {
    /// Free function graphs: CallPath → MajitGraph.
    /// RPython: `funcptr._obj.graph` linkage.
    function_graphs: HashMap<CallPath, MajitGraph>,

    /// Trait impl method graphs: (method_name, impl_type) → MajitGraph.
    /// Used for resolving `handler.method_name()` calls.
    trait_method_graphs: HashMap<(String, String), MajitGraph>,

    /// Trait bindings: method_name → Vec<impl_type>.
    /// Tracks which types implement a given method.
    trait_method_impls: HashMap<String, Vec<String>>,

    /// Candidate targets — graphs we will inline.
    /// RPython: `CallControl.candidate_graphs`.
    candidate_graphs: HashSet<CallPath>,

    /// Portal entry points (recursive call detection).
    /// RPython: `CallControl.jitdrivers_sd`.
    portal_targets: HashSet<CallPath>,

    /// Builtin targets (oopspec operations).
    /// RPython: detected via `funcobj.graph.func.oopspec`.
    builtin_targets: HashSet<CallPath>,

    /// RPython: `CallControl.jitcodes` — map {graph_key: JitCode}.
    /// Tracks which graphs have been assigned JitCode objects.
    jitcodes: HashMap<CallPath, ()>,

    /// RPython: `CallControl.unfinished_graphs` — graphs pending assembly.
    unfinished_graphs: Vec<CallPath>,
}

impl CallControl {
    /// RPython: `CallControl.__init__`.
    pub fn new() -> Self {
        Self {
            function_graphs: HashMap::new(),
            trait_method_graphs: HashMap::new(),
            trait_method_impls: HashMap::new(),
            candidate_graphs: HashSet::new(),
            portal_targets: HashSet::new(),
            builtin_targets: HashSet::new(),
            jitcodes: HashMap::new(),
            unfinished_graphs: Vec::new(),
        }
    }

    /// Register a free function graph.
    /// RPython: graphs are discovered via funcptr linkage.
    pub fn register_function_graph(&mut self, path: CallPath, graph: MajitGraph) {
        self.function_graphs.insert(path, graph);
    }

    /// Register a trait impl method graph.
    ///
    /// Also registers the graph in function_graphs under a synthetic
    /// CallPath so that BFS in find_all_graphs can discover it.
    /// RPython: method graphs are reachable through funcptr._obj.graph
    /// linkage — we emulate this by dual registration.
    pub fn register_trait_method(&mut self, method_name: &str, impl_type: &str, graph: MajitGraph) {
        // Register in trait-specific lookup
        self.trait_method_graphs.insert(
            (method_name.to_string(), impl_type.to_string()),
            graph.clone(),
        );
        self.trait_method_impls
            .entry(method_name.to_string())
            .or_default()
            .push(impl_type.to_string());
        // Also register in function_graphs for BFS reachability.
        // RPython: funcptr._obj.graph makes method graphs discoverable
        // through the same mechanism as free functions.
        let synthetic_path = CallPath::from_segments([method_name]);
        self.function_graphs.entry(synthetic_path).or_insert(graph);
    }

    /// Mark a target as the portal entry point.
    pub fn mark_portal(&mut self, path: CallPath) {
        self.portal_targets.insert(path);
    }

    /// Mark a target as a builtin (oopspec) operation.
    pub fn mark_builtin(&mut self, path: CallPath) {
        self.builtin_targets.insert(path);
    }

    /// Discover candidate graphs by BFS from portal targets.
    ///
    /// RPython: `CallControl.find_all_graphs(policy)` (call.py:49-92).
    ///
    /// Walks from portal graphs transitively: for each Call op,
    /// if the callee has a graph, add it to the candidate set.
    /// Portal must be seeded via `mark_portal()` before calling.
    pub fn find_all_graphs(&mut self) {
        assert!(
            !self.portal_targets.is_empty(),
            "find_all_graphs requires at least one portal target; \
             use find_all_graphs_for_tests() if no portal is available"
        );
        self.find_all_graphs_bfs();
    }

    /// Test-only: include all registered function graphs as candidates.
    /// Production code must use `find_all_graphs()` with portal seeded.
    #[cfg(test)]
    pub fn find_all_graphs_for_tests(&mut self) {
        if self.portal_targets.is_empty() {
            let all_paths: Vec<CallPath> = self.function_graphs.keys().cloned().collect();
            for path in all_paths {
                self.candidate_graphs.insert(path);
            }
            return;
        }
        self.find_all_graphs_bfs();
    }

    fn find_all_graphs_bfs(&mut self) {
        // BFS from portal targets (RPython: call.py:49-92)
        let initial_fn_count = self.function_graphs.len();
        let mut todo: Vec<CallPath> = self.portal_targets.iter().cloned().collect();
        for path in &todo {
            self.candidate_graphs.insert(path.clone());
        }

        while let Some(path) = todo.pop() {
            let graph = match self.function_graphs.get(&path) {
                Some(g) => g.clone(),
                None => continue,
            };
            for block in &graph.blocks {
                for op in &block.ops {
                    let callee_path = match &op.kind {
                        // FunctionPath calls: direct function references.
                        OpKind::Call {
                            target: crate::graph::CallTarget::FunctionPath { segments },
                            ..
                        } => Some(CallPath::from_segments(segments.iter().map(String::as_str))),
                        // Method calls: resolve through trait impls to find the
                        // concrete function graph, matching RPython's treatment
                        // of method dispatch in the flow graph.
                        OpKind::Call {
                            target:
                                crate::graph::CallTarget::Method {
                                    name,
                                    receiver_root,
                                },
                            ..
                        } => {
                            // If the method resolves to a unique impl, treat
                            // its graph as reachable. We use the method name as
                            // a synthetic path since methods don't have CallPaths.
                            if self
                                .resolve_method(name, receiver_root.as_deref())
                                .is_some()
                            {
                                Some(CallPath::from_segments([name.as_str()]))
                            } else {
                                None
                            }
                        }
                        _ => None,
                    };

                    if let Some(callee_path) = callee_path {
                        if !self.candidate_graphs.contains(&callee_path) {
                            // Only add if we actually have the graph
                            if self.function_graphs.contains_key(&callee_path) {
                                self.candidate_graphs.insert(callee_path.clone());
                                todo.push(callee_path);
                            }
                        }
                    }
                }
            }
        }
        #[cfg(test)]
        eprintln!(
            "find_all_graphs_bfs: {} function_graphs, {} candidates (from {} portals)",
            initial_fn_count,
            self.candidate_graphs.len(),
            self.portal_targets.len()
        );
    }

    /// RPython: `CallControl.is_candidate(graph)`.
    /// Used only after `find_all_graphs()`.
    pub fn is_candidate(&self, path: &CallPath) -> bool {
        self.candidate_graphs.contains(path)
    }

    /// RPython: `CallControl.get_jitcode(graph, called_from)`.
    /// Retrieve or create a JitCode entry for the given graph.
    /// Currently tracks presence only — full JitCode objects are
    /// managed by the assembler (future).
    pub fn get_jitcode(&mut self, path: CallPath) {
        if !self.jitcodes.contains_key(&path) {
            self.jitcodes.insert(path.clone(), ());
            self.unfinished_graphs.push(path);
        }
    }

    /// RPython: `CallControl.enum_pending_graphs()`.
    pub fn enum_pending_graphs(&mut self) -> Vec<CallPath> {
        std::mem::take(&mut self.unfinished_graphs)
    }

    /// Classify a call target.
    ///
    /// RPython: `CallControl.guess_call_kind(op)`.
    ///
    /// Decision logic (in priority order):
    /// 1. Portal target → Recursive
    /// 2. Builtin target → Builtin
    /// 3. No graph available → Residual
    /// 4. Is candidate → Regular
    /// 5. Otherwise → Residual
    pub fn guess_call_kind(&self, target: &CallTarget) -> CallKind {
        match target {
            CallTarget::FunctionPath { segments } => {
                let path = CallPath::from_segments(segments.iter().map(String::as_str));
                if self.portal_targets.contains(&path) {
                    return CallKind::Recursive;
                }
                if self.builtin_targets.contains(&path) {
                    return CallKind::Builtin;
                }
                if self.candidate_graphs.contains(&path) {
                    return CallKind::Regular;
                }
                CallKind::Residual
            }
            CallTarget::Method {
                name,
                receiver_root,
            } => {
                // RPython: method calls use the same graphs_from() + is_candidate
                // logic as function calls. resolve_method() finds the graph;
                // candidate_graphs membership determines regular vs residual.
                if self
                    .resolve_method(name, receiver_root.as_deref())
                    .is_none()
                {
                    return CallKind::Residual;
                }
                // Method graphs are registered in function_graphs under
                // synthetic CallPath([method_name]) by register_trait_method().
                let synthetic_path = CallPath::from_segments([name.as_str()]);
                if self.candidate_graphs.contains(&synthetic_path) {
                    return CallKind::Regular;
                }
                CallKind::Residual
            }
            CallTarget::UnsupportedExpr => CallKind::Residual,
        }
    }

    /// Get the callee graph for a call target.
    ///
    /// RPython: `CallControl.graphs_from(op)`.
    pub fn graphs_from(&self, target: &CallTarget) -> Option<&MajitGraph> {
        match target {
            CallTarget::FunctionPath { segments } => {
                let path = CallPath::from_segments(segments.iter().map(String::as_str));
                self.function_graphs.get(&path)
            }
            CallTarget::Method {
                name,
                receiver_root,
            } => self.resolve_method(name, receiver_root.as_deref()),
            CallTarget::UnsupportedExpr => None,
        }
    }

    /// Resolve a method call to a concrete impl graph.
    ///
    /// RPython: method resolution happens at the type system level.
    /// Here we resolve through the trait impl registry. If there's
    /// exactly one impl for the method, return it. If the receiver
    /// is a generic parameter (lowercase or single uppercase letter),
    /// we try all known impls and return the unique one.
    pub fn resolve_method(&self, name: &str, receiver_root: Option<&str>) -> Option<&MajitGraph> {
        let impls = self.trait_method_impls.get(name)?;

        // Filter out default trait method entries (e.g. "<default methods of LocalOpcodeHandler>")
        // — we prefer concrete impls when available.
        let concrete_impls: Vec<&String> = impls
            .iter()
            .filter(|t| !t.starts_with("<default methods of"))
            .collect();

        if concrete_impls.len() == 1 {
            // Unique concrete impl — use it regardless of receiver
            let impl_type = concrete_impls[0];
            return self
                .trait_method_graphs
                .get(&(name.to_string(), impl_type.clone()));
        }

        // Multiple concrete impls — try to match by receiver root
        if let Some(receiver) = receiver_root {
            if !is_generic_receiver(receiver) {
                // Concrete receiver — look for exact match
                return self
                    .trait_method_graphs
                    .get(&(name.to_string(), receiver.to_string()));
            }
        }

        // Generic receiver with multiple concrete impls — can't resolve uniquely.
        // Fall back to default method if available.
        if concrete_impls.is_empty() && impls.len() == 1 {
            let impl_type = &impls[0];
            return self
                .trait_method_graphs
                .get(&(name.to_string(), impl_type.clone()));
        }

        None
    }

    /// Access the function graphs map (for inline pass).
    pub fn function_graphs(&self) -> &HashMap<CallPath, MajitGraph> {
        &self.function_graphs
    }
}

impl Default for CallControl {
    fn default() -> Self {
        Self::new()
    }
}

/// Detect generic type parameter or variable name used as receiver.
///
/// Generic: "H", "T", "handler", "self", "executor"
/// Concrete: "PyFrame", "Code", "Vec"
///
/// Heuristic: single uppercase letter is a type parameter;
/// starts with lowercase is a variable name.
pub fn is_generic_receiver(receiver: &str) -> bool {
    let mut chars = receiver.chars();
    let first = match chars.next() {
        Some(c) => c,
        None => return false,
    };
    if first.is_lowercase() {
        return true;
    }
    // Single uppercase letter: "H", "T", "E" — type parameter
    first.is_uppercase() && chars.next().is_none()
}

// ── Builtin call effect tables ──────────────────────────────────
//
// RPython equivalent: effect classification in `call.py::getcalldescr()`
// combined with the builtin function tables.
// These tables map known function targets to their effect info,
// used by `jtransform::classify_call()` as a fallback when the
// call is not in the explicit `call_effects` config.

use majit_ir::descr::{ExtraEffect, OopSpecIndex};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CallTargetPattern {
    Method {
        name: &'static str,
        receiver_root: Option<&'static str>,
    },
    FunctionPath(&'static [&'static str]),
}

impl CallTargetPattern {
    fn matches(self, target: &CallTarget) -> bool {
        match (self, target) {
            (
                CallTargetPattern::Method {
                    name,
                    receiver_root,
                },
                CallTarget::Method {
                    name: target_name,
                    receiver_root: target_root,
                },
            ) => {
                if target_name != name {
                    return false;
                }
                receiver_root.is_none_or(|root| {
                    target_root.as_deref() == Some(root)
                        || target_root.as_ref().is_some_and(|r| is_generic_receiver(r))
                })
            }
            (CallTargetPattern::FunctionPath(path), CallTarget::FunctionPath { segments }) => {
                segments.iter().map(String::as_str).eq(path.iter().copied())
            }
            _ => false,
        }
    }
}

struct CallDescriptorEntry {
    targets: &'static [CallTargetPattern],
    extra_effect: ExtraEffect,
    oopspec_index: OopSpecIndex,
}

impl CallDescriptorEntry {
    fn effect_info(&self) -> EffectInfo {
        match self.extra_effect {
            ExtraEffect::ElidableCannotRaise => EffectInfo::elidable(),
            extra_effect => EffectInfo::new(extra_effect, self.oopspec_index),
        }
    }
}

const INT_ARITH_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["w_int_add"]),
    CallTargetPattern::FunctionPath(&["w_int_sub"]),
    CallTargetPattern::FunctionPath(&["w_int_mul"]),
    CallTargetPattern::FunctionPath(&["crate", "math", "w_int_add"]),
    CallTargetPattern::FunctionPath(&["crate", "math", "w_int_sub"]),
    CallTargetPattern::FunctionPath(&["crate", "math", "w_int_mul"]),
];

const FLOAT_ARITH_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["w_float_add"]),
    CallTargetPattern::FunctionPath(&["w_float_sub"]),
];

const CALL_DESCRIPTOR_TABLE: &[CallDescriptorEntry] = &[
    CallDescriptorEntry {
        targets: INT_ARITH_TARGETS,
        extra_effect: ExtraEffect::ElidableCannotRaise,
        oopspec_index: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: FLOAT_ARITH_TARGETS,
        extra_effect: ExtraEffect::ElidableCannotRaise,
        oopspec_index: OopSpecIndex::None,
    },
];

fn matches_any(target: &CallTarget, patterns: &[CallTargetPattern]) -> bool {
    patterns
        .iter()
        .copied()
        .any(|pattern| pattern.matches(target))
}

/// Check if a call target is a known int arithmetic function.
/// Used by annotate pass for type inference.
pub fn is_int_arithmetic_target(target: &CallTarget) -> bool {
    matches_any(target, INT_ARITH_TARGETS)
}

/// Look up a call target in the builtin effect table.
///
/// RPython: part of `CallControl.getcalldescr()` — returns effect info
/// for known functions like `w_int_add` (elidable), `w_float_sub` (elidable).
pub fn describe_call(target: &CallTarget) -> Option<CallDescriptor> {
    CALL_DESCRIPTOR_TABLE
        .iter()
        .find(|entry| matches_any(target, entry.targets))
        .map(|entry| CallDescriptor::known(target.clone(), entry.effect_info()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::MajitGraph;

    #[test]
    fn guess_call_kind_function_path() {
        let mut cc = CallControl::new();
        let graph = MajitGraph::new("opcode_load_fast");
        let path = CallPath::from_segments(["opcode_load_fast"]);
        cc.register_function_graph(path, graph);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["opcode_load_fast"]);
        assert_eq!(cc.guess_call_kind(&target), CallKind::Regular);

        let unknown = CallTarget::function_path(["unknown_function"]);
        assert_eq!(cc.guess_call_kind(&unknown), CallKind::Residual);
    }

    #[test]
    fn guess_call_kind_portal() {
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["portal_runner"]);
        cc.mark_portal(path);

        let target = CallTarget::function_path(["portal_runner"]);
        assert_eq!(cc.guess_call_kind(&target), CallKind::Recursive);
    }

    #[test]
    fn guess_call_kind_builtin() {
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["w_int_add"]);
        cc.mark_builtin(path);

        let target = CallTarget::function_path(["w_int_add"]);
        assert_eq!(cc.guess_call_kind(&target), CallKind::Builtin);
    }

    #[test]
    fn resolve_method_unique_impl() {
        let mut cc = CallControl::new();
        let graph = MajitGraph::new("PyFrame::load_local_value");
        cc.register_trait_method("load_local_value", "PyFrame", graph);

        // Unique impl — resolves for any receiver
        assert!(
            cc.resolve_method("load_local_value", Some("handler"))
                .is_some()
        );
        assert!(cc.resolve_method("load_local_value", Some("H")).is_some());
        assert!(cc.resolve_method("load_local_value", None).is_some());
    }

    #[test]
    fn resolve_method_multiple_impls() {
        let mut cc = CallControl::new();
        cc.register_trait_method(
            "push_value",
            "PyFrame",
            MajitGraph::new("PyFrame::push_value"),
        );
        cc.register_trait_method(
            "push_value",
            "MIFrame",
            MajitGraph::new("MIFrame::push_value"),
        );

        // Concrete receiver — resolves to specific impl
        assert!(cc.resolve_method("push_value", Some("PyFrame")).is_some());

        // Generic receiver — can't resolve uniquely
        assert!(cc.resolve_method("push_value", Some("handler")).is_none());
        assert!(cc.resolve_method("push_value", Some("H")).is_none());
    }

    #[test]
    fn is_generic_receiver_examples() {
        assert!(is_generic_receiver("handler"));
        assert!(is_generic_receiver("self"));
        assert!(is_generic_receiver("H"));
        assert!(is_generic_receiver("T"));
        assert!(!is_generic_receiver("PyFrame"));
        assert!(!is_generic_receiver("Code"));
        assert!(!is_generic_receiver("Vec"));
    }
}
