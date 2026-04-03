//! Call control — inline vs residual decision for function calls.
//!
//! RPython equivalent: `rpython/jit/codewriter/call.py` class `CallControl`.
//!
//! Decides which functions should be inlined into JitCode ("regular") and
//! which should remain as opaque calls ("residual").  Also handles builtin
//! (oopspec) and recursive (portal) call classification.

use std::collections::{HashMap, HashSet};

use crate::graph::{CallTarget, MajitGraph, OpKind};
use crate::parse::CallPath;

// Re-export CallDescriptor so callers don't need call_match directly.
pub use crate::call_match::CallDescriptor;

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
    candidate_targets: HashSet<CallPath>,

    /// Portal entry points (recursive call detection).
    /// RPython: `CallControl.jitdrivers_sd`.
    portal_targets: HashSet<CallPath>,

    /// Builtin targets (oopspec operations).
    /// RPython: detected via `funcobj.graph.func.oopspec`.
    builtin_targets: HashSet<CallPath>,
}

impl CallControl {
    /// RPython: `CallControl.__init__`.
    pub fn new() -> Self {
        Self {
            function_graphs: HashMap::new(),
            trait_method_graphs: HashMap::new(),
            trait_method_impls: HashMap::new(),
            candidate_targets: HashSet::new(),
            portal_targets: HashSet::new(),
            builtin_targets: HashSet::new(),
        }
    }

    /// Register a free function graph.
    /// RPython: graphs are discovered via funcptr linkage.
    pub fn register_function_graph(&mut self, path: CallPath, graph: MajitGraph) {
        self.function_graphs.insert(path, graph);
    }

    /// Register a trait impl method graph.
    pub fn register_trait_method(&mut self, method_name: &str, impl_type: &str, graph: MajitGraph) {
        self.trait_method_graphs
            .insert((method_name.to_string(), impl_type.to_string()), graph);
        self.trait_method_impls
            .entry(method_name.to_string())
            .or_default()
            .push(impl_type.to_string());
    }

    /// Mark a target as the portal entry point.
    pub fn mark_portal(&mut self, path: CallPath) {
        self.portal_targets.insert(path);
    }

    /// Mark a target as a builtin (oopspec) operation.
    pub fn mark_builtin(&mut self, path: CallPath) {
        self.builtin_targets.insert(path);
    }

    /// Discover all candidate graphs reachable from portal.
    ///
    /// RPython: `CallControl.find_all_graphs(policy)`.
    ///
    /// Walks from portal graphs transitively: for each Call op,
    /// if `guess_call_kind() == Regular`, add the callee to candidates.
    pub fn find_all_graphs(&mut self) {
        // Start with all registered function graphs as candidates.
        // In RPython, this is filtered by policy.look_inside_graph().
        // Here, we include all graphs by default (equivalent to the
        // default policy where look_inside_function() returns True).
        let all_paths: Vec<CallPath> = self.function_graphs.keys().cloned().collect();
        for path in all_paths {
            self.candidate_targets.insert(path);
        }
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
                if self.candidate_targets.contains(&path) {
                    return CallKind::Regular;
                }
                CallKind::Residual
            }
            CallTarget::Method {
                name,
                receiver_root,
            } => {
                // For method calls, check if we have a unique impl.
                if self
                    .resolve_method(name, receiver_root.as_deref())
                    .is_some()
                {
                    CallKind::Regular
                } else {
                    CallKind::Residual
                }
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
        if impls.len() == 1 {
            // Unique impl — use it regardless of receiver
            let impl_type = &impls[0];
            return self
                .trait_method_graphs
                .get(&(name.to_string(), impl_type.clone()));
        }

        // Multiple impls — try to match by receiver root
        if let Some(receiver) = receiver_root {
            if !is_generic_receiver(receiver) {
                // Concrete receiver — look for exact match
                return self
                    .trait_method_graphs
                    .get(&(name.to_string(), receiver.to_string()));
            }
        }

        // Generic receiver or no receiver — can't resolve uniquely
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
        cc.find_all_graphs();

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
