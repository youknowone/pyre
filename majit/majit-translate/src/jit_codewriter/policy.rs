//! JIT inlining policy.
//!
//! Translated from `rpython/jit/codewriter/policy.py`.
//!
//! `JitPolicy` decides which graphs the codewriter should "look inside"
//! and inline-trace.  RPython models this as a base class with a virtual
//! `look_inside_function`; subclasses (e.g. `StopAtXPolicy`) override that
//! one method.  In Rust we use a trait + state struct so subclasses share
//! the bookkeeping fields.

use std::collections::HashSet;

use crate::front::ast::SemanticFunction;
use crate::model::FunctionGraph;
#[allow(unused_imports)]
use majit_ir::value::Type;

/// policy.py:10-46: shared mutable state and the default classifier.
///
/// `JitPolicy.__init__` initializes:
///   - `self.unsafe_loopy_graphs = set()`
///   - `self.supports_floats = False`
///   - `self.supports_longlong = False`
///   - `self.supports_singlefloats = False`
///   - `self.jithookiface = jithookiface`
#[derive(Debug, Clone, Default)]
pub struct JitPolicyState {
    pub unsafe_loopy_graphs: HashSet<String>,
    pub supports_floats: bool,
    pub supports_longlong: bool,
    pub supports_singlefloats: bool,
    /// policy.py:16: optional `jithookiface`.  Pyre does not yet expose
    /// JIT hooks, so this stays as a marker placeholder.
    pub jithookiface: Option<()>,
}

impl JitPolicyState {
    /// policy.py:11-16: constructor.
    pub fn new() -> Self {
        Self::default()
    }

    /// policy.py:18-19
    pub fn set_supports_floats(&mut self, flag: bool) {
        self.supports_floats = flag;
    }

    /// policy.py:21-22
    pub fn set_supports_longlong(&mut self, flag: bool) {
        self.supports_longlong = flag;
    }

    /// policy.py:24-25
    pub fn set_supports_singlefloats(&mut self, flag: bool) {
        self.supports_singlefloats = flag;
    }

    /// policy.py:27-33 `dump_unsafe_loops`.
    ///
    /// ```python
    /// def dump_unsafe_loops(self):
    ///     f = udir.join("unsafe-loops.txt").open('w')
    ///     strs = [str(graph) for graph in self.unsafe_loopy_graphs]
    ///     strs.sort()
    ///     for graph in strs:
    ///         print(graph, file=f)
    ///     f.close()
    /// ```
    ///
    /// RPython's `udir` is the translator's per-run temp directory; the
    /// Rust port takes the destination path as a parameter so callers can
    /// route the dump anywhere (typically `std::env::temp_dir()`).
    pub fn dump_unsafe_loops(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut strs: Vec<&String> = self.unsafe_loopy_graphs.iter().collect();
        strs.sort();
        let mut f = std::fs::File::create(path)?;
        for graph in strs {
            writeln!(f, "{}", graph)?;
        }
        Ok(())
    }
}

/// `JitPolicy` interface.
///
/// policy.py:10 `class JitPolicy`. `look_inside_function` is the only
/// upstream method designed for subclassing; everything else is a default
/// implementation calling it.
pub trait JitPolicy {
    fn state(&self) -> &JitPolicyState;
    fn state_mut(&mut self) -> &mut JitPolicyState;

    /// policy.py:35-36 — return `True` for every function by default.
    /// `StopAtXPolicy` overrides this.
    fn look_inside_function(&self, _func: &SemanticFunction) -> bool {
        true
    }

    /// policy.py:38-46 `_reject_function(func)`.
    ///
    /// RPython rejects functions tagged `_elidable_function_` (always
    /// opaque) and the `rpython.rtyper.module.*` opaque helpers.  Pyre
    /// has no `rpython.rtyper.module` namespace, so only the `elidable`
    /// hint is consulted.
    fn _reject_function(&self, func: &SemanticFunction) -> bool {
        if func.hints.iter().any(|h| h == "elidable") {
            return true;
        }
        false
    }

    /// policy.py:48-84 `look_inside_graph(graph)`.
    ///
    /// `func._jit_look_inside_` overrides everything; otherwise we
    /// combine `look_inside_function` and `_reject_function`.  Loops
    /// disqualify a graph unless it is `_jit_unroll_safe_`.  A
    /// reject due to loops is recorded in `unsafe_loopy_graphs`.
    fn look_inside_graph(&mut self, func: &SemanticFunction) -> bool {
        let mut contains_loop = !find_backedges(&func.graph).is_empty();
        let see_function = if let Some(flag) = jit_look_inside_hint(&func.hints) {
            // policy.py:56-57: `_jit_look_inside_` override.
            flag
        } else {
            self.look_inside_function(func) && !self._reject_function(func)
        };
        // policy.py:61-62: `_jit_unroll_safe_` opts back in despite a loop.
        contains_loop = contains_loop && !func.hints.iter().any(|h| h == "unroll_safe");

        let res = see_function
            && !contains_unsupported_variable_type(
                &func.graph,
                self.state().supports_floats,
                self.state().supports_longlong,
                self.state().supports_singlefloats,
            );
        if res && contains_loop {
            self.state_mut()
                .unsafe_loopy_graphs
                .insert(func.name.clone());
        }
        let res = res && !contains_loop;
        // policy.py:71-83: access_directly check.  Pyre has no
        // `access_directly` annotation so this branch is omitted; it can
        // be re-added when the virtualizable annotation lands.
        res
    }
}

/// Default policy: equivalent to instantiating `JitPolicy()` in RPython.
#[derive(Debug, Clone, Default)]
pub struct DefaultJitPolicy {
    pub state: JitPolicyState,
}

impl DefaultJitPolicy {
    pub fn new() -> Self {
        Self {
            state: JitPolicyState::new(),
        }
    }
}

impl JitPolicy for DefaultJitPolicy {
    fn state(&self) -> &JitPolicyState {
        &self.state
    }
    fn state_mut(&mut self) -> &mut JitPolicyState {
        &mut self.state
    }
}

/// policy.py:113-119 `class StopAtXPolicy(JitPolicy)`.
///
/// Excludes a fixed list of function names from inlining.  Used by
/// translator tests that need to JIT-compile one half of a graph and
/// keep the other half opaque.
#[derive(Debug, Clone, Default)]
pub struct StopAtXPolicy {
    pub state: JitPolicyState,
    /// policy.py:115-116: `self.funcs = funcs` — list of opaque names.
    pub funcs: Vec<String>,
}

impl StopAtXPolicy {
    pub fn new(funcs: Vec<String>) -> Self {
        Self {
            state: JitPolicyState::new(),
            funcs,
        }
    }
}

impl JitPolicy for StopAtXPolicy {
    fn state(&self) -> &JitPolicyState {
        &self.state
    }
    fn state_mut(&mut self) -> &mut JitPolicyState {
        &mut self.state
    }
    /// policy.py:118-119: `return func not in self.funcs`.
    fn look_inside_function(&self, func: &SemanticFunction) -> bool {
        !self.funcs.iter().any(|f| f == &func.name)
    }
}

/// policy.py:56 `getattr(func, '_jit_look_inside_', ...)`.
///
/// Returns `Some(true|false)` when the explicit `_jit_look_inside_`
/// override is present, otherwise `None`.
///
/// rlib/jit.py wires the override via two decorators:
///   - `@dont_look_inside` (`rlib/jit.py:142`) sets
///     `func._jit_look_inside_ = False`
///   - `@look_inside` (`rlib/jit.py:147`) sets
///     `func._jit_look_inside_ = True`
///
/// `ast.rs::collect_jit_hints` lowers those decorators into the
/// `"dont_look_inside"` and `"jit_look_inside"` hint strings; both forms
/// route through this helper.
fn jit_look_inside_hint(hints: &[String]) -> Option<bool> {
    for h in hints {
        match h.as_str() {
            "dont_look_inside" => return Some(false),
            "jit_look_inside" => return Some(true),
            _ => {}
        }
        if let Some(rest) = h.strip_prefix("jit_look_inside") {
            // Accept the legacy `jit_look_inside=true|false` spelling.
            return match rest.trim_start_matches('=').trim() {
                "" | "true" | "True" => Some(true),
                "false" | "False" => Some(false),
                _ => Some(true),
            };
        }
    }
    None
}

/// policy.py:86-109 `contains_unsupported_variable_type(graph, ...)`.
///
/// PRE-EXISTING-ADAPTATION: pyre's value-id table does not yet carry the
/// per-value lltype information that RPython walks here.  Since pyre
/// always supports floats and the codewriter never produces longlong /
/// singlefloat values, the upstream check would never reject a graph in
/// the current state.  We keep the function signature (and behaviour
/// "everything is supported") so that `look_inside_graph` matches the
/// upstream control flow.  When the IR gains per-value lltype metadata
/// the body must walk every `Block.inputargs` / `SpaceOperation.kind`
/// arg and result identical to `policy.py:90-104`.
pub fn contains_unsupported_variable_type(
    _graph: &FunctionGraph,
    _supports_floats: bool,
    _supports_longlong: bool,
    _supports_singlefloats: bool,
) -> bool {
    false
}

/// `rpython.translator.backendopt.support.find_backedges(graph)`.
///
/// Standard DFS classification: edges from a block back to an ancestor
/// in the current DFS stack are back edges.  Returns the list of back
/// edges as `(from_block, to_block)` pairs.
pub fn find_backedges(graph: &FunctionGraph) -> Vec<(usize, usize)> {
    use std::collections::HashSet;

    let mut backedges = Vec::new();
    let mut seen: HashSet<usize> = HashSet::new();
    let mut seeing: HashSet<usize> = HashSet::new();
    if !graph.blocks.is_empty() {
        let start = graph.startblock.0;
        seen.insert(start);
        find_backedges_dfs(graph, start, &mut seen, &mut seeing, &mut backedges);
    }
    backedges
}

fn find_backedges_dfs(
    graph: &FunctionGraph,
    block_idx: usize,
    seen: &mut std::collections::HashSet<usize>,
    seeing: &mut std::collections::HashSet<usize>,
    backedges: &mut Vec<(usize, usize)>,
) {
    seeing.insert(block_idx);
    for target in block_exit_targets(graph, block_idx) {
        if seen.contains(&target) {
            if seeing.contains(&target) {
                backedges.push((block_idx, target));
            }
        } else {
            seen.insert(target);
            find_backedges_dfs(graph, target, seen, seeing, backedges);
        }
    }
    seeing.remove(&block_idx);
}

fn block_exit_targets(graph: &FunctionGraph, block_idx: usize) -> Vec<usize> {
    use crate::model::Terminator;
    let block = match graph.blocks.get(block_idx) {
        Some(b) => b,
        None => return Vec::new(),
    };
    // RPython `flowspace/model.py:66-76` FunctionGraph.iterblocks derives
    // the successor set from `Block.exits` only; final blocks
    // (`exits == ()`) have no outgoing targets.
    block.exits.iter().map(|link| link.target.0).collect()
}

/// `rpython.jit.metainterp.history.getkind(TYPE, ...)`.
///
/// Returns `"void"`, `"int"`, `"ref"`, `"float"`, or `Err` when the type
/// cannot be encoded under the current `supports_*` flags.  Pyre's
/// [`Type`] is already coarser than RPython's `lltype`, so the long-long
/// and single-float branches are no-ops; we keep the parameters for
/// signature parity.
pub fn getkind(
    ty: &Type,
    supports_floats: bool,
    _supports_longlong: bool,
    _supports_singlefloats: bool,
) -> Result<&'static str, ()> {
    match ty {
        Type::Void => Ok("void"),
        Type::Int => Ok("int"),
        Type::Ref => Ok("ref"),
        Type::Float => {
            if supports_floats {
                Ok("float")
            } else {
                Err(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::FunctionGraph;

    fn make_func(name: &str, hints: Vec<&str>) -> SemanticFunction {
        SemanticFunction {
            name: name.into(),
            graph: FunctionGraph::new(name),
            return_type: None,
            self_ty_root: None,
            hints: hints.into_iter().map(|h| h.to_string()).collect(),
        }
    }

    #[test]
    fn default_look_inside_function_returns_true() {
        let policy = DefaultJitPolicy::new();
        let f = make_func("foo", vec![]);
        assert!(policy.look_inside_function(&f));
    }

    #[test]
    fn elidable_hint_rejects_function() {
        let policy = DefaultJitPolicy::new();
        let f = make_func("foo", vec!["elidable"]);
        assert!(policy._reject_function(&f));
    }

    #[test]
    fn jit_look_inside_overrides_default() {
        let mut policy = DefaultJitPolicy::new();
        let f = make_func("foo", vec!["jit_look_inside=false"]);
        assert!(!policy.look_inside_graph(&f));
    }

    #[test]
    fn stop_at_x_policy_excludes_named_funcs() {
        let policy = StopAtXPolicy::new(vec!["stop_me".into()]);
        let stop = make_func("stop_me", vec![]);
        let other = make_func("other", vec![]);
        assert!(!policy.look_inside_function(&stop));
        assert!(policy.look_inside_function(&other));
    }

    #[test]
    fn unroll_safe_disables_loop_rejection() {
        let mut policy = DefaultJitPolicy::new();
        // Build a graph with a self-loop on block 0.
        let mut g = FunctionGraph::new("loopy");
        let entry = g.startblock;
        g.set_terminator(
            entry,
            crate::model::Terminator::Goto {
                target: entry,
                args: Vec::new(),
            },
        );
        let loopy = SemanticFunction {
            name: "loopy".into(),
            graph: g.clone(),
            return_type: None,
            self_ty_root: None,
            hints: vec![],
        };
        // Without `unroll_safe`, the loop disqualifies the graph.
        assert!(!policy.look_inside_graph(&loopy));
        assert!(policy.state().unsafe_loopy_graphs.contains("loopy"));

        // With `unroll_safe`, the loop is ignored.
        let unroll_safe = SemanticFunction {
            name: "loopy_safe".into(),
            graph: g,
            return_type: None,
            self_ty_root: None,
            hints: vec!["unroll_safe".into()],
        };
        assert!(policy.look_inside_graph(&unroll_safe));
    }

    #[test]
    fn dont_look_inside_hint_overrides_default_to_false() {
        // test_policy.py:61-66 `test_dont_look_inside`.
        let mut policy = DefaultJitPolicy::new();
        let f = make_func("h", vec!["dont_look_inside"]);
        assert!(!policy.look_inside_graph(&f));
    }

    #[test]
    fn jit_look_inside_hint_overrides_subclass_to_true() {
        // test_policy.py:68-80 `test_look_inside`.
        struct NoPolicy(JitPolicyState);
        impl JitPolicy for NoPolicy {
            fn state(&self) -> &JitPolicyState {
                &self.0
            }
            fn state_mut(&mut self) -> &mut JitPolicyState {
                &mut self.0
            }
            fn look_inside_function(&self, _: &SemanticFunction) -> bool {
                false
            }
        }
        let mut policy = NoPolicy(JitPolicyState::new());
        let h1 = make_func("h1", vec![]);
        let h2 = make_func("h2", vec!["jit_look_inside"]);
        assert!(!policy.look_inside_graph(&h1));
        assert!(policy.look_inside_graph(&h2));
    }

    #[test]
    fn dump_unsafe_loops_writes_sorted_names() {
        let mut state = JitPolicyState::new();
        state.unsafe_loopy_graphs.insert("zeta".into());
        state.unsafe_loopy_graphs.insert("alpha".into());
        state.unsafe_loopy_graphs.insert("mu".into());
        let tmp = tempfile::NamedTempFile::new().expect("tmpfile");
        let path = tmp.path();
        state.dump_unsafe_loops(path).expect("write");
        let body = std::fs::read_to_string(path).expect("read");
        assert_eq!(body, "alpha\nmu\nzeta\n");
    }

    #[test]
    fn find_backedges_detects_self_loop() {
        let mut g = FunctionGraph::new("loop");
        let entry = g.startblock;
        g.set_terminator(
            entry,
            crate::model::Terminator::Goto {
                target: entry,
                args: Vec::new(),
            },
        );
        let edges = find_backedges(&g);
        assert_eq!(edges, vec![(entry.0, entry.0)]);
    }
}
