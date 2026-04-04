//! Call control — inline vs residual decision for function calls.
//!
//! RPython equivalent: `rpython/jit/codewriter/call.py` class `CallControl`.
//!
//! Decides which functions should be inlined into JitCode ("regular") and
//! which should remain as opaque calls ("residual").  Also handles builtin
//! (oopspec) and recursive (portal) call classification.

use std::collections::{HashMap, HashSet};

use majit_ir::descr::{EffectInfo, ExtraEffect, OopSpecIndex};
use majit_ir::value::Type;
use serde::{Deserialize, Serialize};

use crate::model::{CallTarget, FunctionGraph, OpKind, Terminator};
use crate::parse::CallPath;

// ── Graph-based analyzers (RPython effectinfo.py + canraise.py) ────
//
// RPython uses BoolGraphAnalyzer subclasses that traverse call graphs
// transitively. Each analyzer checks for specific operations:
//   - RaiseAnalyzer: Abort terminators (canraise.py)
//   - VirtualizableAnalyzer: jit_force_virtualizable/jit_force_virtual ops
//   - QuasiImmutAnalyzer: jit_force_quasi_immutable ops
//   - RandomEffectsAnalyzer: unanalyzable external calls

/// RPython: canraise.py — result of raise analysis.
///
/// `_canraise()` returns True, False, or "mem" (only MemoryError).
/// call.py:337-355.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanRaise {
    /// Function cannot raise any exception.
    No,
    /// Function can only raise MemoryError.
    MemoryErrorOnly,
    /// Function can raise arbitrary exceptions.
    Yes,
}

/// RPython: DependencyTracker equivalent — caches transitive analysis results.
///
/// Each analyzer in RPython has its own `seen` set (via `analyze_direct_call`).
/// We cache the final result per CallPath so repeated queries are O(1).
#[derive(Default)]
pub struct AnalysisCache {
    can_raise: HashMap<CallPath, CanRaise>,
    forces_virtualizable: HashMap<CallPath, bool>,
    random_effects: HashMap<CallPath, bool>,
    can_invalidate: HashMap<CallPath, bool>,
    /// RPython: `cpu.fielddescrof(T, fieldname)` / `cpu.arraydescrof(ARRAY)`.
    /// Assigns sequential, collision-free ei_index values for bitstrings.
    pub descr_indices: DescrIndexRegistry,
}

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

/// RPython: `JitDriverStaticData` — per-jitdriver metadata.
///
/// RPython `metainterp/jitdriver.py`: stores green/red variable names,
/// virtualizable info, portal graph reference, etc.
#[derive(Debug, Clone)]
pub struct JitDriverStaticData {
    /// RPython: `jitdriver_sd.index`
    pub index: usize,
    /// RPython: `jitdriver.greens` — loop-invariant variable names.
    pub greens: Vec<String>,
    /// RPython: `jitdriver.reds` — loop-variant variable names.
    pub reds: Vec<String>,
    /// Portal graph path.
    pub portal_graph: CallPath,
    /// RPython: `jd.mainjitcode` (call.py:147) — JitCode index for the portal.
    /// Set by `grab_initial_jitcodes()`.
    pub mainjitcode: Option<usize>,
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
    /// Free function graphs: CallPath → FunctionGraph.
    /// RPython: `funcptr._obj.graph` linkage.
    function_graphs: HashMap<CallPath, FunctionGraph>,

    /// Trait impl method graphs: (method_name, impl_type) → FunctionGraph.
    /// Used for resolving `handler.method_name()` calls.
    trait_method_graphs: HashMap<(String, String), FunctionGraph>,

    /// Trait bindings: method_name → Vec<impl_type>.
    /// Tracks which types implement a given method.
    trait_method_impls: HashMap<String, Vec<String>>,

    /// Candidate targets — graphs we will inline.
    /// RPython: `CallControl.candidate_graphs`.
    candidate_graphs: HashSet<CallPath>,

    /// Portal entry points (recursive call detection).
    /// RPython: `CallControl.jitdrivers_sd`.
    portal_targets: HashSet<CallPath>,

    /// RPython: `JitDriverStaticData` — metadata for each jitdriver.
    /// `jitdrivers_sd[i]` holds the green/red arg layout for driver i.
    jitdrivers_sd: Vec<JitDriverStaticData>,

    /// Builtin targets (oopspec operations).
    /// RPython: detected via `funcobj.graph.func.oopspec`.
    builtin_targets: HashSet<CallPath>,

    /// RPython: `CallControl.jitcodes` — map {graph_key: jitcode_index}.
    /// Tracks which graphs have been assigned JitCode objects.
    /// The index is assigned sequentially and used by InlineCall ops.
    jitcodes: HashMap<CallPath, usize>,

    /// RPython: `CallControl.unfinished_graphs` — graphs pending assembly.
    unfinished_graphs: Vec<CallPath>,

    /// RPython: `CallControl.callinfocollection` (call.py:31).
    /// Stores oopspec function info for builtin call handling.
    pub callinfocollection: majit_ir::descr::CallInfoCollection,

    /// Next JitCode index to assign.
    next_jitcode_index: usize,

    /// RPython: `getattr(func, "_elidable_function_", False)` (call.py:239).
    /// Targets known to be elidable (pure, no side effects).
    elidable_targets: HashSet<CallPath>,

    /// RPython: `getattr(func, "_jit_loop_invariant_", False)` (call.py:240).
    /// Targets known to be loop-invariant (call once per loop).
    loopinvariant_targets: HashSet<CallPath>,

    /// RPython: known struct types for `get_type_flag(ARRAY.OF)` → FLAG_STRUCT.
    /// If an array's element type is in this set, the array descriptor gets
    /// `ArrayFlag::Struct` (like RPython's `isinstance(TYPE, lltype.Struct)`).
    known_struct_names: HashSet<String>,

    /// RPython: struct field type info — maps struct_name → [(field_name, type_string)].
    /// Used by `resolve_array_identity` to determine the ARRAY element type
    /// when the base of an array access comes from a FieldRead.
    /// Equivalent to `op.args[0].concretetype.TO` in RPython's rtyped graph.
    struct_fields: crate::front::StructFieldRegistry,
}

/// Sequential descriptor index assignment — majit equivalent of
/// `cpu.fielddescrof(T, fieldname).get_ei_index()` /
/// `cpu.arraydescrof(ARRAY).get_ei_index()`.
///
/// RPython: each descriptor gets a globally unique index via
/// `compute_bitstrings()`. The bitstring module creates variable-length
/// bitfields. In majit we use fixed-width u64, so indices wrap at 64.
///
/// Guarantees: same (owner_root, field_name) always gets the same index.
/// Limitation: after 64 unique field descriptors, indices wrap and may
/// alias unrelated descriptors. This matches RPython's bitstring semantics
/// where aliased bits cause conservative over-approximation (safe but
/// imprecise). Array descriptors are keyed by `(item_ty, array_type_id)`,
/// matching RPython's `cpu.arraydescrof(ARRAY)` which distinguishes by
/// ARRAY identity (e.g. `GcArray(Signed)` vs `GcArray(Ptr(STRUCT_X))`).
#[derive(Default)]
pub struct DescrIndexRegistry {
    /// (owner_root, field_name) → bit index (0..63)
    field_indices: HashMap<(Option<String>, String), u32>,
    /// (item_ty_discriminant, array_type_id) → bit index (0..63)
    /// RPython: cpu.arraydescrof(ARRAY).get_ei_index()
    array_indices: HashMap<(u8, Option<String>), u32>,
    next_field_index: u32,
    next_array_index: u32,
}

impl DescrIndexRegistry {
    /// RPython: `cpu.fielddescrof(T, fieldname).get_ei_index()`
    pub fn field_index(&mut self, owner_root: &Option<String>, field_name: &str) -> u32 {
        let key = (owner_root.clone(), field_name.to_string());
        *self.field_indices.entry(key).or_insert_with(|| {
            let idx = self.next_field_index % 64;
            self.next_field_index += 1;
            idx
        })
    }

    /// RPython: `cpu.arraydescrof(ARRAY).get_ei_index()`
    ///
    /// Keys on `(item_ty_discriminant, array_type_id)`.  For non-Ref types
    /// (Int, Float, Void), `array_type_id` is ignored because all arrays
    /// of the same primitive element type share one descriptor — exactly
    /// like RPython's `GcArray(Signed)` being one lltype regardless of
    /// which field holds it.  Only Ref types need `array_type_id` to
    /// distinguish different pointer/struct element types.
    pub fn array_index(&mut self, item_ty_discriminant: u8, array_type_id: &Option<String>) -> u32 {
        // RPython: GcArray(Signed) == GcArray(Signed) for all int arrays.
        // Only Ref (discriminant for Ref/Unknown) needs sub-discrimination.
        let effective_id = if item_ty_discriminant
            == value_type_discriminant(&crate::model::ValueType::Ref)
            || item_ty_discriminant == value_type_discriminant(&crate::model::ValueType::Unknown)
        {
            array_type_id.clone()
        } else {
            None
        };
        let key = (item_ty_discriminant, effective_id);
        *self.array_indices.entry(key).or_insert_with(|| {
            let idx = self.next_array_index % 64;
            self.next_array_index += 1;
            idx
        })
    }
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
            jitdrivers_sd: Vec::new(),
            builtin_targets: HashSet::new(),
            jitcodes: HashMap::new(),
            unfinished_graphs: Vec::new(),
            callinfocollection: majit_ir::descr::CallInfoCollection::new(),
            next_jitcode_index: 0,
            elidable_targets: HashSet::new(),
            loopinvariant_targets: HashSet::new(),
            known_struct_names: HashSet::new(),
            struct_fields: crate::front::StructFieldRegistry::default(),
        }
    }

    /// RPython: register struct type names for get_type_flag(ARRAY.OF).
    pub fn set_known_struct_names(&mut self, names: HashSet<String>) {
        self.known_struct_names = names;
    }

    /// RPython: register struct field types for op.args[0].concretetype resolution.
    pub fn set_struct_fields(&mut self, registry: crate::front::StructFieldRegistry) {
        self.struct_fields = registry;
    }

    /// RPython: isinstance(TYPE, lltype.Struct) check.
    pub fn is_known_struct(&self, name: &str) -> bool {
        self.known_struct_names.contains(name)
    }

    /// RPython: resolve a struct field's type string.
    /// For `owner::field_name`, returns the full type of the field.
    pub fn field_type(&self, owner: &str, field_name: &str) -> Option<&str> {
        self.struct_fields.field_type(owner, field_name)
    }

    /// Register a free function graph.
    /// RPython: graphs are discovered via funcptr linkage.
    pub fn register_function_graph(&mut self, path: CallPath, graph: FunctionGraph) {
        self.function_graphs.insert(path, graph);
    }

    /// Register a trait impl method graph.
    ///
    /// Also registers the graph in function_graphs under a synthetic
    /// CallPath so that BFS in find_all_graphs can discover it.
    /// RPython: method graphs are reachable through funcptr._obj.graph
    /// linkage — we emulate this by dual registration.
    pub fn register_trait_method(
        &mut self,
        method_name: &str,
        impl_type: &str,
        graph: FunctionGraph,
    ) {
        self.trait_method_graphs.insert(
            (method_name.to_string(), impl_type.to_string()),
            graph.clone(),
        );
        self.trait_method_impls
            .entry(method_name.to_string())
            .or_default()
            .push(impl_type.to_string());
        // Register in function_graphs for BFS reachability.
        // RPython: each graph has its own identity via funcptr._obj.graph.
        // We emulate this with CallPath([impl_type, method_name]) —
        // each impl gets its own distinct path, preventing name collisions
        // (e.g. PyFrame::push_value vs MIFrame::push_value).
        let qualified_path = CallPath::from_segments([impl_type, method_name]);
        self.function_graphs.entry(qualified_path).or_insert(graph);
    }

    /// Mark a target as the portal entry point.
    ///
    /// RPython: `setup_jitdriver(jitdriver_sd)` + `grab_initial_jitcodes()`.
    pub fn mark_portal(&mut self, path: CallPath) {
        self.portal_targets.insert(path);
    }

    /// Register a JitDriver with its green/red variable layout.
    ///
    /// RPython: `CodeWriter.setup_jitdriver(jitdriver_sd)` (codewriter.py:96-99).
    /// Each jitdriver gets a sequential index.
    pub fn setup_jitdriver(
        &mut self,
        portal_graph: CallPath,
        greens: Vec<String>,
        reds: Vec<String>,
    ) {
        let index = self.jitdrivers_sd.len();
        self.jitdrivers_sd.push(JitDriverStaticData {
            index,
            greens,
            reds,
            portal_graph: portal_graph.clone(),
            mainjitcode: None,
        });
        self.portal_targets.insert(portal_graph);
    }

    /// RPython: `jitdriver_sd_from_portal_runner_ptr(funcptr)`.
    /// Find the jitdriver that owns a given portal target.
    pub fn jitdriver_sd_from_portal(&self, path: &CallPath) -> Option<&JitDriverStaticData> {
        self.jitdrivers_sd
            .iter()
            .find(|sd| &sd.portal_graph == path)
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
        // RPython call.py:49-92: BFS from portal targets.
        // For each graph, scan all Call ops. If guess_call_kind would
        // return 'regular' (i.e. graphs_from returns a graph AND it's
        // a candidate), add the callee graph to candidates and continue.
        //
        // During BFS we use target_to_path + function_graphs directly
        // (not graphs_from, which checks candidate_graphs — the set
        // we're building).
        let mut todo: Vec<CallPath> = self.portal_targets.iter().cloned().collect();
        for path in &todo {
            self.candidate_graphs.insert(path.clone());
        }

        while let Some(path) = todo.pop() {
            let graph = match self.function_graphs.get(&path) {
                Some(g) => g.clone(),
                None => continue,
            };
            // RPython call.py:77-90: scan all Call ops in the graph.
            // For each call, check guess_call_kind (with BFS-aware
            // is_candidate that treats "has graph" as candidate).
            for block in &graph.blocks {
                for op in &block.operations {
                    let target = match &op.kind {
                        OpKind::Call { target, .. } => target,
                        _ => continue,
                    };
                    let callee_path = match self.target_to_path(target) {
                        Some(p) => p,
                        None => continue,
                    };
                    // RPython call.py:80: kind = self.guess_call_kind(op, is_candidate)
                    // Skip recursive (portal) and builtin calls — these are NOT
                    // followed during BFS. Only "regular" calls are followed.
                    if self.portal_targets.contains(&callee_path) {
                        continue; // recursive — don't follow
                    }
                    if self.builtin_targets.contains(&callee_path) {
                        continue; // builtin — don't follow
                    }
                    if self.candidate_graphs.contains(&callee_path) {
                        continue; // already discovered
                    }
                    // RPython call.py:84,87: callee must have a graph.
                    // is_candidate during BFS = "has a graph" (default policy).
                    if self.function_graphs.contains_key(&callee_path) {
                        self.candidate_graphs.insert(callee_path.clone());
                        todo.push(callee_path);
                    }
                }
            }
        }
    }

    /// RPython: `CallControl.is_candidate(graph)`.
    /// Used only after `find_all_graphs()`.
    pub fn is_candidate(&self, path: &CallPath) -> bool {
        self.candidate_graphs.contains(path)
    }

    /// RPython: `CallControl.get_jitcode(graph, called_from)`.
    ///
    /// Retrieve or create a JitCode index for the given graph.
    /// Returns the index that should be embedded in `InlineCall` ops
    /// so the meta-interpreter can find the callee's bytecode.
    ///
    /// RPython call.py:155-172: creates JitCode(graph.name, fnaddr, calldescr)
    /// and adds graph to unfinished_graphs for later assembly.
    pub fn get_jitcode(&mut self, path: &CallPath) -> usize {
        if let Some(&index) = self.jitcodes.get(path) {
            return index;
        }
        let index = self.next_jitcode_index;
        self.next_jitcode_index += 1;
        self.jitcodes.insert(path.clone(), index);
        self.unfinished_graphs.push(path.clone());
        index
    }

    /// RPython: `CallControl.grab_initial_jitcodes()` (call.py:145-148).
    ///
    /// ```python
    /// def grab_initial_jitcodes(self):
    ///     for jd in self.jitdrivers_sd:
    ///         jd.mainjitcode = self.get_jitcode(jd.portal_graph)
    ///         jd.mainjitcode.jitdriver_sd = jd
    /// ```
    ///
    /// Creates JitCode entries for portal graphs and sets back-references.
    pub fn grab_initial_jitcodes(&mut self) {
        // Collect portal paths first to avoid borrow conflict.
        let portals: Vec<(usize, CallPath)> = self
            .jitdrivers_sd
            .iter()
            .enumerate()
            .map(|(i, jd)| (i, jd.portal_graph.clone()))
            .collect();
        for (jd_index, portal) in portals {
            // RPython: jd.mainjitcode = self.get_jitcode(jd.portal_graph)
            let jitcode_index = self.get_jitcode(&portal);
            // RPython: jd.mainjitcode.jitdriver_sd = jd
            // (In majit, we store the jitcode index on the jitdriver.)
            self.jitdrivers_sd[jd_index].mainjitcode = Some(jitcode_index);
        }
    }

    /// RPython: `CallControl.enum_pending_graphs()` (call.py:150-153).
    ///
    /// ```python
    /// def enum_pending_graphs(self):
    ///     while self.unfinished_graphs:
    ///         graph = self.unfinished_graphs.pop()  # LIFO
    ///         yield graph, self.jitcodes[graph]
    /// ```
    ///
    /// RPython uses a generator that pops one graph at a time (LIFO).
    /// During processing, new graphs may be added to `unfinished_graphs`
    /// via `get_jitcode()`, and the generator picks them up on the next
    /// iteration. We emulate this with `pop_one_graph()`.
    pub fn pop_one_graph(&mut self) -> Option<(CallPath, usize)> {
        let path = self.unfinished_graphs.pop()?; // LIFO, matching RPython
        let index = self.jitcodes[&path];
        Some((path, index))
    }

    /// Classify a call target.
    ///
    /// RPython: `CallControl.guess_call_kind(op, is_candidate)` (call.py:116-139).
    ///
    /// Exact RPython decision logic:
    /// 1. Is portal runner → 'recursive'
    /// 2. Has oopspec → 'builtin'
    /// 3. `graphs_from(target) is None` → 'residual'
    /// 4. Otherwise → 'regular'
    ///
    /// Step 3 is the key: graphs_from returns None when the callee graph
    /// is not available OR not a candidate.
    pub fn guess_call_kind(&self, target: &CallTarget) -> CallKind {
        // Step 1: recursive (RPython call.py:119-120)
        let path = self.target_to_path(target);
        if let Some(ref p) = path {
            if self.portal_targets.contains(p) {
                return CallKind::Recursive;
            }
        }
        // Step 2: builtin (RPython call.py:135-136)
        if let Some(ref p) = path {
            if self.builtin_targets.contains(p) {
                return CallKind::Builtin;
            }
        }
        // Step 3+4: graphs_from check (RPython call.py:137-139)
        // graphs_from returns the graph ONLY if it's a candidate.
        if self.graphs_from(target).is_none() {
            CallKind::Residual
        } else {
            CallKind::Regular
        }
    }

    /// Get the callee graph for a call target, but only if it is a candidate.
    ///
    /// RPython: `CallControl.graphs_from(op, is_candidate)` (call.py:94-114).
    ///
    /// Returns the graph only if:
    /// 1. The graph exists (via function_graphs or resolve_method)
    /// 2. The graph is a candidate (in candidate_graphs)
    ///
    /// This is the gatekeeper: if graphs_from returns None, the call
    /// becomes residual. If it returns Some, the call is regular.
    pub fn graphs_from(&self, target: &CallTarget) -> Option<&FunctionGraph> {
        let path = self.target_to_path(target)?;
        // RPython call.py:100: is_candidate(graph)
        if !self.candidate_graphs.contains(&path) {
            return None;
        }
        // RPython call.py:94-101: returns the actual target graph.
        // For FunctionPath: direct lookup in function_graphs.
        // For Method: resolve_method returns the specific impl graph,
        // NOT whatever was first registered under the synthetic path.
        match target {
            CallTarget::Method {
                name,
                receiver_root,
            } => self.resolve_method(name, receiver_root.as_deref()),
            _ => self.function_graphs.get(&path),
        }
    }

    /// Convert a CallTarget to a CallPath for lookup.
    ///
    /// FunctionPath → direct path.
    /// Method → qualified CallPath([impl_type, method_name]).
    ///
    /// RPython: graph identity is by object pointer, not name.
    /// We emulate this with qualified paths that include the impl type,
    /// so different impls of the same method get distinct paths.
    fn target_to_path(&self, target: &CallTarget) -> Option<CallPath> {
        match target {
            CallTarget::FunctionPath { segments } => {
                Some(CallPath::from_segments(segments.iter().map(String::as_str)))
            }
            CallTarget::Method {
                name,
                receiver_root,
            } => {
                // resolve_method finds the concrete impl type.
                // We use that type to build a qualified path.
                let impl_type = self.resolve_method_impl_type(name, receiver_root.as_deref())?;
                Some(CallPath::from_segments([impl_type, name.as_str()]))
            }
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
    pub fn resolve_method(
        &self,
        name: &str,
        receiver_root: Option<&str>,
    ) -> Option<&FunctionGraph> {
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

    /// Like `resolve_method`, but returns the impl type name instead of the graph.
    /// Used by `target_to_path` to build qualified CallPaths.
    /// Like `resolve_method`, but returns the impl type name.
    /// All returned references borrow from `self`, not from `receiver_root`.
    fn resolve_method_impl_type<'b>(
        &'b self,
        name: &str,
        receiver_root: Option<&str>,
    ) -> Option<&'b str> {
        let impls = self.trait_method_impls.get(name)?;
        let concrete_impls: Vec<&String> = impls
            .iter()
            .filter(|t| !t.starts_with("<default methods of"))
            .collect();

        if concrete_impls.len() == 1 {
            return Some(concrete_impls[0]);
        }
        if let Some(receiver) = receiver_root {
            if !is_generic_receiver(receiver) {
                // Find the matching impl owned by self
                if let Some(impl_name) = impls.iter().find(|t| t.as_str() == receiver) {
                    return Some(impl_name);
                }
            }
        }
        if concrete_impls.is_empty() && impls.len() == 1 {
            return Some(&impls[0]);
        }
        None
    }

    /// Access the function graphs map (for inline pass).
    pub fn function_graphs(&self) -> &HashMap<CallPath, FunctionGraph> {
        &self.function_graphs
    }

    /// Access jitdriver static data.
    pub fn jitdrivers_sd(&self) -> &[JitDriverStaticData] {
        &self.jitdrivers_sd
    }

    // ── Elidable / loop-invariant registration ──────────────────────

    /// RPython: `getattr(func, "_elidable_function_", False)` (call.py:239).
    /// Mark a target as elidable (pure function).
    pub fn mark_elidable(&mut self, path: CallPath) {
        self.elidable_targets.insert(path);
    }

    /// RPython: `getattr(func, "_jit_loop_invariant_", False)` (call.py:240).
    /// Mark a target as loop-invariant.
    pub fn mark_loopinvariant(&mut self, path: CallPath) {
        self.loopinvariant_targets.insert(path);
    }

    /// RPython: call.py:239 — check if target has `_elidable_function_`.
    pub fn is_elidable(&self, target: &CallTarget) -> bool {
        self.target_to_path(target)
            .is_some_and(|p| self.elidable_targets.contains(&p))
    }

    /// RPython: call.py:240 — check if target has `_jit_loop_invariant_`.
    pub fn is_loopinvariant(&self, target: &CallTarget) -> bool {
        self.target_to_path(target)
            .is_some_and(|p| self.loopinvariant_targets.contains(&p))
    }

    // ── Graph-based analyzers (call.py:282-303) ─────────────────────

    /// RPython: RaiseAnalyzer.analyze() — transitive can-raise analysis.
    ///
    /// canraise.py:8-24: RaiseAnalyzer(BoolGraphAnalyzer)
    /// - `analyze_simple_operation`: checks `LL_OPERATIONS[op.opname].canraise`
    /// - `analyze_external_call`: `getattr(fnobj, 'canraise', True)`
    /// - `analyze_exceptblock_in_graph`: checks except blocks
    ///
    /// In majit we check per-operation canraise metadata via `op_can_raise()`,
    /// Abort terminators, and transitive Call analysis.
    fn analyze_can_raise(&self, path: &CallPath, seen: &mut HashSet<CallPath>) -> bool {
        self.analyze_can_raise_impl(path, seen, false)
    }

    /// Shared implementation for both raise analyzers.
    ///
    /// RPython has two separate RaiseAnalyzer instances (call.py:34-36):
    /// - `raise_analyzer`: normal mode
    /// - `raise_analyzer_ignore_memoryerror`: `do_ignore_memory_error()` mode
    ///
    /// `ignore_memoryerror` controls whether ops that can only raise
    /// MemoryError are treated as non-raising (canraise.py:11-17).
    fn analyze_can_raise_impl(
        &self,
        path: &CallPath,
        seen: &mut HashSet<CallPath>,
        ignore_memoryerror: bool,
    ) -> bool {
        if !seen.insert(path.clone()) {
            return false; // cycle → bottom_result
        }
        let graph = match self.function_graphs.get(path) {
            Some(g) => g,
            // RPython: analyze_external_call → getattr(fnobj, 'canraise', True)
            None => return true,
        };
        for block in &graph.blocks {
            // RPython: Abort terminator = except block path.
            // canraise.py:27-41: analyze_exceptblock_in_graph.
            if let Terminator::Abort { reason } = &block.terminator {
                if ignore_memoryerror && is_memoryerror_only(reason) {
                    // RPython: do_ignore_memory_error skips MemoryError-only
                    continue;
                }
                return true;
            }
            // RPython: analyze_simple_operation(op) per operation.
            // canraise.py:14-17: LL_OPERATIONS[op.opname].canraise
            for op in &block.operations {
                match &op.kind {
                    OpKind::Call { target, .. } => {
                        let callee_path = match self.target_to_path(target) {
                            Some(p) => p,
                            None => return true, // unresolvable → conservative
                        };
                        if self.analyze_can_raise_impl(&callee_path, seen, ignore_memoryerror) {
                            return true;
                        }
                    }
                    other => {
                        if op_can_raise(other, ignore_memoryerror) {
                            return true;
                        }
                    }
                }
            }
        }
        false
    }

    /// RPython: VirtualizableAnalyzer.analyze() (effectinfo.py:401-404).
    ///
    /// analyze_simple_operation: op.opname in ('jit_force_virtualizable',
    ///                                         'jit_force_virtual')
    fn analyze_forces_virtualizable(&self, path: &CallPath, seen: &mut HashSet<CallPath>) -> bool {
        if !seen.insert(path.clone()) {
            return false;
        }
        let graph = match self.function_graphs.get(path) {
            Some(g) => g,
            // RPython: external call → analyze_external_call → bottom_result (False).
            // VirtualizableAnalyzer does not override analyze_external_call.
            None => return false,
        };
        for block in &graph.blocks {
            for op in &block.operations {
                match &op.kind {
                    // RPython: jit_force_virtualizable / jit_force_virtual
                    OpKind::VableForce => return true,
                    OpKind::Call { target, .. } => {
                        let callee_path = match self.target_to_path(target) {
                            Some(p) => p,
                            None => continue, // external call → False
                        };
                        if self.analyze_forces_virtualizable(&callee_path, seen) {
                            return true;
                        }
                    }
                    _ => {}
                }
            }
        }
        false
    }

    /// RPython: RandomEffectsAnalyzer.analyze() (effectinfo.py:410-418).
    ///
    /// ```python
    /// class RandomEffectsAnalyzer(BoolGraphAnalyzer):
    ///     def analyze_external_call(self, funcobj, seen=None):
    ///         if funcobj.random_effects_on_gcobjs:
    ///             return True
    ///         return super().analyze_external_call(funcobj, seen)
    ///     def analyze_simple_operation(self, op, graphinfo):
    ///         return False
    /// ```
    ///
    /// Key: `analyze_simple_operation` always returns False. External calls
    /// only return True if `random_effects_on_gcobjs` is set. The default
    /// `analyze_external_call` returns `bottom_result()` = False
    /// (graphanalyze.py:60-69). "No graph" ≠ random effects in RPython.
    ///
    /// In majit: functions without graphs are treated as external calls
    /// → returns False (matching RPython's default external call behavior).
    fn analyze_random_effects(&self, path: &CallPath, seen: &mut HashSet<CallPath>) -> bool {
        if !seen.insert(path.clone()) {
            return false; // cycle → bottom_result
        }
        let graph = match self.function_graphs.get(path) {
            Some(g) => g,
            // RPython: external call → analyze_external_call() → bottom_result
            // (False) unless funcobj.random_effects_on_gcobjs. In majit, we
            // don't have that flag, so external functions default to False.
            None => return false,
        };
        // RPython: analyze_simple_operation always returns False.
        // Only recursive calls into graphs can propagate random effects.
        for block in &graph.blocks {
            for op in &block.operations {
                if let OpKind::Call { target, .. } = &op.kind {
                    let callee_path = match self.target_to_path(target) {
                        Some(p) => p,
                        // Unresolvable target = external call → False
                        None => continue,
                    };
                    if self.analyze_random_effects(&callee_path, seen) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// RPython: QuasiImmutAnalyzer.analyze() (effectinfo.py).
    ///
    /// analyze_simple_operation: op.opname == 'jit_force_quasi_immutable'.
    ///
    /// In majit: we don't have quasi-immutable ops in the model yet,
    /// so this always returns false. The transitive call check is still
    /// performed for future-proofing.
    fn analyze_can_invalidate(&self, path: &CallPath, seen: &mut HashSet<CallPath>) -> bool {
        if !seen.insert(path.clone()) {
            return false;
        }
        let graph = match self.function_graphs.get(path) {
            Some(g) => g,
            None => return false, // no graph → cannot invalidate (not conservative here)
        };
        for block in &graph.blocks {
            for op in &block.operations {
                // RPython: jit_force_quasi_immutable → true
                // majit: no such op yet, but check calls transitively
                if let OpKind::Call { target, .. } = &op.kind {
                    let callee_path = match self.target_to_path(target) {
                        Some(p) => p,
                        None => continue,
                    };
                    if self.analyze_can_invalidate(&callee_path, seen) {
                        return true;
                    }
                }
            }
        }
        false
    }

    // ── Cached analyzer wrappers ────────────────────────────────────

    /// Cached version of _canraise for a CallTarget.
    ///
    /// RPython call.py:337-355 — _canraise uses two analyzers:
    /// 1. raise_analyzer.can_raise(op)
    /// 2. raise_analyzer_ignore_memoryerror.can_raise(op)
    ///
    /// If (1) is True and (2) is False → "mem" (MemoryErrorOnly).
    /// If (1) is True and (2) is True → True.
    /// If (1) is False → False.
    fn cached_can_raise(&self, target: &CallTarget, cache: &mut AnalysisCache) -> CanRaise {
        let path = match self.target_to_path(target) {
            Some(p) => p,
            None => return CanRaise::Yes,
        };
        if let Some(&result) = cache.can_raise.get(&path) {
            return result;
        }
        // RPython call.py:342: self.raise_analyzer.can_raise(op)
        let mut seen = HashSet::new();
        let can_raise = self.analyze_can_raise_impl(&path, &mut seen, false);
        let result = if !can_raise {
            // RPython call.py:348: return False
            CanRaise::No
        } else {
            // RPython call.py:343: self.raise_analyzer_ignore_memoryerror.can_raise(op)
            let mut seen2 = HashSet::new();
            let can_raise_non_memoryerror = self.analyze_can_raise_impl(&path, &mut seen2, true);
            if can_raise_non_memoryerror {
                // RPython call.py:344: return True
                CanRaise::Yes
            } else {
                // RPython: return "mem"
                CanRaise::MemoryErrorOnly
            }
        };
        cache.can_raise.insert(path, result);
        result
    }

    /// Cached version of analyze_forces_virtualizable for a CallTarget.
    /// RPython: VirtualizableAnalyzer external calls → bottom_result (False).
    fn cached_forces_virtualizable(&self, target: &CallTarget, cache: &mut AnalysisCache) -> bool {
        let path = match self.target_to_path(target) {
            Some(p) => p,
            None => return false, // external → False (RPython bottom_result)
        };
        if let Some(&result) = cache.forces_virtualizable.get(&path) {
            return result;
        }
        let mut seen = HashSet::new();
        let result = self.analyze_forces_virtualizable(&path, &mut seen);
        cache.forces_virtualizable.insert(path, result);
        result
    }

    /// Cached version of analyze_random_effects for a CallTarget.
    /// RPython: RandomEffectsAnalyzer defaults to False for external calls.
    fn cached_random_effects(&self, target: &CallTarget, cache: &mut AnalysisCache) -> bool {
        let path = match self.target_to_path(target) {
            Some(p) => p,
            None => return false, // external call → False (RPython default)
        };
        if let Some(&result) = cache.random_effects.get(&path) {
            return result;
        }
        let mut seen = HashSet::new();
        let result = self.analyze_random_effects(&path, &mut seen);
        cache.random_effects.insert(path, result);
        result
    }

    /// Cached version of analyze_can_invalidate for a CallTarget.
    fn cached_can_invalidate(&self, target: &CallTarget, cache: &mut AnalysisCache) -> bool {
        let path = match self.target_to_path(target) {
            Some(p) => p,
            None => return false,
        };
        if let Some(&result) = cache.can_invalidate.get(&path) {
            return result;
        }
        let mut seen = HashSet::new();
        let result = self.analyze_can_invalidate(&path, &mut seen);
        cache.can_invalidate.insert(path, result);
        result
    }

    // ── _canraise + getcalldescr (call.py:210-355) ──────────────────

    /// RPython: CallControl._canraise(op) (call.py:337-355).
    ///
    /// ```python
    /// def _canraise(self, op):
    ///     if op.opname == 'pseudo_call_cannot_raise':
    ///         return False
    ///     try:
    ///         if self.raise_analyzer.can_raise(op):
    ///             if self.raise_analyzer_ignore_memoryerror.can_raise(op):
    ///                 return True
    ///             else:
    ///                 return "mem"
    ///         else:
    ///             return False
    ///     except DelayedPointer:
    ///         return True
    /// ```
    pub fn canraise(&self, target: &CallTarget, cache: &mut AnalysisCache) -> CanRaise {
        self.cached_can_raise(target, cache)
    }

    /// RPython: CallControl.getcalldescr(op, oopspecindex, extraeffect, ...)
    /// (call.py:210-335).
    ///
    /// Determines the effect classification for a call target by running
    /// graph-based analyzers, then builds and returns a CallDescriptor
    /// with the computed EffectInfo.
    ///
    /// ```python
    /// def getcalldescr(self, op, oopspecindex=OS_NONE,
    ///                  extraeffect=None, extradescr=None, calling_graph=None):
    ///     ...
    ///     random_effects = self.randomeffects_analyzer.analyze(op)
    ///     if random_effects:
    ///         extraeffect = EF_RANDOM_EFFECTS
    ///     can_invalidate = random_effects or self.quasiimmut_analyzer.analyze(op)
    ///     if extraeffect is None:
    ///         if self.virtualizable_analyzer.analyze(op):
    ///             extraeffect = EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE
    ///         elif loopinvariant:
    ///             extraeffect = EF_LOOPINVARIANT
    ///         elif elidable:
    ///             cr = self._canraise(op)
    ///             ...
    ///         elif self._canraise(op):
    ///             extraeffect = EF_CAN_RAISE
    ///         else:
    ///             extraeffect = EF_CANNOT_RAISE
    ///     ...
    ///     effectinfo = effectinfo_from_writeanalyze(...)
    ///     return self.cpu.calldescrof(FUNC, NON_VOID_ARGS, RESULT, effectinfo)
    /// ```
    pub fn getcalldescr(
        &self,
        target: &CallTarget,
        arg_types: Vec<Type>,
        result_type: Type,
        oopspecindex: OopSpecIndex,
        extraeffect: Option<ExtraEffect>,
        cache: &mut AnalysisCache,
    ) -> CallDescriptor {
        // RPython call.py:239-240: extract flags
        let elidable = self.is_elidable(target);
        let loopinvariant = self.is_loopinvariant(target);

        // RPython call.py:223-234: check the number and type of arguments.
        //   FUNC = op.args[0].concretetype.TO
        //   if NON_VOID_ARGS != [T for T in FUNC.ARGS if T is not Void]:
        //       raise Exception(...)
        //   if RESULT != FUNC.RESULT:
        //       raise Exception(...)
        //
        // In majit, we validate by looking up the callee's FunctionGraph inputs.
        // RPython raises on mismatch; we warn (callee signatures are approximate
        // in static analysis, so hard errors would be too strict).
        if !arg_types.is_empty() {
            if let Some(path) = self.target_to_path(target) {
                if let Some(graph) = self.function_graphs.get(&path) {
                    let expected_arity = graph.block(graph.startblock).inputargs.len();
                    if arg_types.len() != expected_arity {
                        eprintln!(
                            "[getcalldescr] WARNING: {target} expects {expected_arity} args \
                             but got {} (NON_VOID_ARGS mismatch)",
                            arg_types.len()
                        );
                    }
                }
            }
        }

        // RPython call.py:282: random_effects = self.randomeffects_analyzer.analyze(op)
        let random_effects = self.cached_random_effects(target, cache);
        let mut extraeffect = extraeffect;
        if random_effects {
            extraeffect = Some(ExtraEffect::RandomEffects);
        }

        // RPython call.py:285: can_invalidate = random_effects or quasiimmut_analyzer
        let can_invalidate = random_effects || self.cached_can_invalidate(target, cache);

        // RPython call.py:286-303: determine extraeffect
        if extraeffect.is_none() {
            extraeffect = Some(if self.cached_forces_virtualizable(target, cache) {
                // call.py:288
                ExtraEffect::ForcesVirtualOrVirtualizable
            } else if loopinvariant {
                // call.py:290
                ExtraEffect::LoopInvariant
            } else if elidable {
                // call.py:292-298
                match self.canraise(target, cache) {
                    CanRaise::No => ExtraEffect::ElidableCannotRaise,
                    CanRaise::MemoryErrorOnly => ExtraEffect::ElidableOrMemoryError,
                    CanRaise::Yes => ExtraEffect::ElidableCanRaise,
                }
            } else if matches!(
                self.canraise(target, cache),
                CanRaise::Yes | CanRaise::MemoryErrorOnly
            ) {
                // call.py:299-300
                ExtraEffect::CanRaise
            } else {
                // call.py:302
                ExtraEffect::CannotRaise
            });
        }

        let extraeffect = extraeffect.unwrap_or(ExtraEffect::CanRaise);

        // RPython call.py:249-251: loopinvariant functions must have no args
        if loopinvariant && !arg_types.is_empty() {
            panic!(
                "getcalldescr: arguments not supported for loop-invariant \
                 function {target}"
            );
        }

        // RPython call.py:305-318: check that the result is really as expected
        if loopinvariant && extraeffect != ExtraEffect::LoopInvariant {
            panic!(
                "getcalldescr: {target} is marked loop-invariant but got \
                 extraeffect={extraeffect:?}"
            );
        }
        if elidable {
            if !matches!(
                extraeffect,
                ExtraEffect::ElidableCannotRaise
                    | ExtraEffect::ElidableOrMemoryError
                    | ExtraEffect::ElidableCanRaise
            ) {
                panic!(
                    "getcalldescr: {target} is marked elidable but got \
                     extraeffect={extraeffect:?}"
                );
            }
            // RPython call.py:315-318: elidable function must have a result
            if result_type == Type::Void {
                panic!("getcalldescr: {target} is elidable but has no result");
            }
        }

        // RPython call.py:320-324: effectinfo_from_writeanalyze(...)
        let effectinfo = effectinfo_from_writeanalyze(
            target,
            extraeffect,
            oopspecindex,
            can_invalidate,
            &self.function_graphs,
            self,
            &mut cache.descr_indices,
        );

        // RPython call.py:326-332: assert post-conditions
        if elidable || loopinvariant {
            assert!(
                effectinfo.extra_effect < ExtraEffect::ForcesVirtualOrVirtualizable,
                "getcalldescr: elidable/loopinvariant {target} has \
                 effect {:?} >= ForcesVirtualOrVirtualizable",
                effectinfo.extra_effect
            );
        }

        // RPython call.py:334-335: cpu.calldescrof(FUNC, NON_VOID_ARGS, RESULT, effectinfo)
        CallDescriptor {
            target: target.clone(),
            effect_info: effectinfo,
        }
    }

    /// RPython: calldescr_canraise(calldescr) (call.py:357-359).
    pub fn calldescr_canraise(&self, calldescr: &CallDescriptor) -> bool {
        calldescr.effect_info.check_can_raise(false)
    }
}

impl Default for CallControl {
    fn default() -> Self {
        Self::new()
    }
}

// ── effectinfo_from_writeanalyze (effectinfo.py:276-378) ──────────
//
// RPython: effectinfo_from_writeanalyze(effects, cpu, extraeffect, ...)
// Builds EffectInfo from the write-analysis result set.
//
// In RPython, the ReadWriteAnalyzer produces a set of tuples like:
//   ("struct", T, fieldname), ("readstruct", T, fieldname),
//   ("array", T), ("readarray", T), etc.
// These are converted to field/array descriptor bitsets.
//
// In majit, we scan the callee graph's ops directly for
// FieldRead/FieldWrite/ArrayRead/ArrayWrite and collect their
// descriptor indices into EffectInfo's bitset fields.

/// RPython: effectinfo_from_writeanalyze() (effectinfo.py:276-378).
///
/// Scans the callee's graph for field/array read/write operations
/// and populates the corresponding bitset fields in EffectInfo.
fn effectinfo_from_writeanalyze(
    target: &CallTarget,
    extraeffect: ExtraEffect,
    oopspecindex: OopSpecIndex,
    can_invalidate: bool,
    function_graphs: &HashMap<CallPath, FunctionGraph>,
    cc: &CallControl,
    descr_indices: &mut DescrIndexRegistry,
) -> EffectInfo {
    // RPython effectinfo.py:279-281:
    // if effects is top_set or extraeffect == EF_RANDOM_EFFECTS:
    //     readonly/write = None (=> check_* always returns True)
    if extraeffect == ExtraEffect::RandomEffects {
        return EffectInfo {
            extra_effect: ExtraEffect::RandomEffects,
            oopspec_index: oopspecindex,
            readonly_descrs_fields: !0, // all bits set = top_set
            write_descrs_fields: !0,
            readonly_descrs_arrays: !0,
            write_descrs_arrays: !0,
            can_invalidate,
            single_write_descr_array: None,
        };
    }

    // For elidable/loopinvariant: ignore writes (effectinfo.py:181-186).
    // RPython: if elidable or loopinvariant: write_descrs = frozenset()
    let ignore_writes = matches!(
        extraeffect,
        ExtraEffect::ElidableCannotRaise
            | ExtraEffect::ElidableOrMemoryError
            | ExtraEffect::ElidableCanRaise
            | ExtraEffect::LoopInvariant
    );

    // Collect raw read/write bitsets from graph traversal.
    let mut read_fields: u64 = 0;
    let mut write_fields: u64 = 0;
    let mut read_arrays: u64 = 0;
    let mut write_arrays: u64 = 0;
    // effectinfo.py:201-206: collect actual array write DescrRefs.
    let mut array_write_descrs: Vec<majit_ir::descr::DescrRef> = Vec::new();

    if let Some(path) = cc.target_to_path(target) {
        let mut seen = HashSet::new();
        collect_readwrite_effects(
            &path,
            function_graphs,
            cc,
            descr_indices,
            &mut seen,
            &mut read_fields,
            &mut write_fields,
            &mut read_arrays,
            &mut write_arrays,
            &mut array_write_descrs,
        );
    }

    // RPython effectinfo.py:345-360: readonly = reads that have NO
    // corresponding write. If a field is both read and written, it
    // goes into write_descrs only, NOT readonly_descrs.
    //
    // effectinfo.py:346: tupw = ("struct",) + tup[1:]
    //                    if tupw not in effects:
    //                        add_struct(readonly_descrs_fields, tup)
    let readonly_descrs_fields = read_fields & !write_fields;
    let readonly_descrs_arrays = read_arrays & !write_arrays;
    let mut write_descrs_fields = write_fields;
    let mut write_descrs_arrays = write_arrays;

    // RPython effectinfo.py:169-181: for elidable/loopinvariant,
    // ignore writes (_write_descrs_* = frozenset()).
    // This must clear BOTH the bitset AND the descr set, so that
    // single_write_descr_array becomes None (effectinfo.py:201-206).
    if ignore_writes {
        write_descrs_fields = 0;
        write_descrs_arrays = 0;
        array_write_descrs.clear();
    }

    // effectinfo.py:201-206: single_write_descr_array
    // RPython: if len(_write_descrs_arrays) == 1: [single] = _write_descrs_arrays
    let single_write_descr_array = if array_write_descrs.len() == 1 {
        Some(array_write_descrs.into_iter().next().unwrap())
    } else {
        None
    };

    EffectInfo {
        extra_effect: extraeffect,
        oopspec_index: oopspecindex,
        readonly_descrs_fields,
        write_descrs_fields,
        readonly_descrs_arrays,
        write_descrs_arrays,
        can_invalidate,
        single_write_descr_array,
    }
}

/// RPython: `op.args[0].concretetype` — resolve ARRAY identity.
///
/// In RPython, array identity is the ARRAY lltype (e.g. `GcArray(Signed)`),
/// determined by the ELEMENT TYPE, not by which field holds the array.
/// Two fields `a: GcArray(Signed)` and `b: GcArray(Signed)` share the
/// same descriptor because they have the same element type.
///
/// Resolution order:
/// 1. If the op has `array_type_id` from parser (direct variable access), use it
/// 2. If the base comes from FieldRead, look up the field's type in struct_fields
///    to determine the element type (RPython: op.args[0].concretetype.TO)
/// 3. Otherwise return None (conservative: falls back to item_ty-only keying)
fn resolve_array_identity(
    base: &crate::model::ValueId,
    op_array_type_id: &Option<String>,
    value_producers: &HashMap<crate::model::ValueId, &crate::model::OpKind>,
    cc: &CallControl,
) -> Option<String> {
    // 1. Parser-set element type (from FnArg or typed let binding).
    if op_array_type_id.is_some() {
        return op_array_type_id.clone();
    }
    // 2. Trace back to producer — RPython: op.args[0].concretetype.
    if let Some(producer) = value_producers.get(base) {
        if let OpKind::FieldRead { field, .. } = producer {
            // Look up the field's type in the struct registry.
            // self.points → owner="MyStruct", name="points"
            // struct_fields["MyStruct"]["points"] = "Vec<Point>"
            // → element type = "Point"
            if let Some(owner) = &field.owner_root {
                if let Some(field_type_str) = cc.field_type(owner, &field.name) {
                    // Extract element type from container: "Vec<Point>" → "Point"
                    return extract_element_type_from_str(field_type_str);
                }
            }
        }
    }
    None
}

/// Extract element type from a type string like `"Vec<Point>"` → `"Point"`.
/// RPython equivalent: `ARRAY.OF` from `GcArray(T)` → `T`.
fn extract_element_type_from_str(type_str: &str) -> Option<String> {
    // Find first '<' and matching '>'
    let start = type_str.find('<')?;
    let end = type_str.rfind('>')?;
    if start < end {
        Some(type_str[start + 1..end].trim().to_string())
    } else {
        None
    }
}

/// Transitive read/write effect collection.
///
/// RPython: ReadWriteAnalyzer.analyze() — traverses callee graphs.
/// Produces a set of tuples: ("struct"/"readstruct"/"array"/"readarray", ...).
///
/// We collect raw reads and writes separately into bitsets. The caller
/// (`effectinfo_from_writeanalyze`) then applies the RPython rule:
/// "readonly = reads & ~writes" (effectinfo.py:345-360).
fn collect_readwrite_effects(
    path: &CallPath,
    function_graphs: &HashMap<CallPath, FunctionGraph>,
    cc: &CallControl,
    descr_indices: &mut DescrIndexRegistry,
    seen: &mut HashSet<CallPath>,
    read_fields: &mut u64,
    write_fields: &mut u64,
    read_arrays: &mut u64,
    write_arrays: &mut u64,
    // effectinfo.py:201-206: collect actual array write DescrRefs
    // for single_write_descr_array population.
    array_write_descrs: &mut Vec<majit_ir::descr::DescrRef>,
) {
    if !seen.insert(path.clone()) {
        return;
    }
    let graph = match function_graphs.get(path) {
        Some(g) => g,
        None => {
            // No graph available → top_set (all bits set).
            *read_fields = !0;
            *write_fields = !0;
            *read_arrays = !0;
            *write_arrays = !0;
            return;
        }
    };

    // RPython: the rtyped graph gives op.args[0].concretetype directly.
    // In majit, we build a producer map to resolve ValueId → producing OpKind,
    // so we can determine the array identity from the base operand's provenance.
    let value_producers: HashMap<crate::model::ValueId, &crate::model::OpKind> = graph
        .blocks
        .iter()
        .flat_map(|b| &b.operations)
        .filter_map(|op| op.result.map(|vid| (vid, &op.kind)))
        .collect();

    for block in &graph.blocks {
        for op in &block.operations {
            match &op.kind {
                // RPython: ("readstruct", T, fieldname)
                OpKind::FieldRead { field, .. } => {
                    // RPython: cpu.fielddescrof(T, fieldname).get_ei_index()
                    let idx = descr_indices.field_index(&field.owner_root, &field.name);
                    *read_fields |= 1u64 << idx;
                }
                // RPython: ("struct", T, fieldname)
                OpKind::FieldWrite { field, .. } => {
                    let idx = descr_indices.field_index(&field.owner_root, &field.name);
                    *write_fields |= 1u64 << idx;
                }
                // RPython: ("readarray", T)
                OpKind::ArrayRead {
                    base,
                    item_ty,
                    array_type_id,
                    ..
                } => {
                    // RPython: op.args[0].concretetype → cpu.arraydescrof(ARRAY)
                    let resolved_id =
                        resolve_array_identity(base, array_type_id, &value_producers, cc);
                    let idx =
                        descr_indices.array_index(value_type_discriminant(item_ty), &resolved_id);
                    *read_arrays |= 1u64 << idx;
                }
                // RPython: ("array", T)
                OpKind::ArrayWrite {
                    base,
                    item_ty,
                    array_type_id,
                    ..
                } => {
                    let resolved_id =
                        resolve_array_identity(base, array_type_id, &value_producers, cc);
                    let idx =
                        descr_indices.array_index(value_type_discriminant(item_ty), &resolved_id);
                    *write_arrays |= 1u64 << idx;
                    // effectinfo.py:298,307: cpu.arraydescrof(ARRAY) → DescrRef.
                    // Keyed on (item_ty, resolved_id) for ARRAY identity.
                    // Dedup by descriptor index (frozenset semantics).
                    if !array_write_descrs.iter().any(|d| d.index() == idx) {
                        let ir_type = match item_ty {
                            crate::model::ValueType::Int | crate::model::ValueType::State => {
                                majit_ir::value::Type::Int
                            }
                            crate::model::ValueType::Ref | crate::model::ValueType::Unknown => {
                                majit_ir::value::Type::Ref
                            }
                            crate::model::ValueType::Float => majit_ir::value::Type::Float,
                            crate::model::ValueType::Void => majit_ir::value::Type::Void,
                        };
                        // RPython: get_type_flag(ARRAY.OF) determines the flag.
                        // isinstance(TYPE, lltype.Struct) → FLAG_STRUCT.
                        let elem_name = resolved_id.as_deref();
                        let is_struct = elem_name.is_some_and(|n| cc.is_known_struct(n));
                        let flag = majit_ir::descr::ArrayFlag::from_item_type(ir_type, is_struct);
                        let mut ad = majit_ir::descr::SimpleArrayDescr::with_flag(
                            idx, 0, 8, 0, ir_type, flag,
                        );
                        // RPython: descr.py:372-375 — for array-of-structs:
                        //   descrs = heaptracker.all_interiorfielddescrs(
                        //       gccache, ARRAY, get_field_descr=get_interiorfield_descr)
                        //   arraydescr.all_interiorfielddescrs = descrs
                        if is_struct {
                            if let Some(struct_name) = elem_name {
                                // RPython: InteriorFieldDescr needs the parent arraydescr.
                                // Create Arc first, then build interior field descrs.
                                let ad_arc = std::sync::Arc::new(ad);
                                let fielddescrs =
                                    build_interior_fielddescrs(cc, struct_name, ad_arc.clone());
                                if !fielddescrs.is_empty() {
                                    // Clone the inner to set fielddescrs, then re-wrap.
                                    let mut ad_mut = (*ad_arc).clone();
                                    ad_mut.set_all_fielddescrs(fielddescrs);
                                    array_write_descrs.push(std::sync::Arc::new(ad_mut));
                                } else {
                                    array_write_descrs.push(ad_arc);
                                }
                            } else {
                                array_write_descrs.push(std::sync::Arc::new(ad));
                            }
                        } else {
                            array_write_descrs.push(std::sync::Arc::new(ad));
                        }
                    }
                }
                // Recursive: follow calls.
                OpKind::Call { target, .. } => {
                    if let Some(callee_path) = cc.target_to_path(target) {
                        collect_readwrite_effects(
                            &callee_path,
                            function_graphs,
                            cc,
                            descr_indices,
                            seen,
                            read_fields,
                            write_fields,
                            read_arrays,
                            write_arrays,
                            array_write_descrs,
                        );
                    } else {
                        // Unresolvable call → top_set.
                        *read_fields = !0;
                        *write_fields = !0;
                        *read_arrays = !0;
                        *write_arrays = !0;
                        return;
                    }
                }
                _ => {}
            }
        }
    }
}

/// RPython: `heaptracker.all_interiorfielddescrs(gccache, ARRAY)`.
///
/// For an array-of-structs (element type is a known struct), iterate
/// `STRUCT._names` and create `InteriorFieldDescr(arraydescr, fielddescr)`
/// for each field. This mirrors heaptracker.py:74-92 with
/// `get_field_descr=get_interiorfield_descr` (descr.py:373-375).
///
/// Each interior field descriptor wraps:
/// - The parent `ArrayDescr` (shared Arc)
/// - A `FieldDescr` with actual offset/size/type from the struct definition
fn build_interior_fielddescrs(
    cc: &CallControl,
    struct_name: &str,
    array_descr: std::sync::Arc<majit_ir::descr::SimpleArrayDescr>,
) -> Vec<majit_ir::descr::DescrRef> {
    let fields = match cc.struct_fields.fields.get(struct_name) {
        Some(f) => f,
        None => return Vec::new(),
    };
    // RPython: heaptracker.py:83-91 — iterate STRUCT._names,
    // skip Void fields and typeptr.
    let mut offset: usize = 0;
    let mut result = Vec::new();
    for (i, (field_name, field_type_str)) in fields.iter().enumerate() {
        // RPython: FIELD = getattr(STRUCT, name); if FIELD is Void: continue
        let (field_type, field_size) = field_type_from_rust_type(field_type_str);
        if field_type == majit_ir::value::Type::Void {
            continue;
        }
        // RPython: get_field_descr(gccache, REALARRAY.OF, name)
        // → FieldDescr(name, offset, size, flag, index_in_parent, is_pure)
        let align = field_size;
        if align > 0 {
            offset = (offset + align - 1) & !(align - 1);
        }
        // RPython: index_in_parent = heaptracker.get_fielddescr_index_in(STRUCT, name)
        let index_in_parent = i as u32;
        let is_immutable = false;
        let fd = std::sync::Arc::new(majit_ir::descr::SimpleFieldDescr::new_with_name(
            index_in_parent,
            offset,
            field_size,
            field_type,
            is_immutable,
            format!("{}.{}", struct_name, field_name),
        ));
        // RPython: descr.py:436 — InteriorFieldDescr(arraydescr, fielddescr)
        let ifd = majit_ir::descr::SimpleInteriorFieldDescr::new(
            index_in_parent,
            array_descr.clone(),
            fd,
        );
        result.push(std::sync::Arc::new(ifd) as majit_ir::descr::DescrRef);
        offset += field_size;
    }
    result
}

/// RPython: `get_type_flag(TYPE)` (descr.py:241-254).
///
/// Maps a Rust type string to (IR type, size in bytes).
/// Ptr(gc) → (Ref, 8), Float → (Float, 8), signed int → (Int, N).
fn field_type_from_rust_type(type_str: &str) -> (majit_ir::value::Type, usize) {
    match type_str {
        // RPython: isinstance(TYPE, lltype.Ptr) and TYPE.TO._gckind == 'gc' → FLAG_POINTER
        s if s.starts_with('&')
            || s.starts_with("Box<")
            || s.starts_with("Arc<")
            || s.starts_with("Rc<")
            || s.starts_with("Vec<")
            || s.starts_with("Option<")
            || s == "String" =>
        {
            (majit_ir::value::Type::Ref, 8)
        }
        // RPython: TYPE is lltype.Float → FLAG_FLOAT
        "f64" => (majit_ir::value::Type::Float, 8),
        "f32" => (majit_ir::value::Type::Float, 4),
        // RPython: isinstance(TYPE, lltype.Number) and signed → FLAG_SIGNED
        "i64" | "isize" => (majit_ir::value::Type::Int, 8),
        "i32" => (majit_ir::value::Type::Int, 4),
        "i16" => (majit_ir::value::Type::Int, 2),
        "i8" => (majit_ir::value::Type::Int, 1),
        "u64" | "usize" => (majit_ir::value::Type::Int, 8),
        "u32" => (majit_ir::value::Type::Int, 4),
        "u16" => (majit_ir::value::Type::Int, 2),
        "u8" => (majit_ir::value::Type::Int, 1),
        "bool" => (majit_ir::value::Type::Int, 1),
        // RPython: Void fields are skipped
        "()" => (majit_ir::value::Type::Void, 0),
        // Unknown type — treat as word-sized reference (conservative)
        _ => (majit_ir::value::Type::Ref, 8),
    }
}

/// RPython: `RaiseAnalyzer.analyze_simple_operation(op)` (canraise.py:14-17).
///
/// ```python
/// canraise = LL_OPERATIONS[op.opname].canraise
/// return bool(canraise) and canraise != (self.ignore_exact_class,)
/// ```
///
/// Returns true if the operation itself (not counting transitive calls)
/// can raise an exception. When `ignore_memoryerror` is true, operations
/// that can only raise MemoryError are treated as non-raising.
fn op_can_raise(op: &OpKind, ignore_memoryerror: bool) -> bool {
    // RPython canraise.py:14-17:
    //   canraise = LL_OPERATIONS[op.opname].canraise
    //   return bool(canraise) and canraise != (self.ignore_exact_class,)
    //
    // canraise.py:18: unknown op → log.WARNING + return True
    //
    // Each op has a canraise tuple. When ignore_exact_class == MemoryError,
    // ops that can ONLY raise MemoryError are treated as non-raising.
    match op {
        // ── Known non-raising ops (canraise = ()) ─────────────────
        // RPython LL: getfield_gc, setfield_gc → cannot raise
        OpKind::FieldRead { .. } | OpKind::FieldWrite { .. } => false,
        // RPython LL: getarrayitem_gc, setarrayitem_gc → cannot raise
        OpKind::ArrayRead { .. } | OpKind::ArrayWrite { .. } => false,
        // RPython LL: int_add, int_sub, int_lt, int_and, etc → cannot raise
        // (non-ovf, non-div arithmetic)
        OpKind::BinOp { op, .. }
            if !op.contains("div")
                && !op.contains("mod")
                && !op.contains("rem")
                && !op.contains("ovf") =>
        {
            false
        }
        // RPython LL: int_neg, bool_not → cannot raise
        OpKind::UnaryOp { op, .. } if !op.contains("ovf") => false,
        // RPython LL: same_as, cast_*, hint → cannot raise
        OpKind::Input { .. } | OpKind::ConstInt(_) => false,
        // JIT-specific ops that cannot raise
        OpKind::GuardTrue { .. }
        | OpKind::GuardFalse { .. }
        | OpKind::GuardValue { .. }
        | OpKind::Live => false,
        // Virtualizable field/array access (from boxes, no heap) → cannot raise
        OpKind::VableFieldRead { .. }
        | OpKind::VableFieldWrite { .. }
        | OpKind::VableArrayRead { .. }
        | OpKind::VableArrayWrite { .. } => false,
        // Post-jtransform call ops: raise is determined by their descriptor,
        // not by op_can_raise. These are not "simple operations" in RPython
        // terms — they're handled by analyze() → analyze_direct_call.
        OpKind::CallResidual { .. }
        | OpKind::CallElidable { .. }
        | OpKind::CallMayForce { .. }
        | OpKind::InlineCall { .. }
        | OpKind::RecursiveCall { .. } => false,

        // ── Known raising ops ─────────────────────────────────────
        // RPython LL: jit_force_virtualizable → canraise
        OpKind::VableForce => true,
        // RPython LL: int_floordiv, int_mod → canraise = (ZeroDivisionError,)
        OpKind::BinOp { .. } => true, // div/mod/rem/ovf (others matched above)
        // RPython LL: int_neg_ovf → canraise = (OverflowError,)
        OpKind::UnaryOp { .. } => true, // ovf (others matched above)

        // ── Calls handled by analyze() dispatch, not here ─────────
        // RPython: Call ops dispatch to analyze_direct_call/analyze_external_call.
        // op_can_raise is only for "simple operations" (non-call).
        // But if we see a Call here (shouldn't happen in normal flow),
        // be conservative.
        OpKind::Call { .. } => true,

        // ── Unknown ops: canraise.py:18 → True (conservative) ─────
        // RPython: log.WARNING("Unknown operation: %s" % op.opname)
        //          return True
        OpKind::Unknown { .. } => true,
    }
}

/// Check if an Abort reason indicates MemoryError-only.
///
/// RPython: `do_ignore_memory_error()` sets `ignore_exact_class = MemoryError`.
/// Then `canraise != (self.ignore_exact_class,)` filters it out.
///
/// In majit, Abort carries a reason string. We check for MemoryError
/// indicators. This matches:
/// - "MemoryError" — explicit MemoryError raise
/// - "alloc" / "allocation" — memory allocation failures
fn is_memoryerror_only(reason: &str) -> bool {
    let r = reason.to_lowercase();
    r.contains("memoryerror") || r.contains("out of memory")
}

/// Map ValueType to a small integer for array descriptor indexing.
fn value_type_discriminant(ty: &crate::model::ValueType) -> u8 {
    use crate::model::ValueType;
    match ty {
        ValueType::Int => 0,
        ValueType::Ref => 1,
        ValueType::Float => 2,
        ValueType::Void => 3,
        ValueType::State => 4,
        ValueType::Unknown => 5,
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

// ── Builtin call descriptor table ──
//
// RPython effectinfo.py + call.py parity: pre-classified call targets.
// The codewriter matches function names to determine effect category
// and oopspec index without graph-level analysis.

const INT_ARITH_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["w_int_add"]),
    CallTargetPattern::FunctionPath(&["w_int_sub"]),
    CallTargetPattern::FunctionPath(&["w_int_mul"]),
    CallTargetPattern::FunctionPath(&["int_add"]),
    CallTargetPattern::FunctionPath(&["int_sub"]),
    CallTargetPattern::FunctionPath(&["int_mul"]),
    CallTargetPattern::FunctionPath(&["int_bitand"]),
    CallTargetPattern::FunctionPath(&["int_bitor"]),
    CallTargetPattern::FunctionPath(&["int_bitxor"]),
    // Qualified paths (annotator uses these for type inference).
    CallTargetPattern::FunctionPath(&["crate", "math", "w_int_add"]),
    CallTargetPattern::FunctionPath(&["crate", "math", "w_int_sub"]),
    CallTargetPattern::FunctionPath(&["crate", "math", "w_int_mul"]),
];

const INT_CMP_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["int_lt"]),
    CallTargetPattern::FunctionPath(&["int_le"]),
    CallTargetPattern::FunctionPath(&["int_gt"]),
    CallTargetPattern::FunctionPath(&["int_ge"]),
    CallTargetPattern::FunctionPath(&["int_eq"]),
    CallTargetPattern::FunctionPath(&["int_ne"]),
];

const FLOAT_ARITH_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["w_float_add"]),
    CallTargetPattern::FunctionPath(&["w_float_sub"]),
    CallTargetPattern::FunctionPath(&["float_add"]),
    CallTargetPattern::FunctionPath(&["float_sub"]),
    CallTargetPattern::FunctionPath(&["float_mul"]),
    CallTargetPattern::FunctionPath(&["float_truediv"]),
];

const FLOAT_CMP_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["float_lt"]),
    CallTargetPattern::FunctionPath(&["float_le"]),
    CallTargetPattern::FunctionPath(&["float_gt"]),
    CallTargetPattern::FunctionPath(&["float_ge"]),
    CallTargetPattern::FunctionPath(&["float_eq"]),
    CallTargetPattern::FunctionPath(&["float_ne"]),
];

// effectinfo.py: EF_ELIDABLE_CAN_RAISE — may raise (e.g. ZeroDivisionError)
// int_floordiv and int_mod have distinct oopspec indices (IntPyDiv vs IntPyMod)
// because intbounds.rs optimizes them differently.
const INT_FLOORDIV_TARGETS: &[CallTargetPattern] =
    &[CallTargetPattern::FunctionPath(&["int_floordiv"])];

const INT_MOD_TARGETS: &[CallTargetPattern] = &[CallTargetPattern::FunctionPath(&["int_mod"])];

const FLOAT_DIV_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["float_floordiv"]),
    CallTargetPattern::FunctionPath(&["float_mod"]),
];

const INT_SHIFT_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["int_lshift"]),
    CallTargetPattern::FunctionPath(&["int_rshift"]),
];

const INT_POW_TARGETS: &[CallTargetPattern] = &[CallTargetPattern::FunctionPath(&["int_pow"])];

// effectinfo.py: OS_STR_CONCAT etc. — string operations with oopspec
const STR_CONCAT_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["str_concat"]),
    CallTargetPattern::FunctionPath(&["jit_str_concat"]),
];

const STR_CMP_TARGETS: &[CallTargetPattern] =
    &[CallTargetPattern::FunctionPath(&["jit_str_compare"])];

// effectinfo.py: list operations (may raise IndexError)
const LIST_GETITEM_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["jit_list_getitem"]),
    CallTargetPattern::FunctionPath(&["w_list_getitem"]),
];

const LIST_SETITEM_TARGETS: &[CallTargetPattern] =
    &[CallTargetPattern::FunctionPath(&["jit_list_setitem"])];

const LIST_APPEND_TARGETS: &[CallTargetPattern] =
    &[CallTargetPattern::FunctionPath(&["jit_list_append"])];

// effectinfo.py: tuple access (elidable, cannot raise for valid index)
const TUPLE_GETITEM_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["jit_tuple_getitem"]),
    CallTargetPattern::FunctionPath(&["w_tuple_getitem"]),
];

// effectinfo.py: constructor-like (cannot raise, elidable)
const INT_NEW_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["w_int_new"]),
    CallTargetPattern::FunctionPath(&["jit_w_int_new"]),
];

const FLOAT_NEW_TARGETS: &[CallTargetPattern] = &[
    CallTargetPattern::FunctionPath(&["w_float_new"]),
    CallTargetPattern::FunctionPath(&["jit_w_float_new"]),
];

const BOOL_FROM_TARGETS: &[CallTargetPattern] =
    &[CallTargetPattern::FunctionPath(&["w_bool_from"])];

const CALL_DESCRIPTOR_TABLE: &[CallDescriptorEntry] = &[
    // ── Pure arithmetic (elidable, cannot raise) ──
    CallDescriptorEntry {
        targets: INT_ARITH_TARGETS,
        extra_effect: ExtraEffect::ElidableCannotRaise,
        oopspec_index: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: INT_CMP_TARGETS,
        extra_effect: ExtraEffect::ElidableCannotRaise,
        oopspec_index: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: FLOAT_ARITH_TARGETS,
        extra_effect: ExtraEffect::ElidableCannotRaise,
        oopspec_index: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: FLOAT_CMP_TARGETS,
        extra_effect: ExtraEffect::ElidableCannotRaise,
        oopspec_index: OopSpecIndex::None,
    },
    // ── Elidable but may raise (ZeroDivisionError, OverflowError) ──
    CallDescriptorEntry {
        targets: INT_FLOORDIV_TARGETS,
        extra_effect: ExtraEffect::ElidableCanRaise,
        oopspec_index: OopSpecIndex::IntPyDiv,
    },
    CallDescriptorEntry {
        targets: INT_MOD_TARGETS,
        extra_effect: ExtraEffect::ElidableCanRaise,
        oopspec_index: OopSpecIndex::IntPyMod,
    },
    CallDescriptorEntry {
        targets: FLOAT_DIV_TARGETS,
        extra_effect: ExtraEffect::ElidableCanRaise,
        oopspec_index: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: INT_SHIFT_TARGETS,
        extra_effect: ExtraEffect::ElidableCanRaise,
        oopspec_index: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: INT_POW_TARGETS,
        extra_effect: ExtraEffect::ElidableCanRaise,
        oopspec_index: OopSpecIndex::None,
    },
    // ── String operations with oopspec ──
    CallDescriptorEntry {
        targets: STR_CONCAT_TARGETS,
        extra_effect: ExtraEffect::ElidableCanRaise,
        oopspec_index: OopSpecIndex::StrConcat,
    },
    CallDescriptorEntry {
        targets: STR_CMP_TARGETS,
        extra_effect: ExtraEffect::ElidableCannotRaise,
        oopspec_index: OopSpecIndex::StrCmp,
    },
    // ── List operations (may raise, side effects) ──
    CallDescriptorEntry {
        targets: LIST_GETITEM_TARGETS,
        extra_effect: ExtraEffect::CanRaise,
        oopspec_index: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: LIST_SETITEM_TARGETS,
        extra_effect: ExtraEffect::CanRaise,
        oopspec_index: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: LIST_APPEND_TARGETS,
        extra_effect: ExtraEffect::CanRaise,
        oopspec_index: OopSpecIndex::None,
    },
    // ── Tuple access (elidable for valid indices) ──
    CallDescriptorEntry {
        targets: TUPLE_GETITEM_TARGETS,
        extra_effect: ExtraEffect::ElidableCanRaise,
        oopspec_index: OopSpecIndex::None,
    },
    // ── Allocating constructors (cannot raise, but NOT elidable) ──
    // w_int_new/w_float_new allocate fresh objects — CSE would merge
    // distinct allocations, breaking Python identity (is).
    CallDescriptorEntry {
        targets: INT_NEW_TARGETS,
        extra_effect: ExtraEffect::CannotRaise,
        oopspec_index: OopSpecIndex::None,
    },
    CallDescriptorEntry {
        targets: FLOAT_NEW_TARGETS,
        extra_effect: ExtraEffect::CannotRaise,
        oopspec_index: OopSpecIndex::None,
    },
    // w_bool_from returns singletons (True/False) — safe to CSE.
    CallDescriptorEntry {
        targets: BOOL_FROM_TARGETS,
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
    use crate::model::FunctionGraph;

    #[test]
    fn guess_call_kind_function_path() {
        let mut cc = CallControl::new();
        let graph = FunctionGraph::new("opcode_load_fast");
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
        let graph = FunctionGraph::new("PyFrame::load_local_value");
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
            FunctionGraph::new("PyFrame::push_value"),
        );
        cc.register_trait_method(
            "push_value",
            "MIFrame",
            FunctionGraph::new("MIFrame::push_value"),
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

    // ── getcalldescr tests ───────────────────────────���──────────────

    use crate::model::{Block, SpaceOperation, Terminator, ValueId, ValueType};

    /// Helper: create a FunctionGraph with just a return.
    fn simple_graph(name: &str) -> FunctionGraph {
        let mut g = FunctionGraph::new(name);
        g.set_terminator(g.startblock, Terminator::Return(None));
        g
    }

    /// Helper: create a FunctionGraph with an Abort terminator.
    fn raising_graph(name: &str) -> FunctionGraph {
        let mut g = FunctionGraph::new(name);
        g.set_terminator(
            g.startblock,
            Terminator::Abort {
                reason: "error".into(),
            },
        );
        g
    }

    #[test]
    fn test_getcalldescr_cannot_raise() {
        // A simple function with no Abort → CannotRaise.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["pure_add"]);
        cc.register_function_graph(path.clone(), simple_graph("pure_add"));
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["pure_add"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &target,
            Vec::new(),
            Type::Int,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        assert_eq!(
            descriptor.effect_info.extra_effect,
            ExtraEffect::CannotRaise
        );
        assert!(!descriptor.effect_info.can_invalidate);
    }

    #[test]
    fn test_getcalldescr_can_raise() {
        // A function with Abort terminator → CanRaise.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["failing_func"]);
        cc.register_function_graph(path.clone(), raising_graph("failing_func"));
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["failing_func"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &target,
            Vec::new(),
            Type::Void,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        assert_eq!(descriptor.effect_info.extra_effect, ExtraEffect::CanRaise);
    }

    #[test]
    fn test_getcalldescr_elidable() {
        // An elidable function that cannot raise → ElidableCannotRaise.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["pure_lookup"]);
        cc.register_function_graph(path.clone(), simple_graph("pure_lookup"));
        cc.mark_elidable(path);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["pure_lookup"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &target,
            Vec::new(),
            Type::Int,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        assert_eq!(
            descriptor.effect_info.extra_effect,
            ExtraEffect::ElidableCannotRaise
        );
    }

    #[test]
    fn test_getcalldescr_elidable_can_raise() {
        // An elidable function that CAN raise → ElidableCanRaise.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["elidable_raiser"]);
        cc.register_function_graph(path.clone(), raising_graph("elidable_raiser"));
        cc.mark_elidable(path);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["elidable_raiser"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &target,
            Vec::new(),
            Type::Int,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        assert_eq!(
            descriptor.effect_info.extra_effect,
            ExtraEffect::ElidableCanRaise
        );
    }

    #[test]
    fn test_getcalldescr_loopinvariant() {
        // A loop-invariant function → LoopInvariant.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["get_config"]);
        cc.register_function_graph(path.clone(), simple_graph("get_config"));
        cc.mark_loopinvariant(path);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["get_config"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &target,
            Vec::new(),
            Type::Int,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        assert_eq!(
            descriptor.effect_info.extra_effect,
            ExtraEffect::LoopInvariant
        );
    }

    #[test]
    fn test_getcalldescr_forces_virtualizable() {
        // A function with VableForce → ForcesVirtualOrVirtualizable.
        let mut cc = CallControl::new();
        let mut graph = FunctionGraph::new("forcer");
        graph.push_op(graph.startblock, OpKind::VableForce, false);
        graph.set_terminator(graph.startblock, Terminator::Return(None));
        let path = CallPath::from_segments(["forcer"]);
        cc.register_function_graph(path, graph);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["forcer"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &target,
            Vec::new(),
            Type::Void,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        assert_eq!(
            descriptor.effect_info.extra_effect,
            ExtraEffect::ForcesVirtualOrVirtualizable
        );
    }

    #[test]
    fn test_getcalldescr_extraeffect_override() {
        // When extraeffect is provided, it overrides the analyzers.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["func"]);
        cc.register_function_graph(path, simple_graph("func"));
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["func"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &target,
            Vec::new(),
            Type::Int,
            OopSpecIndex::None,
            Some(ExtraEffect::ElidableCannotRaise),
            &mut cache,
        );
        assert_eq!(
            descriptor.effect_info.extra_effect,
            ExtraEffect::ElidableCannotRaise
        );
    }

    #[test]
    fn test_getcalldescr_transitive_can_raise() {
        // A function that calls another function that raises → CanRaise.
        let mut cc = CallControl::new();

        // callee: raises
        let callee_path = CallPath::from_segments(["callee"]);
        cc.register_function_graph(callee_path, raising_graph("callee"));

        // caller: calls callee (no Abort itself)
        let mut caller = FunctionGraph::new("caller");
        caller.push_op(
            caller.startblock,
            OpKind::Call {
                target: CallTarget::function_path(["callee"]),
                args: Vec::new(),
                result_ty: ValueType::Void,
            },
            false,
        );
        caller.set_terminator(caller.startblock, Terminator::Return(None));
        let caller_path = CallPath::from_segments(["caller"]);
        cc.register_function_graph(caller_path, caller);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["caller"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &target,
            Vec::new(),
            Type::Void,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        assert_eq!(descriptor.effect_info.extra_effect, ExtraEffect::CanRaise);
    }

    #[test]
    fn test_getcalldescr_unknown_target_can_raise() {
        // Unknown target (no graph) treated as external call.
        // RPython: RandomEffectsAnalyzer returns False for external calls
        // (only True if random_effects_on_gcobjs). RaiseAnalyzer returns
        // True (top_result) for unknown graphs → CanRaise.
        let cc = CallControl::new();
        let target = CallTarget::function_path(["unknown_extern"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &target,
            Vec::new(),
            Type::Void,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        assert_eq!(descriptor.effect_info.extra_effect, ExtraEffect::CanRaise);
        // RandomEffects is false, QuasiImmut is false → can_invalidate is false.
        assert!(!descriptor.effect_info.can_invalidate);
    }

    #[test]
    fn test_getcalldescr_readwrite_effects() {
        // A function with FieldRead/FieldWrite → bitsets populated.
        let mut cc = CallControl::new();
        let mut graph = FunctionGraph::new("accessor");
        let base = graph.alloc_value();
        graph.push_op(
            graph.startblock,
            OpKind::FieldRead {
                base,
                field: crate::model::FieldDescriptor::new("x", Some("Point".into())),
                ty: ValueType::Int,
            },
            true,
        );
        graph.push_op(
            graph.startblock,
            OpKind::FieldWrite {
                base,
                field: crate::model::FieldDescriptor::new("y", Some("Point".into())),
                value: base, // dummy
                ty: ValueType::Int,
            },
            false,
        );
        graph.set_terminator(graph.startblock, Terminator::Return(None));
        let path = CallPath::from_segments(["accessor"]);
        cc.register_function_graph(path, graph);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["accessor"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &target,
            Vec::new(),
            Type::Void,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        // Should have non-zero bitsets for field reads and writes.
        assert_ne!(descriptor.effect_info.readonly_descrs_fields, 0);
        assert_ne!(descriptor.effect_info.write_descrs_fields, 0);
    }

    #[test]
    fn test_getcalldescr_elidable_ignores_writes() {
        // Elidable function: write_descrs should be 0 even if graph has writes.
        // RPython effectinfo.py:181-186: ignore writes for elidable.
        let mut cc = CallControl::new();
        let mut graph = FunctionGraph::new("pure_writer");
        let base = graph.alloc_value();
        graph.push_op(
            graph.startblock,
            OpKind::FieldWrite {
                base,
                field: crate::model::FieldDescriptor::new("cache", Some("Obj".into())),
                value: base,
                ty: ValueType::Int,
            },
            false,
        );
        graph.set_terminator(graph.startblock, Terminator::Return(None));
        let path = CallPath::from_segments(["pure_writer"]);
        cc.register_function_graph(path.clone(), graph);
        cc.mark_elidable(path);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["pure_writer"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &target,
            Vec::new(),
            Type::Int,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        assert_eq!(
            descriptor.effect_info.extra_effect,
            ExtraEffect::ElidableCannotRaise
        );
        // Writes should be zeroed out for elidable functions.
        assert_eq!(descriptor.effect_info.write_descrs_fields, 0);
    }

    #[test]
    fn test_canraise_cached() {
        // Verify caching: second call should reuse result.
        let mut cc = CallControl::new();
        let path = CallPath::from_segments(["raiser"]);
        cc.register_function_graph(path, raising_graph("raiser"));

        let target = CallTarget::function_path(["raiser"]);
        let mut cache = AnalysisCache::default();

        let r1 = cc.canraise(&target, &mut cache);
        assert_eq!(r1, CanRaise::Yes);
        assert!(
            cache
                .can_raise
                .contains_key(&CallPath::from_segments(["raiser"]))
        );

        let r2 = cc.canraise(&target, &mut cache);
        assert_eq!(r2, CanRaise::Yes);
    }

    #[test]
    fn test_readonly_excludes_written_fields() {
        // RPython effectinfo.py:345-348: readstruct only goes to readonly
        // if there's no corresponding write ("struct") for that field.
        let mut cc = CallControl::new();
        let mut graph = FunctionGraph::new("rw_same_field");
        let base = graph.alloc_value();
        let field = crate::model::FieldDescriptor::new("x", Some("Point".into()));
        // Both read AND write the same field "x"
        graph.push_op(
            graph.startblock,
            OpKind::FieldRead {
                base,
                field: field.clone(),
                ty: ValueType::Int,
            },
            true,
        );
        graph.push_op(
            graph.startblock,
            OpKind::FieldWrite {
                base,
                field: field.clone(),
                value: base,
                ty: ValueType::Int,
            },
            false,
        );
        graph.set_terminator(graph.startblock, Terminator::Return(None));
        let path = CallPath::from_segments(["rw_same_field"]);
        cc.register_function_graph(path, graph);
        cc.find_all_graphs_for_tests();

        let target = CallTarget::function_path(["rw_same_field"]);
        let mut cache = AnalysisCache::default();
        let descriptor = cc.getcalldescr(
            &target,
            Vec::new(),
            Type::Void,
            OopSpecIndex::None,
            None,
            &mut cache,
        );
        // Write is set, but readonly should NOT have the same bit set.
        // RPython: readonly = reads & ~writes
        assert_ne!(descriptor.effect_info.write_descrs_fields, 0);
        let overlap = descriptor.effect_info.readonly_descrs_fields
            & descriptor.effect_info.write_descrs_fields;
        assert_eq!(
            overlap, 0,
            "readonly and write should not overlap for same field"
        );
    }

    #[test]
    fn test_op_can_raise_division() {
        // Division ops can raise (ZeroDivisionError).
        // RPython: LL_OPERATIONS[int_floordiv].canraise = (ZeroDivisionError,)
        let mut cc = CallControl::new();
        let mut graph = FunctionGraph::new("divider");
        let a = graph.alloc_value();
        let b = graph.alloc_value();
        graph.push_op(
            graph.startblock,
            OpKind::BinOp {
                op: "int_floordiv".to_string(),
                lhs: a,
                rhs: b,
                result_ty: ValueType::Int,
            },
            true,
        );
        graph.set_terminator(graph.startblock, Terminator::Return(None));
        let path = CallPath::from_segments(["divider"]);
        cc.register_function_graph(path, graph);

        let target = CallTarget::function_path(["divider"]);
        let mut cache = AnalysisCache::default();
        let result = cc.canraise(&target, &mut cache);
        assert_eq!(result, CanRaise::Yes);
    }

    #[test]
    fn test_canraise_memoryerror_only() {
        // Abort with "MemoryError" reason → MemoryErrorOnly.
        let mut cc = CallControl::new();
        let mut graph = FunctionGraph::new("allocator");
        graph.set_terminator(
            graph.startblock,
            Terminator::Abort {
                reason: "MemoryError: allocation failed".into(),
            },
        );
        let path = CallPath::from_segments(["allocator"]);
        cc.register_function_graph(path, graph);

        let target = CallTarget::function_path(["allocator"]);
        let mut cache = AnalysisCache::default();
        let result = cc.canraise(&target, &mut cache);
        assert_eq!(result, CanRaise::MemoryErrorOnly);
    }
}
