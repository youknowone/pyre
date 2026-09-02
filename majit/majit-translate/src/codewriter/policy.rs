//! JIT inlining policy.
//!
//! Translated from `rpython/jit/codewriter/policy.py`.
//!
//! `JitPolicy` decides which graphs the codewriter should "look inside"
//! and inline-trace.  RPython models this as a base class with a virtual
//! `look_inside_function`; subclasses (e.g. `StopAtXPolicy`) override that
//! one method.  In Rust we use a trait + state struct so subclasses share
//! the bookkeeping fields.
//!
//! ## Parity shape: allowlist via registration, not blacklist via module name
//!
//! Upstream `pypy/module/pypyjit/policy.py::PyPyJitPolicy.look_inside_function`
//! returns `False` for functions whose Python module matches a rejection
//! list (`pypy.interpreter.astcompiler.*`, `rpython.rlib.rlocale`, …). The
//! base `JitPolicy.look_inside_function` defaults to `True`; the PyPy
//! subclass flips that to `False` for the excluded modules. Un-excluded
//! functions inline; excluded functions stay at the residual-call
//! boundary.
//!
//! Pyre converges on the same observable behaviour through a different
//! mechanism: the `JIT_GRAPH_MODULES` whitelist in
//! `generated.rs` plus `CallControl::register_function_graph` plays the
//! role of the allowed-module set. A callee whose `CallPath` is not
//! present in `CallControl::function_graphs` is treated as residual at
//! BFS time (`call.rs::find_all_graphs_bfs` only pulls callees into
//! `candidate_graphs` when a matching graph exists).
//!
//! Consequence: pyre does **not** need a `PyPyJitPolicy`-style subclass
//! listing excluded Rust modules. The analysed-source set is the policy;
//! anything outside it is residual by construction. Per-graph hints
//! (`_elidable_function_`, `_jit_look_inside_`, `_jit_unroll_safe_`,
//! `access_directly`) still apply identically to upstream — they filter
//! allowed graphs further.
//!
//! The contract is locked down by
//! `tests/test_phase_d_find_all_graphs_parity.rs::
//! find_all_graphs_leaves_unregistered_targets_as_residual`.

use std::collections::HashSet;

use crate::front::semantic::SemanticFunction;
use crate::model::{Block, BlockId, FunctionGraph, OpKind, ValueType};

/// policy.py: shared mutable state and the default classifier.
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

    /// policy.py `dump_unsafe_loops`.
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

    /// policy.py — return `True` for every function by default.
    /// `StopAtXPolicy` overrides this.
    fn look_inside_function(&self, _func: &SemanticFunction) -> bool {
        true
    }

    /// policy.py `_reject_function(func)`.
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

    /// policy.py `look_inside_graph(graph)`.
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
        // policy.py: `_jit_unroll_safe_` opts back in despite a loop.
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
        // policy.py:71-83 `access_directly` virtualizable safety gate.
        //
        // RPython raises `ValueError("access_directly on a function which
        // we don't see ...")` when three conditions meet:
        //   - `see_function` is True (annotator determined the function is
        //     part of the JIT-visible graph set),
        //   - `res` is False (loops or unsupported types mean
        //     `look_inside_graph` decided not to trace into it),
        //   - `graph.access_directly` is True (annotator set this because
        //     an ARGUMENT carried the `access_directly` flag, see
        //     `default_specialize` in `rpython/annotator/specialize.py`).
        //
        // Turning the call into a residual call while the function
        // accesses a virtualizable would silently desynchronise the
        // virtualizable from the JIT's view; upstream therefore aborts
        // translation loudly. Pyre carries the same flag where upstream
        // does, on the graph: `FunctionGraph::access_directly`, beside
        // `hints`. It has to live there because this gate is reached from
        // the codewriter's BFS, which holds a registered `FunctionGraph`
        // and synthesizes the `SemanticFunction` around it — a field on
        // the front end's own record never travels here. The flowspace
        // pipeline writes the flag through `description.rs
        // default_specialize`; the LLBC path writes it through
        // `front::semantic::propagate_access_directly`, which walks the op
        // stream because there is no annotator to carry a flag on an
        // annotation.
        //
        // This is the first of upstream's two gates on the flag. The
        // second, `warmspot.py check_access_directly_sanity`, walks
        // everything reachable from the entry point and asserts that no
        // graph outside the JIT graph set is `access_directly`; it has no
        // port here.
        if see_function && !res && func.graph.access_directly {
            panic!(
                "access_directly on a function which we don't see: {}",
                func.name
            );
        }
        // A `false` here is the policy's refusal to let this callee become
        // a JitCode, and it reaches the caller as a bare `bool` — four
        // structurally different clauses collapsed into one answer.
        // `unsafe_loopy_graphs` already records ONE of them (and only when
        // `res` was still true at that point), so it cannot stand in for
        // the rest.  Re-derive which clause refused; guarded on the census
        // so the extra predicate calls never run on the decision path they
        // measure.
        if !res && crate::decline::enabled() {
            let reason = if !see_function {
                if jit_look_inside_hint(&func.hints) == Some(false) {
                    "dont_look_inside-hint"
                } else if self._reject_function(func) {
                    "elidable-hint"
                } else {
                    "look_inside_function-said-no"
                }
            } else if contains_loop {
                "loop-without-unroll_safe"
            } else {
                "unsupported-variable-type"
            };
            crate::decline::record(
                crate::decline::gate::LOOK_INSIDE_GRAPH,
                reason,
                format_args!("{}", func.name),
            );
        }
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

/// policy.py `class StopAtXPolicy(JitPolicy)`.
///
/// Excludes a fixed list of function names from inlining.  Used by
/// translator tests that need to JIT-compile one half of a graph and
/// keep the other half opaque.
#[derive(Debug, Clone, Default)]
pub struct StopAtXPolicy {
    pub state: JitPolicyState,
    /// policy.py: `self.funcs = funcs` — list of opaque names.
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
    /// policy.py: `return func not in self.funcs`.
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
///   - `@dont_look_inside` (`rlib/jit.py`) sets
///     `func._jit_look_inside_ = False`
///   - `@look_inside` (`rlib/jit.py`) sets
///     `func._jit_look_inside_ = True`
///
/// `front::llbc_hints::harvest_hints_from_llbcs` lowers those decorators
/// into the `"dont_look_inside"` and `"jit_look_inside"` hint strings;
/// both forms route through this helper.
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

/// `policy.py:88-108 contains_unsupported_variable_type(graph, ...)`.
///
/// ```python
/// def contains_unsupported_variable_type(graph, supports_floats,
///                                               supports_longlong,
///                                               supports_singlefloats):
///     getkind = history.getkind
///     try:
///         for block in graph.iterblocks():
///             for v in block.inputargs:
///                 getkind(v.concretetype, ...)
///             for op in block.operations:
///                 for v in op.args:
///                     getkind(v.concretetype, ...)
///                 v = op.result
///                 getkind(v.concretetype, ...)
///     except NotImplementedError as e:
///         log.WARNING('%s, ignoring graph' % (e,))
///         log.WARNING('  %s' % (graph,))
///         return True
///     return False
/// ```
///
/// Upstream reaches every value's type through `v.concretetype`, so it
/// can call `getkind` on `block.inputargs`, `op.args` and `op.result`
/// alike.  Pyre's per-`Variable` type is
/// [`crate::codewriter::type_state::ConcreteType`], a four-way
/// `Signed / GcRef / Float / Void` projection with no width axis: a
/// 128-bit value is indistinguishable from a word-sized one there, and
/// walking `inputargs` would answer `Signed` for both.  The width
/// survives only on the [`ValueType`] an [`OpKind`] declares, which is
/// the same field the two `value_type_to_kind` copies
/// (`codewriter/jtransform.rs`, `codewriter/assembler.rs`) later read to
/// form an opname, and which the sibling `value_type_to_ir_type` /
/// `constvalue_kind` / array-descr arms panic on in the same shape.  So
/// this walks that field set — see [`collect_declared_value_types`] —
/// over the same startblock-reachable block closure
/// `graph.iterblocks()` yields.
///
/// `supports_singlefloats` reaches [`value_type_has_kind`], which is the
/// flag's whole purpose upstream.  `supports_floats` and
/// `supports_longlong` stay unconsulted: neither family they gate is
/// refusable in pyre's `ValueType` domain.  See [`value_type_has_kind`].
///
/// `look_inside_graph` turns a `true` here into a refusal, which makes
/// the call residual; the census records it under the
/// `"unsupported-variable-type"` reason, standing in for upstream's
/// `log.WARNING('%s, ignoring graph')`.
pub fn contains_unsupported_variable_type(
    graph: &FunctionGraph,
    _supports_floats: bool,
    _supports_longlong: bool,
    supports_singlefloats: bool,
) -> bool {
    // `iterblocks()` parity (`rpython/flowspace/model.py:66`): the
    // startblock-reachable closure over `Block.exits`, id-keyed because
    // block ids need not be index-aligned with `blocks` storage order.
    let by_id: std::collections::HashMap<BlockId, &Block> =
        graph.blocks.iter().map(|b| (b.id, b)).collect();
    let mut block_seen: HashSet<BlockId> = HashSet::new();
    let mut stack = vec![graph.startblock];
    let mut declared: Vec<&ValueType> = Vec::new();
    while let Some(bid) = stack.pop() {
        if !block_seen.insert(bid) {
            continue;
        }
        let Some(block) = by_id.get(&bid) else {
            continue;
        };
        for op in &block.operations {
            declared.clear();
            collect_declared_value_types(&op.kind, &mut declared);
            if declared
                .iter()
                .any(|ty| !value_type_has_kind(ty, supports_singlefloats))
            {
                return true;
            }
        }
        stack.extend(block.exits.iter().rev().map(|e| e.target));
    }
    false
}

/// Whether `history.py:56-69 getkind(TYPE, ...)` has a register kind for
/// `ty`, or raises `NotImplementedError` on it.
///
/// `getkind` refuses three families.
///
/// `history.py:58-61` refuses `SingleFloat` unless the CPU supports
/// single floats, which is what `supports_singlefloats` carries.  Pyre's
/// effective value is `false`: `warmspot.py:250`
/// (`policy.set_supports_singlefloats(cpu.supports_singlefloats)`) has no
/// port, so the flag keeps the base backend's answer
/// (`backend/model.py:20`).
///
/// It must stay false for a reason upstream does not have to state.
/// Upstream's "singlefloats are stored in an int" holds because RPython
/// gives `SingleFloat` no arithmetic at all — `rffi` converts to `Float`,
/// computes, and converts back, so the rtyper never emits a float-kind
/// operation over a SingleFloat operand.  Rust source does contain native
/// `f32` arithmetic, and pyre has no `cast_singlefloat_to_float` to
/// bracket it with, so an accepted `f32` graph lowers that arithmetic
/// over the float's bit pattern in the integer bank.  Flipping this flag
/// is sound only once those casts exist and `f32` arithmetic lowers
/// through them.
///
/// `Float` is refused by the same line when `supports_floats` is false,
/// but pyre's kind projection does not model a float-less CPU: both
/// `value_type_to_kind` copies map `Float` to `'f'` unconditionally, so
/// refusing a graph for a float here would refuse one the codewriter goes
/// on to lower.
///
/// The third is `history.py:60-63`, `"type %s is too large"`, for a
/// primitive wider than `Signed`.  [`ValueType::Int128`] /
/// [`ValueType::UInt128`] (RPython `SignedLongLongLong` /
/// `UnsignedLongLongLong`) are 16 bytes, twice a word; upstream's
/// `supports_longlong` arm asserts a width of exactly 8 before handing
/// the value the `'float'` slot, so no flag setting rescues a 16-byte
/// type from the raise.  That arm is the one every downstream kind
/// projection panics on rather than declines — both `value_type_to_kind`
/// copies, `jtransform`'s `value_type_to_ir_type`, `flatten`'s
/// `constvalue_kind`, and `call`'s array-descr item projection — and the
/// one this refuses.
fn value_type_has_kind(ty: &ValueType, supports_singlefloats: bool) -> bool {
    match ty {
        // `raise NotImplementedError("type %s is too large" % TYPE)`
        ValueType::Int128 | ValueType::UInt128 => false,
        // `if TYPE is lltype.SingleFloat and supports_singlefloats`,
        // falling through to `raise NotImplementedError("type %s not
        // supported" % TYPE)` when it does not.
        ValueType::SingleFloat => supports_singlefloats,
        ValueType::Int
        | ValueType::Unsigned
        | ValueType::Bool
        | ValueType::State
        | ValueType::Ref(_)
        | ValueType::Str
        | ValueType::StringBuilder
        | ValueType::Unknown
        | ValueType::Float
        | ValueType::Void => true,
    }
}

/// Append the [`ValueType`]s one operation declares to `out`.
///
/// Two surfaces declare one.  Most variants carry it in a `ty` /
/// `item_ty` / `result_ty` field, which is what the `value_type_to_kind`
/// copies read.  The `ConstInt128` / `ConstUInt128` variants carry no
/// such field — the width is in the variant name and the payload is a
/// Rust literal — but they are the op form of upstream's
/// `Constant(value, SignedLongLongLong)` operand, which `policy.py:96-98`
/// reaches through `op.args` and refuses like any other value.  They
/// report the type they materialise, so a graph holding a 128-bit
/// literal is refused here instead of panicking later in
/// `assembler.rs`'s opname formation.
///
/// The match is exhaustive and deliberately carries no wildcard arm: an
/// `OpKind` variant added with a `ValueType` field has to be classified
/// here, and until it is the crate does not compile.  A wildcard would
/// let a new carrier of a 128-bit type pass the policy gate and reach
/// `value_type_to_kind`, which panics rather than declining.
pub fn collect_declared_value_types<'a>(kind: &'a OpKind, out: &mut Vec<&'a ValueType>) {
    // The 128-bit constant variants have no `ValueType` field to borrow
    // from, and `ValueType` owns a `String` so it cannot be promoted to a
    // `&'static` from a `const`.  Name the two shapes once instead.
    static INT128: ValueType = ValueType::Int128;
    static UINT128: ValueType = ValueType::UInt128;
    static SINGLEFLOAT: ValueType = ValueType::SingleFloat;

    match kind {
        // The type of the value the op reads or writes.
        OpKind::Input { ty, .. }
        | OpKind::ConstSymbolic { ty, .. }
        | OpKind::FieldRead { ty, .. }
        | OpKind::FieldWrite { ty, .. }
        | OpKind::VableFieldRead { ty, .. }
        | OpKind::VableFieldWrite { ty, .. }
        | OpKind::LoadStatic { ty, .. } => out.push(ty),

        // The element type of the array or interior field addressed.
        OpKind::NewArrayClear { item_ty, .. }
        | OpKind::NewListClear { item_ty, .. }
        | OpKind::ArrayRead { item_ty, .. }
        | OpKind::ArrayWrite { item_ty, .. }
        | OpKind::RawLoad { item_ty, .. }
        | OpKind::RawStore { item_ty, .. }
        | OpKind::InteriorFieldRead { item_ty, .. }
        | OpKind::InteriorFieldWrite { item_ty, .. }
        | OpKind::VableArrayRead { item_ty, .. }
        | OpKind::VableArrayWrite { item_ty, .. }
        | OpKind::VableArrayLen { item_ty, .. } => out.push(item_ty),

        // The declared type of the op's result.
        OpKind::Call { result_ty, .. }
        | OpKind::IndirectCall { result_ty, .. }
        | OpKind::BinOp { result_ty, .. }
        | OpKind::UnaryOp { result_ty, .. }
        | OpKind::IsInstance { result_ty, .. } => out.push(result_ty),

        // `policy.py:96-98`'s `for v in op.args: getkind(v.concretetype)`
        // over a `Constant` of the 16-byte primitive.
        OpKind::ConstInt128(_) => out.push(&INT128),
        OpKind::ConstUInt128(_) => out.push(&UINT128),
        // Likewise `Constant(value, SingleFloat)`.  This is the only way
        // the walk sees an `f32` literal: the variant declares no
        // `ValueType` field, and the literal's own type channel is not
        // one this walk reads.
        OpKind::ConstSingleFloat(_) => out.push(&SINGLEFLOAT),

        // No value type declared.  The remaining constant variants carry
        // a Rust literal whose kind is fixed by the variant name, and the
        // call variants downstream of `jtransform` carry a `result_kind`
        // char that `value_type_to_kind` already produced.
        OpKind::ConstInt(_)
        | OpKind::ConstUInt(_)
        | OpKind::ConstBool(_)
        | OpKind::ConstFloat(_)
        | OpKind::ConstStr(_)
        | OpKind::ConstRef(_)
        | OpKind::ConstRefNull
        | OpKind::ConstNone
        | OpKind::ConstRefAddr(_)
        | OpKind::New { .. }
        | OpKind::NewWithVtable { .. }
        | OpKind::ArrayLen { .. }
        | OpKind::GuardTrue { .. }
        | OpKind::GuardFalse { .. }
        | OpKind::GuardValue { .. }
        | OpKind::VtableMethodPtr { .. }
        | OpKind::VableForce { .. }
        | OpKind::Hint { .. }
        | OpKind::CallElidable { .. }
        | OpKind::CallResidual { .. }
        | OpKind::CallMayForce { .. }
        | OpKind::InlineCall { .. }
        | OpKind::RecursiveCall { .. }
        | OpKind::JitDebug { .. }
        | OpKind::AssertGreen { .. }
        | OpKind::CurrentTraceLength
        | OpKind::IsConstant { .. }
        | OpKind::IsVirtual { .. }
        | OpKind::ConditionalCall { .. }
        | OpKind::ConditionalCallValue { .. }
        | OpKind::RecordKnownResult { .. }
        | OpKind::RecordQuasiImmutField { .. }
        | OpKind::Live
        | OpKind::JitMergePoint { .. }
        | OpKind::LoopHeader { .. }
        | OpKind::Abort { .. }
        | OpKind::NewTuple { .. }
        | OpKind::NewList { .. }
        | OpKind::GetSlice { .. }
        | OpKind::LoweredBlackholeOp { .. } => {}
    }
}

/// `rpython.translator.backendopt.support.find_backedges(graph)`.
///
/// Standard DFS classification: edges from a block back to an ancestor
/// in the current DFS stack are back edges.  Returns the list of back
/// edges as `(from_block, to_block)` pairs.
fn find_backedges(graph: &FunctionGraph) -> Vec<(usize, usize)> {
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
    let block = match graph.blocks.get(block_idx) {
        Some(b) => b,
        None => return Vec::new(),
    };
    // RPython `flowspace/model.py` FunctionGraph.iterblocks derives
    // the successor set from `Block.exits` only; final blocks
    // (`exits == ()`) have no outgoing targets.
    block.exits.iter().map(|link| link.target.0).collect()
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
            trait_impl_id: None,
            hints: hints.into_iter().map(|h| h.to_string()).collect(),
            module_path: String::new(),
            trait_root: None,
            trait_qualified: None,
            returns_objectptr: false,
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

    /// `policy.py:71-83`: a graph the codewriter refuses to look inside must
    /// not be `access_directly`. Pins the carrier as well as the gate — the
    /// flag has to reach here on the `FunctionGraph`, because the production
    /// caller (`call.rs`) synthesizes the `SemanticFunction` around a
    /// registered graph and can put nothing else on it.
    #[test]
    #[should_panic(expected = "access_directly on a function which we don't see")]
    fn access_directly_on_a_loopy_graph_aborts() {
        let mut policy = DefaultJitPolicy::new();
        let mut g = FunctionGraph::new("loopy");
        let entry = g.startblock;
        g.set_goto(entry, entry, Vec::new());
        g.access_directly = true;
        policy.look_inside_graph(&SemanticFunction {
            name: "loopy".into(),
            graph: g,
            return_type: None,
            self_ty_root: None,
            trait_impl_id: None,
            hints: vec![],
            module_path: String::new(),
            trait_root: None,
            trait_qualified: None,
            returns_objectptr: false,
        });
    }

    /// The same graph without the flag is an ordinary decline, not an abort.
    #[test]
    fn a_loopy_graph_without_the_flag_only_declines() {
        let mut policy = DefaultJitPolicy::new();
        let mut g = FunctionGraph::new("loopy");
        let entry = g.startblock;
        g.set_goto(entry, entry, Vec::new());
        assert!(!policy.look_inside_graph(&SemanticFunction {
            name: "loopy".into(),
            graph: g,
            return_type: None,
            self_ty_root: None,
            trait_impl_id: None,
            hints: vec![],
            module_path: String::new(),
            trait_root: None,
            trait_qualified: None,
            returns_objectptr: false,
        }));
    }

    /// `policy.py:88-108`: a graph holding a value `history.getkind`
    /// refuses is refused here, not carried to the codewriter — where
    /// `value_type_to_kind` panics rather than declining.
    #[test]
    fn a_128_bit_result_type_is_unsupported() {
        let mut g = FunctionGraph::new("wide");
        let entry = g.startblock;
        g.push_op_var(
            entry,
            OpKind::Input {
                name: "x".into(),
                ty: ValueType::Int128,
                class_root: None,
            },
            true,
        );
        assert!(contains_unsupported_variable_type(&g, true, true, true));
    }

    /// The 128-bit constants declare their width through the variant
    /// name rather than a `ValueType` field, and are refused all the
    /// same — `policy.py:96-98` reaches upstream's equivalent
    /// `Constant(value, SignedLongLongLong)` through `op.args`.
    #[test]
    fn a_128_bit_constant_is_unsupported() {
        let mut g = FunctionGraph::new("wide_const");
        let entry = g.startblock;
        g.push_op_var(entry, OpKind::ConstUInt128(1), true);
        assert!(contains_unsupported_variable_type(&g, true, true, true));
    }

    /// `history.py:58-61`: a singlefloat is refused unless the CPU
    /// supports one, and pyre's effective answer is the base backend's
    /// `False` (`backend/model.py:20`) because `warmspot.py:250` has no
    /// port. The flag is honoured rather than hardcoded, so both answers
    /// are asserted here — the `true` leg is what upstream's x86 gets
    /// (`backend/x86/runner.py:21`), and reaching it in pyre would need
    /// the singlefloat casts first.
    #[test]
    fn a_singlefloat_is_unsupported_unless_the_cpu_supports_one() {
        let mut g = FunctionGraph::new("narrow_float");
        let entry = g.startblock;
        g.push_op_var(
            entry,
            OpKind::Input {
                name: "x".into(),
                ty: ValueType::SingleFloat,
                class_root: None,
            },
            true,
        );
        assert!(contains_unsupported_variable_type(&g, true, true, false));
        assert!(!contains_unsupported_variable_type(&g, true, true, true));
    }

    /// The `f32` literal is the carrier with no `ValueType` field
    /// anywhere on its path — it declares none, and nothing else in a
    /// literal-only graph declares one either. `policy.py:96-98` reaches
    /// upstream's `Constant(value, SingleFloat)` through `op.args`, and
    /// this is how the walk reaches its op form.
    #[test]
    fn a_singlefloat_constant_is_unsupported() {
        let mut g = FunctionGraph::new("narrow_float_const");
        let entry = g.startblock;
        g.push_op_var(entry, OpKind::ConstSingleFloat(1.0f32.to_bits()), true);
        assert!(contains_unsupported_variable_type(&g, true, true, false));
    }

    /// Word-sized and float values keep their kinds, so an ordinary
    /// graph is not refused.  The `supports_*` flags do not enter into
    /// it: `false` for all three answers the same as `true`, because
    /// pyre's kind projection models no float-less CPU.
    #[test]
    fn ordinary_value_types_are_supported() {
        let mut g = FunctionGraph::new("narrow");
        let entry = g.startblock;
        for ty in [
            ValueType::Int,
            ValueType::Unsigned,
            ValueType::Bool,
            ValueType::Float,
            ValueType::Void,
            ValueType::Ref(None),
        ] {
            g.push_op_var(
                entry,
                OpKind::Input {
                    name: "x".into(),
                    ty,
                    class_root: None,
                },
                true,
            );
        }
        assert!(!contains_unsupported_variable_type(&g, true, true, true));
        assert!(!contains_unsupported_variable_type(&g, false, false, false));
    }

    /// The gate that consumes it: a graph the codewriter cannot give a
    /// register kind is declined rather than reaching
    /// `value_type_to_kind`.
    #[test]
    fn look_inside_graph_declines_a_128_bit_graph() {
        let mut policy = DefaultJitPolicy::new();
        let mut g = FunctionGraph::new("wide");
        let entry = g.startblock;
        g.push_op_var(
            entry,
            OpKind::Input {
                name: "x".into(),
                ty: ValueType::UInt128,
                class_root: None,
            },
            true,
        );
        assert!(!policy.look_inside_graph(&SemanticFunction {
            name: "wide".into(),
            graph: g,
            return_type: None,
            self_ty_root: None,
            trait_impl_id: None,
            hints: vec![],
            module_path: String::new(),
            trait_root: None,
            trait_qualified: None,
            returns_objectptr: false,
        }));
    }

    #[test]
    fn unroll_safe_disables_loop_rejection() {
        let mut policy = DefaultJitPolicy::new();
        // Build a graph with a self-loop on block 0.
        let mut g = FunctionGraph::new("loopy");
        let entry = g.startblock;
        g.set_goto(entry, entry, Vec::new());
        let loopy = SemanticFunction {
            name: "loopy".into(),
            graph: g.clone(),
            return_type: None,
            self_ty_root: None,
            trait_impl_id: None,
            hints: vec![],
            module_path: String::new(),
            trait_root: None,
            trait_qualified: None,
            returns_objectptr: false,
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
            trait_impl_id: None,
            hints: vec!["unroll_safe".into()],
            module_path: String::new(),
            trait_root: None,
            trait_qualified: None,
            returns_objectptr: false,
        };
        assert!(policy.look_inside_graph(&unroll_safe));
    }

    #[test]
    fn dont_look_inside_hint_overrides_default_to_false() {
        // test_policy.py `test_dont_look_inside`.
        let mut policy = DefaultJitPolicy::new();
        let f = make_func("h", vec!["dont_look_inside"]);
        assert!(!policy.look_inside_graph(&f));
    }

    #[test]
    fn jit_look_inside_hint_overrides_subclass_to_true() {
        // test_policy.py `test_look_inside`.
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
        g.set_goto(entry, entry, Vec::new());
        let edges = find_backedges(&g);
        assert_eq!(edges, vec![(entry.0, entry.0)]);
    }
}
