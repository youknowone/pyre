//! `rpython/memory/gctransform/framework.py` — which operations can collect.
//!
//! Upstream answers this by construction: `BaseFrameworkGCTransformer` has one
//! `gct_*` handler per operation that can reach the collector, and each of them
//! brackets its operation with
//!
//! ```text
//! livevars = self.push_roots(hop)
//! ...
//! self.pop_roots(hop, livevars)
//! ```
//!
//! (`framework.py:790, 853, 901, 908, 971, 1022, 1050, 1150, …` — about thirty
//! pairs).  The live set is never written by hand either:
//! `get_livevars_for_roots` (`framework.py:1501`) takes `hop.livevars_after_op()`
//! straight off the flow graph, and for a moving GC deliberately drops the
//! current operation's own arguments — *"moving GCs don't borrow, so the caller
//! does not need to keep the arguments alive"*.
//!
//! pyre has no such rewrite, so the same question has to be answered by
//! analysis.  It is answered over the *resolved* ULLBC call graph: a Call
//! terminator names its callee as `Fun::Regular(FunDeclId)`, so closing over
//! callers cannot be widened by two functions sharing a leaf name.
//!
//! # What actually collects
//!
//! Settled in source on 2026-08-19, because getting this backwards is what
//! makes an audit report windows that are not defects:
//!
//! - The **host** allocation path does **not** collect.  `try_gc_alloc`
//!   (`pyre-object/src/gc_hook.rs`) reaches the backend hook
//!   `dynasm_alloc_nursery_typed` (`majit-backend-dynasm/src/runner.rs`), whose
//!   own comment states it: *"host-side allocation must not trigger collection:
//!   the caller holds a raw `*mut u8` on the Rust stack that is NOT registered
//!   as a GC root."*  It routes to `try_alloc_nursery_no_collect_typed`, which
//!   bumps the nursery or spills to old-gen, and an object born old while a
//!   major cycle is marking is born black (`oldgen_birth_flags`).  So neither a
//!   minor nor a sweep can touch what such a call leaves behind.
//! - What collects is **application-level Python**: it can reach JIT-compiled
//!   code, which allocates inline in the nursery and runs a minor when it
//!   fills.  The explicit collection entry points and the collecting nursery
//!   hook (`alloc_nursery_collecting_typed`, reserved for the elidable bigint
//!   payload helpers) are the other two.
//!
//! So the seeds below are *dispatch* entry points, never allocators.

use std::collections::{HashMap, HashSet};

/// A resolved ULLBC call graph: which function calls which.
pub struct CallGraph {
    /// `def_id -> fully qualified name`
    pub names: HashMap<u64, String>,
    /// `def_id -> callees`, from `Fun::Regular` Call terminators only.
    pub callees: HashMap<u64, HashSet<u64>>,
    /// `def_id -> callers`, the reverse index.
    pub callers: HashMap<u64, HashSet<u64>>,
    /// Functions with at least one Call terminator this pass could not resolve
    /// to a `FunDeclId` — a call through a function pointer, a trait object or
    /// a closure.  Reported rather than dropped: an unresolved edge is exactly
    /// where "can this reach Python" stops being decidable from the graph.
    pub indirect: HashSet<u64>,
    /// What the unresolved calls actually were.  A taint that swallows a third
    /// of the graph is only actionable if you know whether it is `dyn Error`
    /// or something that can reach Python.
    pub opaque: OpaqueCensus,
}

/// The unresolved calls, bucketed by what could not be resolved.
#[derive(Default)]
pub struct OpaqueCensus {
    /// `dyn Trait` dispatch, counted per trait name.
    pub dyn_trait: HashMap<String, usize>,
    /// A call of a function-typed value whose trait could not be named — a
    /// `global_hook!` cell read, a function passed in as an argument.
    pub fn_value: usize,
    /// The unresolved calls split by which spelling charon used, so a bucket
    /// is never reasoned about as if it were all one thing.
    pub by_variant: HashMap<&'static str, usize>,
    /// Anything else the reader could not classify.
    pub unknown: usize,
}

/// Whether a `CallKind::Trait` payload's trait reference is a `dyn` one.
///
/// `[trait_ref, method_index, fun_decl_id]` — charon resolves the method and
/// puts its `FunDeclId` third, so a trait call is a real edge.  The exception
/// is a `Dyn` trait reference, where the third element names the trait's own
/// method declaration and the actual body is only known at run time.
/// The discriminant of a `CallKind::Trait` payload's trait reference.
pub fn trait_ref_kind(llbc: &majit_charon_reader::Llbc, tref: &serde_json::Value) -> String {
    let body = if let Some(id) = tref.get("Deduplicated").and_then(|x| x.as_u64()) {
        match llbc.dedup_body(id) {
            Some(b) => b,
            None => return "unresolved-dedup".into(),
        }
    } else if let Some(inline) = tref.pointer("/HashConsedValue/1") {
        inline
    } else {
        return "no-body".into();
    };
    match body.get("kind") {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Object(m)) => m.keys().next().cloned().unwrap_or_default(),
        _ => "none".into(),
    }
}

/// The first `trait_decl_ref.skip_binder.id` anywhere under `v`.
///
/// Both spellings of a virtual call carry the trait the same way — the
/// `CallKind::Trait` payload and the fat pointer's `DynTrait` type — just at
/// different depths, so one search answers for both.
fn first_trait_id(v: &serde_json::Value) -> Option<u64> {
    if let Some(map) = v.as_object() {
        if let Some(r) = map.get("trait_decl_ref")
            && let Some(id) = r.pointer("/skip_binder/id").and_then(|x| x.as_u64())
        {
            return Some(id);
        }
        for sub in map.values() {
            if let Some(id) = first_trait_id(sub) {
                return Some(id);
            }
        }
    } else if let Some(arr) = v.as_array() {
        for sub in arr {
            if let Some(id) = first_trait_id(sub) {
                return Some(id);
            }
        }
    }
    None
}

/// The dispatch entry points application-level Python is reached through.
///
/// Matched on the trailing path segments so a crate rename cannot silently
/// empty the seed set; a seed that matches nothing is reported by
/// [`CallGraph::seed_report`].
pub const PYTHON_DISPATCH_SEEDS: &[&str] = &[
    // The generic callable dispatchers.
    "call::call_callable",
    "call::call_callable_in_ctx",
    "call::call_callable_with_mode",
    "call::call_user_function",
    "call::call_user_function_plain",
    "call::call_user_function_with_args",
    "call::call_user_function_resolved_frameless",
    "call::builtin_code_call_positional",
    "call::call_builtin_code_positional",
    // The gateway's indirect call into a builtin's Rust body.
    "gateway::builtin_code_call",
    // The frame executors.  `execute_frame` and its resumed twin are inherent
    // methods, so the `impl` block sits in the path as an opaque segment and
    // the bare `pyframe::execute_frame` spelling can never match.
    "eval::eval_frame_plain",
    "eval::eval_frame_plain_with_operr",
    "eval::eval_frame_plain_with_resume",
    "eval::eval_loop",
    "pyframe::<Impl>::execute_frame",
    "pyframe::<Impl>::resume_execute_frame",
    // The JIT layer's routes back into interpretation.  An artefact extracted
    // for `pyre-jit` carries only part of `pyre-interpreter`, so most of the
    // entries above are absent from it and the closure comes out at 0% -- a
    // scan over that is vacuous rather than clean.  These are the portal and
    // blackhole entries that artefact does carry.
    "call_jit::ll_portal_runner_shim",
    "call_jit::run_frame_through_portal",
    "call_jit::bh_portal_runner",
    "call_jit::bh_call_self_recursive_portal",
    "eval::portal_runner",
    "eval::portal_runner_dispatch",
    "eval::portal_runner_result",
    "eval::eval_loop_jit",
    // The space-level helpers most builtins reach Python through.
    "baseobjspace::call_function",
    "baseobjspace::call_method",
    "baseobjspace::call_obj_args",
    "descroperation::try_call_special",
];

/// Explicit collection entry points and the collecting host allocator.
///
/// These are the same external symbols `majit-translate`'s call control marks
/// `canmallocgc` (`lib.rs`, the `mark_canmallocgc` loop): the `*_collecting_*`
/// allocators run a minor collection when the nursery cannot satisfy the
/// request, and the `collect_*` entries are requested collections outright.
/// They carry no lowered body, so nothing reaches them transitively — they have
/// to be named, or every direct caller (`pyre_object_gc_alloc_collecting_trampoline`
/// in `pyre-jit/src/eval.rs`, say) is classified non-collecting and so is
/// everything above it.
///
/// ⛔ The plain host allocator is deliberately **not** here.  `try_gc_alloc`
/// routes to `alloc_nursery_typed` and on through the backend's
/// `dynasm_alloc_nursery_typed`, which falls back to old-gen rather than
/// collect precisely because its caller holds an unregistered raw pointer; so
/// no host-side `w_*_new` is a collection point, and seeding one would report
/// every allocating body in the interpreter.  A minor runs from compiled code
/// allocating inline in the nursery, from the elidable bigint payload helpers,
/// and from a requested collection — which is why an artefact carrying only
/// interpreter bodies legitimately matches none of the allocator entries and
/// takes its whole closure from [`PYTHON_DISPATCH_SEEDS`].
pub const COLLECTING_SEEDS: &[&str] = &[
    "majit_gc::alloc_nursery_collecting_typed",
    "majit_gc::alloc_nursery_collecting_typed_rooted",
    "majit_gc::alloc_fast_nursery_collecting_typed_rooted",
    "majit_gc::standalone_alloc_nursery_collecting_typed_rooted",
    "majit_gc::standalone_alloc_fast_nursery_collecting_typed_rooted",
    "majit_gc::collect_full",
    "majit_gc::collect_step",
    "majit_gc::collect_oldgen_nonmoving",
    // The host hook the interpreter reaches them through.
    "gc_hook::try_gc_alloc_collecting_rooted",
    // A requested collection, which is the one collection point an interpreter
    // body reaches without running Python at all (`gc.collect()`).
    "gc_hook::try_gc_collect",
    "gc_hook::try_gc_collect_step",
    "gc_hook::try_gc_collect_oldgen",
];

impl CallGraph {
    /// Every function that can transitively reach one of `seeds`.
    pub fn reaching(&self, seeds: &HashSet<u64>) -> HashSet<u64> {
        let mut out: HashSet<u64> = seeds.clone();
        let mut work: Vec<u64> = seeds.iter().copied().collect();
        while let Some(cur) = work.pop() {
            let Some(ups) = self.callers.get(&cur) else {
                continue;
            };
            for &up in ups {
                if out.insert(up) {
                    work.push(up);
                }
            }
        }
        out
    }

    /// A shortest call chain from `from` down to one of `seeds`, as names.
    ///
    /// A reachability verdict is only as good as the path behind it, and a
    /// path through an error-formatting helper is a very different claim from
    /// one through a dunder invocation.  Adjudicating a finding means reading
    /// this, not trusting the bit.
    pub fn path_to(&self, from: u64, seeds: &HashSet<u64>) -> Option<Vec<String>> {
        let mut prev: HashMap<u64, u64> = HashMap::new();
        let mut seen: HashSet<u64> = HashSet::from([from]);
        let mut queue: std::collections::VecDeque<u64> = std::collections::VecDeque::from([from]);
        while let Some(cur) = queue.pop_front() {
            if seeds.contains(&cur) {
                let mut chain = vec![cur];
                let mut walk = cur;
                while let Some(&up) = prev.get(&walk) {
                    chain.push(up);
                    walk = up;
                }
                chain.reverse();
                return Some(
                    chain
                        .into_iter()
                        .map(|id| {
                            self.names
                                .get(&id)
                                .map(|n| n.rsplit("::").take(2).collect::<Vec<_>>().join("::"))
                                .unwrap_or_else(|| format!("#{id}"))
                        })
                        .collect(),
                );
            }
            for &down in self.callees.get(&cur).into_iter().flatten() {
                if seen.insert(down) {
                    prev.insert(down, cur);
                    queue.push_back(down);
                }
            }
        }
        None
    }

    /// Resolve the configured seed patterns against the real name table.
    pub fn seeds_for(&self, patterns: &[&str]) -> (HashSet<u64>, Vec<String>) {
        let mut ids = HashSet::new();
        let mut unmatched = Vec::new();
        for pat in patterns {
            let mut any = false;
            for (&id, name) in &self.names {
                // Anchored on a path separator: an unanchored `ends_with`
                // would let `fastcall::call_callable` answer for the
                // `call::call_callable` seed, and an over-matched seed widens
                // every verdict downstream of `reaching`.
                if name == pat || name.ends_with(&format!("::{pat}")) {
                    ids.insert(id);
                    any = true;
                }
            }
            if !any {
                unmatched.push((*pat).to_string());
            }
        }
        (ids, unmatched)
    }

    /// A seed pattern that matches nothing is a silently empty analysis, so
    /// name them rather than let the closure come back small and look clean.
    pub fn seed_report(&self, patterns: &[&str]) -> String {
        let (ids, unmatched) = self.seeds_for(patterns);
        let mut s = format!("{} seed function(s)", ids.len());
        if !unmatched.is_empty() {
            s.push_str(&format!("; UNMATCHED patterns: {unmatched:?}"));
        }
        s
    }
}

/// Several artefacts' call graphs, joined on fully qualified name.
///
/// A charon artefact carries only what rustc monomorphised into that crate, so
/// `pyre-jit.ullbc` holds the portal and the blackhole and almost none of the
/// interpreter they call into: its own closure comes out at 1%, and a scan over
/// it reports nothing because nothing in it reaches a seed — not because the
/// crate is clean.  Upstream never meets this: RPython analyses one translated
/// program, and `collect_analyzer` closes over the whole call graph at once.
/// So the artefacts are joined back into one graph here.
///
/// # The join key does not always identify a function
///
/// `ItemMeta::name_path` renders an inherent `impl` block as the opaque segment
/// `<Impl>`, so `PyFrame::new` and `FrameDebugData::new` are both spelled
/// `pyframe::<Impl>::new`.  Joining on that spelling merges them into one node
/// and invents an edge from every caller of one to every callee of the other —
/// which is how a `getorcreatedebug` that only allocates came out reaching
/// `call_user_function_with_args`.  Widening is not automatically safe here:
/// `framework.py` over-brackets rather than under-brackets, but a fabricated
/// path is not conservatism, it is a wrong answer with a citation.
///
/// So a name **one artefact carries more than once** is not a join key at all:
/// each occurrence keeps its own node, exactly as it had before the join.  What
/// is joined is the unambiguous majority — the free functions the JIT reaches
/// the interpreter through (`finditem_str`, `call_function_impl_result`) — and
/// [`Self::ambiguous_names`] reports what that rule held back.
pub struct Joined {
    /// The joined graph.  Its `opaque` census is left empty: an opaque *count*
    /// is per-artefact accounting, and summing two of them double-counts every
    /// body both artefacts carry.  `indirect` is joined, being a per-function
    /// fact.
    pub graph: CallGraph,
    /// `part index -> (that artefact's def id -> node id)`.
    canonical: Vec<HashMap<u64, u64>>,
    /// Unambiguous names carried by more than one artefact — what the join
    /// actually merged.
    pub joined_names: usize,
    /// Names some artefact carries more than once, kept apart.
    pub ambiguous_names: usize,
}

impl Joined {
    pub fn build(parts: &[&CallGraph]) -> Self {
        let mut ambiguous: HashSet<&str> = HashSet::new();
        for part in parts {
            let mut seen: HashSet<&str> = HashSet::new();
            for name in part.names.values() {
                if !seen.insert(name.as_str()) {
                    ambiguous.insert(name.as_str());
                }
            }
        }
        let mut shared: HashMap<&str, u64> = HashMap::new();
        let mut names: HashMap<u64, String> = HashMap::new();
        let mut canonical: Vec<HashMap<u64, u64>> = Vec::with_capacity(parts.len());
        let mut next = 0u64;
        let mut joined_names = 0usize;
        for part in parts {
            // Sorted, so the numbering does not depend on hash order and two
            // runs over the same inputs print the same ids.
            let mut sorted: Vec<(&u64, &String)> = part.names.iter().collect();
            sorted.sort_unstable_by(|a, b| a.1.cmp(b.1).then(a.0.cmp(b.0)));
            let mut mine: HashMap<u64, u64> = HashMap::new();
            for (&def_id, name) in sorted {
                let id = if ambiguous.contains(name.as_str()) {
                    let id = next;
                    next += 1;
                    names.insert(id, name.clone());
                    id
                } else if let Some(&id) = shared.get(name.as_str()) {
                    joined_names += 1;
                    id
                } else {
                    let id = next;
                    next += 1;
                    names.insert(id, name.clone());
                    shared.insert(name.as_str(), id);
                    id
                };
                mine.insert(def_id, id);
            }
            canonical.push(mine);
        }
        let mut callees: HashMap<u64, HashSet<u64>> = HashMap::new();
        let mut callers: HashMap<u64, HashSet<u64>> = HashMap::new();
        let mut indirect: HashSet<u64> = HashSet::new();
        for (part, map) in parts.iter().zip(&canonical) {
            for (caller, cs) in &part.callees {
                let Some(&from) = map.get(caller) else {
                    continue;
                };
                let entry = callees.entry(from).or_default();
                for callee in cs {
                    if let Some(&to) = map.get(callee) {
                        entry.insert(to);
                    }
                }
            }
            for id in &part.indirect {
                if let Some(&c) = map.get(id) {
                    indirect.insert(c);
                }
            }
        }
        for (&caller, cs) in &callees {
            for &c in cs {
                callers.entry(c).or_default().insert(caller);
            }
        }
        Self {
            graph: CallGraph {
                names,
                callees,
                callers,
                indirect,
                opaque: OpaqueCensus::default(),
            },
            canonical,
            joined_names,
            ambiguous_names: ambiguous.len(),
        }
    }

    /// Project a set of joined nodes back onto one artefact's def ids, so the
    /// liveness scan — which walks that artefact's bodies — can be asked the
    /// whole-program question.  `part` indexes the slice `build` was given.
    pub fn project(&self, part: usize, joined: &HashSet<u64>) -> HashSet<u64> {
        self.canonical[part]
            .iter()
            .filter(|(_, cid)| joined.contains(cid))
            .map(|(&def_id, _)| def_id)
            .collect()
    }
}

/// Where each function-pointer call gets its callee from, counted by source.
///
/// A count of unresolved calls says how much is opaque; this says *what* is
/// opaque, which is the part that decides whether modelling it is worth
/// anything.  The walk is flow-insensitive on purpose — it answers "what kind
/// of thing lands in this local", not "which value on this path".
pub fn dynamic_call_sources(llbc: &majit_charon_reader::Llbc) -> HashMap<String, usize> {
    use majit_charon_reader::ullbc::{
        CallFunc, CallKind, FunId, Operand, Place, PlaceKind, Rvalue, StmtKind, TermKind,
    };

    fn root(p: &Place) -> Option<u64> {
        match &p.kind {
            PlaceKind::Local(i) => Some(*i),
            PlaceKind::Projection(b, _) => root(b),
            _ => None,
        }
    }
    fn op_local(o: &Operand) -> Option<u64> {
        match o {
            Operand::Copy(p) | Operand::Move(p) => root(p),
            Operand::Const(_) => None,
        }
    }

    let mut out: HashMap<String, usize> = HashMap::new();
    for fd in llbc.iter_local_fns() {
        let Some(body) = fd.unstructured() else {
            continue;
        };
        let mut defs: HashMap<u64, Rvalue> = HashMap::new();
        let mut from_call: HashMap<u64, String> = HashMap::new();
        let mut projected: HashMap<u64, String> = HashMap::new();
        for bb in &body.body {
            for st in &bb.statements {
                if let Ok(StmtKind::Assign(p, rv)) = st.stmt_kind()
                    && let Some(l) = root(&p)
                {
                    if let Rvalue::Use(Operand::Copy(src) | Operand::Move(src)) = &rv
                        && let PlaceKind::Projection(_, elem) = &src.kind
                    {
                        projected.insert(l, elem.label());
                    }
                    defs.insert(l, rv);
                }
            }
            if let Ok(TermKind::Call { call, .. }) = bb.term()
                && let Some(l) = root(&call.dest)
                && let CallFunc::Regular(reg) = &call.func
                && let CallKind::Fun(FunId::Regular { id }) = &reg.kind
            {
                let name = llbc
                    .fn_by_id(*id)
                    .map(|f| f.item_meta.name_path())
                    .unwrap_or_default();
                from_call.insert(l, name.rsplit("::").take(2).collect::<Vec<_>>().join("::"));
            }
        }
        for bb in &body.body {
            let Ok(TermKind::Call { call, .. }) = bb.term() else {
                continue;
            };
            let CallFunc::Dynamic(op) = &call.func else {
                continue;
            };
            let mut cur = op_local(op);
            let mut seen: HashSet<u64> = HashSet::new();
            let label = loop {
                let Some(l) = cur else {
                    break "operand is not a local".to_string();
                };
                if !seen.insert(l) {
                    break "cycle".to_string();
                }
                if let Some(callee) = from_call.get(&l) {
                    break format!("<- {callee}");
                }
                match defs.get(&l) {
                    None => {
                        break if l >= 1 && l <= body.locals.arg_count {
                            "<- this fn's own parameter".to_string()
                        } else {
                            "<- no definition".to_string()
                        };
                    }
                    Some(Rvalue::Use(o) | Rvalue::Cast(_, o, _)) => {
                        if let Some(nl) = op_local(o) {
                            cur = Some(nl);
                            continue;
                        }
                        break match projected.get(&l) {
                            Some(f) => format!("<- field {f}"),
                            None => "<- constant".to_string(),
                        };
                    }
                    Some(Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. }) => {
                        if let PlaceKind::Projection(_, elem) = &place.kind {
                            break format!("<- &field {}", elem.label());
                        }
                        match root(place) {
                            Some(nl) if nl != l => {
                                cur = Some(nl);
                                continue;
                            }
                            _ => break "<- borrow".to_string(),
                        }
                    }
                    Some(other) => {
                        break format!("<- rvalue {}", rvalue_label(other));
                    }
                }
            };
            *out.entry(label).or_insert(0) += 1;
        }
    }
    out
}

fn rvalue_label(r: &majit_charon_reader::ullbc::Rvalue) -> &'static str {
    use majit_charon_reader::ullbc::Rvalue as R;
    match r {
        R::Use(_) => "Use",
        R::BinaryOp(..) => "BinaryOp",
        R::UnaryOp(..) => "UnaryOp",
        R::Ref { .. } => "Ref",
        R::Aggregate(..) => "Aggregate",
        R::Discriminant(_) => "Discriminant",
        R::Cast(..) => "Cast",
        R::Len(_) => "Len",
        R::Repeat(..) => "Repeat",
        R::ShallowInitBox(..) => "ShallowInitBox",
        R::RawPtr { .. } => "RawPtr",
        R::NullaryOp(..) => "NullaryOp",
        R::Unknown => "Unknown",
    }
}

/// Build the call graph of one charon artefact.
///
/// Only `Call` terminators contribute edges, and only when charon already
/// resolved the callee to a `FunDeclId`.  Everything else — `dyn` dispatch, a
/// function-pointer call, a trait call charon left unresolved — marks the
/// caller in [`CallGraph::indirect`] instead of inventing an edge.
pub fn build(llbc: &majit_charon_reader::Llbc) -> CallGraph {
    use majit_charon_reader::ullbc::{CallFunc, CallKind, FunId, TermKind};

    let mut names = HashMap::new();
    let mut callees: HashMap<u64, HashSet<u64>> = HashMap::new();
    let mut callers: HashMap<u64, HashSet<u64>> = HashMap::new();
    let mut indirect = HashSet::new();
    let mut opaque = OpaqueCensus::default();
    let mut note_opaque = |raw: Option<&serde_json::Value>, variant: &'static str| {
        *opaque.by_variant.entry(variant).or_insert(0) += 1;
        match raw.and_then(first_trait_id) {
            Some(tid) => {
                let name = llbc
                    .trait_by_id(tid)
                    .map(|t| t.item_meta.name_path())
                    .unwrap_or_else(|| format!("trait#{tid}"));
                *opaque.dyn_trait.entry(name).or_insert(0) += 1;
            }
            None => opaque.fn_value += 1,
        }
    };

    for fd in llbc.iter_local_fns() {
        let id = fd.def_id;
        names.insert(id, fd.item_meta.name_path());
        let entry = callees.entry(id).or_default();
        let Some(body) = fd.unstructured() else {
            continue;
        };
        for bb in &body.body {
            let Ok(TermKind::Call { call, .. }) = bb.term() else {
                continue;
            };
            match &call.func {
                CallFunc::Regular(reg) => match &reg.kind {
                    CallKind::Fun(FunId::Regular { id: callee }) => {
                        entry.insert(*callee);
                    }
                    CallKind::Fun(FunId::Other(v)) => {
                        indirect.insert(id);
                        note_opaque(Some(v), "Fun(Other)");
                    }
                    CallKind::Trait(v) => {
                        // `[trait_ref, method_index, fun_decl_id]`.  The third
                        // element is the **trait's method declaration**, not
                        // the impl that will run: measured over this artefact,
                        // 382 of 541 have no body at all, and every trait ref
                        // is a generic `Clause` / `ParentClause` rather than a
                        // `TraitImpl`, so the impl is not recoverable here
                        // without propagating the caller's instantiation.
                        // Turning this into an edge therefore buys false
                        // confidence — it retires the caller's undecidability
                        // by pointing it at a bodyless leaf.
                        //
                        // A default body is still real code the call may run,
                        // so it is added as an edge, which can only widen
                        // "reaches a collection".  The caller stays opaque
                        // either way, because an overriding impl is invisible
                        // from here.
                        if let Some(c) = v.get(2).and_then(|x| x.as_u64())
                            && llbc.fn_by_id(c).and_then(|f| f.unstructured()).is_some()
                        {
                            entry.insert(c);
                        }
                        indirect.insert(id);
                        note_opaque(Some(v), "Trait");
                    }
                    CallKind::Ptr(v) => {
                        indirect.insert(id);
                        note_opaque(Some(v), "Ptr");
                    }
                    CallKind::Unknown => {
                        indirect.insert(id);
                        opaque.unknown += 1;
                    }
                },
                CallFunc::Dynamic(op) => {
                    indirect.insert(id);
                    // A fat pointer's type is as often a dedup id as an inline
                    // body; resolving it is what keeps `dyn Trait` out of the
                    // function-pointer bucket.
                    let ty = match op {
                        majit_charon_reader::ullbc::Operand::Copy(p)
                        | majit_charon_reader::ullbc::Operand::Move(p) => match &p.ty {
                            majit_charon_reader::ullbc::TyRef::Inline { value: (_, v) } => Some(v),
                            majit_charon_reader::ullbc::TyRef::Dedup { id } => llbc.dedup_body(*id),
                            majit_charon_reader::ullbc::TyRef::Other(v) => Some(v),
                        },
                        majit_charon_reader::ullbc::Operand::Const(v) => Some(v),
                    };
                    note_opaque(ty, "Dynamic");
                }
                CallFunc::Unknown => {
                    indirect.insert(id);
                    opaque.unknown += 1;
                }
            }
        }
    }
    for (&caller, cs) in &callees {
        for &c in cs {
            callers.entry(c).or_default().insert(caller);
        }
    }
    drop(note_opaque);
    CallGraph {
        names,
        callees,
        callers,
        indirect,
        opaque,
    }
}
