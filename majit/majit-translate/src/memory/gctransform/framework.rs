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
    // The frame executors.
    "eval::eval_frame_plain",
    "eval::eval_frame_plain_with_operr",
    "eval::eval_frame_plain_with_resume",
    "eval::eval_loop",
    "pyframe::execute_frame",
    // The space-level helpers most builtins reach Python through.
    "baseobjspace::call_function",
    "baseobjspace::call_method",
    "baseobjspace::call_obj_args",
    "descroperation::try_call_special",
];

/// Explicit collection entry points and the one collecting host allocator.
pub const COLLECTING_SEEDS: &[&str] = &[
    "gc_hook::try_gc_alloc_collecting",
    "gc_hook::try_gc_alloc_collecting_rooted",
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
                if name.ends_with(pat) {
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
            match call.func {
                CallFunc::Regular(reg) => match reg.kind {
                    CallKind::Fun(FunId::Regular { id: callee }) => {
                        entry.insert(callee);
                    }
                    _ => {
                        indirect.insert(id);
                    }
                },
                _ => {
                    indirect.insert(id);
                }
            }
        }
    }
    for (&caller, cs) in &callees {
        for &c in cs {
            callers.entry(c).or_default().insert(caller);
        }
    }
    CallGraph {
        names,
        callees,
        callers,
        indirect,
    }
}
