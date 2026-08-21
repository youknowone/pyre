//! `rpython/memory/gctransform/framework.py get_livevars_for_roots` —
//! which GC pointers are live across an operation that can collect.
//!
//! Upstream reads the answer straight off the flow graph with
//! `hop.livevars_after_op()`, then brackets the operation with
//! `push_roots` / `pop_roots`.  pyre's interpreter is compiled by rustc, so
//! there is no flow graph to rewrite; this pass computes the same live set over
//! the ULLBC body and reports the operations where the bracket is *missing*.
//!
//! # One deliberate divergence from upstream
//!
//! `get_livevars_for_roots` drops the current operation's own arguments for a
//! moving GC — *"moving GCs don't borrow, so the caller does not need to keep
//! the arguments alive"*.  That holds upstream because the shadowstack walks
//! the caller's frame and rewrites its copy in place.  pyre's shadow stack
//! holds only what was explicitly pinned, so an argument the caller still uses
//! after the call is just as stale as any other local.  Arguments are therefore
//! kept in the live set, and reported in a separate column so the two shapes
//! stay distinguishable.

use std::collections::{HashMap, HashSet};

use majit_charon_reader::ullbc::{
    CallFunc, CallKind, FunId, Operand, Place, PlaceKind, Rvalue, StmtKind, SwitchTargets,
    TermKind, TyRef,
};

/// One call that can collect, with GC pointers live across it and no bracket.
pub struct Finding {
    pub func: u64,
    pub func_name: String,
    pub line: u64,
    pub callee_id: u64,
    pub callee_name: String,
    /// Live across the call and *not* passed to it — the shape both proven
    /// defects had.
    pub live_non_arg: Vec<String>,
    /// Live across the call because the call itself takes them.
    pub live_arg: Vec<String>,
    /// Of the live pointers, those the function later hands to a callee whose
    /// name says it addresses a `list` or a `dict` — the only two kinds whose
    /// header a minor collection relocates.  A stale pointer that is only
    /// stored or returned is a different (and rarer) problem; this column is
    /// where a stale pointer is actually dereferenced as a movable object.
    pub movable_use: Vec<String>,
}

/// What the scan could and could not account for.
///
/// A finding count means nothing without these: a body whose terminator the
/// reader could not parse loses successors, which shrinks every live set
/// computed from it, and a call sitting downstream of a `push_roots` is
/// withheld rather than cleared.
#[derive(Default)]
pub struct ScanStats {
    pub bodies_scanned: usize,
    /// Bodies holding a terminator this reader could not classify.  Their
    /// liveness is incomplete, so they are reported rather than counted clean.
    pub unparsed_terminator_bodies: usize,
    /// Bodies holding a statement this reader could not classify.  A statement
    /// that does not parse, and one that parses as `StmtKind::Unknown`, both
    /// contribute no uses, so the live set computed over such a body is a lower
    /// bound and a clean result over it is one too.
    pub unparsed_statement_bodies: usize,
    /// Collecting calls withheld because a `push_roots` dominates them.
    /// Whether that scope is still alive at the call is a drop-placement
    /// question this pass cannot answer, so they are neither reported nor
    /// silently dropped.
    pub withheld_under_a_bracket: usize,
}

/// Callee names that say the pointer is addressed as a movable object.
///
/// `list` and `dict` are the only two kinds whose header a minor collection
/// relocates, so a stale `PyObjectRef` handed to one of these is dereferenced
/// as a corpse rather than merely stored or returned.
pub const MOVABLE_GC_MARKERS: &[&str] = &[
    "w_list_",
    "w_dict_",
    "list_concat",
    "list_repeat",
    "sequence_repeat",
    "require_list",
    "require_dict",
    "dict_method_",
    "list_method_",
];

fn addresses_movable(markers: &[&str], name: &str) -> bool {
    markers.iter().any(|m| name.contains(m))
}

fn ty_id(t: &TyRef) -> Option<u64> {
    match t {
        TyRef::Dedup { id } => Some(*id),
        TyRef::Inline { value: (id, _) } => Some(*id),
        TyRef::Other(_) => None,
    }
}

/// The type ids `PyObjectRef` is spelled with in *this* artefact.
///
/// Read off signatures pyre already declares in those terms rather than
/// hard-coded, because a dedup id is artefact-local: `pin_root` takes one and
/// `shadow_stack_get` returns one.  An empty result means the analysis would
/// silently find nothing, so callers must report it.
pub fn gc_ptr_type_ids(llbc: &majit_charon_reader::Llbc) -> HashSet<u64> {
    let mut out = HashSet::new();
    for fd in llbc.iter_local_fns() {
        let name = fd.item_meta.name_path();
        if name.ends_with("gc_roots::pin_root") {
            if let Some(t) = fd.signature.inputs.first().and_then(ty_id) {
                out.insert(t);
            }
        } else if name.ends_with("gc_roots::shadow_stack_get") {
            if let Some(t) = ty_id(&fd.signature.output) {
                out.insert(t);
            }
        }
    }
    out
}

/// The type ids a `PyFrame` pointer is spelled with in *this* artefact.
///
/// The frame is the second thing a minor collection can leave a body holding a
/// corpse of, and it is the harder one: a `PyObjectRef` at least has a root
/// stack it can be pinned on, whereas the running frame is carried as a bare
/// `&mut PyFrame` that no walker reaches.  `eval::FrameAnchor` is the reload
/// point, and a body that takes one and re-reads `live()` kills its stale local
/// at the call, so [`scan`] needs no bracket set for this kind — the liveness
/// answer already distinguishes a reloaded frame from a carried one.
///
/// The interpreter spells the pointer three ways and all three go stale, so
/// each is read off a signature pyre already declares in those terms rather
/// than hard-coded: a dedup id is artefact-local.  An empty result means the
/// scan would silently find nothing, so callers must report it.
pub fn frame_ptr_type_ids(llbc: &majit_charon_reader::Llbc) -> HashSet<u64> {
    // Free functions, so the name is the whole match: an inherent method
    // carries its `impl` block as an opaque segment and cannot be named here.
    const FIRST_INPUT: &[&str] = &[
        "eval::install_current_frame",   // &mut PyFrame
        "eval::handle_exception",        // &mut PyFrame
        "executioncontext::force_frame", // *mut PyFrame
        "call::enter_recursive_frame",   // *const PyFrame
    ];
    let mut out = HashSet::new();
    for fd in llbc.iter_local_fns() {
        let name = fd.item_meta.name_path();
        if FIRST_INPUT
            .iter()
            .any(|p| name == *p || name.ends_with(&format!("::{p}")))
        {
            if let Some(t) = fd.signature.inputs.first().and_then(ty_id) {
                out.insert(t);
            }
        }
    }
    out
}

/// The local a place is rooted at, looking through every projection.
fn place_local(p: &Place) -> Option<u64> {
    match &p.kind {
        PlaceKind::Local(i) => Some(*i),
        PlaceKind::Projection(base, _) => place_local(base),
        _ => None,
    }
}

/// Whether the place names a whole local (so assigning it kills the old value).
fn bare_local(p: &Place) -> Option<u64> {
    match &p.kind {
        PlaceKind::Local(i) => Some(*i),
        _ => None,
    }
}

fn use_operand(o: &Operand, out: &mut HashSet<u64>) {
    match o {
        Operand::Copy(p) | Operand::Move(p) => {
            if let Some(l) = place_local(p) {
                out.insert(l);
            }
        }
        Operand::Const(_) => {}
    }
}

fn use_rvalue(r: &Rvalue, out: &mut HashSet<u64>) {
    match r {
        Rvalue::Use(o) | Rvalue::UnaryOp(_, o) => use_operand(o, out),
        Rvalue::BinaryOp(_, a, b) => {
            use_operand(a, out);
            use_operand(b, out);
        }
        Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. } => {
            if let Some(l) = place_local(place) {
                out.insert(l);
            }
        }
        Rvalue::Aggregate(_, ops) => {
            for o in ops {
                use_operand(o, out);
            }
        }
        Rvalue::Discriminant(p) | Rvalue::Len(p) => {
            if let Some(l) = place_local(p) {
                out.insert(l);
            }
        }
        Rvalue::Cast(_, o, _) | Rvalue::Repeat(o, _, _) | Rvalue::ShallowInitBox(o, _) => {
            use_operand(o, out)
        }
        Rvalue::NullaryOp(_, _) | Rvalue::Unknown => {}
    }
}

fn successors(t: &TermKind) -> Vec<u64> {
    match t {
        TermKind::Goto { target } => vec![*target],
        TermKind::Switch { targets, .. } => match targets {
            SwitchTargets::If(a, b) => vec![*a, *b],
            SwitchTargets::SwitchInt(_, arms, d) => {
                let mut v: Vec<u64> = arms.iter().map(|(_, b)| *b).collect();
                v.push(*d);
                v
            }
        },
        TermKind::Call {
            target, on_unwind, ..
        }
        | TermKind::Assert {
            target, on_unwind, ..
        }
        | TermKind::Drop {
            target, on_unwind, ..
        } => vec![*target, *on_unwind],
        _ => vec![],
    }
}

/// Report every call that can collect and carries an unrooted live GC pointer.
///
/// `push_roots` are the `gc_roots::push_roots` function ids.  Coverage is
/// judged **per call**, not per function: a call is withheld only when every
/// path to it runs through a `push_roots`, which is exactly "this block is
/// unreachable from entry once the bracket blocks are removed".  A function
/// that brackets one branch and leaves another bare is therefore still
/// reported on the bare branch.
///
/// `gc_tys` selects which pointer kind is being judged — [`gc_ptr_type_ids`]
/// for a managed reference, [`frame_ptr_type_ids`] for the running frame — and
/// `movable_markers` ranks the findings for that kind ([`MOVABLE_GC_MARKERS`],
/// or empty where the kind has no such call).
pub fn scan(
    llbc: &majit_charon_reader::Llbc,
    cg: &super::framework::CallGraph,
    reach: &HashSet<u64>,
    push_roots: &HashSet<u64>,
    gc_tys: &HashSet<u64>,
    movable_markers: &[&str],
) -> (Vec<Finding>, ScanStats) {
    let mut findings = Vec::new();
    let mut stats = ScanStats::default();
    for fd in llbc.iter_local_fns() {
        let id = fd.def_id;
        if !reach.contains(&id) {
            continue;
        }
        let Some(body) = fd.unstructured() else {
            continue;
        };
        // Locals whose static type is a GC pointer.  Nothing else can go stale.
        let gc_locals: HashMap<u64, String> = body
            .locals
            .locals
            .iter()
            .filter(|l| ty_id(&l.ty).is_some_and(|t| gc_tys.contains(&t)))
            .map(|l| {
                (
                    l.index,
                    l.name.clone().unwrap_or_else(|| format!("_{}", l.index)),
                )
            })
            .collect();
        if gc_locals.is_empty() {
            continue;
        }

        stats.bodies_scanned += 1;
        let n = body.body.len();
        let terms: Vec<Option<TermKind>> = body.body.iter().map(|b| b.term().ok()).collect();
        if terms
            .iter()
            .any(|t| t.is_none() || matches!(t, Some(TermKind::Unknown)))
        {
            // Successors are unknown for that block, so every live set derived
            // from it is a lower bound.  Count the body; do not pretend it is
            // clean.
            stats.unparsed_terminator_bodies += 1;
        }
        if body.body.iter().any(|blk| {
            blk.statements
                .iter()
                .any(|st| matches!(st.stmt_kind(), Err(_) | Ok(StmtKind::Unknown)))
        }) {
            // `transfer_stmt` reads no uses out of either shape, so the live
            // sets below can only be smaller than the truth.
            stats.unparsed_statement_bodies += 1;
        }

        // Blocks whose terminator opens a root scope, and the blocks that are
        // still reachable from entry without them — those are the ones no
        // bracket can dominate.
        let bracket_blocks: HashSet<usize> = (0..n)
            .filter(|&b| match &terms[b] {
                Some(TermKind::Call { call, .. }) => match &call.func {
                    CallFunc::Regular(reg) => matches!(
                        &reg.kind,
                        CallKind::Fun(FunId::Regular { id }) if push_roots.contains(id)
                    ),
                    _ => false,
                },
                _ => false,
            })
            .collect();
        let unbracketed: HashSet<usize> = if bracket_blocks.is_empty() {
            (0..n).collect()
        } else {
            let mut seen: HashSet<usize> = HashSet::new();
            let mut work = vec![0usize];
            while let Some(cur) = work.pop() {
                if bracket_blocks.contains(&cur) || !seen.insert(cur) {
                    continue;
                }
                if let Some(t) = &terms[cur] {
                    for s in successors(t) {
                        if (s as usize) < n {
                            work.push(s as usize);
                        }
                    }
                }
            }
            seen
        };
        let mut live_in: Vec<HashSet<u64>> = vec![HashSet::new(); n];
        // Backward liveness to a fixed point.  The bodies are small; a plain
        // worklist over predecessors converges in a handful of rounds.
        let mut changed = true;
        while changed {
            changed = false;
            for b in (0..n).rev() {
                let Some(t) = &terms[b] else { continue };
                let mut live: HashSet<u64> = HashSet::new();
                for s in successors(t) {
                    if let Some(sl) = live_in.get(s as usize) {
                        live.extend(sl.iter().copied());
                    }
                }
                transfer_term(t, &mut live);
                for st in body.body[b].statements.iter().rev() {
                    if let Ok(k) = st.stmt_kind() {
                        transfer_stmt(&k, &mut live);
                    }
                }
                live.retain(|l| gc_locals.contains_key(l));
                if live != live_in[b] {
                    live_in[b] = live;
                    changed = true;
                }
            }
        }

        // Locals this body hands to a `list`/`dict`-addressing callee.  Built
        // once: it does not depend on which collecting call is being judged.
        let mut movable_args: HashSet<u64> = HashSet::new();
        for other in &body.body {
            let Ok(TermKind::Call { call: c2, .. }) = other.term() else {
                continue;
            };
            let CallFunc::Regular(r2) = &c2.func else {
                continue;
            };
            let CallKind::Fun(FunId::Regular { id: cid }) = &r2.kind else {
                continue;
            };
            if !cg
                .names
                .get(cid)
                .is_some_and(|n| addresses_movable(movable_markers, n))
            {
                continue;
            }
            for a in &c2.args {
                use_operand(a, &mut movable_args);
            }
        }

        // Now re-walk, and at every collecting Call read the live-after set.
        for (b, bb) in body.body.iter().enumerate() {
            let Some(TermKind::Call {
                call,
                target,
                on_unwind,
            }) = &terms[b]
            else {
                continue;
            };
            let CallFunc::Regular(reg) = &call.func else {
                continue;
            };
            let CallKind::Fun(FunId::Regular { id: callee }) = &reg.kind else {
                continue;
            };
            if !reach.contains(callee) {
                continue;
            }
            if !unbracketed.contains(&b) {
                stats.withheld_under_a_bracket += 1;
                continue;
            }
            let mut after: HashSet<u64> = HashSet::new();
            for s in [*target, *on_unwind] {
                if let Some(sl) = live_in.get(s as usize) {
                    after.extend(sl.iter().copied());
                }
            }
            if let Some(d) = bare_local(&call.dest) {
                after.remove(&d);
            }
            after.retain(|l| gc_locals.contains_key(l));
            if after.is_empty() {
                continue;
            }
            let mut args: HashSet<u64> = HashSet::new();
            for a in &call.args {
                use_operand(a, &mut args);
            }
            let mut non_arg: Vec<String> = after
                .iter()
                .filter(|l| !args.contains(l))
                .map(|l| gc_locals[l].clone())
                .collect();
            let mut in_arg: Vec<String> = after
                .iter()
                .filter(|l| args.contains(l))
                .map(|l| gc_locals[l].clone())
                .collect();
            non_arg.sort();
            in_arg.sort();
            // Does any live pointer reach a list/dict-addressing callee in this
            // body?  Anywhere in the body, not only in the dominated
            // successors — this is a ranking signal, not a proof.
            let mut movable_use: Vec<String> = after
                .iter()
                .filter(|l| movable_args.contains(l))
                .map(|l| gc_locals[l].clone())
                .collect();
            movable_use.sort();
            findings.push(Finding {
                func: id,
                func_name: fd.item_meta.name_path(),
                line: bb
                    .statements
                    .last()
                    .map_or(fd.item_meta.span.data.beg.line, |s| s.span.data.beg.line),
                callee_id: *callee,
                callee_name: cg.names.get(callee).cloned().unwrap_or_default(),
                live_non_arg: non_arg,
                live_arg: in_arg,
                movable_use,
            });
        }
    }
    (findings, stats)
}

fn transfer_stmt(k: &StmtKind, live: &mut HashSet<u64>) {
    match k {
        StmtKind::Assign(p, r) => {
            if let Some(d) = bare_local(p) {
                live.remove(&d);
            } else if let Some(l) = place_local(p) {
                live.insert(l);
            }
            use_rvalue(r, live);
        }
        StmtKind::StorageLive(i) | StmtKind::StorageDead(i) => {
            live.remove(i);
        }
        StmtKind::Assert(a) => use_operand(&a.cond, live),
        StmtKind::PlaceMention(p) => {
            if let Some(l) = place_local(p) {
                live.insert(l);
            }
        }
        StmtKind::Unknown => {}
    }
}

fn transfer_term(t: &TermKind, live: &mut HashSet<u64>) {
    match t {
        TermKind::Call { call, .. } => {
            if let Some(d) = bare_local(&call.dest) {
                live.remove(&d);
            } else if let Some(l) = place_local(&call.dest) {
                live.insert(l);
            }
            for a in &call.args {
                use_operand(a, live);
            }
        }
        TermKind::Switch { discr, .. } => use_operand(discr, live),
        TermKind::Assert { assert, .. } => use_operand(&assert.cond, live),
        _ => {}
    }
}
