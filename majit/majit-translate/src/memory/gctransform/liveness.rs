//! `rpython/memory/gctransform/framework.py get_livevars_for_roots` —
//! which GC pointers are live across an operation that can collect.
//!
//! Upstream reads the answer straight off the flow graph with
//! `hop.livevars_after_op()`, then brackets the operation with
//! `push_roots` / `pop_roots`. Until the framework marker-insertion handlers
//! are ported, this audit computes the same live set over the ULLBC body and
//! reports the operations where the existing source bracket is *missing*.
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
    /// Source path of [`Finding::line`], read out of the same span, so the
    /// two cannot name different files.  Empty when the artefact carries no
    /// file table.
    pub file: String,
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
    /// Of [`Self::withheld_under_a_bracket`], those whose bracket pins every
    /// GC pointer live across the call.  A call with nothing live counts here:
    /// any bracket covers an empty set.
    pub withheld_bracket_covers: usize,
    /// Those where a pointer live across the call is not in the bracket's
    /// pinned set.  The bracket exists, so the finding scan withholds the
    /// call -- and the root it would have needed is not in it.  Listed in
    /// [`Self::short_brackets`].
    pub withheld_bracket_short: usize,
    /// Those whose pinned set could not be read: a body holding a statement or
    /// terminator this reader could not parse, an `enter_roots_frame` whose
    /// slots are filled by later stores rather than named in an argument, or a
    /// pin whose argument does not trace back to a set.  Neither graded nor
    /// claimed clean -- a set read short would accuse correct code.
    pub withheld_contents_opaque: usize,
    /// Of [`Self::withheld_bracket_short`], those missing a root the body
    /// produced itself, which no caller's bracket can be covering.
    pub withheld_bracket_short_body_local: usize,
    /// Of [`Self::withheld_bracket_short`], those missing a root this body goes
    /// on to address as a `list` or `dict`.  A caller's pin does not answer for
    /// these, so they do not depend on the intra-procedural limit.
    pub withheld_bracket_short_movable: usize,
    /// The [`ScanStats::withheld_bracket_short`] calls, named.
    pub short_brackets: Vec<ShortBracket>,
}

/// A bracketed call whose bracket does not pin everything live across it.
///
/// The complement of a [`Finding`]: there the bracket is absent, here it is
/// present and short.  `postprocess_double_check` asserts the same property
/// upstream after `shadowcolor.py` has run; this reads it off the shipped
/// hand-written brackets, which no pass has ever checked.
pub struct ShortBracket {
    pub func_name: String,
    pub file: String,
    pub line: u64,
    pub callee_name: String,
    /// Live across the call and not pinned by the bracket that dominates it.
    pub missing: Vec<String>,
    /// Of [`Self::missing`], those the body produced itself rather than took as
    /// a parameter.  The scan is intra-procedural, so a missing *parameter* may
    /// be pinned by the caller -- `build_class_inner` says in prose that its
    /// only caller keeps its arguments pinned for the whole call, and no read
    /// of this body can see that.  Nothing outside the body can have pinned a
    /// local it produced, so these are the ones a caller cannot account for.
    pub missing_local: Vec<String>,
    /// Of [`Self::missing`], those this body later hands to a callee whose name
    /// says it addresses a `list` or a `dict` -- the two kinds whose header a
    /// minor collection relocates.  A caller's pin cannot rescue one of these:
    /// the collection rewrites the caller's slot, not this body's copy, so the
    /// word in hand is a corpse either way.  The bracket-contents twin of the
    /// `tier 1.5` column, which the gate holds at zero.
    pub missing_movable: Vec<String>,
    /// What the bracket does pin here, so the two columns can be read
    /// together: an empty one is a scope opened over no roots at all.
    pub pinned: Vec<String>,
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

/// How a local that a pin call was handed came to hold its value.
///
/// Only the shapes `pin_roots(&[a, b])` lowers to are followed; anything else
/// leaves the pinned set unread rather than guessed at.
enum PinSrc {
    /// `_t = [a, b, c]` -- the pinned set itself.
    Aggregate(Vec<u64>),
    /// `_t = &_u`, `_t = _u as _`, `_t = _u` -- the same value under a second
    /// name, so keep walking.
    Alias(u64),
}

fn pin_src(r: &Rvalue) -> Option<PinSrc> {
    let one = |o: &Operand| {
        let mut s = HashSet::new();
        use_operand(o, &mut s);
        s.into_iter().next()
    };
    match r {
        Rvalue::Aggregate(_, ops) => {
            let mut v = Vec::new();
            for o in ops {
                v.extend(one(o));
            }
            Some(PinSrc::Aggregate(v))
        }
        Rvalue::Ref { place, .. } | Rvalue::RawPtr { place, .. } => {
            place_local(place).map(PinSrc::Alias)
        }
        Rvalue::Use(o) | Rvalue::Cast(_, o, _) => one(o).map(PinSrc::Alias),
        _ => None,
    }
}

/// Every local the value reaching a pin argument is spelled by.
///
/// The whole alias chain is kept, not only its end: a pin publishes a *value*,
/// so every local holding that value at the pin is covered by it, and the
/// liveness answer names whichever one the body went on to use.
fn chase_pinned(l: u64, defs: &HashMap<u64, PinSrc>, out: &mut HashSet<u64>, depth: u32) {
    if !out.insert(l) || depth > 8 {
        return;
    }
    match defs.get(&l) {
        Some(PinSrc::Aggregate(v)) => {
            for &m in v {
                chase_pinned(m, defs, out, depth + 1);
            }
        }
        Some(PinSrc::Alias(next)) => chase_pinned(*next, defs, out, depth + 1),
        None => {}
    }
}

/// The functions that write a livevar onto the root stack.
///
/// `push_roots` opens the scope and takes no arguments; the set is named
/// separately, so the contents are read off these and not off the opener.
/// `publish_roots` is `pin_roots` without the normalize half, and pins just
/// the same.
fn is_pin_fn(name: &str) -> bool {
    name.ends_with("::pin_root") || name.ends_with("::pin_roots") || name.ends_with("::publish_roots")
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
        let unparsed_terms = terms
            .iter()
            .any(|t| t.is_none() || matches!(t, Some(TermKind::Unknown)));
        if unparsed_terms {
            // Successors are unknown for that block, so every live set derived
            // from it is a lower bound.  Count the body; do not pretend it is
            // clean.
            stats.unparsed_terminator_bodies += 1;
        }
        let unparsed_stmts = body.body.iter().any(|blk| {
            blk.statements
                .iter()
                .any(|st| matches!(st.stmt_kind(), Err(_) | Ok(StmtKind::Unknown)))
        });
        if unparsed_stmts {
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
        // Reachable from entry, brackets and all.  A block no path reaches
        // has no meet to take, and a must-analysis would hand it the universe
        // and read every root as pinned; it is also not a block whose bracket
        // anyone runs.
        let reachable: HashSet<usize> = {
            let mut seen: HashSet<usize> = HashSet::new();
            let mut work = vec![0usize];
            while let Some(cur) = work.pop() {
                if !seen.insert(cur) {
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

        // What each pin call names.  `bracket_blocks` says a scope is open; it
        // does not say what is in it, and a scope that pins the wrong set
        // silences this scan without protecting anything.
        //
        // A local assigned twice is not a chain worth following, so the map is
        // built over single-assignment locals only -- which is every temporary
        // `pin_roots(&[..])` lowers through.
        let mut defs: HashMap<u64, PinSrc> = HashMap::new();
        let mut defined: HashSet<u64> = HashSet::new();
        for (b, blk) in body.body.iter().enumerate() {
            for st in &blk.statements {
                let Ok(StmtKind::Assign(place, rv)) = st.stmt_kind() else {
                    continue;
                };
                let Some(d) = bare_local(&place) else { continue };
                if !defined.insert(d) {
                    defs.remove(&d);
                    continue;
                }
                if let Some(src) = pin_src(&rv) {
                    defs.insert(d, src);
                }
            }
            if let Some(TermKind::Call { call, .. }) = &terms[b] {
                if let Some(d) = bare_local(&call.dest) {
                    if !defined.insert(d) {
                        defs.remove(&d);
                    }
                }
            }
        }

        // A body whose pinned set cannot be read is not a body with an empty
        // one: grading it would turn "not understood" into "root missing".
        let mut opaque_contents = unparsed_terms || unparsed_stmts;
        let mut saw_pin_call = false;
        let mut term_pins: Vec<HashSet<u64>> = vec![HashSet::new(); n];
        for b in 0..n {
            let Some(TermKind::Call { call, .. }) = &terms[b] else {
                continue;
            };
            let CallFunc::Regular(reg) = &call.func else {
                continue;
            };
            let CallKind::Fun(FunId::Regular { id }) = &reg.kind else {
                continue;
            };
            let Some(name) = cg.names.get(id) else { continue };
            if name.ends_with("gc_roots::enter_roots_frame") {
                // The coloured form reserves slots and stores the roots into
                // them afterwards, so there is no argument to read a set off.
                opaque_contents = true;
                continue;
            }
            // Reading a slot back yields the word the slot holds now, which
            // is rooted by whatever pinned it; the index it takes is not a
            // root, so only the result is read here.
            let reads_a_slot_back = name.ends_with("gc_roots::shadow_stack_get");
            if !is_pin_fn(name) && !reads_a_slot_back {
                continue;
            }
            saw_pin_call = true;
            let mut pinned: HashSet<u64> = HashSet::new();
            if !reads_a_slot_back {
                for a in &call.args {
                    let mut seed: HashSet<u64> = HashSet::new();
                    use_operand(a, &mut seed);
                    for l in seed {
                        chase_pinned(l, &defs, &mut pinned, 0);
                    }
                }
            }
            // What a pin hands back is the word now in the slot, so the local
            // it binds is rooted exactly as the argument was.  `pin_root` is
            // `#[must_use]` for that reason -- a collection may have forwarded
            // the value, and the returned word is the one the caller must go
            // on to use.  `let obj = pin_root(obj)` rebinds, so the local the
            // body uses afterwards is a *different* one from the local passed
            // in, and pinning only the argument would read the rebound name as
            // unrooted.
            if let Some(d) = bare_local(&call.dest) {
                pinned.insert(d);
            }
            pinned.retain(|l| gc_locals.contains_key(l));
            if pinned.is_empty() {
                // A pin that named nothing we could resolve is a pin we do not
                // understand, not one that pinned nothing.
                opaque_contents = true;
            }
            term_pins[b] = pinned;
        }
        if !bracket_blocks.is_empty() && !saw_pin_call {
            // A scope is open and nothing in this body names what went into
            // it: the pins run behind a helper that holds the scope itself,
            // as `RootedItems` does.  An unread set, not an empty one.
            opaque_contents = true;
        }

        let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
        for b in 0..n {
            if !reachable.contains(&b) {
                continue;
            }
            if let Some(t) = &terms[b] {
                for s in successors(t) {
                    if (s as usize) < n {
                        preds[s as usize].push(b);
                    }
                }
            }
        }
        // Locals a block reassigns before its terminator runs: the pin still
        // holds the word the local used to carry, which is not the one the
        // call is about to use.
        let mut stmt_kills: Vec<HashSet<u64>> = vec![HashSet::new(); n];
        for (b, blk) in body.body.iter().enumerate() {
            for st in &blk.statements {
                match st.stmt_kind() {
                    Ok(StmtKind::Assign(place, _)) => {
                        if let Some(d) = bare_local(&place) {
                            stmt_kills[b].insert(d);
                        }
                    }
                    Ok(StmtKind::StorageLive(i)) | Ok(StmtKind::StorageDead(i)) => {
                        stmt_kills[b].insert(i);
                    }
                    _ => {}
                }
            }
        }
        // Forward *must*-analysis: a root counts as pinned at a call only when
        // every path reaching it pinned that local and nothing has reassigned
        // it since.  Starts at the universe and intersects down, so a block
        // whose predecessors disagree keeps only what they all hold.
        let universe: HashSet<u64> = gc_locals.keys().copied().collect();
        let out_of = |b: usize, pin: &[HashSet<u64>]| -> HashSet<u64> {
            let mut s: HashSet<u64> = pin[b].difference(&stmt_kills[b]).copied().collect();
            if let Some(TermKind::Call { call, .. }) = &terms[b] {
                if let Some(d) = bare_local(&call.dest) {
                    s.remove(&d);
                }
            }
            s.extend(term_pins[b].iter().copied());
            s
        };
        let mut pinned_in: Vec<HashSet<u64>> = (0..n)
            .map(|b| {
                if b == 0 || !reachable.contains(&b) {
                    HashSet::new()
                } else {
                    universe.clone()
                }
            })
            .collect();
        for _round in 0..n + 8 {
            let outs: Vec<HashSet<u64>> = (0..n).map(|b| out_of(b, &pinned_in)).collect();
            let mut changed = false;
            let mut next: Vec<HashSet<u64>> = Vec::with_capacity(n);
            for b in 0..n {
                if b == 0 || !reachable.contains(&b) {
                    next.push(HashSet::new());
                    continue;
                }
                let mut acc = match preds[b].first() {
                    Some(&p) => outs[p].clone(),
                    None => HashSet::new(),
                };
                for &p in preds[b].iter().skip(1) {
                    acc.retain(|l| outs[p].contains(l));
                }
                if acc != pinned_in[b] {
                    changed = true;
                }
                next.push(acc);
            }
            pinned_in = next;
            if !changed {
                break;
            }
        }

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
            // One span answers every column below: a call with no statement to
            // stand on falls back to the function's own span, and a file taken
            // from anywhere else would then name a different one.
            let at = bb
                .statements
                .last()
                .map_or(&fd.item_meta.span.data, |s| &s.span.data);
            if !unbracketed.contains(&b) {
                stats.withheld_under_a_bracket += 1;
                // Withholding the call is right -- a bracket does dominate it
                // -- but "a bracket is open" and "this root is in it" are two
                // questions, and only the first was ever asked.  Grade the
                // second here so a bracket that pins the wrong set stops
                // reading as coverage.
                if opaque_contents || !reachable.contains(&b) {
                    stats.withheld_contents_opaque += 1;
                    continue;
                }
                let held: HashSet<u64> =
                    pinned_in[b].difference(&stmt_kills[b]).copied().collect();
                let mut missing: Vec<String> = after
                    .iter()
                    .filter(|l| !held.contains(l))
                    .map(|l| gc_locals[l].clone())
                    .collect();
                if missing.is_empty() {
                    stats.withheld_bracket_covers += 1;
                    continue;
                }
                // Locals `1..=arg_count` are this body's parameters; 0 is the
                // return place.
                let params = 1..=body.locals.arg_count;
                let mut missing_local: Vec<String> = after
                    .iter()
                    .filter(|l| !held.contains(l) && !params.contains(*l))
                    .map(|l| gc_locals[l].clone())
                    .collect();
                // A caller's pin keeps the object alive; it does not keep
                // *this* body's copy correct.  A collection rewrites the slot
                // the caller reads back from, not the callee's local, so a
                // missing root of a kind that relocates is stale here however
                // well the caller bracketed it.  `build_class_inner` says as
                // much the other way round -- its borrowed arguments are safe
                // because they are tuples, which never move.
                let mut missing_movable: Vec<String> = after
                    .iter()
                    .filter(|l| !held.contains(l) && movable_args.contains(l))
                    .map(|l| gc_locals[l].clone())
                    .collect();
                missing.sort();
                missing_local.sort();
                missing_movable.sort();
                if !missing_local.is_empty() {
                    stats.withheld_bracket_short_body_local += 1;
                }
                if !missing_movable.is_empty() {
                    stats.withheld_bracket_short_movable += 1;
                }
                let mut pinned: Vec<String> = held
                    .iter()
                    .filter_map(|l| gc_locals.get(l).cloned())
                    .collect();
                pinned.sort();
                stats.withheld_bracket_short += 1;
                stats.short_brackets.push(ShortBracket {
                    func_name: fd.item_meta.name_path(),
                    file: llbc.file_path(at.file_id).unwrap_or_default().to_string(),
                    line: at.beg.line,
                    callee_name: cg.names.get(callee).cloned().unwrap_or_default(),
                    missing,
                    missing_local,
                    missing_movable,
                    pinned,
                });
                continue;
            }
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
            // One span answers both columns: a call with no statement to
            // stand on falls back to the function's own span, and a file
            // taken from anywhere else would then name a different one.
            let at = bb
                .statements
                .last()
                .map_or(&fd.item_meta.span.data, |s| &s.span.data);
            findings.push(Finding {
                func: id,
                func_name: fd.item_meta.name_path(),
                file: llbc.file_path(at.file_id).unwrap_or_default().to_string(),
                line: at.beg.line,
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

#[cfg(test)]
mod tests {
    use super::*;
    use majit_charon_reader::ullbc::{Operand, Place, PlaceKind, Rvalue, TyRef};

    fn ty() -> TyRef {
        TyRef::Dedup { id: 0 }
    }

    fn local(i: u64) -> Place {
        Place {
            kind: PlaceKind::Local(i),
            ty: ty(),
        }
    }

    fn mv(i: u64) -> Operand {
        Operand::Move(local(i))
    }

    fn chased(l: u64, defs: &HashMap<u64, PinSrc>) -> Vec<u64> {
        let mut out = HashSet::new();
        chase_pinned(l, defs, &mut out, 0);
        let mut v: Vec<u64> = out.into_iter().collect();
        v.sort();
        v
    }

    /// The set is named by `pin_roots`, never by the opener.  Reading the
    /// contents off `push_roots` would find no argument at all and report
    /// every bracket as empty.
    #[test]
    fn the_scope_opener_is_not_where_a_root_set_is_named() {
        assert!(is_pin_fn("pyre_object::gc_roots::pin_root"));
        assert!(is_pin_fn("pyre_object::gc_roots::pin_roots"));
        assert!(is_pin_fn("pyre_object::gc_roots::publish_roots"));
        assert!(!is_pin_fn("pyre_object::gc_roots::push_roots"));
        assert!(!is_pin_fn("pyre_object::gc_roots::enter_roots_frame"));
    }

    /// `pin_roots(&[a, b])` lowers to an array build, a borrow, and the call.
    #[test]
    fn a_pin_argument_traces_back_to_the_array_it_was_built_from() {
        let mut defs = HashMap::new();
        defs.insert(2, PinSrc::Aggregate(vec![10, 11]));
        defs.insert(
            3,
            pin_src(&Rvalue::Ref {
                place: local(2),
                kind: serde_json::Value::Null,
                ptr_metadata: serde_json::Value::Null,
            })
            .expect("a borrow is a followable alias"),
        );
        assert_eq!(chased(3, &defs), vec![2, 3, 10, 11]);
    }

    /// The borrow is `&[T; N]` and the parameter is `&[T]`, so an unsize cast
    /// sits between them on some lowerings and on others it does not.
    #[test]
    fn an_unsize_cast_between_the_array_and_the_slice_is_walked_through() {
        let mut defs = HashMap::new();
        defs.insert(2, PinSrc::Aggregate(vec![10]));
        defs.insert(3, PinSrc::Alias(2));
        defs.insert(
            4,
            pin_src(&Rvalue::Cast(serde_json::Value::Null, mv(3), ty()))
                .expect("a cast is a followable alias"),
        );
        assert_eq!(chased(4, &defs), vec![2, 3, 4, 10]);
    }

    /// A pin publishes a value, so every local spelling that value at the pin
    /// is covered by it -- keeping only the end of the chain would read a
    /// covered root as missing and accuse correct code.
    #[test]
    fn a_pin_covers_every_local_the_value_is_spelled_by() {
        let mut defs = HashMap::new();
        defs.insert(1, pin_src(&Rvalue::Use(mv(0))).expect("a use is an alias"));
        assert_eq!(chased(1, &defs), vec![0, 1]);
    }

    /// A local defined by a call has no followable definition; the pin still
    /// names that local and nothing more.
    #[test]
    fn an_argument_with_no_followable_definition_names_only_itself() {
        assert_eq!(chased(7, &HashMap::new()), vec![7]);
    }

    /// Two locals aliasing each other must not walk forever.  A body is read
    /// from an artefact, so no shape can be ruled out by construction.
    #[test]
    fn an_alias_cycle_terminates() {
        let mut defs = HashMap::new();
        defs.insert(0, PinSrc::Alias(1));
        defs.insert(1, PinSrc::Alias(0));
        assert_eq!(chased(0, &defs), vec![0, 1]);
    }

    /// A shape this reader does not model leaves the set unread rather than
    /// guessed at: `RootedItems` fills its slots through a method, and reading
    /// that as an empty pin would report every root in it as missing.
    #[test]
    fn an_unmodelled_rvalue_yields_no_source_at_all() {
        assert!(pin_src(&Rvalue::Unknown).is_none());
        assert!(pin_src(&Rvalue::Len(local(1))).is_none());
    }
}
