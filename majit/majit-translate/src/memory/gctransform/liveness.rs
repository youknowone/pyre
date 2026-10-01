//! `rpython/memory/gctransform/framework.py get_livevars_for_roots` —
//! which GC pointers are live across an operation that can collect.
//!
//! Upstream reads the answer straight off the flow graph with
//! `hop.livevars_after_op()`, then brackets the operation with
//! `push_roots` / `pop_roots`. The flowspace transformer now performs that
//! insertion automatically for translated graphs. Native interpreter paths
//! also run as rustc-compiled Rust, so this audit computes the corresponding
//! live set over ULLBC and reports where their source bracket is *missing*.
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
    BasicBlock, CallFunc, CallKind, FunId, Operand, Place, PlaceKind, ProjectionElem, Rvalue,
    StmtKind, SwitchTargets, TermKind, TyRef,
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
    /// name says it addresses a kind whose header a minor collection
    /// relocates.  A stale pointer that is only
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
    /// Bodies holding two live root scopes at once somewhere in them.  The
    /// pin set carries an owner, so this on its own no longer withholds a
    /// body; the count stays because a nested body that is unread is unread
    /// for one of the other reasons and the distinction is worth measuring.
    pub bodies_with_nested_scopes: usize,
    /// Calls counted in [`Self::withheld_contents_opaque`] that came from a
    /// body in [`Self::bodies_with_nested_scopes`].
    pub withheld_opaque_from_nested: usize,
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
    /// terminator this reader could not parse, a scope-local slot overwrite,
    /// or a pin whose argument does not trace back to a set.  Neither graded
    /// nor claimed clean -- a set read short would accuse correct code.
    pub withheld_contents_opaque: usize,
    /// Of [`Self::withheld_bracket_short`], those missing a root the body
    /// produced itself, which no caller's bracket can be covering.
    pub withheld_bracket_short_body_local: usize,
    /// Of [`Self::withheld_bracket_short`], those missing a root this body goes
    /// on to address as a movable kind.  A caller's pin does not answer for
    /// these, so they do not depend on the intra-procedural limit.
    pub withheld_bracket_short_movable: usize,
    /// The [`ScanStats::withheld_bracket_short`] calls, named.
    pub short_brackets: Vec<ShortBracket>,
    /// Pins whose argument local is still read after the pin normalised the
    /// slot.  Not a missing root -- a stale word.  See [`StalePinRead`].
    pub pin_arg_read_after: usize,
    /// Of those, the ones reading a local this body later addresses as a
    /// movable kind, where the stale word is dereferenced.
    pub pin_arg_read_after_movable: usize,
    /// The [`Self::pin_arg_read_after`] pins, named.
    pub stale_pin_reads: Vec<StalePinRead>,
    /// Bodies that hand at least one GC pointer to a movable-addressing callee.
    ///
    /// The `movable` columns are all filters over this set, so a zero here says
    /// the ranking never had anything to rank -- a different fact from "nothing
    /// ranked".  Worth a counter of its own: the marker list, the id space the
    /// callees are resolved in, and the operand shape a call argument takes each
    /// silence the columns the same way, and only this number separates "the
    /// scan found no such call" from "the scan found them and none was stale".
    pub bodies_with_movable_args: usize,
    /// Locals summed over [`Self::bodies_with_movable_args`], so a body that
    /// contributes one argument is told apart from one that contributes twenty.
    pub movable_arg_locals: usize,
    /// [`Self::withheld_contents_opaque`] split by the first reason the body's
    /// pinned set went unread.
    pub opaque_by_reason: std::collections::BTreeMap<&'static str, usize>,
    /// One entry per body with unread contents: function, file, first
    /// reason, and how many withheld calls it cost.
    pub opaque_bodies: Vec<(String, String, &'static str, usize)>,
}

/// A pin whose argument the body goes on to read.
///
/// `pin_root` returns the word its slot holds *after* the publish, because the
/// publish is a safepoint: a foreign collection can forward the value between
/// the caller's copy and the query, leaving the caller's local pointing at a
/// forwarding stub.  Reading the returned word is the fix; `let _ =` opts out
/// and thereby asserts the kind never moves, which is what
/// [`Self::movable`] checks.  The assertion is worth checking because it has
/// been written down and been wrong: `getitem_tuple` and `getitem_str` each
/// stated in prose that their receiver never moves, and both receivers are
/// nursery-allocated.
pub struct StalePinRead {
    pub func_name: String,
    pub file: String,
    pub line: u64,
    /// Which pin was called.  `with_roots!` expands to `pin_roots` and runs its
    /// body before reading the slots back, so it shows this shape by
    /// construction; a hand-written `let _ = pin_root(x)` does not.
    pub pin_name: String,
    /// The locals handed to the pin and still read afterwards.
    pub locals: Vec<String>,
    /// Of those, the ones this body later addresses as a movable kind,
    /// which is where a stale word is dereferenced rather than merely carried.
    pub movable: Vec<String>,
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
    /// says it addresses a movable kind -- one whose header a
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
/// A stale `PyObjectRef` handed to one of these is dereferenced as a corpse
/// rather than merely stored or returned.  The set is not the whole moving
/// heap: it is the part of it a callee's *name* identifies, and the entries
/// below record which kind each spelling was checked against.
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
    // The families whose headers the interpreter allocates in the nursery, so
    // a pointer reaching one of these is addressed as an object that relocates.
    // Spelled at the moving constructor rather than at the family prefix: a
    // `w_weakref_lifeline_*` is `allocate_stable` and a `w_str_new` is
    // immortal, so the wider spelling would name a pointer that cannot move.
    // `w_dict_view_*` needs no entry of its own -- the `w_dict_` prefix already
    // spells it.
    //
    // `str` has no entry at all, and deliberately.  Its mobility is decided per
    // constructor -- `w_str_from_wtf8_managed_collecting` is the one nursery
    // spelling, while `w_str_from_wtf8_managed` is `try_gc_alloc_stable` and
    // `w_str_new` is immortal -- so the accessors every one of them shares
    // cannot say which kind the pointer reaching them is.  Naming a constructor
    // here would add nothing either: they take a `Wtf8Buf`, never a
    // `PyObjectRef`, so no GC-pointer local is ever an argument of one.  The
    // same reading rules out the items-block and JITFRAME allocators, whose
    // arguments are a capacity and a type id.
    // `w_method_` is spelled at the accessors rather than at the family
    // prefix, because `w_method_new` takes the new method's three members --
    // not a `Method` -- and roots and re-reads them itself.  Under the family
    // prefix every argument of that constructor counted, so an unrelated local
    // passed as a member was reported as addressed through a method it is only
    // stored in.  `w_tuple_new` needs no such split: it takes a slice, so no
    // `PyObjectRef` local is ever one of its arguments.
    "w_tuple_",
    "w_method_get_",
    "w_method_set_",
    "w_gc_weakref_box_",
    "w_weakref_new",
];

/// The callees that say a pointer reaching them is addressed as a movable
/// object, resolved once for the whole scan.
///
/// `hops` is how far past the named callee to look.  At 0 only the callee's own
/// name is read, which is what the `tier 1.5` column has always done -- and a
/// thin wrapper defeats it: `module_ns_store` is one line forwarding to
/// `w_dict_setitem_str_no_proxy` and matches no marker itself.  A zero at 0 hops
/// therefore says the marker was not the immediate callee's name, not that
/// nothing is addressed as a list or dict.  Each hop admits a caller of
/// something already in the set.
pub fn movable_callee_ids(
    cg: &super::framework::CallGraph,
    markers: &[&str],
    hops: u32,
) -> HashSet<u64> {
    let mut out: HashSet<u64> = cg
        .names
        .iter()
        .filter(|(_, n)| markers.iter().any(|m| n.contains(m)))
        .map(|(&id, _)| id)
        .collect();
    for _ in 0..hops {
        let grown: HashSet<u64> = cg
            .callees
            .iter()
            .filter(|(id, cs)| !out.contains(id) && cs.iter().any(|c| out.contains(c)))
            .map(|(&id, _)| id)
            .collect();
        if grown.is_empty() {
            break;
        }
        out.extend(grown);
    }
    out
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
        // `pin_root` / `shadow_stack_get` live in `pyre-object`. An
        // optional-module artefact extracted `--opaque pyre_object`
        // does not carry those names, but it does carry its own
        // `register_module` / `module_ns_store` signatures, which
        // take the same artefact-local `PyObjectRef` id.
        if name.ends_with("gc_roots::pin_root")
            || name.ends_with("::register_module")
            || name.ends_with("module_ns_store")
        {
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

/// The type ids of `&[PyObjectRef]` in *this* artefact.
///
/// A builtin receives its arguments as a native slice: a copy the collector
/// does not rewrite, so an element read after a collecting call is the same
/// stale word a bare local would be.  Each borrow region is its own type id,
/// so the spellings are read off every body's locals, as for `Option`.
pub fn gc_slice_type_ids(llbc: &majit_charon_reader::Llbc, gc_tys: &HashSet<u64>) -> HashSet<u64> {
    let id_of = |v: &serde_json::Value| {
        v.get("Deduplicated")
            .and_then(serde_json::Value::as_u64)
            .or_else(|| v.pointer("/Value/0").and_then(serde_json::Value::as_u64))
    };
    let body_of = |v: &serde_json::Value| {
        v.pointer("/Value/1")
            .cloned()
            .or_else(|| id_of(v).and_then(|id| llbc.dedup_body(id).cloned()))
    };
    let mut seen = HashSet::new();
    let mut out = HashSet::new();
    for fd in llbc.iter_local_fns() {
        let Some(body) = fd.unstructured() else {
            continue;
        };
        for l in &body.locals.locals {
            let Some(t) = ty_id(&l.ty) else { continue };
            if gc_tys.contains(&t) || !seen.insert(t) {
                continue;
            }
            let elem = llbc
                .dedup_body(t)
                .and_then(|b| b.pointer("/Ref/1").cloned())
                .and_then(|inner| body_of(&inner))
                .and_then(|inner| inner.pointer("/Slice/0").and_then(id_of));
            if elem.is_some_and(|e| gc_tys.contains(&e)) {
                out.insert(t);
            }
        }
    }
    out
}

/// The type ids of `Option<PyObjectRef>` in *this* artefact.
///
/// A GC pointer held as `Option<PyObjectRef>` goes stale across a collecting
/// call exactly like a bare one: the payload is the same word, and nothing
/// publishes it.  Read off every body's locals, so the answer covers only
/// the spellings a body actually holds.
pub fn gc_option_type_ids(llbc: &majit_charon_reader::Llbc, gc_tys: &HashSet<u64>) -> HashSet<u64> {
    let mut seen = HashSet::new();
    let mut out = HashSet::new();
    for fd in llbc.iter_local_fns() {
        let Some(body) = fd.unstructured() else {
            continue;
        };
        for l in &body.locals.locals {
            let Some(t) = ty_id(&l.ty) else { continue };
            if gc_tys.contains(&t) || !seen.insert(t) {
                continue;
            }
            let is_option = llbc
                .dedup_to_adt_def_id(t)
                .and_then(|def| llbc.type_by_id(def))
                .is_some_and(|td| td.item_meta.name_path() == "core::option::Option");
            if !is_option {
                continue;
            }
            let arg = llbc
                .dedup_body(t)
                .and_then(|v| v.pointer("/Adt/generics/types/0"))
                .and_then(|v| serde_json::from_value::<TyRef>(v.clone()).ok())
                .and_then(|r| ty_id(&r));
            if arg.is_some_and(|a| gc_tys.contains(&a)) {
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

/// `x.PtrMetadata` -- a fat pointer's length, not the words behind it.
fn is_metadata_place(p: &Place) -> bool {
    matches!(&p.kind, PlaceKind::Projection(_, ProjectionElem::Atom(e)) if e == "PtrMetadata")
}

fn use_operand(o: &Operand, out: &mut HashSet<u64>) {
    match o {
        Operand::Copy(p) | Operand::Move(p) if is_metadata_place(p) => {}
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
        // A slice's length lives in the fat pointer, not in the elements the
        // collector would have to forward.
        Rvalue::Len(_) => {}
        Rvalue::Use(o, _) | Rvalue::UnaryOp(_, o) => use_operand(o, out),
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
        Rvalue::Discriminant(p) => {
            if let Some(l) = place_local(p) {
                out.insert(l);
            }
        }
        Rvalue::Cast(_, o, _) | Rvalue::Repeat(o, _, _, _) | Rvalue::ShallowInitBox(o, _) => {
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
        Rvalue::Use(o, _) | Rvalue::Cast(_, o, _) => one(o).map(PinSrc::Alias),
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

/// Every local an argument to a movable-addressing callee is spelled by.
///
/// The alias half of [`chase_pinned`], and needed for the same reason on the
/// other side of the intersection: a call argument is materialised as its own
/// temporary (`_t = copy _obj; w_tuple_getitem(move _t, ..)`), so matching the
/// argument local against the pinned local compares two spellings of one value
/// and finds nothing.  Aggregates are not followed here -- `pin_roots(&[a, b])`
/// pins each element, but a callee handed a container is addressing the
/// container, not the elements.
fn chase_arg_aliases(l: u64, defs: &HashMap<u64, PinSrc>, out: &mut HashSet<u64>, depth: u32) {
    if !out.insert(l) || depth > 8 {
        return;
    }
    if let Some(PinSrc::Alias(next)) = defs.get(&l) {
        chase_arg_aliases(*next, defs, out, depth + 1);
    }
}

/// The functions that write a livevar onto the root stack.
///
/// `push_roots` opens the scope and takes no arguments; the set is named
/// separately, so the contents are read off these and not off the opener.
/// `publish_roots` is `pin_roots` without the normalize half, and pins just
/// the same.
///
/// The free functions and the scope-local `pin_root` / `publish` forms.
/// `scan` models the matching `RootScope` Drop at the same time, so a method
/// pin cannot leak into calls after its guard was truncated.  Scope-local
/// `set` overwrites a coloured slot and remains opaque until the analysis
/// carries a slot-to-root map; guessing there could claim false coverage.
fn is_pin_fn(name: &str) -> bool {
    name.ends_with("::pin_root")
        || name.ends_with("::pin_roots")
        || name.ends_with("::publish_roots")
        || (name.contains("gc_roots::<Impl>") && name.ends_with("::publish"))
}

fn reads_root_slot(name: &str) -> bool {
    name.ends_with("gc_roots::shadow_stack_get")
        || (name.contains("gc_roots::<Impl>") && name.ends_with("::get"))
}

/// `CallKind::Fun(FunId::Regular)` — the only call shape [`scan`] treats as
/// a resolved callee. `push_roots` and the pin helpers are read off these.
fn regular_fun_id(func: &CallFunc) -> Option<u64> {
    match func {
        CallFunc::Regular(reg) => match &reg.kind {
            CallKind::Fun(FunId::Regular { id }) => Some(*id),
            _ => None,
        },
        _ => None,
    }
}

/// The predicate [`scan`] uses to find `bracket_blocks`: this call opens a
/// root scope.
fn opens_root_scope(func: &CallFunc, push_roots: &HashSet<u64>) -> bool {
    regular_fun_id(func).is_some_and(|id| push_roots.contains(&id))
}

/// A function that pins into its caller's open root scope.
///
/// It calls [`is_pin_fn`] (or another such function) and never calls
/// `push_roots` itself. `pinned_params` are 0-based positions among the call
/// arguments; `returns_pinned` means the return place holds a pin result or
/// a slot read.
#[derive(Clone, Debug, PartialEq, Eq)]
struct PinHelperSummary {
    pinned_params: HashSet<usize>,
    returns_pinned: bool,
}

/// One resolved call, reduced to the locals its arguments name.
struct HelperCallFact {
    callee: u64,
    callee_name: String,
    /// Locals [`use_operand`] named for each argument, before [`chase_pinned`].
    arg_locals: Vec<Vec<u64>>,
}

/// What [`summarize_pin_helpers`] needs from one body. Built by
/// [`helper_body_fact`]; tests construct it directly.
struct HelperBodyFact {
    has_push_roots: bool,
    /// MIR parameters are locals `1..=arg_count`. Local 0 is the return place.
    arg_count: u64,
    /// Locals that received an assignment, so they are no longer the incoming
    /// parameter value even when their number is still in `1..=arg_count`.
    assigned: HashSet<u64>,
    defs: HashMap<u64, PinSrc>,
    calls: Vec<HelperCallFact>,
    /// Bare locals whose single assignment is a pin call or a slot read.
    pin_result_locals: HashSet<u64>,
    /// Per-block indices into [`Self::calls`]. Empty means the tests' single
    /// block: every call runs, then the function returns.
    block_calls: Vec<Vec<usize>>,
    /// Normal successors only (`Call.target`, not `on_unwind`). Empty when
    /// [`Self::block_calls`] is empty.
    successors: Vec<Vec<usize>>,
}

struct PinAssignIndex {
    defs: HashMap<u64, PinSrc>,
    /// Locals written at least once. A parameter local in this set is the
    /// replacement, not the incoming argument.
    assigned: HashSet<u64>,
    /// Bare locals whose single assignment is a call, and that were not
    /// overwritten later. A pin result is one of these.
    call_dests: HashSet<u64>,
    /// `_t = &mut _l` — the callee writes the live word back through `_t`.
    mut_borrow_of: HashMap<u64, u64>,
}

/// Single-assignment locals the pin-argument chase will follow.
///
/// A local assigned twice is not a chain worth following. Call destinations
/// are marked assigned so a later statement does not alias them, and they are
/// not given a [`PinSrc`]: the value came from the callee.
fn index_pin_assigns(blocks: &[BasicBlock], terms: &[Option<TermKind>]) -> PinAssignIndex {
    let mut defs: HashMap<u64, PinSrc> = HashMap::new();
    let mut defined: HashSet<u64> = HashSet::new();
    // `_t = &mut _l`: a callee handed `_t` owns keeping `_l` current, the
    // way `try_dispatch_binary_special` pins both operands and writes the
    // live words back through its `&mut` parameters.
    let mut mut_borrow_of: HashMap<u64, u64> = HashMap::new();
    let mut call_dests: HashSet<u64> = HashSet::new();
    for (b, blk) in blocks.iter().enumerate() {
        for st in &blk.statements {
            let Ok(StmtKind::Assign(place, rv)) = st.stmt_kind() else {
                continue;
            };
            let Some(d) = bare_local(&place) else {
                continue;
            };
            if !defined.insert(d) {
                defs.remove(&d);
                mut_borrow_of.remove(&d);
                call_dests.remove(&d);
                continue;
            }
            // A two-phase call argument reborrows: `_t = &mut _l;
            // _u = &TwoPhaseMut (*_t)`, so a deref of a recorded borrow
            // names the same local.
            if let Rvalue::Ref { place, kind, .. } = &rv
                && matches!(kind.as_str(), Some("Mut" | "TwoPhaseMut"))
                && let Some(l) = bare_local(place).or_else(|| match &place.kind {
                    PlaceKind::Projection(base, ProjectionElem::Atom(elem)) if elem == "Deref" => {
                        bare_local(base).and_then(|t| mut_borrow_of.get(&t).copied())
                    }
                    _ => None,
                })
            {
                mut_borrow_of.insert(d, l);
            }
            if let Some(src) = pin_src(&rv) {
                defs.insert(d, src);
            }
        }
        if let Some(TermKind::Call { call, .. }) = &terms[b] {
            if let Some(d) = bare_local(&call.dest) {
                if !defined.insert(d) {
                    defs.remove(&d);
                    call_dests.remove(&d);
                } else {
                    call_dests.insert(d);
                }
            }
        }
    }
    PinAssignIndex {
        defs,
        assigned: defined,
        call_dests,
        mut_borrow_of,
    }
}

/// Parameter positions (0-based) among `1..=arg_count` that `seeds` reaches.
fn param_positions_reaching(
    seeds: &[u64],
    defs: &HashMap<u64, PinSrc>,
    assigned: &HashSet<u64>,
    arg_count: u64,
) -> HashSet<usize> {
    let mut chased = HashSet::new();
    for &local in seeds {
        chase_pinned(local, defs, &mut chased, 0);
    }
    chased
        .into_iter()
        .filter(|local| *local >= 1 && *local <= arg_count && !assigned.contains(local))
        .map(|local| (local - 1) as usize)
        .collect()
}

fn call_pins_params(
    call: &HelperCallFact,
    body: &HelperBodyFact,
    nested: &HashMap<u64, PinHelperSummary>,
) -> HashSet<usize> {
    let mut out = HashSet::new();
    if is_pin_fn(&call.callee_name) {
        for seeds in &call.arg_locals {
            out.extend(param_positions_reaching(
                seeds,
                &body.defs,
                &body.assigned,
                body.arg_count,
            ));
        }
        return out;
    }
    let Some(summary) = nested.get(&call.callee) else {
        return out;
    };
    for &position in &summary.pinned_params {
        let Some(seeds) = call.arg_locals.get(position) else {
            continue;
        };
        out.extend(param_positions_reaching(
            seeds,
            &body.defs,
            &body.assigned,
            body.arg_count,
        ));
    }
    out
}

fn block_layout(body: &HelperBodyFact) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    if body.block_calls.is_empty() {
        return (vec![(0..body.calls.len()).collect()], vec![Vec::new()]);
    }
    (body.block_calls.clone(), body.successors.clone())
}

/// Parameter positions pinned on every normal path from entry to a return.
/// A pin that lives on only one arm of a switch, or after an early return,
/// is not in the set: the caller scan would otherwise treat the helper as
/// bracketing a use that the skipped arm never rooted.
fn must_pinned_params(
    body: &HelperBodyFact,
    nested: &HashMap<u64, PinHelperSummary>,
) -> HashSet<usize> {
    let (block_calls, successors) = block_layout(body);
    let n = block_calls.len();
    if n == 0 {
        return HashSet::new();
    }
    let block_pins: Vec<HashSet<usize>> = block_calls
        .iter()
        .map(|idxs| {
            let mut pins = HashSet::new();
            for &i in idxs {
                if let Some(call) = body.calls.get(i) {
                    pins.extend(call_pins_params(call, body, nested));
                }
            }
            pins
        })
        .collect();
    // Forward must: join is intersection, transfer is union with this block's
    // pins. Entry has nothing yet.
    let mut in_set: Vec<Option<HashSet<usize>>> = vec![None; n];
    in_set[0] = Some(HashSet::new());
    let mut work: Vec<usize> = vec![0];
    while let Some(b) = work.pop() {
        let mut out = in_set[b].clone().unwrap_or_default();
        out.extend(&block_pins[b]);
        for &s in successors.get(b).map(|v| v.as_slice()).unwrap_or(&[]) {
            if s >= n {
                continue;
            }
            match &in_set[s] {
                None => {
                    in_set[s] = Some(out.clone());
                    work.push(s);
                }
                Some(have) => {
                    let joined: HashSet<usize> = have.intersection(&out).copied().collect();
                    if joined != *have {
                        in_set[s] = Some(joined);
                        work.push(s);
                    }
                }
            }
        }
    }
    let mut exits = Vec::new();
    for b in 0..n {
        if successors.get(b).map(|s| s.is_empty()).unwrap_or(true) && in_set[b].is_some() {
            exits.push(b);
        }
    }
    if exits.is_empty() {
        return HashSet::new();
    }
    let mut must = {
        let mut out = in_set[exits[0]].clone().unwrap_or_default();
        out.extend(&block_pins[exits[0]]);
        out
    };
    for &b in &exits[1..] {
        let mut out = in_set[b].clone().unwrap_or_default();
        out.extend(&block_pins[b]);
        must = must.intersection(&out).copied().collect();
    }
    must
}

fn direct_pinned_params(body: &HelperBodyFact) -> HashSet<usize> {
    must_pinned_params(body, &HashMap::new())
}

fn call_is_collecting(name: &str) -> bool {
    crate::memory::gctransform::framework::COLLECTING_SEEDS
        .iter()
        .any(|seed| name == *seed || name.ends_with(&format!("::{seed}")))
}

/// Local 0 holds a pin result or a slot read, following single-assignment
/// aliases only. An aggregate or a second assignment is not that, and stays
/// unpinned. A collecting call after the pin drops the claim:
/// `framework.py get_livevars_for_roots` is live-at-call.
fn returns_pinned_word(body: &HelperBodyFact) -> bool {
    let mut local = 0u64;
    let mut seen = HashSet::new();
    let aliases_pin = loop {
        if !seen.insert(local) {
            break false;
        }
        if body.pin_result_locals.contains(&local) {
            break true;
        }
        match body.defs.get(&local) {
            Some(PinSrc::Alias(next)) => local = *next,
            _ => break false,
        }
    };
    aliases_pin && !collects_after_pin(body)
}

/// True when a collecting call can run after a pin on the way to the return.
fn collects_after_pin(body: &HelperBodyFact) -> bool {
    if body.block_calls.is_empty() {
        let mut seen_pin = false;
        for call in &body.calls {
            if is_pin_fn(&call.callee_name) || reads_root_slot(&call.callee_name) {
                seen_pin = true;
            } else if seen_pin && call_is_collecting(&call.callee_name) {
                return true;
            }
        }
        return false;
    }
    let n = body.block_calls.len();
    let mut stack = Vec::new();
    for (b, indices) in body.block_calls.iter().enumerate() {
        let mut pin_at = None;
        for &i in indices {
            let name = &body.calls[i].callee_name;
            if is_pin_fn(name) || reads_root_slot(name) {
                pin_at = Some(i);
                break;
            }
        }
        let Some(pin_at) = pin_at else {
            continue;
        };
        for &i in indices {
            if i <= pin_at {
                continue;
            }
            let name = &body.calls[i].callee_name;
            if call_is_collecting(name) && !is_pin_fn(name) && !reads_root_slot(name) {
                return true;
            }
        }
        if let Some(succ) = body.successors.get(b) {
            stack.extend(succ.iter().copied());
        }
    }
    if stack.is_empty() {
        return false;
    }
    let mut seen = vec![false; n];
    while let Some(b) = stack.pop() {
        if seen[b] {
            continue;
        }
        seen[b] = true;
        for &i in &body.block_calls[b] {
            let name = &body.calls[i].callee_name;
            if call_is_collecting(name) && !is_pin_fn(name) && !reads_root_slot(name) {
                return true;
            }
        }
        if let Some(succ) = body.successors.get(b) {
            stack.extend(succ.iter().copied());
        }
    }
    false
}

fn body_calls_pin(body: &HelperBodyFact) -> bool {
    body.calls.iter().any(|call| is_pin_fn(&call.callee_name))
}

/// Which of `bodies` are pin helpers, and which of their parameters they pin.
///
/// A body is a helper when it does not open a root scope and it calls
/// [`is_pin_fn`] or, transitively, another helper. `pinned_params` starts
/// from the locals that reach a pin argument and then grows by parameters
/// handed to a nested helper at one of *its* pinned positions.
fn summarize_pin_helpers(bodies: &HashMap<u64, HelperBodyFact>) -> HashMap<u64, PinHelperSummary> {
    let opens_nested = |body: &HelperBodyFact| {
        body.has_push_roots
            || body
                .calls
                .iter()
                .any(|call| bodies.get(&call.callee).is_some_and(|b| b.has_push_roots))
    };
    let mut helpers: HashSet<u64> = bodies
        .iter()
        .filter(|(_, body)| !opens_nested(body) && body_calls_pin(body))
        .map(|(&id, _)| id)
        .collect();
    loop {
        let grown: Vec<u64> = bodies
            .iter()
            .filter(|(id, body)| {
                !helpers.contains(*id)
                    && !opens_nested(body)
                    && body.calls.iter().any(|call| helpers.contains(&call.callee))
            })
            .map(|(&id, _)| id)
            .collect();
        if grown.is_empty() {
            break;
        }
        helpers.extend(grown);
    }

    let mut summaries: HashMap<u64, PinHelperSummary> = helpers
        .iter()
        .map(|&id| {
            let body = &bodies[&id];
            (
                id,
                PinHelperSummary {
                    pinned_params: direct_pinned_params(body),
                    returns_pinned: returns_pinned_word(body),
                },
            )
        })
        .collect();

    // Nested helpers contribute only on paths that always call them. Each
    // round may add a must-pin, and a helper has finitely many parameters.
    loop {
        let mut changed = false;
        for &id in &helpers {
            let next = must_pinned_params(&bodies[&id], &summaries);
            let have = &mut summaries
                .get_mut(&id)
                .expect("helper id is in the summary map")
                .pinned_params;
            if next != *have {
                *have = next;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    summaries
}

/// Superset of the helper ids, from the call graph alone.
///
/// [`CallGraph::callees`](super::framework::CallGraph::callees) records every
/// regular callee [`opens_root_scope`] would see, plus some trait-default
/// edges. Those extra edges can only add candidates; [`helper_body_fact`]
/// re-reads the body with [`opens_root_scope`] and [`summarize_pin_helpers`]
/// drops whoever does not qualify.
fn helper_candidate_ids(
    callees: &HashMap<u64, HashSet<u64>>,
    names: &HashMap<u64, String>,
    push_roots: &HashSet<u64>,
) -> HashSet<u64> {
    let mut openers: HashSet<u64> = callees
        .iter()
        .filter(|(_, cs)| cs.iter().any(|callee| push_roots.contains(callee)))
        .map(|(&id, _)| id)
        .collect();
    loop {
        let grown: Vec<u64> = callees
            .iter()
            .filter(|(id, cs)| {
                !openers.contains(*id) && cs.iter().any(|callee| openers.contains(callee))
            })
            .map(|(&id, _)| id)
            .collect();
        if grown.is_empty() {
            break;
        }
        openers.extend(grown);
    }
    let pins = |cs: &HashSet<u64>| {
        cs.iter()
            .any(|callee| names.get(callee).is_some_and(|name| is_pin_fn(name)))
    };
    let mut helpers: HashSet<u64> = callees
        .iter()
        .filter(|(id, cs)| !openers.contains(*id) && pins(cs))
        .map(|(&id, _)| id)
        .collect();
    loop {
        let grown: Vec<u64> = callees
            .iter()
            .filter(|(id, cs)| {
                !helpers.contains(*id)
                    && !openers.contains(*id)
                    && cs.iter().any(|callee| helpers.contains(callee))
            })
            .map(|(&id, _)| id)
            .collect();
        if grown.is_empty() {
            break;
        }
        helpers.extend(grown);
    }
    helpers
}

fn helper_body_fact(
    llbc: &majit_charon_reader::Llbc,
    fd: &majit_charon_reader::ullbc::FunDecl,
    push_roots: &HashSet<u64>,
    names: &HashMap<u64, String>,
) -> Option<HelperBodyFact> {
    let body = fd.unstructured()?;
    let terms: Vec<Option<TermKind>> = body.body.iter().map(|blk| blk.term(llbc).ok()).collect();
    // A terminator this reader cannot classify may be the `push_roots` that
    // disqualifies the helper, or the only pin. Either misread is worse than
    // leaving the caller unread.
    if terms
        .iter()
        .any(|term| matches!(term, None | Some(TermKind::Unknown)))
    {
        return None;
    }
    let index = index_pin_assigns(&body.body, &terms);
    let mut has_push_roots = false;
    let mut calls = Vec::new();
    let mut pin_result_locals = HashSet::new();
    let n = terms.len();
    let mut block_calls = vec![Vec::new(); n];
    let mut succ = vec![Vec::new(); n];
    for (b, term) in terms.iter().enumerate() {
        if let Some(t) = term {
            succ[b] = match t {
                TermKind::Call { target, .. }
                | TermKind::Assert { target, .. }
                | TermKind::Drop { target, .. } => vec![*target as usize],
                other => successors(other).into_iter().map(|s| s as usize).collect(),
            };
        }
        let Some(TermKind::Call { call, .. }) = term else {
            continue;
        };
        if opens_root_scope(&call.func, push_roots) {
            has_push_roots = true;
        }
        let Some(callee) = regular_fun_id(&call.func) else {
            continue;
        };
        let callee_name = names.get(&callee).cloned().unwrap_or_default();
        if (is_pin_fn(&callee_name) || reads_root_slot(&callee_name))
            && let Some(dest) = bare_local(&call.dest)
            && index.call_dests.contains(&dest)
        {
            pin_result_locals.insert(dest);
        }
        let mut arg_locals = Vec::with_capacity(call.args.len());
        for arg in &call.args {
            let mut seed = HashSet::new();
            use_operand(arg, &mut seed);
            arg_locals.push(seed.into_iter().collect());
        }
        block_calls[b].push(calls.len());
        calls.push(HelperCallFact {
            callee,
            callee_name,
            arg_locals,
        });
    }
    Some(HelperBodyFact {
        has_push_roots,
        arg_count: body.locals.arg_count,
        assigned: index.assigned,
        defs: index.defs,
        calls,
        pin_result_locals,
        block_calls,
        successors: succ,
    })
}

/// Pin helpers in this artefact, keyed by function id.
///
/// An empty `push_roots` makes every pin-caller a helper, and no bracket is
/// ever active to grade — frame scans pass that empty set. Skip the walk.
fn pin_helper_summaries(
    llbc: &majit_charon_reader::Llbc,
    cg: &super::framework::CallGraph,
    push_roots: &HashSet<u64>,
) -> HashMap<u64, PinHelperSummary> {
    if push_roots.is_empty() {
        return HashMap::new();
    }
    let candidates = helper_candidate_ids(&cg.callees, &cg.names, push_roots);
    let mut bodies = HashMap::with_capacity(candidates.len());
    for id in candidates {
        let Some(fd) = llbc.fn_by_id(id) else {
            continue;
        };
        let Some(fact) = helper_body_fact(llbc, fd, push_roots, &cg.names) else {
            continue;
        };
        bodies.insert(id, fact);
    }
    summarize_pin_helpers(&bodies)
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
/// `movable_callees` ranks the findings for that kind
/// ([`movable_callee_ids`], or empty where the kind has no such call).
pub fn scan(
    llbc: &majit_charon_reader::Llbc,
    cg: &super::framework::CallGraph,
    reach: &HashSet<u64>,
    push_roots: &HashSet<u64>,
    gc_tys: &HashSet<u64>,
    movable_callees: &HashSet<u64>,
) -> (Vec<Finding>, ScanStats) {
    let mut findings = Vec::new();
    let mut stats = ScanStats::default();
    // One summary for the artefact: a helper pins into the caller's open
    // scope, so which parameters it publishes is a property of the callee.
    let pin_helpers = pin_helper_summaries(llbc, cg, push_roots);
    // Calls that read a slice's length or test a word against null.  A moved
    // object's stale address is still non-null, so neither answer changes.
    let metadata_fns: HashSet<u64> = cg
        .names
        .iter()
        .filter(|(_, n)| {
            matches!(
                n.as_str(),
                "core::slice::<Impl>::len"
                    | "core::slice::<Impl>::is_empty"
                    | "core::ptr::mut_ptr::<Impl>::is_null"
                    | "core::ptr::const_ptr::<Impl>::is_null"
            )
        })
        .map(|(id, _)| *id)
        .collect();
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
        let terms: Vec<Option<TermKind>> = body.body.iter().map(|b| b.term(llbc).ok()).collect();
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
                Some(TermKind::Call { call, .. }) => opens_root_scope(&call.func, push_roots),
                _ => false,
            })
            .collect();
        // Active root-scope locals at each block entry, one sorted set per
        // feasible path state.  Merely removing every opener block from the
        // CFG (the old implementation) made its bracket last until function
        // exit: a collecting call after `RootScope::drop` was silently read as
        // covered.  Charon's Drop place tells us exactly which guard ends.
        let mut active_at: Vec<HashSet<Vec<u64>>> = vec![HashSet::new(); n];
        if n > 0 {
            active_at[0].insert(Vec::new());
        }
        let mut scope_work = if n > 0 { vec![0usize] } else { Vec::new() };
        while let Some(b) = scope_work.pop() {
            let states: Vec<Vec<u64>> = active_at[b].iter().cloned().collect();
            let Some(term) = &terms[b] else {
                continue;
            };
            for state in states {
                let mut normal = state.clone();
                let mut unwind = state;
                match term {
                    TermKind::Call {
                        call,
                        target,
                        on_unwind,
                    } => {
                        let opens = opens_root_scope(&call.func, push_roots);
                        // Pushed rather than inserted in id order: a pin is
                        // rewound by the *innermost* live guard, because
                        // `RootScope::drop` truncates the shadow stack to the
                        // length it saved, so ownership needs nesting order.
                        if opens
                            && let Some(scope) = bare_local(&call.dest)
                            && !normal.contains(&scope)
                        {
                            normal.push(scope);
                        }
                        for (succ, next) in [(*target, normal), (*on_unwind, unwind)] {
                            if (succ as usize) < n && active_at[succ as usize].insert(next) {
                                scope_work.push(succ as usize);
                            }
                        }
                        continue;
                    }
                    // Truncated rather than removed, for the same reason the
                    // push above is a push: `RootScope::drop` truncates the
                    // shadow stack to the length the dropped guard saved, so
                    // dropping an outer guard retires every guard opened after
                    // it.  Removing only the named scope would leave the stack
                    // claiming a guard the runtime has already released.
                    TermKind::Drop { place, .. } => {
                        if let Some(scope) = place_local(place) {
                            if let Some(i) = normal.iter().position(|&s| s == scope) {
                                normal.truncate(i);
                            }
                            if let Some(i) = unwind.iter().position(|&s| s == scope) {
                                unwind.truncate(i);
                            }
                        }
                    }
                    _ => {}
                }
                for succ in successors(term) {
                    if (succ as usize) < n && active_at[succ as usize].insert(normal.clone()) {
                        scope_work.push(succ as usize);
                    }
                }
            }
        }
        let unbracketed: HashSet<usize> = (0..n)
            .filter(|&b| active_at[b].iter().any(Vec::is_empty))
            .collect();
        // The scope that owns a pin made at `b`: the innermost guard live
        // there.  `None` where the block is reached with two different scope
        // stacks, because then the owner is path-dependent and no single
        // answer is right.
        let owner_at: Vec<Option<u64>> = (0..n)
            .map(|b| {
                let mut it = active_at[b].iter();
                let first = it.next()?;
                let innermost = *first.last()?;
                for other in it {
                    if other.last() != Some(&innermost) {
                        return None;
                    }
                }
                Some(innermost)
            })
            .collect();
        // The one scope stack live at `b`, innermost last.  `None` where the
        // block is reached with two different stacks.
        let stack_at: Vec<Option<Vec<u64>>> = (0..n)
            .map(|b| {
                let mut it = active_at[b].iter();
                let first = it.next()?;
                if it.next().is_some() {
                    return None;
                }
                Some(first.clone())
            })
            .collect();
        // The scope local a scope-closing Drop names, so the dataflow can
        // retire that guard's pins and leave an enclosing guard's alone.
        let closed_scope: Vec<Option<u64>> = (0..n)
            .map(|b| match &terms[b] {
                Some(TermKind::Drop { place, .. }) => place_local(place)
                    .filter(|scope| active_at[b].iter().any(|a| a.contains(scope))),
                _ => None,
            })
            .collect();
        let has_nested_scopes = active_at
            .iter()
            .flat_map(HashSet::iter)
            .any(|scopes| scopes.len() > 1);
        let term_closes_root_scope: Vec<bool> = (0..n)
            .map(|b| match &terms[b] {
                Some(TermKind::Drop { place, .. }) => place_local(place)
                    .is_some_and(|scope| active_at[b].iter().any(|a| a.contains(&scope))),
                _ => false,
            })
            .collect();
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
        let PinAssignIndex {
            defs,
            mut_borrow_of,
            ..
        } = index_pin_assigns(&body.body, &terms);

        // A body whose pinned set cannot be read is not a body with an empty
        // one: grading it would turn "not understood" into "root missing".
        // The pin-set analysis below carries the owning guard alongside each
        // root local, so an inner Drop retires its own pins and leaves an
        // enclosing guard's alone; nesting on its own no longer withholds a
        // body.  This counts how many still hold two live scopes, because a
        // body that also trips one of the opacity tests below is unread for
        // that reason and the two populations are easy to confuse.
        if has_nested_scopes {
            stats.bodies_with_nested_scopes += 1;
        }
        let mut opaque_reason: Option<&'static str> = if unparsed_terms {
            Some("unparsed-terminator")
        } else if unparsed_stmts {
            Some("unparsed-statement")
        } else {
            None
        };
        let mut saw_pin_call = false;
        let mut term_pins: Vec<HashSet<u64>> = vec![HashSet::new(); n];
        // The locals a pin was *handed*, as distinct from the word it hands
        // back.  `pin_root` returns the normalized word because the publish is
        // itself a safepoint -- a foreign collection can forward the value
        // between the caller's copy and the query -- so a body that goes on
        // reading the local it passed in is reading a possible forwarding stub.
        let mut term_pin_args: Vec<HashSet<u64>> = vec![HashSet::new(); n];
        let mut term_pin_names: Vec<String> = vec![String::new(); n];
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
            let Some(name) = cg.names.get(id) else {
                continue;
            };
            if name.contains("gc_roots::<Impl>") && name.ends_with("::set") {
                // These overwrite an existing coloured slot.  Without a
                // slot→root map, retaining the old root or replacing the
                // wrong one could both claim false coverage.
                opaque_reason.get_or_insert("slot-set");
                continue;
            }
            // Reading a slot back yields the word the slot holds now, which
            // is rooted by whatever pinned it; the index it takes is not a
            // root, so only the result is read here.
            let reads_a_slot_back = reads_root_slot(name);
            if !is_pin_fn(name) && !reads_a_slot_back {
                // A helper pins into this body's open scope. Its own fresh
                // object names no local here, so an empty set is not a pin
                // we failed to read.
                let Some(helper) = pin_helpers.get(id) else {
                    continue;
                };
                saw_pin_call = true;
                let mut pinned: HashSet<u64> = HashSet::new();
                for (i, arg) in call.args.iter().enumerate() {
                    if !helper.pinned_params.contains(&i) {
                        continue;
                    }
                    let mut seed: HashSet<u64> = HashSet::new();
                    use_operand(arg, &mut seed);
                    for local in seed {
                        chase_pinned(local, &defs, &mut pinned, 0);
                    }
                }
                let mut args_only = pinned.clone();
                if let Some(dest) = bare_local(&call.dest) {
                    if helper.returns_pinned {
                        pinned.insert(dest);
                    }
                    // `let obj = helper(obj)` rebinds. The destination is the
                    // word handed back only when the helper returns one;
                    // either way it is not an argument still being read.
                    args_only.remove(&dest);
                }
                args_only.retain(|local| gc_locals.contains_key(local));
                term_pin_args[b] = args_only;
                term_pin_names[b] = name.clone();
                pinned.retain(|local| gc_locals.contains_key(local));
                if pinned.is_empty() && !helper.pinned_params.is_empty() {
                    opaque_reason.get_or_insert("pin-names-nothing");
                }
                term_pins[b] = pinned;
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
            let mut args_only = pinned.clone();
            if let Some(d) = bare_local(&call.dest) {
                pinned.insert(d);
                args_only.remove(&d);
            }
            args_only.retain(|l| gc_locals.contains_key(l));
            term_pin_args[b] = args_only;
            term_pin_names[b] = name.clone();
            pinned.retain(|l| gc_locals.contains_key(l));
            if pinned.is_empty() {
                // A pin that named nothing we could resolve is a pin we do not
                // understand, not one that pinned nothing.
                opaque_reason.get_or_insert("pin-names-nothing");
            }
            term_pins[b] = pinned;
        }
        // A scope-closing Drop this reader cannot name retires an unknown set,
        // and a pin made where two scope stacks meet has a path-dependent
        // owner.  Either would let one guard's Drop release another's pins.
        // Blocks that neither pin nor close a scope need no owner.
        if (0..n).any(|b| {
            (term_closes_root_scope[b] && (closed_scope[b].is_none() || stack_at[b].is_none()))
                || (!term_pins[b].is_empty() && owner_at[b].is_none())
        }) {
            opaque_reason.get_or_insert("scope-owner-ambiguous");
        }
        if !bracket_blocks.is_empty() && !saw_pin_call {
            // A scope is open and nothing in this body names what went into
            // it. A helper that pins into this scope was already recorded
            // above; what remains opens its own scope, as `RootedItems` does.
            // An unread set, not an empty one.
            opaque_reason.get_or_insert("no-pin-in-body");
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
        // Each element carries the guard that owns it, so a Drop retires that
        // guard's pins and leaves an enclosing guard's in place.  The owner is
        // the innermost scope live where the pin was made.
        let all_scopes: HashSet<u64> = active_at
            .iter()
            .flat_map(HashSet::iter)
            .flat_map(|a| a.iter().copied())
            .collect();
        // The element is a subset of `locals x scopes` and the entry value is
        // that whole product, so a `HashSet` of pairs per block costs
        // `blocks * locals * scopes` hash entries -- for the largest
        // interpreter bodies, gigabytes, three times over per round because
        // each round rebuilt both block-indexed vectors.  The same answer is
        // `blocks * locals * scopes` **bits**.
        //
        // Bit `l * scopes.len() + s`, so one local's owners are contiguous:
        // clearing every owner of a local is what a kill and a call
        // destination each do, while retiring a scope is rarer and takes a
        // strided mask built once.
        let mut scopes: Vec<u64> = all_scopes.iter().copied().collect();
        scopes.sort_unstable();
        // The domain carries every local a pin *names*, not only the GC ones:
        // a terminator adds `(local, owner)` for whatever it pinned.  Only the
        // GC locals seed the entry value, which is what starting from
        // `gc_locals x scopes` meant.
        let mut locals: Vec<u64> = gc_locals.keys().copied().collect();
        for pins in &term_pins {
            locals.extend(pins.iter().copied());
        }
        locals.sort_unstable();
        locals.dedup();
        let local_at: HashMap<u64, usize> =
            locals.iter().enumerate().map(|(i, &l)| (l, i)).collect();
        let scope_at: HashMap<u64, usize> =
            scopes.iter().enumerate().map(|(i, &x)| (x, i)).collect();
        let nscopes = scopes.len();
        let words = (locals.len() * nscopes).div_ceil(64);
        let bit_set = |v: &mut [u64], i: usize| v[i / 64] |= 1u64 << (i % 64);
        let bit_get = |v: &[u64], i: usize| v[i / 64] >> (i % 64) & 1 == 1;
        // One scope's pairs, which are strided and so need a mask; a local's
        // are a run and are cleared in place.
        let scope_masks: Vec<Vec<u64>> = (0..nscopes)
            .map(|si| {
                let mut m = vec![0u64; words];
                for li in 0..locals.len() {
                    bit_set(&mut m, li * nscopes + si);
                }
                m
            })
            .collect();
        // What each block does to the set, resolved once so the rounds below
        // are word operations: the pairs it clears by local, the owners it
        // retires, and the pairs its terminator adds.  Clears commute, so the
        // call destination joins the statement kills rather than taking a
        // pass of its own.
        struct BlockEdit {
            kill_locals: Vec<usize>,
            retire_scopes: Vec<usize>,
            retire_all: bool,
            add_bits: Vec<usize>,
        }
        let edits: Vec<BlockEdit> = (0..n)
            .map(|b| {
                let mut kill_locals: Vec<usize> = stmt_kills[b]
                    .iter()
                    .filter_map(|l| local_at.get(l).copied())
                    .collect();
                let mut retire_scopes: Vec<usize> = Vec::new();
                let mut retire_all = false;
                if term_closes_root_scope[b] {
                    // `RootScope::drop` truncates the shadow stack to the
                    // length it saved, so a guard takes its own pins *and*
                    // those of every guard opened after it -- dropping out of
                    // order takes more than one scope's worth.  An enclosing
                    // guard keeps its own.
                    match (closed_scope[b], stack_at[b].as_ref()) {
                        (Some(scope), Some(stack)) => {
                            let from = stack.iter().position(|&x| x == scope).unwrap_or(0);
                            retire_scopes = stack[from..]
                                .iter()
                                .filter_map(|x| scope_at.get(x).copied())
                                .collect();
                        }
                        // A Drop this reader cannot place retires an unknown
                        // set.
                        _ => retire_all = true,
                    }
                }
                if let Some(TermKind::Call { call, .. }) = &terms[b]
                    && let Some(d) = bare_local(&call.dest)
                    && let Some(&di) = local_at.get(&d)
                {
                    kill_locals.push(di);
                }
                let add_bits = match owner_at[b].and_then(|o| scope_at.get(&o).copied()) {
                    Some(si) => term_pins[b]
                        .iter()
                        .filter_map(|l| local_at.get(l).map(|&li| li * nscopes + si))
                        .collect(),
                    None => Vec::new(),
                };
                BlockEdit {
                    kill_locals,
                    retire_scopes,
                    retire_all,
                    add_bits,
                }
            })
            .collect();
        // Block-indexed buffers, allocated once rather than per round.
        let mut pinned_in = vec![0u64; n * words];
        for b in 0..n {
            if b == 0 || !reachable.contains(&b) {
                continue;
            }
            let blk = &mut pinned_in[b * words..(b + 1) * words];
            for (li, l) in locals.iter().enumerate() {
                if !gc_locals.contains_key(l) {
                    continue;
                }
                for si in 0..nscopes {
                    bit_set(blk, li * nscopes + si);
                }
            }
        }
        let mut outs = vec![0u64; n * words];
        let mut next = vec![0u64; n * words];
        for _round in 0..n + 8 {
            for b in 0..n {
                let at = b * words;
                outs[at..at + words].copy_from_slice(&pinned_in[at..at + words]);
                let e = &edits[b];
                let o = &mut outs[at..at + words];
                for &li in &e.kill_locals {
                    for si in 0..nscopes {
                        let i = li * nscopes + si;
                        o[i / 64] &= !(1u64 << (i % 64));
                    }
                }
                if e.retire_all {
                    o.fill(0);
                } else {
                    for &si in &e.retire_scopes {
                        for (w, m) in o.iter_mut().zip(&scope_masks[si]) {
                            *w &= !m;
                        }
                    }
                }
                for &i in &e.add_bits {
                    bit_set(o, i);
                }
            }
            let mut changed = false;
            for b in 0..n {
                let at = b * words;
                if b == 0 || !reachable.contains(&b) {
                    next[at..at + words].fill(0);
                } else if let Some(&first) = preds[b].first() {
                    let src = first * words;
                    for w in 0..words {
                        next[at + w] = outs[src + w];
                    }
                    for &p in preds[b].iter().skip(1) {
                        let src = p * words;
                        for w in 0..words {
                            next[at + w] &= outs[src + w];
                        }
                    }
                } else {
                    next[at..at + words].fill(0);
                }
                if next[at..at + words] != pinned_in[at..at + words] {
                    changed = true;
                }
            }
            std::mem::swap(&mut pinned_in, &mut next);
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
                transfer_term(t, &mut live, &metadata_fns);
                for st in body.body[b].statements.iter().rev() {
                    if let Ok(k) = st.stmt_kind() {
                        transfer_stmt(&k, &mut live, &gc_locals);
                    }
                }
                live.retain(|l| gc_locals.contains_key(l));
                if live != live_in[b] {
                    live_in[b] = live;
                    changed = true;
                }
            }
        }

        // Locals this body hands to a callee that addresses a nursery-allocated
        // kind.  Built once: it does not depend on which collecting call is
        // being judged, so a local that reaches such a callee anywhere in the
        // body is in the set for every call the body makes.
        let mut movable_args: HashSet<u64> = HashSet::new();
        for other in &body.body {
            let Ok(TermKind::Call { call: c2, .. }) = other.term(llbc) else {
                continue;
            };
            let CallFunc::Regular(r2) = &c2.func else {
                continue;
            };
            let CallKind::Fun(FunId::Regular { id: cid }) = &r2.kind else {
                continue;
            };
            if !movable_callees.contains(cid) {
                continue;
            }
            for a in &c2.args {
                let mut seed: HashSet<u64> = HashSet::new();
                use_operand(a, &mut seed);
                for l in seed {
                    chase_arg_aliases(l, &defs, &mut movable_args, 0);
                }
            }
        }
        // Count only the GC pointers: a `Wtf8Buf` or a capacity handed to a
        // marked callee is an argument the `movable` filters can never select,
        // so counting it here would report a live ranking that is not one.
        let movable_gc_args = movable_args
            .iter()
            .filter(|l| gc_locals.contains_key(*l))
            .count();
        if movable_gc_args > 0 {
            stats.bodies_with_movable_args += 1;
            stats.movable_arg_locals += movable_gc_args;
        }

        // Every pin whose argument the body goes on to read.  Independent of
        // the bracket question above: this is not a missing root but a stale
        // *word*, and the pin call is itself where the forwarding could have
        // happened.  `#[must_use]` on `pin_root` puts the choice in the open --
        // use the returned word, or write `let _ =` and thereby claim the kind
        // never moves.  That claim is what this checks.
        for b in 0..n {
            if term_pin_args[b].is_empty() {
                continue;
            }
            let Some(TermKind::Call {
                target, on_unwind, ..
            }) = &terms[b]
            else {
                continue;
            };
            let mut after: HashSet<u64> = HashSet::new();
            for s in [*target, *on_unwind] {
                if let Some(sl) = live_in.get(s as usize) {
                    after.extend(sl.iter().copied());
                }
            }
            let still: Vec<u64> = term_pin_args[b]
                .iter()
                .filter(|l| after.contains(l))
                .copied()
                .collect();
            if still.is_empty() {
                continue;
            }
            stats.pin_arg_read_after += 1;
            let mut movable: Vec<String> = still
                .iter()
                .filter(|l| movable_args.contains(l))
                .map(|l| gc_locals[l].clone())
                .collect();
            if !movable.is_empty() {
                stats.pin_arg_read_after_movable += 1;
            }
            let mut locals: Vec<String> = still.iter().map(|l| gc_locals[l].clone()).collect();
            locals.sort();
            movable.sort();
            let at = term_span(llbc, &body.body[b].terminator.span, &fd.item_meta.span);
            stats.stale_pin_reads.push(StalePinRead {
                func_name: fd.item_meta.name_path(),
                file: llbc.file_path(at.file_id).unwrap_or_default().to_string(),
                line: at.beg.line,
                pin_name: term_pin_names[b].clone(),
                locals,
                movable,
            });
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
            for a in &call.args {
                let mut used: HashSet<u64> = HashSet::new();
                use_operand(a, &mut used);
                for t in used {
                    if let Some(l) = mut_borrow_of.get(&t) {
                        after.remove(l);
                    }
                }
            }
            after.retain(|l| gc_locals.contains_key(l));
            // One span answers every column below, and it is the
            // terminator's own: the call being reported *is* the terminator,
            // so the block's last statement names a line that runs after it.
            // A terminator with no span at all falls back to the function's.
            let at = term_span(llbc, &bb.terminator.span, &fd.item_meta.span);
            if !unbracketed.contains(&b) {
                stats.withheld_under_a_bracket += 1;
                // Withholding the call is right -- a bracket does dominate it
                // -- but "a bracket is open" and "this root is in it" are two
                // questions, and only the first was ever asked.  Grade the
                // second here so a bracket that pins the wrong set stops
                // reading as coverage.
                if opaque_reason.is_some() || !reachable.contains(&b) {
                    stats.withheld_contents_opaque += 1;
                    let reason = opaque_reason.unwrap_or("unreachable-block");
                    *stats.opaque_by_reason.entry(reason).or_default() += 1;
                    let fname = fd.item_meta.name_path();
                    match stats.opaque_bodies.last_mut() {
                        Some(last) if last.0 == fname => last.3 += 1,
                        _ => stats.opaque_bodies.push((
                            fname,
                            llbc.file_path(at.file_id).unwrap_or_default().to_string(),
                            reason,
                            1,
                        )),
                    }
                    if has_nested_scopes {
                        stats.withheld_opaque_from_nested += 1;
                    }
                    continue;
                }
                // Drop the owner here: coverage asks only whether the local
                // is pinned by *some* live guard.
                let blk = &pinned_in[b * words..(b + 1) * words];
                let held: HashSet<u64> = locals
                    .iter()
                    .enumerate()
                    .filter(|(li, l)| {
                        !stmt_kills[b].contains(l)
                            && (0..nscopes).any(|si| bit_get(blk, li * nscopes + si))
                    })
                    .map(|(_, l)| *l)
                    .collect();
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
            // Does any live pointer reach a movable-addressing callee in this
            // body?  Anywhere in the body, not only in the dominated
            // successors — this is a ranking signal, not a proof.
            let mut movable_use: Vec<String> = after
                .iter()
                .filter(|l| movable_args.contains(l))
                .map(|l| gc_locals[l].clone())
                .collect();
            movable_use.sort();
            // One span answers both columns, and it is the terminator's
            // own: the call being reported *is* the terminator, so the
            // block's last statement names a line that runs after it.
            // A terminator with no span at all falls back to the function's.
            let at = term_span(llbc, &bb.terminator.span, &fd.item_meta.span);
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

fn transfer_stmt(k: &StmtKind, live: &mut HashSet<u64>, tracked: &HashMap<u64, String>) {
    match k {
        StmtKind::Assign(p, r) => {
            if let Some(d) = bare_local(p) {
                // `_t = &*args` / `_t = copy x` into a tracked local reads `x`
                // only if `_t` is read later: the reborrow a `args.len()` call
                // takes is dead once the length is out.  Only for a tracked
                // destination -- an untracked one is never in `live`, so its
                // later reads are invisible here.
                let pure = matches!(r, Rvalue::Use(..) | Rvalue::Ref { .. } | Rvalue::Cast(..));
                let was_live = live.remove(&d);
                if pure && tracked.contains_key(&d) && !was_live {
                    return;
                }
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
        StmtKind::Borrowck(_) | StmtKind::Unknown => {}
    }
}

fn term_span<'a>(
    llbc: &'a majit_charon_reader::Llbc,
    term: &'a Option<majit_charon_reader::ullbc::SpanRef>,
    item: &'a majit_charon_reader::ullbc::SpanRef,
) -> &'a majit_charon_reader::ullbc::SpanData {
    let chosen = match term {
        Some(span) => span,
        None => item,
    };
    llbc.span_data(chosen)
        .expect("span id is not in the artefact span table")
}

fn transfer_term(t: &TermKind, live: &mut HashSet<u64>, metadata_fns: &HashSet<u64>) {
    match t {
        TermKind::Call { call, .. } => {
            if let Some(d) = bare_local(&call.dest) {
                live.remove(&d);
            } else if let Some(l) = place_local(&call.dest) {
                live.insert(l);
            }
            let metadata_only = matches!(
                &call.func,
                CallFunc::Regular(reg)
                    if matches!(&reg.kind, CallKind::Fun(FunId::Regular { id }) if metadata_fns.contains(id))
            );
            if metadata_only {
                return;
            }
            for a in &call.args {
                use_operand(a, live);
            }
        }
        TermKind::Switch { discr, .. } => use_operand(discr, live),
        TermKind::Assert { assert, .. } => use_operand(&assert.cond, live),
        TermKind::Drop { place, .. } => {
            if let Some(l) = place_local(place) {
                live.insert(l);
            }
        }
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
        defs.insert(
            1,
            pin_src(&Rvalue::Use(mv(0), serde_json::Value::Null)).expect("a use is an alias"),
        );
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

    fn helper_fact(
        arg_count: u64,
        has_push_roots: bool,
        calls: Vec<(&str, u64, Vec<Vec<u64>>)>,
    ) -> HelperBodyFact {
        HelperBodyFact {
            has_push_roots,
            arg_count,
            assigned: HashSet::new(),
            defs: HashMap::new(),
            pin_result_locals: HashSet::new(),
            calls: calls
                .into_iter()
                .map(|(name, callee, arg_locals)| HelperCallFact {
                    callee,
                    callee_name: name.to_string(),
                    arg_locals,
                })
                .collect(),
            block_calls: Vec::new(),
            successors: Vec::new(),
        }
    }

    fn summary_of(bodies: HashMap<u64, HelperBodyFact>, id: u64) -> Option<PinHelperSummary> {
        summarize_pin_helpers(&bodies).remove(&id)
    }

    /// A pin on only one arm is not a must-pin: the caller scan would
    /// otherwise treat the skipped arm as bracketed.
    #[test]
    fn a_helper_that_pins_on_one_arm_names_no_parameter() {
        let mut body = helper_fact(
            1,
            false,
            vec![("pyre_object::gc_roots::pin_root", 9, vec![vec![1]])],
        );
        // block 0 switches to a pin arm and a bare return.
        body.block_calls = vec![Vec::new(), vec![0], Vec::new()];
        body.successors = vec![vec![1, 2], Vec::new(), Vec::new()];
        let bodies = HashMap::from([(1, body)]);
        assert_eq!(
            summary_of(bodies, 1),
            Some(PinHelperSummary {
                pinned_params: HashSet::new(),
                returns_pinned: false,
            })
        );
    }

    /// `pin_root(param)` — parameter local 1 is argument position 0.
    #[test]
    fn a_helper_that_pins_its_first_parameter_names_that_position() {
        let bodies = HashMap::from([(
            1,
            helper_fact(
                1,
                false,
                vec![("pyre_object::gc_roots::pin_root", 9, vec![vec![1]])],
            ),
        )]);
        assert_eq!(
            summary_of(bodies, 1),
            Some(PinHelperSummary {
                pinned_params: HashSet::from([0]),
                returns_pinned: false,
            })
        );
    }

    /// `pin_root(fresh)` — the pinned local is not a parameter.
    #[test]
    fn a_helper_that_pins_only_a_fresh_value_names_no_parameter() {
        let bodies = HashMap::from([(
            1,
            helper_fact(
                1,
                false,
                vec![("pyre_object::gc_roots::pin_root", 9, vec![vec![5]])],
            ),
        )]);
        assert_eq!(
            summary_of(bodies, 1),
            Some(PinHelperSummary {
                pinned_params: HashSet::new(),
                returns_pinned: false,
            })
        );
    }

    /// A pin into a nested owned scope is not a pin into the caller.
    #[test]
    fn a_pin_caller_that_opens_a_scope_through_another_function_is_not_a_helper() {
        let mut bodies = HashMap::new();
        bodies.insert(
            1,
            helper_fact(
                1,
                true,
                vec![("pyre_object::gc_roots::pin_root", 9, vec![vec![1]])],
            ),
        );
        bodies.insert(
            2,
            helper_fact(
                1,
                false,
                vec![
                    ("pyre_object::gc_roots::pin_root", 9, vec![vec![1]]),
                    ("nested_opener", 1, vec![vec![1]]),
                ],
            ),
        );
        let sums = summarize_pin_helpers(&bodies);
        assert!(!sums.contains_key(&1));
        assert!(!sums.contains_key(&2));
    }

    /// `RootedItems::new` opens its own scope. It is not a helper, and neither
    /// is a function whose only pin goes through it.
    #[test]
    fn a_function_that_opens_its_own_root_scope_is_not_a_helper() {
        let mut bodies = HashMap::new();
        bodies.insert(
            1,
            helper_fact(
                1,
                true,
                vec![("pyre_object::gc_roots::pin_root", 9, vec![vec![1]])],
            ),
        );
        bodies.insert(
            2,
            helper_fact(
                1,
                false,
                vec![("pyre_object::gc_roots::RootedItems::new", 1, vec![vec![1]])],
            ),
        );
        let sums = summarize_pin_helpers(&bodies);
        assert!(!sums.contains_key(&1));
        assert!(!sums.contains_key(&2));
    }

    /// `outer` hands local 3, an alias of its parameter, to `inner`'s pinned
    /// position. The other argument is not pinned.
    #[test]
    fn a_nested_helper_pins_the_parameter_its_caller_handed_over() {
        let mut outer = helper_fact(
            2,
            false,
            vec![(
                "pyre_interpreter::builtins::pin_into_caller",
                1,
                vec![vec![3], vec![2]],
            )],
        );
        outer.defs.insert(3, PinSrc::Alias(1));
        let inner = helper_fact(
            1,
            false,
            vec![("pyre_object::gc_roots::pin_root", 9, vec![vec![1]])],
        );
        let mut bodies = HashMap::new();
        bodies.insert(1, inner);
        bodies.insert(2, outer);
        let sums = summarize_pin_helpers(&bodies);
        assert_eq!(
            sums.get(&1).map(|summary| summary.pinned_params.clone()),
            Some(HashSet::from([0]))
        );
        assert_eq!(
            sums.get(&2),
            Some(&PinHelperSummary {
                pinned_params: HashSet::from([0]),
                returns_pinned: false,
            })
        );
    }

    /// The return place is pinned when it aliases a pin result, and not when
    /// the assignment is an aggregate.
    #[test]
    fn the_return_place_is_pinned_only_when_it_aliases_a_pin_result() {
        let mut aliased = helper_fact(
            1,
            false,
            vec![("pyre_object::gc_roots::pin_root", 9, vec![vec![1]])],
        );
        aliased.pin_result_locals.insert(2);
        aliased.defs.insert(0, PinSrc::Alias(2));
        let mut aggregate = helper_fact(
            0,
            false,
            vec![("pyre_object::gc_roots::pin_root", 9, vec![vec![4]])],
        );
        aggregate.pin_result_locals.insert(2);
        aggregate.defs.insert(0, PinSrc::Aggregate(vec![2]));
        let mut slot = helper_fact(
            0,
            false,
            vec![
                ("pyre_object::gc_roots::pin_root", 9, vec![vec![4]]),
                ("pyre_object::gc_roots::shadow_stack_get", 8, vec![vec![3]]),
            ],
        );
        slot.pin_result_locals.insert(0);
        let bodies = HashMap::from([(1, aliased), (2, aggregate), (3, slot)]);
        let sums = summarize_pin_helpers(&bodies);
        assert!(sums[&1].returns_pinned);
        assert!(!sums[&2].returns_pinned);
        assert!(sums[&3].returns_pinned);
        assert!(sums[&2].pinned_params.is_empty());
    }

    /// A collection between the pin and the return drops `returns_pinned`.
    #[test]
    fn a_collecting_call_after_a_pin_is_not_a_pinned_return() {
        let mut body = helper_fact(
            1,
            false,
            vec![
                ("pyre_object::gc_roots::pin_root", 9, vec![vec![1]]),
                ("gc_hook::try_gc_collect", 11, vec![]),
            ],
        );
        body.pin_result_locals.insert(0);
        body.defs.insert(0, PinSrc::Alias(2));
        body.pin_result_locals.insert(2);
        let bodies = HashMap::from([(1, body)]);
        let sums = summarize_pin_helpers(&bodies);
        assert!(!sums[&1].returns_pinned);
    }

    /// The call-graph prefilter keeps a pin-caller and its non-bracketing
    /// caller, and drops a function that opens a scope.
    #[test]
    fn candidate_ids_skip_a_function_that_opens_a_root_scope() {
        let mut callees: HashMap<u64, HashSet<u64>> = HashMap::new();
        let mut names = HashMap::new();
        callees.insert(1, HashSet::from([9]));
        names.insert(1, "helper".into());
        names.insert(9, "pyre_object::gc_roots::pin_root".into());
        callees.insert(2, HashSet::from([7, 9]));
        names.insert(2, "bracket".into());
        names.insert(7, "pyre_object::gc_roots::push_roots".into());
        callees.insert(3, HashSet::from([1]));
        names.insert(3, "outer".into());
        callees.insert(4, HashSet::from([2]));
        names.insert(4, "caller_of_bracket".into());
        callees.insert(5, HashSet::from([2, 9]));
        names.insert(5, "pin_and_open".into());
        let ids = helper_candidate_ids(&callees, &names, &HashSet::from([7]));
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
        assert!(!ids.contains(&2));
        assert!(!ids.contains(&4));
        assert!(!ids.contains(&5));
        assert!(!ids.contains(&9));
    }
}
