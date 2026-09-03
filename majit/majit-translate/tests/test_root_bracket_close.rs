//! A lowered root bracket closes, in the crate that owns the guard and in a
//! crate that only imports it.
//!
//! The `Drop` of a `RootScope` is the bracket's `pop_roots`: it rewinds the
//! shadow stack to the length the matching `push_roots` captured. A lowering
//! that forwards past that terminator leaves the bracket open, and the slots
//! it pinned stay reachable for the life of the thread — every later
//! collection walks them, so a loop that opens one per iteration turns a
//! constant root set into a growing one.
//!
//! The close is spelled as a call taking the guard by reference rather than as
//! two field reads and a truncate, because a crate that imports `RootScope`
//! sees an opaque stub with no fields. Reading the fields would confine the
//! close to `pyre-object`, which is why `pyre_interpreter` is checked here and
//! not just `pyre_object`.
//!
//! Two things make the close conditional on more than the `Drop` arm existing,
//! and each has a fixture below. The guard has to reach the dropping block:
//! liveness counts a `Drop` place as a use only because the close reads it, and
//! without that the binding survives only when the dropping block happens to be
//! lowered right after the defining one. And the guard has to be one this body
//! still owns: a drop the artefact stamps `Conditional` runs under an
//! initialisation flag it does not carry, so a guard the body moves out of
//! keeps its bracket open rather than rewinding a stack its new owner holds.
//!
//! A third case owes no close at all. `erased_root_bracket_guards` names the
//! brackets the lowering takes out of the jitcode entirely: those open nothing
//! and pin nothing, so there is no shadow stack to rewind. Every assertion here
//! is stated over the brackets that survive that pass, which is why the
//! fixtures below are bodies whose brackets it keeps.

use majit_charon_reader::Llbc;
use majit_charon_reader::ullbc::{PlaceKind, SwitchTargets, TermKind, TyRef, Unstructured};
use majit_translate::front::mir::{erased_root_bracket_guards, lower_fun_decl};
use majit_translate::model::{CallTarget, FunctionGraph, OpKind};
use std::sync::OnceLock;

const OBJECT_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/llbc/pyre-object.ullbc"
);
const INTERPRETER_LLBC: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../build/llbc/pyre-interpreter.ullbc"
);

/// `None` means the artefact is absent, which degrades the tests to a skip
/// rather than a failure on a tree that has not run the extraction.
fn llbc(path: &'static str, slot: &'static OnceLock<Option<Llbc>>) -> Option<&'static Llbc> {
    slot.get_or_init(|| {
        if !std::path::Path::new(path).is_file() {
            eprintln!("skipping: {path} is missing; run `python3 scripts/extract-llbc.py`");
            return None;
        }
        Some(Llbc::load(path).expect("load llbc"))
    })
    .as_ref()
}

fn object_llbc() -> Option<&'static Llbc> {
    static LLBC: OnceLock<Option<Llbc>> = OnceLock::new();
    llbc(OBJECT_LLBC, &LLBC)
}

fn interpreter_llbc() -> Option<&'static Llbc> {
    static LLBC: OnceLock<Option<Llbc>> = OnceLock::new();
    llbc(INTERPRETER_LLBC, &LLBC)
}

fn lower_named(llbc: &Llbc, leaf: &str) -> FunctionGraph {
    let suffix = format!("::{leaf}");
    let fd = llbc
        .iter_local_fns()
        .find(|fd| fd.item_meta.name_path().ends_with(&suffix))
        .unwrap_or_else(|| panic!("{leaf} present in the shipped LLBC"));
    lower_fun_decl(llbc, fd).unwrap_or_else(|e| panic!("lower {leaf}: {e:?}"))
}

/// Count calls whose path ends with `leaf`, over every block of the graph.
fn calls_to(graph: &FunctionGraph, leaf: &str) -> usize {
    graph
        .blocks
        .iter()
        .flat_map(|b| b.operations.iter())
        .filter(|op| match &op.kind {
            OpKind::Call {
                target: CallTarget::FunctionPath { segments },
                ..
            } => segments.last().is_some_and(|s| s == leaf),
            _ => false,
        })
        .count()
}

fn assert_bracket_closes(llbc: &Llbc, leaf: &str) {
    let graph = lower_named(llbc, leaf);
    let opened = calls_to(&graph, "push_roots");
    let closed = calls_to(&graph, "root_scope_close");
    assert!(
        opened > 0,
        "{leaf} is the fixture for a lowered root bracket, but it opens none"
    );
    assert!(
        closed > 0,
        "{leaf} opens {opened} root bracket(s) and closes none: every slot the \
         bracket pins stays reachable for the life of the thread"
    );
}

#[test]
fn bracket_closes_in_the_crate_that_owns_the_guard() {
    let Some(llbc) = object_llbc() else { return };
    assert_bracket_closes(llbc, "w_tuple_items_copy_as_vec");
}

#[test]
fn bracket_closes_in_a_crate_that_only_imports_the_guard() {
    let Some(llbc) = interpreter_llbc() else {
        return;
    };
    assert_bracket_closes(llbc, "call_function_impl_result");
}

/// The bracket a descended `lib.abs(x)` inlines into its trace.
#[test]
fn bracket_closes_in_the_cffi_call_path() {
    let Some(llbc) = interpreter_llbc() else {
        return;
    };
    assert_bracket_closes(llbc, "do_call");
}

/// A body whose guard is dropped several blocks away from where it is bound.
/// The binding reaches that block only through the drop block's `inputargs`,
/// which liveness supplies only because a `Drop` place counts as a use.
///
/// Stated over every body that has such a drop rather than against named
/// fixtures: the shapes the erasure keeps are not the ones it keeps tomorrow,
/// and a fixture list that goes empty asserts nothing while still passing.
#[test]
fn bracket_closes_when_the_drop_is_not_adjacent_to_the_binding() {
    let Some(llbc) = object_llbc() else { return };
    let mut bodies = 0usize;
    for fd in llbc.iter_local_fns() {
        let Some(body) = fd.unstructured() else {
            continue;
        };
        let erased = erased_root_bracket_guards(llbc, &body);
        let moved = moved_out_locals(&body);
        let opens = opener_blocks(llbc, &body);
        let distant = guard_drop_sites(llbc, &body).any(|(bb, local)| {
            !moved.contains(&local)
                && !erased.contains(&(local as usize))
                && opens.get(&local).is_some_and(|open| *open != bb)
        });
        if !distant {
            continue;
        }
        bodies += 1;
        let Ok(graph) = lower_fun_decl(llbc, fd) else {
            continue;
        };
        assert!(
            reachable_closes(&graph) > 0,
            "{} drops a surviving guard in a block that does not bind it and              lowers no close",
            fd.item_meta.name_path()
        );
    }
    assert!(
        bodies > 10,
        "only {bodies} bodies drop a surviving guard away from its binding; the          fixture population is too small to prove anything"
    );
}

/// `local -> block` for each guard a `push_roots` call binds in this body.
fn opener_blocks(llbc: &Llbc, body: &Unstructured) -> std::collections::HashMap<u64, usize> {
    let mut out = std::collections::HashMap::new();
    for (i, bb) in body.body.iter().enumerate() {
        let Ok(TermKind::Call { call, .. }) = bb.term() else {
            continue;
        };
        let PlaceKind::Local(local) = call.dest.kind else {
            continue;
        };
        if ty_is_root_scope(llbc, &call.dest.ty) {
            out.insert(local, i);
        }
    }
    out
}

/// A bracket left open must belong to a guard the body moved out of, or to one
/// the erasure took out of the jitcode.
///
/// From the move onward the bracket is the new owner's, and rewinding it at
/// the moved-from local's `Drop` would truncate a shadow stack that owner is
/// still using — `DictOperationGuard::new` pins before it moves, so those pins
/// are what the rewind would drop. Charon stamps such a drop `Conditional`:
/// the destructor runs only if the place still holds a value, and the flag
/// that decides it is not in the artefact. "This body never moves the guard"
/// is the only proof of definite initialisation available, so a guard without
/// it keeps its bracket open.
///
/// An erased guard is the other way a drop lowers no close, and it is not a
/// retention: the bracket publishes nothing, so there is no slot to rewind.
///
/// Stated over every body that drops a guard rather than against named
/// fixtures: any other function that stops closing is the regression this
/// guards against, whatever it is called.
#[test]
fn only_a_moved_out_or_erased_guard_keeps_its_bracket_open() {
    let Some(llbc) = object_llbc() else { return };
    let mut dropping = 0usize;
    let mut left_open: Vec<String> = Vec::new();
    let mut moved_out = 0usize;
    for fd in llbc.iter_local_fns() {
        let Some(body) = fd.unstructured() else {
            continue;
        };
        if guard_drop_sites(llbc, &body).next().is_none() {
            continue;
        }
        dropping += 1;
        let Ok(graph) = lower_fun_decl(llbc, fd) else {
            continue;
        };
        if calls_to(&graph, "root_scope_close") > 0 {
            continue;
        }
        let name = fd.item_meta.name_path().to_string();
        let moved = moved_out_locals(&body);
        let erased = erased_root_bracket_guards(llbc, &body);
        let excused = guard_drop_sites(llbc, &body)
            .any(|(_, local)| moved.contains(&local) || erased.contains(&(local as usize)));
        assert!(
            excused,
            "{name} drops a root-bracket guard, lowers no close for it, and neither \
             moves the guard out nor has it erased: every slot the bracket pins \
             stays reachable for the life of the thread"
        );
        if guard_drop_sites(llbc, &body).any(|(_, local)| moved.contains(&local)) {
            moved_out += 1;
        }
        left_open.push(name);
    }
    assert!(
        dropping > 100,
        "only {dropping} bodies drop a guard; the artefact looks wrong, so a pass \
         here would prove nothing"
    );
    assert!(
        moved_out > 0,
        "no function left a bracket open by moving its guard out, so that case has \
         no live fixture and this test asserted nothing about it"
    );
}

/// The locals this body moves out of.
fn moved_out_locals(body: &Unstructured) -> std::collections::HashSet<u64> {
    let mut moved = std::collections::HashSet::new();
    let mut stack: Vec<&serde_json::Value> = Vec::new();
    for bb in &body.body {
        stack.extend(bb.statements.iter().map(|st| &st.kind));
        stack.push(&bb.terminator.kind);
    }
    while let Some(node) = stack.pop() {
        match node {
            serde_json::Value::Object(map) => {
                if let Some(local) = map
                    .get("Move")
                    .and_then(|place| place.get("kind"))
                    .and_then(|kind| kind.get("Local"))
                    .and_then(serde_json::Value::as_u64)
                {
                    moved.insert(local);
                }
                stack.extend(map.values());
            }
            serde_json::Value::Array(items) => stack.extend(items),
            _ => {}
        }
    }
    moved
}

/// Whether `ty` resolves to the root-bracket guard's own ADT.
fn ty_is_root_scope(llbc: &Llbc, ty: &TyRef) -> bool {
    let TyRef::Dedup { id } = ty else {
        return false;
    };
    llbc.dedup_to_adt_def_id(*id)
        .and_then(|def_id| llbc.type_by_id(def_id))
        .is_some_and(|decl| decl.item_meta.name_path().ends_with("gc_roots::RootScope"))
}

/// Most of the brackets that survive the erasure must actually close.
///
/// The per-fixture tests above pin named shapes; this pins the aggregate, so a
/// rewrite that starts bypassing the block a close sits in shows up as a
/// coverage drop rather than as silence. A rewrite that redirects a block's
/// exits deletes whatever the bypassed chain carried, and the close is not
/// exempt — measured, three bodies here lose one that way.
///
/// Both artefacts are counted. The erasure answers most of `pyre-object`'s
/// brackets, so that crate alone no longer carries a population large enough
/// for this to prove anything.
#[test]
fn nearly_every_dropped_bracket_closes() {
    let (mut bodies, mut closed) = (0usize, 0usize);
    let mut short: Vec<String> = Vec::new();
    for llbc in [object_llbc(), interpreter_llbc()].into_iter().flatten() {
        for fd in llbc.iter_local_fns() {
            let Some(body) = fd.unstructured() else {
                continue;
            };
            let want = owed_closes(llbc, &body);
            if want == 0 {
                continue;
            }
            bodies += 1;
            let Ok(graph) = lower_fun_decl(llbc, fd) else {
                continue;
            };
            if reachable_closes(&graph) >= want {
                closed += 1;
            } else {
                short.push(fd.item_meta.name_path().to_string());
            }
        }
    }
    assert!(
        bodies > 100,
        "only {bodies} bodies drop a surviving guard; the artefact looks wrong, so \
         a pass here would prove nothing"
    );
    assert!(
        closed * 100 >= bodies * 95,
        "only {closed} of {bodies} guard-dropping bodies close every bracket they \
         drop; short: {short:?}"
    );
}

/// The closes a body owes: one per drop of an unmoved, unerased guard, counting
/// only the drops the front lowers. A cleanup-only drop is not one of them —
/// the front does not follow `on_unwind`.
fn owed_closes(llbc: &Llbc, body: &Unstructured) -> usize {
    let moved = moved_out_locals(body);
    let erased = erased_root_bracket_guards(llbc, body);
    let live = reachable_without_unwind(body);
    guard_drop_sites(llbc, body)
        .filter(|(bb, local)| {
            live[*bb] && !moved.contains(local) && !erased.contains(&(*local as usize))
        })
        .count()
}

/// `(block, local)` for every drop of a root-bracket guard, moved or not.
fn guard_drop_sites<'a>(
    llbc: &'a Llbc,
    body: &'a Unstructured,
) -> impl Iterator<Item = (usize, u64)> + 'a {
    body.body.iter().enumerate().filter_map(move |(i, bb)| {
        let Ok(TermKind::Drop { place, .. }) = bb.term() else {
            return None;
        };
        let PlaceKind::Local(local) = place.kind else {
            return None;
        };
        ty_is_root_scope(llbc, &place.ty).then_some((i, local))
    })
}

/// MIR blocks reachable from the entry without taking an unwind edge.
fn reachable_without_unwind(body: &Unstructured) -> Vec<bool> {
    let mut seen = vec![false; body.body.len()];
    let mut stack = vec![0usize];
    while let Some(b) = stack.pop() {
        if b >= seen.len() || seen[b] {
            continue;
        }
        seen[b] = true;
        let Ok(term) = body.body[b].term() else {
            continue;
        };
        match term {
            TermKind::Goto { target }
            | TermKind::Call { target, .. }
            | TermKind::Assert { target, .. }
            | TermKind::Drop { target, .. } => stack.push(target as usize),
            TermKind::Switch { targets, .. } => match targets {
                SwitchTargets::If(a, b) => stack.extend([a as usize, b as usize]),
                SwitchTargets::SwitchInt(_, arms, default) => {
                    stack.extend(arms.into_iter().map(|(_, t)| t as usize));
                    stack.push(default as usize);
                }
            },
            _ => {}
        }
    }
    seen
}

/// Closes in blocks the lowered graph can still reach. A close in a block a
/// rewrite orphaned is one the function no longer runs.
fn reachable_closes(graph: &FunctionGraph) -> usize {
    let mut seen = vec![false; graph.blocks.len()];
    let mut stack = vec![graph.startblock.0];
    let mut found = 0usize;
    while let Some(b) = stack.pop() {
        if b >= graph.blocks.len() || seen[b] {
            continue;
        }
        seen[b] = true;
        found += graph.blocks[b]
            .operations
            .iter()
            .filter(|op| {
                matches!(&op.kind,
                    OpKind::Call { target: CallTarget::FunctionPath { segments }, .. }
                        if segments.last().is_some_and(|s| s == "root_scope_close"))
            })
            .count();
        stack.extend(graph.blocks[b].exits.iter().map(|e| e.target.0));
    }
    found
}
