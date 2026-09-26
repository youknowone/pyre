//! Take the shadow-stack root bracket out of a body before it is lowered.
//!
//! Upstream never puts a root bracket in a jitcode.  `gc_push_roots` /
//! `gc_pop_roots` are genop'd by `ShadowStackFrameworkGCTransformer`
//! (`memory/gctransform/shadowstack.py` `push_roots` / `pop_roots`), and that
//! transformer runs out of the C backend's database long after
//! `warmspot.py` `make_jitcodes` has read the graphs.  pyre spells the bracket
//! in interpreter source, so this pass removes it from what the codewriter
//! reads, the way the gctransformer's absence does upstream.  The native build
//! keeps executing the source bracket unchanged.
//!
//! The bracket is scalar-replaced: every slot a body publishes becomes one
//! synthetic MIR local.  A pin writes the slot local and answers the value it
//! was handed, a read-back answers the slot local, a `set` rewrites it, and the
//! opener, `base()`, `shadow_stack_len()`, the normalize calls and the close
//! all disappear.  The flow-graph builder then carries each slot through the
//! body's blocks like any other local, so a merge or a loop needs nothing of
//! its own.  The references the slots held stay rooted by the jitcode's own
//! root sets (the jitframe gcmap, the blackhole's `registers_r`, the recorder).
//!
//! The rewrite needs every slot index to be a compile-time offset from the
//! depth the body was entered at, so it runs an abstract interpretation of the
//! stack depth first.  A callee is assumed to leave the depth where it found
//! it: a body whose own depth at `Return` is not zero is refused, so the only
//! functions that could break that assumption are the ones this pass never
//! rewrites.  Anything the interpretation cannot model refuses the whole body,
//! which then lowers exactly as it did before.
//!
//! Unwind edges are not followed: the flow-graph builder does not lower an
//! `on_unwind` cleanup chain either, and one chain is shared by calls made at
//! different depths.

use std::collections::{HashMap, VecDeque};

use majit_charon_reader::Llbc;
use majit_charon_reader::ullbc::{
    FunDecl, Operand, Place, PlaceKind, ProjectionElem, Rvalue, StmtKind, TermKind, Unstructured,
};
use serde_json::{Value, json};

/// One shadow-stack operation, keyed by the callee's path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Leaf {
    Open,
    Close,
    Base,
    Len,
    Pin,
    Publish,
    Normalize,
    NormalizeMoved,
    Get,
    Set,
    ReloadTop,
    /// Touches the stack in a way this pass does not model.
    Unmodeled,
}

/// Classify a `gc_roots` callee, and say whether it takes the guard as its
/// first argument.  `None` for a callee outside the module.
///
/// Charon names an inherent method `gc_roots::<Impl>::leaf`, so the type a
/// method belongs to is read off the call: the receiver's type for a method,
/// the destination's for the guard's constructor.
fn classify_call(
    call: &majit_charon_reader::ullbc::CallPayload,
    llbc: &Llbc,
) -> Option<(Leaf, bool)> {
    let path = callee_path(call, llbc)?;
    let segments: Vec<&str> = path.split("::").collect();
    if !segments.contains(&super::ROOT_SCOPE_MODULE) {
        return None;
    }
    let leaf = *segments.last()?;
    let names_type = |ty: &majit_charon_reader::ullbc::TyRef, name: &str| {
        super::tyref_class_root(ty, llbc).is_some_and(|root| root.rsplit("::").next() == Some(name))
    };
    let receiver_ty = match call.args.first() {
        Some(Operand::Copy(p) | Operand::Move(p)) => Some(&p.ty),
        _ => None,
    };
    let is_method =
        segments.contains(&super::ROOT_SCOPE_TYPE) || segments.iter().any(|s| s.starts_with('<'));
    if is_method {
        if names_type(&call.dest.ty, super::ROOT_SCOPE_TYPE) && leaf == "new" {
            return Some((Leaf::Open, false));
        }
        if !receiver_ty.is_some_and(|ty| names_type(ty, super::ROOT_SCOPE_TYPE)) {
            // `RootedItems`, `RootStack` and the walkers: the stack itself.
            return Some((Leaf::Unmodeled, false));
        }
        let kind = match leaf {
            "base" => Leaf::Base,
            "pin_root" => Leaf::Pin,
            "get" => Leaf::Get,
            "publish" | "pin_roots" => Leaf::Publish,
            "normalize" => Leaf::Normalize,
            "normalize_moved" => Leaf::NormalizeMoved,
            "set" => Leaf::Set,
            "drop" | "drop_in_place" => Leaf::Close,
            _ => Leaf::Unmodeled,
        };
        return Some((kind, true));
    }
    let kind = match leaf {
        "push_roots" => Leaf::Open,
        "root_scope_close" => return Some((Leaf::Close, true)),
        "pin_root" => Leaf::Pin,
        "pin_roots" | "publish_roots" => Leaf::Publish,
        "normalize_roots" => Leaf::Normalize,
        "shadow_stack_len" => Leaf::Len,
        "shadow_stack_get" => Leaf::Get,
        "shadow_stack_set" => Leaf::Set,
        "reload_top_root" => Leaf::ReloadTop,
        "mark_prebuilt_roots_dirty"
        | "prebuilt_roots_dirty"
        | "clear_prebuilt_roots_dirty"
        | "increase_root_stack_depth"
        | "root_stack_depth" => return None,
        // Everything else in the module reads or rewrites the stack itself.
        _ => Leaf::Unmodeled,
    };
    Some((kind, false))
}

/// What a special local stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Special {
    /// A `RootScope` guard opened at this depth.
    Guard(usize),
    /// A borrow of a guard.
    Alias(usize),
    /// A slot index: the entry depth plus this offset.
    Index(usize),
    /// A checked-add pair whose `.0` is this offset.
    IndexPair(usize),
}

/// A call or assert this pass rewrites to a `Goto`, with the statements that
/// replace it.
struct TermRewrite {
    target: u64,
    stmts: Vec<Value>,
}

struct Plan {
    specials: HashMap<usize, Special>,
    /// Statements to drop, per block.
    removed: HashMap<usize, Vec<usize>>,
    /// Statements to insert after a statement, per block.
    inserted_after: HashMap<(usize, usize), Vec<Value>>,
    terms: HashMap<usize, TermRewrite>,
    slot_count: usize,
    slot_ty: Option<Value>,
    /// Blocks no path from the entry reaches once the rewritten calls lose
    /// their unwind edges.
    unreachable: Vec<usize>,
}

type Refusal = &'static str;

fn place_local(place: &Place) -> Option<usize> {
    match place.kind {
        PlaceKind::Local(l) => Some(l as usize),
        _ => None,
    }
}

fn operand_local(op: &Operand) -> Option<usize> {
    match op {
        Operand::Copy(p) | Operand::Move(p) => place_local(p),
        Operand::Const(_) => None,
    }
}

/// The base local of `*local`.
fn deref_of_local(place: &Place) -> Option<usize> {
    let PlaceKind::Projection(inner, ProjectionElem::Atom(elem)) = &place.kind else {
        return None;
    };
    (elem == "Deref").then(|| place_local(inner)).flatten()
}

/// `local.0` / `local.1` of a tuple.
fn tuple_field_of_local(place: &Place) -> Option<(usize, u64)> {
    let PlaceKind::Projection(inner, ProjectionElem::Tagged(elem)) = &place.kind else {
        return None;
    };
    let field = elem.get("Field")?.as_array()?;
    let index = field.get(1)?.as_u64()?;
    Some((place_local(inner)?, index))
}

/// An unsigned scalar constant operand.
fn const_usize(op: &Operand) -> Option<usize> {
    let Operand::Const(v) = op else {
        return None;
    };
    let scalar = v.get("kind")?.get("Literal")?.get("Scalar")?;
    let lit = scalar.get("Unsigned").or_else(|| scalar.get("Signed"))?;
    lit.as_array()?.get(1)?.as_str()?.parse().ok()
}

fn binop_is_add(op: &Value) -> Option<bool> {
    // `"AddChecked"` yields a pair; `"Add"` / `{"Add": "Wrap"}` a value.
    match op {
        Value::String(s) if s == "AddChecked" => Some(true),
        Value::String(s) if s == "Add" => Some(false),
        Value::Object(m) if m.contains_key("Add") => Some(false),
        _ => None,
    }
}

fn copy_operand_json(op: &Value) -> Value {
    match op.as_object() {
        Some(m) if m.contains_key("Move") => json!({ "Copy": m["Move"].clone() }),
        _ => op.clone(),
    }
}

fn local_place_json(local: usize, ty: &Value) -> Value {
    json!({ "kind": { "Local": local }, "ty": ty })
}

fn assign_json(dest: Value, rvalue: Value) -> Value {
    json!({ "Assign": [dest, rvalue] })
}

/// Build the rewrite, or say why this body keeps its bracket.
fn analyze(body: &Unstructured, raw: &Value, llbc: &Llbc) -> Result<Option<Plan>, Refusal> {
    let n_blocks = body.body.len();
    let n_locals = body.locals.locals.len();
    let blocks_json = raw["body"].as_array().ok_or("body-json")?;
    let mut classified: HashMap<usize, (Leaf, bool)> = HashMap::new();
    for (bb, block) in body.body.iter().enumerate() {
        if let Ok(TermKind::Call { call, .. }) = block.term_ref()
            && let Some(class) = classify_call(call, llbc)
        {
            classified.insert(bb, class);
        }
    }
    if classified.is_empty() {
        return Ok(None);
    }
    if !llbc.stack_sensitive_fns_complete() {
        return Err("callee-stack-effects-unknown");
    }
    for block in &body.body {
        if let Ok(TermKind::Call { call, .. }) = block.term_ref()
            && callee_path(call, llbc).is_some_and(|path| llbc.is_stack_sensitive_fn(&path))
        {
            return Err("calls-stack-sensitive-fn");
        }
    }
    if classified
        .values()
        .any(|(leaf, _)| *leaf == Leaf::Unmodeled)
    {
        return Err("unmodeled-stack-op");
    }

    let mut plan = Plan {
        specials: HashMap::new(),
        removed: HashMap::new(),
        inserted_after: HashMap::new(),
        terms: HashMap::new(),
        slot_count: 0,
        slot_ty: None,
        unreachable: Vec::new(),
    };
    let bind =
        |specials: &mut HashMap<usize, Special>, local: usize, special: Special| match specials
            .insert(local, special)
        {
            Some(previous) if previous != special => Err("special-rebound"),
            _ => Ok(()),
        };
    let slot_local = |k: usize| n_locals + k;

    let mut depth_in: Vec<Option<usize>> = vec![None; n_blocks];
    let mut queue: VecDeque<usize> = VecDeque::new();
    if n_blocks == 0 {
        return Ok(None);
    }
    depth_in[0] = Some(0);
    queue.push_back(0);
    let mut visited = vec![false; n_blocks];
    let reach = |depth_in: &mut Vec<Option<usize>>,
                 queue: &mut VecDeque<usize>,
                 target: u64,
                 depth: usize|
     -> Result<(), Refusal> {
        let target = target as usize;
        if target >= n_blocks {
            return Ok(());
        }
        match depth_in[target] {
            Some(d) if d != depth => Err("depth-disagrees-at-merge"),
            Some(_) => Ok(()),
            None => {
                depth_in[target] = Some(depth);
                queue.push_back(target);
                Ok(())
            }
        }
    };

    while let Some(bb) = queue.pop_front() {
        if visited[bb] {
            continue;
        }
        visited[bb] = true;
        let mut depth = depth_in[bb].expect("queued blocks have a depth");
        let block = &body.body[bb];
        // Statements: the index, pair and alias definitions.
        for (si, stmt) in block.statements.iter().enumerate() {
            let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind_ref() else {
                continue;
            };
            let Some(dest) = place_local(place) else {
                continue;
            };
            let special = match rvalue {
                Rvalue::Ref { place: src, .. } | Rvalue::RawPtr { place: src, .. } => {
                    let guard = place_local(src)
                        .and_then(|l| match plan.specials.get(&l) {
                            Some(Special::Guard(_)) => Some(l),
                            _ => None,
                        })
                        .or_else(|| {
                            deref_of_local(src).and_then(|l| match plan.specials.get(&l) {
                                Some(Special::Alias(g)) => Some(*g),
                                _ => None,
                            })
                        });
                    guard.map(Special::Alias)
                }
                Rvalue::Use(op) => match op {
                    Operand::Copy(src) | Operand::Move(src) => {
                        if let Some(l) = place_local(src) {
                            match plan.specials.get(&l) {
                                Some(Special::Index(k)) => Some(Special::Index(*k)),
                                Some(Special::Alias(g)) => Some(Special::Alias(*g)),
                                _ => None,
                            }
                        } else if let Some((pair, 0)) = tuple_field_of_local(src) {
                            match plan.specials.get(&pair) {
                                Some(Special::IndexPair(k)) => Some(Special::Index(*k)),
                                _ => None,
                            }
                        } else {
                            None
                        }
                    }
                    Operand::Const(_) => None,
                },
                Rvalue::BinaryOp(op, lhs, rhs) => match binop_is_add(op) {
                    Some(checked) => {
                        let index_of = |o: &Operand| {
                            operand_local(o).and_then(|l| match plan.specials.get(&l) {
                                Some(Special::Index(k)) => Some(*k),
                                _ => None,
                            })
                        };
                        let offset = match (index_of(lhs), index_of(rhs)) {
                            (Some(k), None) => const_usize(rhs).map(|c| k + c),
                            (None, Some(k)) => const_usize(lhs).map(|c| k + c),
                            _ => None,
                        };
                        offset.map(|k| {
                            if checked {
                                Special::IndexPair(k)
                            } else {
                                Special::Index(k)
                            }
                        })
                    }
                    None => None,
                },
                _ => None,
            };
            if let Some(special) = special {
                bind(&mut plan.specials, dest, special)?;
                plan.removed.entry(bb).or_default().push(si);
            }
        }
        let term_json = &blocks_json[bb]["terminator"]["kind"];
        match block.term_ref() {
            Ok(TermKind::Call { call, target, .. }) => {
                let Some(&(leaf, method)) = classified.get(&bb) else {
                    reach(&mut depth_in, &mut queue, *target, depth)?;
                    continue;
                };
                let args_json = &term_json["Call"]["call"]["args"];
                let dest_json = &term_json["Call"]["call"]["dest"];
                let dest = place_local(&call.dest);
                let arg0 = usize::from(method);
                let guard = if method {
                    let receiver = call.args.first().and_then(operand_local);
                    match receiver.and_then(|l| plan.specials.get(&l)) {
                        Some(Special::Alias(g)) => Some(*g),
                        _ => return Err("receiver-not-a-guard-borrow"),
                    }
                } else {
                    None
                };
                let guard_depth = |g: usize| match plan.specials.get(&g) {
                    Some(Special::Guard(d)) => Ok(*d),
                    _ => Err("guard-not-opened"),
                };
                let index_arg = |i: usize| -> Result<usize, Refusal> {
                    match call
                        .args
                        .get(i)
                        .and_then(operand_local)
                        .and_then(|l| plan.specials.get(&l))
                    {
                        Some(Special::Index(k)) => Ok(*k),
                        _ => Err("index-not-static"),
                    }
                };
                let mut stmts = Vec::new();
                match leaf {
                    Leaf::Open => {
                        let dest = dest.ok_or("guard-dest-projected")?;
                        bind(&mut plan.specials, dest, Special::Guard(depth))?;
                    }
                    Leaf::Close => {
                        let d = guard_depth(guard.ok_or("close-without-guard")?)?;
                        if d > depth {
                            return Err("close-above-depth");
                        }
                        depth = d;
                    }
                    Leaf::Base => {
                        let d = guard_depth(guard.ok_or("base-without-guard")?)?;
                        bind(
                            &mut plan.specials,
                            dest.ok_or("index-dest-projected")?,
                            Special::Index(d),
                        )?;
                    }
                    Leaf::Len => {
                        bind(
                            &mut plan.specials,
                            dest.ok_or("index-dest-projected")?,
                            Special::Index(depth),
                        )?;
                    }
                    Leaf::Pin => {
                        let value = &args_json[arg0];
                        let ty = value
                            .as_object()
                            .and_then(|m| m.values().next())
                            .and_then(|p| p.get("ty"))
                            .cloned()
                            .ok_or("pin-value-untyped")?;
                        plan.slot_ty.get_or_insert(ty.clone());
                        let copied = copy_operand_json(value);
                        stmts.push(assign_json(
                            local_place_json(slot_local(depth), &ty),
                            json!({ "Use": copied }),
                        ));
                        stmts.push(assign_json(dest_json.clone(), json!({ "Use": copied })));
                        depth += 1;
                        plan.slot_count = plan.slot_count.max(depth);
                    }
                    Leaf::Publish => {
                        let slice = call.args.get(arg0).and_then(operand_local);
                        let (agg_si, operands) =
                            resolve_published_array(block, &blocks_json[bb], slice)
                                .ok_or("publish-array-unresolved")?;
                        let mut inserted = Vec::new();
                        for (i, element) in operands.iter().enumerate() {
                            let ty = element
                                .as_object()
                                .and_then(|m| m.values().next())
                                .and_then(|p| p.get("ty"))
                                .cloned()
                                .ok_or("publish-element-untyped")?;
                            plan.slot_ty.get_or_insert(ty.clone());
                            inserted.push(assign_json(
                                local_place_json(slot_local(depth + i), &ty),
                                json!({ "Use": copy_operand_json(element) }),
                            ));
                        }
                        plan.inserted_after
                            .entry((bb, agg_si))
                            .or_default()
                            .extend(inserted);
                        if let Some(dest) = dest {
                            bind(&mut plan.specials, dest, Special::Index(depth))?;
                        }
                        depth += operands.len();
                        plan.slot_count = plan.slot_count.max(depth);
                    }
                    Leaf::Normalize => {}
                    Leaf::NormalizeMoved => {
                        let ty = dest_json["ty"].clone();
                        stmts.push(assign_json(
                            dest_json.clone(),
                            json!({ "Use": { "Const": {
                                "kind": { "Literal": { "Bool": false } },
                                "ty": ty,
                            } } }),
                        ));
                    }
                    Leaf::Get => {
                        let k = index_arg(arg0)?;
                        if k >= depth {
                            return Err("get-above-depth");
                        }
                        let ty = plan.slot_ty.clone().ok_or("get-before-any-pin")?;
                        stmts.push(assign_json(
                            dest_json.clone(),
                            json!({ "Use": { "Copy": local_place_json(slot_local(k), &ty) } }),
                        ));
                    }
                    Leaf::Set => {
                        let k = index_arg(arg0)?;
                        if k >= depth {
                            return Err("set-above-depth");
                        }
                        let ty = plan.slot_ty.clone().ok_or("set-before-any-pin")?;
                        stmts.push(assign_json(
                            local_place_json(slot_local(k), &ty),
                            json!({ "Use": copy_operand_json(&args_json[arg0 + 1]) }),
                        ));
                    }
                    Leaf::ReloadTop => {
                        if depth == 0 {
                            return Err("reload-empty-stack");
                        }
                        let ty = plan.slot_ty.clone().ok_or("reload-before-any-pin")?;
                        stmts.push(assign_json(
                            dest_json.clone(),
                            json!({ "Use": { "Copy": local_place_json(slot_local(depth - 1), &ty) } }),
                        ));
                    }
                    Leaf::Unmodeled => unreachable!("refused above"),
                }
                plan.terms.insert(
                    bb,
                    TermRewrite {
                        target: *target,
                        stmts,
                    },
                );
                reach(&mut depth_in, &mut queue, *target, depth)?;
            }
            Ok(TermKind::Drop { place, target, .. }) => {
                let dropped = place_local(place);
                if let Some(Special::Guard(d)) = dropped.and_then(|l| plan.specials.get(&l)) {
                    // A drop flag the artefact does not carry can leave the
                    // guard unopened on this path; the depth says so.
                    if *d > depth {
                        return Err("close-above-depth");
                    }
                    depth = *d;
                    plan.terms.insert(
                        bb,
                        TermRewrite {
                            target: *target,
                            stmts: Vec::new(),
                        },
                    );
                    reach(&mut depth_in, &mut queue, *target, depth)?;
                } else if dropped.is_some_and(|l| is_root_scope_local(body, llbc, l)) {
                    return Err("drop-of-unopened-guard");
                } else {
                    reach(&mut depth_in, &mut queue, *target, depth)?;
                }
            }
            Ok(TermKind::Assert { assert, target, .. }) => {
                let pair = match &assert.cond {
                    Operand::Copy(p) | Operand::Move(p) => tuple_field_of_local(p),
                    Operand::Const(_) => None,
                };
                if let Some((pair, 1)) = pair
                    && matches!(plan.specials.get(&pair), Some(Special::IndexPair(_)))
                {
                    plan.terms.insert(
                        bb,
                        TermRewrite {
                            target: *target,
                            stmts: Vec::new(),
                        },
                    );
                    reach(&mut depth_in, &mut queue, *target, depth)?;
                } else {
                    reach(&mut depth_in, &mut queue, *target, depth)?;
                }
            }
            Ok(TermKind::Goto { target }) => reach(&mut depth_in, &mut queue, *target, depth)?,
            Ok(TermKind::Switch { targets, .. }) => match targets {
                majit_charon_reader::ullbc::SwitchTargets::If(a, b) => {
                    reach(&mut depth_in, &mut queue, *a, depth)?;
                    reach(&mut depth_in, &mut queue, *b, depth)?;
                }
                majit_charon_reader::ullbc::SwitchTargets::SwitchInt(_, arms, default) => {
                    for (_, t) in arms {
                        reach(&mut depth_in, &mut queue, *t, depth)?;
                    }
                    reach(&mut depth_in, &mut queue, *default, depth)?;
                }
            },
            Ok(TermKind::Return) => {
                if depth != 0 {
                    return Err("returns-with-published-slots");
                }
            }
            Ok(TermKind::UnwindResume | TermKind::Abort(_)) => {}
            _ => return Err("unknown-terminator"),
        }
    }

    // Every surviving mention of a guard, borrow or index has to be one of the
    // shapes rewritten above; anything else would read a local no block binds.
    let mut watched = bit_set::BitSet::new();
    for local in plan.specials.keys() {
        watched.insert(*local);
    }
    for (bb, block) in body.body.iter().enumerate() {
        if !visited[bb] {
            continue;
        }
        let removed = plan.removed.get(&bb);
        for (si, stmt) in block.statements.iter().enumerate() {
            if removed.is_some_and(|r| r.contains(&si)) {
                continue;
            }
            if matches!(
                stmt.stmt_kind_ref(),
                Ok(StmtKind::StorageLive(_) | StmtKind::StorageDead(_))
            ) {
                continue;
            }
            if super::mentions_local(&stmt.kind, &watched) {
                return Err("unmodeled-use-in-statement");
            }
        }
        if plan.terms.contains_key(&bb) {
            continue;
        }
        if super::mentions_local(&block.terminator.kind, &watched) {
            return Err("unmodeled-use-in-terminator");
        }
    }
    plan.unreachable = (0..n_blocks).filter(|bb| !visited[*bb]).collect();
    Ok(Some(plan))
}

pub(super) fn is_root_scope_local(body: &Unstructured, llbc: &Llbc, local: usize) -> bool {
    let Some(decl) = body.locals.locals.get(local) else {
        return false;
    };
    super::output_adt_def_id_free(&decl.ty, llbc)
        .and_then(|id| llbc.type_by_id(id))
        .is_some_and(|t| super::gc_root_scope_type_path(&t.item_meta.name_path()))
}

/// Follow a published slice back to the array literal it borrows, in the same
/// block: `_a = [x, y]; _r = &_a; _s = &*_r; _p = _s as &[_]`.  Returns the
/// literal's statement index and its element operands.
fn resolve_published_array(
    block: &majit_charon_reader::ullbc::BasicBlock,
    block_json: &Value,
    slice: Option<usize>,
) -> Option<(usize, Vec<Value>)> {
    let mut want = slice?;
    for (si, stmt) in block.statements.iter().enumerate().rev() {
        let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind_ref() else {
            continue;
        };
        if place_local(place) != Some(want) {
            continue;
        }
        match rvalue {
            Rvalue::UnaryOp(_, op) | Rvalue::Use(op) | Rvalue::Cast(_, op, _) => {
                want = operand_local(op)?;
            }
            Rvalue::Ref { place: src, .. } => {
                want = place_local(src).or_else(|| deref_of_local(src))?;
            }
            Rvalue::Aggregate(kind, _) if kind.get("Array").is_some() => {
                let operands = block_json["statements"][si]["kind"]["Assign"][1]["Aggregate"][1]
                    .as_array()?
                    .clone();
                return Some((si, operands));
            }
            _ => return None,
        }
    }
    None
}

/// Rewrite `fd`'s body with its root brackets scalar-replaced.
///
/// `Ok(None)`: the body has no bracket.  `Err`: it has one this pass refuses;
/// the reason names the shape.
pub(super) fn erase_shadow_stack(
    fd: &FunDecl,
    body: &Unstructured,
    llbc: &Llbc,
) -> Result<Option<Unstructured>, Refusal> {
    let Some(raw) = fd.body.as_ref() else {
        return Ok(None);
    };
    // The root-stack runtime itself implements the leaves this pass removes;
    // its bodies stay as written for the callers that keep their brackets.
    if fd.item_meta.name_path().contains("::gc_roots::") {
        return Ok(None);
    }
    let mut root: Value = serde_json::from_str(raw.get()).map_err(|_| "body-json")?;
    let plan = {
        let u = root.get("Unstructured").ok_or("body-json")?;
        match analyze(body, u, llbc)? {
            Some(plan) => plan,
            None => return Ok(None),
        }
    };
    let u = root.get_mut("Unstructured").ok_or("body-json")?;
    let span = u["span"].clone();
    // The slot locals.
    if plan.slot_count > 0 {
        let ty = plan.slot_ty.clone().ok_or("slot-untyped")?;
        let locals = u["locals"]["locals"].as_array_mut().ok_or("body-json")?;
        let base = locals.len();
        for k in 0..plan.slot_count {
            locals.push(json!({
                "index": base + k,
                "name": null,
                "span": span,
                "ty": ty,
            }));
        }
    }
    let blocks = u["body"].as_array_mut().ok_or("body-json")?;
    for (bb, block) in blocks.iter_mut().enumerate() {
        // An unreachable block may still name a guard or an index whose
        // definition is gone; nothing runs it.
        if plan.unreachable.contains(&bb) {
            block["statements"] = Value::Array(Vec::new());
            block["terminator"]["kind"] = json!({ "Abort": "UnwindTerminate" });
            continue;
        }
        let removed = plan.removed.get(&bb);
        let term = plan.terms.get(&bb);
        let has_inserts = plan.inserted_after.keys().any(|(b, _)| *b == bb);
        if removed.is_none() && term.is_none() && !has_inserts {
            continue;
        }
        let stmt_span = block["terminator"]["span"].clone();
        let make_stmt = |kind: Value| {
            json!({
                "span": stmt_span,
                "kind": kind,
                "comments_before": [],
            })
        };
        let old = block["statements"].as_array().cloned().unwrap_or_default();
        let mut new = Vec::with_capacity(old.len());
        for (si, stmt) in old.into_iter().enumerate() {
            if !removed.is_some_and(|r| r.contains(&si)) {
                new.push(stmt);
            }
            if let Some(extra) = plan.inserted_after.get(&(bb, si)) {
                new.extend(extra.iter().cloned().map(make_stmt));
            }
        }
        if let Some(term) = term {
            new.extend(term.stmts.iter().cloned().map(make_stmt));
            block["terminator"]["kind"] = json!({ "Goto": { "target": term.target } });
        }
        block["statements"] = Value::Array(new);
    }
    let body = root
        .get_mut("Unstructured")
        .map(Value::take)
        .ok_or("body-json")?;
    serde_json::from_value::<Unstructured>(body)
        .map(Some)
        .map_err(|_| "rewritten-body-unparsable")
}

/// [`erase_shadow_stack`], falling back to the body as extracted.  A refusal is
/// counted under its reason.
pub(super) fn erase_or_keep(fd: &FunDecl, body: Unstructured, llbc: &Llbc) -> Unstructured {
    match erase_shadow_stack(fd, &body, llbc) {
        Ok(Some(erased)) => erased,
        Ok(None) => body,
        Err(reason) => {
            crate::decline::record(
                crate::decline::gate::SHADOW_STACK_ERASE,
                reason,
                format_args!("{}", fd.item_meta.name_path()),
            );
            body
        }
    }
}

/// Every local body with a root bracket, bucketed by what the scalar
/// replacement did with it: `"erased"`, or the refusal reason.  The subject
/// list names the bodies in each bucket.
pub fn census(llbc: &Llbc) -> std::collections::BTreeMap<&'static str, Vec<String>> {
    ensure_stack_sensitive_fns(llbc);
    let mut out: std::collections::BTreeMap<&'static str, Vec<String>> = Default::default();
    for fd in llbc.iter_local_fns() {
        let Some(body) = fd.unstructured() else {
            continue;
        };
        let bucket = match erase_shadow_stack(fd, &body, llbc) {
            Ok(Some(_)) => "erased",
            Ok(None) => continue,
            Err(reason) => reason,
        };
        out.entry(bucket)
            .or_default()
            .push(fd.item_meta.name_path());
    }
    out
}

// ---------------------------------------------------------------------------
// Stack effects a caller cannot see past
// ---------------------------------------------------------------------------

/// Stack depth relative to a body's entry, or unknown once a callee may have
/// left slots behind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Depth {
    Known(usize),
    Unknown,
}

impl Depth {
    fn join(self, other: Depth) -> Depth {
        if self == other { self } else { Depth::Unknown }
    }

    fn add(self, n: usize) -> Depth {
        match self {
            Depth::Known(d) => Depth::Known(d + n),
            Depth::Unknown => Depth::Unknown,
        }
    }
}

/// The callee path a call terminator names, for a statically resolved callee.
fn callee_path(call: &majit_charon_reader::ullbc::CallPayload, llbc: &Llbc) -> Option<String> {
    let majit_charon_reader::ullbc::CallFunc::Regular(reg) = &call.func else {
        return None;
    };
    let id = super::regular_call_fun_decl_id(&reg.kind)?;
    llbc.fn_by_id(id).map(|fd| fd.item_meta.name_path())
}

/// Whether a body can leave slots published past its return, or reads a
/// slot it did not publish itself.  Either makes it unsafe for a caller to
/// scalar-replace its own bracket: the caller's slots would be missing from
/// the real stack the callee reads, or the callee's leftovers would never be
/// rewound.  `sensitive` answers the same question for a callee.  The
/// answer names the first site that makes the body sensitive.
fn stack_sensitivity(
    body: &Unstructured,
    llbc: &Llbc,
    sensitive: &dyn Fn(&str) -> bool,
) -> Option<String> {
    let n_blocks = body.body.len();
    if n_blocks == 0 {
        return None;
    }
    let mut why = String::new();
    let mut depth_in: Vec<Option<Depth>> = vec![None; n_blocks];
    let mut guards: HashMap<usize, Depth> = HashMap::new();
    let mut aliases: HashMap<usize, usize> = HashMap::new();
    let mut indices: HashMap<usize, Depth> = HashMap::new();
    let mut pairs: HashMap<usize, Depth> = HashMap::new();
    depth_in[0] = Some(Depth::Known(0));
    let mut queue: VecDeque<usize> = VecDeque::from([0]);
    let mut reads_below = false;
    let mut leaves = false;
    let mut rounds = 0usize;
    while let Some(bb) = queue.pop_front() {
        rounds += 1;
        if rounds > n_blocks * 8 + 64 {
            // A depth that keeps growing round a loop: slots pile up.
            return Some("depth-grows-in-loop".into());
        }
        let mut depth = depth_in[bb].expect("queued blocks have a depth");
        let block = &body.body[bb];
        for stmt in &block.statements {
            let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind_ref() else {
                continue;
            };
            let Some(dest) = place_local(place) else {
                continue;
            };
            match rvalue {
                Rvalue::Ref { place: src, .. } | Rvalue::RawPtr { place: src, .. } => {
                    let guard = place_local(src)
                        .filter(|l| guards.contains_key(l))
                        .or_else(|| deref_of_local(src).and_then(|l| aliases.get(&l).copied()));
                    if let Some(g) = guard {
                        aliases.insert(dest, g);
                    }
                }
                Rvalue::Use(Operand::Copy(src) | Operand::Move(src)) => {
                    if let Some(l) = place_local(src) {
                        if let Some(k) = indices.get(&l).copied() {
                            indices.insert(dest, k);
                        }
                        if let Some(g) = aliases.get(&l).copied() {
                            aliases.insert(dest, g);
                        }
                    } else if let Some((pair, 0)) = tuple_field_of_local(src)
                        && let Some(k) = pairs.get(&pair).copied()
                    {
                        indices.insert(dest, k);
                    }
                }
                Rvalue::BinaryOp(op, lhs, rhs) => {
                    if let Some(checked) = binop_is_add(op) {
                        let base =
                            |o: &Operand| operand_local(o).and_then(|l| indices.get(&l).copied());
                        let offset = match (base(lhs), base(rhs)) {
                            (Some(k), None) => const_usize(rhs).map(|c| k.add(c)),
                            (None, Some(k)) => const_usize(lhs).map(|c| k.add(c)),
                            _ => None,
                        };
                        if let Some(k) = offset {
                            if checked {
                                pairs.insert(dest, k);
                            } else {
                                indices.insert(dest, k);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        // `None`: the block ends the walk.  Otherwise the successors and the
        // depth they receive.
        let mut successors: Vec<u64> = Vec::new();
        match block.term_ref() {
            Ok(TermKind::Call { call, target, .. }) => {
                successors.push(*target);
                let path = callee_path(call, llbc);
                match classify_call(call, llbc) {
                    Some((leaf, method)) => {
                        let arg0 = usize::from(method);
                        let guard = call
                            .args
                            .first()
                            .and_then(operand_local)
                            .and_then(|l| aliases.get(&l).copied());
                        let index_ok = |i: usize, depth: Depth| {
                            let k = call
                                .args
                                .get(i)
                                .and_then(operand_local)
                                .and_then(|l| indices.get(&l).copied());
                            matches!((k, depth), (Some(Depth::Known(k)), Depth::Known(d)) if k < d)
                        };
                        match leaf {
                            Leaf::Open => {
                                if let Some(dest) = place_local(&call.dest) {
                                    let entry = guards.entry(dest).or_insert(depth);
                                    *entry = entry.join(depth);
                                }
                            }
                            Leaf::Close => {
                                depth = guard
                                    .and_then(|g| guards.get(&g).copied())
                                    .unwrap_or(Depth::Unknown);
                            }
                            Leaf::Base => {
                                if let (Some(dest), Some(g)) = (place_local(&call.dest), guard) {
                                    indices.insert(
                                        dest,
                                        guards.get(&g).copied().unwrap_or(Depth::Unknown),
                                    );
                                }
                            }
                            Leaf::Len => {
                                if let Some(dest) = place_local(&call.dest) {
                                    indices.insert(dest, depth);
                                }
                            }
                            Leaf::Pin => depth = depth.add(1),
                            Leaf::Publish => {
                                let slice = call.args.get(arg0).and_then(operand_local);
                                match resolve_published_array_len(block, slice) {
                                    Some(n) => {
                                        if let Some(dest) = place_local(&call.dest) {
                                            indices.insert(dest, depth);
                                        }
                                        depth = depth.add(n);
                                    }
                                    None => depth = Depth::Unknown,
                                }
                            }
                            Leaf::Normalize | Leaf::NormalizeMoved => {}
                            Leaf::Get | Leaf::Set => {
                                if !index_ok(arg0, depth) {
                                    reads_below = true;
                                    why = format!("index-not-own bb{bb}");
                                }
                            }
                            Leaf::ReloadTop => {
                                if !matches!(depth, Depth::Known(d) if d > 0) {
                                    reads_below = true;
                                    why = "reload-top-below".into();
                                }
                            }
                            Leaf::Unmodeled => {
                                reads_below = true;
                                depth = Depth::Unknown;
                                why = format!("unmodeled {}", path.as_deref().unwrap_or(""));
                            }
                        }
                    }
                    None => {
                        if path.as_deref().is_some_and(sensitive) {
                            // A sensitive callee may read our slots or leave
                            // its own; either way nothing after it is known.
                            reads_below = true;
                            depth = Depth::Unknown;
                            why = format!("callee {}", path.as_deref().unwrap_or(""));
                        }
                    }
                }
            }
            Ok(TermKind::Drop { place, target, .. }) => {
                successors.push(*target);
                if let Some(g) = place_local(place).filter(|l| guards.contains_key(l)) {
                    depth = guards[&g];
                } else if place_local(place).is_some_and(|l| is_root_scope_local(body, llbc, l)) {
                    depth = Depth::Unknown;
                }
            }
            Ok(TermKind::Assert { target, .. }) => successors.push(*target),
            Ok(TermKind::Goto { target }) => successors.push(*target),
            Ok(TermKind::Switch { targets, .. }) => match targets {
                majit_charon_reader::ullbc::SwitchTargets::If(a, b) => {
                    successors.extend([*a, *b]);
                }
                majit_charon_reader::ullbc::SwitchTargets::SwitchInt(_, arms, default) => {
                    successors.extend(arms.iter().map(|(_, t)| *t));
                    successors.push(*default);
                }
            },
            Ok(TermKind::Return) => {
                if depth != Depth::Known(0) {
                    leaves = true;
                    why = format!("returns-at {depth:?}");
                }
            }
            _ => {}
        }
        if reads_below || leaves {
            return Some(why);
        }
        for target in successors {
            let target = target as usize;
            if target >= n_blocks {
                continue;
            }
            let next = match depth_in[target] {
                None => depth,
                Some(old) => old.join(depth),
            };
            if depth_in[target] != Some(next) {
                depth_in[target] = Some(next);
                queue.push_back(target);
            }
        }
    }
    None
}

/// [`resolve_published_array`] without the JSON: only the literal's length.
fn resolve_published_array_len(
    block: &majit_charon_reader::ullbc::BasicBlock,
    slice: Option<usize>,
) -> Option<usize> {
    let mut want = slice?;
    for stmt in block.statements.iter().rev() {
        let Ok(StmtKind::Assign(place, rvalue)) = stmt.stmt_kind_ref() else {
            continue;
        };
        if place_local(place) != Some(want) {
            continue;
        }
        match rvalue {
            Rvalue::UnaryOp(_, op) | Rvalue::Use(op) | Rvalue::Cast(_, op, _) => {
                want = operand_local(op)?;
            }
            Rvalue::Ref { place: src, .. } => {
                want = place_local(src).or_else(|| deref_of_local(src))?;
            }
            Rvalue::Aggregate(kind, operands) if kind.get("Array").is_some() => {
                return Some(operands.len());
            }
            _ => return None,
        }
    }
    None
}

/// The functions of `llbc` whose shadow-stack effect a caller cannot see
/// past (see [`is_stack_sensitive`]).  Callees from artefacts linked earlier
/// are answered by what they registered on `llbc`
/// ([`Llbc::is_stack_sensitive_fn`]).
pub fn discover_stack_sensitive_fns(llbc: &Llbc) -> Vec<String> {
    let fds: Vec<&FunDecl> = llbc
        .iter_local_fns()
        .filter(|fd| fd.body.is_some())
        .collect();
    let mut found: std::collections::HashSet<String> = std::collections::HashSet::new();
    // callee path -> the bodies that call it.
    let mut callers: HashMap<String, Vec<usize>> = HashMap::new();
    let mut queue: VecDeque<usize> = (0..fds.len()).collect();
    let mut queued = vec![true; fds.len()];
    let mut first_pass = true;
    let mut remaining_first = fds.len();
    while let Some(i) = queue.pop_front() {
        queued[i] = false;
        let fd = fds[i];
        let Some(body) = fd.unstructured() else {
            continue;
        };
        if first_pass {
            for block in &body.body {
                if let Ok(TermKind::Call { call, .. }) = block.term_ref()
                    && let Some(path) = callee_path(call, llbc)
                {
                    callers.entry(path).or_default().push(i);
                }
            }
            remaining_first -= 1;
            if remaining_first == 0 {
                first_pass = false;
            }
        }
        let name = fd.item_meta.name_path();
        if found.contains(&name) {
            continue;
        }
        let sensitive = |path: &str| found.contains(path) || llbc.is_stack_sensitive_fn(path);
        if stack_sensitivity(&body, llbc, &sensitive).is_some() {
            for &caller in callers.get(&name).into_iter().flatten() {
                if !queued[caller] {
                    queued[caller] = true;
                    queue.push_back(caller);
                }
            }
            found.insert(name);
        }
    }
    let mut out: Vec<String> = found.into_iter().collect();
    out.sort();
    out
}

/// Classify `llbc`'s own functions and mark its set complete, for a caller
/// that lowers one artefact on its own.  Callees from other artefacts count
/// as insensitive here; the linked translation registers them first.
pub fn ensure_stack_sensitive_fns(llbc: &Llbc) {
    if llbc.stack_sensitive_fns_complete() {
        return;
    }
    let found = discover_stack_sensitive_fns(llbc);
    llbc.register_stack_sensitive_fns(found);
    llbc.mark_stack_sensitive_fns_complete();
}
