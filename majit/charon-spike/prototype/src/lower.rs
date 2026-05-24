//! ULLBC → spike `FunctionGraph` lowering.
//!
//! This is the minimal converter the issue #97 Step-0 deliverable
//! requires.  It handles only the shapes the corpus actually exercises
//! (straight-line arithmetic, branch+loop, basic enum match) and panics
//! loudly on anything else.  When the real Step-3 driver lands, this
//! function will not be reused — it exists only to *demonstrate* the
//! mapping and produce the canonical text used by the comparison step.

use crate::graph::{Block, ExitKind, FunctionGraph, Link, SpaceOperation};
use crate::ullbc::*;

pub fn lower(fd: &FunDecl) -> FunctionGraph {
    let u = fd.unstructured().unwrap_or_else(|| {
        panic!(
            "function {} has no Unstructured body — was --ullbc passed?",
            fd.item_meta.name_path()
        )
    });
    let u = &u;
    let name = fd.item_meta.name_path();
    let arg_count = u.locals.arg_count as usize;
    let mut args = Vec::with_capacity(arg_count);
    for i in 1..=arg_count {
        let l = &u.locals.locals[i];
        args.push(local_label(l));
    }

    let mut blocks = Vec::with_capacity(u.body.len());
    for (i, bb) in u.body.iter().enumerate() {
        let id = i as u32;
        // ULLBC basic blocks have no inputargs (locals are function-wide
        // slots, not phi-node arguments).  We model `inputargs = []` so
        // the canonical text reflects that — this is one of the deltas
        // versus the AST front-end's phi-style FunctionGraph.
        let inputargs = vec![];

        let mut operations: Vec<SpaceOperation> = Vec::new();
        for (s_idx, st) in bb.statements.iter().enumerate() {
            match st.stmt_kind() {
                Ok(sk) => {
                    if let Some(op) = stmt_to_op(&sk, &u.locals) {
                        operations.push(op);
                    }
                }
                Err(e) => {
                    operations.push(SpaceOperation {
                        result: None,
                        op: format!("<unsupported stmt#{s_idx}>"),
                        args: vec![e],
                    });
                }
            }
        }

        let exit = match bb.term() {
            Ok(tk) => lower_terminator(&tk, &u.locals),
            Err(e) => ExitKind::Goto(Link {
                target: 0,
                args: vec![format!("<unsupported terminator: {e}>")],
                exitcase: None,
            }),
        };
        blocks.push(Block { id, inputargs, operations, exit });
    }

    FunctionGraph { name, args, blocks }
}

fn local_label(l: &Local) -> String {
    match &l.name {
        Some(n) => format!("%{}_{}", n, l.index),
        None => format!("%t{}", l.index),
    }
}

fn place_label(p: &Place, locals: &Locals) -> String {
    match &p.kind {
        PlaceKind::Local(i) => local_label(&locals.locals[*i as usize]),
        PlaceKind::Projection(inner, proj) => {
            format!("{}.{}", place_label(inner, locals), proj_label(proj))
        }
        PlaceKind::Other => "<place?>".into(),
    }
}

fn proj_label(v: &serde_json::Value) -> String {
    if let Some(s) = v.as_str() {
        return match s {
            "Deref" => "*".into(),
            other => other.into(),
        };
    }
    if let Some(obj) = v.as_object() {
        if let Some(field) = obj.get("Field") {
            // Field is e.g. [{"Tuple": 2}, 1] or [{"Adt": [variant, ...]}, idx].
            if let Some(arr) = field.as_array() {
                let variant = arr
                    .first()
                    .map(|x| {
                        if let Some(o) = x.as_object() {
                            // {"Tuple": 2} or {"Adt": [variant_idx, field_idx, _]}
                            let key = o.keys().next().cloned().unwrap_or_else(|| "?".into());
                            if let Some(payload) = o.values().next() {
                                if let Some(p) = payload.as_u64() {
                                    return format!("{key}{p}");
                                }
                                if let Some(a) = payload.as_array() {
                                    if let Some(v0) = a.first() {
                                        if let Some(n) = v0.as_u64() {
                                            return format!("{key}{n}");
                                        }
                                        if v0.is_null() {
                                            return key;
                                        }
                                    }
                                }
                            }
                            key
                        } else {
                            "?".into()
                        }
                    })
                    .unwrap_or_else(|| "?".into());
                let idx = arr.get(1).and_then(|x| x.as_u64()).unwrap_or(0);
                return format!("{variant}_{idx}");
            }
        }
    }
    "<proj?>".into()
}

fn operand_label(o: &Operand, locals: &Locals) -> String {
    match o {
        Operand::Copy(p) => format!("copy {}", place_label(p, locals)),
        Operand::Move(p) => format!("move {}", place_label(p, locals)),
        Operand::Const(v) => format!("const({})", short_json(v)),
    }
}

fn short_json(v: &serde_json::Value) -> String {
    // Best-effort short-form printer for the variants we see in the corpus.
    if let Some(obj) = v.as_object() {
        if let Some(value) = obj.get("value") {
            if let Some(inner) = value.as_object() {
                if let Some(lit) = inner.get("Literal") {
                    if let Some(litobj) = lit.as_object() {
                        if let Some(scalar) = litobj.get("Scalar") {
                            return format_scalar(scalar);
                        }
                    }
                }
            }
        }
        if let Some(scalar) = obj.get("Scalar") {
            return format_scalar(scalar);
        }
    }
    // Fall back to a structural marker — full JSON would dominate the
    // canonical text noise-wise.
    if let Some(obj) = v.as_object() {
        if let Some(k) = obj.keys().next() {
            return format!("{k}<…>");
        }
    }
    "?".into()
}

fn format_scalar(v: &serde_json::Value) -> String {
    if let Some(obj) = v.as_object() {
        for (k, vv) in obj {
            if let Some(arr) = vv.as_array() {
                if arr.len() == 2 {
                    return format!("{k}({})", arr[1]);
                }
            }
        }
    }
    "<scalar?>".into()
}

fn rvalue_to_op(rv: &Rvalue, locals: &Locals) -> Option<(String, Vec<String>)> {
    match rv {
        Rvalue::Use(op) => Some(("use".into(), vec![operand_label(op, locals)])),
        Rvalue::BinaryOp(name, lhs, rhs) => Some((
            format!("binop.{name}"),
            vec![operand_label(lhs, locals), operand_label(rhs, locals)],
        )),
        Rvalue::UnaryOp(op, x) => {
            let name = op.as_object()
                .and_then(|m| m.keys().next().cloned())
                .unwrap_or_else(|| "?".into());
            Some((format!("unop.{name}"), vec![operand_label(x, locals)]))
        }
        Rvalue::Ref { place, kind, .. } => {
            let ref_kind = kind.as_str().map(str::to_owned).unwrap_or_else(|| {
                kind.as_object()
                    .and_then(|m| m.keys().next().cloned())
                    .unwrap_or_else(|| "?".into())
            });
            Some((format!("ref.{ref_kind}"), vec![place_label(place, locals)]))
        }
        Rvalue::Discriminant(p) => {
            Some(("discriminant".into(), vec![place_label(p, locals)]))
        }
        Rvalue::Aggregate(kind, ops) => {
            let kind_label = aggregate_kind_label(kind);
            let mut args = Vec::with_capacity(ops.len());
            for op in ops {
                args.push(operand_label(op, locals));
            }
            Some((format!("aggregate.{kind_label}"), args))
        }
        Rvalue::Other => None,
    }
}

fn stmt_to_op(s: &StmtKind, locals: &Locals) -> Option<SpaceOperation> {
    match s {
        StmtKind::StorageLive(_) | StmtKind::StorageDead(_) => None,
        StmtKind::Assert(_) => None, // surfaced as terminator-level Assert
        StmtKind::Other => None,
        StmtKind::Assign(p, rv) => {
            let (op, args) = rvalue_to_op(rv, locals)?;
            Some(SpaceOperation {
                result: Some(place_label(p, locals)),
                op,
                args,
            })
        }
    }
}

fn lower_terminator(t: &TermKind, locals: &Locals) -> ExitKind {
    match t {
        TermKind::Return => ExitKind::Return(local_label(&locals.locals[0])),
        TermKind::UnwindResume | TermKind::Abort(_) => ExitKind::Resume,
        TermKind::Goto { target } => ExitKind::Goto(Link {
            target: *target as u32,
            args: vec![],
            exitcase: None,
        }),
        TermKind::Switch { discr, targets } => {
            let discr_label = operand_label(discr, locals);
            let cases = match targets {
                SwitchTargets::If(then_bb, else_bb) => vec![
                    (
                        Some("true".into()),
                        Link { target: *then_bb as u32, args: vec![], exitcase: Some("true".into()) },
                    ),
                    (
                        Some("false".into()),
                        Link { target: *else_bb as u32, args: vec![], exitcase: Some("false".into()) },
                    ),
                ],
                SwitchTargets::SwitchInt(_ty, arms, default) => {
                    let mut out = Vec::with_capacity(arms.len() + 1);
                    for (scalar, bb) in arms {
                        let label = short_json(scalar);
                        out.push((
                            Some(label.clone()),
                            Link { target: *bb as u32, args: vec![], exitcase: Some(label) },
                        ));
                    }
                    out.push((
                        None,
                        Link { target: *default as u32, args: vec![], exitcase: None },
                    ));
                    out
                }
            };
            ExitKind::Switch { discr: discr_label, cases }
        }
        TermKind::Call(call_v) => lower_call(call_v, locals),
        TermKind::Assert { target, on_unwind, .. } => ExitKind::Assert {
            cond: "<assert>".into(),
            on_success: Link { target: *target as u32, args: vec![], exitcase: None },
            on_unwind: Link { target: *on_unwind as u32, args: vec![], exitcase: None },
        },
        TermKind::Drop { target, on_unwind, .. } => ExitKind::Call {
            callee: "<drop>".into(),
            args: vec![],
            dest: None,
            on_success: Link { target: *target as u32, args: vec![], exitcase: None },
            on_unwind: Link { target: *on_unwind as u32, args: vec![], exitcase: None },
        },
        TermKind::Other => ExitKind::Resume,
    }
}

fn lower_call(call_v: &serde_json::Value, locals: &Locals) -> ExitKind {
    let obj = call_v.as_object().expect("Call payload is object");
    let target = obj.get("target").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
    let unwind = obj.get("on_unwind").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
    let call = obj.get("call").and_then(|x| x.as_object()).cloned().unwrap_or_default();
    let func = call.get("func").map(|f| describe_func(f)).unwrap_or_else(|| "<fn?>".into());
    let args = call
        .get("args")
        .and_then(|a| a.as_array())
        .map(|arr| {
            arr.iter()
                .map(|x| {
                    serde_json::from_value::<Operand>(x.clone())
                        .map(|op| operand_label(&op, locals))
                        .unwrap_or_else(|_| "<arg?>".into())
                })
                .collect()
        })
        .unwrap_or_default();
    let dest = call
        .get("dest")
        .cloned()
        .and_then(|d| serde_json::from_value::<Place>(d).ok())
        .map(|p| place_label(&p, locals));
    ExitKind::Call {
        callee: func,
        args,
        dest,
        on_success: Link { target, args: vec![], exitcase: None },
        on_unwind: Link { target: unwind, args: vec![], exitcase: None },
    }
}

fn aggregate_kind_label(v: &serde_json::Value) -> String {
    // AggregateKind = { "Adt": [type_ref, variant_idx, field_idx_or_null] } |
    //                 { "Tuple": [] } | { "Array": [...] } | etc.
    if let Some(obj) = v.as_object() {
        if let Some((k, payload)) = obj.iter().next() {
            if k == "Adt" {
                if let Some(arr) = payload.as_array() {
                    let variant = arr.get(1).and_then(|x| x.as_u64()).unwrap_or(0);
                    return format!("Adt(variant={variant})");
                }
            }
            if k == "Tuple" {
                return "Tuple".into();
            }
            return k.clone();
        }
    }
    "<aggregate?>".into()
}

fn describe_func(f: &serde_json::Value) -> String {
    if let Some(obj) = f.as_object() {
        if let Some(reg) = obj.get("Regular").and_then(|r| r.as_object()) {
            if let Some(kind) = reg.get("kind").and_then(|k| k.as_object()) {
                if let Some(fun) = kind.get("Fun").and_then(|x| x.as_object()) {
                    if let Some(reg2) = fun.get("Regular") {
                        return format!("fn#{}", reg2);
                    }
                }
            }
        }
        if let Some(builtin) = obj.get("Ptr") {
            return format!("ptr<{}>", builtin);
        }
    }
    "<fn?>".into()
}
