//! AST front-end: build semantic graphs from Rust source.
//!
//! RPython equivalent: flowspace/ — converts source to Block/Link/Variable/SpaceOperation.
//! This module lowers syn AST nodes into MajitGraph ops with proper data flow (ValueId linking).

use quote::quote;
use serde::{Deserialize, Serialize};
use syn::{Item, ItemFn};

use crate::graph::{BasicBlockId, MajitGraph, OpKind, Terminator, ValueId, ValueType};
use crate::ParsedInterpreter;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstGraphOptions {
    pub max_summary_len: usize,
}

impl Default for AstGraphOptions {
    fn default() -> Self {
        Self {
            max_summary_len: 160,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticFunction {
    pub name: String,
    pub graph: MajitGraph,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SemanticProgram {
    pub functions: Vec<SemanticFunction>,
}

pub fn build_semantic_program(parsed: &ParsedInterpreter) -> SemanticProgram {
    build_semantic_program_with_options(parsed, &AstGraphOptions::default())
}

pub fn build_semantic_program_with_options(
    parsed: &ParsedInterpreter,
    options: &AstGraphOptions,
) -> SemanticProgram {
    let mut functions = Vec::new();

    for item in &parsed.file.items {
        match item {
            Item::Fn(func) => functions.push(build_function_graph(func, options)),
            Item::Impl(impl_block) => {
                for item in &impl_block.items {
                    if let syn::ImplItem::Fn(method) = item {
                        let fake_fn = ItemFn {
                            attrs: method.attrs.clone(),
                            vis: syn::Visibility::Inherited,
                            sig: method.sig.clone(),
                            block: Box::new(method.block.clone()),
                        };
                        functions.push(build_function_graph(&fake_fn, options));
                    }
                }
            }
            _ => {}
        }
    }

    SemanticProgram { functions }
}

fn build_function_graph(func: &ItemFn, options: &AstGraphOptions) -> SemanticFunction {
    let mut graph = MajitGraph::new(func.sig.ident.to_string());
    let entry = graph.entry;

    // Register function parameters as Input ops (RPython: Block.inputargs)
    for param in &func.sig.inputs {
        if let syn::FnArg::Typed(pat_type) = param {
            let name = quote!(#pat_type.pat).to_string().replace(' ', "");
            graph.push_op(
                entry,
                OpKind::Input {
                    name,
                    ty: ValueType::Unknown,
                },
                true,
            );
        }
    }

    // Lower function body
    for stmt in &func.block.stmts {
        lower_stmt(&mut graph, entry, stmt, options);
    }

    // Default terminator if none was set
    if graph.block(entry).terminator == Terminator::Unreachable {
        graph.set_terminator(entry, Terminator::Return(None));
    }

    SemanticFunction {
        name: func.sig.ident.to_string(),
        graph,
    }
}

// ── Statement lowering ──────────────────────────────────────────

/// Public entry point for lowering a single statement into a graph.
/// Used by the graph-based classifier in lib.rs to analyze resolved method bodies.
pub fn lower_stmt_pub(graph: &mut MajitGraph, block: BasicBlockId, stmt: &syn::Stmt) {
    lower_stmt(graph, block, stmt, &AstGraphOptions::default());
}

fn lower_stmt(
    graph: &mut MajitGraph,
    block: BasicBlockId,
    stmt: &syn::Stmt,
    options: &AstGraphOptions,
) {
    match stmt {
        syn::Stmt::Expr(expr, _) => {
            lower_expr(graph, block, expr, options);
        }
        syn::Stmt::Local(local) => {
            if let Some(init) = &local.init {
                lower_expr(graph, block, &init.expr, options);
            }
        }
        syn::Stmt::Macro(_) => {
            let summary = truncate(&quote!(#stmt).to_string(), options.max_summary_len);
            graph.push_op(block, OpKind::Unknown { summary }, false);
        }
        syn::Stmt::Item(_) => {}
    }
}

// ── Expression lowering (returns ValueId for data flow) ──────────

/// Lower an expression and return the ValueId of its result.
///
/// RPython equivalent: FlowContext recording SpaceOperations.
/// Each expression becomes one or more ops with connected ValueIds.
fn lower_expr(
    graph: &mut MajitGraph,
    block: BasicBlockId,
    expr: &syn::Expr,
    options: &AstGraphOptions,
) -> Option<ValueId> {
    match expr {
        // ── receiver.field ──
        syn::Expr::Field(field) => {
            let base = lower_expr(graph, block, &field.base, options)
                .unwrap_or_else(|| graph.alloc_value());
            let field_name = member_name(&field.member);
            graph.push_op(
                block,
                OpKind::FieldRead {
                    base,
                    field: field_name,
                    ty: ValueType::Unknown,
                },
                true,
            )
        }

        // ── base[index] ──
        syn::Expr::Index(idx) => {
            let base = lower_expr(graph, block, &idx.expr, options)
                .unwrap_or_else(|| graph.alloc_value());
            let index = lower_expr(graph, block, &idx.index, options)
                .unwrap_or_else(|| graph.alloc_value());
            graph.push_op(
                block,
                OpKind::ArrayRead {
                    base,
                    index,
                    item_ty: ValueType::Unknown,
                },
                true,
            )
        }

        // ── lhs = rhs ──
        syn::Expr::Assign(assign) => {
            let value = lower_expr(graph, block, &assign.right, options)
                .unwrap_or_else(|| graph.alloc_value());

            match &*assign.left {
                syn::Expr::Field(field) => {
                    let base = lower_expr(graph, block, &field.base, options)
                        .unwrap_or_else(|| graph.alloc_value());
                    let field_name = member_name(&field.member);
                    graph.push_op(
                        block,
                        OpKind::FieldWrite {
                            base,
                            field: field_name,
                            value,
                            ty: ValueType::Unknown,
                        },
                        false,
                    );
                }
                syn::Expr::Index(idx) => {
                    let base = lower_expr(graph, block, &idx.expr, options)
                        .unwrap_or_else(|| graph.alloc_value());
                    let index = lower_expr(graph, block, &idx.index, options)
                        .unwrap_or_else(|| graph.alloc_value());
                    graph.push_op(
                        block,
                        OpKind::ArrayWrite {
                            base,
                            index,
                            value,
                            item_ty: ValueType::Unknown,
                        },
                        false,
                    );
                }
                _ => {
                    // Generic assignment — value already lowered
                }
            }
            None
        }

        // ── function call ──
        syn::Expr::Call(call) => {
            let args: Vec<ValueId> = call
                .args
                .iter()
                .filter_map(|a| lower_expr(graph, block, a, options))
                .collect();
            let target = truncate(&quote!(#call.func).to_string(), 80);
            graph.push_op(
                block,
                OpKind::Call {
                    target,
                    args,
                    result_ty: ValueType::Unknown,
                },
                true,
            )
        }

        // ── method call ──
        syn::Expr::MethodCall(mc) => {
            let mut args = Vec::new();
            if let Some(recv) = lower_expr(graph, block, &mc.receiver, options) {
                args.push(recv);
            }
            for a in &mc.args {
                if let Some(v) = lower_expr(graph, block, a, options) {
                    args.push(v);
                }
            }
            graph.push_op(
                block,
                OpKind::Call {
                    target: mc.method.to_string(),
                    args,
                    result_ty: ValueType::Unknown,
                },
                true,
            )
        }

        // ── if/else ──
        syn::Expr::If(if_expr) => {
            let cond = lower_expr(graph, block, &if_expr.cond, options)
                .unwrap_or_else(|| graph.alloc_value());
            graph.push_op(block, OpKind::GuardTrue { cond }, false);
            for stmt in &if_expr.then_branch.stmts {
                lower_stmt(graph, block, stmt, options);
            }
            if let Some((_, else_branch)) = &if_expr.else_branch {
                lower_expr(graph, block, else_branch, options);
            }
            None
        }

        // ── return ──
        syn::Expr::Return(ret) => {
            let val = ret
                .expr
                .as_ref()
                .and_then(|e| lower_expr(graph, block, e, options));
            graph.set_terminator(block, Terminator::Return(val));
            None
        }

        // ── block { stmts } ──
        syn::Expr::Block(blk) => {
            let mut last = None;
            for stmt in &blk.block.stmts {
                lower_stmt(graph, block, stmt, options);
                if let syn::Stmt::Expr(e, None) = stmt {
                    last = lower_expr(graph, block, e, options);
                }
            }
            last
        }

        // ── literals ──
        syn::Expr::Lit(lit) => {
            if let syn::Lit::Int(int_lit) = &lit.lit {
                if let Ok(v) = int_lit.base10_parse::<i64>() {
                    return graph.push_op(block, OpKind::ConstInt(v), true);
                }
            }
            let summary = truncate(&quote!(#expr).to_string(), options.max_summary_len);
            graph.push_op(block, OpKind::Unknown { summary }, true)
        }

        // ── path (variable reference) ──
        syn::Expr::Path(path) => {
            let name = quote!(#path).to_string().replace(' ', "");
            graph.push_op(
                block,
                OpKind::Input {
                    name,
                    ty: ValueType::Unknown,
                },
                true,
            )
        }

        // ── reference &expr ──
        syn::Expr::Reference(r) => lower_expr(graph, block, &r.expr, options),

        // ── parenthesized (expr) ──
        syn::Expr::Paren(p) => lower_expr(graph, block, &p.expr, options),

        // ── unary !x, -x ──
        syn::Expr::Unary(u) => lower_expr(graph, block, &u.expr, options),

        // ── binary a + b ──
        syn::Expr::Binary(bin) => {
            let lhs = lower_expr(graph, block, &bin.left, options);
            let rhs = lower_expr(graph, block, &bin.right, options);
            let mut args = Vec::new();
            if let Some(l) = lhs {
                args.push(l);
            }
            if let Some(r) = rhs {
                args.push(r);
            }
            let op_name = quote!(#bin.op).to_string();
            graph.push_op(
                block,
                OpKind::Call {
                    target: op_name,
                    args,
                    result_ty: ValueType::Unknown,
                },
                true,
            )
        }

        // ── cast: expr as T ──
        syn::Expr::Cast(cast) => lower_expr(graph, block, &cast.expr, options),

        // ── match expr { arms } ──
        syn::Expr::Match(m) => {
            lower_expr(graph, block, &m.expr, options);
            for arm in &m.arms {
                lower_expr(graph, block, &arm.body, options);
            }
            None
        }

        // ── while/loop ──
        syn::Expr::While(w) => {
            lower_expr(graph, block, &w.cond, options);
            for stmt in &w.body.stmts {
                lower_stmt(graph, block, stmt, options);
            }
            None
        }
        syn::Expr::Loop(l) => {
            for stmt in &l.body.stmts {
                lower_stmt(graph, block, stmt, options);
            }
            None
        }
        syn::Expr::ForLoop(f) => {
            lower_expr(graph, block, &f.expr, options);
            for stmt in &f.body.stmts {
                lower_stmt(graph, block, stmt, options);
            }
            None
        }

        // ── break/continue ──
        syn::Expr::Break(b) => {
            if let Some(e) = &b.expr {
                lower_expr(graph, block, e, options);
            }
            None
        }
        syn::Expr::Continue(_) => None,

        // ── closure ──
        syn::Expr::Closure(c) => lower_expr(graph, block, &c.body, options),

        // ── tuple (a, b, c) ──
        syn::Expr::Tuple(t) => {
            let mut last = None;
            for e in &t.elems {
                last = lower_expr(graph, block, e, options);
            }
            last
        }

        // ── try expr? ──
        syn::Expr::Try(t) => lower_expr(graph, block, &t.expr, options),

        // ── fallback ──
        _ => {
            let summary = truncate(&quote!(#expr).to_string(), options.max_summary_len);
            graph.push_op(block, OpKind::Unknown { summary }, true)
        }
    }
}

// ── Helpers ──────────────────────────────────────────────────────

fn member_name(member: &syn::Member) -> String {
    match member {
        syn::Member::Named(ident) => ident.to_string(),
        syn::Member::Unnamed(idx) => idx.index.to_string(),
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("{}...", &s[..max])
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_function_with_data_flow() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example(x: i64, y: i64) -> i64 {
                let z = x + y;
                z
            }
        "#,
        );
        let program = build_semantic_program(&parsed);
        assert_eq!(program.functions.len(), 1);
        let graph = &program.functions[0].graph;
        // Should have Input ops for params + ops for body
        assert!(graph.block(graph.entry).ops.len() >= 2);
    }

    #[test]
    fn lowers_field_access_with_data_flow() {
        let parsed = crate::parse::parse_source(
            r#"
            struct S { x: i64 }
            fn read_field(s: S) -> i64 {
                s.x
            }
        "#,
        );
        let program = build_semantic_program(&parsed);
        let graph = &program.functions[0].graph;
        let ops = &graph.block(graph.entry).ops;
        // Should contain a FieldRead op
        assert!(
            ops.iter().any(|op| matches!(&op.kind, OpKind::FieldRead { field, .. } if field == "x")),
            "expected FieldRead for 'x', got {:?}",
            ops
        );
    }

    #[test]
    fn lowers_method_call_with_args() {
        let parsed = crate::parse::parse_source(
            r#"
            fn call_example(v: Vec<i64>) {
                v.push(42);
            }
        "#,
        );
        let program = build_semantic_program(&parsed);
        let graph = &program.functions[0].graph;
        let ops = &graph.block(graph.entry).ops;
        assert!(
            ops.iter().any(|op| matches!(&op.kind, OpKind::Call { target, .. } if target == "push")),
            "expected Call to 'push', got {:?}",
            ops
        );
    }

    #[test]
    fn builds_impl_methods() {
        let parsed = crate::parse::parse_source(
            r#"
            struct Foo;
            impl Foo {
                fn bar(&self) { }
                fn baz(&self, x: i64) -> i64 { x }
            }
        "#,
        );
        let program = build_semantic_program(&parsed);
        assert_eq!(program.functions.len(), 2);
        assert_eq!(program.functions[0].name, "bar");
        assert_eq!(program.functions[1].name, "baz");
    }
}
