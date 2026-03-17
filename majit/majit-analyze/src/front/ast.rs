//! AST front-end scaffold for building semantic graphs from Rust source.
//!
//! This is a shell builder only.  It intentionally does not replace the
//! current heuristic analyzer path yet.

use quote::quote;
use serde::{Deserialize, Serialize};
use syn::{Item, ItemFn};

use crate::{MajitGraph, ParsedInterpreter, Terminator};

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
            Item::Fn(func) => functions.push(build_function_shell(func, options)),
            Item::Impl(impl_block) => {
                for item in &impl_block.items {
                    if let syn::ImplItem::Fn(method) = item {
                        let fake_fn = ItemFn {
                            attrs: method.attrs.clone(),
                            vis: syn::Visibility::Inherited,
                            sig: method.sig.clone(),
                            block: Box::new(method.block.clone()),
                        };
                        functions.push(build_function_shell(&fake_fn, options));
                    }
                }
            }
            _ => {}
        }
    }

    SemanticProgram { functions }
}

fn build_function_shell(func: &ItemFn, options: &AstGraphOptions) -> SemanticFunction {
    let mut graph = MajitGraph::new(func.sig.ident.to_string());
    let entry = graph.entry;

    // Lower function body statements into semantic ops.
    // This replaces the old body_summary string heuristic with actual AST traversal.
    for stmt in &func.block.stmts {
        lower_stmt_to_graph(&mut graph, entry, stmt, options);
    }

    // If no terminator was set by lowering, add Return.
    if graph.block(entry).terminator == Terminator::Unreachable {
        graph.set_terminator(entry, Terminator::Return(None));
    }

    SemanticFunction {
        name: func.sig.ident.to_string(),
        graph,
    }
}

/// Lower a single statement to semantic graph ops.
///
/// RPython equivalent: FlowContext.handle_bytecode() — converts source
/// constructs into SpaceOperations within a Block.
fn lower_stmt_to_graph(
    graph: &mut MajitGraph,
    block: crate::BasicBlockId,
    stmt: &syn::Stmt,
    options: &AstGraphOptions,
) {
    use crate::graph::{OpKind, ValueType};

    match stmt {
        syn::Stmt::Expr(expr, _) => {
            lower_expr_to_graph(graph, block, expr, options);
        }
        syn::Stmt::Macro(_) => {
            let summary = quote!(#stmt).to_string();
            let summary = truncate(&summary, options.max_summary_len);
            graph.push_op(block, OpKind::Unknown { summary }, false);
        }
        syn::Stmt::Local(local) => {
            // let x = expr;
            if let Some(init) = &local.init {
                lower_expr_to_graph(graph, block, &init.expr, options);
            }
        }
        syn::Stmt::Item(_) => {
            // Nested items (fn, struct, etc.) — skip
        }
    }
}

/// Lower an expression to semantic graph ops.
fn lower_expr_to_graph(
    graph: &mut MajitGraph,
    block: crate::BasicBlockId,
    expr: &syn::Expr,
    options: &AstGraphOptions,
) {
    use crate::graph::{OpKind, ValueType};

    match expr {
        // field.read: receiver.field
        syn::Expr::Field(field) => {
            let base = graph.alloc_value();
            let field_name = match &field.member {
                syn::Member::Named(ident) => ident.to_string(),
                syn::Member::Unnamed(idx) => idx.index.to_string(),
            };
            graph.push_op(
                block,
                OpKind::FieldRead {
                    base,
                    field: field_name,
                    ty: ValueType::Unknown,
                },
                true,
            );
        }
        // array[index]
        syn::Expr::Index(idx) => {
            let base = graph.alloc_value();
            let index = graph.alloc_value();
            graph.push_op(
                block,
                OpKind::ArrayRead {
                    base,
                    index,
                    item_ty: ValueType::Unknown,
                },
                true,
            );
        }
        // assignment: lhs = rhs
        syn::Expr::Assign(assign) => {
            match &*assign.left {
                syn::Expr::Field(field) => {
                    let base = graph.alloc_value();
                    let value = graph.alloc_value();
                    let field_name = match &field.member {
                        syn::Member::Named(ident) => ident.to_string(),
                        syn::Member::Unnamed(idx) => idx.index.to_string(),
                    };
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
                syn::Expr::Index(_idx_expr) => {
                    let base = graph.alloc_value();
                    let index = graph.alloc_value();
                    let value = graph.alloc_value();
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
                    // Generic assignment — lower RHS
                    lower_expr_to_graph(graph, block, &assign.right, options);
                }
            }
        }
        // function/method call
        syn::Expr::Call(call) => {
            let target = quote!(#call.func).to_string();
            let target = truncate(&target, 80);
            graph.push_op(
                block,
                OpKind::Call {
                    target,
                    args: Vec::new(),
                    result_ty: ValueType::Unknown,
                },
                true,
            );
        }
        syn::Expr::MethodCall(mc) => {
            let target = mc.method.to_string();
            graph.push_op(
                block,
                OpKind::Call {
                    target,
                    args: Vec::new(),
                    result_ty: ValueType::Unknown,
                },
                true,
            );
        }
        // if/else → Branch
        syn::Expr::If(if_expr) => {
            let cond = graph.alloc_value();
            graph.push_op(block, OpKind::GuardTrue { cond }, false);
            // Lower then/else bodies into current block (simplified)
            for stmt in &if_expr.then_branch.stmts {
                lower_stmt_to_graph(graph, block, stmt, options);
            }
        }
        // return
        syn::Expr::Return(ret) => {
            let val = ret.expr.as_ref().map(|_| graph.alloc_value());
            graph.set_terminator(block, Terminator::Return(val));
        }
        // block expression
        syn::Expr::Block(blk) => {
            for stmt in &blk.block.stmts {
                lower_stmt_to_graph(graph, block, stmt, options);
            }
        }
        // Fallback: record as Unknown
        _ => {
            let summary = quote!(#expr).to_string();
            let summary = truncate(&summary, options.max_summary_len);
            graph.push_op(block, OpKind::Unknown { summary }, false);
        }
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
    fn builds_function_shells_for_free_functions() {
        let parsed = crate::parse::parse_source(
            r#"
            fn mainloop() { let x = 1; }
            fn helper() { return; }
        "#,
        );
        let program = build_semantic_program(&parsed);
        assert_eq!(program.functions.len(), 2);
        assert_eq!(program.functions[0].graph.blocks.len(), 1);
    }
}
