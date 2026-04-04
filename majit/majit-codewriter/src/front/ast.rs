//! AST front-end: build semantic graphs from Rust source.
//!
//! RPython equivalent: flowspace/ — converts source to Block/Link/Variable/SpaceOperation.
//! This module lowers syn AST nodes into FunctionGraph ops with proper data flow (ValueId linking).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use syn::{Item, ItemFn};

use crate::ParsedInterpreter;
use crate::model::{
    BlockId, CallTarget, FunctionGraph, OpKind, Terminator, UnknownKind, ValueId, ValueType,
};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AstGraphOptions;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticFunction {
    pub name: String,
    pub graph: FunctionGraph,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SemanticProgram {
    pub functions: Vec<SemanticFunction>,
    /// RPython: known struct types for `get_type_flag(ARRAY.OF)` → FLAG_STRUCT.
    /// If the element type of an array is in this set, the array is
    /// array-of-structs (inline struct elements, not pointers).
    pub known_struct_names: std::collections::HashSet<String>,
}

pub fn build_semantic_program(parsed: &ParsedInterpreter) -> SemanticProgram {
    build_semantic_program_with_options(parsed, &AstGraphOptions::default())
}

pub fn build_semantic_program_from_parsed_files(
    parsed_files: &[ParsedInterpreter],
) -> SemanticProgram {
    build_semantic_program_from_parsed_files_with_options(parsed_files, &AstGraphOptions::default())
}

pub fn build_semantic_program_with_options(
    parsed: &ParsedInterpreter,
    options: &AstGraphOptions,
) -> SemanticProgram {
    let mut functions = Vec::new();
    let mut known_struct_names = std::collections::HashSet::new();

    for item in &parsed.file.items {
        match item {
            Item::Fn(func) => functions.push(build_function_graph(func, options, None)),
            Item::Impl(impl_block) => {
                let self_ty_root = type_root_ident(&impl_block.self_ty);
                for item in &impl_block.items {
                    if let syn::ImplItem::Fn(method) = item {
                        let fake_fn = ItemFn {
                            attrs: method.attrs.clone(),
                            vis: syn::Visibility::Inherited,
                            sig: method.sig.clone(),
                            block: Box::new(method.block.clone()),
                        };
                        functions.push(build_function_graph(
                            &fake_fn,
                            options,
                            self_ty_root.clone(),
                        ));
                    }
                }
            }
            // RPython: collect struct type names for get_type_flag(ARRAY.OF).
            // If an array's element type is a known struct, the array gets
            // FLAG_STRUCT instead of FLAG_POINTER.
            Item::Struct(s) => {
                known_struct_names.insert(s.ident.to_string());
            }
            _ => {}
        }
    }

    SemanticProgram {
        functions,
        known_struct_names,
    }
}

pub fn build_semantic_program_from_parsed_files_with_options(
    parsed_files: &[ParsedInterpreter],
    options: &AstGraphOptions,
) -> SemanticProgram {
    let mut program = SemanticProgram::default();
    for parsed in parsed_files {
        let sub = build_semantic_program_with_options(parsed, options);
        program.functions.extend(sub.functions);
        program.known_struct_names.extend(sub.known_struct_names);
    }
    program
}

/// Public entry for building a graph from a single function AST node.
/// Lower a standalone expression into an existing graph.
/// Used to build semantic graphs from opcode match arm bodies.
pub fn lower_expr_into_graph(graph: &mut FunctionGraph, expr: &syn::Expr) {
    let mut block = graph.startblock;
    let ctx = GraphBuildContext::default();
    let result = lower_expr(graph, &mut block, expr, &AstGraphOptions::default(), &ctx);
    if let Some(val) = result {
        graph.set_terminator(block, crate::model::Terminator::Return(Some(val)));
    } else {
        graph.set_terminator(block, crate::model::Terminator::Return(None));
    }
}

pub fn build_function_graph_pub(func: &ItemFn) -> SemanticFunction {
    build_function_graph(func, &AstGraphOptions::default(), None)
}

pub fn build_function_graph_with_self_ty_pub(
    func: &ItemFn,
    self_ty_root: Option<String>,
) -> SemanticFunction {
    build_function_graph(func, &AstGraphOptions::default(), self_ty_root)
}

#[derive(Debug, Clone, Default)]
struct GraphBuildContext {
    local_type_roots: HashMap<String, String>,
    /// RPython: ARRAY element type identity — maps variable name to the
    /// element type of its array (e.g. "arr" → "Point" for `arr: Vec<Point>`).
    /// This is the Rust equivalent of RPython's `GcArray(T)` where T is the
    /// element type that determines the ARRAY identity for `cpu.arraydescrof()`.
    local_element_types: HashMap<String, String>,
}

fn build_function_graph(
    func: &ItemFn,
    options: &AstGraphOptions,
    self_ty_root: Option<String>,
) -> SemanticFunction {
    let mut graph = FunctionGraph::new(func.sig.ident.to_string());
    let mut entry = graph.startblock;
    let mut ctx = GraphBuildContext::default();

    // Register function parameters as Input ops (RPython: Block.inputargs)
    for param in &func.sig.inputs {
        match param {
            syn::FnArg::Receiver(_) => {
                if let Some(self_ty_root) = &self_ty_root {
                    ctx.local_type_roots
                        .insert("self".to_string(), self_ty_root.clone());
                }
                if let Some(vid) = graph.push_op(
                    entry,
                    OpKind::Input {
                        name: "self".to_string(),
                        ty: ValueType::Unknown,
                    },
                    true,
                ) {
                    graph.name_value(vid, "self".to_string());
                }
            }
            syn::FnArg::Typed(pat_type) => {
                let name = canonical_pat_name(&pat_type.pat);
                if let Some(type_root) = type_root_ident(&pat_type.ty) {
                    ctx.local_type_roots.insert(name.clone(), type_root);
                }
                if let Some(elem_type) = array_element_type(&pat_type.ty) {
                    ctx.local_element_types.insert(name.clone(), elem_type);
                }
                if let Some(vid) = graph.push_op(
                    entry,
                    OpKind::Input {
                        name: name.clone(),
                        ty: ValueType::Unknown,
                    },
                    true,
                ) {
                    graph.name_value(vid, name);
                }
            }
        }
    }

    // Lower function body
    for stmt in &func.block.stmts {
        lower_stmt(&mut graph, &mut entry, stmt, options, &ctx);
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
pub fn lower_stmt_pub(graph: &mut FunctionGraph, block: BlockId, stmt: &syn::Stmt) {
    let mut block = block;
    lower_stmt(
        graph,
        &mut block,
        stmt,
        &AstGraphOptions::default(),
        &GraphBuildContext::default(),
    );
}

fn lower_stmt(
    graph: &mut FunctionGraph,
    block: &mut BlockId,
    stmt: &syn::Stmt,
    options: &AstGraphOptions,
    ctx: &GraphBuildContext,
) {
    match stmt {
        syn::Stmt::Expr(expr, _) => {
            lower_expr(graph, block, expr, options, ctx);
        }
        syn::Stmt::Local(local) => {
            if let Some(init) = &local.init {
                let result = lower_expr(graph, block, &init.expr, options, ctx);
                // Record variable name (RPython Variable._name)
                if let Some(vid) = result {
                    if let syn::Pat::Ident(pat_ident) = &local.pat {
                        graph.name_value(vid, pat_ident.ident.to_string());
                    }
                }
            }
        }
        syn::Stmt::Macro(_) => {
            graph.push_op(
                *block,
                OpKind::Unknown {
                    kind: UnknownKind::MacroStmt,
                },
                false,
            );
        }
        syn::Stmt::Item(_) => {}
    }
}

// ── Expression lowering (block-splitting for control flow) ───────

/// Lower an expression, potentially splitting blocks for control flow.
///
/// RPython equivalent: FlowContext.handle_bytecode() + guessbool().
/// When `if`/`match` is encountered, the current block is terminated
/// with a Branch, new blocks are created for each arm, and `block`
/// is updated to the merge/continuation block.
fn lower_expr(
    graph: &mut FunctionGraph,
    block: &mut BlockId,
    expr: &syn::Expr,
    options: &AstGraphOptions,
    ctx: &GraphBuildContext,
) -> Option<ValueId> {
    match expr {
        // ── receiver.field ──
        syn::Expr::Field(field) => {
            let base = lower_expr(graph, block, &field.base, options, ctx)
                .unwrap_or_else(|| graph.alloc_value());
            let field_name = member_name(&field.member);
            graph.push_op(
                *block,
                OpKind::FieldRead {
                    base,
                    field: crate::model::FieldDescriptor::new(
                        field_name,
                        receiver_type_root(&field.base, ctx),
                    ),
                    ty: ValueType::Unknown,
                },
                true,
            )
        }

        // ── base[index] ──
        syn::Expr::Index(idx) => {
            let base = lower_expr(graph, block, &idx.expr, options, ctx)
                .unwrap_or_else(|| graph.alloc_value());
            let index = lower_expr(graph, block, &idx.index, options, ctx)
                .unwrap_or_else(|| graph.alloc_value());
            let array_type_id = array_type_id_from_expr(&idx.expr, ctx);
            graph.push_op(
                *block,
                OpKind::ArrayRead {
                    base,
                    index,
                    item_ty: ValueType::Unknown,
                    array_type_id,
                },
                true,
            )
        }

        // ── lhs = rhs ──
        syn::Expr::Assign(assign) => {
            let value = lower_expr(graph, block, &assign.right, options, ctx)
                .unwrap_or_else(|| graph.alloc_value());

            match &*assign.left {
                syn::Expr::Field(field) => {
                    let base = lower_expr(graph, block, &field.base, options, ctx)
                        .unwrap_or_else(|| graph.alloc_value());
                    let field_name = member_name(&field.member);
                    graph.push_op(
                        *block,
                        OpKind::FieldWrite {
                            base,
                            field: crate::model::FieldDescriptor::new(
                                field_name,
                                receiver_type_root(&field.base, ctx),
                            ),
                            value,
                            ty: ValueType::Unknown,
                        },
                        false,
                    );
                }
                syn::Expr::Index(idx) => {
                    let base = lower_expr(graph, block, &idx.expr, options, ctx)
                        .unwrap_or_else(|| graph.alloc_value());
                    let index = lower_expr(graph, block, &idx.index, options, ctx)
                        .unwrap_or_else(|| graph.alloc_value());
                    let array_type_id = array_type_id_from_expr(&idx.expr, ctx);
                    graph.push_op(
                        *block,
                        OpKind::ArrayWrite {
                            base,
                            index,
                            value,
                            item_ty: ValueType::Unknown,
                            array_type_id,
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
                .filter_map(|a| lower_expr(graph, block, a, options, ctx))
                .collect();
            let target = canonical_call_target(&call.func);
            graph.push_op(
                *block,
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
            if let Some(recv) = lower_expr(graph, block, &mc.receiver, options, ctx) {
                args.push(recv);
            }
            for a in &mc.args {
                if let Some(v) = lower_expr(graph, block, a, options, ctx) {
                    args.push(v);
                }
            }
            graph.push_op(
                *block,
                OpKind::Call {
                    target: CallTarget::method(
                        mc.method.to_string(),
                        receiver_type_root(&mc.receiver, ctx),
                    ),
                    args,
                    result_ty: ValueType::Unknown,
                },
                true,
            )
        }

        // ── if/else → block split (RPython FlowContext.guessbool) ──
        //
        // Creates: then_block, else_block, merge_block
        // If both branches produce a value, merge_block gets an inputarg
        // (Phi node) that receives the value from each branch via Link args.
        syn::Expr::If(if_expr) => {
            let cond = lower_expr(graph, block, &if_expr.cond, options, ctx)
                .unwrap_or_else(|| graph.alloc_value());

            let mut then_block = graph.create_block();
            let mut else_block = graph.create_block();

            graph.set_terminator(
                *block,
                Terminator::Branch {
                    cond,
                    if_true: then_block,
                    true_args: vec![],
                    if_false: else_block,
                    false_args: vec![],
                },
            );

            // Lower then branch — collect result value
            let mut then_result = None;
            for stmt in &if_expr.then_branch.stmts {
                lower_stmt(graph, &mut then_block, stmt, options, ctx);
            }
            // Last expression in then_branch is the result (if no explicit return)
            if let Some(last) = if_expr.then_branch.stmts.last() {
                if let syn::Stmt::Expr(e, None) = last {
                    then_result = lower_expr(graph, &mut then_block, e, options, ctx);
                }
            }

            // Lower else branch
            let mut else_result = None;
            if let Some((_, else_branch)) = &if_expr.else_branch {
                else_result = lower_expr(graph, &mut else_block, else_branch, options, ctx);
            }

            // Create merge block with Phi if both branches have values
            let (merge_block, phi_result) = if then_result.is_some() && else_result.is_some() {
                let (merge, phi_args) = graph.create_block_with_args(1);
                // Link args: then → merge(then_result), else → merge(else_result)
                if graph.block(then_block).terminator == Terminator::Unreachable {
                    graph.set_terminator(
                        then_block,
                        Terminator::Goto {
                            target: merge,
                            args: vec![then_result.unwrap()],
                        },
                    );
                }
                if graph.block(else_block).terminator == Terminator::Unreachable {
                    graph.set_terminator(
                        else_block,
                        Terminator::Goto {
                            target: merge,
                            args: vec![else_result.unwrap()],
                        },
                    );
                }
                (merge, Some(phi_args[0]))
            } else {
                let merge = graph.create_block();
                if graph.block(then_block).terminator == Terminator::Unreachable {
                    graph.set_terminator(
                        then_block,
                        Terminator::Goto {
                            target: merge,
                            args: vec![],
                        },
                    );
                }
                if graph.block(else_block).terminator == Terminator::Unreachable {
                    graph.set_terminator(
                        else_block,
                        Terminator::Goto {
                            target: merge,
                            args: vec![],
                        },
                    );
                }
                (merge, None)
            };

            *block = merge_block;
            phi_result
        }

        // ── return ──
        syn::Expr::Return(ret) => {
            let val = ret
                .expr
                .as_ref()
                .and_then(|e| lower_expr(graph, block, e, options, ctx));
            graph.set_terminator(*block, Terminator::Return(val));
            None
        }

        // ── block { stmts } ──
        syn::Expr::Block(blk) => {
            let mut last = None;
            for stmt in &blk.block.stmts {
                lower_stmt(graph, block, stmt, options, ctx);
                if let syn::Stmt::Expr(e, None) = stmt {
                    last = lower_expr(graph, block, e, options, ctx);
                }
            }
            last
        }

        // ── literals ──
        syn::Expr::Lit(lit) => {
            if let syn::Lit::Int(int_lit) = &lit.lit {
                if let Ok(v) = int_lit.base10_parse::<i64>() {
                    return graph.push_op(*block, OpKind::ConstInt(v), true);
                }
            }
            graph.push_op(
                *block,
                OpKind::Unknown {
                    kind: UnknownKind::UnsupportedLiteral,
                },
                true,
            )
        }

        // ── path (variable reference) ──
        syn::Expr::Path(path) => {
            let name = path
                .path
                .segments
                .iter()
                .map(|seg| seg.ident.to_string())
                .collect::<Vec<_>>()
                .join("::");
            graph.push_op(
                *block,
                OpKind::Input {
                    name,
                    ty: ValueType::Unknown,
                },
                true,
            )
        }

        // ── reference &expr ──
        syn::Expr::Reference(r) => lower_expr(graph, block, &r.expr, options, ctx),

        // ── parenthesized (expr) ──
        syn::Expr::Paren(p) => lower_expr(graph, block, &p.expr, options, ctx),

        // ── unary !x, -x ──
        syn::Expr::Unary(u) => {
            let operand = lower_expr(graph, block, &u.expr, options, ctx)
                .unwrap_or_else(|| graph.alloc_value());
            graph.push_op(
                *block,
                OpKind::UnaryOp {
                    op: unary_op_name(&u.op).into(),
                    operand,
                    result_ty: ValueType::Unknown,
                },
                true,
            )
        }

        // ── binary a + b ──
        syn::Expr::Binary(bin) => {
            let lhs = lower_expr(graph, block, &bin.left, options, ctx)
                .unwrap_or_else(|| graph.alloc_value());
            let rhs = lower_expr(graph, block, &bin.right, options, ctx)
                .unwrap_or_else(|| graph.alloc_value());
            graph.push_op(
                *block,
                OpKind::BinOp {
                    op: binary_op_name(&bin.op).into(),
                    lhs,
                    rhs,
                    result_ty: ValueType::Unknown,
                },
                true,
            )
        }

        // ── cast: expr as T ──
        syn::Expr::Cast(cast) => lower_expr(graph, block, &cast.expr, options, ctx),

        // ── match expr { arms } → multi-block (RPython switch) ──
        syn::Expr::Match(m) => {
            let scrutinee = lower_expr(graph, block, &m.expr, options, ctx)
                .unwrap_or_else(|| graph.alloc_value());

            if m.arms.is_empty() {
                return None;
            }

            let merge = graph.create_block();
            let mut arm_results = Vec::new();

            for arm in &m.arms {
                let mut arm_block = graph.create_block();
                // Each arm gets its own block (simplified: no pattern matching guards)
                let result = lower_expr(graph, &mut arm_block, &arm.body, options, ctx);
                arm_results.push((arm_block, result));
                if graph.block(arm_block).terminator == Terminator::Unreachable {
                    let goto_args = result.map_or(vec![], |v| vec![v]);
                    graph.set_terminator(
                        arm_block,
                        Terminator::Goto {
                            target: merge,
                            args: goto_args,
                        },
                    );
                }
            }

            // First arm as default branch (simplified)
            if m.arms.len() == 1 {
                graph.set_terminator(
                    *block,
                    Terminator::Goto {
                        target: arm_results[0].0,
                        args: vec![],
                    },
                );
            } else {
                // Binary branch on scrutinee for first arm, else second
                let first_block = arm_results[0].0;
                let second_block = arm_results.get(1).map(|a| a.0).unwrap_or(merge);
                graph.set_terminator(
                    *block,
                    Terminator::Branch {
                        cond: scrutinee,
                        if_true: first_block,
                        true_args: vec![],
                        if_false: second_block,
                        false_args: vec![],
                    },
                );
                // Chain remaining arms as fallthrough branches
                for i in 1..arm_results.len().saturating_sub(1) {
                    let next = arm_results.get(i + 1).map(|a| a.0).unwrap_or(merge);
                    // Each arm could branch to next or fall to merge
                    // (simplified: all goto merge)
                }
            }

            *block = merge;
            None
        }

        // ── while → header block + body block + exit block ──
        syn::Expr::While(w) => {
            let mut header = graph.create_block();
            let mut body = graph.create_block();
            let exit = graph.create_block();

            // Current block → header
            graph.set_terminator(
                *block,
                Terminator::Goto {
                    target: header,
                    args: vec![],
                },
            );

            // Header: evaluate condition, branch to body or exit
            let cond = lower_expr(graph, &mut header, &w.cond, options, ctx)
                .unwrap_or_else(|| graph.alloc_value());
            graph.set_terminator(
                header,
                Terminator::Branch {
                    cond,
                    if_true: body,
                    true_args: vec![],
                    if_false: exit,
                    false_args: vec![],
                },
            );

            // Body → back to header
            for stmt in &w.body.stmts {
                lower_stmt(graph, &mut body, stmt, options, ctx);
            }
            if graph.block(body).terminator == Terminator::Unreachable {
                graph.set_terminator(
                    body,
                    Terminator::Goto {
                        target: header,
                        args: vec![],
                    },
                );
            }

            *block = exit;
            None
        }
        syn::Expr::Loop(l) => {
            let mut body = graph.create_block();
            let exit = graph.create_block();

            graph.set_terminator(
                *block,
                Terminator::Goto {
                    target: body,
                    args: vec![],
                },
            );

            for stmt in &l.body.stmts {
                lower_stmt(graph, &mut body, stmt, options, ctx);
            }
            if graph.block(body).terminator == Terminator::Unreachable {
                graph.set_terminator(
                    body,
                    Terminator::Goto {
                        target: body,
                        args: vec![],
                    },
                );
            }

            *block = exit;
            None
        }
        syn::Expr::ForLoop(f) => {
            let mut header = graph.create_block();
            let mut body = graph.create_block();
            let exit = graph.create_block();

            graph.set_terminator(
                *block,
                Terminator::Goto {
                    target: header,
                    args: vec![],
                },
            );

            lower_expr(graph, &mut header, &f.expr, options, ctx);
            let iter_cond = graph.alloc_value();
            graph.set_terminator(
                header,
                Terminator::Branch {
                    cond: iter_cond,
                    if_true: body,
                    true_args: vec![],
                    if_false: exit,
                    false_args: vec![],
                },
            );

            for stmt in &f.body.stmts {
                lower_stmt(graph, &mut body, stmt, options, ctx);
            }
            if graph.block(body).terminator == Terminator::Unreachable {
                graph.set_terminator(
                    body,
                    Terminator::Goto {
                        target: header,
                        args: vec![],
                    },
                );
            }

            *block = exit;
            None
        }

        // ── break/continue ──
        syn::Expr::Break(b) => {
            if let Some(e) = &b.expr {
                lower_expr(graph, block, e, options, ctx);
            }
            None
        }
        syn::Expr::Continue(_) => None,

        // ── closure ──
        syn::Expr::Closure(c) => lower_expr(graph, block, &c.body, options, ctx),

        // ── tuple (a, b, c) ──
        syn::Expr::Tuple(t) => {
            let mut last = None;
            for e in &t.elems {
                last = lower_expr(graph, block, e, options, ctx);
            }
            last
        }

        // ── try expr? ──
        syn::Expr::Try(t) => lower_expr(graph, block, &t.expr, options, ctx),

        // ── fallback ──
        _ => graph.push_op(
            *block,
            OpKind::Unknown {
                kind: UnknownKind::UnsupportedExpr,
            },
            true,
        ),
    }
}

// ── Helpers ──────────────────────────────────────────────────────

fn unary_op_name(op: &syn::UnOp) -> &'static str {
    match op {
        syn::UnOp::Deref(_) => "deref",
        syn::UnOp::Not(_) => "not",
        syn::UnOp::Neg(_) => "neg",
        _ => "unknown_unary",
    }
}

fn binary_op_name(op: &syn::BinOp) -> &'static str {
    match op {
        syn::BinOp::Add(_) => "add",
        syn::BinOp::Sub(_) => "sub",
        syn::BinOp::Mul(_) => "mul",
        syn::BinOp::Div(_) => "div",
        syn::BinOp::Rem(_) => "mod",
        syn::BinOp::And(_) => "and",
        syn::BinOp::Or(_) => "or",
        syn::BinOp::BitXor(_) => "bitxor",
        syn::BinOp::BitAnd(_) => "bitand",
        syn::BinOp::BitOr(_) => "bitor",
        syn::BinOp::Shl(_) => "lshift",
        syn::BinOp::Shr(_) => "rshift",
        syn::BinOp::Eq(_) => "eq",
        syn::BinOp::Lt(_) => "lt",
        syn::BinOp::Le(_) => "le",
        syn::BinOp::Ne(_) => "ne",
        syn::BinOp::Ge(_) => "ge",
        syn::BinOp::Gt(_) => "gt",
        syn::BinOp::AddAssign(_) => "add_assign",
        syn::BinOp::SubAssign(_) => "sub_assign",
        syn::BinOp::MulAssign(_) => "mul_assign",
        syn::BinOp::DivAssign(_) => "div_assign",
        syn::BinOp::RemAssign(_) => "mod_assign",
        syn::BinOp::BitXorAssign(_) => "bitxor_assign",
        syn::BinOp::BitAndAssign(_) => "bitand_assign",
        syn::BinOp::BitOrAssign(_) => "bitor_assign",
        syn::BinOp::ShlAssign(_) => "lshift_assign",
        syn::BinOp::ShrAssign(_) => "rshift_assign",
        _ => "unknown_binop",
    }
}

fn member_name(member: &syn::Member) -> String {
    match member {
        syn::Member::Named(ident) => ident.to_string(),
        syn::Member::Unnamed(idx) => idx.index.to_string(),
    }
}

fn canonical_call_target(expr: &syn::Expr) -> CallTarget {
    match expr {
        syn::Expr::Path(path) => CallTarget::function_path(
            path.path
                .segments
                .iter()
                .map(|seg| seg.ident.to_string())
                .collect::<Vec<_>>(),
        ),
        _ => CallTarget::UnsupportedExpr,
    }
}

fn receiver_type_root(expr: &syn::Expr, ctx: &GraphBuildContext) -> Option<String> {
    match expr {
        syn::Expr::Path(path) => path
            .path
            .get_ident()
            .and_then(|ident| ctx.local_type_roots.get(&ident.to_string()).cloned()),
        syn::Expr::Reference(reference) => receiver_type_root(&reference.expr, ctx),
        syn::Expr::Paren(paren) => receiver_type_root(&paren.expr, ctx),
        syn::Expr::Field(field) => receiver_type_root(&field.base, ctx),
        syn::Expr::Index(index) => receiver_type_root(&index.expr, ctx),
        _ => None,
    }
}

fn canonical_pat_name(pat: &syn::Pat) -> String {
    match pat {
        syn::Pat::Ident(ident) => ident.ident.to_string(),
        syn::Pat::Reference(reference) => canonical_pat_name(&reference.pat),
        syn::Pat::Type(typed) => canonical_pat_name(&typed.pat),
        syn::Pat::TupleStruct(tuple_struct) => tuple_struct
            .path
            .segments
            .iter()
            .map(|seg| seg.ident.to_string())
            .collect::<Vec<_>>()
            .join("::"),
        syn::Pat::Struct(strukt) => strukt
            .path
            .segments
            .iter()
            .map(|seg| seg.ident.to_string())
            .collect::<Vec<_>>()
            .join("::"),
        syn::Pat::Tuple(_) => "tuple_pat".into(),
        syn::Pat::Slice(_) => "slice_pat".into(),
        syn::Pat::Lit(_) => "lit_pat".into(),
        syn::Pat::Path(_) => "path_pat".into(),
        syn::Pat::Wild(_) => "_".into(),
        syn::Pat::Or(_) => "or_pat".into(),
        syn::Pat::Range(_) => "range_pat".into(),
        syn::Pat::Macro(_) => "macro_pat".into(),
        syn::Pat::Paren(paren) => canonical_pat_name(&paren.pat),
        _ => "unsupported_pat".into(),
    }
}

fn type_root_ident(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(path) => path.path.segments.last().map(|seg| seg.ident.to_string()),
        syn::Type::Reference(reference) => type_root_ident(&reference.elem),
        syn::Type::Paren(paren) => type_root_ident(&paren.elem),
        syn::Type::Group(group) => type_root_ident(&group.elem),
        _ => None,
    }
}

/// RPython: `GcArray(T)` — extract the element type `T` from an
/// array-like container type.
///
/// For `Vec<Point>`, returns `"Point"`.  For `[i64]`, returns `"i64"`.
/// This is the Rust equivalent of RPython's ARRAY.OF, which is the
/// element type that determines the ARRAY identity.  Two arrays with
/// the same element type share one `cpu.arraydescrof()` descriptor.
fn array_element_type(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(path) => {
            // Vec<T>, Box<[T]>, etc. — extract first generic type arg
            let last_seg = path.path.segments.last()?;
            if let syn::PathArguments::AngleBracketed(args) = &last_seg.arguments {
                for arg in &args.args {
                    if let syn::GenericArgument::Type(inner_ty) = arg {
                        return full_type_string(inner_ty);
                    }
                }
            }
            None
        }
        syn::Type::Reference(r) => array_element_type(&r.elem),
        syn::Type::Paren(p) => array_element_type(&p.elem),
        syn::Type::Group(g) => array_element_type(&g.elem),
        // [T] slice → element type is T
        syn::Type::Slice(s) => full_type_string(&s.elem),
        // [T; N] array → element type is T
        syn::Type::Array(a) => full_type_string(&a.elem),
        _ => None,
    }
}

/// Canonical type string for a syn::Type.
///
/// Produces a string that includes generic arguments,
/// e.g. `Vec<Point>` → `"Vec<Point>"`, `Point` → `"Point"`.
fn full_type_string(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(path) => {
            let segments: Vec<String> = path
                .path
                .segments
                .iter()
                .map(|seg| {
                    let name = seg.ident.to_string();
                    match &seg.arguments {
                        syn::PathArguments::None => name,
                        syn::PathArguments::AngleBracketed(args) => {
                            let inner: Vec<String> = args
                                .args
                                .iter()
                                .filter_map(|arg| match arg {
                                    syn::GenericArgument::Type(t) => full_type_string(t),
                                    _ => None,
                                })
                                .collect();
                            if inner.is_empty() {
                                name
                            } else {
                                format!("{}<{}>", name, inner.join(","))
                            }
                        }
                        syn::PathArguments::Parenthesized(_) => name,
                    }
                })
                .collect();
            Some(segments.join("::"))
        }
        syn::Type::Reference(r) => full_type_string(&r.elem),
        syn::Type::Paren(p) => full_type_string(&p.elem),
        syn::Type::Group(g) => full_type_string(&g.elem),
        syn::Type::Slice(s) => full_type_string(&s.elem).map(|t| format!("[{}]", t)),
        syn::Type::Array(a) => full_type_string(&a.elem).map(|t| format!("[{}]", t)),
        _ => None,
    }
}

/// RPython: resolve ARRAY identity from an expression.
///
/// For `arr[idx]`, returns the ELEMENT TYPE of `arr` from context.
/// This is the Rust equivalent of RPython's `op.args[0].concretetype.TO`
/// which gives `GcArray(T)` — the `T` is what distinguishes array types.
fn array_type_id_from_expr(expr: &syn::Expr, ctx: &GraphBuildContext) -> Option<String> {
    match expr {
        syn::Expr::Path(path) => path
            .path
            .get_ident()
            .and_then(|ident| ctx.local_element_types.get(&ident.to_string()).cloned()),
        syn::Expr::Reference(r) => array_type_id_from_expr(&r.expr, ctx),
        syn::Expr::Paren(p) => array_type_id_from_expr(&p.expr, ctx),
        // For field access (self.array) or other complex exprs,
        // type is not statically known — return None (conservative).
        _ => None,
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
        assert!(graph.block(graph.startblock).operations.len() >= 2);
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
        let ops = &graph.block(graph.startblock).operations;
        // Should contain a FieldRead op
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::FieldRead { field, .. } if field.name == "x"
            )),
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
        let ops = &graph.block(graph.startblock).operations;
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target, .. } if target == &CallTarget::method("push", Some("Vec".into()))
            )),
            "expected Call to 'push', got {:?}",
            ops
        );
    }

    #[test]
    fn lowers_impl_self_method_call_with_concrete_self_type() {
        let parsed = crate::parse::parse_source(
            r#"
            struct Foo;
            impl Foo {
                fn helper(&self) {}
                fn run(&self) {
                    self.helper();
                }
            }
        "#,
        );
        let program = build_semantic_program(&parsed);
        let run = program
            .functions
            .iter()
            .find(|func| func.name == "run")
            .expect("run graph");
        let ops = &run.graph.block(run.graph.startblock).operations;
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target, .. }
                    if target == &CallTarget::method("helper", Some("Foo".into()))
            )),
            "expected helper call with concrete self type, got {:?}",
            ops
        );
    }

    #[test]
    fn lowers_path_call_to_canonical_symbol() {
        let parsed = crate::parse::parse_source(
            r#"
            fn call_example(x: i64) -> i64 {
                crate::math::w_int_add(x)
            }
        "#,
        );
        let program = build_semantic_program(&parsed);
        let graph = &program.functions[0].graph;
        let ops = &graph.block(graph.startblock).operations;
        assert!(
            ops.iter().any(|op| matches!(
                &op.kind,
                OpKind::Call { target, .. }
                    if target == &CallTarget::function_path(["crate", "math", "w_int_add"])
            )),
            "expected canonical Call target path, got {:?}",
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

    #[test]
    fn if_creates_multiple_blocks() {
        let parsed = crate::parse::parse_source(
            r#"
            fn branch(x: bool) -> i64 {
                if x { 1 } else { 2 }
            }
        "#,
        );
        let program = build_semantic_program(&parsed);
        let graph = &program.functions[0].graph;
        // entry + then + else + merge = at least 4 blocks
        assert!(
            graph.blocks.len() >= 4,
            "if/else should create >=4 blocks, got {}",
            graph.blocks.len()
        );
        // Entry block should have a Branch terminator
        assert!(
            matches!(
                &graph.block(graph.startblock).terminator,
                Terminator::Branch { .. }
            ),
            "entry should end with Branch, got {:?}",
            graph.block(graph.startblock).terminator
        );
    }

    #[test]
    fn while_creates_header_body_exit() {
        let parsed = crate::parse::parse_source(
            r#"
            fn loop_fn(mut x: i64) -> i64 {
                while x > 0 { x = x - 1; }
                x
            }
        "#,
        );
        let program = build_semantic_program(&parsed);
        let graph = &program.functions[0].graph;
        // entry + header + body + exit = at least 4 blocks
        assert!(
            graph.blocks.len() >= 4,
            "while should create >=4 blocks, got {}",
            graph.blocks.len()
        );
    }

    #[test]
    fn lowers_binary_ops_to_exact_names_without_token_strings() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example(x: i64, y: i64) -> i64 {
                x + y
            }
        "#,
        );
        let program = build_semantic_program(&parsed);
        let graph = &program.functions[0].graph;
        let op = graph
            .block(graph.startblock)
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::BinOp { op, .. } => Some(op.clone()),
                _ => None,
            })
            .expect("binop");
        assert_eq!(op, "add");
    }

    #[test]
    fn lowers_unary_ops_to_exact_names_without_token_strings() {
        let parsed = crate::parse::parse_source(
            r#"
            fn example(x: i64) -> i64 {
                -x
            }
        "#,
        );
        let program = build_semantic_program(&parsed);
        let graph = &program.functions[0].graph;
        let op = graph
            .block(graph.startblock)
            .operations
            .iter()
            .find_map(|op| match &op.kind {
                OpKind::UnaryOp { op, .. } => Some(op.clone()),
                _ => None,
            })
            .expect("unary op");
        assert_eq!(op, "neg");
    }
}
