//! Match arm classifier: analyzes each arm of the opcode dispatch and
//! classifies it into a structural category.
//!
//! Most arms are `Lowerable` and handled by the JitCode lowerer.
//! Only a handful of structurally special patterns need dedicated codegen.
//!
//! ## Supported CFG patterns
//!
//! The classifier and lowerer handle these control flow patterns within
//! match arms:
//!
//! - **Sequential operations**: straight-line code (push, pop, binop, etc.)
//! - **Binary if/else**: `if cond { ... } else { ... }` with guard emission
//! - **Nested if/else**: multiple levels of if/else within a match arm
//! - **match on opcode** (the main dispatch): the top-level opcode match
//! - **Nested match in let-binding**: `let x = match ... { ... };` is
//!   recognized as a `BranchGroup` (BRPOP/JMP/BRZ pattern)
//! - **Standalone match**: `match x { 1 => ..., 2 => ..., _ => ... }` is
//!   lowered to a chained if-else guard sequence
//!
//! ## Loop CFG patterns
//!
//! Loops within match arms are lowered to JitCode branch sequences:
//!
//! - **while loops**: `while cond { body }` → branch-zero exit check + back-edge jump
//! - **loop with break**: `loop { ... break ... }` → unconditional back-edge + break targets
//! - **for loops**: `for x in iter { body }` → the lowerer falls back to `None`
//!   (not lowered), which makes the entire arm opaque to the JIT
//!
//! ## Unsupported CFG patterns
//!
//! These patterns within match arms will cause the arm to be classified as
//! `Unsupported` and produce a compile-time error:
//!
//! - **goto/label patterns**: not applicable in safe Rust, but any
//!   `unsafe` block with computed jumps is unsupported

use syn::{Arm, Expr, ExprBlock, Pat, Stmt};

/// Classification of a match arm for trace code generation.
pub enum ArmPattern {
    /// Branch group — contains a nested match (BRPOP/JMP/BRZ).
    BranchGroup,
    /// Abort permanently — contains untraceable I/O input operations.
    AbortPermanent,
    /// No-op — empty body.
    Nop,
    /// Halt — `break`.
    Halt,
    /// Lowerable — the Lowerer handles this arm generically.
    Lowerable,
    /// Unsupported — contains control flow that cannot be traced.
    Unsupported(String),
}

/// A classified match arm with its original pattern and body.
pub struct ClassifiedArm {
    pub pat: Pat,
    pub pattern: ArmPattern,
    pub original_body: Expr,
}

/// Classify all arms of the opcode dispatch match.
pub fn classify_arms(arms: &[Arm]) -> Vec<ClassifiedArm> {
    arms.iter()
        .map(|arm| {
            let pattern = classify_arm_body(&arm.body);
            ClassifiedArm {
                pat: arm.pat.clone(),
                pattern,
                original_body: (*arm.body).clone(),
            }
        })
        .collect()
}

fn classify_arm_body(body: &Expr) -> ArmPattern {
    if is_break_expr(body) {
        return ArmPattern::Halt;
    }
    if is_empty_block(body) {
        return ArmPattern::Nop;
    }

    let stmts = extract_stmts(body);

    if detect_abort_pattern(&stmts) {
        return ArmPattern::AbortPermanent;
    }
    if detect_branch_group_pattern(&stmts) {
        return ArmPattern::BranchGroup;
    }
    if let Some(reason) = detect_unsupported_pattern(&stmts) {
        return ArmPattern::Unsupported(reason);
    }

    ArmPattern::Lowerable
}

// ── Pattern detection helpers ────────────────────────────────────────

fn is_break_expr(expr: &Expr) -> bool {
    matches!(expr, Expr::Break(_))
}

fn is_empty_block(expr: &Expr) -> bool {
    if let Expr::Block(block) = expr {
        block.block.stmts.is_empty()
    } else {
        false
    }
}

fn extract_stmts(expr: &Expr) -> Vec<Stmt> {
    match expr {
        Expr::Block(ExprBlock { block, .. }) => block.stmts.clone(),
        _ => vec![Stmt::Expr(expr.clone(), None)],
    }
}

/// Detect abort pattern: contains input read operations.
fn detect_abort_pattern(stmts: &[Stmt]) -> bool {
    stmts.iter().any(stmt_contains_abort_call)
}

fn stmt_contains_abort_call(stmt: &Stmt) -> bool {
    match stmt {
        Stmt::Expr(expr, _) => expr_contains_abort_call(expr),
        Stmt::Local(local) => local
            .init
            .as_ref()
            .is_some_and(|init| expr_contains_abort_call(&init.expr)),
        _ => false,
    }
}

fn expr_contains_abort_call(expr: &Expr) -> bool {
    match expr {
        Expr::Call(call) => {
            path_ends_with(&call.func, &["read_number", "read_utf8"])
                || call.args.iter().any(expr_contains_abort_call)
        }
        Expr::MethodCall(call) => {
            call.method == "read_number"
                || call.method == "read_utf8"
                || expr_contains_abort_call(&call.receiver)
                || call.args.iter().any(expr_contains_abort_call)
        }
        Expr::Block(block) => block.block.stmts.iter().any(stmt_contains_abort_call),
        Expr::If(expr_if) => {
            expr_contains_abort_call(&expr_if.cond)
                || expr_if
                    .then_branch
                    .stmts
                    .iter()
                    .any(stmt_contains_abort_call)
                || expr_if
                    .else_branch
                    .as_ref()
                    .is_some_and(|(_, else_expr)| expr_contains_abort_call(else_expr))
        }
        Expr::Match(expr_match) => {
            expr_contains_abort_call(&expr_match.expr)
                || expr_match.arms.iter().any(|arm| {
                    expr_contains_abort_call(&arm.body)
                        || arm
                            .guard
                            .as_ref()
                            .is_some_and(|(_, guard)| expr_contains_abort_call(guard))
                })
        }
        Expr::Loop(expr_loop) => expr_loop.body.stmts.iter().any(stmt_contains_abort_call),
        Expr::While(expr_while) => {
            expr_contains_abort_call(&expr_while.cond)
                || expr_while.body.stmts.iter().any(stmt_contains_abort_call)
        }
        Expr::ForLoop(expr_for) => {
            expr_contains_abort_call(&expr_for.expr)
                || expr_for.body.stmts.iter().any(stmt_contains_abort_call)
        }
        Expr::Assign(expr_assign) => {
            expr_contains_abort_call(&expr_assign.left)
                || expr_contains_abort_call(&expr_assign.right)
        }
        Expr::Binary(expr_binary) => {
            expr_contains_abort_call(&expr_binary.left)
                || expr_contains_abort_call(&expr_binary.right)
        }
        Expr::Unary(expr_unary) => expr_contains_abort_call(&expr_unary.expr),
        Expr::Paren(expr_paren) => expr_contains_abort_call(&expr_paren.expr),
        Expr::Return(expr_return) => expr_return
            .expr
            .as_ref()
            .is_some_and(|expr| expr_contains_abort_call(expr)),
        Expr::Break(expr_break) => expr_break
            .expr
            .as_ref()
            .is_some_and(|expr| expr_contains_abort_call(expr)),
        Expr::Array(expr_array) => expr_array.elems.iter().any(expr_contains_abort_call),
        Expr::Tuple(expr_tuple) => expr_tuple.elems.iter().any(expr_contains_abort_call),
        Expr::Struct(expr_struct) => {
            expr_struct
                .fields
                .iter()
                .any(|field| expr_contains_abort_call(&field.expr))
                || expr_struct
                    .rest
                    .as_ref()
                    .is_some_and(|expr| expr_contains_abort_call(expr))
        }
        Expr::Index(expr_index) => {
            expr_contains_abort_call(&expr_index.expr)
                || expr_contains_abort_call(&expr_index.index)
        }
        Expr::Field(expr_field) => expr_contains_abort_call(&expr_field.base),
        Expr::Reference(expr_ref) => expr_contains_abort_call(&expr_ref.expr),
        Expr::Cast(expr_cast) => expr_contains_abort_call(&expr_cast.expr),
        _ => false,
    }
}

fn path_ends_with(expr: &Expr, names: &[&str]) -> bool {
    match expr {
        Expr::Path(path) => path
            .path
            .segments
            .last()
            .is_some_and(|seg| names.iter().any(|name| seg.ident == *name)),
        _ => false,
    }
}

/// Detect branch group pattern: contains a nested match in a let-binding.
fn detect_branch_group_pattern(stmts: &[Stmt]) -> bool {
    for stmt in stmts {
        if let Stmt::Local(local) = stmt {
            if let Some(init) = &local.init {
                if matches!(&*init.expr, Expr::Match(_)) {
                    return true;
                }
            }
        }
    }
    false
}

/// Detect unsupported CFG patterns within match arm statements.
///
/// Returns `Some(reason)` if an unsupported pattern is found.
fn detect_unsupported_pattern(stmts: &[Stmt]) -> Option<String> {
    for stmt in stmts {
        if let Some(reason) = check_stmt_unsupported(stmt) {
            return Some(reason);
        }
    }
    None
}

fn check_stmt_unsupported(stmt: &Stmt) -> Option<String> {
    match stmt {
        Stmt::Expr(expr, _) => check_expr_unsupported(expr),
        Stmt::Local(local) => {
            if let Some(init) = &local.init {
                // Nested match in let-binding is handled as BranchGroup,
                // so we only flag it if it's not detected as a branch group.
                check_expr_unsupported(&init.expr)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn check_expr_unsupported(expr: &Expr) -> Option<String> {
    match expr {
        // Standalone match expression — lowered to an if-else guard chain.
        // Recurse into the arm bodies to check for unsupported constructs.
        Expr::Match(match_expr) => {
            for arm in &match_expr.arms {
                let arm_stmts = extract_stmts(&arm.body);
                for s in &arm_stmts {
                    if let Some(reason) = check_stmt_unsupported(s) {
                        return Some(reason);
                    }
                }
            }
            None
        }
        // Loops within arms are lowered to JitCode branch sequences
        // by the Lowerer (while/loop) or fall back to opaque (for).
        Expr::Loop(loop_expr) => {
            // Recurse into loop body to check for nested unsupported constructs
            for s in &loop_expr.body.stmts {
                if let Some(reason) = check_stmt_unsupported(s) {
                    return Some(reason);
                }
            }
            None
        }
        Expr::While(while_expr) => {
            // Recurse into while body to check for nested unsupported constructs
            for s in &while_expr.body.stmts {
                if let Some(reason) = check_stmt_unsupported(s) {
                    return Some(reason);
                }
            }
            None
        }
        Expr::ForLoop(for_expr) => {
            // Recurse into for body to check for nested unsupported constructs
            for s in &for_expr.body.stmts {
                if let Some(reason) = check_stmt_unsupported(s) {
                    return Some(reason);
                }
            }
            None
        }
        // Recurse into if/else branches — if/else itself is supported,
        // but it may contain unsupported constructs within.
        Expr::If(if_expr) => {
            for s in &if_expr.then_branch.stmts {
                if let Some(reason) = check_stmt_unsupported(s) {
                    return Some(reason);
                }
            }
            if let Some((_, else_expr)) = &if_expr.else_branch {
                check_expr_unsupported(else_expr)?;
            }
            None
        }
        Expr::Block(block) => {
            for s in &block.block.stmts {
                if let Some(reason) = check_stmt_unsupported(s) {
                    return Some(reason);
                }
            }
            None
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_arm(code: &str) -> Arm {
        let match_code = format!("match x {{ {} }}", code);
        let expr: syn::ExprMatch = syn::parse_str(&match_code).expect("failed to parse match");
        expr.arms.into_iter().next().unwrap()
    }

    #[test]
    fn classify_break_as_halt() {
        let arm = parse_arm("0 => break,");
        let result = classify_arm_body(&arm.body);
        assert!(matches!(result, ArmPattern::Halt));
    }

    #[test]
    fn classify_empty_block_as_nop() {
        let arm = parse_arm("0 => {},");
        let result = classify_arm_body(&arm.body);
        assert!(matches!(result, ArmPattern::Nop));
    }

    #[test]
    fn classify_simple_call_as_lowerable() {
        let arm = parse_arm("0 => { foo(); },");
        let result = classify_arm_body(&arm.body);
        assert!(matches!(result, ArmPattern::Lowerable));
    }

    #[test]
    fn classify_if_else_as_lowerable() {
        let arm = parse_arm("0 => { if cond { a(); } else { b(); } },");
        let result = classify_arm_body(&arm.body);
        assert!(matches!(result, ArmPattern::Lowerable));
    }

    #[test]
    fn classify_nested_if_else_as_lowerable() {
        let arm = parse_arm("0 => { if a { if b { x(); } else { y(); } } else { z(); } },");
        let result = classify_arm_body(&arm.body);
        assert!(matches!(result, ArmPattern::Lowerable));
    }

    #[test]
    fn classify_let_match_as_branch_group() {
        let arm = parse_arm("0 => { let x = match op { 1 => a, _ => b }; },");
        let result = classify_arm_body(&arm.body);
        assert!(matches!(result, ArmPattern::BranchGroup));
    }

    #[test]
    fn classify_standalone_match_as_lowerable() {
        let arm = parse_arm("0 => { match op { 1 => a(), _ => b() } },");
        let result = classify_arm_body(&arm.body);
        assert!(matches!(result, ArmPattern::Lowerable));
    }

    #[test]
    fn classify_nested_match_with_loop_as_lowerable() {
        let arm = parse_arm("0 => { match op { 1 => { loop { break; } }, _ => b() } },");
        let result = classify_arm_body(&arm.body);
        assert!(matches!(result, ArmPattern::Lowerable));
    }

    #[test]
    fn classify_loop_as_lowerable() {
        let arm = parse_arm("0 => { loop { break; } },");
        let result = classify_arm_body(&arm.body);
        assert!(matches!(result, ArmPattern::Lowerable));
    }

    #[test]
    fn classify_while_as_lowerable() {
        let arm = parse_arm("0 => { while cond { x(); } },");
        let result = classify_arm_body(&arm.body);
        assert!(matches!(result, ArmPattern::Lowerable));
    }

    #[test]
    fn classify_for_as_lowerable() {
        let arm = parse_arm("0 => { for i in 0..10 { x(); } },");
        let result = classify_arm_body(&arm.body);
        assert!(matches!(result, ArmPattern::Lowerable));
    }

    #[test]
    fn classify_nested_loop_in_if_as_lowerable() {
        let arm = parse_arm("0 => { if cond { loop { break; } } else { x(); } },");
        let result = classify_arm_body(&arm.body);
        assert!(matches!(result, ArmPattern::Lowerable));
    }

    #[test]
    fn classify_read_number_as_abort() {
        let arm = parse_arm("0 => { read_number(); },");
        let result = classify_arm_body(&arm.body);
        assert!(matches!(result, ArmPattern::AbortPermanent));
    }

    #[test]
    fn classify_method_read_utf8_as_abort() {
        let arm = parse_arm("0 => { io.read_utf8(); },");
        let result = classify_arm_body(&arm.body);
        assert!(matches!(result, ArmPattern::AbortPermanent));
    }
}
