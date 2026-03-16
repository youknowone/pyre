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
//! ## Unsupported CFG patterns
//!
//! These patterns within match arms will cause the arm to be classified as
//! `Unsupported` and produce a compile-time error:
//!
//! - **loop/while within arms**: iteration inside an opcode handler
//!   (extract to a `#[dont_look_inside]` helper)
//! - **for loops within arms**: same as loop/while
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
    stmts.iter().any(|s| {
        let s_str = quote::quote!(#s).to_string();
        s_str.contains("read_number") || s_str.contains("read_utf8")
    })
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
        // Loop/while/for within an arm
        Expr::Loop(_) => Some(
            "loop within a match arm is not supported in JIT-traced arms; \
             extract to a #[dont_look_inside] helper function"
                .to_string(),
        ),
        Expr::While(_) => Some(
            "while loop within a match arm is not supported in JIT-traced arms; \
             extract to a #[dont_look_inside] helper function"
                .to_string(),
        ),
        Expr::ForLoop(_) => Some(
            "for loop within a match arm is not supported in JIT-traced arms; \
             extract to a #[dont_look_inside] helper function"
                .to_string(),
        ),
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
    fn classify_nested_match_with_loop_as_unsupported() {
        let arm = parse_arm("0 => { match op { 1 => { loop { break; } }, _ => b() } },");
        let result = classify_arm_body(&arm.body);
        assert!(matches!(result, ArmPattern::Unsupported(ref msg) if msg.contains("loop")));
    }

    #[test]
    fn classify_loop_as_unsupported() {
        let arm = parse_arm("0 => { loop { break; } },");
        let result = classify_arm_body(&arm.body);
        assert!(matches!(result, ArmPattern::Unsupported(ref msg) if msg.contains("loop")),);
    }

    #[test]
    fn classify_while_as_unsupported() {
        let arm = parse_arm("0 => { while cond { x(); } },");
        let result = classify_arm_body(&arm.body);
        assert!(matches!(result, ArmPattern::Unsupported(ref msg) if msg.contains("while")),);
    }

    #[test]
    fn classify_for_as_unsupported() {
        let arm = parse_arm("0 => { for i in 0..10 { x(); } },");
        let result = classify_arm_body(&arm.body);
        assert!(matches!(result, ArmPattern::Unsupported(ref msg) if msg.contains("for")),);
    }

    #[test]
    fn classify_nested_loop_in_if_as_unsupported() {
        let arm = parse_arm("0 => { if cond { loop { break; } } else { x(); } },");
        let result = classify_arm_body(&arm.body);
        assert!(matches!(result, ArmPattern::Unsupported(ref msg) if msg.contains("loop")),);
    }

    #[test]
    fn classify_read_number_as_abort() {
        let arm = parse_arm("0 => { read_number(); },");
        let result = classify_arm_body(&arm.body);
        assert!(matches!(result, ArmPattern::AbortPermanent));
    }
}
