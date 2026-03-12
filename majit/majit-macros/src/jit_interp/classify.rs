//! Match arm classifier: analyzes each arm of the opcode dispatch and
//! classifies it into a structural category.
//!
//! Most arms are `Lowerable` and handled by the JitCode lowerer.
//! Only a handful of structurally special patterns need dedicated codegen.

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

/// Detect branch group pattern: contains a nested match.
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
