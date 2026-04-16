//! Format the SSA representation as text for tests.
//!
//! Translated from `rpython/jit/codewriter/format.py`. Pyre's
//! [`SSARepr`](crate::flatten::SSARepr) holds typed [`FlatOp`] variants
//! rather than raw tuples, so the formatter produces the same textual
//! shape as upstream's `format_assembler` while reading the Rust types
//! directly.
//!
//! - Registers print as `%i<n>`, `%r<n>`, `%f<n>`.
//! - Constants print as `$<value>`.
//! - Labels print as `L<index>` (assigned in textual order, matching
//!   `format.py`'s `getlabelname`).
//!
//! **Partial port — known gaps versus `format.py:12-81`:**
//!
//! - Argument printing only handles `OpKind::Call`; every other
//!   `OpKind` variant currently emits its name with an empty argument
//!   list.  Callers who need precise per-variant fidelity should
//!   extend [`op_args_repr`] before relying on the output.
//! - Constants are not yet special-cased; upstream prints `$<repr>`
//!   for `Constant` values, but pyre routes constants through the
//!   `ValueId` / `RegKind` pair so the helper would need a
//!   constant-pool lookup that does not exist yet.
//! - Descrs and `ListOfKind` argument groups are not formatted.
//! - The `_insns_pos` map (`format.py:62-66`) and the trailing
//!   "missing label sentinel" branch are not reproduced.
//!
//! The reverse direction (`unformat_assembler`) is intentionally not
//! ported — the parity tests we run end-to-end build SSARepr through
//! the codewriter pipeline rather than parsing assembler text.  When a
//! caller actually needs to round-trip text → SSARepr the missing
//! function should be ported here mirroring `format.py:104-167` line
//! by line.

use std::collections::HashMap;
use std::fmt::Write;

use crate::flatten::{FlatOp, Label, RegKind, SSARepr};
use crate::model::ValueId;

/// format.py:12-81 `format_assembler(ssarepr)`.
pub fn format_assembler(ssarepr: &SSARepr) -> String {
    // First pass: collect every label that appears as a target so the
    // numbering matches format.py's getlabelname (labels are numbered in
    // first-seen order).
    let mut seenlabels: HashMap<Label, usize> = HashMap::new();
    let mut next_label = 0usize;
    let mut name_label = |label: Label, seen: &mut HashMap<Label, usize>, next: &mut usize| {
        *seen.entry(label).or_insert_with(|| {
            *next += 1;
            *next
        })
    };
    for op in &ssarepr.insns {
        match op {
            FlatOp::Jump(label) | FlatOp::GotoIfNot { target: label, .. } => {
                name_label(*label, &mut seenlabels, &mut next_label);
            }
            _ => {}
        }
    }

    let mut out = String::new();
    for op in &ssarepr.insns {
        match op {
            FlatOp::Label(label) => {
                if let Some(num) = seenlabels.get(label) {
                    let _ = writeln!(out, "L{num}:");
                }
            }
            FlatOp::Op(space_op) => {
                let _ = writeln!(
                    out,
                    "{} {}",
                    op_name(space_op),
                    op_args_repr(space_op, &ssarepr.value_kinds)
                );
            }
            FlatOp::Jump(label) => {
                let num = name_label(*label, &mut seenlabels, &mut next_label);
                let _ = writeln!(out, "goto L{num}");
            }
            FlatOp::GotoIfNot { cond, target } => {
                let num = name_label(*target, &mut seenlabels, &mut next_label);
                let _ = writeln!(
                    out,
                    "goto_if_not {}, L{num}",
                    register_repr(*cond, &ssarepr.value_kinds)
                );
            }
            FlatOp::Move { dst, src } => {
                let _ = writeln!(
                    out,
                    "move {} -> {}",
                    register_repr(*src, &ssarepr.value_kinds),
                    register_repr(*dst, &ssarepr.value_kinds)
                );
            }
            FlatOp::Live { live_values } => {
                let mut names: Vec<String> = live_values
                    .iter()
                    .map(|v| register_repr(*v, &ssarepr.value_kinds))
                    .collect();
                // format.py:76: `if asm[0] == '-live-': lst.sort()`.
                names.sort();
                let _ = writeln!(out, "-live- {}", names.join(", "));
            }
            FlatOp::Unreachable => {
                let _ = writeln!(out, "---");
            }
        }
    }
    out
}

/// format.py:83-102 `assert_format(ssarepr, expected)`.
///
/// Compares the formatted SSARepr with `expected` line by line.  When a
/// line differs we emit the same `Got:` / `Expected:` diff format as
/// upstream so failing tests are easy to read.
pub fn assert_format(ssarepr: &SSARepr, expected: &str) {
    let asm = format_assembler(ssarepr);
    let expected = if expected.is_empty() {
        String::new()
    } else {
        // Normalize multiline raw-string indentation the way
        // py.code.Source(expected).strip() does in upstream.
        normalize_expected(expected)
    };
    let asm_lines: Vec<&str> = asm.split('\n').collect();
    let exp_lines: Vec<&str> = expected.split('\n').collect();
    for (asm_line, exp_line) in asm_lines.iter().zip(exp_lines.iter()) {
        if asm_line != exp_line {
            let mut msg = String::new();
            msg.push_str("\n");
            let _ = writeln!(msg, "Got:      {asm_line}");
            let _ = writeln!(msg, "Expected: {exp_line}");
            let mut common = 0usize;
            for (a, e) in asm_line.chars().zip(exp_line.chars()) {
                if a == e {
                    common += 1;
                } else {
                    break;
                }
            }
            let _ = writeln!(msg, "          {}^^^^", " ".repeat(common));
            panic!("{msg}");
        }
    }
    assert_eq!(asm_lines.len(), exp_lines.len(), "line-count mismatch");
}

fn normalize_expected(expected: &str) -> String {
    // Strip the leading and trailing blank lines, then trim the common
    // indentation (similar to py.code.Source(...).strip()).
    let raw_lines: Vec<&str> = expected.split('\n').collect();
    let trimmed: Vec<&str> = raw_lines
        .iter()
        .copied()
        .skip_while(|l| l.trim().is_empty())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .skip_while(|l| l.trim().is_empty())
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    let indent = trimmed
        .iter()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.len() - l.trim_start().len())
        .min()
        .unwrap_or(0);
    let mut out = trimmed
        .iter()
        .map(|l| if l.len() >= indent { &l[indent..] } else { *l })
        .collect::<Vec<_>>()
        .join("\n");
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out
}

fn register_repr(v: ValueId, kinds: &HashMap<ValueId, RegKind>) -> String {
    let kind = kinds.get(&v).copied().unwrap_or(RegKind::Ref);
    let prefix = match kind {
        RegKind::Int => 'i',
        RegKind::Ref => 'r',
        RegKind::Float => 'f',
    };
    format!("%{prefix}{}", v.0)
}

fn op_name(op: &crate::model::SpaceOperation) -> String {
    use crate::model::OpKind;
    match &op.kind {
        OpKind::Call { .. } => "call".to_string(),
        // For the rest, fall back on a stable Debug-derived discriminant.
        other => format!("{:?}", other)
            .split('{')
            .next()
            .unwrap_or("?")
            .split('(')
            .next()
            .unwrap_or("?")
            .trim()
            .to_lowercase(),
    }
}

fn op_args_repr(op: &crate::model::SpaceOperation, kinds: &HashMap<ValueId, RegKind>) -> String {
    use crate::model::OpKind;
    let mut out = String::new();
    match &op.kind {
        OpKind::Call { args, .. } => {
            let parts: Vec<String> = args.iter().map(|v| register_repr(*v, kinds)).collect();
            out.push_str(&parts.join(", "));
        }
        _ => {
            // **Stub branch.**  Pyre's `OpKind` carries typed payloads
            // rather than positional argument tuples, so an upstream-
            // shaped formatter would need a per-variant projection
            // (constants → `$<repr>`, descrs, `ListOfKind` groups,
            // `Variable` regs).  Emitting an empty arg list keeps the
            // op name + result direction printable without lying about
            // the args; extend this branch when a parity test demands
            // it.
        }
    }
    if let Some(result) = op.result {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str("-> ");
        out.push_str(&register_repr(result, kinds));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flatten::{FlatOp, Label, SSARepr};

    fn empty_ssa() -> SSARepr {
        SSARepr {
            name: "test".into(),
            insns: Vec::new(),
            num_values: 0,
            num_blocks: 0,
            value_kinds: HashMap::new(),
        }
    }

    #[test]
    fn format_jump_emits_label() {
        let mut ssa = empty_ssa();
        let target = Label(0);
        ssa.insns.push(FlatOp::Jump(target));
        let text = format_assembler(&ssa);
        assert!(text.contains("goto L1"));
    }

    #[test]
    fn format_label_uses_first_seen_numbering() {
        let mut ssa = empty_ssa();
        ssa.insns.push(FlatOp::Jump(Label(7)));
        ssa.insns.push(FlatOp::Label(Label(7)));
        let text = format_assembler(&ssa);
        assert!(text.contains("goto L1"));
        assert!(text.contains("L1:"));
    }

    #[test]
    fn format_unreachable_marker() {
        let mut ssa = empty_ssa();
        ssa.insns.push(FlatOp::Unreachable);
        let text = format_assembler(&ssa);
        assert_eq!(text.trim(), "---");
    }

    #[test]
    fn assert_format_matches_simple_program() {
        let mut ssa = empty_ssa();
        ssa.insns.push(FlatOp::Jump(Label(0)));
        ssa.insns.push(FlatOp::Label(Label(0)));
        ssa.insns.push(FlatOp::Unreachable);
        assert_format(
            &ssa,
            "
            goto L1
            L1:
            ---
            ",
        );
    }
}
