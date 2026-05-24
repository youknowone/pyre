//! Tiny **FunctionGraph-like** target IR for the spike.
//!
//! Mirrors the *shape* of `majit-translate::model::FunctionGraph` (Block,
//! SpaceOperation, Link, ExitSwitch, ExitCase) without depending on the
//! real crate.  The point is to (a) demonstrate the ULLBC → CFG mapping
//! and (b) emit a canonical text form that can be diffed against a
//! hand-authored expected file.
//!
//! Deliberate divergences from the real `FunctionGraph`:
//!   - No Variable-identity bridge — we use plain `var_<i>` strings.
//!   - No `framestate`, no `last_exception` / `llexitcase` on Links.
//!   - No `returnblock` / `exceptblock` sentinels — `return` is a real
//!     terminator at the per-block level, matching MIR directly.
//!   - Operations carry a single `OpKind` string rather than the full
//!     `OpKind` enum tree.  Sufficient for canonical diffing.

#[derive(Debug, Clone, Default)]
pub struct FunctionGraph {
    pub name: String,
    pub args: Vec<String>,
    pub blocks: Vec<Block>,
}

#[derive(Debug, Clone)]
pub struct Block {
    pub id: u32,
    pub inputargs: Vec<String>,
    pub operations: Vec<SpaceOperation>,
    pub exit: ExitKind,
}

#[derive(Debug, Clone)]
pub struct SpaceOperation {
    pub result: Option<String>,
    pub op: String,
    pub args: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum ExitKind {
    Return(String),
    /// Unwind out of the function (panic propagation).  Maps to a
    /// FunctionGraph link to the `exceptblock` sentinel; the spike
    /// flattens it to a marker.
    Resume,
    /// Unconditional control-flow.
    Goto(Link),
    /// Two-way boolean / two-way enum discriminant.
    Switch {
        discr: String,
        cases: Vec<(Option<String>, Link)>,
    },
    /// Function call terminator with success + unwind continuations.
    Call {
        callee: String,
        args: Vec<String>,
        dest: Option<String>,
        on_success: Link,
        on_unwind: Link,
    },
    /// Assertion (overflow check etc.) — pass-through on success.
    Assert {
        cond: String,
        on_success: Link,
        on_unwind: Link,
    },
}

#[derive(Debug, Clone)]
pub struct Link {
    pub target: u32,
    pub args: Vec<String>,
    /// Optional label used by Switch arms (e.g. `"0"`, `"true"`, `"default"`).
    /// Stored on the Link for symmetry with the real `model::Link.exitcase`;
    /// the prototype's canonical printer renders the case label on the
    /// Switch arm itself, so this field is set at construction but not
    /// currently read.
    #[allow(dead_code)]
    pub exitcase: Option<String>,
}

impl FunctionGraph {
    pub fn canonical(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!("graph {} ({})\n", self.name, self.args.join(", ")));
        for b in &self.blocks {
            out.push_str(&format!(
                "  block bb{} ({})\n",
                b.id,
                b.inputargs.join(", ")
            ));
            for op in &b.operations {
                let lhs = op.result.as_deref().unwrap_or("_");
                out.push_str(&format!(
                    "    {} = {}({})\n",
                    lhs,
                    op.op,
                    op.args.join(", ")
                ));
            }
            match &b.exit {
                ExitKind::Return(v) => out.push_str(&format!("    return {v}\n")),
                ExitKind::Resume => out.push_str("    resume\n"),
                ExitKind::Goto(l) => out.push_str(&format!("    goto {}\n", fmt_link(l))),
                ExitKind::Switch { discr, cases } => {
                    out.push_str(&format!("    switch {discr}\n"));
                    for (case, link) in cases {
                        let label = case.as_deref().unwrap_or("default");
                        out.push_str(&format!("      case {label}: {}\n", fmt_link(link)));
                    }
                }
                ExitKind::Call {
                    callee,
                    args,
                    dest,
                    on_success,
                    on_unwind,
                } => {
                    let dst = dest.as_deref().unwrap_or("_");
                    out.push_str(&format!(
                        "    call {dst} = {callee}({})\n",
                        args.join(", ")
                    ));
                    out.push_str(&format!("      ok:     {}\n", fmt_link(on_success)));
                    out.push_str(&format!("      unwind: {}\n", fmt_link(on_unwind)));
                }
                ExitKind::Assert {
                    cond,
                    on_success,
                    on_unwind,
                } => {
                    out.push_str(&format!("    assert {cond}\n"));
                    out.push_str(&format!("      ok:     {}\n", fmt_link(on_success)));
                    out.push_str(&format!("      unwind: {}\n", fmt_link(on_unwind)));
                }
            }
        }
        out
    }
}

fn fmt_link(l: &Link) -> String {
    if l.args.is_empty() {
        format!("-> bb{}", l.target)
    } else {
        format!("-> bb{}({})", l.target, l.args.join(", "))
    }
}
