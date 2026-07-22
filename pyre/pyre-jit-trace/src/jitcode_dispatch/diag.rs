//! pc-map inversion helpers and `PYRE_PCMAP_*` audit probes.
//!
//! **Parity:** pyre-specific — the jitcode-pc <-> python-pc inversion and
//! the `PYRE_PCMAP_*` audit probes have no `rpython/jit/metainterp/`
//! counterpart (PyPy's pc handling is codewriter-side).
//!
//! Extracted verbatim from `jitcode_dispatch/mod.rs`: the jitcode-pc ->
//! python-pc inversion (`python_pc_for_jitcode_pc` + floor-boundary
//! helpers), the `skip_python_trivia_forward` boundary walker, and the
//! report-only `PYRE_PCMAP_*` audit probes.

use super::*;

/// `PYRE_PCMAP_RECIPE_RESULTCOLOR_AUDIT` is a report-only census for the
/// recipe resume-coordinate result-color reader and the multi-frame callee
/// diagnostic's inversion. The optional `_PROBE` receives a fire row followed
/// by its verdict, since `check.py` discards diagnostic stderr.
pub(crate) fn pcmap_recipe_resultcolor_audit_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_PCMAP_RECIPE_RESULTCOLOR_AUDIT").is_some())
}

/// `PYRE_PCMAP_CONTAINING_AUDIT`: assert the Slice-B floor-only depth twin
/// (`depth_containing_for_jitcode_pc`) equals the raw
/// `depth_at_py_pc[vstack_containing_py_pc(jit_pc)]` read at both consumer
/// seams. Off in production; the gated branch is the only added code.
pub(crate) fn pcmap_containing_audit_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_PCMAP_CONTAINING_AUDIT").is_some())
}

/// `PYRE_PCMAP_AFTERRESIDUAL_AUDIT`: assert the Slice-C after-residual depth
/// twin (`depth_after_residual_for_jitcode_pc`) equals the raw
/// `depth_at_py_pc[semantic_fallthrough_pc(python_pc_for_jitcode_pc(jit_pc))]`
/// read at each consumer seam. Off in production; the gated branch is the only
/// added code.
pub(crate) fn pcmap_afterresidual_audit_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_PCMAP_AFTERRESIDUAL_AUDIT").is_some())
}

pub(crate) fn pcmap_recipe_resultcolor_audit_probe(site: &'static str, verdict: &'static str) {
    if let Some(path) = std::env::var_os("PYRE_PCMAP_RECIPE_RESULTCOLOR_AUDIT_PROBE") {
        use std::io::Write;

        if let Ok(mut probe) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = writeln!(probe, "recipe_resultcolor\t{site}\t{verdict}");
        }
    }
}

/// Resolve the authoritative paused-parent coordinate at a Python-native
/// consumer. A missing JitCode/code object is reported to the caller as the
/// same multi-frame snapshot decline used by nearby unavailable-coordinate
/// paths; it is never a panic.
pub(crate) fn resolve_parent_resume_py_pc(parent: &InlineParentFrame) -> Option<u32> {
    match parent.resume_coord {
        ParentResumeCoord::Backxlat(jitcode_pc) => Some(crate::state::backxlat_py_pc(
            parent.jitcode_index as i32,
            jitcode_pc as i32,
        ) as u32),
        ParentResumeCoord::CallFallthrough(call_jit_pc) => {
            let Some(pjc) = crate::state::pyjitcode_for_jitcode_index(parent.jitcode_index as i32)
            else {
                return None;
            };
            if pjc.code_ptr.is_null() {
                return None;
            }
            let call_py_pc = python_pc_for_jitcode_pc(&pjc.metadata, call_jit_pc) as usize;
            let code = unsafe { &*pjc.code_ptr };
            Some(crate::pyjitpl::semantic_fallthrough_pc(code, call_py_pc) as u32)
        }
    }
}

pub(crate) fn floor_boundary_at_or_after(
    metadata: &crate::PyJitCodeMetadata,
    jit_pc: usize,
) -> Option<(usize, u32)> {
    let table = &metadata.py_floor_by_jit_pc;
    let idx = table.partition_point(|&(off, _)| (off as usize) < jit_pc);
    table.get(idx).map(|&(off, py)| (off as usize, py))
}

pub(crate) fn first_floor_boundary_for_py(
    metadata: &crate::PyJitCodeMetadata,
    py_pc: u32,
) -> Option<(usize, u32)> {
    metadata
        .py_floor_by_jit_pc
        .iter()
        .find(|&&(_, py)| py == py_pc)
        .map(|&(off, py)| (off as usize, py))
}

pub(crate) fn python_pc_for_jitcode_pc(metadata: &crate::PyJitCodeMetadata, jit_pc: usize) -> u32 {
    if !metadata.py_floor_by_jit_pc.is_empty() {
        let pivot = metadata
            .block_head_py_by_jit_pc
            .binary_search_by_key(&jit_pc, |&(off, _)| off)
            .ok()
            .map(|i| metadata.block_head_py_by_jit_pc[i].1)
            .or_else(|| {
                crate::pyjitcode::floor_segment_for_jitcode_pc(&metadata.py_floor_by_jit_pc, jit_pc)
                    .map(|(_, py)| py)
            })
            .expect("drained JitCode PC floor pivot must begin at byte offset zero");
        return pivot;
    }
    0
}

/// Resolve an in-flight body channel exactly where a stash match needs its
/// Python body pc. A missing JitCode entry is deliberately `None`: callers
/// treat it as no match and retain the legacy replay/delivery fallback.
pub(crate) fn inflight_foriter_body_pc(body: InflightForiterBody) -> Option<usize> {
    match body {
        InflightForiterBody::Py(body_pc) => Some(body_pc),
        InflightForiterBody::Jit {
            outer_jitcode_index,
            op_pc,
        } => crate::state::pyjitcode_for_jitcode_index(outer_jitcode_index as i32)
            .map(|jc| python_pc_for_jitcode_pc(&jc.metadata, op_pc) as usize + 1),
    }
}

/// Capture the native coordinates that identify a `for_iter_next` residual.
/// The Python continue-arm fallthrough is intentionally not derived here.
pub(crate) fn fbw_foriter_body_from_op_pc<Sym: WalkSym>(
    snapshot_sym: *const Sym,
    op_pc: usize,
) -> Option<InflightForiterBody> {
    if snapshot_sym.is_null() {
        return None;
    }
    // SAFETY: the snapshot root stays live for the full-body walk. Only the
    // immutable JitCode identity is read here.
    let sym = unsafe { &*snapshot_sym };
    if sym.jitcode().is_null() {
        return None;
    }
    Some(InflightForiterBody::Jit {
        outer_jitcode_index: unsafe { (*sym.jitcode()).index as u32 },
        op_pc,
    })
}

/// Forward-skip Python trivia (`Cache` / `ExtendedArg` / `Resume` / `Nop`
/// / `NotTaken`) from `py_pc` to the next executable opcode.  Mirrors the
/// forward trivia walk in [`crate::pyjitpl::semantic_fallthrough_pc`]
/// but starts AT `py_pc` (not `py_pc + 1`) so a coordinate that already
/// points at trivia is advanced.  A resume coordinate must be a real
/// opcode boundary; the resume reader's own backtrack walks trivia
/// BACKWARD, which is wrong for a `NOT_TAKEN` branch-target coordinate.
pub fn skip_python_trivia_forward(code: &pyre_interpreter::CodeObject, mut py_pc: usize) -> usize {
    use pyre_interpreter::bytecode::Instruction;
    loop {
        match pyre_interpreter::decode_instruction_at(code, py_pc) {
            Some((
                Instruction::ExtendedArg
                | Instruction::Resume { .. }
                | Instruction::Nop
                | Instruction::Cache
                | Instruction::NotTaken,
                _,
            )) => py_pc += 1,
            _ => return py_pc,
        }
    }
}
