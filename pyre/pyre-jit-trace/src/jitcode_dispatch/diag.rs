//! pc-map audit probes.
//!
//! **Parity:** pyre-specific — the `PYRE_PCMAP_*` audit probes have no
//! `rpython/jit/metainterp/` counterpart (PyPy's pc handling is
//! codewriter-side).
//!
//! Extracted verbatim from `jitcode_dispatch/mod.rs`: the
//! `skip_python_trivia_forward` boundary walker and the report-only
//! `PYRE_PCMAP_*` audit probes.

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
/// `depth_at_py_pc[semantic_fallthrough_pc(containing_py_pc_for_jitcode_pc(jit_pc))]`
/// read at each consumer seam. Off in production; the gated branch is the only
/// added code.
pub(crate) fn pcmap_afterresidual_audit_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_PCMAP_AFTERRESIDUAL_AUDIT").is_some())
}

/// `PYRE_FBW_STRICT_DIAG`: name the opcode that rejected a callee from the
/// strict straight-line fast path.
///
/// Cached like the audit flags above, and for the reason `majit-backend-dynasm`
/// states at `majit_log_enabled`: `std::env::var_os` takes a global lock and
/// walks the env table on every call, and this one is read from inside
/// `callee_fast_path_inlinable`'s per-opcode decode loop.
pub(crate) fn fbw_strict_diag_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_FBW_STRICT_DIAG").is_some())
}

/// `PYRE_FBW_INLINE_DIAG`: name the reason a call site declined to inline.
pub(crate) fn fbw_inline_diag_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_FBW_INLINE_DIAG").is_some())
}

/// `PYRE_FBW_MF_DIAG`: multi-frame callee window diagnostics.
pub(crate) fn fbw_mf_diag_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_FBW_MF_DIAG").is_some())
}

/// `PYRE_P2_DIAG`: self-recursive CALL_ASSEMBLER fold diagnostics.
pub(crate) fn p2_diag_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PYRE_P2_DIAG").is_some())
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
        ParentResumeCoord::Backxlat(jitcode_pc) => {
            // #73: read the forward py_pc twin, a codewriter-built
            // trivia-normalized twin of the containing coordinate. The
            // containing lookup survives only for the empty-twin class.
            let twin = crate::state::pyjitcode_for_jitcode_index(parent.jitcode_index as i32)
                .and_then(|pjc| pjc.forward_py_pc_for_jitcode_pc(jitcode_pc));
            Some(twin.unwrap_or_else(|| {
                crate::py_coord::trivia_normalized_py_pc_for_jitcode_pc(
                    parent.jitcode_index as i32,
                    jitcode_pc as i32,
                ) as u32
            }))
        }
        ParentResumeCoord::CallFallthrough(call_jit_pc) => {
            let Some(pjc) = crate::state::pyjitcode_for_jitcode_index(parent.jitcode_index as i32)
            else {
                return None;
            };
            if pjc.code_ptr.is_null() {
                return None;
            }
            // #73 Slice 4: read the forward after-residual fallthrough twin.
            // The containing lookup survives only for the empty-twin class
            // (populated code with no Python map) and as the audit oracle.
            let legacy = || {
                let call_py_pc =
                    crate::py_coord::containing_py_pc_for_jitcode_pc(&pjc.metadata, call_jit_pc)
                        as usize;
                let code = unsafe { &*pjc.code_ptr };
                crate::pyjitpl::semantic_fallthrough_pc(code, call_py_pc) as u32
            };
            let twin = pjc
                .after_residual_fallthrough_py_pc_populated()
                .then(|| pjc.after_residual_fallthrough_py_pc_for_jitcode_pc(call_jit_pc))
                .flatten();
            match twin {
                Some(ft) => {
                    if pcmap_afterresidual_audit_enabled() {
                        assert_eq!(
                            ft,
                            legacy(),
                            "PYRE_PCMAP_AFTERRESIDUAL_AUDIT: parent-resume fallthrough-py twin diverged at jit_pc {call_jit_pc}"
                        );
                    }
                    Some(ft)
                }
                None => Some(legacy()),
            }
        }
    }
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
        } => crate::state::pyjitcode_for_jitcode_index(outer_jitcode_index as i32).map(|jc| {
            crate::py_coord::containing_py_pc_for_jitcode_pc(&jc.metadata, op_pc) as usize + 1
        }),
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
