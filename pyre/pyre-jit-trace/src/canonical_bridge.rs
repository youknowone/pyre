//! Portal-bridge wrapper.
//!
//! Wraps the build-time portal `JitCode` (`portal_jitcode()` in
//! `jitcode_runtime`) in a `PyJitCode` whose pyre-specific
//! `metadata` is the per-CodeObject derivation product (depth tables,
//! stack-base, slot color identity).  After the
//! canonical/runtime JitCode unification (`majit_translate::jitcode`
//! is the single type), this module owns just the
//! `install_portal_for` entry point — the body merely clones the
//! canonical's body and overrides `exec` to empty (the source-only
//! portal bypasses the per-jitcode `exec.descrs` pool entirely; descr
//! resolution goes through the global `ALL_DESCRS`).
//!
//! ## PRE-EXISTING-ADAPTATION
//!
//! RPython has no per-CodeObject portal wrapper. Upstream
//! `pypy/module/pypyjit/interp_jit.py portal_runner` *is* the single
//! portal jitcode shared by every Python frame — `pycode` is just a
//! green argument threaded through the trace and the interpreter
//! reads `code.co_code[pc]` directly during dispatch. Pyre cannot
//! reuse that shape because:
//!
//! 1. **Bytecode PC vs. JitCode PC mismatch.** CPython 3.13/3.14
//!    bytecodes (the only target Pyre supports) carry per-PC
//!    `LOAD_FAST_LOAD_FAST` / `LOAD_FAST_BORROW` super-instructions
//!    that the linear `eval_loop_jit` body cannot bias against. The
//!    portal-bridge wrapper exists so each user `CodeObject` carries
//!    its own `depth_at_py_pc` / `stack_base` / `pc_map` derivation
//!    (`pyre-jit-trace/src/liveness.rs::depth_at_py_pc`) the encoders
//!    and decoders consult against `pycode.instructions[py_pc]` —
//!    information PyPy 3.11 does not need because its bytecode has
//!    one PC per instruction.
//!
//! 2. **`metainterp_sd.jitcodes` lookup uses `frame.jitcode`
//!    identity.** `opencoder.py:777` writes `frame.jitcode.index`
//!    into the snapshot; the resume reader looks the JitCode up by
//!    that index. With a single shared portal jitcode every snapshot
//!    in pyre would collide, so the wrapper carries a per-CodeObject
//!    `Arc<JitCode>` identity that the SD list addresses
//!    independently.
//!
//! Both reasons sit in Pyre's "Python 3.11 vs 3.14 differences"
//! adapter bucket per the project parity policy. Each
//! `install_portal_for` call still performs a strict `clone()` of the
//! canonical body, so canonical RPython invariants (`startpoints`,
//! `c_num_regs_*`, constants pool) reach the runtime install
//! verbatim; the per-CodeObject delta is only the `PyJitCodeMetadata`
//! sidecar.
//!
//! ## Source-only descr resolution
//!
//! The build-time portal jitcode (`eval_loop_jit`) is produced by the
//! source-only translator
//! (`analyze_multiple_pipeline_with_vinfo_and_fnaddr_bindings`), so
//! 100% of its `'d'`/`'j'` operand sites resolve through the **global
//! pool**. The per-jitcode `exec.descrs` is dead weight for portal
//! execution and stays empty.
//!
//! ## RPython references
//!
//! - `rpython/jit/codewriter/jitcode.py:9-43` — the single `JitCode`
//!   class.
//! - `rpython/jit/metainterp/blackhole.py:103,288` — global
//!   `BlackholeInterpBuilder.descrs` / `BlackholeInterpreter.descrs`,
//!   the analog of pyre's `ALL_DESCRS`.
//! - `pypy/module/pypyjit/interp_jit.py portal_runner` — the
//!   single-jitcode portal shape pyre adapts away from.

use std::sync::Arc;

use majit_metainterp::jitcode::JitCode;

use crate::pyjitcode::{PyJitCode, PyJitCodeMetadata};

/// G.3a — Wrap the build-time portal canonical jitcode in a
/// `PyJitCode` whose pyre-specific `metadata` is the portal-bridge
/// derivation product.
///
/// Body: `portal_jitcode()` (canonical) → fresh `Arc<JitCode>` whose
/// body mirrors the canonical (code, c_num_regs_*, constants_*,
/// calldescr) with `exec` cleared and `jitdriver_sd` left unset →
/// wrap in `PyJitCode` with a `PyJitCodeMetadata` whose `pc_map` is
/// **empty** (the discriminator `PyJitCode::is_portal_bridge()`
/// checks) but whose other fields are populated from the user
/// `CodeObject`:
///   * `depth_at_py_pc` from the `LiveVars` forward stack analysis
///     (G.4.2 — `pyre-jit-trace/src/liveness.rs::depth_at_py_pc`).
///   * `portal_frame_reg` / `portal_ec_reg` sentinel-skipped via
///     `u16::MAX` until the portal-bridge readers consult the
///     canonical's own portal_red_colors directly.
///   * `stack_base` from `code.varnames.len() + ncells(code)` to
///     match the setup-time CodeWriter derivation.
///
/// The empty metadata is **not** an oversight — it is the marker that
/// distinguishes a portal-bridged `PyJitCode` from a drained
/// CodeWriter result. Drained jitcodes own a `pc_map` mapping Python
/// PCs to JitCode byte offsets because their bytecode IS the linear
/// translation of one Python function. The portal jitcode
/// (`eval_loop_jit` / `execute_opcode_step`) dispatches on
/// `pycode.instructions[pc]` *at runtime* via its own dispatch arms —
/// user PCs do not map to portal byte offsets.
///
/// ## Caller invariant (G.3a → G.3b → G.3c gate)
///
/// A portal-bridged `PyJitCode` MUST NOT reach readers that index
/// `metadata.pc_map[pc]`. The category audit is documented at
/// `PyJitCode::is_portal_bridge`; readers in Categories A and C are
/// portal-mode-aware via that discriminator.
///
/// ## jitdriver_sd invariant
///
/// G.4.3a-fix: `jitdriver_sd` left unset on the per-CodeObject
/// install. Upstream `call.py:147` `jd.mainjitcode = self.get_jitcode(
/// jd.portal_graph)` followed by `jd.mainjitcode.jitdriver_sd = jd`
/// assigns the flag to **exactly one** jitcode in
/// `metainterp_sd.jitcodes`. Pyre's portal-bridge wrapper produces a
/// fresh Arc per CodeObject; if every clone retained `jitdriver_sd`,
/// every per-CodeObject install would impersonate the canonical
/// mainjitcode and violate the single-portal invariant.
pub fn install_portal_for(
    code_ptr: *const pyre_interpreter::CodeObject,
    w_code: *const (),
) -> Arc<PyJitCode> {
    let canonical = crate::jitcode_runtime::portal_jitcode()
        .expect("install_portal_for: build-time portal canonical jitcode must exist");

    // Clone the canonical body so the runtime install owns its own
    // calldescr / code / constants pools.  The cloned body inherits the
    // canonical's `code` / `constants_*` / `c_num_regs_*` / `startpoints`
    // / etc. — RPython `jitcode.py:22-42 setup` fields — verbatim. The
    // top-level `exec` (descrs pool, RPython
    // `BlackholeInterpBuilder.descrs` analog) is intentionally left
    // empty: the source-only portal jitcode bypasses the per-jitcode
    // pool entirely and resolves descrs through the global
    // `ALL_DESCRS`.
    let body = canonical.body().clone();
    let mut runtime = JitCode::new(canonical.name.clone());
    runtime.set_body(body);
    runtime.fnaddr = canonical.fnaddr;
    // `jitcode.index` / `jitdriver_sd` are intentionally left unset —
    // each per-CodeObject install is a fresh entry in the SD's jitcodes
    // list, and `state::jitcode_for` is the single setter for the
    // SD-local position (RPython `codewriter.py:68 jitcode.index =
    // index` matches this single-set discipline).  The single-portal
    // invariant (`call.py:147` — exactly one jitcode in
    // `metainterp_sd.jitcodes` carries `jitdriver_sd`) likewise rules
    // out copying the canonical's value.
    let runtime = Arc::new(runtime);

    // Portal-bridge installs do not run the per-CodeObject regalloc, so
    // the post-regalloc colors of the portal red inputargs (`pypy/module/
    // pypyjit/interp_jit.py:67 reds = ['frame', 'ec']`) are unknown
    // here.  The snapshot serializer at
    // `trace_opcode::get_list_of_active_boxes` sentinel-skips
    // `u16::MAX` so portal-bridge frames that hit guard capture fall
    // through to the locals/stack-only resolution path.
    //
    // RPython parity: `rpython/jit/metainterp/warmspot.py setup_jit`
    // derives the red arg inputarg indices from the jitdriver's
    // `reds=[]` declaration, not from the canonical jitcode's name.
    // Pyre lacks the jitdriver greens/reds → register-slot
    // infrastructure to do the analogous derivation here; the sentinel
    // is the safe default until the portal-bridge readers consult the
    // canonical's own portal_red_colors directly.
    let portal_frame_reg = u16::MAX;
    let portal_ec_reg = u16::MAX;

    // `stack_base` is the absolute index of the operand stack within
    // `PyFrame.locals_cells_stack_w`, matching the setup-time
    // CodeWriter derivation.
    //
    // G.4.2 also derives `depth_at_py_pc` from the same user
    // `CodeObject` via the cached `LiveVars` forward stack analysis
    // (`pyre-jit-trace/src/liveness.rs:225-260`).  This populates the
    // portal-bridge metadata table that G.4.3 routes both the encoder
    // (`trace_opcode.rs:get_list_of_active_boxes`) and the decoder
    // (`state.rs:frame_value_count_at` / `restore_guard_failure_values`)
    // through, restoring the upstream packed-liveness invariant for
    // portal-bridge frames.  Both sides consume `stack_base +
    // depth_at_py_pc[pc]` Ref-typed boxes (G.4.3a-fix — pre-fix used
    // `nlocals_from_code() = varnames.len()` which dropped `ncells`).
    //
    // `pc_map` stays empty so `PyJitCode::is_portal_bridge()`
    // (`pyjitcode.rs:117-119`) keeps reporting `true` for portal-bridge
    // installs — the discriminator the G.3 reader audit branches on.
    // Portal-bridge install does not run regalloc, so the post-color
    // maps are populated as identity over the PyFrame slot ranges. Both
    // map lengths must match the runtime PyFrame allocation
    // (`pyframe.rs:1576` `nlocals + ncells + max_stackdepth`) so the
    // bridge fallback at `state.rs::setup_bridge_sym`
    // (`stack_base + stack_slot_color_map.len()`) reconstructs the full
    // `len(locals_cells_stack_w)` virtualizable shape rather than
    // collapsing to `stack_base + 0`. The contract is documented at
    // `PyreJitCode::stack_slot_color_map`. With no regalloc the color of each slot is
    // its own pre-color, hence the identity fill.
    let (stack_base, depth_at_py_pc, stack_slot_color_map, pyre_color_for_semantic_local) =
        if code_ptr.is_null() {
            (0, Vec::new(), Vec::new(), Vec::new())
        } else {
            let code = unsafe { &*code_ptr };
            let nlocals = code.varnames.len();
            let stack_base = nlocals + pyre_interpreter::pyframe::ncells(code);
            let depth_at_py_pc = crate::liveness::liveness_for(code_ptr).depth_at_py_pc();
            let max_stackdepth = code.max_stackdepth as usize;
            let stack_slot_color_map: Vec<u16> = (0..max_stackdepth as u16)
                .map(|d| stack_base as u16 + d)
                .collect();
            let pyre_color_for_semantic_local: Vec<u16> = (0..nlocals as u16).collect();
            (
                stack_base,
                depth_at_py_pc,
                stack_slot_color_map,
                pyre_color_for_semantic_local,
            )
        };

    Arc::new(PyJitCode::from_parts(
        runtime,
        PyJitCodeMetadata {
            pc_map: Vec::new(),
            depth_at_py_pc,
            portal_frame_reg,
            portal_ec_reg,
            stack_base,
            stack_slot_color_map,
            pyre_color_for_semantic_local,
        },
        code_ptr,
        w_code,
        false,
        None,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::jitcode_runtime::portal_jitcode;

    /// G.3a — `install_portal_for` returns a fresh `Arc<PyJitCode>`
    /// each call (no shared interior state between installs).
    #[test]
    fn install_portal_for_returns_independent_arcs() {
        let r1 = install_portal_for(std::ptr::null(), std::ptr::null());
        let r2 = install_portal_for(std::ptr::null(), std::ptr::null());
        assert!(
            !Arc::ptr_eq(&r1, &r2),
            "each install_portal_for call must produce a fresh Arc"
        );
    }

    /// G.3a — A portal-bridged `PyJitCode` keeps `pc_map` empty so
    /// `is_portal_bridge()` (`pyjitcode.rs:117-119`) reports `true`.
    /// This is the **portal-mode marker** the G.3 reader audit
    /// branches on.  G.4.2 added the `depth_at_py_pc` derivation but
    /// only populates it when `code_ptr` is non-null — the
    /// null-fixture path preserves the original empty observation so
    /// the marker invariant still surfaces here.
    #[test]
    fn install_portal_for_metadata_is_empty_marker() {
        let pyjit = install_portal_for(std::ptr::null(), std::ptr::null());
        assert!(
            pyjit.metadata.pc_map.is_empty(),
            "portal-bridged pc_map must be empty (G.3a marker invariant)"
        );
        // Null code_ptr ⇒ no user CodeObject to analyse ⇒
        // `depth_at_py_pc` falls back to `Vec::new()`.  Production
        // callers always pass a non-null wrapper; that path is
        // exercised by
        // `install_portal_for_populates_depth_at_py_pc_for_user_code`.
        assert!(pyjit.metadata.depth_at_py_pc.is_empty());
        // RPython-orthodox revert: `portal_frame_reg` /
        // `portal_ec_reg` are sentinel-skipped pending the proper
        // greens/reds → register-slot derivation port.  See
        // `install_portal_for` for the rationale.
        assert_eq!(pyjit.metadata.portal_frame_reg, u16::MAX);
        assert_eq!(pyjit.metadata.portal_ec_reg, u16::MAX);
        // Test fixture passes null code_ptr; install_portal_for falls
        // back to stack_base = 0 (production callers always pass a
        // non-null wrapper, so stack_base = nlocals + ncells).
        assert_eq!(pyjit.metadata.stack_base, 0);
        assert!(!pyjit.has_abort);
        assert!(pyjit.merge_point_pc.is_none());
        assert!(
            !pyjit.is_populated(),
            "is_populated() must report false for portal-bridged jitcode \
             so existing skeleton-aware readers fall back gracefully"
        );
    }

    /// G.4.2 — Non-null `code_ptr` path: `install_portal_for` derives
    /// `depth_at_py_pc` from the user `CodeObject` via the cached
    /// `LiveVars` forward stack analysis.  The vector must be sized
    /// to `code.instructions.len()` so the portal-bridge
    /// encoder/decoder (G.4.3+) can index it by user PC.  `pc_map`
    /// stays empty and `is_portal_bridge()` keeps reporting `true` —
    /// the discriminator is independent of the depth table.
    ///
    /// Sanity bounds:
    ///   * Length matches `code.instructions.len()`.
    ///   * Maximum derived depth never exceeds `code.max_stackdepth`
    ///     (CPython compiler invariant — the max recorded by the
    ///     compiler is an upper bound on the runtime stack height at
    ///     any reachable PC).
    ///   * `is_portal_bridge()` stays true.
    #[test]
    fn install_portal_for_populates_depth_at_py_pc_for_user_code() {
        let source = "\
def f(x, y):
    a = x + y
    b = a * x
    return b - y
";
        let module = pyre_interpreter::compile_exec(source).expect("compile failed");
        let user_code = module
            .constants
            .iter()
            .find_map(|c| match c {
                pyre_interpreter::bytecode::ConstantData::Code { code } => Some((**code).clone()),
                _ => None,
            })
            .expect("expected nested function code object");
        let user_code_ptr = Box::into_raw(Box::new(user_code));

        let pyjit = install_portal_for(user_code_ptr, std::ptr::null());

        let code_ref = unsafe { &*user_code_ptr };
        assert_eq!(
            pyjit.metadata.depth_at_py_pc.len(),
            code_ref.instructions.len(),
            "depth_at_py_pc must have one entry per Python PC so the \
             portal-bridge encoder/decoder can index by user PC"
        );
        let max_observed = pyjit
            .metadata
            .depth_at_py_pc
            .iter()
            .copied()
            .max()
            .unwrap_or(0);
        assert!(
            (max_observed as u32) <= code_ref.max_stackdepth,
            "derived depth {} exceeds compiler's max_stackdepth {}",
            max_observed,
            code_ref.max_stackdepth
        );
        assert!(
            pyjit.is_portal_bridge(),
            "populating depth_at_py_pc must NOT flip the portal-bridge \
             discriminator — pc_map stays empty"
        );
        assert!(
            pyjit.metadata.pc_map.is_empty(),
            "portal-bridge pc_map must remain empty (G.3a marker invariant)"
        );
        // stack_base = nlocals + ncells for a non-null code_ptr; for
        // this fixture (no closures / cell vars), ncells = 0.
        assert_eq!(
            pyjit.metadata.stack_base,
            code_ref.varnames.len() + pyre_interpreter::pyframe::ncells(code_ref)
        );

        // Drop the leaked Box now the wrapper has been examined.
        drop(unsafe { Box::from_raw(user_code_ptr) });
    }

    /// G.3b — `PyJitCode::is_portal_bridge()` distinguishes a
    /// portal-bridged install from `PyJitCode::skeleton(...)` and
    /// from a fully-drained per-CodeObject `PyJitCode`. This is the
    /// hook G.3c readers branch on.
    ///
    /// * Portal-bridge: `jitcode.code` non-empty + `pc_map` empty.
    /// * Skeleton: `jitcode.code` empty + `pc_map` empty.
    /// * Drained CodeWriter result: `jitcode.code` non-empty +
    ///   `pc_map` populated by the setup-time drain.
    ///
    /// The discriminator must be true for portal-bridge and false
    /// for the other two states.
    #[test]
    fn is_portal_bridge_distinguishes_install_from_skeleton() {
        let portal = install_portal_for(std::ptr::null(), std::ptr::null());
        assert!(
            portal.is_portal_bridge(),
            "install_portal_for product must report is_portal_bridge() == true"
        );

        let skeleton = PyJitCode::skeleton(std::ptr::null(), std::ptr::null(), None);
        assert!(
            !skeleton.is_portal_bridge(),
            "skeleton has empty jitcode.code — must NOT report portal-bridge"
        );

        // Drained CodeWriter simulation: clone the portal jitcode
        // (so `code` is non-empty) and stamp a synthesized `pc_map`
        // entry. This mirrors a setup-time drain result — both
        // fields populated.
        let drained_runtime = (*portal.jitcode).clone();
        // Defensively assert the precondition the discriminator
        // relies on — if a future refactor empties `code` here, the
        // assertion below would still pass but would no longer
        // exercise the intended case.
        assert!(
            !drained_runtime.code.is_empty(),
            "test setup: cloned portal jitcode must have non-empty code"
        );
        let drained = PyJitCode::from_parts(
            std::sync::Arc::new(drained_runtime),
            PyJitCodeMetadata {
                pc_map: vec![0],
                depth_at_py_pc: Vec::new(),
                portal_frame_reg: 0,
                portal_ec_reg: 0,
                stack_base: 0,
                stack_slot_color_map: Vec::new(),
                pyre_color_for_semantic_local: Vec::new(),
            },
            std::ptr::null(),
            std::ptr::null(),
            false,
            None,
        );
        assert!(
            !drained.is_portal_bridge(),
            "drained per-CodeObject (pc_map populated) must NOT report portal-bridge"
        );
    }

    /// `install_portal_for` preserves every canonical field on the
    /// wrapped runtime jitcode and clears `jitdriver_sd` per
    /// G.4.3a-fix (call.py:147 single-jitdriver_sd invariant). The
    /// `exec.descrs` pool stays empty — the source-only portal
    /// bypasses the per-jitcode descr pool entirely.
    #[test]
    fn install_portal_for_preserves_canonical_fields() {
        let canonical = portal_jitcode().expect("build-time portal jitcode must exist");
        let pyjit = install_portal_for(std::ptr::null(), std::ptr::null());
        let body = canonical.body();
        assert_eq!(pyjit.jitcode.name, canonical.name);
        assert_eq!(pyjit.jitcode.code, body.code);
        assert_eq!(pyjit.jitcode.c_num_regs_i, body.c_num_regs_i);
        assert_eq!(pyjit.jitcode.c_num_regs_r, body.c_num_regs_r);
        assert_eq!(pyjit.jitcode.c_num_regs_f, body.c_num_regs_f);
        assert_eq!(pyjit.jitcode.constants_i, body.constants_i);
        assert_eq!(pyjit.jitcode.constants_r, body.constants_r);
        assert_eq!(pyjit.jitcode.constants_f, body.constants_f);
        assert_eq!(pyjit.jitcode.fnaddr, canonical.fnaddr);
        assert!(
            canonical.jitdriver_sd().is_some(),
            "build-time canonical carries jitdriver_sd (it IS the portal); \
             this test pins the precondition the install_portal_for clear \
             invariant rests on"
        );
        assert!(
            pyjit.jitcode.jitdriver_sd().is_none(),
            "install_portal_for must clear jitdriver_sd so the per-CodeObject \
             clone does not impersonate the canonical mainjitcode \
             (call.py:147 single-jitdriver_sd invariant)"
        );
        assert!(
            pyjit.jitcode.exec.descrs.is_empty(),
            "exec.descrs stays empty for the portal-bridge install — the \
             source-only portal bypasses the per-jitcode descr pool"
        );
    }

    /// Constants-f bit-pattern fidelity — both canonical and the
    /// portal-bridge install store the float pool as `i64` (RPython
    /// `longlong.float2longlong`); cloning the body must preserve
    /// every entry verbatim.
    #[test]
    fn install_portal_for_constants_f_round_trip_via_bits() {
        let canonical = portal_jitcode().expect("build-time portal jitcode must exist");
        let body = canonical.body();
        let pyjit = install_portal_for(std::ptr::null(), std::ptr::null());
        for (orig, bridged) in body
            .constants_f
            .iter()
            .zip(pyjit.jitcode.constants_f.iter())
        {
            assert_eq!(
                orig, bridged,
                "constants_f entries must round-trip identically through \
                 the portal-bridge install (longlong.float2longlong parity)"
            );
        }
    }
}
