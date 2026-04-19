/// Python bytecode → JitCode compiler.
///
/// RPython: rpython/jit/codewriter/codewriter.py
///
/// RPython's CodeWriter transforms flow graphs → JitCode via a 4-step pipeline:
///   1. jtransform  — rewrite operations
///   2. regalloc    — assign registers
///   3. flatten     — CFG → linear SSARepr
///   4. assemble    — SSARepr → JitCode bytecode
///
/// For pyre, Python bytecodes are already linearized and register-allocated
/// (fast locals = registers, value stack = runtime stack). Steps 1-3 collapse
/// into a single bytecode-to-bytecode translation.
///
/// The Assembler (majit JitCodeBuilder) is RPython's assembler.py equivalent.
use std::cell::{OnceCell, RefCell, UnsafeCell};
use std::collections::HashMap;

use majit_ir::OpCode;
use majit_metainterp::jitcode::{JitCode, JitCodeBuilder, LivenessInfo};
use pyre_jit_trace::{PyJitCode, PyJitCodeMetadata};

use super::assembler::Assembler;
use super::ssa_emitter::SSAReprEmitter;
use pyre_interpreter::bytecode::{CodeObject, Instruction, OpArgState};
use pyre_interpreter::runtime_ops::{binary_op_tag, compare_op_tag};

use super::flatten::{
    CallFlavor, DescrOperand, Insn, Kind, ListOfKind, Operand, Register, ResKind, SSARepr, TLabel,
};

// ---------------------------------------------------------------------------
// RPython: codewriter/flatten.py KINDS = ['int', 'ref', 'float']
// ---------------------------------------------------------------------------

/// Python `var_num` → flat index into the `locals_cells_stack_w`
/// virtualizable array.
///
/// PyFrame lays out locals, cells, and the value stack in a single
/// vector; `var_num` from `LOAD_FAST`/`STORE_FAST` is already a direct
/// offset into that vector (no indirection).
///
/// jtransform.py:1877 do_fixed_list_getitem / :1898 do_fixed_list_setitem
/// rewrite portal local accesses into `getarrayitem_vable_r` /
/// `setarrayitem_vable_r` against the virtualizable array.
#[inline]
fn local_to_vable_slot(var_num: usize) -> usize {
    var_num
}

// `PyJitCode` and `PyJitCodeMetadata` live in `pyre_jit_trace::pyjitcode`
// so both the codewriter (here) and the trace/blackhole runtime can hold
// the same `Arc<PyJitCode>` instances.

#[derive(Clone, Copy)]
struct ExceptionCatchSite {
    landing_label: u16,
    handler_py_pc: usize,
    stack_depth: u16,
    push_lasti: bool,
    lasti_py_pc: usize,
}

/// RPython: per-graph output of `perform_register_allocation` over the
/// three register kinds (codewriter.py:46-48). pyre's regalloc is
/// trivial — fast locals occupy the bottom of the ref register file
/// and the value stack stacks above them — so the "allocation" reduces
/// to a handful of constant offsets derived from `code.varnames` /
/// `code.max_stackdepth`. `RegisterLayout::compute` runs the same
/// arithmetic the inline section of `transform_graph_to_jitcode` used
/// to do directly; its only purpose is to give the layout a name and
/// pull the calculation out of the 1400-line dispatch loop.
#[derive(Clone, Copy, Debug)]
struct RegisterLayout {
    /// `code.varnames.len()` — number of fast locals.
    nlocals: usize,
    /// pyre-only: number of cell + free vars (`pyframe::ncells`).
    ncells: usize,
    /// Absolute index where the operand stack begins in
    /// `PyFrame.locals_cells_stack_w` — `nlocals + ncells`.
    stack_base_absolute: usize,
    /// Compile-time depth bound from `code.max_stackdepth` (clamped to ≥ 1).
    max_stackdepth: usize,
    /// Ref register index where the operand stack begins
    /// (`stack_base = nlocals` since locals occupy the first registers).
    stack_base: u16,
    /// Scratch ref register #0 — sits at `nlocals + max_stackdepth`.
    obj_tmp0: u16,
    /// Scratch ref register #1.
    obj_tmp1: u16,
    /// First ref register reserved for portal red-arg shuffling.
    arg_regs_start: u16,
    /// Ref register holding the `NULL` sentinel value.
    null_ref_reg: u16,
    /// `interp_jit.py:64` portal red `frame` register.
    portal_frame_reg: u16,
    /// `interp_jit.py:64` portal red `ec` register.
    portal_ec_reg: u16,
    /// Scratch int register #0.
    int_tmp0: u16,
    /// Scratch int register #1.
    int_tmp1: u16,
    /// Int register holding the current opcode value during dispatch.
    op_code_reg: u16,
}

impl RegisterLayout {
    /// Pure arithmetic over `code` — no allocation, no side effects.
    /// Mirrors the constant block at the top of
    /// `transform_graph_to_jitcode`.
    fn compute(code: &CodeObject) -> Self {
        let nlocals = code.varnames.len();
        let ncells = pyre_interpreter::pyframe::ncells(code);
        let stack_base_absolute = nlocals + ncells;
        let max_stackdepth = code.max_stackdepth.max(1) as usize;
        let stack_base = nlocals as u16;
        let obj_tmp0 = (nlocals + max_stackdepth) as u16;
        let obj_tmp1 = (nlocals + max_stackdepth + 1) as u16;
        let arg_regs_start = (nlocals + max_stackdepth + 2) as u16;
        let null_ref_reg = (nlocals + max_stackdepth + 10) as u16;
        let portal_frame_reg = null_ref_reg + 1;
        let portal_ec_reg = null_ref_reg + 2;
        Self {
            nlocals,
            ncells,
            stack_base_absolute,
            max_stackdepth,
            stack_base,
            obj_tmp0,
            obj_tmp1,
            arg_regs_start,
            null_ref_reg,
            portal_frame_reg,
            portal_ec_reg,
            int_tmp0: 0,
            int_tmp1: 1,
            op_code_reg: 2,
        }
    }
}

/// Indices returned by `assembler.add_fn_ptr` for every blackhole
/// helper fn pointer the dispatch loop references. Mirrors the slot
/// shape of RPython's `_callinfo_for_oopspec`-derived index table —
/// the helpers are interned in a fixed order so the dispatch handlers
/// can capture the indices once and reuse them across emit sites.
///
/// PRE-EXISTING-ADAPTATION: the order matches the historical
/// inline sequence (`call_fn`, then the per-opcode helpers, then the
/// per-arity `call_fn_n`). Changing the order would shift every
/// `assembler.add_fn_ptr` index — RPython's `assembler.see_raw_object`
/// path has the same constraint.
#[derive(Clone, Copy, Debug)]
struct FnPtrIndices {
    call_fn: u16,
    load_global_fn: u16,
    compare_fn: u16,
    binary_op_fn: u16,
    box_int_fn: u16,
    truth_fn: u16,
    load_const_fn: u16,
    store_subscr_fn: u16,
    build_list_fn: u16,
    call_fn_0: u16,
    call_fn_2: u16,
    call_fn_3: u16,
    call_fn_4: u16,
    call_fn_5: u16,
    call_fn_6: u16,
    call_fn_7: u16,
    call_fn_8: u16,
}

/// Register every blackhole helper fn pointer with the assembler in
/// the canonical order. Returns the per-helper index table used by
/// the dispatch loop.
fn register_helper_fn_pointers(
    assembler: &mut SSAReprEmitter,
    cpu: &super::cpu::Cpu,
) -> FnPtrIndices {
    // RPython: CallControl manages fn addresses; assembler.finished()
    // writes them into callinfocollection. pyre adds them inline so
    // each handler can capture the index it needs.
    let call_fn = assembler.add_fn_ptr(cpu.call_fn as *const ());
    let load_global_fn = assembler.add_fn_ptr(cpu.load_global_fn as *const ());
    let compare_fn = assembler.add_fn_ptr(cpu.compare_fn as *const ());
    let binary_op_fn = assembler.add_fn_ptr(cpu.binary_op_fn as *const ());
    let box_int_fn = assembler.add_fn_ptr(cpu.box_int_fn as *const ());
    let truth_fn = assembler.add_fn_ptr(cpu.truth_fn as *const ());
    let load_const_fn = assembler.add_fn_ptr(cpu.load_const_fn as *const ());
    let store_subscr_fn = assembler.add_fn_ptr(cpu.store_subscr_fn as *const ());
    let build_list_fn = assembler.add_fn_ptr(cpu.build_list_fn as *const ());
    // Per-arity call helpers (appended AFTER existing fn_ptrs to preserve indices).
    let call_fn_0 = assembler.add_fn_ptr(cpu.call_fn_0 as *const ());
    let call_fn_2 = assembler.add_fn_ptr(cpu.call_fn_2 as *const ());
    let call_fn_3 = assembler.add_fn_ptr(cpu.call_fn_3 as *const ());
    let call_fn_4 = assembler.add_fn_ptr(cpu.call_fn_4 as *const ());
    let call_fn_5 = assembler.add_fn_ptr(cpu.call_fn_5 as *const ());
    let call_fn_6 = assembler.add_fn_ptr(cpu.call_fn_6 as *const ());
    let call_fn_7 = assembler.add_fn_ptr(cpu.call_fn_7 as *const ());
    let call_fn_8 = assembler.add_fn_ptr(cpu.call_fn_8 as *const ());
    FnPtrIndices {
        call_fn,
        load_global_fn,
        compare_fn,
        binary_op_fn,
        box_int_fn,
        truth_fn,
        load_const_fn,
        store_subscr_fn,
        build_list_fn,
        call_fn_0,
        call_fn_2,
        call_fn_3,
        call_fn_4,
        call_fn_5,
        call_fn_6,
        call_fn_7,
        call_fn_8,
    }
}

/// Patch the per-`-live-` slots in `assembler` with the
/// register-byte vectors derived from `liveness`.
///
/// RPython: `assembler.py:fill_live_args` is called inline by
/// `assembler.assemble`'s `Insn::Live` branch.
///
/// Note: the caller previously checked PRE-rename register indices
/// against the 256-byte cap in `liveness.py:139` and routed overflow
/// through a pyre-only `has_abort` fallback. After the post-pass
/// register allocator (`super::regalloc::allocate_registers`) lands,
/// the encoded indices are POST-rename and the cap fires only on
/// pathological functions whose `nlocals` alone already exceeds 256
/// — the same condition that crashes the RPython translator at
/// `encode_liveness`'s `assert 0 <= char < 256`. We adopt the
/// RPython-orthodox behavior (panic on overflow) by deleting the
/// fallback path.
fn fill_assembler_liveness(
    assembler: &mut SSAReprEmitter,
    liveness: &[LivenessInfo],
    live_patches: &[(usize, usize)],
) {
    for (entry, &(_, patch_offset)) in liveness.iter().zip(live_patches.iter()) {
        assembler.fill_live_args(
            patch_offset,
            &entry.live_i_regs,
            &entry.live_r_regs,
            &entry.live_f_regs,
        );
    }
}

/// RPython: `liveness.py:19-80` `compute_liveness(ssarepr)` —
/// backward dataflow over the populated `SSARepr` that produces the
/// per-PC `LivenessInfo` triple consumed by both
/// `get_list_of_active_boxes` (pyjitpl.py:177) and
/// `consume_one_section` (resume.py:1381).
///
/// Phase 4: the dataflow now runs on the walker's SSARepr via
/// `liveness::compute_liveness_preserve_positions`, which skips
/// `remove_repeated_live` so `live_patches` insn indices stay valid
/// offsets into `ssarepr.insns`.
///
/// Post-filters on the SSA output:
///   - Only Ref-kind registers: pyre Int/Float regs are scratch
///     tmps (int_tmp0/1, op_code_reg) or constant-as-reg encodings
///     inside `jit_merge_point`; they never carry a Python box
///     across py_pc boundaries. Leaving live_i / live_f empty keeps
///     the tracer / blackhole (which index box arrays by raw
///     register index at `trace_opcode.rs:229-263` /
///     `call_jit.rs:965-982`) from pulling a Ref box through an
///     Int/Float slot.
///   - Only indices inside the Python-frame range: locals
///     `0..nlocals` or in-depth stack slots
///     `stack_base..stack_base+depth`. Helper Ref regs above that
///     range (obj_tmp0/1, arg_regs_start, null_ref_reg,
///     portal_{frame,ec}_reg) are correctly dead across py_pcs and
///     would desynchronise box layout if leaked.
///
/// Intersecting with LiveVars narrows SSA's superset back to the
/// live set pyre's runtime contract currently assumes: symbolic
/// locals / blackhole resume are built on LiveVars' answer, which
/// omits function parameters at pc=0. Making SSA authoritative
/// requires teaching the runtime to handle SSA-extra live params —
/// see `phase4_ssa_liveness_blocker_2026_04_18.md`.
///
/// Unreachable PCs (the `LiveVars` dataflow leaves `usize::MAX`
/// sentinels from `liveness.rs:150`) get an empty entry so the
/// downstream `liveness.zip(live_patches)` alignment in dispatch
/// finalisation stays one-to-one.
fn compute_liveness_table(
    assembler: &mut SSAReprEmitter,
    live_patches: &[(usize, usize)],
    code: &CodeObject,
    num_instrs: usize,
    nlocals: usize,
    stack_base: u16,
    depth_at_pc: &[u16],
    pc_map: &[usize],
) -> Vec<LivenessInfo> {
    use super::flatten::{Insn, Kind as SsaKind, Operand as SsaOperand};
    super::liveness::compute_liveness_preserve_positions(&mut assembler.ssarepr);
    let live_vars = pyre_jit_trace::state::liveness_for(code as *const _);
    let mut liveness = Vec::with_capacity(num_instrs);
    for &(py_pc, insn_idx) in live_patches.iter() {
        let jit_pc = pc_map[py_pc];
        if !live_vars.is_reachable(py_pc) {
            liveness.push(LivenessInfo {
                pc: jit_pc as u16,
                live_i_regs: Vec::new(),
                live_r_regs: Vec::new(),
                live_f_regs: Vec::new(),
            });
            continue;
        }
        let depth = depth_at_pc[py_pc];
        let stack_limit = stack_base as usize + depth as usize;
        let mut seen: std::collections::BTreeSet<u16> = std::collections::BTreeSet::new();
        let mut live_r: Vec<u16> = Vec::new();
        if let Some(Insn::Live(args)) = assembler.ssarepr.insns.get(insn_idx) {
            for op in args {
                if let SsaOperand::Register(reg) = op {
                    if reg.kind != SsaKind::Ref {
                        continue;
                    }
                    let idx = reg.index as usize;
                    let in_locals = idx < nlocals;
                    let in_stack = idx >= stack_base as usize && idx < stack_limit;
                    if (in_locals || in_stack) && seen.insert(reg.index) {
                        live_r.push(reg.index);
                    }
                }
            }
        }
        for d in 0..depth {
            let idx = stack_base + d;
            if seen.insert(idx) {
                live_r.push(idx);
            }
        }
        let lv_live: std::collections::BTreeSet<u16> = {
            let mut s: std::collections::BTreeSet<u16> = (0..nlocals)
                .filter(|&idx| live_vars.is_local_live(py_pc, idx))
                .map(|idx| idx as u16)
                .collect();
            for d in 0..depth {
                s.insert(stack_base + d);
            }
            s
        };
        live_r.retain(|idx| lv_live.contains(idx));
        liveness.push(LivenessInfo {
            pc: jit_pc as u16,
            live_i_regs: Vec::new(),
            live_r_regs: live_r,
            live_f_regs: Vec::new(),
        });
    }
    liveness
}

/// Decode `code.exceptiontable` into the structures the dispatch loop
/// consumes:
/// - `catch_for_pc[py_pc]` — `Some(landing_label)` for every PC that
///   falls inside an exception range, mapping to the landing label
///   the dispatch loop will branch to on raise.
/// - `catch_sites` — one entry per active range, holding the handler
///   PC, the saved stack depth, and the `push_lasti` flag. The
///   dispatch loop emits a landing block per entry at the end.
/// - `handler_depth_at[handler_pc]` — the stack depth Python sets on
///   exception-handler entry (`entry.depth + 1` plus another `+1`
///   when `push_lasti`); used by the dispatch loop to fix
///   `current_depth` at the handler's first instruction.
///
/// PRE-EXISTING-ADAPTATION: RPython has no analog because RPython
/// flow graphs already carry exception-handling links; pyre's input
/// is raw CPython bytecode + the packed exception table, so this
/// preprocessing step is pyre-specific.
fn decode_exception_catch_sites(
    assembler: &mut SSAReprEmitter,
    code: &CodeObject,
    num_instrs: usize,
) -> (
    Vec<Option<u16>>,
    Vec<ExceptionCatchSite>,
    std::collections::HashMap<usize, u16>,
) {
    let exception_entries =
        pyre_interpreter::bytecode::decode_exception_table(&code.exceptiontable);
    let mut catch_for_pc: Vec<Option<u16>> = vec![None; num_instrs];
    let mut catch_sites: Vec<ExceptionCatchSite> = Vec::new();
    for py_pc in 0..num_instrs {
        let Some(entry) = exception_entries
            .iter()
            .find(|entry| py_pc >= entry.start as usize && py_pc < entry.end as usize)
        else {
            continue;
        };
        let handler_py_pc = entry.target as usize;
        if handler_py_pc >= num_instrs {
            continue;
        }
        let landing_label = assembler.new_label();
        catch_for_pc[py_pc] = Some(landing_label);
        catch_sites.push(ExceptionCatchSite {
            landing_label,
            handler_py_pc,
            stack_depth: entry.depth,
            push_lasti: entry.push_lasti,
            lasti_py_pc: py_pc,
        });
    }
    let handler_depth_at: std::collections::HashMap<usize, u16> = exception_entries
        .iter()
        .map(|e| {
            let extra = if e.push_lasti { 1u16 } else { 0 };
            (e.target as usize, e.depth as u16 + extra + 1)
        })
        .collect();
    (catch_for_pc, catch_sites, handler_depth_at)
}

// Note: the legacy `liveness_regs_to_u8_sorted` helper that returned
// `Option<Vec<u8>>` to flag the 256-register cap is gone. The cap is
// now enforced by `majit_translate::liveness::encode_liveness`'s
// `assert!(char_ < 256)` (RPython `liveness.py:147-166` parity), and
// the post-pass register allocator
// (`super::regalloc::allocate_registers`) compresses the indices so
// the cap fires only on pathological functions whose `nlocals` alone
// exceeds 256 — the same condition that crashes the RPython
// translator.

/// B6 Phase 3b dual emission entry point for residual calls.
///
/// Each per-PC handler that previously called one of `assembler.call_*_typed`
/// directly now funnels through this helper. Two steps:
///   1. `emit_residual_call_shape` pushes the upstream-canonical
///      `residual_call_{kinds}_{reskind}` Insn into the walker-local
///      `ssarepr` (the Phase 3c authoritative input).
///   2. The matching `assembler.{...}_typed` call is dispatched below to
///      preserve the runtime path (the `SSAReprEmitter::ssarepr` consumed
///      by `Assembler::assemble` today still carries the pyre-only
///      `call_*` opname). Once Phase 3c switches the assembler input,
///      step 2 can be removed.
fn emit_residual_call(
    ssarepr: &mut SSARepr,
    assembler: &mut SSAReprEmitter,
    flavor: CallFlavor,
    fn_idx: u16,
    call_args: &[majit_metainterp::jitcode::JitCallArg],
    reskind: ResKind,
    dst: Option<u16>,
) {
    emit_residual_call_shape(ssarepr, flavor, fn_idx, call_args, reskind, dst);
    match (flavor, reskind) {
        (CallFlavor::Plain, ResKind::Int) => {
            assembler.call_int_typed(fn_idx, call_args, dst.unwrap());
        }
        (CallFlavor::Plain, ResKind::Ref) => {
            assembler.call_ref_typed(fn_idx, call_args, dst.unwrap());
        }
        (CallFlavor::MayForce, ResKind::Ref) => {
            assembler.call_may_force_ref_typed(fn_idx, call_args, dst.unwrap());
        }
        (CallFlavor::MayForce, ResKind::Void) => {
            assembler.call_may_force_void_typed_args(fn_idx, call_args);
        }
        (flavor, reskind) => panic!(
            "emit_residual_call: unsupported runtime (flavor, reskind) = ({:?}, {:?})",
            flavor, reskind
        ),
    }
}

/// Upstream-shape Insn builder for the walker-local `ssarepr`. See
/// `emit_residual_call` for the dual-emit policy.
///
/// `rpython/jit/codewriter/jtransform.py:414-435 rewrite_call` emits
///
/// ```text
/// SpaceOperation('%s_%s_%s' % (namebase, kinds, reskind),
///                [fn, ListOfKind('int',   args_i),
///                     ListOfKind('ref',   args_r),   # only if 'r' in kinds
///                     ListOfKind('float', args_f),   # only if 'f' in kinds
///                     calldescr],
///                result)
/// ```
///
/// where `kinds` is the smallest cover of actually-present arg kinds
/// (`'r'` if only ref args, `'ir'` if int+ref, `'irf'` if floats are
/// present or the result itself is float) and `reskind` ∈ {`i`, `r`,
/// `f`, `v`}. `namebase = 'residual_call'` for the direct-call helper
/// (`jtransform.py:460-471 handle_residual_call`).
///
/// This helper is the external-`SSARepr` side of the Phase 3b dual
/// emission: every call site pairs a direct `assembler.call_*_typed(...)`
/// (pushing a pyre-only `call_*` Insn into the runtime-consumed
/// `SSAReprEmitter::ssarepr`) with one call here, which pushes the
/// upstream-canonical shape into the walker-local `ssarepr` that
/// Phase 3c will wire up as the authoritative `Assembler::assemble`
/// input. Upstream stores `calldescr` (an `AbstractDescr` carrying
/// `EffectInfo`) in the trailing slot; pyre threads a
/// `DescrOperand::CallFlavor` there so Phase 3c's assembler dispatch
/// can recover the (flavor, reskind) pair the runtime path resolves
/// statically today.
fn emit_residual_call_shape(
    ssarepr: &mut SSARepr,
    flavor: CallFlavor,
    fn_idx: u16,
    // Arguments in the C function's parameter order. `JitCallArg` carries
    // its own kind tag so pyre's runtime builder can interleave kinds on
    // the machine-code call. The SSARepr side projects this list into
    // kind-separated sublists to match upstream shape — upstream itself
    // loses per-C-parameter order in the SSARepr and relies on
    // `calldescr` at `bh_call_*` time to reconstruct it.
    call_args: &[majit_metainterp::jitcode::JitCallArg],
    reskind: ResKind,
    dst: Option<u16>,
) {
    use majit_metainterp::jitcode::JitArgKind;

    // `rpython/jit/codewriter/jtransform.py:437-445 make_three_lists`.
    let mut args_i: Vec<u16> = Vec::new();
    let mut args_r: Vec<u16> = Vec::new();
    let mut args_f: Vec<u16> = Vec::new();
    for arg in call_args {
        match arg.kind {
            JitArgKind::Int => args_i.push(arg.reg),
            JitArgKind::Ref => args_r.push(arg.reg),
            JitArgKind::Float => args_f.push(arg.reg),
        }
    }

    // `rewrite_call` kinds selection (jtransform.py:423-426).
    let kinds: &str = if !args_f.is_empty() || reskind == ResKind::Float {
        "irf"
    } else if !args_i.is_empty() {
        "ir"
    } else {
        "r"
    };
    let reskind_ch = reskind.as_char();
    let opname = format!("residual_call_{kinds}_{reskind_ch}");

    // SSARepr arg list: [Const(fn), ListI?, ListR, ListF?, Descr(flavor)].
    let mut args: Vec<Operand> = Vec::with_capacity(5);
    args.push(Operand::ConstInt(fn_idx as i64));
    let reg_list = |kind: Kind, regs: &[u16]| {
        Operand::ListOfKind(ListOfKind::new(
            kind,
            regs.iter().map(|&r| Operand::reg(kind, r)).collect(),
        ))
    };
    if kinds.contains('i') {
        args.push(reg_list(Kind::Int, &args_i));
    }
    if kinds.contains('r') {
        args.push(reg_list(Kind::Ref, &args_r));
    }
    if kinds.contains('f') {
        args.push(reg_list(Kind::Float, &args_f));
    }
    args.push(Operand::descr(DescrOperand::CallFlavor(flavor)));

    let insn = match (reskind.to_kind(), dst) {
        (Some(kind), Some(d)) => Insn::op_with_result(opname, args, Register::new(kind, d)),
        (None, None) => Insn::op(opname, args),
        (Some(_), None) => panic!("residual_call with non-void reskind requires dst"),
        (None, Some(_)) => panic!("residual_call with void reskind must not have dst"),
    };
    ssarepr.insns.push(insn);
}

// ---------------------------------------------------------------------------
// RPython: codewriter/codewriter.py — class CodeWriter
// ---------------------------------------------------------------------------

/// Compiles Python CodeObjects into JitCode for blackhole execution.
///
/// RPython: `rpython/jit/codewriter/codewriter.py::CodeWriter`.
/// `codewriter.py:20-23` stores `self.assembler = Assembler()` and
/// `self.callcontrol = CallControl(cpu, jitdrivers_sd)` once on the
/// CodeWriter and reuses them across every `transform_graph_to_jitcode`
/// call so `all_liveness` / `num_liveness_ops` and the `jitcodes` dict
/// accumulate over the whole translator session.
///
/// pyre mirrors that ownership via a per-thread singleton: the process
/// holds a single `CodeWriter` instance (one per thread) reachable via
/// [`CodeWriter::instance`], matching `warmspot.py:245`
/// `codewriter = CodeWriter(cpu, [jd])`. The owned `Assembler` lives on
/// a `RefCell<Assembler>` field so `transform_graph_to_jitcode` can
/// still mutate it under the immutable-by-default singleton borrow.
pub struct CodeWriter {
    /// `codewriter.py:22` `self.assembler = Assembler()`.
    ///
    /// Single Assembler instance shared across every `transform_graph_to_jitcode`
    /// call on this CodeWriter. `all_liveness` / `all_liveness_positions` /
    /// `num_liveness_ops` accumulate here just like the upstream object.
    assembler: RefCell<Assembler>,
    /// RPython: `self.callcontrol = CallControl(cpu, jitdrivers_sd)`
    /// (codewriter.py:23). Owned in a `UnsafeCell` so `&CodeWriter` can
    /// mint `&mut CallControl` through [`Self::callcontrol`] — matches
    /// the legacy `JITCODE_CACHE` interior-mutability contract.
    callcontrol: UnsafeCell<super::call::CallControl>,
}

impl CodeWriter {
    /// RPython: `CodeWriter.__init__(cpu, jitdrivers_sd)` (codewriter.py:20-23).
    ///
    /// Phase A: the cpu helpers are fixed module-level functions in
    /// `crate::call_jit`; Phase D.2 wired `callcontrol` as a field so
    /// `writer.callcontrol()` matches `self.callcontrol` in upstream.
    pub fn new() -> Self {
        // codewriter.py:21-23 `self.cpu = cpu; self.assembler = Assembler();
        //   self.callcontrol = CallControl(cpu, jitdrivers_sd)`.
        // pyre owns the single `Cpu` on `CallControl`; `CodeWriter::cpu()`
        // returns a borrow back out so the upstream attribute access
        // pattern (`self.cpu`) still works.
        let cpu = super::cpu::Cpu::new();
        // Register the trace-side `jitcode_for` compile callback the
        // first time any `CodeWriter` is constructed in this process.
        // This is the lazy analog of jtransform's eager call to
        // `cc.callcontrol.get_jitcode(callee_graph)` (call.py:155):
        // when the tracer references a callee's `JitCode`, the same
        // `make_jitcodes` pipeline (codewriter.py:74-89) runs for that
        // one entry so `CallControl.find_jitcode` and
        // `MetaInterpStaticData.jitcodes` agree before the blackhole
        // resume looks anything up (resume.py:1338).
        static INIT_COMPILE_CALLBACK: std::sync::Once = std::sync::Once::new();
        INIT_COMPILE_CALLBACK.call_once(|| {
            pyre_jit_trace::set_compile_jitcode_fn(compile_jitcode_via_w_code);
        });
        Self {
            assembler: RefCell::new(Assembler::new()),
            callcontrol: UnsafeCell::new(super::call::CallControl::new(cpu, Vec::new())),
        }
    }

    /// `codewriter.py:21` `self.cpu = cpu`.
    ///
    /// Convenience accessor — pyre owns the single `Cpu` on
    /// `CallControl` (call.py:27 `self.cpu = cpu`); upstream both
    /// attributes point at the same object.
    pub fn cpu(&self) -> &super::cpu::Cpu {
        &self.callcontrol().cpu
    }

    /// RPython: `CodeWriter.setup_vrefinfo(self, vrefinfo)`
    /// (codewriter.py:91-94).
    ///
    /// ```python
    /// def setup_vrefinfo(self, vrefinfo):
    ///     # must be called at most once
    ///     assert self.callcontrol.virtualref_info is None
    ///     self.callcontrol.virtualref_info = vrefinfo
    /// ```
    ///
    /// PRE-EXISTING-ADAPTATION: pyre has no `virtualref` machinery
    /// (no `@jit.virtual_ref`, no `vref_info` lookup); the slot is
    /// preserved so future warmspot wiring can call through with the
    /// same name.
    pub fn setup_vrefinfo(&self, vrefinfo: ()) {
        // codewriter.py:93 `assert self.callcontrol.virtualref_info is None`.
        assert!(self.callcontrol().virtualref_info.is_none());
        // codewriter.py:94 `self.callcontrol.virtualref_info = vrefinfo`.
        self.callcontrol().virtualref_info = Some(vrefinfo);
    }

    /// RPython: `CodeWriter.setup_jitdriver(self, jitdriver_sd)`
    /// (codewriter.py:96-99).
    ///
    /// ```python
    /// def setup_jitdriver(self, jitdriver_sd):
    ///     # Must be called once per jitdriver.
    ///     self.callcontrol.jitdrivers_sd.append(jitdriver_sd)
    /// ```
    ///
    /// PRE-EXISTING-ADAPTATION: RPython appends unconditionally because
    /// each `@jit_callback` decoration calls `setup_jitdriver` exactly
    /// once at translation time. pyre's portal discovery is lazy and
    /// fires on every JIT entry, so the same `portal_graph` would be
    /// pushed repeatedly without the `find` guard below — `jitdrivers_sd`
    /// would grow linearly with JIT entries instead of staying bounded
    /// by the number of unique portals. The dedup updates the existing
    /// jd's `merge_point_pc` so the refinement hint propagates into
    /// the next `grab_initial_jitcodes` pass.
    pub fn setup_jitdriver(&self, jitdriver_sd: super::call::JitDriverStaticData) {
        let cc = self.callcontrol();
        if let Some(existing) = cc
            .jitdrivers_sd
            .iter_mut()
            .find(|j| j.portal_graph == jitdriver_sd.portal_graph)
        {
            if jitdriver_sd.merge_point_pc.is_some() {
                existing.merge_point_pc = jitdriver_sd.merge_point_pc;
            }
            return;
        }
        // codewriter.py:99 `self.callcontrol.jitdrivers_sd.append(jitdriver_sd)`.
        cc.jitdrivers_sd.push(jitdriver_sd);
    }

    /// RPython: `self.callcontrol` (codewriter.py:23).
    ///
    /// Returns a mutable reference to the owned `CallControl`. Safe under
    /// the same invariant as the legacy `JITCODE_CACHE` thread_local: the
    /// caller must not re-enter `callcontrol()` while the returned borrow
    /// is live.
    #[allow(clippy::mut_from_ref)]
    pub fn callcontrol(&self) -> &mut super::call::CallControl {
        // SAFETY: `CodeWriter` is only accessed via `instance()` which
        // returns a thread-local reference; all callers execute on the
        // owning thread.
        unsafe { &mut *self.callcontrol.get() }
    }

    /// Access the process-wide single `CodeWriter` — analog of the
    /// single `codewriter` owned by `warmspot.py:245-281` for the
    /// lifetime of the JIT.
    ///
    /// Implemented as a per-thread singleton: pyre's JIT currently runs
    /// one interpreter per thread and function pointers in `Self` are
    /// `Sync`, so a thread-local provides the RPython "one CodeWriter
    /// per warmspot" invariant without a global lock.
    pub fn instance() -> &'static CodeWriter {
        thread_local! {
            static INSTANCE: CodeWriter = CodeWriter::new();
        }
        INSTANCE.with(|cw| unsafe { &*(cw as *const CodeWriter) })
    }

    /// Transform a Python CodeObject into a JitCode.
    ///
    /// RPython: CodeWriter.transform_graph_to_jitcode(graph, jitcode, verbose, index)
    ///
    /// Python bytecodes serve as the "graph". Since they are already linear
    /// and register-allocated, jtransform/regalloc/flatten are identity
    /// transforms. We go directly to assembly.
    pub fn transform_graph_to_jitcode(
        &self,
        code: &CodeObject,
        w_code: *const (),
        merge_point_pc: Option<usize>,
    ) -> PyJitCode {
        // RPython codewriter.py:46-48 `regallocs[kind] = perform_register_allocation(graph, kind)`.
        // pyre's regalloc is trivial — fast locals occupy the bottom of
        // the ref register file and the operand stack stacks above
        // them — so the "allocation" reduces to a `RegisterLayout`
        // computed from `code.varnames` / `code.max_stackdepth`.
        let layout = RegisterLayout::compute(code);
        let RegisterLayout {
            nlocals,
            ncells: _,
            stack_base_absolute,
            max_stackdepth,
            stack_base,
            obj_tmp0,
            obj_tmp1,
            arg_regs_start,
            null_ref_reg,
            portal_frame_reg,
            portal_ec_reg,
            int_tmp0,
            int_tmp1,
            op_code_reg,
        } = layout;
        // jtransform.py: virtualizable field indices for getfield_vable_*
        // interp_jit.py:25-31 / virtualizable_spec.rs parity:
        //   0=last_instr, 1=code, 2=vsd, 3=debugdata, 4=lastblock, 5=namespace
        const VABLE_CODE_FIELD_IDX: u16 = 1;
        const VABLE_VALUESTACKDEPTH_FIELD_IDX: u16 = 2;
        const VABLE_NAMESPACE_FIELD_IDX: u16 = 5;

        // regalloc.py: compile-time stack depth counter — tracks which
        // stack register (stack_base + depth) is the current TOS.
        let mut current_depth: u16 = 0;

        // RPython: self.assembler = Assembler() + JitCode(graph.name, ...)
        // (rpython/jit/codewriter/jitcode.py:14-15 takes name as the first
        // __init__ argument; majit's JitCodeBuilder::set_name mirrors that).
        let mut assembler = SSAReprEmitter::new();
        assembler.set_name(code.obj_name.to_string());
        // B6 Phase 3b scaffolding: grow an `SSARepr` alongside the direct
        // `JitCodeBuilder` calls. Currently only a handful of handlers
        // (`ref_return` below) dual-emit an `Insn::Op`; the remaining
        // bytecode handlers still route through the builder only. When
        // every handler has been converted, `ssarepr` becomes the
        // authoritative input to `jit::assembler::Assembler::assemble`
        // (Phase 3c switchover) and the direct builder calls disappear.
        // See `pyre/pyre-jit/src/jit/B6_CODEWRITER_PIPELINE_PLAN.md`.
        let mut ssarepr = SSARepr::new(code.obj_name.to_string());

        // RPython regalloc.py: keep kind-separated register files.
        assembler.ensure_r_regs(portal_ec_reg + 1);
        assembler.ensure_i_regs(op_code_reg + 1);

        // Register helper fn pointers in the canonical order; the
        // returned struct names every index so the dispatch handlers
        // below can reference them by field instead of an opaque local.
        let FnPtrIndices {
            call_fn: call_fn_idx,
            load_global_fn: load_global_fn_idx,
            compare_fn: compare_fn_idx,
            binary_op_fn: binary_op_fn_idx,
            box_int_fn: box_int_fn_idx,
            truth_fn: truth_fn_idx,
            load_const_fn: load_const_fn_idx,
            store_subscr_fn: store_subscr_fn_idx,
            build_list_fn: build_list_fn_idx,
            call_fn_0: call_fn_0_idx,
            call_fn_2: call_fn_2_idx,
            call_fn_3: call_fn_3_idx,
            call_fn_4: call_fn_4_idx,
            call_fn_5: call_fn_5_idx,
            call_fn_6: call_fn_6_idx,
            call_fn_7: call_fn_7_idx,
            call_fn_8: call_fn_8_idx,
        } = register_helper_fn_pointers(&mut assembler, self.cpu());

        // RPython flatten.py: pre-create labels for each block.
        // Python bytecodes are linear, so each instruction index gets a label.
        let num_instrs = code.instructions.len();
        let mut labels: Vec<u16> = Vec::with_capacity(num_instrs);
        for _ in 0..num_instrs {
            labels.push(assembler.new_label());
        }

        let (catch_for_pc, catch_sites, handler_depth_at) =
            decode_exception_catch_sites(&mut assembler, code, num_instrs);

        // interp_jit.py:118 `pypyjitdriver.can_enter_jit(...)` is called in
        // `jump_absolute` (`jumpto < next_instr` branch), i.e. at each
        // Python backward jump.  jtransform.py:1714-1723
        // `handle_jit_marker__can_enter_jit = handle_jit_marker__loop_header`
        // lowers each one to a `loop_header` jitcode op.  Pyre has no
        // `jump_absolute` Python wrapper — the equivalent is to pre-scan
        // `JumpBackward` opcodes and record their targets; each target PC
        // becomes a `loop_header` site.
        let loop_header_pcs = find_loop_header_pcs(code);

        // pc_map: Python PC → JitCode byte offset
        let mut pc_map = vec![0usize; num_instrs];

        // RPython: flatten_graph() walks blocks and emits instruction tuples.
        // RPython: assembler.assemble(ssarepr, jitcode, num_regs) emits bytecodes.
        // For pyre, we combine both steps: walk Python bytecodes and emit
        // JitCode bytecodes directly.
        let mut arg_state = OpArgState::default();
        // liveness.py parity: record stack depth at each Python PC for
        // precise liveness generation. Stack registers stack_base..stack_base+depth
        // are live at each PC.
        let mut depth_at_pc: Vec<u16> = vec![0; num_instrs];
        let mut live_patches: Vec<(usize, usize)> = Vec::with_capacity(num_instrs);

        // codewriter.py:37 `portal_jd = self.callcontrol.jitdriver_sd_from_portal_graph(graph)`
        // — RPython looks up portal-ness in the registry that
        // `setup_jitdriver` populates. pyre matches that: a code is a
        // portal iff it is in `CallControl.jitdrivers_sd`. The portal
        // path (`register_portal_jitdriver`) registers before the drain
        // runs `transform_graph_to_jitcode`, so the lookup sees the
        // entry; the callee path (`compile_jitcode_for_callee`) never
        // touches `jitdrivers_sd`, so the lookup returns `None`. This
        // replaces the older "any backedge → portal" heuristic.
        let is_portal = self
            .callcontrol()
            .jitdriver_sd_from_portal_graph(code as *const CodeObject)
            .is_some();
        // RPython parity: every backward jump goes through dispatch() →
        // jit_merge_point(). The blackhole's bhimpl_jit_merge_point raises
        // ContinueRunningNormally at the bottommost level. Ideally all
        // loop headers should emit BC_JIT_MERGE_POINT, but the
        // CRN→interpreter→JIT-reentry cycle crashes in JIT compiled code
        // because blackhole-modified frame locals can contain values
        // incompatible with the compiled trace's assumptions. Until the
        // full RPython CRN→portal_ptr restart is implemented, only the
        // first loop header is a merge point.
        // RPython jtransform.py:1690: jit_merge_point only in the portal
        // graph. merge_point_pc is the trace entry PC (from bound_reached).
        // Other loop headers use loop_header (no-op in the blackhole).
        let merge_point_pc = if is_portal {
            merge_point_pc.or_else(|| loop_header_pcs.iter().copied().min())
        } else {
            // Callee — no jit_merge_point emit. RPython's jtransform.py:1690
            // `jit_merge_point only in the portal graph` is the matching
            // statement.
            None
        };

        // pyframe.py:379-417 pushvalue/popvalue_maybe_none parity:
        // Each push/pop writes self.valuestackdepth = depth ± 1.
        // jtransform.py:923-928 lowers this to setfield_vable_i.
        // This macro emits the equivalent BC_SETFIELD_VABLE_I after
        // every current_depth mutation so the frame's valuestackdepth
        // stays in sync at every guard/call point — matching RPython's
        // per-push/per-pop semantics.
        macro_rules! emit_vsd {
            ($asm:expr, $depth:expr) => {
                if is_portal {
                    emit_load_const_i!(
                        ssarepr,
                        $asm,
                        int_tmp0,
                        (stack_base_absolute + $depth as usize) as i64
                    );
                    emit_vable_setfield_int!(
                        ssarepr,
                        $asm,
                        VABLE_VALUESTACKDEPTH_FIELD_IDX,
                        int_tmp0
                    );
                }
            };
        }

        // PRE-EXISTING-ADAPTATION: the `BC_ABORT_PERMANENT` runtime
        // bytecode does not appear in `rpython/jit/codewriter/` or
        // `rpython/jit/metainterp/`. RPython refuses to build jitcode for
        // bytecodes it cannot translate (the translator surfaces the
        // failure at build time); pyre must always produce runnable
        // jitcode because bytecode translation is lazy at runtime. We
        // therefore keep the runtime-side adaptation (assembler emits
        // `BC_ABORT_PERMANENT` so the blackhole interpreter falls back to
        // CPython evaluation) but never surface the pyre-only opname into
        // the RPython-parity SSARepr layer — `flatten.py:106` uses plain
        // `Label` for loop headers and `assembler.py:159` does not encode
        // unsupported bytecodes as named opnames.

        // B6 Phase 3b dual emission for `last_exc_value`. RPython parity:
        // `flatten.py:347` `self.emitline("last_exc_value", "->",
        // self.getcolor(w))` — `assembler.py:220` turns it into
        // `last_exc_value/>r`. pyre emits this once per catch site to
        // load the thread-local exception into the handler's input
        // register.
        macro_rules! emit_last_exc_value {
            ($ssarepr:expr, $asm:expr, $dst:expr) => {{
                let dst = $dst;
                $ssarepr.insns.push(Insn::op_with_result(
                    "last_exc_value",
                    Vec::new(),
                    Register::new(Kind::Ref, dst),
                ));
                $asm.last_exc_value(dst);
            }};
        }

        // PRE-EXISTING-ADAPTATION: the `BC_JUMP_TARGET` runtime opcode
        // does not appear in `rpython/jit/codewriter/`. RPython marks
        // loop-header block entries with a plain `Insn::Label` and lets
        // the blackhole's dispatch loop recognise them via the label
        // position; pyre emits a dedicated `BC_JUMP_TARGET` opcode so the
        // runtime inner-loop can cheaply identify back-edge targets
        // without consulting a label table. The runtime-side adaptation
        // stays (assembler dispatch at `assembler.rs:367-372`) but the
        // pyre-only opname is not surfaced into the RPython-parity
        // SSARepr layer — `flatten.py:106` uses plain `Label` for loop
        // headers.

        // B6 Phase 3b dual emission for `int_copy` / `ref_copy` /
        // `float_copy` with a Constant source. RPython parity:
        // `flatten.py:333` `self.emitline('%s_copy' % kind, v, "->", w)`
        // — `v` is resolved via `getcolor(v)` which returns either a
        // `Register` or an unchanged `Constant` (see `flatten.py:382-384`).
        // The `assembler.py:140-222` dispatch handles both: the Register
        // source emits an `int_copy/ii` entry, and the Constant source
        // emits an `int_copy/ci` entry (argcode `'c'` for a compact
        // Constant). pyre's legacy `load_const_{i,r,f}_value` emits the
        // same runtime bytes under pyre-only `load_const_*` opnames; the
        // SSARepr now carries the RPython-parity `*_copy` name with a
        // ConstInt/ConstRef/ConstFloat source operand.
        // `int_eq` is NOT dual-emitted by itself.
        //
        // pyre's only emitter of `OpCode::IntEq` via `record_binop_i` is
        // the `PopJumpIfTrue` branch-folding chain below — which
        // `rpython/jit/codewriter/jtransform.py:1212` rewrites as
        // `int_is_zero(x)` and `flatten.py:247` further specialises into
        // `goto_if_not_int_is_zero`. Emitting a standalone `int_eq` at
        // the PopJumpIfTrue site would replicate pyre's builder-ABI
        // sequence into the SSARepr and lock in a pyre-only shape.
        // Keep the handler on the direct builder path until the
        // assembler learns the RPython specialisation.

        // Call family intentionally has NO dual-emit.
        //
        // `rpython/jit/codewriter/jtransform.py:414-435` `rewrite_call()`
        // emits `residual_call_{kinds}_{reskind}` with
        // `[fnptr_constant, ListOfKind(int)?, ListOfKind(ref),
        //   ListOfKind(float)?, calldescr]`. pyre's runtime ABI uses
        // a caller-order Register list plus a u16 helper-table index and
        // encodes `may_force` in the target bytecode — none of which
        // fit into RPython's SSA tuple shape. Reviewer guidance:
        //   "codewriter는 원본 RPython tuple 을 만들고, pyre 적응은
        //    assembler 가 해야 한다."
        // Rather than baking the pyre shape into the SSA (which would
        // ossify a pyre-only SSA vocabulary), we keep the call handlers
        // on the direct builder path until assembler.rs grows exact
        // `residual_call_*` dispatch that can reconstruct the pyre
        // caller-order list. See `B6_CODEWRITER_PIPELINE_PLAN.md`.

        macro_rules! emit_load_const_i {
            ($ssarepr:expr, $asm:expr, $dst:expr, $value:expr $(,)?) => {{
                let dst = $dst;
                let value: i64 = $value;
                $ssarepr.insns.push(Insn::op_with_result(
                    "int_copy",
                    vec![Operand::ConstInt(value)],
                    Register::new(Kind::Int, dst),
                ));
                $asm.load_const_i_value(dst, value);
            }};
        }

        // B6 Phase 3b dual emission for `ref_return`. Every site that used
        // to invoke `assembler.ref_return(src)` now also appends an
        // `Insn::Op { opname: "ref_return", args: [Register(Ref, src)] }`
        // to the SSARepr so the future `Assembler::assemble` path
        // (`assembler.rs::dispatch_op:374`) can reproduce the same byte at
        // the Phase 3c switchover. The direct builder call stays until the
        // switchover runs so the emitted JitCode remains bit-identical.
        // RPython parity: `rpython/jit/codewriter/jtransform.py` emits
        // `op_ref_return(v)` via `rewrite_op_jit_return` for the portal
        // return path; `assembler.py:221` turns that into the `ref_return/r`
        // bytecode key.
        macro_rules! emit_ref_return {
            ($ssarepr:expr, $asm:expr, $src:expr) => {{
                let src = $src;
                $ssarepr
                    .insns
                    .push(Insn::op("ref_return", vec![Operand::reg(Kind::Ref, src)]));
                $asm.ref_return(src);
            }};
        }

        // B6 Phase 3b dual emission for `goto`. RPython parity:
        // `flatten.py:161` `self.emitline('goto', TLabel(link.target))` —
        // `assembler.py:220` turns the op into `goto/L`. Pyre labels are
        // integer indices into `labels[]`, one per Python PC; the
        // `TLabel` carries the synthetic name `pc{target_py_pc}` so the
        // Phase 3c dispatch (`assembler.rs::dispatch_op:345`) can resolve
        // it against `builder_label`.
        macro_rules! emit_goto {
            ($ssarepr:expr, $asm:expr, $labels:expr, $target_py_pc:expr) => {{
                let target_py_pc = $target_py_pc;
                $ssarepr.insns.push(Insn::op(
                    "goto",
                    vec![Operand::TLabel(TLabel::new(format!("pc{}", target_py_pc)))],
                ));
                $asm.jump($labels[target_py_pc]);
            }};
        }

        // B6 Phase 3b dual emission for `raise`. RPython parity:
        // `flatten.py` emits `self.emitline("raise", self.getcolor(args[1]))`
        // inside the exception-link handler; `assembler.py:220` turns it
        // into `raise/r`. pyre's single `emit_raise(exc_reg)` call site
        // (RAISE_VARARGS with argc >= 1) corresponds to the same edge.
        macro_rules! emit_raise {
            ($ssarepr:expr, $asm:expr, $src:expr) => {{
                let src = $src;
                $ssarepr
                    .insns
                    .push(Insn::op("raise", vec![Operand::reg(Kind::Ref, src)]));
                $asm.emit_raise(src);
            }};
        }

        // B6 Phase 3b dual emission for `reraise`. RPython parity:
        // `flatten.py` emits the zero-arg `self.emitline("reraise")` for
        // the re-raise edge; `assembler.py:220` turns it into
        // `reraise/`. pyre emits this for RAISE_VARARGS with argc == 0.
        macro_rules! emit_reraise {
            ($ssarepr:expr, $asm:expr) => {{
                $ssarepr.insns.push(Insn::op("reraise", Vec::new()));
                $asm.emit_reraise();
            }};
        }

        // B6 Phase 3b dual emission for `catch_exception`. RPython parity:
        // `flatten.py` emits `self.emitline('catch_exception',
        // TLabel(block.exits[0]))` when a block has an exception edge;
        // `assembler.py:220` turns it into `catch_exception/L`. pyre
        // emits this after each Python PC that has an exception handler.
        // The catch landing block lives after the main loop
        // (`mark_label(site.landing_label)`), so the `TLabel` carries
        // `catch_landing_{landing_label}` — distinct from the
        // `pc{py_pc}` naming used for PC-indexed labels.
        macro_rules! emit_catch_exception {
            ($ssarepr:expr, $asm:expr, $catch_label:expr) => {{
                let catch_label = $catch_label;
                $ssarepr.insns.push(Insn::op(
                    "catch_exception",
                    vec![Operand::TLabel(TLabel::new(format!(
                        "catch_landing_{}",
                        catch_label
                    )))],
                ));
                $asm.catch_exception(catch_label);
            }};
        }

        // B6 Phase 3b dual emission for block `Label`. RPython parity:
        // `flatten.py:180` `self.emitline(Label(block))` marks block
        // entry; `assembler.py:157-158` records the label position in
        // `self.label_positions`. pyre marks a label at every Python PC
        // (`mark_label(labels[py_pc])`) and at each catch landing
        // block's entry. The two naming schemes (`pc{py_pc}` vs
        // `catch_landing_{u16}`) match the TLabel schemes used by
        // `emit_goto!` and `emit_catch_exception!`.
        macro_rules! emit_mark_label_pc {
            ($ssarepr:expr, $asm:expr, $labels:expr, $py_pc:expr) => {{
                let py_pc = $py_pc;
                $ssarepr
                    .insns
                    .push(Insn::Label(super::flatten::Label::new(format!(
                        "pc{}",
                        py_pc
                    ))));
                $asm.mark_label($labels[py_pc]);
            }};
        }
        macro_rules! emit_mark_label_catch_landing {
            ($ssarepr:expr, $asm:expr, $landing_label:expr) => {{
                let landing_label = $landing_label;
                $ssarepr
                    .insns
                    .push(Insn::Label(super::flatten::Label::new(format!(
                        "catch_landing_{}",
                        landing_label
                    ))));
                $asm.mark_label(landing_label);
            }};
        }

        // B6 Phase 3b dual emission for `-live-`. RPython parity:
        // `flatten.py` inserts `self.emitline('-live-')` at every block
        // entry and at every point with a live resume set; `assembler.py:146-158`
        // allocates an `all_liveness` slot and encodes the live-register
        // triple. pyre patches the live-offset after the assembler
        // computes liveness for the Python PC, so the SSARepr Insn::Live
        // is empty until Phase 4's compute_liveness replaces it.
        macro_rules! emit_live_placeholder {
            ($ssarepr:expr, $asm:expr) => {{
                $ssarepr.insns.push(Insn::Live(Vec::new()));
                $asm.live_placeholder()
            }};
        }

        // flatten.py:240-260 boolean exitswitch emission. When the bool is a
        // plain variable (truth_fn result), flatten emits `goto_if_not <v> L`
        // (alias of bhimpl_goto_if_not_int_is_true per blackhole.py:913).
        // PopJumpIfTrue inverts the polarity via jtransform.py:1212
        // `_rewrite_equality` + flatten.py:247 specialisation
        // `goto_if_not_int_is_zero <v> L` (blackhole.py:916-920).
        macro_rules! emit_goto_if_not {
            ($ssarepr:expr, $asm:expr, $labels:expr, $cond:expr, $py_pc:expr) => {{
                let cond = $cond;
                let py_pc = $py_pc;
                $ssarepr.insns.push(Insn::op(
                    "goto_if_not",
                    vec![
                        Operand::reg(Kind::Int, cond),
                        Operand::TLabel(TLabel::new(format!("pc{}", py_pc))),
                    ],
                ));
                $asm.branch_reg_zero(cond, $labels[py_pc]);
            }};
        }
        macro_rules! emit_goto_if_not_int_is_zero {
            ($ssarepr:expr, $asm:expr, $labels:expr, $cond:expr, $py_pc:expr) => {{
                let cond = $cond;
                let py_pc = $py_pc;
                $ssarepr.insns.push(Insn::op(
                    "goto_if_not_int_is_zero",
                    vec![
                        Operand::reg(Kind::Int, cond),
                        Operand::TLabel(TLabel::new(format!("pc{}", py_pc))),
                    ],
                ));
                $asm.goto_if_not_int_is_zero(cond, $labels[py_pc]);
            }};
        }

        // B6 Phase 3b dual emission for the `*_vable_*` field/array
        // accessors. RPython parity: `jtransform.py:844`
        // `SpaceOperation('getfield_vable_%s' % kind, ...)` and
        // `jtransform.py:923` `SpaceOperation('setfield_vable_%s' % kind,
        // ...)` use the FULL kind name (`int` / `ref` / `float`);
        // `jtransform.py` `SpaceOperation('getarrayitem_vable_%s' %
        // kind[0], ...)` / `SpaceOperation('setarrayitem_vable_%s' %
        // kind[0], ...)` use the SHORT form (`i` / `r` / `f`).
        //
        // The runtime dispatch arms in `assembler.rs:486-524` were
        // already written against short-form field names; the SSA
        // emission here uses the RPython-parity full-form names and
        // relies on the Phase 3c dispatch aliases (added alongside)
        // to route the full-form opnames to the same builder methods.
        macro_rules! emit_vable_getfield_ref {
            ($ssarepr:expr, $asm:expr, $dst:expr, $field_idx:expr) => {{
                let dst = $dst;
                let field_idx = $field_idx;
                $ssarepr.insns.push(Insn::op_with_result(
                    "getfield_vable_ref",
                    vec![Operand::ConstInt(field_idx as i64)],
                    Register::new(Kind::Ref, dst),
                ));
                $asm.vable_getfield_ref(dst, field_idx);
            }};
        }
        macro_rules! emit_vable_setfield_int {
            ($ssarepr:expr, $asm:expr, $field_idx:expr, $src:expr) => {{
                let field_idx = $field_idx;
                let src = $src;
                $ssarepr.insns.push(Insn::op(
                    "setfield_vable_int",
                    vec![
                        Operand::ConstInt(field_idx as i64),
                        Operand::reg(Kind::Int, src),
                    ],
                ));
                $asm.vable_setfield_int(field_idx, src);
            }};
        }
        macro_rules! emit_vable_getarrayitem_ref {
            ($ssarepr:expr, $asm:expr, $dst:expr, $field_idx:expr, $index:expr) => {{
                let dst = $dst;
                let field_idx = $field_idx;
                let index = $index;
                $ssarepr.insns.push(Insn::op_with_result(
                    "getarrayitem_vable_r",
                    vec![
                        Operand::ConstInt(field_idx as i64),
                        Operand::reg(Kind::Int, index),
                    ],
                    Register::new(Kind::Ref, dst),
                ));
                $asm.vable_getarrayitem_ref(dst, field_idx, index);
            }};
        }
        macro_rules! emit_vable_setarrayitem_ref {
            ($ssarepr:expr, $asm:expr, $field_idx:expr, $index:expr, $src:expr) => {{
                let field_idx = $field_idx;
                let index = $index;
                let src = $src;
                $ssarepr.insns.push(Insn::op(
                    "setarrayitem_vable_r",
                    vec![
                        Operand::ConstInt(field_idx as i64),
                        Operand::reg(Kind::Int, index),
                        Operand::reg(Kind::Ref, src),
                    ],
                ));
                $asm.vable_setarrayitem_ref(field_idx, index, src);
            }};
        }

        // B6 Phase 3b dual emission for `ref_copy`. RPython parity:
        // `flatten.py:334` `self.emitline('%s_copy' % kind, v, "->", w)`
        // emits the register-to-register move as `ref_copy` when
        // `kind == 'ref'`; `assembler.py:220` turns it into `ref_copy/r>r`.
        // pyre's `move_r(dst, src)` is the single ref-move primitive.
        // The SSARepr arg list follows the RPython `(src, '->', dst)`
        // shape via `op_with_result`.
        macro_rules! emit_move_r {
            ($ssarepr:expr, $asm:expr, $dst:expr, $src:expr) => {{
                let dst = $dst;
                let src = $src;
                $ssarepr.insns.push(Insn::op_with_result(
                    "ref_copy",
                    vec![Operand::reg(Kind::Ref, src)],
                    Register::new(Kind::Ref, dst),
                ));
                $asm.move_r(dst, src);
            }};
        }

        // B6 Phase 3b dual emission for `jit_merge_point`. RPython parity:
        // `jtransform.py:rewrite_op_jit_merge_point` emits
        // `SpaceOperation('jit_merge_point', args, None)` with
        // ListOfKind-tagged greens/reds; `assembler.py:220` turns it into
        // `jit_merge_point/IRR` (int-list, ref-list, ref-list). pyre
        // emits this at the portal loop header.
        macro_rules! emit_jit_merge_point {
            ($ssarepr:expr, $asm:expr, $greens_i:expr, $greens_r:expr, $reds_r:expr) => {{
                let greens_i: &[u8] = $greens_i;
                let greens_r: &[u8] = $greens_r;
                let reds_r: &[u8] = $reds_r;
                let to_list = |kind: Kind, regs: &[u8]| {
                    ListOfKind::new(
                        kind,
                        regs.iter()
                            .map(|&r| Operand::Register(Register::new(kind, r as u16)))
                            .collect(),
                    )
                };
                $ssarepr.insns.push(Insn::op(
                    "jit_merge_point",
                    vec![
                        Operand::ListOfKind(to_list(Kind::Int, greens_i)),
                        Operand::ListOfKind(to_list(Kind::Ref, greens_r)),
                        Operand::ListOfKind(to_list(Kind::Ref, reds_r)),
                    ],
                ));
                $asm.jit_merge_point(greens_i, greens_r, reds_r);
            }};
        }

        for py_pc in 0..num_instrs {
            // Exception handler entry: Python resets stack depth to the
            // handler's specified depth. Correct current_depth to match.
            if let Some(&handler_depth) = handler_depth_at.get(&py_pc) {
                current_depth = handler_depth;
            }
            // RPython flatten.py: Label(block) at block entry
            emit_mark_label_pc!(ssarepr, assembler, labels, py_pc);
            pc_map[py_pc] = assembler.current_pos();
            let live_patch = emit_live_placeholder!(ssarepr, assembler);
            live_patches.push((py_pc, live_patch));
            depth_at_pc[py_pc] = current_depth;

            if loop_header_pcs.contains(&py_pc) {
                if merge_point_pc == Some(py_pc) {
                    // interp_jit.py:64 portal contract:
                    //   greens = ['next_instr', 'is_being_profiled', 'pycode']
                    //   reds = ['frame', 'ec']
                    assembler.emit_portal_jit_merge_point(
                        py_pc,
                        w_code as i64,
                        portal_frame_reg,
                        portal_ec_reg,
                    );
                } else {
                    // pyre has a single jitdriver (PyPyJitDriver), index 0.
                    assembler.loop_header(0);
                }
            }

            let code_unit = code.instructions[py_pc];
            let (instruction, op_arg) = arg_state.get(code_unit);

            // pyframe.py:379-417 pushvalue/popvalue_maybe_none parity:
            // RPython's push/pop each write `self.valuestackdepth = depth +/- 1`.
            // On the JIT, these map to per-push `setfield_vable_i`. pyre's
            // codewriter stores stack values in typed registers rather than
            // the `locals_cells_stack_w` array, so we cannot emit a vable
            // setitem for each push. As the coarsest RPython-compatible
            // approximation we flush `valuestackdepth` once at opcode entry,
            // reflecting the pre-opcode stack depth — which is what the
            // interpreter (eval.rs:92 `target_depth = frame.nlocals() +
            // frame.ncells() + entry.depth`) uses when an exception handler
            // unwinds the frame.
            //
            // RPython interp_jit.py keeps `next_instr` as a green portal
            // argument and updates `last_instr` in the interpreter loop; it
            // does not lower a per-bytecode virtualizable write here. pyre's
            // portal entry / guard-resume paths already restore
            // `frame.next_instr`, and the interpreter updates `last_instr`
            // once execution returns there. Emitting `py_pc + 1` here only
            // grows the int constant pool linearly with function size and
            // trips assembler.py's 256-entry cap.
            // pyframe.py:379-417: valuestackdepth is written per-push/per-pop
            // via setfield_vable_i (jtransform.py:923-928), NOT once at opcode
            // entry. The per-push/per-pop emit_vsd! calls below mirror that.
            // (The old single-entry flush is removed.)

            // RPython jtransform.py: rewrite_operation() dispatches per opname.
            // Each match arm is the pyre equivalent of rewrite_op_*.
            match instruction {
                Instruction::Resume { .. }
                | Instruction::Nop
                | Instruction::Cache
                | Instruction::NotTaken
                | Instruction::ExtendedArg => {
                    // RPython: no-op operations produce no jitcode output
                }

                // jtransform.py:1877 do_fixed_list_getitem vable case:
                // portal locals are virtualizable array items — emit
                // vable_getarrayitem_ref so the optimizer folds the read
                // against virtualizable_boxes and the blackhole pulls the
                // live frame value into stack_base+current_depth on
                // resume. Non-portal frames keep move_r (no virtualizable
                // in scope).
                Instruction::LoadFast { var_num } | Instruction::LoadFastBorrow { var_num } => {
                    let reg = var_num.get(op_arg).as_usize() as u16;
                    if is_portal {
                        emit_load_const_i!(
                            ssarepr,
                            assembler,
                            int_tmp0,
                            local_to_vable_slot(reg as usize) as i64
                        );
                        emit_vable_getarrayitem_ref!(
                            ssarepr,
                            assembler,
                            stack_base + current_depth,
                            0_u16,
                            int_tmp0
                        );
                    } else {
                        emit_move_r!(ssarepr, assembler, stack_base + current_depth, reg);
                    }
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                // jtransform.py:1898 do_fixed_list_setitem vable case:
                // Portal frames treat `locals_cells_stack_w` as the sole
                // storage for locals — setarrayitem_vable_r writes from
                // the value-stack slot directly, so no register-per-local
                // shadow exists. Non-portal frames keep move_r (no vable
                // in scope).
                Instruction::StoreFast { var_num } => {
                    let reg = var_num.get(op_arg).as_usize() as u16;
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    if is_portal {
                        emit_load_const_i!(
                            ssarepr,
                            assembler,
                            int_tmp0,
                            local_to_vable_slot(reg as usize) as i64
                        );
                        emit_vable_setarrayitem_ref!(
                            ssarepr,
                            assembler,
                            0_u16,
                            int_tmp0,
                            stack_base + current_depth
                        );
                    } else {
                        emit_move_r!(ssarepr, assembler, reg, stack_base + current_depth);
                    }
                }

                Instruction::LoadSmallInt { i } => {
                    let val = i.get(op_arg) as u32 as i64;
                    emit_load_const_i!(ssarepr, assembler, int_tmp0, val);
                    emit_residual_call(
                        &mut ssarepr,
                        &mut assembler,
                        CallFlavor::Plain,
                        box_int_fn_idx,
                        &[majit_metainterp::jitcode::JitCallArg::int(int_tmp0)],
                        ResKind::Ref,
                        Some(obj_tmp0),
                    );
                    emit_move_r!(ssarepr, assembler, stack_base + current_depth, obj_tmp0);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                Instruction::LoadConst { consti } => {
                    let idx = consti.get(op_arg).as_usize();
                    // jtransform.py: getfield_vable_r for pycode (field 1)
                    emit_vable_getfield_ref!(ssarepr, assembler, obj_tmp0, VABLE_CODE_FIELD_IDX);
                    emit_load_const_i!(ssarepr, assembler, int_tmp0, idx as i64);
                    emit_residual_call(
                        &mut ssarepr,
                        &mut assembler,
                        CallFlavor::Plain,
                        load_const_fn_idx,
                        &[
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0),
                            majit_metainterp::jitcode::JitCallArg::int(int_tmp0),
                        ],
                        ResKind::Ref,
                        Some(obj_tmp0),
                    );
                    emit_move_r!(ssarepr, assembler, stack_base + current_depth, obj_tmp0);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                // CPython 3.13 super-instructions LOAD_FAST_LOAD_FAST /
                // LOAD_FAST_BORROW_LOAD_FAST_BORROW decompose to two plain
                // LOAD_FAST reads. Portal parity with plain LoadFast would
                // route both halves through vable_getarrayitem_ref, but
                // flipping this arm (attempted 2026-04-19 after P1 Step 1
                // + P3 seed helper + unroll reserve_pos refactor
                // `66d1f7212d`) still breaks spectral_norm/nbody/fannkuch
                // on both backends. The heap-vs-symbolic-state gap
                // persists — the compiled loop's vable reads still see
                // stale slots that the bridge / blackhole resume path
                // has not re-synchronized. Keep move_r here until the
                // liveness pipeline rework (Priority 4) lands; the full
                // `flatten → compute_liveness(ssarepr) → assemble`
                // sequence should close the gap by ensuring the vable
                // mirror is write-back-consistent before each compiled
                // entry.
                Instruction::LoadFastBorrowLoadFastBorrow { var_nums }
                | Instruction::LoadFastLoadFast { var_nums } => {
                    let pair = var_nums.get(op_arg);
                    let reg_a = u32::from(pair.idx_1()) as u16;
                    let reg_b = u32::from(pair.idx_2()) as u16;
                    emit_move_r!(ssarepr, assembler, stack_base + current_depth, reg_a);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                    emit_move_r!(ssarepr, assembler, stack_base + current_depth, reg_b);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                // Super-instruction STORE_FAST; LOAD_FAST: pop TOS into
                // idx_1 (store), then push idx_2 (load). Net depth 0.
                // Portal: store via setarrayitem_vable_r, load via
                // getarrayitem_vable_r. Non-portal: move_r for both halves.
                Instruction::StoreFastLoadFast { var_nums } => {
                    let pair = var_nums.get(op_arg);
                    let store_reg = u32::from(pair.idx_1()) as u16;
                    let load_reg = u32::from(pair.idx_2()) as u16;
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    if is_portal {
                        emit_load_const_i!(
                            ssarepr,
                            assembler,
                            int_tmp0,
                            local_to_vable_slot(store_reg as usize) as i64
                        );
                        emit_vable_setarrayitem_ref!(
                            ssarepr,
                            assembler,
                            0_u16,
                            int_tmp0,
                            stack_base + current_depth
                        );
                        emit_load_const_i!(
                            ssarepr,
                            assembler,
                            int_tmp0,
                            local_to_vable_slot(load_reg as usize) as i64
                        );
                        emit_vable_getarrayitem_ref!(
                            ssarepr,
                            assembler,
                            stack_base + current_depth,
                            0_u16,
                            int_tmp0
                        );
                    } else {
                        emit_move_r!(ssarepr, assembler, store_reg, stack_base + current_depth);
                        emit_move_r!(ssarepr, assembler, stack_base + current_depth, load_reg);
                    }
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                // STORE_SUBSCR: stack [value, obj, key] → obj[key] = value
                Instruction::StoreSubscr => {
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    emit_move_r!(ssarepr, assembler, obj_tmp1, stack_base + current_depth); // key
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    emit_move_r!(ssarepr, assembler, obj_tmp0, stack_base + current_depth); // obj
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    emit_move_r!(
                        ssarepr,
                        assembler,
                        arg_regs_start,
                        stack_base + current_depth
                    ); // value
                    emit_residual_call(
                        &mut ssarepr,
                        &mut assembler,
                        CallFlavor::MayForce,
                        store_subscr_fn_idx,
                        &[
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0),
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp1),
                            majit_metainterp::jitcode::JitCallArg::reference(arg_regs_start),
                        ],
                        ResKind::Void,
                        None,
                    );
                }

                Instruction::PopTop => {
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    // regalloc.py: discard = just decrement depth, no bytecode
                }

                Instruction::PushNull => {
                    emit_move_r!(ssarepr, assembler, stack_base + current_depth, null_ref_reg);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                // jtransform.py: rewrite_op_int_add etc.
                //
                // Call reads stack slots DIRECTLY rather than copying through
                // obj_tmp0/obj_tmp1 temps. This keeps the call's argument
                // registers inside the trace-tracked range (`symbolic_locals`
                // + `symbolic_stack`), so guards fired across the op (e.g.
                // `GUARD_NOT_FORCED_2` after a helper call) capture the
                // lhs/rhs values in fail_args. See
                // `memory/pyre_trace_temp_reg_tracking_gap_2026_04_19.md`.
                Instruction::BinaryOp { op } => {
                    let op_val = binary_op_tag(op.get(op_arg))
                        .expect("unsupported binary op tag in jitcode lowering")
                        as u32;
                    // Pop rhs (blackhole will see vsd reflect this pop).
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    let rhs_reg = stack_base + current_depth;
                    // Pop lhs.
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    let lhs_reg = stack_base + current_depth;
                    emit_load_const_i!(ssarepr, assembler, op_code_reg, op_val as i64);
                    emit_residual_call(
                        &mut ssarepr,
                        &mut assembler,
                        CallFlavor::MayForce,
                        binary_op_fn_idx,
                        &[
                            majit_metainterp::jitcode::JitCallArg::reference(lhs_reg),
                            majit_metainterp::jitcode::JitCallArg::reference(rhs_reg),
                            majit_metainterp::jitcode::JitCallArg::int(op_code_reg),
                        ],
                        ResKind::Ref,
                        Some(obj_tmp0),
                    );
                    emit_move_r!(ssarepr, assembler, stack_base + current_depth, obj_tmp0);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                // jtransform.py: rewrite_op_int_lt, optimize_goto_if_not
                Instruction::CompareOp { opname } => {
                    // Same stack-direct pattern as BinaryOp — see its comment.
                    let op_val = compare_op_tag(opname.get(op_arg)) as u32;
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    let rhs_reg = stack_base + current_depth;
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    let lhs_reg = stack_base + current_depth;
                    emit_load_const_i!(ssarepr, assembler, op_code_reg, op_val as i64);
                    emit_residual_call(
                        &mut ssarepr,
                        &mut assembler,
                        CallFlavor::MayForce,
                        compare_fn_idx,
                        &[
                            majit_metainterp::jitcode::JitCallArg::reference(lhs_reg),
                            majit_metainterp::jitcode::JitCallArg::reference(rhs_reg),
                            majit_metainterp::jitcode::JitCallArg::int(op_code_reg),
                        ],
                        ResKind::Ref,
                        Some(obj_tmp0),
                    );
                    emit_move_r!(ssarepr, assembler, stack_base + current_depth, obj_tmp0);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                // flatten.py:240-260 + blackhole.py:865-869. truth_fn returns
                // a bool-as-int; emit plain `goto_if_not <bool> L` — the
                // unfused form flatten.py takes when the exitswitch is a
                // plain variable (not a tuple of a foldable comparison op).
                // bhimpl_goto_if_not takes the target when `a == 0`.
                Instruction::PopJumpIfFalse { delta } => {
                    let target_py_pc = jump_target_forward(
                        code,
                        num_instrs,
                        py_pc + 1,
                        delta.get(op_arg).as_usize(),
                    );
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    emit_move_r!(ssarepr, assembler, obj_tmp0, stack_base + current_depth);
                    emit_residual_call(
                        &mut ssarepr,
                        &mut assembler,
                        CallFlavor::Plain,
                        truth_fn_idx,
                        &[majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0)],
                        ResKind::Int,
                        Some(int_tmp0),
                    );
                    if target_py_pc < num_instrs {
                        emit_goto_if_not!(ssarepr, assembler, labels, int_tmp0, target_py_pc);
                    }
                }

                // jtransform.py:1212 `_rewrite_equality` rewrites
                // `int_eq(x, 0)` → `int_is_zero(x)`; flatten.py:247
                // specialises into `goto_if_not_int_is_zero <v> L`
                // (blackhole.py:916-920). Polarity is inverted vs
                // PopJumpIfFalse: target taken iff the truth-fn result
                // is non-zero (truthy).
                Instruction::PopJumpIfTrue { delta } => {
                    let target_py_pc = jump_target_forward(
                        code,
                        num_instrs,
                        py_pc + 1,
                        delta.get(op_arg).as_usize(),
                    );
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    emit_move_r!(ssarepr, assembler, obj_tmp0, stack_base + current_depth);
                    emit_residual_call(
                        &mut ssarepr,
                        &mut assembler,
                        CallFlavor::Plain,
                        truth_fn_idx,
                        &[majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0)],
                        ResKind::Int,
                        Some(int_tmp0),
                    );
                    if target_py_pc < num_instrs {
                        emit_goto_if_not_int_is_zero!(
                            ssarepr,
                            assembler,
                            labels,
                            int_tmp0,
                            target_py_pc
                        );
                    }
                }

                // RPython flatten.py: goto Label
                Instruction::JumpForward { delta } => {
                    let target_py_pc = jump_target_forward(
                        code,
                        num_instrs,
                        py_pc + 1,
                        delta.get(op_arg).as_usize(),
                    );
                    if target_py_pc < num_instrs {
                        emit_goto!(ssarepr, assembler, labels, target_py_pc);
                    }
                }

                instr @ Instruction::JumpBackward { .. } => {
                    if let Some(target_py_pc) = backward_jump_target(code, py_pc, instr, op_arg) {
                        if target_py_pc < num_instrs {
                            assembler.jump(labels[target_py_pc]);
                        }
                    }
                }

                // flatten.py: int_return / ref_return
                Instruction::ReturnValue => {
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    emit_move_r!(ssarepr, assembler, obj_tmp0, stack_base + current_depth);
                    emit_ref_return!(ssarepr, assembler, obj_tmp0);
                }

                // RPython jtransform.py: rewrite_op_direct_call (residual)
                Instruction::LoadGlobal { namei } => {
                    let raw_namei = namei.get(op_arg) as usize as i64;
                    // jtransform.py: getfield_vable_r for w_globals (field 3)
                    // and pycode (field 1) — namespace for lookup, code for names.
                    emit_vable_getfield_ref!(
                        ssarepr,
                        assembler,
                        obj_tmp0,
                        VABLE_NAMESPACE_FIELD_IDX
                    );
                    emit_vable_getfield_ref!(ssarepr, assembler, obj_tmp1, VABLE_CODE_FIELD_IDX);
                    emit_load_const_i!(ssarepr, assembler, int_tmp0, raw_namei);
                    emit_residual_call(
                        &mut ssarepr,
                        &mut assembler,
                        CallFlavor::MayForce,
                        load_global_fn_idx,
                        &[
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0),
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp1),
                            majit_metainterp::jitcode::JitCallArg::int(int_tmp0),
                        ],
                        ResKind::Ref,
                        Some(obj_tmp0),
                    );
                    // LOAD_GLOBAL with (namei >> 1) & 1: push NULL first
                    if raw_namei & 1 != 0 {
                        emit_move_r!(ssarepr, assembler, stack_base + current_depth, null_ref_reg);
                        current_depth += 1;
                        emit_vsd!(assembler, current_depth);
                    }
                    emit_move_r!(ssarepr, assembler, stack_base + current_depth, obj_tmp0);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                // RPython jtransform.py: rewrite_op_direct_call →
                // call_may_force / residual_call
                //
                // RPython blackhole.py: bhimpl_recursive_call_i calls
                // portal_runner directly, bypassing JIT entry.
                // Here we pop args and callable from the stack into
                // registers, then call the helper with explicit args.
                //
                // Stack layout before CALL(argc):
                //   [NULL, callable, arg0, arg1, ..., arg(argc-1)]
                // We pop in reverse: args first, then callable, then NULL.
                Instruction::Call { argc } => {
                    let nargs = argc.get(op_arg) as usize;
                    for i in (0..nargs).rev() {
                        current_depth -= 1;
                        emit_vsd!(assembler, current_depth);
                        emit_move_r!(
                            ssarepr,
                            assembler,
                            arg_regs_start + i as u16,
                            stack_base + current_depth
                        );
                    }
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    emit_move_r!(ssarepr, assembler, obj_tmp1, stack_base + current_depth); // callable
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth); // NULL (discard)

                    // RPython: bhimpl_recursive_call_i(jdindex, greens, reds)
                    // call_fn(callable, arg0, ...) → result
                    // Parent frame accessed via BH_VABLE_PTR thread-local.
                    let mut call_args =
                        vec![majit_metainterp::jitcode::JitCallArg::reference(obj_tmp1)];
                    for i in 0..nargs {
                        call_args.push(majit_metainterp::jitcode::JitCallArg::reference(
                            arg_regs_start + i as u16,
                        ));
                    }
                    // Select the correct arity-specific call helper.
                    // RPython blackhole.py: call_int_function transmutes
                    // to the correct arity. Each nargs needs a matching
                    // extern "C" fn with that many i64 parameters.
                    // nargs > 8 → abort_permanent (no matching helper).
                    if nargs > 8 {
                        assembler.abort_permanent();
                    } else {
                        let fn_idx = match nargs {
                            0 => call_fn_0_idx,
                            1 => call_fn_idx,
                            2 => call_fn_2_idx,
                            3 => call_fn_3_idx,
                            4 => call_fn_4_idx,
                            5 => call_fn_5_idx,
                            6 => call_fn_6_idx,
                            7 => call_fn_7_idx,
                            _ => call_fn_8_idx,
                        };
                        emit_residual_call(
                            &mut ssarepr,
                            &mut assembler,
                            CallFlavor::MayForce,
                            fn_idx,
                            &call_args,
                            ResKind::Ref,
                            Some(obj_tmp0),
                        );
                    }
                    emit_move_r!(ssarepr, assembler, stack_base + current_depth, obj_tmp0);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                // Python 3.13: ToBool converts TOS to bool before branch.
                // No-op in JitCode: the value is already truthy/falsy and
                // the following PopJumpIfFalse guards on it.
                Instruction::ToBool => {}

                // RPython bhimpl_int_neg: -obj via binary_op(0, obj, NB_SUBTRACT)
                Instruction::UnaryNegative => {
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    emit_move_r!(ssarepr, assembler, obj_tmp0, stack_base + current_depth);
                    emit_load_const_i!(ssarepr, assembler, int_tmp0, 0);
                    emit_residual_call(
                        &mut ssarepr,
                        &mut assembler,
                        CallFlavor::MayForce,
                        box_int_fn_idx,
                        &[majit_metainterp::jitcode::JitCallArg::int(int_tmp0)],
                        ResKind::Ref,
                        Some(obj_tmp1),
                    );
                    emit_load_const_i!(
                        ssarepr,
                        assembler,
                        int_tmp0,
                        binary_op_tag(pyre_interpreter::bytecode::BinaryOperator::Subtract)
                            .expect("subtract must have a jit binary-op tag"),
                    );
                    emit_residual_call(
                        &mut ssarepr,
                        &mut assembler,
                        CallFlavor::MayForce,
                        binary_op_fn_idx,
                        &[
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp1),
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0),
                            majit_metainterp::jitcode::JitCallArg::int(int_tmp0),
                        ],
                        ResKind::Ref,
                        Some(obj_tmp0),
                    );
                    emit_move_r!(ssarepr, assembler, stack_base + current_depth, obj_tmp0);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                // JumpBackwardNoInterrupt reuses `backward_jump_target`:
                // the encoding differs from JumpBackward (no skip_caches
                // on the next-PC base) but the helper routes each variant
                // to its correct arithmetic so pre-scan and emit stay in
                // lockstep.  interp_jit.py:103 + jtransform.py:1714.
                instr @ Instruction::JumpBackwardNoInterrupt { .. } => {
                    if let Some(target_py_pc) = backward_jump_target(code, py_pc, instr, op_arg) {
                        if target_py_pc < num_instrs {
                            assembler.jump(labels[target_py_pc]);
                        }
                    }
                }

                // RPython bhimpl_newlist: build list from N items on stack.
                Instruction::BuildList { count } => {
                    let argc = count.get(op_arg) as usize;
                    for i in (0..argc.min(2)).rev() {
                        current_depth -= 1;
                        emit_vsd!(assembler, current_depth);
                        emit_move_r!(
                            ssarepr,
                            assembler,
                            arg_regs_start + i as u16,
                            stack_base + current_depth
                        );
                    }
                    // Discard extra items beyond 2 (helper supports 0-2).
                    for _ in 2..argc {
                        current_depth -= 1;
                        emit_vsd!(assembler, current_depth);
                    }
                    // build_list_fn(argc, item0, item1) → list
                    emit_load_const_i!(ssarepr, assembler, int_tmp0, argc as i64);
                    let item0 = if argc >= 1 {
                        majit_metainterp::jitcode::JitCallArg::reference(arg_regs_start)
                    } else {
                        majit_metainterp::jitcode::JitCallArg::int(int_tmp0) // dummy
                    };
                    let item1 = if argc >= 2 {
                        majit_metainterp::jitcode::JitCallArg::reference(arg_regs_start + 1)
                    } else {
                        majit_metainterp::jitcode::JitCallArg::int(int_tmp0) // dummy
                    };
                    emit_residual_call(
                        &mut ssarepr,
                        &mut assembler,
                        CallFlavor::MayForce,
                        build_list_fn_idx,
                        &[
                            majit_metainterp::jitcode::JitCallArg::int(int_tmp0),
                            item0,
                            item1,
                        ],
                        ResKind::Ref,
                        Some(obj_tmp0),
                    );
                    emit_move_r!(ssarepr, assembler, stack_base + current_depth, obj_tmp0);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                // Exception handling: residual calls to frame helpers.
                // RPython blackhole.py handles exceptions via dedicated
                // bhimpl_* functions. In pyre, we delegate to the frame's
                // exception machinery via call_fn.
                Instruction::RaiseVarargs { argc } => {
                    let n = argc.get(op_arg) as i64;
                    if n >= 1 {
                        current_depth -= 1;
                        emit_vsd!(assembler, current_depth);
                        emit_move_r!(ssarepr, assembler, obj_tmp0, stack_base + current_depth);
                        emit_raise!(ssarepr, assembler, obj_tmp0);
                    } else {
                        // reraise: re-raise exception_last_value
                        emit_reraise!(ssarepr, assembler);
                    }
                }

                Instruction::PushExcInfo => {
                    // flatten.py: dup = ref_copy TOS → TOS+1
                    emit_move_r!(
                        ssarepr,
                        assembler,
                        stack_base + current_depth,
                        stack_base + current_depth - 1
                    );
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                Instruction::CheckExcMatch => {
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    emit_move_r!(ssarepr, assembler, obj_tmp1, stack_base + current_depth); // match type
                    current_depth -= 1;
                    emit_vsd!(assembler, current_depth);
                    emit_move_r!(ssarepr, assembler, obj_tmp0, stack_base + current_depth); // exception
                    // isinstance check via compare_fn(exc, type, ISINSTANCE_OP)
                    emit_load_const_i!(ssarepr, assembler, int_tmp0, 10); // isinstance op
                    emit_residual_call(
                        &mut ssarepr,
                        &mut assembler,
                        CallFlavor::MayForce,
                        compare_fn_idx,
                        &[
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp0),
                            majit_metainterp::jitcode::JitCallArg::reference(obj_tmp1),
                            majit_metainterp::jitcode::JitCallArg::int(int_tmp0),
                        ],
                        ResKind::Ref,
                        Some(obj_tmp0),
                    );
                    emit_move_r!(ssarepr, assembler, stack_base + current_depth, obj_tmp0);
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                Instruction::PopExcept => {
                    // Clear current exception — no-op in blackhole
                    // (exception state is in TLS, cleared by interpreter).
                }

                Instruction::Reraise { .. } => {
                    // Exception path: abort_permanent.
                    assembler.abort_permanent();
                }

                Instruction::Copy { i } => {
                    let d = i.get(op_arg) as usize;
                    if d == 1 {
                        assembler
                            .move_r(stack_base + current_depth, stack_base + current_depth - 1);
                        current_depth += 1;
                        emit_vsd!(assembler, current_depth);
                    } else {
                        // COPY(d>1): exception handler pattern only.
                        // Use abort_permanent (BC_ABORT_PERMANENT=14) so it
                        // doesn't trigger the has_abort(BC_ABORT=13) check.
                        assembler.abort_permanent();
                    }
                }

                Instruction::LoadName { .. }
                | Instruction::StoreName { .. }
                | Instruction::MakeFunction { .. } => {
                    // Module-level only: abort_permanent (won't block blackhole).
                    assembler.abort_permanent();
                }

                // CPython 3.13 superinstruction: STORE_FAST_STORE_FAST.
                // jtransform.py:1898 — each local write → setarrayitem_vable_r
                // in portal, move_r in non-portal. Mirrors plain StoreFast.
                Instruction::StoreFastStoreFast { var_nums } => {
                    let pair = var_nums.get(op_arg);
                    let reg_a = u32::from(pair.idx_1()) as u16;
                    let reg_b = u32::from(pair.idx_2()) as u16;
                    for reg in [reg_a, reg_b] {
                        current_depth -= 1;
                        emit_vsd!(assembler, current_depth);
                        if is_portal {
                            emit_load_const_i!(
                                ssarepr,
                                assembler,
                                int_tmp0,
                                local_to_vable_slot(reg as usize) as i64,
                            );
                            emit_vable_setarrayitem_ref!(
                                ssarepr,
                                assembler,
                                0_u16,
                                int_tmp0,
                                stack_base + current_depth
                            );
                        } else {
                            emit_move_r!(ssarepr, assembler, reg, stack_base + current_depth);
                        }
                    }
                }

                // CPython 3.13 UNPACK_SEQUENCE: pop 1 (seq), push `count`.
                // Emit abort_permanent (no getitem helper yet) but
                // adjust current_depth so subsequent instructions don't
                // underflow.
                Instruction::UnpackSequence { count } => {
                    let n = count.get(op_arg) as u16;
                    assembler.abort_permanent();
                    // Stack effect: pop 1 + push n = net (n - 1)
                    if current_depth > 0 {
                        current_depth -= 1;
                        emit_vsd!(assembler, current_depth);
                    }
                    current_depth += n;
                }

                // CPython 3.13 iterator protocol — emit abort_permanent
                // with correct depth tracking so subsequent instructions
                // don't underflow.
                Instruction::GetIter => {
                    // pop iterable, push iterator: net 0
                    assembler.abort_permanent();
                }

                Instruction::ForIter { .. } => {
                    // push next item: net +1
                    assembler.abort_permanent();
                    current_depth += 1;
                    emit_vsd!(assembler, current_depth);
                }

                Instruction::EndFor => {
                    // pop iterator + last value: net -2
                    assembler.abort_permanent();
                    current_depth = current_depth.saturating_sub(2);
                    // No emit_vsd: after abort_permanent, depth is
                    // simulation-only for subsequent compile-time tracking.
                }

                Instruction::PopIter => {
                    // pop iterator: net -1
                    current_depth = current_depth.saturating_sub(1);
                    emit_vsd!(assembler, current_depth);
                }

                // Unsupported instruction: abort_permanent.
                _other => {
                    assembler.abort_permanent();
                }
            }
            if let Some(catch_label) = catch_for_pc[py_pc] {
                emit_catch_exception!(ssarepr, assembler, catch_label);
            }
        }

        // RPython flatten.py parity: every code path ends with an explicit
        // return/raise/goto/unreachable. No end-of-code sentinel needed —
        // falling off the end is unreachable if all bytecodes are covered.

        // pyre-only PyJitCode.has_abort: a "this jitcode cannot be
        // blackhole-dispatched, pipe straight to the interpreter" flag.
        // RPython has no such flag (rpython/jit/codewriter/jitcode.py:14
        // — no abort tracking on JitCode). Upstream's `Assembler.abort()`
        // (assembler.py:177-181, bhimpl_abort) emits BC_ABORT so the
        // blackhole raises SwitchToBlackhole(ABORT_ESCAPE) at runtime;
        // `abort_permanent()` is a different pyre-only bytecode we emit
        // for genuinely unsupported Python opcodes, and its execution
        // path already raises/aborts correctly from the blackhole. We
        // keep has_abort narrowly scoped to `abort()` emissions (matches
        // the JitCodeBuilder flag shape) so the flag's meaning doesn't
        // drift into "assembler overflow" or "abort_permanent present"
        // — both of which the assembler/blackhole already handle without
        // a front-end gate.
        let mut has_abort = assembler.has_abort_flag();

        // liveness.py:19-80 parity: generate LivenessInfo at each
        // bytecode PC via SSA backward dataflow over the populated
        // SSARepr, with a LiveVars intersection at each pc to match
        // the runtime contract. See `compute_liveness_table`.
        let liveness = compute_liveness_table(
            &mut assembler,
            &live_patches,
            code,
            num_instrs,
            nlocals,
            stack_base,
            &depth_at_pc,
            &pc_map,
        );

        // liveness.py:19-23 parity: patch each SSARepr `-live-` slot
        // with the per-PC register triple. The post-pass register
        // allocator below renames these registers to colors that fit
        // in `encode_liveness`'s 8-bit precondition; the legacy
        // pyre-only `has_abort` fallback for `register index ≥ 256`
        // is gone (RPython parity: `liveness.py:147-166` panics if a
        // post-regalloc register exceeds 256, and so do we).
        fill_assembler_liveness(&mut assembler, &liveness, &live_patches);

        for site in catch_sites {
            emit_mark_label_catch_landing!(ssarepr, assembler, site.landing_label);
            let mut exc_slot = stack_base + site.stack_depth;
            if site.push_lasti {
                emit_load_const_i!(ssarepr, assembler, int_tmp0, site.lasti_py_pc as i64);
                emit_residual_call(
                    &mut ssarepr,
                    &mut assembler,
                    CallFlavor::Plain,
                    box_int_fn_idx,
                    &[majit_metainterp::jitcode::JitCallArg::int(int_tmp0)],
                    ResKind::Ref,
                    Some(obj_tmp0),
                );
                emit_move_r!(ssarepr, assembler, exc_slot, obj_tmp0);
                exc_slot += 1;
            }
            emit_last_exc_value!(ssarepr, assembler, exc_slot);
            emit_goto!(ssarepr, assembler, labels, site.handler_py_pc);
        }

        // codewriter.py:45-47 `for kind in KINDS:
        //   regallocs[kind] = perform_register_allocation(graph, kind)`
        // The pyre-side post-pass scans the populated SSARepr to build
        // per-kind dependency graphs, runs chordal coloring on each,
        // and returns the rename map applied to the SSARepr in place.
        let max_stack_depth_observed = depth_at_pc.iter().copied().max().unwrap_or(0);
        let inputs = super::regalloc::ExternalInputs {
            portal_frame_reg,
            portal_ec_reg,
            // Portal frames carry a virtualizable + ec red argument
            // pair (interp_jit.py:64-69). Non-portal callees pass red
            // args via the call assembler edge; the dispatch loop
            // does not pre-load them into Ref registers.
            portal_inputs: portal_frame_reg != u16::MAX,
            stack_base,
            max_stack_depth: max_stack_depth_observed,
        };
        let alloc_result =
            super::regalloc::allocate_registers(&assembler.ssarepr, code.varnames.len(), inputs);
        let mut assembler = assembler;
        super::regalloc::apply_rename(&mut assembler.ssarepr, &alloc_result.rename);

        // codewriter.py:62-67 num_regs[kind] = max(coloring)+1
        // (or 0 if coloring is empty). Pass through to the Assembler
        // step so `JitCode.num_regs_*` reflect the post-regalloc
        // ceiling rather than the pre-regalloc PyFrame-slot range.
        let num_regs = super::assembler::NumRegs {
            int: alloc_result
                .num_regs
                .get(&super::flatten::Kind::Int)
                .copied()
                .unwrap_or(0),
            ref_: alloc_result
                .num_regs
                .get(&super::flatten::Kind::Ref)
                .copied()
                .unwrap_or(0),
            float: alloc_result
                .num_regs
                .get(&super::flatten::Kind::Float)
                .copied()
                .unwrap_or(0),
        };

        // codewriter.py:67-72 step 4 — assemble the SSARepr into an
        // owned JitCode, translate pc_map insn indices to byte offsets,
        // and stamp the per-graph metadata. See `Self::finalize_jitcode`.
        self.finalize_jitcode(
            assembler,
            code,
            pc_map,
            depth_at_pc,
            liveness,
            portal_frame_reg,
            portal_ec_reg,
            has_abort,
            merge_point_pc,
            num_regs,
        )
    }

    /// RPython: `codewriter.py:62-72` step 4 — produce the
    /// owned `JitCode` from the populated `SSARepr` and stamp the
    /// per-graph metadata.
    ///
    /// ```python
    /// num_regs = {kind: ... for kind in KINDS}
    /// self.assembler.assemble(ssarepr, jitcode, num_regs)
    /// jitcode.index = index
    /// ```
    ///
    /// pyre's combined step:
    ///   - `SSAReprEmitter::finish_with_positions` runs the
    ///     `assembler.py:assemble` analog through the shared
    ///     `self.assembler`, returning the owned `JitCode` plus the
    ///     translated `pc_map` byte offsets.
    ///   - jitdriver_sd / calldescr / fnaddr are stamped onto the
    ///     `JitCode` (call.py:148, call.py:174-187, call.py:167).
    ///   - `PyJitCodeMetadata` is bundled with the ref-count-stable
    ///     `Arc<JitCode>` plus the pyre-only `has_abort` /
    ///     `merge_point_pc` fields into the returned `PyJitCode`.
    fn finalize_jitcode(
        &self,
        assembler: SSAReprEmitter,
        code: &CodeObject,
        pc_map: Vec<usize>,
        depth_at_pc: Vec<u16>,
        liveness: Vec<LivenessInfo>,
        portal_frame_reg: u16,
        portal_ec_reg: u16,
        has_abort: bool,
        merge_point_pc: Option<usize>,
        num_regs: super::assembler::NumRegs,
    ) -> PyJitCode {
        // pc_map[py_pc] currently holds SSARepr insn indices (returned by
        // SSAReprEmitter::current_pos()). Translate them to JitCode byte
        // offsets via ssarepr.insns_pos, populated during
        // Assembler::assemble (assembler.py:41-44). Runtime readers
        // (get_live_vars_info, resume dispatch) expect byte offsets.
        //
        // `codewriter.py:67` `self.assembler.assemble(ssarepr, jitcode, num_regs)`
        // parity: borrow the CodeWriter's single Assembler so
        // `all_liveness` / `num_liveness_ops` continue to accumulate
        // across every jitcode compiled on this thread.
        let (mut jitcode, pc_map_bytes) = {
            let mut asm = self.assembler.borrow_mut();
            assembler.finish_with_positions(&mut *asm, &pc_map, num_regs)
        };

        // call.py:148 `jd.mainjitcode.jitdriver_sd = jd` is assigned by
        // `assign_portal_jitdriver_indices` after the drain, where the
        // jdindex is known from this code's position in
        // `CallControl.jitdrivers_sd`. RPython's `JitCode` constructor
        // (jitcode.py:18) leaves `jitdriver_sd = None` for non-portals;
        // `grab_initial_jitcodes` is the single source of truth for
        // the portal assignment, so we leave it `None` here too rather
        // than guessing an index from a local `is_portal` flag (which
        // would collide once a second portal lands or a non-portal
        // slips in first).
        jitcode.jitdriver_sd = None;
        // call.py:167 `(fnaddr, calldescr) = self.get_jitcode_calldescr(graph)`.
        // The values are constant across CodeObjects in pyre — see
        // [`super::call::CallControl::get_jitcode_calldescr`] for the
        // PRE-EXISTING-ADAPTATION rationale.
        let (fnaddr, calldescr) = self
            .callcontrol()
            .get_jitcode_calldescr(code as *const CodeObject);
        jitcode.fnaddr = fnaddr;
        jitcode.calldescr = calldescr;
        // Per-code stack base in `locals_cells_stack_w`. RPython's JitCode
        // does not carry PyFrame layout data; keep it in PyJitCodeMetadata
        // and attach it to BlackholeInterpreter setup when pyre needs it.
        let frame_stack_base = code.varnames.len() + pyre_interpreter::pyframe::ncells(code);

        let metadata = PyJitCodeMetadata {
            pc_map: pc_map_bytes,
            depth_at_py_pc: depth_at_pc,
            portal_frame_reg,
            portal_ec_reg,
            stack_base: frame_stack_base,
            liveness,
        };

        PyJitCode {
            jitcode: std::sync::Arc::new(jitcode),
            metadata,
            has_abort,
            merge_point_pc,
        }
    }

    /// RPython: `CodeWriter.make_jitcodes(verbose)` (codewriter.py:74-89).
    ///
    /// ```python
    /// def make_jitcodes(self, verbose=False):
    ///     log.info("making JitCodes...")
    ///     self.callcontrol.grab_initial_jitcodes()
    ///     count = 0
    ///     all_jitcodes = []
    ///     for graph, jitcode in self.callcontrol.enum_pending_graphs():
    ///         self.transform_graph_to_jitcode(graph, jitcode, verbose, len(all_jitcodes))
    ///         all_jitcodes.append(jitcode)
    ///         count += 1
    ///         if not count % 500:
    ///             log.info("Produced %d jitcodes" % count)
    ///     self.assembler.finished(self.callcontrol.callinfocollection)
    ///     log.info("There are %d JitCode instances." % count)
    ///     log.info("There are %d -live- ops. Size of liveness is %s bytes" % (
    ///         self.assembler.num_liveness_ops, self.assembler.all_liveness_length))
    ///     return all_jitcodes
    /// ```
    ///
    /// Each freshly-compiled `PyJitCode` is `Arc`-wrapped before being
    /// inserted into `CallControl.jitcodes`; the next trace-side
    /// `state::jitcode_for(w_code)` callback hands the same `Arc`
    /// to `MetaInterpStaticData.jitcodes`, so both stores reference
    /// one allocation — the Rust analog of RPython's two stores
    /// referencing the same Python `JitCode` via refcount semantics.
    ///
    /// `grab_initial_jitcodes` reads its seed list from
    /// [`super::call::CallControl::jitdrivers_sd`]; callers register
    /// portals with [`Self::setup_jitdriver`] before invoking this
    /// method (matching codewriter.py:74 — `setup_jitdriver` followed
    /// by `make_jitcodes` is the upstream order).
    pub fn make_jitcodes(&self) -> Vec<*const PyJitCode> {
        // codewriter.py:75 `log.info("making JitCodes...")` — pyre has no
        // codewriter.py log channel, intentionally elided.

        // codewriter.py:76 `self.callcontrol.grab_initial_jitcodes()`.
        self.callcontrol().grab_initial_jitcodes();
        // codewriter.py:79-84 drain + per-jitcode assemble.
        let all_jitcodes = self.drain_unfinished_graphs();
        // call.py:148 `jd.mainjitcode.jitdriver_sd = jd` — assign
        // jdindex to each portal's populated `PyJitCode` AFTER the
        // drain so we use the actual position in
        // `CallControl.jitdrivers_sd` instead of a hardcoded `Some(0)`.
        self.assign_portal_jitdriver_indices();
        // codewriter.py:86-88 final log lines — elided.
        // codewriter.py:89 `return all_jitcodes`.
        all_jitcodes
    }

    /// Drain `CallControl.unfinished_graphs` — the body shared between
    /// [`Self::make_jitcodes`] (portal entry) and
    /// [`compile_jitcode_for_callee`] (trace-side callee path).
    ///
    /// RPython's `make_jitcodes` (codewriter.py:79-85) drains the queue
    /// once and then calls `assembler.finished()`. Both pyre adapters
    /// run the same drain so each batch ends with `assembler.finished()`,
    /// matching `codewriter.py:85`.
    pub(crate) fn drain_unfinished_graphs(&self) -> Vec<*const PyJitCode> {
        let mut all_jitcodes: Vec<*const PyJitCode> = Vec::new();
        // codewriter.py:79 `for graph, jitcode in enum_pending_graphs():`.
        loop {
            let popped = self.callcontrol().enum_pending_graphs();
            let Some((code_ptr, w_code, merge_point_pc)) = popped else {
                break;
            };
            // codewriter.py:80 `self.transform_graph_to_jitcode(graph,
            //                     jitcode, verbose, len(all_jitcodes))`.
            //
            // PRE-EXISTING-ADAPTATION: RPython mutates the empty JitCode
            // skeleton in place (`assembler.assemble(ssarepr, jitcode,
            // num_regs)` at codewriter.py:67); pyre's
            // `transform_graph_to_jitcode` returns a fresh `PyJitCode`
            // and we replace the skeleton entry in `jitcodes` with the
            // populated one so the "jitcode in dict has its body filled
            // after drain" invariant is preserved.
            let pyjitcode =
                self.transform_graph_to_jitcode(unsafe { &*code_ptr }, w_code, merge_point_pc);
            let key = code_ptr as usize;
            let arc = std::sync::Arc::new(pyjitcode);
            let raw_ptr = std::sync::Arc::as_ptr(&arc);
            self.callcontrol().jitcodes.insert(key, arc);
            // codewriter.py:81 `all_jitcodes.append(jitcode)`.
            all_jitcodes.push(raw_ptr);
        }
        // codewriter.py:85 `self.assembler.finished(self.callcontrol.callinfocollection)`.
        self.assembler
            .borrow_mut()
            .finished(&self.callcontrol().callinfocollection);
        all_jitcodes
    }

    /// call.py:148 `jd.mainjitcode.jitdriver_sd = jd` — propagate the
    /// jdindex from `CallControl.jitdrivers_sd` onto each portal's
    /// inner `JitCode.jitdriver_sd`. RPython mutates the same Python
    /// `JitCode` object the dict references; pyre uses
    /// `Arc::get_mut` on the freshly-installed `Arc<PyJitCode>` (and
    /// the inner `Arc<JitCode>`) for the same effect. Idempotent — a
    /// later `make_jitcodes` call may find the `Arc` already cloned
    /// into `MetaInterpStaticData.jitcodes` and skip the assignment;
    /// the previously-set value still holds.
    fn assign_portal_jitdriver_indices(&self) {
        let cc = self.callcontrol();
        // Snapshot the (key, jdindex) pairs first so the borrow on
        // `cc.jitdrivers_sd` is released before we mutate `cc.jitcodes`.
        let assignments: Vec<(usize, usize)> = cc
            .jitdrivers_sd
            .iter()
            .enumerate()
            .map(|(idx, jd)| (jd.portal_graph as usize, idx))
            .collect();
        for (key, idx) in assignments {
            if let Some(arc) = cc.jitcodes.get_mut(&key) {
                if let Some(pyjitcode) = std::sync::Arc::get_mut(arc) {
                    if let Some(inner) = std::sync::Arc::get_mut(&mut pyjitcode.jitcode) {
                        inner.jitdriver_sd = Some(idx);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Jump target calculation (RPython: flatten.py link following)
// ---------------------------------------------------------------------------

/// Forward jump target: skip_caches(next_instr) + delta.
/// Must match pyre-interpreter/pyopcode.rs:jump_target_forward.
fn jump_target_forward(
    code: &CodeObject,
    num_instrs: usize,
    next_instr: usize,
    delta: usize,
) -> usize {
    let target = skip_caches(code, next_instr) + delta;
    target.min(num_instrs)
}

/// Single-source-of-truth backward-jump target calculation used by both
/// the loop-header pre-scan (pypy/module/pypyjit/interp_jit.py:103) and
/// the emitter (jtransform.py:1714 `handle_jit_marker__loop_header`).
///
/// Returns the target PC for `JumpBackward` (with `skip_caches` on the
/// next-PC base) and `JumpBackwardNoInterrupt` (direct `py_pc + 1 - delta`
/// arithmetic to match the interpreter's dispatch in pyopcode.rs).
/// Returns `None` for any non-backward-jump opcode.
fn backward_jump_target(
    code: &CodeObject,
    py_pc: usize,
    instr: Instruction,
    op_arg: pyre_interpreter::OpArg,
) -> Option<usize> {
    match instr {
        Instruction::JumpBackward { delta } => {
            Some(skip_caches(code, py_pc + 1).saturating_sub(delta.get(op_arg).as_usize()))
        }
        Instruction::JumpBackwardNoInterrupt { delta } => {
            Some((py_pc + 1).saturating_sub(delta.get(op_arg).as_usize()))
        }
        _ => None,
    }
}

/// Match pyre-interpreter/pyopcode.rs:skip_caches.
fn skip_caches(code: &CodeObject, mut pos: usize) -> usize {
    let mut state = OpArgState::default();
    while pos < code.instructions.len() {
        let (instruction, _) = state.get(code.instructions[pos]);
        if matches!(instruction, Instruction::Cache) {
            pos += 1;
        } else {
            break;
        }
    }
    pos
}

// ---------------------------------------------------------------------------
// JitCode cache — RPython: `CallControl.get_jitcode` (call.py:155-172).
// The cache + `unfinished_graphs` queue live on `super::call::CallControl`;
// `CallControl::get_jitcode` is the canonical entry point.
// ---------------------------------------------------------------------------

/// Portal entry path: `setup_jitdriver` followed by `make_jitcodes` —
/// the warmspot order at codewriter.py:74-99. RPython runs this once
/// per `@jit_callback` decoration; pyre's portal discovery is lazy,
/// so this adapter fires per JIT entry. `setup_jitdriver` dedups by
/// `portal_graph` so `jitdrivers_sd` stays bounded by the number of
/// unique portals (see [`CodeWriter::setup_jitdriver`] for the
/// PRE-EXISTING-ADAPTATION rationale).
///
/// `make_jitcodes` is then the canonical RPython no-arg call: it
/// pulls its seed list from `CallControl.jitdrivers_sd` and runs
/// `grab_initial_jitcodes` → drain → `assembler.finished()` →
/// `assign_portal_jitdriver_indices`. The final stages short-circuit
/// on the second-and-later registration of any given portal because
/// `get_jitcode` skips already-built entries.
///
/// **Use this only for true portals.** Trace-side callee compiles go
/// through [`compile_jitcode_for_callee`] so they never touch
/// `jitdrivers_sd` — see `feedback_setup_jitdriver_portal_only`.
pub fn register_portal_jitdriver(
    code: &pyre_interpreter::CodeObject,
    w_code: *const (),
    merge_point_pc: Option<usize>,
) {
    let writer = CodeWriter::instance();
    // codewriter.py:96-99 `setup_jitdriver(jd)` — register the
    // portal so `grab_initial_jitcodes` finds it.
    writer.setup_jitdriver(super::call::JitDriverStaticData {
        portal_graph: code as *const pyre_interpreter::CodeObject,
        w_code,
        merge_point_pc,
    });
    // codewriter.py:74 `make_jitcodes()` — drain everything pending.
    writer.make_jitcodes();
}

/// Callee compile path: `CallControl.get_jitcode(graph)` followed by
/// the shared drain — the analog of jtransform's
/// `cc.callcontrol.get_jitcode(callee_graph)` (call.py:155-172) that
/// inserts the callee into `CallControl.jitcodes` and lets the
/// surrounding `make_jitcodes` drain transform it. RPython never
/// touches `jitdrivers_sd` here; pyre matches that by going through
/// `get_jitcode` + the shared drain helper directly, *not* through
/// `setup_jitdriver`.
pub fn compile_jitcode_for_callee(code: &pyre_interpreter::CodeObject, w_code: *const ()) {
    let writer = CodeWriter::instance();
    // call.py:155 `get_jitcode(graph)` — insert skeleton + queue.
    let _ = writer.callcontrol().get_jitcode(code, w_code, None);
    // codewriter.py:79-85 drain + assembler.finished() for the
    // newly-queued entry. No portal-jitdriver assignment because
    // this code is a callee, not a portal.
    writer.drain_unfinished_graphs();
}

/// Trace-side hook registered with `pyre_jit_trace::set_compile_jitcode_fn`
/// from `CodeWriter::new()`. Resolves the `w_code` (a `PyObjectRef`
/// wrapping a `CodeObject`) and routes it through
/// [`compile_jitcode_for_callee`] so the lazy compile pipeline runs
/// for one entry without polluting `jitdrivers_sd`.
///
/// RPython parity: jtransform's call to `cc.callcontrol.get_jitcode(callee)`
/// (call.py:155-172) inserts the callee into `CallControl.jitcodes`
/// and queues it on `unfinished_graphs`; the surrounding
/// `make_jitcodes` drain (codewriter.py:79-85) then transforms the
/// queued graph and pipes the result through `assembler.finished()`.
/// Pyre fires this callback per trace-side `state::jitcode_for(w_code)`
/// so the same transitive closure the RPython warmspot builds eagerly
/// is built lazily here, one callee at a time. The callback explicitly
/// must NOT call `setup_jitdriver` — see
/// `feedback_setup_jitdriver_portal_only`.
fn compile_jitcode_via_w_code(w_code: *const ()) -> Option<std::sync::Arc<PyJitCode>> {
    if w_code.is_null() {
        return None;
    }
    let raw_code = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    if raw_code.is_null() {
        return None;
    }
    let code = unsafe { &*raw_code };
    compile_jitcode_for_callee(code, w_code);
    // Hand the populated `Arc` back to the trace-side caller so the
    // SD entry stores the same allocation as `CallControl.jitcodes`.
    CodeWriter::instance()
        .callcontrol()
        .find_jitcode_arc(code as *const _)
}

/// Scan `code` for JUMP_BACKWARD targets — the PCs where
/// `transform_graph_to_jitcode` would emit `BC_JUMP_TARGET` and where
/// `jit_merge_point` is evaluated.
///
/// RPython parity: corresponds to `jtransform.py:1714-1718`
/// `handle_jit_marker__loop_header`, which walks the flow graph looking
/// for `jit_marker('loop_header', ...)` operations. pyre's "graph" is
/// raw Python bytecode, so the equivalent scan looks for
/// `JUMP_BACKWARD` instructions and resolves their target PCs.
///
/// Shared between `transform_graph_to_jitcode` (which uses the set to
/// decide emit locations) and `is_portal` (which tests emptiness to
/// classify the CodeObject without triggering a compile).
pub fn find_loop_header_pcs(
    code: &pyre_interpreter::CodeObject,
) -> std::collections::HashSet<usize> {
    let num_instrs = code.instructions.len();
    let mut loop_header_pcs: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut scan_state = OpArgState::default();
    for scan_pc in 0..num_instrs {
        let (scan_instr, scan_arg) = scan_state.get(code.instructions[scan_pc]);
        if let Some(target) = backward_jump_target(code, scan_pc, scan_instr, scan_arg) {
            if target < num_instrs {
                loop_header_pcs.insert(target);
            }
        }
    }
    loop_header_pcs
}

// `liveness_regs_to_u8_sorted` tests removed alongside the helper.
// The 256-register cap is now enforced inside `encode_liveness` and
// covered by `majit_translate::liveness::encode_liveness*` tests.
