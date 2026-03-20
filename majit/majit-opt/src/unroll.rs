/// Loop unrolling pass (peel one iteration).
///
/// Detects loops ending with a `Jump` back-edge and peels one iteration:
/// the loop body is duplicated to create a "preamble" that executes once
/// before the main loop. Guards in the peeled preamble serve as initial
/// type checks, enabling downstream passes to remove redundant guards
/// from the main loop body.
///
/// The peeled structure looks like:
///
/// ```text
/// [peeled body]          ← first iteration (preamble), guards act as type checks
///   Label(...)           ← loop header
/// [original body]        ← main loop body
///   Jump(...)            ← back-edge to Label
/// ```
///
/// OpRefs in the peeled iteration are remapped to new positions so they
/// don't collide with the original ops.
use std::collections::HashMap;

use majit_ir::{Op, OpCode, OpRef};

use crate::{OptContext, Optimization, OptimizationResult};

/// unroll.py: UnrollOptimizer — high-level loop optimization controller.
///
/// Wraps the streaming OptUnroll pass with RPython's UnrollOptimizer API:
/// - optimize_preamble: process and optimize the first iteration
/// - optimize_peeled_loop: optimize the main loop body
/// - optimize_bridge: compile a bridge into an unrolled loop
pub struct UnrollOptimizer {
    /// Maximum retrace attempts for a single loop location.
    /// unroll.py: retrace_limit from WarmEnterState.
    pub retrace_limit: u32,
    /// Number of guards in the bridge (for retrace_limit checks).
    pub bridge_guard_count: u32,
    /// The short preamble from the preamble optimization pass.
    pub short_preamble: Option<crate::shortpreamble::ShortPreamble>,
    /// The exported virtual state at the loop header.
    pub exported_state: Option<crate::virtualstate::VirtualState>,
}

impl UnrollOptimizer {
    pub fn new() -> Self {
        UnrollOptimizer {
            retrace_limit: 0,
            bridge_guard_count: 0,
            short_preamble: None,
            exported_state: None,
        }
    }

    /// unroll.py: optimize_preamble(trace, runtime_boxes)
    /// Optimize the preamble (first iteration) of a loop trace.
    /// Returns the optimized preamble ops + the peeled loop ops.
    pub fn optimize_preamble(&mut self, ops: &[Op]) -> Vec<Op> {
        let mut optimizer = crate::optimizer::Optimizer::default_pipeline();
        optimizer.add_pass(Box::new(OptUnroll::new()));
        optimizer.optimize(ops)
    }

    /// unroll.py: optimize_peeled_loop(trace)
    /// Optimize the loop body AFTER preamble peeling.
    /// The peeled preamble has already established the type/class/bounds
    /// information; this method optimizes the repeating body.
    pub fn optimize_peeled_loop(&mut self, ops: &[Op]) -> Vec<Op> {
        let mut optimizer = crate::optimizer::Optimizer::default_pipeline();
        optimizer.optimize(ops)
    }

    /// unroll.py: optimize_bridge(trace, runtime_boxes, call_pure_results)
    /// unroll.py: optimize_bridge(trace, runtime_boxes, call_pure_results)
    /// Optimize a bridge trace that enters an existing loop.
    ///
    /// If a short preamble is available, its ops are prepended to the bridge
    /// so the optimizer can re-establish the invariants that the loop body
    /// depends on (type checks, bounds, cached values).
    pub fn optimize_bridge(&mut self, bridge_ops: &[Op]) -> Vec<Op> {
        self.optimize_bridge_with_label_args(bridge_ops, &[])
    }

    /// Optimize a bridge with explicit label args for short preamble instantiation.
    pub fn optimize_bridge_with_label_args(
        &mut self,
        bridge_ops: &[Op],
        label_args: &[OpRef],
    ) -> Vec<Op> {
        let mut optimizer = crate::optimizer::Optimizer::default_pipeline();
        let mut full_ops = Vec::new();

        // unroll.py: inline short preamble ops at the start of the bridge.
        if let Some(ref sp) = self.short_preamble {
            if !sp.is_empty() && !label_args.is_empty() {
                let preamble_ops = sp.instantiate(label_args);
                full_ops.extend(preamble_ops);
            }
        }

        full_ops.extend_from_slice(bridge_ops);
        optimizer.optimize(&full_ops)
    }

    /// Whether we've exceeded the retrace limit for this loop location.
    /// unroll.py: checks retrace_limit in optimize_bridge.
    pub fn should_give_up(&self, retrace_count: u32) -> bool {
        retrace_count >= self.retrace_limit
    }

    /// unroll.py: export_state(target_token)
    /// Export the virtual state at the current loop header.
    pub fn set_exported_state(&mut self, state: crate::virtualstate::VirtualState) {
        self.exported_state = Some(state);
    }

    /// unroll.py: get_virtual_state()
    /// Get the exported virtual state for this loop.
    pub fn get_exported_state(&self) -> Option<&crate::virtualstate::VirtualState> {
        self.exported_state.as_ref()
    }

    /// unroll.py: set_short_preamble(sp)
    pub fn set_short_preamble(&mut self, sp: crate::shortpreamble::ShortPreamble) {
        self.short_preamble = Some(sp);
    }

    /// unroll.py: get_short_preamble()
    pub fn get_short_preamble(&self) -> Option<&crate::shortpreamble::ShortPreamble> {
        self.short_preamble.as_ref()
    }

    /// unroll.py: check_retrace_count(retrace_count, max_retrace_guards)
    /// Whether this bridge has too many guards for retrace to be worthwhile.
    pub fn too_many_guards_for_retrace(&self, retrace_count: u32, max_retrace_guards: u32) -> bool {
        if retrace_count >= self.retrace_limit {
            return true;
        }
        self.bridge_guard_count > max_retrace_guards
    }

    /// unroll.py: optimize_trace(trace, call_pure_results)
    /// Full trace optimization: peel → optimize preamble → optimize body.
    /// Returns the optimized peeled+body trace.
    pub fn optimize_trace(&mut self, ops: &[Op]) -> Vec<Op> {
        let result = self.optimize_preamble(ops);
        // After peeling, extract short preamble from the result.
        let sp = crate::shortpreamble::extract_short_preamble(&result);
        if !sp.is_empty() {
            self.short_preamble = Some(sp);
        }
        result
    }

    /// unroll.py: optimize_trace_with_constants
    /// Same as optimize_trace but with known constants.
    pub fn optimize_trace_with_constants(
        &mut self,
        ops: &[Op],
        constants: &mut std::collections::HashMap<u32, i64>,
    ) -> Vec<Op> {
        let mut optimizer = crate::optimizer::Optimizer::default_pipeline();
        optimizer.add_pass(Box::new(OptUnroll::new()));
        let result = optimizer.optimize_with_constants(ops, constants);
        let sp = crate::shortpreamble::extract_short_preamble(&result);
        if !sp.is_empty() {
            self.short_preamble = Some(sp);
        }
        result
    }

    /// optimize_trace with constants AND explicit num_inputs.
    /// compile.py: compile_loop → optimize with preamble peeling.
    pub fn optimize_trace_with_constants_and_inputs(
        &mut self,
        ops: &[Op],
        constants: &mut std::collections::HashMap<u32, i64>,
        num_inputs: usize,
    ) -> (Vec<Op>, usize) {
        self.optimize_trace_with_constants_and_inputs_vable(ops, constants, num_inputs, None)
    }

    /// compile.py:275-308: compile_loop — 2-phase preamble peeling.
    /// compile.py:275-338: 2-phase preamble peeling (RPython parity).
    ///
    /// Phase 1 (optimize_preamble): full pipeline on trace → preamble_ops.
    /// export_state: flatten virtual JUMP args → label_args.
    /// Phase 2 (optimize_peeled_loop): import_state + full pipeline → body_ops.
    /// Assembly: [preamble_no_jump] + Label(label_args) + [body_with_jump].
    pub fn optimize_trace_with_constants_and_inputs_vable(
        &mut self,
        ops: &[Op],
        constants: &mut std::collections::HashMap<u32, i64>,
        num_inputs: usize,
        vable_config: Option<crate::virtualize::VirtualizableConfig>,
    ) -> (Vec<Op>, usize) {
        // ── Phase 1: PreambleCompileData.optimize() ──
        // ── Phase 1: optimize_preamble (compile.py:275-276) ──
        let mut consts_p1 = constants.clone();
        let mut opt_p1 = match vable_config.as_ref() {
            Some(c) => crate::optimizer::Optimizer::default_pipeline_with_virtualizable(c.clone()),
            None => crate::optimizer::Optimizer::default_pipeline(),
        };
        let p1_ops = opt_p1.optimize_with_constants_and_inputs(ops, &mut consts_p1, num_inputs);
        let p1_ni = opt_p1.final_num_inputs();
        let jump_virtuals = std::mem::take(&mut opt_p1.exported_jump_virtuals);

        if jump_virtuals.is_empty() {
            *constants = consts_p1;
            let sp = crate::shortpreamble::extract_short_preamble(&p1_ops);
            if !sp.is_empty() { self.short_preamble = Some(sp); }
            return (p1_ops, p1_ni);
        }

        // ── export_state (unroll.py:452-477) ──
        let p1_jump = match p1_ops.iter().rfind(|op| op.opcode == OpCode::Jump) {
            Some(j) => j.clone(),
            None => { *constants = consts_p1; return (p1_ops, p1_ni); }
        };

        // ── Phase 2: optimize_peeled_loop (compile.py:291-292) ──
        // RPython import_state: same trace, optimizer pre-populated with
        // imported_virtual_heads. GetfieldGcR(pool) → virtual head forwarding
        // in OptVirtualize eliminates the head load and enables virtualization.
        // NO trace modification (no build_body_trace). Original trace as-is.
        let label_args = export_flatten_jump_args(&p1_ops, &p1_jump, &jump_virtuals);
        // Body needs extra inputargs for flattened virtual fields.
        // label_args.len() > num_inputs when virtuals are flattened.
        let body_num_inputs = label_args.len();

        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!(
                "[jit] preamble peeling: {} virtual(s), label_args={}",
                jump_virtuals.len(), label_args.len(),
            );
        }

        let imported = build_imported_virtuals(&jump_virtuals);

        // Remap original trace ops to avoid collisions with field inputargs
        let (remapped_ops, mut consts_p2) =
            remap_trace_for_body(ops, &consts_p1, num_inputs, body_num_inputs);

        let mut opt_p2 = match vable_config.as_ref() {
            Some(c) => crate::optimizer::Optimizer::default_pipeline_with_virtualizable(c.clone()),
            None => crate::optimizer::Optimizer::default_pipeline(),
        };
        opt_p2.imported_virtuals = imported;
        opt_p2.set_flatten_virtuals_at_jump(true);

        if std::env::var_os("MAJIT_LOG").is_some() {
            let gc_before = remapped_ops.iter().filter(|o| o.opcode.is_guard()).count();
            eprintln!("[jit] phase 2 input: {} ops, {} guards, body_ni={}", remapped_ops.len(), gc_before, body_num_inputs);
        }

        let p2_ops = opt_p2.optimize_with_constants_and_inputs(
            &remapped_ops, &mut consts_p2, body_num_inputs,
        );
        let p2_ni = opt_p2.final_num_inputs();

        if std::env::var_os("MAJIT_LOG").is_some() {
            let nc = p2_ops.iter()
                .filter(|o| o.opcode == OpCode::New || o.opcode == OpCode::NewWithVtable).count();
            let gc = p2_ops.iter().filter(|o| o.opcode.is_guard()).count();
            eprintln!("[jit] phase 2: {} ops, {} New, {} guards", p2_ops.len(), nc, gc);
        }

        // ── Assembly (compile.py:310-338) ──
        let combined = assemble_peeled_trace(&p1_ops, &p2_ops, &label_args, p2_ni);
        *constants = consts_p2;
        let sp = crate::shortpreamble::extract_short_preamble(&combined);
        if !sp.is_empty() { self.short_preamble = Some(sp); }
        (combined, p2_ni)
    }

    /// Count the guards in an optimized trace (for retrace_limit checks).
    pub fn count_guards(ops: &[Op]) -> u32 {
        ops.iter().filter(|op| op.opcode.is_guard()).count() as u32
    }

    /// unroll.py: _map_args(mapping, arglist)
    /// Remap a list of OpRefs through a forwarding mapping.
    /// Constants (OpRef >= 10000) are left unchanged.
    pub fn map_args(
        mapping: &std::collections::HashMap<OpRef, OpRef>,
        args: &[OpRef],
    ) -> Vec<OpRef> {
        args.iter()
            .map(|&arg| {
                if arg.0 >= 10_000 {
                    arg // constant, keep as-is
                } else {
                    mapping.get(&arg).copied().unwrap_or(arg)
                }
            })
            .collect()
    }

    /// unroll.py: _check_no_forwarding(lsts)
    /// Debug assertion: verify no OpRef in the lists has been forwarded.
    pub fn check_no_forwarding(ctx: &crate::OptContext, oprefs: &[OpRef]) -> bool {
        oprefs.iter().all(|&r| ctx.get_replacement(r) == r)
    }

    /// unroll.py: jump_to_preamble(cell_token, jump_op)
    /// Redirect a Jump to target the preamble instead of the loop body.
    /// Used when the virtual state doesn't match any existing target token.
    pub fn jump_to_preamble(jump_op: &Op) -> Op {
        // In RPython this changes the jump's descr to the preamble's target token.
        // Here we just return a copy of the jump op (the caller attaches the token).
        jump_op.clone()
    }

    /// unroll.py: disable_retracing_if_max_retrace_guards(ops, target_token)
    /// If the trace has too many guards, disable retracing for this location.
    /// Returns true if retracing was disabled.
    pub fn disable_retracing_if_max_retrace_guards(ops: &[Op], max_retrace_guards: u32) -> bool {
        let guard_count = Self::count_guards(ops);
        guard_count > max_retrace_guards
    }

    /// unroll.py: get_virtual_state(args)
    /// Build a VirtualState from the optimizer's current knowledge about args.
    pub fn get_virtual_state(
        args: &[OpRef],
        ctx: &crate::OptContext,
        ptr_info: &[Option<crate::info::PtrInfo>],
    ) -> crate::virtualstate::VirtualState {
        crate::virtualstate::export_state(args, ctx, ptr_info)
    }
}

impl Default for UnrollOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// unroll.py: ExportedState — snapshot of optimizer state at the end of
/// the preamble, used to initialize the peeled loop body.
///
/// Contains the virtual state, short preamble boxes, arg mappings,
/// and exported infos needed to resume optimization after peeling.
#[derive(Clone, Debug)]
pub struct ExportedState {
    /// Label args at the end of the preamble (after forcing).
    pub end_args: Vec<OpRef>,
    /// Args for the next iteration (before forcing).
    pub next_iteration_args: Vec<OpRef>,
    /// Virtual state at the loop boundary.
    pub virtual_state: crate::virtualstate::VirtualState,
    /// Short preamble builder for bridge entry.
    pub short_preamble: Option<crate::shortpreamble::ShortPreamble>,
    /// Renamed inputargs from the preamble.
    pub renamed_inputargs: Vec<OpRef>,
    /// Short inputargs for the short preamble.
    pub short_inputargs: Vec<OpRef>,
}

impl ExportedState {
    /// unroll.py: ExportedState.__init__
    pub fn new(
        end_args: Vec<OpRef>,
        next_iteration_args: Vec<OpRef>,
        virtual_state: crate::virtualstate::VirtualState,
        renamed_inputargs: Vec<OpRef>,
        short_inputargs: Vec<OpRef>,
    ) -> Self {
        ExportedState {
            end_args,
            next_iteration_args,
            virtual_state,
            short_preamble: None,
            renamed_inputargs,
            short_inputargs,
        }
    }

    /// unroll.py: final() — ExportedState is never final (loop continues).
    pub fn is_final(&self) -> bool {
        false
    }
}

impl OptUnroll {
    /// unroll.py: export_state — capture optimizer state at end of preamble.
    ///
    /// After the preamble is optimized, snapshot:
    /// - end_args: forced versions of label args
    /// - virtual_state: abstract info for loop-carried values
    /// - short boxes: mapping of preamble ops to label args
    pub fn export_state(
        &self,
        original_label_args: &[OpRef],
        renamed_inputargs: &[OpRef],
        ctx: &OptContext,
    ) -> ExportedState {
        let end_args: Vec<OpRef> = original_label_args
            .iter()
            .map(|&a| ctx.get_replacement(a))
            .collect();

        let virtual_state = crate::virtualstate::VirtualState::new(
            end_args
                .iter()
                .map(|_| crate::virtualstate::VirtualStateInfo::Unknown)
                .collect(),
        );

        let label_args = virtual_state.make_inputargs(&end_args);

        ExportedState::new(
            label_args.clone(),
            end_args,
            virtual_state,
            renamed_inputargs.to_vec(),
            label_args,
        )
    }

    /// unroll.py: inline_short_preamble — replay short preamble ops
    /// to re-populate the optimizer's cache when entering from a bridge.
    ///
    /// Maps short preamble input args to the jump args, then emits
    /// each short preamble op with remapped arguments.
    pub fn inline_short_preamble(
        jump_args: &[OpRef],
        short_preamble: &crate::shortpreamble::ShortPreamble,
        ctx: &mut OptContext,
    ) -> Vec<OpRef> {
        let mut mapping: HashMap<OpRef, OpRef> = HashMap::new();

        // Map short preamble input args to jump args
        // (short_preamble.ops[i].arg_mapping tells us which args to remap)
        for sp_op in &short_preamble.ops {
            let mut new_op = sp_op.op.clone();
            // Remap args from arg_mapping (label idx → jump arg)
            for &(arg_pos, label_idx) in &sp_op.arg_mapping {
                if arg_pos < new_op.args.len() && label_idx < jump_args.len() {
                    new_op.args[arg_pos] = jump_args[label_idx];
                }
            }
            // Remap remaining args from the mapping table
            for arg in &mut new_op.args {
                if let Some(&mapped) = mapping.get(arg) {
                    *arg = mapped;
                }
            }
            let new_ref = ctx.emit(new_op.clone());
            mapping.insert(sp_op.op.pos, new_ref);
        }

        // Return remapped jump args
        jump_args
            .iter()
            .map(|&a| *mapping.get(&a).unwrap_or(&a))
            .collect()
    }

    /// unroll.py: jump_to_existing_trace — check if a compiled trace
    /// exists for this loop and redirect the jump to it.
    pub fn jump_to_existing_trace(jump_op: &Op, exported_state: &ExportedState) -> Option<Op> {
        // Check if virtual states are compatible
        // (simplified: just check arg count matches)
        if jump_op.args.len() != exported_state.end_args.len() {
            return None;
        }
        Some(jump_op.clone())
    }

    /// unroll.py: import_state — restore optimizer state for peeled loop.
    ///
    /// Maps target args (from the new label) to the exported state's
    /// next_iteration_args, carrying forward type info and virtuals.
    pub fn import_state(
        &self,
        targetargs: &[OpRef],
        exported_state: &ExportedState,
        ctx: &mut OptContext,
    ) -> Vec<OpRef> {
        assert_eq!(
            exported_state.next_iteration_args.len(),
            targetargs.len(),
            "import_state: arg count mismatch"
        );

        // Map each source arg to its target via forwarding
        for (i, &target) in exported_state.next_iteration_args.iter().enumerate() {
            let source = targetargs[i];
            if source != target {
                ctx.replace_op(source, target);
            }
        }

        // Create label args from virtual state
        exported_state.virtual_state.make_inputargs(targetargs)
    }
}

/// unroll.py: pick_virtual_state(my_vs, label_vs, target_tokens)
///
/// Given the current virtual state and available target tokens,
/// find a compatible target to jump to. Returns the target index
/// or None if no match.
/// RPython unroll.py: import_state + _generate_virtual.
///
// ── RPython-parity helper functions for 2-phase preamble peeling ──

/// unroll.py:452-477 export_state + make_inputargs_and_virtuals:
/// Replace virtual ptr in Phase 1 JUMP args with field values from SetfieldGc.
fn export_flatten_jump_args(
    p1_ops: &[Op],
    p1_jump: &Op,
    jump_virtuals: &[crate::optimizer::ExportedJumpVirtual],
) -> Vec<OpRef> {
    let mut args = p1_jump.args.clone();
    for virt in jump_virtuals.iter().rev() {
        if virt.jump_arg_index >= args.len() { continue; }
        let vref = args[virt.jump_arg_index];
        let mut fvals = Vec::new();
        for (descr, _) in &virt.fields {
            let fv = p1_ops.iter()
                .find(|op| op.opcode == OpCode::SetfieldGc
                    && op.args.first() == Some(&vref)
                    && op.descr.as_ref().map_or(false, |d| d.index() == descr.index()))
                .and_then(|op| op.args.get(1).copied())
                .unwrap_or(OpRef::NONE);
            fvals.push(fv);
        }
        args.remove(virt.jump_arg_index);
        for (i, fv) in fvals.into_iter().enumerate() {
            args.insert(virt.jump_arg_index + i, fv);
        }
    }
    args.into_vec()
}

/// unroll.py:479-504 import_state: build ImportedVirtual for Phase 2.
fn build_imported_virtuals(
    jump_virtuals: &[crate::optimizer::ExportedJumpVirtual],
) -> Vec<crate::optimizer::ImportedVirtual> {
    let mut imported = Vec::new();
    let mut offset = 0;
    for virt in jump_virtuals {
        let base = virt.jump_arg_index + offset;
        let fields: Vec<_> = virt.fields.iter().enumerate()
            .map(|(i, (descr, _))| (descr.clone(), base + i))
            .collect();
        imported.push(crate::optimizer::ImportedVirtual {
            inputarg_index: base,
            size_descr: virt.size_descr.clone(),
            fields,
            head_load_descr_index: virt.head_load_descr_index,
        });
        offset += virt.fields.len() - 1;
    }
    imported
}

/// Remap original trace op positions that collide with body's extra inputargs.
fn remap_trace_for_body(
    ops: &[Op],
    constants: &std::collections::HashMap<u32, i64>,
    orig_num_inputs: usize,
    body_num_inputs: usize,
) -> (Vec<Op>, std::collections::HashMap<u32, i64>) {
    let extra = body_num_inputs.saturating_sub(orig_num_inputs) as u32;
    if extra == 0 {
        return (ops.to_vec(), constants.clone());
    }
    let mut remapped = ops.to_vec();
    let mut remap: HashMap<OpRef, OpRef> = HashMap::new();
    for op in &mut remapped {
        if op.pos.0 != u32::MAX
            && op.pos.0 >= orig_num_inputs as u32
            && op.pos.0 < orig_num_inputs as u32 + extra
        {
            let new_pos = OpRef(op.pos.0 + extra);
            remap.insert(op.pos, new_pos);
            op.pos = new_pos;
        }
    }
    for op in &mut remapped {
        for arg in &mut op.args {
            if let Some(&r) = remap.get(arg) { *arg = r; }
        }
        if let Some(ref mut fa) = op.fail_args {
            for a in fa.iter_mut() {
                if let Some(&r) = remap.get(a) { *a = r; }
            }
        }
    }
    let mut new_consts = std::collections::HashMap::new();
    for (&k, &v) in constants {
        let nk = remap.get(&OpRef(k)).map_or(k, |r| r.0);
        new_consts.insert(nk, v);
    }
    (remapped, new_consts)
}

/// compile.py:310-338: [preamble_no_jump] + Label(label_args) + [body_with_jump]
fn assemble_peeled_trace(
    p1_ops: &[Op],
    p2_ops: &[Op],
    label_args: &[OpRef],
    body_num_inputs: usize,
) -> Vec<Op> {
    let mut result = Vec::with_capacity(p1_ops.len() + p2_ops.len() + 1);

    // Preamble: everything except Jump
    for op in p1_ops {
        if op.opcode == OpCode::Jump { break; }
        result.push(op.clone());
    }

    // Label position
    let max_pos = result.iter()
        .map(|op| op.pos.0).filter(|&p| p != u32::MAX).max().unwrap_or(0);
    let label_pos = (max_pos + 1).max(body_num_inputs as u32);
    let mut label_op = Op::new(OpCode::Label, label_args);
    label_op.pos = OpRef(label_pos);
    result.push(label_op);

    // Body: 2-pass remap (inputarg refs → label_args, op positions → after label)
    let body_base = label_pos + 1;
    let mut remap: HashMap<OpRef, OpRef> = HashMap::new();

    // Pass 1: collect all remappings
    for (i, &la) in label_args.iter().enumerate() {
        if (i as u32) < body_num_inputs as u32 {
            remap.insert(OpRef(i as u32), la);
        }
    }
    for (idx, op) in p2_ops.iter().enumerate() {
        if op.pos.0 != u32::MAX {
            remap.insert(op.pos, OpRef(body_base + idx as u32));
        }
    }

    // Pass 2: apply
    for (idx, op) in p2_ops.iter().enumerate() {
        let mut new_op = op.clone();
        if new_op.pos.0 != u32::MAX {
            new_op.pos = OpRef(body_base + idx as u32);
        }
        for arg in &mut new_op.args {
            if let Some(&m) = remap.get(arg) { *arg = m; }
        }
        // Jump args: if a remapped arg doesn't point to a label_arg or
        // a body-produced op, fall back to the corresponding label slot.
        // This handles Phase 2 flattened field values that may reference
        // stale pre-remap OpRefs.
        if new_op.opcode == OpCode::Jump {
            let body_positions: std::collections::HashSet<OpRef> = (0..p2_ops.len() as u32)
                .map(|i| OpRef(body_base + i))
                .collect();
            let label_set: std::collections::HashSet<OpRef> = label_args.iter().copied().collect();
            for (i, arg) in new_op.args.iter_mut().enumerate() {
                if i < label_args.len()
                    && !label_set.contains(arg)
                    && !body_positions.contains(arg)
                {
                    *arg = label_args[i];
                }
            }
        }
        if let Some(ref mut fa) = new_op.fail_args {
            for a in fa.iter_mut() {
                if let Some(&m) = remap.get(a) { *a = m; }
            }
        }
        result.push(new_op);
    }

    result
}

// ── Legacy helpers (kept for non-peeling paths) ──

/// Build a modified trace for Phase 2 optimization by prepending
/// New+SetfieldGc reconstruction ops that rebuild virtual objects
/// from flattened field inputargs (unused — kept for reference).
#[allow(dead_code)]
fn build_body_trace_with_virtual_import(
    ops: &[Op],
    jump_virtuals: &[crate::optimizer::ExportedJumpVirtual],
    num_inputs: usize,
) -> Option<(Vec<Op>, usize)> {
    // Find the Jump op
    let jump_idx = ops.iter().rposition(|op| op.opcode == OpCode::Jump)?;
    let jump_op = &ops[jump_idx];

    let mut next_pos = num_inputs as u32;
    for op in ops {
        if op.pos.0 != u32::MAX {
            next_pos = next_pos.max(op.pos.0 + 1);
        }
    }

    let mut expanded_input_index = 0usize;
    let mut virtual_to_reconstructed: HashMap<usize, (OpRef, Vec<OpRef>)> = HashMap::new();
    let mut inputarg_replacement: HashMap<OpRef, OpRef> = HashMap::new();

    let mut jump_virtual_map = HashMap::new();
    for virt in jump_virtuals {
        jump_virtual_map.insert(virt.jump_arg_index, virt);
    }

    for arg_idx in 0..num_inputs {
        if let Some(virt) = jump_virtual_map.get(&arg_idx) {
            let mut field_inputargs = Vec::new();
            for _ in &virt.fields {
                let field_ref = OpRef(expanded_input_index as u32);
                field_inputargs.push(field_ref);
                expanded_input_index += 1;
            }
            virtual_to_reconstructed
                .insert(arg_idx, (OpRef(next_pos), field_inputargs));
            inputarg_replacement.insert(OpRef(arg_idx as u32), OpRef(next_pos));
            next_pos += 1;
        } else {
            inputarg_replacement.insert(OpRef(arg_idx as u32), OpRef(expanded_input_index as u32));
            expanded_input_index += 1;
        }
    }

    if virtual_to_reconstructed.is_empty() {
        return None;
    }

    let body_num_inputs = expanded_input_index;
    let mut result =
        Vec::with_capacity(ops.len() + jump_virtuals.len() * 4);

    // Emit New+SetfieldGc reconstruction at trace start (before original ops)
    for virt in jump_virtuals {
        if let Some((new_ref, field_refs)) = virtual_to_reconstructed.get(&virt.jump_arg_index)
        {
            let mut new_op = Op::new(OpCode::New, &[]);
            new_op.pos = *new_ref;
            new_op.descr = Some(virt.size_descr.clone());
            result.push(new_op);

            for (i, (descr, _concrete)) in virt.fields.iter().enumerate() {
                let mut set_op = Op::new(OpCode::SetfieldGc, &[*new_ref, field_refs[i]]);
                set_op.pos = OpRef(next_pos);
                next_pos += 1;
                set_op.descr = Some(descr.clone());
                result.push(set_op);
            }
        }
    }

    // Original ops with virtual refs replaced by reconstructed refs
    for (idx, op) in ops.iter().enumerate() {
        let mut new_op = op.clone();
        for arg in &mut new_op.args {
            if let Some(&repl) = inputarg_replacement.get(arg) {
                *arg = repl;
            }
        }
        if let Some(ref mut fa) = new_op.fail_args {
            for arg in fa.iter_mut() {
                if let Some(&repl) = inputarg_replacement.get(arg) {
                    *arg = repl;
                }
            }
        }
        result.push(new_op);
    }

    Some((result, body_num_inputs))
}

/// compile.py:310-338: combine preamble + Label + body into final trace.
/// compile.py:310-338: combine preamble + Label + body.
///
/// RPython parity: the preamble JUMP is extended with virtual field values
/// so its args match the body's expected inputargs (including flattened fields).
/// Body ops are remapped so inputarg refs point to the Label args.
fn combine_preamble_and_body(
    preamble: &[Op],
    body: &[Op],
    _preamble_num_inputs: usize,
    body_num_inputs: usize,
    jump_virtuals: &[crate::optimizer::ExportedJumpVirtual],
) -> Vec<Op> {
    let mut result = Vec::with_capacity(preamble.len() + body.len() + 1);

    // Preamble ops (Phase 1 result, up to but not including Jump)
    for op in preamble {
        if op.opcode == OpCode::Jump {
            break;
        }
        result.push(op.clone());
    }

    let preamble_jump = preamble.iter().rfind(|op| op.opcode == OpCode::Jump);
    let Some(jump) = preamble_jump else {
        result.extend_from_slice(body);
        return result;
    };

    let mut label_args = Vec::new();
    let mut virt_by_arg = HashMap::new();
    for virt in jump_virtuals {
        virt_by_arg.insert(virt.jump_arg_index, virt);
    }
    for (arg_idx, &arg) in jump.args.iter().enumerate() {
        if let Some(virt) = virt_by_arg.get(&arg_idx) {
            let mut field_vals = Vec::new();
            for (descr, _concrete) in &virt.fields {
                let field_val = preamble
                    .iter()
                    .find(|op| {
                        op.opcode == OpCode::SetfieldGc
                            && op.args.first() == Some(&arg)
                            && op.descr.as_ref().map_or(false, |d| d.index() == descr.index())
                    })
                    .and_then(|op| op.args.get(1).copied())
                    .unwrap_or(OpRef::NONE);
                label_args.push(field_val);
                field_vals.push(field_val);
            }
        } else {
            label_args.push(arg);
        }
    }

    let label_pos = OpRef(result.len() as u32 + body_num_inputs as u32);
    let mut label = Op::new(OpCode::Label, &label_args);
    label.pos = label_pos;
    result.push(label);

    let mut body_remap: HashMap<OpRef, OpRef> = HashMap::new();
    for (i, &label_arg) in label_args.iter().enumerate() {
        if i < body_num_inputs {
            body_remap.insert(OpRef(i as u32), label_arg);
        }
    }

    let body_ops: Vec<&Op> = body.iter().filter(|op| op.opcode != OpCode::Label).collect();
    let body_base = result.len() as u32 + body_num_inputs as u32 + 1;
    for (idx, op) in body_ops.iter().enumerate() {
        if op.pos.0 != u32::MAX {
            let new_pos = OpRef(body_base + idx as u32);
            body_remap.insert(op.pos, new_pos);
        }
    }

    for (idx, op) in body_ops.iter().enumerate() {
        let mut new_op = (*op).clone();
        if new_op.pos.0 != u32::MAX {
            new_op.pos = OpRef(body_base + idx as u32);
        }
        for arg in &mut new_op.args {
            if let Some(&mapped) = body_remap.get(arg) {
                *arg = mapped;
            }
        }
        if let Some(ref mut fa) = new_op.fail_args {
            for arg in fa.iter_mut() {
                if let Some(&mapped) = body_remap.get(arg) {
                    *arg = mapped;
                }
            }
        }
        // Remap descr fail_arg_types if present (for guard descriptors)
        result.push(new_op);
    }

    result
}

/// RPython unroll.py: make_inputargs_and_virtuals + _generate_virtual.
///
/// Transform the Phase 1 peeled trace to flatten virtual JUMP args into
/// their constituent field values, and insert reconstruction ops at body start.
///
/// Input:  Phase 1 optimized peeled trace with materialized virtuals at JUMP
/// Output: Modified trace where:
///   - JUMP carries field VALUES instead of virtual pointer
///   - LABEL has extra field-value inputargs
///   - Body start has New+SetfieldGc that reconstruct the virtual
///   - Body refs to old virtual replaced with reconstructed ref
///
/// Phase 2 optimizer will then virtualize the reconstruction ops.
fn flatten_virtuals_in_peeled_trace(
    ops: &[Op],
    jump_virtuals: &[crate::optimizer::ExportedJumpVirtual],
    num_inputs: usize,
) -> Option<(Vec<Op>, usize)> {
    let label_idx = ops.iter().position(|op| op.opcode == OpCode::Label)?;
    let jump_idx = ops.iter().rposition(|op| op.opcode == OpCode::Jump)?;
    if label_idx >= jump_idx {
        return None;
    }

    let label_op = &ops[label_idx];
    let jump_op = &ops[jump_idx];

    // Map each virtual JUMP arg → (New position, [SetfieldGc field value OpRefs])
    let mut virtual_info: HashMap<usize, (OpRef, Vec<(majit_ir::DescrRef, OpRef)>)> =
        HashMap::new();
    let mut new_positions: std::collections::HashSet<OpRef> = std::collections::HashSet::new();

    for virt in jump_virtuals {
        if virt.jump_arg_index >= jump_op.args.len() {
            continue;
        }
        let jump_arg = jump_op.args[virt.jump_arg_index];

        // Find New op that produced this JUMP arg
        let new_op = ops[label_idx + 1..jump_idx]
            .iter()
            .find(|op| op.pos == jump_arg && op.opcode == OpCode::New)?;
        new_positions.insert(new_op.pos);

        // Collect SetfieldGc field values for this New
        let mut field_values = Vec::new();
        for op in &ops[label_idx + 1..jump_idx] {
            if op.opcode == OpCode::SetfieldGc
                && op.args.first() == Some(&new_op.pos)
                && op.args.len() >= 2
            {
                if let Some(descr) = &op.descr {
                    field_values.push((descr.clone(), op.args[1]));
                }
            }
        }
        if field_values.is_empty() {
            continue;
        }
        virtual_info.insert(virt.jump_arg_index, (new_op.pos, field_values));
    }

    if virtual_info.is_empty() {
        return None;
    }

    // Compute next available OpRef position
    let mut next_pos = num_inputs as u32;
    for op in ops.iter() {
        if op.pos.0 != u32::MAX {
            next_pos = next_pos.max(op.pos.0 + 1);
        }
    }

    let mut result = Vec::with_capacity(ops.len() + virtual_info.len() * 4);
    let mut virtual_to_reconstructed: HashMap<OpRef, OpRef> = HashMap::new();
    let label_orig_len = label_op.args.len();

    // ── Preamble ops (before Label) — unchanged ──
    for op in &ops[..label_idx] {
        result.push(op.clone());
    }

    // ── Modified Label: add extra field-value inputargs ──
    let mut new_label = label_op.clone();
    let mut extra_field_count = 0;
    // Track: for each virtual, which extra Label arg indices hold its fields
    let mut virt_field_label_indices: HashMap<usize, Vec<usize>> = HashMap::new();

    for (&virt_idx, (_new_pos, fields)) in &virtual_info {
        let mut indices = Vec::new();
        for _ in fields {
            let idx = label_orig_len + extra_field_count;
            new_label.args.push(OpRef(idx as u32));
            indices.push(idx);
            extra_field_count += 1;
        }
        virt_field_label_indices.insert(virt_idx, indices);
    }
    result.push(new_label);

    // ── Reconstruction: New+SetfieldGc after Label ──
    // RPython unroll.py: _generate_virtual — rebuild virtual from flattened fields
    for (&virt_idx, (_new_pos, fields)) in &virtual_info {
        let original_label_arg = label_op.args.get(virt_idx).copied()?;

        // Find size_descr from the original New op
        let orig_new = ops[label_idx + 1..jump_idx]
            .iter()
            .find(|op| op.pos == *_new_pos)?;
        let size_descr = orig_new.descr.clone()?;

        // Emit New
        let reconstructed_ref = OpRef(next_pos);
        next_pos += 1;
        let mut new_op = Op::new(OpCode::New, &[]);
        new_op.pos = reconstructed_ref;
        new_op.descr = Some(size_descr);
        result.push(new_op);

        // Emit SetfieldGc for each field, using the extra Label args
        let field_indices = virt_field_label_indices.get(&virt_idx)?;
        for (i, (descr, _orig_value)) in fields.iter().enumerate() {
            let field_input_ref = OpRef(field_indices[i] as u32);
            let mut set_op = Op::new(OpCode::SetfieldGc, &[reconstructed_ref, field_input_ref]);
            set_op.pos = OpRef(next_pos);
            next_pos += 1;
            set_op.descr = Some(descr.clone());
            result.push(set_op);
        }

        virtual_to_reconstructed.insert(original_label_arg, reconstructed_ref);
    }

    // ── Body ops: replace refs to old virtual with reconstructed ──
    for op in &ops[label_idx + 1..jump_idx] {
        // Skip the original New+SetfieldGc for the virtual (now reconstructed above)
        if new_positions.contains(&op.pos) {
            continue;
        }
        if op.opcode == OpCode::SetfieldGc {
            if let Some(&first) = op.args.first() {
                if new_positions.contains(&first) {
                    continue;
                }
            }
        }

        let mut new_op = op.clone();
        for arg in &mut new_op.args {
            if let Some(&r) = virtual_to_reconstructed.get(arg) {
                *arg = r;
            }
        }
        if let Some(ref mut fa) = new_op.fail_args {
            for arg in fa.iter_mut() {
                if let Some(&r) = virtual_to_reconstructed.get(arg) {
                    *arg = r;
                }
            }
        }
        result.push(new_op);
    }

    // ── Modified Jump: replace virtual ref with field values ──
    let mut new_jump = jump_op.clone();
    // Replace virtual args with placeholder and add field values as extra args
    for (&virt_idx, (_new_pos, fields)) in &virtual_info {
        for (_descr, value_ref) in fields {
            new_jump.args.push(*value_ref);
        }
    }
    new_jump.pos = OpRef(next_pos);
    result.push(new_jump);

    let new_num_inputs = num_inputs + extra_field_count;
    Some((result, new_num_inputs))
}

pub fn pick_virtual_state(
    my_vs: &crate::virtualstate::VirtualState,
    target_states: &[crate::virtualstate::VirtualState],
) -> Option<usize> {
    for (i, target_vs) in target_states.iter().enumerate() {
        if my_vs.is_compatible(target_vs) {
            return Some(i);
        }
    }
    None
}

/// unroll.py: _jump_to_existing_trace(jump_op, label_op, runtime_boxes)
///
/// Check if any existing compiled trace (target_token) has a compatible
/// virtual state. If so, generate extra guards, inline the short preamble,
/// and redirect the jump.
///
/// Returns Some(ops_to_emit) if a match was found, None otherwise.
pub fn jump_to_existing_trace_full(
    jump_args: &[OpRef],
    target_states: &[crate::virtualstate::VirtualState],
    target_short_preambles: &[Option<crate::shortpreamble::ShortPreamble>],
    ctx: &mut OptContext,
) -> Option<(usize, Vec<OpRef>)> {
    let my_vs = crate::virtualstate::VirtualState::new(
        jump_args
            .iter()
            .map(|_| crate::virtualstate::VirtualStateInfo::Unknown)
            .collect(),
    );

    for (i, target_vs) in target_states.iter().enumerate() {
        if !my_vs.is_compatible(target_vs) {
            continue;
        }
        // Generate extra guards from virtual state mismatch
        let _extra_guards = target_vs.generate_guards(&my_vs);

        // Make input args matching the target's virtual state
        let args = target_vs.make_inputargs(jump_args);

        // Inline short preamble if available
        let final_args = if let Some(Some(sp)) = target_short_preambles.get(i) {
            OptUnroll::inline_short_preamble(&args, sp, ctx)
        } else {
            args.clone()
        };

        return Some((i, final_args));
    }
    None
}

pub struct OptUnroll {
    /// Buffer of ops received before the Jump back-edge.
    buffer: Vec<Op>,
    /// Whether a Jump was already seen (avoid double-unrolling).
    seen_jump: bool,
}

impl OptUnroll {
    pub fn new() -> Self {
        OptUnroll {
            buffer: Vec::new(),
            seen_jump: false,
        }
    }

    /// Peel one iteration of the loop body.
    ///
    /// Given the buffered ops (everything before the Jump), this emits:
    /// 1. The peeled (duplicated) body with remapped OpRefs
    /// 2. A Label op marking the loop header
    /// 3. The original body ops
    ///
    /// The caller is responsible for emitting the final Jump.
    fn peel_iteration(&self, jump_op: &Op, ctx: &mut OptContext) {
        if self.buffer.is_empty() {
            return;
        }

        // Build the OpRef remapping for peeled ops.
        // Each op in the buffer gets a new position in the peeled iteration.
        // The offset is: original positions map to new sequential positions
        // starting from the current emission point.
        let peel_base = ctx.new_operations.len() as u32;
        let mut ref_map: HashMap<OpRef, OpRef> = HashMap::new();

        // First pass: assign new positions for peeled ops.
        for (i, op) in self.buffer.iter().enumerate() {
            let new_pos = OpRef(peel_base + i as u32);
            ref_map.insert(op.pos, new_pos);
        }

        // Emit peeled iteration with remapped refs.
        for (i, op) in self.buffer.iter().enumerate() {
            let mut peeled = op.clone();
            peeled.pos = OpRef(peel_base + i as u32);

            // Remap argument references.
            for arg in &mut peeled.args {
                if let Some(&new_ref) = ref_map.get(arg) {
                    *arg = new_ref;
                }
                // Args referencing ops outside the buffer (e.g., input args)
                // are kept as-is.
            }

            // Remap fail_args references.
            if let Some(ref mut fa) = peeled.fail_args {
                for arg in fa.iter_mut() {
                    if let Some(&new_ref) = ref_map.get(arg) {
                        *arg = new_ref;
                    }
                }
            }

            ctx.emit(peeled);
        }

        // Emit Label between peeled and original body.
        // The Label's args match the Jump's args, forming the loop header.
        let label_pos = OpRef(peel_base + self.buffer.len() as u32);
        let mut label_op = Op::new(OpCode::Label, &jump_op.args);
        label_op.pos = label_pos;
        ctx.emit(label_op);

        // Emit original body ops with updated positions.
        let body_base = peel_base + self.buffer.len() as u32 + 1; // +1 for Label
        let mut orig_ref_map: HashMap<OpRef, OpRef> = HashMap::new();

        for (i, op) in self.buffer.iter().enumerate() {
            orig_ref_map.insert(op.pos, OpRef(body_base + i as u32));
        }

        for (i, op) in self.buffer.iter().enumerate() {
            let mut body_op = op.clone();
            body_op.pos = OpRef(body_base + i as u32);

            // Remap argument references within the body.
            for arg in &mut body_op.args {
                if let Some(&new_ref) = orig_ref_map.get(arg) {
                    *arg = new_ref;
                }
            }

            if let Some(ref mut fa) = body_op.fail_args {
                for arg in fa.iter_mut() {
                    if let Some(&new_ref) = orig_ref_map.get(arg) {
                        *arg = new_ref;
                    }
                }
            }

            ctx.emit(body_op);
        }
    }
}

impl Default for OptUnroll {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimization for OptUnroll {
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        // Only peel once per trace, and only for Jump (back-edge).
        if op.opcode == OpCode::Jump && !self.seen_jump {
            self.seen_jump = true;

            if self.buffer.is_empty() {
                // Empty loop body, nothing to peel.
                return OptimizationResult::Emit(op.clone());
            }

            // Perform the peeling: emit peeled body + Label + original body.
            self.peel_iteration(op, ctx);

            // Emit the final Jump with remapped args pointing to the
            // original body's ops.
            let body_base = ctx.new_operations.len() as u32 - self.buffer.len() as u32;
            let mut jump = op.clone();

            // Build ref map for Jump args (same as orig_ref_map).
            let mut orig_ref_map: HashMap<OpRef, OpRef> = HashMap::new();
            for (i, buffered_op) in self.buffer.iter().enumerate() {
                orig_ref_map.insert(buffered_op.pos, OpRef(body_base + i as u32));
            }

            for arg in &mut jump.args {
                if let Some(&new_ref) = orig_ref_map.get(arg) {
                    *arg = new_ref;
                }
            }

            jump.pos = OpRef(ctx.new_operations.len() as u32);
            return OptimizationResult::Emit(jump);
        }

        // For non-Jump ops (or after we've already unrolled), buffer them.
        if !self.seen_jump {
            self.buffer.push(op.clone());
            return OptimizationResult::Remove;
        }

        // After unrolling, pass everything through.
        OptimizationResult::Emit(op.clone())
    }

    fn setup(&mut self) {
        self.buffer.clear();
        self.seen_jump = false;
    }

    fn name(&self) -> &'static str {
        "unroll"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::Optimizer;

    /// Assign sequential positions to ops starting from `base`.
    fn assign_positions(ops: &mut [Op], base: u32) {
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef(base + i as u32);
        }
    }

    fn run_unroll_pass(ops: &[Op]) -> Vec<Op> {
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptUnroll::new()));
        opt.optimize(ops)
    }

    // ── Basic peeling ─────────────────────────────────────────────────

    #[test]
    fn test_no_jump_no_unroll() {
        // Without a Jump back-edge, the pass just buffers and nothing is emitted.
        // (In practice, traces always end with Jump or Finish.)
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Finish, &[OpRef(0)]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // IntAdd gets buffered, Finish is not a Jump so it gets buffered too.
        // Nothing is emitted because there's no Jump to trigger peeling.
        // The buffered ops are lost (which is correct: no loop = no unrolling).
        assert!(
            result.is_empty(),
            "no Jump means no loop to unroll, ops are buffered but never emitted"
        );
    }

    #[test]
    fn test_empty_loop_body() {
        // Jump with no prior ops: nothing to peel.
        let mut ops = vec![Op::new(OpCode::Jump, &[])];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::Jump);
    }

    #[test]
    fn test_simple_loop_peeled() {
        // A simple loop: one add op, then Jump.
        // Expected output: peeled_add, Label, original_add, Jump
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        assert_eq!(
            result.len(),
            4,
            "expected: peeled_add, Label, original_add, Jump"
        );
        assert_eq!(result[0].opcode, OpCode::IntAdd); // peeled
        assert_eq!(result[1].opcode, OpCode::Label);
        assert_eq!(result[2].opcode, OpCode::IntAdd); // original body
        assert_eq!(result[3].opcode, OpCode::Jump);
    }

    #[test]
    fn test_peeled_ops_have_different_positions() {
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::IntSub, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // 2 peeled + Label + 2 original + Jump = 6
        assert_eq!(result.len(), 6);

        // All positions should be unique.
        let positions: Vec<OpRef> = result.iter().map(|op| op.pos).collect();
        for (i, pos) in positions.iter().enumerate() {
            for (j, other) in positions.iter().enumerate() {
                if i != j {
                    assert_ne!(
                        pos, other,
                        "positions at index {} and {} should differ",
                        i, j
                    );
                }
            }
        }
    }

    // ── OpRef remapping ───────────────────────────────────────────────

    #[test]
    fn test_internal_refs_remapped_in_peeled_copy() {
        // op0: v0 = IntAdd(v100, v101)  -- uses input args
        // op1: v1 = IntMul(v0, v101)    -- uses result of op0
        // Jump()
        //
        // After peeling:
        // peeled_v0 = IntAdd(v100, v101)     -- input refs unchanged
        // peeled_v1 = IntMul(peeled_v0, v101) -- v0 remapped to peeled_v0
        // Label()
        // body_v0 = IntAdd(v100, v101)
        // body_v1 = IntMul(body_v0, v101)    -- v0 remapped to body_v0
        // Jump()
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::IntMul, &[OpRef(0), OpRef(101)]), // references op0
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        assert_eq!(result.len(), 6); // 2 peeled + Label + 2 body + Jump

        // Peeled iteration:
        let peeled_add = &result[0];
        let peeled_mul = &result[1];
        assert_eq!(peeled_add.opcode, OpCode::IntAdd);
        assert_eq!(peeled_mul.opcode, OpCode::IntMul);
        // peeled_mul should reference peeled_add's position, not original op0.
        assert_eq!(peeled_mul.args[0], peeled_add.pos);
        // Second arg (input ref) should be unchanged.
        assert_eq!(peeled_mul.args[1], OpRef(101));

        // Original body:
        let body_add = &result[3];
        let body_mul = &result[4];
        assert_eq!(body_add.opcode, OpCode::IntAdd);
        assert_eq!(body_mul.opcode, OpCode::IntMul);
        // body_mul should reference body_add's position.
        assert_eq!(body_mul.args[0], body_add.pos);
        assert_eq!(body_mul.args[1], OpRef(101));
    }

    #[test]
    fn test_external_refs_preserved() {
        // Refs to ops outside the buffer (input arguments) should not be remapped.
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // Peeled add should still reference v100 and v101.
        assert_eq!(result[0].args[0], OpRef(100));
        assert_eq!(result[0].args[1], OpRef(101));

        // Body add should also reference v100 and v101.
        assert_eq!(result[2].args[0], OpRef(100));
        assert_eq!(result[2].args[1], OpRef(101));
    }

    // ── Guard preservation ────────────────────────────────────────────

    #[test]
    fn test_guards_duplicated_in_peel() {
        // Guards in the preamble serve as type checks.
        let mut ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef(100)]),
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // peeled_guard, peeled_add, Label, body_guard, body_add, Jump
        assert_eq!(result.len(), 6);

        let guard_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardTrue)
            .count();
        assert_eq!(
            guard_count, 2,
            "guard should appear in both peeled and body"
        );
    }

    #[test]
    fn test_guard_fail_args_remapped() {
        // Guards with fail_args should have those refs remapped too.
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            {
                let mut guard = Op::new(OpCode::GuardTrue, &[OpRef(100)]);
                guard.fail_args = Some(vec![OpRef(0)].into()); // refs op0
                guard
            },
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // Check peeled guard's fail_args.
        let peeled_guard = &result[1];
        assert_eq!(peeled_guard.opcode, OpCode::GuardTrue);
        let peeled_add_pos = result[0].pos;
        assert_eq!(
            peeled_guard.fail_args.as_ref().unwrap()[0],
            peeled_add_pos,
            "peeled guard's fail_args should reference peeled add"
        );

        // Check body guard's fail_args.
        let body_guard = &result[4]; // after Label (idx 3) and body_add (idx 3)
        assert_eq!(body_guard.opcode, OpCode::GuardTrue);
        let body_add_pos = result[3].pos;
        assert_eq!(
            body_guard.fail_args.as_ref().unwrap()[0],
            body_add_pos,
            "body guard's fail_args should reference body add"
        );
    }

    // ── Jump args remapping ───────────────────────────────────────────

    #[test]
    fn test_jump_args_remapped_to_body() {
        // Jump args should reference the body's ops, not the original positions.
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Jump, &[OpRef(0)]), // carries v0 (the add result)
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // peeled_add, Label, body_add, Jump
        assert_eq!(result.len(), 4);

        let jump = result.last().unwrap();
        assert_eq!(jump.opcode, OpCode::Jump);

        let body_add_pos = result[2].pos;
        assert_eq!(
            jump.args[0], body_add_pos,
            "Jump arg should reference body add, not original"
        );
    }

    #[test]
    fn test_label_args_match_jump_args() {
        // The Label should carry the same args as the Jump.
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Jump, &[OpRef(0), OpRef(100)]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        let label = result.iter().find(|o| o.opcode == OpCode::Label).unwrap();
        let jump_args = &ops.last().unwrap().args;
        assert_eq!(
            label.args.as_slice(),
            jump_args.as_slice(),
            "Label args should match original Jump args"
        );
    }

    // ── Multiple ops in loop body ─────────────────────────────────────

    #[test]
    fn test_multi_op_loop() {
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::IntSub, &[OpRef(0), OpRef(101)]),
            Op::new(OpCode::IntMul, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::GuardTrue, &[OpRef(2)]),
            Op::new(OpCode::Jump, &[OpRef(2)]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // 4 peeled + Label + 4 body + Jump = 10
        assert_eq!(result.len(), 10);

        // Verify structure: peeled body, then Label, then body, then Jump.
        assert_eq!(result[0].opcode, OpCode::IntAdd);
        assert_eq!(result[1].opcode, OpCode::IntSub);
        assert_eq!(result[2].opcode, OpCode::IntMul);
        assert_eq!(result[3].opcode, OpCode::GuardTrue);
        assert_eq!(result[4].opcode, OpCode::Label);
        assert_eq!(result[5].opcode, OpCode::IntAdd);
        assert_eq!(result[6].opcode, OpCode::IntSub);
        assert_eq!(result[7].opcode, OpCode::IntMul);
        assert_eq!(result[8].opcode, OpCode::GuardTrue);
        assert_eq!(result[9].opcode, OpCode::Jump);
    }

    // ── No double-unrolling ───────────────────────────────────────────

    #[test]
    fn test_second_jump_not_unrolled() {
        // If there are multiple Jumps (unusual, but defensive), only the first
        // triggers peeling.
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Jump, &[]),
            Op::new(OpCode::IntSub, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // First Jump triggers peeling of the IntAdd.
        // After that, IntSub and second Jump pass through.
        let jump_count = result.iter().filter(|o| o.opcode == OpCode::Jump).count();
        assert_eq!(jump_count, 2, "both jumps should be in output");

        let label_count = result.iter().filter(|o| o.opcode == OpCode::Label).count();
        assert_eq!(label_count, 1, "only one Label from the first peeling");
    }

    // ── Setup resets state ────────────────────────────────────────────

    #[test]
    fn test_setup_resets_state() {
        let mut pass = OptUnroll::new();

        // Simulate some state.
        pass.buffer.push(Op::new(OpCode::IntAdd, &[OpRef(0)]));
        pass.seen_jump = true;

        pass.setup();

        assert!(pass.buffer.is_empty());
        assert!(!pass.seen_jump);
    }

    // ── Integration with optimizer ────────────────────────────────────

    #[test]
    fn test_unroll_standalone_optimizer() {
        // Run the unroll pass through the optimizer infrastructure.
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::GuardTrue, &[OpRef(0)]),
            Op::new(OpCode::Jump, &[OpRef(0)]),
        ];
        assign_positions(&mut ops, 0);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptUnroll::new()));
        let result = opt.optimize(&ops);

        // Expect: peeled_add, peeled_guard, Label, body_add, body_guard, Jump = 6
        assert_eq!(result.len(), 6);

        // All ops should have valid (non-NONE) positions.
        for op in &result {
            assert!(
                !op.pos.is_none(),
                "op {:?} should have a valid pos",
                op.opcode
            );
        }
    }

    #[test]
    fn test_unroll_preserves_descr() {
        // Descriptors on ops should be preserved in both copies.
        use majit_ir::{Descr, DescrRef};
        use std::sync::Arc;

        #[derive(Debug)]
        struct TestDescr(u32);
        impl Descr for TestDescr {
            fn index(&self) -> u32 {
                self.0
            }
        }

        let descr: DescrRef = Arc::new(TestDescr(42));
        let mut ops = vec![
            Op::with_descr(OpCode::GuardTrue, &[OpRef(100)], descr.clone()),
            Op::new(OpCode::Jump, &[]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // Both guards (peeled and body) should have the descriptor.
        let guards: Vec<&Op> = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardTrue)
            .collect();
        assert_eq!(guards.len(), 2);
        for guard in &guards {
            assert!(guard.descr.is_some(), "guard should have a descriptor");
            assert_eq!(guard.descr.as_ref().unwrap().index(), 42);
        }
    }

    // ── Chain of references ───────────────────────────────────────────

    #[test]
    fn test_chain_of_refs_correctly_remapped() {
        // v0 = IntAdd(v100, v101)
        // v1 = IntAdd(v0, v100)
        // v2 = IntAdd(v1, v0)
        // Jump(v2)
        //
        // In the peeled copy, all internal refs must point to peeled positions.
        // In the body copy, all internal refs must point to body positions.
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(100)]),
            Op::new(OpCode::IntAdd, &[OpRef(1), OpRef(0)]),
            Op::new(OpCode::Jump, &[OpRef(2)]),
        ];
        assign_positions(&mut ops, 0);

        let result = run_unroll_pass(&ops);

        // 3 peeled + Label + 3 body + Jump = 8
        assert_eq!(result.len(), 8);

        // Peeled iteration refs:
        let p0 = result[0].pos;
        let p1 = result[1].pos;
        let p2 = result[2].pos;
        assert_eq!(result[1].args[0], p0, "peeled v1 should ref peeled v0");
        assert_eq!(result[2].args[0], p1, "peeled v2 should ref peeled v1");
        assert_eq!(result[2].args[1], p0, "peeled v2 should ref peeled v0");

        // Body refs:
        let b0 = result[4].pos;
        let b1 = result[5].pos;
        let b2 = result[6].pos;
        assert_eq!(result[5].args[0], b0, "body v1 should ref body v0");
        assert_eq!(result[6].args[0], b1, "body v2 should ref body v1");
        assert_eq!(result[6].args[1], b0, "body v2 should ref body v0");

        // Jump should reference body v2.
        let jump = &result[7];
        assert_eq!(jump.args[0], b2, "Jump should ref body v2");
    }

    #[test]
    fn test_unroll_optimizer_optimize_trace() {
        let mut unroll_opt = UnrollOptimizer::new();
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::Jump, &[OpRef(0)]),
        ];
        assign_positions(&mut ops, 0);
        let result = unroll_opt.optimize_trace(&ops);
        // The optimizer processes the trace; result should not be empty
        assert!(!result.is_empty(), "optimize_trace should produce output");
    }

    #[test]
    fn test_unroll_optimizer_count_guards() {
        let ops = vec![
            Op::new(OpCode::GuardTrue, &[OpRef(0)]),
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::GuardNonnull, &[OpRef(0)]),
            Op::new(OpCode::Jump, &[OpRef(0)]),
        ];
        assert_eq!(UnrollOptimizer::count_guards(&ops), 2);
    }

    #[test]
    fn test_unroll_optimizer_should_give_up() {
        let opt = UnrollOptimizer::new();
        assert!(!opt.should_give_up(0));
        assert!(!opt.should_give_up(4));
        assert!(opt.should_give_up(5)); // default retrace_limit = 5
        assert!(opt.should_give_up(10));
    }

    #[test]
    fn test_unroll_optimizer_too_many_guards() {
        let mut opt = UnrollOptimizer::new();
        opt.bridge_guard_count = 20;
        assert!(opt.too_many_guards_for_retrace(0, 15));
        assert!(!opt.too_many_guards_for_retrace(0, 25));
    }
}
