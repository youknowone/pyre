/// RPython-parity preamble peeling patch.
///
/// This file contains the replacement implementation for:
///   1. UnrollOptimizer::optimize_trace_with_constants_and_inputs_vable
///   2. combine_preamble_and_body (helper)
///
/// RPython reference:
///   compile.py:275-338 — compile_loop orchestrates 2-phase
///   unroll.py:100-110  — optimize_preamble (Phase 1)
///   unroll.py:112-181  — optimize_peeled_loop (Phase 2)
///   unroll.py:452-477  — export_state
///   unroll.py:479-504  — import_state
///
/// Design:
///   Phase 1: full pipeline optimization of trace.
///     - JUMP passes through optimizer (OptVirtualize forces virtuals).
///     - export_virtual_for_preamble records virtual structure at JUMP.
///     - Result: preamble_ops including Jump with materialized virtuals.
///
///   export_state: from Phase 1 JUMP args, flatten virtual ptrs → field values.
///     - For each virtual JUMP arg, find SetfieldGc ops in preamble.
///     - Replace virtual ptr arg with field value args → label_args.
///     - label_args has same count as body's expected inputargs.
///
///   Phase 2: full pipeline optimization of SAME trace with import_state.
///     - imported_virtuals marks flattened field inputargs as VirtualStruct.
///     - flatten_virtuals_at_jump=true: body JUMP also flattens → same layout.
///     - Result: body_ops including Jump with flattened args.
///
///   Assembly: [preamble_ops_no_jump] + Label(label_args) + [body_ops_with_jump]
///     - Body ops' inputarg refs (0..body_num_inputs) remapped to label_args.
///     - Body op positions remapped to after Label.
///     - Body Jump args remapped → targets Label (loop back).
///
/// KEY INVARIANT: label_args count == body Jump args count.
///   Both use flattened layout: virtual ptrs replaced by field values.

// ── optimize_trace_with_constants_and_inputs_vable ──
//
// Replace lines 191-355 of unroll.rs with:

/*
    pub fn optimize_trace_with_constants_and_inputs_vable(
        &mut self,
        ops: &[Op],
        constants: &mut std::collections::HashMap<u32, i64>,
        num_inputs: usize,
        vable_config: Option<crate::virtualize::VirtualizableConfig>,
    ) -> (Vec<Op>, usize) {
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
        // Flatten: replace virtual ptr with field values in JUMP args → label_args.
        let p1_jump = match p1_ops.iter().rfind(|op| op.opcode == OpCode::Jump) {
            Some(j) => j.clone(),
            None => { *constants = consts_p1; return (p1_ops, p1_ni); }
        };
        let label_args = export_flatten_jump_args(&p1_ops, &p1_jump, &jump_virtuals);

        if std::env::var_os("MAJIT_LOG").is_some() {
            eprintln!("[jit] preamble peeling: {} virtual(s), label_args={}", jump_virtuals.len(), label_args.len());
        }

        // ── Phase 2: optimize_peeled_loop (compile.py:291-292) ──
        let body_num_inputs = label_args.len();

        // import_state: build ImportedVirtual for each flattened virtual
        let imported = build_imported_virtuals(&jump_virtuals, body_num_inputs, num_inputs);

        // Remap original trace ops to avoid collision with new body inputargs
        let (remapped_ops, mut consts_p2, _remap) =
            remap_trace_for_body(ops, &consts_p1, num_inputs, body_num_inputs);

        let mut opt_p2 = match vable_config.as_ref() {
            Some(c) => crate::optimizer::Optimizer::default_pipeline_with_virtualizable(c.clone()),
            None => crate::optimizer::Optimizer::default_pipeline(),
        };
        opt_p2.imported_virtuals = imported;
        opt_p2.set_flatten_virtuals_at_jump(true);
        let p2_ops = opt_p2.optimize_with_constants_and_inputs(
            &remapped_ops, &mut consts_p2, body_num_inputs,
        );
        let p2_ni = opt_p2.final_num_inputs();

        if std::env::var_os("MAJIT_LOG").is_some() {
            let nc = p2_ops.iter().filter(|o| o.opcode == OpCode::New || o.opcode == OpCode::NewWithVtable).count();
            eprintln!("[jit] phase 2: {} ops, {} New remaining", p2_ops.len(), nc);
        }

        // ── Assembly (compile.py:310-338) ──
        let combined = assemble_peeled_trace(&p1_ops, &p2_ops, &label_args, p2_ni);

        *constants = consts_p2;
        let sp = crate::shortpreamble::extract_short_preamble(&combined);
        if !sp.is_empty() { self.short_preamble = Some(sp); }
        (combined, p2_ni)
    }
*/

// ── Helper: export_flatten_jump_args ──
// Replace virtual ptr in Phase 1 JUMP args with field values from SetfieldGc.
// Returns label_args with same count as body expects.

/*
fn export_flatten_jump_args(
    p1_ops: &[Op],
    p1_jump: &Op,
    jump_virtuals: &[crate::optimizer::ExportedJumpVirtual],
) -> Vec<OpRef> {
    let mut args = p1_jump.args.clone();
    // Process in reverse to keep indices stable
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
    args
}
*/

// ── Helper: build_imported_virtuals ──
// Create ImportedVirtual entries for Phase 2 import_state.

/*
fn build_imported_virtuals(
    jump_virtuals: &[crate::optimizer::ExportedJumpVirtual],
    _body_num_inputs: usize,
    _orig_num_inputs: usize,
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
        });
        offset += virt.fields.len() - 1;
    }
    imported
}
*/

// ── Helper: remap_trace_for_body ──
// Remap original trace op positions that collide with body's extra inputargs.

/*
fn remap_trace_for_body(
    ops: &[Op],
    constants: &std::collections::HashMap<u32, i64>,
    orig_num_inputs: usize,
    body_num_inputs: usize,
) -> (Vec<Op>, std::collections::HashMap<u32, i64>, HashMap<OpRef, OpRef>) {
    let extra = body_num_inputs.saturating_sub(orig_num_inputs) as u32;
    if extra == 0 {
        return (ops.to_vec(), constants.clone(), HashMap::new());
    }
    let mut remapped = ops.to_vec();
    let mut remap = HashMap::new();
    for op in &mut remapped {
        if op.pos.0 != u32::MAX
            && op.pos.0 >= orig_num_inputs as u32
            && op.pos.0 < (orig_num_inputs as u32 + extra)
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
    (remapped, new_consts, remap)
}
*/

// ── Helper: assemble_peeled_trace ──
// compile.py:310-338: [preamble_no_jump] + Label(label_args) + [body_with_jump]

/*
fn assemble_peeled_trace(
    p1_ops: &[Op],
    p2_ops: &[Op],
    label_args: &[OpRef],
    body_num_inputs: usize,
) -> Vec<Op> {
    let mut result = Vec::with_capacity(p1_ops.len() + p2_ops.len() + 1);

    // Preamble: everything except the Jump
    for op in p1_ops {
        if op.opcode == OpCode::Jump { break; }
        result.push(op.clone());
    }

    // Label position: after all preamble ops, beyond all existing positions
    let max_preamble_pos = result.iter()
        .map(|op| op.pos.0).filter(|&p| p != u32::MAX).max().unwrap_or(0);
    let label_pos = (max_preamble_pos + 1).max(body_num_inputs as u32);
    let mut label_op = Op::new(OpCode::Label, label_args);
    label_op.pos = OpRef(label_pos);
    result.push(label_op);

    // Body: remap inputarg refs → label_args, op positions → after label
    let body_base = label_pos + 1;
    let mut remap: HashMap<OpRef, OpRef> = HashMap::new();

    // Inputargs (0..body_num_inputs) → corresponding label_args
    for (i, &la) in label_args.iter().enumerate() {
        if (i as u32) < body_num_inputs as u32 {
            remap.insert(OpRef(i as u32), la);
        }
    }

    // Op positions → sequential after body_base
    for (idx, op) in p2_ops.iter().enumerate() {
        if op.pos.0 != u32::MAX {
            remap.insert(op.pos, OpRef(body_base + idx as u32));
        }
    }

    for (idx, op) in p2_ops.iter().enumerate() {
        let mut new_op = op.clone();
        if new_op.pos.0 != u32::MAX {
            new_op.pos = OpRef(body_base + idx as u32);
        }
        for arg in &mut new_op.args {
            if let Some(&m) = remap.get(arg) { *arg = m; }
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
*/
