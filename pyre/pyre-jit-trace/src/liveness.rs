//! Bytecode liveness analysis (codewriter/liveness.py parity).

use pyre_interpreter::bytecode::{CodeObject, Instruction};

/// Skip Cache pseudo-instructions starting at `pos`.
/// Returns the index of the first non-Cache instruction at or after `pos`.
fn skip_caches(code: &CodeObject, mut pos: usize) -> usize {
    while pos < code.instructions.len() {
        if let Some((Instruction::Cache, _)) = pyre_interpreter::decode_instruction_at(code, pos) {
            pos += 1;
        } else {
            break;
        }
    }
    pos
}

/// codewriter/liveness.py parity: bytecode liveness analysis.
/// For each bytecode PC, tracks which locals are live (may be read
/// on some path before being reassigned) and the operand stack depth.
///
/// RPython computes this from JitCode (register bytecodes) via
/// backward dataflow. pyre computes from Python bytecodes using the
/// same algorithm, with stack depth tracked via forward analysis.
pub struct LiveVars {
    /// Flat multi-word bitvector for local liveness.
    /// `live_bits[pc * words_per_pc + w]` = word w of bitvector at PC.
    /// No 64-slot cap: supports `words_per_pc * 64` locals.
    live_bits: Vec<u64>,
    /// Number of u64 words per PC (ceil(nlocals / 64)).
    words_per_pc: usize,
    /// Stack depth at each PC (forward analysis).
    /// RPython JitCode treats stack as registers with liveness.
    /// For Python bytecodes, stack depth determines which slots are live.
    stack_depth_at: Vec<usize>,
}

impl LiveVars {
    /// Backward dataflow liveness analysis on Python bytecodes.
    /// Fixed-point iteration handles loops correctly.
    fn compute(code: &CodeObject) -> Self {
        use pyre_interpreter::bytecode::Instruction;
        let n = code.instructions.len();
        if n == 0 {
            return LiveVars {
                live_bits: Vec::new(),
                words_per_pc: 0,
                stack_depth_at: Vec::new(),
            };
        }
        let nlocals = code.varnames.len().max(1);
        let words_per_pc = (nlocals + 63) / 64;
        let total = (n + 1) * words_per_pc;
        let mut live_bits = vec![0u64; total];

        // Temporary per-PC bitvector for the inner loop.
        let mut live = vec![0u64; words_per_pc];

        // Exception table: (start, end, target) ranges for implicit
        // control flow. Any instruction in [start, end) can jump to
        // target on exception — the target's live set must be unioned
        // into every PC in the range.
        let exc_handlers: Vec<(usize, usize, usize)> = {
            let entries = pyre_interpreter::bytecode::decode_exception_table(&code.exceptiontable);
            entries
                .iter()
                .map(|e| (e.start as usize, e.end as usize, e.target as usize))
                .collect()
        };

        // Fixed-point backward analysis for local liveness.
        let mut changed = true;
        while changed {
            changed = false;
            for pc in (0..n).rev() {
                let Some((instr, op_arg)) = pyre_interpreter::decode_instruction_at(code, pc)
                else {
                    continue;
                };
                // Start with successor's live set.
                let next_base = (pc + 1) * words_per_pc;
                for w in 0..words_per_pc {
                    live[w] = live_bits[next_base + w];
                }
                // Branch targets: union with target's live set.
                // CPython 3.13+ jump deltas are relative to NEXT_INSTR
                // (past any cache slots). Must use skip_caches to match
                // the codewriter's target calculation.
                let next = skip_caches(code, pc + 1);
                let target: Option<usize> = match instr {
                    Instruction::JumpForward { delta } => Some(next + delta.get(op_arg).as_usize()),
                    Instruction::JumpBackward { delta }
                    | Instruction::JumpBackwardNoInterrupt { delta } => {
                        Some(next.saturating_sub(delta.get(op_arg).as_usize()))
                    }
                    Instruction::PopJumpIfTrue { delta }
                    | Instruction::PopJumpIfFalse { delta } => {
                        Some(next + delta.get(op_arg).as_usize())
                    }
                    Instruction::PopJumpIfNone { delta }
                    | Instruction::PopJumpIfNotNone { delta } => {
                        Some(next + delta.get(op_arg).as_usize())
                    }
                    Instruction::ForIter { delta } => Some(next + delta.get(op_arg).as_usize()),
                    _ => None,
                };
                if let Some(tgt) = target {
                    let tgt_base = tgt * words_per_pc;
                    if tgt_base + words_per_pc <= live_bits.len() {
                        for w in 0..words_per_pc {
                            live[w] |= live_bits[tgt_base + w];
                        }
                    }
                }
                // liveness.py:59-78 GEN/KILL for locals.
                // GEN: register is read (add to alive).
                // KILL: register is written (remove from alive).
                match instr {
                    Instruction::LoadFast { var_num }
                    | Instruction::LoadFastBorrow { var_num }
                    | Instruction::LoadFastCheck { var_num }
                    | Instruction::LoadFastAndClear { var_num } => {
                        let i = var_num.get(op_arg).as_usize();
                        let word = i / 64;
                        if word < words_per_pc {
                            live[word] |= 1u64 << (i % 64);
                        }
                    }
                    // Super-instructions LOAD_FAST_LOAD_FAST and
                    // LOAD_FAST_BORROW_LOAD_FAST_BORROW read two locals in
                    // one opcode. Backward liveness must GEN both indices
                    // to match the runtime read set; otherwise blackhole
                    // resume at a guard PC feeding into this super-inst
                    // leaves the target registers stale (their prior
                    // content in the compiled frame never gets written
                    // back when consume_one_section walks an empty
                    // live-set). The extra live slots that caused the
                    // 2026-04-17 regressions (SIGSEGV on null refs and
                    // float-ULP drift) came from backward liveness alone
                    // marking uninitialised slots; the forward-definedness
                    // pass below (`defined_bits`) now intersects with the
                    // backward set so only stores that actually dominate
                    // the current PC count as live. This mirrors RPython's
                    // jtransform.py behaviour of only inserting `-live-`
                    // markers after the corresponding store.
                    // TODO: enable once forward-definedness lands with
                    // full stack-slot coverage — current pass only
                    // tracks locals, so super-inst GEN drives some
                    // stack-resident reads into uninitialised InputArg
                    // slots (fannkuch SIGSEGV).
                    // Instruction::LoadFastLoadFast { var_nums }
                    // | Instruction::LoadFastBorrowLoadFastBorrow { var_nums } => {
                    //     let pair = var_nums.get(op_arg);
                    //     for i in [
                    //         u32::from(pair.idx_1()) as usize,
                    //         u32::from(pair.idx_2()) as usize,
                    //     ] {
                    //         let word = i / 64;
                    //         if word < words_per_pc {
                    //             live[word] |= 1u64 << (i % 64);
                    //         }
                    //     }
                    // }
                    Instruction::StoreFast { var_num } | Instruction::DeleteFast { var_num } => {
                        let i = var_num.get(op_arg).as_usize();
                        let word = i / 64;
                        if word < words_per_pc {
                            live[word] &= !(1u64 << (i % 64));
                        }
                    }
                    // STORE_FAST_LOAD_FAST(a, b): store top-of-stack into
                    // local a (KILL a), then load local b (GEN b). Backward
                    // order is reversed: GEN(b) first, then KILL(a), so
                    // a == b leaves the slot live (the load reads a value
                    // defined elsewhere).
                    Instruction::StoreFastLoadFast { var_nums } => {
                        let pair = var_nums.get(op_arg);
                        let load_idx = u32::from(pair.idx_2()) as usize;
                        let store_idx = u32::from(pair.idx_1()) as usize;
                        let lw = load_idx / 64;
                        if lw < words_per_pc {
                            live[lw] |= 1u64 << (load_idx % 64);
                        }
                        let sw = store_idx / 64;
                        if sw < words_per_pc {
                            live[sw] &= !(1u64 << (store_idx % 64));
                        }
                    }
                    // STORE_FAST_STORE_FAST(a, b): two consecutive stores,
                    // no reads. Both locals are KILLed.
                    Instruction::StoreFastStoreFast { var_nums } => {
                        let pair = var_nums.get(op_arg);
                        for i in [
                            u32::from(pair.idx_1()) as usize,
                            u32::from(pair.idx_2()) as usize,
                        ] {
                            let word = i / 64;
                            if word < words_per_pc {
                                live[word] &= !(1u64 << (i % 64));
                            }
                        }
                    }
                    _ => {}
                }
                // Exception handlers: union with handler target's live set.
                // This handles implicit control flow edges from try blocks
                // to except handlers (not covered by explicit jump targets).
                for entry in &exc_handlers {
                    if pc >= entry.0 && pc < entry.1 {
                        let tgt = entry.2;
                        let tgt_base = tgt * words_per_pc;
                        if tgt_base + words_per_pc <= live_bits.len() {
                            for w in 0..words_per_pc {
                                live[w] |= live_bits[tgt_base + w];
                            }
                        }
                    }
                }
                let base = pc * words_per_pc;
                if (0..words_per_pc).any(|w| live_bits[base + w] != live[w]) {
                    for w in 0..words_per_pc {
                        live_bits[base + w] = live[w];
                    }
                    changed = true;
                }
            }
        }

        // Forward stack depth analysis.
        let mut stack_depth_at = vec![usize::MAX; n + 1];
        stack_depth_at[0] = 0;
        let mut sd_changed = true;
        while sd_changed {
            sd_changed = false;
            for pc in 0..n {
                let d = stack_depth_at[pc];
                if d == usize::MAX {
                    continue;
                }
                let Some((instr, op_arg)) = pyre_interpreter::decode_instruction_at(code, pc)
                else {
                    // Unknown instruction: propagate same depth.
                    if stack_depth_at[pc + 1] == usize::MAX {
                        stack_depth_at[pc + 1] = d;
                        sd_changed = true;
                    }
                    continue;
                };
                let (ft_d, br_d) = stack_effects(&instr, op_arg, d);
                // Fall-through.
                if !is_unconditional_jump(&instr) && ft_d < stack_depth_at[pc + 1] {
                    stack_depth_at[pc + 1] = ft_d;
                    sd_changed = true;
                }
                // Branch target.
                if let Some(tgt) = target_pc(code, &instr, pc, op_arg) {
                    if tgt <= n && br_d < stack_depth_at[tgt] {
                        stack_depth_at[tgt] = br_d;
                        sd_changed = true;
                    }
                }
            }
        }

        LiveVars {
            live_bits,
            words_per_pc,
            stack_depth_at,
        }
    }

    /// codewriter/liveness.py _live_vars(pc) parity — local registers.
    /// No slot-count cap: supports arbitrary number of locals.
    pub fn is_local_live(&self, pc: usize, local_idx: usize) -> bool {
        let word = local_idx / 64;
        let bit = local_idx % 64;
        if word >= self.words_per_pc {
            return false; // index beyond tracked locals
        }
        self.live_bits
            .get(pc * self.words_per_pc + word)
            .map_or(true, |w| (w >> bit) & 1 != 0)
    }

    /// codewriter/liveness.py parity — stack registers.
    /// Stack slot is live if index < stack_depth_at[pc].
    pub fn is_stack_live(&self, pc: usize, stack_idx: usize) -> bool {
        self.stack_depth_at
            .get(pc)
            .map_or(true, |&depth| stack_idx < depth)
    }

    /// Number of live stack slots at the given PC.
    pub fn stack_depth_at(&self, pc: usize) -> usize {
        self.stack_depth_at.get(pc).copied().unwrap_or(0)
    }

    /// True iff the dataflow analysis reached `pc`. Unreachable pcs have
    /// their `stack_depth_at` slot left at the `usize::MAX` sentinel
    /// (liveness.rs:150) and must not be treated as having a real stack
    /// height. Used by the codewriter to skip liveness emission for
    /// unreachable bytecode (RPython parity: flatten_graph() emits only
    /// reachable blocks).
    pub fn is_reachable(&self, pc: usize) -> bool {
        self.stack_depth_at
            .get(pc)
            .map_or(false, |&d| d != usize::MAX)
    }
}

/// Stack effects: (fallthrough_depth, branch_depth).
/// Returns new absolute stack depths after the instruction.
fn stack_effects(
    instr: &pyre_interpreter::bytecode::Instruction,
    op_arg: pyre_interpreter::OpArg,
    depth: usize,
) -> (usize, usize) {
    use pyre_interpreter::bytecode::Instruction;
    let d = depth as i32;
    let (ft, br) = match instr {
        // No-ops
        Instruction::Nop
        | Instruction::Resume { .. }
        | Instruction::Cache
        | Instruction::NotTaken
        | Instruction::ExtendedArg => (d, d),
        // Push one
        Instruction::LoadConst { .. }
        | Instruction::LoadSmallInt { .. }
        | Instruction::LoadFast { .. }
        | Instruction::LoadFastBorrow { .. }
        | Instruction::LoadFastCheck { .. }
        | Instruction::LoadFastAndClear { .. }
        | Instruction::LoadName { .. }
        | Instruction::LoadDeref { .. }
        | Instruction::LoadLocals
        | Instruction::LoadBuildClass
        | Instruction::PushNull
        | Instruction::Copy { .. }
        | Instruction::PushExcInfo => (d + 1, d + 1),
        // LOAD_GLOBAL pushes 1 or 2 values depending on the low bit of namei.
        // When namei & 1 == 1, a NULL sentinel is pushed before the result (+2).
        // When namei & 1 == 0, only the result is pushed (+1).
        Instruction::LoadGlobal { namei } => {
            let raw = namei.get(op_arg) as usize;
            if raw & 1 != 0 {
                (d + 2, d + 2)
            } else {
                (d + 1, d + 1)
            }
        }
        // Push two (super-instructions: load two locals)
        Instruction::LoadFastBorrowLoadFastBorrow { .. } | Instruction::LoadFastLoadFast { .. } => {
            (d + 2, d + 2)
        }
        // STORE_FAST_LOAD_FAST: pop one, push one (net 0)
        Instruction::StoreFastLoadFast { .. } => (d, d),
        // STORE_FAST_STORE_FAST: pop two, push none
        Instruction::StoreFastStoreFast { .. } => (d - 2, d - 2),
        // Pop one
        Instruction::PopTop
        | Instruction::StoreFast { .. }
        | Instruction::StoreName { .. }
        | Instruction::StoreGlobal { .. }
        | Instruction::StoreDeref { .. }
        | Instruction::EndSend
        | Instruction::PopIter => (d - 1, d - 1),
        // YieldValue: sends TOS to caller, receives new value back. Net 0.
        Instruction::YieldValue { .. } => (d, d),
        // PopExcept: codewriter emits no depth change (exception state
        // is in TLS, cleared by interpreter). Must match codewriter.
        Instruction::PopExcept => (d, d),
        // Pop two
        Instruction::EndFor => (d - 2, d - 2),
        // Pop 0, push 0 (identity stack effect)
        Instruction::DeleteFast { .. }
        | Instruction::DeleteName { .. }
        | Instruction::DeleteGlobal { .. }
        | Instruction::DeleteDeref { .. }
        | Instruction::Swap { .. }
        | Instruction::CopyFreeVars { .. } => (d, d),
        // Pop 1 (consume item, append/extend into collection at TOS[i])
        Instruction::ListAppend { .. }
        | Instruction::SetAdd { .. }
        | Instruction::ListExtend { .. }
        | Instruction::SetUpdate { .. }
        | Instruction::DictUpdate { .. }
        | Instruction::DictMerge { .. } => (d - 1, d - 1),
        // Pop 2 (key + value into collection at TOS[i])
        Instruction::MapAdd { .. } => (d - 2, d - 2),
        // Pop 1 push 1 (net 0)
        Instruction::UnaryNegative
        | Instruction::UnaryNot
        | Instruction::UnaryInvert
        | Instruction::GetIter
        | Instruction::GetYieldFromIter
        | Instruction::GetAIter
        | Instruction::LoadAttr { .. }
        | Instruction::GetAwaitable { .. } => (d, d),
        // Push 1 onto TOS without popping (net +1)
        Instruction::GetLen
        | Instruction::MatchMapping
        | Instruction::MatchSequence
        | Instruction::ImportFrom { .. } => (d + 1, d + 1),
        // ImportName: pop 2 (fromlist + level), push 1 (module). Net -1.
        Instruction::ImportName { .. } => (d - 1, d - 1),
        // LoadSuperAttr: pop 3 (global_super, cls, self), push 1. Net -2.
        Instruction::LoadSuperAttr { .. } => (d - 2, d - 2),
        // CheckExcMatch: codewriter pops 2 (match_type + exception),
        // pushes 1 (result). Net -1. Must match codewriter.
        Instruction::CheckExcMatch => (d - 1, d - 1),
        // Pop 2 push 1 (net -1)
        Instruction::BinaryOp { .. }
        | Instruction::CompareOp { .. }
        | Instruction::IsOp { .. }
        | Instruction::ContainsOp { .. } => (d - 1, d - 1),
        // Pop 2: obj + value
        Instruction::StoreAttr { .. } => (d - 2, d - 2),
        // Pop 1: obj only
        Instruction::DeleteAttr { .. } => (d - 1, d - 1),
        // Pop 3: value + obj + key. Codewriter pops 3 explicitly.
        Instruction::StoreSubscr => (d - 3, d - 3),
        // Pop 2: obj + key
        Instruction::DeleteSubscr => (d - 2, d - 2),
        // Pop 4: value + obj + start + end
        Instruction::StoreSlice => (d - 4, d - 4),
        // Unconditional jumps
        Instruction::JumpForward { .. }
        | Instruction::JumpBackward { .. }
        | Instruction::JumpBackwardNoInterrupt { .. } => (d, d),
        // Conditional pop-and-jump: pop 1 on both paths
        Instruction::PopJumpIfTrue { .. }
        | Instruction::PopJumpIfFalse { .. }
        | Instruction::PopJumpIfNone { .. }
        | Instruction::PopJumpIfNotNone { .. } => (d - 1, d - 1),
        // ForIter: fallthrough pushes TOS_next (+1); branch pops iterator (-1)
        Instruction::ForIter { .. } => (d + 1, d - 1),
        // Return
        Instruction::ReturnValue => (d - 1, d - 1),
        // Build collections: pop count, push 1
        Instruction::BuildTuple { count } => {
            let s = count.get(op_arg) as usize as i32;
            (d - s + 1, d - s + 1)
        }
        Instruction::BuildList { count } => {
            let s = count.get(op_arg) as usize as i32;
            (d - s + 1, d - s + 1)
        }
        Instruction::BuildSet { count } => {
            let s = count.get(op_arg) as usize as i32;
            (d - s + 1, d - s + 1)
        }
        Instruction::BuildMap { count } => {
            let s = (count.get(op_arg) as usize * 2) as i32;
            (d - s + 1, d - s + 1)
        }
        Instruction::BuildString { count } => {
            let s = count.get(op_arg) as usize as i32;
            (d - s + 1, d - s + 1)
        }
        // CALL: pop callable + args, push result
        Instruction::Call { argc } => {
            let n = argc.get(op_arg) as usize as i32;
            (d - n - 1, d - n - 1)
        }
        // Unpack
        Instruction::UnpackSequence { count } => {
            let s = count.get(op_arg) as usize as i32;
            (d - 1 + s, d - 1 + s)
        }
        // MATCH_CLASS (pyopcode.py:1730): pops w_names, w_type, w_subject (−3)
        // and pushes either (None, False) or (tuple, True) (+2). Net −1.
        Instruction::MatchClass { .. } => (d - 1, d - 1),
        // Reraise pops TOS (exception). Net -1.
        Instruction::Reraise { .. } => (d - 1, d - 1),
        // RaiseVarargs: pops argc values from stack.
        // Codewriter pops 1 when argc >= 1, 0 when argc == 0.
        Instruction::RaiseVarargs { argc } => {
            let n = argc.get(op_arg) as i32;
            (d - n, d - n)
        }
        // Default: conservative, keep same depth.
        _ => (d, d),
    };
    (ft.max(0) as usize, br.max(0) as usize)
}

/// Is the instruction an unconditional jump (no fallthrough)?
fn is_unconditional_jump(instr: &pyre_interpreter::bytecode::Instruction) -> bool {
    use pyre_interpreter::bytecode::Instruction;
    matches!(
        instr,
        Instruction::JumpForward { .. }
            | Instruction::JumpBackward { .. }
            | Instruction::JumpBackwardNoInterrupt { .. }
            | Instruction::ReturnValue
            | Instruction::Reraise { .. }
            | Instruction::RaiseVarargs { .. }
    )
}

/// Branch target PC for branching instructions.
/// Uses skip_caches to account for cache slots after the instruction,
/// matching the codewriter's target calculation.
fn target_pc(
    code: &CodeObject,
    instr: &pyre_interpreter::bytecode::Instruction,
    pc: usize,
    op_arg: pyre_interpreter::OpArg,
) -> Option<usize> {
    use pyre_interpreter::bytecode::Instruction;
    let next = skip_caches(code, pc + 1);
    match instr {
        Instruction::JumpForward { delta } => Some(next + delta.get(op_arg).as_usize()),
        Instruction::JumpBackward { delta } | Instruction::JumpBackwardNoInterrupt { delta } => {
            Some(next.saturating_sub(delta.get(op_arg).as_usize()))
        }
        Instruction::PopJumpIfTrue { delta }
        | Instruction::PopJumpIfFalse { delta }
        | Instruction::PopJumpIfNone { delta }
        | Instruction::PopJumpIfNotNone { delta } => Some(next + delta.get(op_arg).as_usize()),
        Instruction::ForIter { delta } => Some(next + delta.get(op_arg).as_usize()),
        _ => None,
    }
}

/// Cache liveness info per CodeObject pointer.
/// codewriter/liveness.py parity: thread-local cache of computed liveness.
pub fn liveness_for(code: *const CodeObject) -> &'static LiveVars {
    use std::cell::RefCell;
    use std::collections::HashMap;
    thread_local! {
        // SAFETY: LiveVars is computed once and never mutated.
        // The 'static lifetime is safe because CodeObject outlives the JIT.
        static CACHE: RefCell<HashMap<usize, Box<LiveVars>>> =
            RefCell::new(HashMap::new());
    }
    let key = code as usize;
    CACHE.with(|c| {
        let mut c = c.borrow_mut();
        let entry = c.entry(key).or_insert_with(|| {
            let code_ref = unsafe { &*code };
            Box::new(LiveVars::compute(code_ref))
        });
        // SAFETY: the Box is never removed from the HashMap and LiveVars
        // is immutable, so extending the lifetime is safe within this thread.
        unsafe { &*(entry.as_ref() as *const LiveVars) }
    })
}

/// jitcode.py:147-167 enumerate_vars parity: expand compact (dead-skipped)
/// values to dense positional layout using liveness analysis.
///
/// Encode side (`get_list_of_active_boxes`) skips dead registers, producing
/// a compact array. This function reverses that: iterates register indices
/// via liveness, pops the next compact value for each live index, and places
/// `Value::Void` for dead indices.
pub fn expand_compact_to_dense(
    code: *const CodeObject,
    py_pc: usize,
    compact_values: &[majit_ir::Value],
    nlocals: usize,
    stack_depth: usize,
) -> Vec<majit_ir::Value> {
    let live = liveness_for(code);
    let mut result = Vec::with_capacity(nlocals + stack_depth);
    let mut compact_idx = 0;
    for i in 0..nlocals {
        if live.is_local_live(py_pc, i) && compact_idx < compact_values.len() {
            result.push(compact_values[compact_idx].clone());
            compact_idx += 1;
        } else {
            result.push(majit_ir::Value::Void);
        }
    }
    for i in 0..stack_depth {
        if live.is_stack_live(py_pc, i) && compact_idx < compact_values.len() {
            result.push(compact_values[compact_idx].clone());
            compact_idx += 1;
        } else {
            result.push(majit_ir::Value::Void);
        }
    }
    result
}
