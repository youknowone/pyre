/// Pure operation optimization (Common Subexpression Elimination).
///
/// Translated from rpython/jit/metainterp/optimizeopt/pure.py.
///
/// When the same pure operation is seen again with the same arguments,
/// the cached result is returned instead of recomputing.
use std::collections::HashMap;

use majit_ir::{Op, OpCode, OpRef};

use crate::{OptContext, Optimization, OptimizationResult};

/// Key for looking up a previously computed pure operation.
///
/// Identifies an operation by its opcode and argument OpRefs.
/// For operations with a descriptor, the descriptor identity is not tracked
/// here because pure ops in the always-pure range don't use descriptors
/// for identity purposes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct PureOpKey {
    opcode: OpCode,
    args: Vec<OpRef>,
}

impl PureOpKey {
    fn from_op(op: &Op) -> Self {
        PureOpKey {
            opcode: op.opcode,
            args: op.args.to_vec(),
        }
    }

    /// Build a key with swapped arguments (for commutative operations).
    fn commuted(&self) -> Self {
        debug_assert!(self.args.len() == 2);
        let mut swapped = self.args.clone();
        swapped.swap(0, 1);
        PureOpKey {
            opcode: self.opcode,
            args: swapped,
        }
    }
}

/// Cache mapping (opcode, args) -> result OpRef for recently seen pure operations.
///
/// Translated from RecentPureOps in pure.py.
/// Uses a HashMap for O(1) lookup instead of the Python linear-scan ring buffer,
/// combined with an LRU eviction list capped at `limit` entries.
struct RecentPureOps {
    map: HashMap<PureOpKey, OpRef>,
    /// Ring buffer of keys for LRU eviction.
    order: Vec<Option<PureOpKey>>,
    next_index: usize,
}

impl RecentPureOps {
    fn new(limit: usize) -> Self {
        RecentPureOps {
            map: HashMap::with_capacity(limit),
            order: vec![None; limit],
            next_index: 0,
        }
    }

    /// Record that `key` produces the result `result`.
    fn insert(&mut self, key: PureOpKey, result: OpRef) {
        // Evict the oldest entry if the slot is occupied.
        if let Some(old_key) = self.order[self.next_index].take() {
            self.map.remove(&old_key);
        }
        self.order[self.next_index] = Some(key.clone());
        self.map.insert(key, result);
        self.next_index += 1;
        if self.next_index >= self.order.len() {
            self.next_index = 0;
        }
    }

    /// Look up a previously recorded result for the given key.
    fn lookup(&self, key: &PureOpKey) -> Option<OpRef> {
        self.map.get(key).copied()
    }

    /// pure.py: lookup1(opt, box0, descr) — look up a unary pure operation.
    /// Searches for any cached op with the given opcode and single arg.
    fn lookup1(&self, opcode: OpCode, arg0: OpRef) -> Option<OpRef> {
        let key = PureOpKey {
            opcode,
            args: vec![arg0],
        };
        self.map.get(&key).copied()
    }

    /// pure.py: lookup2(opt, box0, box1, descr, commutative)
    /// Look up a binary pure operation, optionally checking swapped args.
    fn lookup2(
        &self,
        opcode: OpCode,
        arg0: OpRef,
        arg1: OpRef,
        commutative: bool,
    ) -> Option<OpRef> {
        let key = PureOpKey {
            opcode,
            args: vec![arg0, arg1],
        };
        if let Some(result) = self.map.get(&key).copied() {
            return Some(result);
        }
        if commutative {
            let key_swapped = PureOpKey {
                opcode,
                args: vec![arg1, arg0],
            };
            return self.map.get(&key_swapped).copied();
        }
        None
    }
}

/// The OptPure optimization pass.
///
/// pure.py: OptPure class.
/// For pure operations (is_always_pure), checks if the same operation was
/// computed before. If yes, replaces the current op with the cached result
/// (CSE). If no, records the operation for future lookups.
///
/// Also handles:
/// - CALL_PURE -> CALL demotion when arguments aren't all constant.
/// - CALL_LOOPINVARIANT_* caching (result persists for entire loop iteration).
/// - OVF operation postponement (INT_ADD_OVF etc. are deferred until GUARD_NO_OVERFLOW).
/// - GUARD_NO_EXCEPTION removal after eliminated CALL_PURE.
/// - RECORD_KNOWN_RESULT for pre-recorded call_pure results.
pub struct OptPure {
    cache: RecentPureOps,
    /// Per-loop-iteration cache for CALL_LOOPINVARIANT_* results.
    loopinvariant_cache: HashMap<PureOpKey, OpRef>,
    /// Postponed OVF operation: INT_ADD_OVF, INT_SUB_OVF, INT_MUL_OVF.
    /// pure.py: postponed_op — deferred until GUARD_NO_OVERFLOW is seen.
    postponed_op: Option<Op>,
    /// Indices into new_operations of emitted CALL_PURE ops.
    /// pure.py: call_pure_positions — tracked for short preamble generation.
    call_pure_positions: Vec<usize>,
    /// RPython pure.py / shortpreamble.py: pure ops that phase 2 should be
    /// able to reproduce from the preamble via optimizer state, not by
    /// textual body replay.
    short_preamble_pure_ops: Vec<Op>,
    /// RPython shortpreamble.py: CALL_LOOPINVARIANT ops tracked separately
    /// from regular pure ops and re-imported into rewrite state.
    short_preamble_loopinvariant_ops: Vec<Op>,
    /// Whether the last emitted operation was removed (for GUARD_NO_EXCEPTION elimination).
    /// pure.py: last_emitted_operation is REMOVED check.
    last_emitted_was_removed: bool,
    /// Pre-recorded CALL_PURE results from RECORD_KNOWN_RESULT.
    /// pure.py: known_result_call_pure
    known_result_call_pure: Vec<(PureOpKey, OpRef)>,
    /// pure.py: extra_call_pure — CALL_PURE results from the previous
    /// loop iteration (fed from Optimizer.call_pure_results).
    /// These are checked before the regular cache for cross-iteration CSE.
    extra_call_pure: Vec<(PureOpKey, OpRef)>,
}

impl OptPure {
    pub fn new() -> Self {
        OptPure {
            cache: RecentPureOps::new(4096),
            loopinvariant_cache: HashMap::new(),
            postponed_op: None,
            call_pure_positions: Vec::new(),
            short_preamble_pure_ops: Vec::new(),
            short_preamble_loopinvariant_ops: Vec::new(),
            last_emitted_was_removed: false,
            known_result_call_pure: Vec::new(),
            extra_call_pure: Vec::new(),
        }
    }

    /// pure.py: inject extra_call_pure from optimizer.call_pure_results.
    /// Called before optimization starts to seed cross-iteration CSE data.
    pub fn set_extra_call_pure(&mut self, results: Vec<(Vec<OpRef>, OpRef)>) {
        self.extra_call_pure = results
            .into_iter()
            .map(|(args, result)| {
                let key = PureOpKey {
                    opcode: OpCode::CallPureI,
                    args,
                };
                (key, result)
            })
            .collect();
    }

    /// Whether this opcode is commutative (order of args doesn't matter).
    fn is_commutative(opcode: OpCode) -> bool {
        matches!(
            opcode,
            OpCode::IntAdd
                | OpCode::IntAddOvf
                | OpCode::IntMul
                | OpCode::IntMulOvf
                | OpCode::IntAnd
                | OpCode::IntOr
                | OpCode::IntXor
        )
    }

    /// Try to find a cached result for this operation, considering commutativity.
    fn lookup_pure(&self, key: &PureOpKey) -> Option<OpRef> {
        if let Some(result) = self.cache.lookup(key) {
            return Some(result);
        }
        // For commutative binary ops, also check with swapped arguments.
        if key.args.len() == 2 && Self::is_commutative(key.opcode) {
            let swapped = key.commuted();
            return self.cache.lookup(&swapped);
        }
        None
    }

    /// Handle CALL_PURE: demote to plain CALL since we can't constant-fold.
    fn handle_call_pure(&self, op: &Op) -> OptimizationResult {
        OptimizationResult::Emit(self.demote_call_pure(op))
    }

    /// Record a pure operation in the CSE cache.
    /// pure.py: pure(opnum, op)
    pub fn pure(&mut self, op: &Op) {
        let key = PureOpKey::from_op(op);
        self.cache.insert(key, op.pos);
    }

    /// Record a pure operation with explicit args.
    /// pure.py: pure_from_args(opnum, args, op, descr=None)
    pub fn pure_from_args(&mut self, opcode: OpCode, args: &[OpRef], result: OpRef) {
        let key = PureOpKey {
            opcode,
            args: args.to_vec(),
        };
        self.cache.insert(key, result);
    }

    /// pure.py: pure_from_args1(opnum, arg0, op)
    /// Specialized version for unary operations.
    pub fn pure_from_args1(&mut self, opcode: OpCode, arg0: OpRef, result: OpRef) {
        self.pure_from_args(opcode, &[arg0], result);
    }

    /// pure.py: pure_from_args2(opnum, arg0, arg1, op)
    /// Specialized version for binary operations.
    pub fn pure_from_args2(&mut self, opcode: OpCode, arg0: OpRef, arg1: OpRef, result: OpRef) {
        self.pure_from_args(opcode, &[arg0, arg1], result);
    }

    /// Look up a previously recorded pure operation result.
    /// pure.py: get_pure_result(op)
    pub fn get_pure_result(&self, op: &Op) -> Option<OpRef> {
        let key = PureOpKey::from_op(op);
        self.lookup_pure(&key)
    }

    /// pure.py: lookup1(opt, box0, descr) — look up a unary pure op result.
    pub fn lookup1(&self, opcode: OpCode, arg0: OpRef) -> Option<OpRef> {
        self.cache.lookup1(opcode, arg0)
    }

    /// pure.py: lookup2(opt, box0, box1, descr, commutative)
    pub fn lookup2(
        &self,
        opcode: OpCode,
        arg0: OpRef,
        arg1: OpRef,
        commutative: bool,
    ) -> Option<OpRef> {
        self.cache.lookup2(opcode, arg0, arg1, commutative)
    }

    /// Record a CALL_PURE result from a RECORD_KNOWN_RESULT hint.
    /// pure.py: optimize_RECORD_KNOWN_RESULT
    pub fn record_known_result(&mut self, args: &[OpRef], result: OpRef) {
        let key = PureOpKey {
            opcode: OpCode::CallPureI,
            args: args.to_vec(),
        };
        self.known_result_call_pure.push((key, result));
    }

    /// Get the positions of emitted CALL_PURE ops (for short preamble generation).
    /// pure.py: call_pure_positions
    pub fn call_pure_positions(&self) -> &[usize] {
        &self.call_pure_positions
    }

    /// Check known_result_call_pure for a matching call.
    fn lookup_known_result(&self, key: &PureOpKey) -> Option<OpRef> {
        // Check known_result_call_pure first (from RECORD_KNOWN_RESULT)
        for (k, result) in &self.known_result_call_pure {
            if k.args == key.args {
                return Some(*result);
            }
        }
        // pure.py: also check extra_call_pure (from previous loop iteration)
        for (k, result) in &self.extra_call_pure {
            if k.args == key.args {
                return Some(*result);
            }
        }
        None
    }

    /// Handle CALL_LOOPINVARIANT_*: cache the result for the loop iteration.
    fn handle_call_loopinvariant(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        let key = PureOpKey::from_op(op);

        // Check if we've already computed this loop-invariant call.
        if let Some(cached_ref) = self.loopinvariant_cache.get(&key).copied() {
            let cached_ref = ctx.get_replacement(cached_ref);
            ctx.replace_op(op.pos, cached_ref);
            return OptimizationResult::Remove;
        }

        // Also check the commutative case (unlikely for calls, but consistent).
        // Not applicable for calls — skip.

        // Cache the result and demote to plain CALL_*.
        self.loopinvariant_cache.insert(key, op.pos);

        let new_op = self.demote_call_loopinvariant(op);
        self.short_preamble_loopinvariant_ops.push(new_op.clone());
        OptimizationResult::Emit(new_op)
    }

    fn demote_call_pure(&self, op: &Op) -> Op {
        let mut new_op = op.clone();
        new_op.opcode = OpCode::call_for_type(op.result_type());
        new_op
    }

    fn demote_call_loopinvariant(&self, op: &Op) -> Op {
        let mut new_op = op.clone();
        new_op.opcode = OpCode::call_for_type(op.result_type());
        new_op
    }

    fn call_pure_can_raise(op: &Op) -> bool {
        op.descr
            .as_ref()
            .and_then(|d| d.as_call_descr())
            .map(|cd| cd.effect_info().check_can_raise(true))
            .unwrap_or(true)
    }

    fn lookup_imported_short_pure(&self, op: &Op, ctx: &OptContext) -> Option<OpRef> {
        ctx.imported_short_pure_ops.iter().find_map(|entry| {
            if entry.opcode != op.opcode {
                return None;
            }
            if entry.descr_idx != op.descr.as_ref().map(|d| d.index()) {
                return None;
            }
            if entry.args.len() != op.args.len() {
                return None;
            }
            for (expected, &arg) in entry.args.iter().zip(op.args.iter()) {
                match expected {
                    crate::ImportedShortPureArg::OpRef(expected_ref) => {
                        if ctx.get_replacement(arg) != *expected_ref {
                            return None;
                        }
                    }
                    crate::ImportedShortPureArg::Const(expected_value) => {
                        if ctx.get_constant(arg) != Some(expected_value) {
                            return None;
                        }
                    }
                }
            }
            Some(entry.result)
        })
    }
}

/// Try to constant-fold a pure operation when all arguments are constants.
///
/// RPython equivalent: pure.py constant folding in optimize_default().
/// Returns the constant result value if successful.
fn try_constant_fold_value(op: &Op, ctx: &OptContext) -> Option<i64> {
    // Only fold binary int operations for now (most common case).
    if op.num_args() != 2 {
        return None;
    }
    let a = ctx.get_constant_int(op.arg(0))?;
    let b = ctx.get_constant_int(op.arg(1))?;

    let result = match op.opcode {
        OpCode::IntAdd | OpCode::IntAddOvf => a.checked_add(b)?,
        OpCode::IntSub | OpCode::IntSubOvf => a.checked_sub(b)?,
        OpCode::IntMul | OpCode::IntMulOvf => a.checked_mul(b)?,
        OpCode::IntAnd => Some(a & b)?,
        OpCode::IntOr => Some(a | b)?,
        OpCode::IntXor => Some(a ^ b)?,
        OpCode::IntLshift if b >= 0 && b < 64 => Some(a << b)?,
        OpCode::IntRshift if b >= 0 && b < 64 => Some(a >> b)?,
        OpCode::UintRshift if b >= 0 && b < 64 => Some((a as u64 >> b as u64) as i64)?,
        OpCode::IntLt => Some(if a < b { 1 } else { 0 })?,
        OpCode::IntLe => Some(if a <= b { 1 } else { 0 })?,
        OpCode::IntGt => Some(if a > b { 1 } else { 0 })?,
        OpCode::IntGe => Some(if a >= b { 1 } else { 0 })?,
        OpCode::IntEq => Some(if a == b { 1 } else { 0 })?,
        OpCode::IntNe => Some(if a != b { 1 } else { 0 })?,
        OpCode::UintLt => Some(if (a as u64) < (b as u64) { 1 } else { 0 })?,
        OpCode::UintLe => Some(if (a as u64) <= (b as u64) { 1 } else { 0 })?,
        OpCode::UintGe => Some(if (a as u64) >= (b as u64) { 1 } else { 0 })?,
        OpCode::UintGt => Some(if (a as u64) > (b as u64) { 1 } else { 0 })?,
        OpCode::IntFloorDiv if b != 0 => {
            // Python-style floor division
            let (q, r) = (a / b, a % b);
            if (r != 0) && ((r ^ b) < 0) {
                Some(q - 1)
            } else {
                Some(q)
            }
        }?,
        OpCode::IntMod if b != 0 => {
            let r = a % b;
            if (r != 0) && ((r ^ b) < 0) {
                Some(r + b)
            } else {
                Some(r)
            }
        }?,
        _ => return None,
    };

    Some(result)
}

impl Default for OptPure {
    fn default() -> Self {
        Self::new()
    }
}

impl Optimization for OptPure {
    fn propagate_forward(&mut self, op: &Op, ctx: &mut OptContext) -> OptimizationResult {
        // Don't reset for GUARD_NO_EXCEPTION — it needs the previous state.
        if op.opcode != OpCode::GuardNoException {
            self.last_emitted_was_removed = false;
        }

        // pure.py: OVF operation postponement.
        // INT_ADD_OVF, INT_SUB_OVF, INT_MUL_OVF are deferred until we see
        // GUARD_NO_OVERFLOW, so we can try CSE on the OVF op + guard pair.
        if op.opcode.is_ovf() {
            self.postponed_op = Some(op.clone());
            return OptimizationResult::Remove;
        }

        // Handle the postponed OVF op when we see GUARD_NO_OVERFLOW.
        if let Some(postponed) = self.postponed_op.take() {
            if op.opcode == OpCode::GuardNoOverflow {
                // Try constant folding on the OVF op.
                if let Some(folded) = try_constant_fold_value(&postponed, ctx) {
                    ctx.find_or_record_constant_int(postponed.pos, folded);
                    self.last_emitted_was_removed = true;
                    return OptimizationResult::Remove; // guard also removed
                }

                if let Some(cached_ref) = self.lookup_imported_short_pure(&postponed, ctx) {
                    let cached_ref = ctx.get_replacement(cached_ref);
                    ctx.replace_op(postponed.pos, cached_ref);
                    self.last_emitted_was_removed = true;
                    return OptimizationResult::Remove; // guard also removed
                }

                // pure.py: CSE on the OVF op.
                // _can_reuse_oldop: OVF ops can only reuse results from
                // other OVF ops (not regular INT_ADD etc.), because the
                // guard pairing requires the OVF semantic.
                let key = PureOpKey::from_op(&postponed);
                if let Some(cached_ref) = self.lookup_pure(&key) {
                    let cached_ref = ctx.get_replacement(cached_ref);
                    ctx.replace_op(postponed.pos, cached_ref);
                    self.last_emitted_was_removed = true;
                    return OptimizationResult::Remove; // guard also removed
                }
                // Also check the non-OVF version (INT_ADD for INT_ADD_OVF, etc.)
                let non_ovf_opcode = match postponed.opcode {
                    OpCode::IntAddOvf => Some(OpCode::IntAdd),
                    OpCode::IntSubOvf => Some(OpCode::IntSub),
                    OpCode::IntMulOvf => Some(OpCode::IntMul),
                    _ => None,
                };
                if let Some(non_ovf) = non_ovf_opcode {
                    let non_ovf_key = PureOpKey {
                        opcode: non_ovf,
                        args: postponed.args.to_vec(),
                    };
                    if let Some(cached_ref) = self.lookup_pure(&non_ovf_key) {
                        // A non-OVF version exists. We CAN'T reuse it for OVF
                        // because the guard needs the OVF semantics. But we
                        // record the OVF result as also being the non-OVF result.
                        let _ = cached_ref;
                    }
                }

                // Record and emit both the OVF op and the guard.
                self.cache.insert(key, postponed.pos);
                ctx.emit(postponed);
                return OptimizationResult::PassOn; // guard passes through
            } else {
                // Not a GUARD_NO_OVERFLOW: emit the postponed op now.
                ctx.emit(postponed);
            }
        }

        // pure.py: GUARD_NO_EXCEPTION — remove if last emitted was removed
        // (CALL_PURE was constant-folded or CSE'd away).
        if op.opcode == OpCode::GuardNoException {
            if self.last_emitted_was_removed {
                return OptimizationResult::Remove;
            }
            return OptimizationResult::PassOn;
        }

        // pure.py: RECORD_KNOWN_RESULT — record for later CALL_PURE lookup.
        if op.opcode == OpCode::RecordKnownResult {
            if op.num_args() >= 2 {
                let result = op.arg(0);
                let key = PureOpKey {
                    opcode: OpCode::CallPureI,
                    args: op.args[1..].to_vec(),
                };
                self.known_result_call_pure.push((key, result));
            }
            return OptimizationResult::Remove;
        }

        if op.opcode.is_always_pure() {
            // Constant folding: all args are constants → compute at opt time.
            if let Some(folded_value) = try_constant_fold_value(op, ctx) {
                let const_ref = ctx.find_or_record_constant_int(op.pos, folded_value);
                if const_ref != op.pos {
                    ctx.replace_op(op.pos, const_ref);
                }
                self.last_emitted_was_removed = true;
                return OptimizationResult::Remove;
            }

            if let Some(cached_ref) = self.lookup_imported_short_pure(op, ctx) {
                let cached_ref = ctx.get_replacement(cached_ref);
                ctx.replace_op(op.pos, cached_ref);
                self.last_emitted_was_removed = true;
                return OptimizationResult::Remove;
            }

            let key = PureOpKey::from_op(op);

            // CSE: exact same operation already computed?
            if let Some(cached_ref) = self.lookup_pure(&key) {
                let cached_ref = ctx.get_replacement(cached_ref);
                ctx.replace_op(op.pos, cached_ref);
                self.last_emitted_was_removed = true;
                return OptimizationResult::Remove;
            }

            self.cache.insert(key, op.pos);
            self.short_preamble_pure_ops.push(op.clone());
            return OptimizationResult::PassOn;
        }

        // CALL_PURE_* -> CSE or known_result lookup, then demote to CALL_*.
        if op.opcode.is_call_pure() {
            if let Some(cached_ref) = self.lookup_imported_short_pure(op, ctx) {
                let cached_ref = ctx.get_replacement(cached_ref);
                ctx.replace_op(op.pos, cached_ref);
                self.last_emitted_was_removed = true;
                return OptimizationResult::Remove;
            }

            let key = PureOpKey::from_op(op);

            // CSE: same call_pure with same args → reuse result.
            if let Some(cached_ref) = self.lookup_pure(&key) {
                let cached_ref = ctx.get_replacement(cached_ref);
                ctx.replace_op(op.pos, cached_ref);
                self.last_emitted_was_removed = true;
                return OptimizationResult::Remove;
            }

            // Check RECORD_KNOWN_RESULT cache.
            if let Some(result_ref) = self.lookup_known_result(&key) {
                let result_ref = ctx.get_replacement(result_ref);
                ctx.replace_op(op.pos, result_ref);
                self.last_emitted_was_removed = true;
                return OptimizationResult::Remove;
            }

            self.cache.insert(key, op.pos);
            // Track position for short preamble generation.
            self.call_pure_positions.push(ctx.new_operations.len());
            let new_op = self.demote_call_pure(op);
            if !Self::call_pure_can_raise(op) {
                self.short_preamble_pure_ops.push(new_op.clone());
            }
            return OptimizationResult::Emit(new_op);
        }

        // COND_CALL_VALUE_I/R → CSE like CALL_PURE, but skip arg[0]
        // (the condition value). pure.py: optimize_COND_CALL_VALUE_I.
        if op.opcode.is_cond_call_value() {
            // Build CSE key from args[1..] (skip condition value at arg[0])
            let key = PureOpKey {
                opcode: OpCode::CallPureI,
                args: op.args[1..].to_vec(),
            };

            if let Some(cached_ref) = self.lookup_pure(&key) {
                let cached_ref = ctx.get_replacement(cached_ref);
                ctx.replace_op(op.pos, cached_ref);
                self.last_emitted_was_removed = true;
                return OptimizationResult::Remove;
            }

            if let Some(result_ref) = self.lookup_known_result(&key) {
                let result_ref = ctx.get_replacement(result_ref);
                ctx.replace_op(op.pos, result_ref);
                self.last_emitted_was_removed = true;
                return OptimizationResult::Remove;
            }

            self.cache.insert(key, op.pos);
            self.call_pure_positions.push(ctx.new_operations.len());
            // Unlike CALL_PURE, COND_CALL_VALUE is NOT demoted — emit as-is.
            return OptimizationResult::Emit(op.clone());
        }

        // CALL_LOOPINVARIANT_* -> cache result, demote to CALL_*.
        if op.opcode.is_call_loopinvariant() {
            return self.handle_call_loopinvariant(op, ctx);
        }

        // pure.py: COND_CALL_VALUE_I/R — treated like CALL_PURE but
        // with args starting at index 1 (index 0 is the condition).
        if op.opcode == OpCode::CondCallValueI || op.opcode == OpCode::CondCallValueR {
            if op.num_args() >= 2 {
                // CSE key uses args[1..] (skip condition arg)
                let key = PureOpKey {
                    opcode: op.opcode,
                    args: op.args[1..].to_vec(),
                };
                if let Some(cached_ref) = self.lookup_pure(&key) {
                    let cached_ref = ctx.get_replacement(cached_ref);
                    ctx.replace_op(op.pos, cached_ref);
                    self.last_emitted_was_removed = true;
                    return OptimizationResult::Remove;
                }
                self.cache.insert(key, op.pos);
            }
            return OptimizationResult::PassOn;
        }

        OptimizationResult::PassOn
    }

    fn setup(&mut self) {
        let limit = self.cache.order.len();
        self.cache = RecentPureOps::new(limit);
        self.loopinvariant_cache.clear();
        self.postponed_op = None;
        self.call_pure_positions.clear();
        self.short_preamble_pure_ops.clear();
        self.short_preamble_loopinvariant_ops.clear();
        self.last_emitted_was_removed = false;
        self.known_result_call_pure.clear();
        // Note: extra_call_pure is NOT cleared on setup — it persists
        // across optimization runs (set by set_extra_call_pure before opt).
    }

    fn name(&self) -> &'static str {
        "pure"
    }

    /// pure.py: produce_potential_short_preamble_ops(sb)
    /// Add pure operations and CALL_PURE results to the short preamble.
    fn produce_potential_short_preamble_ops(&self, sb: &mut crate::shortpreamble::ShortBoxes) {
        for op in &self.short_preamble_pure_ops {
            if let Some(label_arg_idx) = sb.lookup_label_arg(op.pos) {
                sb.add_pure_op(label_arg_idx, op.clone());
            }
        }
        for op in &self.short_preamble_loopinvariant_ops {
            if let Some(label_arg_idx) = sb.lookup_label_arg(op.pos) {
                sb.add_loopinvariant_op(label_arg_idx, op.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::Optimizer;
    /// Helper: assign sequential positions to ops.
    fn assign_positions(ops: &mut [Op]) {
        for (i, op) in ops.iter_mut().enumerate() {
            op.pos = OpRef(i as u32);
        }
    }

    #[test]
    fn test_cse_int_add() {
        // i2 = int_add(i0, i1)
        // i3 = int_add(i0, i1)  <- should be eliminated, replaced by i2
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt.optimize(&ops);

        // Only the first IntAdd should remain.
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
    }

    #[test]
    fn test_cse_different_args_not_eliminated() {
        // i2 = int_add(i0, i1)
        // i3 = int_add(i0, i2)  <- different args, should NOT be eliminated
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(2)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt.optimize(&ops);

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_cse_commutative() {
        // i2 = int_add(i0, i1)
        // i3 = int_add(i1, i0)  <- commutative, should be eliminated
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::IntAdd, &[OpRef(1), OpRef(0)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt.optimize(&ops);

        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_cse_non_commutative() {
        // i2 = int_sub(i0, i1)
        // i3 = int_sub(i1, i0)  <- NOT commutative, should NOT be eliminated
        let mut ops = vec![
            Op::new(OpCode::IntSub, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::IntSub, &[OpRef(1), OpRef(0)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt.optimize(&ops);

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_cse_multiple_opcodes() {
        // i2 = int_add(i0, i1)
        // i3 = int_mul(i0, i1)  <- different opcode, should NOT be eliminated
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::IntMul, &[OpRef(0), OpRef(1)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt.optimize(&ops);

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_cse_three_duplicates() {
        // Use input arg OpRefs (100, 101) that don't collide with op positions (0, 1, 2).
        // i2 = int_add(i100, i101)
        // i3 = int_add(i100, i101)  <- eliminated
        // i4 = int_add(i100, i101)  <- eliminated
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt.optimize(&ops);

        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_call_pure_demoted_to_call() {
        // call_pure_i(args...) -> should become call_i(args...)
        let mut ops = vec![Op::new(OpCode::CallPureI, &[OpRef(0), OpRef(1)])];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt.optimize(&ops);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::CallI);
        assert_eq!(result[0].args.as_slice(), &[OpRef(0), OpRef(1)]);
    }

    #[test]
    fn test_call_pure_r_demoted() {
        let mut ops = vec![Op::new(OpCode::CallPureR, &[OpRef(0)])];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt.optimize(&ops);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::CallR);
    }

    #[test]
    fn test_non_pure_op_passes_through() {
        // setfield_gc is not pure, should pass through unchanged
        let mut ops = vec![Op::new(OpCode::SetfieldGc, &[OpRef(0), OpRef(1)])];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt.optimize(&ops);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
    }

    #[test]
    fn test_cse_unary_ops() {
        // i1 = int_neg(i0)
        // i2 = int_neg(i0)  <- should be eliminated
        let mut ops = vec![
            Op::new(OpCode::IntNeg, &[OpRef(0)]),
            Op::new(OpCode::IntNeg, &[OpRef(0)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt.optimize(&ops);

        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_cse_float_ops() {
        // f2 = float_add(f0, f1)
        // f3 = float_add(f0, f1)  <- should be eliminated
        let mut ops = vec![
            Op::new(OpCode::FloatAdd, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::FloatAdd, &[OpRef(0), OpRef(1)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt.optimize(&ops);

        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_cache_eviction() {
        // Force a tiny cache so eviction behavior is deterministic even if
        // the production default changes.
        let mut ops = Vec::new();
        for i in 0..17u32 {
            ops.push(Op::new(OpCode::IntAdd, &[OpRef(i), OpRef(i + 100)]));
        }
        // Re-insert op #0: same args as ops[0]
        ops.push(Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(100)]));
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure {
            cache: RecentPureOps::new(16),
            loopinvariant_cache: HashMap::new(),
            postponed_op: None,
            call_pure_positions: Vec::new(),
            short_preamble_pure_ops: Vec::new(),
            short_preamble_loopinvariant_ops: Vec::new(),
            last_emitted_was_removed: false,
            known_result_call_pure: Vec::new(),
            extra_call_pure: Vec::new(),
        }));
        let result = opt.optimize(&ops);

        // All 17 unique ops should be emitted, plus the re-inserted one
        // (since the first was evicted from the LRU cache of size 16).
        assert_eq!(result.len(), 18);
    }

    #[test]
    fn test_cse_with_forwarding() {
        // Test that CSE works correctly when OpRef forwarding is involved.
        let mut ctx = OptContext::new(10);
        let mut pass = OptPure::new();

        // Simulate: op0 = int_add(a, b)
        let op0 = Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]);
        let mut op0 = op0;
        op0.pos = OpRef(2);
        let result0 = pass.propagate_forward(&op0, &mut ctx);
        assert!(matches!(result0, OptimizationResult::PassOn));

        // Simulate: op1 = int_add(a, b) with same args
        let op1 = Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]);
        let mut op1 = op1;
        op1.pos = OpRef(3);
        let result1 = pass.propagate_forward(&op1, &mut ctx);
        assert!(matches!(result1, OptimizationResult::Remove));
    }

    #[test]
    fn test_pure_op_key_equality() {
        let key1 = PureOpKey {
            opcode: OpCode::IntAdd,
            args: vec![OpRef(0), OpRef(1)],
        };
        let key2 = PureOpKey {
            opcode: OpCode::IntAdd,
            args: vec![OpRef(0), OpRef(1)],
        };
        let key3 = PureOpKey {
            opcode: OpCode::IntAdd,
            args: vec![OpRef(1), OpRef(0)],
        };
        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_commutative_xor() {
        let mut ops = vec![
            Op::new(OpCode::IntXor, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::IntXor, &[OpRef(1), OpRef(0)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt.optimize(&ops);

        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_commutative_int_and() {
        let mut ops = vec![
            Op::new(OpCode::IntAnd, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::IntAnd, &[OpRef(1), OpRef(0)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt.optimize(&ops);

        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_comparison_cse() {
        // i2 = int_lt(i0, i1)
        // i3 = int_lt(i0, i1)  <- should be eliminated
        let mut ops = vec![
            Op::new(OpCode::IntLt, &[OpRef(0), OpRef(1)]),
            Op::new(OpCode::IntLt, &[OpRef(0), OpRef(1)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt.optimize(&ops);

        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_call_pure_f_n_demoted() {
        let mut ops = vec![
            Op::new(OpCode::CallPureF, &[OpRef(0)]),
            Op::new(OpCode::CallPureN, &[OpRef(0)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt.optimize(&ops);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::CallF);
        assert_eq!(result[1].opcode, OpCode::CallN);
    }

    #[test]
    fn test_mixed_pure_and_non_pure() {
        // Mix of pure and non-pure operations, only duplicated pure ops get CSE'd.
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]), // pure, kept
            Op::new(OpCode::SetfieldGc, &[OpRef(0), OpRef(1)]), // not pure, kept
            Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]), // pure duplicate, eliminated
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt.optimize(&ops);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
        assert_eq!(result[1].opcode, OpCode::SetfieldGc);
    }

    #[test]
    fn test_call_loopinvariant_cse() {
        // Two identical CALL_LOOPINVARIANT_I calls → second eliminated.
        let mut ops = vec![
            Op::new(OpCode::CallLoopinvariantI, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::CallLoopinvariantI, &[OpRef(100), OpRef(101)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt.optimize(&ops);

        // Only the first call should remain, demoted to CallI.
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::CallI);
    }

    #[test]
    fn test_call_loopinvariant_different_args() {
        // CALL_LOOPINVARIANT_I with different args → both kept.
        let mut ops = vec![
            Op::new(OpCode::CallLoopinvariantI, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::CallLoopinvariantI, &[OpRef(100), OpRef(102)]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt.optimize(&ops);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::CallI);
        assert_eq!(result[1].opcode, OpCode::CallI);
    }

    #[test]
    fn test_call_loopinvariant_all_types() {
        for (loopinv_op, expected_op) in [
            (OpCode::CallLoopinvariantI, OpCode::CallI),
            (OpCode::CallLoopinvariantR, OpCode::CallR),
            (OpCode::CallLoopinvariantF, OpCode::CallF),
            (OpCode::CallLoopinvariantN, OpCode::CallN),
        ] {
            let mut ops = vec![Op::new(loopinv_op, &[OpRef(0)])];
            assign_positions(&mut ops);

            let mut opt = Optimizer::new();
            opt.add_pass(Box::new(OptPure::new()));
            let result = opt.optimize(&ops);

            assert_eq!(result.len(), 1);
            assert_eq!(result[0].opcode, expected_op);
        }
    }

    #[test]
    fn test_call_loopinvariant_no_eviction() {
        // Unlike pure CSE (LRU limit 16), loop-invariant cache has no eviction.
        // Create 20 unique calls, then re-check the first one.
        let mut ops = Vec::new();
        for i in 0..20u32 {
            ops.push(Op::new(
                OpCode::CallLoopinvariantI,
                &[OpRef(i + 100), OpRef(200)],
            ));
        }
        // Re-insert call #0: same args as ops[0]
        ops.push(Op::new(
            OpCode::CallLoopinvariantI,
            &[OpRef(100), OpRef(200)],
        ));
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt.optimize(&ops);

        // 20 unique calls + the duplicate (#0) should be eliminated → 20 total
        assert_eq!(result.len(), 20);
    }

    #[test]
    fn test_call_loopinvariant_mixed_with_pure() {
        // Loop-invariant and pure CSE should coexist.
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]), // pure
            Op::new(OpCode::CallLoopinvariantI, &[OpRef(200), OpRef(201)]), // loopinvariant
            Op::new(OpCode::IntAdd, &[OpRef(100), OpRef(101)]), // pure dup → removed
            Op::new(OpCode::CallLoopinvariantI, &[OpRef(200), OpRef(201)]), // loopinvariant dup → removed
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt.optimize(&ops);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
        assert_eq!(result[1].opcode, OpCode::CallI);
    }

    #[test]
    fn test_constant_fold_int_add() {
        // IntAdd(const(3), const(4)) → eliminated, result = const(7)
        let mut ops = vec![
            Op::new(OpCode::IntAdd, &[OpRef(10_000), OpRef(10_001)]),
            // Use the result in a guard to prevent dead code elimination
            Op::new(OpCode::Finish, &[OpRef(0)]),
        ];
        assign_positions(&mut ops);

        let mut constants = std::collections::HashMap::new();
        constants.insert(10_000, 3i64);
        constants.insert(10_001, 4i64);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt.optimize_with_constants(&ops, &mut constants);

        // IntAdd should be folded away (only Finish remains)
        assert_eq!(result.len(), 1, "IntAdd(3,4) should be constant-folded");
        assert_eq!(result[0].opcode, OpCode::Finish);
    }

    #[test]
    fn test_constant_fold_int_lt() {
        // IntLt(const(3), const(5)) → const(1) (true)
        let mut ops = vec![
            Op::new(OpCode::IntLt, &[OpRef(10_000), OpRef(10_001)]),
            Op::new(OpCode::Finish, &[OpRef(0)]),
        ];
        assign_positions(&mut ops);

        let mut constants = std::collections::HashMap::new();
        constants.insert(10_000, 3i64);
        constants.insert(10_001, 5i64);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt.optimize_with_constants(&ops, &mut constants);

        assert_eq!(result.len(), 1, "IntLt(3,5) should be constant-folded");
    }

    #[test]
    fn test_ovf_postponement_cse() {
        // INT_ADD_OVF(a, b) + GUARD_NO_OVERFLOW
        // then same INT_ADD_OVF(a, b) + GUARD_NO_OVERFLOW → CSE'd away
        let mut ops = vec![
            Op::new(OpCode::IntAddOvf, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::GuardNoOverflow, &[]),
            Op::new(OpCode::IntAddOvf, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::GuardNoOverflow, &[]),
            Op::new(OpCode::Finish, &[]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt.optimize(&ops);

        // First pair stays, second pair CSE'd → 3 ops total
        let ovf_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::IntAddOvf)
            .count();
        assert_eq!(ovf_count, 1, "duplicate OVF should be CSE'd");
    }

    #[test]
    fn test_ovf_constant_fold() {
        // INT_ADD_OVF(const(3), const(4)) + GUARD_NO_OVERFLOW → both removed
        let mut ops = vec![
            Op::new(OpCode::IntAddOvf, &[OpRef(10_000), OpRef(10_001)]),
            Op::new(OpCode::GuardNoOverflow, &[]),
            Op::new(OpCode::Finish, &[]),
        ];
        assign_positions(&mut ops);

        let mut constants = std::collections::HashMap::new();
        constants.insert(10_000, 3i64);
        constants.insert(10_001, 4i64);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt.optimize_with_constants(&ops, &mut constants);

        // Both OVF and guard should be folded away
        let ovf_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::IntAddOvf)
            .count();
        assert_eq!(ovf_count, 0, "OVF(3,4) should be constant-folded");
    }

    #[test]
    fn test_guard_no_exception_after_removed_call_pure() {
        // CALL_PURE_I(same args) × 2 → second removed → GUARD_NO_EXCEPTION after removed
        let mut ops = vec![
            Op::new(OpCode::CallPureI, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::GuardNoException, &[]),
            Op::new(OpCode::CallPureI, &[OpRef(100), OpRef(101)]),
            Op::new(OpCode::GuardNoException, &[]), // should be removed
            Op::new(OpCode::Finish, &[]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt.optimize(&ops);

        // Second CALL_PURE → removed (CSE), its GUARD_NO_EXCEPTION → removed
        let gne_count = result
            .iter()
            .filter(|o| o.opcode == OpCode::GuardNoException)
            .count();
        assert_eq!(
            gne_count, 1,
            "GUARD_NO_EXCEPTION after removed CALL_PURE should be eliminated"
        );
    }

    #[test]
    fn test_pure_and_pure_from_args() {
        let mut pass = OptPure::new();

        // Manually record a pure operation via the API
        let mut op = Op::new(OpCode::IntAdd, &[OpRef(10), OpRef(20)]);
        op.pos = OpRef(0);
        pass.pure(&op);

        // Should find it via get_pure_result
        let lookup_op = Op::new(OpCode::IntAdd, &[OpRef(10), OpRef(20)]);
        assert!(pass.get_pure_result(&lookup_op).is_some());

        // pure_from_args
        pass.pure_from_args(OpCode::IntMul, &[OpRef(30), OpRef(40)], OpRef(5));
        let mut lookup_mul = Op::new(OpCode::IntMul, &[OpRef(30), OpRef(40)]);
        lookup_mul.pos = OpRef(99);
        assert!(pass.get_pure_result(&lookup_mul).is_some());
    }

    #[test]
    fn test_extra_call_pure() {
        let mut pass = OptPure::new();

        // Inject extra_call_pure from a previous loop iteration
        let args = vec![OpRef(100), OpRef(101)];
        pass.set_extra_call_pure(vec![(args.clone(), OpRef(50))]);

        // The lookup should find the injected result
        let key = super::PureOpKey {
            opcode: OpCode::CallPureI,
            args,
        };
        assert_eq!(pass.lookup_known_result(&key), Some(OpRef(50)));
    }

    #[test]
    fn test_imported_short_pure_result_replays_into_pure_cache() {
        let mut pass = OptPure::new();
        let mut ctx = OptContext::with_num_inputs(6, 0);
        ctx.make_constant(OpRef(10), majit_ir::Value::Int(7));
        ctx.imported_short_pure_ops.push(crate::ImportedShortPureOp {
            opcode: OpCode::IntAdd,
            descr_idx: None,
            args: vec![
                crate::ImportedShortPureArg::OpRef(OpRef(0)),
                crate::ImportedShortPureArg::Const(majit_ir::Value::Int(7)),
            ],
            result: OpRef(1),
        });

        pass.setup();

        let mut op = Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(10)]);
        op.pos = OpRef(2);
        let result = pass.propagate_forward(&op, &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(ctx.get_replacement(OpRef(2)), OpRef(1));
    }

    #[test]
    fn test_imported_short_call_pure_result_replays_into_pure_cache() {
        let mut pass = OptPure::new();
        let mut ctx = OptContext::with_num_inputs(8, 0);
        ctx.make_constant(OpRef(10), majit_ir::Value::Int(0x1234));
        let call_descr = majit_ir::descr::make_call_descr_full(
            77,
            vec![majit_ir::Type::Int, majit_ir::Type::Int],
            majit_ir::Type::Int,
            8,
            majit_ir::EffectInfo::elidable(),
        );
        ctx.imported_short_pure_ops.push(crate::ImportedShortPureOp {
            opcode: OpCode::CallPureI,
            descr_idx: Some(77),
            args: vec![
                crate::ImportedShortPureArg::Const(majit_ir::Value::Int(0x1234)),
                crate::ImportedShortPureArg::OpRef(OpRef(0)),
            ],
            result: OpRef(1),
        });

        pass.setup();

        let mut op = Op::new(OpCode::CallPureI, &[OpRef(10), OpRef(0)]);
        op.pos = OpRef(2);
        op.descr = Some(call_descr);
        let result = pass.propagate_forward(&op, &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(ctx.get_replacement(OpRef(2)), OpRef(1));
    }

    #[test]
    fn test_short_preamble_collects_pure_op_candidate() {
        let mut pass = OptPure::new();
        let mut ctx = OptContext::with_num_inputs(4, 0);
        pass.setup();

        let mut op = Op::new(OpCode::IntAdd, &[OpRef(0), OpRef(1)]);
        op.pos = OpRef(2);
        let result = pass.propagate_forward(&op, &mut ctx);
        assert!(matches!(result, OptimizationResult::PassOn));

        let mut sb = crate::shortpreamble::ShortBoxes::with_label_args(&[OpRef(2)]);
        pass.produce_potential_short_preamble_ops(&mut sb);
        let collected = sb.produced_ops();
        assert_eq!(collected.len(), 1);
        assert!(matches!(
            collected[0].1.kind,
            crate::shortpreamble::PreambleOpKind::Pure
        ));
        assert_eq!(collected[0].1.preamble_op.opcode, OpCode::IntAdd);
    }

    #[test]
    fn test_short_preamble_collects_non_raising_call_pure_candidate() {
        let mut pass = OptPure::new();
        let mut ctx = OptContext::with_num_inputs(6, 0);
        pass.setup();

        let mut op = Op::new(OpCode::CallPureI, &[OpRef(100), OpRef(0), OpRef(1)]);
        op.pos = OpRef(2);
        op.descr = Some(majit_ir::descr::make_call_descr(
            vec![majit_ir::Type::Int, majit_ir::Type::Int, majit_ir::Type::Int],
            majit_ir::Type::Int,
            majit_ir::EffectInfo::elidable(),
        ));
        let result = pass.propagate_forward(&op, &mut ctx);
        match result {
            OptimizationResult::Emit(emitted) => assert_eq!(emitted.opcode, OpCode::CallI),
            other => panic!("expected emitted demoted call, got {other:?}"),
        }

        let mut sb = crate::shortpreamble::ShortBoxes::with_label_args(&[OpRef(2)]);
        pass.produce_potential_short_preamble_ops(&mut sb);
        let collected = sb.produced_ops();
        assert_eq!(collected.len(), 1);
        assert!(matches!(
            collected[0].1.kind,
            crate::shortpreamble::PreambleOpKind::Pure
        ));
        assert_eq!(collected[0].1.preamble_op.opcode, OpCode::CallPureI);
    }

    #[test]
    fn test_short_preamble_collects_loopinvariant_candidate() {
        let mut pass = OptPure::new();
        let mut ctx = OptContext::with_num_inputs(6, 0);
        pass.setup();

        let mut op = Op::new(OpCode::CallLoopinvariantI, &[OpRef(100), OpRef(0)]);
        op.pos = OpRef(2);
        op.descr = Some(majit_ir::descr::make_call_descr(
            vec![majit_ir::Type::Int, majit_ir::Type::Int],
            majit_ir::Type::Int,
            majit_ir::EffectInfo::elidable(),
        ));
        let result = pass.propagate_forward(&op, &mut ctx);
        match result {
            OptimizationResult::Emit(emitted) => assert_eq!(emitted.opcode, OpCode::CallI),
            other => panic!("expected emitted demoted call, got {other:?}"),
        }

        let mut sb = crate::shortpreamble::ShortBoxes::with_label_args(&[OpRef(2)]);
        pass.produce_potential_short_preamble_ops(&mut sb);
        let collected = sb.produced_ops();
        assert_eq!(collected.len(), 1);
        assert!(matches!(
            collected[0].1.kind,
            crate::shortpreamble::PreambleOpKind::LoopInvariant
        ));
        assert_eq!(
            collected[0].1.preamble_op.opcode,
            OpCode::CallLoopinvariantI
        );
    }

    #[test]
    fn test_lookup1_lookup2() {
        let mut pass = OptPure::new();

        // Record via pure_from_args
        pass.pure_from_args(OpCode::IntAdd, &[OpRef(10), OpRef(20)], OpRef(30));

        // lookup2 should find it
        assert!(
            pass.lookup2(OpCode::IntAdd, OpRef(10), OpRef(20), false)
                .is_some()
        );
        // lookup2 with commutative should find swapped
        assert!(
            pass.lookup2(OpCode::IntAdd, OpRef(20), OpRef(10), true)
                .is_some()
        );
        // Non-commutative swapped should NOT find it
        assert!(
            pass.lookup2(OpCode::IntAdd, OpRef(20), OpRef(10), false)
                .is_none()
        );

        // lookup1 for a unary op
        pass.pure_from_args(OpCode::IntNeg, &[OpRef(10)], OpRef(40));
        assert!(pass.lookup1(OpCode::IntNeg, OpRef(10)).is_some());
        assert!(pass.lookup1(OpCode::IntNeg, OpRef(99)).is_none());
    }

    #[test]
    fn test_cond_call_value_cse() {
        // COND_CALL_VALUE_I(cond, func, arg) → CSE using args[1..]
        // A second COND_CALL_VALUE_I with same func+arg should reuse result.
        let mut ops = vec![
            Op::new(
                OpCode::CondCallValueI,
                &[OpRef(100), OpRef(200), OpRef(300)],
            ),
            Op::new(
                OpCode::CondCallValueI,
                &[OpRef(101), OpRef(200), OpRef(300)],
            ),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt.optimize(&ops);

        // First COND_CALL_VALUE emitted, second removed by CSE
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::CondCallValueI);
    }
}
