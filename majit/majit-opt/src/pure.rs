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
}

/// The OptPure optimization pass.
///
/// For pure operations (is_always_pure), checks if the same operation was
/// computed before. If yes, replaces the current op with the cached result
/// (CSE). If no, records the operation for future lookups.
///
/// Also handles:
/// - CALL_PURE -> CALL demotion when arguments aren't all constant.
/// - CALL_LOOPINVARIANT_* caching: the result of a loop-invariant call is
///   cached for the entire loop iteration (unlike pure ops, no LRU eviction).
///   Translated from rpython/jit/metainterp/optimizeopt/pure.py
///   optimize_CALL_LOOPINVARIANT_I/R/F/N.
pub struct OptPure {
    cache: RecentPureOps,
    /// Per-loop-iteration cache for CALL_LOOPINVARIANT_* results.
    /// Key: (func_ptr_opref, arg0, arg1, ...) → result OpRef.
    /// This cache persists across the whole loop body without eviction,
    /// as loop-invariant calls produce the same result for one iteration.
    loopinvariant_cache: HashMap<PureOpKey, OpRef>,
}

impl OptPure {
    pub fn new() -> Self {
        OptPure {
            // Aheui traces routinely repeat the same comparisons hundreds of
            // ops apart; a tiny ring buffer misses most of those CSE wins.
            cache: RecentPureOps::new(4096),
            loopinvariant_cache: HashMap::new(),
        }
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
        let call_opcode = match op.opcode {
            OpCode::CallPureI => OpCode::CallI,
            OpCode::CallPureR => OpCode::CallR,
            OpCode::CallPureF => OpCode::CallF,
            OpCode::CallPureN => OpCode::CallN,
            _ => unreachable!(),
        };
        let mut new_op = op.clone();
        new_op.opcode = call_opcode;
        OptimizationResult::Emit(new_op)
    }

    /// Handle CALL_LOOPINVARIANT_*: cache the result for the loop iteration.
    ///
    /// If the same call (same function + arguments) was already seen, replace
    /// with the cached result. Otherwise, emit and cache the result.
    /// The call is demoted to a plain CALL_* for the backend.
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

        let call_opcode = OpCode::call_for_type(op.result_type());
        let mut new_op = op.clone();
        new_op.opcode = call_opcode;
        OptimizationResult::Emit(new_op)
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
        OpCode::IntAdd => a.checked_add(b)?,
        OpCode::IntSub => a.checked_sub(b)?,
        OpCode::IntMul => a.checked_mul(b)?,
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
            if (r != 0) && ((r ^ b) < 0) { Some(q - 1) } else { Some(q) }
        }?,
        OpCode::IntMod if b != 0 => {
            let r = a % b;
            if (r != 0) && ((r ^ b) < 0) { Some(r + b) } else { Some(r) }
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
        if op.opcode.is_always_pure() {
            // RPython pure.py: constant folding — if all args are constants,
            // compute the result at optimization time and replace with constant.
            if let Some(folded_value) = try_constant_fold_value(op, ctx) {
                // Find or create a constant OpRef for this value.
                // Check if any existing constant matches.
                let const_ref = ctx.find_or_record_constant_int(op.pos, folded_value);
                if const_ref != op.pos {
                    ctx.replace_op(op.pos, const_ref);
                }
                return OptimizationResult::Remove;
            }

            let key = PureOpKey::from_op(op);

            // CSE: did we do the exact same operation already?
            if let Some(cached_ref) = self.lookup_pure(&key) {
                let cached_ref = ctx.get_replacement(cached_ref);
                ctx.replace_op(op.pos, cached_ref);
                return OptimizationResult::Remove;
            }

            // Record this operation for future lookups.
            self.cache.insert(key, op.pos);
            return OptimizationResult::PassOn;
        }

        // CALL_PURE_* -> demote to plain CALL_*
        if op.opcode.is_call_pure() {
            return self.handle_call_pure(op);
        }

        // CALL_LOOPINVARIANT_* -> cache result, demote to CALL_*
        if op.opcode.is_call_loopinvariant() {
            return self.handle_call_loopinvariant(op, ctx);
        }

        OptimizationResult::PassOn
    }

    fn name(&self) -> &'static str {
        "pure"
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
}
