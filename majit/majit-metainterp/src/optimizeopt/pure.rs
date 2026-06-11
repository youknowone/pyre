/// Pure operation optimization (Common Subexpression Elimination).
///
/// Translated from rpython/jit/metainterp/optimizeopt/pure.py.
///
/// When the same pure operation is seen again with the same arguments,
/// the cached result is returned instead of recomputing.
use majit_ir::{GcRef, Op, OpCode, OpRef, Value};

use crate::r#box::BoxRef;
use crate::optimizeopt::info::{PreambleOp, PtrInfoExt};
use crate::optimizeopt::{OptContext, Optimization, OptimizationResult};

/// pure.py:104,204-210: extra_call_pure entry.
/// RPython stores AbstractResOp (or PreambleOp) directly in the list.
/// isinstance(old_op, PreambleOp) check → force_op_from_preamble → replace.
#[derive(Clone, Debug)]
enum ExtraCallPureEntry {
    Direct { key: PureOpKey, result: OpRef },
    Preamble { key: PureOpKey, pop: PreambleOp },
}

/// Key for looking up a previously computed pure operation.
///
/// Identifies an operation by its opcode, argument OpRefs, and descriptor.
///
/// RPython's optimizeopt/pure.py includes the descriptor in pure-op identity
/// checks for operations like GETFIELD_GC_PURE_*; otherwise distinct immutable
/// fields on the same object can be incorrectly CSE'd together.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct PureOpKey {
    opcode: OpCode,
    args: Vec<OpRef>,
    descr_identity: Option<usize>,
}

impl PureOpKey {
    fn from_op(op: &Op) -> Self {
        PureOpKey {
            opcode: op.opcode,
            args: op.getarglist().iter().map(|a| a.to_opref()).collect(),
            descr_identity: op.getdescr().as_ref().map(majit_ir::descr::descr_identity),
        }
    }

    /// pure.py:185 parity: COND_CALL_VALUE uses start_index=1 to skip arg[0].
    /// The key uses CALL_PURE opcode so that COND_CALL_VALUE and CALL_PURE
    /// share the same cache namespace (RPython uses the same dict).
    fn from_call_op(op: &Op, start_index: usize) -> Self {
        PureOpKey {
            opcode: OpCode::call_pure_for_type(op.result_type()),
            args: op.getarglist()[start_index..]
                .iter()
                .map(|a| a.to_opref())
                .collect(),
            descr_identity: op.getdescr().as_ref().map(majit_ir::descr::descr_identity),
        }
    }
}

/// pure.py:213: known_result_call_pure entry.
/// RPython stores the full RECORD_KNOWN_RESULT op and compares by descr +
/// _same_args(known_op, query_op, 1, start_index). We pre-extract the
/// fields to avoid storing a dummy PureOpKey with an opcode.
#[derive(Clone, Debug)]
struct KnownResultEntry {
    descr_identity: Option<usize>,
    /// args[1..] from the RECORD_KNOWN_RESULT op (the call arguments).
    args: Vec<OpRef>,
    /// arg(0) from the RECORD_KNOWN_RESULT op (the known result).
    result: OpRef,
}

/// pure.py:36-95 RecentPureOps — fixed-size ring buffer with linear scan.
///
/// RPython uses a flat array of Op references, scanned linearly on lookup.
/// At limit=16 (pureop_historylength), linear scan beats HashMap because:
/// - No hashing overhead or Vec<OpRef> allocation per lookup
/// - Cache-friendly sequential memory access
/// - Typical hit is within first few entries
struct RecentPureOps {
    /// Ring buffer of (key, result) pairs. None = empty slot.
    lst: Vec<Option<(PureOpKey, OpRef)>>,
    next_index: usize,
}

impl RecentPureOps {
    fn new(limit: usize) -> Self {
        RecentPureOps {
            lst: vec![None; limit],
            next_index: 0,
        }
    }

    /// pure.py:41-48 — add(op): record a pure operation result.
    fn insert(&mut self, key: PureOpKey, result: OpRef) {
        self.lst[self.next_index] = Some((key, result));
        self.next_index += 1;
        if self.next_index >= self.lst.len() {
            self.next_index = 0;
        }
    }

    /// pure.py:81-95 lookup(self, optimizer, op, commutative=False).
    ///
    /// Dispatches to `lookup1` / `lookup2` by arg count; any other
    /// numargs hits the upstream `assert False` because OptPure only
    /// runs against ALWAYS_PURE ops with 1 or 2 args (`LOAD_EFFECTIVE_ADDRESS`
    /// has 4 args but is emitted by `rewrite.py` after OptPure).
    ///
    /// `same_box(query, stored)` mirrors RPython's `box.same_box(other)`
    /// (history.py:204-205, :244): identity for non-Const Boxes (the
    /// caller is expected to apply `get_box_replacement` to both sides
    /// first), value equality for Const subclasses. Without this hook,
    /// raw `OpRef ==` would miss CSE for two distinct constant slots
    /// holding the same value — a divergence introduced by variant-
    /// aware OpRef Eq.
    fn lookup<F: Fn(OpRef, OpRef) -> bool>(
        &self,
        key: &PureOpKey,
        same_box: F,
        commutative: bool,
    ) -> Option<OpRef> {
        match key.args.len() {
            1 => self.lookup1(key.opcode, key.args[0], key.descr_identity, same_box),
            2 => self.lookup2(
                key.opcode,
                key.args[0],
                key.args[1],
                key.descr_identity,
                commutative,
                same_box,
            ),
            _ => panic!(
                "RecentPureOps::lookup: numargs must be 1 or 2, got {}",
                key.args.len()
            ),
        }
    }

    /// pure.py:57-65 lookup1(opt, box0, descr).
    ///
    /// RPython: `box0.same_box(get_box_replacement(op.getarg(0)))`.
    /// `same_box` is identity for non-constants, value equality for constants.
    /// The `same_box` callback combines get_box_replacement + value comparison.
    fn lookup1(
        &self,
        opcode: OpCode,
        arg0: OpRef,
        descr_identity: Option<usize>,
        same_box: impl Fn(OpRef, OpRef) -> bool,
    ) -> Option<OpRef> {
        for entry in &self.lst {
            let Some((k, result)) = entry else { break };
            if k.opcode != opcode || k.args.len() != 1 {
                continue;
            }
            if k.descr_identity != descr_identity {
                continue;
            }
            // pure.py:62 — box0.same_box(get_box_replacement(op.getarg(0)))
            if same_box(arg0, k.args[0]) {
                return Some(*result);
            }
        }
        None
    }

    /// pure.py:67-79 lookup2(opt, box0, box1, descr, commutative).
    ///
    /// `same_box` applies get_box_replacement internally and uses
    /// value equality for constants (history.py:204-205 Const.same_box).
    fn lookup2(
        &self,
        opcode: OpCode,
        arg0: OpRef,
        arg1: OpRef,
        descr_identity: Option<usize>,
        commutative: bool,
        same_box: impl Fn(OpRef, OpRef) -> bool,
    ) -> Option<OpRef> {
        for entry in &self.lst {
            let Some((k, result)) = entry else { break };
            if k.opcode != opcode || k.args.len() != 2 {
                continue;
            }
            if k.descr_identity != descr_identity {
                continue;
            }
            // pure.py:72-75 — same_box includes get_box_replacement
            if (same_box(arg0, k.args[0]) && same_box(arg1, k.args[1]))
                || (commutative && same_box(arg1, k.args[0]) && same_box(arg0, k.args[1]))
            {
                return Some(*result);
            }
        }
        None
    }
}

struct RecentPureOpTable {
    buckets: Vec<Option<RecentPureOps>>,
    history_length: usize,
}

impl RecentPureOpTable {
    fn new(limit: usize) -> Self {
        let bucket_count = Self::bucket_count();
        let buckets = std::iter::repeat_with(|| None).take(bucket_count).collect();
        RecentPureOpTable {
            buckets,
            history_length: limit,
        }
    }

    fn bucket_count() -> usize {
        (OpCode::LoadEffectiveAddress as usize - OpCode::IntAdd as usize + 1) + 3
    }

    fn bucket_index(opcode: OpCode) -> Option<usize> {
        if opcode.is_ovf() {
            return Some(opcode as usize - OpCode::IntAddOvf as usize);
        }
        match opcode {
            OpCode::GetfieldGcPureI => {
                Some(OpCode::LoadEffectiveAddress as usize - OpCode::IntAdd as usize + 1)
            }
            OpCode::GetfieldGcPureR => {
                Some(OpCode::LoadEffectiveAddress as usize - OpCode::IntAdd as usize + 2)
            }
            OpCode::GetfieldGcPureF => {
                Some(OpCode::LoadEffectiveAddress as usize - OpCode::IntAdd as usize + 3)
            }
            _ if opcode.is_always_pure() => Some(opcode as usize - OpCode::IntAdd as usize),
            _ => None,
        }
    }

    fn bucket(&self, opcode: OpCode) -> Option<&RecentPureOps> {
        let idx = Self::bucket_index(opcode)?;
        self.buckets.get(idx)?.as_ref()
    }

    fn bucket_mut(&mut self, opcode: OpCode) -> Option<&mut RecentPureOps> {
        let idx = Self::bucket_index(opcode)?;
        if self.buckets[idx].is_none() {
            self.buckets[idx] = Some(RecentPureOps::new(self.history_length));
        }
        self.buckets[idx].as_mut()
    }

    fn lookup<F: Fn(OpRef, OpRef) -> bool>(
        &self,
        key: &PureOpKey,
        same_box: F,
        commutative: bool,
    ) -> Option<OpRef> {
        self.bucket(key.opcode)?.lookup(key, same_box, commutative)
    }

    fn insert(&mut self, key: PureOpKey, result: OpRef) {
        if let Some(bucket) = self.bucket_mut(key.opcode) {
            bucket.insert(key, result);
        }
    }

    fn lookup1(
        &self,
        opcode: OpCode,
        arg0: OpRef,
        descr_identity: Option<usize>,
        same_box: impl Fn(OpRef, OpRef) -> bool,
    ) -> Option<OpRef> {
        self.bucket(opcode)
            .and_then(|bucket| bucket.lookup1(opcode, arg0, descr_identity, same_box))
    }

    fn lookup2(
        &self,
        opcode: OpCode,
        arg0: OpRef,
        arg1: OpRef,
        descr_identity: Option<usize>,
        commutative: bool,
        same_box: impl Fn(OpRef, OpRef) -> bool,
    ) -> Option<OpRef> {
        self.bucket(opcode).and_then(|bucket| {
            bucket.lookup2(opcode, arg0, arg1, descr_identity, commutative, same_box)
        })
    }

    fn history_length(&self) -> usize {
        self.history_length
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
/// - OVF operation postponement (INT_ADD_OVF etc. are deferred until GUARD_NO_OVERFLOW).
/// - GUARD_NO_EXCEPTION removal after eliminated CALL_PURE.
/// - RECORD_KNOWN_RESULT for pre-recorded call_pure results.
pub struct OptPure {
    cache: RecentPureOpTable,
    /// Postponed OVF operation: INT_ADD_OVF, INT_SUB_OVF, INT_MUL_OVF.
    /// pure.py: postponed_op — deferred until GUARD_NO_OVERFLOW is seen.
    postponed_op: Option<Op>,
    /// Bound `BoxRef` of `postponed_op`, captured from the live `OpRc` at
    /// postponement. The OVF op is `Remove`d before emit, so its head box
    /// is never bound by the emit path; capturing it here (where the op
    /// object is live) gives `make_equal_to` a bound receiver without an
    /// `materialize_box_at` round-trip through the opref.
    postponed_box: Option<crate::r#box::BoxRef>,
    /// Indices into new_operations of emitted CALL_PURE ops.
    /// pure.py: call_pure_positions — tracked for short preamble generation.
    call_pure_positions: Vec<usize>,
    /// RPython pure.py / shortpreamble.py: pure ops that phase 2 should be
    /// able to reproduce from the preamble via optimizer state, not by
    /// textual body replay.
    short_preamble_pure_ops: Vec<Op>,
    /// Whether the last emitted operation was removed (for GUARD_NO_EXCEPTION elimination).
    /// pure.py: last_emitted_operation is REMOVED check.
    last_emitted_was_removed: bool,
    /// Pre-recorded CALL_PURE results from RECORD_KNOWN_RESULT.
    /// pure.py: known_result_call_pure — stores the full RECORD_KNOWN_RESULT op.
    /// RPython lookup: descr + _same_args(known_op, op, 1, start_index).
    /// We store (descr_identity, args_from_1, result) — no opcode comparison.
    known_result_call_pure: Vec<KnownResultEntry>,
    /// pure.py:104: extra_call_pure — CALL_PURE results from the previous
    /// loop iteration and preamble import. May contain PreambleOp entries
    /// (RPython isinstance check → force_op_from_preamble → replace in-place).
    extra_call_pure: Vec<ExtraCallPureEntry>,
    /// optimizer.py: call_pure_results passed into propagate_all_forward.
    /// RPython keys are lists of constant boxes (value-based equality).
    /// Keys are the constant Values that _can_optimize_call_pure builds.
    call_pure_results: crate::optimizeopt::vec_assoc::VecAssoc<Vec<Value>, Value>,
    /// shortpreamble.py:124-126: PureOp.produce_op stores PreambleOp in
    /// optpure's cache. In majit, PreambleOp entries stored here are
    /// searched with forwarding-aware matching (force_preamble_op pattern).
    /// Body CSE uses `RecentPureOpTable` (the `cache` field above) — a
    /// per-opcode bucket array, not a hashmap.
    preamble_pure_ops: Vec<PreamblePureEntry>,
}

/// shortpreamble.py:124-126: PreambleOp stored in OptPure for always-pure ops.
/// Searched with forwarding-aware matching during body optimization.
#[derive(Clone, Debug)]
struct PreamblePureEntry {
    opcode: OpCode,
    args: Vec<OpRef>,
    descr_identity: Option<usize>,
    pop: PreambleOp,
    /// Forced flag: after first match, replaced with Direct result.
    forced_result: Option<OpRef>,
}

impl OptPure {
    pub fn new() -> Self {
        OptPure {
            cache: RecentPureOpTable::new(crate::jit::PARAMETERS.pureop_historylength as usize),
            postponed_op: None,
            postponed_box: None,
            call_pure_positions: Vec::new(),
            short_preamble_pure_ops: Vec::new(),
            last_emitted_was_removed: false,
            known_result_call_pure: Vec::new(),
            extra_call_pure: Vec::new(),
            call_pure_results: crate::optimizeopt::vec_assoc::VecAssoc::new(),
            preamble_pure_ops: Vec::new(),
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
                    descr_identity: None,
                };
                ExtraCallPureEntry::Direct { key, result }
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

    /// PyPy RecentPureOps stores AbstractResOp / Const boxes directly, so a
    /// reused result necessarily carries the same `box.type` as the query op.
    /// majit's imported preamble caches store only OpRefs, so recover the
    /// typed-constant path first (`ConstPtr(NULL)` etc.) before falling back
    /// to opref_type metadata.
    fn matches_result_type(op: &Op, result: OpRef, ctx: &OptContext) -> bool {
        if let Some(result_box) = ctx.get_box_replacement_box(result) {
            if let Some((_raw, result_type)) = ctx.getconst(&result_box) {
                return result_type == op.result_type();
            }
        }
        match ctx.opref_type(result) {
            Some(result_type) => result_type == op.result_type(),
            None => false,
        }
    }

    /// Try to find a cached result for this operation, considering commutativity.
    ///
    /// pure.py:81-95 `RecentPureOps.lookup` dispatches to `lookup1` /
    /// `lookup2` so that arg comparison goes through `box.same_box`.
    /// Pyre's typed OpRef Eq matches `same_box` for non-constants
    /// (identity), but constants need value equality
    /// (history.py:204-205): two distinct ConstInt slots with the same
    /// value are `same_box` true even though `OpRef ==` is false.
    fn lookup_pure(&self, key: &PureOpKey, ctx: &OptContext) -> Option<OpRef> {
        // pure.py:62 / :72-73 — `box.same_box(get_box_replacement(other))`
        // routes through `OptContext::same_box`, which walks both sides
        // through `get_box_replacement` and dispatches Const value equality
        // for the `history.py:204-205 Const.same_box → same_constant` overload.
        let same_box = |query: OpRef, stored: OpRef| -> bool { ctx.same_box(query, stored) };
        // pure.py:88-93 — `commutative` is forwarded into `lookup2`,
        // which checks both `(arg0, arg1)` and `(arg1, arg0)` orderings.
        let commutative = Self::is_commutative(key.opcode);
        self.cache.lookup(key, &same_box, commutative)
    }

    /// Record a pure operation in the CSE cache.
    /// pure.py: pure(opnum, op)
    pub fn pure(&mut self, op: &Op) {
        let key = PureOpKey::from_op(op);
        self.cache.insert(key, op.pos.get());
    }

    /// Record a pure operation with explicit args.
    /// pure.py: pure_from_args(opnum, args, op, descr=None)
    pub fn pure_from_args(&mut self, opcode: OpCode, args: &[OpRef], result: OpRef) {
        let key = PureOpKey {
            opcode,
            args: args.to_vec(),
            descr_identity: None,
        };
        self.cache.insert(key, result);
    }

    /// pure.py: pure_from_args1(opnum, arg0, op)
    /// Specialized version for unary operations.
    pub fn pure_from_args1(&mut self, opcode: OpCode, arg0: OpRef, result: OpRef) {
        self.pure_from_args(opcode, &[arg0], result);
    }

    /// pure.py: pure_from_args1(opnum, arg0, op, descr=...) — same as
    /// pure_from_args1 but keys the pure cache on `descr_identity`
    /// (Arc::as_ptr) so distinct descrs don't collide on the same
    /// (opcode, arg0) pair. Used for opcodes like ARRAYLEN_GC where
    /// the array descr discriminates which array's length is stored
    /// (virtualize.py:220).
    pub fn pure_from_args1_with_descr(
        &mut self,
        opcode: OpCode,
        arg0: OpRef,
        result: OpRef,
        descr: majit_ir::DescrRef,
    ) {
        let key = PureOpKey {
            opcode,
            args: vec![arg0],
            descr_identity: Some(majit_ir::descr::descr_identity(&descr)),
        };
        self.cache.insert(key, result);
    }

    /// pure.py: pure_from_args2(opnum, arg0, arg1, op)
    /// Specialized version for binary operations.
    pub fn pure_from_args2(&mut self, opcode: OpCode, arg0: OpRef, arg1: OpRef, result: OpRef) {
        self.pure_from_args(opcode, &[arg0, arg1], result);
    }

    /// Look up a previously recorded pure operation result.
    /// pure.py: get_pure_result(op)
    pub fn get_pure_result(&self, op: &Op, ctx: &OptContext) -> Option<OpRef> {
        let key = PureOpKey::from_op(op);
        self.lookup_pure(&key, ctx)
    }

    /// pure.py:57-65 lookup1(opt, box0, descr).
    ///
    /// `same_box(a, b)`: should apply get_box_replacement to `b` and then
    /// compare — identity for ops, value equality for constants.
    pub fn lookup1(
        &self,
        opcode: OpCode,
        arg0: OpRef,
        descr_identity: Option<usize>,
        same_box: impl Fn(OpRef, OpRef) -> bool,
    ) -> Option<OpRef> {
        self.cache.lookup1(opcode, arg0, descr_identity, same_box)
    }

    /// pure.py:67-79 lookup2(opt, box0, box1, descr, commutative).
    pub fn lookup2(
        &self,
        opcode: OpCode,
        arg0: OpRef,
        arg1: OpRef,
        descr_identity: Option<usize>,
        commutative: bool,
        same_box: impl Fn(OpRef, OpRef) -> bool,
    ) -> Option<OpRef> {
        self.cache
            .lookup2(opcode, arg0, arg1, descr_identity, commutative, same_box)
    }

    /// Get the positions of emitted CALL_PURE ops (for short preamble generation).
    /// pure.py: call_pure_positions
    pub fn call_pure_positions(&self) -> &[usize] {
        &self.call_pure_positions
    }

    /// pure.py:211-220 — check known_result_call_pure for a matching call.
    ///
    /// RPython iterates known_result_call_pure and compares:
    ///   `op.getdescr() is not known_result_op.getdescr()` → descr check
    ///   `self._same_args(known_result_op, op, 1, start_index)` → args check
    /// No opcode comparison.
    fn lookup_known_result(&self, op: &Op, start_index: usize, ctx: &OptContext) -> Option<OpRef> {
        let op_descr_identity = op.getdescr().as_ref().map(majit_ir::descr::descr_identity);
        for entry in &self.known_result_call_pure {
            if entry.descr_identity != op_descr_identity {
                continue;
            }
            // _same_args(known_op, op, 1, start_index):
            // entry.args is already known_op.args[1..], so compare from 0.
            let op_args: Vec<OpRef> = op.getarglist().iter().map(|a| a.to_opref()).collect();
            if Self::_same_args(&entry.args, &op_args, 0, start_index, ctx) {
                return Some(entry.result);
            }
        }
        None
    }

    fn demote_call_pure(&self, op: &Op) -> Op {
        let mut new_op = op.clone();
        new_op.opcode = OpCode::call_for_type(op.result_type());
        new_op
    }

    fn call_pure_can_raise(op: &Op) -> bool {
        op.with_call_descr(|cd| cd.get_extra_info().check_can_raise(true))
            .unwrap_or(true)
    }

    /// pure.py:50-55: RecentPureOps.force_preamble_op
    /// Searches preamble entries with forwarding-aware arg matching.
    /// On match, forces PreambleOp (in-place replacement) and returns result.
    fn force_preamble_op(&mut self, op: &Op, ctx: &mut OptContext) -> Option<OpRef> {
        let descr_identity = op.getdescr().as_ref().map(majit_ir::descr::descr_identity);
        for entry in &mut self.preamble_pure_ops {
            if entry.opcode != op.opcode {
                continue;
            }
            if entry.descr_identity != descr_identity {
                continue;
            }
            if entry.args.len() != op.num_args() {
                continue;
            }
            // pure.py:62 lookup1: `box0.same_box(get_box_replacement(op.getarg(0)))`.
            // Both stored and query are walked through the forwarding chain
            // via `OptContext::same_box` (pure.py:62, :72-73 +
            // history.py:204-205 Const.same_box → same_constant).
            let args_match = entry
                .args
                .iter()
                .zip(op.getarglist().iter())
                .all(|(&stored, query)| ctx.same_box(stored, query.to_opref()));
            if args_match {
                // pure.py:50-55: force_preamble_op — isinstance check → force → replace
                if let Some(result) = entry.forced_result {
                    if Self::matches_result_type(op, result, ctx) {
                        return Some(result);
                    }
                    continue;
                }
                let forced = ctx.force_op_from_preamble_op(&entry.pop);
                if !Self::matches_result_type(op, forced, ctx) {
                    continue;
                }
                entry.forced_result = Some(forced);
                return Some(forced);
            }
        }
        // Fallback: search ctx.imported_short_pure_ops directly.
        // Active until install_preamble_pure_ops is enabled, which
        // transfers these entries into preamble_pure_ops above.
        let mut matched_entry = None;
        for entry in &ctx.imported_short_pure_ops {
            if entry.opcode != op.opcode {
                continue;
            }
            if entry.descr.as_ref().map(majit_ir::descr::descr_identity) != descr_identity {
                continue;
            }
            if entry.args.len() != op.num_args() {
                continue;
            }
            // same_box: identity for non-constants, same_constant for constants.
            let mut args_match = true;
            for (expected, arg) in entry.args.iter().zip(op.getarglist().iter()) {
                let arg = arg.to_opref();
                match expected {
                    crate::optimizeopt::ImportedShortPureArg::OpRef(expected_ref) => {
                        if arg != *expected_ref {
                            args_match = false;
                            break;
                        }
                    }
                    crate::optimizeopt::ImportedShortPureArg::Const(expected_value, _source) => {
                        // Normalize through forwarding so a box that became
                        // constant via `make_constant` / replace_op chains
                        // still matches, mirroring `_same_args` and
                        // `preamble_pure_ops` upstream paths
                        // (optimizer.py:343 get_box_replacement).
                        match ctx
                            .get_box_replacement_box(arg)
                            .and_then(|b| b.const_value())
                        {
                            Some(v) if v == *expected_value => {}
                            _ => {
                                args_match = false;
                                break;
                            }
                        }
                    }
                }
            }
            if !args_match {
                continue;
            }
            matched_entry = Some(entry.clone());
            break;
        }
        if let Some(matched_entry) = matched_entry {
            let forced = ctx.force_op_from_preamble_op(&matched_entry.pop);
            if Self::matches_result_type(op, forced, ctx) {
                return Some(forced);
            }
        }
        None
    }

    /// Store PreambleOp in OptPure for always-pure ops.
    /// RPython shortpreamble.py:124-126: opt.pure(op.getopnum(), PreambleOp(...))
    pub fn pure_preamble(
        &mut self,
        opcode: OpCode,
        args: Vec<OpRef>,
        descr_identity: Option<usize>,
        pop: PreambleOp,
    ) {
        self.preamble_pure_ops.push(PreamblePureEntry {
            opcode,
            args,
            descr_identity,
            pop,
            forced_result: None,
        });
    }

    /// Store PreambleOp in extra_call_pure for CALL_PURE preamble imports.
    /// RPython shortpreamble.py:122-123: optpure.extra_call_pure.append(PreambleOp(...))
    pub fn extra_call_pure_preamble(
        &mut self,
        opcode: OpCode,
        args: Vec<OpRef>,
        descr_identity: Option<usize>,
        pop: PreambleOp,
    ) {
        let key = PureOpKey {
            opcode,
            args,
            descr_identity,
        };
        self.extra_call_pure
            .push(ExtraCallPureEntry::Preamble { key, pop });
    }

    /// pure.py:162-171 _can_reuse_oldop
    ///
    /// OVF guard pairing requires that an overflow-tracking op only reuse
    /// the result of another overflow-tracking op of the same opnum (since
    /// a regular `INT_ADD` may have overflowed). The non-OVF case is
    /// always safe.
    fn _can_reuse_oldop(oldop_opcode: OpCode, op_opcode: OpCode, ovf: bool) -> bool {
        if ovf {
            return oldop_opcode == op_opcode;
        }
        true
    }

    /// pure.py:240-247 _same_args
    ///
    /// Compare two argument lists with optional skip-prefixes on each side
    /// (used by COND_CALL_VALUE so its leading condition slot is not
    /// matched against a CALL_PURE's first real argument).
    fn _same_args(
        op1_args: &[OpRef],
        op2_args: &[OpRef],
        start_index1: usize,
        start_index2: usize,
        ctx: &OptContext,
    ) -> bool {
        if op1_args.len() - start_index1 != op2_args.len() - start_index2 {
            return false;
        }
        let mut j = start_index2;
        for i in start_index1..op1_args.len() {
            // pure.py:240-247 — same_box(op1_args[i], op2_args[j])
            // applies `get_box_replacement` to both sides, then dispatches
            // identity vs Const value equality (`history.py:204-205`).
            if !ctx.same_box(op1_args[i], op2_args[j]) {
                return false;
            }
            j += 1;
        }
        true
    }

    /// pure.py:249-265 optimize_call_pure_old
    ///
    /// Try to fuse `op` with a previously emitted `old_op` (either an
    /// inline call_pure recorded in `call_pure_positions` or an entry from
    /// `extra_call_pure`). Returns true when a match is found and the
    /// caller should mark the op REMOVED.
    fn optimize_call_pure_old(
        op: &Op,
        old_op_opcode: OpCode,
        old_op_args: &[OpRef],
        old_op_descr_identity: Option<usize>,
        op_descr_identity: Option<usize>,
        start_index: usize,
        ctx: &OptContext,
    ) -> bool {
        // pure.py:250-251: descr identity check.
        if op_descr_identity != old_op_descr_identity {
            return false;
        }
        // RPython relies on each CALL_PURE having a unique descriptor that
        // already encodes the return type, so a separate result-type check
        // is unnecessary upstream. majit allows tests with `descr = None`,
        // so we keep the implicit invariant explicit here: never CSE across
        // different return types.
        if op.opcode.result_type() != old_op_opcode.result_type() {
            return false;
        }
        // pure.py:254: old_start_index = OpHelpers.is_cond_call_value(old_op.opnum)
        let old_start_index = if old_op_opcode.is_cond_call_value() {
            1
        } else {
            0
        };
        // pure.py:255: self._same_args(old_op, op, old_start_index, start_index)
        let op_args: Vec<OpRef> = op.getarglist().iter().map(|a| a.to_opref()).collect();
        Self::_same_args(old_op_args, &op_args, old_start_index, start_index, ctx)
    }
}

impl Default for OptPure {
    fn default() -> Self {
        Self::new()
    }
}

impl OptPure {
    fn force_box(&mut self, opref: OpRef, ctx: &mut OptContext) -> OpRef {
        // Single resolve through the BoxRef terminal; the OpRef view is the
        // terminal's `to_opref()` (keystone equivalence, #113), so the prior
        // paired `get_box_replacement` + `get_box_replacement_box` of the
        // same `opref` was a redundant double walk.
        let resolved_box = ctx.get_box_replacement_box(opref);
        let resolved = resolved_box.as_ref().map(|b| b.to_opref()).unwrap_or(opref);
        if resolved_box.as_ref().map_or(false, |b| ctx.is_virtual(b)) {
            let resolved_box = resolved_box.expect("recorder-populated");
            let mut info = ctx.take_ptr_info(&resolved_box).unwrap();
            let forced = info.force_box(resolved_box, ctx);
            return ctx
                .get_box_replacement_box(forced)
                .map(|b| b.to_opref())
                .unwrap_or(forced);
        }
        resolved
    }

    /// optimizer.py:215-226 _can_optimize_call_pure.
    ///
    /// RPython: for each arg, `get_constant_box(arg)` returns the constant
    /// value (ConstInt/ConstPtr/ConstFloat), then uses those values as the
    /// lookup key in call_pure_results. Value-based equality, not identity.
    fn lookup_call_pure_result(
        &mut self,
        op: &Op,
        start_index: usize,
        ctx: &mut OptContext,
    ) -> Option<Value> {
        let mut arg_consts = Vec::with_capacity(op.num_args().saturating_sub(start_index));
        for i in start_index..op.num_args() {
            let forced = self.force_box(op.arg(i).to_opref(), ctx);
            let Some(const_value) = ctx
                .get_box_replacement_box(forced)
                .and_then(|b| ctx.get_constant_box(&b))
            else {
                return None;
            };
            arg_consts.push(const_value);
        }
        self.call_pure_results
            .iter()
            .find(|(k, _)| k.as_slice() == arg_consts.as_slice())
            .map(|(_, v)| v.clone())
    }
}

impl Optimization for OptPure {
    fn set_pureop_historylength(&mut self, limit: usize) {
        self.cache = RecentPureOpTable::new(limit);
    }

    fn propagate_forward(
        &mut self,
        op: &Op,
        op_rc: &majit_ir::OpRc,
        ctx: &mut OptContext,
    ) -> OptimizationResult {
        // optimizer.py: pure_from_args1 parity — consume pending registrations
        // from rewrite pass (CAST_*, CONVERT_* reverse-pure relationships)
        // and virtualize pass (ARRAYLEN_GC with array descr keying per
        // virtualize.py:220).
        for (opcode, arg0, result, descr) in ctx.pending_pure_from_args.drain(..) {
            match descr {
                Some(d) => self.pure_from_args1_with_descr(opcode, arg0, result, d),
                None => self.pure_from_args1(opcode, arg0, result),
            }
        }
        // optimizer.py: pure_from_args2 parity — consume binary registrations
        // (INSTANCE_PTR_EQ/NE swapped-args from rewrite.py:565,571).
        for (opcode, arg0, arg1, result) in ctx.pending_pure_from_args2.drain(..) {
            self.pure_from_args2(opcode, arg0, arg1, result);
        }

        // Don't reset for GUARD_NO_EXCEPTION — it needs the previous state.
        if op.opcode != OpCode::GuardNoException {
            self.last_emitted_was_removed = false;
        }

        // pure.py: OVF operation postponement.
        // INT_ADD_OVF, INT_SUB_OVF, INT_MUL_OVF are deferred until we see
        // GUARD_NO_OVERFLOW, so we can try CSE on the OVF op + guard pair.
        if op.opcode.is_ovf() {
            self.postponed_op = Some(op.clone());
            self.postponed_box = Some(BoxRef::from_bound_op(op_rc));
            return OptimizationResult::Remove;
        }

        // Handle the postponed OVF op when we see GUARD_NO_OVERFLOW.
        if let Some(mut postponed) = self.postponed_op.take() {
            let postponed_box = self
                .postponed_box
                .take()
                .expect("postponed_box is set whenever postponed_op is set");
            if op.opcode == OpCode::GuardNoOverflow {
                // pure.py:126-136 — only call `constant_fold` when every
                // arg has resolved to a `Const*` via `get_constant_box`
                // (the `for...else:` gate).  Without this pre-check,
                // `constant_fold` would panic on the now-strict
                // `expect("arg must be Const")` since postponed OVF
                // ops can have non-const args (e.g. `IntMulOvf(p, 1)`
                // where p is an inputarg).
                let all_args_const = (0..postponed.num_args()).all(|i| {
                    ctx.get_constant_box(&postponed.arg(i).get_box_replacement(false))
                        .is_some()
                });
                if all_args_const {
                    if let Some(Value::Int(folded)) = ctx.constant_fold(&postponed) {
                        ctx.make_constant(postponed.pos.get(), Value::Int(folded));
                        self.last_emitted_was_removed = true;
                        return OptimizationResult::Remove; // guard also removed
                    }
                }

                // pure.py:50-55: force_preamble_op replaces the OVF op
                // with the preamble's cached result.
                if let Some(cached_ref) = self.force_preamble_op(&postponed, ctx) {
                    let b_old = postponed_box.clone();
                    let b_cached = ctx.get_box_replacement(cached_ref);
                    ctx.make_equal_to(&b_old, &b_cached);
                    self.last_emitted_was_removed = true;
                    return OptimizationResult::Remove; // guard also removed
                }

                // pure.py:139-154 + 162-171 _can_reuse_oldop:
                // The lookup may surface a non-OVF op of the same shape
                // (e.g. INT_ADD vs INT_ADD_OVF). _can_reuse_oldop accepts
                // it only when the cached opnum matches our OVF opnum.
                let key = PureOpKey::from_op(&postponed);
                if let Some(cached_ref) = self.lookup_pure(&key, ctx) {
                    if Self::_can_reuse_oldop(postponed.opcode, postponed.opcode, true) {
                        let b_old = postponed_box.clone();
                        let b_cached = ctx.get_box_replacement(cached_ref);
                        ctx.make_equal_to(&b_old, &b_cached);
                        self.last_emitted_was_removed = true;
                        return OptimizationResult::Remove; // guard also removed
                    }
                }
                // pure.py:162-171: an OVF op cannot reuse a non-OVF result
                // even when the args/descr are identical (the prior op
                // may have overflowed silently). Discard the non-OVF
                // lookup but document the inverse case for future readers.
                let non_ovf_opcode = match postponed.opcode {
                    OpCode::IntAddOvf => Some(OpCode::IntAdd),
                    OpCode::IntSubOvf => Some(OpCode::IntSub),
                    OpCode::IntMulOvf => Some(OpCode::IntMul),
                    _ => None,
                };
                if let Some(non_ovf) = non_ovf_opcode {
                    let non_ovf_key = PureOpKey {
                        opcode: non_ovf,
                        args: postponed
                            .getarglist()
                            .iter()
                            .map(|a| a.to_opref())
                            .collect(),
                        descr_identity: None,
                    };
                    if let Some(cached_ref) = self.lookup_pure(&non_ovf_key, ctx) {
                        // _can_reuse_oldop(non_ovf, ovf=true) is false:
                        // skip even though the keys would otherwise match.
                        debug_assert!(!Self::_can_reuse_oldop(non_ovf, postponed.opcode, true));
                        let _ = cached_ref;
                    }
                }

                // RPython emits the postponed op through Optimizer.emit(),
                // which force_box()es every arg before final emission.
                // ctx.emit() bypasses that optimizer path, so mirror the
                // force_box step here before recording the postponed op.
                for i in 0..postponed.num_args() {
                    let forced = self.force_box(postponed.arg(i).to_opref(), ctx);
                    postponed.setarg(i, ctx.materialize_box_at(forced));
                }
                // Record and emit both the OVF op and the guard.
                self.cache.insert(key, postponed.pos.get());
                ctx.emit(postponed);
                return OptimizationResult::PassOn; // guard passes through
            } else {
                // Not a GUARD_NO_OVERFLOW: emit the postponed op now.
                for i in 0..postponed.num_args() {
                    let forced = self.force_box(postponed.arg(i).to_opref(), ctx);
                    postponed.setarg(i, ctx.materialize_box_at(forced));
                }
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

        // pure.py:211-220: RECORD_KNOWN_RESULT — record for later CALL_PURE lookup.
        // pure.py:214: `self.known_result_call_pure.append(op)`
        // Lookup compares descr + _same_args(known_op, query_op, 1, start_index).
        if op.opcode == OpCode::RecordKnownResult {
            if op.num_args() >= 2 {
                self.known_result_call_pure.push(KnownResultEntry {
                    descr_identity: op.getdescr().as_ref().map(majit_ir::descr::descr_identity),
                    args: op.getarglist()[1..].iter().map(|a| a.to_opref()).collect(),
                    result: op.arg(0).to_opref(),
                });
            }
            return OptimizationResult::Remove;
        }

        if op.opcode.is_always_pure() {
            // pure.py:121-136:
            //     for i in range(op.numargs()):
            //         if self.get_constant_box(op.getarg(i)) is None:
            //             break
            //     else:
            //         # all constant arguments: constant-fold away
            //         resbox = self.optimizer.constant_fold(op)
            //         self.optimizer.make_constant(op, resbox)
            //         return
            let all_args_const = (0..op.num_args()).all(|i| {
                ctx.get_constant_box(&op.arg(i).get_box_replacement(false))
                    .is_some()
            });
            if all_args_const {
                // Upstream `pure.py:130-136 for...else:` calls
                // `optimizer.constant_fold(op)` unconditionally and
                // feeds the result straight into `make_constant`.
                // Pyre's `constant_fold` now narrows to two
                // documented `None` paths only:
                //   1. `protect_speculative_operation`'s
                //      `supports_guard_gc_type == false` gate on
                //      memory-reading folds (mod.rs).  Upstream's
                //      comment at `optimizer.py:822-825` skips
                //      unrolling in that mode; pyre's
                //      `constant_fold` runs outside unroll too, so
                //      the fold itself must decline.
                //   2. Helper-internal narrow skips on `IntAddOvf` /
                //      `IntSubOvf` / `IntMulOvf` overflow, shift
                //      count outside `0..64`, `IntFloorDiv` /
                //      `IntMod` divide-by-zero, and CAST_FLOAT_TO_INT
                //      non-finite (`execute_binary_int_const` /
                //      `execute_cast_const` return `Ok(None)`).
                //      Upstream `do_int_add_ovf` would `assert
                //      metainterp is not None` on overflow with
                //      `metainterp=None` (executor.py:287); pyre
                //      prefers the softer skip so the op stays in
                //      the trace and the runtime guard fires.
                // Every caller-invariant violation (missing box,
                // descr, wrong Value variant) now panics inside
                // `constant_fold` / `protect_speculative_operation`,
                // matching upstream's `AttributeError`.  Genuine
                // SpeculativeError paths panic via
                // `panic_any(SpeculativeError)` and are caught at
                // `optimize_peeled_loop` /
                // `optimize_with_constants_and_inputs_at` per
                // `unroll.py:119-123`.
                if let Some(folded_value) = ctx.constant_fold(op) {
                    ctx.make_constant(op.pos.get(), folded_value);
                    self.last_emitted_was_removed = true;
                    return OptimizationResult::Remove;
                }
            }

            if let Some(cached_ref) = self.force_preamble_op(op, ctx) {
                let b_old = crate::r#box::BoxRef::from_bound_op(op_rc);
                let b_cached = ctx.get_box_replacement(cached_ref);
                ctx.make_equal_to(&b_old, &b_cached);
                self.last_emitted_was_removed = true;
                return OptimizationResult::Remove;
            }

            let key = PureOpKey::from_op(op);

            // CSE: exact same operation already computed?
            if let Some(cached_ref) = self.lookup_pure(&key, ctx) {
                let b_old = crate::r#box::BoxRef::from_bound_op(op_rc);
                let b_cached = ctx.get_box_replacement(cached_ref);
                ctx.make_equal_to(&b_old, &b_cached);
                self.last_emitted_was_removed = true;
                return OptimizationResult::Remove;
            }

            self.cache.insert(key, op.pos.get());
            self.short_preamble_pure_ops.push(op.clone());
            return OptimizationResult::PassOn;
        }

        // pure.py:185-228 optimize_call_pure(op, start_index)
        // pure.py:230-234: optimize_CALL_PURE_I/R/F/N → start_index=0
        // pure.py:236-238: optimize_COND_CALL_VALUE_I/R → start_index=1
        if op.opcode.is_call_pure() || op.opcode.is_cond_call_value() {
            let start_index: usize = if op.opcode.is_cond_call_value() { 1 } else { 0 };
            let op_descr_identity = op.getdescr().as_ref().map(majit_ir::descr::descr_identity);

            // pure.py:191-196: _can_optimize_call_pure(op, start_index=1).
            if let Some(value) = self.lookup_call_pure_result(op, start_index, ctx) {
                ctx.make_constant(op.pos.get(), value);
                self.last_emitted_was_removed = true;
                return OptimizationResult::Remove;
            }

            // pure.py:200-203: iterate call_pure_positions, try
            // optimize_call_pure_old with adjusted start_index.
            for &pos in &self.call_pure_positions {
                if let Some(old_op) = ctx.new_operations.get(pos) {
                    let old_descr_identity = old_op
                        .getdescr()
                        .as_ref()
                        .map(majit_ir::descr::descr_identity);
                    let old_op_args: Vec<OpRef> =
                        old_op.getarglist().iter().map(|a| a.to_opref()).collect();
                    if Self::optimize_call_pure_old(
                        op,
                        old_op.opcode,
                        &old_op_args,
                        old_descr_identity,
                        op_descr_identity,
                        start_index,
                        ctx,
                    ) {
                        let cached_src = old_op.pos.get();
                        let b_old = crate::r#box::BoxRef::from_bound_op(op_rc);
                        let b_cached = ctx.get_box_replacement(cached_src);
                        ctx.make_equal_to(&b_old, &b_cached);
                        self.last_emitted_was_removed = true;
                        return OptimizationResult::Remove;
                    }
                }
            }
            // pure.py:204-210: iterate extra_call_pure entries.
            //   if isinstance(old_op, PreambleOp):
            //       old_op = self.optimizer.force_op_from_preamble(old_op)
            //       self.extra_call_pure[i] = old_op
            // Force runs after the match so use_box / potential_extra_ops /
            // forwarded-info side effects are tied to the actual fuse, and
            // the entry is rewritten so subsequent matches reuse the forced
            // result without re-forcing.
            let mut matched: Option<(usize, Option<PreambleOp>)> = None;
            for (i, entry) in self.extra_call_pure.iter().enumerate() {
                let (entry_opcode, entry_args, entry_descr_identity, pop) = match entry {
                    ExtraCallPureEntry::Direct { key, .. } => {
                        (key.opcode, &key.args, key.descr_identity, None)
                    }
                    ExtraCallPureEntry::Preamble { key, pop } => {
                        (key.opcode, &key.args, key.descr_identity, Some(pop.clone()))
                    }
                };
                if Self::optimize_call_pure_old(
                    op,
                    entry_opcode,
                    entry_args,
                    entry_descr_identity,
                    op_descr_identity,
                    start_index,
                    ctx,
                ) {
                    matched = Some((i, pop));
                    break;
                }
            }
            if let Some((i, pop)) = matched {
                let entry_result = if let Some(pop) = pop {
                    let forced = ctx.force_op_from_preamble_op(&pop);
                    let key = match &self.extra_call_pure[i] {
                        ExtraCallPureEntry::Preamble { key, .. } => key.clone(),
                        _ => unreachable!("matched index must still hold the Preamble entry"),
                    };
                    self.extra_call_pure[i] = ExtraCallPureEntry::Direct {
                        key,
                        result: forced,
                    };
                    forced
                } else {
                    match &self.extra_call_pure[i] {
                        ExtraCallPureEntry::Direct { result, .. } => *result,
                        _ => unreachable!("non-preamble matched index must be Direct"),
                    }
                };
                let b_old = crate::r#box::BoxRef::from_bound_op(op_rc);
                let b_cached = ctx.get_box_replacement(entry_result);
                ctx.make_equal_to(&b_old, &b_cached);
                self.last_emitted_was_removed = true;
                return OptimizationResult::Remove;
            }
            // pure.py:211-220: known_result_call_pure.
            if let Some(result_ref) = self.lookup_known_result(op, start_index, ctx) {
                let b_old = crate::r#box::BoxRef::from_bound_op(op_rc);
                let b_result = ctx.get_box_replacement(result_ref);
                ctx.make_equal_to(&b_old, &b_result);
                self.last_emitted_was_removed = true;
                return OptimizationResult::Remove;
            }

            let key = PureOpKey::from_call_op(op, start_index);
            self.cache.insert(key, op.pos.get());
            self.call_pure_positions.push(ctx.new_operations.len());
            if start_index == 0 {
                // pure.py:222-225: replace CALL_PURE with CALL.
                let new_op = self.demote_call_pure(op);
                if !Self::call_pure_can_raise(op) {
                    self.short_preamble_pure_ops.push(new_op.clone());
                }
                return OptimizationResult::Emit(new_op);
            } else {
                // pure.py:226-227: COND_CALL_VALUE is NOT demoted.
                return OptimizationResult::Emit(op.clone());
            }
        }

        OptimizationResult::PassOn
    }

    fn setup(&mut self) {
        let limit = self.cache.history_length();
        self.cache = RecentPureOpTable::new(limit);
        self.postponed_op = None;
        self.postponed_box = None;
        self.call_pure_positions.clear();
        self.short_preamble_pure_ops.clear();
        self.last_emitted_was_removed = false;
        self.known_result_call_pure.clear();
        // Note: extra_call_pure is NOT cleared on setup — it persists
        // across optimization runs (set by set_extra_call_pure before opt).
        // preamble_pure_ops also NOT cleared — populated during import.
    }

    fn set_call_pure_results(
        &mut self,
        results: &crate::optimizeopt::vec_assoc::VecAssoc<Vec<Value>, Value>,
    ) {
        self.call_pure_results = results.clone();
    }

    fn name(&self) -> &'static str {
        "pure"
    }

    /// pure.py: produce_potential_short_preamble_ops(sb)
    /// Add pure operations and CALL_PURE results to the short preamble.
    /// shortpreamble.py:112-126: PureOp.produce_op stores PreambleOp in
    /// optpure. In RPython, produce_op accesses opt.optimizer.optpure directly.
    /// In majit, import_short_preamble_ops stores in ctx.imported_short_pure_ops,
    /// then this method transfers them into OptPure's preamble caches.
    fn install_preamble_pure_ops(&mut self, ctx: &OptContext) {
        for entry in &ctx.imported_short_pure_ops {
            // heap.py:640-643: GetfieldGcPure on constant objects are
            // handled by constant_fold in the heap optimizer. Skip these
            // to avoid conflicting with the heap path. Non-constant
            // GetfieldGcPure ops go through the pure cache normally.
            if matches!(
                entry.opcode,
                OpCode::GetfieldGcPureI | OpCode::GetfieldGcPureR | OpCode::GetfieldGcPureF
            ) {
                let arg0_is_const = entry.args.first().map_or(false, |a| {
                    matches!(a, crate::optimizeopt::ImportedShortPureArg::Const(..))
                });
                if arg0_is_const {
                    continue;
                }
            }
            // The replay `preamble_op` was built by `ImportedShortPureOp::new`
            // from the same arg list with producer-bound operands
            // (shortpreamble.py:425 — the replay op carries the same Box
            // objects); reuse them instead of re-deriving position-only
            // echoes from the OpRef table.
            let imported_args = entry.pop.preamble_op.getarglist();
            let mut imported_op = Op::new(entry.opcode, &imported_args);
            imported_op.pos.set(entry.result);
            if let Some(d) = entry.descr.clone() {
                imported_op.setdescr(d);
            }
            self.short_preamble_pure_ops.push(imported_op);
            let resolved_args: Vec<OpRef> = entry
                .args
                .iter()
                .map(|a| match a {
                    crate::optimizeopt::ImportedShortPureArg::OpRef(r) => *r,
                    crate::optimizeopt::ImportedShortPureArg::Const(_v, source) => {
                        // RPython: Const args have a registered OpRef from
                        // make_constant. Use the source OpRef for matching.
                        *source
                    }
                })
                .collect();
            let descr_identity = entry.descr.as_ref().map(majit_ir::descr::descr_identity);
            let pop = entry.pop.clone();
            if entry.opcode.is_call_pure() || entry.opcode.is_call() {
                // shortpreamble.py:122-123: optpure.extra_call_pure.append(PreambleOp(...))
                self.extra_call_pure_preamble(entry.opcode, resolved_args, descr_identity, pop);
            } else {
                // shortpreamble.py:124-126: opt.pure(opnum, PreambleOp(...))
                self.pure_preamble(entry.opcode, resolved_args, descr_identity, pop);
            }
        }
    }

    fn produce_potential_short_preamble_ops(
        &self,
        sb: &mut crate::optimizeopt::shortpreamble::ShortBoxes,
        _ctx: &mut OptContext,
    ) {
        for op in &self.short_preamble_pure_ops {
            sb.add_pure_op(op.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn initialize_imported_short_pure_builder(
        ctx: &mut OptContext,
        preamble_op: Op,
        label_arg_idx: Option<usize>,
    ) {
        let source = preamble_op.pos.get();
        let short_inputargs: Vec<OpRef> = match label_arg_idx {
            Some(idx) => (0..=idx as u32).map(OpRef::int_op).collect(),
            None => vec![OpRef::int_op(0)],
        };
        ctx.initialize_imported_short_preamble_builder(
            &short_inputargs,
            &short_inputargs,
            &[crate::optimizeopt::shortpreamble::PreambleOp {
                op: preamble_op,
                kind: crate::optimizeopt::shortpreamble::PreambleOpKind::Pure,
                label_arg_idx,
                invented_name: false,
                same_as_source: None,
            }],
        );
        // Keep the source result available to use_box() exactly like the
        // imported short preamble path does after unroll import.
        if source != OpRef::NONE {
            // PreambleOp.op carries the Box itself (shortpreamble.py:12).
            let source_box = ctx.materialize_box_at(source);
            let pop = crate::optimizeopt::info::PreambleOp {
                op: source_box.clone(),
                invented_name: false,
                preamble_op: {
                    let mut same_as = Op::new(OpCode::SameAsI, &[source_box]);
                    same_as.pos.set(source);
                    std::rc::Rc::new(same_as)
                },
            };
            ctx.set_potential_extra_op(source, pop);
        }
    }
    use crate::optimizeopt::optimizer::Optimizer;
    use majit_ir::Type;
    use majit_ir::descr::make_field_descr_full;
    /// Helper: assign sequential positions to ops.
    fn assign_positions(ops: &mut [Op]) {
        for (i, op) in ops.iter_mut().enumerate() {
            // Type-tag op.pos via the result-type-aware factory so the
            // OpRef variant carries `Box.type` (history.py:220 /
            // resoperation.py:1693 parity). Argument OpRefs in these
            // fixtures must use the matching typed factory at the same
            // raw N to satisfy variant-aware Eq against `op.pos`.
            op.pos.set(OpRef::op_typed(i as u32, op.result_type()));
        }
    }

    #[test]
    fn test_cse_int_add() {
        // i2 = int_add(i0, i1)
        // i3 = int_add(i0, i1)  <- should be eliminated, replaced by i2
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[
                    BoxRef::from_opref(OpRef::int_op(0)),
                    BoxRef::from_opref(OpRef::int_op(1)),
                ],
            ),
            Op::new(
                OpCode::IntAdd,
                &[
                    BoxRef::from_opref(OpRef::int_op(0)),
                    BoxRef::from_opref(OpRef::int_op(1)),
                ],
            ),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecAssoc::new(), 1024);

        // Only the first IntAdd should remain.
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
    }

    #[test]
    fn test_cse_different_args_not_eliminated() {
        // i2 = int_add(i0, i1)
        // i3 = int_add(i0, i2)  <- different args, should NOT be eliminated
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[
                    BoxRef::from_opref(OpRef::int_op(0)),
                    BoxRef::from_opref(OpRef::int_op(1)),
                ],
            ),
            Op::new(
                OpCode::IntAdd,
                &[
                    BoxRef::from_opref(OpRef::int_op(0)),
                    BoxRef::from_opref(OpRef::int_op(2)),
                ],
            ),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecAssoc::new(), 1024);

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_cse_commutative() {
        // i2 = int_add(i0, i1)
        // i3 = int_add(i1, i0)  <- commutative, should be eliminated
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[
                    BoxRef::from_opref(OpRef::int_op(0)),
                    BoxRef::from_opref(OpRef::int_op(1)),
                ],
            ),
            Op::new(
                OpCode::IntAdd,
                &[
                    BoxRef::from_opref(OpRef::int_op(1)),
                    BoxRef::from_opref(OpRef::int_op(0)),
                ],
            ),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecAssoc::new(), 1024);

        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_cse_non_commutative() {
        // i2 = int_sub(i0, i1)
        // i3 = int_sub(i1, i0)  <- NOT commutative, should NOT be eliminated
        let mut ops = vec![
            Op::new(
                OpCode::IntSub,
                &[
                    BoxRef::from_opref(OpRef::int_op(0)),
                    BoxRef::from_opref(OpRef::int_op(1)),
                ],
            ),
            Op::new(
                OpCode::IntSub,
                &[
                    BoxRef::from_opref(OpRef::int_op(1)),
                    BoxRef::from_opref(OpRef::int_op(0)),
                ],
            ),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecAssoc::new(), 1024);

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_cse_multiple_opcodes() {
        // i2 = int_add(i0, i1)
        // i3 = int_mul(i0, i1)  <- different opcode, should NOT be eliminated
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[
                    BoxRef::from_opref(OpRef::int_op(0)),
                    BoxRef::from_opref(OpRef::int_op(1)),
                ],
            ),
            Op::new(
                OpCode::IntMul,
                &[
                    BoxRef::from_opref(OpRef::int_op(0)),
                    BoxRef::from_opref(OpRef::int_op(1)),
                ],
            ),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecAssoc::new(), 1024);

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_cse_three_duplicates() {
        // Use input arg OpRefs (100, 101) that don't collide with op positions (0, 1, 2).
        // i2 = int_add(i100, i101)
        // i3 = int_add(i100, i101)  <- eliminated
        // i4 = int_add(i100, i101)  <- eliminated
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[
                    BoxRef::from_opref(OpRef::int_op(100)),
                    BoxRef::from_opref(OpRef::int_op(101)),
                ],
            ),
            Op::new(
                OpCode::IntAdd,
                &[
                    BoxRef::from_opref(OpRef::int_op(100)),
                    BoxRef::from_opref(OpRef::int_op(101)),
                ],
            ),
            Op::new(
                OpCode::IntAdd,
                &[
                    BoxRef::from_opref(OpRef::int_op(100)),
                    BoxRef::from_opref(OpRef::int_op(101)),
                ],
            ),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecAssoc::new(), 1024);

        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_call_pure_demoted_to_call() {
        // call_pure_i(args...) -> should become call_i(args...)
        let mut ops = vec![Op::new(
            OpCode::CallPureI,
            &[
                BoxRef::from_opref(OpRef::int_op(0)),
                BoxRef::from_opref(OpRef::int_op(1)),
            ],
        )];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecAssoc::new(), 1024);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::CallI);
        assert_eq!(
            result[0]
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            &[OpRef::int_op(0), OpRef::int_op(1)]
        );
    }

    #[test]
    fn test_call_pure_r_demoted() {
        // CallPureR / CallR carry RPython `RefOp.type = 'r'` parity
        // (resoperation.py:638): the result is a Ref-typed Box, and
        // the function-pointer arg is also Ref-typed.
        let mut ops = vec![Op::new(
            OpCode::CallPureR,
            &[BoxRef::from_opref(OpRef::ref_op(0))],
        )];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecAssoc::new(), 1024);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::CallR);
        assert_eq!(result[0].pos.get().ty(), Some(Type::Ref));
    }

    #[test]
    fn test_non_pure_op_passes_through() {
        // setfield_gc is not pure, should pass through unchanged
        let mut ops = vec![Op::new(
            OpCode::SetfieldGc,
            &[
                BoxRef::from_opref(OpRef::void_op(0)),
                BoxRef::from_opref(OpRef::int_op(1)),
            ],
        )];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecAssoc::new(), 1024);

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::SetfieldGc);
    }

    #[test]
    fn test_cse_unary_ops() {
        // i1 = int_neg(i0)
        // i2 = int_neg(i0)  <- should be eliminated
        let mut ops = vec![
            Op::new(OpCode::IntNeg, &[BoxRef::from_opref(OpRef::int_op(0))]),
            Op::new(OpCode::IntNeg, &[BoxRef::from_opref(OpRef::int_op(0))]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecAssoc::new(), 1024);

        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_cse_float_ops() {
        // f2 = float_add(f0, f1)
        // f3 = float_add(f0, f1)  <- should be eliminated
        let mut ops = vec![
            Op::new(
                OpCode::FloatAdd,
                &[
                    BoxRef::from_opref(OpRef::float_op(0)),
                    BoxRef::from_opref(OpRef::float_op(1)),
                ],
            ),
            Op::new(
                OpCode::FloatAdd,
                &[
                    BoxRef::from_opref(OpRef::float_op(0)),
                    BoxRef::from_opref(OpRef::float_op(1)),
                ],
            ),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecAssoc::new(), 1024);

        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_cache_eviction() {
        // Force a tiny cache so eviction behavior is deterministic even if
        // the production default changes.
        let mut ops = Vec::new();
        for i in 0..17u32 {
            ops.push(Op::new(
                OpCode::IntAdd,
                &[
                    BoxRef::from_opref(OpRef::int_op(i)),
                    BoxRef::from_opref(OpRef::int_op(i + 100)),
                ],
            ));
        }
        // Re-insert op #0: same args as ops[0]
        ops.push(Op::new(
            OpCode::IntAdd,
            &[
                BoxRef::from_opref(OpRef::int_op(0)),
                BoxRef::from_opref(OpRef::int_op(100)),
            ],
        ));
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure {
            cache: RecentPureOpTable::new(16),
            postponed_op: None,
            postponed_box: None,
            call_pure_positions: Vec::new(),
            short_preamble_pure_ops: Vec::new(),
            last_emitted_was_removed: false,
            known_result_call_pure: Vec::new(),
            extra_call_pure: Vec::new(),
            call_pure_results: crate::optimizeopt::vec_assoc::VecAssoc::new(),
            preamble_pure_ops: Vec::new(),
        }));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecAssoc::new(), 1024);

        // All 17 unique ops should be emitted, plus the re-inserted one
        // (since the first was evicted from the LRU cache of size 16).
        assert_eq!(result.len(), 18);
    }

    #[test]
    fn test_cse_with_forwarding() {
        // Test that CSE works correctly when OpRef forwarding is involved.
        let mut ctx = OptContext::new(10);
        let mut pass = OptPure::new();

        // Production binds the input operands a, b before the int_add is
        // processed; bind them canonically here so same_box resolves both
        // ops' args to one shared box.
        ctx.materialize_box_at(OpRef::int_op(0));
        ctx.materialize_box_at(OpRef::int_op(1));

        // Simulate: op0 = int_add(a, b)
        let op0 = Op::new(
            OpCode::IntAdd,
            &[
                BoxRef::from_opref(OpRef::int_op(0)),
                BoxRef::from_opref(OpRef::int_op(1)),
            ],
        );
        let mut op0 = op0;
        op0.pos.set(OpRef::int_op(2));
        let result0 = pass.propagate_forward(&op0, &std::rc::Rc::new(op0.clone()), &mut ctx);
        assert!(matches!(result0, OptimizationResult::PassOn));

        // Simulate: op1 = int_add(a, b) with same args
        let op1 = Op::new(
            OpCode::IntAdd,
            &[
                BoxRef::from_opref(OpRef::int_op(0)),
                BoxRef::from_opref(OpRef::int_op(1)),
            ],
        );
        let mut op1 = op1;
        op1.pos.set(OpRef::int_op(3));
        let result1 = pass.propagate_forward(&op1, &std::rc::Rc::new(op1.clone()), &mut ctx);
        assert!(matches!(result1, OptimizationResult::Remove));
    }

    #[test]
    fn test_pure_op_key_equality() {
        let key1 = PureOpKey {
            opcode: OpCode::IntAdd,
            args: vec![OpRef::int_op(0), OpRef::int_op(1)],
            descr_identity: None,
        };
        let key2 = PureOpKey {
            opcode: OpCode::IntAdd,
            args: vec![OpRef::int_op(0), OpRef::int_op(1)],
            descr_identity: None,
        };
        let key3 = PureOpKey {
            opcode: OpCode::IntAdd,
            args: vec![OpRef::int_op(1), OpRef::int_op(0)],
            descr_identity: None,
        };
        assert_eq!(key1, key2);
        assert_ne!(key1, key3);
    }

    #[test]
    fn test_commutative_xor() {
        let mut ops = vec![
            Op::new(
                OpCode::IntXor,
                &[
                    BoxRef::from_opref(OpRef::int_op(0)),
                    BoxRef::from_opref(OpRef::int_op(1)),
                ],
            ),
            Op::new(
                OpCode::IntXor,
                &[
                    BoxRef::from_opref(OpRef::int_op(1)),
                    BoxRef::from_opref(OpRef::int_op(0)),
                ],
            ),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecAssoc::new(), 1024);

        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_commutative_int_and() {
        let mut ops = vec![
            Op::new(
                OpCode::IntAnd,
                &[
                    BoxRef::from_opref(OpRef::int_op(0)),
                    BoxRef::from_opref(OpRef::int_op(1)),
                ],
            ),
            Op::new(
                OpCode::IntAnd,
                &[
                    BoxRef::from_opref(OpRef::int_op(1)),
                    BoxRef::from_opref(OpRef::int_op(0)),
                ],
            ),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecAssoc::new(), 1024);

        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_comparison_cse() {
        // i2 = int_lt(i0, i1)
        // i3 = int_lt(i0, i1)  <- should be eliminated
        let mut ops = vec![
            Op::new(
                OpCode::IntLt,
                &[
                    BoxRef::from_opref(OpRef::int_op(0)),
                    BoxRef::from_opref(OpRef::int_op(1)),
                ],
            ),
            Op::new(
                OpCode::IntLt,
                &[
                    BoxRef::from_opref(OpRef::int_op(0)),
                    BoxRef::from_opref(OpRef::int_op(1)),
                ],
            ),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecAssoc::new(), 1024);

        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_call_pure_f_n_demoted() {
        // CallPureF / CallF carry RPython `FloatOp.type = 'f'` parity
        // (resoperation.py:589). CallPureN / CallN are void-result —
        // `AbstractResOp.type = 'v'` (resoperation.py:260) — and pyre
        // mints them as `OpRef::VoidOp(pos)` whose `ty()` is
        // `Some(Type::Void)`.
        let mut ops = vec![
            Op::new(OpCode::CallPureF, &[BoxRef::from_opref(OpRef::float_op(0))]),
            Op::new(OpCode::CallPureN, &[BoxRef::from_opref(OpRef::float_op(0))]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecAssoc::new(), 1024);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::CallF);
        assert_eq!(result[0].pos.get().ty(), Some(Type::Float));
        assert_eq!(result[1].opcode, OpCode::CallN);
        assert_eq!(result[1].pos.get().ty(), Some(Type::Void));
    }

    #[test]
    fn test_mixed_pure_and_non_pure() {
        // Mix of pure and non-pure operations, only duplicated pure ops get CSE'd.
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[
                    BoxRef::from_opref(OpRef::int_op(0)),
                    BoxRef::from_opref(OpRef::void_op(1)),
                ],
            ), // pure, kept
            Op::new(
                OpCode::SetfieldGc,
                &[
                    BoxRef::from_opref(OpRef::int_op(0)),
                    BoxRef::from_opref(OpRef::void_op(1)),
                ],
            ), // not pure, kept
            Op::new(
                OpCode::IntAdd,
                &[
                    BoxRef::from_opref(OpRef::int_op(0)),
                    BoxRef::from_opref(OpRef::void_op(1)),
                ],
            ), // pure duplicate, eliminated
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecAssoc::new(), 1024);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
        assert_eq!(result[1].opcode, OpCode::SetfieldGc);
    }

    #[test]
    fn test_call_loopinvariant_cse() {
        // Two identical CALL_LOOPINVARIANT_I calls → second eliminated.
        let mut ops = vec![
            Op::new(
                OpCode::CallLoopinvariantI,
                &[
                    BoxRef::from_opref(OpRef::int_op(100)),
                    BoxRef::from_opref(OpRef::int_op(101)),
                ],
            ),
            Op::new(
                OpCode::CallLoopinvariantI,
                &[
                    BoxRef::from_opref(OpRef::int_op(100)),
                    BoxRef::from_opref(OpRef::int_op(101)),
                ],
            ),
        ];
        assign_positions(&mut ops);

        let mut constants: majit_ir::VecAssoc<u32, majit_ir::Value> = majit_ir::VecAssoc::new();
        constants.insert(100u32, majit_ir::Value::Int(0xCAFE)); // func pointer must be a known constant
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(crate::optimizeopt::rewrite::OptRewrite::new()));
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 1024);

        // Only the first call should remain, demoted to CallI.
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::CallI);
    }

    #[test]
    fn test_call_loopinvariant_different_args() {
        // CALL_LOOPINVARIANT_I with different args → both kept.
        let mut ops = vec![
            Op::new(
                OpCode::CallLoopinvariantI,
                &[
                    BoxRef::from_opref(OpRef::int_op(100)),
                    BoxRef::from_opref(OpRef::int_op(101)),
                ],
            ),
            Op::new(
                OpCode::CallLoopinvariantI,
                &[
                    BoxRef::from_opref(OpRef::int_op(100)),
                    BoxRef::from_opref(OpRef::int_op(102)),
                ],
            ),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(crate::optimizeopt::rewrite::OptRewrite::new()));
        opt.add_pass(Box::new(OptPure::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecAssoc::new(), 1024);

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
            let mut ops = vec![Op::new(loopinv_op, &[BoxRef::from_opref(OpRef::int_op(0))])];
            assign_positions(&mut ops);

            let mut opt = Optimizer::new();
            opt.add_pass(Box::new(crate::optimizeopt::rewrite::OptRewrite::new()));
            opt.add_pass(Box::new(OptPure::new()));
            let result =
                opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecAssoc::new(), 1024);

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
                &[
                    BoxRef::from_opref(OpRef::int_op(i + 100)),
                    BoxRef::from_opref(OpRef::int_op(200)),
                ],
            ));
        }
        // Re-insert call #0: same args as ops[0]
        ops.push(Op::new(
            OpCode::CallLoopinvariantI,
            &[
                BoxRef::from_opref(OpRef::int_op(100)),
                BoxRef::from_opref(OpRef::int_op(200)),
            ],
        ));
        assign_positions(&mut ops);

        // Each func pointer must be a known constant for OptRewrite CSE.
        let mut constants: majit_ir::VecAssoc<u32, majit_ir::Value> = majit_ir::VecAssoc::new();
        for i in 0..20u32 {
            constants.insert(i + 100, majit_ir::Value::Int((i + 100) as i64));
        }
        let mut opt = Optimizer::new();
        for i in 0..20u32 {}
        opt.add_pass(Box::new(crate::optimizeopt::rewrite::OptRewrite::new()));
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 1024);

        // 20 unique calls + the duplicate (#0) should be eliminated → 20 total
        assert_eq!(result.len(), 20);
    }

    #[test]
    fn test_call_loopinvariant_mixed_with_pure() {
        // Loop-invariant and pure CSE should coexist.
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[
                    BoxRef::from_opref(OpRef::int_op(100)),
                    BoxRef::from_opref(OpRef::int_op(101)),
                ],
            ), // pure
            Op::new(
                OpCode::CallLoopinvariantI,
                &[
                    BoxRef::from_opref(OpRef::int_op(200)),
                    BoxRef::from_opref(OpRef::int_op(201)),
                ],
            ), // loopinvariant
            Op::new(
                OpCode::IntAdd,
                &[
                    BoxRef::from_opref(OpRef::int_op(100)),
                    BoxRef::from_opref(OpRef::int_op(101)),
                ],
            ), // pure dup → removed
            Op::new(
                OpCode::CallLoopinvariantI,
                &[
                    BoxRef::from_opref(OpRef::int_op(200)),
                    BoxRef::from_opref(OpRef::int_op(201)),
                ],
            ), // loopinvariant dup → removed
        ];
        assign_positions(&mut ops);

        let mut constants: majit_ir::VecAssoc<u32, majit_ir::Value> = majit_ir::VecAssoc::new();
        constants.insert(200u32, majit_ir::Value::Int(0xBEEF)); // func pointer must be a known constant
        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(crate::optimizeopt::rewrite::OptRewrite::new()));
        opt.add_pass(Box::new(OptPure::new()));
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 1024);

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].opcode, OpCode::IntAdd);
        assert_eq!(result[1].opcode, OpCode::CallI);
    }

    #[test]
    fn test_constant_fold_int_add() {
        // IntAdd(const(3), const(4)) → eliminated, result = const(7)
        let mut ops = vec![
            Op::new(
                OpCode::IntAdd,
                &[
                    BoxRef::from_opref(OpRef::int_op(10_000)),
                    BoxRef::from_opref(OpRef::int_op(10_001)),
                ],
            ),
            // Use the result in a guard to prevent dead code elimination
            Op::new(OpCode::Finish, &[BoxRef::from_opref(OpRef::int_op(0))]),
        ];
        assign_positions(&mut ops);

        let mut constants: majit_ir::VecAssoc<u32, majit_ir::Value> = majit_ir::VecAssoc::new();
        constants.insert(10_000u32, majit_ir::Value::Int(3));
        constants.insert(10_001u32, majit_ir::Value::Int(4));

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
            Op::new(
                OpCode::IntLt,
                &[
                    BoxRef::from_opref(OpRef::int_op(10_000)),
                    BoxRef::from_opref(OpRef::int_op(10_001)),
                ],
            ),
            Op::new(OpCode::Finish, &[BoxRef::from_opref(OpRef::int_op(0))]),
        ];
        assign_positions(&mut ops);

        let mut constants: majit_ir::VecAssoc<u32, majit_ir::Value> = majit_ir::VecAssoc::new();
        constants.insert(10_000u32, majit_ir::Value::Int(3));
        constants.insert(10_001u32, majit_ir::Value::Int(5));

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
            Op::new(
                OpCode::IntAddOvf,
                &[
                    BoxRef::from_opref(OpRef::int_op(100)),
                    BoxRef::from_opref(OpRef::int_op(101)),
                ],
            ),
            Op::new(OpCode::GuardNoOverflow, &[]),
            Op::new(
                OpCode::IntAddOvf,
                &[
                    BoxRef::from_opref(OpRef::int_op(100)),
                    BoxRef::from_opref(OpRef::int_op(101)),
                ],
            ),
            Op::new(OpCode::GuardNoOverflow, &[]),
            Op::new(OpCode::Finish, &[]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        opt.snapshot_boxes = snapshots;
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecAssoc::new(), 1024);

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
            Op::new(
                OpCode::IntAddOvf,
                &[
                    BoxRef::from_opref(OpRef::int_op(10_000)),
                    BoxRef::from_opref(OpRef::int_op(10_001)),
                ],
            ),
            Op::new(OpCode::GuardNoOverflow, &[]),
            Op::new(OpCode::Finish, &[]),
        ];
        assign_positions(&mut ops);

        let mut constants: majit_ir::VecAssoc<u32, majit_ir::Value> = majit_ir::VecAssoc::new();
        constants.insert(10_000u32, majit_ir::Value::Int(3));
        constants.insert(10_001u32, majit_ir::Value::Int(4));

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

    #[repr(C)]
    struct TestIntFieldObject {
        value: i64,
    }

    #[repr(C)]
    struct TestFloatFieldObject {
        value: f64,
    }

    #[repr(C)]
    struct TestRefFieldObject {
        value: usize,
    }

    #[test]
    fn test_constant_fold_getfield_gc_pure_i_from_constant_object() {
        let object = Box::new(TestIntFieldObject { value: 123 });
        let ptr = Box::into_raw(object) as usize;

        let descr = make_field_descr_full(1, 0, 8, Type::Int, true);
        let mut op = Op::with_descr(
            OpCode::GetfieldGcPureI,
            &[BoxRef::from_opref(OpRef::ref_op(10))],
            descr,
        );
        op.pos.set(OpRef::int_op(0));

        let mut pass = OptPure::new();
        let mut ctx = OptContext::with_num_inputs(4, 0);
        ctx.make_constant(OpRef::ref_op(10), Value::Ref(GcRef(ptr)));
        pass.setup();

        // Resolve forwarded args (mirrors propagate_from_pass_range) so the op
        // carries the canonical const box the pass reads via get_constant_box.
        let resolved = ctx
            .resolve_box_box_opt(&op.arg(0))
            .expect("constant arg resolves");
        op.setarg(0, resolved);

        assert_eq!(ctx.constant_fold(&op), Some(Value::Int(123)));
        let result = pass.propagate_forward(&op, &std::rc::Rc::new(op.clone()), &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(
            ctx.get_box_replacement_box(OpRef::int_op(0))
                .and_then(|cb| cb.const_int()),
            Some(123)
        );

        unsafe {
            drop(Box::from_raw(ptr as *mut TestIntFieldObject));
        }
    }

    #[test]
    fn test_constant_fold_getfield_gc_pure_f_from_constant_object() {
        let object = Box::new(TestFloatFieldObject { value: 3.5 });
        let ptr = Box::into_raw(object) as usize;

        let descr = make_field_descr_full(2, 0, 8, Type::Float, true);
        let mut op = Op::with_descr(
            OpCode::GetfieldGcPureF,
            &[BoxRef::from_opref(OpRef::ref_op(10))],
            descr,
        );
        op.pos.set(OpRef::float_op(0));

        let mut pass = OptPure::new();
        let mut ctx = OptContext::with_num_inputs(4, 0);
        ctx.make_constant(OpRef::ref_op(10), Value::Ref(GcRef(ptr)));
        pass.setup();

        // Resolve forwarded args (mirrors propagate_from_pass_range) so the op
        // carries the canonical const box the pass reads via get_constant_box.
        let resolved = ctx
            .resolve_box_box_opt(&op.arg(0))
            .expect("constant arg resolves");
        op.setarg(0, resolved);

        assert_eq!(ctx.constant_fold(&op), Some(Value::Float(3.5)));
        let result = pass.propagate_forward(&op, &std::rc::Rc::new(op.clone()), &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(
            ctx.get_box_replacement_box(OpRef::float_op(0))
                .and_then(|b| ctx.get_constant_float_box(&b)),
            Some(3.5)
        );

        unsafe {
            drop(Box::from_raw(ptr as *mut TestFloatFieldObject));
        }
    }

    #[test]
    fn test_constant_fold_getfield_gc_pure_r_from_constant_object() {
        let object = Box::new(TestRefFieldObject {
            value: 0x1234_5678usize,
        });
        let ptr = Box::into_raw(object) as usize;

        let descr = make_field_descr_full(3, 0, std::mem::size_of::<usize>(), Type::Ref, true);
        let mut op = Op::with_descr(
            OpCode::GetfieldGcPureR,
            &[BoxRef::from_opref(OpRef::ref_op(10))],
            descr,
        );
        op.pos.set(OpRef::ref_op(0));

        let mut pass = OptPure::new();
        let mut ctx = OptContext::with_num_inputs(4, 0);
        ctx.make_constant(OpRef::ref_op(10), Value::Ref(GcRef(ptr)));
        pass.setup();

        // Resolve forwarded args (mirrors propagate_from_pass_range) so the op
        // carries the canonical const box the pass reads via get_constant_box.
        let resolved = ctx
            .resolve_box_box_opt(&op.arg(0))
            .expect("constant arg resolves");
        op.setarg(0, resolved);

        assert_eq!(
            ctx.constant_fold(&op),
            Some(Value::Ref(GcRef(0x1234_5678usize)))
        );
        let result = pass.propagate_forward(&op, &std::rc::Rc::new(op.clone()), &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(
            ctx.get_box_replacement_box(OpRef::ref_op(0))
                .and_then(|cb| cb.const_value()),
            Some(Value::Ref(GcRef(0x1234_5678usize)))
        );

        unsafe {
            drop(Box::from_raw(ptr as *mut TestRefFieldObject));
        }
    }

    #[test]
    #[should_panic(expected = "must be a gcref")]
    fn test_constant_fold_getfield_gc_pure_does_not_treat_int_constant_as_gc_pointer() {
        // Upstream `optimizer.py:829-832 protect_speculative_operation`
        // derefs `op.getarg(0)` via `getref_base()` — only `ConstPtr`
        // supports that.  RPython's type system makes a `GETFIELD_GC_
        // PURE_I` with `ConstInt` arg0 unconstructible.  Pyre's
        // strict-orthodoxy port panics on the variant mismatch instead
        // of returning `None`; this test pins that behavior.
        let descr = make_field_descr_full(4, 0, 8, Type::Int, true);
        let mut op = Op::with_descr(
            OpCode::GetfieldGcPureI,
            &[BoxRef::from_opref(OpRef::int_op(10))],
            descr,
        );
        op.pos.set(OpRef::int_op(0));

        let mut ctx = OptContext::with_num_inputs(4, 0);
        ctx.make_constant(OpRef::int_op(10), Value::Int(2));

        // Resolve forwarded args (mirrors propagate_from_pass_range) so the op
        // carries the canonical const box the pass reads via get_constant_box.
        let resolved = ctx
            .resolve_box_box_opt(&op.arg(0))
            .expect("constant arg resolves");
        op.setarg(0, resolved);

        let _ = ctx.constant_fold(&op);
    }

    #[test]
    fn test_guard_no_exception_after_removed_call_pure() {
        // CALL_PURE_I(same args) × 2 → second removed → GUARD_NO_EXCEPTION after removed
        let mut ops = vec![
            Op::new(
                OpCode::CallPureI,
                &[
                    BoxRef::from_opref(OpRef::int_op(100)),
                    BoxRef::from_opref(OpRef::int_op(101)),
                ],
            ),
            Op::new(OpCode::GuardNoException, &[]),
            Op::new(
                OpCode::CallPureI,
                &[
                    BoxRef::from_opref(OpRef::int_op(100)),
                    BoxRef::from_opref(OpRef::int_op(101)),
                ],
            ),
            Op::new(OpCode::GuardNoException, &[]), // should be removed
            Op::new(OpCode::Finish, &[]),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let (ops, snapshots) = super::super::seed_empty_guard_snapshots(&ops);
        opt.snapshot_boxes = snapshots;
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecAssoc::new(), 1024);

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

    /// pure.py:62 / :72-74 same_box semantics for constant args.
    /// history.py:204-205 / :244 — `same_box(a, b) == same_constant(a, b)`
    /// for Const subclasses, so cache hits are value-equality. With
    /// inline `ConstInt.value`, two `make_constant_int(5)` calls return
    /// the same `OpRef::ConstInt(5)` and the cache hit is by
    /// OpRef equality (which is now value equality).
    #[test]
    fn lookup_pure_matches_same_value_constants_across_slots() {
        let mut pass = OptPure::new();
        let mut ctx = OptContext::new(0);

        let c5_a = ctx.make_constant_int(5);
        let c5_b = ctx.make_constant_int(5);
        assert_eq!(
            c5_a, c5_b,
            "two ConstInt(5) boxes are same_constant-equal (history.py:251)"
        );
        assert_eq!(ctx.get_constant(c5_a), ctx.get_constant(c5_b));

        // Cache `IntAdd(c5_a, x)` and look up `IntAdd(c5_b, x)`.
        let x = OpRef::int_op(7);
        ctx.materialize_box_at(x);
        pass.pure_from_args2(OpCode::IntAdd, c5_a, x, OpRef::int_op(42));

        let mut q = Op::new(
            OpCode::IntAdd,
            &[BoxRef::from_opref(c5_b), BoxRef::from_opref(x)],
        );
        q.pos.set(OpRef::int_op(99));
        assert_eq!(
            pass.get_pure_result(&q, &ctx),
            Some(OpRef::int_op(42)),
            "lookup_pure must use same_box for constant args (history.py:204)"
        );

        // A non-constant slot mismatch must still miss.
        let mut q_miss = Op::new(
            OpCode::IntAdd,
            &[
                BoxRef::from_opref(c5_b),
                BoxRef::from_opref(OpRef::int_op(8)),
            ],
        );
        q_miss.pos.set(OpRef::int_op(100));
        assert_eq!(pass.get_pure_result(&q_miss, &ctx), None);
    }

    /// pure.py:81-93 applies get_box_replacement to the lookup-side op args
    /// before comparing them with stored pure op args.
    #[test]
    fn lookup_pure_replaces_query_args_before_matching() {
        let mut pass = OptPure::new();
        let mut ctx = OptContext::new(0);

        let query_arg = OpRef::int_op(7);
        let canonical_arg = OpRef::int_op(8);
        let other_arg = OpRef::int_op(9);
        let result = OpRef::int_op(42);
        let b_query = ctx.materialize_box_at(query_arg);
        let b_canonical = ctx.materialize_box_at(canonical_arg);
        ctx.make_equal_to(&b_query, &b_canonical);
        ctx.materialize_box_at(other_arg);

        pass.pure_from_args2(OpCode::IntAdd, canonical_arg, other_arg, result);

        let mut q = Op::new(
            OpCode::IntAdd,
            &[BoxRef::from_opref(query_arg), BoxRef::from_opref(other_arg)],
        );
        q.pos.set(OpRef::int_op(99));
        assert_eq!(
            pass.get_pure_result(&q, &ctx),
            Some(result),
            "lookup_pure must apply get_box_replacement to query args"
        );
    }

    #[test]
    fn test_pure_and_pure_from_args() {
        let mut pass = OptPure::new();

        // Manually record a pure operation via the API
        let mut op = Op::new(
            OpCode::IntAdd,
            &[
                BoxRef::from_opref(OpRef::int_op(10)),
                BoxRef::from_opref(OpRef::int_op(20)),
            ],
        );
        op.pos.set(OpRef::int_op(0));
        pass.pure(&op);

        let mut ctx = OptContext::new(0);
        // Bind the operand positions canonically so same_box resolves the
        // looked-up op's args to the same boxes the cache recorded.
        for p in [10, 20, 30, 40] {
            ctx.materialize_box_at(OpRef::int_op(p));
        }

        // Should find it via get_pure_result
        let lookup_op = Op::new(
            OpCode::IntAdd,
            &[
                BoxRef::from_opref(OpRef::int_op(10)),
                BoxRef::from_opref(OpRef::int_op(20)),
            ],
        );
        assert!(pass.get_pure_result(&lookup_op, &ctx).is_some());

        // pure_from_args
        pass.pure_from_args(
            OpCode::IntMul,
            &[OpRef::int_op(30), OpRef::int_op(40)],
            OpRef::int_op(5),
        );
        let mut lookup_mul = Op::new(
            OpCode::IntMul,
            &[
                BoxRef::from_opref(OpRef::int_op(30)),
                BoxRef::from_opref(OpRef::int_op(40)),
            ],
        );
        lookup_mul.pos.set(OpRef::int_op(99));
        assert!(pass.get_pure_result(&lookup_mul, &ctx).is_some());
    }

    #[test]
    fn test_extra_call_pure() {
        let mut pass = OptPure::new();

        // Inject extra_call_pure from a previous loop iteration
        let args = vec![OpRef::int_op(100), OpRef::int_op(101)];
        pass.set_extra_call_pure(vec![(args.clone(), OpRef::int_op(50))]);

        // extra_call_pure entries are searched via optimize_call_pure_old
        // in the CALL_PURE handler (pure.py:204-210), not via
        // lookup_known_result (which only searches known_result_call_pure).
        let key = super::PureOpKey {
            opcode: OpCode::CallPureI,
            args,
            descr_identity: None,
        };
        // Verify entries exist in extra_call_pure
        assert_eq!(pass.extra_call_pure.len(), 1);
        match &pass.extra_call_pure[0] {
            super::ExtraCallPureEntry::Direct { key: k, result } => {
                assert_eq!(k, &key);
                assert_eq!(*result, OpRef::int_op(50));
            }
            _ => panic!("expected Direct entry"),
        }
    }

    #[test]
    fn test_known_result_call_pure_lookup() {
        let mut pass = OptPure::new();
        let mut ctx = OptContext::with_num_inputs(4, 0);
        // Bind the matched call args canonically so same_box resolves them.
        ctx.materialize_box_at(OpRef::int_op(100));
        ctx.materialize_box_at(OpRef::int_op(101));

        // pure.py:214: self.known_result_call_pure.append(op)
        pass.known_result_call_pure.push(super::KnownResultEntry {
            descr_identity: None,
            args: vec![OpRef::int_op(100), OpRef::int_op(101)],
            result: OpRef::int_op(50),
        });

        // CALL_PURE lookup: start_index=0, descr matches (both None), args match
        let op = Op::new(
            OpCode::CallPureI,
            &[
                BoxRef::from_opref(OpRef::int_op(100)),
                BoxRef::from_opref(OpRef::int_op(101)),
            ],
        );
        assert_eq!(
            pass.lookup_known_result(&op, 0, &ctx),
            Some(OpRef::int_op(50))
        );

        // COND_CALL_VALUE lookup: start_index=1, skip arg(0)
        let cond_op = Op::new(
            OpCode::CondCallValueI,
            &[
                BoxRef::from_opref(OpRef::int_op(999)),
                BoxRef::from_opref(OpRef::int_op(100)),
                BoxRef::from_opref(OpRef::int_op(101)),
            ],
        );
        assert_eq!(
            pass.lookup_known_result(&cond_op, 1, &ctx),
            Some(OpRef::int_op(50))
        );

        // Args mismatch → None
        let bad_args = Op::new(
            OpCode::CallPureI,
            &[
                BoxRef::from_opref(OpRef::int_op(100)),
                BoxRef::from_opref(OpRef::int_op(999)),
            ],
        );
        assert_eq!(pass.lookup_known_result(&bad_args, 0, &ctx), None);
    }

    #[test]
    fn test_imported_short_pure_result_is_reexported_to_short_preamble() {
        // Imported pure ops (from previous peeling cycle) should be
        // re-exported to ShortBoxes via short_preamble_pure_ops.
        let mut pass = OptPure::new();
        let mut ctx = OptContext::with_num_inputs(6, 0);
        // history.py:227 — the imported pure op carries an inline `Const`
        // arg (`ConstInt.value`). seed_constant takes the inline branch (no
        // const_pool), and the constant is recognised downstream via
        // `is_constant()`, so the short-preamble producer re-exports the op
        // without any const_pool / known_constants bridge.
        let const_opref = OpRef::const_int(7);
        ctx.seed_constant(const_opref, majit_ir::Value::Int(7));
        let imported = crate::optimizeopt::ImportedShortPureOp::new(
            &mut ctx,
            OpCode::IntAdd,
            None,
            vec![
                crate::optimizeopt::ImportedShortPureArg::OpRef(OpRef::int_op(0)),
                crate::optimizeopt::ImportedShortPureArg::Const(
                    majit_ir::Value::Int(7),
                    const_opref,
                ),
            ],
            OpRef::int_op(2),
            OpRef::int_op(2),
            false,
        );
        ctx.imported_short_pure_ops.push(imported);

        pass.setup();
        pass.install_preamble_pure_ops(&ctx);

        // Label args don't include OpRef::int_op(2), so the pure op should be produced.
        let mut sb = crate::optimizeopt::shortpreamble::ShortBoxes::with_label_args(&[
            OpRef::int_op(0),
            OpRef::int_op(1),
        ]);
        pass.produce_potential_short_preamble_ops(&mut sb, &mut ctx);
        let collected = sb.produced_ops(&mut ctx);
        assert_eq!(collected.len(), 1);
        assert!(matches!(
            collected[0].1.kind,
            crate::optimizeopt::shortpreamble::PreambleOpKind::Pure
        ));
        assert_eq!(collected[0].1.preamble_op.opcode, OpCode::IntAdd);
        assert_eq!(collected[0].1.preamble_op.pos.get(), OpRef::int_op(2));
    }

    #[test]
    fn test_imported_short_call_pure_result_replays_into_pure_cache() {
        let mut pass = OptPure::new();
        let mut ctx = OptContext::with_num_inputs(8, 0);
        // Bind the non-const call arg position so same_box resolves the
        // dispatched op's arg to the same box the imported op recorded.
        ctx.materialize_box_at(OpRef::int_op(0));
        let const_opref = OpRef::const_int(0x1234);
        let call_descr = majit_ir::descr::make_call_descr_full(
            77,
            vec![majit_ir::Type::Int, majit_ir::Type::Int],
            majit_ir::Type::Int,
            true,
            8,
            majit_ir::EffectInfo::new(
                majit_ir::ExtraEffect::ElidableCannotRaise,
                majit_ir::OopSpecIndex::None,
            ),
        );
        let imported = crate::optimizeopt::ImportedShortPureOp::new(
            &mut ctx,
            OpCode::CallPureI,
            Some(call_descr.clone()),
            vec![
                crate::optimizeopt::ImportedShortPureArg::Const(
                    majit_ir::Value::Int(0x1234),
                    const_opref,
                ),
                crate::optimizeopt::ImportedShortPureArg::OpRef(OpRef::int_op(0)),
            ],
            OpRef::int_op(1),
            OpRef::int_op(1),
            false,
        );
        initialize_imported_short_pure_builder(
            &mut ctx,
            (*imported.pop.preamble_op).clone(),
            Some(1),
        );
        ctx.imported_short_pure_ops.push(imported);

        pass.setup();
        pass.install_preamble_pure_ops(&ctx);

        let mut op = Op::new(
            OpCode::CallPureI,
            &[
                BoxRef::from_opref(const_opref),
                BoxRef::from_opref(OpRef::int_op(0)),
            ],
        );
        op.pos.set(OpRef::int_op(2));
        op.setdescr(call_descr);
        // Register the dispatched op as the producer at its position so the
        // collapsed `from_bound_op(op_rc)` resolves to the same box the test
        // reads back via `get_box_replacement_box(pos)` (mirrors production
        // input-op binding).
        let op_rc = std::rc::Rc::new(op.clone());
        ctx.bind_input_resops(std::slice::from_ref(&op_rc));
        let result = pass.propagate_forward(&op, &op_rc, &mut ctx);
        assert!(matches!(result, OptimizationResult::Remove));
        assert_eq!(
            ctx.get_box_replacement_box(OpRef::int_op(2))
                .map(|b| b.to_opref()),
            Some(OpRef::int_op(1))
        );
    }

    #[test]
    fn test_short_preamble_collects_pure_op_candidate() {
        let mut pass = OptPure::new();
        let mut ctx = OptContext::with_num_inputs(4, 0);
        pass.setup();

        let mut op = Op::new(
            OpCode::IntAdd,
            &[
                BoxRef::from_opref(OpRef::int_op(0)),
                BoxRef::from_opref(OpRef::int_op(1)),
            ],
        );
        op.pos.set(OpRef::int_op(2));
        let result = pass.propagate_forward(&op, &std::rc::Rc::new(op.clone()), &mut ctx);
        assert!(matches!(result, OptimizationResult::PassOn));

        let mut sb = crate::optimizeopt::shortpreamble::ShortBoxes::with_label_args(&[
            OpRef::int_op(0),
            OpRef::int_op(1),
            OpRef::int_op(2),
        ]);
        pass.produce_potential_short_preamble_ops(&mut sb, &mut ctx);
        let collected = sb.produced_ops(&mut ctx);
        assert_eq!(collected.len(), 1);
        assert!(matches!(
            collected[0].1.kind,
            crate::optimizeopt::shortpreamble::PreambleOpKind::Pure
        ));
        assert_eq!(collected[0].1.preamble_op.opcode, OpCode::IntAdd);
    }

    #[test]
    fn test_short_preamble_collects_non_raising_call_pure_candidate() {
        let mut pass = OptPure::new();
        let mut ctx = OptContext::with_num_inputs(6, 0);
        pass.setup();

        let mut op = Op::new(
            OpCode::CallPureI,
            &[
                BoxRef::from_opref(OpRef::int_op(100)),
                BoxRef::from_opref(OpRef::int_op(0)),
                BoxRef::from_opref(OpRef::int_op(1)),
            ],
        );
        op.pos.set(OpRef::int_op(2));
        op.setdescr(majit_ir::descr::make_call_descr(
            vec![
                majit_ir::Type::Int,
                majit_ir::Type::Int,
                majit_ir::Type::Int,
            ],
            majit_ir::Type::Int,
            majit_ir::EffectInfo::new(
                majit_ir::ExtraEffect::ElidableCannotRaise,
                majit_ir::OopSpecIndex::None,
            ),
        ));
        let result = pass.propagate_forward(&op, &std::rc::Rc::new(op.clone()), &mut ctx);
        match result {
            OptimizationResult::Emit(emitted) => assert_eq!(emitted.opcode, OpCode::CallI),
            other => panic!("expected emitted demoted call, got {other:?}"),
        }

        let mut sb = crate::optimizeopt::shortpreamble::ShortBoxes::with_label_args(&[
            OpRef::int_op(0),
            OpRef::int_op(1),
            OpRef::int_op(2),
            OpRef::int_op(100),
        ]);
        pass.produce_potential_short_preamble_ops(&mut sb, &mut ctx);
        let collected = sb.produced_ops(&mut ctx);
        assert_eq!(collected.len(), 1);
        assert!(matches!(
            collected[0].1.kind,
            crate::optimizeopt::shortpreamble::PreambleOpKind::Pure
        ));
        assert_eq!(collected[0].1.preamble_op.opcode, OpCode::CallPureI);
    }

    #[test]
    fn test_short_preamble_collects_loopinvariant_candidate() {
        let mut rewrite = crate::optimizeopt::rewrite::OptRewrite::new();
        let mut pass = OptPure::new();
        let mut ctx = OptContext::with_num_inputs(6, 0);
        // func pointer arg must be a known constant for OptRewrite tracking
        ctx.seed_constant(OpRef::int_op(100), majit_ir::Value::Int(0xCAFE));
        rewrite.setup();
        pass.setup();

        let mut op = Op::new(
            OpCode::CallLoopinvariantI,
            &[
                BoxRef::from_opref(OpRef::int_op(100)),
                BoxRef::from_opref(OpRef::int_op(0)),
            ],
        );
        op.pos.set(OpRef::int_op(2));
        op.setdescr(majit_ir::descr::make_call_descr(
            vec![majit_ir::Type::Int, majit_ir::Type::Int],
            majit_ir::Type::Int,
            majit_ir::EffectInfo::new(
                majit_ir::ExtraEffect::ElidableCannotRaise,
                majit_ir::OopSpecIndex::None,
            ),
        ));
        // optimizer.py:651-652 setarg loop parity: canonicalize args the way
        // propagate_from_pass_range does before propagate_forward.
        for i in 0..op.num_args() {
            op.setarg(
                i,
                ctx.resolve_box_box_opt(&op.arg(i))
                    .unwrap_or_else(|| op.arg(i).clone()),
            );
        }
        // OptRewrite demotes CallLoopinvariantI → CallI
        let rewrite_result =
            rewrite.propagate_forward(&op, &std::rc::Rc::new(op.clone()), &mut ctx);
        let demoted = match rewrite_result {
            OptimizationResult::Emit(emitted) => emitted,
            other => panic!("expected OptRewrite to emit demoted call, got {other:?}"),
        };
        assert_eq!(demoted.opcode, OpCode::CallI);
        // OptPure sees the demoted CallI
        let result = pass.propagate_forward(&demoted, &std::rc::Rc::new(demoted.clone()), &mut ctx);
        match result {
            OptimizationResult::Emit(emitted) => assert_eq!(emitted.opcode, OpCode::CallI),
            OptimizationResult::PassOn => {} // PassOn is also acceptable
            other => panic!("expected emitted or pass-on from OptPure, got {other:?}"),
        }

        // OptRewrite tracks loopinvariant for short preamble collection
        let mut sb = crate::optimizeopt::shortpreamble::ShortBoxes::with_label_args(&[
            OpRef::int_op(0),
            OpRef::int_op(2),
            OpRef::int_op(100),
        ]);
        rewrite.produce_potential_short_preamble_ops(&mut sb, &mut ctx);
        let collected = sb.produced_ops(&mut ctx);
        assert_eq!(collected.len(), 1);
        assert!(matches!(
            collected[0].1.kind,
            crate::optimizeopt::shortpreamble::PreambleOpKind::LoopInvariant
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
        pass.pure_from_args(
            OpCode::IntAdd,
            &[OpRef::int_op(10), OpRef::int_op(20)],
            OpRef::int_op(30),
        );

        // same_box: identity comparison (no constants, no forwarding)
        let sb = |a: OpRef, b: OpRef| a == b;
        // lookup2 should find it
        assert!(
            pass.lookup2(
                OpCode::IntAdd,
                OpRef::int_op(10),
                OpRef::int_op(20),
                None,
                false,
                sb
            )
            .is_some()
        );
        // lookup2 with commutative should find swapped
        assert!(
            pass.lookup2(
                OpCode::IntAdd,
                OpRef::int_op(20),
                OpRef::int_op(10),
                None,
                true,
                sb
            )
            .is_some()
        );
        // Non-commutative swapped should NOT find it
        assert!(
            pass.lookup2(
                OpCode::IntAdd,
                OpRef::int_op(20),
                OpRef::int_op(10),
                None,
                false,
                sb
            )
            .is_none()
        );

        // lookup1 for a unary op
        pass.pure_from_args(OpCode::IntNeg, &[OpRef::int_op(10)], OpRef::int_op(40));
        assert!(
            pass.lookup1(OpCode::IntNeg, OpRef::int_op(10), None, sb)
                .is_some()
        );
        assert!(
            pass.lookup1(OpCode::IntNeg, OpRef::int_op(99), None, sb)
                .is_none()
        );
    }

    #[test]
    fn test_cond_call_value_cse() {
        // COND_CALL_VALUE_I(cond, func, arg) → CSE using args[1..]
        // A second COND_CALL_VALUE_I with same func+arg should reuse result.
        let mut ops = vec![
            Op::new(
                OpCode::CondCallValueI,
                &[
                    BoxRef::from_opref(OpRef::int_op(100)),
                    BoxRef::from_opref(OpRef::int_op(200)),
                    BoxRef::from_opref(OpRef::int_op(300)),
                ],
            ),
            Op::new(
                OpCode::CondCallValueI,
                &[
                    BoxRef::from_opref(OpRef::int_op(101)),
                    BoxRef::from_opref(OpRef::int_op(200)),
                    BoxRef::from_opref(OpRef::int_op(300)),
                ],
            ),
        ];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.add_pass(Box::new(OptPure::new()));
        let result =
            opt.optimize_with_constants_and_inputs(&ops, &mut majit_ir::VecAssoc::new(), 1024);

        // First COND_CALL_VALUE emitted, second removed by CSE
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, OpCode::CondCallValueI);
    }

    #[test]
    fn test_cond_call_value_uses_call_pure_results_starting_at_arg1() {
        let mut ops = vec![Op::new(
            OpCode::CondCallValueI,
            &[
                BoxRef::from_opref(OpRef::int_op(100)),
                BoxRef::from_opref(OpRef::const_int(0xCAFE)),
                BoxRef::from_opref(OpRef::const_int(7)),
            ],
        )];
        assign_positions(&mut ops);

        let mut opt = Optimizer::new();
        opt.record_call_pure_result(vec![Value::Int(0xCAFE), Value::Int(7)], Value::Int(42));
        opt.add_pass(Box::new(OptPure::new()));

        let mut constants: majit_ir::VecAssoc<u32, majit_ir::Value> = majit_ir::VecAssoc::new();
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 1);

        assert!(result.is_empty());
        assert_eq!(
            constants.get(&ops[0].pos.get().raw()),
            Some(&majit_ir::Value::Int(42))
        );
    }

    /// REF analog of `test_cond_call_value_uses_call_pure_results_*` /
    /// the Int `test_call_pure_results`: a `CALL_PURE_R` whose all-const
    /// args (funcptr + Ref) hit a seeded `call_pure_results` entry must
    /// be removed by OptPure and its result made a Ref constant.
    ///
    /// pure.py:230-234 `optimize_CALL_PURE_R` → start_index=0 (the
    /// funcptr is part of the lookup key).  `_can_optimize_call_pure`
    /// (optimizer.py:215-226) reads `get_constant_box(arg)` per arg →
    /// `Value::Ref(ConstPtr)` for the Ref arg — value equality, so the
    /// 2nd identical call collapses to the 1st's stored result.
    ///
    /// resoperation.py:638 `RefOp.type = 'r'`: the folded constant must
    /// be `Value::Ref` (history.py:314 ConstPtr), NOT `Value::Int`
    /// (history.py:220 ConstInt) of the same numeric value.
    #[test]
    fn test_call_pure_r_results_folds_second_identical_call() {
        let func_const = OpRef::const_int(0xCAFE);
        let ref_arg_const = OpRef::const_ptr(GcRef(0x4242));
        let result_ptr = GcRef(0xDEAD_BEEF);

        let mut ops = vec![Op::new(
            OpCode::CallPureR,
            &[
                BoxRef::from_opref(func_const),
                BoxRef::from_opref(ref_arg_const),
            ],
        )];
        assign_positions(&mut ops);
        // `assign_positions` types op.pos via result_type() → CallPureR
        // gives a Ref-typed position OpRef.
        assert_eq!(
            ops[0].pos.get().ty(),
            Some(Type::Ref),
            "CallPureR position must be Ref-typed (resoperation.py:638)"
        );

        let mut opt = Optimizer::new();
        // Seed the cross-call result keyed on [funcptr, ref_arg] (Ref
        // value), as record_result_of_call_pure does at trace time.
        opt.record_call_pure_result(
            vec![Value::Int(0xCAFE), Value::Ref(GcRef(0x4242))],
            Value::Ref(result_ptr),
        );
        opt.add_pass(Box::new(OptPure::new()));

        let mut constants: majit_ir::VecAssoc<u32, majit_ir::Value> = majit_ir::VecAssoc::new();
        let result = opt.optimize_with_constants_and_inputs(&ops, &mut constants, 1);

        // The 2nd identical CallPureR collapses to the cached constant —
        // the op is removed entirely.
        assert!(
            result.is_empty(),
            "identical-const-args CallPureR must be removed by call_pure_results fold; got {result:?}"
        );
        let folded = constants.get(&ops[0].pos.get().raw());
        // The folded constant must be Ref-typed, value == result_ptr.
        assert_eq!(
            folded,
            Some(&Value::Ref(result_ptr)),
            "CallPureR fold must yield ConstPtr(result_ptr), got {folded:?}"
        );
        // Critically: NOT a ConstInt of the same numeric value.
        assert!(
            !matches!(folded, Some(Value::Int(_))),
            "folded CallPureR constant aliased to Value::Int — ConstPtr/ConstInt distinction lost"
        );
    }
}
