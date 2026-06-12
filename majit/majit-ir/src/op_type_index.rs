use std::borrow::Cow;

use crate::resoperation::{Op, OpRef};
use crate::value::{InputArg, Type};

/// O(1) `OpRef → Type` lookup over a trace's inputargs + ops + constant pool.
///
/// rpython/jit/metainterp/history.py:220 `ConstInt.type = INT`,
/// rpython/jit/metainterp/resoperation.py:567 `IntOp.type = 'i'` —
/// RPython reads `box.type` directly from the Box object. Pyre's typed
/// `OpRef` variants carry that type tag for value boxes; this index is
/// the compatibility boundary for raw/index-keyed consumers that still
/// need to recover type from `op.type_`, `inputarg.tp`, or caller-supplied
/// `constant_types` (resume.py ResumeDataLoopMemo).
///
/// `inputarg_pos` and `op_pos` are stored as `Cow` so callers may
/// either let `new` build them eagerly or share pre-built indexes that
/// outlive the trace (e.g. `RegAlloc<'a>`).
pub struct OpTypeIndex<'a, T: AsRef<Op> = Op> {
    inputargs: &'a [InputArg],
    ops: &'a [T],
    /// `inputarg_pos[raw] = slice index in inputargs`, sentinel
    /// [`NO_POS`] for unset slots. `arg.index` raw uniqueness is
    /// enforced at build time, mirroring RPython's backend uniqueness
    /// assertion (x86/assembler.py:516-518 + aarch64/assembler.py:54-56
    /// `assert len(set(inputargs)) == len(inputargs)`).
    inputarg_pos: Cow<'a, [u32]>,
    /// `op_pos[raw] = slice index in ops`, sentinel [`NO_POS`] for
    /// unset slots and Void/None ops. `op.pos.raw()` raw uniqueness is
    /// enforced at build time per RPython Box identity (Box `is`
    /// semantics in `rpython/jit/metainterp/resoperation.py:38`).
    op_pos: Cow<'a, [u32]>,
}

/// Sentinel for "no entry at this raw u32 slot" in `inputarg_pos` /
/// `op_pos` arrays. Production raw u32 values come from monotonic
/// counters; the constant-pool side sets `CONST_BIT = 1 << 31` and is
/// gated out before reaching these arrays, so raw values land well
/// below `u32::MAX`.
pub const NO_POS: u32 = u32::MAX;

impl<'a, T: AsRef<Op>> OpTypeIndex<'a, T> {
    pub fn new(inputargs: &'a [InputArg], ops: &'a [T]) -> Self {
        let inputarg_pos = Self::build_inputarg_pos(inputargs);
        let op_pos = Self::build_op_pos(ops);
        Self {
            inputargs,
            ops,
            inputarg_pos: Cow::Owned(inputarg_pos),
            op_pos: Cow::Owned(op_pos),
        }
    }

    /// Construct from pre-built indexes (e.g. owned by `RegAlloc<'a>`).
    /// O(1) — borrows slices instead of rebuilding the position arrays.
    pub fn from_parts(
        inputargs: &'a [InputArg],
        ops: &'a [T],
        inputarg_pos: &'a [u32],
        op_pos: &'a [u32],
    ) -> Self {
        Self {
            inputargs,
            ops,
            inputarg_pos: Cow::Borrowed(inputarg_pos),
            op_pos: Cow::Borrowed(op_pos),
        }
    }

    /// Build `inputarg_pos` indexed by `arg.index`.
    ///
    /// RPython's backend enforces `assert len(set(inputargs)) == len(inputargs)`
    /// at loop/bridge entry (x86/assembler.py:516-518 +
    /// aarch64/assembler.py:54-56). Raw u32 uniqueness is the dual
    /// invariant in pyre's flat OpRef namespace: an `InputArgInt(7)` +
    /// `InputArgRef(7)` collision would silently keep only the later
    /// one in any variant-blind reader. Hard-panic on raw collision so
    /// the violation surfaces here rather than as a wrong-type guard
    /// fail much further along.
    pub fn build_inputarg_pos(inputargs: &[InputArg]) -> Vec<u32> {
        if inputargs.is_empty() {
            return Vec::new();
        }
        let max_raw = inputargs.iter().map(|a| a.index).max().unwrap_or(0);
        let mut pos: Vec<u32> = vec![NO_POS; max_raw as usize + 1];
        for (idx, arg) in inputargs.iter().enumerate() {
            let r = arg.index as usize;
            if pos[r] != NO_POS {
                panic!(
                    "OpTypeIndex: raw inputarg index {} bound to inputargs[{}] {:?} and inputargs[{}] {:?} — backend uniqueness violated",
                    arg.index, pos[r], inputargs[pos[r] as usize].tp, idx, arg.tp,
                );
            }
            pos[r] = idx as u32;
        }
        pos
    }

    /// Build `op_pos` indexed by `op.pos.raw()`. Filters out Void-typed
    /// ops because RPython's `box.type` only exists on Box-bearing ops;
    /// a Void op is not a Box and must never shadow an inputarg slot.
    ///
    /// RPython Box identity gives a one-to-one map from a Box object to
    /// its producing ResOperation; two Box-bearing ops sharing the same
    /// raw OpRef payload (e.g. `IntOp(7)` + `RefOp(7)`) is a Box-identity
    /// violation even though the typed variants disambiguate the type
    /// tag, because pyre's backend boundary keys by raw u32 and would
    /// silently keep only the later op. Hard-panic on raw collision so
    /// the violation surfaces here.
    pub fn build_op_pos(ops: &[T]) -> Vec<u32> {
        let max_raw = ops
            .iter()
            .map(|op| op.as_ref())
            .filter(|op| !op.pos.get().is_none() && op.type_ != Type::Void)
            .map(|op| op.pos.get().raw())
            .max();
        let Some(max_raw) = max_raw else {
            return Vec::new();
        };
        let mut pos: Vec<u32> = vec![NO_POS; max_raw as usize + 1];
        for (idx, op) in ops.iter().enumerate() {
            let op = op.as_ref();
            if op.pos.get().is_none() || op.type_ == Type::Void {
                continue;
            }
            let r = op.pos.get().raw() as usize;
            if pos[r] != NO_POS {
                panic!(
                    "OpTypeIndex: raw {} bound to ops[{}] {:?} and ops[{}] {:?} — Box identity broken",
                    op.pos.get().raw(),
                    pos[r],
                    ops[pos[r] as usize].as_ref().opcode,
                    idx,
                    op.opcode,
                );
            }
            pos[r] = idx as u32;
        }
        pos
    }

    /// `box.type` lookup. resoperation.py:29 / history.py:182: a typed
    /// Box carries its `.type` on the object itself, and pyre encodes
    /// that on the `OpRef` variant tag (`ConstInt`/`ConstPtr`/`ConstFloat`,
    /// `InputArg{Int,Ref,Float}`, `{Int,Float,Ref,Void}Op`). The tag IS
    /// the answer — no positional or side-table lookup is needed. Returns
    /// `None` for `OpRef::NONE`, `TempVar` (neither is a Box), or a
    /// `Void`-tagged op (`Void` is not a valid Box type).
    pub fn opref_type(&self, opref: OpRef) -> Option<Type> {
        self.opref_type_at_or_after(opref, None)
    }

    /// `box.type` lookup with a trace-position hint.
    ///
    /// The `op_index` hint is vestigial. It once disambiguated a flat
    /// `OpRef(u32)` namespace in which an InputArg and a later
    /// ResOperation could share an id, so the inputarg type was used
    /// until the op's result was defined. Typed `OpRef` variants now
    /// resolve that directly — `InputArgInt(0)` and `IntOp(0)` are
    /// distinct values — so this is identical to `opref_type`; the
    /// parameter is retained for call-site compatibility.
    pub fn opref_type_at(&self, opref: OpRef, op_index: usize) -> Option<Type> {
        self.opref_type_at_or_after(opref, Some(op_index))
    }

    fn opref_type_at_or_after(&self, opref: OpRef, _op_index: Option<usize>) -> Option<Type> {
        // history.py:182 / resoperation.py:29: `box.type` lives on the Box
        // object itself; pyre's typed OpRef variants carry the matching
        // type tag intrinsically, so the tag IS the answer. The only
        // tag-less oprefs are `OpRef::None` and `TempVar` — neither is a
        // Box — and both resolve to `None`, as does a `Void`-tagged op
        // (`Void` is not a valid Box type).
        opref.ty().filter(|tp| *tp != Type::Void)
    }

    /// Direct `OpRef → &Op` lookup; returns `None` for constants,
    /// inputargs, or `OpRef::NONE`.
    ///
    /// resoperation.py:29 `AbstractResOp` vs history.py:182 `AbstractValue`:
    /// only `*Op` variants (Int/Float/Ref/VoidOp) are produced by a
    /// `ResOperation`; `Const*` and `InputArg*` boxes have no producing
    /// op. Filter on the variant tag so a flat-`OpRef(u32)` collision
    /// (e.g. `InputArgInt(0)` sharing raw=0 with `IntOp(0)`) cannot
    /// surface a producer record for an inputarg or a constant via
    /// the raw-u32 position array.
    pub fn op_at(&self, opref: OpRef) -> Option<&Op> {
        if !matches!(
            opref,
            OpRef::IntOp(_) | OpRef::FloatOp(_) | OpRef::RefOp(_) | OpRef::VoidOp(_)
        ) {
            return None;
        }
        let idx = op_pos_lookup(&self.op_pos, opref.raw() as usize)?;
        Some(self.ops[idx].as_ref())
    }

    /// Inputarg-only type lookup; returns `None` if `opref` does not
    /// reference an inputarg. Used by callers that need to roll back to
    /// an OpRef's pre-redefinition type when an op later overwrote the
    /// same OpRef with a different type.
    pub fn inputarg_type(&self, opref: OpRef) -> Option<Type> {
        let idx = op_pos_lookup(&self.inputarg_pos, opref.raw() as usize)?;
        Some(self.inputargs[idx].tp)
    }

    /// Raw-keyed companion of `inputarg_type`. Used by callers that hold
    /// the inputarg position as a `u32` (e.g. backend op-var indices) and
    /// would otherwise need to mint a typed `OpRef` solely for the lookup.
    /// The `inputarg_pos` position array uses `OpRef::raw()` (less the
    /// inputarg base) internally, so the round-trip carries no information.
    pub fn inputarg_type_raw(&self, raw: u32) -> Option<Type> {
        let idx = op_pos_lookup(&self.inputarg_pos, raw as usize)?;
        Some(self.inputargs[idx].tp)
    }
}

/// Look up `raw` in a position array (`inputarg_pos` / `op_pos`).
/// Returns `Some(idx)` for a populated slot, `None` for an out-of-range
/// raw or a sentinel slot.
#[inline]
fn op_pos_lookup(pos: &[u32], raw: usize) -> Option<usize> {
    let entry = *pos.get(raw)?;
    if entry == NO_POS {
        return None;
    }
    Some(entry as usize)
}
