//! Liveness computation for flattened JitCode instructions.
//!
//! RPython equivalent: `rpython/jit/codewriter/liveness.py`.
//!
//! Expands `-live-` markers in the flattened instruction sequence to
//! include all values that are alive at that point (written before and
//! read afterwards). This information is used by guard operations in the
//! meta-interpreter to know which values to save on failure.
//!
//! The algorithm is a backward dataflow analysis that iterates to fixpoint.

use std::collections::{HashMap, HashSet};

use crate::flatten::{FlatOp, Label, RegKind, Register, SSARepr};
use crate::regalloc::RegAllocResult;

/// Compute liveness for a flattened function.
///
/// RPython: `liveness.py::compute_liveness(ssarepr)`.
///
/// Modifies the flattened ops in place: each `FlatOp::Live` marker
/// gets its `live_values` set populated with all [`Register`]s alive
/// at that point in the instruction sequence.
///
/// `regallocs` supplies the per-`Variable` `(kind, color)` mapping used
/// to convert FlatOp::Op operand `Variable`s to [`Register`]s for
/// the alive set — RPython works directly on Registers because
/// `serialize_op` already projects `Variable`→`Register` via
/// `getcolor`; pyre still walks the conversion here at the liveness
/// boundary because `FlatOp::Op` carries the pre-flatten
/// `SpaceOperation` whose operand reads are kind-driven.
/// RPython liveness.py:19-23.
///
/// Operand kinds read through `Variable.concretetype` directly via
/// `variable_to_register`, so this entry no longer needs the
/// surrounding `FunctionGraph` — the kind lives on every operand
/// Variable itself (upstream `Variable.concretetype` parity).
pub fn compute_liveness(flattened: &mut SSARepr, regallocs: &HashMap<RegKind, RegAllocResult>) {
    let mut label2alive: HashMap<Label, HashSet<Register>> = HashMap::new();

    loop {
        if !compute_liveness_pass(&mut flattened.insns, &mut label2alive, regallocs) {
            break;
        }
    }
    remove_repeated_live(&mut flattened.insns);
}

/// Resolve a `Variable` from a `FlatOp::Op` operand to its
/// [`Register`].
///
/// **Structural divergence (TODO)**: PyPy `liveness.py:67` walks
/// instructions whose register operands are already
/// [`Register`] / `ListOfKind` because `flatten_list()`
/// (`flatten.py:355-371`) projected `Variable` → `Register` at
/// flatten time.  Pyre's `FlatOp::Op` still carries
/// [`crate::model::SpaceOperation`] with `Variable` slots, so the
/// liveness pass has to redo the `getcolor` lookup here.  The fix
/// is to migrate `SpaceOperation` slots to `Register` so liveness
/// can read the kind off the operand directly; until that lands
/// this helper preserves the same `(kind, color)` answer.
///
/// **RPython invariant** (`flatten.py:382` `getcolor`): every
/// `Variable` has a single `(kind, color)` via
/// `getkind(v.concretetype)` + `regallocs[kind]`.  Reads kind via
/// `FunctionGraph::concretetype_of(var)` and color via
/// `RegAllocResult::color_for_variable(var)`; a miss panics — PyPy
/// would never fall back to other classes.
fn variable_to_register(
    var: &crate::flowspace::model::Variable,
    regallocs: &HashMap<RegKind, RegAllocResult>,
) -> Option<Register> {
    use crate::model::ConcreteType;
    use crate::model::FunctionGraph;
    let declared = FunctionGraph::concretetype_of(var);
    let kind = match declared {
        ConcreteType::Signed => Some(RegKind::Int),
        ConcreteType::GcRef => Some(RegKind::Ref),
        ConcreteType::Float => Some(RegKind::Float),
        ConcreteType::Void | ConcreteType::Unknown => None,
    };
    if let Some(kind) = kind {
        let ra = regallocs.get(&kind).unwrap_or_else(|| {
            panic!(
                "variable_to_register: graph declared kind {kind:?} for {var:?} \
                 but regallocs map is missing the entry",
            )
        });
        let color = ra.color_for_variable(var).unwrap_or_else(|| {
            let other_classes: Vec<_> = [RegKind::Int, RegKind::Ref, RegKind::Float]
                .iter()
                .filter(|k| **k != kind)
                .filter(|k| {
                    regallocs
                        .get(*k)
                        .is_some_and(|ra| ra.contains_variable(var))
                })
                .copied()
                .collect();
            panic!(
                "variable_to_register: graph declared kind {kind:?} for {var:?} \
                 but regallocs[{kind:?}] has no coloring (other classes with a \
                 coloring: {other_classes:?})",
            )
        });
        return Some(Register::new(kind, color));
    }
    // Void / Unknown — fall through to KINDS scan.
    let mut found: Option<Register> = None;
    for kind in [RegKind::Int, RegKind::Ref, RegKind::Float] {
        if let Some(ra) = regallocs.get(&kind) {
            if let Some(color) = ra.color_for_variable(var) {
                if let Some(prev) = found {
                    panic!(
                        "variable_to_register: Variable {var:?} colored in multiple \
                         regalloc classes ({:?} and {kind:?}) — RPython `getkind` must \
                         give exactly one",
                        prev.kind,
                    );
                }
                found = Some(Register::new(kind, color));
            }
        }
    }
    found
}

/// RPython liveness.py:82-116: remove_repeated_live.
///
/// Merges consecutive `-live-` markers into a single one (union of
/// all live registers). Labels between them are preserved.
fn remove_repeated_live(ops: &mut Vec<FlatOp>) {
    let mut result: Vec<FlatOp> = Vec::new();
    let mut i = 0;
    while i < ops.len() {
        if !matches!(&ops[i], FlatOp::Live { .. }) {
            result.push(ops[i].clone());
            i += 1;
            continue;
        }
        let mut labels = Vec::new();
        let mut merged_live: HashSet<Register> = HashSet::new();
        while i < ops.len() {
            match &ops[i] {
                FlatOp::Live { live_values } => {
                    merged_live.extend(live_values.iter().copied());
                    i += 1;
                }
                FlatOp::Label(_) => {
                    labels.push(ops[i].clone());
                    i += 1;
                }
                _ => break,
            }
        }
        result.extend(labels);
        // Stable order so the final bytecode encoding is reproducible.
        let mut merged: Vec<Register> = merged_live.into_iter().collect();
        merged.sort_by_key(|r| (r.kind as u8, r.index));
        result.push(FlatOp::Live {
            live_values: merged,
        });
    }
    *ops = result;
}

/// One backward pass of liveness analysis.
/// Returns true if any label's alive set grew (needs another iteration).
///
/// RPython: `_compute_liveness_must_continue(ssarepr, label2alive)`.
///
/// Walks backward through the instruction sequence. At each `-live-`
/// marker, expands it to include all values alive at that point.
/// Reads each `FlatOp::Op` operand's kind via
/// `FunctionGraph::concretetype_of(&var)` and color via
/// `RegAllocResult::color_for_variable(&var)`, matching upstream
/// `flatten.py:382 getcolor` line-for-line.
fn compute_liveness_pass(
    ops: &mut [FlatOp],
    label2alive: &mut HashMap<Label, HashSet<Register>>,
    regallocs: &HashMap<RegKind, RegAllocResult>,
) -> bool {
    let mut alive: HashSet<Register> = HashSet::new();
    let mut must_continue = false;

    // `def_value` and `use_value` route a `FlatOp::Op`-side
    // [`Variable`] through regalloc to its [`Register`] before
    // joining the alive set.  Void / unallocated values are silently
    // dropped (RPython's `flatten.py:325 if v.concretetype is not
    // lltype.Void` makes the same filter at flatten time).
    let def_value = |alive: &mut HashSet<Register>, var: &crate::flowspace::model::Variable| {
        if let Some(r) = variable_to_register(var, regallocs) {
            alive.remove(&r);
        }
    };
    let use_value = |alive: &mut HashSet<Register>, var: &crate::flowspace::model::Variable| {
        if let Some(r) = variable_to_register(var, regallocs) {
            alive.insert(r);
        }
    };

    for i in (0..ops.len()).rev() {
        match &ops[i] {
            FlatOp::Label(label) => {
                let label = *label;
                let alive_at_point = label2alive.entry(label).or_default();
                let prev_len = alive_at_point.len();
                alive_at_point.extend(alive.iter());
                if alive_at_point.len() != prev_len {
                    must_continue = true;
                }
            }
            FlatOp::Live { live_values } => {
                // RPython liveness.py:44-52: `-live-` markers are
                // expanded to the full set of [`Register`]s alive at
                // this point.  Pre-seeded values (e.g. forced by
                // jtransform) merge in here.
                for r in live_values {
                    alive.insert(*r);
                }
                ops[i] = FlatOp::Live {
                    live_values: alive.iter().copied().collect(),
                };
            }
            FlatOp::EndOfBlock => {
                alive.clear();
            }
            FlatOp::Unreachable => {
                // Same liveness semantics as `EndOfBlock`: the
                // instruction stream past this point cannot execute,
                // so registers that are only live in the dead tail
                // must NOT leak backward into earlier `-live-`
                // markers.  Without this clear, regalloc / resume
                // would over-pin those slots for no observable use.
                alive.clear();
            }
            FlatOp::Op(inner_op) => {
                if let Some(result) = inner_op.result.as_ref() {
                    def_value(&mut alive, result);
                }
                for var in crate::inline::op_variable_refs(&inner_op.kind) {
                    use_value(&mut alive, &var);
                }
            }
            FlatOp::Jump(label) => {
                let label = *label;
                if let Some(alive_at_target) = label2alive.get(&label) {
                    alive.extend(alive_at_target.iter());
                }
            }
            FlatOp::CatchException { target } => {
                let target = *target;
                if let Some(alive_at_target) = label2alive.get(&target) {
                    alive.extend(alive_at_target.iter());
                }
            }
            FlatOp::GotoIfExceptionMismatch { target, .. } => {
                let target = *target;
                if let Some(alive_at_target) = label2alive.get(&target) {
                    alive.extend(alive_at_target.iter());
                }
            }
            FlatOp::GotoIfNot { cond, target } => {
                let target = *target;
                alive.insert(*cond);
                if let Some(alive_at_target) = label2alive.get(&target) {
                    alive.extend(alive_at_target.iter());
                }
            }
            FlatOp::Switch { value, targets } => {
                alive.insert(*value);
                for (_, target) in targets {
                    if let Some(alive_at_target) = label2alive.get(target) {
                        alive.extend(alive_at_target.iter());
                    }
                }
            }
            FlatOp::IntBinOpJumpIfOvf {
                target,
                lhs,
                rhs,
                dst,
                ..
            } => {
                alive.remove(dst);
                alive.insert(*lhs);
                alive.insert(*rhs);
                if let Some(alive_at_target) = label2alive.get(target) {
                    alive.extend(alive_at_target.iter());
                }
            }
            FlatOp::Move { dst, src } => {
                // `flatten.py:333` — `int_copy %src -> %dst`.
                // Backward: dst is defined, register source is used,
                // constant source contributes nothing.
                alive.remove(dst);
                if let crate::flatten::RegOrConst::Reg(r) = src {
                    alive.insert(*r);
                }
            }
            FlatOp::Push(src) => {
                // `flatten.py:329` — `int_push %src` reads `src` into
                // the per-kind tmpreg.  Backward: a pure use of src.
                alive.insert(*src);
            }
            FlatOp::Pop(dst) => {
                // `flatten.py:331` — `int_pop -> %dst` writes tmpreg
                // into dst.  Backward: a pure def of dst.
                alive.remove(dst);
            }
            FlatOp::LastException { dst } | FlatOp::LastExcValue { dst } => {
                // Register operand carries (kind, color); the alive
                // set is Register-keyed so the def removes the
                // matching slot directly without any Variable bridge.
                alive.remove(dst);
            }
            FlatOp::Reraise => {}
            FlatOp::IntReturn(v) | FlatOp::RefReturn(v) | FlatOp::FloatReturn(v) => {
                // Backward: the return value is alive at this point;
                // after it (forward) nothing is.  RegOrConst::Reg
                // contributes its Register to the alive set;
                // Constants don't.
                alive.clear();
                if let crate::flatten::RegOrConst::Reg(r) = v {
                    alive.insert(*r);
                }
            }
            FlatOp::VoidReturn => {
                alive.clear();
            }
            FlatOp::Raise(v) => {
                alive.clear();
                if let crate::flatten::RegOrConst::Reg(r) = v {
                    alive.insert(*r);
                }
            }
        }
    }

    must_continue
}

// ____________________________________________________________
// helper functions for compactly encoding and decoding liveness info
//
// liveness is encoded as a 2 byte offset into the single string all_liveness
// (which is stored on the metainterp_sd)

/// RPython liveness.py:125 `OFFSET_SIZE`.
pub const OFFSET_SIZE: usize = 2;

/// RPython liveness.py:127-131 `encode_offset(pos, code)`.
pub fn encode_offset(pos: usize, code: &mut Vec<u8>) {
    assert_eq!(OFFSET_SIZE, 2);
    code.push((pos & 0xff) as u8);
    code.push(((pos >> 8) & 0xff) as u8);
    assert_eq!(pos >> 16, 0);
}

/// RPython liveness.py:133-136 `decode_offset(jitcode, pc)`.
pub fn decode_offset(jitcode: &[u8], pc: usize) -> usize {
    assert_eq!(OFFSET_SIZE, 2);
    (jitcode[pc] as usize) | ((jitcode[pc + 1] as usize) << 8)
}

// within the string of all_liveness, we encode the bitsets of which of the 256
// registers are live as follows: first three byte with the number of set bits
// for each of the categories ints, refs, floats followed by the necessary
// number of bytes to store them (this number of bytes is implicit), for each of
// the categories
// | len live_i | len live_r | len live_f
// | bytes for live_i | bytes for live_r | bytes for live_f

/// RPython liveness.py:147-166 `encode_liveness(live)`.
///
/// Encodes a single register-kind bitset: `live` is a list of register
/// indices (each `< 256`). Returns the packed bitset bytes (no length
/// header — the caller is responsible for emitting the three
/// `len_i/len_r/len_f` header bytes).
///
/// Mirrors RPython's `live = sorted(live)` (liveness.py:148): the input
/// is sorted internally so callers can pass arbitrary-order or
/// duplicated slices without normalization.
pub fn encode_liveness(live: &[u8]) -> Vec<u8> {
    // RPython liveness.py:148 `live = sorted(live)`.
    let mut sorted: Vec<u8> = live.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut liveness: Vec<u8> = Vec::new();
    let mut offset: u32 = 0;
    let mut char_: u32 = 0;
    let mut i = 0;
    while i < sorted.len() {
        let x = sorted[i] as u32;
        let x = x.wrapping_sub(offset);
        if x >= 8 {
            liveness.push(char_ as u8);
            char_ = 0;
            offset += 8;
            continue;
        }
        char_ |= 1 << x;
        assert!(char_ < 256);
        i += 1;
    }
    if char_ != 0 {
        liveness.push(char_ as u8);
    }
    liveness
}

/// RPython liveness.py:170-200 `LivenessIterator`.
///
/// Iterates set bit positions from a bitset stored in `all_liveness`
/// starting at `offset`, producing `length` indices total.
#[derive(Debug, Clone)]
pub struct LivenessIterator<'a> {
    pub all_liveness: &'a [u8],
    pub offset: usize,
    pub length: u32,
    pub curr_byte: u32,
    pub count: u32,
}

impl<'a> LivenessIterator<'a> {
    /// RPython liveness.py:172-178 `__init__(self, offset, length, all_liveness)`.
    pub fn new(offset: usize, length: u32, all_liveness: &'a [u8]) -> Self {
        assert!(length != 0);
        LivenessIterator {
            all_liveness,
            offset,
            length,
            curr_byte: 0,
            count: 0,
        }
    }
}

impl<'a> Iterator for LivenessIterator<'a> {
    type Item = u32;

    /// RPython liveness.py:184-200 `next(self)`.
    fn next(&mut self) -> Option<u32> {
        if self.length == 0 {
            return None;
        }
        self.length -= 1;
        let mut count = self.count;
        let all_liveness = self.all_liveness;
        let mut curr_byte = self.curr_byte;
        // find next bit set
        loop {
            if (count & 7) == 0 {
                curr_byte = all_liveness[self.offset] as u32;
                self.curr_byte = curr_byte;
                self.offset += 1;
            }
            if (curr_byte >> (count & 7)) & 1 != 0 {
                self.count = count + 1;
                return Some(count);
            }
            count += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flatten::FlatOp;
    use crate::model::{OpKind, SpaceOperation, ValueType};

    #[test]
    fn basic_liveness() {
        // v0 = Input
        // v1 = ConstInt(42)
        // v2 = BinOp(v0, v1)
        // Return v2
        let regallocs: HashMap<RegKind, RegAllocResult> = HashMap::new();
        let v0 = crate::flowspace::model::Variable::new();
        let v1 = crate::flowspace::model::Variable::new();
        let v2 = crate::flowspace::model::Variable::new();
        let mut flat = SSARepr {
            name: "test".into(),
            insns: vec![
                FlatOp::Label(Label(0)),
                FlatOp::Op(SpaceOperation {
                    result: Some(v0.clone()),
                    kind: OpKind::Input {
                        name: "a".into(),
                        ty: ValueType::Int,
                        class_root: None,
                    },
                }),
                FlatOp::Op(SpaceOperation {
                    result: Some(v1.clone()),
                    kind: OpKind::ConstInt(42),
                }),
                FlatOp::Op(SpaceOperation {
                    result: Some(v2),
                    kind: OpKind::BinOp {
                        op: "add".into(),
                        lhs: v0,
                        rhs: v1,
                        result_ty: ValueType::Int,
                    },
                }),
            ],
            num_blocks: 1,
            insns_pos: None,
        };

        // Should not panic.  Phase 3 added the `regallocs` parameter
        // for the FlatOp::Op `Variable → Register` bridge; pass an
        // empty map since this fixture has no inputargs that exercise
        // the conversion.
        compute_liveness(&mut flat, &regallocs);
    }

    #[test]
    fn encode_decode_offset_roundtrip() {
        let mut code: Vec<u8> = Vec::new();
        encode_offset(0x1234, &mut code);
        assert_eq!(code, vec![0x34, 0x12]);
        assert_eq!(decode_offset(&code, 0), 0x1234);
    }

    #[test]
    fn encode_liveness_empty() {
        assert_eq!(encode_liveness(&[]), Vec::<u8>::new());
    }

    #[test]
    fn encode_liveness_small() {
        // live = [0, 1, 7] -> first byte has bits 0, 1, 7 set = 0b1000_0011 = 0x83
        assert_eq!(encode_liveness(&[0u8, 1, 7]), vec![0x83]);
    }

    #[test]
    fn encode_liveness_multi_byte() {
        // live = [0, 8, 15] -> byte 0 = 0x01, byte 1 = 0b1000_0001 = 0x81
        assert_eq!(encode_liveness(&[0u8, 8, 15]), vec![0x01, 0x81]);
    }

    #[test]
    fn liveness_iterator_roundtrip() {
        let live = [0u8, 3, 5, 9, 12, 17];
        let encoded = encode_liveness(&live);
        let mut it = LivenessIterator::new(0, live.len() as u32, &encoded);
        let decoded: Vec<u32> = (&mut it).collect();
        assert_eq!(decoded, live.iter().map(|&i| i as u32).collect::<Vec<_>>());
    }
}
