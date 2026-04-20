//! Flatten pass: CFG → linear instruction sequence.
//!
//! RPython equivalent: `jit/codewriter/flatten.py` flatten_graph().
//!
//! Converts a multi-block FunctionGraph into a linear sequence of
//! FlatOps with Labels and Jumps. This is the last graph pass
//! before register allocation and JitCode assembly.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::model::{BlockId, FunctionGraph, SpaceOperation, Terminator, ValueId};
use crate::regalloc::RegAllocResult;

/// A label in the flattened instruction stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Label(pub usize);

/// A flattened instruction (post-CFG).
///
/// RPython equivalent: SSARepr instruction tuples from flatten.py.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlatOp {
    /// Label definition (target for jumps).
    Label(Label),
    /// Semantic op (from the graph).
    Op(SpaceOperation),
    /// Unconditional jump to label.
    /// RPython: `('goto', TLabel(target))`.
    Jump(Label),
    /// Conditional jump: if cond is false (zero), jump to label.
    /// RPython: `('goto_if_not', cond, TLabel(false_path))`.
    /// There is NO goto_if_true — RPython only uses goto_if_not.
    /// The true path is always the fallthrough.
    GotoIfNot { cond: ValueId, target: Label },
    /// Copy value (for Phi-node resolution: Link.args → target.inputargs).
    ///
    /// RPython `flatten.py:333` `self.emitline('%s_copy' % kind, v, "->", w)`.
    Move { dst: ValueId, src: ValueId },
    /// Save a value into the per-kind tmpreg, to break a cycle in a
    /// link renaming. Always paired with a later `Pop`.
    ///
    /// RPython `flatten.py:329` `self.emitline('%s_push' % kind, v)`.
    /// Blackhole handler: `blackhole.py:661-669` `bhimpl_{int,ref,float}_push`.
    Push(ValueId),
    /// Restore a value from the per-kind tmpreg into `dst`, completing
    /// a cycle break started by a prior `Push`.
    ///
    /// RPython `flatten.py:331` `self.emitline('%s_pop' % kind, "->", w)`.
    /// Blackhole handler: `blackhole.py:671-679` `bhimpl_{int,ref,float}_pop`.
    Pop(ValueId),
    /// Liveness marker — expanded by `compute_liveness()` to include
    /// all values alive at this point.
    ///
    /// RPython: `-live-` operation. Inserted by jtransform after calls
    /// that may need guard resumption (call_may_force, residual_call,
    /// inline_call, recursive_call). The liveness pass expands the
    /// `live_values` set to include all registers alive at this point.
    Live {
        /// Values known to be live (forced by jtransform).
        /// `compute_liveness()` expands this set.
        live_values: Vec<ValueId>,
    },
    /// Unreachable marker — marks the end of a code path.
    /// RPython: `---` operation. Resets the alive set in liveness analysis.
    Unreachable,
}

/// Register kind for a value (RPython regalloc).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegKind {
    Int,
    Ref,
    Float,
}

/// Result of the flatten pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSARepr {
    pub name: String,
    pub insns: Vec<FlatOp>,
    /// Total number of values used (for register allocation).
    pub num_values: usize,
    /// Number of basic blocks in the source graph.
    pub num_blocks: usize,
    /// Value kinds inferred from the type resolution pass.
    #[serde(default)]
    pub value_kinds: std::collections::HashMap<ValueId, RegKind>,
    /// flatten.py / assembler.py `ssarepr._insns_pos` — byte position
    /// of each instruction in the final bytecode, populated by the
    /// assembler.  `format.py:57-60` uses it to prefix every line with
    /// the position when set.  `None` when the SSARepr has not yet been
    /// assembled, matching upstream's `if ssarepr._insns_pos:` guard.
    #[serde(default)]
    pub insns_pos: Option<Vec<usize>>,
}

/// Flatten a FunctionGraph into a linear instruction sequence.
///
/// RPython equivalent: `flatten_graph(graph, regallocs)` from flatten.py.
///
/// `regallocs` is the per-kind register-allocation result produced by
/// the preceding `perform_all_register_allocations` pass. Upstream's
/// `insert_renamings` reads it via `getcolor(v)` to decide cycle-break
/// on the assigned color, not on the pre-regalloc ValueId identity.
///
/// Block ordering: entry first, then BFS order. Back-edges (loops)
/// become jumps to earlier labels.
pub fn flatten(graph: &FunctionGraph, regallocs: &HashMap<RegKind, RegAllocResult>) -> SSARepr {
    let mut ops = Vec::new();
    let mut block_labels: std::collections::HashMap<BlockId, Label> =
        std::collections::HashMap::new();
    let mut next_label = 0usize;

    // Assign labels to all blocks
    let order = block_order(graph);
    for &bid in &order {
        block_labels.insert(bid, Label(next_label));
        next_label += 1;
    }

    // Emit instructions in block order
    for &bid in &order {
        let block = graph.block(bid);
        let label = block_labels[&bid];

        // Label
        ops.push(FlatOp::Label(label));

        // Ops
        for op in &block.operations {
            if matches!(&op.kind, crate::model::OpKind::Live) {
                // RPython: -live- op becomes FlatOp::Live marker
                ops.push(FlatOp::Live {
                    live_values: Vec::new(),
                });
            } else {
                ops.push(FlatOp::Op(op.clone()));
            }
        }

        // Terminator → jumps + moves (Phi resolution)
        match &block.terminator {
            Terminator::Goto { target, args } => {
                // RPython `flatten.py:306` `insert_renamings(link)`.
                let target_block = graph.block(*target);
                insert_renamings(args, &target_block.inputargs, regallocs, &mut ops);
                ops.push(FlatOp::Jump(block_labels[target]));
            }
            Terminator::Branch {
                cond,
                if_true,
                true_args,
                if_false,
                false_args,
            } => {
                // RPython flatten.py:240-267: Two exits with boolean condition.
                //
                // Layout:
                //   -live-
                //   goto_if_not(cond, TLabel(false_path))
                //   [true path: phi moves + goto true_block]
                //   Label(false_path)
                //   [false path: phi moves + goto false_block]
                //
                // The true/false block bodies are emitted separately in
                // block order. Here we only emit the branch + phi moves.
                let true_label = block_labels[if_true];
                let false_label = block_labels[if_false];

                // RPython flatten.py:259: -live- before goto_if_not
                ops.push(FlatOp::Live {
                    live_values: Vec::new(),
                });
                // Allocate a fresh label for the false-path landing pad.
                // This avoids re-using the block's own label and creating
                // a self-jump. RPython uses TLabel(linkfalse) which is
                // distinct from Label(linkfalse.target).
                let false_landing = Label(next_label);
                next_label += 1;

                // RPython flatten.py:260: goto_if_not(cond, TLabel(false))
                ops.push(FlatOp::GotoIfNot {
                    cond: *cond,
                    target: false_landing,
                });

                // RPython flatten.py:264: true path (fallthrough)
                // insert_renamings(linktrue) + make_bytecode_block(linktrue.target)
                let true_block = graph.block(*if_true);
                insert_renamings(true_args, &true_block.inputargs, regallocs, &mut ops);
                ops.push(FlatOp::Jump(true_label));

                // RPython flatten.py:266-267: false path
                // Label(linkfalse) then insert_renamings + make_bytecode_block
                ops.push(FlatOp::Label(false_landing));
                let false_block = graph.block(*if_false);
                insert_renamings(false_args, &false_block.inputargs, regallocs, &mut ops);
                ops.push(FlatOp::Jump(false_label));
            }
            Terminator::Return(_val) => {
                // Return is implicit at the end (no jump needed)
                // Could emit a FlatOp::Return if needed
            }
            Terminator::Abort { .. } | Terminator::Unreachable => {
                // Terminal — no jump
            }
        }
    }

    // Count total values
    let mut max_value = 0usize;
    for op in &ops {
        match op {
            FlatOp::Op(op) => {
                if let Some(ValueId(v)) = op.result {
                    max_value = max_value.max(v + 1);
                }
            }
            FlatOp::Move {
                dst: ValueId(d),
                src: ValueId(s),
            } => {
                max_value = max_value.max(*d + 1);
                max_value = max_value.max(*s + 1);
            }
            FlatOp::Push(ValueId(v)) | FlatOp::Pop(ValueId(v)) => {
                max_value = max_value.max(*v + 1);
            }
            _ => {}
        }
    }

    SSARepr {
        name: graph.name.clone(),
        insns: ops,
        num_values: max_value,
        num_blocks: graph.blocks.len(),
        value_kinds: std::collections::HashMap::new(),
        insns_pos: None,
    }
}

/// Flatten with type information from rtype pass.
///
/// Like `flatten()` but populates `value_kinds` from the TypeResolutionState.
pub fn flatten_with_types(
    graph: &FunctionGraph,
    types: &crate::translate_legacy::rtyper::rtyper::TypeResolutionState,
    regallocs: &HashMap<RegKind, RegAllocResult>,
) -> SSARepr {
    let mut result = flatten(graph, regallocs);
    result.value_kinds = crate::translate_legacy::rtyper::rtyper::build_value_kinds(types);
    result
}

/// Compute block ordering (entry first, then BFS).
fn block_order(graph: &FunctionGraph) -> Vec<BlockId> {
    let mut order = Vec::new();
    let mut visited = std::collections::HashSet::new();
    let mut queue = std::collections::VecDeque::new();

    queue.push_back(graph.startblock);
    visited.insert(graph.startblock);

    while let Some(bid) = queue.pop_front() {
        order.push(bid);
        let block = graph.block(bid);
        for succ in successors(&block.terminator) {
            if visited.insert(succ) {
                queue.push_back(succ);
            }
        }
    }

    // Add any unreachable blocks (shouldn't happen in well-formed graphs)
    for block in &graph.blocks {
        if !visited.contains(&block.id) {
            order.push(block.id);
        }
    }

    order
}

/// Get successor block IDs from a terminator.
fn successors(term: &Terminator) -> Vec<BlockId> {
    match term {
        Terminator::Goto { target, .. } => vec![*target],
        Terminator::Branch {
            if_true, if_false, ..
        } => vec![*if_true, *if_false],
        Terminator::Return(_) | Terminator::Abort { .. } | Terminator::Unreachable => vec![],
    }
}

/// `flatten.py:306-334` `def insert_renamings(self, link)`.
///
/// Emits the ordered series of `%s_copy` / `%s_push` / `%s_pop` ops
/// that resolve a link's argument-to-inputarg renaming, breaking any
/// cycles via `reorder_renaming_list`. Upstream groups by register
/// kind so it can emit `int_copy` / `ref_copy` / `float_copy` under
/// different opnames; this helper emits generic `FlatOp::Move` /
/// `Push` / `Pop` and defers the per-kind opname selection to the
/// assembler (`assembler.rs::write_insn`), which looks up each value's
/// kind via the RegAllocResult coloring. Cycle-break correctness is
/// preserved because cycles live entirely within one kind's register
/// bank — a single global `reorder_renaming_list` call produces the
/// same (src, dst) sequence as three per-kind calls would.
///
/// `regallocs` supplies `getcolor(v)` per upstream — cycle detection
/// operates on colors, not ValueIds, so it remains correct whether
/// coalescing merged ValueIds into one color or split them across
/// separate colors.
///
/// Upstream:
/// ```py
/// def insert_renamings(self, link):
///     renamings = {}
///     lst = [(self.getcolor(v), self.getcolor(link.target.inputargs[i]))
///            for i, v in enumerate(link.args)
///            if v.concretetype is not lltype.Void and
///               v not in (link.last_exception, link.last_exc_value)]
///     lst.sort(key=lambda(v, w): w.index)
///     for v, w in lst:
///         if v == w:
///             continue
///         frm, to = renamings.setdefault(w.kind, ([], []))
///         frm.append(v)
///         to.append(w)
///     for kind in KINDS:
///         if kind in renamings:
///             frm, to = renamings[kind]
///             result = reorder_renaming_list(frm, to)
///             for v, w in result:
///                 if w is None:
///                     self.emitline('%s_push' % kind, v)
///                 elif v is None:
///                     self.emitline('%s_pop' % kind, "->", w)
///                 else:
///                     self.emitline('%s_copy' % kind, v, "->", w)
///     self.generate_last_exc(link, link.target.inputargs)
/// ```
///
/// `last_exception` / `last_exc_value` handling is not yet ported;
/// majit's jtransform doesn't surface the per-link exception extras.
/// When that lands, `generate_last_exc` will be a sibling helper.
pub fn insert_renamings(
    link_args: &[ValueId],
    inputargs: &[ValueId],
    regallocs: &HashMap<RegKind, RegAllocResult>,
    ops: &mut Vec<FlatOp>,
) {
    // Resolve each ValueId to its regalloc-assigned color, mirroring
    // upstream's `self.getcolor(v)` + `self.getcolor(inputargs[i])`.
    // Cycle-break must operate on colors, not ValueIds: two distinct
    // ValueIds can share a color after `coalesce_variables`, and two
    // kind-separated banks use independent color spaces.
    let get_color = |v: ValueId| -> Option<usize> {
        for ra in regallocs.values() {
            if let Some(&c) = ra.coloring.get(&v) {
                return Some(c);
            }
        }
        None
    };

    // Graph-construction divergence (not this helper's fault):
    // majit's synthesized graphs occasionally produce a `Terminator`
    // with `args.len() > target.inputargs.len()` — upstream rejects
    // that at `FunctionGraph` build time, so `insert_renamings` can
    // assume equality. The root fix lives in the graph builder;
    // `.zip()` truncation here keeps the prefix renaming parity
    // correct for the well-formed portion and matches the original
    // (pre-insert_renamings) flatten() behavior exactly.
    //
    // Parallel color lists for `reorder_renaming_list` + representative
    // ValueIds for emitting back into `FlatOp::Move/Push/Pop`. Each
    // kept pair satisfies `v_color != w_color` (upstream `if v == w:
    // continue` — after regalloc, equal colors mean equal variables).
    let cap = link_args.len().min(inputargs.len());
    let mut frm_colors: Vec<usize> = Vec::with_capacity(cap);
    let mut to_colors: Vec<usize> = Vec::with_capacity(cap);
    let mut src_vids: Vec<ValueId> = Vec::with_capacity(cap);
    let mut dst_vids: Vec<ValueId> = Vec::with_capacity(cap);
    for (v, w) in link_args.iter().zip(inputargs.iter()) {
        let Some(v_col) = get_color(*v) else { continue };
        let Some(w_col) = get_color(*w) else { continue };
        if v_col == w_col {
            continue;
        }
        frm_colors.push(v_col);
        to_colors.push(w_col);
        src_vids.push(*v);
        dst_vids.push(*w);
    }
    if frm_colors.is_empty() {
        return;
    }

    // `result = reorder_renaming_list(frm, to)` — cycle-safe ordering
    // over the color pairs.
    let result = reorder_renaming_list(&frm_colors, &to_colors);

    // Map a color back to a representative ValueId so downstream passes
    // that still use ValueId identity (liveness backward walk, assembler
    // `lookup_reg_with_kind`) keep working. Every Some(color) in the
    // output corresponds to at least one entry in the `frm_colors` or
    // `to_colors` list, respectively.
    let find_src_vid = |c: usize| -> ValueId {
        let idx = frm_colors
            .iter()
            .position(|&fc| fc == c)
            .expect("reorder_renaming_list src color must come from frm_colors");
        src_vids[idx]
    };
    let find_dst_vid = |c: usize| -> ValueId {
        let idx = to_colors
            .iter()
            .position(|&tc| tc == c)
            .expect("reorder_renaming_list dst color must come from to_colors");
        dst_vids[idx]
    };

    for (v, w) in result {
        match (v, w) {
            // `if w is None: self.emitline('%s_push' % kind, v)`.
            (Some(src_c), None) => ops.push(FlatOp::Push(find_src_vid(src_c))),
            // `elif v is None: self.emitline('%s_pop' % kind, "->", w)`.
            (None, Some(dst_c)) => ops.push(FlatOp::Pop(find_dst_vid(dst_c))),
            // `else: self.emitline('%s_copy' % kind, v, "->", w)`.
            (Some(src_c), Some(dst_c)) => ops.push(FlatOp::Move {
                src: find_src_vid(src_c),
                dst: find_dst_vid(dst_c),
            }),
            (None, None) => unreachable!("reorder_renaming_list never yields (None, None)"),
        }
    }
}

/// `flatten.py:395-414` `def reorder_renaming_list(frm, to):`.
///
/// Line-by-line port. Given two equal-length sequences `frm[i] -> to[i]`,
/// return an ordered list of `(src, dst)` pairs so that each move runs
/// after every read of its `dst` register has happened. Cycles are
/// broken by a `(src, None)` save and `(None, dst)` load pair:
///
/// ```py
/// def reorder_renaming_list(frm, to):
///     result = []
///     pending_indices = range(len(to))
///     while pending_indices:
///         not_read = dict.fromkeys([frm[i] for i in pending_indices])
///         still_pending_indices = []
///         for i in pending_indices:
///             if to[i] not in not_read:
///                 result.append((frm[i], to[i]))
///             else:
///                 still_pending_indices.append(i)
///         if len(pending_indices) == len(still_pending_indices):
///             # no progress -- there is a cycle
///             assert None not in not_read
///             result.append((frm[pending_indices[0]], None))
///             frm[pending_indices[0]] = None
///             continue
///         pending_indices = still_pending_indices
///     return result
/// ```
///
/// Each `(src, dst)` entry maps to one `%s_copy src -> dst` operation
/// emitted by `insert_renamings`; `(src, None)` maps to `%s_push src`
/// and `(None, dst)` maps to `%s_pop -> dst` (flatten.py:326-335).
///
/// `T: Eq + Copy + Hash` so the algorithm works for any register
/// representation — RPython uses `Register` objects keyed by identity,
/// we'll typically instantiate with `Register` or `u16` color indices.
pub fn reorder_renaming_list<T>(frm: &[T], to: &[T]) -> Vec<(Option<T>, Option<T>)>
where
    T: Eq + Copy + std::hash::Hash,
{
    // Mutable copy so the `frm[pending_indices[0]] = None` cycle-break
    // write has a home. In Rust we use `Option<T>` in the working
    // buffer; `None` is the "register already saved on the stack"
    // marker, matching RPython's `frm[...] = None`.
    let mut frm: Vec<Option<T>> = frm.iter().copied().map(Some).collect();
    let to: Vec<T> = to.to_vec();
    assert_eq!(frm.len(), to.len(), "frm and to must have equal length");

    let mut result: Vec<(Option<T>, Option<T>)> = Vec::new();
    // `pending_indices = range(len(to))`.
    let mut pending_indices: Vec<usize> = (0..to.len()).collect();

    // `while pending_indices:`.
    while !pending_indices.is_empty() {
        // `not_read = dict.fromkeys([frm[i] for i in pending_indices])`.
        // RPython builds a dict keyed on `frm[i]`; `None` entries mean
        // "already saved via push", which `to[i] not in not_read` checks
        // against.
        let not_read: std::collections::HashSet<Option<T>> =
            pending_indices.iter().map(|&i| frm[i]).collect();
        let mut still_pending_indices: Vec<usize> = Vec::new();
        // `for i in pending_indices:`.
        for &i in &pending_indices {
            // `if to[i] not in not_read`.
            if !not_read.contains(&Some(to[i])) {
                // `result.append((frm[i], to[i]))`.
                result.push((frm[i], Some(to[i])));
            } else {
                // `still_pending_indices.append(i)`.
                still_pending_indices.push(i);
            }
        }
        // `if len(pending_indices) == len(still_pending_indices):`.
        if pending_indices.len() == still_pending_indices.len() {
            // `assert None not in not_read`.
            debug_assert!(
                !not_read.contains(&None),
                "reorder_renaming_list: duplicate cycle break"
            );
            // `result.append((frm[pending_indices[0]], None))`.
            let head = pending_indices[0];
            result.push((frm[head], None));
            // `frm[pending_indices[0]] = None`.
            frm[head] = None;
            continue;
        }
        pending_indices = still_pending_indices;
    }

    // After the main loop finishes, every `(src, None)` push needs a
    // matching `(None, dst)` pop at the tail of its cycle. RPython's
    // loop emits the pop naturally when the cycle's final read slot
    // becomes safe — but because `frm[head] = None` is an in-place
    // rewrite, the next iteration sees `frm[...] = None` and emits
    // `(None, to[...])` directly as part of `(frm[i], to[i])` above.
    // No separate pop stage needed.
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{FunctionGraph, OpKind, Terminator};

    /// Test helper — build a `regallocs` map that assigns each
    /// `ValueId(n)` the color `n` in `RegKind::Int`. This turns the
    /// color-based `insert_renamings` cycle-break into pure ValueId
    /// identity, matching the pre-regalloc reasoning used by the
    /// `insert_renamings_*` unit tests below.
    fn identity_regallocs(
        max_id: usize,
    ) -> std::collections::HashMap<RegKind, crate::regalloc::RegAllocResult> {
        let coloring: std::collections::HashMap<ValueId, usize> =
            (0..=max_id).map(|n| (ValueId(n), n)).collect();
        let num_regs = max_id + 1;
        let mut m = std::collections::HashMap::new();
        m.insert(
            RegKind::Int,
            crate::regalloc::RegAllocResult { coloring, num_regs },
        );
        m
    }

    #[test]
    fn flatten_single_block() {
        let mut graph = FunctionGraph::new("simple");
        let entry = graph.startblock;
        let v = graph.push_op(entry, OpKind::ConstInt(42), true).unwrap();
        graph.set_terminator(entry, Terminator::Return(Some(v)));

        let flat = flatten(&graph, &identity_regallocs(8));
        assert_eq!(flat.name, "simple");
        // Label + ConstInt op = 2 flat ops
        assert!(flat.insns.len() >= 2);
        assert!(matches!(flat.insns[0], FlatOp::Label(Label(0))));
    }

    #[test]
    fn flatten_if_else_produces_jumps() {
        let mut graph = FunctionGraph::new("branch");
        let entry = graph.startblock;
        let cond = graph.push_op(entry, OpKind::ConstInt(1), true).unwrap();
        let then_block = graph.create_block();
        let else_block = graph.create_block();
        let merge = graph.create_block();

        graph.set_terminator(
            entry,
            Terminator::Branch {
                cond,
                if_true: then_block,
                true_args: vec![],
                if_false: else_block,
                false_args: vec![],
            },
        );
        graph.set_terminator(
            then_block,
            Terminator::Goto {
                target: merge,
                args: vec![],
            },
        );
        graph.set_terminator(
            else_block,
            Terminator::Goto {
                target: merge,
                args: vec![],
            },
        );
        graph.set_terminator(merge, Terminator::Return(None));

        let flat = flatten(&graph, &identity_regallocs(8));
        // Should have labels + jumps
        let has_jump = flat
            .insns
            .iter()
            .any(|op| matches!(op, FlatOp::Jump(_) | FlatOp::GotoIfNot { .. }));
        assert!(has_jump, "flattened if/else should have jumps");
        // Should have 4 labels (one per block)
        let label_count = flat
            .insns
            .iter()
            .filter(|op| matches!(op, FlatOp::Label(_)))
            .count();
        // 4 block labels + 1 false-path label from Branch (RPython goto_if_not convention)
        assert!(
            label_count >= 4,
            "should have at least 4 labels, got {label_count}"
        );
    }

    #[test]
    fn flatten_while_loop_has_back_edge() {
        let mut graph = FunctionGraph::new("loop");
        let entry = graph.startblock;
        let header = graph.create_block();
        let body = graph.create_block();
        let exit = graph.create_block();

        graph.set_terminator(
            entry,
            Terminator::Goto {
                target: header,
                args: vec![],
            },
        );
        let cond = graph.push_op(header, OpKind::ConstInt(1), true).unwrap();
        graph.set_terminator(
            header,
            Terminator::Branch {
                cond,
                if_true: body,
                true_args: vec![],
                if_false: exit,
                false_args: vec![],
            },
        );
        graph.set_terminator(
            body,
            Terminator::Goto {
                target: header,
                args: vec![],
            },
        );
        graph.set_terminator(exit, Terminator::Return(None));

        let flat = flatten(&graph, &identity_regallocs(8));
        // Body should jump back to header label
        let jumps: Vec<_> = flat
            .insns
            .iter()
            .filter(|op| matches!(op, FlatOp::Jump(_)))
            .collect();
        assert!(
            jumps.len() >= 2,
            "loop should have >=2 jumps (entry→header, body→header)"
        );
    }

    #[test]
    fn flatten_phi_produces_move_ops() {
        // When a Goto carries Link args to a target with inputargs,
        // flatten should emit Move ops for Phi resolution.
        let mut graph = FunctionGraph::new("phi");
        let entry = graph.startblock;
        let val = graph.push_op(entry, OpKind::ConstInt(42), true).unwrap();

        let (target, phi_args) = graph.create_block_with_args(1);
        let _phi = phi_args[0];

        graph.set_terminator(
            entry,
            Terminator::Goto {
                target,
                args: vec![val],
            },
        );
        graph.set_terminator(target, Terminator::Return(None));

        let flat = flatten(&graph, &identity_regallocs(8));
        let moves: Vec<_> = flat
            .insns
            .iter()
            .filter(|op| matches!(op, FlatOp::Move { .. }))
            .collect();
        assert_eq!(moves.len(), 1, "should have 1 Move for Phi resolution");
    }

    // `rpython/jit/codewriter/test/test_flatten.py:115-128` `test_reorder_renaming_list`.
    #[test]
    fn reorder_renaming_list_empty() {
        let result: Vec<(Option<i32>, Option<i32>)> = reorder_renaming_list::<i32>(&[], &[]);
        assert_eq!(result, Vec::<(Option<i32>, Option<i32>)>::new());
    }

    #[test]
    fn reorder_renaming_list_all_independent() {
        // No overlap between frm and to → identity order.
        let result = reorder_renaming_list(&[1, 2, 3], &[4, 5, 6]);
        assert_eq!(
            result,
            vec![(Some(1), Some(4)), (Some(2), Some(5)), (Some(3), Some(6)),]
        );
    }

    #[test]
    fn reorder_renaming_list_chain() {
        // 4→1, 5→2, 1→3, 2→4. Safe order: do (1→3) and (2→4) first
        // (their destinations aren't read later), then (4→1) and
        // (5→2). RPython expected: [(1,3), (4,1), (2,4), (5,2)].
        let result = reorder_renaming_list(&[4, 5, 1, 2], &[1, 2, 3, 4]);
        assert_eq!(
            result,
            vec![
                (Some(1), Some(3)),
                (Some(4), Some(1)),
                (Some(2), Some(4)),
                (Some(5), Some(2)),
            ]
        );
    }

    #[test]
    fn reorder_renaming_list_swap_cycle() {
        // 1↔2 is a cycle of length 2. Save 1 with push, do 2→1,
        // then pop→2. RPython expected: [(1,None), (2,1), (None,2)].
        let result = reorder_renaming_list(&[1, 2], &[2, 1]);
        assert_eq!(
            result,
            vec![(Some(1), None), (Some(2), Some(1)), (None, Some(2))]
        );
    }

    #[test]
    fn reorder_renaming_list_long_chain_and_two_cycles() {
        // Chain + two independent cycles: (7→8) safe;
        // (4→1, 3→2, 1→3, 2→4) is a 4-cycle; (6→5, 5→6) is a 2-cycle.
        let result = reorder_renaming_list(&[4, 3, 6, 1, 2, 5, 7], &[1, 2, 5, 3, 4, 6, 8]);
        assert_eq!(
            result,
            vec![
                (Some(7), Some(8)),
                (Some(4), None),
                (Some(2), Some(4)),
                (Some(3), Some(2)),
                (Some(1), Some(3)),
                (None, Some(1)),
                (Some(6), None),
                (Some(5), Some(6)),
                (None, Some(5)),
            ]
        );
    }

    // `rpython/jit/codewriter/test/test_flatten.py` exercises
    // `insert_renamings` indirectly via whole-graph tests; majit covers
    // the standalone helper below. Identity coloring (ValueId(n) →
    // color n) is used so the cycle-break reasoning under test matches
    // the ValueId-level intuition these cases document.
    #[test]
    fn insert_renamings_emits_nothing_for_identity() {
        // `for i, v in enumerate(link.args): if v == w: continue`.
        let regallocs = identity_regallocs(8);
        let mut ops: Vec<FlatOp> = Vec::new();
        let args = vec![ValueId(0), ValueId(1), ValueId(2)];
        let inputargs = args.clone();
        insert_renamings(&args, &inputargs, &regallocs, &mut ops);
        assert_eq!(ops, Vec::<FlatOp>::new());
    }

    #[test]
    fn insert_renamings_emits_move_for_acyclic_rename() {
        // Simple `%0 -> %1` phi resolution.
        let regallocs = identity_regallocs(8);
        let mut ops: Vec<FlatOp> = Vec::new();
        insert_renamings(&[ValueId(0)], &[ValueId(1)], &regallocs, &mut ops);
        assert_eq!(
            ops,
            vec![FlatOp::Move {
                dst: ValueId(1),
                src: ValueId(0),
            }]
        );
    }

    #[test]
    fn insert_renamings_breaks_swap_cycle_with_push_pop() {
        // Swap `%0 <-> %1` — reorder_renaming_list returns
        // [(0,None), (1,0), (None,1)] which maps to push/copy/pop.
        let regallocs = identity_regallocs(8);
        let mut ops: Vec<FlatOp> = Vec::new();
        insert_renamings(
            &[ValueId(0), ValueId(1)],
            &[ValueId(1), ValueId(0)],
            &regallocs,
            &mut ops,
        );
        assert_eq!(
            ops,
            vec![
                FlatOp::Push(ValueId(0)),
                FlatOp::Move {
                    dst: ValueId(0),
                    src: ValueId(1),
                },
                FlatOp::Pop(ValueId(1)),
            ]
        );
    }

    /// Two ValueIds that regalloc coalesced to the same color must NOT
    /// emit a Move — upstream `flatten.py:314` `if v == w: continue`
    /// tests color identity, not ValueId identity. This is the key
    /// difference between the pre- and post-regalloc insert_renamings.
    #[test]
    fn insert_renamings_skips_coalesced_same_color() {
        // ValueId(0) and ValueId(1) both colored 7 — the rename is a
        // no-op even though the ValueIds differ.
        let mut coloring: std::collections::HashMap<ValueId, usize> =
            std::collections::HashMap::new();
        coloring.insert(ValueId(0), 7);
        coloring.insert(ValueId(1), 7);
        let mut regallocs = std::collections::HashMap::new();
        regallocs.insert(
            RegKind::Int,
            crate::regalloc::RegAllocResult {
                coloring,
                num_regs: 8,
            },
        );
        let mut ops: Vec<FlatOp> = Vec::new();
        insert_renamings(&[ValueId(0)], &[ValueId(1)], &regallocs, &mut ops);
        assert_eq!(ops, Vec::<FlatOp>::new());
    }

    /// Four distinct ValueIds colored so that the color-level rename is
    /// a 2-cycle swap. Pre-regalloc cycle-break (ValueId identity) would
    /// not see any cycle — two unrelated moves — and emit only Moves,
    /// corrupting the color bank. Post-regalloc cycle-break (colors)
    /// emits Push/Move/Pop just like the pure 2-cycle test above.
    #[test]
    fn insert_renamings_detects_cycle_at_color_level() {
        // Coloring:
        //   v0 → c0, v2 → c1 (link.args)
        //   v1 → c1, v3 → c0 (target.inputargs)
        // Color pairs: (c0 → c1), (c1 → c0) — a swap cycle on colors.
        let mut coloring: std::collections::HashMap<ValueId, usize> =
            std::collections::HashMap::new();
        coloring.insert(ValueId(0), 0);
        coloring.insert(ValueId(1), 1);
        coloring.insert(ValueId(2), 1);
        coloring.insert(ValueId(3), 0);
        let mut regallocs = std::collections::HashMap::new();
        regallocs.insert(
            RegKind::Int,
            crate::regalloc::RegAllocResult {
                coloring,
                num_regs: 2,
            },
        );
        let mut ops: Vec<FlatOp> = Vec::new();
        insert_renamings(
            &[ValueId(0), ValueId(2)],
            &[ValueId(1), ValueId(3)],
            &regallocs,
            &mut ops,
        );
        // Must emit push/copy/pop — NOT two naive Moves.
        assert!(
            ops.iter().any(|o| matches!(o, FlatOp::Push(_))),
            "color-level 2-cycle must emit a Push, got {:?}",
            ops
        );
        assert!(
            ops.iter().any(|o| matches!(o, FlatOp::Pop(_))),
            "color-level 2-cycle must emit a Pop, got {:?}",
            ops
        );
    }
}
