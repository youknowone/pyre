//! Line-by-line port of `rpython/jit/codewriter/liveness.py`.
//!
//! The encoder/decoder helpers (`encode_offset`, `decode_offset`,
//! `encode_liveness`, `LivenessIterator`, `OFFSET_SIZE`) already live in
//! `majit_translate::liveness`; this module re-exports them and adds the
//! `compute_liveness` / `remove_repeated_live` passes operating on the
//! pyre-local `SSARepr` from `super::flatten`.
//!
//! RPython's `liveness.py` works on an untyped list of Python tuples. The
//! pyre port walks the typed `Insn` enum instead, but the ordering, case
//! analysis, and set-update semantics reproduce `liveness.py:19-116`
//! line-for-line.

use std::collections::{HashMap, HashSet};

use super::flatten::{DescrOperand, Insn, Operand, Register, SSARepr, TLabel};

pub use majit_translate::liveness::{
    LivenessIterator, OFFSET_SIZE, decode_offset, encode_liveness, encode_offset,
};

/// `liveness.py:19-23` `def compute_liveness(ssarepr)`.
///
/// Python:
/// ```py
/// def compute_liveness(ssarepr):
///     label2alive = {}
///     while _compute_liveness_must_continue(ssarepr, label2alive):
///         pass
///     remove_repeated_live(ssarepr)
/// ```
pub fn compute_liveness(ssarepr: &mut SSARepr) {
    let mut label2alive: HashMap<String, HashSet<Register>> = HashMap::new();
    while _compute_liveness_must_continue(ssarepr, &mut label2alive) {}
    remove_repeated_live(ssarepr);
}

/// Fixpoint dataflow only, skipping `remove_repeated_live` so external
/// `Insn::Live` indices stay stable. Used by the codewriter's Phase 4
/// migration where `live_patches` caches insn indices.
pub fn compute_liveness_preserve_positions(ssarepr: &mut SSARepr) {
    let mut label2alive: HashMap<String, HashSet<Register>> = HashMap::new();
    while _compute_liveness_must_continue(ssarepr, &mut label2alive) {}
}

/// `liveness.py:25-80` `_compute_liveness_must_continue(ssarepr, label2alive)`.
///
/// Walks the instruction list backwards, maintaining the set of live
/// registers at each point. When a `-live-` marker is reached, the set
/// is snapshot into the marker's argument list (the "expansion" described
/// in the module docstring).
///
/// Returns `True` iff any `label2alive[label]` grew during this pass —
/// the caller iterates to fixpoint.
fn _compute_liveness_must_continue(
    ssarepr: &mut SSARepr,
    label2alive: &mut HashMap<String, HashSet<Register>>,
) -> bool {
    // `liveness.py:26` `alive = set()`.
    let mut alive: HashSet<Register> = HashSet::new();
    // `liveness.py:27` `must_continue = False`.
    let mut must_continue = false;

    // `liveness.py:29-31` `def follow_label(lbl)`.
    //
    // Captured closures in Rust fight the borrow checker, so the helper
    // is inlined at its two call sites (`-live-` TLabels, and regular-
    // instruction TLabels / SwitchDictDescr). Inlining preserves the
    // `alive_at_point = label2alive.get(lbl.name, ()); alive.update(...)`
    // semantic exactly.
    #[inline]
    fn follow_label(
        alive: &mut HashSet<Register>,
        label2alive: &HashMap<String, HashSet<Register>>,
        lbl: &TLabel,
    ) {
        if let Some(alive_at_point) = label2alive.get(&lbl.name) {
            alive.extend(alive_at_point.iter().copied());
        }
    }

    // `liveness.py:33` `for i in range(len(ssarepr.insns)-1, -1, -1):`.
    for i in (0..ssarepr.insns.len()).rev() {
        // `liveness.py:34` `insn = ssarepr.insns[i]`.
        // Clone so the borrow on `ssarepr` is released before the later
        // `ssarepr.insns[i] = ...` write at `liveness.py:52`.
        let insn = ssarepr.insns[i].clone();

        // `liveness.py:36-42` `if isinstance(insn[0], Label)`.
        if let Insn::Label(label) = &insn {
            let alive_at_point = label2alive.entry(label.name.clone()).or_default();
            // `liveness.py:38` `prevlength = len(alive_at_point)`.
            let prevlength = alive_at_point.len();
            // `liveness.py:39` `alive_at_point.update(alive)`.
            alive_at_point.extend(alive.iter().copied());
            // `liveness.py:40-41`.
            if prevlength != alive_at_point.len() {
                must_continue = true;
            }
            // `liveness.py:42` `continue`.
            continue;
        }

        // `liveness.py:44-53` `if insn[0] == '-live-':`.
        if let Insn::Live(args) = &insn {
            // `liveness.py:45` `labels = []`.
            let mut labels: Vec<Operand> = Vec::new();
            // `liveness.py:46-51` — iterate args, add registers to alive,
            // follow TLabels, and remember the TLabels for the rewrite.
            for x in args {
                match x {
                    Operand::Register(reg) => {
                        // `liveness.py:47-48`.
                        alive.insert(*reg);
                    }
                    Operand::TLabel(lbl) => {
                        // `liveness.py:49-51`.
                        follow_label(&mut alive, label2alive, lbl);
                        labels.push(Operand::TLabel(lbl.clone()));
                    }
                    _ => {}
                }
            }
            // `liveness.py:52`:
            //   ssarepr.insns[i] = insn[:1] + tuple(alive) + tuple(labels)
            //
            // The first slot is the `-live-` tag itself (preserved by
            // writing another `Insn::Live(...)`), followed by the alive
            // set, followed by the labels.
            let mut new_args: Vec<Operand> = alive.iter().copied().map(Operand::Register).collect();
            new_args.extend(labels);
            ssarepr.insns[i] = Insn::Live(new_args);
            // `liveness.py:53` `continue`.
            continue;
        }

        // `liveness.py:55-57` `if insn[0] == '---':`.
        if let Insn::Unreachable = &insn {
            // `liveness.py:56` `alive = set()`.
            alive.clear();
            // `liveness.py:57` `continue`.
            continue;
        }

        // PRE-EXISTING-ADAPTATION: `Insn::PcAnchor(py_pc)` carries no
        // operands or liveness side effect (see `flatten.rs::Insn::PcAnchor`).
        // Skip without affecting `alive` so the SSARepr-position marker
        // is a no-op for the RPython `liveness.py:19-78` walk.
        if let Insn::PcAnchor(_) = &insn {
            continue;
        }

        // Regular instruction (`liveness.py:59-78`).
        //
        // `liveness.py:59-65` pre-strips the `-> result` suffix: the
        // trailing `'->', reg` pair means `reg` is defined here, so we
        // `alive.discard(reg)` and drop the last two slots before
        // consuming operands.
        let (args, result) = match &insn {
            Insn::Op { args, result, .. } => (args.as_slice(), result),
            // `Label`, `Live`, `Unreachable` were handled above.
            _ => unreachable!("non-Op insn after branches"),
        };
        if let Some(reg) = result {
            // `liveness.py:63-64`
            //   assert isinstance(reg, Register)
            //   alive.discard(reg)
            alive.remove(reg);
        }
        // `liveness.py:67-78` — scan remaining args, add register reads
        // to alive, recurse through ListOfKind, and follow TLabels /
        // SwitchDictDescr label tables.
        for x in args {
            match x {
                Operand::Register(reg) => {
                    // `liveness.py:68-69`.
                    alive.insert(*reg);
                }
                Operand::ListOfKind(lst) => {
                    // `liveness.py:70-73`.
                    for y in &lst.content {
                        if let Operand::Register(reg) = y {
                            alive.insert(*reg);
                        }
                    }
                }
                Operand::TLabel(lbl) => {
                    // `liveness.py:74-75`.
                    follow_label(&mut alive, label2alive, lbl);
                }
                Operand::Descr(rc) => match &**rc {
                    DescrOperand::SwitchDict(descr) => {
                        // `liveness.py:76-78`:
                        //   elif isinstance(x, SwitchDictDescr):
                        //       for key, label in x._labels:
                        //           follow_label(label)
                        for (_, label) in &descr.labels {
                            follow_label(&mut alive, label2alive, label);
                        }
                    }
                    DescrOperand::Bh(_) | DescrOperand::CallDescrStub(_) => {
                        // RPython `liveness.py:59-78` ignores non-`SwitchDictDescr`
                        // descrs — they contribute no control-flow edges.
                        // `CallDescrStub` is a pyre-only dispatch tag at the
                        // calldescr slot and likewise carries no edges.
                    }
                },
                _ => {}
            }
        }
    }

    must_continue
}

/// `live[1:]` element type, the Rust stand-in for the Python-side
/// heterogeneous tuple that `liveness.py:111-113` unions into a `set()`.
///
/// RPython's `liveset = set(); liveset.update(live[1:])` consumes any
/// element appearing in a `-live-` tuple after the opname slot — that
/// includes `Register` objects (always expanded by
/// `_compute_liveness_must_continue`) and `TLabel` objects (the
/// `labels` list preserved at `liveness.py:44-52`). Rust needs a single
/// hashable type to back the same `HashSet` semantics.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum LiveItem {
    Register(Register),
    TLabel(TLabel),
}

impl LiveItem {
    fn from_operand(op: &Operand) -> Option<Self> {
        match op {
            Operand::Register(r) => Some(Self::Register(*r)),
            Operand::TLabel(t) => Some(Self::TLabel(t.clone())),
            _ => None,
        }
    }

    fn into_operand(self) -> Operand {
        match self {
            Self::Register(r) => Operand::Register(r),
            Self::TLabel(t) => Operand::TLabel(t),
        }
    }
}

/// `liveness.py:82-116` `def remove_repeated_live(ssarepr)`.
///
/// Merges runs of consecutive `-live-` markers (optionally with intervening
/// `Label` markers) into a single `-live-` whose arguments are the union
/// of all collapsed markers.
pub fn remove_repeated_live(ssarepr: &mut SSARepr) {
    // `liveness.py:83-85` `last_i_pos = None; i = 0; res = []`.
    let mut res: Vec<Insn> = Vec::with_capacity(ssarepr.insns.len());
    let mut i = 0usize;

    while i < ssarepr.insns.len() {
        // `liveness.py:87` `insn = ssarepr.insns[i]`.
        let insn = ssarepr.insns[i].clone();
        // `liveness.py:88-91`.
        if !matches!(insn, Insn::Live(_)) {
            res.push(insn);
            i += 1;
            continue;
        }
        // `liveness.py:92-95` — collect `lives` and `labels` runs.
        let _last_i_pos = i;
        i += 1;
        let mut labels: Vec<Insn> = Vec::new();
        let mut lives: Vec<Insn> = vec![insn];

        // `liveness.py:97-106` inner loop.
        while i < ssarepr.insns.len() {
            let next = ssarepr.insns[i].clone();
            if matches!(next, Insn::Live(_)) {
                lives.push(next);
                i += 1;
            } else if matches!(next, Insn::Label(_)) {
                labels.push(next);
                i += 1;
            } else {
                break;
            }
        }

        // `liveness.py:107-110`.
        if lives.len() == 1 {
            res.extend(labels);
            res.push(lives.into_iter().next().unwrap());
            continue;
        }

        // `liveness.py:111-115` — union of all live sets.
        //
        // RPython `live[1:]` includes every tuple element after the
        // `-live-` tag: Register objects expanded by the backward pass
        // and any TLabel objects that survived from the original marker
        // (`liveness.py:44-52`). Both must be dedup'd by identity and
        // re-emitted, otherwise the merged marker loses control-flow
        // edges — this drops `L1` in `test_live_with_label` /
        // `test_live_duplicate`.
        let mut liveset: HashSet<LiveItem> = HashSet::new();
        for live in &lives {
            if let Insn::Live(args) = live {
                for a in args {
                    if let Some(item) = LiveItem::from_operand(a) {
                        liveset.insert(item);
                    }
                }
            }
        }
        res.extend(labels);
        // `liveness.py:115` `res.append(('-live-', ) + tuple(sorted(liveset)))`.
        //
        // Python's `sorted(set_of_mixed_objects)` raises `TypeError` in
        // Python 3 but RPython's `set()` elements are homogeneous per
        // call site (all `Register` or all `TLabel`-like), so a total
        // order is fine. Mirror the intent with a tiered comparator:
        // Registers first (by kind then index), TLabels after (by name).
        let mut sorted: Vec<LiveItem> = liveset.into_iter().collect();
        sorted.sort_by(|a, b| match (a, b) {
            (LiveItem::Register(ra), LiveItem::Register(rb)) => {
                (ra.kind as u32, ra.index).cmp(&(rb.kind as u32, rb.index))
            }
            (LiveItem::Register(_), LiveItem::TLabel(_)) => std::cmp::Ordering::Less,
            (LiveItem::TLabel(_), LiveItem::Register(_)) => std::cmp::Ordering::Greater,
            (LiveItem::TLabel(la), LiveItem::TLabel(lb)) => la.name.cmp(&lb.name),
        });
        res.push(Insn::Live(
            sorted.into_iter().map(LiveItem::into_operand).collect(),
        ));
    }

    // `liveness.py:116` `ssarepr.insns = res`.
    ssarepr.insns = res;
}

#[cfg(test)]
mod tests {
    use super::super::flatten::{Kind, Label as FLabel};
    use super::*;

    fn r_i(index: u16) -> Register {
        Register::new(Kind::Int, index)
    }

    #[test]
    fn empty_ssa_is_noop() {
        let mut s = SSARepr::new("t");
        compute_liveness(&mut s);
        assert!(s.insns.is_empty());
    }

    #[test]
    fn single_live_after_op_contains_operand() {
        // seq: (int_add, r1, r2, '->', r3)
        //      ('-live-',)
        //      ('int_return/i', r3)
        let mut s = SSARepr::new("t");
        s.insns.push(Insn::op_with_result(
            "int_add",
            vec![Operand::Register(r_i(1)), Operand::Register(r_i(2))],
            r_i(3),
        ));
        s.insns.push(Insn::Live(Vec::new()));
        s.insns
            .push(Insn::op("int_return", vec![Operand::Register(r_i(3))]));
        compute_liveness(&mut s);
        // The `-live-` marker should now hold `r3` (read by `int_return`
        // and live across the placeholder).
        let expanded = s
            .insns
            .iter()
            .find_map(|insn| match insn {
                Insn::Live(args) => Some(args.clone()),
                _ => None,
            })
            .expect("Live marker survives compute_liveness");
        let regs: Vec<Register> = expanded
            .iter()
            .filter_map(|o| {
                if let Operand::Register(r) = o {
                    Some(*r)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(regs, vec![r_i(3)]);
    }

    #[test]
    fn unreachable_resets_alive_set() {
        // seq: ('-live-',) ('---',) (int_return, r1)
        // After Unreachable, no registers are alive for the earlier Live.
        let mut s = SSARepr::new("t");
        s.insns.push(Insn::Live(Vec::new()));
        s.insns.push(Insn::Unreachable);
        s.insns
            .push(Insn::op("int_return", vec![Operand::Register(r_i(1))]));
        compute_liveness(&mut s);
        let expanded = match &s.insns[0] {
            Insn::Live(args) => args.clone(),
            _ => panic!("expected Live first"),
        };
        assert!(
            expanded.is_empty(),
            "Live before '---' should have empty alive set, got {:?}",
            expanded
        );
    }

    #[test]
    fn remove_repeated_live_merges_runs() {
        // Two consecutive -live- markers collapse into one.
        let mut s = SSARepr::new("t");
        s.insns.push(Insn::Live(vec![Operand::Register(r_i(1))]));
        s.insns.push(Insn::Live(vec![Operand::Register(r_i(2))]));
        remove_repeated_live(&mut s);
        assert_eq!(s.insns.len(), 1);
        let Insn::Live(args) = &s.insns[0] else {
            panic!("expected single Live")
        };
        let regs: Vec<Register> = args
            .iter()
            .filter_map(|o| {
                if let Operand::Register(r) = o {
                    Some(*r)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(regs, vec![r_i(1), r_i(2)]);
    }

    #[test]
    fn backward_label_flows_through_jump_argument() {
        // Simulate:
        //   (Label('L0'),)
        //   ('-live-',)
        //   ('int_return/i', r1)
        //   ('---',)
        //   (Label('L1'),)       <- unreached in layout, but label-based
        //   ('-live-',)
        //
        // We only check that `-live-` after the unreachable marker picks
        // up nothing (`alive` reset) and that the one before `int_return`
        // picks up r1.
        let mut s = SSARepr::new("t");
        s.insns.push(Insn::Label(FLabel::new("L0")));
        s.insns.push(Insn::Live(Vec::new()));
        s.insns
            .push(Insn::op("int_return", vec![Operand::Register(r_i(1))]));
        s.insns.push(Insn::Unreachable);
        s.insns.push(Insn::Label(FLabel::new("L1")));
        s.insns.push(Insn::Live(Vec::new()));
        compute_liveness(&mut s);
        let regs_at = |idx: usize| -> Vec<Register> {
            match &s.insns[idx] {
                Insn::Live(args) => args
                    .iter()
                    .filter_map(|o| {
                        if let Operand::Register(r) = o {
                            Some(*r)
                        } else {
                            None
                        }
                    })
                    .collect(),
                _ => panic!("expected Live"),
            }
        };
        assert_eq!(regs_at(1), vec![r_i(1)]);
        assert!(regs_at(5).is_empty());
    }

    // rpython/jit/codewriter/test/test_liveness.py:224-237 `test_live_with_label`.
    //
    // Source:
    //     -live- L1
    //     foo %i0
    //     ---
    //     L1:
    //     bar %i1
    // Expected:
    //     -live- %i0, %i1, L1
    //     foo %i0 ...
    #[test]
    fn live_with_tlabel_keeps_label() {
        let mut s = SSARepr::new("t");
        s.insns
            .push(Insn::Live(vec![Operand::TLabel(TLabel::new("L1"))]));
        s.insns
            .push(Insn::op("foo", vec![Operand::Register(r_i(0))]));
        s.insns.push(Insn::Unreachable);
        s.insns.push(Insn::Label(FLabel::new("L1")));
        s.insns
            .push(Insn::op("bar", vec![Operand::Register(r_i(1))]));
        compute_liveness(&mut s);

        let Insn::Live(args) = &s.insns[0] else {
            panic!("expected live")
        };
        assert!(
            args.iter()
                .any(|a| matches!(a, Operand::Register(r) if *r == r_i(0)))
        );
        assert!(
            args.iter()
                .any(|a| matches!(a, Operand::Register(r) if *r == r_i(1)))
        );
        assert!(
            args.iter()
                .any(|a| matches!(a, Operand::TLabel(t) if t.name == "L1"))
        );
    }

    // rpython/jit/codewriter/test/test_liveness.py:239-253 `test_live_duplicate`.
    //
    // Source:
    //     -live- L1
    //     -live- %i12
    //     foo %i0
    //     ---
    //     L1:
    //     bar %i1
    // Expected:
    //     -live- %i0, %i1, %i12, L1
    //     foo %i0 ...
    #[test]
    fn repeated_live_merges_registers_and_tlabels() {
        let mut s = SSARepr::new("t");
        s.insns
            .push(Insn::Live(vec![Operand::TLabel(TLabel::new("L1"))]));
        s.insns.push(Insn::Live(vec![Operand::Register(r_i(12))]));
        s.insns
            .push(Insn::op("foo", vec![Operand::Register(r_i(0))]));
        s.insns.push(Insn::Unreachable);
        s.insns.push(Insn::Label(FLabel::new("L1")));
        s.insns
            .push(Insn::op("bar", vec![Operand::Register(r_i(1))]));
        compute_liveness(&mut s);

        let live_idx = s
            .insns
            .iter()
            .position(|insn| matches!(insn, Insn::Live(_)))
            .unwrap();
        let Insn::Live(args) = &s.insns[live_idx] else {
            panic!("expected live")
        };
        assert!(
            args.iter()
                .any(|a| matches!(a, Operand::Register(r) if *r == r_i(0)))
        );
        assert!(
            args.iter()
                .any(|a| matches!(a, Operand::Register(r) if *r == r_i(1)))
        );
        assert!(
            args.iter()
                .any(|a| matches!(a, Operand::Register(r) if *r == r_i(12)))
        );
        assert!(
            args.iter()
                .any(|a| matches!(a, Operand::TLabel(t) if t.name == "L1"))
        );
    }

    // rpython/jit/codewriter/test/test_liveness.py:165-192 `test_switch`.
    //
    // Source:
    //     goto_maybe L1
    //     -live-
    //     fooswitch <SwitchDictDescr 4:L2, 5:L3>
    //     ---
    //     L3:  int_return %i7
    //     ---
    //     L1:  int_return %i4
    //     ---
    //     L2:  int_return %i3
    // Expected `-live-` picks up %i3 (L2) + %i7 (L3) because those are the
    // switch-table targets. %i4 is reached only via the default L1 edge,
    // which is NOT part of `SwitchDictDescr._labels` and so is not added
    // to the alive set at the switch.
    #[test]
    fn switch_descr_follows_target_labels_for_liveness() {
        let mut descr = super::super::flatten::SwitchDictDescr::new();
        descr.labels.push((4, TLabel::new("L2")));
        descr.labels.push((5, TLabel::new("L3")));

        let mut s = SSARepr::new("t");
        s.insns.push(Insn::op(
            "goto_maybe",
            vec![Operand::TLabel(TLabel::new("L1"))],
        ));
        s.insns.push(Insn::Live(Vec::new()));
        s.insns.push(Insn::op(
            "fooswitch",
            vec![Operand::descr(
                super::super::flatten::DescrOperand::SwitchDict(descr),
            )],
        ));
        s.insns.push(Insn::Unreachable);
        s.insns.push(Insn::Label(FLabel::new("L3")));
        s.insns
            .push(Insn::op("int_return", vec![Operand::Register(r_i(7))]));
        s.insns.push(Insn::Unreachable);
        s.insns.push(Insn::Label(FLabel::new("L1")));
        s.insns
            .push(Insn::op("int_return", vec![Operand::Register(r_i(4))]));
        s.insns.push(Insn::Unreachable);
        s.insns.push(Insn::Label(FLabel::new("L2")));
        s.insns
            .push(Insn::op("int_return", vec![Operand::Register(r_i(3))]));

        compute_liveness(&mut s);

        let live_idx = s
            .insns
            .iter()
            .position(|insn| matches!(insn, Insn::Live(_)))
            .unwrap();
        let Insn::Live(args) = &s.insns[live_idx] else {
            panic!("expected live")
        };
        assert!(
            args.iter()
                .any(|a| matches!(a, Operand::Register(r) if *r == r_i(3)))
        );
        assert!(
            args.iter()
                .any(|a| matches!(a, Operand::Register(r) if *r == r_i(7)))
        );
        assert!(
            !args
                .iter()
                .any(|a| matches!(a, Operand::Register(r) if *r == r_i(4)))
        );
    }
}
