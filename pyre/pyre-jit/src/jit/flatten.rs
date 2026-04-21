//! Line-by-line port of `rpython/jit/codewriter/flatten.py` lines 1-60.
//!
//! Covers the data structures the rest of the codewriter pipeline
//! (`liveness.py`, `assembler.py`) consumes: `SSARepr`, `Label`, `TLabel`,
//! `Register`, `ListOfKind`, `IndirectCallTargets`, and the `KINDS`
//! constant. The `GraphFlattener` driver at `flatten.py:60-350` has no
//! 1:1 analog in pyre because pyre's input is a CPython `CodeObject`
//! rather than an RPython `FunctionGraph`; the equivalent walker lives
//! in `codewriter.rs` and produces an `SSARepr` whose `insns` contents
//! follow the shapes declared here.
//!
//! The `Insn` enum and `Operand` enum do not exist as separate classes
//! in RPython — `ssarepr.insns` is a list of heterogeneous Python tuples.
//! Rust needs a typed representation, so the tuple shape is captured by
//! `Insn` (with cases for the well-known string markers `Label`, `-live-`,
//! `---` and generic `Op` instructions) plus `Operand` for everything
//! that appears inside a tuple.

use std::rc::Rc;

use majit_translate::jitcode::BhDescr;

/// `rpython/jit/codewriter/flatten.py:59` `KINDS = ['int', 'ref', 'float']`.
///
/// RPython stores register kinds as strings; the `Kind` enum is the Rust
/// analog. The `as_str` method yields the exact RPython string so callers
/// that stringify ("int"/"ref"/"float") continue to behave identically.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    Int,
    Ref,
    Float,
}

impl Kind {
    pub const ALL: [Kind; 3] = [Kind::Int, Kind::Ref, Kind::Float];

    /// Yields the RPython string for this kind ("int", "ref", "float").
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Int => "int",
            Kind::Ref => "ref",
            Kind::Float => "float",
        }
    }

    /// First character of the kind name ("i", "r", "f") — matches
    /// `self.kind[0]` in RPython `Register.__repr__` and the `argcodes`
    /// alphabet used by `assembler.py`.
    pub fn first_char(self) -> char {
        match self {
            Kind::Int => 'i',
            Kind::Ref => 'r',
            Kind::Float => 'f',
        }
    }
}

/// `flatten.py:6-10` `class SSARepr(object)`.
///
/// Python:
/// ```py
/// class SSARepr(object):
///     def __init__(self, name):
///         self.name = name
///         self.insns = []
///         self._insns_pos = None     # after being assembled
/// ```
#[derive(Debug, Clone)]
pub struct SSARepr {
    pub name: String,
    pub insns: Vec<Insn>,
    /// `flatten.py:10` `self._insns_pos = None # after being assembled`.
    /// `assembler.py:41` populates this with the byte position of each
    /// instruction after `assemble()`.
    pub insns_pos: Option<Vec<usize>>,
}

impl SSARepr {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            insns: Vec::new(),
            insns_pos: None,
        }
    }
}

/// `flatten.py:12-18` `class Label(object)`.
///
/// Python:
/// ```py
/// class Label(object):
///     def __init__(self, name):
///         self.name = name
///     def __repr__(self):
///         return "Label(%r)" % (self.name, )
///     def __eq__(self, other):
///         return isinstance(other, Label) and other.name == self.name
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Label {
    pub name: String,
}

impl Label {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

/// `flatten.py:20-26` `class TLabel(object)`.
///
/// Python:
/// ```py
/// class TLabel(object):
///     def __init__(self, name):
///         self.name = name
///     def __repr__(self):
///         return "TLabel(%r)" % (self.name, )
///     def __eq__(self, other):
///         return isinstance(other, TLabel) and other.name == self.name
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TLabel {
    pub name: String,
}

impl TLabel {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

/// `flatten.py:28-33` `class Register(object)`.
///
/// Python:
/// ```py
/// class Register(object):
///     def __init__(self, kind, index):
///         self.kind = kind          # 'int', 'ref' or 'float'
///         self.index = index
///     def __repr__(self):
///         return "%%%s%d" % (self.kind[0], self.index)
/// ```
///
/// RPython uses Python ints for `index`; pyre uses `u16` because pyre's
/// register allocator can exceed 255 in unusual cases (see
/// `liveness_regs_to_u8_sorted` in `codewriter.rs`). The RPython-orthodox
/// assertion `index < 256` is enforced at `assembler.emit_reg` time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Register {
    pub kind: Kind,
    pub index: u16,
}

impl Register {
    pub fn new(kind: Kind, index: u16) -> Self {
        Self { kind, index }
    }
}

/// `flatten.py:35-51` `class ListOfKind(object)`.
///
/// Python:
/// ```py
/// class ListOfKind(object):
///     # a list of Regs/Consts, all of the same 'kind'.
///     # We cannot use a plain list, because we wouldn't know what 'kind' of
///     # Regs/Consts would be expected in case the list is empty.
///     def __init__(self, kind, content):
///         assert kind in KINDS
///         self.kind = kind
///         self.content = tuple(content)
/// ```
#[derive(Debug, Clone)]
pub struct ListOfKind {
    pub kind: Kind,
    pub content: Vec<Operand>,
}

impl ListOfKind {
    pub fn new(kind: Kind, content: Vec<Operand>) -> Self {
        Self { kind, content }
    }

    /// `flatten.py:47` `def __nonzero__(self): return bool(self.content)`.
    pub fn is_empty(&self) -> bool {
        self.content.is_empty()
    }

    /// `flatten.py:45-46` `def __iter__(self): return iter(self.content)`.
    pub fn iter(&self) -> std::slice::Iter<'_, Operand> {
        self.content.iter()
    }
}

/// `flatten.py:53-57` `class IndirectCallTargets(object)`.
///
/// Python:
/// ```py
/// class IndirectCallTargets(object):
///     def __init__(self, lst):
///         self.lst = lst       # list of JitCodes
/// ```
///
/// The list carries `JitCode` references (RPython `list of JitCodes`, in
/// which each entry is a Python-object reference shared with other call
/// sites and with the assembler's `indirectcalltargets` set).  pyre still
/// stores the runtime adapter `majit_metainterp::jitcode::JitCode` here;
/// `Arc` preserves the shared-reference semantics of the Python object
/// reference, but the canonical codewriter `majit_translate::jitcode::JitCode`
/// has not reached this runtime path yet.
#[derive(Debug, Clone, Default)]
pub struct IndirectCallTargets {
    pub lst: Vec<std::sync::Arc<majit_metainterp::jitcode::JitCode>>,
}

/// `rpython/jit/codewriter/jitcode.py:131-143` `class SwitchDictDescr`
/// as populated by `flatten.py:282-298`.
///
/// Python:
/// ```py
/// from rpython.jit.codewriter.jitcode import SwitchDictDescr
/// switchdict = SwitchDictDescr()
/// switchdict._labels = []
/// ...
/// for switch in switches:
///     key = lltype.cast_primitive(lltype.Signed, switch.llexitcase)
///     switchdict._labels.append((key, TLabel(switch)))
/// ```
///
/// The SSARepr-side descr carries `_labels` because the liveness pass
/// (`liveness.py:76-78`) iterates this list to follow switch-target
/// edges. RPython's runtime `SwitchDictDescr.dict` is set later by
/// `Assembler.fix_labels` (`assembler.py:258-263`, via `attach`); pyre
/// lowers the SSARepr-side `SwitchDictDescr` into the runtime
/// `BhDescr::SwitchDict` at assemble time.
#[derive(Debug, Clone, Default)]
pub struct SwitchDictDescr {
    /// `flatten.py:284,298` `switchdict._labels.append((key, TLabel(...)))`.
    pub labels: Vec<(i64, TLabel)>,
}

impl SwitchDictDescr {
    pub fn new() -> Self {
        Self { labels: Vec::new() }
    }
}

/// Descr operand shape visible inside an `SSARepr`.
///
/// RPython's `assembler.py:197-206` handles both regular descrs (runtime
/// `AbstractDescr`) and the not-yet-attached `SwitchDictDescr` carried as
/// `isinstance(x, SwitchDictDescr)` (checked at `liveness.py:76`). The
/// `DescrOperand` enum preserves that distinction so liveness sees
/// `_labels` and the assembler sees a finalised runtime descr.
#[derive(Debug, Clone)]
pub enum DescrOperand {
    /// Runtime descr already materialised as `BhDescr`.
    Bh(BhDescr),
    /// SSARepr-side `SwitchDictDescr` before `attach()`; liveness reads
    /// its `labels` field to follow control-flow edges.
    SwitchDict(SwitchDictDescr),
    /// `rpython/jit/codewriter/jtransform.py:414-435 rewrite_call` appends
    /// an `AbstractDescr` (the `calldescr`) at the end of every
    /// `residual_call_*` / `inline_call_*` arg list. The descr carries
    /// `EffectInfo` that downstream (`rpython/jit/metainterp/optimizeopt/
    /// rewrite.py`) consults to pick between `call_may_force_*`,
    /// `call_release_gil_*`, `call_loopinvariant_*`, `call_pure_*`, and
    /// `call_assembler_*`, plus `arg_types` that `bh_call_*` reads to
    /// reconstruct the C-function parameter order.
    ///
    /// pyre does not (yet) thread `EffectInfo` through the codewriter
    /// layer, so this variant stands in for the calldescr and carries the
    /// flavor plus per-arg kind sequence directly. The assembler dispatch
    /// consumes both: flavor picks the same builder method the optimizeopt
    /// layer would have selected; `arg_kinds` lets `dispatch_op` re-
    /// interleave the kind-separated `ListOfKind` sublists into pyre's
    /// flat `&[JitCallArg]` call order (pyre helpers have varied param
    /// sequences — `ref,int`, `int,ref,ref`, `ref,ref,int` etc.). SSARepr
    /// shape still matches upstream 1:1: one descr operand per residual
    /// call, final argument position.
    CallDescrStub(CallDescrStub),
}

/// Pyre-local stand-in for `rpython/jit/codewriter/effectinfo.py
/// AbstractDescr` on a residual call. Upstream's calldescr carries both
/// the flavor (EffectInfo) and the ordered arg-types used by `bh_call_*`
/// to rebuild the C-function parameter list from `args_i` / `args_r` /
/// `args_f` pools (`rpython/jit/backend/llsupport/llmodel.py:816-839
/// bh_call_*` + `calldescr.call_stub_*`). pyre needs both pieces at
/// dispatch time.
#[derive(Debug, Clone)]
pub struct CallDescrStub {
    pub flavor: CallFlavor,
    /// Per-arg kind sequence in C-function parameter order. Exact length
    /// equals the sum of the int/ref/float `ListOfKind` sublists for the
    /// same residual_call Insn.
    pub arg_kinds: Vec<Kind>,
}

/// `rpython/jit/metainterp/optimizeopt/rewrite.py` `Rewrite.optimize_CALL_XXX`
/// branches on `op.getdescr().effectinfo.extraeffect` to select between
/// `call_may_force`, `call_release_gil`, `call_loopinvariant`, `call_pure`,
/// and `call_assembler`. In pyre the codewriter knows statically which
/// branch applies for each per-PC helper, so the enum is emitted directly
/// into the calldescr slot rather than derived from `EffectInfo`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CallFlavor {
    /// Plain residual call with no extra effect
    /// (`rpython/jit/codewriter/effectinfo.py EF_CANNOT_RAISE` /
    /// `EF_CAN_RAISE`).
    Plain,
    /// `EF_FORCES_VIRTUAL_OR_VIRTUALIZABLE`. The builder emits
    /// `call_may_force_*` so the metainterp forces virtualizable state
    /// before the call. Maps to `JitCodeBuilder::call_may_force_*_typed`.
    MayForce,
    /// `EF_LOOPINVARIANT`. One-shot call memoised across the trace loop.
    /// Maps to `JitCodeBuilder::call_loopinvariant_*_typed`.
    LoopInvariant,
    /// `EF_RELEASES_GIL`. Maps to `JitCodeBuilder::call_release_gil_*_typed`.
    ReleaseGil,
    /// `EF_ELIDABLE_*`. Maps to `JitCodeBuilder::call_pure_*_typed`.
    Pure,
    /// `jit.dont_look_inside` portal call — `bhimpl_call_assembler_*`.
    /// Maps to `JitCodeBuilder::call_assembler_*_typed`.
    Assembler,
}

/// `rpython/jit/codewriter/jtransform.py:423` `reskind =
/// getkind(op.result.concretetype)[0]`. The four result-kind suffixes
/// used by `residual_call_{kinds}_{reskind}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResKind {
    Int,
    Ref,
    Float,
    Void,
}

impl ResKind {
    /// The single-character suffix used in `residual_call_{kinds}_{reskind}`.
    /// `rpython/jit/codewriter/jtransform.py:434`
    /// `'%s_%s_%s' % (namebase, kinds, reskind)`.
    pub fn as_char(self) -> char {
        match self {
            ResKind::Int => 'i',
            ResKind::Ref => 'r',
            ResKind::Float => 'f',
            ResKind::Void => 'v',
        }
    }

    /// The non-void reskinds map to a `Kind` for the result `Register`.
    pub fn to_kind(self) -> Option<Kind> {
        match self {
            ResKind::Int => Some(Kind::Int),
            ResKind::Ref => Some(Kind::Ref),
            ResKind::Float => Some(Kind::Float),
            ResKind::Void => None,
        }
    }
}

// --------------------------------------------------------------------------
// Instruction shape.
//
// RPython's `ssarepr.insns` is a list of Python tuples with five possible
// shapes:
//   1. `(Label(name),)`                                              — block header
//   2. `('-live-', arg1, arg2, ...)`                                 — liveness marker
//   3. `('---',)`                                                    — unreachable
//   4. `(opname, arg1, arg2, ..., '->', result_register)`            — op with result
//   5. `(opname, arg1, arg2, ...)`                                   — op without result
//
// Rust has no untyped tuples, so `Insn` is an enum that captures these
// five shapes explicitly. The `Op` variant retains a separate `result`
// field to match RPython's `'->'` marker rather than requiring the caller
// to scan the args list.
// --------------------------------------------------------------------------

/// `flatten.py` / `assembler.py` tuple-argument types.
///
/// Exhaustive variant set for anything that can appear inside an Insn's
/// argument list. Constants follow `assembler.py:157-175` — int, ref
/// (GCREF pointer as `i64`), and float (bit-pattern stored as `i64`).
#[derive(Debug, Clone)]
pub enum Operand {
    Register(Register),
    /// `Constant(Signed)` in RPython.
    ConstInt(i64),
    /// `Constant(GCREF)` in RPython — a boxed reference stored as a
    /// pointer-sized int. `PY_NULL` is represented as `0`.
    ConstRef(i64),
    /// `Constant(Float)` in RPython — stored as bit-pattern for
    /// parity with `assembler.emit_const`'s float path.
    ConstFloat(i64),
    /// Forward reference to a `Label`.
    TLabel(TLabel),
    /// A same-kind list of Registers/Constants.
    ListOfKind(ListOfKind),
    /// `assembler.py:197-206` `elif isinstance(x, AbstractDescr)` /
    /// `liveness.py:76` `elif isinstance(x, SwitchDictDescr)` — either a
    /// runtime-resolved descr (`BhDescr`) or an SSARepr-side
    /// `SwitchDictDescr` with a live `_labels` table.
    ///
    /// Wrapped in `Rc` so cloning the `Operand` preserves Python object
    /// identity: `assembler.py:197-199` keys `self._descr_dict` on
    /// `id(x)`, and two SSARepr sites that share the SAME descr object
    /// MUST dedup to the same `descrs` index. With `Rc`, callers that
    /// build an SSARepr `clone()` the `Rc` (pointer-preserving) while
    /// still being able to construct distinct descrs with
    /// `Rc::new(...)` when identity should differ. A plain
    /// `DescrOperand` value field would lose this distinction on every
    /// `Clone`.
    Descr(Rc<DescrOperand>),
    /// `IndirectCallTargets` — list of jitcodes for `indirect_call`.
    IndirectCallTargets(IndirectCallTargets),
    /// majit IR `OpCode` passed verbatim to `record_binop_*` /
    /// `record_unary_*` dispatch arms. RPython's equivalent is a
    /// `Constant(Address)` holding the low-level helper pointer; pyre's
    /// in-process path carries the typed `majit_ir::OpCode` so the
    /// assembler can forward it to `JitCodeBuilder::record_*` without a
    /// lossy int-tag round-trip.
    OpCode(majit_ir::OpCode),
}

impl Operand {
    pub fn reg(kind: Kind, index: u16) -> Self {
        Operand::Register(Register::new(kind, index))
    }

    /// Wrap a `DescrOperand` into a fresh `Rc` and build an `Operand`.
    /// Callers that want two `Operand::Descr`s to dedup to the same
    /// `descrs` index must `Rc::clone` the `Rc` returned by this call
    /// rather than invoking `descr()` twice with equal values.
    pub fn descr(value: DescrOperand) -> Self {
        Operand::Descr(Rc::new(value))
    }

    /// Build an `Operand::Descr` from an existing `Rc<DescrOperand>`.
    /// Preserves identity for dedup (`assembler.py:197-199`).
    pub fn descr_rc(value: Rc<DescrOperand>) -> Self {
        Operand::Descr(value)
    }
}

/// RPython `-live-` marker opname (`liveness.py:5-12`). Stored as the
/// first tuple element in RPython's ssarepr; pyre stores it as the
/// `opname` field of `Insn::Op`, matching the tuple-shape exactly.
pub const OPNAME_LIVE: &str = "-live-";

/// Instruction tuple (`ssarepr.insns[i]`).
///
/// The four RPython tuple shapes enumerated above (`Label`, `-live-`,
/// `---`, regular op), plus one pyre-specific `PcAnchor` variant — see
/// its docstring for rationale. `-live-` shares the `Op` variant with
/// regular operations, matching RPython's tuple representation where
/// `insn[0] == '-live-'` is the discriminator.
#[derive(Debug, Clone)]
pub enum Insn {
    /// `(Label(name),)` — block-entry marker.
    Label(Label),
    /// `('---',)` — unreachable marker; clears the liveness pass's alive
    /// set (`liveness.py:70`).
    Unreachable,
    /// `(opname, args..., ['->' result])` — regular operation, including
    /// `-live-` liveness markers (`opname == OPNAME_LIVE`). `result` is
    /// `Some(register)` iff the RPython tuple contains a trailing
    /// `'->' result` pair; the assembler emits the `>` argcode in that
    /// case (`assembler.py:210-219`). `-live-` always has
    /// `result == None`.
    Op {
        opname: String,
        args: Vec<Operand>,
        result: Option<Register>,
    },
    /// PRE-EXISTING-ADAPTATION: pyre-only marker recording the SSARepr
    /// position where a Python bytecode (py_pc) starts. RPython has no
    /// equivalent because its jitcode is graph-derived (not Python-
    /// bytecode-1:1) and Python PCs do not appear in jitcode space.
    ///
    /// pyre's dispatch loop emits one `PcAnchor` at every Python PC so
    /// the trace-time dispatch can map `next_instr` to the JitCode byte
    /// offset post-assemble. The `compute_liveness` and `regalloc`
    /// passes ignore anchors entirely (no liveness, no interference, no
    /// rename); `assembler.assemble` records each anchor's byte offset
    /// without emitting any bytecode. This replaces the older
    /// dispatch-time `pc_map[py_pc] = current_pos()` snapshot, which
    /// became stale whenever `compute_liveness::remove_repeated_live`
    /// merged consecutive `-live-` markers and shifted insn indices.
    ///
    /// Closest RPython analog: `Label(block)` markers used to anchor
    /// merge-point block entries (`flatten.py`); pyre's anchor is the
    /// same idea applied per Python bytecode rather than per graph block.
    PcAnchor(usize),
}

impl Insn {
    /// `Insn::Op` with no result register.
    pub fn op(opname: impl Into<String>, args: Vec<Operand>) -> Self {
        Insn::Op {
            opname: opname.into(),
            args,
            result: None,
        }
    }

    /// `Insn::Op` with a trailing `'->' result` pair.
    pub fn op_with_result(opname: impl Into<String>, args: Vec<Operand>, result: Register) -> Self {
        Insn::Op {
            opname: opname.into(),
            args,
            result: Some(result),
        }
    }

    /// `('-live-', args...)` marker, RPython `liveness.py` parity.
    pub fn live(args: Vec<Operand>) -> Self {
        Insn::Op {
            opname: OPNAME_LIVE.to_string(),
            args,
            result: None,
        }
    }

    /// `true` iff this instruction is a `-live-` marker.
    pub fn is_live(&self) -> bool {
        matches!(self, Insn::Op { opname, .. } if opname == OPNAME_LIVE)
    }

    /// `Some(&args)` if this instruction is a `-live-` marker, else `None`.
    pub fn live_args(&self) -> Option<&[Operand]> {
        match self {
            Insn::Op { opname, args, .. } if opname == OPNAME_LIVE => Some(args),
            _ => None,
        }
    }

    /// `Some(&mut args)` if this instruction is a `-live-` marker, else `None`.
    pub fn live_args_mut(&mut self) -> Option<&mut Vec<Operand>> {
        match self {
            Insn::Op { opname, args, .. } if opname == OPNAME_LIVE => Some(args),
            _ => None,
        }
    }
}

// --------------------------------------------------------------------------
// `flatten.py:306-413` renaming helpers.
//
// These are the graph-level link renaming helpers that Step 6.3 (Task
// #214) will call at each terminator edge.  They are added here as
// pure functions — unwired to the walker yet — so that the walker
// integration lands as a separate, reviewable step.
//
// Upstream `GraphFlattener.insert_renamings` + `generate_last_exc` +
// `reorder_renaming_list` (`flatten.py:306-413`) expect a
// `regallocs[kind]` mapping from `Variable → colour index`, driven by
// `self.getcolor(v)` (`flatten.py:382-391`) which returns a `Constant`
// for Constants and a `Register(kind, colour)` for Variables.  pyre's
// production regalloc keys on SSARepr `ValueId`, not graph
// `Variable` — so the public API here takes an explicit
// `getcolor_variable: FnMut(&Variable) -> Register` closure.  A
// future session migrating regalloc onto the graph will supply that
// closure via the `AllocationResult`; for now callers provide a
// synthetic mapping for testing.
// --------------------------------------------------------------------------

use std::collections::HashMap as StdHashMap;
use std::collections::HashSet as StdHashSet;
use std::hash::Hash as StdHash;

/// `flatten.py:395-413` `reorder_renaming_list(frm, to)`.
///
/// Reorders a parallel `(frm[i], to[i])` renaming list so that each
/// copy `frm[i] -> to[i]` fires only after every rename whose source
/// matches `to[i]` has already been emitted.  Cycles are broken by
/// turning the first cycle edge into a `(Some(frm), None)` push +
/// later `(None, Some(to))` pop pair, letting the assembler route
/// through the kind-typed scratch slot (`{kind}_push` / `{kind}_pop`).
///
/// Generic over the key type `T` so callers can pass either colour
/// indices (upstream Python) or typed `Register` / enum keys (pyre).
pub fn reorder_renaming_list<T>(frm: &[T], to: &[T]) -> Vec<(Option<T>, Option<T>)>
where
    T: Clone + Eq + StdHash,
{
    assert_eq!(
        frm.len(),
        to.len(),
        "reorder_renaming_list: length mismatch"
    );
    let mut frm: Vec<Option<T>> = frm.iter().cloned().map(Some).collect();
    let to: Vec<Option<T>> = to.iter().cloned().map(Some).collect();
    let mut result: Vec<(Option<T>, Option<T>)> = Vec::new();
    let mut pending: Vec<usize> = (0..to.len()).collect();
    while !pending.is_empty() {
        // `not_read = dict.fromkeys([frm[i] for i in pending_indices])`.
        let not_read: StdHashSet<&T> = pending.iter().filter_map(|&i| frm[i].as_ref()).collect();
        let mut still_pending: Vec<usize> = Vec::new();
        for &i in &pending {
            let to_val = to[i].as_ref();
            let safe = match to_val {
                Some(w) => !not_read.contains(w),
                None => true,
            };
            if safe {
                result.push((frm[i].clone(), to[i].clone()));
            } else {
                still_pending.push(i);
            }
        }
        if still_pending.len() == pending.len() {
            // `flatten.py:406-411` — no progress, cycle present.  Break
            // the cycle at the first pending entry by converting its
            // source into a `None` (emit a push), then retry.
            let cycle_head = still_pending[0];
            debug_assert!(
                frm[cycle_head].is_some(),
                "reorder_renaming_list: cycle detected but head is already None"
            );
            result.push((frm[cycle_head].clone(), None));
            frm[cycle_head] = None;
            continue;
        }
        pending = still_pending;
    }
    result
}

/// `flatten.py:336-347` `GraphFlattener.generate_last_exc`.
///
/// After `insert_renamings` has emitted its copy/push/pop sequence,
/// `generate_last_exc` emits the `last_exception` / `last_exc_value`
/// loads that populate the `link.target.inputargs` slots mapped to
/// `link.last_exception` / `link.last_exc_value`.
///
/// Mirrors upstream's two-pass structure (types first, values second),
/// matching the order in which an exception handler reads them back.
pub fn generate_last_exc<F>(
    link: &super::flow::Link,
    target_inputargs: &[super::flow::FlowValue],
    mut getcolor_variable: F,
    ssarepr: &mut SSARepr,
) where
    F: FnMut(&super::flow::Variable) -> Register,
{
    use super::flow::FlowValue;
    if link.last_exception.is_none() && link.last_exc_value.is_none() {
        return;
    }
    let pairs: Vec<(&super::flow::FlowValue, &super::flow::FlowValue)> = link
        .args
        .iter()
        .zip(target_inputargs.iter())
        .filter_map(|(a, w)| {
            let a = a.as_ref()?;
            Some((a, w))
        })
        .collect();

    // `flatten.py:342-344` `for v, w in zip(link.args, inputargs): if
    // v is link.last_exception: emitline("last_exception", "->", getcolor(w))`.
    if let Some(last_exc_type) = link.last_exception {
        for (v, w) in &pairs {
            if let FlowValue::Variable(var) = v {
                if *var == last_exc_type {
                    if let FlowValue::Variable(dst_var) = w {
                        let dst_reg = getcolor_variable(dst_var);
                        ssarepr.insns.push(Insn::op_with_result(
                            "last_exception",
                            Vec::new(),
                            dst_reg,
                        ));
                    }
                }
            }
        }
    }
    // `flatten.py:345-347` symmetric pass for `last_exc_value`.
    if let Some(last_exc_val) = link.last_exc_value {
        for (v, w) in &pairs {
            if let FlowValue::Variable(var) = v {
                if *var == last_exc_val {
                    if let FlowValue::Variable(dst_var) = w {
                        let dst_reg = getcolor_variable(dst_var);
                        ssarepr.insns.push(Insn::op_with_result(
                            "last_exc_value",
                            Vec::new(),
                            dst_reg,
                        ));
                    }
                }
            }
        }
    }
}

/// `flatten.py:306-334` `GraphFlattener.insert_renamings`.
///
/// Emits the `{kind}_copy` / `{kind}_push` / `{kind}_pop` Insns that
/// implement `link.args -> link.target.inputargs` at a CFG edge.
/// Sorts the renamings by destination register index, groups by kind,
/// and feeds each group through `reorder_renaming_list` for cycle
/// detection.
///
/// Step 6.3 (Task #214) will wire this at each terminator emitted by
/// `transform_graph_to_jitcode`; until that lands, pyre relies on the
/// post-hoc SSARepr-level `coalesce_variables` scanner
/// (`pyre-jit/src/jit/regalloc.rs`) as a documented
/// PRE-EXISTING-ADAPTATION.
pub fn insert_renamings<F>(
    link: &super::flow::Link,
    target_inputargs: &[super::flow::FlowValue],
    mut getcolor_variable: F,
    ssarepr: &mut SSARepr,
) where
    F: FnMut(&super::flow::Variable) -> Register,
{
    use super::flow::{Constant as FlowConstant, FlowValue};

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    enum RenameKey {
        /// Source or destination that carries a kind-typed colour.
        Reg(Register),
        /// Source is a graph-level `Constant`; kept as its own key so
        /// cycle detection never matches it against a `Reg` and so the
        /// emission loop recovers the original `Constant` for
        /// `constant_to_copy_source`.  Destinations are always `Reg`.
        Const(FlowConstant),
    }

    // `flatten.py:308-311` — build the `(src_colour, dst_colour)` list,
    // filtering out void-kind entries (pyre has no void, nothing to
    // skip there) and the `last_exception` / `last_exc_value` extras
    // that `generate_last_exc` handles separately.
    assert_eq!(
        link.args.len(),
        target_inputargs.len(),
        "insert_renamings: link.args and target.inputargs must have equal length"
    );

    let mut lst: Vec<(RenameKey, Register)> = Vec::new();
    for (arg, target_input) in link.args.iter().zip(target_inputargs.iter()) {
        let src = match arg.as_ref() {
            Some(v) => v,
            None => continue, // dead merge-path slot
        };
        // Skip extravars handled by `generate_last_exc`.
        if let FlowValue::Variable(v) = src {
            if Some(*v) == link.last_exception || Some(*v) == link.last_exc_value {
                continue;
            }
        }
        let dst_var = match target_input {
            FlowValue::Variable(v) => v,
            // Target inputargs are always Variables in a well-formed
            // graph.  A Constant in an inputarg slot would be a
            // front-end bug (see `model.py:185-188`).  Skip defensively.
            FlowValue::Constant(_) => continue,
        };
        let dst_reg = getcolor_variable(dst_var);

        let src_key = match src {
            FlowValue::Variable(v) => {
                let src_reg = getcolor_variable(v);
                // `flatten.py:314` `if v == w: continue`.  After regalloc,
                // equal registers mean the renaming is the identity.
                if src_reg == dst_reg {
                    continue;
                }
                RenameKey::Reg(src_reg)
            }
            FlowValue::Constant(c) => RenameKey::Const(c.clone()),
        };
        lst.push((src_key, dst_reg));
    }

    // `flatten.py:312` `lst.sort(key=lambda (v, w): w.index)`.
    lst.sort_by_key(|(_, to)| to.index);

    // `flatten.py:313-318` group by destination kind.
    let mut by_kind: StdHashMap<Kind, (Vec<RenameKey>, Vec<RenameKey>)> = StdHashMap::new();
    for (src_key, dst_reg) in lst {
        let entry = by_kind
            .entry(dst_reg.kind)
            .or_insert_with(|| (Vec::new(), Vec::new()));
        entry.0.push(src_key);
        entry.1.push(RenameKey::Reg(dst_reg));
    }

    // `flatten.py:319-333` emit per-kind `{kind}_copy` / `{kind}_push`
    // / `{kind}_pop` via `reorder_renaming_list`.
    for kind in Kind::ALL {
        let Some((frm_keys, to_keys)) = by_kind.remove(&kind) else {
            continue;
        };
        let ordered = reorder_renaming_list(&frm_keys, &to_keys);
        for (from, to) in ordered {
            match (from, to) {
                // `flatten.py:328-329` `if w is None: emitline("%s_push", v)`.
                (Some(key), None) => {
                    let src_reg = match key {
                        RenameKey::Reg(r) => r,
                        RenameKey::Const(_) => unreachable!(
                            "reorder_renaming_list: Const source cannot participate in a cycle push"
                        ),
                    };
                    ssarepr.insns.push(Insn::op(
                        format!("{}_push", kind.as_str()),
                        vec![Operand::Register(src_reg)],
                    ));
                }
                // `flatten.py:330-331` `elif v is None: emitline("%s_pop", "->", w)`.
                (None, Some(RenameKey::Reg(dst_reg))) => {
                    ssarepr.insns.push(Insn::op_with_result(
                        format!("{}_pop", kind.as_str()),
                        Vec::new(),
                        dst_reg,
                    ));
                }
                // `flatten.py:332-333` `else: emitline("%s_copy", v, "->", w)`.
                (Some(src_key), Some(RenameKey::Reg(dst_reg))) => {
                    let src_operand = match src_key {
                        RenameKey::Reg(r) => Operand::Register(r),
                        RenameKey::Const(c) => constant_to_copy_source(&c),
                    };
                    ssarepr.insns.push(Insn::op_with_result(
                        format!("{}_copy", kind.as_str()),
                        vec![src_operand],
                        dst_reg,
                    ));
                }
                (None, None) | (_, Some(RenameKey::Const(_))) => {
                    unreachable!("reorder_renaming_list produced malformed entry");
                }
            }
        }
    }

    // `flatten.py:334` — tail-emit the `last_exception` / `last_exc_value`
    // loads after the copy sequence.
    generate_last_exc(link, target_inputargs, &mut getcolor_variable, ssarepr);
}

/// Map a graph-level `Constant` into the `Operand` shape expected by a
/// `{kind}_copy` source slot.  Mirrors `assembler.py:162-174` which
/// encodes constants as `i`/`c` argcode variants depending on kind.
fn constant_to_copy_source(c: &super::flow::Constant) -> Operand {
    use super::flow::ConstantValue;
    match &c.value {
        ConstantValue::None => Operand::ConstRef(0),
        ConstantValue::Bool(b) => Operand::ConstInt(if *b { 1 } else { 0 }),
        ConstantValue::Signed(v) => Operand::ConstInt(*v),
        // Str / Atom / Opaque constants do not currently appear as
        // `{kind}_copy` sources — pyre's walker only threads None /
        // Bool / Signed / opaque host-object refs into link args.  An
        // unexpected kind here means slice 2h extended the constant
        // space without updating this mapping.
        ConstantValue::Str(_) | ConstantValue::Atom(_) | ConstantValue::Opaque(_) => {
            panic!(
                "constant_to_copy_source: unsupported constant kind {:?}",
                c.value
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_repr_matches_rpython() {
        // RPython `flatten.py:33` `return "%%%s%d" % (self.kind[0], self.index)`.
        assert_eq!(Register::new(Kind::Int, 0).kind.first_char(), 'i');
        assert_eq!(Register::new(Kind::Ref, 3).kind.first_char(), 'r');
        assert_eq!(Register::new(Kind::Float, 7).kind.first_char(), 'f');
    }

    #[test]
    fn kind_as_str_roundtrip() {
        // RPython `flatten.py:59` `KINDS = ['int', 'ref', 'float']`.
        assert_eq!(Kind::Int.as_str(), "int");
        assert_eq!(Kind::Ref.as_str(), "ref");
        assert_eq!(Kind::Float.as_str(), "float");
    }

    #[test]
    fn label_equality_follows_name() {
        // RPython `flatten.py:17-18` eq compares `.name`.
        assert_eq!(Label::new("foo"), Label::new("foo"));
        assert_ne!(Label::new("foo"), Label::new("bar"));
    }

    #[test]
    fn tlabel_equality_follows_name() {
        assert_eq!(TLabel::new("foo"), TLabel::new("foo"));
        assert_ne!(TLabel::new("foo"), TLabel::new("bar"));
    }

    // --------------------------------------------------------------------
    // Renaming helpers (`reorder_renaming_list` / `insert_renamings` /
    // `generate_last_exc`) ported from `flatten.py:306-413`.  Tests
    // mirror `rpython/jit/codewriter/test/test_flatten.py:83-128`.
    // --------------------------------------------------------------------

    use super::super::flow::{Constant as FlowConstant, FlowValue, Link, Variable, VariableId};

    fn var(id: u32, kind: Kind) -> Variable {
        Variable::new(VariableId(id), kind)
    }

    fn reg(kind: Kind, index: u16) -> Register {
        Register::new(kind, index)
    }

    /// Synthetic getcolor: maps `VariableId(n)` to `Register(kind, n as u16)`.
    fn trivial_getcolor(v: &Variable) -> Register {
        let kind = v.kind.expect("test variable must be typed");
        Register::new(kind, v.id.0 as u16)
    }

    #[test]
    fn reorder_renaming_list_identity_pass_through() {
        // No aliasing — every (frm, to) fires in the original order.
        let frm = vec![1u16, 2, 3];
        let to = vec![10u16, 20, 30];
        let result = reorder_renaming_list(&frm, &to);
        assert_eq!(
            result,
            vec![
                (Some(1), Some(10)),
                (Some(2), Some(20)),
                (Some(3), Some(30)),
            ]
        );
    }

    #[test]
    fn reorder_renaming_list_reorders_to_avoid_overwrite() {
        // `frm = [1, 2]`, `to = [2, 3]` — copy 2->3 first, then 1->2.
        let frm = vec![1u16, 2];
        let to = vec![2u16, 3];
        let result = reorder_renaming_list(&frm, &to);
        assert_eq!(result, vec![(Some(2), Some(3)), (Some(1), Some(2))]);
    }

    #[test]
    fn reorder_renaming_list_breaks_two_cycle_via_push_pop() {
        // Swap: frm=[1,2], to=[2,1]. Upstream breaks at the first
        // pending slot: push(1); copy 2->1; pop(->2).
        let frm = vec![1u16, 2];
        let to = vec![2u16, 1];
        let result = reorder_renaming_list(&frm, &to);
        assert_eq!(
            result,
            vec![(Some(1), None), (Some(2), Some(1)), (None, Some(2)),]
        );
    }

    #[test]
    fn insert_renamings_emits_sorted_kind_grouped_copies() {
        // Build a link that renames r0 -> r2 (Ref) and r1 -> r3 (Ref).
        let v_src_a = var(0, Kind::Ref);
        let v_src_b = var(1, Kind::Ref);
        let v_dst_a = var(2, Kind::Ref);
        let v_dst_b = var(3, Kind::Ref);
        let link = Link::new(
            vec![FlowValue::from(v_src_a), FlowValue::from(v_src_b)],
            None,
            None,
        );
        let inputargs = vec![FlowValue::from(v_dst_a), FlowValue::from(v_dst_b)];
        let mut ssarepr = SSARepr::new("t");
        insert_renamings(&link, &inputargs, trivial_getcolor, &mut ssarepr);

        assert_eq!(ssarepr.insns.len(), 2);
        for insn in &ssarepr.insns {
            match insn {
                Insn::Op {
                    opname,
                    args,
                    result,
                } => {
                    assert_eq!(opname, "ref_copy");
                    assert_eq!(args.len(), 1);
                    assert!(result.is_some());
                }
                other => panic!("unexpected insn {:?}", other),
            }
        }
    }

    #[test]
    fn insert_renamings_skips_identity_copies() {
        // Source and destination share the same colour — emit nothing.
        let v_same = var(0, Kind::Ref);
        let link = Link::new(vec![FlowValue::from(v_same)], None, None);
        let inputargs = vec![FlowValue::from(v_same)];
        let mut ssarepr = SSARepr::new("t");
        insert_renamings(&link, &inputargs, trivial_getcolor, &mut ssarepr);
        assert!(ssarepr.insns.is_empty());
    }

    #[test]
    fn insert_renamings_breaks_ref_swap_via_push_pop() {
        // Swap r0 <-> r1.  `reorder_renaming_list` sees `frm=[1,0]`,
        // `to=[0,1]` (after `lst.sort_by_key(w.index)`), cycle head
        // becomes the first item → push(1), copy 0->1, pop(->0).
        let v_a = var(0, Kind::Ref);
        let v_b = var(1, Kind::Ref);
        let link = Link::new(
            // args in original order: (r0 -> ?, r1 -> ?)
            vec![FlowValue::from(v_a), FlowValue::from(v_b)],
            None,
            None,
        );
        // target inputargs: slot 0 wants r1's value, slot 1 wants r0's value.
        let inputargs = vec![FlowValue::from(v_b), FlowValue::from(v_a)];
        // ... but trivial_getcolor maps Variable(id=n) -> Register(Ref, n),
        // so the rename list is (r0 -> r1, r1 -> r0) — a 2-cycle.
        let mut ssarepr = SSARepr::new("t");
        insert_renamings(&link, &inputargs, trivial_getcolor, &mut ssarepr);

        assert_eq!(ssarepr.insns.len(), 3);
        let opnames: Vec<String> = ssarepr
            .insns
            .iter()
            .filter_map(|i| match i {
                Insn::Op { opname, .. } => Some(opname.clone()),
                _ => None,
            })
            .collect();
        // Exactly one push, one copy, one pop — order as above.
        assert_eq!(opnames, vec!["ref_push", "ref_copy", "ref_pop"]);
    }

    #[test]
    fn insert_renamings_groups_by_kind() {
        // Int and Ref renamings co-exist on the same link.  Both groups
        // emit copies independently — no cross-kind interference.
        let int_src = var(0, Kind::Int);
        let int_dst = var(1, Kind::Int);
        let ref_src = var(2, Kind::Ref);
        let ref_dst = var(3, Kind::Ref);
        let link = Link::new(
            vec![FlowValue::from(int_src), FlowValue::from(ref_src)],
            None,
            None,
        );
        let inputargs = vec![FlowValue::from(int_dst), FlowValue::from(ref_dst)];
        let mut ssarepr = SSARepr::new("t");
        insert_renamings(&link, &inputargs, trivial_getcolor, &mut ssarepr);

        let opnames: Vec<String> = ssarepr
            .insns
            .iter()
            .filter_map(|i| match i {
                Insn::Op { opname, .. } => Some(opname.clone()),
                _ => None,
            })
            .collect();
        // `Kind::ALL = [Int, Ref, Float]` ordering guarantees int_copy
        // fires before ref_copy.
        assert_eq!(opnames, vec!["int_copy", "ref_copy"]);
    }

    #[test]
    fn insert_renamings_constant_source_emits_copy_with_constint() {
        // Constant source → emitted as `{kind}_copy ConstInt -> dst`.
        let dst = var(0, Kind::Int);
        let link = Link::new(vec![FlowValue::from(FlowConstant::signed(42))], None, None);
        let inputargs = vec![FlowValue::from(dst)];
        let mut ssarepr = SSARepr::new("t");
        insert_renamings(&link, &inputargs, trivial_getcolor, &mut ssarepr);

        assert_eq!(ssarepr.insns.len(), 1);
        match &ssarepr.insns[0] {
            Insn::Op {
                opname,
                args,
                result,
            } => {
                assert_eq!(opname, "int_copy");
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0], Operand::ConstInt(42)));
                assert_eq!(*result, Some(reg(Kind::Int, 0)));
            }
            other => panic!("unexpected insn {:?}", other),
        }
    }

    #[test]
    fn insert_renamings_skips_last_exception_extras() {
        // Extravars (`last_exception` / `last_exc_value`) must be routed
        // through `generate_last_exc`, not the copy loop.
        let exc_type = var(10, Kind::Ref);
        let exc_value = var(11, Kind::Ref);
        let dst_type = var(0, Kind::Ref);
        let dst_value = var(1, Kind::Ref);
        let mut link = Link::new(
            vec![FlowValue::from(exc_type), FlowValue::from(exc_value)],
            None,
            None,
        );
        link.extravars(Some(exc_type), Some(exc_value));
        let inputargs = vec![FlowValue::from(dst_type), FlowValue::from(dst_value)];
        let mut ssarepr = SSARepr::new("t");
        insert_renamings(&link, &inputargs, trivial_getcolor, &mut ssarepr);

        // Copy loop: both args are extravars and skipped.  Tail
        // `generate_last_exc`: emits last_exception -> r0, last_exc_value -> r1.
        let opnames: Vec<String> = ssarepr
            .insns
            .iter()
            .filter_map(|i| match i {
                Insn::Op { opname, .. } => Some(opname.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(opnames, vec!["last_exception", "last_exc_value"]);
    }

    #[test]
    fn generate_last_exc_emits_nothing_when_extras_absent() {
        let v = var(0, Kind::Ref);
        let link = Link::new(vec![FlowValue::from(v)], None, None);
        let inputargs = vec![FlowValue::from(v)];
        let mut ssarepr = SSARepr::new("t");
        generate_last_exc(&link, &inputargs, trivial_getcolor, &mut ssarepr);
        assert!(ssarepr.insns.is_empty());
    }
}
