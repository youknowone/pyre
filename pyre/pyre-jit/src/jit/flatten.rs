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

use std::{collections::HashMap, rc::Rc};

use majit_translate::jit_codewriter::flatten::reorder_renaming_list;
use majit_translate::jitcode::BhDescr;

use super::flow::{
    BlockRef, Constant, ConstantValue, FlowValue, LinkRef, SpaceOperation, SpaceOperationArg,
    Variable,
};

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

/// Minimal production slice of `rpython/jit/codewriter/flatten.py:
/// 60-350` `GraphFlattener`.
///
/// Upstream owns the whole `FunctionGraph -> SSARepr` lowering. pyre is
/// still in the transitional dual-write phase, so this helper currently
/// serializes individual graph-level `SpaceOperation`s into `Insn`s and is
/// used only for the first production op migrated off direct SSA emission.
/// Expand this helper as more ops move from `codewriter.rs` into the
/// flow-graph + flatten pipeline.
pub struct GraphFlattener<'a, F, C = fn(&Constant) -> Operand> {
    ssarepr: &'a mut SSARepr,
    get_register: F,
    lower_constant: C,
    seen_blocks: HashMap<BlockRef, bool>,
    block_names: HashMap<BlockRef, String>,
    link_names: HashMap<LinkRef, String>,
    next_label_id: usize,
    include_all_exc_links: bool,
}

impl<'a, F> GraphFlattener<'a, F>
where
    F: FnMut(Variable) -> Register,
{
    pub fn new(ssarepr: &'a mut SSARepr, get_register: F) -> Self {
        Self {
            ssarepr,
            get_register,
            lower_constant: flatten_constant_operand,
            seen_blocks: HashMap::new(),
            block_names: HashMap::new(),
            link_names: HashMap::new(),
            next_label_id: 0,
            include_all_exc_links: false,
        }
    }
}

impl<'a, F, C> GraphFlattener<'a, F, C>
where
    F: FnMut(Variable) -> Register,
    C: FnMut(&Constant) -> Operand,
{
    pub fn new_with_constant_lowering(
        ssarepr: &'a mut SSARepr,
        get_register: F,
        lower_constant: C,
    ) -> Self {
        Self {
            ssarepr,
            get_register,
            lower_constant,
            seen_blocks: HashMap::new(),
            block_names: HashMap::new(),
            link_names: HashMap::new(),
            next_label_id: 0,
            include_all_exc_links: false,
        }
    }

    pub fn emit_space_operation(&mut self, op: &SpaceOperation) {
        let insn = self.flatten_space_operation(op);
        self.ssarepr.insns.push(insn);
    }

    fn emitline(&mut self, insn: Insn) {
        self.ssarepr.insns.push(insn);
    }

    fn label_name_for_block(&mut self, block: &BlockRef) -> String {
        if let Some(name) = self.block_names.get(block) {
            return name.clone();
        }
        let name = format!("block{}", self.next_label_id);
        self.next_label_id += 1;
        self.block_names.insert(block.clone(), name.clone());
        name
    }

    fn label_name_for_link(&mut self, link: &LinkRef) -> String {
        if let Some(name) = self.link_names.get(link) {
            return name.clone();
        }
        let name = format!("link{}", self.next_label_id);
        self.next_label_id += 1;
        self.link_names.insert(link.clone(), name.clone());
        name
    }

    fn tlabel_for_block(&mut self, block: &BlockRef) -> Operand {
        Operand::TLabel(TLabel::new(self.label_name_for_block(block)))
    }

    fn tlabel_for_link(&mut self, link: &LinkRef) -> Operand {
        Operand::TLabel(TLabel::new(self.label_name_for_link(link)))
    }

    fn label_for_block(&mut self, block: &BlockRef) -> Insn {
        Insn::Label(Label::new(self.label_name_for_block(block)))
    }

    fn label_for_link(&mut self, link: &LinkRef) -> Insn {
        Insn::Label(Label::new(self.label_name_for_link(link)))
    }

    fn flow_kind(value: &FlowValue) -> Option<Kind> {
        match value {
            FlowValue::Variable(variable) => variable.kind,
            FlowValue::Constant(constant) => constant.kind,
        }
    }

    fn rename_operand(&mut self, value: &FlowValue) -> RenameOperand {
        match self.flatten_value(value) {
            Operand::Register(register) => RenameOperand::Register(register),
            Operand::ConstInt(value) => RenameOperand::ConstInt(value),
            Operand::ConstRef(value) => RenameOperand::ConstRef(value),
            Operand::ConstFloat(value) => RenameOperand::ConstFloat(value),
            other => panic!("insert_renamings expects register/constant, got {other:?}"),
        }
    }

    fn make_return(&mut self, args: &[FlowValue]) {
        match args {
            [value] => match Self::flow_kind(value) {
                None => self.emitline(Insn::op("void_return", Vec::new())),
                Some(kind) => {
                    let opname = format!("{}_return", kind.as_str());
                    let operand = self.flatten_value(value);
                    self.emitline(Insn::op(opname, vec![operand]));
                }
            },
            [_, exc_value] => {
                if exc_value.as_variable().is_some() {
                    self.emitline(Insn::live(Vec::new()));
                }
                let operand = self.flatten_value(exc_value);
                self.emitline(Insn::op("raise", vec![operand]));
            }
            _ => panic!("make_return expects 1 or 2 args, got {}", args.len()),
        }
        self.emitline(Insn::Unreachable);
    }

    fn make_link(&mut self, link: &LinkRef, handling_ovf: bool) {
        let (target, args, last_exception, last_exc_value, can_return_directly) = {
            let link_borrow = link.borrow();
            let target = link_borrow
                .target
                .clone()
                .expect("link target required for make_link");
            let target_is_final = target.borrow().exits.is_empty();
            let uses_last_exception = link_borrow.args.iter().any(|arg| {
                arg.as_ref()
                    .and_then(FlowValue::as_variable)
                    .is_some_and(|value| Some(value) == link_borrow.last_exception)
            });
            let uses_last_exc_value = link_borrow.args.iter().any(|arg| {
                arg.as_ref()
                    .and_then(FlowValue::as_variable)
                    .is_some_and(|value| Some(value) == link_borrow.last_exc_value)
            });
            (
                target,
                link_borrow
                    .args
                    .iter()
                    .flatten()
                    .cloned()
                    .collect::<Vec<_>>(),
                link_borrow.last_exception,
                link_borrow.last_exc_value,
                target_is_final && !uses_last_exception && !uses_last_exc_value,
            )
        };
        if can_return_directly {
            self.make_return(&args);
            return;
        }
        let _ = (last_exception, last_exc_value, handling_ovf);
        self.insert_renamings(link);
        self.make_bytecode_block(target, handling_ovf);
    }

    fn make_exception_link(&mut self, link: &LinkRef, handling_ovf: bool) {
        let should_reraise = {
            let link_borrow = link.borrow();
            let Some(last_exception) = link_borrow.last_exception else {
                panic!("make_exception_link requires last_exception");
            };
            let Some(last_exc_value) = link_borrow.last_exc_value else {
                panic!("make_exception_link requires last_exc_value");
            };
            let target = link_borrow
                .target
                .clone()
                .expect("link target required for make_exception_link");
            target.borrow().operations.is_empty()
                && target.borrow().exits.is_empty()
                && link_borrow.args.len() == 2
                && link_borrow.args[0] == Some(last_exception.into())
                && link_borrow.args[1] == Some(last_exc_value.into())
        };
        if should_reraise {
            assert!(
                !handling_ovf,
                "overflow exception edges are not modeled in pyre flatten_graph yet"
            );
            self.emitline(Insn::op("reraise", Vec::new()));
            self.emitline(Insn::Unreachable);
            return;
        }
        self.make_link(link, handling_ovf);
    }

    fn insert_exits(&mut self, block: &BlockRef, handling_ovf: bool) {
        let exits = block.borrow().exits.clone();
        if exits.len() == 1 {
            self.make_link(&exits[0], handling_ovf);
            return;
        }
        if block.borrow().canraise() {
            if !self.include_all_exc_links && block.borrow().raising_op().is_none() {
                self.make_link(&exits[0], false);
                return;
            }
            let catch_label = self.tlabel_for_link(&exits[0]);
            self.emitline(Insn::op("catch_exception", vec![catch_label]));
            self.make_link(&exits[0], false);
            let normal_label = self.label_for_link(&exits[0]);
            self.emitline(normal_label);
            let mut captured_all = false;
            for link in exits.iter().skip(1) {
                let llexitcase = link.borrow().llexitcase.clone();
                if let Some(case) = llexitcase {
                    let case_operand = self.flatten_value(&case);
                    let mismatch_label = self.tlabel_for_link(link);
                    self.emitline(Insn::op(
                        "goto_if_exception_mismatch",
                        vec![case_operand, mismatch_label],
                    ));
                    self.make_exception_link(link, false);
                    let link_label = self.label_for_link(link);
                    self.emitline(link_label);
                } else {
                    self.make_exception_link(link, false);
                    captured_all = true;
                    break;
                }
            }
            if !captured_all {
                self.emitline(Insn::op("reraise", Vec::new()));
                self.emitline(Insn::Unreachable);
            }
            return;
        }
        if exits.len() == 2 {
            let Some(exitswitch) = block.borrow().exitswitch.clone() else {
                panic!("flatten_graph: 2-exit block missing exitswitch");
            };
            let mut linkfalse = exits[0].clone();
            let mut linktrue = exits[1].clone();
            if linkfalse.borrow().llexitcase == Some(Constant::bool(true).into()) {
                std::mem::swap(&mut linkfalse, &mut linktrue);
            }
            let opargs = match exitswitch {
                super::flow::ExitSwitch::Value(value) => {
                    vec![self.flatten_value(&value), self.tlabel_for_link(&linkfalse)]
                }
                super::flow::ExitSwitch::Tuple(_) => {
                    panic!("flatten_graph: tuple exitswitch not wired yet")
                }
            };
            self.emitline(Insn::live(Vec::new()));
            self.emitline(Insn::op("goto_if_not", opargs));
            self.make_link(&linktrue, handling_ovf);
            let false_label = self.label_for_link(&linkfalse);
            self.emitline(false_label);
            self.make_link(&linkfalse, handling_ovf);
            return;
        }
        panic!(
            "flatten_graph: unsupported exits shape for block with {} exits",
            exits.len()
        );
    }

    fn insert_renamings(&mut self, link: &LinkRef) {
        let (target_inputargs, last_exception, last_exc_value, link_args) = {
            let link_borrow = link.borrow();
            let target = link_borrow
                .target
                .clone()
                .expect("link target required for insert_renamings");
            (
                target.borrow().inputargs.clone(),
                link_borrow.last_exception,
                link_borrow.last_exc_value,
                link_borrow.args.clone(),
            )
        };

        let mut pairs: Vec<(RenameOperand, Register)> = Vec::new();
        for (index, arg) in link_args.iter().enumerate() {
            let Some(src_value) = arg.as_ref() else {
                continue;
            };
            let Some(dst_variable) = target_inputargs[index].as_variable() else {
                continue;
            };
            let src_variable = src_value.as_variable();
            if src_variable == last_exception || src_variable == last_exc_value {
                continue;
            }
            let src = self.rename_operand(src_value);
            let dst = (self.get_register)(dst_variable);
            if src == RenameOperand::Register(dst) {
                continue;
            }
            pairs.push((src, dst));
        }
        pairs.sort_by_key(|(_, dst)| dst.index);

        let mut renamings: HashMap<Kind, (Vec<RenameOperand>, Vec<RenameOperand>)> = HashMap::new();
        for (src, dst) in pairs {
            let (frm, to) = renamings.entry(dst.kind).or_default();
            frm.push(src);
            to.push(RenameOperand::Register(dst));
        }
        for &kind in &Kind::ALL {
            if let Some((frm, to)) = renamings.get(&kind) {
                for (src, dst) in reorder_renaming_list(frm, to) {
                    match (src, dst) {
                        (Some(src), Some(RenameOperand::Register(dst))) => {
                            self.emitline(Insn::op_with_result(
                                format!("{}_copy", kind.as_str()),
                                vec![src.into_operand()],
                                dst,
                            ));
                        }
                        (Some(RenameOperand::Register(src)), None) => {
                            self.emitline(Insn::op(
                                format!("{}_push", kind.as_str()),
                                vec![Operand::Register(src)],
                            ));
                        }
                        (None, Some(RenameOperand::Register(dst))) => {
                            self.emitline(Insn::op_with_result(
                                format!("{}_pop", kind.as_str()),
                                Vec::new(),
                                dst,
                            ));
                        }
                        other => panic!("unexpected renaming step {other:?}"),
                    }
                }
            }
        }
        let link_borrow = link.borrow();
        self.generate_last_exc(&link_borrow, &target_inputargs);
    }

    fn generate_last_exc(&mut self, link: &super::flow::Link, inputargs: &[FlowValue]) {
        if link.last_exception.is_none() && link.last_exc_value.is_none() {
            return;
        }
        for (arg, inputarg) in link.args.iter().zip(inputargs) {
            if arg.as_ref().and_then(FlowValue::as_variable) == link.last_exception {
                let dst = inputarg
                    .as_variable()
                    .expect("last_exception target must be a Variable");
                let dst_reg = (self.get_register)(dst);
                self.emitline(Insn::op_with_result("last_exception", Vec::new(), dst_reg));
            }
        }
        for (arg, inputarg) in link.args.iter().zip(inputargs) {
            if arg.as_ref().and_then(FlowValue::as_variable) == link.last_exc_value {
                let dst = inputarg
                    .as_variable()
                    .expect("last_exc_value target must be a Variable");
                let dst_reg = (self.get_register)(dst);
                self.emitline(Insn::op_with_result("last_exc_value", Vec::new(), dst_reg));
            }
        }
    }

    fn make_bytecode_block(&mut self, block: BlockRef, handling_ovf: bool) {
        if block.borrow().exits.is_empty() {
            let args = block.borrow().inputargs.clone();
            self.make_return(&args);
            return;
        }
        if self.seen_blocks.contains_key(&block) {
            let target = self.tlabel_for_block(&block);
            self.emitline(Insn::op("goto", vec![target]));
            self.emitline(Insn::Unreachable);
            return;
        }
        self.seen_blocks.insert(block.clone(), true);
        let block_label = self.label_for_block(&block);
        self.emitline(block_label);
        let operations = block.borrow().operations.clone();
        for op in &operations {
            self.emit_space_operation(op);
        }
        self.insert_exits(&block, handling_ovf);
    }

    fn flatten_space_operation(&mut self, op: &SpaceOperation) -> Insn {
        let args = op.args.iter().map(|arg| self.flatten_arg(arg)).collect();
        match op.result {
            None => Insn::op(op.opname.clone(), args),
            Some(FlowValue::Variable(variable)) => {
                let result = (self.get_register)(variable);
                Insn::op_with_result(op.opname.clone(), args, result)
            }
            Some(FlowValue::Constant(ref constant)) => {
                panic!(
                    "GraphFlattener: op {} has Constant result {:?}; \
                     flow graph results must be Variables",
                    op.opname, constant
                )
            }
        }
    }

    fn flatten_arg(&mut self, arg: &SpaceOperationArg) -> Operand {
        match arg {
            SpaceOperationArg::Value(value) => self.flatten_value(value),
            SpaceOperationArg::ListOfKind(list) => Operand::ListOfKind(ListOfKind::new(
                list.kind,
                list.content
                    .iter()
                    .map(|value| self.flatten_value(value))
                    .collect(),
            )),
            // `flatten.py:365-367` passes AbstractDescr through
            // unchanged. `flow::SpaceOperationArg::Descr` already carries
            // the closed-world SSA-side `DescrOperand`, so flattening is
            // the direct identity-preserving `Operand::Descr` wrap.
            SpaceOperationArg::Descr(descr) => Operand::descr_rc(Rc::clone(&descr.0)),
            // `flatten.py:365-367` also passes IndirectCallTargets
            // through unchanged.  `Operand::IndirectCallTargets` takes a
            // value, so clone the inner (the `Vec<Arc<JitCode>>` clone
            // is cheap — it bumps Arc refcounts).
            SpaceOperationArg::IndirectCallTargets(targets) => {
                Operand::IndirectCallTargets((*targets.0).clone())
            }
        }
    }

    fn flatten_value(&mut self, value: &FlowValue) -> Operand {
        match value {
            FlowValue::Variable(variable) => Operand::Register((self.get_register)(*variable)),
            FlowValue::Constant(constant) => (self.lower_constant)(constant),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum RenameOperand {
    Register(Register),
    ConstInt(i64),
    ConstRef(i64),
    ConstFloat(i64),
}

impl RenameOperand {
    fn into_operand(self) -> Operand {
        match self {
            Self::Register(register) => Operand::Register(register),
            Self::ConstInt(value) => Operand::ConstInt(value),
            Self::ConstRef(value) => Operand::ConstRef(value),
            Self::ConstFloat(value) => Operand::ConstFloat(value),
        }
    }
}

fn flatten_constant_operand(constant: &super::flow::Constant) -> Operand {
    match (&constant.value, constant.kind) {
        (ConstantValue::None, Some(Kind::Ref)) => Operand::ConstRef(0),
        (ConstantValue::Bool(value), Some(Kind::Int)) => Operand::ConstInt(i64::from(*value)),
        (ConstantValue::Signed(value), Some(Kind::Int)) => Operand::ConstInt(*value),
        (ConstantValue::Opaque(_), Some(Kind::Ref)) => {
            panic!("GraphFlattener: opaque ref constants need runtime lowering support")
        }
        other => panic!("GraphFlattener: unsupported constant operand {other:?}"),
    }
}

/// Block-level walker matching `rpython/jit/codewriter/flatten.py:60
/// flatten_graph(graph, regallocs)`.  Walks every block in `graph`,
/// emits a `Label` for each block and an `Insn` for each
/// `SpaceOperation`, producing the SSARepr that the assembler consumes.
///
/// `regallocs` keyed by `Kind` provides the per-kind
/// `GraphAllocationResult` that the caller computed via
/// `regalloc::perform_graph_register_allocation_all_kinds(graph)`
/// (upstream `codewriter.py:44-46`'s `regallocs` dict).  `get_register`
/// projects `Variable` to `Register` using the appropriate per-kind
/// coloring; `lower_constant` handles non-raw constants (pycode opaque
/// refs, jitdriver descrs, etc.) that default `flatten_constant_operand`
/// cannot express on its own.
///
/// **Phase 1 scaffold for Task #224**: currently covers the ops whose
/// graph shape exists in production (`loop_header`, `jit_merge_point`
/// with the 7-arg upstream shape); ops that pyre's walker still emits
/// directly into the SSARepr (the bulk of opcodes in
/// `codewriter.rs::transform_graph_to_jitcode`) do not yet have a
/// corresponding graph-side `SpaceOperation`, so `flatten_graph`
/// cannot reproduce the walker's full output.  Wiring Phase 1c
/// replaces the direct SSARepr emission with `record_graph_op` at
/// every walker emit point and then switches production to call this
/// function in place of the walker's interleaved
/// `ssarepr.insns.push(...)` calls.
///
/// Matches upstream structure:
/// - `flatten_graph`: driver entry point (this function)
/// - `generate_ssa_form`: block iteration + per-op dispatch
///   (delegated here to `GraphFlattener::emit_space_operation`)
/// - `make_bytecode_block`/`make_link`/`insert_exits`: block boundary
///   handling — not yet implemented; `Label` insertion happens at
///   block entry only, `insert_exits` equivalent is not yet wired.
pub fn flatten_graph<F, C>(
    graph: &super::flow::FunctionGraph,
    ssarepr: &mut SSARepr,
    get_register: F,
    lower_constant: C,
) where
    F: FnMut(Variable) -> Register,
    C: FnMut(&Constant) -> Operand,
{
    let mut flattener =
        GraphFlattener::new_with_constant_lowering(ssarepr, get_register, lower_constant);
    flattener.make_bytecode_block(graph.startblock.clone(), false);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::flow::{FlowListOfKind, VariableId};

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

    #[test]
    fn graph_flattener_emits_loop_header_from_graph_op() {
        let op = SpaceOperation::new("loop_header", vec![Constant::signed(0).into()], None, 17);
        let mut ssarepr = SSARepr::new("test");
        let mut flattener = GraphFlattener::new(&mut ssarepr, |_| {
            Register::new(Kind::Ref, VariableId(0).0 as u16)
        });

        flattener.emit_space_operation(&op);

        match &ssarepr.insns[..] {
            [
                Insn::Op {
                    opname,
                    args,
                    result,
                },
            ] => {
                assert_eq!(opname, "loop_header");
                assert!(result.is_none());
                assert!(matches!(args.as_slice(), [Operand::ConstInt(0)]));
            }
            other => panic!("unexpected insns: {other:?}"),
        }
    }

    #[test]
    fn graph_flattener_preserves_jit_merge_point_graph_shape() {
        let frame = Variable::new(VariableId(10), Kind::Ref);
        let ec = Variable::new(VariableId(11), Kind::Ref);
        let op = SpaceOperation::new(
            "jit_merge_point",
            vec![
                Constant::signed(0).into(),
                FlowListOfKind::new(
                    Kind::Int,
                    vec![Constant::signed(17).into(), Constant::signed(0).into()],
                )
                .into(),
                FlowListOfKind::new(
                    Kind::Ref,
                    vec![Constant::opaque("pycode", Some(Kind::Ref)).into()],
                )
                .into(),
                FlowListOfKind::new(Kind::Float, vec![]).into(),
                FlowListOfKind::new(Kind::Int, vec![]).into(),
                FlowListOfKind::new(Kind::Ref, vec![frame.into(), ec.into()]).into(),
                FlowListOfKind::new(Kind::Float, vec![]).into(),
            ],
            None,
            3,
        );
        let mut ssarepr = SSARepr::new("test");
        let mut flattener = GraphFlattener::new_with_constant_lowering(
            &mut ssarepr,
            |v| Register::new(v.kind.expect("typed variable"), v.id.0 as u16),
            |constant| match (&constant.value, constant.kind) {
                (ConstantValue::Signed(value), Some(Kind::Int)) => Operand::ConstInt(*value),
                (ConstantValue::Opaque(_), Some(Kind::Ref)) => Operand::ConstRef(99),
                other => panic!("unexpected test constant {other:?}"),
            },
        );

        flattener.emit_space_operation(&op);

        match &ssarepr.insns[..] {
            [
                Insn::Op {
                    opname,
                    args,
                    result,
                },
            ] => {
                assert_eq!(opname, "jit_merge_point");
                assert!(result.is_none());
                assert_eq!(args.len(), 7);
                assert!(matches!(args[0], Operand::ConstInt(0)));
                assert!(matches!(
                    &args[1],
                    Operand::ListOfKind(ListOfKind { kind: Kind::Int, content })
                        if matches!(content.as_slice(), [Operand::ConstInt(17), Operand::ConstInt(0)])
                ));
                assert!(matches!(
                    &args[2],
                    Operand::ListOfKind(ListOfKind { kind: Kind::Ref, content })
                        if matches!(content.as_slice(), [Operand::ConstRef(99)])
                ));
                assert!(matches!(
                    &args[5],
                    Operand::ListOfKind(ListOfKind { kind: Kind::Ref, content })
                        if matches!(
                            content.as_slice(),
                            [
                                Operand::Register(Register { kind: Kind::Ref, index: 10 }),
                                Operand::Register(Register { kind: Kind::Ref, index: 11 })
                            ]
                        )
                ));
            }
            other => panic!("unexpected insns: {other:?}"),
        }
    }

    #[test]
    fn flatten_graph_walks_all_blocks_and_emits_each_op() {
        // Synthetic graph with two blocks; each contains a loop_header
        // op with a distinct offset tag.  flatten_graph must walk every
        // block and emit one Insn per SpaceOperation.
        use crate::jit::flow::{Block, FunctionGraph};
        let start_arg = Variable::new(VariableId(0), Kind::Ref);
        let next_arg = Variable::new(VariableId(1), Kind::Ref);
        let start = Block::shared(vec![start_arg.into()]);
        let mut graph = FunctionGraph::new("flat_walk", start.clone(), None);
        let next = graph.new_block(vec![next_arg.into()]);
        super::super::flow::push_op(
            &start,
            SpaceOperation::new("loop_header", vec![Constant::signed(0).into()], None, 0),
        );
        super::super::flow::push_op(
            &next,
            SpaceOperation::new("loop_header", vec![Constant::signed(0).into()], None, 1),
        );
        start.closeblock(vec![
            super::super::flow::Link::new(vec![start_arg.into()], Some(next.clone()), None)
                .into_ref(),
        ]);
        next.closeblock(vec![
            super::super::flow::Link::new(
                vec![next_arg.into()],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        let mut ssarepr = SSARepr::new("flat_walk");
        flatten_graph(
            &graph,
            &mut ssarepr,
            |v| Register::new(v.kind.unwrap_or(Kind::Ref), v.id.0 as u16),
            flatten_constant_operand,
        );

        // Two loop_header Insns emitted — one per block.
        let header_count = ssarepr
            .insns
            .iter()
            .filter(|i| matches!(i, Insn::Op { opname, .. } if opname == "loop_header"))
            .count();
        assert_eq!(
            header_count, 2,
            "flatten_graph should emit one Insn per SpaceOperation across all blocks; got {:?}",
            ssarepr.insns
        );
    }

    #[test]
    fn flatten_graph_inserts_renamings_for_fallthrough_links() {
        use crate::jit::flow::{Block, FunctionGraph, Link};
        let src = Variable::new(VariableId(0), Kind::Ref);
        let dst = Variable::new(VariableId(1), Kind::Ref);
        let start = Block::shared(vec![src.into()]);
        let mut graph = FunctionGraph::new("renaming", start.clone(), Some(dst));
        let middle = graph.new_block(vec![dst.into()]);
        start.closeblock(vec![
            Link::new(vec![src.into()], Some(middle.clone()), None).into_ref(),
        ]);
        middle.closeblock(vec![
            Link::new(vec![dst.into()], Some(graph.returnblock.clone()), None).into_ref(),
        ]);

        let mut ssarepr = SSARepr::new("renaming");
        flatten_graph(
            &graph,
            &mut ssarepr,
            |v| Register::new(v.kind.expect("typed variable"), v.id.0 as u16),
            flatten_constant_operand,
        );

        assert!(ssarepr.insns.iter().any(|insn| {
            matches!(
                insn,
                Insn::Op {
                    opname,
                    args,
                    result: Some(Register {
                        kind: Kind::Ref,
                        index: 1
                    }),
                } if opname == "ref_copy"
                    && matches!(
                        args.as_slice(),
                        [Operand::Register(Register {
                            kind: Kind::Ref,
                            index: 0
                        })]
                    )
            )
        }));
    }

    #[test]
    fn flatten_graph_emits_exception_dispatch_and_last_exc_loads() {
        use crate::jit::flow::{Block, ExitSwitch, FunctionGraph, Link, c_last_exception};
        let normal = Variable::new(VariableId(0), Kind::Ref);
        let exc_type = Variable::new(VariableId(1), Kind::Int);
        let exc_value = Variable::new(VariableId(2), Kind::Ref);
        let catch_type = Variable::new(VariableId(3), Kind::Int);
        let catch_value = Variable::new(VariableId(4), Kind::Ref);
        let start = Block::shared(Vec::new());
        let mut graph = FunctionGraph::new("exc_dispatch", start.clone(), Some(normal));
        let typed_handler = graph.new_block(vec![exc_type.into(), exc_value.into()]);
        let catchall_handler = graph.new_block(vec![catch_type.into(), catch_value.into()]);
        super::super::flow::push_op(
            &start,
            SpaceOperation::new("call_can_raise", Vec::new(), None, 0),
        );

        typed_handler.closeblock(vec![
            Link::new(
                vec![exc_value.into()],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);
        catchall_handler.closeblock(vec![
            Link::new(
                vec![catch_value.into()],
                Some(graph.returnblock.clone()),
                None,
            )
            .into_ref(),
        ]);

        start.borrow_mut().exitswitch = Some(ExitSwitch::Value(c_last_exception().into()));
        let normal_link = Link::new(
            vec![Constant::none().into()],
            Some(graph.returnblock.clone()),
            None,
        )
        .into_ref();
        let mut typed_link = Link::new(
            vec![exc_type.into(), exc_value.into()],
            Some(typed_handler.clone()),
            None,
        )
        .with_llexitcase(Constant::signed(7).into());
        typed_link.extravars(Some(exc_type), Some(exc_value));
        let mut catchall_link = Link::new(
            vec![catch_type.into(), catch_value.into()],
            Some(catchall_handler.clone()),
            None,
        );
        catchall_link.extravars(Some(catch_type), Some(catch_value));
        start.closeblock(vec![
            normal_link,
            typed_link.into_ref(),
            catchall_link.into_ref(),
        ]);

        let mut ssarepr = SSARepr::new("exc_dispatch");
        flatten_graph(
            &graph,
            &mut ssarepr,
            |v| Register::new(v.kind.expect("typed variable"), v.id.0 as u16),
            flatten_constant_operand,
        );

        assert!(
            ssarepr
                .insns
                .iter()
                .any(|insn| matches!(insn, Insn::Op { opname, .. } if opname == "catch_exception"))
        );
        assert!(ssarepr.insns.iter().any(|insn| {
            matches!(
                insn,
                Insn::Op { opname, args, .. }
                    if opname == "goto_if_exception_mismatch"
                        && matches!(args.as_slice(), [Operand::ConstInt(7), Operand::TLabel(_)])
            )
        }));
        assert!(
            ssarepr
                .insns
                .iter()
                .any(|insn| matches!(insn, Insn::Op { opname, .. } if opname == "last_exception"))
        );
        assert!(
            ssarepr
                .insns
                .iter()
                .any(|insn| matches!(insn, Insn::Op { opname, .. } if opname == "last_exc_value"))
        );
    }

    #[test]
    fn flatten_graph_emits_boolean_branch_exits() {
        use crate::jit::flow::{Block, Constant, ExitSwitch, FunctionGraph, Link};

        let cond = Variable::new(VariableId(0), Kind::Int);
        let retval = Variable::new(VariableId(1), Kind::Int);
        let start = Block::shared(vec![cond.into()]);
        let mut graph = FunctionGraph::new("bool_branch", start.clone(), Some(retval));

        start.borrow_mut().exitswitch = Some(ExitSwitch::Value(cond.into()));
        let false_link = Link::new(
            vec![Constant::signed(0).into()],
            Some(graph.returnblock.clone()),
            Some(Constant::bool(false).into()),
        )
        .with_llexitcase(Constant::bool(false).into())
        .into_ref();
        let true_link = Link::new(
            vec![Constant::signed(1).into()],
            Some(graph.returnblock.clone()),
            Some(Constant::bool(true).into()),
        )
        .with_llexitcase(Constant::bool(true).into())
        .into_ref();
        start.closeblock(vec![false_link, true_link]);

        let mut ssarepr = SSARepr::new("bool_branch");
        flatten_graph(
            &graph,
            &mut ssarepr,
            |v| Register::new(v.kind.expect("typed variable"), v.id.0 as u16),
            flatten_constant_operand,
        );

        assert!(ssarepr.insns.iter().any(|insn| {
            matches!(
                insn,
                Insn::Op { opname, args, .. }
                    if opname == "goto_if_not"
                        && matches!(
                            args.as_slice(),
                            [Operand::Register(Register { kind: Kind::Int, index: 0 }), Operand::TLabel(_)]
                        )
            )
        }));
        assert!(ssarepr.insns.iter().any(|insn| {
            matches!(insn, Insn::Op { opname, args, .. }
                if opname == "int_return"
                    && matches!(args.as_slice(), [Operand::ConstInt(0)] | [Operand::ConstInt(1)]))
        }));
    }

    #[test]
    fn graph_flattener_emits_generic_result_op() {
        let src = Variable::new(VariableId(0), Kind::Ref);
        let dst = Variable::new(VariableId(1), Kind::Ref);
        let op = SpaceOperation::new("type", vec![src.into()], Some(dst.into()), 23);
        let mut ssarepr = SSARepr::new("generic");
        let mut flattener = GraphFlattener::new(&mut ssarepr, |variable| {
            Register::new(
                variable.kind.expect("test variable kind"),
                variable.id.0 as u16,
            )
        });

        flattener.emit_space_operation(&op);

        match &ssarepr.insns[..] {
            [
                Insn::Op {
                    opname,
                    args,
                    result: Some(result),
                },
            ] => {
                assert_eq!(opname, "type");
                assert!(matches!(
                    args.as_slice(),
                    [Operand::Register(Register {
                        kind: Kind::Ref,
                        index: 0
                    })]
                ));
                assert_eq!(*result, Register::new(Kind::Ref, 1));
            }
            other => panic!("unexpected insns: {other:?}"),
        }
    }

    #[test]
    fn graph_flattener_passes_descr_args_through_by_identity() {
        let int_arg = Variable::new(VariableId(0), Kind::Int);
        let ref_arg = Variable::new(VariableId(1), Kind::Ref);
        let dst = Variable::new(VariableId(2), Kind::Ref);
        let descr = Rc::new(DescrOperand::CallDescrStub(CallDescrStub {
            flavor: CallFlavor::MayForce,
            arg_kinds: vec![Kind::Ref, Kind::Int],
        }));
        let op = SpaceOperation::new(
            "residual_call_ir_r",
            vec![
                Constant::signed(17).into(),
                crate::jit::flow::FlowListOfKind::new(Kind::Int, vec![int_arg.into()]).into(),
                crate::jit::flow::FlowListOfKind::new(Kind::Ref, vec![ref_arg.into()]).into(),
                descr.clone().into(),
            ],
            Some(dst.into()),
            29,
        );
        let mut ssarepr = SSARepr::new("descr_passthrough");
        let mut flattener = GraphFlattener::new(&mut ssarepr, |variable| {
            Register::new(
                variable.kind.expect("test variable kind"),
                variable.id.0 as u16,
            )
        });

        flattener.emit_space_operation(&op);

        match &ssarepr.insns[..] {
            [Insn::Op { args, .. }] => {
                let Operand::Descr(emitted) = &args[3] else {
                    panic!("expected descr operand, got {:?}", args[3]);
                };
                assert!(Rc::ptr_eq(emitted, &descr));
            }
            other => panic!("unexpected insns: {other:?}"),
        }
    }
}
