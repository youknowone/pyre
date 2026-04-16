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

use majit_codewriter::jitcode::BhDescr;

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
/// The list carries `JitCode` references for the `indirect_call` opcode.
/// pyre does not currently emit `indirect_call` against user Python
/// functions, so this type is unused but kept for structural parity.
#[derive(Debug, Clone, Default)]
pub struct IndirectCallTargets {
    pub lst: Vec<majit_metainterp::jitcode::JitCode>,
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
    /// `SwitchDictDescr` with a live `_labels` table. The assembler
    /// records the descr in its table and emits its 16-bit index;
    /// liveness reads `_labels` to follow switch-target edges.
    Descr(DescrOperand),
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
}

/// Instruction tuple (`ssarepr.insns[i]`).
///
/// The five RPython tuple shapes enumerated above.
#[derive(Debug, Clone)]
pub enum Insn {
    /// `(Label(name),)` — block-entry marker.
    Label(Label),
    /// `('-live-', arg1, ...)` — liveness placeholder.
    ///
    /// `compute_liveness` (see `liveness.rs`) rewrites this in place with
    /// the full set of registers alive at this program point.
    Live(Vec<Operand>),
    /// `('---',)` — unreachable marker; clears the liveness pass's alive
    /// set (`liveness.py:70`).
    Unreachable,
    /// `(opname, args..., ['->' result])` — regular operation.
    ///
    /// `result` is `Some(register)` iff the RPython tuple contains a
    /// trailing `'->' result` pair; the assembler emits the `>` argcode
    /// in that case (`assembler.py:210-219`).
    Op {
        opname: String,
        args: Vec<Operand>,
        result: Option<Register>,
    },
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
}
