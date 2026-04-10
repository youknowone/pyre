//! JitCode — assembled bytecode + register/constant pools.
//!
//! RPython equivalent: `rpython/jit/codewriter/jitcode.py` class `JitCode`.
//!
//! In RPython this is a single shared type used by both the codewriter
//! (which writes into it via `Assembler.assemble`) and the metainterp
//! (which reads from it via `BlackholeInterpreter.dispatch_loop` and
//! `MetaInterp.handle_call_assembler`). majit currently has two `JitCode`
//! types — this `codewriter::jitcode::JitCode` (RPython orthodox encoding,
//! `insns` dict, dynamic argcodes) and `metainterp::jitcode::JitCode`
//! (pyre-specific BC_* hardcoded opcode set). Phase D will line-by-line
//! port `BlackholeInterpreter.setup_insns` so the metainterp can consume
//! this type directly, eliminating the fork.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

/// Assembled JitCode — the output of the assembler.
///
/// RPython parity (`rpython/jit/codewriter/jitcode.py:9-43`):
///
/// ```python
/// class JitCode(AbstractDescr):
///     def __init__(self, name, fnaddr=None, calldescr=None, called_from=None):
///         self.name = name
///         self.fnaddr = fnaddr
///         self.calldescr = calldescr
///         self.jitdriver_sd = None
///         self._called_from = called_from
///         self._ssarepr = None
///
///     def setup(self, code='', constants_i=[], constants_r=[], constants_f=[],
///               num_regs_i=255, num_regs_r=255, num_regs_f=255,
///               startpoints=None, alllabels=None, resulttypes=None):
///         self.code = code
///         self.constants_i = constants_i or self._empty_i
///         self.constants_r = constants_r or self._empty_r
///         self.constants_f = constants_f or self._empty_f
///         self.c_num_regs_i = chr(num_regs_i)
///         self.c_num_regs_r = chr(num_regs_r)
///         self.c_num_regs_f = chr(num_regs_f)
///         self._startpoints = startpoints
///         self._alllabels = alllabels
///         self._resulttypes = resulttypes
/// ```
///
/// Field-by-field mapping below preserves the RPython names. Where
/// RPython uses `chr(int)` to pack a 0..255 register count into a single
/// byte we use `u8` directly; the value range is identical.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JitCode {
    /// RPython `jitcode.py:15` `self.name = name`.
    pub name: String,
    /// RPython `jitcode.py:18` `self.jitdriver_sd = None`.
    /// `Some(index)` for portal jitcodes (set by `grab_initial_jitcodes`).
    pub jitdriver_sd: Option<usize>,
    /// RPython `jitcode.py:26` `self.code = code` — bytecode bytes.
    pub code: Vec<u8>,
    /// RPython `jitcode.py:32` `self.constants_i`.
    pub constants_i: Vec<i64>,
    /// RPython `jitcode.py:33` `self.constants_r`.
    /// RPython uses GCREF (opaque pointer); pyre uses `u64` until lltype
    /// is ported.
    pub constants_r: Vec<u64>,
    /// RPython `jitcode.py:34` `self.constants_f`.
    /// RPython packs the float as `longlong.FLOATSTORAGE`; pyre uses `f64`.
    pub constants_f: Vec<f64>,
    /// RPython `jitcode.py:37-39` `self.c_num_regs_i = chr(num_regs_i)`.
    /// RPython packs into a single chr because of `assert num_regs_i < 256`;
    /// majit uses u8 directly with the same range.
    pub c_num_regs_i: u8,
    /// RPython `jitcode.py:38` `self.c_num_regs_r = chr(num_regs_r)`.
    pub c_num_regs_r: u8,
    /// RPython `jitcode.py:39` `self.c_num_regs_f = chr(num_regs_f)`.
    pub c_num_regs_f: u8,
    /// RPython `jitcode.py:40` `self._startpoints = startpoints` —
    /// debug-only set of bytecode offsets where instructions start.
    pub startpoints: HashSet<usize>,
    /// RPython `jitcode.py:41` `self._alllabels = alllabels` — debug-only
    /// set of bytecode offsets that are label targets.
    pub alllabels: HashSet<usize>,
    /// RPython `jitcode.py:42` `self._resulttypes = resulttypes` —
    /// debug-only map from bytecode offset to result type char.
    pub resulttypes: HashMap<usize, char>,
    /// RPython `codewriter.py:68` `jitcode.index = index` — sequential
    /// position in `all_jitcodes[]`. Set by `CodeWriter` after assembly.
    /// Used by `inline_call` ops to reference callee jitcode by index.
    pub index: usize,
    /// RPython `jitcode.py:19` `self._called_from = called_from` — debug:
    /// which call graph first triggered this jitcode's creation. In RPython
    /// this is a graph object; pyre uses an optional CallPath string.
    #[serde(default, skip_serializing)]
    pub _called_from: Option<String>,
    /// RPython `jitcode.py:20` `self._ssarepr = None` — debug: the
    /// flattened SSA representation, kept for `dump()` output. Set by
    /// `Assembler.assemble` (assembler.py:49 `jitcode._ssarepr = ssarepr`).
    #[serde(skip)]
    pub _ssarepr: Option<crate::passes::flatten::SSARepr>,
    /// RPython: `BlackholeInterpBuilder.descrs` is a shared list of AbstractDescr
    /// objects, populated once via `setup_descrs(asm.descrs)`. All JitCodes share
    /// the same global descrs table.
    /// pyre: per-jitcode Assembler produces per-jitcode descrs, so we store them
    /// directly on the JitCode. The blackhole loads them via `setposition()`.
    #[serde(skip)]
    pub descrs: Vec<BhDescr>,
}

impl JitCode {
    /// RPython `jitcode.py:14-20` `JitCode.__init__(name, fnaddr=None,
    /// calldescr=None, called_from=None)`.
    ///
    /// Constructs a JitCode with name + default-initialized state. The
    /// `setup()` step (RPython `jitcode.py:22-42`) populates `code`,
    /// `constants_*`, `c_num_regs_*`, `startpoints`, etc. via the
    /// assembler.
    ///
    /// `fnaddr`, `calldescr`, `_called_from`, and `_ssarepr` from RPython
    /// are not yet ported (pyre lacks lltype function pointers and uses
    /// `CallPath` instead of `funcptr._obj.graph`).
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            jitdriver_sd: None,
            code: Vec::new(),
            constants_i: Vec::new(),
            constants_r: Vec::new(),
            constants_f: Vec::new(),
            c_num_regs_i: 255,
            c_num_regs_r: 255,
            c_num_regs_f: 255,
            startpoints: HashSet::new(),
            alllabels: HashSet::new(),
            resulttypes: HashMap::new(),
            index: 0,
            _called_from: None,
            _ssarepr: None,
            descrs: Vec::new(),
        }
    }

    /// RPython `jitcode.py:114-119` `def dump(self)`:
    ///
    /// ```python
    /// def dump(self):
    ///     if self._ssarepr is None:
    ///         return '<no dump available for %r>' % (self.name,)
    ///     else:
    ///         from rpython.jit.codewriter.format import format_assembler
    ///         return format_assembler(self._ssarepr)
    /// ```
    pub fn dump(&self) -> String {
        match &self._ssarepr {
            None => format!("<no dump available for {:?}>", self.name),
            Some(ssarepr) => format_assembler(ssarepr),
        }
    }

    /// RPython `jitcode.py:47-48` `def num_regs_i(self): return ord(self.c_num_regs_i)`.
    pub fn num_regs_i(&self) -> usize {
        self.c_num_regs_i as usize
    }

    /// RPython `jitcode.py:50-51` `def num_regs_r(self): return ord(self.c_num_regs_r)`.
    pub fn num_regs_r(&self) -> usize {
        self.c_num_regs_r as usize
    }

    /// RPython `jitcode.py:53-54` `def num_regs_f(self): return ord(self.c_num_regs_f)`.
    pub fn num_regs_f(&self) -> usize {
        self.c_num_regs_f as usize
    }

    /// RPython `jitcode.py:56-57` `def num_regs_and_consts_i(self):
    /// return ord(self.c_num_regs_i) + len(self.constants_i)`.
    pub fn num_regs_and_consts_i(&self) -> usize {
        self.num_regs_i() + self.constants_i.len()
    }

    /// RPython `jitcode.py:59-60` `def num_regs_and_consts_r(self):
    /// return ord(self.c_num_regs_r) + len(self.constants_r)`.
    pub fn num_regs_and_consts_r(&self) -> usize {
        self.num_regs_r() + self.constants_r.len()
    }

    /// RPython `jitcode.py:62-63` `def num_regs_and_consts_f(self):
    /// return ord(self.c_num_regs_f) + len(self.constants_f)`.
    pub fn num_regs_and_consts_f(&self) -> usize {
        self.num_regs_f() + self.constants_f.len()
    }

    /// RPython `jitcode.py:102-112` `def follow_jump(self, position)`:
    /// "Assuming that 'position' points just after a bytecode instruction
    /// that ends with a label, follow that label."
    ///
    /// ```python
    /// def follow_jump(self, position):
    ///     code = self.code
    ///     position -= 2
    ///     assert position >= 0
    ///     labelvalue = ord(code[position]) | (ord(code[position+1])<<8)
    ///     assert labelvalue < len(code)
    ///     return labelvalue
    /// ```
    pub fn follow_jump(&self, position: usize) -> usize {
        let position = position - 2;
        let labelvalue = (self.code[position] as usize) | ((self.code[position + 1] as usize) << 8);
        assert!(labelvalue < self.code.len(), "follow_jump out of range");
        labelvalue
    }

    /// RPython `jitcode.py:82-93` `get_live_vars_info(pc, op_live)`:
    ///
    /// ```python
    /// def get_live_vars_info(self, pc, op_live):
    ///     # either this, or the previous instruction must be -live-
    ///     if not we_are_translated():
    ///         assert pc in self._startpoints
    ///     if ord(self.code[pc]) != op_live:
    ///         pc -= OFFSET_SIZE + 1
    ///         if not we_are_translated():
    ///             assert pc in self._startpoints
    ///         if ord(self.code[pc]) != op_live:
    ///             self._missing_liveness(pc)
    ///     return decode_offset(self.code, pc + 1)
    /// ```
    ///
    /// `op_live` is the runtime opcode byte for `live/` (assigned by the
    /// blackhole interpreter at `setup_insns` time, RPython
    /// `blackhole.py:72`). The result is the offset into the metainterp's
    /// `all_liveness` table.
    pub fn get_live_vars_info(&self, pc: usize, op_live: u8) -> usize {
        debug_assert!(self.startpoints.contains(&pc), "pc not in startpoints");
        let mut pc = pc;
        if self.code[pc] != op_live {
            pc -= crate::liveness::OFFSET_SIZE + 1;
            debug_assert!(self.startpoints.contains(&pc), "pc not in startpoints");
            if self.code[pc] != op_live {
                self.missing_liveness(pc);
            }
        }
        crate::liveness::decode_offset(&self.code, pc + 1)
    }

    /// RPython `jitcode.py:95-100` `_missing_liveness(self, pc)`:
    ///
    /// ```python
    /// def _missing_liveness(self, pc):
    ///     msg = "missing liveness[%d] in %s" % (pc, self.name)
    ///     if we_are_translated():
    ///         print(msg)
    ///         raise AssertionError
    ///     raise MissingLiveness(...)
    /// ```
    fn missing_liveness(&self, pc: usize) -> ! {
        panic!("missing liveness[{pc}] in {}", self.name);
    }
}

// RPython `jitcode.py:121-122` `def __repr__(self): return '<JitCode %r>' % self.name`.
impl std::fmt::Display for JitCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<JitCode {:?}>", self.name)
    }
}

/// RPython `jitcode.py:146-167` module-level `enumerate_vars(offset,
/// all_liveness, callback_i, callback_r, callback_f, spec)`:
///
/// ```python
/// @specialize.arg(5)
/// def enumerate_vars(offset, all_liveness, callback_i, callback_r, callback_f, spec):
///     length_i = ord(all_liveness[offset])
///     length_r = ord(all_liveness[offset + 1])
///     length_f = ord(all_liveness[offset + 2])
///     offset += 3
///     if length_i:
///         it = LivenessIterator(offset, length_i, all_liveness)
///         for index in it: callback_i(index)
///         offset = it.offset
///     if length_r:
///         it = LivenessIterator(offset, length_r, all_liveness)
///         for index in it: callback_r(index)
///         offset = it.offset
///     if length_f:
///         it = LivenessIterator(offset, length_f, all_liveness)
///         for index in it: callback_f(index)
/// ```
///
/// Reads the `[len_i][len_r][len_f]` header at `offset`, then walks the
/// three packed bitsets (int, ref, float) via `LivenessIterator`, invoking
/// the matching callback for each live register index.
///
/// RPython places this in `rpython/jit/codewriter/jitcode.py` (not in
/// metainterp). majit follows the same module placement.
pub fn enumerate_vars(
    mut offset: usize,
    all_liveness: &[u8],
    mut callback_i: impl FnMut(u32),
    mut callback_r: impl FnMut(u32),
    mut callback_f: impl FnMut(u32),
) {
    use crate::liveness::LivenessIterator;
    // jitcode.py:149-151
    let length_i = all_liveness[offset] as u32;
    let length_r = all_liveness[offset + 1] as u32;
    let length_f = all_liveness[offset + 2] as u32;
    // jitcode.py:152
    offset += 3;
    // jitcode.py:153-157
    if length_i != 0 {
        let mut it = LivenessIterator::new(offset, length_i, all_liveness);
        for index in &mut it {
            callback_i(index);
        }
        offset = it.offset;
    }
    // jitcode.py:158-162
    if length_r != 0 {
        let mut it = LivenessIterator::new(offset, length_r, all_liveness);
        for index in &mut it {
            callback_r(index);
        }
        offset = it.offset;
    }
    // jitcode.py:163-166
    if length_f != 0 {
        let mut it = LivenessIterator::new(offset, length_f, all_liveness);
        for index in &mut it {
            callback_f(index);
        }
    }
}

/// RPython `jitcode.py:127-128` `class MissingLiveness(Exception): pass`.
///
/// Raised by `JitCode::get_live_vars_info` when a `-live-` op is missing
/// at the expected PC. Currently we panic instead of returning a typed
/// error since pyre's blackhole has no exception-based error path yet.
pub struct MissingLiveness {
    pub message: String,
}

/// RPython `jitcode.py:131-143` `class SwitchDictDescr(AbstractDescr)`:
///
/// ```python
/// class SwitchDictDescr(AbstractDescr):
///     "Get a 'dict' attribute mapping integer values to bytecode positions."
///
///     def attach(self, as_dict):
///         self.dict = as_dict
///         self.const_keys_in_order = map(ConstInt, sorted(as_dict.keys()))
///
///     def __repr__(self):
///         dict = getattr(self, 'dict', '?')
///         return '<SwitchDictDescr %s>' % (dict,)
///
///     def _clone_if_mutable(self):
///         raise NotImplementedError
/// ```
///
/// Used by the assembler to encode `switch` ops as a side-table mapping
/// integer values to bytecode positions. Currently a placeholder — pyre
/// has no `switch` op users yet, but the type lives here so the
/// codewriter::jitcode module shape stays parity-aligned with RPython.
#[derive(Debug, Clone, Default)]
pub struct SwitchDictDescr {
    /// RPython `attach`: integer key → bytecode position map.
    pub dict: std::collections::HashMap<i64, usize>,
    /// RPython `attach`: sorted ConstInt keys for replay/serialization.
    pub const_keys_in_order: Vec<i64>,
}

impl SwitchDictDescr {
    /// RPython `jitcode.py:134-136` `def attach(self, as_dict)`.
    pub fn attach(&mut self, as_dict: std::collections::HashMap<i64, usize>) {
        let mut keys: Vec<i64> = as_dict.keys().copied().collect();
        keys.sort();
        self.const_keys_in_order = keys;
        self.dict = as_dict;
    }
}

impl std::fmt::Display for SwitchDictDescr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "<SwitchDictDescr {:?}>", self.dict)
    }
}

/// RPython `history.py:AbstractDescr` — base class for all descriptor
/// objects stored in the assembler's `descrs` list. Read at runtime via
/// 'd'/'j' argcodes in the blackhole interpreter.
///
/// RPython uses a class hierarchy (`FieldDescr`, `ArrayDescr`, `CallDescr`,
/// `JitCode(AbstractDescr)`, `SwitchDictDescr`). pyre uses an enum to
/// represent the same heterogeneous list, shared between the codewriter
/// assembler and the metainterp blackhole.
#[derive(Debug, Clone)]
pub enum BhDescr {
    /// Field descriptor: for getfield/setfield.
    /// RPython: `FieldDescr(AbstractDescr)` — carries `offset`, `field_size`.
    /// `name` + `owner` identify the field for runtime offset resolution.
    /// `offset` is populated when known (0 = unresolved placeholder).
    Field {
        offset: usize,
        name: String,
        owner: String,
    },
    /// Array descriptor: for getarrayitem/setarrayitem/arraylen.
    /// RPython: `ArrayDescr` with `itemsize`, `basesize` attributes.
    /// `itemsize` is populated when known (8 = default placeholder).
    Array { itemsize: usize },
    /// Call descriptor: for residual_call. Carries calling convention.
    /// RPython: `CallDescr`.
    Call,
    /// JitCode descriptor: for inline_call_*.
    /// RPython: `JitCode(AbstractDescr)` — carries `fnaddr` + `calldescr`.
    /// `jitcode_index` indexes into `all_jitcodes[]` (set by CodeWriter).
    /// `fnaddr` is resolved at runtime from the callee's function address.
    JitCode {
        /// Index into all_jitcodes[]. Used by the blackhole to find the
        /// callee's bytecode for frame-chain push.
        jitcode_index: usize,
        /// Function address for cpu.bh_call_*. Resolved at runtime.
        fnaddr: i64,
    },
    /// SwitchDictDescr: maps int values to bytecode positions.
    Switch {
        dict: std::collections::HashMap<i64, usize>,
    },
    /// Virtualizable field descriptor: index into VirtualizableInfo.static_fields.
    /// NOT a byte offset — the blackhole resolves it via `vinfo.static_fields[index].offset`.
    VableField { index: usize },
    /// Virtualizable array descriptor: index into VirtualizableInfo.array_fields.
    VableArray { index: usize },
}

impl BhDescr {
    /// Extract byte offset for field/array operations (FieldDescr/ArrayDescr).
    /// Panics on VableField/VableArray — those must use `as_vable_field_index`.
    pub fn as_offset(&self) -> usize {
        match self {
            BhDescr::Field { offset, .. } => *offset,
            BhDescr::Array { itemsize } => *itemsize,
            _ => panic!("BhDescr::as_offset called on {:?}", self),
        }
    }

    /// Get field name (for runtime offset resolution).
    pub fn field_name(&self) -> &str {
        match self {
            BhDescr::Field { name, .. } => name,
            _ => panic!("BhDescr::field_name called on {:?}", self),
        }
    }

    /// Get field owner type name.
    pub fn field_owner(&self) -> &str {
        match self {
            BhDescr::Field { owner, .. } => owner,
            _ => panic!("BhDescr::field_owner called on {:?}", self),
        }
    }

    /// Extract virtualizable field index.
    pub fn as_vable_field_index(&self) -> usize {
        match self {
            BhDescr::VableField { index } => *index,
            _ => panic!("BhDescr::as_vable_field_index called on {:?}", self),
        }
    }

    /// Extract virtualizable array index.
    pub fn as_vable_array_index(&self) -> usize {
        match self {
            BhDescr::VableArray { index } => *index,
            _ => panic!("BhDescr::as_vable_array_index called on {:?}", self),
        }
    }

    /// Extract JitCode index for inline_call.
    pub fn as_jitcode_index(&self) -> usize {
        match self {
            BhDescr::JitCode { jitcode_index, .. } => *jitcode_index,
            _ => panic!("BhDescr::as_jitcode_index called on {:?}", self),
        }
    }

    /// Extract function address for inline_call cpu.bh_call_* fallback.
    pub fn as_jitcode_fnaddr(&self) -> i64 {
        match self {
            BhDescr::JitCode { fnaddr, .. } => *fnaddr,
            _ => 0,
        }
    }

    /// Lookup switch value → position.
    pub fn switch_lookup(&self, value: i64) -> Option<usize> {
        match self {
            BhDescr::Switch { dict } => dict.get(&value).copied(),
            _ => None,
        }
    }
}

/// RPython `format.py:12-80` `format_assembler(ssarepr)`.
///
/// Minimal port: formats each FlatOp in the SSARepr into human-readable
/// text. RPython uses this for debug output and testing.
///
/// ```python
/// def format_assembler(ssarepr):
///     """For testing: format a SSARepr as a multiline string."""
///     ...
///     return buf.getvalue()
/// ```
pub fn format_assembler(ssarepr: &crate::passes::flatten::SSARepr) -> String {
    use crate::passes::flatten::FlatOp;
    use std::fmt::Write;

    let mut out = String::new();
    writeln!(out, "{}", ssarepr.name).ok();
    for op in &ssarepr.insns {
        match op {
            FlatOp::Label(label) => {
                writeln!(out, "L{}:", label.0).ok();
            }
            FlatOp::Live { live_values } => {
                let regs: Vec<String> = live_values.iter().map(|v| format!("%i{}", v.0)).collect();
                writeln!(out, "  -live- {}", regs.join(", ")).ok();
            }
            FlatOp::Unreachable => {
                writeln!(out, "  ---").ok();
            }
            FlatOp::Op(space_op) => {
                let result = space_op
                    .result
                    .map(|v| format!(" -> %i{}", v.0))
                    .unwrap_or_default();
                writeln!(out, "  {:?}{result}", space_op.kind).ok();
            }
            FlatOp::Jump(label) => {
                writeln!(out, "  goto L{}", label.0).ok();
            }
            FlatOp::GotoIfNot { cond, target } => {
                writeln!(out, "  goto_if_not %i{}, L{}", cond.0, target.0).ok();
            }
            FlatOp::Move { dst, src } => {
                writeln!(out, "  %i{} = %i{}", dst.0, src.0).ok();
            }
        }
    }
    out
}
