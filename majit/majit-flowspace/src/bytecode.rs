//! Bytecode handling classes and functions for use by the flow space.
//!
//! RPython basis: `rpython/flowspace/bytecode.py`.
//!
//! Phase 2 F2.1 landed the bare re-export. F2.2 added the `HostCode`
//! wrapper and its `cpython_code_signature` companion. F2.3 (this
//! commit) adds `BytecodeCorruption` and the `HostCode::read` decoder.
//! Upstream has a single `HostCode.read`; the roadmap's companion name
//! `next_bytecode_instruction` has no upstream symbol — the `read`
//! method IS the instruction walker.
//!
//! ## Deviation from the plan (parity rule #1)
//!
//! The roadmap asked for `pyre_interpreter::bytecode::{OpCode, Instruction,
//! CodeObject}`. There is no standalone `OpCode` type in pyre-interpreter:
//! RustPython's `Instruction` enum already carries the opcode tag inline
//! (see `rustpython-compiler-core::bytecode::Instruction`). Splitting a
//! parallel `OpCode` out would introduce a majit-local alias with no
//! upstream basis, so we re-export `Instruction` on its own — each variant
//! IS an opcode.
//!
//! Upstream `bytecode.py` solves the same problem by importing
//! `HAVE_ARGUMENT` / `EXTENDED_ARG` from the stdlib `opcode` module and
//! walking raw bytes; the CPython-3.14 equivalent is the `Instruction`
//! enum's own `decode` path, already implemented inside the RustPython
//! compiler core and driven through `pyre_interpreter::bytecode`.

pub use pyre_interpreter::bytecode::{
    BinaryOperator, CodeFlags, CodeObject, CodeUnit, ComparisonOperator, ConstantData,
    ExceptionTableEntry, Instruction, MakeFunctionFlag, OpArg, OpArgState,
};

use crate::argument::Signature;

/// Error raised when the bytecode stream cannot be decoded.
///
/// RPython basis: `rpython/flowspace/bytecode.py:BytecodeCorruption`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BytecodeCorruption(pub String);

impl BytecodeCorruption {
    pub fn new<S: Into<String>>(msg: S) -> Self {
        Self(msg.into())
    }
}

impl core::fmt::Display for BytecodeCorruption {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "BytecodeCorruption: {}", self.0)
    }
}

impl std::error::Error for BytecodeCorruption {}

/// RPython: `bytecode.py:CO_GENERATOR` (0x0020).
pub const CO_GENERATOR: u32 = 0x0020;
/// RPython: `bytecode.py:CO_VARARGS` (0x0004).
pub const CO_VARARGS: u32 = 0x0004;
/// RPython: `bytecode.py:CO_VARKEYWORDS` (0x0008).
pub const CO_VARKEYWORDS: u32 = 0x0008;

/// Compute `([list-of-arg-names], vararg-name-or-None, kwarg-name-or-None)`.
///
/// RPython basis: `rpython/flowspace/bytecode.py:cpython_code_signature`.
pub fn cpython_code_signature(code: &HostCode) -> Signature {
    let mut argcount = code.co_argcount as usize;
    let argnames: Vec<String> = code.co_varnames.iter().take(argcount).cloned().collect();
    let varargname = if code.co_flags & CO_VARARGS != 0 {
        let name = code.co_varnames[argcount].clone();
        argcount += 1;
        Some(name)
    } else {
        None
    };
    let kwargname = if code.co_flags & CO_VARKEYWORDS != 0 {
        let name = code.co_varnames[argcount].clone();
        Some(name)
    } else {
        None
    };
    Signature::new(argnames, varargname, kwargname)
}

/// A wrapper around a native code object of the host interpreter.
///
/// RPython basis: `rpython/flowspace/bytecode.py:HostCode`.
///
/// Fields copy the upstream `co_*` attribute names verbatim. The
/// `opnames` class-level table has no Rust analogue — Phase 3
/// `flowcontext.rs` dispatches on the `Instruction` enum directly,
/// which already carries the opcode tag. The upstream
/// `HostCode.opnames` string list is a stdlib-`opcode` module artifact
/// with no independent meaning.
///
/// ### Field-shape deviation (parity rule #1)
///
/// RPython's `__init__` stores `code` as the raw `co_code` byte string
/// because PyPy-2.7 carried a pre-wordcode bytecode format. The CPython
/// 3.14 equivalent in RustPython is `CodeUnits` (a
/// two-byte-per-instruction container); we store that directly as
/// `co_code` to avoid a second copy into raw bytes. The RPython walker
/// (F2.3 port of `HostCode.read`) becomes a `CodeUnits` iterator
/// instead of a byte-level `ord(co_code[offset])` loop.
#[derive(Clone, Debug)]
pub struct HostCode {
    pub co_argcount: u32,
    pub co_nlocals: u32,
    pub co_stacksize: u32,
    pub co_flags: u32,
    pub co_code: pyre_interpreter::bytecode::CodeUnits,
    pub consts: Vec<ConstantData>,
    pub names: Vec<String>,
    pub co_varnames: Vec<String>,
    pub co_freevars: Vec<String>,
    pub co_filename: String,
    pub co_name: String,
    pub co_firstlineno: u32,
    pub co_lnotab: Vec<u8>,
    pub exceptiontable: Box<[u8]>,
    pub signature: Signature,
}

impl PartialEq for HostCode {
    fn eq(&self, other: &Self) -> bool {
        self.co_argcount == other.co_argcount
            && self.co_nlocals == other.co_nlocals
            && self.co_stacksize == other.co_stacksize
            && self.co_flags == other.co_flags
            && self.co_code.original_bytes() == other.co_code.original_bytes()
            && self.consts == other.consts
            && self.names == other.names
            && self.co_varnames == other.co_varnames
            && self.co_freevars == other.co_freevars
            && self.co_filename == other.co_filename
            && self.co_name == other.co_name
            && self.co_firstlineno == other.co_firstlineno
            && self.co_lnotab == other.co_lnotab
            && self.exceptiontable == other.exceptiontable
            && self.signature == other.signature
    }
}

impl Eq for HostCode {}

impl HostCode {
    /// RPython: `HostCode.__init__`.
    ///
    /// Structural note: RPython takes 13 positional arguments matching
    /// CPython's code-object constructor order. We mirror that order here
    /// so call-site translations stay line-by-line, even though a
    /// builder struct would be more idiomatic.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        argcount: u32,
        nlocals: u32,
        stacksize: u32,
        flags: u32,
        code: pyre_interpreter::bytecode::CodeUnits,
        consts: Vec<ConstantData>,
        names: Vec<String>,
        varnames: Vec<String>,
        filename: String,
        name: String,
        firstlineno: u32,
        lnotab: Vec<u8>,
        freevars: Vec<String>,
        exceptiontable: Box<[u8]>,
    ) -> Self {
        assert!(nlocals as i64 >= 0, "nlocals must be non-negative");
        let mut host = Self {
            co_argcount: argcount,
            co_nlocals: nlocals,
            co_stacksize: stacksize,
            co_flags: flags,
            co_code: code,
            consts,
            names,
            co_varnames: varnames,
            co_freevars: freevars,
            co_filename: filename,
            co_name: name,
            co_firstlineno: firstlineno,
            co_lnotab: lnotab,
            exceptiontable,
            // Temporary placeholder until we fill it below — RPython
            // assigns inside the same __init__ call.
            signature: Signature::new(Vec::new(), None, None),
        };
        host.signature = cpython_code_signature(&host);
        host
    }

    /// Initialize the `HostCode` from a real (CPython) code object.
    ///
    /// RPython basis: `HostCode._from_code`.
    pub fn from_code(code: &CodeObject) -> Self {
        let nlocals = code
            .localspluskinds
            .len()
            .saturating_sub(code.cellvars.len())
            .saturating_sub(code.freevars.len()) as u32;
        let firstlineno = code.first_line_number.map_or(0, |n| n.get() as u32);
        Self::new(
            code.arg_count,
            nlocals,
            code.max_stackdepth,
            code.flags.bits(),
            code.instructions.clone(),
            code.constants.iter().cloned().collect(),
            code.names.iter().cloned().collect(),
            code.varnames.iter().cloned().collect(),
            code.source_path.clone(),
            code.obj_name.clone(),
            firstlineno,
            code.linetable.iter().copied().collect(),
            code.freevars.iter().cloned().collect(),
            code.exceptiontable.clone(),
        )
    }

    /// Total number of arguments passed into the frame, including
    /// `*vararg` and `**varkwarg` if they exist.
    ///
    /// RPython basis: `HostCode.formalargcount`.
    pub fn formalargcount(&self) -> usize {
        self.signature.scope_length()
    }

    /// RPython: `HostCode.is_generator`.
    pub fn is_generator(&self) -> bool {
        self.co_flags & CO_GENERATOR != 0
    }

    /// Find the exception-table handler covering the given byte offset.
    ///
    /// RustPython stores exception-table offsets in code-unit indices.
    /// `HostCode.read` and flowspace operate in byte offsets for
    /// parity with RPython's `co_code` walk, so the adapter converts
    /// to/from byte offsets here.
    pub fn find_exception_handler(&self, offset: u32) -> Option<ExceptionTableEntry> {
        if !offset.is_multiple_of(2) {
            return None;
        }
        let unit_offset = offset / 2;
        pyre_interpreter::bytecode::find_exception_handler(&self.exceptiontable, unit_offset).map(
            |entry| ExceptionTableEntry {
                start: entry.start * 2,
                end: entry.end * 2,
                target: entry.target * 2,
                depth: entry.depth,
                push_lasti: entry.push_lasti,
            },
        )
    }

    /// Decode the entire exception table into byte-offset entries.
    pub fn decode_exception_table(&self) -> Vec<ExceptionTableEntry> {
        pyre_interpreter::bytecode::decode_exception_table(&self.exceptiontable)
            .into_iter()
            .map(|entry| ExceptionTableEntry {
                start: entry.start * 2,
                end: entry.end * 2,
                target: entry.target * 2,
                depth: entry.depth,
                push_lasti: entry.push_lasti,
            })
            .collect()
    }

    /// Decode the instruction starting at byte position `offset`.
    ///
    /// Returns `(next_offset, op, oparg)` where `next_offset` is the byte
    /// position of the following instruction, `op` is the decoded opcode,
    /// and `oparg` is the concatenated argument (with any preceding
    /// `EXTENDED_ARG` already folded in).
    ///
    /// RPython basis: `rpython/flowspace/bytecode.py:HostCode.read`.
    ///
    /// ### Deviation from RPython (parity rule #1)
    ///
    /// RPython returns the opcode **name** as a string (`opname`); we
    /// return the typed `Instruction` enum because CPython 3.14's
    /// wordcode format is enum-typed at the RustPython layer. The
    /// string name was used upstream to drive attribute lookup
    /// (`self.opcode_<NAME>(oparg)`), which is not idiomatic in Rust
    /// — the Phase 3 `flowcontext.rs` port matches on the enum
    /// directly. No information is lost.
    pub fn read(&self, offset: u32) -> Result<(u32, Instruction, u32), BytecodeCorruption> {
        let byte_offset = offset as usize;
        if !byte_offset.is_multiple_of(2) {
            return Err(BytecodeCorruption::new(format!(
                "unaligned bytecode offset {offset}"
            )));
        }
        let units: &[CodeUnit] = &self.co_code;
        let mut idx = byte_offset / 2;
        if idx >= units.len() {
            return Err(BytecodeCorruption::new(format!(
                "bytecode offset {offset} out of range (codeunits={})",
                units.len()
            )));
        }
        let mut arg_state = OpArgState::default();
        loop {
            let instruction_index = idx as u32;
            let unit = units[idx];
            let (op, oparg) = arg_state.get(unit);
            idx += 1;
            if !matches!(op, Instruction::ExtendedArg) {
                let next_offset = (idx * 2) as u32;
                let arg = absolutize_jump_target(instruction_index, &op, u32::from(oparg))
                    .unwrap_or_else(|| u32::from(oparg));
                return Ok((next_offset, op, arg));
            }
            if idx >= units.len() {
                return Err(BytecodeCorruption::new(
                    "ExtendedArg at end of bytecode stream",
                ));
            }
        }
    }
}

fn absolutize_jump_target(offset_units: u32, op: &Instruction, arg: u32) -> Option<u32> {
    let after = offset_units
        .checked_add(1)?
        .checked_add(op.cache_entries() as u32)?;
    let target_units = match op {
        Instruction::JumpForward { .. }
        | Instruction::PopJumpIfFalse { .. }
        | Instruction::PopJumpIfTrue { .. }
        | Instruction::PopJumpIfNone { .. }
        | Instruction::PopJumpIfNotNone { .. }
        | Instruction::ForIter { .. }
        | Instruction::Send { .. } => after.checked_add(arg)?,
        Instruction::JumpBackward { .. } | Instruction::JumpBackwardNoInterrupt { .. } => {
            after.checked_sub(arg)?
        }
        _ => return None,
    };
    target_units.checked_mul(2)
}

#[cfg(test)]
mod test {
    use super::*;
    use pyre_interpreter::compile::{self, Mode};

    fn compile_source(src: &str) -> CodeObject {
        compile::compile_source(src, Mode::Exec).expect("compile should succeed")
    }

    // RPython basis: test_objspace.py shapes these round-trips by calling
    // `build_flow_graph`. We pin the minimum field-copy invariants here
    // so F2.3 / F3 can rely on `HostCode::from_code` preserving upstream
    // attribute semantics. See F2.4 (upstream test ports) for the full
    // bytecode-shape coverage.

    #[test]
    fn from_code_preserves_basic_fields() {
        let code = compile_source("def f(x, y):\n    return x + y\n");
        // Extract the inner function's code object.
        let inner = code
            .constants
            .iter()
            .find_map(|c| match c {
                ConstantData::Code { code } => Some(&**code),
                _ => None,
            })
            .expect("function body should be a code constant");
        let host = HostCode::from_code(inner);
        assert_eq!(host.co_argcount, 2);
        assert_eq!(host.co_name, "f");
        assert_eq!(host.co_varnames[0], "x");
        assert_eq!(host.co_varnames[1], "y");
        assert_eq!(host.formalargcount(), 2);
        assert!(!host.is_generator());
        assert_eq!(host.signature.argnames, vec!["x".to_string(), "y".into()]);
        assert!(host.signature.varargname.is_none());
        assert!(host.signature.kwargname.is_none());
    }

    #[test]
    fn signature_tracks_varargs_and_kwargs() {
        let code = compile_source("def f(a, *args, **kw):\n    return a\n");
        let inner = code
            .constants
            .iter()
            .find_map(|c| match c {
                ConstantData::Code { code } => Some(&**code),
                _ => None,
            })
            .expect("function body");
        let host = HostCode::from_code(inner);
        assert_eq!(host.signature.argnames, vec!["a".to_string()]);
        assert_eq!(host.signature.varargname.as_deref(), Some("args"));
        assert_eq!(host.signature.kwargname.as_deref(), Some("kw"));
        assert_eq!(host.formalargcount(), 3);
    }

    #[test]
    fn is_generator_detects_yield() {
        let code = compile_source("def g():\n    yield 1\n");
        let inner = code
            .constants
            .iter()
            .find_map(|c| match c {
                ConstantData::Code { code } => Some(&**code),
                _ => None,
            })
            .expect("function body");
        let host = HostCode::from_code(inner);
        assert!(host.is_generator());
    }

    // RPython basis: bytecode.py:HostCode.read. Upstream has no standalone
    // test for `read`; the shape is verified indirectly through
    // test_objspace.py:build_flow_graph. These pin the wordcode-level
    // decoder directly so F3 (flowcontext.py port) can rely on the
    // `(next_offset, Instruction, oparg)` shape.

    #[test]
    fn read_walks_each_instruction_in_order() {
        let code = compile_source("def f(x):\n    return x\n");
        let inner = code
            .constants
            .iter()
            .find_map(|c| match c {
                ConstantData::Code { code } => Some(&**code),
                _ => None,
            })
            .expect("function body");
        let host = HostCode::from_code(inner);
        let mut offset = 0u32;
        let mut seen = Vec::new();
        let total = (host.co_code.len() * 2) as u32;
        while offset < total {
            let (next, op, _arg) = host.read(offset).expect("decoder");
            assert!(next > offset, "read must advance");
            seen.push(op);
            offset = next;
        }
        assert!(!seen.is_empty());
        // Last instruction of a `return x` body is ReturnValue on CPython
        // 3.14 wordcode.
        assert!(
            matches!(seen.last(), Some(Instruction::ReturnValue)),
            "expected terminal ReturnValue, got {:?}",
            seen.last()
        );
    }

    #[test]
    fn read_rejects_unaligned_offset() {
        let code = compile_source("def f():\n    return 1\n");
        let inner = code
            .constants
            .iter()
            .find_map(|c| match c {
                ConstantData::Code { code } => Some(&**code),
                _ => None,
            })
            .expect("function body");
        let host = HostCode::from_code(inner);
        let err = host.read(1).expect_err("unaligned offset must fail");
        assert!(
            err.to_string().contains("unaligned"),
            "expected unaligned diagnostic, got {err}"
        );
    }

    #[test]
    fn read_rejects_out_of_range_offset() {
        let code = compile_source("def f():\n    return 1\n");
        let inner = code
            .constants
            .iter()
            .find_map(|c| match c {
                ConstantData::Code { code } => Some(&**code),
                _ => None,
            })
            .expect("function body");
        let host = HostCode::from_code(inner);
        let past_end = (host.co_code.len() * 2) as u32;
        assert!(host.read(past_end).is_err());
    }

    #[test]
    fn read_absolutizes_relative_conditional_jump_targets() {
        let code = compile_source("def f(x):\n    if x:\n        return 1\n    return 0\n");
        let inner = code
            .constants
            .iter()
            .find_map(|c| match c {
                ConstantData::Code { code } => Some(&**code),
                _ => None,
            })
            .expect("function body");
        let host = HostCode::from_code(inner);
        let total = (host.co_code.len() * 2) as u32;
        let mut offset = 0u32;
        let mut saw_branch = false;
        while offset < total {
            let (next, op, arg) = host.read(offset).expect("decoder");
            if matches!(
                op,
                Instruction::PopJumpIfFalse { .. }
                    | Instruction::PopJumpIfTrue { .. }
                    | Instruction::PopJumpIfNone { .. }
                    | Instruction::PopJumpIfNotNone { .. }
            ) {
                assert!(arg > next, "relative jump target should be absolutized");
                saw_branch = true;
                break;
            }
            offset = next;
        }
        assert!(
            saw_branch,
            "expected a conditional jump in the function body"
        );
    }
}
