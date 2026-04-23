//! Runtime access to the build-time `pipeline.jitcodes` table.
//!
//! RPython: `MetaInterpStaticData.jitcodes` (warmspot.py:281-282) — the list
//! of `JitCode` objects produced by `CodeWriter.make_jitcodes()`
//! (codewriter.py:89). In RPython this list is passed by reference from
//! `CallControl.jitcodes` directly into `MetaInterpStaticData`; the two
//! stores reference the same Python objects.
//!
//! majit's build-time side lives in `majit_translate::jitcode::JitCode`
//! (serde-serializable, emitted by `build.rs` into
//! `$OUT_DIR/opcode_jitcodes.bin`). This module deserializes that blob once
//! on first access and hands out `Arc<JitCode>` shells. The `PipelineOpcodeArm
//! .entry_jitcode_index` field (already present in the canonical pipeline
//! result) indexes into this table.
//!
//! No side-table serialization: the only persisted collection is
//! `pipeline.jitcodes`, in allocation order, matching RPython's single-store
//! model (`feedback_single_jitcodes_store`).
//!
//! Phase D-1 Step 2 of the eval-loop automation plan.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use majit_translate::jitcode::{BhDescr, JitCode};
use majit_translate::opcode_dispatch::PipelineOpcodeArm;
use majit_translate::{CallPath, OpcodeDispatchSelector};
use pyre_interpreter::bytecode::Instruction;

/// Deserialized `pipeline.jitcodes` — RPython `all_jitcodes[]` from
/// codewriter.py:89.  Dense: `ALL_JITCODES[i].index == i` (RPython
/// codewriter.py:80 invariant, preserved by
/// `collect_jitcodes_in_alloc_order`). Use `get_jitcode_by_index` or
/// direct indexing for lookup by `entry_jitcode_index`.
static ALL_JITCODES: LazyLock<Vec<Arc<JitCode>>> = LazyLock::new(|| {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/opcode_jitcodes.bin"));
    let vec: Vec<Arc<JitCode>> = bincode::deserialize(BYTES).unwrap_or_else(|e| {
        panic!(
            "pyre-jit-trace: failed to deserialize opcode_jitcodes.bin \
             ({} bytes): {e}",
            BYTES.len(),
        )
    });
    // RPython codewriter.py:80: `all_jitcodes[jitcode.index] is jitcode`.
    // Check at load time so any regression in
    // `collect_jitcodes_in_alloc_order` is caught immediately.
    for (i, jc) in vec.iter().enumerate() {
        assert_eq!(
            jc.index(),
            i,
            "pyre-jit-trace: jitcode[{i}].index = {} (expected {i}); \
             RPython invariant `all_jitcodes[i].index == i` broken",
            jc.index(),
        );
    }
    vec
});

/// Deserialized `pipeline.opcode_dispatch` — the arm table. Each entry
/// carries `arm_id`, `selector`, and `entry_jitcode_index` (logical index
/// into `ALL_JITCODES` via `.index`).
static ALL_OPCODE_ARMS: LazyLock<Vec<PipelineOpcodeArm>> = LazyLock::new(|| {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/opcode_dispatch.bin"));
    bincode::deserialize(BYTES).unwrap_or_else(|e| {
        panic!(
            "pyre-jit-trace: failed to deserialize opcode_dispatch.bin \
             ({} bytes): {e}",
            BYTES.len(),
        )
    })
});

/// RPython: `metainterp_sd.jitcodes` — full all_jitcodes table.
pub fn all_jitcodes() -> &'static [Arc<JitCode>] {
    &ALL_JITCODES
}

/// RPython: `metainterp_sd.jitcodes[index]` where `index == jitcode.index`.
///
/// Build-time `PipelineOpcodeArm.entry_jitcode_index` is this logical
/// index. The dense invariant (`ALL_JITCODES[i].index == i`) is asserted
/// at load time, so direct vec indexing is correct.
pub fn get_jitcode_by_index(index: usize) -> Option<Arc<JitCode>> {
    ALL_JITCODES.get(index).cloned()
}

/// RPython: opcode dispatch arm table (analogue of PyPy's per-opcode
/// Python methods). One `PipelineOpcodeArm` per Rust `match` arm.
pub fn all_opcode_arms() -> &'static [PipelineOpcodeArm] {
    &ALL_OPCODE_ARMS
}

/// Returns the arm with the given `arm_id`.
pub fn get_arm(arm_id: usize) -> Option<&'static PipelineOpcodeArm> {
    ALL_OPCODE_ARMS.iter().find(|a| a.arm_id == arm_id)
}

/// Convenience: resolve `arm_id` → entry jitcode. Returns `None` if arm
/// doesn't exist or the arm has no body graph.
pub fn jitcode_for_arm(arm_id: usize) -> Option<Arc<JitCode>> {
    let arm = get_arm(arm_id)?;
    let idx = arm.entry_jitcode_index?;
    get_jitcode_by_index(idx)
}

/// Variant-name → arm_id index built from `ALL_OPCODE_ARMS` selectors.
/// Multi-pattern `Instruction::A | Instruction::B` arms expand each
/// variant to the same arm_id, matching the RPython model where a
/// single Python method is registered under each dispatched opcode.
///
/// Keyed by the bare variant name (e.g. `"PopTop"`, `"LoadFast"`) — the
/// last segment of `OpcodeDispatchSelector::Path.segments`. Derived Debug
/// on `Instruction` yields the same variant prefix, so
/// `arm_id_for_instruction` uses this to resolve an `Instruction` value
/// at runtime.
static ARM_ID_BY_VARIANT: LazyLock<HashMap<String, usize>> = LazyLock::new(|| {
    let mut map = HashMap::new();
    for arm in all_opcode_arms() {
        collect_variant_names(&arm.selector, arm.arm_id, &mut map);
    }
    map
});

fn collect_variant_names(
    sel: &OpcodeDispatchSelector,
    arm_id: usize,
    out: &mut HashMap<String, usize>,
) {
    match sel {
        OpcodeDispatchSelector::Path(cp) => {
            if let Some(name) = variant_name_from_path(cp) {
                out.insert(name.to_string(), arm_id);
            }
        }
        OpcodeDispatchSelector::Or(cases) => {
            for c in cases {
                collect_variant_names(c, arm_id, out);
            }
        }
        OpcodeDispatchSelector::Wildcard | OpcodeDispatchSelector::Unsupported => {}
    }
}

fn variant_name_from_path(cp: &CallPath) -> Option<&str> {
    cp.segments.last().map(String::as_str)
}

/// Extracts the variant identifier from `format!("{instr:?}")`. The
/// derived `Debug` for an `Instruction` variant starts with the variant
/// name, optionally followed by ` { .. }` for struct variants or ` (..)`
/// for tuple variants.
fn extract_variant_name(instr_debug: &str) -> &str {
    instr_debug
        .split(|c: char| c.is_whitespace() || c == '(' || c == '{')
        .next()
        .unwrap_or(instr_debug)
}

/// Runtime variant name for `Instruction` — matches the last segment of
/// the build-time `OpcodeDispatchSelector::Path` emitted by the parser
/// (e.g. `Instruction::PopTop` → `"PopTop"`).
pub fn instruction_variant_name(instruction: &Instruction) -> String {
    extract_variant_name(&format!("{instruction:?}")).to_string()
}

/// Resolve an `Instruction` to its `PipelineOpcodeArm.arm_id`.
///
/// Returns `None` for variants not covered by any dispatch arm — either
/// because the parser emitted `Wildcard`/`Unsupported` for that arm, or
/// the variant has no match arm in `execute_opcode_step`.
pub fn arm_id_for_instruction(instruction: &Instruction) -> Option<usize> {
    ARM_ID_BY_VARIANT
        .get(&instruction_variant_name(instruction))
        .copied()
}

/// Resolve an `Instruction` directly to its entry jitcode. This is the
/// MIFrame-side entry for Phase D-2 shadow dispatch.
pub fn jitcode_for_instruction(instruction: &Instruction) -> Option<Arc<JitCode>> {
    jitcode_for_arm(arm_id_for_instruction(instruction)?)
}

/// Deserialized `pipeline.insns` — the opname → u8 table
/// `Assembler.write_insn` grew during assembly. `JitCode.code[i]` bytes
/// can be mapped back to opnames through the inverted view exposed by
/// `opname_for_byte`. Matches RPython `setup_insns(insns)` consumption
/// at pyjitpl.py:2227-2243.
static INSNS_OPNAME_TO_BYTE: LazyLock<HashMap<String, u8>> = LazyLock::new(|| {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/opcode_insns.bin"));
    bincode::deserialize(BYTES).unwrap_or_else(|e| {
        panic!(
            "pyre-jit-trace: failed to deserialize opcode_insns.bin \
             ({} bytes): {e}",
            BYTES.len(),
        )
    })
});

/// Inverted view: `u8` opcode byte → opname string. Built lazily on
/// first access from `INSNS_OPNAME_TO_BYTE`. Duplicate u8 values should
/// never occur (assembler.py assigns sequential ids), and we assert that
/// at load time.
static INSNS_BYTE_TO_OPNAME: LazyLock<HashMap<u8, String>> = LazyLock::new(|| {
    let mut map = HashMap::with_capacity(INSNS_OPNAME_TO_BYTE.len());
    for (name, &byte) in INSNS_OPNAME_TO_BYTE.iter() {
        assert!(
            map.insert(byte, name.clone()).is_none(),
            "duplicate opcode byte {byte} in pipeline.insns table"
        );
    }
    map
});

/// RPython `setup_insns(insns)` — full opname → opcode-byte table.
pub fn insns_opname_to_byte() -> &'static HashMap<String, u8> {
    &INSNS_OPNAME_TO_BYTE
}

/// Inverse lookup: `JitCode.code[i]` byte → opname. None for unknown
/// bytes (e.g. operand bytes, not opcode bytes).
pub fn opname_for_byte(byte: u8) -> Option<&'static str> {
    INSNS_BYTE_TO_OPNAME.get(&byte).map(String::as_str)
}

/// Inverse of `insns_opname_to_byte()` — full `u8 -> opname/argcodes` table.
pub fn insns_byte_to_opname() -> &'static HashMap<u8, String> {
    &INSNS_BYTE_TO_OPNAME
}

/// Deserialized `pipeline.descrs` — RPython `Assembler.descrs`
/// (assembler.py:23). Handed to `BlackholeInterpBuilder.setup_descrs`
/// at builder construction (blackhole.py:59 `self.setup_descrs(asm.descrs)`,
/// :102-103 `def setup_descrs(self, descrs): self.descrs = descrs`).
///
/// Each 'd'/'j' argcode in a `JitCode.code` byte stream is a 2-byte
/// little-endian index into this pool. The resolved `BhDescr` is what
/// every `bhimpl_*` handler reads for field offsets, call descriptors,
/// sub-JitCodes, and switch dicts.
static ALL_DESCRS: LazyLock<Vec<BhDescr>> = LazyLock::new(|| {
    const BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/opcode_descrs.bin"));
    bincode::deserialize(BYTES).unwrap_or_else(|e| {
        panic!(
            "pyre-jit-trace: failed to deserialize opcode_descrs.bin \
             ({} bytes): {e}",
            BYTES.len(),
        )
    })
});

/// RPython: `metainterp_sd.all_descrs` — full shared descr pool.
pub fn all_descrs() -> &'static [BhDescr] {
    &ALL_DESCRS
}

/// Build a `BlackholeInterpBuilder` pre-configured for this binary's
/// jitcodes.
///
/// RPython: `BlackholeInterpBuilder.__init__` (blackhole.py:55-61) runs
/// `setup_insns(asm.insns)` + `setup_descrs(asm.descrs)` immediately
/// and `setup_insns` (blackhole.py:66) resolves each opname via
/// `_get_method` eagerly, raising `AttributeError` if any `bhimpl_*` is
/// missing.
///
/// pyre mirrors that fail-fast contract: after `setup_insns` +
/// `wire_bhimpl_handlers`, assert that every opname in the insns table
/// has an explicit handler. If any remain unwired we panic here
/// instead of letting dispatch surface a confusing runtime error.
///
/// The shared descr pool is handed over via `setup_descrs` — same call
/// order as RPython. Every 'd'/'j' argcode in a `JitCode.code` byte
/// stream resolves against `builder.descrs[index]`, which matches the
/// RPython single-store model at `metainterp_sd.descrs`.
pub fn build_default_bh_builder() -> majit_metainterp::blackhole::BlackholeInterpBuilder {
    let mut builder = majit_metainterp::blackhole::BlackholeInterpBuilder::new();
    // blackhole.py:58-59 order: setup_insns, then setup_descrs.
    builder.setup_insns(insns_opname_to_byte());
    builder.setup_descrs(all_descrs().to_vec());
    majit_metainterp::blackhole::wire_bhimpl_handlers(&mut builder);
    let unwired = builder.unwired_opnames();
    if !unwired.is_empty() {
        panic!(
            "build_default_bh_builder: {} insns opnames have no bhimpl_* \
             handler (RPython blackhole.py:66 raises AttributeError here): \
             {:?}",
            unwired.len(),
            unwired,
        );
    }
    builder
}

/// Decoded one jitcode instruction. Mirrors the static slice that RPython
/// `BlackholeInterpBuilder._get_method` would walk over, without any
/// execution of `bhimpl_*`. Lifetime is tied to the `insns` table, so the
/// opname stays valid while the runtime is alive (`'static`).
///
/// RPython parity: `blackhole.py:105-232` `_get_method.handler` consumes
/// operand bytes per `argcodes` char; this struct captures the same byte
/// layout without executing.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct DecodedOp {
    /// The `opname/argcodes` key from the `insns` table.
    pub key: &'static str,
    /// `opname` part of `key` (before the `/`).
    pub opname: &'static str,
    /// `argcodes` part of `key` (after the `/`). Empty for `live/`.
    pub argcodes: &'static str,
    /// Position of the opcode byte in the jitcode.
    pub pc: usize,
    /// Position of the first byte *after* this instruction. `None` if the
    /// instruction reads a variable-length operand (`I`/`R`/`F`) that
    /// overflowed the code slice.
    pub next_pc: usize,
}

/// Statically walk one jitcode instruction starting at `pc`. Returns
/// `None` when `pc` is beyond `code.len()` or when the opcode byte at
/// `pc` is unknown to the `insns` table.
///
/// The `argcodes` char-byte mapping follows `blackhole.py:112-157`:
///
/// ```text
///   'i'|'c'|'r'|'f'  -> 1 byte (register index or signed const byte)
///   'L'              -> 2 bytes (unsigned label)
///   'd'|'j'          -> 2 bytes (descr index)
///   'I'|'R'|'F'      -> 1 + N bytes (N = first byte = list length)
///   '>' + ('i'|'r'|'f') -> 1 byte (result destination register)
/// ```
///
/// `live/` is special-cased to advance by `liveness::OFFSET_SIZE` per
/// `blackhole.py:1603-1605` (`bhimpl_live(pc): return pc + OFFSET_SIZE`).
pub fn decode_op_at(code: &[u8], pc: usize) -> Option<DecodedOp> {
    let opcode_byte = *code.get(pc)?;
    let key: &'static str = INSNS_BYTE_TO_OPNAME.get(&opcode_byte)?.as_str();
    let (opname, argcodes) = split_key(key);

    let mut cursor = pc + 1;
    if opname == "live" {
        // blackhole.py:1603-1605 bhimpl_live(pc): position += OFFSET_SIZE.
        // The `live/` key has empty argcodes so the generic walker would
        // advance 0 bytes, but dispatch skips 2 bytes of liveness offset.
        cursor += majit_translate::liveness::OFFSET_SIZE;
        if cursor > code.len() {
            return None;
        }
        return Some(DecodedOp {
            key,
            opname,
            argcodes,
            pc,
            next_pc: cursor,
        });
    }

    let mut chars = argcodes.chars();
    while let Some(c) = chars.next() {
        match c {
            'i' | 'c' | 'r' | 'f' => cursor += 1,
            'L' | 'd' | 'j' => cursor += 2,
            'I' | 'R' | 'F' => {
                // blackhole.py:139-143: varlist opens with a 1-byte length,
                // followed by that many 1-byte register indices.
                let list_len = *code.get(cursor)? as usize;
                cursor += 1 + list_len;
            }
            '>' => {
                // blackhole.py:185-209: result destination is 1 byte,
                // following `>i`, `>r`, or `>f`.
                let rt = chars.next()?;
                if !matches!(rt, 'i' | 'r' | 'f') {
                    return None;
                }
                cursor += 1;
            }
            _ => return None,
        }
    }

    if cursor > code.len() {
        return None;
    }
    Some(DecodedOp {
        key,
        opname,
        argcodes,
        pc,
        next_pc: cursor,
    })
}

/// Iterator over every instruction in a jitcode `code` slice. Yields
/// `DecodedOp` in linear order (fallthrough layout — branch targets in
/// `L`-typed operands are not followed). Stops on the first decode
/// failure, which surfaces either end-of-code or an insns-table miss.
pub fn decoded_ops(code: &[u8]) -> impl Iterator<Item = DecodedOp> + '_ {
    let mut pc = 0;
    std::iter::from_fn(move || {
        let op = decode_op_at(code, pc)?;
        pc = op.next_pc;
        Some(op)
    })
}

fn split_key(key: &str) -> (&str, &str) {
    match key.split_once('/') {
        Some((name, codes)) => (name, codes),
        None => (key, ""),
    }
}

/// Where a resolved operand came from and the value read at that slot.
///
/// RPython `blackhole.py:112-157` argcodes consume a register index byte
/// or a small-constant byte and produce the value the `bhimpl_*` method
/// receives. This enum captures both the source byte(s) and the resolved
/// value, so diagnostics and shadow-execution paths can surface either
/// without re-walking the code.
#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedOperand {
    /// `i`: `self.registers_i[code[pc]]` (blackhole.py:120).
    IntReg { reg: u8, value: i64 },
    /// `c`: signed byte treated as a small constant (blackhole.py:121-123,
    /// `signedord`).
    ConstByte { byte: i8 },
    /// `r`: `self.registers_r[code[pc]]` (blackhole.py:124-126).
    RefReg { reg: u8, value: i64 },
    /// `f`: `self.registers_f[code[pc]]` (blackhole.py:127-129).
    FloatReg { reg: u8, value: i64 },
    /// `L`: 2-byte little-endian unsigned label (blackhole.py:133-138).
    Label { target: u16 },
    /// `d`|`j`: 2-byte little-endian descr index into
    /// `metainterp_sd.descrs` (blackhole.py:150-157). `j` carries the
    /// additional RPython assertion `isinstance(value, JitCode)`; the
    /// resolver here records the index only.
    DescrIdx { index: u16, is_jitcode: bool },
    /// `I`: `[registers_i[idx] for idx in list]` (blackhole.py:139-143 via
    /// `_get_list_of_values`).
    IntList(Vec<(u8, i64)>),
    /// `R`: ref-list variant of the above.
    RefList(Vec<(u8, i64)>),
    /// `F`: float-list variant of the above.
    FloatList(Vec<(u8, i64)>),
}

/// Where a `bhimpl_*` result would be written back.
///
/// RPython `blackhole.py:185-223` handles `>i`, `>r`, `>f` result slots
/// (and the `iL` split for `goto_if_*` which the resolver treats as
/// `Int` here — the shadow layer can interpret the `L`-branch later).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedResult {
    Int { reg: u8 },
    Ref { reg: u8 },
    Float { reg: u8 },
}

/// One `DecodedOp` enriched with resolved operand values and the result
/// destination (if any). Construction is pure — walking the byte stream
/// with a read-only register file.
///
/// RPython parity: `_get_method.handler` up to the `unboundmethod(*args)`
/// call. The bhimpl dispatch itself is intentionally left out; this
/// struct is the data the shadow-record layer (Phase D-2) or a diff-only
/// analyzer can inspect without executing any side effect.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedOp {
    pub decoded: DecodedOp,
    pub operands: Vec<ResolvedOperand>,
    pub result: Option<ResolvedResult>,
}

/// Read-only view over the three register files a `bhimpl_*` would
/// consume. Empty slots read as 0 (RPython has no corresponding case —
/// in-range reads are a translation-time invariant). The resolver here
/// returns `None` for out-of-range indices so the caller can treat an
/// oversized jitcode as a decode failure rather than silently misread.
#[derive(Debug, Clone, Copy)]
pub struct RegisterFileView<'a> {
    pub registers_i: &'a [i64],
    pub registers_r: &'a [i64],
    pub registers_f: &'a [i64],
}

impl<'a> RegisterFileView<'a> {
    pub fn empty() -> Self {
        Self {
            registers_i: &[],
            registers_r: &[],
            registers_f: &[],
        }
    }
}

/// Port of `_get_method.handler` operand-resolution phase (everything up
/// to the `unboundmethod(*args)` call at blackhole.py:170).
///
/// Decodes the op at `pc`, then walks `argcodes` once more — this time
/// reading each operand byte *and* resolving it via the register files.
/// Returns `None` on any of:
///   - `pc` beyond `code.len()` or unknown opcode byte (same as
///     `decode_op_at`).
///   - register index out of range for the active file.
///   - unrecognized argcode char.
///
/// The function is intentionally stateless; the caller supplies a
/// `RegisterFileView` borrowing from whatever concrete register storage
/// the shadow layer uses.
pub fn resolve_op_at(code: &[u8], pc: usize, regs: RegisterFileView<'_>) -> Option<ResolvedOp> {
    let decoded = decode_op_at(code, pc)?;
    if decoded.opname == "live" {
        // bhimpl_live consumes OFFSET_SIZE operand bytes but produces no
        // resolved operands — decode_op_at already advanced past them.
        return Some(ResolvedOp {
            decoded,
            operands: Vec::new(),
            result: None,
        });
    }

    let mut cursor = pc + 1;
    let mut operands: Vec<ResolvedOperand> = Vec::new();
    let mut result: Option<ResolvedResult> = None;
    let mut chars = decoded.argcodes.chars();
    while let Some(c) = chars.next() {
        match c {
            'i' => {
                let reg = *code.get(cursor)?;
                cursor += 1;
                let value = *regs.registers_i.get(reg as usize)?;
                operands.push(ResolvedOperand::IntReg { reg, value });
            }
            'c' => {
                // blackhole.py:121-123 `signedord`: signed byte constant.
                let byte = *code.get(cursor)? as i8;
                cursor += 1;
                operands.push(ResolvedOperand::ConstByte { byte });
            }
            'r' => {
                let reg = *code.get(cursor)?;
                cursor += 1;
                let value = *regs.registers_r.get(reg as usize)?;
                operands.push(ResolvedOperand::RefReg { reg, value });
            }
            'f' => {
                let reg = *code.get(cursor)?;
                cursor += 1;
                let value = *regs.registers_f.get(reg as usize)?;
                operands.push(ResolvedOperand::FloatReg { reg, value });
            }
            'L' => {
                let lo = *code.get(cursor)? as u16;
                let hi = *code.get(cursor + 1)? as u16;
                cursor += 2;
                operands.push(ResolvedOperand::Label {
                    target: lo | (hi << 8),
                });
            }
            'd' | 'j' => {
                let lo = *code.get(cursor)? as u16;
                let hi = *code.get(cursor + 1)? as u16;
                cursor += 2;
                operands.push(ResolvedOperand::DescrIdx {
                    index: lo | (hi << 8),
                    is_jitcode: c == 'j',
                });
            }
            'I' | 'R' | 'F' => {
                let list_len = *code.get(cursor)? as usize;
                cursor += 1;
                let mut entries = Vec::with_capacity(list_len);
                let file: &[i64] = match c {
                    'I' => regs.registers_i,
                    'R' => regs.registers_r,
                    'F' => regs.registers_f,
                    _ => unreachable!(),
                };
                for _ in 0..list_len {
                    let reg = *code.get(cursor)?;
                    cursor += 1;
                    let value = *file.get(reg as usize)?;
                    entries.push((reg, value));
                }
                operands.push(match c {
                    'I' => ResolvedOperand::IntList(entries),
                    'R' => ResolvedOperand::RefList(entries),
                    'F' => ResolvedOperand::FloatList(entries),
                    _ => unreachable!(),
                });
            }
            '>' => {
                let rt = chars.next()?;
                let reg = *code.get(cursor)?;
                cursor += 1;
                result = Some(match rt {
                    'i' => ResolvedResult::Int { reg },
                    'r' => ResolvedResult::Ref { reg },
                    'f' => ResolvedResult::Float { reg },
                    _ => return None,
                });
            }
            _ => return None,
        }
    }

    // Sanity: the walker here must land on the same next_pc decode_op_at
    // computed. If not, our argcodes handling disagrees with decode_op_at
    // and something silently miscounted operand bytes.
    debug_assert_eq!(
        cursor, decoded.next_pc,
        "resolve_op_at cursor {cursor} != decode_op_at next_pc {} for key {}",
        decoded.next_pc, decoded.key,
    );
    Some(ResolvedOp {
        decoded,
        operands,
        result,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserializes_jitcodes_without_error() {
        let jitcodes = all_jitcodes();
        assert!(!jitcodes.is_empty(), "expected at least one jitcode");
    }

    #[test]
    fn deserializes_arms_without_error() {
        let arms = all_opcode_arms();
        assert!(!arms.is_empty(), "expected at least one opcode arm");
    }

    #[test]
    fn pop_top_lookup() {
        // Phase D-1 MVP target: Instruction::PopTop is arm_id=13 at build
        // time. Confirm arm → jitcode resolution works end-to-end and the
        // jitcode carries bytecode bytes (not an empty shell).
        let jc = jitcode_for_arm(13).expect("PopTop arm should resolve to a jitcode");
        assert!(
            !jc.code.is_empty(),
            "PopTop jitcode should have non-empty bytecode"
        );
        assert_eq!(
            jc.name, "Instruction::PopTop#13",
            "jitcode name should match the arm selector"
        );
    }

    #[test]
    fn extract_variant_name_handles_unit_and_struct_variants() {
        assert_eq!(extract_variant_name("PopTop"), "PopTop");
        assert_eq!(extract_variant_name("LoadFast { var_num: 3 }"), "LoadFast");
        assert_eq!(extract_variant_name("Resume { arg: 0 }"), "Resume");
    }

    #[test]
    fn instruction_variant_name_round_trips_via_debug() {
        let instr = Instruction::PopTop;
        assert_eq!(instruction_variant_name(&instr), "PopTop");
    }

    #[test]
    fn arm_id_for_pop_top_matches_arm_13() {
        // PopTop is a single-variant arm; `Instruction::PopTop` must
        // resolve to the same arm_id as the direct `jitcode_for_arm(13)`
        // lookup above.
        let arm_id =
            arm_id_for_instruction(&Instruction::PopTop).expect("PopTop must resolve to an arm_id");
        assert_eq!(arm_id, 13);
    }

    #[test]
    fn jitcode_for_instruction_matches_arm_lookup() {
        let jc = jitcode_for_instruction(&Instruction::PopTop)
            .expect("PopTop must resolve to a jitcode");
        assert_eq!(jc.name, "Instruction::PopTop#13");
        assert!(!jc.code.is_empty());
    }

    #[test]
    fn insns_table_is_populated() {
        let table = insns_opname_to_byte();
        assert!(
            !table.is_empty(),
            "pipeline.insns should contain at least the core ops"
        );
    }

    #[test]
    fn opname_round_trips_through_byte() {
        // RPython assembler.py keys are `opname/argcode` (the argcode is
        // appended during `write_insn`). `live/` — the canonical BC_LIVE
        // marker emitted by liveness insertion — is the stablest key to
        // assert on, since every flattened graph touches it at least
        // once. If this test fails the assembler stopped emitting BC_LIVE,
        // which is a bigger parity break.
        let byte = *insns_opname_to_byte()
            .get("live/")
            .expect("`live/` opcode must be in the insns table");
        assert_eq!(opname_for_byte(byte), Some("live/"));
    }

    #[test]
    fn first_byte_of_pop_top_jitcode_decodes() {
        // End-to-end: pop top's jitcode bytes must start with an opcode
        // byte that `opname_for_byte` can decode.
        let jc = jitcode_for_instruction(&Instruction::PopTop)
            .expect("PopTop must resolve to a jitcode");
        let first = *jc
            .code
            .first()
            .expect("PopTop jitcode should have at least one opcode byte");
        assert!(
            opname_for_byte(first).is_some(),
            "first byte {first} of PopTop jitcode is unknown to the insns table",
        );
    }

    #[test]
    fn decode_live_skips_offset_size() {
        // `live/` is special-cased: empty argcodes but +OFFSET_SIZE (2)
        // bytes per blackhole.py:1603-1605.
        let live_byte = *insns_opname_to_byte()
            .get("live/")
            .expect("`live/` must be in insns table");
        let code = [live_byte, 0x00, 0x00];
        let op = decode_op_at(&code, 0).expect("live/ must decode");
        assert_eq!(op.opname, "live");
        assert_eq!(op.argcodes, "");
        assert_eq!(op.pc, 0);
        assert_eq!(
            op.next_pc,
            1 + majit_translate::liveness::OFFSET_SIZE,
            "live/ must advance by OFFSET_SIZE past the opcode byte",
        );
    }

    #[test]
    fn decode_int_add_reads_ii_operands_and_one_result_byte() {
        // `int_add/ii>i` — 1+1 operand bytes + 1 result byte = 3 bytes
        // after the opcode.
        let op_byte = *insns_opname_to_byte()
            .get("int_add/ii>i")
            .expect("int_add/ii>i must be in insns table");
        let code = [op_byte, 0x01, 0x02, 0x03];
        let op = decode_op_at(&code, 0).expect("int_add must decode");
        assert_eq!(op.opname, "int_add");
        assert_eq!(op.argcodes, "ii>i");
        assert_eq!(op.next_pc, 4);
    }

    #[test]
    fn decode_pop_top_jitcode_walks_to_end() {
        // PopTop's jitcode is ~41 bytes. Walking with `decoded_ops`
        // must reach exactly code.len() if every byte decodes cleanly.
        let jc = jitcode_for_instruction(&Instruction::PopTop)
            .expect("PopTop must resolve to a jitcode");
        let mut last_next = 0;
        let mut step_count = 0;
        for op in decoded_ops(&jc.code) {
            last_next = op.next_pc;
            step_count += 1;
        }
        assert!(step_count > 0, "should decode at least one op");
        assert_eq!(
            last_next,
            jc.code.len(),
            "decoded stream must end exactly at code.len() for PopTop \
             (stopped at {last_next} after {step_count} ops, code.len()={})",
            jc.code.len(),
        );
    }

    #[test]
    fn decode_varlist_reads_length_byte_plus_items() {
        // Synthetic: inline_call_ir_r/dIR>r — d(2) + I(1+N) + R(1+M) + r(1).
        let op_byte = *insns_opname_to_byte()
            .get("inline_call_ir_r/dIR>r")
            .expect("inline_call_ir_r/dIR>r must be in insns table");
        // opcode, d=0x0001, I-list len=2 + 2 regs, R-list len=1 + 1 reg, dst
        // = 1 + 2 + (1+2) + (1+1) + 1 = 9 bytes
        let code = [
            op_byte, 0x01, 0x00, // d
            0x02, 0x00, 0x01, // I: len=2, [0,1]
            0x01, 0x00, // R: len=1, [0]
            0x03, // >r: dst=3
        ];
        let op = decode_op_at(&code, 0).expect("inline_call_ir_r must decode");
        assert_eq!(op.opname, "inline_call_ir_r");
        assert_eq!(op.next_pc, 9);
    }

    #[test]
    fn decode_unknown_opcode_returns_none() {
        // Byte 0xFF should not be a valid opcode — 21 entries go 0..=20.
        let code = [0xFF];
        assert!(
            decode_op_at(&code, 0).is_none(),
            "unknown opcode byte must yield None",
        );
    }

    #[test]
    fn resolve_int_add_reads_both_register_values() {
        // `int_add/ii>i`: canonical — both operands read from int-regs,
        // result written to int-reg. RPython
        // `blackhole.py:@arguments("i", "i", returns="i")`.
        let op_byte = *insns_opname_to_byte()
            .get("int_add/ii>i")
            .expect("int_add/ii>i must be in insns table");
        // code: [opcode, i_idx=2, i_idx=1, dst=0]
        let code = [op_byte, 0x02, 0x01, 0x00];
        let regs = RegisterFileView {
            registers_i: &[0, 7, 42, 0],
            registers_r: &[],
            registers_f: &[],
        };
        let op = resolve_op_at(&code, 0, regs).expect("int_add must resolve");
        assert_eq!(op.decoded.opname, "int_add");
        assert_eq!(op.operands.len(), 2);
        assert_eq!(
            op.operands[0],
            ResolvedOperand::IntReg { reg: 2, value: 42 }
        );
        assert_eq!(op.operands[1], ResolvedOperand::IntReg { reg: 1, value: 7 });
        assert_eq!(op.result, Some(ResolvedResult::Int { reg: 0 }));
    }

    #[test]
    fn resolve_live_yields_no_operands() {
        // `live/`: empty argcodes, OFFSET_SIZE skipped; no operands, no
        // result.
        let live_byte = *insns_opname_to_byte()
            .get("live/")
            .expect("live/ must be in insns table");
        let code = [live_byte, 0x00, 0x00];
        let op = resolve_op_at(&code, 0, RegisterFileView::empty()).expect("live/ must resolve");
        assert!(op.operands.is_empty());
        assert!(op.result.is_none());
        assert_eq!(op.decoded.opname, "live");
    }

    #[test]
    fn resolve_out_of_range_int_reg_returns_none() {
        // int_add/ii>i: opcode reads registers_i[5], but registers_i is
        // only 2 wide. Must surface as decode failure, not a silent 0.
        let op_byte = *insns_opname_to_byte()
            .get("int_add/ii>i")
            .expect("int_add/ii>i must be in insns table");
        let code = [op_byte, 0x00, 0x05, 0x00];
        let regs = RegisterFileView {
            registers_i: &[10, 20],
            registers_r: &[],
            registers_f: &[],
        };
        assert!(resolve_op_at(&code, 0, regs).is_none());
    }

    #[test]
    fn resolve_varlist_reads_each_member() {
        // inline_call_ir_r/dIR>r: d(2) + I(len+items) + R(len+items) + >r(1).
        let op_byte = *insns_opname_to_byte()
            .get("inline_call_ir_r/dIR>r")
            .expect("inline_call_ir_r/dIR>r must be in insns table");
        // descr=0x0102, I=[reg1, reg2], R=[reg0], dst=4
        let code = [
            op_byte, 0x02, 0x01, //
            0x02, 0x01, 0x02, //
            0x01, 0x00, //
            0x04,
        ];
        let regs = RegisterFileView {
            registers_i: &[0, 111, 222],
            registers_r: &[333],
            registers_f: &[],
        };
        let op = resolve_op_at(&code, 0, regs).expect("inline_call_ir_r must resolve");
        assert_eq!(op.operands.len(), 3);
        assert_eq!(
            op.operands[0],
            ResolvedOperand::DescrIdx {
                index: 0x0102,
                is_jitcode: false,
            },
        );
        assert_eq!(
            op.operands[1],
            ResolvedOperand::IntList(vec![(1, 111), (2, 222)]),
        );
        assert_eq!(op.operands[2], ResolvedOperand::RefList(vec![(0, 333)]));
        assert_eq!(op.result, Some(ResolvedResult::Ref { reg: 4 }));
    }

    #[test]
    fn arm_id_covers_or_grouped_variants() {
        // arm_id=3 is the `Instruction::LoadFast | Instruction::LoadFastBorrow`
        // group (see build-time opcode_dispatch). Both variants must land on
        // the same arm_id — matching the RPython model where `Or` selectors
        // register a single Python method under each opcode.
        use pyre_interpreter::bytecode::Arg;
        let id_a = arm_id_for_instruction(&Instruction::LoadFast {
            var_num: Arg::marker(),
        })
        .expect("LoadFast must resolve");
        let id_b = arm_id_for_instruction(&Instruction::LoadFastBorrow {
            var_num: Arg::marker(),
        })
        .expect("LoadFastBorrow must resolve");
        assert_eq!(id_a, id_b, "Or-grouped variants must share an arm_id");
    }

    #[test]
    fn build_default_bh_builder_matches_insns_table() {
        // Slice 3a: the runtime-side `BlackholeInterpBuilder` is reachable
        // from pyre-jit-trace. After `setup_insns + wire_bhimpl_handlers`
        // it must carry the same byte<->opname mapping as the build-time
        // insns bincode, and it must resolve the three well-known
        // opcodes (`live/`, `catch_exception/L`, `rvmprof_code/ii`) when
        // they appear in the table.
        let builder = build_default_bh_builder();
        let expected_live = insns_opname_to_byte().get("live/").copied();
        assert_eq!(Some(builder.op_live), expected_live);
        // Reverse mapping parity: every opname in the build-time table
        // must appear at the same byte index in builder._insns.
        for (key, &byte) in insns_opname_to_byte() {
            assert_eq!(
                &builder._insns[byte as usize], key,
                "builder._insns[{byte}] disagrees with build-time key {key:?}",
            );
        }
    }

    #[test]
    fn default_bh_builder_handler_coverage_report() {
        // Diagnostic: surface the opnames in the real insns table that
        // `wire_bhimpl_handlers` did NOT override. These fall back to
        // the `setup_insns` placeholder and would panic on dispatch.
        //
        // `BlackholeInterpBuilder::unwired_opnames()` is the accessor
        // that returns the gap. Each unwired opname is a concrete
        // bhimpl port that Phase D-2 must land before flipping the
        // production dispatch path to insns-table-driven.
        //
        // The test does NOT fail on unwired opnames — it just reports
        // them. Gating turns on later once the table path is the sole
        // production dispatch route.
        let builder = build_default_bh_builder();
        let total = insns_opname_to_byte().len();
        let mut unwired: Vec<&str> = builder.unwired_opnames();
        unwired.sort_unstable();
        let wired = total - unwired.len();
        eprintln!(
            "[jitcode_runtime] coverage: {wired}/{total} opnames wired; \
             {} unwired: {:?}",
            unwired.len(),
            unwired,
        );
        // Sanity: `live/` must be present in the insns table and wired.
        // Any binary that lacks it is structurally broken.
        assert!(
            insns_opname_to_byte().contains_key("live/"),
            "live/ missing from real insns table — broken build",
        );
        assert!(
            !unwired.iter().any(|k| *k == "live/"),
            "live/ must be wired by wire_bhimpl_handlers",
        );
    }

    #[test]
    fn pop_top_jitcode_is_complete_in_canonical_store() {
        // RPython parity target: the deserialized `ALL_JITCODES` entries
        // are themselves the canonical objects produced by
        // `CodeWriter.make_jitcodes()`. Avoid the transitional
        // build→runtime `From` adapter here and assert directly on the
        // canonical object that build.rs persisted.
        let arm_id =
            arm_id_for_instruction(&Instruction::PopTop).expect("PopTop must resolve to an arm_id");
        let arm = get_arm(arm_id).expect("PopTop arm must exist");
        let bt_jc = jitcode_for_instruction(&Instruction::PopTop)
            .expect("PopTop must resolve to a jitcode");
        assert!(!bt_jc.code.is_empty());
        assert_eq!(bt_jc.name, "Instruction::PopTop#13");
        assert_eq!(arm.entry_jitcode_index, Some(bt_jc.index()));
        assert_eq!(
            bt_jc.num_regs_and_consts_i(),
            bt_jc.num_regs_i() + bt_jc.constants_i.len()
        );
        assert_eq!(
            bt_jc.num_regs_and_consts_r(),
            bt_jc.num_regs_r() + bt_jc.constants_r.len()
        );
        assert_eq!(
            bt_jc.num_regs_and_consts_f(),
            bt_jc.num_regs_f() + bt_jc.constants_f.len()
        );
    }
}
