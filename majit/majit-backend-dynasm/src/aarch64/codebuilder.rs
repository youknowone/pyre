/// aarch64/codebuilder.py parity: low-level AArch64 instruction builders.
///
/// The Python backend keeps raw machine-code helpers in `codebuilder.py` and
/// the trace/op lowering in `assembler.py`.  The dynasm backend still stores
/// the dynasm assembler inside `AssemblerARM64`, but the codebuilder-shaped
/// helpers live here so PyPy backend patches have the same file boundary.
use dynasmrt::{DynasmApi, dynasm};

use super::assembler::AssemblerARM64;

impl<'a> AssemblerARM64<'a> {
    /// codebuilder.py:509 `gen_load_int`.
    ///
    /// Materialise a 64-bit immediate into `reg`. This is variable length:
    /// callers whose emitted block is rewritten in place afterwards (the
    /// `frame_depth_to_patch` sites, patched by `patch_frame_depth` which
    /// always writes 4 words) MUST use [`emit_mov_imm64_fixed4`] instead,
    /// which always emits exactly four words.
    pub(crate) fn emit_mov_imm64(&mut self, reg: u32, val: i64) {
        let r = reg as u8;
        if val < 0 {
            if val >= -65536 {
                let inv = ((!val) as u64 & 0xFFFF) as u32;
                dynasm!(self.mc ; .arch aarch64 ; movn X(r), inv);
                return;
            }
            let inv = ((!val) as u64 & 0xFFFF) as u32;
            dynasm!(self.mc ; .arch aarch64 ; movn X(r), inv);
            let mut value = val >> 16;
            let mut shift = 16;
            while shift < 64 {
                let hw = (value & 0xFFFF) as u32;
                if hw != 0xFFFF {
                    match shift {
                        16 => dynasm!(self.mc ; .arch aarch64 ; movk X(r), hw, lsl 16),
                        32 => dynasm!(self.mc ; .arch aarch64 ; movk X(r), hw, lsl 32),
                        48 => dynasm!(self.mc ; .arch aarch64 ; movk X(r), hw, lsl 48),
                        _ => unreachable!(),
                    }
                }
                shift += 16;
                value >>= 16;
            }
            return;
        }
        let mut value = val as u64;
        dynasm!(self.mc ; .arch aarch64 ; movz X(r), (value & 0xFFFF) as u32);
        value >>= 16;
        let mut shift = 16;
        while value != 0 {
            let hw = (value & 0xFFFF) as u32;
            match shift {
                16 => dynasm!(self.mc ; .arch aarch64 ; movk X(r), hw, lsl 16),
                32 => dynasm!(self.mc ; .arch aarch64 ; movk X(r), hw, lsl 32),
                48 => dynasm!(self.mc ; .arch aarch64 ; movk X(r), hw, lsl 48),
                _ => unreachable!(),
            }
            shift += 16;
            value >>= 16;
        }
    }

    /// Materialise a 64-bit immediate as a fixed four-word
    /// `movz`/`movk lsl 16`/`movk lsl 32`/`movk lsl 48` block, byte-identical
    /// to [`encode_mov_imm64_words`].  Used only at the `frame_depth_to_patch`
    /// sites, whose block is rewritten in place by `patch_frame_depth` (always
    /// 16 bytes); the variable-length [`emit_mov_imm64`] would let the patch
    /// overrun into the following instructions.
    pub(super) fn emit_mov_imm64_fixed4(&mut self, reg: u32, val: i64) {
        let v = val as u64;
        let r = reg as u8;
        dynasm!(self.mc ; .arch aarch64
            ; movz X(r), (v & 0xFFFF) as u32
            ; movk X(r), ((v >> 16) & 0xFFFF) as u32, lsl 16
            ; movk X(r), ((v >> 32) & 0xFFFF) as u32, lsl 32
            ; movk X(r), ((v >> 48) & 0xFFFF) as u32, lsl 48
        );
    }

    /// Hand-encode the four ARM64 words `emit_mov_imm64_fixed4` produces for
    /// `(rd, val)`: `MOVZ`/`MOVK lsl 16`/`MOVK lsl 32`/`MOVK lsl 48` (64-bit
    /// variants).  Kept byte-identical to the dynasm output by
    /// `frame_depth_patch_words_match_emit_mov_imm64`.
    pub(super) fn encode_mov_imm64_words(rd: u32, val: i64) -> [u32; 4] {
        let v = val as u64;
        let rd = rd & 0x1F;
        let imm16 = |shift: u32| (((v >> shift) & 0xFFFF) as u32) << 5;
        [
            0xD280_0000 | imm16(0) | rd,              // movz Xrd, #imm, lsl 0
            0xF280_0000 | (1 << 21) | imm16(16) | rd, // movk Xrd, #imm, lsl 16
            0xF280_0000 | (2 << 21) | imm16(32) | rd, // movk Xrd, #imm, lsl 32
            0xF280_0000 | (3 << 21) | imm16(48) | rd, // movk Xrd, #imm, lsl 48
        ]
    }
}

#[cfg(test)]
mod tests {
    use dynasmrt::aarch64::Assembler;
    use dynasmrt::{DynasmApi, dynasm};

    use super::*;

    /// `patch_frame_depth` rewrites the depth placeholder by hand-encoding
    /// four `movz/movk` words.  They must be byte-identical to the fixed
    /// four-word block `emit_mov_imm64_fixed4` (dynasm) emits at the patch
    /// sites, otherwise the patched bridge would load a corrupt depth into
    /// the realloc slowpath.  (The inline sequence below is exactly that
    /// block; the general `emit_mov_imm64` is variable length and is not
    /// used at patch sites.)
    #[test]
    fn frame_depth_patch_words_match_emit_mov_imm64() {
        // (rd, val): the CMP site targets x17, the ARG1 site x1; cover the
        // full 64-bit range so every `movk` hw position is exercised.
        let cases: &[(u32, i64)] = &[
            (17, 0),
            (1, 0xffffff),
            (17, 56),
            (1, 0x1234_5678),
            (17, -1),
            (1, 0x7fff_ffff_ffff_ffff),
            (17, 0x0001_0000_0001_0000),
        ];
        for &(rd, val) in cases {
            let mut mc = Assembler::new().unwrap();
            let r = rd as u8;
            let v = val as u64;
            dynasm!(mc ; .arch aarch64
                ; movz X(r), (v & 0xFFFF) as u32
                ; movk X(r), ((v >> 16) & 0xFFFF) as u32, lsl 16
                ; movk X(r), ((v >> 32) & 0xFFFF) as u32, lsl 32
                ; movk X(r), ((v >> 48) & 0xFFFF) as u32, lsl 48
            );
            let buf = mc.finalize().unwrap();
            let mut expected = [0u32; 4];
            for (i, w) in expected.iter_mut().enumerate() {
                let b = i * 4;
                *w = u32::from_le_bytes([buf[b], buf[b + 1], buf[b + 2], buf[b + 3]]);
            }
            let got = AssemblerARM64::encode_mov_imm64_words(rd, val);
            assert_eq!(got, expected, "rd={rd} val={val:#x}");
        }
    }
}
