//! Memory-operand encoders from `rpython/jit/backend/x86/rx86.py`.
//!
//! `encode_mem_reg_plus_const` covers every base, including the rsp/r12 SIB
//! byte (`encode_stack_sp`) and the rbp/r13 forced displacement
//! (`encode_stack_bp` and the 64-bit special cases). A SIB byte is emitted
//! only for those bases or when an index is present. Displacement is omitted
//! when the offset is 0, `disp8` when `single_byte`, and `disp32` otherwise.
//!
//! Instruction emitters write into the same [`dynasmrt::DynasmApi`] buffer the
//! assembler uses.

use dynasmrt::DynasmApi;

/// `rx86.py` `REX_W`.
pub const REX_W: u8 = 8;
/// `rx86.py` `REX_R`.
pub const REX_R: u8 = 4;
/// `rx86.py` `REX_X`.
pub const REX_X: u8 = 2;
/// `rx86.py` `REX_B`.
pub const REX_B: u8 = 1;

const RSP: u8 = 4;
const RBP: u8 = 5;

/// `rx86.py` `single_byte`.
pub fn single_byte(value: i32) -> bool {
    (-128..128).contains(&value)
}

/// `rx86.py` `fits_in_32bits`.
pub fn fits_in_32bits(value: i64) -> bool {
    i32::try_from(value).is_ok()
}

/// `rx86.py` `rex_register`. `factor` is 1 (`REX_B`) or 8 (`REX_R`).
pub fn rex_register(reg: u8, factor: u8) -> u8 {
    if reg >= 8 {
        if factor == 1 { REX_B } else { REX_R }
    } else {
        0
    }
}

/// `rx86.py` `rex_byte_register`.
///
/// `reg` is the low-byte register number (`al` = 0 … `r15b` = 15), the same
/// numbering `Rb` uses. `spl`/`bpl`/`sil`/`dil` are 4…7 and `r8b`…`r15b` are
/// 8…15; those forms need a REX prefix (`rex_fw` on the byte-register
/// instructions, plus `REX_R`/`REX_B` when `reg >= 8`).
pub fn rex_byte_register(reg: u8, factor: u8) -> u8 {
    rex_register(reg, factor)
}

/// `spl`/`bpl`/`sil`/`dil` are encoded as `ah`…`bh` unless a REX prefix is present.
fn byte_reg_forces_rex(reg: u8) -> bool {
    (4..8).contains(&reg)
}

/// `rx86.py` `rex_mem_reg_plus_const`.
pub fn rex_mem_reg_plus_const(reg: u8) -> u8 {
    if reg >= 8 { REX_B } else { 0 }
}

/// `rx86.py` `rex_mem_reg_plus_scaled_reg_plus_const`.
pub fn rex_mem_reg_plus_scaled_reg_plus_const(base: u8, index: u8) -> u8 {
    let mut rex = 0;
    if base >= 8 {
        rex |= REX_B;
    }
    if index >= 8 {
        rex |= REX_X;
    }
    rex
}

fn reg3(reg: u8) -> u8 {
    reg & 7
}

fn reg_field(reg: u8) -> u8 {
    reg3(reg) << 3
}

/// `rx86.py` `encode_mem_reg_plus_const`, extended to rsp/rbp the way
/// `encode_stack_sp` / `encode_stack_bp` encode them. `orbyte` is the
/// ModRM reg field (already shifted by 3).
pub fn encode_mem_reg_plus_const(mc: &mut impl DynasmApi, reg: u8, offset: i32, orbyte: u8) {
    let reg1 = reg3(reg);
    let mut no_offset = offset == 0;
    // rsp/r12 look like esp in the low 3 bits and need a SIB with no index.
    // rbp/r13 look like ebp and cannot use mod=00 (that encoding is RIP-relative).
    let sib = if reg1 == RSP {
        Some((RSP << 3) | RSP)
    } else {
        None
    };
    if reg1 == RBP {
        no_offset = false;
    }
    if no_offset {
        mc.push(orbyte | reg1);
        if let Some(sib) = sib {
            mc.push(sib);
        }
    } else if single_byte(offset) {
        mc.push(0x40 | orbyte | reg1);
        if let Some(sib) = sib {
            mc.push(sib);
        }
        mc.push(offset as u8);
    } else {
        mc.push(0x80 | orbyte | reg1);
        if let Some(sib) = sib {
            mc.push(sib);
        }
        mc.push_i32(offset);
    }
}

fn scale_shift(scale: u8) -> u8 {
    match scale {
        1 => 0,
        2 => 1,
        4 => 2,
        8 => 3,
        _ => panic!("rx86 scale must be 1, 2, 4 or 8, got {scale}"),
    }
}

/// `rx86.py` `encode_mem_reg_plus_scaled_reg_plus_const`.
/// `scaleshift` is 0…3 for scales 1/2/4/8. `orbyte` is the ModRM reg field.
pub fn encode_mem_reg_plus_scaled_reg_plus_const(
    mc: &mut impl DynasmApi,
    base: u8,
    index: u8,
    scaleshift: u8,
    offset: i32,
    orbyte: u8,
) {
    let reg1 = reg3(base);
    let reg2 = reg3(index);
    let sib = (scaleshift << 6) | (reg2 << 3) | reg1;
    let mut no_offset = offset == 0;
    // r13/rbp in the SIB base field cannot use mod=00.
    if reg1 == RBP {
        no_offset = false;
    }
    if no_offset {
        mc.push(0x04 | orbyte);
        mc.push(sib);
    } else if single_byte(offset) {
        mc.push(0x44 | orbyte);
        mc.push(sib);
        mc.push(offset as u8);
    } else {
        mc.push(0x84 | orbyte);
        mc.push(sib);
        mc.push_i32(offset);
    }
}

fn emit_rex(mc: &mut impl DynasmApi, rex: u8, rex_w: bool, force: bool) {
    if rex_w || force || rex != 0 {
        let w = if rex_w { REX_W } else { 0 };
        mc.push(0x40 | w | rex);
    }
}

fn emit_prefixes(mc: &mut impl DynasmApi, prefixes: &[u8]) {
    for byte in prefixes {
        mc.push(*byte);
    }
}

fn emit_imm8(mc: &mut impl DynasmApi, imm: i8) {
    mc.push(imm as u8);
}

fn emit_imm16(mc: &mut impl DynasmApi, imm: i16) {
    mc.push_i16(imm);
}

fn emit_imm32(mc: &mut impl DynasmApi, imm: i32) {
    mc.push_i32(imm);
}

fn gpr_rm(
    mc: &mut impl DynasmApi,
    prefixes: &[u8],
    rex_w: bool,
    opcode: &[u8],
    reg: u8,
    base: u8,
    offset: i32,
) {
    emit_prefixes(mc, prefixes);
    emit_rex(
        mc,
        rex_register(reg, 8) | rex_mem_reg_plus_const(base),
        rex_w,
        false,
    );
    for byte in opcode {
        mc.push(*byte);
    }
    encode_mem_reg_plus_const(mc, base, offset, reg_field(reg));
}

fn gpr_mr_byte(mc: &mut impl DynasmApi, byte_reg: u8, base: u8, offset: i32) {
    let rex = rex_byte_register(byte_reg, 8) | rex_mem_reg_plus_const(base);
    // `MOV8_*r` is `rex_fw`: a REX prefix is mandatory for spl/bpl/sil/dil.
    emit_rex(mc, rex, false, byte_reg_forces_rex(byte_reg));
    mc.push(0x88);
    encode_mem_reg_plus_const(mc, base, offset, reg_field(byte_reg));
}

fn gpr_ra(
    mc: &mut impl DynasmApi,
    prefixes: &[u8],
    rex_w: bool,
    opcode: &[u8],
    reg: u8,
    base: u8,
    index: u8,
    scale: u8,
    offset: i32,
) {
    emit_prefixes(mc, prefixes);
    emit_rex(
        mc,
        rex_register(reg, 8) | rex_mem_reg_plus_scaled_reg_plus_const(base, index),
        rex_w,
        false,
    );
    for byte in opcode {
        mc.push(*byte);
    }
    encode_mem_reg_plus_scaled_reg_plus_const(
        mc,
        base,
        index,
        scale_shift(scale),
        offset,
        reg_field(reg),
    );
}

fn byte_ar(mc: &mut impl DynasmApi, byte_reg: u8, base: u8, index: u8, scale: u8, offset: i32) {
    let rex = rex_byte_register(byte_reg, 8) | rex_mem_reg_plus_scaled_reg_plus_const(base, index);
    emit_rex(mc, rex, false, byte_reg_forces_rex(byte_reg));
    mc.push(0x88);
    encode_mem_reg_plus_scaled_reg_plus_const(
        mc,
        base,
        index,
        scale_shift(scale),
        offset,
        reg_field(byte_reg),
    );
}

fn xmm_xm(mc: &mut impl DynasmApi, prefix: u8, opcode: u8, xmm: u8, base: u8, offset: i32) {
    mc.push(prefix);
    emit_rex(
        mc,
        rex_register(xmm, 8) | rex_mem_reg_plus_const(base),
        false,
        false,
    );
    mc.push(0x0f);
    mc.push(opcode);
    encode_mem_reg_plus_const(mc, base, offset, reg_field(xmm));
}

fn xmm_xa(
    mc: &mut impl DynasmApi,
    prefix: u8,
    opcode: u8,
    xmm: u8,
    base: u8,
    index: u8,
    scale: u8,
    offset: i32,
) {
    mc.push(prefix);
    emit_rex(
        mc,
        rex_register(xmm, 8) | rex_mem_reg_plus_scaled_reg_plus_const(base, index),
        false,
        false,
    );
    mc.push(0x0f);
    mc.push(opcode);
    encode_mem_reg_plus_scaled_reg_plus_const(
        mc,
        base,
        index,
        scale_shift(scale),
        offset,
        reg_field(xmm),
    );
}

fn mem_imm(
    mc: &mut impl DynasmApi,
    prefixes: &[u8],
    rex_w: bool,
    force_rex: bool,
    opcode: u8,
    modrm_reg: u8,
    base: u8,
    offset: i32,
) {
    emit_prefixes(mc, prefixes);
    emit_rex(mc, rex_mem_reg_plus_const(base), rex_w, force_rex);
    mc.push(opcode);
    encode_mem_reg_plus_const(mc, base, offset, modrm_reg << 3);
}

fn mem_imm_scaled(
    mc: &mut impl DynasmApi,
    prefixes: &[u8],
    rex_w: bool,
    opcode: u8,
    modrm_reg: u8,
    base: u8,
    index: u8,
    scale: u8,
    offset: i32,
) {
    emit_prefixes(mc, prefixes);
    emit_rex(
        mc,
        rex_mem_reg_plus_scaled_reg_plus_const(base, index),
        rex_w,
        false,
    );
    mc.push(opcode);
    encode_mem_reg_plus_scaled_reg_plus_const(
        mc,
        base,
        index,
        scale_shift(scale),
        offset,
        modrm_reg << 3,
    );
}

/// `rx86.py` `MOV_rm` — `mov r64, r/m64`.
pub fn mov_rm(mc: &mut impl DynasmApi, dst: u8, base: u8, offset: i32) {
    gpr_rm(mc, &[], true, &[0x8b], dst, base, offset);
}

/// `rx86.py` `MOV_mr` — `mov r/m64, r64`.
pub fn mov_mr(mc: &mut impl DynasmApi, base: u8, offset: i32, src: u8) {
    gpr_rm(mc, &[], true, &[0x89], src, base, offset);
}

/// `rx86.py` `MOV32_rm` — `mov r32, r/m32` (zero-extends into r64).
pub fn mov32_rm(mc: &mut impl DynasmApi, dst: u8, base: u8, offset: i32) {
    gpr_rm(mc, &[], false, &[0x8b], dst, base, offset);
}

/// `rx86.py` `MOV32_mr` — `mov r/m32, r32`.
pub fn mov32_mr(mc: &mut impl DynasmApi, base: u8, offset: i32, src: u8) {
    gpr_rm(mc, &[], false, &[0x89], src, base, offset);
}

/// `rx86.py` `MOV16_mr` — `mov r/m16, r16`.
pub fn mov16_mr(mc: &mut impl DynasmApi, base: u8, offset: i32, src: u8) {
    gpr_rm(mc, &[0x66], false, &[0x89], src, base, offset);
}

/// `rx86.py` `MOV8_mr` — `mov r/m8, r8`.
pub fn mov8_mr(mc: &mut impl DynasmApi, base: u8, offset: i32, src: u8) {
    gpr_mr_byte(mc, src, base, offset);
}

/// `rx86.py` `MOV_mi` — `mov r/m64, imm32` (sign-extended).
pub fn mov_mi(mc: &mut impl DynasmApi, base: u8, offset: i32, imm: i32) {
    mem_imm(mc, &[], true, false, 0xc7, 0, base, offset);
    emit_imm32(mc, imm);
}

/// `rx86.py` `MOV32_mi` — `mov r/m32, imm32`.
pub fn mov32_mi(mc: &mut impl DynasmApi, base: u8, offset: i32, imm: i32) {
    mem_imm(mc, &[], false, false, 0xc7, 0, base, offset);
    emit_imm32(mc, imm);
}

/// `rx86.py` `MOV16_mi` — `mov r/m16, imm16`.
pub fn mov16_mi(mc: &mut impl DynasmApi, base: u8, offset: i32, imm: i16) {
    mem_imm(mc, &[0x66], false, false, 0xc7, 0, base, offset);
    emit_imm16(mc, imm);
}

/// `rx86.py` `MOV8_mi` — `mov r/m8, imm8`.
pub fn mov8_mi(mc: &mut impl DynasmApi, base: u8, offset: i32, imm: i8) {
    mem_imm(mc, &[], false, false, 0xc6, 0, base, offset);
    emit_imm8(mc, imm);
}

/// `rx86.py` `MOV_ra` — `mov r64, [base + index*scale + disp]`.
pub fn mov_ra(mc: &mut impl DynasmApi, dst: u8, base: u8, index: u8, scale: u8, offset: i32) {
    gpr_ra(mc, &[], true, &[0x8b], dst, base, index, scale, offset);
}

/// `rx86.py` `MOV_ar` — `mov [base + index*scale + disp], r64`.
pub fn mov_ar(mc: &mut impl DynasmApi, base: u8, index: u8, scale: u8, offset: i32, src: u8) {
    gpr_ra(mc, &[], true, &[0x89], src, base, index, scale, offset);
}

/// `rx86.py` `MOV32_ra`.
pub fn mov32_ra(mc: &mut impl DynasmApi, dst: u8, base: u8, index: u8, scale: u8, offset: i32) {
    gpr_ra(mc, &[], false, &[0x8b], dst, base, index, scale, offset);
}

/// `rx86.py` `MOV32_ar`.
pub fn mov32_ar(mc: &mut impl DynasmApi, base: u8, index: u8, scale: u8, offset: i32, src: u8) {
    gpr_ra(mc, &[], false, &[0x89], src, base, index, scale, offset);
}

/// `rx86.py` `MOV16_ar`.
pub fn mov16_ar(mc: &mut impl DynasmApi, base: u8, index: u8, scale: u8, offset: i32, src: u8) {
    gpr_ra(mc, &[0x66], false, &[0x89], src, base, index, scale, offset);
}

/// `rx86.py` `MOV8_ar`.
pub fn mov8_ar(mc: &mut impl DynasmApi, base: u8, index: u8, scale: u8, offset: i32, src: u8) {
    byte_ar(mc, src, base, index, scale, offset);
}

/// `rx86.py` `MOV_ai` — `mov r/m64, imm32` with a scaled address.
pub fn mov_ai(mc: &mut impl DynasmApi, base: u8, index: u8, scale: u8, offset: i32, imm: i32) {
    mem_imm_scaled(mc, &[], true, 0xc7, 0, base, index, scale, offset);
    emit_imm32(mc, imm);
}

/// `rx86.py` `MOV32_ai`.
pub fn mov32_ai(mc: &mut impl DynasmApi, base: u8, index: u8, scale: u8, offset: i32, imm: i32) {
    mem_imm_scaled(mc, &[], false, 0xc7, 0, base, index, scale, offset);
    emit_imm32(mc, imm);
}

/// `rx86.py` `MOV16_ai`.
pub fn mov16_ai(mc: &mut impl DynasmApi, base: u8, index: u8, scale: u8, offset: i32, imm: i16) {
    mem_imm_scaled(mc, &[0x66], false, 0xc7, 0, base, index, scale, offset);
    emit_imm16(mc, imm);
}

/// `rx86.py` `MOV8_ai`.
pub fn mov8_ai(mc: &mut impl DynasmApi, base: u8, index: u8, scale: u8, offset: i32, imm: i8) {
    mem_imm_scaled(mc, &[], false, 0xc6, 0, base, index, scale, offset);
    emit_imm8(mc, imm);
}

/// `rx86.py` `LEA_rm` — `lea r64, [base + disp]`.
pub fn lea_rm(mc: &mut impl DynasmApi, dst: u8, base: u8, offset: i32) {
    gpr_rm(mc, &[], true, &[0x8d], dst, base, offset);
}

/// `rx86.py` `LEA_ra` — `lea r64, [base + index*scale + disp]`.
pub fn lea_ra(mc: &mut impl DynasmApi, dst: u8, base: u8, index: u8, scale: u8, offset: i32) {
    gpr_ra(mc, &[], true, &[0x8d], dst, base, index, scale, offset);
}

/// `rx86.py` `MOVZX8_rm` — `movzx r64, r/m8`.
pub fn movzx8_rm(mc: &mut impl DynasmApi, dst: u8, base: u8, offset: i32) {
    gpr_rm(mc, &[], true, &[0x0f, 0xb6], dst, base, offset);
}

/// `rx86.py` `MOVZX16_rm` — `movzx r64, r/m16`.
pub fn movzx16_rm(mc: &mut impl DynasmApi, dst: u8, base: u8, offset: i32) {
    gpr_rm(mc, &[], true, &[0x0f, 0xb7], dst, base, offset);
}

/// `rx86.py` `MOVSX8_rm` — `movsx r64, r/m8`.
pub fn movsx8_rm(mc: &mut impl DynasmApi, dst: u8, base: u8, offset: i32) {
    gpr_rm(mc, &[], true, &[0x0f, 0xbe], dst, base, offset);
}

/// `rx86.py` `MOVSX16_rm` — `movsx r64, r/m16`.
pub fn movsx16_rm(mc: &mut impl DynasmApi, dst: u8, base: u8, offset: i32) {
    gpr_rm(mc, &[], true, &[0x0f, 0xbf], dst, base, offset);
}

/// `rx86.py` `MOVSX32_rm` — `movsxd r64, r/m32`.
pub fn movsx32_rm(mc: &mut impl DynasmApi, dst: u8, base: u8, offset: i32) {
    gpr_rm(mc, &[], true, &[0x63], dst, base, offset);
}

/// `rx86.py` `MOVZX8_ra`.
pub fn movzx8_ra(mc: &mut impl DynasmApi, dst: u8, base: u8, index: u8, scale: u8, offset: i32) {
    gpr_ra(
        mc,
        &[],
        true,
        &[0x0f, 0xb6],
        dst,
        base,
        index,
        scale,
        offset,
    );
}

/// `rx86.py` `MOVZX16_ra`.
pub fn movzx16_ra(mc: &mut impl DynasmApi, dst: u8, base: u8, index: u8, scale: u8, offset: i32) {
    gpr_ra(
        mc,
        &[],
        true,
        &[0x0f, 0xb7],
        dst,
        base,
        index,
        scale,
        offset,
    );
}

/// `rx86.py` `MOVSX8_ra`.
pub fn movsx8_ra(mc: &mut impl DynasmApi, dst: u8, base: u8, index: u8, scale: u8, offset: i32) {
    gpr_ra(
        mc,
        &[],
        true,
        &[0x0f, 0xbe],
        dst,
        base,
        index,
        scale,
        offset,
    );
}

/// `rx86.py` `MOVSX16_ra`.
pub fn movsx16_ra(mc: &mut impl DynasmApi, dst: u8, base: u8, index: u8, scale: u8, offset: i32) {
    gpr_ra(
        mc,
        &[],
        true,
        &[0x0f, 0xbf],
        dst,
        base,
        index,
        scale,
        offset,
    );
}

/// `rx86.py` `MOVSX32_ra`.
pub fn movsx32_ra(mc: &mut impl DynasmApi, dst: u8, base: u8, index: u8, scale: u8, offset: i32) {
    gpr_ra(mc, &[], true, &[0x63], dst, base, index, scale, offset);
}

/// `rx86.py` `MOVSD_xm` — `movsd xmm, m64`.
pub fn movsd_xm(mc: &mut impl DynasmApi, dst: u8, base: u8, offset: i32) {
    xmm_xm(mc, 0xf2, 0x10, dst, base, offset);
}

/// `rx86.py` `MOVSD_mx` — `movsd m64, xmm`.
pub fn movsd_mx(mc: &mut impl DynasmApi, base: u8, offset: i32, src: u8) {
    xmm_xm(mc, 0xf2, 0x11, src, base, offset);
}

/// `rx86.py` `MOVSS_xm` — `movss xmm, m32`.
pub fn movss_xm(mc: &mut impl DynasmApi, dst: u8, base: u8, offset: i32) {
    xmm_xm(mc, 0xf3, 0x10, dst, base, offset);
}

/// `rx86.py` `MOVSS_mx` — `movss m32, xmm`.
pub fn movss_mx(mc: &mut impl DynasmApi, base: u8, offset: i32, src: u8) {
    xmm_xm(mc, 0xf3, 0x11, src, base, offset);
}

/// `rx86.py` `MOVSD_xa`.
pub fn movsd_xa(mc: &mut impl DynasmApi, dst: u8, base: u8, index: u8, scale: u8, offset: i32) {
    xmm_xa(mc, 0xf2, 0x10, dst, base, index, scale, offset);
}

/// `rx86.py` `MOVSD_ax`.
pub fn movsd_ax(mc: &mut impl DynasmApi, base: u8, index: u8, scale: u8, offset: i32, src: u8) {
    xmm_xa(mc, 0xf2, 0x11, src, base, index, scale, offset);
}

/// `rx86.py` `MOVSS_xa`.
pub fn movss_xa(mc: &mut impl DynasmApi, dst: u8, base: u8, index: u8, scale: u8, offset: i32) {
    xmm_xa(mc, 0xf3, 0x10, dst, base, index, scale, offset);
}

/// `rx86.py` `MOVSS_ax`.
pub fn movss_ax(mc: &mut impl DynasmApi, base: u8, index: u8, scale: u8, offset: i32, src: u8) {
    xmm_xa(mc, 0xf3, 0x11, src, base, index, scale, offset);
}

/// `rx86.py` `MOVUPS_mx` — `movups m128, xmm`.
pub fn movups_mx(mc: &mut impl DynasmApi, base: u8, offset: i32, src: u8) {
    emit_rex(
        mc,
        rex_register(src, 8) | rex_mem_reg_plus_const(base),
        false,
        false,
    );
    mc.push(0x0f);
    mc.push(0x11);
    encode_mem_reg_plus_const(mc, base, offset, reg_field(src));
}

/// `rx86.py` `TEST8_mi` — `test r/m8, imm8`.
pub fn test8_mi(mc: &mut impl DynasmApi, base: u8, offset: i32, imm: i8) {
    mem_imm(mc, &[], false, false, 0xf6, 0, base, offset);
    emit_imm8(mc, imm);
}

/// `rx86.py` `CMP_mi` — `cmp r/m64, imm8/imm32` (`select_8_or_32_bit_immed`).
pub fn cmp_mi(mc: &mut impl DynasmApi, base: u8, offset: i32, imm: i32) {
    if single_byte(imm) {
        mem_imm(mc, &[], true, false, 0x83, 7, base, offset);
        emit_imm8(mc, imm as i8);
    } else {
        mem_imm(mc, &[], true, false, 0x81, 7, base, offset);
        emit_imm32(mc, imm);
    }
}

/// `rx86.py` `CMP_rm` — `cmp r64, r/m64`.
pub fn cmp_rm(mc: &mut impl DynasmApi, reg: u8, base: u8, offset: i32) {
    gpr_rm(mc, &[], true, &[0x3b], reg, base, offset);
}

/// `rx86.py` `CMP_mr` — `cmp r/m64, r64`.
pub fn cmp_mr(mc: &mut impl DynasmApi, base: u8, offset: i32, reg: u8) {
    gpr_rm(mc, &[], true, &[0x39], reg, base, offset);
}

/// `rx86.py` `SUB_rm` — `sub r64, r/m64`.
pub fn sub_rm(mc: &mut impl DynasmApi, reg: u8, base: u8, offset: i32) {
    gpr_rm(mc, &[], true, &[0x2b], reg, base, offset);
}

/// `rx86.py` `SUB_mi8` / `SUB` memory immediate (`common_modes` group 5).
/// An immediate that fits in a signed byte uses `83 /5 ib`; otherwise `81 /5 id`.
pub fn sub_mi(mc: &mut impl DynasmApi, base: u8, offset: i32, imm: i32) {
    if single_byte(imm) {
        mem_imm(mc, &[], true, false, 0x83, 5, base, offset);
        emit_imm8(mc, imm as i8);
    } else {
        mem_imm(mc, &[], true, false, 0x81, 5, base, offset);
        emit_imm32(mc, imm);
    }
}

/// `rx86.py` `OR8_mi` — `or r/m8, imm8`.
pub fn or8_mi(mc: &mut impl DynasmApi, base: u8, offset: i32, imm: i8) {
    mem_imm(mc, &[], false, false, 0x80, 1, base, offset);
    emit_imm8(mc, imm);
}
