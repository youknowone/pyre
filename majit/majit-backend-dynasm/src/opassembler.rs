/// aarch64/opassembler.py parity: arch-specific operation emission.
///
/// RPython separates instruction emission from assembly control flow:
/// - `llsupport/assembler.py` → common (BaseAssembler)
/// - `aarch64/opassembler.py` → arch-specific emit_op_* (ResOpAssembler)
/// - `aarch64/assembler.py` → control flow (AssemblerARM64)
///
/// This module contains arch-gated emit helpers called from the
/// regalloc-aware dispatch in assembler.rs. Each method is a single
/// emit_op_* entry point with `#[cfg(target_arch)]` inside.
/// No `dynasm!(.arch ...)` should appear in assembler.rs outside of
/// already-guarded blocks.
use dynasmrt::{DynasmApi, dynasm};

use crate::assembler::Assembler386;
use crate::regloc::{Loc, RegLoc};

impl Assembler386 {
    // ── emit_op_setfield_gc ──
    // aarch64/opassembler.py: emit_op_setfield_gc
    // x86/assembler.py: genop_discard_setfield_gc

    pub(crate) fn emit_op_setfield_regalloc(
        &mut self,
        base: &RegLoc,
        val_loc: &Loc,
        ofs: i32,
        field_size: usize,
    ) {
        #[cfg(target_arch = "x86_64")]
        match val_loc {
            Loc::Reg(v) if v.is_xmm => {
                dynasm!(self.mc ; .arch x64 ; movsd [Rq(base.value) + ofs], Rx(v.value));
            }
            Loc::Reg(v) => match field_size {
                1 => dynasm!(self.mc ; .arch x64 ; mov BYTE [Rq(base.value) + ofs], Rb(v.value)),
                2 => dynasm!(self.mc ; .arch x64 ; mov WORD [Rq(base.value) + ofs], Rw(v.value)),
                4 => dynasm!(self.mc ; .arch x64 ; mov DWORD [Rq(base.value) + ofs], Rd(v.value)),
                _ => dynasm!(self.mc ; .arch x64 ; mov [Rq(base.value) + ofs], Rq(v.value)),
            },
            _ => {
                let scratch = crate::regloc::X86_64_SCRATCH_REG;
                self.regalloc_mov(val_loc, &Loc::Reg(scratch));
                dynasm!(self.mc ; .arch x64 ; mov [Rq(base.value) + ofs], Rq(scratch.value));
            }
        }
        #[cfg(target_arch = "aarch64")]
        match val_loc {
            Loc::Reg(v) if v.is_xmm => {
                dynasm!(self.mc ; .arch aarch64 ; str D(v.value), [X(base.value), ofs as u32]);
            }
            Loc::Reg(v) => match field_size {
                1 => {
                    dynasm!(self.mc ; .arch aarch64 ; strb W(v.value), [X(base.value), ofs as u32])
                }
                2 => {
                    dynasm!(self.mc ; .arch aarch64 ; strh W(v.value), [X(base.value), ofs as u32])
                }
                4 => dynasm!(self.mc ; .arch aarch64 ; str W(v.value), [X(base.value), ofs as u32]),
                _ => dynasm!(self.mc ; .arch aarch64 ; str X(v.value), [X(base.value), ofs as u32]),
            },
            _ => {
                let scratch = RegLoc::new(16, false);
                self.regalloc_mov(val_loc, &Loc::Reg(scratch));
                dynasm!(self.mc ; .arch aarch64 ; str X(scratch.value), [X(base.value), ofs as u32]);
            }
        }
    }

    // ── emit_op_gc_load ──
    // aarch64/opassembler.py: emit_op_gc_load_i/r/f

    pub(crate) fn emit_op_gcload_regalloc(&mut self, base: &RegLoc, ofs_loc: &Loc, dst: &RegLoc) {
        match ofs_loc {
            Loc::Immed(i) => {
                let o = i.value as i32;
                #[cfg(target_arch = "x86_64")]
                if dst.is_xmm {
                    dynasm!(self.mc ; .arch x64 ; movsd Rx(dst.value), [Rq(base.value) + o]);
                } else {
                    dynasm!(self.mc ; .arch x64 ; mov Rq(dst.value), [Rq(base.value) + o]);
                }
                #[cfg(target_arch = "aarch64")]
                if dst.is_xmm {
                    dynasm!(self.mc ; .arch aarch64 ; ldr D(dst.value), [X(base.value), o as u32]);
                } else {
                    dynasm!(self.mc ; .arch aarch64 ; ldr X(dst.value), [X(base.value), o as u32]);
                }
            }
            Loc::Reg(ofs_r) => {
                #[cfg(target_arch = "x86_64")]
                if dst.is_xmm {
                    dynasm!(self.mc ; .arch x64 ; movsd Rx(dst.value), [Rq(base.value) + Rq(ofs_r.value)]);
                } else {
                    dynasm!(self.mc ; .arch x64 ; mov Rq(dst.value), [Rq(base.value) + Rq(ofs_r.value)]);
                }
                #[cfg(target_arch = "aarch64")]
                if dst.is_xmm {
                    dynasm!(self.mc ; .arch aarch64 ; ldr D(dst.value), [X(base.value), X(ofs_r.value)]);
                } else {
                    dynasm!(self.mc ; .arch aarch64 ; ldr X(dst.value), [X(base.value), X(ofs_r.value)]);
                }
            }
            _ => {}
        }
    }

    // ── emit_op_gc_store ──
    // aarch64/opassembler.py: emit_op_gc_store

    pub(crate) fn emit_op_gcstore_regalloc(&mut self, base: &RegLoc, ofs_loc: &Loc, val: &RegLoc) {
        match ofs_loc {
            Loc::Immed(i) => {
                let o = i.value as i32;
                #[cfg(target_arch = "x86_64")]
                dynasm!(self.mc ; .arch x64 ; mov [Rq(base.value) + o], Rq(val.value));
                #[cfg(target_arch = "aarch64")]
                dynasm!(self.mc ; .arch aarch64 ; str X(val.value), [X(base.value), o as u32]);
            }
            Loc::Reg(ofs_r) => {
                #[cfg(target_arch = "x86_64")]
                dynasm!(self.mc ; .arch x64 ; mov [Rq(base.value) + Rq(ofs_r.value)], Rq(val.value));
                #[cfg(target_arch = "aarch64")]
                dynasm!(self.mc ; .arch aarch64 ; str X(val.value), [X(base.value), X(ofs_r.value)]);
            }
            _ => {}
        }
    }
}
