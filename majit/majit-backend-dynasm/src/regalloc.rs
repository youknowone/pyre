/// regalloc.py: Register allocation for the dynasm backend.
///
/// Classes:
///   X86RegisterManager — GPR allocation
///   X86XMMRegisterManager — XMM (float) allocation
///   X86FrameManager — frame slot allocation
///   RegAlloc — main allocator (walk ops, allocate regs)
use crate::regloc::*;
use majit_ir::{Op, OpRef, Type};
use std::collections::HashMap;

/// regalloc.py:1310 RegAlloc — main register allocator.
pub struct RegAlloc {
    /// GPR manager
    pub rm: RegisterManager,
    /// XMM manager
    pub xrm: RegisterManager,
    /// Frame manager
    pub fm: FrameManager,
    /// OpRef → location mapping
    longevity: HashMap<OpRef, (usize, usize)>,
}

/// regalloc.py RegisterManager — tracks which regs are free/allocated.
pub struct RegisterManager {
    /// Available registers for allocation.
    pub free_regs: Vec<RegLoc>,
    /// OpRef → register mapping.
    pub reg_bindings: HashMap<OpRef, RegLoc>,
    /// All allocatable registers.
    pub all_regs: Vec<RegLoc>,
}

/// regalloc.py X86FrameManager — frame slot allocation.
pub struct FrameManager {
    /// Next available frame slot index.
    next_slot: usize,
    /// OpRef → frame slot mapping.
    bindings: HashMap<OpRef, usize>,
}

impl RegAlloc {
    pub fn new() -> Self {
        // regalloc.py:1320-1330 — x86_64 allocatable registers
        // Callee-saved: rbx, r12-r15 (not allocatable without save/restore)
        // Caller-saved: rax, rcx, rdx, rsi, rdi, r8-r10 (freely allocatable)
        // Reserved: rsp (stack), rbp (frame), r11 (scratch)
        let gpr_regs = vec![
            EAX, ECX, EDX, EBX, ESI, EDI, R8, R9, R10, R12, R13, R14, R15,
        ];
        let xmm_regs = vec![
            XMM0, XMM1, XMM2, XMM3, XMM4, XMM5, XMM6, XMM7, XMM8, XMM9, XMM10, XMM11, XMM12, XMM13,
            XMM14, XMM15,
        ];

        RegAlloc {
            rm: RegisterManager::new(gpr_regs),
            xrm: RegisterManager::new(xmm_regs),
            fm: FrameManager::new(),
            longevity: HashMap::new(),
        }
    }

    /// regalloc.py consider_* dispatcher.
    pub fn walk_operations(&mut self, _ops: &[Op]) {
        // TODO: compute longevity, walk ops, call consider_*
    }

    /// regalloc.py:make_sure_var_in_reg
    pub fn make_sure_var_in_reg(&mut self, opref: OpRef, tp: Type) -> Loc {
        if tp == Type::Float {
            if let Some(reg) = self.xrm.get_reg(opref) {
                return Loc::Reg(reg);
            }
            let reg = self.xrm.allocate(opref);
            Loc::Reg(reg)
        } else {
            if let Some(reg) = self.rm.get_reg(opref) {
                return Loc::Reg(reg);
            }
            let reg = self.rm.allocate(opref);
            Loc::Reg(reg)
        }
    }

    /// regalloc.py:loc — get current location of an OpRef.
    pub fn loc(&self, opref: OpRef) -> Loc {
        if let Some(reg) = self.rm.get_reg(opref) {
            return Loc::Reg(reg);
        }
        if let Some(reg) = self.xrm.get_reg(opref) {
            return Loc::Reg(reg);
        }
        if let Some(slot) = self.fm.get_slot(opref) {
            return Loc::Frame(FrameLoc::new(slot, (slot as i32) * 8, false));
        }
        // Constants → immediate
        Loc::Immed(ImmedLoc::new(0))
    }
}

impl RegisterManager {
    pub fn new(all_regs: Vec<RegLoc>) -> Self {
        let free = all_regs.clone();
        RegisterManager {
            free_regs: free,
            reg_bindings: HashMap::new(),
            all_regs,
        }
    }

    pub fn get_reg(&self, opref: OpRef) -> Option<RegLoc> {
        self.reg_bindings.get(&opref).copied()
    }

    pub fn allocate(&mut self, opref: OpRef) -> RegLoc {
        let reg = self.free_regs.pop().expect("no free registers");
        self.reg_bindings.insert(opref, reg);
        reg
    }

    pub fn free(&mut self, opref: OpRef) {
        if let Some(reg) = self.reg_bindings.remove(&opref) {
            self.free_regs.push(reg);
        }
    }
}

impl FrameManager {
    pub fn new() -> Self {
        FrameManager {
            next_slot: 0,
            bindings: HashMap::new(),
        }
    }

    pub fn get_slot(&self, opref: OpRef) -> Option<usize> {
        self.bindings.get(&opref).copied()
    }

    pub fn allocate(&mut self, opref: OpRef) -> usize {
        let slot = self.next_slot;
        self.next_slot += 1;
        self.bindings.insert(opref, slot);
        slot
    }
}
