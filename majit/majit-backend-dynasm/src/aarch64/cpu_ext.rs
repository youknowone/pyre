//! aarch64-specific per-CPU assembler state held by `DynasmBackend`.
//!
//! PyPy stores `self.malloc_slowpath` and
//! `self.propagate_exception_path` on its one `AssemblerARM64` per CPU.
//! Pyre creates a trace assembler per compilation, so these code buffers live
//! one level higher here and every loop/bridge receives the cached address.

use crate::codebuf::ArenaExecutableBuffer;
use crate::guard::CpuDescrHandle;
use majit_backend::AsmMemoryManager;
use std::sync::Arc;

pub(crate) struct Aarch64CpuExt {
    asm_memory_manager: Arc<AsmMemoryManager>,
    malloc_slowpath_fixed: Option<usize>,
    _malloc_slowpath_fixed_buffer: Option<ArenaExecutableBuffer>,
    propagate_exception_path: Option<usize>,
    _propagate_exception_path_buffer: Option<ArenaExecutableBuffer>,
}

impl Aarch64CpuExt {
    pub(crate) fn new(asm_memory_manager: Arc<AsmMemoryManager>) -> Self {
        Self {
            asm_memory_manager,
            malloc_slowpath_fixed: None,
            _malloc_slowpath_fixed_buffer: None,
            propagate_exception_path: None,
            _propagate_exception_path_buffer: None,
        }
    }

    fn ensure_propagate_exception_path(&mut self, cpu_handle: &CpuDescrHandle) -> usize {
        if let Some(addr) = self.propagate_exception_path {
            return addr;
        }
        let (buffer, addr) =
            super::assembler::build_propagate_exception_path(cpu_handle, &self.asm_memory_manager);
        debug_assert_ne!(addr, 0);
        self._propagate_exception_path_buffer = Some(buffer);
        self.propagate_exception_path = Some(addr);
        addr
    }

    /// `aarch64/assembler.py setup_once` / `_build_malloc_slowpath`: build
    /// once per CPU and reuse for fixed and varsize-frame nursery probes.
    pub(crate) fn ensure_malloc_slowpath_fixed(&mut self, cpu_handle: &CpuDescrHandle) -> usize {
        if let Some(addr) = self.malloc_slowpath_fixed {
            return addr;
        }
        let propagate_path = self.ensure_propagate_exception_path(cpu_handle);
        let (buffer, addr) =
            super::assembler::build_malloc_slowpath_fixed(propagate_path, &self.asm_memory_manager);
        debug_assert_ne!(addr, 0);
        self._malloc_slowpath_fixed_buffer = Some(buffer);
        self.malloc_slowpath_fixed = Some(addr);
        addr
    }

    pub(crate) fn has_propagate_dependent_caches(&self) -> bool {
        self.malloc_slowpath_fixed.is_some() || self.propagate_exception_path.is_some()
    }
}
