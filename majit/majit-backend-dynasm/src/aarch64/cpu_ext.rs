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
    /// `AssemblerARM64.wb_slowpath`; `0` where no helper is built yet.
    wb_slowpath: [usize; 5],
    _wb_slowpath_buffers: Vec<ArenaExecutableBuffer>,
}

impl Aarch64CpuExt {
    pub(crate) fn new(asm_memory_manager: Arc<AsmMemoryManager>) -> Self {
        Self {
            asm_memory_manager,
            malloc_slowpath_fixed: None,
            _malloc_slowpath_fixed_buffer: None,
            propagate_exception_path: None,
            _propagate_exception_path_buffer: None,
            wb_slowpath: [0; 5],
            _wb_slowpath_buffers: Vec::new(),
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

    /// `aarch64/assembler.py setup_once`: `_build_wb_slowpath` for every
    /// `withcards`/`withfloats` pair and the `for_frame` helper.
    pub(crate) fn ensure_wb_slowpath(&mut self) -> [usize; 5] {
        // `_build_wb_slowpath(False)`, `(True)`, `(False, for_frame=True)`,
        // then the `withfloats=True` pair.
        for (withcards, withfloats, for_frame) in [
            (false, false, false),
            (true, false, false),
            (false, false, true),
            (false, true, false),
            (true, true, false),
        ] {
            let helper_num = if for_frame {
                4
            } else {
                usize::from(withcards) + 2 * usize::from(withfloats)
            };
            // `_write_barrier_fastpath`: `if self.wb_slowpath[helper_num] ==
            // 0` builds it there. A collector installed after the first
            // build (`set_gc_allocator`) can be the first with a write
            // barrier, so an entry left at 0 is retried, never cached.
            if self.wb_slowpath[helper_num] != 0 {
                continue;
            }
            let Some((buffer, addr)) = super::assembler::build_wb_slowpath(
                withcards,
                withfloats,
                for_frame,
                &self.asm_memory_manager,
            ) else {
                continue;
            };
            self.wb_slowpath[helper_num] = addr;
            self._wb_slowpath_buffers.push(buffer);
        }
        self.wb_slowpath
    }

    pub(crate) fn has_propagate_dependent_caches(&self) -> bool {
        self.malloc_slowpath_fixed.is_some() || self.propagate_exception_path.is_some()
    }
}
