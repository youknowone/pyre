//! codebuf.py: Code buffer management.
//!
//! In RPython, MachineCodeBlockWrapper wraps rx86 code builders with block
//! management. Here dynasm emits into a temporary mapping, which is copied
//! into the CPU's `AsmMemoryManager` to return the final arena-owned buffer.

/// codebuf.py MachineCodeBlockWrapper
/// The arena/free-list ownership below is the `MachineCodeBlockWrapper` role.
use std::sync::Arc;

use dynasmrt::relocations::Relocation;
use dynasmrt::{AssemblyOffset, DynasmApi};
use majit_backend::{AsmMemoryBlock, AsmMemoryManager, BackendError};

/// Executable code owned by pyre's reusable assembler arena.
pub struct ArenaExecutableBuffer {
    block: AsmMemoryBlock,
}

impl ArenaExecutableBuffer {
    pub fn ptr(&self, offset: AssemblyOffset) -> *const u8 {
        assert!(offset.0 <= self.block.len());
        unsafe { self.block.ptr().add(offset.0) }
    }

    pub fn len(&self) -> usize {
        self.block.len()
    }

    pub fn is_empty(&self) -> bool {
        self.block.is_empty()
    }

    pub fn size(&self) -> usize {
        self.block.capacity()
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.block.as_mut_slice()
    }

    pub fn make_executable(&mut self) -> Result<(), BackendError> {
        self.block
            .make_executable()
            .map_err(|error| BackendError::CompilationFailed(error.to_string()))
    }
}

impl std::ops::Deref for ArenaExecutableBuffer {
    type Target = [u8];

    fn deref(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.block.ptr(), self.block.len()) }
    }
}

/// Copy the assembled code into an arena-owned RW block. The caller may patch
/// backend placeholders before switching the block to RX.
///
/// Relocating finalized code to a different address is only sound while no
/// relocation encodes an address. dynasm keeps those (`RelToAbs`, `AbsToRel`)
/// in a private `managed` list and re-applies them when it moves its own
/// buffer; nothing outside that crate can re-apply them here. This backend
/// emits neither kind: aarch64 encodes every jump target PC-relative, and the
/// x86 emitter uses no `extern` targets and no 8-byte label operands — the two
/// forms that produce one. `tests/relocation_guard.rs` holds that in place.
pub fn finalize_writable<R: Relocation>(
    mut assembler: dynasmrt::Assembler<R>,
    arena: &Arc<AsmMemoryManager>,
) -> Result<ArenaExecutableBuffer, BackendError> {
    let len = assembler.offset().0;
    // Surface relocation errors as an `Err`; `finalize` panics on them.
    assembler
        .commit()
        .map_err(|error| BackendError::CompilationFailed(error.to_string()))?;
    let source = assembler.finalize().map_err(|_| {
        BackendError::CompilationFailed("assembler is still borrowed by an Executor".to_string())
    })?;
    let mut block = arena
        .allocate(len, len)
        .map_err(|error| BackendError::CompilationFailed(error.to_string()))?;
    {
        // `copy_to_raw_memory` runs inside the caller's bracket when there is
        // one (`assemble_loop` / `assemble_bridge`); the slow-path stubs built
        // at `setup_once` have none, so open one here too.
        let _writing = majit_backend::AssemblerWriting::enter();
        block.as_mut_slice()[..len].copy_from_slice(&source[..len]);
    }
    Ok(ArenaExecutableBuffer { block })
}

pub fn finalize_executable<R: Relocation>(
    assembler: dynasmrt::Assembler<R>,
    arena: &Arc<AsmMemoryManager>,
) -> Result<ArenaExecutableBuffer, BackendError> {
    let mut buffer = finalize_writable(assembler, arena)?;
    buffer.make_executable()?;
    Ok(buffer)
}

/// Patch finished code in place: `rmmap.enter_assembler_writing()` around the
/// write (`aarch64/runner.py` `invalidate_loop`), then `clear_cache` over the
/// patched bytes (`aarch64/codebuilder.py` `copy_to_raw_memory`). The mapping
/// is RWX throughout, so no neighbouring block changes protection.
pub fn with_writable<F: FnOnce()>(addr: *mut u8, len: usize, f: F) {
    {
        let _writing = majit_backend::AssemblerWriting::enter();
        f();
    }
    majit_backend::flush_instruction_cache(addr as *const u8, len);
}

/// Get the raw pointer to an arena buffer's code.
pub fn buffer_ptr(buffer: &ArenaExecutableBuffer) -> *const u8 {
    buffer.ptr(dynasmrt::AssemblyOffset(0))
}
