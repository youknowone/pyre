//! codebuf.py: Code buffer management.
//!
//! In RPython, MachineCodeBlockWrapper wraps rx86 code builders with block
//! management. Here dynasm emits into a byte vector (`VecAssembler`), which is
//! copied into the CPU's `AsmMemoryManager` to return the final arena-owned
//! buffer (`copy_to_raw_memory`).

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
/// Placing finalized code at an address other than the one it was assembled
/// for (`VecAssembler::new(0)`) is only sound while no relocation encodes an
/// address: dynasm resolves those (`RelToAbs`, `AbsToRel`) against the base
/// address it was given, and nothing here re-applies them. This backend
/// emits neither kind: aarch64 encodes every jump target PC-relative, and the
/// x86 emitter uses no `extern` targets and no 8-byte label operands — the two
/// forms that produce one. `tests/relocation_guard.rs` holds that in place.
pub fn finalize_writable<R: Relocation>(
    assembler: dynasmrt::VecAssembler<R>,
    arena: &Arc<AsmMemoryManager>,
) -> Result<ArenaExecutableBuffer, BackendError> {
    let len = assembler.offset().0;
    // Surface relocation errors as an `Err`.
    let source = assembler
        .finalize()
        .map_err(|error| BackendError::CompilationFailed(error.to_string()))?;
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
    assembler: dynasmrt::VecAssembler<R>,
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

/// `x86/runner.py invalidate_loop` / `aarch64/runner.py invalidate_loop` —
/// activate the `GUARD_NOT_INVALIDATED` sites recorded for a loop and its
/// attached bridges by writing the branch to each guard's recovery stub over
/// the placeholder the emitter left there.
///
/// Upstream aarch64 brackets the same walk with `rmmap.enter_assembler_writing()`
/// / `leave_assembler_writing()`; [`with_writable`] is that bracket.  One
/// naturally-aligned four-byte store per site and nothing else, so a thread
/// executing the loop reads either the placeholder or the finished branch:
/// on x86-64 the four bytes are the `JMP rel32` displacement, on aarch64 they
/// are the whole `B imm26` instruction word, one of the encodings the
/// architecture names as safe to modify while another PE executes it.
///
/// The store is what has to be indivisible; the page being RW around it is a
/// property of [`with_writable`] itself, shared with the bridge attachment in
/// `patch_jump_for_descr`, which patches live code the same way.
pub fn write_invalidate_positions(positions: &[majit_backend::InvalidatePosition]) {
    for position in positions {
        debug_assert_eq!(
            position.addr % 4,
            0,
            "invalidate position {:#x} is not word-aligned; the store would not be atomic",
            position.addr
        );
        with_writable(position.addr as *mut u8, 4, || unsafe {
            // Require one indivisible word store even when LLVM knows more
            // about the instruction bytes than the native code reader does.
            // The cache flush in with_writable publishes it to instruction fetch.
            std::sync::atomic::AtomicU32::from_ptr(position.addr as *mut u32)
                .store(position.word, std::sync::atomic::Ordering::Relaxed);
        });
    }
}

/// Get the raw pointer to an arena buffer's code.
pub fn buffer_ptr(buffer: &ArenaExecutableBuffer) -> *const u8 {
    buffer.ptr(dynasmrt::AssemblyOffset(0))
}
