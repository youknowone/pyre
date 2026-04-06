/// codebuf.py: Code buffer management.
///
/// In RPython, MachineCodeBlockWrapper wraps rx86 code builders with
/// block management. In the dynasm backend, dynasmrt::Assembler
/// handles this natively — this module provides any extra utilities.

/// codebuf.py:34 MachineCodeBlockWrapper
/// dynasm-rs handles code buffer management internally.
/// This module is kept for RPython naming parity and any extra utilities.
use dynasmrt::ExecutableBuffer;

/// Make a memory region writable for patching, execute the closure,
/// then restore execute permission.
///
/// On macOS ARM64, W^X requires explicit permission toggling.
/// On x86_64 Linux, mprotect is used.
pub fn with_writable<F: FnOnce()>(addr: *mut u8, len: usize, f: F) {
    // For now, assume memory is already writable (dynasm manages this).
    // TODO: implement proper W^X handling for aarch64 macOS
    f();
}

/// Get the raw pointer to an ExecutableBuffer's code.
pub fn buffer_ptr(buffer: &ExecutableBuffer) -> *const u8 {
    buffer.ptr(dynasmrt::AssemblyOffset(0))
}
