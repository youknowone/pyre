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
/// Make a memory region writable for in-place patching, execute the
/// closure, then restore execute permission. dynasmrt uses memmap2
/// which maps executable pages as PROT_READ|PROT_EXEC after finalize,
/// so we must use mprotect to toggle.
pub fn with_writable<F: FnOnce()>(addr: *mut u8, len: usize, f: F) {
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let page_start = (addr as usize) & !(page_size - 1);
    let page_end = ((addr as usize + len) + page_size - 1) & !(page_size - 1);
    let mprotect_len = page_end - page_start;
    let page_ptr = page_start as *mut libc::c_void;

    // macOS aarch64 disallows W+X. Switch to RW, patch, then RX.
    let rc1 = unsafe { libc::mprotect(page_ptr, mprotect_len, libc::PROT_READ | libc::PROT_WRITE) };
    if rc1 != 0 {
        eprintln!(
            "[dynasm] mprotect RW failed: addr={:?} len={} errno={}",
            page_ptr,
            mprotect_len,
            std::io::Error::last_os_error()
        );
    }
    f();
    let rc2 = unsafe { libc::mprotect(page_ptr, mprotect_len, libc::PROT_READ | libc::PROT_EXEC) };
    if rc2 != 0 {
        eprintln!(
            "[dynasm] mprotect RX failed: addr={:?} len={} errno={}",
            page_ptr,
            mprotect_len,
            std::io::Error::last_os_error()
        );
    }

    #[cfg(target_arch = "aarch64")]
    {
        // Flush instruction cache after patching.
        flush_icache_range(page_start as *const u8, mprotect_len);
    }
}

#[cfg(target_arch = "aarch64")]
fn flush_icache_range(addr: *const u8, len: usize) {
    #[cfg(target_os = "macos")]
    {
        unsafe extern "C" {
            fn sys_icache_invalidate(start: *mut u8, size: usize);
        }
        unsafe { sys_icache_invalidate(addr as *mut u8, len) };
    }
    #[cfg(target_os = "linux")]
    {
        unsafe extern "C" {
            fn __clear_cache(start: *mut u8, end: *mut u8);
        }
        unsafe { __clear_cache(addr as *mut u8, (addr as *mut u8).add(len)) };
    }
}

/// Get the raw pointer to an ExecutableBuffer's code.
pub fn buffer_ptr(buffer: &ExecutableBuffer) -> *const u8 {
    buffer.ptr(dynasmrt::AssemblyOffset(0))
}
