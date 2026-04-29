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
    #[cfg(unix)]
    {
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let page_start = (addr as usize) & !(page_size - 1);
        let page_end = ((addr as usize + len) + page_size - 1) & !(page_size - 1);
        let mprotect_len = page_end - page_start;
        let page_ptr = page_start as *mut libc::c_void;

        let rc1 = unsafe { libc::mprotect(page_ptr, mprotect_len, libc::PROT_READ | libc::PROT_WRITE) };
        assert!(
            rc1 == 0,
            "[dynasm] mprotect RW failed: addr={:?} len={} errno={}",
            page_ptr,
            mprotect_len,
            std::io::Error::last_os_error()
        );

        struct RestoreRx { ptr: *mut libc::c_void, len: usize }
        impl Drop for RestoreRx {
            fn drop(&mut self) {
                let rc = unsafe { libc::mprotect(self.ptr, self.len, libc::PROT_READ | libc::PROT_EXEC) };
                assert!(
                    rc == 0,
                    "[dynasm] mprotect RX failed: addr={:?} len={} errno={}",
                    self.ptr,
                    self.len,
                    std::io::Error::last_os_error()
                );
                #[cfg(target_arch = "aarch64")]
                {
                    flush_icache_range(self.ptr as *const u8, self.len);
                }
            }
        }
        let _guard = RestoreRx { ptr: page_ptr, len: mprotect_len };
        f();
    }

    #[cfg(windows)]
    {
        use std::os::windows::raw::HANDLE;

        let page_size = {
            #[repr(C)]
            struct SystemInfo { _pad: [u8; 4], dw_page_size: u32, _rest: [u8; 60] }
            let mut si = std::mem::MaybeUninit::<SystemInfo>::zeroed();
            unsafe { GetSystemInfo(si.as_mut_ptr() as *mut u8) };
            unsafe { si.assume_init().dw_page_size as usize }
        };
        let page_start = (addr as usize) & !(page_size - 1);
        let page_end = ((addr as usize + len) + page_size - 1) & !(page_size - 1);
        let region_len = page_end - page_start;
        let page_ptr = page_start as *mut u8;

        const PAGE_READWRITE: u32 = 0x04;
        const PAGE_EXECUTE_READ: u32 = 0x20;

        unsafe extern "system" {
            fn VirtualProtect(addr: *mut u8, size: usize, new: u32, old: *mut u32) -> i32;
            fn GetSystemInfo(info: *mut u8);
        }

        let mut old_protect: u32 = 0;
        let rc1 = unsafe { VirtualProtect(page_ptr, region_len, PAGE_READWRITE, &mut old_protect) };
        assert!(
            rc1 != 0,
            "[dynasm] VirtualProtect RW failed: addr={:?} len={} err={}",
            page_ptr, region_len, std::io::Error::last_os_error()
        );

        struct RestoreRx { ptr: *mut u8, len: usize }
        impl Drop for RestoreRx {
            fn drop(&mut self) {
                const PAGE_EXECUTE_READ: u32 = 0x20;
                unsafe extern "system" {
                    fn VirtualProtect(addr: *mut u8, size: usize, new: u32, old: *mut u32) -> i32;
                    fn FlushInstructionCache(process: isize, addr: *const u8, size: usize) -> i32;
                    fn GetCurrentProcess() -> isize;
                }
                let mut old: u32 = 0;
                let rc = unsafe { VirtualProtect(self.ptr, self.len, PAGE_EXECUTE_READ, &mut old) };
                assert!(
                    rc != 0,
                    "[dynasm] VirtualProtect RX failed: addr={:?} len={} err={}",
                    self.ptr, self.len, std::io::Error::last_os_error()
                );
                unsafe { FlushInstructionCache(GetCurrentProcess(), self.ptr, self.len) };
            }
        }
        let _guard = RestoreRx { ptr: page_ptr, len: region_len };
        f();
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
