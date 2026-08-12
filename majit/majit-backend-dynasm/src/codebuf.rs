//! codebuf.py: Code buffer management.
//!
//! In RPython, MachineCodeBlockWrapper wraps rx86 code builders with block
//! management. Here dynasm emits into a temporary mapping, which is copied
//! into the CPU's `AsmMemoryManager` to return the final arena-owned buffer.

/// codebuf.py:34 MachineCodeBlockWrapper
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
    block.as_mut_slice()[..len].copy_from_slice(&source[..len]);
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

/// Make a memory region writable for in-place patching, execute the
/// closure, then restore execute permission. Arena blocks are page-isolated,
/// so this does not change a neighboring live block's protection.
pub fn with_writable<F: FnOnce()>(addr: *mut u8, len: usize, f: F) {
    #[cfg(unix)]
    {
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        let page_start = (addr as usize) & !(page_size - 1);
        let page_end = ((addr as usize + len) + page_size - 1) & !(page_size - 1);
        let mprotect_len = page_end - page_start;
        let page_ptr = page_start as *mut libc::c_void;

        let rc1 =
            unsafe { libc::mprotect(page_ptr, mprotect_len, libc::PROT_READ | libc::PROT_WRITE) };
        assert!(
            rc1 == 0,
            "[dynasm] mprotect RW failed: addr={:?} len={} errno={}",
            page_ptr,
            mprotect_len,
            std::io::Error::last_os_error()
        );

        struct RestoreRx {
            ptr: *mut libc::c_void,
            len: usize,
        }
        impl Drop for RestoreRx {
            fn drop(&mut self) {
                let rc = unsafe {
                    libc::mprotect(self.ptr, self.len, libc::PROT_READ | libc::PROT_EXEC)
                };
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
        let _guard = RestoreRx {
            ptr: page_ptr,
            len: mprotect_len,
        };
        f();
    }

    #[cfg(windows)]
    {
        use std::os::windows::raw::HANDLE;

        // Win64 SYSTEM_INFO is 48 bytes; we only read dwPageSize at offset 4.
        #[repr(C)]
        struct SystemInfo {
            _arch: u32,
            dw_page_size: u32,
            _rest: [u8; 40],
        }

        let page_size = {
            let mut si = std::mem::MaybeUninit::<SystemInfo>::zeroed();
            unsafe { GetSystemInfo(si.as_mut_ptr()) };
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
            fn GetSystemInfo(info: *mut SystemInfo);
        }

        let mut old_protect: u32 = 0;
        let rc1 = unsafe { VirtualProtect(page_ptr, region_len, PAGE_READWRITE, &mut old_protect) };
        assert!(
            rc1 != 0,
            "[dynasm] VirtualProtect RW failed: addr={:?} len={} err={}",
            page_ptr,
            region_len,
            std::io::Error::last_os_error()
        );

        struct RestoreRx {
            ptr: *mut u8,
            len: usize,
        }
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
                    self.ptr,
                    self.len,
                    std::io::Error::last_os_error()
                );
                unsafe { FlushInstructionCache(GetCurrentProcess(), self.ptr, self.len) };
            }
        }
        let _guard = RestoreRx {
            ptr: page_ptr,
            len: region_len,
        };
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

/// Get the raw pointer to an arena buffer's code.
pub fn buffer_ptr(buffer: &ArenaExecutableBuffer) -> *const u8 {
    buffer.ptr(dynasmrt::AssemblyOffset(0))
}
