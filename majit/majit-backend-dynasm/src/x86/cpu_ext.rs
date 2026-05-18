//! x86-specific per-CPU assembler state held by `DynasmBackend`.
//!
//! PyPy stores `self.malloc_slowpath` / `self.propagate_exception_path`
//! on `Assembler386` (`rpython/jit/backend/x86/assembler.py:63,344`);
//! the assembler is one-per-CPU and lives for the CPU's lifetime, so
//! the trampolines built at `setup_once` (`llsupport/assembler.py:124-138`)
//! persist on it.
//!
//! Pyre's `Assembler386` is constructed per-`compile_loop`/`compile_bridge`
//! (`runner.rs::compile_loop`, `compile_bridge`), so the per-CPU stash
//! moves up one level to `DynasmBackend` via this struct.  Aarch64 has
//! its own equivalent (`aarch64::cpu_ext::Aarch64CpuExt`) which is
//! currently a no-op placeholder — aarch64 inlines the slowpath
//! sequences today and has no per-CPU trampoline to memoise.

use crate::guard::CpuDescrHandle;

/// Lazily-materialised per-CPU x86 trampoline addresses.
///
/// Both fields are set once on first use and reused for every
/// subsequent `compile_loop` / `compile_bridge` on this CPU.  The
/// underlying executable buffers are `mem::forget`-leaked into
/// executable memory inside `build_*`, matching PyPy's `asmmemmgr`
/// rooting the buffer for the CPU's lifetime.
pub(crate) struct X86CpuExt {
    /// `assembler.py:63 self.malloc_slowpath` parity.  Address of the
    /// fixed-size malloc slowpath helper built by
    /// `build_malloc_slowpath_fixed`.
    malloc_slowpath_fixed: Option<usize>,
    /// `assembler.py:344 self.propagate_exception_path` parity.
    /// Standalone trampoline that the malloc slowpath (and, in PyPy,
    /// the stack check slowpath) JMPs to on OOM / propagate.
    propagate_exception_path: Option<usize>,
}

impl X86CpuExt {
    pub(crate) fn new() -> Self {
        Self {
            malloc_slowpath_fixed: None,
            propagate_exception_path: None,
        }
    }

    /// `assembler.py:328 _build_propagate_exception_path` parity:
    /// materialise the standalone propagate trampoline that
    /// `_store_and_reset_exception`s, writes `jf_guard_exc` / `jf_descr`,
    /// and tail-calls `_call_footer`.  The malloc slowpath (and, in
    /// PyPy, the stack check slowpath) JMP into this single entry
    /// point.  Materialised lazily; the address is then memoised here
    /// so every slowpath built on this CPU shares the same propagate
    /// path (matches PyPy's `self.propagate_exception_path` attribute).
    pub(crate) fn ensure_propagate_exception_path(&mut self, cpu_handle: &CpuDescrHandle) -> usize {
        if let Some(addr) = self.propagate_exception_path {
            return addr;
        }
        let addr = super::assembler::build_propagate_exception_path(cpu_handle);
        self.propagate_exception_path = Some(addr);
        addr
    }

    /// `assembler.py:231 _build_malloc_slowpath` parity: materialise
    /// the fixed-size malloc slowpath helper on first use and stash
    /// its address here.  Subsequent `compile_loop` / `compile_bridge`
    /// invocations reuse the same helper, matching PyPy's
    /// `setup_once` semantics where the helper is built once per CPU
    /// and referenced as `self.malloc_slowpath` thereafter.
    ///
    /// Ensures the propagate trampoline exists first so the slowpath's
    /// OOM branch can `JMP` to it (matches PyPy's `setup_once` ordering:
    /// `_build_propagate_exception_path` then `_build_malloc_slowpath`).
    pub(crate) fn ensure_malloc_slowpath_fixed(&mut self, cpu_handle: &CpuDescrHandle) -> usize {
        if let Some(addr) = self.malloc_slowpath_fixed {
            return addr;
        }
        let propagate_path = self.ensure_propagate_exception_path(cpu_handle);
        let addr = super::assembler::build_malloc_slowpath_fixed(cpu_handle, propagate_path);
        self.malloc_slowpath_fixed = Some(addr);
        addr
    }
}
