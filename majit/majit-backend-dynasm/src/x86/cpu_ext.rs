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
use dynasmrt::ExecutableBuffer;

/// Lazily-materialised per-CPU x86 trampolines.
///
/// Both addresses are set once on first use and reused for every
/// subsequent `compile_loop` / `compile_bridge` on this CPU.  The
/// owning `ExecutableBuffer`s are kept alongside so the RX pages are
/// unmapped exactly when the CPU is dropped — matching PyPy's
/// `asmmemmgr`, which roots helper buffers on the CPU.
pub(crate) struct X86CpuExt {
    /// `assembler.py:63 self.malloc_slowpath` parity.  Entry address
    /// of the fixed-size malloc slowpath helper built by
    /// `build_malloc_slowpath_fixed`; `_buffer` is the matching RX
    /// mapping kept for the lifetime of this struct.
    malloc_slowpath_fixed: Option<usize>,
    _malloc_slowpath_fixed_buffer: Option<ExecutableBuffer>,
    /// `assembler.py:344 self.propagate_exception_path` parity.
    /// Standalone trampoline that the malloc slowpath (and, in PyPy,
    /// the stack check slowpath) JMPs to on OOM / propagate.
    propagate_exception_path: Option<usize>,
    _propagate_exception_path_buffer: Option<ExecutableBuffer>,
}

impl X86CpuExt {
    pub(crate) fn new() -> Self {
        Self {
            malloc_slowpath_fixed: None,
            _malloc_slowpath_fixed_buffer: None,
            propagate_exception_path: None,
            _propagate_exception_path_buffer: None,
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
        let (buffer, addr) = super::assembler::build_propagate_exception_path(cpu_handle);
        debug_assert!(
            addr != 0,
            "build_propagate_exception_path returned NULL entry address — \
             dynasm finalize is expected to yield a non-zero buffer_ptr"
        );
        self._propagate_exception_path_buffer = Some(buffer);
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
        let (buffer, addr) =
            super::assembler::build_malloc_slowpath_fixed(cpu_handle, propagate_path);
        debug_assert!(
            addr != 0,
            "build_malloc_slowpath_fixed returned NULL entry address — \
             dynasm finalize is expected to yield a non-zero buffer_ptr"
        );
        self._malloc_slowpath_fixed_buffer = Some(buffer);
        self.malloc_slowpath_fixed = Some(addr);
        addr
    }

    /// Whether either trampoline that bakes `propagate_exception_descr`
    /// as an immediate has already been materialised.  Used by
    /// `DynasmBackend::set_propagate_exception_descr` to refuse a
    /// non-identical `Arc` swap after the bake: such a swap would
    /// leave previously-compiled loops/bridges (whose `JMP` immediates
    /// point at this buffer's RX pages and whose helpers carry the
    /// old descr pointer) referencing a now-orphaned descr.  PyPy
    /// attaches `propagate_exception_descr` once before
    /// `cpu.setup_once()` and never replaces it
    /// (`pyjitpl.py:2273-2283` precedes `pyjitpl.py:2292-2303`); pyre
    /// upholds the same invariant by panicking instead of dropping
    /// the buffer.
    pub(crate) fn has_propagate_dependent_caches(&self) -> bool {
        self.malloc_slowpath_fixed.is_some() || self.propagate_exception_path.is_some()
    }
}
