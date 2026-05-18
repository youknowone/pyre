//! aarch64-specific per-CPU assembler state held by `DynasmBackend`.
//!
//! Mirror of `x86::cpu_ext::X86CpuExt` for `target_arch = "aarch64"`.
//! PyPy's `aarch64/assembler.py:559-577 _build_propagate_exception_path`
//! and `aarch64/assembler.py:605-... _build_malloc_slowpath` produce
//! per-CPU trampolines just like x86 does; pyre's aarch64 backend
//! currently inlines the equivalent slowpath sequences per call site
//! (`aarch64/assembler.rs::CallMallocNursery*` arms), so there is no
//! address to memoise yet.
//!
//! This placeholder exists so `runner.rs::DynasmBackend` can hold one
//! arch-specific extension under a single `arch_cpu_ext` field name
//! regardless of `target_arch`.  When the aarch64 backend grows
//! `_build_*` builders, port the x86 layout onto this struct.

pub(crate) struct Aarch64CpuExt;

impl Aarch64CpuExt {
    pub(crate) fn new() -> Self {
        Self
    }
}
