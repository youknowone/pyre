//! `rpython/rlib/rmmap.py`: the assembler-writing bracket.
//!
//! On darwin/arm64 the JIT mapping is `MAP_JIT`, and
//! `pthread_jit_write_protect_np` switches this thread's view of it between
//! executable and writable. Everywhere else the mapping is RWX and the
//! bracket only counts. It lives in this crate because both the assembler
//! (`asmmemmgr.py`) and the reference-constant tracer (`gcreftracer.py`
//! `gcrefs_trace`) take it: the tracer forwards slots that sit inside a
//! machine-code block.

use std::cell::Cell;

// `rmmap.py` `Nester`: the depth of `enter_assembler_writing` brackets on
// this thread. `pthread_jit_write_protect_np` is per-thread state, so the
// counter is too.
thread_local! {
    static ASSEMBLER_WRITING_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// `rmmap.py` `enter_assembler_writing`: on darwin/arm64 switch this thread's
/// `MAP_JIT` pages from executable to writable; elsewhere the JIT mapping is
/// RWX and this only counts the bracket.
pub fn enter_assembler_writing() {
    ASSEMBLER_WRITING_DEPTH.with(|depth| {
        if depth.get() == 0 {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            unsafe {
                libc::pthread_jit_write_protect_np(0);
            }
        }
        depth.set(depth.get() + 1);
    });
}

/// `rmmap.py` `leave_assembler_writing`.
pub fn leave_assembler_writing() {
    ASSEMBLER_WRITING_DEPTH.with(|depth| {
        depth.set(depth.get() - 1);
        if depth.get() == 0 {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            unsafe {
                libc::pthread_jit_write_protect_np(1);
            }
        }
    });
}

/// The `try: ... finally: rmmap.leave_assembler_writing()` bracket
/// (`aarch64/assembler.py` `assemble_loop`, `assemble_bridge`,
/// `redirect_call_assembler`; `aarch64/runner.py` `invalidate_loop`;
/// `gcreftracer.py` `gcrefs_trace`) as a guard, so every return path leaves.
pub struct AssemblerWriting(());

impl AssemblerWriting {
    pub fn enter() -> Self {
        enter_assembler_writing();
        Self(())
    }
}

impl Drop for AssemblerWriting {
    fn drop(&mut self) {
        leave_assembler_writing();
    }
}
