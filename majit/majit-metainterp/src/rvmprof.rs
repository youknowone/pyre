//! Disabled-runtime adapter for RPython's `rpython.rlib.rvmprof` surface.
//!
//! The `cintf` submodule exposes the `jit_rvmprof_code(leaving, unique_id)`
//! function that blackhole calls directly, like RPython's
//! `rpython.rlib.rvmprof.cintf`. This file does not implement RPython's
//! vmprof stack manipulation; by default it is the explicit "rvmprof disabled"
//! adapter, and embedders that ship an rvmprof runtime replace it via
//! `cintf::set_hook`.
//!
//! Blackhole consumers must use `cintf::jit_rvmprof_code` exclusively;
//! the hook-registry indirection is an implementation detail of this
//! module and is never surfaced to dispatch code.

pub mod cintf {
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// RPython `rvmprof.cintf` hook signature — `leaving ∈ {0, 1}` and
    /// `unique_id` identifies the jitted code object (in RPython, a
    /// pointer to the function's `ll_func`).
    pub type RvmprofHook = fn(leaving: i64, unique_id: i64);

    static HOOK: AtomicUsize = AtomicUsize::new(0);

    /// Install the rvmprof runtime hook. Passing `None` reverts to the
    /// default no-op. Called once at startup by embedders that ship an
    /// rvmprof runtime; majit never calls it itself.
    pub fn set_hook(hook: Option<RvmprofHook>) {
        let raw = hook.map_or(0, |h| h as usize);
        HOOK.store(raw, Ordering::Release);
    }

    /// Disabled-runtime adapter for
    /// `rpython.rlib.rvmprof.cintf.jit_rvmprof_code`.
    ///
    /// Invoked by `bhimpl_rvmprof_code` and by `handle_rvmprof_enter`
    /// inside the blackhole interpreter. The no-op default intentionally has
    /// no vmprof stack side effects.
    #[inline]
    pub fn jit_rvmprof_code(leaving: i64, unique_id: i64) {
        let raw = HOOK.load(Ordering::Acquire);
        if raw != 0 {
            // SAFETY: `HOOK` only receives function pointers passed
            // through `set_hook`, which accepts `RvmprofHook`.
            let hook: RvmprofHook = unsafe { std::mem::transmute(raw) };
            hook(leaving, unique_id);
        }
    }
}
