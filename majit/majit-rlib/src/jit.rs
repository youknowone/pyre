//! `rpython/rlib/jit.py` — the interpreter-facing JIT hint surface.
//!
//! `isconstant` / `isvirtual` return false when not tracing.  The translator
//! rewrites calls marked `oopspec("jit.isvirtual")` / `jit.isconstant` into
//! the `ref_isvirtual` / `*_isconstant` ops; the bodies here are the
//! untranslated residual (`rlib/jit.py isconstant` / `isvirtual`).
//!
//! `we_are_jitted` is a hook because this crate cannot depend on
//! `majit-backend`.  `pyre-jit` / the metainterp install
//! [`majit_backend::we_are_jitted`] at process start.

use std::sync::atomic::{AtomicPtr, Ordering};

static WE_ARE_JITTED: AtomicPtr<()> = AtomicPtr::new(std::ptr::null_mut());

/// `rlib/jit.py we_are_jitted`.
pub fn we_are_jitted() -> bool {
    let p = WE_ARE_JITTED.load(Ordering::Relaxed);
    if p.is_null() {
        return false;
    }
    let f: fn() -> bool = unsafe { std::mem::transmute(p) };
    f()
}

/// Install the process-wide `we_are_jitted` reader.  Called once from
/// `init_jit_hooks`.
pub fn install_we_are_jitted(f: fn() -> bool) {
    WE_ARE_JITTED.store(f as *mut (), Ordering::Relaxed);
}

/// `rlib/jit.py isconstant`.
#[majit_macros::oopspec("jit.isconstant(value)")]
pub fn isconstant<T: ?Sized>(_value: &T) -> bool {
    false
}

/// `rlib/jit.py isvirtual`.
#[majit_macros::oopspec("jit.isvirtual(value)")]
pub fn isvirtual<T: ?Sized>(_value: &T) -> bool {
    false
}

/// `rlib/jit.py loop_unrolling_heuristic`.
///
/// `isvirtual(lst)` is often lying for a resizable list (it reports the
/// containing struct, not the whole list), so size must also be constant.
pub fn loop_unrolling_heuristic<T: ?Sized>(lst: &T, size: usize, cutoff: usize) -> bool {
    size == 0 || (isconstant(&size) && (isvirtual(lst) || size <= cutoff))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loop_unrolling_heuristic_empty_unrolls() {
        // `rlib/jit.py loop_unrolling_heuristic`: size == 0 is always true.
        let lst: &[i32] = &[];
        assert!(loop_unrolling_heuristic(lst, 0, 2));
    }

    #[test]
    fn loop_unrolling_heuristic_needs_constant_size() {
        // Residual `isconstant` is false, so a non-empty list does not unroll.
        let lst = &[1, 2];
        assert!(!loop_unrolling_heuristic(lst, 2, 2));
    }
}
