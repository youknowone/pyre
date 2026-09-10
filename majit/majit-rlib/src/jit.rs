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

/// `rlib/jit.py conditional_call` residual body (`if condition: function(*args)`).
///
/// Translated graphs rewrite a `oopspec("jit.conditional_call")` call to
/// `conditional_call_ir_v` (`jtransform.py rewrite_op_jit_conditional_call`).
/// Upstream is one `*args` function with `_always_inline_ = 'try'`; Rust
/// spells one arity per residual word-count (`jtransform.py` rejects more
/// than 4 function arguments).
#[majit_macros::oopspec("jit.conditional_call")]
pub fn conditional_call0(condition: bool, function: unsafe fn()) {
    if condition {
        unsafe { function() }
    }
}

/// [`conditional_call0`] with one function argument.
#[majit_macros::oopspec("jit.conditional_call")]
pub fn conditional_call1<A>(condition: bool, function: unsafe fn(A), a: A) {
    if condition {
        unsafe { function(a) }
    }
}

/// [`conditional_call0`] with two function arguments.
#[majit_macros::oopspec("jit.conditional_call")]
pub fn conditional_call2<A, B>(condition: bool, function: unsafe fn(A, B), a: A, b: B) {
    if condition {
        unsafe { function(a, b) }
    }
}

/// [`conditional_call0`] with three function arguments.
///
/// `rlist.py _ll_list_resize_ge` uses this shape:
/// `conditional_call(cond, _ll_list_resize_hint_really, l, newsize, True)`.
#[majit_macros::oopspec("jit.conditional_call")]
pub fn conditional_call3<A, B, C>(condition: bool, function: unsafe fn(A, B, C), a: A, b: B, c: C) {
    if condition {
        unsafe { function(a, b, c) }
    }
}

/// [`conditional_call0`] with four function arguments.
#[majit_macros::oopspec("jit.conditional_call")]
pub fn conditional_call4<A, B, C, D>(
    condition: bool,
    function: unsafe fn(A, B, C, D),
    a: A,
    b: B,
    c: C,
    d: D,
) {
    if condition {
        unsafe { function(a, b, c, d) }
    }
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

    #[test]
    fn conditional_call_runs_when_true() {
        // Residual `rlib/jit.py conditional_call`: `if condition: function(*args)`.
        fn mark(flag: &mut bool) {
            *flag = true;
        }
        let mut flag = false;
        conditional_call1(true, mark, &mut flag);
        assert!(flag);
        let mut flag = true;
        conditional_call1(false, mark, &mut flag);
        assert!(flag);
    }
}
