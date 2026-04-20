//! `sys` module interpreter-level state.
//!
//! PyPy stores `sys.recursionlimit` on the `sys` module instance itself
//! (`pypy/module/sys/moduledef.py:25 self.recursionlimit = 1000`,
//! read/written through `space.sys.recursionlimit` in
//! `pypy/module/sys/vm.py:92-96`). Pyre is single-space so the module
//! singleton maps 1:1 to a static here, but the storage is scoped to
//! the `sys` module (not to the stack-check subsystem) so the data
//! structure lives in the same namespace as its upstream owner.

use std::sync::atomic::{AtomicI32, Ordering};

/// Default recursion limit, matching CPython / PyPy. `Module.__init__`
/// sets `self.recursionlimit = 1000` in the PyPy tree
/// (`pypy/module/sys/moduledef.py:25`).
pub const DEFAULT_RECURSION_LIMIT: i32 = 1000;

/// Hard upper bound for `sys.setrecursionlimit`, matching the silent
/// clamp at `pypy/module/sys/vm.py:82-87`.
pub const MAX_RECURSION_LIMIT: i32 = 1_000_000;

/// `space.sys.recursionlimit` parity. `sys.setrecursionlimit` writes
/// here, `sys.getrecursionlimit` reads from here. The stack-check
/// subsystem (`crate::stack_check`) consults this value when the user
/// raises/lowers the budget, but otherwise keeps its own derived
/// byte-budget (`PYRE_STACKTOOBIG.length`) hot in L1.
static RECURSION_LIMIT: AtomicI32 = AtomicI32::new(DEFAULT_RECURSION_LIMIT);

/// `space.sys.recursionlimit` getter. Matches
/// `pypy/module/sys/vm.py:102 return space.newint(space.sys.recursionlimit)`.
#[inline]
pub fn recursion_limit() -> i32 {
    RECURSION_LIMIT.load(Ordering::Relaxed)
}

/// `space.sys.recursionlimit = new_limit` parity
/// (`pypy/module/sys/vm.py:96`).
#[inline]
pub fn set_recursion_limit(new_limit: i32) {
    RECURSION_LIMIT.store(new_limit, Ordering::Relaxed);
}

/// Reset to the default value. Used by unit tests that need a clean
/// recursion-limit state between runs.
#[cfg(test)]
pub fn reset_recursion_limit_for_tests() {
    RECURSION_LIMIT.store(DEFAULT_RECURSION_LIMIT, Ordering::Relaxed);
}
