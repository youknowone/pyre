//! Translation-visible JIT hint carriers.
//!
//! These functions are runtime identities.  Their call paths survive in LLBC
//! long enough for `majit-translate` to replace them with the corresponding
//! RPython `jit.hint` operation.  They live in `majit-ir`, below both
//! `pyre-object` and `majit-metainterp`, so source-level decorators do not
//! invert the crate dependency graph merely to spell a translation hint.

/// `rpython.rlib.jit.promote(x)` — `hint(x, promote=True)`.
#[inline(always)]
pub fn promote<T: Copy>(value: T) -> T {
    value
}
