//! `rpython/jit/metainterp/quasiimmut.py` — the residual half of an
//! `_immutable_fields_` entry spelled with a `?`.
//!
//! Only `do_force_quasi_immutable` lives here.  The recording half
//! (`QuasiImmutDescr`, `get_current_qmut_instance`, `register_loop_token`)
//! is reached through [`majit_ir::QuasiImmutHandle`], which the host
//! implements; this is the other end, the one a `jit_force_quasi_immutable`
//! bytecode runs when the blackhole executes the write that revokes the
//! folded loops.
//!
//! Upstream's hidden `mutate_<name>` field holds a `GCREF` to a `QuasiImmut`
//! instance, so the force reads the pointer, nulls it and invalidates what it
//! points at.  A host may embed the instance's state in the owner instead, in
//! which case unlinking and flipping every recorded loop flag is one call at
//! the field's address.  Both shapes fit the same signature — an address
//! derived from the struct and the mutate field descr — so majit computes the
//! address and the host performs the pair.

use std::sync::atomic::{AtomicUsize, Ordering};

/// `quasiimmut.py`'s
/// ```python
/// cpu.bh_setfield_gc_r(p, ConstPtr.value, mutatefielddescr)
/// qmut = cast_gcref_to_instance(QuasiImmut, qmut_ref)
/// qmut.invalidate(mutatefielddescr.repr_of_descr())
/// ```
/// as one host call taking the address of the hidden mutate field.
pub type ForceQuasiImmutable = extern "C" fn(field_addr: i64);

static FORCE_QUASI_IMMUTABLE: AtomicUsize = AtomicUsize::new(0);

/// Install the host's invalidation routine.  A frontend whose codewriter
/// never emits `jit_force_quasi_immutable` needs none.
pub fn set_force_quasi_immutable_hook(hook: Option<ForceQuasiImmutable>) {
    FORCE_QUASI_IMMUTABLE.store(hook.map_or(0, |f| f as usize), Ordering::Release);
}

/// `quasiimmut.py`:
/// ```python
/// def do_force_quasi_immutable(cpu, p, mutatefielddescr):
///     qmut_ref = cpu.bh_getfield_gc_r(p, mutatefielddescr)
///     if qmut_ref:
///         cpu.bh_setfield_gc_r(p, ConstPtr.value, mutatefielddescr)
///         qmut = cast_gcref_to_instance(QuasiImmut, qmut_ref)
///         qmut.invalidate(mutatefielddescr.repr_of_descr())
/// ```
///
/// The `if qmut_ref` test is the host's: it owns the representation that
/// answers whether any loop ever folded a read of this field, and the whole
/// point of the test is that an object no loop watches pays nothing.
pub fn do_force_quasi_immutable(
    struct_ptr: i64,
    mutatefielddescr: &majit_translate::jitcode::BhDescr,
) {
    if struct_ptr == 0 {
        return;
    }
    let hook = FORCE_QUASI_IMMUTABLE.load(Ordering::Acquire);
    if hook == 0 {
        return;
    }
    let field_addr = struct_ptr.wrapping_add(mutatefielddescr.as_offset() as i64);
    // Safety: the only stored values come from a `ForceQuasiImmutable`
    // function pointer in `set_force_quasi_immutable_hook`.
    let hook: ForceQuasiImmutable = unsafe { std::mem::transmute(hook) };
    hook(field_addr);
}
