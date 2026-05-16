//! `ResOpWithDescr.{getdescr,setdescr,cleardescr}` accessors plus
//! per-trait `with_*_descr` shortcuts and `rd_*` Arc-returning
//! resolvers, separated from `resoperation.rs` so the build-script
//! source analyzer in `pyre-jit-trace/build.rs` (which reads
//! `resoperation.rs` for the `RdVirtualInfo` enum declarations) does
//! not need to lower the closure-bearing accessor surface.
//!
//! `Op.descr` is still a plain `Option<DescrRef>` field (the build-
//! script translator's `expr_unary_not_operand_kind` classifier does
//! not yet recognise `RefCell<...>` field types).  Until the
//! translator gains parity, `setdescr` / `cleardescr` take `&mut
//! self` rather than `&self`; the `Vec<Rc<Op>>`-era flip to interior
//! mutability is coordinated with the translator update.

use std::rc::Rc;
use std::sync::Arc;

use crate::descr::{
    ArrayDescr, CallDescr, Descr, DescrRef, FailDescr, FieldDescr, InteriorFieldDescr,
    LoopTargetDescr, LoopTokenDescr, SizeDescr,
};
use crate::resoperation::{GuardPendingFieldEntry, Op, RdVirtualInfo};
use crate::value::Const;

impl Op {
    /// `resoperation.py:244 AbstractResOpOrInputArg.getdescr` + `:462
    /// ResOpWithDescr.getdescr` parity. Returns an owned clone of the
    /// `Option<DescrRef>` so callers can chain `.as_ref()`, `.expect()`,
    /// or pattern-match without holding a reference across the call.
    pub fn getdescr(&self) -> Option<DescrRef> {
        self.descr.clone()
    }

    /// `resoperation.py:465 ResOpWithDescr.setdescr` parity — overwrites
    /// the descr slot.
    pub fn setdescr(&mut self, descr: DescrRef) {
        self.descr = Some(descr);
    }

    /// `resoperation.py:474 ResOpWithDescr.cleardescr` parity — clears
    /// the descr slot.
    pub fn cleardescr(&mut self) {
        self.descr = None;
    }

    // `has_descr` lives in `resoperation.rs` so the build-script
    // source analyzer can resolve the bool return type when callers
    // appear in `!op.has_descr()` patterns inside that file.

    /// Project the descr (if any) through a closure operating on a
    /// `&dyn Descr`. `f` may freely return owned values derived from
    /// borrowed projections (`as_field_descr`, `as_array_descr`, etc.).
    pub fn project_descr<R>(&self, f: impl FnOnce(&dyn Descr) -> R) -> Option<R> {
        self.descr.as_ref().map(|d| f(&**d))
    }

    /// `as_field_descr` shortcut.
    pub fn with_field_descr<R>(&self, f: impl FnOnce(&dyn FieldDescr) -> R) -> Option<R> {
        self.project_descr(|d| d.as_field_descr().map(f)).flatten()
    }

    /// `as_array_descr` shortcut.
    pub fn with_array_descr<R>(&self, f: impl FnOnce(&dyn ArrayDescr) -> R) -> Option<R> {
        self.project_descr(|d| d.as_array_descr().map(f)).flatten()
    }

    /// `as_call_descr` shortcut.
    pub fn with_call_descr<R>(&self, f: impl FnOnce(&dyn CallDescr) -> R) -> Option<R> {
        self.project_descr(|d| d.as_call_descr().map(f)).flatten()
    }

    /// `as_loop_target_descr` shortcut.
    pub fn with_loop_target_descr<R>(
        &self,
        f: impl FnOnce(&dyn LoopTargetDescr) -> R,
    ) -> Option<R> {
        self.project_descr(|d| d.as_loop_target_descr().map(f))
            .flatten()
    }

    /// `as_size_descr` shortcut.
    pub fn with_size_descr<R>(&self, f: impl FnOnce(&dyn SizeDescr) -> R) -> Option<R> {
        self.project_descr(|d| d.as_size_descr().map(f)).flatten()
    }

    /// `as_fail_descr` shortcut.
    pub fn with_fail_descr<R>(&self, f: impl FnOnce(&dyn FailDescr) -> R) -> Option<R> {
        self.project_descr(|d| d.as_fail_descr().map(f)).flatten()
    }

    /// `as_loop_token_descr` shortcut.
    pub fn with_loop_token_descr<R>(
        &self,
        f: impl FnOnce(&dyn LoopTokenDescr) -> R,
    ) -> Option<R> {
        self.project_descr(|d| d.as_loop_token_descr().map(f))
            .flatten()
    }

    /// `as_interior_field_descr` shortcut.
    pub fn with_interior_field_descr<R>(
        &self,
        f: impl FnOnce(&dyn InteriorFieldDescr) -> R,
    ) -> Option<R> {
        self.project_descr(|d| d.as_interior_field_descr().map(f))
            .flatten()
    }

    /// `compile.py:849 ResumeGuardCopiedDescr.get_resumestorage(): return prev`
    /// parity. Reads `rd_numb` from `op.descr` — `ResumeGuardCopiedDescr`
    /// chases `prev` automatically.  Returns `Arc<[u8]>` so the slice
    /// stays valid even after the `op.descr` field is later wrapped in
    /// `RefCell` (`Vec<Rc<Op>>`-era migration).
    pub fn resolved_rd_numb(&self) -> Option<Arc<[u8]>> {
        self.descr.as_ref()?.as_fail_descr()?.rd_numb_arc()
    }

    /// Same as `resolved_rd_numb` but for the `rd_consts` const pool.
    pub fn resolved_rd_consts(&self) -> Option<Arc<[Const]>> {
        self.descr.as_ref()?.as_fail_descr()?.rd_consts_arc()
    }

    /// Same as `resolved_rd_numb` but for the `rd_virtuals` table.
    pub fn resolved_rd_virtuals(&self) -> Option<Arc<[Rc<RdVirtualInfo>]>> {
        self.descr.as_ref()?.as_fail_descr()?.rd_virtuals_arc()
    }

    /// Same as `resolved_rd_numb` but for the `rd_pendingfields` table.
    pub fn resolved_rd_pendingfields(&self) -> Option<Arc<[GuardPendingFieldEntry]>> {
        self.descr.as_ref()?.as_fail_descr()?.rd_pendingfields_arc()
    }

    /// `resoperation.py:299/489 AbstractResOp/GuardResOp.getfailargs`
    /// parity. Returns a borrow into the fail_args slot — None for
    /// non-guard ops.
    pub fn getfailargs(&self) -> Option<&[crate::resoperation::OpRef]> {
        self.fail_args.as_deref()
    }

    /// `resoperation.py:492 GuardResOp.getfailargs_copy` parity.
    /// Returns an owned `Vec` copy of the fail_args slot — equivalent
    /// to RPython's `self._fail_args[:]`.
    pub fn getfailargs_copy(&self) -> Vec<crate::resoperation::OpRef> {
        match &self.fail_args {
            Some(fa) => fa.iter().copied().collect(),
            None => Vec::new(),
        }
    }

    /// `resoperation.py:495 GuardResOp.setfailargs` parity — overwrite
    /// the fail_args slot.  Takes `&mut self` until the `Vec<Rc<Op>>`
    /// migration flips the field to interior-mutable.
    pub fn setfailargs(
        &mut self,
        fail_args: smallvec::SmallVec<[crate::resoperation::OpRef; 3]>,
    ) {
        self.fail_args = Some(fail_args);
    }

    /// Clear the fail_args slot.  PyPy has no separate `clearfailargs`
    /// method; the pattern is `op.setfailargs(None)` in RPython, but
    /// pyre's signature distinguishes the two paths (set vs clear) for
    /// clarity.
    pub fn clearfailargs(&mut self) {
        self.fail_args = None;
    }

    /// True iff the fail_args slot is populated.
    pub fn has_failargs(&self) -> bool {
        self.fail_args.is_some()
    }

    /// Per-failarg type vector accessor.  Pyre's `fail_arg_types` slot
    /// caches the types the optimizer assigned to each `fail_arg` (the
    /// `compile.py:855 _attrs_` set lives on the descr, but the
    /// per-op view is kept here for backend dispatch convenience).
    pub fn get_fail_arg_types(&self) -> Option<&[crate::value::Type]> {
        self.fail_arg_types.as_deref()
    }

    /// Owned-clone variant — RPython would write `fail_arg_types[:]`.
    pub fn get_fail_arg_types_copy(&self) -> Vec<crate::value::Type> {
        match &self.fail_arg_types {
            Some(t) => t.clone(),
            None => Vec::new(),
        }
    }

    /// Overwrite the per-failarg type vector.
    pub fn set_fail_arg_types(&mut self, types: Vec<crate::value::Type>) {
        self.fail_arg_types = Some(types);
    }

    /// Clear the per-failarg type vector.
    pub fn clear_fail_arg_types(&mut self) {
        self.fail_arg_types = None;
    }

    /// True iff the per-failarg type vector slot is populated.
    pub fn has_fail_arg_types(&self) -> bool {
        self.fail_arg_types.is_some()
    }
}
