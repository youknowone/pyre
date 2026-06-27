//! `ResOpWithDescr.{getdescr,setdescr,cleardescr}` accessors plus
//! per-trait `with_*_descr` shortcuts and `rd_*` Arc-returning
//! resolvers, separated from `resoperation.rs` so the build-script
//! source analyzer in `pyre-jit-trace/build.rs` (which reads
//! `resoperation.rs` for the `RdVirtualInfo` enum declarations) does
//! not need to lower the closure-bearing accessor surface.
//!
//! `Op.descr` is now `RefCell<Option<DescrRef>>` so the optimizer can
//! stamp a `ResumeGuardDescr` onto a shared `Op` (reached through
//! `Rc<Op>`) the same way RPython's `op.setdescr(d)` writes on a shared
//! Python object. `setdescr` / `cleardescr` therefore take `&self`.

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
    /// ResOpWithDescr.getdescr` parity. Returns an owned `Arc` clone of
    /// the `DescrRef` so callers can chain `.as_ref()`, `.expect()`, or
    /// pattern-match without holding a `RefCell` borrow across the call.
    pub fn getdescr(&self) -> Option<DescrRef> {
        self.descr.borrow().clone()
    }

    /// `resoperation.py:465 ResOpWithDescr.setdescr` parity — overwrites
    /// the descr slot.  Takes `&self` (interior mutability) so callers
    /// holding a shared `Op` (e.g., through `Rc<Op>`) can stamp a fresh
    /// descr the same way RPython's `op.setdescr(d)` writes on a shared
    /// Python object.
    pub fn setdescr(&self, descr: DescrRef) {
        *self.descr.borrow_mut() = Some(descr);
    }

    /// `resoperation.py:474 ResOpWithDescr.cleardescr` parity — clears
    /// the descr slot.
    pub fn cleardescr(&self) {
        *self.descr.borrow_mut() = None;
    }

    // `has_descr` lives in `resoperation.rs` so the build-script
    // source analyzer can resolve the bool return type when callers
    // appear in `!op.has_descr()` patterns inside that file.

    /// `resoperation.py:156-200 VectorizationInfo` slot accessor —
    /// returns an owned clone of the per-op vector metadata installed
    /// by the vectorizer.
    pub fn get_vecinfo(&self) -> Option<crate::resoperation::VectorizationInfo> {
        self.vecinfo.borrow().as_deref().cloned()
    }

    /// Overwrite the per-op vector metadata slot.  Takes `&self` —
    /// interior mutability through `RefCell` matches RPython's
    /// `op._vector_info = …` write on a shared object.
    pub fn set_vecinfo(&self, vecinfo: crate::resoperation::VectorizationInfo) {
        *self.vecinfo.borrow_mut() = Some(Box::new(vecinfo));
    }

    /// Clear the per-op vector metadata slot.
    pub fn clear_vecinfo(&self) {
        *self.vecinfo.borrow_mut() = None;
    }

    /// True iff the per-op vector metadata slot is populated.
    pub fn has_vecinfo(&self) -> bool {
        self.vecinfo.borrow().is_some()
    }

    /// Project the descr (if any) through a closure operating on a
    /// `&dyn Descr`. `f` may freely return owned values derived from
    /// borrowed projections (`as_field_descr`, `as_array_descr`, etc.).
    /// Holds the descr `Ref` for the duration of `f`; the inner trait
    /// object lives behind `Arc`, so callers can safely keep an owned
    /// clone if they need to outlive the borrow.
    pub fn project_descr<R>(&self, f: impl FnOnce(&dyn Descr) -> R) -> Option<R> {
        self.descr.borrow().as_ref().map(|d| f(&**d))
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
    pub fn with_loop_token_descr<R>(&self, f: impl FnOnce(&dyn LoopTokenDescr) -> R) -> Option<R> {
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
    /// stays valid once the borrow on `op.descr` drops.
    pub fn resolved_rd_numb(&self) -> Option<Arc<[u8]>> {
        self.descr.borrow().as_ref()?.as_fail_descr()?.rd_numb_arc()
    }

    /// Same as `resolved_rd_numb` but for the `rd_consts` const pool.
    pub fn resolved_rd_consts(&self) -> Option<Arc<[Const]>> {
        self.descr
            .borrow()
            .as_ref()?
            .as_fail_descr()?
            .rd_consts_arc()
    }

    /// Same as `resolved_rd_numb` but for the `rd_virtuals` table.
    pub fn resolved_rd_virtuals(&self) -> Option<Arc<[Rc<RdVirtualInfo>]>> {
        self.descr
            .borrow()
            .as_ref()?
            .as_fail_descr()?
            .rd_virtuals_arc()
    }

    /// Same as `resolved_rd_numb` but for the `rd_pendingfields` table.
    pub fn resolved_rd_pendingfields(&self) -> Option<Arc<[GuardPendingFieldEntry]>> {
        self.descr
            .borrow()
            .as_ref()?
            .as_fail_descr()?
            .rd_pendingfields_arc()
    }

    /// `resoperation.py:299/489 AbstractResOp/GuardResOp.getfailargs`
    /// parity. Returns an owned `SmallVec` clone of the fail_args slot —
    /// None for non-guard ops.  Clone is cheap because `Operand` is an `Rc`
    /// bump and fail_args almost always fits inline (≤3 entries).  Owned
    /// return avoids the `Ref<[T]>` ergonomics tax for callers that chain
    /// through `.into_iter().flatten()` or `.iter()` patterns.
    pub fn getfailargs(&self) -> Option<smallvec::SmallVec<[crate::operand::Operand; 3]>> {
        self.fail_args.borrow().as_ref().map(|fa| fa.clone())
    }

    /// `resoperation.py:492 GuardResOp.getfailargs_copy` parity.
    /// Returns an owned `Vec` copy of the fail_args slot — equivalent
    /// to RPython's `self._fail_args[:]`, which raises `TypeError:
    /// 'NoneType' object is not subscriptable` when `_fail_args` is
    /// `None`.  Mirror that fail-loud behaviour here so a missing-
    /// fail-args bug surfaces at the call site rather than returning
    /// a silently-empty vector.
    pub fn getfailargs_copy(&self) -> Vec<crate::operand::Operand> {
        let borrow = self.fail_args.borrow();
        let fa = borrow.as_ref().unwrap_or_else(|| {
            panic!(
                "getfailargs_copy on op with fail_args=None — RPython \
                 `self._fail_args[:]` raises TypeError; pyre matches the \
                 fail-loud shape (resoperation.py:492)"
            )
        });
        fa.iter().cloned().collect()
    }

    /// `resoperation.py:495 GuardResOp.setfailargs` parity — overwrite
    /// the fail_args slot.  Takes `&self` (interior mutability) so the
    /// optimizer can stamp fail_args onto a shared `Op` reached through
    /// `Rc<Op>`.
    pub fn setfailargs(&self, fail_args: smallvec::SmallVec<[crate::operand::Operand; 3]>) {
        // `GuardResOp._fail_args` holds the producer operands themselves
        // (resoperation.py:483), exactly like `Op.args`: a bound fail-arg
        // is its `Operand::Op`/`Operand::InputArg` producer, a constant is
        // `Operand::Const`. An unbound position-only fail-arg is a contract
        // violation (every writer binds its producer) — `Operand::from_opref`
        // panics for it at the call site.
        *self.fail_args.borrow_mut() = Some(fail_args);
    }

    /// In-place mutable view of the fail_args slot.  Lets callers iterate
    /// the SmallVec mutably (`fa.iter_mut()`, `fa[i] = …`) without going
    /// through a clone/setfailargs round-trip.  Returns `None` when the
    /// slot is empty.  Uses `RefCell::get_mut` so it requires `&mut Op`;
    /// shared-`Op` callers should clone via `getfailargs_copy`, mutate
    /// the copy, and call `setfailargs`.
    pub fn fail_args_mut(
        &mut self,
    ) -> Option<&mut smallvec::SmallVec<[crate::operand::Operand; 3]>> {
        self.fail_args.get_mut().as_mut()
    }

    /// Clear the fail_args slot.  PyPy has no separate `clearfailargs`
    /// method; the pattern is `op.setfailargs(None)` in RPython, but
    /// pyre's signature distinguishes the two paths (set vs clear) for
    /// clarity.
    pub fn clearfailargs(&self) {
        *self.fail_args.borrow_mut() = None;
    }

    /// True iff the fail_args slot is populated.
    pub fn has_failargs(&self) -> bool {
        self.fail_args.borrow().is_some()
    }

    /// Per-failarg type vector accessor.  Pyre's `fail_arg_types` slot
    /// caches the types the optimizer assigned to each `fail_arg` (the
    /// `compile.py:855 _attrs_` set lives on the descr, but the
    /// per-op view is kept here for backend dispatch convenience).
    /// Returns an owned `Vec` clone now that the slot is `RefCell`-
    /// wrapped; callers no longer hold a borrow across other `Op`
    /// accesses.
    pub fn get_fail_arg_types(&self) -> Option<Vec<crate::value::Type>> {
        self.fail_arg_types.borrow().clone()
    }

    /// Owned-clone variant — RPython would write `fail_arg_types[:]`.
    pub fn get_fail_arg_types_copy(&self) -> Vec<crate::value::Type> {
        self.fail_arg_types.borrow().clone().unwrap_or_default()
    }

    /// Overwrite the per-failarg type vector.  Takes `&self`
    /// (interior mutability through `RefCell`) so shared `Op` instances
    /// can be re-stamped without `&mut`.
    pub fn set_fail_arg_types(&self, types: Vec<crate::value::Type>) {
        *self.fail_arg_types.borrow_mut() = Some(types);
    }

    /// Clear the per-failarg type vector.
    pub fn clear_fail_arg_types(&self) {
        *self.fail_arg_types.borrow_mut() = None;
    }

    /// True iff the per-failarg type vector slot is populated.
    pub fn has_fail_arg_types(&self) -> bool {
        self.fail_arg_types.borrow().is_some()
    }

    /// `resoperation.py:281 AbstractResOp.getarglist` parity — the operand
    /// list as `BoxRef`s.  Subclass mixins (`UnaryOp`, `BinaryOp`, ...,
    /// `N_aryOp`) implement this differently; pyre collapses them into a
    /// single SmallVec slot.
    ///
    /// `#9` operand-union flip: `args` now stores [`Operand`], so this
    /// materializes an owned `SmallVec` of `BoxRef`s (convert-on-read via
    /// `Operand::to_boxref`) rather than borrowing the slot — the old `Ref`
    /// view is no longer expressible. The owned `SmallVec` derefs to
    /// `&[BoxRef]`, so iterate/index/`&`-coercion callers are unchanged; it
    /// also drops the `RefCell` borrow at the call boundary, so a caller may
    /// freely `setarg` afterwards.
    pub fn getarglist(&self) -> smallvec::SmallVec<[crate::box_ref::BoxRef; 3]> {
        self.args.borrow().iter().map(|o| o.to_boxref()).collect()
    }

    /// `Operand`-yielding form of [`getarglist`](Self::getarglist): the stored
    /// `Operand` args directly, with no `to_boxref` round-trip.
    pub fn getarglist_operand(&self) -> smallvec::SmallVec<[crate::operand::Operand; 3]> {
        self.args.borrow().iter().cloned().collect()
    }

    /// `resoperation.py:284 AbstractResOp.getarglist_copy` parity —
    /// `N_aryOp.getarglist_copy` returns `self._args[:]`; pyre returns
    /// an owned `SmallVec` of `BoxRef`s (convert-on-read).
    pub fn getarglist_copy(&self) -> smallvec::SmallVec<[crate::box_ref::BoxRef; 3]> {
        self.args.borrow().iter().map(|o| o.to_boxref()).collect()
    }

    /// `resoperation.py:277 AbstractResOp.initarglist` parity — bulk
    /// store the operand list.  In RPython this is "supposed to be
    /// called only just after the ResOp has been created"
    /// (resoperation.py:278); pyre matches both the name and intent.
    ///
    /// One existing post-creation caller in upstream:
    /// `unroll.py:301 label_op.initarglist(label_op.getarglist() +
    ///  sb.used_boxes)` extends the LABEL arg list when finishing the
    /// peel pass.  pyre's matching call lives in `unroll.rs` and
    /// rebuilds the SmallVec rather than pushing onto `args`.
    pub fn initarglist(&self, args: smallvec::SmallVec<[crate::operand::Operand; 3]>) {
        *self.args.borrow_mut() = args;
    }

    /// `resoperation.py:290 AbstractResOp.setarg` parity — position-wise
    /// in-place arg mutation.  Subclass mixins index `_arg0/_arg1/...`
    /// or `_args[i]`; pyre indexes the SmallVec directly.
    pub fn setarg(&self, i: usize, arg: crate::operand::Operand) {
        self.args.borrow_mut()[i] = arg;
    }
}
