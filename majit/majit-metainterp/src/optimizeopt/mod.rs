/// JIT optimization pipeline.
///
/// Translated from rpython/jit/metainterp/optimizeopt/.
///
/// The optimizer chains multiple passes, each implementing the Optimization trait.
/// Operations flow through the chain: IntBounds → Rewrite → Virtualize → String →
/// Pure → Guard → Simplify → Heap (configurable).
pub mod autogenintrules;
pub mod bridgeopt;
pub mod dependency;
pub mod earlyforce;
pub mod guard;
pub mod heap;
pub mod info;
pub mod intbounds;
pub mod intdiv;
pub mod intutils;
// optimize module is at crate::optimize (RPython: metainterp/optimize.py)
pub mod optimizer;
pub use optimizer::{Optimization, OptimizationResult};
pub mod pure;
pub mod rawbuffer;
pub mod renamer;
pub mod rewrite;
pub mod schedule;
pub mod shortpreamble;
pub mod simplify;
pub mod unroll;
pub mod util;
pub mod vector;
pub mod version;
pub mod virtualize;
pub mod virtualstate;
pub mod vstring;
// walkvirtual moved to crate::walkvirtual (RPython: metainterp/walkvirtual.py)

use crate::optimizeopt::intutils::{IntBound, IntBoundMakeGuards};
use crate::resume::SnapshotBox;
use info::{EnsuredPtrInfo, PtrInfo};
use majit_ir::operand::Operand;
use majit_ir::{DescrRef, GcRef, Op, OpCode, OpRef, Type, Value};
use std::collections::VecDeque;

pub type SnapshotBoxes = Vec<Option<Vec<SnapshotBox>>>;
pub type SnapshotFrameSizes = Vec<Option<Vec<usize>>>;
pub type SnapshotFramePcs = Vec<Option<Vec<(i32, i32, i32)>>>;

pub(crate) fn snapshot_get<T>(store: &[Option<T>], pos: i32) -> Option<&T> {
    if pos < 0 {
        return None;
    }
    store.get(pos as usize).and_then(Option::as_ref)
}

pub(crate) fn snapshot_contains<T>(store: &[Option<T>], pos: i32) -> bool {
    snapshot_get(store, pos).is_some()
}

pub(crate) fn snapshot_insert<T>(store: &mut Vec<Option<T>>, pos: i32, value: T) {
    assert!(pos >= 0, "snapshot position must be non-negative");
    let idx = pos as usize;
    if store.len() <= idx {
        store.resize_with(idx + 1, || None);
    }
    store[idx] = Some(value);
}

pub(crate) fn next_snapshot_pos<T>(store: &[Option<T>]) -> i32 {
    store.len() as i32
}

pub(crate) fn majit_log_enabled() -> bool {
    std::env::var_os("MAJIT_LOG").is_some()
}

/// info.py:865-894 `getrawptrinfo` / `getptrinfo` return shape, with
/// RPython `_forwarded` object identity preserved.
///
/// Two variants mirror upstream's two return paths:
///   - `Const(PtrInfo)` — fresh `ConstPtrInfo(op)` synthesis
///     (info.py:870-871 / 888-889).  Upstream allocates a brand-new
///     `ConstPtrInfo` per call; pyre carries the freshly built
///     `PtrInfo::Constant(_)` inline.
///   - `Live(Rc<RefCell<PtrInfo>>)` — the `return fw` arm
///     (info.py:875-877 / 890-893).  Carries the live `Rc` handle
///     into the chain terminal's `_forwarded` cell so RPython object
///     identity is preserved: two `Live` handles cloned from the
///     same cell observe each other's in-place mutations
///     (`Rc::ptr_eq` ≡ Python `is`).  Holding a handle keeps the cell
///     alive even if the terminal `BoxRef` later swaps its
///     `_forwarded` slot to a different info — mirroring Python local
///     variables that keep a previously read `fw` alive.
///
/// Read access:
///   - `handle.borrow()` → `PtrInfoHandleRef<'_>` which `Deref`s into
///     `&PtrInfo` for ergonomic method calls.  The `Live` arm holds
///     a `Ref` on the underlying `RefCell`; caller must drop the
///     guard before any sibling `borrow_mut` on the same cell.
///   - `handle.borrow_mut()` → `Option<RefMut<'_, PtrInfo>>` for the
///     `Live` arm; `None` for `Const` (mutating a freshly minted
///     `ConstPtrInfo` snapshot would not propagate).
///   - `handle.same_info(&other)` → RPython `same_info` parity:
///     non-constant live infos compare by object identity, while
///     ConstPtrInfo compares the wrapped constant value.
pub enum PtrInfoHandle {
    Const(PtrInfo),
    Live(std::rc::Rc<std::cell::RefCell<PtrInfo>>),
}

impl PtrInfoHandle {
    /// Wrap a freshly synthesized `ConstPtrInfo` (info.py:870-871 /
    /// 888-889 return path).
    pub fn const_(info: PtrInfo) -> Self {
        PtrInfoHandle::Const(info)
    }

    /// Wrap a live `_forwarded` cell handle (info.py:875-877 /
    /// 890-893 return path).
    pub fn live(rc: std::rc::Rc<std::cell::RefCell<PtrInfo>>) -> Self {
        PtrInfoHandle::Live(rc)
    }

    /// RPython `PtrInfo.same_info(other)` parity.
    ///
    /// Base `PtrInfo.same_info` is object identity (`self is other`,
    /// info.py:71-72), so non-constant live infos must share the same
    /// `_forwarded` cell. `ConstPtrInfo` overrides this and compares
    /// the wrapped constant value (`_const.same_constant`, info.py:774-777),
    /// so two independently synthesized ConstPtrInfo handles for the
    /// same pointer are `same_info`.
    pub fn same_info(&self, other: &PtrInfoHandle) -> bool {
        if let (PtrInfoHandle::Live(a), PtrInfoHandle::Live(b)) = (self, other) {
            if std::rc::Rc::ptr_eq(a, b) {
                return true;
            }
        }

        fn constptr_same_info(a: &PtrInfo, b: &PtrInfo) -> bool {
            match (a, b) {
                (PtrInfo::Constant(left), PtrInfo::Constant(right)) => left == right,
                _ => false,
            }
        }

        match (self, other) {
            (PtrInfoHandle::Const(a), PtrInfoHandle::Const(b)) => constptr_same_info(a, b),
            (PtrInfoHandle::Const(a), PtrInfoHandle::Live(b)) => {
                let b = b.borrow();
                constptr_same_info(a, &b)
            }
            (PtrInfoHandle::Live(a), PtrInfoHandle::Const(b)) => {
                let a = a.borrow();
                constptr_same_info(&a, b)
            }
            (PtrInfoHandle::Live(a), PtrInfoHandle::Live(b)) => {
                let a = a.borrow();
                let b = b.borrow();
                constptr_same_info(&a, &b)
            }
        }
    }

    /// Read access — yields a guard that `Deref`s into `&PtrInfo`.
    pub fn borrow(&self) -> PtrInfoHandleRef<'_> {
        match self {
            PtrInfoHandle::Const(info) => PtrInfoHandleRef::Const(info),
            PtrInfoHandle::Live(rc) => PtrInfoHandleRef::Live(rc.borrow()),
        }
    }

    /// Mutable access — `Some(RefMut<'_, PtrInfo>)` for `Live`,
    /// `None` for `Const`.
    pub fn borrow_mut(&self) -> Option<std::cell::RefMut<'_, PtrInfo>> {
        match self {
            PtrInfoHandle::Const(_) => None,
            PtrInfoHandle::Live(rc) => Some(rc.borrow_mut()),
        }
    }

    /// Convert to an owned `PtrInfo` snapshot.  Clones for `Live`;
    /// destructures for `Const`.
    pub fn into_ptr_info(self) -> PtrInfo {
        match self {
            PtrInfoHandle::Const(info) => info,
            PtrInfoHandle::Live(rc) => rc.borrow().clone(),
        }
    }

    /// Cheap clone-as-snapshot for read-only callsites that only need
    /// an owned `PtrInfo` and don't care about identity.
    pub fn snapshot(&self) -> PtrInfo {
        match self {
            PtrInfoHandle::Const(info) => info.clone(),
            PtrInfoHandle::Live(rc) => rc.borrow().clone(),
        }
    }
}

impl std::fmt::Debug for PtrInfoHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PtrInfoHandle::Const(info) => {
                f.debug_tuple("PtrInfoHandle::Const").field(info).finish()
            }
            PtrInfoHandle::Live(rc) => f
                .debug_tuple("PtrInfoHandle::Live")
                .field(&*rc.borrow())
                .finish(),
        }
    }
}

/// Borrow guard returned by `PtrInfoHandle::borrow()`.
/// `Deref<Target = PtrInfo>` lets callers call any `PtrInfo` method
/// uniformly without first matching on the variant.
pub enum PtrInfoHandleRef<'a> {
    Const(&'a PtrInfo),
    Live(std::cell::Ref<'a, PtrInfo>),
}

impl std::ops::Deref for PtrInfoHandleRef<'_> {
    type Target = PtrInfo;
    fn deref(&self) -> &PtrInfo {
        match self {
            PtrInfoHandleRef::Const(info) => info,
            PtrInfoHandleRef::Live(r) => r,
        }
    }
}

/// IntBound counterpart to [`PtrInfoHandle`].
///
/// `optimizer.py:99-113 getintbound(op)` returns the live `IntBound`
/// object stored on `box._forwarded`; downstream code calling
/// `getintbound(box).intersect(b)` mutates that same object so any
/// other holder observes the change.  In pyre, `OpInfo::IntBound`
/// carries `Rc<RefCell<IntBound>>` (Phase 1A), so sharing the cell
/// between two BoxRefs reproduces the RPython object-identity
/// behaviour.  This handle is the public API for that identity:
///
///   - `Const(IntBound)` — a freshly synthesized `IntBound` from a
///     `ConstInt` (`optimizer.py:102-103 from_constant`).  Two
///     `Const` handles never compare equal under `ptr_eq` even when
///     they wrap the same value.
///   - `Live(Rc<RefCell<IntBound>>)` — the actual `_forwarded` cell.
///     `Rc::ptr_eq` ≡ Python `is`.  In-place mutation via
///     `handle.borrow_mut()` propagates to every other live handle
///     cloned from the same cell.
pub enum IntBoundHandle {
    /// Freshly synthesized `IntBound::from_constant(_)` object
    /// (optimizer.py:102-103 return path). Wrapped in `Rc<RefCell<>>`
    /// so callers retain Python `from_constant(...)` reference
    /// semantics — the object is mutable, and clones of *this same
    /// handle* share the cell. Two independent `getintbound_handle`
    /// calls on the same ConstInt mint distinct Rcs (PyPy: two
    /// `from_constant(7)` calls return two distinct objects), so
    /// mutations do not propagate across calls.
    Const(std::rc::Rc<std::cell::RefCell<crate::optimizeopt::intutils::IntBound>>),
    /// Live `_forwarded` cell — mutations propagate to every handle
    /// cloned from the same Rc and through `box._forwarded`.
    Live(std::rc::Rc<std::cell::RefCell<crate::optimizeopt::intutils::IntBound>>),
}

impl IntBoundHandle {
    /// Wrap a freshly synthesized `IntBound::from_constant(_)`
    /// (`optimizer.py:102-103` return path). Mints a fresh `Rc` so
    /// each call produces a distinct object (Python `is` semantics).
    pub fn const_(b: crate::optimizeopt::intutils::IntBound) -> Self {
        IntBoundHandle::Const(std::rc::Rc::new(std::cell::RefCell::new(b)))
    }

    /// Wrap a live `_forwarded` cell handle (`optimizer.py:111-112`
    /// return path).
    pub fn live(
        rc: std::rc::Rc<std::cell::RefCell<crate::optimizeopt::intutils::IntBound>>,
    ) -> Self {
        IntBoundHandle::Live(rc)
    }

    /// Identity comparison. Two handles are `ptr_eq` iff they hold
    /// the same `Rc` — Python `is` parity. Const/Live cross-arm pairs
    /// are never equal because they live in disjoint cell namespaces.
    pub fn ptr_eq(&self, other: &IntBoundHandle) -> bool {
        match (self, other) {
            (IntBoundHandle::Const(a), IntBoundHandle::Const(b))
            | (IntBoundHandle::Live(a), IntBoundHandle::Live(b)) => std::rc::Rc::ptr_eq(a, b),
            _ => false,
        }
    }

    /// Read access — yields a guard that `Deref`s into `&IntBound`.
    pub fn borrow(&self) -> IntBoundHandleRef<'_> {
        match self {
            IntBoundHandle::Const(rc) => IntBoundHandleRef::Const(rc.borrow()),
            IntBoundHandle::Live(rc) => IntBoundHandleRef::Live(rc.borrow()),
        }
    }

    /// Mutable access for both arms. PyPy `getintbound(ConstInt)`
    /// returns a mutable `IntBound.from_constant(...)` object whose
    /// mutations are private to that object (no propagation back to
    /// the box). The `Const` arm mirrors that: borrow_mut yields a
    /// RefMut into the fresh per-call cell.
    pub fn borrow_mut(
        &self,
    ) -> Option<std::cell::RefMut<'_, crate::optimizeopt::intutils::IntBound>> {
        match self {
            IntBoundHandle::Const(rc) | IntBoundHandle::Live(rc) => Some(rc.borrow_mut()),
        }
    }

    /// Convert to an owned `IntBound` snapshot. Clones for both arms
    /// since both wrap `Rc<RefCell<_>>`.
    pub fn into_int_bound(self) -> crate::optimizeopt::intutils::IntBound {
        match self {
            IntBoundHandle::Const(rc) | IntBoundHandle::Live(rc) => rc.borrow().clone(),
        }
    }

    /// Cheap clone-as-snapshot for read-only callsites that only need
    /// an owned `IntBound` and don't care about identity.
    pub fn snapshot(&self) -> crate::optimizeopt::intutils::IntBound {
        match self {
            IntBoundHandle::Const(rc) | IntBoundHandle::Live(rc) => rc.borrow().clone(),
        }
    }
}

impl std::fmt::Debug for IntBoundHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IntBoundHandle::Const(rc) => f
                .debug_tuple("IntBoundHandle::Const")
                .field(&*rc.borrow())
                .finish(),
            IntBoundHandle::Live(rc) => f
                .debug_tuple("IntBoundHandle::Live")
                .field(&*rc.borrow())
                .finish(),
        }
    }
}

/// Borrow guard returned by `IntBoundHandle::borrow()`.
/// `Deref<Target = IntBound>` lets callers call any `IntBound`
/// method uniformly without first matching on the variant.
pub enum IntBoundHandleRef<'a> {
    Const(std::cell::Ref<'a, crate::optimizeopt::intutils::IntBound>),
    Live(std::cell::Ref<'a, crate::optimizeopt::intutils::IntBound>),
}

impl std::ops::Deref for IntBoundHandleRef<'_> {
    type Target = crate::optimizeopt::intutils::IntBound;
    fn deref(&self) -> &crate::optimizeopt::intutils::IntBound {
        match self {
            IntBoundHandleRef::Const(r) | IntBoundHandleRef::Live(r) => r,
        }
    }
}

/// info.py:13-15 INFO_NULL / INFO_NONNULL / INFO_UNKNOWN constants.
///
/// Re-exported from `majit_ir::optimize` so the bound / info types
/// that reference them can be hosted there without a circular dep.
pub use majit_ir::optimize::{INFO_NONNULL, INFO_NULL, INFO_UNKNOWN};

/// Create a ResumeAtPositionDescr for optimizer-generated guards.
///
/// Delegates to compile::make_resume_at_position_descr which wraps a
/// real ResumeGuardDescr — clone_descr() preserves resume data (RPython
/// ResumeAtPositionDescr is a plain subclass of ResumeGuardDescr).
pub fn make_resume_at_position_descr() -> DescrRef {
    crate::compile::make_resume_at_position_descr()
}

/// optimizer.py:47-54: deferred postprocess for GUARD_CLASS/GUARD_NONNULL_CLASS.
/// RPython's postprocess_GUARD_CLASS runs after the guard is emitted to
/// _newoperations. In majit, recorded here by rewrite and executed by
/// emit_operation.
#[derive(Debug)]
pub struct PendingGuardClassPostprocess {
    pub obj: majit_ir::OpRef,
    pub class_val: i64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ImportedShortPureArg {
    OpRef(OpRef),
    /// Const arg with source OpRef for matching in force_preamble_op.
    /// RPython: Const Box has identity; get_box_replacement returns itself.
    Const(Value, OpRef),
}

#[derive(Clone, Debug)]
pub struct ImportedShortPureOp {
    pub opcode: OpCode,
    pub descr: Option<DescrRef>,
    pub args: Vec<ImportedShortPureArg>,
    pub result: OpRef,
    /// RPython: PreambleOp stored in pure cache. Used by force_op_from_preamble.
    pub pop: crate::optimizeopt::info::PreambleOp,
}

impl ImportedShortPureOp {
    /// Construct with auto-generated PreambleOp from fields.
    ///
    /// `ctx` binds the replay op's operands to their canonical producers
    /// (`materialize_box_at`) — shortpreamble.py:425 seeds the replay
    /// `preamble_op` with the SAME Box objects the body sees, so the
    /// operands must carry producer identity, not a position-only echo.
    pub fn new(
        ctx: &mut OptContext,
        opcode: OpCode,
        descr: Option<DescrRef>,
        args: Vec<ImportedShortPureArg>,
        result: OpRef,
        source: OpRef,
        invented_name: bool,
        same_as_source: Option<majit_ir::box_ref::BoxRef>,
    ) -> Self {
        let replay_args: Vec<OpRef> = args
            .iter()
            .map(|a| match a {
                ImportedShortPureArg::OpRef(r) => *r,
                ImportedShortPureArg::Const(_, src) => *src,
            })
            .collect();
        let replay_arg_boxes: Vec<Operand> = replay_args
            .iter()
            .map(|a| ctx.materialize_operand_at(*a))
            .collect();
        let mut replay = majit_ir::Op::new(opcode, &replay_arg_boxes);
        // shortpreamble.py:112-126 PureOp.produce_op constructs TWO distinct
        // RPython Op objects:
        //
        //   * `op` — the alt identifier. For invented_name it is
        //     `self.orig_op.copy_and_change(...).set_forwarded(self.res)`,
        //     a freshly-allocated Op with its own `_forwarded` slot.
        //   * `preamble_op` — the replay Op passed in from `add_op_to_short`.
        //     Its `_forwarded` slot is seeded with the exported info by
        //     `ShortPreambleBuilder.__init__` (shortpreamble.py:425).
        //
        // The two `_forwarded` slots live on two different RPython Op
        // objects, so `op.set_forwarded(self.res)` and
        // `preamble_op.set_forwarded(info)` never collide.
        //
        // pyre's flat-OpRef model has only one slot per OpRef. To preserve
        // PyPy parity we must allocate TWO different OpRefs for the two
        // identities when they would collide:
        //
        //   * non-invented Pure: `op = self.res`. PyPy does NOT install a
        //     forwarding on `op`, so the slot at `source` is free to hold
        //     `info`. We can leave `replay.pos = source` — both objects
        //     point at one slot, which is allowed because only `info`
        //     occupies it.
        //   * invented Pure: PyPy installs `op.set_forwarded(self.res)` on
        //     the alt. In pyre, `produce_pure` calls
        //     `make_equal_to(source, canonical)` (shortpreamble.rs:1279) which
        //     overwrites the source box's `_forwarded` slot with
        //     a forwarding redirect to `canonical_box`.
        //     If `replay.pos` also pointed at `source`, the alt's
        //     replacement chain and the replay's info would share one slot
        //     and the info would be lost. We move `replay.pos` to the
        //     pre-allocated body-visible OpRef (`result`) so it has its
        //     own slot.
        replay.pos.set(if invented_name { result } else { source });
        if let Some(d) = descr.clone() {
            replay.setdescr(d);
        }
        // shortpreamble.py:116-120: pop.op = self.orig_op.copy_and_change(...)
        // for invented (the alt identifier) or self.res for non-invented.
        // pyre's `source` IS the alt identifier for invented (the synthetic
        // alias allocated by the compound-dedup pass at
        // shortpreamble.rs:478-491) and IS self.res for non-invented.
        let pop_op = ctx.materialize_box_at(source);
        ImportedShortPureOp {
            opcode,
            descr,
            args,
            result,
            pop: crate::optimizeopt::info::PreambleOp {
                op: pop_op,
                invented_name,
                preamble_op: std::rc::Rc::new(replay),
                same_as_source,
            },
        }
    }
}

impl PartialEq for ImportedShortPureOp {
    fn eq(&self, other: &Self) -> bool {
        self.opcode == other.opcode
            && self.descr.as_ref().map(|d| d.index()) == other.descr.as_ref().map(|d| d.index())
            && self.args == other.args
            && self.result == other.result
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImportedShortAlias {
    pub result: OpRef,
    /// MIGRATION (#9): the canonical box of the SameAs source operand,
    /// carried directly off `extra_same_as` instead of a positional
    /// round-trip.
    pub same_as_source: majit_ir::box_ref::BoxRef,
    pub same_as_opcode: OpCode,
}

/// optimizer.py:787-789: constant_fold — allocate an immutable object at
/// compile time when all fields are constants. The callback receives the
/// SizeDescr size_bytes, and returns a raw pointer (GcRef) to freshly
/// allocated memory. The optimizer writes field values directly.
pub type ConstantFoldAllocFn = Box<dyn Fn(usize) -> majit_ir::GcRef>;

/// Re-export of `info::StringLengthResolver` for callers that import
/// `optimizeopt::StringLengthResolver`. The runtime hook signature is
/// `Arc<dyn Fn(GcRef, u8) -> Option<i64> + Send + Sync>`. See
/// `info::EnsuredPtrInfo::getlenbound` for the consumer side.
pub use crate::optimizeopt::info::StringLengthResolver;
pub use crate::optimizeopt::info::{StringConstantAllocator, StringContentResolver};

use crate::optimizeopt::info::PtrInfoExt;

/// Context provided to optimization passes.
///
/// Holds the shared state that passes read from and write to.
pub struct OptContext {
    /// The output operation list being built.
    pub new_operations: Vec<majit_ir::OpRc>,
    /// optimizer.py:246 `self._emittedoperations = {}` — the result boxes
    /// of ops emitted by THIS optimizer run, keyed by box identity
    /// (`Rc::ptr_eq`) to mirror the upstream dict keyed by the `op` object.
    /// `as_operation` (optimizer.py:369-377) only treats a box as a producer
    /// when `op in self._emittedoperations`, so pattern-matching rewrites
    /// (e.g. autogenintrules.py int_add chain reassociation) never reach
    /// across an optimization-run boundary into a previous phase's ops.
    /// A Phase-2 lookup that followed a label arg to its Phase-1 producer
    /// would re-express a per-iteration value in terms of the PREAMBLE's
    /// entry value, which the loop header does not carry per-iteration.
    pub emitted_operations: majit_ir::vec_set::VecSet<majit_ir::operand::Operand>,
    /// Number of input arguments, used to offset emitted op positions
    /// so that variable indices don't collide with input arg indices.
    num_inputs: u32,
    /// opencoder.py:259-267 inputarg base in the OpRef namespace.
    ///
    /// RPython lets each TraceIterator allocate fresh inputarg boxes whose
    /// Python identity (`is`) distinguishes them from any other phase's
    /// boxes; majit needs to encode that as a numeric offset because OpRef
    /// IS the identity. Phase 1 uses `inputarg_base = 0` (legacy positional
    /// layout); Phase 2/bridges shift inputarg OpRefs above the parent
    /// trace's high water mark by setting `inputarg_base = parent_high_water`.
    /// `reserve_pos` floors `next_pos` at `inputarg_base + num_inputs +
    /// new_operations.len()` so freshly emitted ops never collide with
    /// inputargs or imported high-water marks.
    pub(crate) inputarg_base: u32,
    /// Next unique op position for newly emitted or queued extra operations.
    pub(crate) next_pos: u32,
    /// RPython emit_extra(op, emit=False) parity: ops queued to be
    /// processed starting from a specific pass index (skipping earlier passes).
    /// Used by heap's force_lazy_set to route ops through remaining passes
    /// without re-entering the heap pass itself.
    /// Held as `OpRc` (resoperation.py: emit_extra appends a ResOperation
    /// object) so the queued op carries object identity into the drain.
    pub(crate) extra_operations_after: VecDeque<(usize, majit_ir::OpRc)>,
    /// optimizer.py:47-54: deferred postprocess for GUARD_CLASS.
    /// Set by rewrite pass, executed by emit_operation after the guard
    /// is added to new_operations (matching RPython's callback pattern).
    pub(crate) pending_guard_class_postprocess: Option<PendingGuardClassPostprocess>,
    /// rewrite.py:282: postprocess_GUARD_NONNULL → mark_last_guard.
    /// Deferred until emit adds the guard to new_operations.
    pub(crate) pending_mark_last_guard: Option<OpRef>,
    /// virtualize.py:84-90 postprocess_FINISH queues the stashed
    /// GUARD_NOT_FORCED_2 here so the outer optimizer can insert it at
    /// `len(_newoperations) - 1` with full `store_final_boxes_in_guard`
    /// semantics.
    pub(crate) pending_finish_guard_postprocess: Option<Op>,
    // ptr_info merged into forwarded (Forwarded::Info variant)
    //
    // RPython parity: per-OpRef IntBound storage lives ENTIRELY on
    // `box._forwarded` (Forwarded::IntBound), accessed via getintbound /
    // setintbound. The previous `int_lower_bounds` (heap.py array length
    // hint), `int_bounds` (per-pass snapshot), and `imported_int_bounds`
    // (preamble import) maps were a majit-only divergence from RPython's
    // single source of truth. They've been merged into Forwarded::IntBound
    // at write time so reads naturally see all sources via getintbound.
    /// RPython shortpreamble.py / pure.py: imported pure-operation results from
    /// the preamble. Phase 2 uses these as cross-iteration CSE facts.
    pub imported_short_pure_ops: Vec<ImportedShortPureOp>,
    /// (base_len, short_args): virtual field values start at base_len
    /// within short_args. Used by install_imported_virtuals.
    pub imported_virtual_args: Option<(usize, Vec<OpRef>)>,
    /// `rewrite.py:39` `self.loop_invariant_results = {}` — keyed by
    /// constant function pointer. PyPy uses a dict; pyre replaces it
    /// with a Vec of `(func_ptr, source_opref)` pairs and linear-scan
    /// dedup. CALL_LOOPINVARIANT is rare and the live set per trace is
    /// tiny, so O(n) lookup is acceptable.
    pub imported_loop_invariant_results: Vec<(i64, OpRef)>,
    /// Phase 2 imported virtuals (from Phase 1 export). Used by
    /// store_final_boxes_in_guard to resolve NONE positions
    /// inherited from Phase 1 virtualization.
    pub imported_virtuals: Vec<crate::optimizeopt::optimizer::ImportedVirtual>,
    /// Phase 2 imported label args (OpRefs in Phase 2 namespace).
    pub imported_label_args: Option<Vec<OpRef>>,
    /// RPython shortpreamble.py: active phase-2 short preamble builder.
    /// Tracks which imported short facts are actually consumed by the body.
    pub imported_short_preamble_builder:
        Option<crate::optimizeopt::shortpreamble::ShortPreambleBuilder>,
    /// `optimizer.py:243` `self.quasi_immutable_deps = None` (initialized
    /// lazily as a dict in `heap.py:806-808`). Each entry pairs an
    /// `(object_ptr, field_index)` quasi-immutable slot the trace
    /// depends on; PyPy uses `dict[k] = None` for set semantics, but the
    /// HashMap house rule forbids that — pyre uses a Vec with
    /// linear-scan dedup. Typical size is small (< a few dozen entries
    /// per trace), so O(n) inserts are acceptable.
    pub quasi_immutable_deps: Vec<(u64, u32)>,
    /// `info.py:722` `optheap.const_infos.get(ref, None)` /
    /// `info.py:725` `optheap.const_infos[ref] = info`. Stores
    /// `StructPtrInfo` / `ArrayPtrInfo` for constant GC objects keyed
    /// by pointer address. PyPy uses `new_ref_dict()`; the house rule
    /// forbids hash containers, so pyre uses a Vec-backed associative
    /// container with linear-scan lookup.
    pub const_infos: majit_ir::VecMap<usize, crate::optimizeopt::info::PtrInfo>,
    /// Dedup imported short fact uses so the builder stays in first-use
    /// order. PyPy uses dict-as-set; pyre uses a Vec with linear-scan
    /// dedup (small per trace).
    imported_short_preamble_used: Vec<OpRef>,
    /// `unroll.py:37` `self.optunroll.potential_extra_ops[op] = preamble_op` /
    /// `optimizer.py:354` `preamble_op = self.optunroll.potential_extra_ops.pop(op)`.
    /// PyPy uses a dict keyed by the box; pyre uses a Vec of `(OpRef,
    /// PreambleOp)` with linear-scan insert/pop/contains. The pool stays
    /// small per trace (one entry per imported pure short-preamble op),
    /// so O(n) operations are acceptable.
    ///
    /// The key stays the box's `OpRef` position rather than PyPy's box
    /// identity (`unroll.py:37` keys by the box object). This is a
    /// PRE-EXISTING-ADAPTATION beyond the Vec-vs-dict shape: the insert key
    /// is the Phase-1 preamble source box (`force_op_from_preamble_op`,
    /// `preamble_op.op`) while the `force_box` pop key is the Phase-2 body
    /// box resolved to a position (`get_replacement_opref`) — two distinct
    /// `Rc`s sharing one position across the peel boundary. A `BoxRef`
    /// `Rc::ptr_eq` key would silent-miss the pop. Re-keying to box identity
    /// is gated on the same short-preamble / InputArg identity unification
    /// that defers `resolve_box_box`'s InputArg arm (#9/S9).
    pub(crate) potential_extra_ops: Vec<(OpRef, crate::optimizeopt::info::PreambleOp)>,
    /// RPython unroll.py: live ExtendedShortPreambleBuilder while replaying an
    /// existing target token's short preamble.
    active_short_preamble_producer:
        Option<crate::optimizeopt::shortpreamble::ExtendedShortPreambleBuilder>,
    /// RPython shortpreamble.py: pass-collected preamble producers aligned to
    /// the exported loop-header inputargs.
    pub exported_short_boxes: Vec<crate::optimizeopt::shortpreamble::PreambleOp>,
    /// unroll.py:480 `short_inputargs = sb.create_short_inputargs(label_args
    /// + virtuals)` — the ShortBoxes-stored renamed inputarg boxes
    /// themselves, carried from the preview pass (optimizer.rs, where the
    /// ShortBoxes object lives) to `export_state_with_bounds` through the
    /// same channel as `exported_short_boxes`. Position projection equals
    /// the export-site `label_args + virtuals` recompute (measured
    /// identical across the corpus, 2026-06-11).
    pub exported_short_inputargs: Vec<majit_ir::box_ref::BoxRef>,
    /// Rooted `InputArgRc` carriers for `exported_short_inputargs`, index-
    /// aligned with that vector. Each renamed short-preamble input box
    /// (`exported_short_inputargs[i]`) holds a WEAK handle to
    /// `exported_short_inputarg_refs[i]`; keeping the strong Rc alive here
    /// lets the box resolve to a real bound `InputArg` (so the operand binds
    /// to `Operand::InputArg` instead of an unbound position-only box, which
    /// `from_boxref` now rejects). Carried alongside `exported_short_inputargs`
    /// through the same export channel.
    pub exported_short_inputarg_refs: Vec<majit_ir::InputArgRc>,
    /// optimizer.py: `can_replace_guards` — disable guard replacement during
    /// bridge compilation. Defaults to true for preamble.
    pub can_replace_guards: bool,
    /// RPython optimizer.py: `patchguardop` — the last GUARD_FUTURE_CONDITION op.
    /// Used by unroll to attach resume data to extra guards from short preamble.
    pub patchguardop: Option<Op>,
    /// RPython optimizer.py: end_args after force_at_the_end_of_preamble().
    /// export_state() prefers this over a raw get_replacement() snapshot.
    pub preamble_end_args: Option<Vec<OpRef>>,
    /// Phase-2 loop-body mode from optimizer.skip_flush.
    /// RPython unroll.py relies on this distinction so virtualize can keep
    /// body-side allocations concrete when guard recovery cannot rebuild them.
    pub skip_flush_mode: bool,
    /// Index of the pass currently executing propagate_forward.
    /// Used by passes to call send_extra_operation_after(self_idx, ..)
    /// matching RPython's emit_extra(op, emit=False) which routes to
    /// self.next_optimization.
    pub current_pass_idx: usize,
    /// earlyforce.py:32: self.optimizer.optearlyforce = self
    /// Index of the OptEarlyForce pass in the pass chain.
    /// Used by force_at_the_end_of_preamble and force_box to route
    /// forced operations starting from earlyforce.next (= heap).
    pub optearlyforce_idx: usize,
    /// optimizer.py: pendingfields — deferred SetfieldGc/SetarrayitemGc ops
    /// where the stored value is virtual. Set by OptHeap.emitting_operation()
    /// before a guard, consumed by emit_operation() to encode into
    /// the guard's rd_pendingfields.
    pub pending_for_guard: Vec<Op>,
    /// optimizer.py: pure_from_args1 parity — reverse-pure relationships
    /// registered by rewrite pass (CAST_*, CONVERT_*) and consumed by pure pass.
    /// Each entry: (opcode, arg0, result, descr) meaning
    /// pure((opcode, arg0), descr) = result. `descr` is `None` for
    /// the common case (no descr); `Some(DescrRef)` matches upstream
    /// `pure_from_args(rop.OPNUM, [arg], result, descr=op.getdescr())`
    /// (e.g. virtualize.py:220 ARRAYLEN_GC keying on the array descr).
    pub pending_pure_from_args: Vec<(OpCode, OpRef, OpRef, Option<majit_ir::DescrRef>)>,
    /// optimizer.py: pure_from_args2 parity — binary reverse-pure relationships
    /// registered by rewrite pass (INSTANCE_PTR_EQ/NE swapped-args). Consumed
    /// by OptPure. Each entry: (opcode, arg0, arg1, result) meaning
    /// pure(opcode, arg0, arg1) = result.
    pub pending_pure_from_args2: Vec<(OpCode, OpRef, OpRef, OpRef)>,
    /// optimizer.py:787: constant_fold allocator callback.
    /// When set, the optimizer can fold immutable virtuals filled with
    /// constants into compile-time constant pointers (info.py:140-145).
    pub constant_fold_alloc: Option<ConstantFoldAllocFn>,
    /// info.py:810-822 `ConstPtrInfo.getstrlen1(mode)` — runtime hook for
    /// constant byte-string / unicode-string length lookup. Set by the
    /// host runtime (pyre etc.) at OptContext construction time. When
    /// `None`, `EnsuredPtrInfo::getlenbound(Some(_))` falls back to
    /// `IntBound::nonnegative()`.
    pub string_length_resolver: Option<StringLengthResolver>,
    /// info.py:788-790 `ConstPtrInfo._unpack_str(mode)` — runtime hook for
    /// extracting character data from a constant string GcRef.
    pub string_content_resolver: Option<StringContentResolver>,
    /// history.py:377 `get_const_ptr_for_string(s)` — runtime hook for
    /// creating a constant string GcRef from char values (used by
    /// force_box constant-folding path, vstring.py:79-90).
    pub string_constant_alloc: Option<StringConstantAllocator>,
    /// True while optimizer.py:_emit_operation equivalent is forcing args
    /// just before final emission. In this phase, virtual forcing must emit
    /// directly into new_operations instead of re-entering the pass chain.
    pub in_final_emission: bool,
    /// effectinfo.py: CallInfoCollection — maps oopspec indices to
    /// (calldescr, func_ptr) pairs. Used by generate_modified_call
    /// (vstring.py:853) to emit specialized string comparison calls.
    pub callinfocollection: Option<std::sync::Arc<majit_ir::CallInfoCollection>>,
    /// resume.py parity: per-guard snapshot boxes from tracing time.
    /// Used by emit() to call store_final_boxes_in_guard inline (RPython
    /// calls this during optimization, not post-assembly).
    pub snapshot_boxes: SnapshotBoxes,
    /// Per-frame box counts for multi-frame snapshots.
    /// opencoder.py:819 capture_resumedata encodes multiple frames;
    /// this tracks the boundary between callee and caller sections.
    pub snapshot_frame_sizes: SnapshotFrameSizes,
    /// Per-guard virtualizable boxes from tracing-time snapshots.
    pub snapshot_vable_boxes: SnapshotBoxes,
    /// Per-guard virtualref boxes from tracing-time snapshots.
    /// resume.py:243-247 _number_boxes consumes vref_array as a section
    /// after vable_array. opencoder.py:767 records vref_boxes here.
    pub snapshot_vref_boxes: SnapshotBoxes,
    /// Per-guard per-frame (jitcode_index, pc) from tracing-time snapshots.
    pub snapshot_frame_pcs: SnapshotFramePcs,
    /// optimizer.py:34 `self.inputargs = inputargs` parity.
    /// Typed InputArg OpRefs; slot `i` is `OpRef::input_arg_typed(i, tp)`.
    pub inputargs: Vec<majit_ir::OpRef>,
    /// Strong `InputArgRc` ownership for the inputargs seeded by
    /// `with_inputarg_types`. Production traces own their `InputArgRc`s
    /// via `TreeLoop.inputargs`; the test-and-fallback helper
    /// `with_inputarg_types` has no upstream `TreeLoop`, so it stashes
    /// fresh `InputArgRc`s here to keep the `Weak<InputArg>` stored
    /// inside each `BoxRef.inputarg_handle` upgradable. `make_equal_to`
    /// then routes the chain step through `Forwarded::InputArg(_)`
    /// (`optimizer.py:394 op.set_forwarded(newop)`) instead of the
    /// retired orphan-box forwarding fallback.
    pub(crate) inputarg_refs: Vec<majit_ir::InputArgRc>,
    /// Synthetic `OpRc` stand-ins for ResOp BoxRef placeholders whose
    /// real producer has not been (and may never be) emitted, indexed
    /// sparsely by `OpRef::raw()`. `materialize_box_at` falls back to
    /// synthesising a `SameAsI/F/R` (or `Jump`) Op with the requested
    /// type and binding the BoxRef to it so `make_equal_to` routes a
    /// chain step that targets such a placeholder through
    /// `Forwarded::Op(_)`. When a real producer Op is later emitted at
    /// the same OpRef position, `emit()` re-binds the BoxRef to that
    /// Op (carrying forwarded state across) and the synthetic stand-in
    /// becomes unreferenced from the BoxRef but is still retained here
    /// for the OptContext's lifetime so any lingering `Weak<Op>`
    /// upgrades (e.g. in already-installed `Forwarded::Op` chains)
    /// stay valid.
    ///
    /// Keyed by the full type-tagged `OpRef`, so a typed and an untyped
    /// (or differently-typed) position sharing a raw `u32` are distinct
    /// entries instead of evicting each other in a raw-indexed slot.
    // Insertion-ordered map (`IndexMap`) rather than the Vec-backed
    // `VecMap`: `find_producer_op` does `resop_refs.get(&opref)` once per
    // live box of every guard inside `store_final_boxes_in_guard`, and a
    // Vec-backed `get` is O(n), making the box-numbering O(n^2) on very
    // large traces (aheui's logo loop spends ~all its compile time there).
    // `IndexMap` keeps the same insertion-ordered `values()` semantics the
    // earlier container guaranteed but resolves `get` in O(1).  Same O(1)
    // acceleration rationale as `input_ops_index`; no PyPy counterpart
    // (upstream keys producers on `box._forwarded`, not a positional map).
    pub(crate) resop_refs: indexmap::IndexMap<OpRef, majit_ir::resoperation::OpRc>,
    /// Live synthetic stand-ins (mint_synthetic_resop / bind_input_resops
    /// products) that have NOT been superseded by an `emit` at their
    /// position. The end-of-Phase-1 orphan-binding pass drains this into
    /// `phase1_emit_ops` so retrace's `Weak<Op>` upgrades stay valid; an
    /// `emit` that rebinds a position's box off its synthetic removes the
    /// synthetic here (it stays strongly held by `resop_refs` for lookup,
    /// but is no longer an orphan needing carry). Tracking liveness
    /// incrementally by `OpRc` identity sidesteps the flat-OpRef raw/type
    /// collision that makes the final `new_operations` state ambiguous
    /// about which type-tagged value won a shared raw slot.
    pub(crate) live_synthetics: Vec<majit_ir::resoperation::OpRc>,
    /// Phase 1 emit ops carried into Phase 2's lookup surface.
    ///
    /// In RPython, a Box referenced cross-phase keeps its `.type` attribute
    /// because the Box object survives both phases (history.py:220 parity).
    /// Pyre's flat `OpRef(u32)` requires explicit carry: Phase 1's emit ops
    /// — at positions in `[num_inputs, phase2_inputarg_base)` — appear as
    /// `imported_label_args`, fail-arg sources, and `record_same_as` source
    /// arguments inside Phase 2, but never inside Phase 2's own `new_operations`.
    /// `op_at` falls back to this slice so `op.type_` stays the single source
    /// of truth for Phase 1 emit OpRef types.
    pub phase1_emit_ops: Vec<majit_ir::OpRc>,
    /// `pos -> producer` index over `phase1_emit_ops`, mirroring the `rfind`
    /// (last-occurrence-wins) lookup in `find_producer_op`. This OptContext
    /// field is written exactly once — the Phase 1→2 handoff
    /// (`ctx.phase1_emit_ops = mem::take(...)` in `optimizer.rs`) — and never
    /// mutated afterwards, so the index is rebuilt in lockstep there via
    /// `rebuild_phase1_emit_ops_index`. Without it, `find_producer_op` scans
    /// the full cross-phase carry (~60k ops on aheui's logo loop) per live
    /// box of every Phase 2 guard, the dominant O(n^2) compile cost. Same
    /// derived-index rationale as `input_ops_index` (no PyPy counterpart:
    /// upstream keys producers on `box._forwarded`, not a positional map).
    pub(crate) phase1_emit_ops_index: std::collections::HashMap<OpRef, majit_ir::OpRc>,
    /// Recorder trace ops that carry the input operands' producer `Op`
    /// (e.g. the `IntLt`/`GetfieldGcPureI` operands of a recorded loop),
    /// shared by `Rc` with the canonical stores but absent from
    /// `new_operations` / `phase1_emit_ops` / `resop_refs`. Seeded at
    /// optimizer setup from the recorder's `Rc<Op>` slice (`TreeLoop.ops`
    /// at the loop-finish / simple-loop sites, or the Phase-2 threaded
    /// `explicit_input_ops_seed`). `find_producer_op` consults this as the
    /// lowest-priority store so a later emission at the same position always
    /// wins.
    pub(crate) input_ops: Vec<majit_ir::OpRc>,
    /// `pos -> producer` index over `input_ops`, mirroring the `rfind`
    /// (last-occurrence-wins) lookup in `find_producer_op`. `input_ops` is
    /// seeded once at optimizer setup and never mutated afterwards, so this
    /// is rebuilt in lockstep with the seeding assignment via
    /// `rebuild_input_ops_index`. Eliminates the O(n) scan of the full
    /// recorder trace that `find_producer_op` otherwise performs on every
    /// miss of the higher-priority stores (the dominant O(n^2) cost on very
    /// large traces, e.g. aheui's logo loop).
    ///
    /// No PyPy counterpart: upstream keys producer information on the box
    /// itself (`box._forwarded` / `PtrInfo`, `optimizer.py`), so a
    /// positional producer scan never exists there. Pyre's flat
    /// `OpRef(u32)` has no such per-box slot, so `find_producer_op` scans
    /// by position; this map is a pure O(1) acceleration of that scan, not
    /// a new data model. Permitted under the AGENTS.md HashMap rule (3)
    /// because it is a derived index — its sole invariant is that
    /// `get(pos)` equals `input_ops.iter().rfind(|o| o.pos == pos)`
    /// (last occurrence wins), enforced by rebuilding forward (a later
    /// `insert` at the same key overwrites the earlier) and covered by
    /// `input_ops_index_last_occurrence_wins`.
    pub(crate) input_ops_index: std::collections::HashMap<OpRef, majit_ir::OpRc>,
    /// optimizer.py:644,679 _last_guard_op — index of the last guard in
    /// new_operations that had full resume data built. Consecutive guards
    /// share resume data via _copy_resume_data_from (ResumeGuardCopiedDescr).
    ///
    /// Production runs through `Optimizer::emit_operation`, which owns the
    /// guard chain via its own `last_guard_op_idx` (optimizer.rs:3584).
    /// This field tracks the chain for the standalone OptContext entry
    /// (unit tests that drive `OptContext::emit` without an `Optimizer`);
    /// `OptContext::emit` gates its guard handling on `!in_final_emission`
    /// to avoid duplicating Optimizer's bookkeeping.
    last_guard_idx: Option<usize>,
    /// Last rd_resume_position with a valid snapshot. Used as fallback
    /// for optimizer-created guards that can't share from a previous guard.
    /// resume.py parity: RPython guards always get a snapshot via
    /// capture_resumedata; pyre tracks the nearest valid position.
    pub last_seen_snapshot_pos: Option<i32>,
    /// `optimizer.cpu` (`model.py:39 AbstractCPU`) backref — the
    /// shared backend services entry point Optimization sub-classes
    /// reach via `self.optimizer.cpu.<method>()` in RPython.  Pyre's
    /// OptContext (the shared state holder Optimization sub-classes
    /// route through) carries an `Arc<dyn Cpu>` clone of `Optimizer.cpu`
    /// because it has no direct backref to the surrounding Optimizer.
    /// Used by `cls_of_box(&BoxRef)` (mod.rs body) and reachable by
    /// future `bh_*` ports.
    pub cpu: std::sync::Arc<dyn crate::cpu::Cpu>,
    /// llmodel.py:55 `self.remove_gctypeptr =
    /// translator.config.translation.gcremovetypeptr`. model.py:26
    /// default is `False`; PyPy x86 enables `--gcremovetypeptr` which
    /// flips this to `True` so the GC header carries the typeid and
    /// `obj[0]` no longer holds the rclass typeptr.
    ///
    /// Pyre's PyObject layout keeps `ob_type` at offset 0 BUT many
    /// static singletons (INSTANCE_TYPE, INT_TYPE, …) carry no GC
    /// header — `GUARD_IS_OBJECT` reads `obj - GcHeader::SIZE` and
    /// SIGBUSes on them. The pyre default matches the `True` branch
    /// (skip GUARD_IS_OBJECT, emit GUARD_NONNULL_CLASS as a single
    /// op). Consumed by `info.py:338/:348 InstancePtrInfo.make_guards`.
    pub remove_gctypeptr: bool,
    /// optimizer.py:84-92 `Optimization.last_emitted_operation` — set
    /// to the just-emitted op (or `REMOVED` sentinel) by the base
    /// class's `_emit_operation`, read by callers like
    /// `optimize_GUARD_NO_EXCEPTION` (rewrite.py:712-718) to check
    /// whether the preceding op was dropped. The slot is updated on
    /// every emit across all passes, so a remove in pass N is visible
    /// to pass N+1.
    ///
    /// Pyre folds the REMOVED sentinel into a `bool` and lifts the
    /// slot from per-pass storage (where it was OptRewrite-local) to
    /// `OptContext` so the cross-pass scope matches the upstream
    /// base-class contract. Set by `propagate_from_pass_range` on
    /// `OptimizationResult::Remove`, reset on every successful
    /// `emit_operation` (see `Optimizer::emit_operation`).
    pub last_op_removed: bool,
    /// Deferred `InvalidLoop` signal.  RPython `raise InvalidLoop`
    /// abandons the trace; pyre carries the same signal as a
    /// `Result<_, InvalidLoop>` threaded through the optimizer driver so
    /// it works under `panic=abort` (wasm32) as well as `panic=unwind`.
    /// Leaf sites that cannot return `Result` directly (e.g.
    /// `get_const_info_mut_box`, deep inside a pass) record the signal
    /// here and return a benign value; the driver checks it at the next
    /// barrier (`propagate_from_pass_range`) and converts it to `Err`.
    /// First write wins so the innermost reason is preserved.  `Cell` so
    /// `&self` leaf methods (e.g. `protect_speculative_operation`) can
    /// record without threading `&mut`.
    pub pending_invalid_loop: std::cell::Cell<Option<crate::optimize::InvalidLoop>>,
}

/// heaptracker.py:66: `if name == 'typeptr': continue`
/// Uses FieldDescr.is_typeptr() which checks `field_name() == "typeptr"`,
/// matching RPython's name-based filtering.
#[inline(always)]
pub(crate) fn is_typeptr_field(
    field_idx: u32,
    field_descrs: &[majit_ir::DescrRef],
    _descr: &majit_ir::DescrRef,
) -> bool {
    field_descrs
        .get(field_idx as usize)
        .and_then(|d| d.as_field_descr())
        .map(|fd| fd.is_typeptr())
        .unwrap_or(false)
}

/// resume.py:192-226 parity — BoxEnv for optimizer context.
///
/// Wraps an immutable reference to OptContext, implementing the BoxEnv
/// trait so that ResumeDataLoopMemo.number() can tag boxes during
/// store_final_boxes_in_guard.
pub struct OptBoxEnv<'a> {
    pub ctx: &'a OptContext,
}

impl<'a> majit_ir::BoxEnv for OptBoxEnv<'a> {
    fn get_box_replacement(&self, opref: OpRef) -> OpRef {
        self.ctx.get_replacement_opref(opref)
    }

    fn get_box_replacement_operand(&self, opref: OpRef) -> Operand {
        // resume.py:202 box.get_box_replacement() as a box OBJECT. The canonical
        // host is the producer Op / InputArg, so two reaches of one logical box
        // return the same producer Rc (ptr_eq) — the #160/S11 livebox dedup key.
        // `get_box_replacement_operand_opt` carries the debug-build tripwire that
        // the native Operand walk agrees with the BoxRef form on presence and
        // identity, so the resume-numbering path validates the BoxRef→Operand
        // equivalence across the corpus. The fallback PANICS on a producerless
        // position (E3 dropped the position-only Operand variant), the armed
        // hazard-5 tripwire: a non-Const numbering key with no findable producer
        // would otherwise mint a fresh, non-ptr_eq box and corrupt the livebox
        // dedup. #157 drained these fires to zero across the corpus.
        if opref.is_none() {
            return Operand::None;
        }
        self.ctx
            .get_box_replacement_operand_opt(opref)
            .unwrap_or_else(|| Operand::from_opref(self.ctx.get_replacement_opref(opref)))
    }

    fn get_box_replacement_not_const(&self, opref: OpRef) -> OpRef {
        // resoperation.py:64-65 not_const arm. The resume liveboxes boundary
        // is legitimately OpRef-keyed (rd_numb wire format), so resolve
        // through the OpRef glue directly.
        self.ctx.get_box_replacement_impl(opref, true)
    }

    fn is_const(&self, opref: OpRef) -> bool {
        if self
            .ctx
            .get_box_replacement_operand_opt(opref)
            .and_then(|cb| cb.const_value())
            .is_some()
        {
            return true;
        }
        matches!(
            self.ctx
                .get_box_replacement_operand_opt(opref)
                .as_ref()
                .and_then(|b| self.ctx.peek_ptr_info(b)),
            Some(crate::optimizeopt::info::PtrInfo::Constant(_))
        )
    }

    fn get_const(&self, opref: OpRef) -> (i64, majit_ir::Type) {
        match self
            .ctx
            .get_box_replacement_operand_opt(opref)
            .and_then(|cb| cb.const_value())
        {
            Some(Value::Int(v)) => (v, majit_ir::Type::Int),
            Some(Value::Float(f)) => (f.to_bits() as i64, majit_ir::Type::Float),
            Some(Value::Ref(r)) => (r.0 as i64, majit_ir::Type::Ref),
            Some(Value::Void) => (0, majit_ir::Type::Int),
            None => {
                if let Some(crate::optimizeopt::info::PtrInfo::Constant(gcref)) = self
                    .ctx
                    .get_box_replacement_operand_opt(opref)
                    .as_ref()
                    .and_then(|b| self.ctx.peek_ptr_info(b))
                {
                    (gcref.0 as i64, majit_ir::Type::Ref)
                } else {
                    (0, majit_ir::Type::Int)
                }
            }
        }
    }

    fn get_type(&self, opref: OpRef) -> majit_ir::Type {
        // `BoxEnv::get_type` is the resume-serdes `box.type` reader
        // (resume.py:201 `box = box.get_box_replacement()` before any
        // type judgement). It is the non-`Option` adapter over
        // `OptContext::opref_type`, the single 5-layer `box.type`
        // resolver that mirrors upstream `AbstractValue.type`
        // (resoperation.py:29). A VoidOp (`Some(Void)`) or a type-less
        // OpRef (`None`) maps to the `Int` default the resume encoder
        // expects for non-typed slots.
        match self.ctx.opref_type(opref) {
            Some(tp) if tp != majit_ir::Type::Void => tp,
            _ => majit_ir::Type::Int,
        }
    }

    fn is_virtual_ref(&self, opref: OpRef) -> bool {
        // info.py:880-886 getptrinfo(op) first applies get_box_replacement(op)
        // before reading PtrInfo. Guard resume numbering walks ORIGINAL
        // snapshot boxes, so virtual classification must follow the same
        // replacement chain or forwarded virtual boxes get mis-tagged as
        // ordinary liveboxes.
        let resolved_box = self.ctx.get_box_replacement_operand_opt(opref);
        resolved_box
            .as_ref()
            .and_then(|b| self.ctx.peek_ptr_info(b))
            .is_some_and(|info| info.is_virtual())
    }

    fn is_virtual_raw(&self, opref: OpRef) -> bool {
        // info.py:865 `RawBufferPtrInfo` / RawSlicePtrInfo — Int-typed
        // virtuals.  `get_type()` already classifies these as Int; mirror
        // the classification here so resume encoding (`resume.rs:3672`)
        // picks them up via TAGVIRTUAL instead of TAGBOX.
        let resolved_box = self.ctx.get_box_replacement_operand_opt(opref);
        resolved_box
            .as_ref()
            .and_then(|b| self.ctx.peek_ptr_info(b))
            .is_some_and(|info| {
                matches!(
                    info,
                    crate::optimizeopt::info::PtrInfo::VirtualRawBuffer(_)
                        | crate::optimizeopt::info::PtrInfo::VirtualRawSlice(_)
                )
            })
    }

    fn has_known_class(&self, opref: OpRef) -> bool {
        // bridgeopt.py:79-80: getptrinfo(box).get_known_class(cpu) is not None
        let resolved_box = self.ctx.get_box_replacement_operand_opt(opref);
        resolved_box
            .as_ref()
            .and_then(|b| self.ctx.peek_ptr_info(b))
            .and_then(|info| info.get_known_class(self.ctx.cpu.as_ref()))
            .is_some()
    }

    fn get_virtual_fields(&self, opref: OpRef) -> Option<majit_ir::VirtualFieldsInfo> {
        let resolved_box = self.ctx.get_box_replacement_operand_opt(opref);
        let info = resolved_box
            .as_ref()
            .and_then(|b| self.ctx.peek_ptr_info(b))?;
        let fielddescrs = info.all_fielddescrs_from_descr();
        match info {
            PtrInfo::Virtual(vi) => Some(majit_ir::VirtualFieldsInfo {
                descr: Some(vi.descr.clone()),
                known_class: vi.known_class,
                // info.py:243-247 `_visitor_walk_recursive` registers the
                // full `_fields` list in descriptor order, leaving unfilled
                // slots as `None`. Preserve that shape so `fieldnums` aligns
                // 1:1 with `descr.get_all_fielddescrs()` for `_cached_vinfo`
                // reuse at resume.py:307-315.
                field_oprefs: fielddescrs
                    .iter()
                    .enumerate()
                    .map(|(fi, _)| {
                        vi.fields
                            .iter()
                            .find(|(field_idx, _)| *field_idx == fi as u32)
                            .map(|(_, vref)| self.ctx.get_replacement_opref(vref.to_opref()))
                            .unwrap_or(OpRef::NONE)
                    })
                    .collect(),
            }),
            PtrInfo::VirtualStruct(vi) => Some(majit_ir::VirtualFieldsInfo {
                descr: Some(vi.descr.clone()),
                known_class: None,
                field_oprefs: fielddescrs
                    .iter()
                    .enumerate()
                    .map(|(fi, _)| {
                        vi.fields
                            .iter()
                            .find(|(field_idx, _)| *field_idx == fi as u32)
                            .map(|(_, vref)| self.ctx.get_replacement_opref(vref.to_opref()))
                            .unwrap_or(OpRef::NONE)
                    })
                    .collect(),
            }),
            PtrInfo::VirtualArray(vi) => Some(majit_ir::VirtualFieldsInfo {
                descr: Some(vi.descr.clone()),
                known_class: None,
                field_oprefs: vi
                    .items
                    .iter()
                    .map(|vref| self.ctx.get_replacement_opref(vref.to_opref()))
                    .collect(),
            }),
            PtrInfo::VirtualArrayStruct(vi) => Some(majit_ir::VirtualFieldsInfo {
                descr: Some(vi.descr.clone()),
                known_class: None,
                field_oprefs: vi
                    .element_fields
                    .iter()
                    .flat_map(|ef| {
                        vi.fielddescrs.iter().enumerate().map(|(fi, _)| {
                            ef.iter()
                                .find(|(field_idx, _)| *field_idx == fi as u32)
                                .map(|(_, vref)| self.ctx.get_replacement_opref(vref.to_opref()))
                                .unwrap_or(OpRef::NONE)
                        })
                    })
                    .collect(),
            }),
            PtrInfo::VirtualRawBuffer(vi) => Some(majit_ir::VirtualFieldsInfo {
                descr: None,
                known_class: None,
                field_oprefs: vi
                    .buffer
                    .values()
                    .iter()
                    .map(|vref| self.ctx.get_replacement_opref(*vref))
                    .collect(),
            }),
            // `info.py:478-482` `RawSlicePtrInfo._visitor_walk_recursive`:
            //
            // ```python
            // def _visitor_walk_recursive(self, op, visitor):
            //     source_op = get_box_replacement(op.getarg(0))
            //     visitor.register_virtual_fields(op, [source_op])
            //     if self.parent.is_virtual():
            //         self.parent.visitor_walk_recursive(source_op, visitor)
            // ```
            //
            // pyre's consumer (`resume.rs::encode_*` worklist at
            // `resume.rs:3517`) drives the recursion off `get_virtual_fields`
            // — registering the parent OpRef here lets the worklist enqueue
            // the parent and re-enter `get_virtual_fields` on it, which
            // matches RPython's `parent.visitor_walk_recursive(source_op,
            // visitor)` follow-up.  Only fires while the slice is still
            // virtual (`slice.parent` non-NONE); after `force_box_impl`
            // materializes the slice the gate in `is_virtual` flips False
            // and the caller drops the entry.
            PtrInfo::VirtualRawSlice(vi) if !vi.parent.is_none() => {
                Some(majit_ir::VirtualFieldsInfo {
                    descr: None,
                    known_class: None,
                    field_oprefs: vec![self.ctx.get_replacement_opref(vi.parent.to_opref())],
                })
            }
            // vstring.py:207-208 VStringPlainInfo._visitor_walk_recursive:
            //   `visitor.register_virtual_fields(instbox, self._chars)`
            //
            // vstring.py:255-260 VStringSliceInfo._visitor_walk_recursive:
            //   `visitor.register_virtual_fields(instbox, [self.s, self.start, self.lgtop])`
            //   (then recurses into the parent string if it is itself virtual).
            //
            // vstring.py:319-325 VStringConcatInfo._visitor_walk_recursive:
            //   `visitor.register_virtual_fields(instbox, [self.vleft, self.vright])`
            //
            // Only virtual StrPtrInfo variants register fields; a non-virtual
            // `VStringVariant::Ptr` skips this arm (falls through to None).
            PtrInfo::Str(sinfo) if sinfo.is_virtual() => {
                use crate::optimizeopt::info::VStringVariant;
                let field_oprefs: Vec<OpRef> = match &sinfo.variant {
                    // vstring.py:207-208: self._chars. `None` slots represent
                    // unfilled positions — the resume encoder later tags those
                    // with UNINITIALIZED (matching how RPython treats missing
                    // char boxes, since STRSETITEM may not have run for every
                    // index yet at guard time).
                    VStringVariant::Plain(p) => p
                        ._chars
                        .iter()
                        .map(|slot| {
                            slot.as_ref()
                                .map(|r| self.ctx.get_replacement_opref(r.to_opref()))
                                .unwrap_or(OpRef::NONE)
                        })
                        .collect(),
                    // vstring.py:255-257: [self.s, self.start, self.lgtop].
                    VStringVariant::Slice(s) => vec![
                        self.ctx.get_replacement_opref(s.s.to_opref()),
                        self.ctx.get_replacement_opref(s.start.to_opref()),
                        self.ctx.get_replacement_opref(s.lgtop.to_opref()),
                    ],
                    // vstring.py:319-324: [self.vleft, self.vright].
                    VStringVariant::Concat(c) => vec![
                        self.ctx.get_replacement_opref(c.vleft.to_opref()),
                        self.ctx.get_replacement_opref(c.vright.to_opref()),
                    ],
                    // Non-virtual `VStringVariant::Ptr` would not reach
                    // here because of the `is_virtual()` guard above.
                    VStringVariant::Ptr => Vec::new(),
                };
                Some(majit_ir::VirtualFieldsInfo {
                    descr: None,
                    known_class: None,
                    field_oprefs,
                })
            }
            _ => None,
        }
    }

    fn make_virtual_info(
        &self,
        opref: OpRef,
        fieldnums: Vec<i16>,
    ) -> Option<std::rc::Rc<majit_ir::RdVirtualInfo>> {
        let resolved_box = self.ctx.get_box_replacement_operand_opt(opref);
        let info = resolved_box
            .as_ref()
            .and_then(|b| self.ctx.peek_ptr_info(b))?;
        // resume.py:307-315 `ResumeDataVirtualAdder.make_virtual_info`:
        //
        //     vinfo = info._cached_vinfo
        //     if vinfo is not None and vinfo.equals(fieldnums):
        //         return vinfo
        //     vinfo = info.visitor_dispatch_virtual_type(self)
        //     vinfo.set_content(fieldnums)
        //     info._cached_vinfo = vinfo
        //     return vinfo
        //
        // The cache stores an `Rc<RdVirtualInfo>` so that cache hits return
        // the same shared handle (matching RPython's Python object identity
        // on cache hit, info.py:124-128). Downstream storage in
        // `storage.rd_virtuals` keeps the shared handle so two guards that
        // reference the same virtual with the same fieldnums end up pointing
        // at the same `RdVirtualInfo` object.
        if let Some(cache) = info.cached_vinfo() {
            if let Some(vinfo) = cache.borrow().as_ref() {
                if vinfo.equals(&fieldnums) {
                    return Some(std::rc::Rc::clone(vinfo));
                }
            }
        }
        let mut builder = RdVirtualInfoBuilder;
        let mut vinfo = info.visitor_dispatch_virtual_type(&mut builder)??;
        // resume.py:313: vinfo.set_content(fieldnums)
        vinfo.set_content(fieldnums);
        let shared = std::rc::Rc::new(vinfo);
        // resume.py:314: info._cached_vinfo = vinfo — store the shared handle
        // so a later equals-hit returns the SAME object.
        if let Some(cache) = info.cached_vinfo() {
            *cache.borrow_mut() = Some(std::rc::Rc::clone(&shared));
            // `info` is a clone returned from `peek_ptr_info`;
            // mutating its independent `cached_vinfo` RefCell does not feed
            // back into the BoxRef canonical slot. Project the cached Rc
            // handle directly onto the BoxRef PtrInfo so subsequent
            // BoxRef-routing readers (`virtual_info_would_be_reused`)
            // observe the cached vinfo.
            if let Some(b) = resolved_box.as_ref() {
                if let Some(pi) = b.ptr_info_mut() {
                    if let Some(c) = pi.cached_vinfo() {
                        *c.borrow_mut() = Some(std::rc::Rc::clone(&shared));
                    }
                }
            }
        }
        Some(shared)
    }

    fn virtual_info_would_be_reused(&self, opref: OpRef, fieldnums: &[i16]) -> bool {
        // BoxRef-routing reader; cached_vinfo's RefCell clones shallowly so the
        // inner Rc<RdVirtualInfo> is shared with the canonical PtrInfo — read of
        // .borrow() yields the same content as the original cache.
        let resolved_box = self.ctx.get_box_replacement_operand_opt(opref);
        let Some(info) = resolved_box
            .as_ref()
            .and_then(|b| self.ctx.peek_ptr_info(b))
        else {
            return false;
        };
        let Some(cache) = info.cached_vinfo() else {
            return false;
        };
        cache
            .borrow()
            .as_ref()
            .is_some_and(|vinfo| vinfo.equals(fieldnums))
    }
}

/// resume.py:298-357 `ResumeDataVirtualAdder` in its role as a
/// `VirtualVisitor` — each `visit_*` builds a fresh `RdVirtualInfo`
/// subclass without fieldnums (the caller attaches those via
/// `set_content`). In pyre, `make_virtual_info` lives on the
/// `BoxEnv` impl (not on `ResumeDataVirtualAdder`) because PtrInfo
/// lookup is the optimizer's responsibility, so the visitor adapter
/// is this zero-sized helper instead of `ResumeDataVirtualAdder`
/// itself.
struct RdVirtualInfoBuilder;

impl crate::walkvirtual::VirtualVisitor for RdVirtualInfoBuilder {
    type VInfo = Option<majit_ir::RdVirtualInfo>;

    fn visit_not_virtual(&mut self, _value: OpRef) -> Self::VInfo {
        // resume.py:317-318 `visit_not_virtual` asserts unreachable.
        debug_assert!(false, "visit_not_virtual reached via virtual dispatch");
        None
    }

    // resume.py:320-321 visit_virtual → VirtualInfo(descr, fielddescrs)
    fn visit_virtual(
        &mut self,
        descr: &majit_ir::DescrRef,
        _fielddescr_indices: &[u32],
        fielddescrs: &[majit_ir::DescrRef],
    ) -> Self::VInfo {
        // `FieldDescrInfo.index` must carry the stable descriptor index
        // (tagged offset) so resume-data readers can identify the field by
        // byte offset. Previously stored `fi as u32` (iteration counter),
        // which made `extract_pyre_field_offset` always fail for virtuals
        // being materialized on bridge entry.
        let built_fielddescrs: Vec<majit_ir::FieldDescrInfo> = fielddescrs
            .iter()
            .map(|descr| {
                let fd = descr.as_field_descr();
                majit_ir::FieldDescrInfo {
                    index: descr.index(),
                    offset: fd.map(|f| f.offset()).unwrap_or(0),
                    field_type: fd.map(|f| f.field_type()).unwrap_or(majit_ir::Type::Int),
                    field_size: fd.map(|f| f.field_size()).unwrap_or(8),
                }
            })
            .collect();
        let sd = descr.as_size_descr();
        Some(majit_ir::RdVirtualInfo::VirtualInfo {
            descr: Some(descr.clone()),
            type_id: sd.map(|s| s.type_id()).unwrap_or(0),
            // resume.py:619 allocate_with_vtable(descr=self.descr) — the
            // vtable is derived from descr; majit mirrors by reading it
            // off the SizeDescr when the descr carries class info.
            known_class: sd.map(|s| s.vtable() as i64).filter(|&v| v != 0),
            fielddescrs: built_fielddescrs,
            fieldnums: Vec::new(),
            descr_size: sd.map(|s| s.size()).unwrap_or(0),
        })
    }

    // resume.py:323-324 visit_vstruct → VStructInfo(typedescr, fielddescrs)
    fn visit_vstruct(
        &mut self,
        typedescr: &majit_ir::DescrRef,
        _fielddescr_indices: &[u32],
        fielddescrs: &[majit_ir::DescrRef],
    ) -> Self::VInfo {
        // See `visit_virtual` — index must be the stable descriptor index,
        // not the iteration counter.
        let built_fielddescrs: Vec<majit_ir::FieldDescrInfo> = fielddescrs
            .iter()
            .map(|descr| {
                let fd = descr.as_field_descr();
                majit_ir::FieldDescrInfo {
                    index: descr.index(),
                    offset: fd.map(|f| f.offset()).unwrap_or(0),
                    field_type: fd.map(|f| f.field_type()).unwrap_or(majit_ir::Type::Int),
                    field_size: fd.map(|f| f.field_size()).unwrap_or(8),
                }
            })
            .collect();
        let sd = typedescr.as_size_descr();
        Some(majit_ir::RdVirtualInfo::VStructInfo {
            typedescr: Some(typedescr.clone()),
            type_id: sd.map(|s| s.type_id()).unwrap_or(0),
            fielddescrs: built_fielddescrs,
            fieldnums: Vec::new(),
            descr_size: sd.map(|s| s.size()).unwrap_or(0),
        })
    }

    // resume.py:326-330 visit_varray → VArrayInfoClear / VArrayInfoNotClear
    fn visit_varray(&mut self, arraydescr: &majit_ir::DescrRef, clear: bool) -> Self::VInfo {
        let kind = arraydescr
            .as_array_descr()
            .map(|ad| match ad.item_type() {
                majit_ir::Type::Float => 2u8,
                majit_ir::Type::Int => 1u8,
                _ => 0u8,
            })
            .unwrap_or(0);
        let ad = Some(arraydescr.clone());
        Some(if clear {
            majit_ir::RdVirtualInfo::VArrayInfoClear {
                arraydescr: ad,
                kind,
                fieldnums: Vec::new(),
            }
        } else {
            majit_ir::RdVirtualInfo::VArrayInfoNotClear {
                arraydescr: ad,
                kind,
                fieldnums: Vec::new(),
            }
        })
    }

    // resume.py:332-333 visit_varraystruct → VArrayStructInfo
    fn visit_varraystruct(
        &mut self,
        arraydescr: &majit_ir::DescrRef,
        length: usize,
        _fielddescr_indices: &[u32],
        fielddescrs: &[majit_ir::DescrRef],
    ) -> Self::VInfo {
        // info.py:701-704: visitor_dispatch_virtual_type always hands
        // down the canonical get_all_interiorfielddescrs() list; fall
        // back to the variant's cached fielddescrs when descr lacks it.
        let canonical_fielddescrs: Vec<majit_ir::DescrRef> = arraydescr
            .as_array_descr()
            .and_then(|ad| ad.get_all_interiorfielddescrs())
            .map(|fds| fds.to_vec())
            .unwrap_or_else(|| fielddescrs.to_vec());
        let mut fo = Vec::new();
        let mut fs = Vec::new();
        let mut ft = Vec::new();
        for fd in &canonical_fielddescrs {
            if let Some(ifd) = fd.as_interior_field_descr() {
                let fld = ifd.field_descr();
                fo.push(fld.offset());
                fs.push(fld.field_size());
                ft.push(match fld.field_type() {
                    majit_ir::Type::Float => 2u8,
                    majit_ir::Type::Int => 1u8,
                    _ => 0u8,
                });
            } else {
                fo.push(fo.len() * 8);
                fs.push(8);
                ft.push(0);
            }
        }
        if ft.is_empty() {
            ft = vec![0u8; canonical_fielddescrs.len()];
        }
        let ad = arraydescr.as_array_descr();
        let fielddescr_count = canonical_fielddescrs.len();
        Some(majit_ir::RdVirtualInfo::VArrayStructInfo {
            arraydescr: Some(arraydescr.clone()),
            size: length,
            fielddescrs: canonical_fielddescrs,
            fielddescr_indices: (0..fielddescr_count).map(|i| i as u32).collect(),
            field_types: ft,
            base_size: ad.map(|a| a.base_size()).unwrap_or(0),
            item_size: ad.map(|a| a.item_size()).unwrap_or(0),
            field_offsets: fo,
            field_sizes: fs,
            fieldnums: Vec::new(),
        })
    }

    // resume.py:335-336 visit_vrawbuffer → VRawBufferInfo
    fn visit_vrawbuffer(
        &mut self,
        func: i64,
        size: usize,
        offsets: &[i64],
        descrs: &[majit_ir::DescrRef],
    ) -> Self::VInfo {
        let descr_infos: Vec<majit_ir::ArrayDescrInfo> = descrs
            .iter()
            .map(|d| {
                let ad = d.as_array_descr();
                majit_ir::ArrayDescrInfo {
                    index: d.index(),
                    base_size: ad.map_or(0, |a| a.base_size()),
                    item_size: ad.map_or(8, |a| a.item_size()),
                    // descr.py:277 ArrayDescr.lendescr.offset — preserved
                    // across the resume summary so the materialize path
                    // sees the same length word offset the producer used.
                    len_offset: ad.and_then(|a| a.len_descr().map(|fd| fd.offset())),
                    item_type: ad.map_or(1, |a| {
                        if a.is_array_of_pointers() {
                            0
                        } else if a.is_array_of_floats() {
                            2
                        } else {
                            1
                        }
                    }),
                    is_signed: ad.map_or(true, |a| a.is_item_signed()),
                }
            })
            .collect();
        Some(majit_ir::RdVirtualInfo::VRawBufferInfo {
            func,
            size,
            offsets: offsets.to_vec(),
            descrs: descr_infos,
            fieldnums: Vec::new(),
        })
    }

    // resume.py:338-339 visit_vrawslice → VRawSliceInfo
    fn visit_vrawslice(&mut self, offset: i64) -> Self::VInfo {
        Some(majit_ir::RdVirtualInfo::VRawSliceInfo {
            offset,
            fieldnums: Vec::new(),
        })
    }

    // resume.py:341-345 visit_vstrplain → VStrPlainInfo / VUniPlainInfo
    fn visit_vstrplain(&mut self, is_unicode: bool) -> Self::VInfo {
        Some(if is_unicode {
            majit_ir::RdVirtualInfo::VUniPlainInfo {
                fieldnums: Vec::new(),
            }
        } else {
            majit_ir::RdVirtualInfo::VStrPlainInfo {
                fieldnums: Vec::new(),
            }
        })
    }

    // resume.py:347-351 visit_vstrconcat → VStrConcatInfo / VUniConcatInfo
    fn visit_vstrconcat(&mut self, is_unicode: bool) -> Self::VInfo {
        // resume.py:347-351 — visitor constructs the shell variant with
        // no funcptr; the decoder looks up OS_STR_CONCAT / OS_UNI_CONCAT
        // via `callinfocollection.funcptr_for_oopspec(...)` at
        // materialization time (resume.py:1467-1468 / 1494-1495).
        Some(if is_unicode {
            majit_ir::RdVirtualInfo::VUniConcatInfo {
                fieldnums: Vec::new(),
            }
        } else {
            majit_ir::RdVirtualInfo::VStrConcatInfo {
                fieldnums: Vec::new(),
            }
        })
    }

    // resume.py:353-357 visit_vstrslice → VStrSliceInfo / VUniSliceInfo
    fn visit_vstrslice(&mut self, is_unicode: bool) -> Self::VInfo {
        Some(if is_unicode {
            majit_ir::RdVirtualInfo::VUniSliceInfo {
                fieldnums: Vec::new(),
            }
        } else {
            majit_ir::RdVirtualInfo::VStrSliceInfo {
                fieldnums: Vec::new(),
            }
        })
    }

    fn register_virtual_fields(&mut self, _virtualbox: OpRef, _fieldboxes: &[OpRef]) {
        // resume.py:359-368 — field registration happens elsewhere in pyre
        // (via resume.rs worklist + get_virtual_fields), not through the
        // visitor. This adapter is only a builder for RdVirtualInfo.
    }

    fn already_seen_virtual(&mut self, _virtualbox: OpRef) -> bool {
        // resume.py:380-386 — same split as register_virtual_fields; the
        // pyre worklist tracks visited boxes directly.
        false
    }
}

impl OptContext {
    /// `optimizer.py:243` quasi-immutable dep registration with
    /// dict-as-set semantics (`heap.py:807-808`
    /// `self.optimizer.quasi_immutable_deps[qmutdescr.qmut] = None`).
    /// Vec-backed set with linear-scan dedup.
    pub fn add_quasi_immutable_dep(&mut self, dep: (u64, u32)) {
        if !self.quasi_immutable_deps.contains(&dep) {
            self.quasi_immutable_deps.push(dep);
        }
    }

    /// Record an `InvalidLoop` abandon signal from a leaf site that cannot
    /// return `Result` directly.  First write wins so the innermost reason
    /// survives.  The driver converts it to `Err` at the next barrier.
    pub fn signal_invalid_loop(&self, reason: &'static str) {
        let cur = self.pending_invalid_loop.take();
        self.pending_invalid_loop
            .set(Some(cur.unwrap_or(crate::optimize::InvalidLoop(reason))));
    }

    /// Consume any pending `InvalidLoop` signal recorded via
    /// [`signal_invalid_loop`](Self::signal_invalid_loop).
    pub fn take_invalid_loop(&self) -> Option<crate::optimize::InvalidLoop> {
        self.pending_invalid_loop.take()
    }

    /// Non-consuming check for a pending `InvalidLoop` signal.  Used by
    /// leaf sites (e.g. `constant_fold`) that must bail immediately after a
    /// failed `protect_speculative_operation` without consuming the signal,
    /// so the driver still aborts at its barrier.
    pub fn has_pending_invalid_loop(&self) -> bool {
        let v = self.pending_invalid_loop.take();
        let present = v.is_some();
        self.pending_invalid_loop.set(v);
        present
    }

    pub fn new(estimated_ops: usize) -> Self {
        OptContext {
            new_operations: Vec::with_capacity(estimated_ops),
            emitted_operations: majit_ir::vec_set::VecSet::new(),
            num_inputs: 0,
            inputarg_base: 0,
            next_pos: 0,
            extra_operations_after: VecDeque::new(),
            pending_guard_class_postprocess: None,
            pending_mark_last_guard: None,
            pending_finish_guard_postprocess: None,
            imported_short_pure_ops: Vec::new(),
            imported_virtual_args: None,
            imported_loop_invariant_results: Vec::new(),
            imported_short_preamble_builder: None,
            const_infos: majit_ir::VecMap::new(),
            imported_short_preamble_used: Vec::new(),

            potential_extra_ops: Vec::new(),
            active_short_preamble_producer: None,
            exported_short_boxes: Vec::new(),
            exported_short_inputargs: Vec::new(),
            exported_short_inputarg_refs: Vec::new(),

            imported_virtuals: Vec::new(),
            imported_label_args: None,
            can_replace_guards: true,
            patchguardop: None,
            preamble_end_args: None,
            skip_flush_mode: false,
            current_pass_idx: 0,
            optearlyforce_idx: 0,

            in_final_emission: false,
            callinfocollection: None,
            pending_for_guard: Vec::new(),
            pending_pure_from_args: Vec::new(),
            pending_pure_from_args2: Vec::new(),
            constant_fold_alloc: None,
            string_length_resolver: None,
            string_content_resolver: None,
            string_constant_alloc: None,
            quasi_immutable_deps: Vec::new(),
            snapshot_boxes: Vec::new(),
            snapshot_frame_sizes: Vec::new(),
            snapshot_vable_boxes: Vec::new(),
            snapshot_vref_boxes: Vec::new(),
            snapshot_frame_pcs: Vec::new(),

            inputargs: Vec::new(),
            inputarg_refs: Vec::new(),
            resop_refs: indexmap::IndexMap::new(),
            live_synthetics: Vec::new(),
            phase1_emit_ops: Vec::new(),
            phase1_emit_ops_index: std::collections::HashMap::new(),
            input_ops: Vec::new(),
            input_ops_index: std::collections::HashMap::new(),
            last_guard_idx: None,
            last_seen_snapshot_pos: None,
            cpu: crate::cpu::default_cpu(),
            remove_gctypeptr: true,
            last_op_removed: false,
            pending_invalid_loop: std::cell::Cell::new(None),
        }
    }

    /// Test-only inputarg-free constructor. Production paths must always go
    /// through [`Self::with_inputarg_types`] so every inputarg slot lands a
    /// typed `BoxRef` matching `opencoder.py:259 inputarg_from_tp(arg.type)`;
    /// passing `num_inputs > 0` here would silently drop the type tag and
    /// produce `Type::Void` reads. Sealed under `#[cfg(test)]` to make the
    /// rule structural rather than discipline-only.
    #[cfg(test)]
    pub fn with_num_inputs(estimated_ops: usize, num_inputs: usize) -> Self {
        debug_assert_eq!(
            num_inputs, 0,
            "with_num_inputs(_, {num_inputs}) — non-zero M requires typed seeding; \
             use `with_inputarg_types(estimated_ops, &[Type; M])` instead \
             (opencoder.py:259 inputarg_from_tp parity)",
        );
        Self::with_inputarg_types(estimated_ops, &[])
    }

    /// Construct an `OptContext` and seed `inputarg_refs` with one canonical
    /// `InputArg::from_type(tp, i)` per entry of `inputarg_types`.
    ///
    /// Mirrors `TraceIterator::new` (`opencoder.rs:373-426`, parity with
    /// `opencoder.py:259-262` `inputarg_from_tp(arg.type)`). Test fixtures
    /// that construct via this helper exercise the optimizer's BoxRef-direct
    /// routing — the production path.
    ///
    /// `inputarg_types` carries the type tags needed to round-trip
    /// `OpRef::input_arg_typed(i, tp)` on read; `with_num_inputs` is the
    /// untyped delegate that defaults every slot to `Type::Void`.
    pub fn with_inputarg_types(estimated_ops: usize, inputarg_types: &[majit_ir::Type]) -> Self {
        let num_inputs = inputarg_types.len();
        let mut ctx =
            Self::with_num_inputs_and_start_pos(estimated_ops, num_inputs, 0, num_inputs as u32);
        // Seed a fresh canonical `InputArgRc` per slot so the optimizer's
        // `make_equal_to` routes an InputArg-targeted chain step through
        // `Forwarded::InputArg(_)` (`optimizer.py:394 op.set_forwarded(newop)`).
        // The strong `InputArgRc`s are stashed in `ctx.inputarg_refs` so the
        // `Weak<InputArg>` each bound box later carries stays upgradable for
        // the OptContext's lifetime. Production traces own these via
        // `TreeLoop.inputargs`; this test-and-fallback helper has no upstream
        // `TreeLoop`.
        ctx.inputarg_refs = inputarg_types
            .iter()
            .enumerate()
            .map(|(i, &tp)| std::rc::Rc::new(majit_ir::InputArg::from_type(tp, i as u32)))
            .collect();
        // Seed `ctx.inputargs` so strict accessors like
        // `inputarg_type_at_strict` return `Some(tp)` matching slot i. Each
        // entry is `OpRef::input_arg_typed(i, tp)` so the variant tag IS the
        // type (resoperation.py:719/727/739).
        ctx.inputargs = majit_ir::OpRef::inputarg_refs(inputarg_types);
        ctx
    }

    /// Mirror every InputArg position into `inputarg_refs[pos]`: an
    /// already-bound slot keeps the `InputArgRc` it binds; an unbound slot
    /// gets a fresh one (bound here) carrying the position's type. Stashing
    /// the strong ref in `inputarg_refs` keeps each bound `Weak<InputArg>`
    /// upgradable for the OptContext's lifetime AND gives the canonical-host
    /// readers (`resolve_to_boxref` / `read_forwarded` / `clear_forwarded`)
    /// a live `InputArg.forwarded` host to resolve to.
    ///
    /// Phase 2 enters with a fresh per-iteration inputarg set whose earlier
    /// `Weak<InputArg>` owners (the previous OptContext's `inputarg_refs`)
    /// were dropped, leaving them dangling. Re-binding here restores
    /// `Forwarded::InputArg(_)` reachability for every InputArg BoxRef the
    /// optimizer will hand to `make_equal_to` (`optimizer.py:394
    /// op.set_forwarded(newop)`, unroll.py:497). Idempotent — re-running
    /// re-mirrors each slot to the same `InputArgRc`.
    pub(crate) fn ensure_inputarg_bindings(&mut self) {
        // Derive the materialized InputArg positions from `ctx` state.
        // The InputArg positions are exactly the
        // canonical/inherited set (`self.inputargs` = `optimizer.py:34
        // self.inputargs`, positions `[0, num_inputs)` carried across Phase 1 →
        // Phase 2 by `opt_p2.trace_inputargs = self.trace_inputargs`) UNION the
        // fresh per-iteration label set at `[inputarg_base, inputarg_base +
        // num_inputs)` (`TraceIterator` allocates these; their types match
        // `self.inputargs` because Phase 2 walks the body half with the same
        // per-arg types). Pre-populating `inputarg_refs` for both subsets makes
        // every InputArg OpRef resolve through `inputarg_refs` (read path:
        // `resolve_to_boxref` / `read_forwarded`; write path: `materialize_box_at`'s
        // InputArg branch). `materialize_box_at` type-repairs any position this derive
        // misses. Both loops no-op when `self.inputargs` is empty
        // (`seed_boxes_canonical` fixtures populate `inputarg_refs` directly).
        // Void slots are skipped: `InputArg{Int,Ref,Float}` has no Void
        // encoding (resoperation.py:719/727/739), so a Void sentinel in
        // `inputargs` is not a real input-arg host (mirrors the retired
        // box_pool scan's `!b.is_inputarg()` skip).
        for op in self.inputargs.clone() {
            match op.ty() {
                Some(tp) if tp != majit_ir::Type::Void => {
                    self.bind_canonical_inputarg(op.raw() as usize, tp);
                }
                _ => {}
            }
        }
        let base = self.inputarg_base as usize;
        for i in 0..self.num_inputs as usize {
            match self.inputargs.get(i).and_then(|op| op.ty()) {
                Some(tp) if tp != majit_ir::Type::Void => {
                    self.bind_canonical_inputarg(base + i, tp);
                }
                _ => {}
            }
        }
    }

    /// Ensure `inputarg_refs[pos]` holds a canonical `InputArgRc` of type
    /// `tp` (the `_forwarded` host that `resolve_to_boxref` / `read_forwarded`
    /// / `clear_forwarded` / `materialize_box_at` route the matching InputArg OpRef
    /// through). Idempotent: keeps an existing same-shape host (preserving any
    /// `_forwarded` chain / live `Weak<InputArg>` chain targets on it) and only
    /// (re)allocates when the slot is absent or its type/index mismatch (mirrors
    /// the `materialize_box_at` InputArg arm).
    fn bind_canonical_inputarg(&mut self, pos: usize, tp: majit_ir::Type) {
        if pos >= self.inputarg_refs.len() {
            self.inputarg_refs
                .resize_with(pos + 1, || std::rc::Rc::new(majit_ir::InputArg::new_int(0)));
            self.inputarg_refs[pos] =
                std::rc::Rc::new(majit_ir::InputArg::from_type(tp, pos as u32));
        } else if self.inputarg_refs[pos].tp != tp || self.inputarg_refs[pos].index != pos as u32 {
            self.inputarg_refs[pos] =
                std::rc::Rc::new(majit_ir::InputArg::from_type(tp, pos as u32));
        }
    }

    /// Find the canonical producer `OpRc`
    /// for `opref` by scanning the current phase's `new_operations`
    /// first, then cross-phase `phase1_emit_ops` (`history.py:220`
    /// box.type parity for Phase 1 emit OpRefs visible from Phase 2),
    /// then synthetic stand-ins / pre-bound input ops in
    /// `resop_refs`. Reverse scan on the first two lists mirrors
    /// `op_at`'s ordering so a later replacement at the same `pos`
    /// wins. Returns `None` for inputargs, constants, OpRefs without
    /// a producer in any of the three stores, and sentinel
    /// `OpRef::none()`.
    pub(crate) fn find_producer_op(&self, opref: OpRef) -> Option<majit_ir::OpRc> {
        if opref.is_none() || opref.is_constant() {
            return None;
        }
        if let Some(op) = self.new_operations.iter().rfind(|op| op.pos.get() == opref) {
            return Some(op.clone());
        }
        if let Some(op) = self.phase1_emit_ops_index.get(&opref).cloned() {
            return Some(op);
        }
        if let Some(op) = self.resop_refs.get(&opref).cloned() {
            return Some(op);
        }
        // Lowest-priority store: the recorder's input ops (seeded at setup
        // from the recorder's `Rc<Op>` slice). Full-OpRef match (collision-safe)
        // so a type-tagged value never aliases a different one at the same raw.
        // Consulted last so any live emission / synthetic above wins.
        //
        // `input_ops_index` mirrors the `input_ops.iter().rfind(...)` scan
        // (last-occurrence-wins) in O(1); `input_ops` is set once at setup so
        // the index stays consistent for the OptContext's lifetime.
        self.input_ops_index.get(&opref).cloned()
    }

    /// Rebuild `input_ops_index` from the current `input_ops`. Must be called
    /// after the one-shot `input_ops` seeding assignment at optimizer setup.
    /// Forward iteration with `insert` makes the last occurrence at each
    /// position win, matching the `rfind` the index replaces.
    pub(crate) fn rebuild_input_ops_index(&mut self) {
        self.input_ops_index.clear();
        self.input_ops_index.reserve(self.input_ops.len());
        for op in &self.input_ops {
            self.input_ops_index.insert(op.pos.get(), op.clone());
        }
    }

    /// Rebuild `phase1_emit_ops_index` from `phase1_emit_ops`. Called once
    /// at the Phase 1→2 handoff right after `phase1_emit_ops` is assigned;
    /// the field is never mutated afterwards. Forward iteration with
    /// `insert` makes the last occurrence at each position win, matching the
    /// `rfind` the index replaces.
    pub(crate) fn rebuild_phase1_emit_ops_index(&mut self) {
        self.phase1_emit_ops_index.clear();
        self.phase1_emit_ops_index
            .reserve(self.phase1_emit_ops.len());
        for op in &self.phase1_emit_ops {
            self.phase1_emit_ops_index.insert(op.pos.get(), op.clone());
        }
    }

    /// Mint a `SameAsI/F/R` (or `Jump` for `Void`) synthetic
    /// stand-in `OpRc` for `opref` with the correct result type and
    /// stash it in `resop_refs[opref]` so `emit()`'s
    /// `bound_is_synthetic` rebind path later upgrades the binding to
    /// the real producer via `bind_op`'s carry-over. The synthetic
    /// stays referenced from `resop_refs` for the OptContext's
    /// lifetime so lingering `Forwarded::Op(_)` `Weak<Op>` upgrades
    /// stay valid until rebind.
    pub(crate) fn mint_synthetic_resop(
        &mut self,
        opref: OpRef,
        ty: majit_ir::Type,
    ) -> majit_ir::OpRc {
        use majit_ir::resoperation::{Op, OpCode};
        let opcode = match ty {
            majit_ir::Type::Int => OpCode::SameAsI,
            majit_ir::Type::Float => OpCode::SameAsF,
            majit_ir::Type::Ref => OpCode::SameAsR,
            majit_ir::Type::Void => OpCode::Jump,
        };
        let synthetic = std::rc::Rc::new(Op::new(opcode, &[]));
        synthetic.pos.set(opref);
        self.resop_refs.insert(opref, synthetic.clone());
        self.live_synthetics.push(synthetic.clone());
        synthetic
    }

    /// `replace_op_with` (optimizer.py:replace_op_with) parity for the
    /// dispatcher's `Restart` arm: the rewritten op supersedes the original
    /// as the producer at its (preserved) position. RPython performs
    /// `original.set_forwarded(newop)`, after which every read of that
    /// position resolves to `newop` and `setintbound`/`setptrinfo` writes
    /// decorate `newop` directly. In pyre's producer-registry model the
    /// equivalent is to make `restart_op` the SOLE registered producer at
    /// its position: `find_producer_op(pos)` then returns `restart_op`, so
    /// `materialize_box_at(pos)` / `from_bound_op(restart_op)` agree on one canonical
    /// `_forwarded` host, and the bound the re-dispatch accumulates survives
    /// `emit`'s `live_synthetics` catch-up onto the emitted op.
    ///
    /// Without this the re-dispatch runs against a fresh, unregistered
    /// `Rc<Op>` (`Rc::new(restart_op)`): writes such as `setintbound` land on
    /// a host absent from `live_synthetics`, the superseded original keeps its
    /// stale stand-in slot, and the carry-over at emit migrates the wrong
    /// (empty) `_forwarded` — dropping the rewrite's bound.
    pub(crate) fn supersede_restart_producer(&mut self, restart_op: &majit_ir::OpRc) {
        self.install_canonical_producer(restart_op);
    }

    /// Make `op` the sole registered producer at its result position,
    /// inheriting any superseded stand-in's accumulated `_forwarded`.
    ///
    /// optimizer.py:403-411 `replace_op_with`: after `newop =
    /// op.copy_and_change(..)` it carries the prior host's info forward —
    /// `opinfo = get_box_replacement(op).get_forwarded(); if opinfo is not
    /// None: newop.set_forwarded(opinfo)` — and only then `op.set_forwarded(
    /// newop)` makes `newop` the single box at that position. In pyre's
    /// producer registry the equivalent is to drop the stand-in
    /// `live_synthetics` entry, migrate its `_forwarded` onto `op` (so the
    /// `IntBound`/`PtrInfo`/const facts accumulated before the rewrite survive),
    /// then register `op` as the lone producer. The blind `_forwarded` clone
    /// mirrors `emit`'s live_synthetics catch-up: a producer-less stand-in only
    /// ever holds `None`/`Info`, never a redirect.
    fn install_canonical_producer(&mut self, op: &majit_ir::OpRc) {
        let pos = op.pos.get();
        if pos.is_none() || pos.is_constant() {
            return;
        }
        // InputArg positions have no producing op (handled by
        // `ensure_inputarg_bindings`); a rewrite never targets one.
        if matches!(
            pos,
            OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_)
        ) {
            return;
        }
        // Drop the superseded stand-in at this position and carry its
        // accumulated `_forwarded` onto `op` (replace_op_with's `opinfo`
        // hand-off). At most one live stand-in exists per position
        // (mod.rs::emit invariant), so a single removal is sufficient. The
        // `ptr_eq` guard skips the self-clone — and its `RefCell` double-borrow
        // — when `op` is already the registered stand-in.
        if let Some(i) = self.live_synthetics.iter().position(|s| s.pos.get() == pos) {
            let superseded = self.live_synthetics.swap_remove(i);
            if !std::rc::Rc::ptr_eq(&superseded, op) {
                let carried = superseded.forwarded.borrow().clone();
                *op.forwarded.borrow_mut() = carried;
                // replace_op_with parity (optimizer.py:403-411): forward the
                // superseded stand-in to `op`. A consumer dispatched before
                // this supersession bound its operand to `superseded` (the
                // producer registered at its dispatch-entry rebind); without
                // this link its box-native walk freezes on the orphaned
                // stand-in while `find_producer_op` resolves `op`, so a later
                // operand fold lands on a different host than the OpRef path
                // observes. `superseded` stays alive in operand `Operand::Op`
                // strong refs, so the `Weak` upgrades and the chain reaches
                // `op`. Mirrors `emit`'s live-synthetic catch-up so a
                // supersession chain (stand-in → extra-producer → emitted op)
                // stays transitively resolvable to the final producer.
                use majit_ir::box_ref::ForwardingHost;
                superseded.set_forwarded_op(op);
            }
        }
        self.resop_refs.insert(pos, op.clone());
        self.live_synthetics.push(op.clone());
    }

    /// Read `_forwarded` for `opref` directly off the canonical
    /// host (`op.forwarded` / `inputarg.forwarded`). Mirrors
    /// `BoxRef::get_forwarded` semantics
    /// but bypasses the wrapper allocation. Returns `Forwarded::None`
    /// for constants (`resoperation.py:50` `Const._forwarded` is
    /// permanently `None`), `None` for sentinel `OpRef::none()` and
    /// for ResOp positions whose producer is not in any canonical
    /// store (`new_operations` / `phase1_emit_ops` / `resop_refs`).
    pub(crate) fn read_forwarded(&self, opref: OpRef) -> Option<majit_ir::box_ref::Forwarded> {
        if opref.is_none() {
            return None;
        }
        if opref.is_constant() {
            return Some(majit_ir::box_ref::Forwarded::None);
        }
        match opref {
            OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_) => {
                let idx = opref.raw() as usize;
                self.inputarg_refs
                    .get(idx)
                    .map(|ia| ia.forwarded.borrow().clone())
            }
            _ => self
                .find_producer_op(opref)
                .map(|op| op.forwarded.borrow().clone()),
        }
    }

    /// Resolve `opref` to a `BoxRef` bound to its canonical
    /// `_forwarded` host (`Op` / `InputArg`). Materialises a fresh
    /// `BoxRef::from_bound_op` / `from_bound_inputarg` per call; the
    /// bound handle ensures every `set_forwarded_*` / `get_forwarded`
    /// routes through the same `Op.forwarded` / `InputArg.forwarded`
    /// slot, so two calls for the same `opref` observe each other's
    /// writes via the canonical host even though the `BoxRef` wrapper
    /// identities differ.
    /// Const variants return `BoxRef::new_const`. Returns
    /// `None` for sentinel `OpRef::none()` and for ResOp positions
    /// without a producer in any canonical store.
    ///
    /// Production paths populate `inputarg_refs` via `bind_input_resops`
    /// plus emit-time `bind_op`, so every
    /// chain-walker-reachable position resolves to its bound `BoxRef`.
    pub(crate) fn resolve_to_boxref(&self, opref: OpRef) -> Option<majit_ir::box_ref::BoxRef> {
        if opref.is_none() {
            return None;
        }
        if opref.is_constant() {
            // history.py:227/268/314 — Const variants carry the value on the
            // OpRef directly; mint a fresh inline-Const BoxRef so the chain
            // round-trip (`box_to_opref`) reconstructs it from the value.
            return match opref {
                OpRef::ConstInt(v) => Some(majit_ir::box_ref::BoxRef::new_const(Value::Int(v))),
                OpRef::ConstFloat(v) => Some(majit_ir::box_ref::BoxRef::new_const(Value::Float(v))),
                OpRef::ConstPtr(v) => Some(majit_ir::box_ref::BoxRef::new_const(Value::Ref(v))),
                _ => None,
            };
        }
        if let Some(op) = self.find_producer_op(opref) {
            return Some(majit_ir::box_ref::BoxRef::from_bound_op(&op));
        }
        let idx = opref.raw() as usize;
        // InputArg variants resolve through the canonical `inputarg_refs`
        // store — symmetric with the `clear_forwarded` write path
        // (`inputarg_refs[idx].forwarded`). `ensure_inputarg_bindings`
        // populates `inputarg_refs[idx]` with the canonical `InputArgRc`, so
        // this returns the `InputArg.forwarded` host every other reader and
        // writer observes.
        match opref {
            OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_) => {
                if let Some(ia) = self.inputarg_refs.get(idx) {
                    return Some(majit_ir::box_ref::BoxRef::from_bound_inputarg(ia));
                }
            }
            _ => {}
        }
        // A ResOp position with no producer in any canonical store resolves
        // to `None`: the caller's `materialize_box_at` mints a `SameAs*` synthetic
        // into `resop_refs[opref]` and binds it, so the next `find_producer_op`
        // (and hence the next `resolve_to_boxref` / `make_constant` chain)
        // reaches that same `_forwarded` host. Routing a ResOp OpRef to
        // `inputarg_refs[idx]` here would re-introduce the raw-position
        // collapse (`int_op(p)` aliasing `input_arg_int(p)`) the
        // `find_producer_op` / `inputarg_refs` split exists to eliminate.
        None
    }

    /// [`Operand`]-yielding sibling of [`resolve_to_boxref`](Self::resolve_to_boxref):
    /// resolve a position to its canonical producer as an `Operand`
    /// (`Op` / `InputArg` host, or inline-`Const`) without minting a `BoxRef`.
    /// 1:1 mirror — `BoxRef::new_const`/`from_bound_op`/`from_bound_inputarg`
    /// become `Operand::const_from_value`/`from_bound_op`/`from_bound_inputarg`.
    pub(crate) fn resolve_to_operand(&self, opref: OpRef) -> Option<Operand> {
        if opref.is_none() {
            return None;
        }
        if opref.is_constant() {
            return match opref {
                OpRef::ConstInt(v) => Some(Operand::const_from_value(Value::Int(v))),
                OpRef::ConstFloat(v) => Some(Operand::const_from_value(Value::Float(v))),
                OpRef::ConstPtr(v) => Some(Operand::const_from_value(Value::Ref(v))),
                _ => None,
            };
        }
        if let Some(op) = self.find_producer_op(opref) {
            return Some(Operand::from_bound_op(&op));
        }
        let idx = opref.raw() as usize;
        match opref {
            OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_) => {
                if let Some(ia) = self.inputarg_refs.get(idx) {
                    return Some(Operand::from_bound_inputarg(ia));
                }
            }
            _ => {}
        }
        None
    }

    /// Write `Forwarded::None` to the canonical host for
    /// `opref` (`resoperation.py:240` `set_forwarded(None)` /
    /// `:50` clear semantics). No-op for sentinel `OpRef::none()`,
    /// constants (whose `_forwarded` is permanently `None`), and
    /// positions without a canonical Op/InputArg.
    pub(crate) fn clear_forwarded(&self, opref: OpRef) {
        if opref.is_none() || opref.is_constant() {
            return;
        }
        match opref {
            OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_) => {
                let idx = opref.raw() as usize;
                if let Some(ia) = self.inputarg_refs.get(idx) {
                    *ia.forwarded.borrow_mut() = majit_ir::box_ref::Forwarded::None;
                }
            }
            _ => {
                if let Some(op) = self.find_producer_op(opref) {
                    *op.forwarded.borrow_mut() = majit_ir::box_ref::Forwarded::None;
                }
            }
        }
    }

    /// Test-only: seed the canonical producer stores (`resop_refs` /
    /// `inputarg_refs`) from a list of already-bound `BoxRef`s, mirroring
    /// what the production recorder→optimizer handoff populates. Each box
    /// is distributed by its bound identity: InputArg boxes land in
    /// `inputarg_refs[index]`, ResOp boxes in `resop_refs[pos]`. This
    /// replaces the retired `ctx.box_pool = vec![..]` fixture pattern so
    /// `resolve_to_boxref` / `materialize_box_at` / `find_producer_op` resolve each
    /// OpRef through the same canonical hosts production uses, returning a
    /// fresh `BoxRef` bound to the seeded `Op` / `InputArg`.
    #[cfg(test)]
    pub(crate) fn seed_boxes_canonical(&mut self, boxes: &[majit_ir::box_ref::BoxRef]) {
        for b in boxes {
            if let Some(ia) = b.bound_inputarg() {
                let idx = ia.index as usize;
                if idx >= self.inputarg_refs.len() {
                    self.inputarg_refs
                        .resize_with(idx + 1, || std::rc::Rc::new(majit_ir::InputArg::new_int(0)));
                }
                self.inputarg_refs[idx] = ia;
            } else if let Some(op) = b.bound_op() {
                self.resop_refs.insert(op.pos.get(), op);
            }
        }
    }

    /// Record every input op's resop producer `OpRc` into
    /// `resop_refs` so any `Forwarded::Op(_)` chain step targeting the
    /// slot has an upgradable `Weak<Op>` from the start of the
    /// optimization run. Absent this pre-pass, `getintbound_box` →
    /// `get_box_replacement_box` (a `&self` reader) can land on an
    /// unbound terminal and a subsequent `set_forwarded_info` write
    /// trips `BoxRef::write_forwarded`'s bound-precondition assert.
    ///
    /// The producer `OpRc` is stashed in `resop_refs[pos]` so `emit()`'s
    /// `bound_is_synthetic` check (`mod.rs::emit` rebind path) later
    /// upgrades the binding to the emitted post-pass producer `OpRc`
    /// via `bind_op`'s carry-over (forwarded state preserved).
    ///
    /// InputArg slots are skipped (handled by `ensure_inputarg_bindings`);
    /// only resop positions land here. Each phase's input `ops` carry that
    /// phase's own positions (Phase 2 body ops sit above the Phase 1 emit
    /// namespace, so they never collide with an inherited bound slot), and
    /// the `resop_refs[opref]` dedup covers intra-`ops` repeats.
    pub(crate) fn bind_input_resops(&mut self, ops: &[majit_ir::OpRc]) {
        // The loop over the caller-threaded `ops` self-guards (no-op on an
        // empty slice) and records each resop producer into `resop_refs` /
        // `live_synthetics` — the collision-safe stores `find_producer_op`
        // consults.
        for op in ops {
            let pos = op.pos.get();
            if pos.is_none() || pos.is_constant() {
                continue;
            }
            // InputArg slots are handled by `ensure_inputarg_bindings`; only
            // resop positions land in `resop_refs`.
            if matches!(
                pos,
                OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_)
            ) {
                continue;
            }
            // Dedup: a producer for this exact pos is already recorded.
            if self.resop_refs.contains_key(&pos) {
                continue;
            }
            // Record the caller-threaded `OpRc` so the iterated input op,
            // `resop_refs`, and `live_synthetics` share one `Op` identity
            // (no second private copy).
            let op_rc = op.clone();
            self.resop_refs.insert(pos, op_rc.clone());
            self.live_synthetics.push(op_rc);
        }
    }

    /// Construct an `OptContext` whose inputarg / fresh-OpRef numbering is
    /// shifted to start above a parent trace's high water mark.
    ///
    /// `inputarg_base` corresponds to RPython's `start = trace._start` for
    /// `TraceIterator`: it is the smallest OpRef this iteration may use, and
    /// inputargs occupy `[inputarg_base, inputarg_base + num_inputs)`.
    /// `start_next_pos` is the value of `_index` after the inputargs were
    /// pre-allocated, i.e. the first fresh OpRef the optimizer will assign
    /// to a non-void op result. Phase 1 / standalone passes use
    /// `inputarg_base = 0`, `start_next_pos = num_inputs`; Phase 2 / bridges
    /// pass `inputarg_base = parent_high_water`,
    /// `start_next_pos = parent_high_water + num_inputs`.
    pub fn with_num_inputs_and_start_pos(
        estimated_ops: usize,
        num_inputs: usize,
        inputarg_base: u32,
        start_next_pos: u32,
    ) -> Self {
        OptContext {
            new_operations: Vec::with_capacity(estimated_ops),
            emitted_operations: majit_ir::vec_set::VecSet::new(),
            num_inputs: num_inputs as u32,
            inputarg_base,
            next_pos: start_next_pos,
            extra_operations_after: VecDeque::new(),
            pending_guard_class_postprocess: None,
            pending_mark_last_guard: None,
            pending_finish_guard_postprocess: None,
            imported_short_pure_ops: Vec::new(),
            imported_virtual_args: None,
            imported_loop_invariant_results: Vec::new(),
            imported_short_preamble_builder: None,
            const_infos: majit_ir::VecMap::new(),
            imported_short_preamble_used: Vec::new(),

            potential_extra_ops: Vec::new(),
            active_short_preamble_producer: None,
            exported_short_boxes: Vec::new(),
            exported_short_inputargs: Vec::new(),
            exported_short_inputarg_refs: Vec::new(),

            imported_virtuals: Vec::new(),
            imported_label_args: None,
            can_replace_guards: true,
            patchguardop: None,
            preamble_end_args: None,
            skip_flush_mode: false,
            current_pass_idx: 0,
            optearlyforce_idx: 0,

            in_final_emission: false,
            callinfocollection: None,
            pending_for_guard: Vec::new(),
            pending_pure_from_args: Vec::new(),
            pending_pure_from_args2: Vec::new(),
            constant_fold_alloc: None,
            string_length_resolver: None,
            string_content_resolver: None,
            string_constant_alloc: None,
            quasi_immutable_deps: Vec::new(),
            snapshot_boxes: Vec::new(),
            snapshot_frame_sizes: Vec::new(),
            snapshot_vable_boxes: Vec::new(),
            snapshot_vref_boxes: Vec::new(),
            snapshot_frame_pcs: Vec::new(),

            inputargs: Vec::new(),
            inputarg_refs: Vec::new(),
            resop_refs: indexmap::IndexMap::new(),
            live_synthetics: Vec::new(),
            phase1_emit_ops: Vec::new(),
            phase1_emit_ops_index: std::collections::HashMap::new(),
            input_ops: Vec::new(),
            input_ops_index: std::collections::HashMap::new(),
            last_guard_idx: None,
            last_seen_snapshot_pos: None,
            cpu: crate::cpu::default_cpu(),
            remove_gctypeptr: true,
            last_op_removed: false,
            pending_invalid_loop: std::cell::Cell::new(None),
        }
    }

    pub fn num_inputs(&self) -> usize {
        self.num_inputs as usize
    }

    /// Allocate a fresh OpRef position with the producer's result type
    /// stamped on the variant tag. Callers always know `box.type` —
    /// the resulting OpRef is recognized at priority 0 by `opref_type` /
    /// `OptBoxEnv::get_type`, and there is no type side-table to grow
    /// for these positions.
    pub fn alloc_op_position_typed(&mut self, tp: majit_ir::Type) -> OpRef {
        self.reserve_pos_typed(tp)
    }

    /// Allocate a fresh OpRef position and eagerly mint its canonical
    /// `_forwarded` host — a `SameAs*`/`Jump` synthetic in `resop_refs` —
    /// returning both the position and a `BoxRef` bound to that host.
    ///
    /// This is the explicit creation primitive for producer-less
    /// synthetics: importers that allocate a position purely to carry a
    /// forwarded write (PtrInfo / IntBound / Const for an imported virtual
    /// state leaf) get a bound write target in one step, instead of
    /// `alloc_op_position_typed` followed by a lazy `materialize_box_at(opref)`
    /// re-materialization. The minted synthetic is identical to the one
    /// `materialize_box_at`'s producer-less arm mints (`mint_synthetic_resop`), so
    /// a later `emit()` for the same position supersedes it through the
    /// same `live_synthetics` catch-up. Reserve a bare position via
    /// `alloc_op_position_typed` instead when no forwarded write follows
    /// (e.g. an `Unknown` leaf), to avoid an eager synthetic for a
    /// position that is never written.
    /// Explicit "create" half of the find-or-create `materialize_box_at`:
    /// mint a `SameAs*` synthetic at `opref` and return a BoxRef bound to it,
    /// so a subsequent `set_forwarded_*` lands on the canonical `Op._forwarded`
    /// host. `opref` must be a non-const, non-sentinel resop position whose
    /// producer is not yet emitted (a virgin alias). Callers reach this only on
    /// the `None` arm of `get_box_replacement_box`; an already-minted or
    /// producer-backed opref resolves there. Mirrors `materialize_box_at`'s resop
    /// lazy-alloc arm (`mint_synthetic_resop` + bind).
    pub(crate) fn mint_box_at(&mut self, opref: OpRef) -> majit_ir::box_ref::BoxRef {
        let tp = opref.ty().unwrap_or(majit_ir::Type::Void);
        let synthetic = self.mint_synthetic_resop(opref, tp);
        majit_ir::box_ref::BoxRef::from_bound_op(&synthetic)
    }

    pub(crate) fn reserve_virtual_box(&mut self, tp: majit_ir::Type) -> (OpRef, Operand) {
        let opref = self.reserve_pos_typed(tp);
        let b = Operand::from_boxref(&self.mint_box_at(opref));
        (opref, b)
    }

    /// Dispatch on a `Value`'s type tag and produce a typed `*Op` OpRef
    /// at the given position (resoperation.py:564-638
    /// IntOp/FloatOp/RefOp/VoidOp mixins).
    pub(crate) fn op_ref_for_value(pos: u32, value: &Value) -> OpRef {
        OpRef::op_typed(pos, value.get_type())
    }

    /// info.py:148,226 emit during force_box: routes through emit_extra
    /// normally, or direct emit when in_final_emission is true.
    pub fn emit_for_force(&mut self, op: Op) -> OpRef {
        if self.in_final_emission {
            self.emit(op)
        } else {
            self.emit_extra(self.current_pass_idx, op)
        }
    }

    /// Emit a boxed integer constant through the optimizer pipeline and return
    /// the resulting OpRef.
    pub fn emit_constant_int(&mut self, value: i64) -> OpRef {
        let pos_ref = self.reserve_pos_typed(Type::Int);
        // The SAME_AS source is the constant itself (`make_constant_box` below
        // forwards the result to the same `Const`, so the op is a tautology
        // `result = ConstInt(value)`). A constant operand binds directly; a
        // position-only `from_opref(pos_ref)` self-reference would have no live
        // producer and `from_boxref` rejects it (#9).
        let mut op = Op::new(
            OpCode::SameAsI,
            &[Operand::const_from_value(Value::Int(value))],
        );
        op.pos.set(pos_ref);
        let opref = self.emit_extra(self.current_pass_idx, op);
        let b = self.materialize_operand_at(opref);
        self.make_constant_box(&b, Value::Int(value));
        opref
    }

    /// Emit a boxed reference constant through the optimizer pipeline and
    /// return the resulting OpRef.
    pub fn emit_constant_ref(&mut self, value: GcRef) -> OpRef {
        let pos_ref = self.reserve_pos_typed(Type::Ref);
        // SAME_AS source is the constant ref itself; see `emit_constant_int`.
        let mut op = Op::new(
            OpCode::SameAsR,
            &[Operand::const_from_value(Value::Ref(value))],
        );
        op.pos.set(pos_ref);
        let opref = self.emit_extra(self.current_pass_idx, op);
        let b = self.materialize_operand_at(opref);
        self.make_constant_box(&b, Value::Ref(value));
        opref
    }

    /// Emit a boxed float constant through the optimizer pipeline and return
    /// the resulting OpRef.
    pub fn emit_constant_float(&mut self, value: f64) -> OpRef {
        let pos_ref = self.reserve_pos_typed(Type::Float);
        // SAME_AS source is the constant float itself; see `emit_constant_int`.
        let mut op = Op::new(
            OpCode::SameAsF,
            &[Operand::const_from_value(Value::Float(value))],
        );
        op.pos.set(pos_ref);
        let opref = self.emit_extra(self.current_pass_idx, op);
        let b = self.materialize_operand_at(opref);
        self.make_constant_box(&b, Value::Float(value));
        opref
    }

    /// optimizer.py:509-515 new_const_item(arraydescr) — default value for
    /// the given item type. Uses emit_extra (downstream-only) so this is
    /// safe to call during force_box / force_virtual.
    pub fn new_const_item(&mut self, item_type: Type) -> OpRef {
        match item_type {
            Type::Int | Type::Void => self.emit_constant_int(0),
            Type::Ref => self.emit_constant_ref(GcRef::NULL),
            Type::Float => self.emit_constant_float(0.0),
        }
    }

    /// vstring.py:110-119 / 171-175 / 251-253 / 281-295
    /// Per-subclass getstrlen() dispatch — returns a cached lgtop OpRef if
    /// available, or computes/emits the length and caches in StrPtrInfo.lgtop.
    /// Always returns a box (OpRef), never an i64 summary.
    ///
    /// Delegates to `getstrlen_for(opref, opref, mode)`.
    pub fn getstrlen_opref(&mut self, opref: OpRef, mode: u8) -> OpRef {
        self.getstrlen_for(opref, opref, mode)
    }

    /// vstring.py:110-119 StrPtrInfo.getstrlen(op, optstring, mode)
    ///
    /// Matches RPython's method dispatch where `self` (info) and `op` may
    /// differ: info lookup comes from `info_opref`, but the fallback STRLEN
    /// emission uses `op_opref`.  Cached lgtop is stored on `info_opref`'s
    /// PtrInfo.
    ///
    /// When both are the same, use `getstrlen_opref(opref, mode)` instead.
    pub fn getstrlen_for(&mut self, info_opref: OpRef, op_opref: OpRef, mode: u8) -> OpRef {
        let resolved_box = self.get_box_replacement_operand_opt(info_opref);
        // vstring.py:112/283: if self.lgtop is not None: return self.lgtop
        if let Some(info) = resolved_box.as_ref().and_then(|b| self.getptrinfo(b)) {
            if let Some(lgtop) = info.get_cached_lgtop() {
                return lgtop;
            }
        }
        // vstring.py:174/253: constant or structurally-known length
        let known_len = resolved_box
            .as_ref()
            .and_then(|b| self.getptrinfo(b))
            .and_then(|info| info.get_known_str_length(self, mode));
        if let Some(len) = known_len {
            let len_opref = self.make_constant_int(len);
            // BoxRef shim — write path through `materialize_box_at` per the
            // "Box always exists" invariant for set_forwarded mirrors.
            if let Some(b) = self.get_box_replacement_operand_opt(info_opref) {
                self.set_str_lgtop(&b, len_opref);
            }
            return len_opref;
        }
        // vstring.py:281-295: VStringConcatInfo.getstrlen — recursive
        // dispatch: getstrlen on each child, then _int_add.
        // Borrow-checker adaptation: extract vleft/vright before &mut self calls.
        let concat_children = resolved_box
            .as_ref()
            .and_then(|b| self.getptrinfo(b))
            .and_then(|info| {
                use crate::optimizeopt::info::VStringVariant;
                if let PtrInfo::Str(sinfo) = info {
                    if let VStringVariant::Concat(c) = sinfo.variant {
                        return Some((c.vleft.to_opref(), c.vright.to_opref()));
                    }
                }
                None
            });
        if let Some((vleft, vright)) = concat_children {
            // vstring.py:286-293
            let left_len = self.getstrlen_for(vleft, vleft, mode);
            let right_len = self.getstrlen_for(vright, vright, mode);
            let left_len = self.materialize_operand_at(left_len);
            let right_len = self.materialize_operand_at(right_len);
            let result =
                crate::optimizeopt::vstring::_int_add(&left_len, &right_len, self).to_opref();
            // vstring.py:293: self.lgtop = _int_add(optstring, len1box, len2box)
            if let Some(b) = self.get_box_replacement_operand_opt(info_opref) {
                self.set_str_lgtop(&b, result);
            }
            return result;
        }
        // vstring.py:115-118: base StrPtrInfo — emit STRLEN/UNICODELEN
        // RPython: lengthop = ResOperation(mode.STRLEN, [op])
        // `op` comes from op_opref (the first arg to getstrlen in RPython).
        let op_resolved = self.get_replacement_opref(op_opref);
        let strlen_opcode = if mode != 0 {
            majit_ir::OpCode::Unicodelen
        } else {
            majit_ir::OpCode::Strlen
        };
        let arg1 = self.materialize_operand_at(op_resolved);
        let strlen_op = majit_ir::Op::new(strlen_opcode, &[arg1]);
        let result = self.emit_extra(self.current_pass_idx, strlen_op);
        // vstring.py:116: lengthop.set_forwarded(self.getlenbound(mode))
        // `set_forwarded` writes the bound unconditionally; route through
        // `materialize_box_at` so the new STRLEN/UNICODELEN box materializes for
        // the IntBound install ("Box always exists" per resoperation.py:233-248).
        // BoxRef shim for `get_str_lenbound(&BoxRef)`; lazy-install of
        // lenbound on the StrPtrInfo is a PtrInfo-internal mutation that
        // RPython performs on the StrPtrInfo instance directly. Route
        // through `materialize_box_at` so the BoxRef exists for the chain walk.
        let lenbound = self
            .get_box_replacement_operand_opt(info_opref)
            .as_ref()
            .and_then(|b| self.get_str_lenbound(b));
        if let Some(bound) = lenbound {
            if let Some(result_box) = self.get_box_replacement_operand_opt(result) {
                self.setintbound(&result_box, &bound);
            }
        }
        // vstring.py:117: self.lgtop = lengthop
        if let Some(b) = self.get_box_replacement_operand_opt(info_opref) {
            self.set_str_lgtop(&b, result);
        }
        result
    }

    /// `vstring.py:117/174/293 self.lgtop = lengthop` — cache the length
    /// box in `StrPtrInfo.lgtop`. Direct PtrInfo field write,
    /// unconditional per `info.py:432`.
    ///
    /// `op: &Operand` is the StrPtrInfo-bearing box; `lgtop: OpRef` is the
    /// length op's position, materialized to its bound producer before the
    /// cache write so the field carries an `Operand` (never a position-only
    /// box).
    pub(crate) fn set_str_lgtop(&mut self, op: &Operand, lgtop: OpRef) {
        // optimizer.py `get_box_replacement` chain walk before mutation.
        let resolved = op.get_box_replacement(false);
        if resolved.is_constant() {
            return;
        }
        let lgtop_op = self.materialize_operand_at(lgtop);
        self.with_ptr_info_mut(&resolved, |info| {
            if let PtrInfo::Str(si) = info {
                si.lgtop = Some(lgtop_op.clone());
            }
        });
    }

    /// `vstring.py:62-70 StrPtrInfo.getlenbound(mode)` — get lenbound
    /// from StrPtrInfo, lazily initializing it from self.length:
    /// ```python
    /// def getlenbound(self, mode):
    ///     if self.lenbound is None:
    ///         if self.length == -1:
    ///             self.lenbound = IntBound.nonnegative()
    ///         else:
    ///             self.lenbound = IntBound.from_constant(self.length)
    ///     return self.lenbound
    /// ```
    /// Const inputs short-circuit (RPython getlenbound is called on a
    /// StrPtrInfo instance, never on a Const).
    fn get_str_lenbound(&self, op: &Operand) -> Option<crate::optimizeopt::intutils::IntBound> {
        // optimizer.py-style chain walk; mirror PyPy `getptrinfo(op)` shape
        // by reading the chain terminal via BoxRef::get_box_replacement.
        let resolved = op.get_box_replacement(false);
        if resolved.is_constant() {
            return None;
        }
        self.with_ptr_info_mut(&resolved, |info| {
            if let PtrInfo::Str(si) = info {
                // vstring.py:65-70
                if si.lenbound.is_none() {
                    si.lenbound = Some(if si.length == -1 {
                        crate::optimizeopt::intutils::IntBound::nonnegative()
                    } else {
                        crate::optimizeopt::intutils::IntBound::from_constant(si.length as i64)
                    });
                }
                si.lenbound.clone()
            } else {
                None
            }
        })
        .flatten()
    }

    /// Typed `reserve_pos`. Tags the resulting OpRef with the producer's
    /// result type so `AbstractValue.type` parity (resoperation.py:29)
    /// is set at allocation time. Readers consult
    /// `opref.ty()` at priority 0 in `opref_type` / `OptBoxEnv::get_type`,
    /// so typed positions never grow `value_types`.
    pub(crate) fn reserve_pos_typed(&mut self, tp: majit_ir::Type) -> OpRef {
        let raw = self.allocate_next_pos_raw();
        // The position's canonical host is materialized lazily on first
        // *write* access (`materialize_box_at` mints a `SameAs*` synthetic
        // into `resop_refs[raw]` keyed by the full OpRef; `resolve_to_boxref`
        // is `&self` and returns `None` until then). No eager pre-mint here:
        // an eager synthetic for a position that is reserved but never
        // emitted (label / jump positions on an empty trace) would leak into
        // `phase1_emit_ops` via `live_synthetics`; the emitted op, when it
        // arrives, supersedes the lazily-minted synthetic the same way.
        // Callers whose position is *read* before any write (imported
        // virtual-state leaves and heads) must use `reserve_virtual_box`
        // instead — a bare position read mints a fresh position-only box per
        // resolution (bind-at-alloc).
        // PyPy/RPython has no Box for positions that no `ResOperation()` /
        // `InputArg()` call produced (`resoperation.py:233-248`).
        OpRef::op_typed(raw, tp)
    }

    /// opencoder.py:271 `_index` parity: floor at the iteration's inputarg
    /// base + num_inputs + emitted-op count, so a freshly allocated raw
    /// position never lands inside the inputarg slice or below the parent
    /// trace's high water mark when called from a Phase 2 / bridge
    /// OptContext.
    fn allocate_next_pos_raw(&mut self) -> u32 {
        self.next_pos = self
            .next_pos
            .max(self.inputarg_base + self.num_inputs + self.new_operations.len() as u32);
        // Skip positions already claimed by a constant forwarding
        // (`make_constant`/`seed_constant`'s `set_forwarded_const` write) —
        // those positions' canonical host is already a constant identity and
        // cannot be reused for a fresh op. Reads the canonical `_forwarded`
        // host for the position (`resop_refs[pos]` / `inputarg_refs[pos]`):
        // `make_constant` writes `Forwarded::Const` to that host
        // (resoperation.py:233).
        while self.position_is_const_forwarded(self.next_pos) {
            self.next_pos += 1;
        }
        debug_assert!(
            !OpRef::raw_is_constant(self.next_pos),
            "reserve_pos overflowed into constant namespace: {}",
            self.next_pos
        );
        let raw = self.next_pos;
        self.next_pos += 1;
        raw
    }

    /// Whether the canonical `_forwarded` host for raw position `raw`
    /// (any `resop_refs` entry whose `OpRef` shares this raw, or
    /// `inputarg_refs[raw]` for an InputArg slot) carries `Forwarded::Const`.
    /// The position-keyed replacement for the retired
    /// `box_pool.get_at_position(raw)` const probe in `allocate_next_pos_raw`.
    fn position_is_const_forwarded(&self, raw: u32) -> bool {
        use majit_ir::box_ref::Forwarded;
        let idx = raw as usize;
        // `resop_refs` is keyed by the full type-tagged `OpRef`; a raw `u32`
        // can host more than one entry (typed vs untyped). Any host at this
        // raw carrying `Forwarded::Const` claims the position.
        let resop_const = self.resop_refs.values().any(|op| {
            op.pos.get().raw() == raw && matches!(*op.forwarded.borrow(), Forwarded::Const(_))
        });
        let inputarg_const = self
            .inputarg_refs
            .get(idx)
            .is_some_and(|ia| matches!(*ia.forwarded.borrow(), Forwarded::Const(_)));
        resop_const || inputarg_const
    }

    /// RPython `box.type` invariant (history.py:220
    /// `InputArg{Int,Ref,Float}.type`, resoperation.py:1693
    /// `opclasses[opnum].type`): every emitted op's intrinsic
    /// `Op.type_` must agree with both the producer's
    /// `OpCode::result_type()` and the `OpRef.pos` variant tag.
    /// Checked at the emit/emit_extra producer surfaces, where both the
    /// `Op` result type and the typed `OpRef` are available.
    fn debug_assert_box_type_invariant(op: &Op) {
        let pos = op.pos.get();
        debug_assert_eq!(
            op.type_,
            op.opcode.result_type(),
            "Op.type_ ({:?}) disagrees with opcode.result_type() ({:?}) at \
             {:?} (opcode={:?}) — dual-source contract violation",
            op.type_,
            op.opcode.result_type(),
            pos,
            op.opcode,
        );
        if let Some(variant_tp) = pos.ty() {
            debug_assert_eq!(
                variant_tp, op.type_,
                "OpRef variant tag ({:?}) disagrees with Op.type_ ({:?}) at \
                 {:?} (opcode={:?}) — typed-factory mismatch \
                 (history.py:220 / resoperation.py:1693 Box.type parity)",
                variant_tp, op.type_, pos, op.opcode,
            );
        }
    }

    /// Emit an operation to the output.
    ///
    /// If the op has no pos assigned (NONE), sets it to `num_inputs + idx`
    /// so the backend's variable numbering stays consistent.
    pub fn emit(&mut self, op: Op) -> OpRef {
        self.emit_impl(op, false)
    }

    /// `emit` variant that REUSES the recorder input op (the `live_synthetics`
    /// entry at this position) as the emitted producer instead of cloning into
    /// a fresh `Rc<Op>`. The input op is the object later ops' operands already
    /// wrap; making it the producer collapses the two-Op-per-position
    /// duplication that the `live_synthetics` catch-up otherwise bridges by
    /// copying `_forwarded` and linking input -> clone. One box per value, the
    /// op IS the box (resoperation.py:233). Falls back to the clone path when
    /// no structurally-matching input op is live at this position.
    pub(crate) fn emit_reusing(&mut self, op: Op) -> OpRef {
        self.emit_impl(op, true)
    }

    fn emit_impl(&mut self, mut op: Op, reuse: bool) -> OpRef {
        if op.pos.get().is_none() || op.pos.get().is_constant() {
            // Tag the freshly allocated position with the producer op's
            // result type so the variant-tag readers
            // (`opref_type`/`OptBoxEnv::get_type`) resolve at priority 0
            // (resoperation.py:1693 `opclasses[opnum].type` parity).
            op.pos.set(self.reserve_pos_typed(op.result_type()));
        } else {
            // Position invariants for an op that already carries a position:
            //
            // (a) body optimization runs through a fresh TraceIterator whose
            //     `_index` starts at `next_global_opref`, so Phase 2 op
            //     results live in a disjoint `[next_global_opref..)`
            //     range that no prior `emit` has touched.
            //
            // (b) preamble / standalone runs start `next_pos` at
            //     `max(num_inputs, max_raw_pos + 1)`, and `reserve_pos`
            //     is monotonic, so fresh positions are always above any
            //     raw trace op.pos the trace carries.
            //
            // (c) `import_state` only creates op/inputarg forwarding chains
            //     on inputarg slots (in `[inputarg_base..inputarg_base +
            //     num_inputs)`) — never on op-result positions that a
            //     later `emit` would try to use.
            //
            // Together these guarantee that:
            //   - `new_operations` never contains two ops at the same pos
            //   - an op being emitted whose pos is a non-void result does
            //     not already have an op-forwarding redirect set
            //
            // Earlier majit revisions compensated for violations of this invariant
            // with two reactive branches in `emit()` (a collision reassign
            // that called `reserve_pos` again, and a forwarding-redirect
            // that called `reserve_pos` + `make_equal_to(old, new)` to route
            // downstream readers to the fresh position). Both branches
            // are not part of this model. Hard-assert the invariants here so
            // any regression is caught at the emit site rather than at a
            // downstream symptom.
            debug_assert!(
                !self
                    .new_operations
                    .iter()
                    .any(|e| e.pos.get() == op.pos.get()),
                "emit: OpRef collision at {:?} — new_operations already contains this position. \
                 Body optimization should use a fresh TraceIterator and reserve_pos() should be \
                 monotonic above all raw trace positions.",
                op.pos.get(),
            );
            let has_op_fwd = self
                .get_box_replacement_operand_opt(op.pos.get())
                .map_or(false, |b| self.has_op_forwarding(&b));
            debug_assert!(
                !(has_op_fwd && op.result_type() != majit_ir::Type::Void),
                "emit: op-forwarding redirect set on non-void result position {:?} — \
                 import_state should only forward inputarg slots in \
                 [inputarg_base..inputarg_base + num_inputs), and body op results \
                 live in a disjoint range above the parent trace high-water mark.",
                op.pos.get(),
            );
            self.next_pos = self.next_pos.max(op.pos.get().raw().saturating_add(1));
        }
        let pos_ref = op.pos.get();
        // RPython parity: emit() does NOT clear forwarding.
        // In RPython, Box._forwarded is never cleared by emit — each Box
        // has unique identity. The forwarding set by import_box must
        // survive body op emission for consumer switchover to work.

        // RPython optimizer.py:652-686 emit_guard_operation — guard resume
        // data sharing via _copy_resume_data_from / ResumeGuardCopiedDescr.
        //
        // RPython has exactly one emit path (`Optimizer._emit_operation`,
        // optimizer.py:614).  Pyre's `Optimizer::emit_operation`
        // (optimizer.rs:3259) handles guard dispatch (force_box on args,
        // emit_guard_operation, force_box on fail_args, _maybe_replace_guard_value)
        // and then calls `ctx.emit(op.clone())` for the
        // `_newoperations.append(op)` step (optimizer.py:646).  The
        // OptContext-side emit_guard_operation is the standalone path
        // used by unit tests that drive `OptContext::emit` directly
        // without going through `Optimizer::emit_operation`.
        //
        // Skip the OptContext guard handling when the Optimizer is
        // mid-flight (`in_final_emission == true`) so production runs
        // through one emit_guard_operation (RPython parity).  The
        // OptContext path remains the sole guard handler for the
        // standalone test entry point.
        //
        // optimizer.py:639-644: side-effectful non-guard ops clear the
        // sharing chain.  Only relevant for the OptContext-managed
        // `last_guard_idx`; in_final_emission runs use
        // `Optimizer::last_guard_op_idx` instead.
        if op.opcode.is_guard() {
            if !self.in_final_emission {
                self.emit_guard_operation(&mut op);
            }
        } else if !self.in_final_emission {
            // optimizer.py:705-711: is_call_pure_pure_canraise — CallPure that
            // can_raise(ignore_memoryerror=True) counts as side-effectful even
            // though has_no_side_effect is true for call_pure opcodes.
            let dominated_by_side_effect = if (op.opcode.has_no_side_effect()
                || op.opcode.is_ovf()
                || op.opcode.is_jit_debug())
                && !Self::is_call_pure_pure_canraise(&op)
            {
                false
            } else {
                true
            };
            if dominated_by_side_effect {
                self.last_guard_idx = None;
            }
        }

        // The op is about to be pushed into `new_operations`,
        // after which `op_at` resolves its intrinsic `op.type_`
        // (resoperation.py:1693 parity) and `opref_type` returns it via
        // the primary fast path. The Box.type invariant is checked by
        // `debug_assert_box_type_invariant` below.
        Self::debug_assert_box_type_invariant(&op);
        let op_pos = op.pos.get();
        // Reuse path: when an unchanged op flows through to final emission, reuse
        // the recorder input op (the `live_synthetics` entry at this position) as
        // the emitted producer rather than cloning. The input op is what later
        // ops' operands already wrap, and the passes already wrote its
        // `_forwarded` host via `find_producer_op`, so overwriting its args with
        // the resolved operands and copying the descr makes it the single box per
        // value (resoperation.py:233 the op IS the box). This is the structural
        // collapse the clone path's catch-up only approximates by copying
        // `_forwarded` onto a fresh clone and redirecting input -> clone.
        // Guards are excluded (the guard path mutates `op` in emit_guard_operation
        // before reaching here). The opcode/num_args match guards against reusing
        // a non-matching stand-in; on any mismatch we fall through to the clone
        // path below, which is always correct.
        if reuse && !op.opcode.is_guard() {
            if let Some(i) = self.live_synthetics.iter().position(|s| {
                s.pos.get() == op_pos && s.opcode == op.opcode && s.num_args() == op.num_args()
            }) {
                let reused = self.live_synthetics.swap_remove(i);
                for k in 0..op.num_args() {
                    reused.setarg(k, op.arg(k));
                }
                match op.getdescr() {
                    Some(d) => reused.setdescr(d),
                    None => reused.cleardescr(),
                }
                // optimizer.py:674 `self._emittedoperations[op] = None`. The
                // clone path below records this too; `get_producing_op` only
                // admits producers present here, so the reused op must be
                // marked emitted or it stays invisible to producer matching.
                self.emitted_operations
                    .insert(majit_ir::operand::Operand::from_bound_op(&reused));
                self.new_operations.push(reused);
                return op_pos;
            }
        }
        let op_rc = std::rc::Rc::new(op);
        // Catch up any BoxRef placeholder that `materialize_box_at` created for
        // `op_pos` ahead of this emit (forward-reference path).
        // `resoperation.py:233 _forwarded` lives on the operation
        // object; late binding establishes that connection so
        // subsequent `box.set_forwarded` reaches `op.forwarded`.
        //
        // The synthetic stand-in registered for `op_pos` by `materialize_box_at` /
        // `bind_input_resops` is the `live_synthetics` entry at this position.
        // Migrate its `_forwarded` onto the real producer (resoperation.py:233
        // `_forwarded` lives on the op) and drop it from `live_synthetics` so
        // the superseded stand-in is not drained into `phase1_emit_ops`. Each
        // `op_pos` has at most one live stand-in, so the position match is
        // unambiguous. This is the sole carry-over path: the synthetic is the
        // `_forwarded` host every `find_producer_op` reaches before this emit
        // supersedes it.
        if let Some(i) = self
            .live_synthetics
            .iter()
            .position(|s| s.pos.get() == op_pos)
        {
            let synth = self.live_synthetics.swap_remove(i);
            *op_rc.forwarded.borrow_mut() = synth.forwarded.borrow().clone();
            // replace_op_with parity (optimizer.py): forward the superseded
            // stand-in to the emitted producer. A consumer dispatched BEFORE
            // this producer emitted (forward reference) bound its operand to
            // `synth`; without this link its box-native walk freezes on the
            // orphaned stand-in while the OpRef path resolves `op_rc`, so a
            // later fold (e.g. GUARD_TRUE constant-folding the operand) lands
            // on a different host and `resolve_box_box`'s witness diverges.
            // `synth` stays alive in `resop_refs` (seeded there alongside
            // `live_synthetics`), so its `Weak` upgrades and the chain reaches
            // `op_rc`.
            if !std::rc::Rc::ptr_eq(&synth, &op_rc) {
                use majit_ir::box_ref::ForwardingHost;
                synth.set_forwarded_op(&op_rc);
            }
        }
        // optimizer.py:674 `self._emittedoperations[op] = None`.
        self.emitted_operations
            .insert(majit_ir::operand::Operand::from_bound_op(&op_rc));
        self.new_operations.push(op_rc);
        pos_ref
    }

    /// Register an op dispatched through the passes (`emit_extra` /
    /// `send_extra_operation`) as the producer for its result position,
    /// mirroring `bind_input_resops`. `find_producer_op(pos)` then resolves
    /// box lookups for this position — `materialize_box_at(pos)` and operand
    /// resolution alike — to this `OpRc`'s `_forwarded` host
    /// (resoperation.py:233) instead of a freshly-minted stand-in, keeping a
    /// single box identity per position. Without this, a pass that folds the
    /// dispatched op via `make_equal_to(from_bound_op(op_rc), ..)` writes the
    /// forwarding onto a private `Rc<Op>` that `find_producer_op` cannot
    /// reach, so the replacement is silently dropped (RPython needs no analog:
    /// the op IS its box, resoperation.py:233-248).
    ///
    /// `emit`'s `live_synthetics` catch-up upgrades the binding to the real
    /// producer once the op is emitted; a folded op stays as the chain host.
    /// When `materialize_box_at` already minted a stand-in for this position,
    /// `op_rc` supersedes it (carrying its `_forwarded`) rather than skipping
    /// registration — otherwise queued-pass reads resolve to the stale stand-in
    /// while writes land on the unregistered `op_rc`, and emit's blind catch-up
    /// then clobbers those writes with the stand-in's slot. (InputArg / const /
    /// sentinel positions are excluded.)
    pub(crate) fn register_extra_producer(&mut self, op_rc: &majit_ir::OpRc) {
        self.install_canonical_producer(op_rc);
    }

    /// RPython emit_extra(op, emit=False) parity: queue an operation to
    /// be processed through passes AFTER the calling pass. Skips earlier
    /// passes (including the caller) to avoid re-absorption loops.
    /// `after_pass_idx`: index of the calling pass (op starts from idx+1).
    pub fn emit_extra(&mut self, after_pass_idx: usize, mut op: Op) -> OpRef {
        if op.pos.get().is_none() {
            // Typed allocation, same rationale as `emit`.
            op.pos.set(self.reserve_pos_typed(op.result_type()));
        } else {
            self.next_pos = self.next_pos.max(op.pos.get().raw().saturating_add(1));
        }
        let pos_ref = op.pos.get();
        // Queued ops carry their intrinsic `op.type_`
        // (resoperation.py:1693 parity). Once the queued op flushes
        // through `propagate_one` into `new_operations`, `op_at` resolves
        // its type without the side-table detour.
        Self::debug_assert_box_type_invariant(&op);
        let op_rc = std::rc::Rc::new(op);
        // Register the queued op as the producer for its position so a fold
        // through `make_equal_to(from_bound_op(op_rc), ..)` reaches a host
        // `find_producer_op` resolves to.
        self.register_extra_producer(&op_rc);
        self.extra_operations_after
            .push_back((after_pass_idx + 1, op_rc));
        pos_ref
    }

    pub fn initialize_imported_short_preamble_builder(
        &mut self,
        label_args: &[OpRef],
        short_inputargs: &[majit_ir::box_ref::BoxRef],
        exported_short_boxes: &[crate::optimizeopt::shortpreamble::PreambleOp],
    ) {
        let produced: Vec<(
            majit_ir::box_ref::BoxRef,
            crate::optimizeopt::shortpreamble::ProducedShortOp,
        )> = exported_short_boxes
            .iter()
            .map(|entry| {
                // `materialize_box_at`, not `from_bound_op`: a
                // const-folded entry carries an inline-Const pos,
                // which resolves to its Const box. The map keys by this res
                // box (#146/S8); the single-op re-export lookup (pure.rs)
                // reproduces it via `materialize_box_at(source)`.
                //
                // materialize_box_at is NON-idempotent for an unregistered
                // ResOp pos: the first call mints a fresh `new_resop` box and
                // registers a synthetic producer; subsequent calls resolve
                // that synthetic to a stable `from_bound_op` box. The lookup is
                // itself a subsequent call, so the warm-up call registers the
                // synthetic and the second (stable) box is the real key.
                let pos = entry.op.pos.get();
                let _ = self.materialize_box_at(pos);
                let res = self.materialize_box_at(pos);
                (
                    res.clone(),
                    crate::optimizeopt::shortpreamble::ProducedShortOp {
                        kind: entry.kind.clone(),
                        res,
                        preamble_op: std::rc::Rc::new((*entry.op).clone()),
                        invented_name: entry.invented_name,
                        same_as_source: entry.same_as_source.clone(),
                        label_arg_idx: entry.label_arg_idx,
                    },
                )
            })
            .collect();
        let builder = crate::optimizeopt::shortpreamble::ShortPreambleBuilder::new(
            label_args,
            &produced,
            short_inputargs,
        );
        self.imported_short_preamble_builder = Some(builder);
        self.imported_short_preamble_used.clear();
    }

    /// shortpreamble.py:409-430 ShortPreambleBuilder constructor parity.
    ///
    /// Reads `exported_state.short_boxes` (RPython `ExportedState.short_boxes`)
    /// directly, classifying each `ProducedShortOp.preamble_op.args` as
    /// Slot/Const/Produced at consume time and rebuilding the
    /// `ShortPreambleBuilder` keyed by Phase 2 OpRefs.
    ///
    /// Replaces the legacy `_from_exported_ops` function which read the
    /// `Vec<ExportedShortOp>` enum-serialization path. The new path
    /// matches RPython literally — no intermediate enum, polymorphism via
    /// `produce_op`-side data on `ProducedShortOp` itself.
    ///
    /// The Phase-2 result OpRef for each entry is read from `result_map`,
    /// computed before this constructor just as RPython already has the
    /// target `Box` identities before `ProducedShortOp.produce_op` runs.
    /// Args are resolved via local `produced_results` (source pos →
    /// resolved OpRef) and the caller-owned `imported_constants` map
    /// (source const OpRef → seeded fresh slot), which is reused by the
    /// following produce-op loop.
    pub fn initialize_imported_short_preamble_builder_from_short_boxes(
        &mut self,
        short_args: &[OpRef],
        short_inputargs: &[majit_ir::box_ref::BoxRef],
        short_boxes: &[(OpRef, crate::optimizeopt::shortpreamble::ProducedShortOp)],
        short_box_const_values: &majit_ir::VecMap<OpRef, majit_ir::Value>,
        result_map: &majit_ir::VecMap<OpRef, OpRef>,
        mut imported_constants: &mut majit_ir::VecMap<OpRef, OpRef>,
        exported_infos: &majit_ir::VecMap<
            majit_ir::box_ref::BoxRef,
            crate::optimizeopt::info::OpInfo,
        >,
    ) -> bool {
        use crate::optimizeopt::shortpreamble::{
            PreambleOpKind, ProducedShortOp, ShortPreambleBuilder,
        };

        // shortpreamble.py:414-426 ShortPreambleBuilder.__init__ parity:
        //
        //   for produced_op in short_boxes:
        //       op = produced_op.short_op.res
        //       preamble_op = produced_op.preamble_op
        //       if isinstance(op, Const):
        //           info = optimizer.getinfo(op)
        //       else:
        //           info = exported_infos.get(op, None)
        //           if info is None:
        //               info = empty_info
        //       preamble_op.set_forwarded(info)
        //
        // RPython sets `_forwarded = info` on each `preamble_op` so a
        // later `use_box` (≡ majit `force_op_from_preamble_op`) reads
        // the info via `get_forwarded()` and routes it through
        // `setinfo_from_preamble(box, info, None)` + `info.make_guards`.
        //
        // majit's `force_op_from_preamble_op` reads the equivalent via
        // `self.get_ptr_info(preamble_source)`; without this pre-seed,
        // info from `exported_infos` for sources outside
        // `next_iteration_args` (Pure / Heap / LoopInvariant short-op
        // sources) never reaches the use_box path.
        //
        // Guard against clobbering existing forwarding (set by an
        // earlier `make_equal_to` from `import_state` for sources that
        // happen to coincide with `next_iteration_args`): only seed
        // when no forwarding is recorded yet.
        //
        // Replay slot rule, matching `ImportedShortPureOp::new` (mod.rs:194)
        // and the producer-side `pop.preamble_op.pos` written by
        // `produce_pure` / `produce_heap_field` / `produce_heap_array_item` /
        // `produce_loop_invariant` in shortpreamble.rs.
        //
        // The rule reduces to: `replay slot = result_opref` iff the producer
        // installs `make_equal_to(source, result_opref)`, otherwise `source`.
        // PyPy `shortpreamble.py:401, 414` calls `preamble_op.set_forwarded(info)`
        // on the replay Op object — distinct from `PreambleOp.op = self.res`.
        // pyre's flat-OpRef model collapses the two onto one slot per OpRef,
        // so when `make_equal_to` is installed at `source`, the replay's
        // `_forwarded` slot must be moved to `result_opref` to avoid the
        // op-forwarding chain clobbering the seeded info.
        //
        //   * invented Pure → result_opref. `produce_pure` installs
        //     `make_equal_to(source, result_opref)` (Fix #3).
        //   * Heap (field + array) → result_opref. `produce_heap_field` /
        //     `produce_heap_array_item` install `make_equal_to` (Cat-2.2 B/C).
        //   * LoopInvariant → result_opref. `produce_loop_invariant` installs
        //     `make_equal_to` (Cat-2.2 A); the synthetic `SameAsI` replay built
        //     by `optimize_CALL_LOOPINVARIANT_*` in `rewrite.rs` writes
        //     `replay.pos = ctx.get_box_replacement(source)` to land on the
        //     same slot.
        //   * non-invented Pure → source. PyPy `shortpreamble.py:120 op = self.res`
        //     with no forwarding installed; pyre keeps source's slot free for
        //     the seeded info.
        //
        // The same rule drives `seed_at` (`set_preamble_forwarded_info`),
        // the BUILDER's `replay.pos`, and the builder-side `produced_results`
        // dependency map so all four sources of replay identity stay in
        // lockstep with PyPy `shortpreamble.py:414-426` (`__init__`).
        let replay_pos = |source: OpRef, produced_op: &ProducedShortOp| -> OpRef {
            let installs_replace_op = match produced_op.kind {
                PreambleOpKind::Pure => produced_op.invented_name,
                PreambleOpKind::Heap | PreambleOpKind::LoopInvariant => true,
                PreambleOpKind::InputArg | PreambleOpKind::Guard => false,
            };
            if installs_replace_op {
                *result_map.get(&source).unwrap_or(&source)
            } else {
                source
            }
        };
        for (source, produced_op) in short_boxes {
            // shortpreamble.py:417-421: op = produced_op.short_op.res;
            //     if isinstance(op, Const): info = optimizer.getinfo(op)
            //     else: info = exported_infos.get(op, None)
            // Look up by the result BOX (`produced_op.res`) — the exporter keyed
            // `_expand_info(produced_op.res)` by the same Rc (cloned from the
            // shared `exported_short_boxes` entry), so the box-identity lookup
            // hits. Const results are skipped (unroll.py:483 export skips them).
            if produced_op.res.to_opref().is_constant() {
                continue;
            }
            if let Some(info) = exported_infos.get(&produced_op.res) {
                self.set_preamble_forwarded_info(replay_pos(*source, produced_op), info);
            }
        }

        // `produced` (OpRef dual-key: source + result_opref) is internal
        // scaffolding for `dep_or_materialize` below, which resolves a
        // dependency arg by its replay position. `builder_entries` is the
        // #146/S8 builder map: ONE entry per short box keyed by the Phase-1
        // carried res box (`produced_op.res`, the same Rc the produce loop
        // reads as `self.res`). The carried box is invariant to the
        // invented-name replay-position aliasing the dual key compensates for,
        // so the builder map collapses to a single box-identity key.
        let mut produced: Vec<(OpRef, ProducedShortOp)> = Vec::with_capacity(short_boxes.len());
        let mut builder_entries: Vec<(majit_ir::box_ref::BoxRef, ProducedShortOp)> =
            Vec::with_capacity(short_boxes.len());
        let mut produced_results: majit_ir::VecMap<OpRef, OpRef> = majit_ir::VecMap::new();
        // shortpreamble.py:PreambleOp.add_op_to_short — Pure ops whose
        // opcode is a Call get rewritten to the CallPure* equivalent so
        // the short preamble can replay the cached call without
        // re-executing arbitrary side effects.
        let pure_call_opcode = |op: OpCode| -> OpCode {
            match op {
                OpCode::CallI => OpCode::CallPureI,
                OpCode::CallR => OpCode::CallPureR,
                OpCode::CallF => OpCode::CallPureF,
                OpCode::CallN => OpCode::CallPureN,
                other => other,
            }
        };
        // shortpreamble.py:PreambleOp.add_op_to_short — LoopInvariant ops
        // become CallLoopinvariant* so the short preamble re-executes the
        // call exactly once per loop iteration.
        let loop_invariant_opcode = |result_type: majit_ir::Type| -> OpCode {
            match result_type {
                majit_ir::Type::Int => OpCode::CallLoopinvariantI,
                majit_ir::Type::Ref => OpCode::CallLoopinvariantR,
                majit_ir::Type::Float => OpCode::CallLoopinvariantF,
                majit_ir::Type::Void => OpCode::CallLoopinvariantN,
            }
        };
        // shortpreamble.py:283 `ShortBoxes.produce_arg` — classify an arg
        // through the shared classifier, then collapse the Slot/Const/
        // Produced variants down to the Phase-2 OpRef the builder needs.
        // Sharing the classifier with `ProducedShortOp::produce_op` (which
        // also dispatches via `classify_short_arg`) keeps the two consume
        // sites locked to a single rule, mirroring RPython's single
        // `produce_arg` path.
        let resolve_arg = |arg: OpRef,
                           ctx: &mut Self,
                           produced_results: &majit_ir::VecMap<OpRef, OpRef>,
                           imported_constants: &mut majit_ir::VecMap<OpRef, OpRef>|
         -> Option<OpRef> {
            crate::optimizeopt::shortpreamble::classify_short_arg(
                ctx,
                arg,
                short_inputargs,
                short_args,
                produced_results,
                imported_constants,
                short_box_const_values,
            )
            .map(|cls| match cls {
                crate::optimizeopt::ImportedShortPureArg::OpRef(r) => r,
                crate::optimizeopt::ImportedShortPureArg::Const(_, r) => r,
            })
        };
        // shortpreamble.py:283-296 produce_arg object-carry: a dependency
        // arg is the dep's replay op OBJECT (upstream returns
        // `produced_short_boxes[op].preamble_op`). Bind dep args to the
        // dep entry's replay Rc (the same dual-key dict the builder will
        // hold — last insert wins, matching VecMap overwrite), so
        // `use_box` reads deps off the operand binding instead of a
        // position-keyed side map. Slot / Const args keep the positional
        // materialization.
        let dep_or_materialize =
            |ctx: &mut Self, produced: &[(OpRef, ProducedShortOp)], r: OpRef| {
                // shortpreamble.py:288 Const arm: the arg is the Const box
                // itself — a const-folded entry carries an inline-Const
                // pos/key, which must not bind to the replay op.
                if r.is_constant() {
                    return ctx.materialize_operand_at(r);
                }
                produced
                    .iter()
                    .rev()
                    .find(|(k, _)| *k == r)
                    .map(|(_, dep)| Operand::from_bound_op(&dep.preamble_op))
                    .unwrap_or_else(|| ctx.materialize_operand_at(r))
            };

        for (source, produced_op) in short_boxes {
            // Some ProducedShortOps (PreambleOpKind::Heap with non-getfield /
            // non-getarrayitem opcodes, or other non-emitting entries) have
            // no Phase-2 result. They are no-ops for the imported builder.
            let result_opref = match result_map.get(source).copied() {
                Some(r) => r,
                None => continue,
            };
            match produced_op.kind {
                PreambleOpKind::Pure => {
                    let mut resolved_args = Vec::with_capacity(produced_op.preamble_op.num_args());
                    for arg in produced_op.preamble_op.getarglist().iter() {
                        let Some(resolved) = resolve_arg(
                            arg.to_opref(),
                            self,
                            &produced_results,
                            &mut imported_constants,
                        ) else {
                            return false;
                        };
                        resolved_args.push(resolved);
                    }
                    let resolved_arg_boxes: Vec<Operand> = resolved_args
                        .iter()
                        .map(|a| dep_or_materialize(self, &produced, *a))
                        .collect();
                    let mut op = Op::new(
                        pure_call_opcode(produced_op.preamble_op.opcode),
                        &resolved_arg_boxes,
                    );
                    op.pos.set(replay_pos(*source, produced_op));
                    if let Some(d) = produced_op.preamble_op.getdescr() {
                        op.setdescr(d);
                    }
                    let res = self.materialize_box_at(op.pos.get());
                    let new_pop = ProducedShortOp {
                        kind: PreambleOpKind::Pure,
                        res,
                        preamble_op: std::rc::Rc::new(op),
                        invented_name: produced_op.invented_name,
                        same_as_source: produced_op.same_as_source.clone(),
                        label_arg_idx: produced_op.label_arg_idx,
                    };
                    builder_entries.push((produced_op.res.clone(), new_pop.clone()));
                    produced.push((*source, new_pop.clone()));
                    if *source != result_opref {
                        produced.push((result_opref, new_pop));
                    }
                    produced_results.insert(*source, replay_pos(*source, produced_op));
                }
                PreambleOpKind::Heap => {
                    let result_type = produced_op.preamble_op.result_type();
                    let descr = match produced_op.preamble_op.getdescr() {
                        Some(d) => d,
                        None => continue,
                    };
                    let object_arg = produced_op.preamble_op.arg(0);
                    let Some(obj) = resolve_arg(
                        object_arg.to_opref(),
                        self,
                        &produced_results,
                        &mut imported_constants,
                    ) else {
                        return false;
                    };
                    let new_pop = match produced_op.preamble_op.opcode {
                        OpCode::GetfieldGcI | OpCode::GetfieldGcR | OpCode::GetfieldGcF => {
                            let opcode = match result_type {
                                majit_ir::Type::Int => OpCode::GetfieldGcI,
                                majit_ir::Type::Ref => OpCode::GetfieldGcR,
                                majit_ir::Type::Float => OpCode::GetfieldGcF,
                                majit_ir::Type::Void => return false,
                            };
                            let obj_b = dep_or_materialize(self, &produced, obj);
                            let mut op = Op::new(opcode, &[obj_b]);
                            op.pos.set(replay_pos(*source, produced_op));
                            op.setdescr(descr);
                            let res = self.materialize_box_at(op.pos.get());
                            ProducedShortOp {
                                kind: PreambleOpKind::Heap,
                                res,
                                preamble_op: std::rc::Rc::new(op),
                                invented_name: produced_op.invented_name,
                                same_as_source: produced_op.same_as_source.clone(),
                                label_arg_idx: produced_op.label_arg_idx,
                            }
                        }
                        OpCode::GetarrayitemGcI
                        | OpCode::GetarrayitemGcR
                        | OpCode::GetarrayitemGcF => {
                            let opcode = match result_type {
                                majit_ir::Type::Int => OpCode::GetarrayitemGcI,
                                majit_ir::Type::Ref => OpCode::GetarrayitemGcR,
                                majit_ir::Type::Float => OpCode::GetarrayitemGcF,
                                majit_ir::Type::Void => return false,
                            };
                            // shortpreamble.py:81 `g.getarg(1).getint()`:
                            // pull the integer VALUE through the shared
                            // `classify_short_arg` rule, which checks
                            // `short_box_const_values` (producer snapshot)
                            // first then the consumer ctx const pool.
                            // `OpRef::raw()` is a tagged trace position —
                            // not the constant integer value.
                            let index_arg = produced_op.preamble_op.arg(1);
                            let index_opref = match resolve_arg(
                                index_arg.to_opref(),
                                self,
                                &produced_results,
                                imported_constants,
                            ) {
                                Some(r) => r,
                                None => return false,
                            };
                            let obj_b = dep_or_materialize(self, &produced, obj);
                            let index_b = dep_or_materialize(self, &produced, index_opref);
                            let mut op = Op::new(opcode, &[obj_b, index_b]);
                            op.pos.set(replay_pos(*source, produced_op));
                            op.setdescr(descr);
                            let res = self.materialize_box_at(op.pos.get());
                            ProducedShortOp {
                                kind: PreambleOpKind::Heap,
                                res,
                                preamble_op: std::rc::Rc::new(op),
                                invented_name: produced_op.invented_name,
                                same_as_source: produced_op.same_as_source.clone(),
                                label_arg_idx: produced_op.label_arg_idx,
                            }
                        }
                        _ => continue,
                    };
                    builder_entries.push((produced_op.res.clone(), new_pop.clone()));
                    produced.push((*source, new_pop.clone()));
                    if *source != result_opref {
                        produced.push((result_opref, new_pop));
                    }
                    produced_results.insert(*source, replay_pos(*source, produced_op));
                }
                PreambleOpKind::LoopInvariant => {
                    let result_type = produced_op.preamble_op.result_type();
                    let Some(func_opref) = resolve_arg(
                        produced_op.preamble_op.arg(0).to_opref(),
                        self,
                        &produced_results,
                        imported_constants,
                    ) else {
                        return false;
                    };
                    if self
                        .get_box_replacement_operand_opt(func_opref)
                        .and_then(|cb| cb.const_int())
                        .is_none()
                    {
                        return false;
                    }
                    let func_b = dep_or_materialize(self, &produced, func_opref);
                    let mut op = Op::new(loop_invariant_opcode(result_type), &[func_b]);
                    op.pos.set(replay_pos(*source, produced_op));
                    let res = self.materialize_box_at(op.pos.get());
                    let new_pop = ProducedShortOp {
                        kind: PreambleOpKind::LoopInvariant,
                        res,
                        preamble_op: std::rc::Rc::new(op),
                        invented_name: produced_op.invented_name,
                        same_as_source: produced_op.same_as_source.clone(),
                        label_arg_idx: produced_op.label_arg_idx,
                    };
                    builder_entries.push((produced_op.res.clone(), new_pop.clone()));
                    produced.push((*source, new_pop.clone()));
                    if *source != result_opref {
                        produced.push((result_opref, new_pop));
                    }
                    produced_results.insert(*source, replay_pos(*source, produced_op));
                }
                PreambleOpKind::InputArg | PreambleOpKind::Guard => {}
            }
        }

        let mut builder = ShortPreambleBuilder::new(short_args, &builder_entries, short_inputargs);
        for &opref in imported_constants.values() {
            builder.note_known_constant(opref);
        }
        self.imported_short_preamble_builder = Some(builder);
        self.imported_short_preamble_used.clear();
        true
    }

    /// unroll.py:26-39: force_op_from_preamble(preamble_op)
    ///
    /// RPython receives a PreambleOp with invented_name already set.
    /// Calls use_box then registers in potential_extra_ops.
    pub fn force_op_from_preamble_op(
        &mut self,
        preamble_op: &crate::optimizeopt::info::PreambleOp,
    ) -> OpRef {
        let preamble_source = preamble_op.op.to_opref();
        // RPython `return preamble_op.op` returns the carried Box. In majit,
        // `pop.op` carries the Phase 1 source box; `make_equal_to(source,
        // body_visible)` is called by the producer for invented Pure / Heap /
        // LoopInvariant, so walking the forwarding chain reaches the
        // body-visible OpRef. Non-invented Pure has no forwarding installed,
        // so `get_box_replacement(source) == source` and the body references
        // source directly (RPython parity for non-invented `op = self.res`).
        let resolved = self.resolve_box_box(&preamble_op.op);
        let result = resolved.to_opref();
        let result_type = preamble_op.preamble_op.result_type();
        let is_constant = resolved.const_value().is_some();
        let first_use = !self.imported_short_preamble_used.contains(&preamble_source);
        if first_use {
            self.imported_short_preamble_used.push(preamble_source);
        }
        if first_use {
            // unroll.py:32: use_box(op, preamble_op.preamble_op, self).
            // RPython passes the preamble_op directly — no lookup miss possible.
            // majit prefers the produced_short_boxes lookup (Phase-2 remapped pos)
            // with fallback to info::PreambleOp.preamble_op.
            let (arg_guards, result_guards) = self.collect_use_box_guards(&preamble_op.preamble_op);
            // unroll.py:28: assert self.short_preamble_producer is not None
            if let Some(mut builder) = self.active_short_preamble_producer.take() {
                builder.use_box(
                    preamble_source,
                    &preamble_op.preamble_op,
                    &arg_guards,
                    &result_guards,
                );
                self.active_short_preamble_producer = Some(builder);
            } else if let Some(mut builder) = self.imported_short_preamble_builder.take() {
                builder.use_box(
                    preamble_source,
                    &preamble_op.preamble_op,
                    &arg_guards,
                    &result_guards,
                );
                self.imported_short_preamble_builder = Some(builder);
            } else {
                unreachable!("force_op_from_preamble_op: no short_preamble_producer");
            }
            // shortpreamble.py:401-405: info = preamble_op.get_forwarded();
            // preamble_op.set_forwarded(None);
            // optimizer.setinfo_from_preamble(box, info, None)
            //
            // RPython reads `_forwarded` from the replay Op object
            // (`preamble_op`), NOT from `preamble_op.op` (= self.res or
            // the alt for invented). pyre's flat-OpRef equivalent is
            // `pop.preamble_op.pos` — the OpRef the replay Op was
            // constructed at by `ImportedShortPureOp::new` (mod.rs:144).
            // For invented Pure that OpRef differs from `pop.op` (the
            // alt) so the alt's `make_equal_to(...)` chain at
            // `forwarded[pop.op]` does not collide with the replay's
            // info at `forwarded[pop.preamble_op.pos]`.
            if let Some(info) =
                self.take_preamble_forwarded_opinfo(preamble_op.preamble_op.pos.get())
            {
                self.setinfo_from_preamble_item_option(result, &info, None);
            }
            // RPython PreambleOp carries Box.type intrinsically.
            // the replay `result` OpRef is typed via the upstream factory
            // (`op_typed`); priority 0 of `opref_type`
            // resolves it from the variant tag without a side-table seed.
            let _ = result_type;
            // unroll.py:34-37: potential_extra_ops[op] = preamble_op
            if !is_constant {
                // unroll.py:35-36: invented_name → get_box_replacement(op)
                let key = if preamble_op.invented_name {
                    result
                } else {
                    preamble_source
                };
                if crate::optimizeopt::majit_log_enabled() {
                    eprintln!(
                        "[jit] potential_extra_ops.insert key={key:?} source={preamble_source:?} result={result:?} invented={}",
                        preamble_op.invented_name
                    );
                }
                // `unroll.py:37` dict-assign semantics — overwrite if the
                // key already exists, otherwise append.
                if let Some(entry) = self.potential_extra_ops.iter_mut().find(|(k, _)| *k == key) {
                    entry.1 = preamble_op.clone();
                } else {
                    self.potential_extra_ops.push((key, preamble_op.clone()));
                }
            }
        }
        // unroll.py:38 `return preamble_op.op`. RPython's `preamble_op.op`
        // equals `self.res` (shortpreamble.py:120 `op = self.res`); pyre's
        // Phase 1 source IS `self.res` for the imported short box. Return
        // it directly so non-invented Pure body references resolve through
        // the Phase 1 OpRef. The use-before-def pass at LABEL emission
        // (unroll.rs `assemble_peeled_trace_with_jump_args`) extends
        // `LABEL.arglist` with that Phase 1 OpRef when the body actually
        // uses it; the orthodox `force_box → potential_extra_ops.pop →
        // add_preamble_op` path (shortpreamble.py:432-440) handles
        // `used_boxes` / `short_preamble_jump` / `extra_same_as` for the
        // imported short box.
        let _ = result;
        preamble_source
    }

    /// shortpreamble.py:383-396,401-406: collect guards from the forwarded
    /// info of preamble_op's args and result. RPython's `info = arg.get_forwarded()`
    /// returns whatever is stored — PtrInfo *or* IntBound — and calls
    /// `info.make_guards(...)` uniformly.
    fn collect_use_box_guards(&mut self, preamble_op: &Op) -> (Vec<Op>, Vec<Op>) {
        // shortpreamble.py:383-401 line-by-line:
        //
        //   for arg in preamble_op.getarglist():
        //       if isinstance(arg, Const):
        //           continue
        //       if isinstance(arg, AbstractInputArg):
        //           info = arg.get_forwarded()
        //           if info is not None and info is not empty_info:
        //               info.make_guards(arg, self.short, optimizer)
        //       elif arg.get_forwarded() is None:
        //           pass
        //       else:
        //           self.short.append(arg)
        //           info = arg.get_forwarded()
        //           if info is not empty_info:
        //               info.make_guards(arg, self.short, optimizer)
        //           arg.set_forwarded(None)
        //
        // RPython has three branches per arg:
        //   * Const → skip (pyre: `OpRef::is_constant()`).
        //   * AbstractInputArg with forwarded info → emit guards, do NOT
        //     clear (info lives across iterations on input args).
        //   * non-input non-Const with forwarded info → also append the
        //     arg op to `self.short` (handled by the builder's `use_box`
        //     dependency walk at shortpreamble.rs:1660-1688), emit
        //     guards, AND clear the slot to prevent double-emission.
        //
        // pyre's `take_preamble_forwarded_opinfo` is the take-clear
        // primitive matching `arg.set_forwarded(None)`. We use it for the
        // non-input branch only; input args use the read-only snapshot.
        enum ForwardedInfo {
            // shortpreamble.py:376-379 EmptyInfo / empty_info sentinel.
            // Its presence is meaningful for short-preamble dedup, but it
            // intentionally emits no guards.
            Empty,
            // info.py:600 PtrInfo + ConstPtrInfo (info.py:706). PtrInfo
            // dispatches further to ConstPtrInfo::make_guards when the
            // PtrInfo is a Constant variant.
            Ptr(PtrInfo),
            // intutils.py:1264 IntBound::make_guards. Constant ints come
            // through this arm via IntBound::is_constant().
            Int(crate::optimizeopt::intutils::IntBound),
            // info.py:851 FloatConstInfo carries a single ConstFloat;
            // make_guards (info.py:861) emits a GUARD_VALUE pinning `op`
            // to the ConstFloat. `set_preamble_forwarded_info` plants this
            // shape per shortpreamble.py:416
            // `preamble_op.set_forwarded(info)`.
            FloatConst(f64),
        }
        let snapshot_forwarded = |ctx: &Self, arg: OpRef| -> Option<ForwardedInfo> {
            // shortpreamble.py:387 `info = arg.get_forwarded()` — PyPy
            // returns the AbstractInfo subtype stored in `_forwarded`.
            // Pyre's canonical `_forwarded` host carries:
            //   `Forwarded::Info(OpInfo::Ptr(_))` — info.py:600 PtrInfo
            //   `Forwarded::Info(OpInfo::IntBound(_))` — intutils.py
            //   `Forwarded::Info(OpInfo::FloatConstInfo(_))` — info.py:851
            //       FloatConstInfo planted via set_preamble_forwarded_info.
            //   `Forwarded::Info(OpInfo::EmptyInfo(_))` —
            //       shortpreamble.py:379 empty_info.
            let forwarded = ctx.read_forwarded(arg)?;
            use crate::optimizeopt::info::OpInfo;
            match &forwarded {
                majit_ir::box_ref::Forwarded::Info(OpInfo::EmptyInfo(_)) => {
                    Some(ForwardedInfo::Empty)
                }
                majit_ir::box_ref::Forwarded::Info(OpInfo::Ptr(info)) => {
                    Some(ForwardedInfo::Ptr(info.borrow().clone()))
                }
                majit_ir::box_ref::Forwarded::Info(OpInfo::IntBound(b)) => {
                    Some(ForwardedInfo::Int(b.borrow().clone()))
                }
                majit_ir::box_ref::Forwarded::Info(OpInfo::FloatConstInfo(f)) => {
                    Some(ForwardedInfo::FloatConst(f.getconst()))
                }
                _ => None,
            }
        };
        // Phase 1 (read-only): classify each arg per the PyPy three-branch
        // shape and snapshot the info-bearing slot.
        struct ArgEntry {
            arg: OpRef,
            info: ForwardedInfo,
            is_input: bool,
        }
        let mut arg_entries: Vec<ArgEntry> = Vec::new();
        for arg in preamble_op.getarglist().iter() {
            let arg = arg.to_opref();
            // Branch 1: shortpreamble.py:384 `isinstance(arg, Const): continue`.
            if arg.is_constant() || arg.is_none() {
                continue;
            }
            // shortpreamble.py:386 `isinstance(arg, AbstractInputArg)` — the
            // classification is the arg's TYPE, not membership in
            // `short_inputargs`. A renamed short inputarg and an original
            // label arg are both AbstractInputArg; either way an input arg's
            // forwarded info must NOT be cleared (it lives across iterations).
            let is_input = matches!(
                arg,
                OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_)
            );
            if let Some(info) = snapshot_forwarded(self, arg) {
                arg_entries.push(ArgEntry {
                    arg,
                    info,
                    is_input,
                });
            }
            // shortpreamble.py:391 `elif arg.get_forwarded() is None: pass`
            // is the no-info branch; falling out of `snapshot_forwarded`
            // returning None is the equivalent.
        }
        let result_info: Option<(OpRef, ForwardedInfo)> =
            snapshot_forwarded(self, preamble_op.pos.get())
                .map(|info| (preamble_op.pos.get(), info));

        // Phase 2 (mutable): clear non-input arg slots — PyPy
        // `arg.set_forwarded(None)` (shortpreamble.py:397). Branch 2 (input
        // args) keeps its info; branch 3 (non-input) clears.
        for entry in &arg_entries {
            if !entry.is_input {
                let _ = self.take_preamble_forwarded_opinfo(entry.arg);
            }
        }

        // Phase 3: generate guards — `make_guards` takes `&mut self`
        // directly. Constants seed via reserve_const_ref + seed_constant
        // (mirroring `ConstInt` / `ConstPtr` inline construction); producer
        // OpRefs come from `alloc_op_position_typed`.
        let mut arg_guards = Vec::new();
        // info.py:861 FloatConstInfo.make_guards / ConstPtrInfo path —
        // single-value info classes emit a GUARD_VALUE that pins `op` to
        // the recorded constant.
        let emit_const_guard = |arg: OpRef, value: &Value, guards: &mut Vec<Op>, ctx: &mut Self| {
            // history.py:227/268/314 Const{Int,Float,Ptr}.value inline —
            // GUARD_VALUE second operand is the inline-Const directly.
            let c = match value {
                Value::Int(v) => OpRef::const_int(*v),
                Value::Float(v) => OpRef::const_float(*v),
                Value::Ref(v) => OpRef::const_ptr(*v),
                Value::Void => panic!("emit_const_guard: ConstVoid not allowed"),
            };
            // ConstInt/Float/Ptr value rides inline on `c` (history.py:227/
            // 268/314); no `seed_constant` step (its const arm is a no-op).
            let arg_b = ctx.materialize_operand_at(arg);
            let c_b = ctx.materialize_operand_at(c);
            guards.push(Op::new(OpCode::GuardValue, &[arg_b, c_b]));
        };
        for entry in &arg_entries {
            match &entry.info {
                ForwardedInfo::Empty => {}
                ForwardedInfo::Ptr(p) => p.make_guards(entry.arg, &mut arg_guards, self),
                ForwardedInfo::Int(b) => b.make_guards(entry.arg, &mut arg_guards, self),
                ForwardedInfo::FloatConst(f) => {
                    emit_const_guard(entry.arg, &Value::Float(*f), &mut arg_guards, self)
                }
            }
        }
        let mut result_guards = Vec::new();
        if let Some((result_ref, info)) = &result_info {
            match info {
                ForwardedInfo::Empty => {}
                ForwardedInfo::Ptr(p) => p.make_guards(*result_ref, &mut result_guards, self),
                ForwardedInfo::Int(b) => b.make_guards(*result_ref, &mut result_guards, self),
                ForwardedInfo::FloatConst(f) => {
                    emit_const_guard(*result_ref, &Value::Float(*f), &mut result_guards, self)
                }
            }
        }
        (arg_guards, result_guards)
    }

    /// shortpreamble.py:425 `preamble_op.set_forwarded(info)` for imported
    /// short preamble ops. Store the same family of info values that RPython
    /// stores in `_forwarded`, without transforming them through
    /// `setinfo_from_preamble` yet.
    fn set_preamble_forwarded_info(
        &mut self,
        source: OpRef,
        info: &crate::optimizeopt::info::OpInfo,
    ) {
        use crate::optimizeopt::info::OpInfo;
        if source.is_constant() {
            return;
        }
        if let Some(b) = self.get_box_replacement_operand_opt(source) {
            if self.has_forwarding(&b) {
                return;
            }
        }
        // shortpreamble.py:425 `preamble_op.set_forwarded(info)`. The replay
        // OpRef is a short-preamble op whose producer may not be registered
        // yet (the Pure / Heap / LoopInvariant replay slot is seeded here,
        // before the short-preamble body that builds the producing op).
        // `materialize_box_at` returns the canonical host, minting a `SameAs*`
        // synthetic into `resop_refs` when absent; `emit()` later re-binds it
        // to the real producer, carrying the forwarded state across.
        let b = self.materialize_operand_at(source);
        match info {
            OpInfo::Unknown => b.clear_forwarded(),
            OpInfo::EmptyInfo(_) => b.set_forwarded_info(info.clone()),
            other => b.set_forwarded_info(other.clone()),
        }
    }

    /// shortpreamble.py:401-405 line-by-line:
    ///
    /// ```python
    /// info = preamble_op.get_forwarded()
    /// preamble_op.set_forwarded(None)
    /// if optimizer is not None:
    ///     optimizer.setinfo_from_preamble(box, info, None)
    /// ```
    ///
    /// RPython reads `_forwarded` from the `preamble_op` Op object directly
    /// — `get_box_replacement` is NOT applied to the slot. Box replacement
    /// only matters for the `box` argument that subsequently receives the
    /// info via `setinfo_from_preamble(box, info, None)`. Walking the
    /// replacement chain on the source side would point at the body-visible
    /// OpRef whose slot is empty (the seed at `forwarded[source]` was
    /// installed by `set_preamble_forwarded_info`).
    ///
    /// `set_forwarded(None)` clears the slot so a second `use_box` for the
    /// same preamble op never re-fires `info.make_guards`. In majit's flat
    /// OpRef model the slot is shared with the box replacement chain
    /// (`Forwarded::Op` / `Forwarded::InputArg`), which other code follows
    /// via `get_box_replacement`; clearing that variant would silently break
    /// downstream replacement, so only the info-bearing variants
    /// (Info / IntBound / Const) take + clear, matching PyPy's clear
    /// semantics on the info-bearing branches.
    fn take_preamble_forwarded_opinfo(
        &mut self,
        source: OpRef,
    ) -> Option<crate::optimizeopt::info::OpInfo> {
        use crate::optimizeopt::info::OpInfo;
        // BoxRef-authoritative read. PyPy stores the replay op's forwarded
        // info directly on `preamble_op._forwarded`; pyre stores the same
        // state in the BoxRef slot keyed by `source`. Non-constant
        // `Forwarded::Op`/`InputArg` is a replacement chain and is excluded.
        // Const targets can still appear from legacy bridge/fixture replay
        // paths; normalize them to the OpInfo shape consumed by
        // `setinfo_from_preamble_item_option`.
        let result = {
            let fwd = self.read_forwarded(source)?;
            match &fwd {
                majit_ir::box_ref::Forwarded::Info(OpInfo::EmptyInfo(e)) => {
                    Some(OpInfo::EmptyInfo(*e))
                }
                majit_ir::box_ref::Forwarded::Info(OpInfo::Ptr(p)) => Some(OpInfo::Ptr(p.clone())),
                majit_ir::box_ref::Forwarded::Info(OpInfo::IntBound(ib)) => {
                    Some(OpInfo::IntBound(ib.clone()))
                }
                // info.py:851 FloatConstInfo planted via
                // `set_preamble_forwarded_info` (shortpreamble.py:416
                // `preamble_op.set_forwarded(info)`).
                majit_ir::box_ref::Forwarded::Info(OpInfo::FloatConstInfo(f)) => {
                    Some(OpInfo::FloatConstInfo(*f))
                }
                majit_ir::box_ref::Forwarded::Const(c) => {
                    // optimizer.py:329-338 `getinfo` parity for the Const
                    // terminal — Refs surface as `ConstPtrInfo`, Floats as
                    // `FloatConstInfo`, Ints as `IntBound::from_constant`.
                    match *c {
                        majit_ir::Const::Ref(gcref) => Some(OpInfo::ptr(
                            crate::optimizeopt::info::PtrInfo::Constant(gcref),
                        )),
                        majit_ir::Const::Float(f) => Some(OpInfo::FloatConstInfo(
                            crate::optimizeopt::info::FloatConstInfo::new(f),
                        )),
                        majit_ir::Const::Int(i) => Some(OpInfo::int_bound(
                            crate::optimizeopt::intutils::IntBound::from_constant(i),
                        )),
                    }
                }
                _ => None,
            }
        };
        if result.is_some() {
            // shortpreamble.py:401 preamble_op.set_forwarded(None) —
            // write directly to the canonical host so we don't
            // re-fetch the BoxRef wrapper.
            self.clear_forwarded(source);
        }
        result
    }

    /// unroll.py:53-98: setinfo_from_preamble(op, preamble_info, exported_infos)
    /// RPython uses sequential `if` (not elif) so multiple properties accumulate.
    /// `exported_infos`: None from use_box path (shortpreamble.py:404),
    /// Some from import_state path. When None, virtual branch does NOT recurse.
    ///
    /// `preamble_info_handle` is the live `Rc<RefCell<PtrInfo>>` cell from
    /// the exporter's `_forwarded` slot (or the shortpreamble entry's
    /// `OpInfo::Ptr(rc)`). The virtual branch shares the SAME `Rc`
    /// (`unroll.py:61` `op.set_forwarded(preamble_info)`) so future
    /// mutations to virtual fields propagate through both export and
    /// import sides — RPython object identity. Non-virtual branches
    /// snapshot the inner `PtrInfo` once because they intentionally
    /// mint fresh info objects per upstream (`unroll.py:71` etc.).
    fn setinfo_from_preamble(
        &mut self,
        op: OpRef,
        preamble_info_handle: &std::rc::Rc<std::cell::RefCell<PtrInfo>>,
        exported_infos: Option<
            &majit_ir::VecMap<majit_ir::box_ref::BoxRef, crate::optimizeopt::info::OpInfo>,
        >,
    ) {
        let op = self.get_replacement_opref(op);
        // unroll.py:55: if op.get_forwarded() is not None: return
        // (covers Op redirect + Info + IntBound + Const states uniformly,
        // matching the sibling setinfo_from_preamble_item pattern below.)
        if let Some(b) = self.get_box_replacement_operand_opt(op) {
            if self.has_forwarding(&b) {
                return;
            }
        }
        // unroll.py:57: if op.is_constant(): return
        if self
            .get_box_replacement_operand_opt(op)
            .and_then(|cb| cb.const_value())
            .is_some()
        {
            return;
        }
        // BoxRef shim for `set_ptr_info` / `make_nonnull` calls below.
        // RPython `unroll.py:54` `op = get_box_replacement(op)` followed
        // by `op.set_forwarded(...)` writes unconditionally; `op` was
        // chain-resolved and checked non-forwarded / non-constant above, so
        // `materialize_box_at` returns its canonical `_forwarded` host
        // (minting one only for an unbound preamble/test slot).
        let op_box = self.materialize_operand_at(op);

        // unroll.py:60-64: virtual — set_forwarded + recurse, then return.
        // Identity-preserving install: clone the `Rc` (not the inner
        // `PtrInfo`) so that the exporter, the importer, and the recursive
        // virtual fields all observe the same `RefCell<PtrInfo>` cell,
        // matching PyPy `op.set_forwarded(preamble_info)` object sharing.
        let is_virtual = preamble_info_handle.borrow().is_virtual();
        if is_virtual {
            let resolved = op_box.get_box_replacement(false);
            if !resolved.is_constant() {
                resolved.set_forwarded_info(crate::optimizeopt::info::OpInfo::Ptr(
                    std::rc::Rc::clone(preamble_info_handle),
                ));
            }
            if let Some(infos) = exported_infos {
                // unroll.py:61-62: setinfo_from_preamble_list(
                //     preamble_info.all_items(), exported_infos).
                // Read field BOXES via `all_items()` — the SAME accessor the
                // exporter (`_expand_infos_from_virtual`) walks on this shared
                // `PtrInfo` cell, so the box-identity (`Rc::ptr_eq`) lookup hits.
                // Using `all_items()` (not a per-variant arm) keeps export/import
                // symmetric: `VirtualRawBuffer`/`VirtualArrayStruct` return `[]`
                // here exactly as on the export side (RPython
                // `RawBufferPtrInfo.all_items()` is also empty), so the import
                // never iterates raw-buffer slots and never calls
                // `clear_forwarded` on a `from_opref`-minted unbound box.
                let items: Vec<majit_ir::box_ref::BoxRef> = preamble_info_handle
                    .borrow()
                    .all_items()
                    .iter()
                    .map(|(_, e)| e.as_seen_box())
                    .collect();
                self.setinfo_from_preamble_list(&items, infos);
            }
            return;
        }

        // Snapshot the non-virtual PtrInfo once. Non-virtual paths
        // mint fresh info objects on each install (`unroll.py:71` etc.),
        // so identity sharing is not required (and matches PyPy by
        // intentionally not sharing).
        let preamble_info_owned = preamble_info_handle.borrow().clone();
        let preamble_info: &PtrInfo = &preamble_info_owned;

        // unroll.py:65-68: constant — return early
        if let PtrInfo::Constant(gcref) = preamble_info {
            let b = self.materialize_operand_at(op);
            self.make_constant_box(&b, Value::Ref(*gcref));
            return;
        }

        // --- Sequential checks (RPython: NOT elif, all accumulate) ---

        // unroll.py:69-74: Struct/Instance with descr → set_forwarded
        if preamble_info.get_descr().is_some() {
            if let PtrInfo::Struct(sinfo) = preamble_info {
                self.set_ptr_info(&op_box, PtrInfo::struct_ptr(sinfo.descr.clone()));
            }
            if let PtrInfo::Instance(iinfo) = preamble_info {
                self.set_ptr_info(&op_box, PtrInfo::instance(iinfo.descr.clone(), None));
            }
        }

        // unroll.py:75-77: known_class → make_constant_class(op, class, False)
        if let Some(cls) = preamble_info.get_known_class(self.cpu.as_ref()) {
            crate::optimizeopt::optimizer::Optimizer::make_constant_class(
                self, &op_box, cls, false, // update_last_guard=False (unroll.py:77)
            );
        }

        // unroll.py:79-84: ArrayPtrInfo → set_forwarded(ArrayPtrInfo(descr, lenbound))
        if let PtrInfo::Array(ainfo) = preamble_info {
            self.set_ptr_info(
                &op_box,
                PtrInfo::array(ainfo.descr.clone(), ainfo.lenbound.clone()),
            );
        }

        // unroll.py:85-89: StrPtrInfo — clone lenbound
        if let PtrInfo::Str(sinfo) = preamble_info {
            let mut new_info = crate::optimizeopt::info::StrPtrInfo {
                lenbound: sinfo.lenbound.clone(),
                lgtop: None,
                mode: sinfo.mode,
                length: -1,
                // unroll.py:86: StrPtrInfo(preamble_info.mode) — always
                // rebuild a plain non-virtual StrPtrInfo; never carry
                // the previous iteration's virtual variant across.
                variant: crate::optimizeopt::info::VStringVariant::Ptr,
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            };
            if new_info.lenbound.is_none() {
                new_info.lenbound = Some(crate::optimizeopt::intutils::IntBound::nonnegative());
            }
            self.set_ptr_info(&op_box, PtrInfo::Str(new_info));
            return;
        }

        // unroll.py:91-92: is_nonnull → make_nonnull
        if preamble_info.is_nonnull() {
            self.make_nonnull(&op_box);
        }
    }

    /// unroll.py:41-51 setinfo_from_preamble_list(lst, infos):
    ///
    /// ```python
    /// def setinfo_from_preamble_list(self, lst, infos):
    ///     for item in lst:
    ///         if item is None:
    ///             continue
    ///         i = infos.get(item, None)
    ///         if i is not None:
    ///             self.setinfo_from_preamble(item, i, infos)
    ///         else:
    ///             item.set_forwarded(None)
    ///             # let's not inherit stuff we don't
    ///             # know anything about
    /// ```
    ///
    /// Every `infos.get(item) is not None` branch funnels through
    /// `setinfo_from_preamble`, which starts with the early-return checks
    /// at unroll.py:54-58 (`get_box_replacement` + `get_forwarded` +
    /// `is_constant`). A shortcut that applies IntBound / FloatConst /
    /// Constant without those checks overwrites already-forwarded boxes.
    /// `setinfo_from_preamble_item` below is the shared dispatcher: it
    /// does the checks once and then routes to the variant-specific
    /// logic, so this method becomes the literal unroll.py loop body.
    fn setinfo_from_preamble_list(
        &mut self,
        items: &[majit_ir::box_ref::BoxRef],
        exported_infos: &majit_ir::VecMap<
            majit_ir::box_ref::BoxRef,
            crate::optimizeopt::info::OpInfo,
        >,
    ) {
        for item in items {
            // unroll.py:42-43: if item is None: continue
            if item.to_opref().is_none() {
                continue;
            }
            // unroll.py:45-46: i = infos.get(item, None) — keyed by the field
            // box. `item` is read from the SAME shared virtual `PtrInfo` that
            // the exporter walked (`_expand_infos_from_virtual`), so the
            // box-identity (`Rc::ptr_eq`) lookup hits exactly when RPython's
            // box-keyed dict does.
            match exported_infos.get(item).cloned() {
                Some(info) => {
                    // unroll.py:47: self.setinfo_from_preamble(item, i, infos)
                    self.setinfo_from_preamble_item(item.to_opref(), &info, exported_infos);
                }
                None => {
                    // unroll.py:49: item.set_forwarded(None)
                    // "let's not inherit stuff we don't know anything about".
                    // Clears `item`'s OWN forwarded slot directly — `item` is
                    // the raw field box (RPython passes the unresolved
                    // `all_items()` box here). `clear_forwarded` is a no-op for
                    // a Const box (Const has no `_forwarded`), matching RPython
                    // where `Const.set_forwarded` raises.
                    item.clear_forwarded();
                }
            }
        }
    }

    /// unroll.py:53-98 `setinfo_from_preamble(op, preamble_info, exported_infos)`.
    ///
    /// Shared dispatcher covering the `isinstance(preamble_info, ...)` chain
    /// at unroll.py:59, 93, 97 — used by both `setinfo_from_preamble_list`
    /// (mod.rs recursive virtual field walker) and `OptUnroll::import_state`
    /// (unroll.rs top-level import). Centralising the dispatch avoids
    /// diverging shortcuts that skip the early-return checks.
    fn setinfo_from_preamble_item(
        &mut self,
        op: OpRef,
        preamble_info: &crate::optimizeopt::info::OpInfo,
        exported_infos: &majit_ir::VecMap<
            majit_ir::box_ref::BoxRef,
            crate::optimizeopt::info::OpInfo,
        >,
    ) {
        use crate::optimizeopt::info::OpInfo;
        // unroll.py:53-54 `op = get_box_replacement(op)`
        let target = self.get_replacement_opref(op);
        // unroll.py:55-56 `if op.get_forwarded() is not None: return`
        if let Some(b) = self.get_box_replacement_operand_opt(op) {
            if self.has_forwarding(&b) {
                return;
            }
        }
        // unroll.py:57-58 `if op.is_constant(): return`
        if self
            .get_box_replacement_operand_opt(target)
            .and_then(|cb| cb.const_value())
            .is_some()
        {
            return;
        }
        match preamble_info {
            // unroll.py:65-68 ConstPtrInfo: set_forwarded(preamble_info.getconst())
            // unroll.py:59-92 general PtrInfo dispatch. The `Ptr` arm now
            // carries the `Rc<RefCell<PtrInfo>>` handle; borrow once to
            // dispatch on the inner variant.
            OpInfo::Ptr(rc) => {
                let const_gcref = match &*rc.borrow() {
                    crate::optimizeopt::info::PtrInfo::Constant(gcref) => Some(*gcref),
                    _ => None,
                };
                if let Some(gcref) = const_gcref {
                    let b = self.materialize_operand_at(target);
                    self.make_constant_box(&b, Value::Ref(gcref));
                } else {
                    // Pass the Rc handle so the virtual branch can
                    // preserve `_forwarded` object identity per
                    // unroll.py:61.
                    self.setinfo_from_preamble(target, rc, Some(exported_infos));
                }
            }
            // unroll.py:93-96 IntBound with widen(): intersect unconditionally.
            OpInfo::IntBound(bound) => {
                let widened = bound.borrow().widen();
                let target_box = self.materialize_operand_at(target);
                self.with_intbound_mut(&target_box, |bm| {
                    let _ = bm.intersect(&widened);
                });
            }
            // unroll.py:97-98 FloatConstInfo: op.set_forwarded(preamble_info._const)
            OpInfo::FloatConstInfo(f) => {
                let b = self.materialize_operand_at(target);
                self.make_constant_box(&b, Value::Float(f.getconst()));
            }
            // unroll.py:53-98 has no dispatch arm for "no info" — the
            // caller never stores an `Unknown` entry in `exported_infos`
            // (see `collect_exported_info`'s `None` return at
            // unroll.rs:2889 mirroring unroll.py:440 `if info:`).
            OpInfo::Unknown | OpInfo::EmptyInfo(_) => unreachable!(
                "exported_infos must never contain OpInfo::Unknown/EmptyInfo; \
                 the absent-entry branch (clear_forwarded) handles that case"
            ),
        }
    }

    fn setinfo_from_preamble_item_option(
        &mut self,
        op: OpRef,
        preamble_info: &crate::optimizeopt::info::OpInfo,
        exported_infos: Option<
            &majit_ir::VecMap<majit_ir::box_ref::BoxRef, crate::optimizeopt::info::OpInfo>,
        >,
    ) {
        use crate::optimizeopt::info::OpInfo;
        let target = self.get_replacement_opref(op);
        if self
            .get_box_replacement_operand_opt(target)
            .and_then(|cb| cb.const_value())
            .is_some()
        {
            return;
        }
        if let Some(b) = self.get_box_replacement_operand_opt(op) {
            if self.has_forwarding(&b) {
                return;
            }
        }
        match preamble_info {
            OpInfo::Ptr(rc) => {
                // Pass the Rc handle (unroll.py:61 identity preservation).
                self.setinfo_from_preamble(target, rc, exported_infos);
            }
            OpInfo::IntBound(bound) => {
                let widened = bound.borrow().widen();
                let target_box = self.materialize_operand_at(target);
                self.with_intbound_mut(&target_box, |bm| {
                    let _ = bm.intersect(&widened);
                });
            }
            OpInfo::FloatConstInfo(f) => {
                let b = self.materialize_operand_at(target);
                self.make_constant_box(&b, Value::Float(f.getconst()));
            }
            OpInfo::Unknown | OpInfo::EmptyInfo(_) => {}
        }
    }

    /// `optimizer.py:354` `preamble_op = self.optunroll.potential_extra_ops.pop(op)`.
    pub fn take_potential_extra_op(
        &mut self,
        result: OpRef,
    ) -> Option<crate::optimizeopt::info::PreambleOp> {
        let idx = self
            .potential_extra_ops
            .iter()
            .position(|(k, _)| *k == result)?;
        Some(self.potential_extra_ops.swap_remove(idx).1)
    }

    /// `unroll.py:37` `self.optunroll.potential_extra_ops[op] = preamble_op` —
    /// dict-assign semantics: overwrite if `key` exists, else append.
    pub fn set_potential_extra_op(
        &mut self,
        key: OpRef,
        preamble_op: crate::optimizeopt::info::PreambleOp,
    ) {
        if let Some(entry) = self.potential_extra_ops.iter_mut().find(|(k, _)| *k == key) {
            entry.1 = preamble_op;
        } else {
            self.potential_extra_ops.push((key, preamble_op));
        }
    }

    /// Dict-`in` parity for `potential_extra_ops`.
    pub fn has_potential_extra_op(&self, key: OpRef) -> bool {
        self.potential_extra_ops.iter().any(|(k, _)| *k == key)
    }

    pub fn activate_short_preamble_producer(
        &mut self,
        builder: crate::optimizeopt::shortpreamble::ExtendedShortPreambleBuilder,
    ) {
        self.active_short_preamble_producer = Some(builder);
    }

    pub fn active_short_preamble_producer_mut(
        &mut self,
    ) -> Option<&mut crate::optimizeopt::shortpreamble::ExtendedShortPreambleBuilder> {
        self.active_short_preamble_producer.as_mut()
    }

    pub fn build_active_short_preamble(
        &self,
    ) -> Option<crate::optimizeopt::shortpreamble::ShortPreamble> {
        self.active_short_preamble_producer.as_ref().map(|builder| {
            // history.py:227/268/314 — `Const{Int,Float,Ptr}.value` rides
            // inline on the OpRef; producer-side `make_constant` writes
            // inline-Const variants into `op.args` directly, so no
            // `loop_constants` snapshot is needed at this boundary.
            builder.build_short_preamble_struct()
        })
    }

    pub fn take_active_short_preamble_producer(
        &mut self,
    ) -> Option<crate::optimizeopt::shortpreamble::ExtendedShortPreambleBuilder> {
        self.active_short_preamble_producer.take()
    }

    pub fn build_imported_short_preamble(
        &self,
    ) -> Option<crate::optimizeopt::shortpreamble::ShortPreamble> {
        self.imported_short_preamble_builder
            .as_ref()
            .map(|builder| {
                // history.py:227/268/314 — `Const{Int,Float,Ptr}.value`
                // rides inline on the OpRef. `make_constant` mints
                // `Const*` variants directly into `op.args`, so the
                // cross-compile `loop_constants` snapshot is empty
                // along every production path and the builder's
                // `known_constants` set picks up Const operands intrinsically
                // via `OpRef::is_constant()`.
                builder.build_short_preamble_struct()
            })
    }

    pub fn used_imported_short_aliases(&self) -> Vec<ImportedShortAlias> {
        self.imported_short_preamble_builder
            .as_ref()
            .map(|builder| {
                builder
                    .extra_same_as()
                    .iter()
                    .map(|op| ImportedShortAlias {
                        result: op.pos.get(),
                        same_as_source: op.arg(0).to_boxref(),
                        same_as_opcode: op.opcode,
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// optimizer.py: pure_from_args1 parity.
    /// Register reverse-pure: pure(opcode, result) = arg0.
    /// Consumed by OptPure at flush time.
    pub fn register_pure_from_args1(&mut self, opcode: OpCode, result: OpRef, arg0: OpRef) {
        self.pending_pure_from_args
            .push((opcode, result, arg0, None));
    }

    /// optimizer.py: pure_from_args1 parity with explicit descr keying.
    /// Mirrors upstream `pure_from_args(opnum, [arg], result, descr=...)`
    /// — descr discriminates the pure cache slot so cross-descr
    /// collisions (e.g. ARRAYLEN_GC across distinct array descrs at
    /// virtualize.py:220) don't collapse onto the same key.
    pub fn register_pure_from_args1_with_descr(
        &mut self,
        opcode: OpCode,
        result: OpRef,
        arg0: OpRef,
        descr: majit_ir::DescrRef,
    ) {
        self.pending_pure_from_args
            .push((opcode, result, arg0, Some(descr)));
    }

    /// info.py:557 pure_from_args(ARRAYLEN_GC, [op], ConstInt(len))
    pub fn pure_from_args_arraylen(&mut self, array_ref: OpRef, length: i64) {
        let len_ref = self.emit_constant_int(length);
        self.register_pure_from_args1(OpCode::ArraylenGc, array_ref, len_ref);
    }

    /// optimizer.py: pure_from_args2 parity.
    /// Register binary reverse-pure: pure(opcode, arg0, arg1) = result.
    /// Consumed by OptPure at flush time.
    pub fn register_pure_from_args2(
        &mut self,
        opcode: OpCode,
        result: OpRef,
        arg0: OpRef,
        arg1: OpRef,
    ) {
        self.pending_pure_from_args2
            .push((opcode, arg0, arg1, result));
    }

    /// `optimizer.py:390-401 make_equal_to(op, newop)` (line-by-line port):
    ///
    /// ```python
    /// def make_equal_to(self, op, newop):
    ///     op = get_box_replacement(op)
    ///     if op is newop: return
    ///     opinfo = op.get_forwarded()
    ///     op.set_forwarded(newop)
    ///     if opinfo is not None and not newop.is_constant():
    ///         newop.set_forwarded(opinfo)
    /// ```
    pub fn make_equal_to(&mut self, op: &Operand, newop: &Operand) {
        // optimizer.py:381 Const.set_forwarded asserts; pyre no-ops the
        // chain head when `op` is itself a Const so callers can fold const
        // sources without an explicit guard.
        if op.is_constant() {
            return;
        }
        // optimizer.py:391 op = get_box_replacement(op)
        let op = op.get_box_replacement(false);
        // optimizer.py:387 Box.type invariant: cross-type forwards would
        // silently retype the chain head. Always-on (not `debug_assert_eq!`)
        // for parity with the Const-invariant `assert!`s in `set_forwarded_*`;
        // asserted on the already-chain-walked `op` so it costs no extra walk.
        assert_eq!(
            op.type_(),
            newop.type_(),
            "make_equal_to: cross-type forward (Box.type invariant)",
        );
        // optimizer.py:392 if op is newop: return
        if &op == newop {
            return;
        }
        if op.is_constant() {
            return;
        }
        // optimizer.py:393 opinfo = op.get_forwarded()
        use crate::optimizeopt::info::OpInfo;
        let info_to_transfer: Option<OpInfo> = match &op.get_forwarded() {
            majit_ir::box_ref::Forwarded::Info(
                opinfo @ (OpInfo::Ptr(_) | OpInfo::IntBound(_) | OpInfo::FloatConstInfo(_)),
            ) => Some(opinfo.clone()),
            _ => None,
        };
        // optimizer.py:394 op.set_forwarded(newop). RPython's `newop` is
        // always a real box object. A position-only unbound BoxRef — the
        // flat-OpRef artifact where a Phase-1 producer was not carried into
        // this rebuilt Phase-2 OptContext (`find_producer_op` miss, the
        // cross-phase resolution gap; e.g. `import_state`'s next-iteration
        // target, a heap/virtual cache entry) — is materialized to its
        // canonical `SameAs*` stand-in (`resoperation.py:233-248` "the box
        // always exists"). The forward then targets a bound `Op`/`InputArg`
        // and never a position-only unbound-box redirect. An unbound
        // `newop` has no producer at its position, so it cannot alias the
        // bound chain head `op` (no self-cycle).
        let materialized_newop;
        let newop: &Operand = if newop.is_constant()
            || newop.bound_op().is_some()
            || newop.bound_inputarg().is_some()
        {
            newop
        } else {
            materialized_newop = self.materialize_operand_at(newop.to_opref());
            &materialized_newop
        };
        if newop.is_constant() {
            let value = newop
                .const_value()
                .expect("is_constant() implies const_value() Some");
            op.set_forwarded_const(majit_ir::Const::from_value(value));
        } else if let Some(target_op) = newop.bound_op() {
            // Op-target chain step: route through Forwarded::Op(Weak<Op>)
            // so the chain refers to the canonical Rc<Op> (PyPy
            // resoperation.py:240 set_forwarded(forwarded_to) where
            // forwarded_to is an AbstractResOp), retiring the
            // BoxKind::ResOp-as-chain-target carrier.
            //
            // `optimizer.py:392 if op is newop: return` — PyPy's
            // identity check uses Python `is`; after `bind_op`, two
            // separate `Rc<Box>` wrappers can share the same canonical
            // `OpRc`, so `&op == newop` (which compares the `Rc<Box>`)
            // misses that case and falls through to `set_forwarded_op`,
            // tripping `set_forwarded_op`'s self-cycle assert. Honour
            // the upstream `is` semantics by comparing the bound `Op`
            // identities first.
            if op
                .bound_op()
                .is_some_and(|o| std::rc::Rc::ptr_eq(&o, &target_op))
            {
                return;
            }
            op.set_forwarded_op(&target_op);
        } else if let Some(target_ia) = newop.bound_inputarg() {
            // InputArg-target chain step (compile.py:478, unroll.py:497).
            // Same `optimizer.py:392` idempotent gate as the
            // `bound_op` arm above, against the bound `InputArg`
            // identities.
            if op
                .bound_inputarg()
                .is_some_and(|i| std::rc::Rc::ptr_eq(&i, &target_ia))
            {
                return;
            }
            op.set_forwarded_inputarg(&target_ia);
        } else {
            // Unreachable: `newop` was normalized above to a constant or a
            // bound `Op`/`InputArg`, so the dispatch always resolves through
            // `set_forwarded_const` / `set_forwarded_op` / `set_forwarded_inputarg`.
            // The orphan unbound-box redirect that the deleted `Forwarded::Box`
            // variant once encoded no longer exists.
            unreachable!(
                "make_equal_to: newop must be constant or bound after \
                 materialize_box_at normalization (orphan unbound-box \
                 redirect retired)"
            );
        }
        // optimizer.py:395-396
        //   if opinfo is not None and not newop.is_constant():
        //       newop.set_forwarded(opinfo)
        if let Some(opinfo) = info_to_transfer
            && !newop.is_constant()
        {
            newop.set_forwarded_info(opinfo);
        }
    }

    /// info.py:111-118 `mark_last_guard(optimizer)` parity (line-by-line port).
    ///
    /// ```python
    /// def mark_last_guard(self, optimizer):
    ///     if (optimizer.getlastop() is None or
    ///             not optimizer.getlastop().is_guard()):
    ///         return
    ///     self.last_guard_pos = len(optimizer._newoperations) - 1
    ///     assert self.get_last_guard(optimizer).is_guard()
    /// ```
    ///
    /// Upstream defines this as a method ON `PtrInfo`
    /// (`opinfo.mark_last_guard(self.optimizer)` per optimizer.py:151);
    /// pyre keeps it at the `OptContext` level so the chain walk and
    /// `ptr_info_mut` interior-mutability stay together. Semantics match
    /// the upstream method: no-op when the last emitted op is not a guard,
    /// otherwise stamps `last_guard_pos = len(_newoperations) - 1` on the
    /// terminal box's PtrInfo.
    pub fn mark_last_guard(&self, op: &Operand) {
        // info.py:112-116: optimizer.getlastop().is_guard() check
        let pos = match self.new_operations.last() {
            Some(o) if o.opcode.is_guard() => (self.new_operations.len() - 1) as i32,
            _ => return,
        };
        // info.py:117: self.last_guard_pos = pos
        // `_forwarded` PtrInfo lives on the terminal of the chain.
        let resolved = op.get_box_replacement(false);
        if let Some(mut info) = resolved.ptr_info_mut() {
            info.set_last_guard_pos(pos);
        }
    }

    /// info.py:100-103 `get_last_guard(optimizer)` parity (line-by-line port).
    ///
    /// ```python
    /// def get_last_guard(self, optimizer):
    ///     if self.last_guard_pos == -1:
    ///         return None
    ///     return optimizer._newoperations[self.last_guard_pos]
    /// ```
    ///
    /// Upstream is a `PtrInfo` method; pyre keeps it at the `OptContext`
    /// level so the chain walk and `ptr_info()` read stay together with
    /// the `_newoperations` index. Returns the guard `Op` at the PtrInfo's
    /// stored `last_guard_pos`, or `None` when the slot is `-1` (no guard
    /// recorded) or the BoxRef has no PtrInfo.
    pub fn get_last_guard(&self, op: &Operand) -> Option<&Op> {
        // info.py:100-103: read last_guard_pos from terminal PtrInfo.
        let resolved = op.get_box_replacement(false);
        let pos = resolved.ptr_info().and_then(|p| p.get_last_guard_pos())?;
        self.new_operations.get(pos).map(|rc| rc.as_ref())
    }

    /// resoperation.py:57-68 get_box_replacement: follow the forwarding
    /// chain (op._forwarded) until we reach a terminal. RPython: walks
    /// op → op._forwarded → ... until None or Info instance.
    ///
    /// RPython invariant: get_box_replacement NEVER returns None.
    /// `_forwarded = None` means "no forwarding" (terminal), NOT
    /// "forwarded to None".
    ///
    /// NEVER consults mapping dicts — RPython's get_box_replacement only
    /// follows the _forwarded chain on the box itself.
    ///
    /// `_forwarded` is a single slot per `BoxRef` (matching RPython's
    /// single Python slot per box). The walker advances through
    /// `Forwarded::Op`/`Forwarded::InputArg` and terminates at `None` /
    /// `Forwarded::Info(_)` / a Const target's reconstructed
    /// `OpRef::const_int/float/ptr`.
    fn get_box_replacement_impl(&self, opref: OpRef, not_const: bool) -> OpRef {
        if opref.is_constant() || opref.is_none() {
            return opref;
        }
        let Some(start) = self.resolve_to_boxref(opref) else {
            return opref;
        };
        // resoperation.py:57-68: walk box._forwarded on the box itself.
        let terminal = start.get_box_replacement(not_const);
        // When the walker did not advance — chain root has Forwarded::None,
        // Forwarded::Info(_), or (not_const=true and the immediate target
        // is Const) — return the source OpRef variant unchanged. The
        // original walker terminated before reading position()/type_(),
        // so callers expect the OpRef shape they passed in.
        if start == terminal {
            return opref;
        }
        // Const targets reconstruct their `source_opref` from the inline
        // value (history.py:227/268/314), so `box_to_opref` reconstruction
        // is direct — every `BoxRef::new_const(value)` reconstructs an
        // inline-Const source_opref via `source_opref()`'s value arm.
        self.box_to_opref(&terminal, opref)
    }

    /// Convert a chain-walk terminal `BoxRef` back into an `OpRef`. This
    /// is the OpRef-side glue around `BoxRef::get_box_replacement`; PyPy
    /// callers hold the box directly and skip this step.
    ///
    /// `BoxKind::Const` carries its `source_opref` (the OpRef the Box was
    /// minted from), so reconstruction is direct — mirrors RPython where
    /// the Box object IS the reference.
    fn box_to_opref(&self, terminal: &majit_ir::box_ref::BoxRef, source: OpRef) -> OpRef {
        if let Some(src) = terminal.source_opref() {
            return src;
        }
        if let Some(pos) = terminal.position() {
            let tp = terminal.type_();
            // `Type::Void` targets are lazy-allocated phantom placeholders
            // (`materialize_box_at` fallback for OpRef variants with no `ty()`); the
            // placeholder carries no type information, so preserve the source variant via `with_raw`
            // instead of promoting to `void_op` / `input_arg_typed(_, Void)`.
            if matches!(tp, majit_ir::Type::Void) {
                return source.with_raw(pos);
            }
            if terminal.is_inputarg() {
                return OpRef::input_arg_typed(pos, tp);
            }
            if terminal.is_resop() {
                return OpRef::op_typed(pos, tp);
            }
        }
        source
    }

    /// resoperation.py:57-68 `get_box_replacement` — walk `op._forwarded`
    /// to the terminal box and return it. PyPy returns the box object
    /// directly; pyre returns the `BoxRef` view. `OpRef`-keyed callers that
    /// still need an integer handle bridge back with `.to_opref()` (the
    /// remaining `Op.args` / fail-args boundary).
    ///
    /// The resolver is total (resoperation.py:57-68): a box with no
    /// forwarding is its own replacement. bind-at-alloc (#194) binds every
    /// value-bearing position's canonical host at allocation
    /// (`reserve_virtual_box` / emit-time `bind_op` / `bind_input_resops`),
    /// so the position-only `BoxRef::from_opref` fallback is unreachable in
    /// practice (probe-verified 0 fires over the corpus); a fire signals a
    /// bookkeeping bug. The env-gated S9 probe stays for triage, but
    /// production returns the identity box rather than aborting the trace —
    /// matching upstream totality and the `a-producerless` arm of
    /// `classify_s9_fallback` ("an unforwarded box returns itself").
    pub fn get_box_replacement(&self, opref: OpRef) -> majit_ir::box_ref::BoxRef {
        // The sentinel `OpRef::none()` (absent operand: empty fail_arg slot,
        // unset lazy setfield) is not a value-bearing position — its total
        // resolution is the `none()` box itself, which `from_opref` already
        // round-trips below. Resolve it here so the "box must exist" assert
        // and the `s9_probe`/`from_opref` fallback apply only to value-bearing
        // positions, matching the explicit NONE handling in `resolve_to_boxref`
        // / `materialize_box_at` / `clear_forwarded`.
        if opref.is_none() {
            return majit_ir::box_ref::BoxRef::none();
        }
        if let Some(b) = self.get_box_replacement_box(opref) {
            return b;
        }
        // S9 probe (env-gated, no-op unless `PYRE_S9_PROBE` is set): classify
        // any fire for triage. Then return the identity box (resoperation.py:57-68).
        self.s9_probe_fire(opref);
        majit_ir::box_ref::BoxRef::from_opref(opref)
    }

    /// [`Operand`]-yielding total mirror of [`get_box_replacement`](Self::get_box_replacement):
    /// resolve a position's `_forwarded` terminal directly as an `Operand`
    /// (canonical `Op` / `InputArg` host, or inline-`Const`) without minting a
    /// `BoxRef`. Walks the chain via [`resolve_to_operand`](Self::resolve_to_operand)
    /// + [`Operand::get_box_replacement`].
    ///
    /// The fallback arm differs from the `BoxRef` sibling by design: a position
    /// that resolves to neither a producer `Op`, an `inputarg_refs` slot, nor a
    /// Const has no `Operand` representation (E3 removed the position-only
    /// `Operand::Box` variant), so `Operand::from_opref` PANICS there — the
    /// armed hazard-5 tripwire. Every value-bearing op-arg position has a
    /// findable producer, so the panic arm is unreachable for arg-resolution
    /// callers (a fire signals an unbound operand fed as an op arg).
    pub(crate) fn get_box_replacement_operand(&self, opref: OpRef) -> Operand {
        if opref.is_none() {
            return Operand::None;
        }
        if let Some(start) = self.resolve_to_operand(opref) {
            return start.get_box_replacement(false);
        }
        self.s9_probe_fire(opref);
        Operand::from_opref(opref)
    }

    /// Operand-yielding sibling of [`materialize_box_at`](Self::materialize_box_at):
    /// the canonical bound host as an [`Operand`] (`Op` / `InputArg`), or an
    /// `Operand::Const` for a const-namespace OpRef. `materialize_box_at` never
    /// returns a position-only box, so the lowering is panic-free.
    pub(crate) fn materialize_operand_at(&mut self, opref: OpRef) -> Operand {
        // `materialize_box_at` always yields a box bound to a canonical host
        // (a producing `Op`, an `InputArg`, or a `Const`). Read that host
        // directly to build the `Operand` natively — the same lowering
        // `Operand::from_boxref` performs for a bound box, but without the
        // wrapper round-trip and without leaning on the `box_cache` memo for
        // identity (the `Rc<Op>` / `Rc<InputArg>` itself is the identity).
        let b = self.materialize_box_at(opref);
        if let Some(op) = b.bound_op() {
            Operand::from_bound_op(&op)
        } else if let Some(ia) = b.bound_inputarg() {
            Operand::from_bound_inputarg(&ia)
        } else {
            Operand::const_from_value(
                b.const_value()
                    .expect("materialize_box_at yields a bound or const box"),
            )
        }
    }

    /// Migration bridge: lower a (possibly position-only) box to its canonical
    /// bound [`Operand`] without the `from_boxref` position-only panic. A box
    /// already carrying a bound producer or a const wraps directly; a bare
    /// position (a `from_opref` / `get_box_replacement` fallback mint with no
    /// findable producer) materializes to its canonical `SameAs*` stand-in
    /// (`resoperation.py:233-248` "the box always exists"). This mirrors the
    /// position-only normalization [`make_equal_to`] applies to its `newop`
    /// target. Use wherever a `resolve_*` / `get_box_replacement` result (whose
    /// box can be position-only) feeds an `Operand` reader.
    pub(crate) fn operand_of_box(&mut self, b: &majit_ir::box_ref::BoxRef) -> Operand {
        if b.bound_op().is_some() || b.bound_inputarg().is_some() || b.is_constant() {
            Operand::from_boxref(b)
        } else {
            self.materialize_operand_at(b.to_opref())
        }
    }

    /// S9 probe classifier for a `from_opref` fallback fire (see
    /// [`OptContext::get_box_replacement`]). `&self`-only, no mutation.
    ///
    /// Precondition: `find_producer_op` already missed (full type-tagged
    /// `OpRef` match across `new_operations` / `phase1_emit_ops` / `resop_refs`
    /// / `input_ops`) and `resolve_to_boxref` returned `None`; `opref` is
    /// non-`none`, non-`Const`.
    ///
    /// - `b-inputarg`: an InputArg position whose `inputarg_refs` slot is unbound.
    /// - `b-typetag`: a producer Op exists at the same RAW position under a
    ///   different type tag (the #97 type-tag divergence signal).
    /// - `b-rekey-*`: the OpRef is named by a carried Phase-1 registry
    ///   (`inputargs` / `imported_label_args` / `imported_virtual_args`), so a
    ///   producer exists in Phase 1 but is not in a `find_producer_op` store.
    /// - `b-shortpure`: the OpRef is the `result` of a carried
    ///   `imported_short_pure_ops` entry — a short-preamble producer survives
    ///   but `find_producer_op` does not consult that table.
    /// - `a-producerless`: no producer Op for this position is reachable in any
    ///   OptContext-local registry — correct-as-is (an unforwarded box returns
    ///   itself, `resoperation.py:57-68`).
    fn classify_s9_fallback(&self, opref: OpRef) -> &'static str {
        match opref {
            OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_) => {
                return "b-inputarg";
            }
            _ => {}
        }
        let raw = opref.raw();
        let raw_hit = self
            .new_operations
            .iter()
            .chain(self.phase1_emit_ops.iter())
            .chain(self.input_ops.iter())
            .any(|op| op.pos.get().raw() == raw);
        if raw_hit {
            return "b-typetag";
        }
        if self.inputargs.iter().any(|o| *o == opref) {
            return "b-rekey-inputargs";
        }
        if let Some(label_args) = self.imported_label_args.as_ref() {
            if label_args.iter().any(|o| *o == opref) {
                return "b-rekey-label";
            }
        }
        if let Some((_, vargs)) = self.imported_virtual_args.as_ref() {
            if vargs.iter().any(|o| *o == opref) {
                return "b-rekey-vargs";
            }
        }
        if self
            .imported_short_pure_ops
            .iter()
            .any(|sp| sp.result == opref)
        {
            return "b-shortpure";
        }
        "a-producerless"
    }

    /// S9 probe sink (env-gated). No-op unless `PYRE_S9_PROBE` is set; appends
    /// one classified line per `from_opref` fallback fire to
    /// `/tmp/pyre_s9_probe.log`. Process-global file sink because the caller is
    /// `&self` (no context-local counter). When the flag is unset, only a
    /// cached bool is read and the sink is never opened.
    fn s9_probe_fire(&self, opref: OpRef) {
        use std::sync::{LazyLock, Mutex};
        static ENABLED: LazyLock<bool> =
            LazyLock::new(|| std::env::var_os("PYRE_S9_PROBE").is_some());
        if !*ENABLED {
            return;
        }
        static SINK: LazyLock<Mutex<std::fs::File>> = LazyLock::new(|| {
            Mutex::new(
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open("/tmp/pyre_s9_probe.log")
                    .expect("open /tmp/pyre_s9_probe.log"),
            )
        });
        let cat = self.classify_s9_fallback(opref);
        // `PYRE_S9_PROBE=bt` additionally captures the caller backtrace per
        // fire, to attribute fallback fires to their `get_box_replacement`
        // call sites.
        static BT: LazyLock<bool> = LazyLock::new(|| {
            std::env::var("PYRE_S9_PROBE")
                .map(|v| v == "bt")
                .unwrap_or(false)
        });
        if let Ok(mut f) = SINK.lock() {
            use std::io::Write;
            let _ = writeln!(f, "cat={} opref={:?} raw={}", cat, opref, opref.raw());
            if *BT {
                let _ = writeln!(f, "{}", std::backtrace::Backtrace::force_capture());
            }
        }
    }

    /// `BoxRef`-addressed operand resolution. Callers holding the
    /// operand Box (`op.arg(i)`) resolve it here instead of collapsing to
    /// an `OpRef` and re-resolving.
    ///
    /// resoperation.py:57-68: a bound operand carries its producer op (or
    /// is a Const) and walks its own `_forwarded` chain directly — the Box
    /// object IS the reference. An unbound operand (no producer handle:
    /// short-preamble replay / test baseline position-only box, or an
    /// InputArg) has no producer chain to walk from the Box alone, so it
    /// resolves through the `OpRef` position store as before.
    ///
    /// A bound InputArg is deliberately NOT walked box-native here: the
    /// canonical InputArg identity lives in `inputarg_refs` (the position
    /// store), and a consumer's operand box can wrap a non-canonical
    /// `InputArgRc` duplicate whose `inputarg.forwarded` is stale relative
    /// to the registered host's. Resolving such a box through its own chain
    /// diverges from the canonical position resolution (the divergence
    /// witness fires on `InputArgInt(0)` in the jitdriver pipeline). Routing
    /// InputArgs through the `OpRef` store keeps resolution canonical until
    /// InputArg identity is unified (the InputArg analog of the ResOp
    /// emit-rebind keystone in `materialize_box_at` / `set_forwarded_op`).
    pub fn resolve_box_box(&self, arg: &majit_ir::box_ref::BoxRef) -> majit_ir::box_ref::BoxRef {
        self.heal_arg_to_canonical(arg);
        if arg.bound_op().is_some() || arg.is_constant() {
            let resolved = arg.get_box_replacement(false);
            // A non-canonical duplicate operand box resolves to itself
            // box-native while its position's canonical forwarding lives in
            // the `OpRef` store (see `resolve_box_box_opt`); defer to the
            // store, which returns the box unchanged when it holds no entry.
            if resolved.same_box(arg) {
                return self.get_box_replacement(arg.to_opref());
            }
            // The box-native walk forwarded: that chain is canonical and the
            // `OpRef` store must agree (migration tripwire).
            #[cfg(debug_assertions)]
            {
                let legacy = self.get_box_replacement(arg.to_opref());
                debug_assert!(
                    resolved.same_box(&legacy) || resolved.to_opref() == legacy.to_opref(),
                    "resolve_box_box: box-native walk diverged from OpRef path for {:?}",
                    arg.to_opref()
                );
            }
            resolved
        } else {
            self.get_box_replacement(arg.to_opref())
        }
    }

    /// `Option`-returning sibling of [`OptContext::resolve_box_box`], the
    /// box-native form of `get_box_replacement_box`. resoperation.py:58
    /// `get_box_replacement(op)` walks the box's `_forwarded` chain; a bound
    /// operand (or Const) walks it directly here. The `None` arm is pyre's
    /// unbound-operand adaptation (RPython has no unbound boxes): an unbound
    /// operand resolves through the `OpRef` store and surfaces `None` when the
    /// root does not resolve (sentinel / baseline) so callers can branch on it.
    /// A bound InputArg stays on the `OpRef` path for the same canonical-
    /// identity reason documented on [`OptContext::resolve_box_box`].
    pub fn resolve_box_box_opt(
        &self,
        arg: &majit_ir::box_ref::BoxRef,
    ) -> Option<majit_ir::box_ref::BoxRef> {
        self.heal_arg_to_canonical(arg);
        if arg.bound_op().is_some() || arg.is_constant() {
            let resolved = arg.get_box_replacement(false);
            // A non-canonical duplicate operand box (short-preamble replay
            // handle / position-only export box — the bound-ResOp analog of
            // the stale `InputArgRc` duplicate documented on `resolve_box_box`)
            // carries no forwarding on its own `_forwarded` chain, so the
            // box-native walk resolves it to itself even though the canonical
            // forwarding for its position lives in the `OpRef` store. When the
            // box resolves to itself the store is the canonical position
            // resolution — defer to it, as the InputArg `else` arm and the
            // "resolve positionally" short-preamble export both do.
            if resolved.same_box(arg) {
                return Some(
                    self.get_box_replacement_box(arg.to_opref())
                        .unwrap_or(resolved),
                );
            }
            // The box-native walk forwarded: that chain is canonical and the
            // `OpRef` store must agree (migration tripwire).
            #[cfg(debug_assertions)]
            {
                let legacy = self.get_box_replacement_box(arg.to_opref());
                let agrees = match &legacy {
                    Some(l) => resolved.same_box(l) || resolved.to_opref() == l.to_opref(),
                    None => false,
                };
                debug_assert!(
                    agrees,
                    "resolve_box_box_opt: box-native walk diverged from OpRef path for {:?}",
                    arg.to_opref()
                );
            }
            Some(resolved)
        } else {
            self.get_box_replacement_box(arg.to_opref())
        }
    }

    /// Operand-input sibling of [`OptContext::resolve_box_box`]: resolves an
    /// [`Operand`] (the op-arg carrier) to its `_forwarded` terminal `BoxRef`.
    /// The `BoxRef` RETURN is kept — its consumers (`get_box_replacement` /
    /// `.to_opref()` / the `&BoxRef` sinks) stay BoxRef-typed — so only the
    /// INPUT side carries `Operand` directly, letting a caller holding
    /// `op.arg(i)` drop the `.to_boxref()` bridge. The internal `to_boxref()`
    /// retires when `resolve_box_box` itself flips its parameter.
    pub fn resolve_operand_box(&self, arg: &Operand) -> majit_ir::box_ref::BoxRef {
        self.resolve_box_box(&arg.to_boxref())
    }

    /// Operand-input sibling of [`OptContext::resolve_box_box_opt`] (`None`
    /// when the operand is a Const / NONE / unresolved position).
    pub fn resolve_operand_box_opt(&self, arg: &Operand) -> Option<majit_ir::box_ref::BoxRef> {
        self.resolve_box_box_opt(&arg.to_boxref())
    }

    /// Native `Operand`-in / `Operand`-out resolver: the [`Operand`] form of
    /// [`resolve_box_box`](Self::resolve_box_box). Resolves an operand to its
    /// `_forwarded` terminal WITHOUT minting a `BoxRef` — the box-native walk
    /// (`arg.get_box_replacement`) and the `OpRef`-store fallback
    /// ([`get_box_replacement_operand`](Self::get_box_replacement_operand)) both
    /// stay on the `Operand` carrier. Mirrors `resolve_box_box`'s two arms: a
    /// bound / const operand walks its own chain and defers to the store only
    /// when it self-resolves; an unbound operand resolves positionally.
    ///
    /// The heal still keys on the bound `Op` host, so it is driven through a
    /// transient `to_boxref()` (heal re-homes onto Op/InputArg in a later slice);
    /// `BoxRef::get_box_replacement` delegates to the `Operand` walk
    /// (`box.rs`), so the native walk is byte-identical to the legacy
    /// resolve-then-rewrap, asserted below.
    pub fn resolve_operand_operand(&self, arg: &Operand) -> Operand {
        self.heal_arg_to_canonical(&arg.to_boxref());
        let native = if arg.bound_op().is_some() || arg.is_constant() {
            let resolved = arg.get_box_replacement(false);
            // Self-resolved box-native: the canonical forwarding for this
            // position lives in the `OpRef` store (see `resolve_box_box`).
            if resolved.same_box(arg) {
                self.get_box_replacement_operand(arg.to_opref())
            } else {
                resolved
            }
        } else {
            self.get_box_replacement_operand(arg.to_opref())
        };
        // Migration tripwire: the native walk must agree with the legacy
        // resolve-then-rewrap (`Const` mints a fresh `Rc` so `same_box` fails
        // by ptr but the resolved position `to_opref()` matches).
        #[cfg(debug_assertions)]
        {
            let legacy = Operand::from_boxref(&self.resolve_operand_box(arg));
            debug_assert!(
                native.same_box(&legacy) || native.to_opref() == legacy.to_opref(),
                "resolve_operand_operand: native walk diverged from legacy for {:?}",
                arg.to_opref()
            );
        }
        native
    }

    /// `Option`-returning native sibling of [`resolve_operand_operand`], the
    /// [`Operand`] form of [`resolve_box_box_opt`](Self::resolve_box_box_opt):
    /// `None` when the operand is a NONE / unresolved position so callers can
    /// supply their own unbound fallback (a sentinel arg box, a
    /// `materialize_*` mint) instead of tripping the position-only panic in the
    /// total [`get_box_replacement_operand`](Self::get_box_replacement_operand).
    pub fn resolve_operand_operand_opt(&self, arg: &Operand) -> Option<Operand> {
        self.heal_arg_to_canonical(&arg.to_boxref());
        let native = if arg.bound_op().is_some() || arg.is_constant() {
            let resolved = arg.get_box_replacement(false);
            if resolved.same_box(arg) {
                Some(
                    self.get_box_replacement_operand_opt(arg.to_opref())
                        .unwrap_or(resolved),
                )
            } else {
                Some(resolved)
            }
        } else {
            self.get_box_replacement_operand_opt(arg.to_opref())
        };
        // Migration tripwire: the native walk must agree with the legacy
        // resolve-then-rewrap on both presence and identity.
        #[cfg(debug_assertions)]
        {
            let legacy = self
                .resolve_operand_box_opt(arg)
                .map(|b| Operand::from_boxref(&b));
            let agrees = match (&native, &legacy) {
                (Some(n), Some(l)) => n.same_box(l) || n.to_opref() == l.to_opref(),
                (None, None) => true,
                _ => false,
            };
            debug_assert!(
                agrees,
                "resolve_operand_operand_opt: native walk diverged from legacy for {:?}",
                arg.to_opref()
            );
        }
        native
    }

    /// Box-canonicalization heal (#189 keystone, phase 1). When a position's
    /// canonical producer (the `find_producer_op` / OpRef-store resolution)
    /// has received a forwarding — const-fold (`make_constant_box` /
    /// `seed_constant`) or CSE (`make_equal_to`) — that did NOT route through
    /// `emit`'s `live_synthetics` catch-up (mod.rs:2673), the recorder input
    /// op the operands carry at that position stays a DISTINCT, still-
    /// unforwarded `Op`. A box-native walk from such an operand then freezes
    /// on the unforwarded input op while the OpRef path resolves the canonical
    /// (the witnessed `resolve_box_box` divergence). Link `input_op ->
    /// canonical` here, the non-emitted analogue of `emit`'s synth->op_rc link,
    /// so both paths agree.
    ///
    /// Cycle-safe: only an UNFORWARDED bound ResOp whose store canonical is a
    /// genuinely distinct `Op` is linked. Const and InputArg operands resolve
    /// canonically already and are skipped.
    ///
    /// The cycle guard keys on the bound `Op` identity, NOT the position: a
    /// position routinely carries TWO distinct `Op` objects — the recorder
    /// input op the operands bind to, and the emitted/canonical op
    /// `find_producer_op` returns (the host on which `heap`/`virtualize` set the
    /// position's PtrInfo via `set_forwarded_info`). Both share the same
    /// `to_opref()` (position), so a position-equality guard would skip exactly
    /// this same-position duplication, leaving the operand's box-native walk
    /// stranded on the info-less input op while the OpRef store path reaches the
    /// info-bearing canonical. Linking those two distinct ops is the whole point
    /// of the heal. But `get_box_replacement_box` can also re-mint a DIFFERENT
    /// `BoxRef` wrapping the SAME bound `Op` as `arg` (a store wrapper vs the
    /// memoized operand box); `same_box` (an `Rc::ptr_eq` on the boxes) misses
    /// that, so an explicit `Rc::ptr_eq` on the bound ops is needed — without it
    /// `set_forwarded_op` self-cycles (`arg.op -> arg.op`). The canonical is a
    /// `get_box_replacement_box` terminal (`Forwarded::None`/`Info`, never a
    /// `Box`), so once a genuinely distinct op is linked no chain cycle forms.
    fn heal_arg_to_canonical(&self, arg: &majit_ir::box_ref::BoxRef) {
        if arg.bound_op().is_none() {
            return;
        }
        if !matches!(arg.get_forwarded(), majit_ir::box_ref::Forwarded::None) {
            return;
        }
        let Some(canon) = self.get_box_replacement_box(arg.to_opref()) else {
            return;
        };
        if arg.same_box(&canon) {
            return;
        }
        // Skip when `canon` wraps the same bound `Op` as `arg` under a distinct
        // `BoxRef` — linking would be a one-node self-cycle.
        if let (Some(ao), Some(co)) = (arg.bound_op(), canon.bound_op()) {
            if std::rc::Rc::ptr_eq(&ao, &co) {
                return;
            }
        }
        if let Some(value) = canon.const_value() {
            arg.set_forwarded_const(majit_ir::Const::from_value(value));
        } else if let Some(canon_op) = canon.bound_op() {
            arg.set_forwarded_op(&canon_op);
        } else if let Some(canon_ia) = canon.bound_inputarg() {
            arg.set_forwarded_inputarg(&canon_ia);
        }
    }

    /// Box-native `not_const=True` sibling of
    /// [`OptContext::resolve_box_box_opt`] (resoperation.py:64-65): the
    /// `_forwarded` walk stops before stepping into a `Const` target, so a
    /// guard fail_arg keeps the runtime Box identity while resume numbering
    /// carries constants as TAGCONST.
    ///
    /// `None` when `op` is itself a Const / NONE, or (for an unbound operand)
    /// the position resolves to no canonical box; the caller keeps its
    /// element unchanged then. A bound producer walks `_forwarded` directly;
    /// an InputArg / unbound operand stays on the OpRef store path for
    /// canonical identity (see [`OptContext::resolve_box_box`]). The
    /// `debug_assertions` arm witnesses the box-native walk against that path.
    pub fn get_box_replacement_not_const_box(
        &self,
        op: &majit_ir::box_ref::BoxRef,
    ) -> Option<majit_ir::box_ref::BoxRef> {
        if op.is_constant() || op.is_none() {
            return None;
        }
        if op.bound_op().is_some() {
            let resolved = op.get_box_replacement(true);
            #[cfg(debug_assertions)]
            {
                if let Some(start) = self.resolve_to_boxref(op.to_opref()) {
                    let legacy = start.get_box_replacement(true);
                    debug_assert!(
                        resolved.same_box(&legacy) || resolved.to_opref() == legacy.to_opref(),
                        "get_box_replacement_not_const_box: box-native walk diverged \
                         from OpRef path for {:?}",
                        op.to_opref()
                    );
                }
            }
            Some(resolved)
        } else {
            let start = self.resolve_to_boxref(op.to_opref())?;
            Some(start.get_box_replacement(true))
        }
    }

    /// [`Operand`]-in / [`Operand`]-out sibling of
    /// [`get_box_replacement_not_const_box`](Self::get_box_replacement_not_const_box):
    /// resolve past const-folds (`get_box_replacement(true)`) directly on the
    /// `Operand` carrier, returning `None` for a const / NONE / unresolved
    /// operand. Lets a fail-arg consumer resolve `op.arg(i)` without the
    /// `to_boxref()` → `from_boxref()` round-trip.
    pub fn get_box_replacement_not_const_operand(&self, op: &Operand) -> Option<Operand> {
        if op.is_constant() || op.is_none() {
            return None;
        }
        if op.bound_op().is_some() {
            let resolved = op.get_box_replacement(true);
            #[cfg(debug_assertions)]
            {
                if let Some(start) = self.resolve_to_operand(op.to_opref()) {
                    let legacy = start.get_box_replacement(true);
                    debug_assert!(
                        resolved.same_box(&legacy) || resolved.to_opref() == legacy.to_opref(),
                        "get_box_replacement_not_const_operand: native walk diverged \
                         from OpRef path for {:?}",
                        op.to_opref()
                    );
                }
            }
            Some(resolved)
        } else {
            let start = self.resolve_to_operand(op.to_opref())?;
            Some(start.get_box_replacement(true))
        }
    }

    /// `OpRef` round-trip view of [`OptContext::get_box_replacement`] for
    /// the remaining flat-`OpRef` bridge callers (the
    /// `get_box_replacement(o).to_opref()` idiom — RPython callers hold the
    /// box and skip this step; the bridge retires with the operand union,
    /// #9). Identical to that idiom observationally — an unresolvable root
    /// returns `opref` unchanged (`from_opref(o).to_opref() == o`) — but
    /// without fabricating the intermediate position-only box.
    pub(crate) fn get_replacement_opref(&self, opref: OpRef) -> OpRef {
        match self.get_box_replacement_box(opref) {
            Some(b) => b.to_opref(),
            None => opref,
        }
    }

    /// `Option`-exposing sibling of [`OptContext::get_box_replacement`]:
    /// walks the `_forwarded` chain rooted at the Box for `opref` and returns
    /// the terminal `BoxRef`, or `None` when the root does not resolve.
    ///
    /// `resoperation.py:57-68 get_box_replacement(self, op)` walks
    /// `op._forwarded` until `None | AbstractInfo`, returning the terminal
    /// Box object. `get_box_replacement` above is total (an unresolvable root
    /// falls back to `BoxRef::from_opref`); this variant instead surfaces the
    /// `None` so callers that must distinguish "no bound box" — a sentinel,
    /// or a test / retrace baseline with no upstream binding — can branch on
    /// it rather than act on a position-only placeholder.
    ///
    /// `BoxRef._forwarded` (`box_ref.rs`) is the authoritative storage; both
    /// readers walk the same chain and agree by construction.
    pub fn get_box_replacement_box(&self, opref: OpRef) -> Option<majit_ir::box_ref::BoxRef> {
        // Resolve the chain root through `resolve_to_boxref`, the
        // variant-aware canonical-host resolver (producer `Op` for ResOp
        // variants, `inputarg_refs` for InputArg, inline-Const for Const),
        // rather than `box_pool.get` whose position-collapse merges a ResOp
        // and an InputArg sharing a raw slot index. Production `materialize_box_at`
        // binds every resop slot to the same producer `Op`, so reads here
        // and writes through `materialize_box_at` agree by hitting the identical
        // `Op.forwarded` / `InputArg.forwarded` host. A `None` resolve
        // (sentinel, or a position with no producer / inputarg / const)
        // leaves callers on the OpRef-returning walker fallback.
        let start = self.resolve_to_boxref(opref)?;
        Some(start.get_box_replacement(false))
    }

    /// [`Operand`]-yielding sibling of [`get_box_replacement_box`](Self::get_box_replacement_box):
    /// walk the `_forwarded` chain rooted at `opref` and return the terminal as
    /// an `Operand`, or `None` when the root does not resolve to a producer /
    /// inputarg / const. Native form of `get_box_replacement_operand`'s body
    /// without the position-only panic fallback.
    pub fn get_box_replacement_operand_opt(&self, opref: OpRef) -> Option<Operand> {
        let native = self
            .resolve_to_operand(opref)
            .map(|start| start.get_box_replacement(false));
        // Migration tripwire: the native `Operand` walk must agree with the
        // `BoxRef`-form `get_box_replacement_box` on both presence and identity
        // (`Const` mints a fresh `Rc` so `same_box` fails by ptr but the
        // resolved position `to_opref()` matches). Covers direct callers that
        // do not flow through the `resolve_operand_operand_opt` tripwire.
        #[cfg(debug_assertions)]
        {
            let legacy = self
                .get_box_replacement_box(opref)
                .map(|b| Operand::from_boxref(&b));
            let agrees = match (&native, &legacy) {
                (Some(n), Some(l)) => n.same_box(l) || n.to_opref() == l.to_opref(),
                (None, None) => true,
                _ => false,
            };
            debug_assert!(
                agrees,
                "get_box_replacement_operand_opt: native walk diverged from box form for {opref:?}"
            );
        }
        native
    }

    /// "Box always exists" materializer (`resoperation.py:233-248
    /// AbstractResOpOrInputArg._forwarded`). Returns the canonical
    /// `Op` / `InputArg` `_forwarded` host for `opref`, minting a `SameAs*`
    /// synthetic into `resop_refs` when no producer is registered yet (the
    /// lazy-alloc arm). For a const-namespace OpRef returns a fresh
    /// `BoxRef::new_const` (`history.py:220` no-dedup; Const boxes have no
    /// `_forwarded`, so any write the caller attempts is a no-op). Unlike
    /// `resolve_to_boxref` it never returns `None` for a value-bearing OpRef
    /// — the explicit-mint endpoint (#47) at find-or-create write sites whose
    /// receiver may be unbound (test fixtures, short-preamble replay slots).
    /// The sentinel `OpRef::none()` has no box (debug-asserted); resolve it
    /// with `resolve_to_boxref` / `get_box_replacement_box` instead.
    pub(crate) fn materialize_box_at(&mut self, opref: OpRef) -> majit_ir::box_ref::BoxRef {
        debug_assert!(
            !opref.is_none(),
            "materialize_box_at: sentinel OpRef::none() has no box"
        );
        if opref.is_constant() {
            // history.py:220/261/307: a Const carries its Value on the OpRef.
            let value = self.get_constant(opref).unwrap_or_else(|| {
                panic!(
                    "materialize_box_at: const OpRef {opref:?} carries no Value — \
                     a Const carries its Value (history.py:220/261/307)"
                )
            });
            return majit_ir::box_ref::BoxRef::new_const(value);
        }
        // Align the write-path host with `resolve_to_boxref`
        // (the read path behind `get_box_replacement_box`). For ResOp
        // variants, resolve to the producing `Op`'s canonical `_forwarded`
        // host first. `find_producer_op` distinguishes the ResOp namespace
        // from the InputArg namespace by full `OpRef` (`op.pos == opref`),
        // so a raw-slot position-collapse — where one slot index served
        // both a ResOp and an InputArg — no longer routes a ResOp write to
        // `inputarg_refs[idx].forwarded` while the matching read routes to
        // `op.forwarded`. Returns `None` for InputArg / input positions
        // (no producing op), falling through to the InputArg / lazy-alloc
        // paths below unchanged.
        if let Some(op_rc) = self.find_producer_op(opref) {
            return majit_ir::box_ref::BoxRef::from_bound_op(&op_rc);
        }
        // InputArg write path: route through the canonical `inputarg_refs`
        // host (symmetric with `resolve_to_boxref`'s InputArg branch and the
        // `read_forwarded` / `clear_forwarded` writers). The returned BoxRef is
        // bound to `inputarg_refs[idx]`, so a `set_forwarded_*` write lands the
        // same `InputArg.forwarded` slot a later `resolve_to_boxref` read
        // observes — without returning a position-collapsed InputArg slot
        // whose write would silently vanish in a release build where the
        // `BoxRef::write_forwarded` bound-precondition assert is off.
        // Materialize / repair the slot's canonical `InputArgRc` by the
        // canonical `inputargs` slot type, mirroring the lazy-alloc path below.
        #[cfg(not(test))]
        if let OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_) = opref {
            let idx = opref.raw() as usize;
            // Type the slot from the canonical `inputargs` slot type, falling
            // back to the OpRef variant tag only when no canonical type is
            // recorded — `opref.ty()` can disagree across a phase boundary (a
            // Phase-2 OpRef referencing a Phase-1 low slot), so it is not the
            // authoritative source (mirrors the lazy-alloc arm's `inputarg_type`
            // sourcing below).
            let tp = self
                .inputarg_type(opref)
                .unwrap_or_else(|| opref.ty().unwrap_or(majit_ir::Type::Void));
            if idx >= self.inputarg_refs.len() {
                self.inputarg_refs
                    .resize_with(idx + 1, || std::rc::Rc::new(majit_ir::InputArg::new_int(0)));
                self.inputarg_refs[idx] =
                    std::rc::Rc::new(majit_ir::InputArg::from_type(tp, idx as u32));
            } else if self.inputarg_refs[idx].tp != tp
                || self.inputarg_refs[idx].index != idx as u32
            {
                self.inputarg_refs[idx] =
                    std::rc::Rc::new(majit_ir::InputArg::from_type(tp, idx as u32));
            }
            return majit_ir::box_ref::BoxRef::from_bound_inputarg(&self.inputarg_refs[idx]);
        }
        // Existing entries keep their construction-time shape (the recorder
        // / `with_inputarg_types` plant authoritative BoxRefs upstream);
        // only newly materialized placeholders pick the shape AND type from
        // the OpRef variant tag. `OpRef::InputArg{Int,Float,Ref}(i)` ⇒
        // `BoxRef::new_inputarg` (resoperation.py:719/727/739 + :233 the
        // `_forwarded` host); `OpRef::{Int,Float,Ref,Void}Op(p)` ⇒
        // `BoxRef::new_resop` (history.py:220 `op.type` parity).  Without
        // this variant-aware lazy-alloc, an `InputArg*` lookup would
        // synthesize a body-namespace `new_resop` shape and `boxref_to_opref`
        // would round-trip to `op_at(pos)` (None) instead of
        // `inputargs[i]`.
        // A resop reaching here has no producer in any `find_producer_op`
        // store (else it returned above), so it falls through to the
        // lazy-alloc arm below, which mints a `SameAs*` synthetic into
        // `resop_refs[opref]` and binds a BoxRef to it. A subsequent
        // `materialize_box_at` / `find_producer_op` for the same OpRef re-resolves to
        // that synthetic (`resop_refs[opref].pos == opref`), so the synthetic is
        // the stable `_forwarded` host across calls; no memoization side-table
        // is needed.
        let idx = opref.raw() as usize;
        let placeholder_type = opref.ty().unwrap_or(majit_ir::Type::Void);
        let placeholder = match opref {
            OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_) => {
                // `resoperation.py:719/727/739`: `InputArg{Int,Ref,Float}`'s
                // `datatype` is the box's intrinsic type. The canonical slot
                // type lives in `inputargs` (`loop.inputargs` parity); the
                // OpRef variant tag can disagree across a phase boundary (a
                // Phase-2 OpRef referencing a Phase-1 low slot), so the
                // materialized box takes its type from the canonical slot via
                // `inputarg_type`, not from `opref.ty()`. The variant tag is a
                // fallback only when no canonical slot type is recorded (test
                // contexts that bypass `setup_optimizations`).
                let canonical_type = self.inputarg_type(opref).unwrap_or(placeholder_type);
                let p = majit_ir::box_ref::BoxRef::new_inputarg(canonical_type, idx as u32);
                // Bind to the canonical `InputArgRc` for this slot. When
                // `inputarg_refs[idx]` is already populated (e.g. by
                // `with_inputarg_types`), use it; otherwise allocate a
                // fresh `InputArgRc`, stash it in `inputarg_refs`, and
                // bind. This keeps the `Forwarded::InputArg(_)` chain
                // shape (`optimizer.py:394 op.set_forwarded(newop)`
                // where `newop` is an `AbstractInputArg`) reachable for
                // lazy-allocated InputArg placeholders too.
                if idx >= self.inputarg_refs.len() {
                    self.inputarg_refs
                        .resize_with(idx + 1, || std::rc::Rc::new(majit_ir::InputArg::new_int(0)));
                    // Replace the placeholder filler at this exact slot
                    // with one matching the slot's canonical type/index.
                    self.inputarg_refs[idx] =
                        std::rc::Rc::new(majit_ir::InputArg::from_type(canonical_type, idx as u32));
                } else if self.inputarg_refs[idx].tp != canonical_type
                    || self.inputarg_refs[idx].index != idx as u32
                {
                    // Replace a mismatched filler (e.g. `new_int(0)` set
                    // by an earlier resize-fill on a different slot).
                    self.inputarg_refs[idx] =
                        std::rc::Rc::new(majit_ir::InputArg::from_type(canonical_type, idx as u32));
                }
                p.bind_inputarg(&self.inputarg_refs[idx]);
                p
            }
            _ => {
                let p = majit_ir::box_ref::BoxRef::new_resop(placeholder_type, idx as u32);
                // Bind to the producing OpRc when present so
                // `box.set_forwarded` dual-writes to `op.forwarded`
                // (resoperation.py:233 `_forwarded` host).
                if let Some(op_rc) = self.find_producer_op(opref) {
                    p.bind_op(&op_rc);
                } else {
                    // No producer Op yet — synthesise a `SameAsI/F/R`
                    // (or `Jump` for Void) stand-in with the correct
                    // result type and bind so chain steps targeting
                    // this BoxRef route through `Forwarded::Op(_)`
                    // (`optimizer.py:394 op.set_forwarded(newop)`
                    // where `newop` is an `AbstractResOp`). `emit()`
                    // re-binds to the real producer when it arrives,
                    // carrying the forwarded state across via
                    // `BoxRef::bind_op`'s carry-over.
                    let synthetic = self.mint_synthetic_resop(opref, placeholder_type);
                    p.bind_op(&synthetic);
                }
                p
            }
        };
        // The placeholder is bound to its producer / freshly-minted
        // `resop_refs` synthetic (resops) or `inputarg_refs` host (the
        // InputArg arm, only reachable in `#[cfg(test)]` since production
        // InputArgs resolve through the `inputarg_refs` branch above), so it
        // carries the canonical `_forwarded` host.
        placeholder
    }

    /// `optimizer.py:1009 getptrinfo + info.is_virtual()` BoxRef-routing
    /// helper. Returns whether the box at `opref` (after chain walk)
    /// carries a `PtrInfo` whose `is_virtual()` is true. Reads via
    /// `BoxRef::ptr_info()` on the chain-walked terminal box; an
    /// unresolvable opref (synthetic test paths) returns `false`.
    /// `optimizer.py:884-886 is_virtual(op)`:
    /// ```python
    /// def is_virtual(self, op):
    ///     opinfo = getptrinfo(op)
    ///     return opinfo is not None and opinfo.is_virtual()
    /// ```
    /// BoxRef-direct read — chain walks via
    /// `BoxRef::get_box_replacement` then queries `ptr_info().is_virtual()`.
    pub fn is_virtual(&self, op: &Operand) -> bool {
        op.get_box_replacement(false)
            .ptr_info()
            .map_or(false, |p| p.is_virtual())
    }

    /// `info.py:41-42 PtrInfo.is_nonnull` (base False) + subclass
    /// overrides — true when the box at `op` carries a non-null
    /// `PtrInfo` in its `_forwarded` Info slot. Chain walks via
    /// `BoxRef::get_box_replacement` then reads `ptr_info()`.
    pub fn is_nonnull(&self, op: &Operand) -> bool {
        op.get_box_replacement(false)
            .ptr_info()
            .map_or(false, |p| p.is_nonnull())
    }

    /// `optimizer.py:99-113 getintbound(op)` read variant — returns an
    /// owned `IntBound` snapshot from the chain terminal's `_forwarded`
    /// slot, plus the ConstInt arm:
    ///
    /// ```python
    /// op = get_box_replacement(op)
    /// if isinstance(op, ConstInt):
    ///     return IntBound.from_constant(op.getint())
    /// fw = op.get_forwarded()
    /// if isinstance(fw, IntBound): return fw
    /// return None     # upstream returns IntBound.unbounded(); the
    ///                 # peek variant signals "no specific bound" instead.
    /// ```
    ///
    /// The full lazy-install path (missing-info → `IntBound.unbounded()`)
    /// lives in [`Self::getintbound`]; this snapshot is the side-effect-
    /// free reader used by gates and read-only intersect comparisons.
    pub fn peek_intbound_box(
        &self,
        op: &Operand,
    ) -> Option<crate::optimizeopt::intutils::IntBound> {
        let resolved = op.get_box_replacement(false);
        if let Some(Value::Int(v)) = resolved.const_value() {
            return Some(crate::optimizeopt::intutils::IntBound::from_constant(
                v as i64,
            ));
        }
        resolved.int_bound().map(|ib| ib.clone())
    }

    /// `info.py:432 op.get_forwarded()` + `isinstance(fw, PtrInfo)` —
    /// snapshot read of the chain terminal's `_forwarded` PtrInfo.
    /// Clones the inner `PtrInfo` out of its `Rc<RefCell<>>` cell, so
    /// the result is independent of subsequent mutations.  For RPython
    /// object identity (`same_info`, in-place mutation propagation),
    /// use [`peek_ptr_info_handle`] which returns the live `Rc`.
    pub fn peek_ptr_info(&self, op: &Operand) -> Option<crate::optimizeopt::info::PtrInfo> {
        op.get_box_replacement(false).ptr_info().map(|p| p.clone())
    }

    /// info.py: getptrinfo(op) — mutable variant. Walks the chain on `op`
    /// and runs the closure against the terminal BoxRef's `_forwarded`
    /// PtrInfo via `ptr_info_mut()`. The BoxRef slot is the authoritative
    /// storage; no separate mirror step is needed.
    ///
    /// Closure semantics: returns `Some(f(info))` when a `PtrInfo` exists
    /// at the terminal box, `None` otherwise (no closure invocation).
    pub fn with_ptr_info_mut<R>(
        &self,
        op: &Operand,
        f: impl FnOnce(&mut PtrInfo) -> R,
    ) -> Option<R> {
        let resolved = op.get_box_replacement(false);
        let mut pi = resolved.ptr_info_mut()?;
        let result = f(&mut *pi);
        Some(result)
    }

    /// Closure-style wrapper around [`Self::ensure_ptr_info_arg0`].
    ///
    /// Closure mutations through `EnsuredPtrInfo::as_mut()` land on the
    /// BoxRef's `RefCell<Forwarded>` directly — single-slot RPython parity
    /// with `optimizer.py:467 ensure_ptr_info_arg0`'s mutate-in-place
    /// behavior.
    pub fn with_ensured_ptr_info_arg0<R>(
        &mut self,
        op: &Op,
        f: impl FnOnce(crate::optimizeopt::info::EnsuredPtrInfo) -> R,
    ) -> R {
        f(self.ensure_ptr_info_arg0(op))
    }

    /// `info.py:91-103 PtrInfo.get_last_guard_pos` BoxRef-direct reader.
    /// Walks chain to terminal and reads its `_forwarded` PtrInfo slot.
    pub fn last_guard_pos(&self, op: &Operand) -> Option<usize> {
        op.get_box_replacement(false)
            .ptr_info()
            .and_then(|p| p.get_last_guard_pos())
    }

    /// `info.py:880-894 getptrinfo(op) is not None` parity — true when
    /// the box carries any `PtrInfo` in its chain-terminal `_forwarded`
    /// Info slot. Walks via `BoxRef::get_box_replacement(false)` then
    /// queries `ptr_info().is_some()`.
    pub fn has_ptr_info(&self, op: &Operand) -> bool {
        // Mirror `getptrinfo(op).is_some()` so the gate behaves
        // identically. info.py:881-885 dispatches by `op.type`: only
        // Int and Ref boxes can carry PtrInfo (raw-ptr Int via
        // `getrawptrinfo`, regular Ref via `getptrinfo`). Float and
        // Void return None / are rejected upstream — short-circuit
        // here so callers of `has_ptr_info` can pass any typed BoxRef
        // without first guarding on the type.
        match op.type_() {
            majit_ir::Type::Int | majit_ir::Type::Ref => self.getptrinfo(op).is_some(),
            majit_ir::Type::Float | majit_ir::Type::Void => false,
        }
    }

    /// TODO: RPython's virtualizable handling lives
    /// tracing-side (`pyjitpl.py:1120-1145 _nonstandard_virtualizable`),
    /// not in optimizeopt — there is no direct `is_virtualizable` helper
    /// in `optimizer.py`. The pyre dedicated `PtrInfo::Virtualizable`
    /// variant + this helper exist because pyre routes virtualizable
    /// field tracking through the optimizer's `_forwarded` PtrInfo slot.
    /// Returns true when the chain-terminal carries `PtrInfo::Virtualizable`.
    pub fn is_virtualizable(&self, op: &Operand) -> bool {
        use crate::optimizeopt::info::PtrInfo;
        op.get_box_replacement(false)
            .ptr_info()
            .map_or(false, |p| matches!(*p, PtrInfo::Virtualizable(_)))
    }

    /// resoperation.py: op.get_forwarded() is not None — check if OpRef
    /// has any forwarding entry (Op, Info, IntBound, Const).
    ///
    /// `Const.get_forwarded()` returns `None` in RPython
    /// (`resoperation.py:1162`); short-circuit on the const-namespace
    /// `OpRef` so the caller doesn't index a raw-keyed store with a
    /// CONST_BIT `raw()` — which would either miss (large-index) or
    /// alias an unrelated slot.
    pub fn has_forwarding(&self, op: &Operand) -> bool {
        // `resoperation.py:1162 Const.get_forwarded()` returns None;
        // Const boxes carry no `_forwarded` slot upstream.
        if op.is_constant() {
            return false;
        }
        // `resoperation.py:235 _forwarded = None` — slot is None until
        // `set_forwarded` writes. `op.get_forwarded() is not None`.
        !matches!(op.get_forwarded(), majit_ir::box_ref::Forwarded::None)
    }

    /// True only when opref has a non-const forwarding redirect.
    ///
    /// `make_equal_to(_, non_const)` writes either `Forwarded::Op(_)` or
    /// `Forwarded::InputArg(_)`; the const branches go through
    /// `Forwarded::Const`. Splitting on the variant excludes the
    /// const-target shape so this returns true only for the AbstractValue
    /// redirect case used by `import_state`.
    ///
    /// `Const.get_forwarded()` returns `None` upstream
    /// (`resoperation.py:1162`); short-circuit on the const-namespace
    /// `OpRef` so the caller doesn't index a raw-keyed store with a
    /// CONST_BIT `raw()`.
    pub fn has_op_forwarding(&self, op: &Operand) -> bool {
        if op.is_constant() {
            return false;
        }
        matches!(
            &op.get_forwarded(),
            majit_ir::box_ref::Forwarded::Op(_) | majit_ir::box_ref::Forwarded::InputArg(_)
        )
    }

    /// Bulk-seed entry for the recorder / backend constant import. NOT a
    /// substitute for the RPython `make_constant(box, constbox)`
    /// (`optimizer.py:413`); production optimizer-time const promotions
    /// must go through `OptContext::make_constant_box`, which overwrites any
    /// existing forwarding per upstream.
    ///
    /// `box_` is the canonical `_forwarded` host — the caller resolves it
    /// via `materialize_box_at` (for a Const-namespace OpRef that is a
    /// fresh `new_const`; for a body OpRef the producing `Op` / `InputArg`
    /// host). A Const box carries its value intrinsically
    /// (history.py:220/261/307), so the const arm only asserts the box and
    /// value types agree. A non-Const box is forwarded to a fresh Const
    /// target, but only when the slot is `Forwarded::None`, preserving any
    /// PtrInfo / IntBound / Box(Const) forwarding installed by an earlier
    /// pass (PyPy's `make_constant` short-circuits on `box.is_constant()`
    /// before reaching `set_forwarded`; this is the recorder/bulk-seed
    /// entry where the forwarding slot is authoritative when present). The
    /// recorder calls this before optimization passes run, so the
    /// no-clobber rule never collides with a real `make_constant` caller.
    ///
    /// RPython parity: `ConstInt`, `ConstPtr`, `ConstFloat` are distinct
    /// Box subclasses (history.py:220/261/307); a typed seed over a slot
    /// already holding a different-typed value is a bug (typical source:
    /// `Value::Ref(0)` reseeded where `Value::Int(0)` lives, flipping
    /// `opref_type` Int→Ref). Assert the invariant rather than overwrite.
    pub fn seed_constant(&mut self, box_: &Operand, value: Value) {
        if box_.is_constant() {
            debug_assert!(
                box_.type_() == value.get_type(),
                "seed_constant: Const box type {:?} mismatches value {:?}",
                box_.type_(),
                value,
            );
        } else {
            // optimizer.py:432 `box.set_forwarded(constbox)`, gated on
            // `Forwarded::None` per the no-clobber rule documented above.
            if matches!(box_.get_forwarded(), majit_ir::box_ref::Forwarded::None) {
                box_.set_forwarded_const(majit_ir::Const::from_value(value));
            }
        }
    }

    /// Read-only variant of `getintbound` — returns the IntBound stored on
    /// `box._forwarded` without materializing an unbounded one on first
    /// access. Returns `None` for boxes that have no IntBound forwarding.
    /// Used by exporters that take `&OptContext` and cannot mutate.
    pub(crate) fn peek_intbound(
        &self,
        opref: OpRef,
    ) -> Option<crate::optimizeopt::intutils::IntBound> {
        // optimizer.py:99-100: assert op.type == 'i'
        // None is allowed for test fixtures that don't seed value_types.
        assert!(
            matches!(self.opref_type(opref), Some(majit_ir::Type::Int) | None),
            "peek_intbound: expected 'i'-typed OpRef, got {:?}",
            self.opref_type(opref)
        );
        let replaced = self.get_replacement_opref(opref);
        if let Some(Value::Int(v)) = self
            .get_box_replacement_operand_opt(replaced)
            .and_then(|cb| cb.const_value())
        {
            return Some(crate::optimizeopt::intutils::IntBound::from_constant(
                v as i64,
            ));
        }
        // optimizer.py:107 second `assert op.type == 'i'` — Box.type is
        // immutable in RPython, so the replaced op must still be int-typed.
        assert!(
            matches!(self.opref_type(replaced), Some(majit_ir::Type::Int) | None),
            "peek_intbound: replaced OpRef must be int-typed, got {:?}",
            self.opref_type(replaced)
        );
        if replaced.is_constant() {
            return None;
        }
        // BoxRef-authoritative reader. IntBound writers populate the
        // BoxRef via `materialize_box_at`.
        let b = self.get_box_replacement_operand_opt(replaced)?;
        b.int_bound().map(|ib| ib.clone())
    }

    /// `optimizer.py:99-113 getintbound(op)` orthodox identity variant.
    ///
    /// Walks `op.get_box_replacement(false)` and returns:
    ///   - `IntBoundHandle::Const(from_constant(v))` for a `ConstInt`
    ///     terminal (line 102-103).
    ///   - `IntBoundHandle::Const(unbounded)` when the terminal carries
    ///     a non-IntBound `_forwarded` slot (raw-pointer Int with
    ///     PtrInfo etc.; `optimizer.py:106` non-IntBound `fw` branch).
    ///   - `IntBoundHandle::Live(rc)` for an existing
    ///     `OpInfo::IntBound(rc)` slot, **or** a freshly installed
    ///     `unbounded` cell when the slot was `Forwarded::None` —
    ///     mirroring RPython's lazy `op.set_forwarded(IntBound())`
    ///     side-effect at line 111.
    pub fn getintbound_handle(&mut self, op: &Operand) -> IntBoundHandle {
        use crate::optimizeopt::info::OpInfo;
        // optimizer.py:100 `assert op.type == 'i'`. Void admitted as the
        // pyre placeholder-box tolerance shared with `setintbound`.
        assert!(
            matches!(op.type_(), majit_ir::Type::Int | majit_ir::Type::Void),
            "getintbound_handle: expected 'i'-typed BoxRef, got {:?}",
            op.type_()
        );
        let resolved = op.get_box_replacement(false);
        // optimizer.py:107 `assert op.type == 'i'` lifted ahead of every
        // post-walker branch.  PyPy's structural-error catch fires when a
        // non-`ConstInt` terminal reaches the `fw is None` arm and the
        // assert sees a non-`'i'` `op.type` — pyre matches that position
        // up-front so a cross-type-forwarded Int box (Int → ConstPtr /
        // ConstFloat) panics here instead of being silently absorbed by
        // the unbounded branch below.  Mirrors `peek_intbound_handle`'s
        // assert shape.
        assert!(
            matches!(resolved.type_(), majit_ir::Type::Int | majit_ir::Type::Void),
            "getintbound_handle: chain terminal lost 'i' type, got {:?}",
            resolved.type_()
        );
        if let Some(Value::Int(v)) = resolved.const_value() {
            return IntBoundHandle::const_(crate::optimizeopt::intutils::IntBound::from_constant(
                v as i64,
            ));
        }
        match &resolved.get_forwarded() {
            majit_ir::box_ref::Forwarded::Info(OpInfo::IntBound(rc)) => {
                return IntBoundHandle::live(std::rc::Rc::clone(rc));
            }
            majit_ir::box_ref::Forwarded::None => {}
            _ => {
                return IntBoundHandle::const_(crate::optimizeopt::intutils::IntBound::unbounded());
            }
        }
        // optimizer.py:110-112 lazy install — the new cell is the live
        // identity that downstream `intersect`/`make_*` mutations
        // propagate through.
        let intbound = crate::optimizeopt::intutils::IntBound::unbounded();
        resolved.set_forwarded_info(OpInfo::int_bound(intbound));
        let rc = resolved
            .int_bound_handle()
            .expect("just installed OpInfo::IntBound");
        IntBoundHandle::live(rc)
    }

    /// optimizer.py:115-125: setintbound(op, bound) line-by-line port.
    ///
    /// ```python
    /// def setintbound(self, op, bound):
    ///     assert op.type == 'i'
    ///     op = get_box_replacement(op)
    ///     if op.is_constant():
    ///         return
    ///     cur = op.get_forwarded()
    ///     if cur is not None:
    ///         if isinstance(cur, IntBound):
    ///             cur.intersect(bound)
    ///     else:
    ///         op.set_forwarded(bound)
    /// ```
    pub fn setintbound(&self, op: &Operand, bound: &crate::optimizeopt::intutils::IntBound) {
        use crate::optimizeopt::info::OpInfo;
        // optimizer.py:116: assert op.type == 'i' — structural assert,
        // matches RPython's release-build invariant. Type::Void boxes are
        // pyre-only phantom placeholders surfaced by `materialize_box_at` when the
        // recorder has not yet typed the position; accept them as the pyre
        // equivalent of RPython's "the trace typing hasn't reached this
        // OpRef yet" tolerance.
        assert!(
            matches!(op.type_(), majit_ir::Type::Int | majit_ir::Type::Void),
            "setintbound: expected 'i'-typed BoxRef, got {:?}",
            op.type_()
        );
        // optimizer.py:117: op = get_box_replacement(op)
        let op = op.get_box_replacement(false);
        // optimizer.py:118-119: if op.is_constant(): return
        if op.is_constant() {
            return;
        }
        // optimizer.py:120-122: cur = op.get_forwarded()
        //                       if cur is not None and isinstance(cur, IntBound):
        //                           cur.intersect(bound)
        if let Some(mut cur) = op.int_bound_mut() {
            let _ = cur.intersect(bound);
            return;
        }
        // optimizer.py:123-125: else (cur is None): op.set_forwarded(bound)
        // When cur is a non-None non-IntBound (e.g. RawBufferPtrInfo on a
        // raw-pointer Int), upstream's outer `if cur is not None` already
        // consumed control; the else branch only runs when cur is None.
        use majit_ir::box_ref::Forwarded as BoxFwd;
        if matches!(op.get_forwarded(), BoxFwd::None) {
            op.set_forwarded_info(OpInfo::int_bound(bound.clone()));
        }
    }

    /// In-place mutation helper for the IntBound stored on `box._forwarded`.
    ///
    /// RPython pattern equivalence: where RPython writes
    /// `self.getintbound(box).<method>(...)` and the method mutates the
    /// `IntBound` returned from `box.get_forwarded()` directly, the Rust
    /// borrow checker forces us to materialize the bound, mutate it, and
    /// store it back. This helper performs that read-modify-write atomically
    /// and threads through any return value from the closure (e.g. the
    /// `Result<bool, InvalidLoop>` flag from `intersect`/`make_*`).
    ///
    /// For Constant boxes the bound is "fixed" — RPython's `getintbound`
    /// returns `IntBound.from_constant(...)` and any `intersect` is a
    /// no-op (the constant value is already in range or InvalidLoop). This
    /// helper mirrors that by running the closure on a temporary that is
    /// discarded after — the constant cannot be widened.
    ///
    /// For non-IntBound forwarded info (RawBufferPtrInfo etc.), RPython's
    /// `getintbound` falls through to "return IntBound.unbounded()" without
    /// overwriting forwarding. We mirror by running the closure on a
    /// temporary unbounded that is discarded.
    pub fn with_intbound_mut<F, R>(&self, op: &Operand, f: F) -> R
    where
        F: FnOnce(&mut crate::optimizeopt::intutils::IntBound) -> R,
    {
        use crate::optimizeopt::info::OpInfo;
        use majit_ir::box_ref::Forwarded;
        // optimizer.py:99-100: assert op.type == 'i'. Active in release
        // builds per upstream. Void-typed phantoms (`materialize_box_at` lazy-alloc)
        // are accepted because they are placeholder boxes pending recorder
        // typing — their chain walk may still terminate at an int-typed
        // Const/InputArg.
        assert!(
            matches!(op.type_(), majit_ir::Type::Int | majit_ir::Type::Void),
            "with_intbound_mut: expected 'i'-typed BoxRef, got {:?}",
            op.type_()
        );
        // optimizer.py:101: op = get_box_replacement(op)
        let resolved = op.get_box_replacement(false);
        // optimizer.py:102-103: ConstInt → IntBound.from_constant(...).
        // RPython's getintbound returns a fresh bound; intersect on it is
        // a no-op (already at the constant value), and the bound is
        // discarded after the closure — the Const box stays canonical.
        if let Some(Value::Int(v)) = resolved.const_value() {
            let mut tmp = crate::optimizeopt::intutils::IntBound::from_constant(v as i64);
            return f(&mut tmp);
        }
        if resolved.is_constant() {
            // Non-Int constant (Float / Ref) — getintbound's "assert op.type
            // == 'i'" would fail upstream; majit returns unbounded for
            // raw-pointer Int constants and Type::Void phantoms, both of
            // which surface here when the typed-namespace OpRef is forced
            // through `with_intbound_mut`.
            let mut tmp = crate::optimizeopt::intutils::IntBound::unbounded();
            return f(&mut tmp);
        }
        // optimizer.py:104-109: branch on forwarded slot.
        let needs_init = matches!(resolved.get_forwarded(), Forwarded::None);
        if needs_init {
            // optimizer.py:110-112 first-access: materialize unbounded,
            // mutate via closure, install on `_forwarded`.
            let mut new_bound = crate::optimizeopt::intutils::IntBound::unbounded();
            let result = f(&mut new_bound);
            resolved.set_forwarded_info(OpInfo::int_bound(new_bound));
            return result;
        }
        if let Some(mut bound) = resolved.int_bound_mut() {
            // optimizer.py:106-107: existing IntBound — mutate in place.
            return f(&mut *bound);
        }
        // optimizer.py:108-109 rare case: forwarded is AbstractInfo other
        // than IntBound (RawBufferPtrInfo, FloatConst, etc.) — return a
        // temporary unbounded that gets discarded after the closure.
        let mut tmp = crate::optimizeopt::intutils::IntBound::unbounded();
        f(&mut tmp)
    }

    /// `Operand` form of [`OptContext::make_constant_box`]. optimizer.py:413
    /// `make_constant(box, constbox)` does `box = get_box_replacement(box)` then
    /// forwards the constant; this takes that first resolve operand-native via
    /// `resolve_operand_operand_opt` instead of collapsing the operand to an `OpRef`.
    pub fn make_constant_arg(&mut self, arg: &Operand, value: Value) {
        let b = self.resolve_operand_operand_opt(arg).or_else(|| {
            let opref = arg.to_opref();
            (!opref.is_none() && !opref.is_constant()).then(|| self.materialize_operand_at(opref))
        });
        if let Some(b) = b {
            // `resolve_operand_operand_opt` / `materialize_operand_at` yield a
            // bound-or-const operand, so the lowering is panic-free.
            self.make_constant_box(&b, value);
        }
    }

    /// optimizer.py:413-435 make_constant(box, constbox)
    pub fn make_constant_box(&mut self, op: &Operand, value: Value) {
        // optimizer.py:415: box = get_box_replacement(box)
        let op = op.get_box_replacement(false);
        // optimizer.py:418-429: IntBound safety check
        if let Value::Int(intval) = value {
            if let Some(mut bound) = op.int_bound_mut() {
                if !bound.contains(intval as i64) {
                    drop(bound);
                    self.signal_invalid_loop(
                        "constant int is outside the range allowed for that box",
                    );
                    return;
                }
                let _ = bound.make_eq_const(intval as i64);
            }
        }
        // optimizer.py:430: if box.is_constant(): return
        if op.is_constant() || op.const_value().is_some() {
            return;
        }
        // optimizer.py:432-434: copy_fields_to_const for Ref
        if let Value::Ref(gcref) = value {
            if let Some(pos) = op.position() {
                let opref = majit_ir::OpRef::ref_op(pos);
                self.copy_fields_to_const(opref, gcref);
            }
        }
        // optimizer.py:432: box.set_forwarded(constbox). Terminate the
        // chain in an inline value-typed Const payload (history.py:227/
        // 268/314) — no separate BoxKind::Const carrier and no pool index.
        // `get_box_replacement` rematerializes the const and `box_to_opref`
        // recovers the inline-Const OpRef via `source_opref()`'s
        // value-derived branch.
        if matches!(value, Value::Void) {
            panic!("make_constant: Value::Void has no ConstVoid upstream (history.py:220/261/307)");
        }
        op.set_forwarded_const(value.to_const());
    }

    /// info.py:194-198 (AbstractStructPtrInfo) + info.py:533-538 (ArrayPtrInfo)
    /// `copy_fields_to_const(constinfo, optheap)`.
    ///
    /// ```text
    /// # AbstractStructPtrInfo
    /// def copy_fields_to_const(self, constinfo, optheap):
    ///     if self._fields is not None:
    ///         info = constinfo._get_info(self.descr, optheap)
    ///         assert isinstance(info, AbstractStructPtrInfo)
    ///         info._fields = self._fields[:]
    ///
    /// # ArrayPtrInfo
    /// def copy_fields_to_const(self, constinfo, optheap):
    ///     descr = self.descr
    ///     if self._items is not None:
    ///         info = constinfo._get_array_info(descr, optheap)
    ///         assert isinstance(info, ArrayPtrInfo)
    ///         info._items = self._items[:]
    /// ```
    ///
    /// majit folds both per-type entries into a single helper because the
    /// per-source dispatch happens via the PtrInfo enum match. The
    /// `_get_info`/`_get_array_info` half is `const_infos.entry(...)`
    /// (RPython: `optheap.const_infos[ref]`).
    fn copy_fields_to_const(&mut self, source: OpRef, gcref: majit_ir::GcRef) {
        use crate::optimizeopt::info::{ArrayPtrInfo, FieldEntry, PtrInfo, StructPtrInfo};
        // `source` is always chain-walked by the caller (`make_constant`),
        // so peek's chain
        // walk is a no-op — owned PtrInfo clone here matches the prior
        // `Forwarded::Info(info)` immediate-slot read.
        let source_box = self.get_box_replacement_operand_opt(source);
        let Some(info) = source_box.as_ref().and_then(|b| self.peek_ptr_info(b)) else {
            return;
        };
        let key = gcref.as_usize();
        match info {
            // info.py:194-198 AbstractStructPtrInfo.copy_fields_to_const →
            // constinfo._get_info(self.descr, optheap) → StructPtrInfo(descr).
            PtrInfo::Instance(v) if !v.fields.is_empty() => {
                let Some(descr) = v.descr.clone() else {
                    return;
                };
                let fields = v.fields.clone();
                let ci = self.const_infos.entry(key).or_insert_with(|| {
                    PtrInfo::Struct(StructPtrInfo {
                        descr,
                        fields: Vec::new(),

                        last_guard_pos: -1,
                    })
                });
                if let PtrInfo::Struct(s) = ci {
                    s.fields = fields;
                }
            }
            PtrInfo::Struct(v) if !v.fields.is_empty() => {
                let descr = v.descr.clone();
                let fields = v.fields.clone();
                let ci = self.const_infos.entry(key).or_insert_with(|| {
                    PtrInfo::Struct(StructPtrInfo {
                        descr,
                        fields: Vec::new(),

                        last_guard_pos: -1,
                    })
                });
                if let PtrInfo::Struct(s) = ci {
                    s.fields = fields;
                }
            }
            PtrInfo::Virtual(v) if !v.fields.is_empty() => {
                let descr = v.descr.clone();
                let fields: Vec<(u32, FieldEntry)> = v
                    .fields
                    .iter()
                    .map(|(k, r)| (*k, FieldEntry::Value(r.clone())))
                    .collect();
                let ci = self.const_infos.entry(key).or_insert_with(|| {
                    PtrInfo::Struct(StructPtrInfo {
                        descr,
                        fields: Vec::new(),

                        last_guard_pos: -1,
                    })
                });
                if let PtrInfo::Struct(s) = ci {
                    s.fields = fields;
                }
            }
            PtrInfo::VirtualStruct(v) if !v.fields.is_empty() => {
                let descr = v.descr.clone();
                let fields: Vec<(u32, FieldEntry)> = v
                    .fields
                    .iter()
                    .map(|(k, r)| (*k, FieldEntry::Value(r.clone())))
                    .collect();
                let ci = self.const_infos.entry(key).or_insert_with(|| {
                    PtrInfo::Struct(StructPtrInfo {
                        descr,
                        fields: Vec::new(),

                        last_guard_pos: -1,
                    })
                });
                if let PtrInfo::Struct(s) = ci {
                    s.fields = fields;
                }
            }
            // info.py:533-538 ArrayPtrInfo.copy_fields_to_const →
            // constinfo._get_array_info(descr, optheap) → ArrayPtrInfo(descr).
            PtrInfo::Array(v) if !v.items.is_empty() => {
                let descr = v.descr.clone();
                let lenbound = v.lenbound.clone();
                let items = v.items.clone();
                let ci = self.const_infos.entry(key).or_insert_with(|| {
                    PtrInfo::Array(ArrayPtrInfo {
                        descr,
                        lenbound,
                        items: Vec::new(),
                        last_guard_pos: -1,
                    })
                });
                if let PtrInfo::Array(a) = ci {
                    a.items = items;
                }
            }
            PtrInfo::VirtualArray(v) if !v.items.is_empty() => {
                let descr = v.descr.clone();
                let len = v.items.len() as i64;
                let items: Vec<FieldEntry> = v
                    .items
                    .iter()
                    .map(|r| FieldEntry::Value(r.clone()))
                    .collect();
                let ci = self.const_infos.entry(key).or_insert_with(|| {
                    PtrInfo::Array(ArrayPtrInfo {
                        descr,
                        lenbound: IntBound::from_constant(len),
                        items: Vec::new(),
                        last_guard_pos: -1,
                    })
                });
                if let PtrInfo::Array(a) = ci {
                    a.items = items;
                }
            }
            _ => {}
        }
    }

    /// resume.py:157 getconst parity for synthetic rd_numb encoding.
    /// Returns the (raw bits, type) of a constant BoxRef, or None if it
    /// is not a constant. Type comes from the `Value` variant directly;
    /// raw-pointer Int constants live as `BoxKind::Const` with
    /// `Value::Ref` (Ref-typed) per the typed-pointer model, so
    /// `Value::Int` is always a real integer here.
    pub fn getconst(&self, op: &Operand) -> Option<(i64, majit_ir::Type)> {
        // Walk the chain and read the terminal's const_value (Const Box).
        let resolved = op.get_box_replacement(false);
        if let Some(val) = resolved.const_value() {
            let (raw, tp) = match val {
                Value::Int(v) => (v, majit_ir::Type::Int),
                Value::Float(f) => (f.to_bits() as i64, majit_ir::Type::Float),
                Value::Ref(r) => (r.0 as i64, majit_ir::Type::Ref),
                _ => return None,
            };
            return Some((raw, tp));
        }
        // info.py: ConstPtrInfo — GcRef constant stored in PtrInfo.
        if let Some(crate::optimizeopt::info::PtrInfo::Constant(gcref)) = self.peek_ptr_info(op) {
            return Some((gcref.0 as i64, majit_ir::Type::Ref));
        }
        None
    }

    /// Actual-Const reader: `box = box.get_box_replacement(); isinstance(box, Const)`.
    ///
    /// This intentionally does not use `get_constant_box`, because PyPy's
    /// `optimizer.get_constant_box` also synthesizes `ConstInt` from constant
    /// `IntBound`. Call this only for source sites that literally test
    /// `isinstance(..., Const)` / `box.is_constant()` / direct `ConstInt`.
    pub fn get_constant(&self, opref: OpRef) -> Option<Value> {
        match opref {
            OpRef::ConstInt(v) => return Some(Value::Int(v)),
            OpRef::ConstFloat(v) => return Some(Value::Float(v)),
            OpRef::ConstPtr(v) => return Some(Value::Ref(v)),
            // Non-constant OpRefs walk the forwarding chain below to find a
            // value forwarded onto them by `make_constant`.
            _ => {}
        }
        self.get_box_replacement_operand_opt(opref)
            .and_then(|b| b.const_value())
    }

    /// resoperation.py:691-720 `InputArg*.getint/getref_base/getfloatstorage`
    /// — extract the concrete runtime value carried by an OpRef's OWN box.
    ///
    /// virtualstate.py:400 `runtime_box.constbox()`, :494 `.getint()`, :579
    /// `.nonnull()`, :601/:608 `cpu.cls_of_box(runtime_box)` read the runtime
    /// box object itself: `getint`/`getref_base` (resoperation.py:691) return
    /// `_resint`/`_resref` — the box's own value slot, set when the box was
    /// created — and never walk `_forwarded`. This resolves the box at
    /// `opref`'s own position (`resolve_to_boxref`, the canonical host WITHOUT
    /// the `get_box_replacement` chain walk) and reads its value directly; an
    /// optimizer forwarding (`make_equal_to` / `make_constant`) never takes
    /// precedence over the box's own observed value.
    ///
    /// `None` for an own slot that carries no value: pyre's value slots are
    /// `Option<Value>` (resoperation.rs / box_ref.rs) where RPython's are
    /// `_resint=0` / `_resref=NULL` defaults, so an unobserved box reads as
    /// `None` here. Callers (`runtime_nonnull`, IntBounded, `runtime_cls_of`,
    /// `get_runtime_field`) treat `None` as "no runtime guidance" and refuse
    /// the guard — the conservative direction. The runtime boxes threaded into
    /// generate_guards are the recorded JUMP args (unroll.py:105), whose
    /// producing ops carry their own observed values, so this reads the real
    /// runtime value without consulting `_forwarded`.
    pub fn runtime_value_of(&self, opref: OpRef) -> Option<Value> {
        let own = self.resolve_to_boxref(opref)?;
        own.const_value().or_else(|| own.get_value())
    }

    /// `runtime_box.nonnull()` — resoperation.py:583 `IntOp.nonnull`
    /// (`self._resint != 0`), :609 `FloatOp.nonnull`
    /// (`bool(extract_bits(self._resfloat))`), `RefOp.nonnull`
    /// (`bool(self.getref_base())`). Reads the runtime box's carried value
    /// (`runtime_value_of`) and applies the per-type rule. Returns `false`
    /// when no runtime value is plumbed: a box with no observed value must
    /// not be claimed nonnull (virtualstate.py:579 gates GUARD_NONNULL on
    /// `runtime_box.nonnull()`, so a null/absent value refuses the guard).
    pub fn runtime_nonnull(&self, opref: OpRef) -> bool {
        match self.runtime_value_of(opref) {
            Some(Value::Int(i)) => i != 0,
            Some(Value::Float(f)) => f.to_bits() != 0,
            Some(Value::Ref(r)) => !r.is_null(),
            Some(Value::Void) | None => false,
        }
    }

    /// `cpu.cls_of_box(runtime_box)` — virtualstate.py:601/608/620,
    /// model.py:199-201. Reads the runtime box's OWN ref value
    /// (`getref_base`, resoperation.py:691 — the box's own `_resref` slot,
    /// never the `_forwarded` chain) via `runtime_value_of`, then returns
    /// `ptr2int(typeptr)`, the immortal vtable address as a plain integer.
    ///
    /// Unlike `cls_of_box(&BoxRef)` (which walks `get_box_replacement` to
    /// reach a Const terminal), this resolves through the no-forward
    /// `runtime_value_of`: virtualstate's KnownClass arms read the runtime
    /// box itself, with no optimizer-tracked / forwarded precedence. Returns
    /// `None` for non-Ref / null / unobserved values, so a KnownClass guard
    /// refuses rather than reading a forwarded class.
    pub fn runtime_cls_of(&self, opref: OpRef) -> Option<i64> {
        match self.runtime_value_of(opref)? {
            Value::Ref(gcref) if !gcref.is_null() => {
                let synth = majit_ir::box_ref::BoxRef::new_const(Value::Ref(gcref));
                let typeptr = self.cpu.cls_of_box(&synth);
                if typeptr == 0 { None } else { Some(typeptr) }
            }
            _ => None,
        }
    }

    /// resoperation.py:38 `AbstractResOpOrInputArg.same_box`: `self is other`
    /// — Python object identity, NOT the value-aware `Const.same_box`.
    ///
    /// Walks both operands through `get_box_replacement` (resoperation.py:58)
    /// then compares the resolved `OpRef`s.
    ///
    /// IDENTITY CAVEAT: `OpRef::Const*` carries the constant VALUE inline
    /// (history.py:227), not a pool-index slot, so two independently-minted
    /// `ConstInt(5)` resolve to the *same* `OpRef` and ARE `box_is`-equal.
    /// For constants this therefore matches PyPy's value-based
    /// `Const.same_box` (history.py:211), NOT PyPy's object-identity `is`
    /// (two distinct `ConstInt(5)` objects are `is`-False). For non-constant
    /// boxes (InputArg*/`*-Op` positions) the variant tag still encodes a
    /// unique position, so `box_is` remains a faithful 1:1 encoding of `is`.
    ///
    /// USAGE / HAZARD: use this where RPython writes `arg0 is arg1`; use
    /// `same_box` where RPython writes `arg0.same_box(arg1)`. Because
    /// constants collapse by value here, only call `box_is` at an `is`-site
    /// where treating two equal-valued constants as identical is correct (or
    /// conservatively safe). The current `is`-on-constant call sites are
    /// value-safe:
    ///   - rewrite.rs `_optimize_oois_ooisnot` `elif arg0 is arg1`
    ///     (rewrite.py:542): folding equal `ConstPtr`/`ConstInt` to "equal"
    ///     is the correct result.
    ///   - heap.rs `lookup_cached` `cached_index is indexbox` (heap.py:322):
    ///     a hit on an equal constant index is valid (the var-index cache is
    ///     write-invalidated, so no stale hit can survive).
    /// A future `is`-site that must treat equal-valued *distinct* constants
    /// as DISTINCT cannot use `box_is` as-is.
    pub fn box_is(&self, a: OpRef, b: OpRef) -> bool {
        self.get_replacement_opref(a) == self.get_replacement_opref(b)
    }

    /// resoperation.py:38 `same_box` (non-Const: `self is other`) +
    /// history.py:211 `Const.same_box` (value comparison via
    /// `same_constant`). Resolves both operands through
    /// `get_box_replacement_box` then delegates to `BoxRef::same_box`. Falls
    /// back to resolved-`OpRef` identity plus constant-value comparison
    /// when either box is absent: two references to the same unresolved
    /// variable are still the same box (`self is other`), and a
    /// producer-less position has no canonical `BoxRef` to compare by `Rc`
    /// identity, so resolved-`OpRef` equality stands in for object identity.
    pub fn same_box(&self, query: OpRef, stored: OpRef) -> bool {
        match (
            self.get_box_replacement_operand_opt(query),
            self.get_box_replacement_operand_opt(stored),
        ) {
            (Some(ref a), Some(ref b)) => a.same_box(b),
            _ => {
                let query = self.get_replacement_opref(query);
                let stored = self.get_replacement_opref(stored);
                if query == stored {
                    return true;
                }
                match (
                    self.get_box_replacement_operand_opt(query)
                        .and_then(|cb| cb.const_value()),
                    self.get_box_replacement_operand_opt(stored)
                        .and_then(|cb| cb.const_value()),
                ) {
                    (Some(a), Some(b)) => a == b,
                    _ => false,
                }
            }
        }
    }

    /// vstring.py:237 `optstring.getintbound(box).is_constant()` pattern.
    /// Returns the constant value if known either from the constant pool
    /// or from IntBound analysis.
    pub fn get_constant_int_or_bound_box(&self, b: &Operand) -> Option<i64> {
        if let Some(Value::Int(i)) = self.get_constant_box(b) {
            return Some(i);
        }
        self.peek_intbound_box(b)
            .filter(|ib| ib.is_constant())
            .map(|ib| ib.get_constant_int())
    }

    /// history.py:361 CONST_NULL = ConstPtr(ConstPtr.value).
    /// `CONST_NULL.same_constant(op)` parity (history.py:361 `CONST_NULL =
    /// ConstPtr(ConstPtr.value)`). True iff `op` resolves to a Ref-typed
    /// null constant. Walks the chain and reads the terminal's
    /// `const_value()` directly — Const-namespace OpRefs whose
    /// forwarding chain terminates at a `Forwarded::Const`
    /// with `Value::Ref(GcRef(0))` are detected here.
    pub fn is_const_null(&self, op: &Operand) -> bool {
        matches!(
            op.get_box_replacement(false).const_value(),
            Some(Value::Ref(r)) if r.0 == 0
        )
    }

    /// optimizer.py:705-711: is_call_pure_pure_canraise — a CallPure op whose
    /// effectinfo says check_can_raise(ignore_memoryerror=True). These ops are
    /// formally side-effect-free (has_no_side_effect), but their potential to
    /// raise means they break guard resume-data sharing.
    fn is_call_pure_pure_canraise(op: &Op) -> bool {
        if !op.opcode.is_call_pure() {
            return false;
        }
        let Some(descr) = op.getdescr() else {
            return false;
        };
        let Some(cd) = descr.as_call_descr() else {
            return false;
        };
        cd.get_extra_info().check_can_raise(true)
    }

    /// optimizer.py:652-686 emit_guard_operation — decide whether to share
    /// resume data from the previous guard (_copy_resume_data_from) or build
    /// new resume data (store_final_boxes_in_guard).
    fn emit_guard_operation(&mut self, op: &mut Op) {
        let opnum = op.opcode;

        // optimizer.py:655-664: GUARD_(NO_)EXCEPTION following a guard that
        // is NOT GUARD_NOT_FORCED — give up sharing.  GUARD_NOT_FORCED_2
        // is excluded for the same reason as in the Optimizer path:
        // pyjitpl.py:3236 emits it at finish() only, so no exception
        // guard can follow.
        if opnum == OpCode::GuardNoException || opnum == OpCode::GuardException {
            if let Some(idx) = self.last_guard_idx {
                if self.new_operations[idx].opcode != OpCode::GuardNotForced {
                    self.last_guard_idx = None;
                }
            }
        }

        // optimizer.py:665-670: GUARD_ALWAYS_FAILS must never share.
        if opnum == OpCode::GuardAlwaysFails {
            self.last_guard_idx = None;
        }

        // optimizer.py:672: `self._last_guard_op and guard_op.getdescr() is None`
        // getdescr() is None only for optimizer-created guards in RPython.
        // Pyre stores resume snapshots in side tables keyed by
        // rd_resume_position; unroll clones those side-table entries and then
        // strips descrs to match opencoder.py.  A guard that already carries a
        // cloned rd_resume_position must therefore be finalized from its own
        // snapshot instead of sharing the previous guard's resume descr.
        // compile.py:925-926: GUARD_NOT_FORCED* must never share —
        // invent_fail_descr_for_op asserts copied_from_descr is None.
        let can_share = self.last_guard_idx.is_some()
            && !op.has_descr()
            && op.rd_resume_position.get() < 0
            && opnum != OpCode::GuardNotForced
            && opnum != OpCode::GuardNotForced2;

        if can_share {
            let idx = self.last_guard_idx.unwrap();
            // compile.py:832 ResumeGuardCopiedDescr(prev) parity: stamp
            // a `ResumeGuardCopiedDescr` whose `prev` references the
            // donor's descr.  Readers go through
            // `FailDescr::rd_*()` which chases `prev` automatically
            // (compile.py:849 `get_resumestorage(): return prev`).
            // GUARD_EXCEPTION / GUARD_NO_EXCEPTION mint the exc variant.
            //
            // optimizer.py:691 `assert isinstance(last_descr,
            // compile.ResumeGuardDescr)` — the donor must be a finalized
            // ResumeGuardDescr (or subclass).  RPython enforces this on
            // every sharing emit; pyre's standalone OptContext path
            // matched the production Optimizer in name only and used to
            // silently leave `op.descr = None` when the donor lacked a
            // descr.  Tighten to RPython parity.
            let donor_descr = self.new_operations[idx].getdescr().expect(
                "optimizer.py:691 assert isinstance(last_descr, \
                     ResumeGuardDescr): donor guard has no descr",
            );
            assert!(
                donor_descr.is_resume_guard(),
                "optimizer.py:691 assert isinstance(last_descr, \
                 ResumeGuardDescr): donor descr_index={} is not a \
                 ResumeGuardDescr subclass",
                donor_descr.index()
            );
            op.setdescr(match opnum {
                OpCode::GuardException | OpCode::GuardNoException => {
                    crate::compile::make_resume_guard_copied_exc_descr(donor_descr)
                }
                _ => crate::compile::make_resume_guard_copied_descr(donor_descr),
            });
            // optimizer.py:722: guard_op.setfailargs(last_guard_op.getfailargs())
            match self.new_operations[idx].getfailargs() {
                Some(fa) => op.setfailargs(fa.iter().cloned().collect()),
                None => op.clearfailargs(),
            }
            // bridgeopt.py parity: fail_arg_types carry the types the
            // serializer used when writing the class-knowledge bitfield in
            // rd_numb (memo.finish() uses numb_state.livebox_types). A
            // shared guard's rd_numb encodes the donor's livebox type
            // layout, so the sharer must inherit fail_arg_types too —
            // otherwise `deserialize_optimizer_knowledge` (bridgeopt.rs:911)
            // reconstructs a different Ref-set and reads past the buffer.
            match self.new_operations[idx].get_fail_arg_types() {
                Some(types) => op.set_fail_arg_types(types.to_vec()),
                None => op.clear_fail_arg_types(),
            }
            // optimizer.py:698-699: _maybe_replace_guard_value after copy.
            if op.opcode == OpCode::GuardValue {
                self.maybe_replace_guard_value(op);
            }
            // Don't update last_guard_idx — copied guards don't become sources.
        } else {
            // optimizer.py:678: store_final_boxes_in_guard.  This is
            // the standalone OptContext path (used by tests and the
            // direct ctx.emit_guard hook); it has no `pending_for_guard`
            // staging, so pass an empty Vec for the descr-side
            // set_rd_pendingfields write.
            self.store_final_boxes_in_guard(op, None, Vec::new());
            self.last_guard_idx = Some(self.new_operations.len());
            // optimizer.py:680-683: force_box on fail_args for unrolling.
            // Mirrors Optimizer.force_box contract: resolve replacement,
            // handle tracked preamble ops, force virtuals.
            if let Some(fa) = op.getfailargs() {
                let fargs: Vec<OpRef> = fa.iter().map(|b| b.to_opref()).collect();
                for farg in fargs {
                    if !farg.is_none() {
                        // regalloc.py:1206: Const objects skip forcing.
                        // Constant OpRefs may collide with virtual positions;
                        // forcing would corrupt the virtual's PtrInfo.
                        if !self
                            .get_box_replacement_operand_opt(farg)
                            .and_then(|cb| cb.const_value())
                            .is_some()
                        {
                            self.force_box_inline(farg);
                        }
                    }
                }
            }
            // optimizer.py:750-751: _maybe_replace_guard_value after store.
            if op.opcode == OpCode::GuardValue {
                self.maybe_replace_guard_value(op);
            }
        }

        // optimizer.py:684-685: GUARD_EXCEPTION clears sharing.
        if opnum == OpCode::GuardException {
            self.last_guard_idx = None;
        }
    }

    /// optimizer.py:754-778 _maybe_replace_guard_value — turn
    /// guard_value(bool) into guard_true/guard_false.
    fn maybe_replace_guard_value(&self, op: &mut Op) {
        let arg0 = op.arg(0);
        // optimizer.py:755: if op.getarg(0).type == 'i'
        let arg0_resolved = self.resolve_operand_box(&arg0).to_opref();
        if self.opref_type(arg0_resolved) != Some(majit_ir::Type::Int) {
            return;
        }
        // optimizer.py:756: b = self.getintbound(op.getarg(0))
        let Some(bound) = self
            .get_box_replacement_operand_opt(arg0_resolved)
            .and_then(|b| self.peek_intbound_box(&b))
        else {
            return;
        };
        if !bound.is_bool() {
            return;
        }
        let arg1 = op.arg(1);
        let Some(constvalue) = self
            .resolve_operand_operand_opt(&arg1)
            .and_then(|cb| cb.const_int())
        else {
            return;
        };
        let new_opcode = match constvalue {
            0 => OpCode::GuardFalse,
            1 => OpCode::GuardTrue,
            _ => return, // optimizer.py:775: strange code, just disable
        };
        // optimizer.py:803 newop = self.replace_op_with(op, opnum,
        //                                  [op.getarg(0)], descr)
        // — produce a fresh op with new opcode and trimmed args, descr
        // unchanged.  copy_and_change preserves fail_args / rd_resume_position
        // / fail_arg_types for guard ops (resoperation.py:498-503).
        *op = op.copy_and_change(new_opcode, Some(&[arg0]), None);
    }

    /// optimizer.py:345-364 force_box — inline equivalent for
    /// emit_guard_operation's fail_arg forcing (optimizer.py:680-683).
    /// Mirrors Optimizer.force_box contract: handle tracked preamble ops,
    /// then force virtuals to concrete. Body refs route through the preamble
    /// source directly, so the prior reverse-lookup 3rd key is no longer
    /// needed.
    fn force_box_inline(&mut self, opref: OpRef) -> OpRef {
        let resolved = self.get_replacement_opref(opref);
        // optimizer.py:351-359: a result that folded to an inline Const can
        // never be a `potential_extra_ops` key (the pool is keyed by the pure
        // op's result Box; the Const inlines at use sites instead of being
        // produced by a short box), so skip the short-preamble recording for
        // const-resolved results — otherwise the Const reaches `used_boxes`
        // and the carried label slot trips `OpRef::raw()` in unroll.rs.
        if !resolved.is_constant() {
            let tracked = self
                .take_potential_extra_op(resolved)
                .or_else(|| self.take_potential_extra_op(opref));
            if let Some(preamble_op) = tracked {
                // shortpreamble.py:434 `op = preamble_op.op.get_box_replacement()`
                // — the resolved Box itself is handed to the builder.
                let resolved_for_pop = self.resolve_box_box(&preamble_op.op);
                if let Some(builder) = self.active_short_preamble_producer_mut() {
                    builder.add_preamble_op_from_pop(&preamble_op, resolved_for_pop);
                } else if let Some(builder) = self.imported_short_preamble_builder.as_mut() {
                    builder.add_preamble_op_from_pop(&preamble_op, resolved_for_pop);
                }
            }
        }
        // optimizer.py:361-362: if op.type == 'i' and info.is_constant():
        //     return ConstInt(info.get_constant_int())
        // Mirrors Optimizer::force_box — a forced operand with an already-constant
        // IntBound materializes as ConstInt; peek the bound without installing.
        if let Some(rb) = self.get_box_replacement_operand_opt(resolved) {
            if rb.const_value().is_none() && rb.type_() == Type::Int {
                if let Some(bound) = self.peek_intbound_box(&rb) {
                    if bound.is_constant() {
                        return self.make_constant_int(bound.get_constant_int());
                    }
                }
            }
        }
        let resolved_op = self.get_box_replacement_operand_opt(opref);
        if let Some(mut info) = resolved_op.as_ref().and_then(|b| self.peek_ptr_info(b)) {
            if info.is_virtual() {
                let resolved_op = resolved_op
                    .clone()
                    .expect("is_virtual implies resolved_op is Some");
                let forced = info.force_box(&resolved_op, self);
                return self.get_replacement_opref(forced);
            }
        }
        resolved
    }

    /// RPython optimizer.py:722-752 store_final_boxes_in_guard inline.
    /// Called from emit() for every guard during optimization. Produces
    /// rd_numb via memo.number() using the CURRENT optimizer state
    /// (replacement chain, constants, virtual info).
    /// resume.py ResumeDataVirtualAdder.finish() parity:
    /// Generate rd_numb + rd_consts + rd_virtuals for a guard.
    /// Called from store_final_boxes_in_guard in optimizer.rs.
    /// Uses snapshot data (vable_boxes, frame_pcs, multi-frame) when available.
    pub fn finalize_guard_resume_data(
        &mut self,
        op: &mut Op,
        knowledge: Option<crate::resume::OptimizerKnowledgeForResume>,
        pending_setfields: Vec<majit_ir::GuardPendingFieldEntry>,
    ) {
        self.store_final_boxes_in_guard(op, knowledge, pending_setfields);
    }

    fn store_final_boxes_in_guard(
        &mut self,
        op: &mut Op,
        knowledge: Option<crate::resume::OptimizerKnowledgeForResume>,
        mut pending_setfields: Vec<majit_ir::GuardPendingFieldEntry>,
    ) {
        use crate::resume::{ResumeDataLoopMemo, Snapshot};

        // optimizer.py:722-730 store_final_boxes_in_guard parity:
        //   if op.getdescr() is not None:
        //       descr = op.getdescr()
        //       assert isinstance(descr, compile.ResumeGuardDescr)
        //   else:
        //       descr = compile.invent_fail_descr_for_op(op.getopnum(), self)
        //       op.setdescr(descr)
        //
        // RPython has exactly one emit path, so this function never
        // sees a `ResumeGuardCopiedDescr` (sibling, not subclass —
        // compile.py:832) nor an already-finalized `ResumeGuardDescr`
        // (resume.py:397 `assert not storage.rd_numb` ensures finish()
        // runs at most once per descr).
        //
        // OptContext::emit gates its emit_guard_operation on
        // `!in_final_emission` so production runs through
        // `Optimizer::emit_guard_operation` once; the OptContext path is
        // limited to the standalone test entry.  Either way only fresh
        // descrs reach this function.
        assert!(
            op.getdescr().map_or(true, |d| d.is_resume_guard()),
            "optimizer.py:723 store_final_boxes_in_guard expects \
             ResumeGuardDescr, got non-resume descr (kind={:?}, copied={})",
            op.getdescr().map(|d| d.index()),
            op.getdescr().map_or(false, |d| d.is_resume_guard_copied())
        );

        // resume.py:397 `assert not storage.rd_numb` — finish() runs at
        // most once per ResumeGuardDescr.  RPython makes this invariant
        // load-bearing: a second call would clobber an already-numbered
        // livebox set and break bridge attachment.  Promoted from
        // debug_assert! so release builds catch double-finish too.
        assert!(
            op.resolved_rd_numb().is_none(),
            "resume.py:397 finish() invoked twice on the same ResumeGuardDescr"
        );

        // resume.py:396-397:
        //   resume_position = self.guard_op.rd_resume_position
        //   assert resume_position >= 0
        // RPython: every guard has a valid rd_resume_position set by either
        // capture_resumedata (tracer guards) or patchguardop copy
        // (unroll.py:336/409). No fallback — the position is always set
        // before store_final_boxes_in_guard runs.
        let resume_pos = op.rd_resume_position.get();
        let has_snapshot = snapshot_contains(&self.snapshot_boxes, resume_pos);
        // resume.py:396-397: `assert resume_position >= 0` —
        // RPython asserts the position is set before calling
        // store_final_boxes_in_guard. Every guard from the production
        // pyre tracer captures its own snapshot via generate_guard /
        // capture_resumedata, and `gen_store_back_in_vable` inherits
        // the previous guard's snapshot id so its GUARD_NOT_FORCED_2
        // also has valid resume data. Hard-assert the invariant when
        // production snapshot data is wired through; the empty
        // `snapshot_boxes` case marks an isolated optimizer unit test
        // that constructs synthetic guards without going through the
        // pyre snapshot path, where the silent drop is acceptable.
        if !has_snapshot {
            // unroll.py:336/409 parity: when unroll creates a new guard from
            // a short preamble / virtual state import, it copies
            // rd_resume_position from patchguardop. If the new guard arrives
            // here without a snapshot, it must come from a patchguardop
            // context — inherit the patchguardop's resume_position.
            // resume.py:396-397: RPython asserts resume_position >= 0.
            let fallback_pos = self
                .patchguardop
                .as_ref()
                .map(|p| p.rd_resume_position.get())
                .filter(|&p| snapshot_contains(&self.snapshot_boxes, p));
            if let Some(fb_pos) = fallback_pos {
                op.rd_resume_position.set(fb_pos);
                // resume.py:570 _add_optimizer_sections: forward knowledge
                // to the patchguardop snapshot so heap/class/loopinvariant
                // sections are serialized into rd_numb. RPython's finish()
                // always serializes current optimizer knowledge regardless
                // of which snapshot provides the frame boxes.
                self.finalize_guard_resume_data(op, knowledge, pending_setfields);
                return;
            }
            // resume.py:396-397 parity:
            //   `resume_position = self.guard_op.rd_resume_position`
            //   `assert resume_position >= 0`
            // RPython asserts unconditionally here.  Tests that construct
            // guards directly must seed snapshot_boxes explicitly instead of
            // inventing a fail_args-derived fallback in this path.
            panic!(
                "store_final_boxes_in_guard: guard {:?} (pos={:?}, \
                 resume_pos={}) has no snapshot and no patchguardop \
                 ancestor — RPython resume.py:397 \
                 `assert resume_position >= 0` parity",
                op.opcode,
                op.pos.get(),
                op.rd_resume_position.get()
            );
        }

        // RPython parity: snapshot path handles ALL guards with snapshots,
        // including guards with rd_virtuals. The snapshot uses original boxes
        // and PtrInfo to correctly assign TAGVIRTUAL via _number_boxes.
        // _number_virtuals then builds rd_virtuals from PtrInfo.
        let snapshot_boxes = snapshot_get(&self.snapshot_boxes, op.rd_resume_position.get())
            .cloned()
            .unwrap_or_default();
        let vable_oprefs = snapshot_get(&self.snapshot_vable_boxes, op.rd_resume_position.get())
            .cloned()
            .unwrap_or_default();
        let vref_oprefs = snapshot_get(&self.snapshot_vref_boxes, op.rd_resume_position.get())
            .cloned()
            .unwrap_or_default();
        let frame_pcs = snapshot_get(&self.snapshot_frame_pcs, op.rd_resume_position.get())
            .cloned()
            .unwrap_or_default();

        // resume.py:201-202 get_box_replacement parity:
        // Pass ORIGINAL (unresolved) snapshot boxes. _number_boxes calls
        // env.get_box_replacement per-box, which resolves through the
        // replacement chain while preserving virtual identity.
        let frame_sizes = snapshot_get(&self.snapshot_frame_sizes, op.rd_resume_position.get());
        let mut snapshot = if let Some(sizes) = frame_sizes.filter(|s| s.len() > 1) {
            // Multi-frame: split snapshot_boxes into per-frame chunks.
            let mut frames = Vec::new();
            let mut offset = 0;
            for (i, &size) in sizes.iter().enumerate() {
                let end = (offset + size).min(snapshot_boxes.len());
                let frame_boxes: Vec<SnapshotBox> = snapshot_boxes[offset..end].to_vec();
                let (jitcode_index, pc, jitcode_pc) = frame_pcs.get(i).copied().unwrap_or((
                    0,
                    0,
                    majit_ir::resumedata::NO_JITCODE_PC,
                ));
                frames.push((jitcode_index, pc, jitcode_pc, frame_boxes));
                offset = end;
            }
            Snapshot::multi_frame_boxes_with_jitcode_pc(frames)
        } else {
            let (jitcode_index, pc, jitcode_pc) =
                frame_pcs
                    .first()
                    .copied()
                    .unwrap_or((0, 0, majit_ir::resumedata::NO_JITCODE_PC));
            Snapshot::single_frame_boxes_with_jitcode_pc(
                jitcode_index,
                pc,
                jitcode_pc,
                snapshot_boxes.clone(),
            )
        };
        // pyjitpl.py:2588: vable_array stores virtualizable_boxes.
        // ni/vsd are constants (TAGINT/TAGCONST) so they don't affect
        // TAGBOX numbering. The same OpRefs also appear in fail_args —
        // _number_boxes deduplicates via liveboxes HashMap.
        snapshot.vable_array = vable_oprefs;
        // resume.py:243-247 _number_boxes also reads vref_array as a
        // separate section after vable_array. opencoder.py:767
        // create_top_snapshot writes both arrays into the snapshot.
        snapshot.vref_array = vref_oprefs;

        if majit_log_enabled() && op.opcode == OpCode::GuardNotForced2 {
            let env = OptBoxEnv { ctx: self };
            let snapshot_debug: Vec<(OpRef, OpRef, bool, Type)> = snapshot_boxes
                .iter()
                .map(|boxref| {
                    let boxref = boxref.opref();
                    let resolved = self.get_replacement_opref(boxref);
                    let is_virtual = self
                        .get_box_replacement_operand_opt(boxref)
                        .as_ref()
                        .map_or(false, |b| self.is_virtual(b));
                    let tp = majit_ir::BoxEnv::get_type(&env, boxref);
                    (boxref, resolved, is_virtual, tp)
                })
                .collect();
            let vable_debug: Vec<(OpRef, OpRef, bool, Type)> = snapshot
                .vable_array
                .iter()
                .map(|boxref| {
                    let boxref = boxref.opref();
                    let resolved = self.get_replacement_opref(boxref);
                    let is_virtual = self
                        .get_box_replacement_operand_opt(boxref)
                        .as_ref()
                        .map_or(false, |b| self.is_virtual(b));
                    let tp = majit_ir::BoxEnv::get_type(&env, boxref);
                    (boxref, resolved, is_virtual, tp)
                })
                .collect();
            eprintln!(
                "[jit][guard-resume] pos={:?} snapshot={:?} vable={:?}",
                op.pos.get(),
                snapshot_debug,
                vable_debug
            );
        }

        // resume.py:389-452: delegate to ResumeDataVirtualAdder.finish()
        let env = OptBoxEnv { ctx: self };
        let mut memo = ResumeDataLoopMemo::new();
        let Ok(numb_state) = memo.number(&snapshot, &env, -1) else {
            return;
        };

        // resume.py:428-445, 520-558: pending_setfields are passed to finish()
        // which handles register_box, visitor_walk_recursive, and tagging.
        let (rd_numb, rd_consts, rd_virtuals, liveboxes, livebox_types) =
            memo.finish(numb_state, &env, &mut pending_setfields, knowledge.as_ref());

        if majit_log_enabled() && op.opcode == OpCode::GuardNotForced2 {
            eprintln!(
                "[jit][guard-resume] pos={:?} liveboxes={:?} rd_virtuals={} livebox_types={:?}",
                op.pos.get(),
                liveboxes,
                rd_virtuals.len(),
                livebox_types
            );
        }

        // RPython Box.type parity: types captured at numbering time via
        // env.get_type(), equivalent to RPython's intrinsic Box.type.
        // Replaces the fragile 7-level type resolution cascade.
        let new_types: Vec<majit_ir::Type> = liveboxes
            .iter()
            .map(|opref| {
                if opref.is_none() {
                    return majit_ir::Type::Ref;
                }
                livebox_types
                    .get(opref)
                    .copied()
                    .unwrap_or(majit_ir::Type::Ref)
            })
            .collect();

        // optimizer.py:712 liveboxes are the canonical Box objects returned
        // by `resumedata.finish()`; resolve each numbering position to its
        // canonical (possibly producer-bound) box so `store_final_boxes`
        // can shed to a live-tracking operand. NONE holes and positions
        // with no canonical box stay position-only.
        let liveboxes_b: Vec<majit_ir::box_ref::BoxRef> = liveboxes
            .iter()
            .map(|a| {
                self.resolve_to_boxref(*a)
                    .unwrap_or_else(|| majit_ir::box_ref::BoxRef::from_opref(*a))
            })
            .collect();
        op.store_final_boxes(liveboxes_b);
        op.set_fail_arg_types(new_types.clone());
        // optimizer.py:722-730 `store_final_boxes_in_guard` parity:
        //   if op.getdescr() is not None:
        //       descr = op.getdescr()
        //       assert isinstance(descr, compile.ResumeGuardDescr)
        //   else:
        //       descr = compile.invent_fail_descr_for_op(op.getopnum(), self)
        //       op.setdescr(descr)
        // RPython preserves the existing descr object (and its
        // `fail_index`, subtype, vector_info) and only mutates its
        // `fail_arg_types`. Pyre's MetaFailDescr / ResumeGuardDescr /
        // ResumeAtPositionDescr / CompileLoopVersionDescr keep `types`
        // in `UnsafeCell<Vec<Type>>`, exposed via
        // `FailDescr::set_fail_arg_types`, so we mutate in place — the
        // load-bearing contract that subtype markers
        // (`is_resume_at_position()`, `loop_version()`) survive
        // `store_final_boxes_in_guard` (compile.py:1035-1043, mirrored
        // at pyjitpl.rs:6799 `is_resume_at_position()`).
        match op.getdescr() {
            Some(existing) => {
                if let Some(fd) = existing.as_fail_descr() {
                    fd.set_fail_arg_types(new_types);
                }
            }
            None => {
                // RPython compile.py:919-937 `invent_fail_descr_for_op`
                // dispatches on opcode:
                //   GUARD_NOT_FORCED / GUARD_NOT_FORCED_2 → ResumeGuardForcedDescr
                //   GUARD_EXCEPTION  / GUARD_NO_EXCEPTION → ResumeGuardExcDescr
                //   else                                  → ResumeGuardDescr
                // The exception-flow / async-forcing special cases at
                // `pyjitpl.rs` opcode-check sites (e.g. the
                // GUARD_EXCEPTION → `is_exception_guard` and
                // GUARD_NOT_FORCED chains) can migrate to descr-keyed
                // dispatch via `is_guard_exc()` / `is_guard_forced()`
                // without reshaping this match arm.
                use majit_ir::OpCode;
                op.setdescr(match op.opcode {
                    OpCode::GuardNotForced | OpCode::GuardNotForced2 => {
                        crate::compile::make_resume_guard_forced_descr_typed(new_types)
                    }
                    OpCode::GuardException | OpCode::GuardNoException => {
                        crate::compile::make_resume_guard_exc_descr_typed(new_types)
                    }
                    _ => crate::compile::make_resume_guard_descr_typed(new_types),
                });
            }
        }
        // compile.py:855 ResumeGuardDescr `_attrs_` parity: write the
        // post-numbering resume payload onto the descr that
        // store_final_boxes_in_guard just minted (or onto the existing
        // ResumeGuardDescr / ResumeAtPositionDescr / ResumeGuardForcedDescr
        // / ResumeGuardExcDescr that capture_resumedata stamped earlier).
        // The descr is the single source of truth — readers go through
        // FailDescr::rd_*().
        let descr_rd_virtuals = if rd_virtuals.is_empty() {
            None
        } else {
            Some(rd_virtuals)
        };
        let descr_pending = if pending_setfields.is_empty() {
            None
        } else {
            Some(pending_setfields)
        };
        let __descr_arc = op.getdescr();
        if let Some(fd) = __descr_arc.as_ref().and_then(|d| d.as_fail_descr()) {
            fd.set_rd_numb(Some(rd_numb));
            fd.set_rd_consts(Some(rd_consts));
            fd.set_rd_virtuals(descr_rd_virtuals);
            fd.set_rd_pendingfields(descr_pending);
        }
        // resume.py: RPython does NOT carry frame sizes out-of-band.
        // The decoder reads jitcode liveness (jitcode.position_info) at
        // each frame's resume pc. majit routes this through the global
        // `frame_value_count_at` callback registered by pyre-jit-trace.
        let _ = frame_sizes;
    }

    /// Construct a constant OpRef whose value rides inline on the variant
    /// tag.
    ///
    /// RPython equivalent: `ConstInt(value)` — constants in RPython are
    /// first-class Const objects, not boxes. Post-S0 inline-Const the value
    /// is carried on the `OpRef::const_*` variant itself: no pool
    /// reservation and no `seed_constant` step (its const arm is a
    /// tautological no-op, asserting only that the variant tag the caller
    /// already minted from `value` matches `value`'s type).
    ///
    /// NOTE: do NOT route through `make_constant`. That helper is the
    /// `optimizer.py:make_constant(box, constbox)` analogue and is meant
    /// to forward an existing **box** OpRef to a constant value. It bails
    /// out early when the input is already a constant OpRef
    /// (`is_constant()` true), which would silently drop the new entry.
    pub fn make_constant_int(&mut self, value: i64) -> OpRef {
        // history.py:227 ConstInt.value inline.
        OpRef::const_int(value)
    }

    pub fn make_constant_ref(&mut self, value: GcRef) -> OpRef {
        // history.py:314 ConstPtr.value inline. The GC walker forwards the
        // inline GcRef across minor collection.
        OpRef::const_ptr(value)
    }

    pub fn make_constant_float(&mut self, value: f64) -> OpRef {
        // history.py:268 ConstFloat.value inline.
        OpRef::const_float(value)
    }

    /// Look up the operation that produces a given box.
    /// Used for pattern matching nested operations (e.g., int_add(int_add(x, C1), C2)).
    /// Returns a clone to avoid borrow conflicts with mutable ctx methods.
    ///
    /// optimizer.py:369-377 `as_operation` ("You should never check
    /// isinstance(op, AbstractResOp) directly. Instead, use this
    /// helper.") admits a producer only when `op in
    /// self._emittedoperations` — ops emitted by THIS optimizer run.
    /// Without the gate, a Phase-2 rule can follow a label arg to its
    /// Phase-1 producer and re-express a per-iteration value in terms
    /// of the preamble's entry box (e.g. int_add reassociation turning
    /// `loop_arg - 1` into `preamble_entry - 2`), a box the loop header
    /// does not carry per-iteration.
    pub fn get_producing_op(&self, op: &Operand) -> Option<Op> {
        // resoperation.py:233 `_forwarded` host: a box's producing op is its
        // bound op (set at emit, mod.rs bind_op before new_operations.push).
        // Walk the forwarding chain first (resoperation.py:58) so the
        // replacement box's producer is read.
        let producer = op.get_box_replacement(false).bound_op()?;
        // optimizer.py:369-377 `op in self._emittedoperations` — keyed by the
        // producer's box identity (`from_bound_op` memoizes one box per op Rc,
        // so this is the same box recorded at emit).
        if !self
            .emitted_operations
            .contains(&majit_ir::operand::Operand::from_bound_op(&producer))
        {
            return None;
        }
        Some((*producer).clone())
    }

    /// Number of emitted operations so far.
    pub fn num_emitted(&self) -> usize {
        self.new_operations.len()
    }

    /// Get the last emitted operation, if any.
    pub fn last_emitted_operation(&self) -> Option<&Op> {
        self.new_operations.last().map(|rc| rc.as_ref())
    }

    /// `optimizer.py:379-387 get_constant_box`:
    /// ```python
    /// def get_constant_box(self, box):
    ///     box = get_box_replacement(box)
    ///     if isinstance(box, Const):
    ///         return box
    ///     if box.type == 'i':
    ///         info = box.get_forwarded()
    ///         if isinstance(info, IntBound) and info.is_constant():
    ///             return ConstInt(info.get_constant_int())
    ///     return None
    /// ```
    pub fn get_constant_box(&self, op: &Operand) -> Option<Value> {
        // optimizer.py:380: box = get_box_replacement(box)
        let resolved = op.get_box_replacement(false);
        // optimizer.py:381-382: isinstance(box, Const) → return box
        if let Some(v) = resolved.const_value() {
            return Some(v);
        }
        // optimizer.py:383-386: box.type == 'i' + IntBound + is_constant
        if resolved.type_() == majit_ir::Type::Int {
            if let Some(b) = resolved.int_bound() {
                if b.is_constant() {
                    return Some(Value::Int(b.get_constant_int()));
                }
            }
        }
        None
    }

    pub fn get_constant_int_box(&self, op: &Operand) -> Option<i64> {
        match self.get_constant_box(op)? {
            Value::Int(i) => Some(i),
            _ => None,
        }
    }

    pub fn get_constant_float_box(&self, op: &Operand) -> Option<f64> {
        match self.get_constant_box(op)? {
            Value::Float(f) => Some(f),
            _ => None,
        }
    }

    /// `isinstance(opref, Const)` parity — True only when the OpRef itself
    /// is in the constant namespace. Does NOT walk the forwarding chain;
    /// a body-namespace OpRef forwarded to a Const via `make_constant` returns
    /// None here. Use `get_constant_box` for `getrawconstoption` semantics
    /// (includes IntBound synthesis and chain walk).
    pub fn isinstance_const(&self, opref: OpRef) -> Option<Value> {
        if !opref.is_constant() {
            return None;
        }
        // Const variants carry the value in the variant itself
        // (history.py:227 `Const.value`).
        opref.inline_const_to_value()
    }

    /// `isinstance(opref, ConstInt)` parity — narrow check without chain walk.
    pub fn isinstance_const_int(&self, opref: OpRef) -> Option<i64> {
        match self.isinstance_const(opref)? {
            Value::Int(i) => Some(i),
            _ => None,
        }
    }

    /// optimizer.py:810-816 `constant_fold(op)`:
    ///
    /// ```python
    /// def constant_fold(self, op):
    ///     self.protect_speculative_operation(op)
    ///     argboxes = [self.get_constant_box(op.getarg(i))
    ///                 for i in range(op.numargs())]
    ///     return execute_nonspec_const(self.cpu, None,
    ///                                    op.getopnum(), argboxes,
    ///                                    op.getdescr(), op.type)
    /// ```
    ///
    /// Returns `None` only when:
    ///  - `supports_guard_gc_type == false` and the op is a memory-
    ///    reading fold (array/string/unicode).  Upstream `optimizer.py:
    ///    822-825` relies on "we don't unroll in that case"; pyre's
    ///    `constant_fold` runs outside the unroll pass too, so this
    ///    gate is placed here at the call site, NOT inside
    ///    `protect_speculative_operation` (which matches upstream as a
    ///    plain `()` function).
    ///  - Helper-internal `Ok(None)` for OVF/shift/divide-by-zero/
    ///    non-finite cast (see `pure.rs:993`).
    /// Every other path panics (caller-invariant, NotImplemented).
    pub fn constant_fold(&self, op: &Op) -> Option<Value> {
        // optimizer.py:822-825: "if cpu.supports_guard_gc_type is
        // false, we can't really do this check at all, but then we
        // don't unroll in that case."  Gate memory-reading ops here
        // so protect_speculative_operation stays a plain () function.
        if !majit_gc::supports_guard_gc_type() {
            use majit_ir::OpCode;
            if matches!(
                op.opcode,
                OpCode::GetarrayitemGcPureI
                    | OpCode::GetarrayitemGcPureR
                    | OpCode::GetarrayitemGcPureF
                    | OpCode::ArraylenGc
                    | OpCode::Strgetitem
                    | OpCode::Strlen
                    | OpCode::Unicodegetitem
                    | OpCode::Unicodelen
            ) {
                return None;
            }
        }
        self.protect_speculative_operation(op);
        // protect_speculative_operation defers an `InvalidLoop` instead of
        // raising; bail before reading the (now unvalidated) memory so the
        // fold never dereferences an ill-typed speculative pointer.
        if self.has_pending_invalid_loop() {
            return None;
        }
        let mut argboxes: Vec<Value> = Vec::with_capacity(op.num_args());
        for i in 0..op.num_args() {
            argboxes.push(
                self.get_constant_box(&op.arg(i).get_box_replacement(false))
                    .expect("constant_fold: arg must be Const (pure.rs:993-1006 pre-check)"),
            );
        }
        match crate::executor::execute_nonspec_const(
            self.cpu.as_ref(),
            op.opcode,
            &argboxes,
            op.getdescr().as_ref(),
            op.result_type(),
        ) {
            Ok(folded) => folded,
            Err(crate::executor::NotImplemented) => panic!(
                "execute_nonspec_const: no helper registered for opcode {:?} \
                 (executor.py:610 NotImplementedError)",
                op.opcode
            ),
        }
    }

    /// optimizer.py:818-867 `protect_speculative_operation(op)` — when
    /// constant-folding a pure operation that reads memory from a
    /// gcref, validate the gcref is non-null and of a valid type;
    /// signal `InvalidLoop` otherwise.
    ///
    /// Returns `()` — matching upstream's Python `def protect_
    /// speculative_operation(self, op):` which has no return value.
    /// Either returns normally (validation passed) or records a deferred
    /// `InvalidLoop` signal on the context (via `signal_invalid_loop`),
    /// which `constant_fold` observes to bail out and the driver surfaces
    /// as `Err(InvalidLoop)` at its next barrier (`unroll.py:119-123`).
    ///
    /// The `supports_guard_gc_type == false` gate that was previously
    /// inside this function has been moved to `constant_fold` (the
    /// only caller), matching upstream's architectural invariant:
    /// *"if cpu.supports_guard_gc_type is false, we can't really do
    /// this check at all, but then we don't unroll in that case"*
    /// (optimizer.py:822-825).
    ///
    /// Caller-invariant violations (missing box, descr, wrong Value
    /// variant) panic — upstream would `AttributeError`.
    ///
    /// Branches mirror the upstream `if / elif / elif / elif / else`
    /// chain line-for-line:
    ///  - pure GETFIELD_GC_PURE_*  → `protect_speculative_field`
    ///  - GETARRAYITEM_GC_PURE_* / ARRAYLEN_GC → `protect_speculative_array`
    ///  - STRGETITEM / STRLEN → `protect_speculative_string`
    ///  - UNICODEGETITEM / UNICODELEN → `protect_speculative_unicode`
    ///  - default → no validation needed (return early).
    ///
    /// For the get*item branches, `cpu.bh_arraylen_gc / bh_strlen /
    /// bh_unicodelen` reads the container length and the routine
    /// checks `0 <= index < length`.  When `bh_strlen / bh_unicodelen`
    /// returns `None` (pyre has no fold-time str/unicode layout), the
    /// bounds check is skipped — equivalent to RPython where the
    /// optimizer falls back to runtime evaluation in that case.
    fn protect_speculative_operation(&self, op: &Op) {
        use majit_ir::OpCode;

        let opnum = op.opcode;
        let arraylength: i64;

        let descr = op.getdescr();
        if opnum.is_getfield() {
            // optimizer.py:829-832 pure-getfield branch.
            let gcref = match self
                .get_constant_box(&op.arg(0).get_box_replacement(false))
                .expect("protect_speculative_operation: arg0 must be Const")
            {
                Value::Ref(r) => r,
                v => unreachable!(
                    "GETFIELD_GC_PURE_* arg0 must be a gcref (Value::Ref); got {:?}",
                    v
                ),
            };
            let fd = descr
                .as_ref()
                .and_then(|d| d.as_field_descr())
                .expect("GETFIELD_GC_PURE_* descr must be a FieldDescr");
            if self.cpu.protect_speculative_field(gcref, fd).is_err() {
                self.signal_invalid_loop("protect_speculative_field");
            }
            return;
        }

        if matches!(
            opnum,
            OpCode::GetarrayitemGcPureI
                | OpCode::GetarrayitemGcPureR
                | OpCode::GetarrayitemGcPureF
                | OpCode::ArraylenGc
        ) {
            // optimizer.py:834-841 array branch.
            let array = match self
                .get_constant_box(&op.arg(0).get_box_replacement(false))
                .expect("protect_speculative_operation: array arg0 must be Const")
            {
                Value::Ref(r) => r,
                v => unreachable!(
                    "GETARRAYITEM_GC_PURE_* / ARRAYLEN_GC arg0 must be a gcref; got {:?}",
                    v
                ),
            };
            let ad = descr
                .as_ref()
                .and_then(|d| d.as_array_descr())
                .expect("array op descr must be an ArrayDescr");
            if self.cpu.protect_speculative_array(array, ad).is_err() {
                self.signal_invalid_loop("protect_speculative_array");
                return;
            }
            if opnum == OpCode::ArraylenGc {
                return;
            }
            arraylength = self
                .cpu
                .bh_arraylen_gc(array, ad)
                .expect("bh_arraylen_gc must succeed after protect_speculative_array");
        } else if matches!(opnum, OpCode::Strgetitem | OpCode::Strlen) {
            // optimizer.py:843-848 string branch.
            let string = match self
                .get_constant_box(&op.arg(0).get_box_replacement(false))
                .expect("protect_speculative_operation: string arg0 must be Const")
            {
                Value::Ref(r) => r,
                v => unreachable!("STRGETITEM / STRLEN arg0 must be a gcref; got {:?}", v),
            };
            if self.cpu.protect_speculative_string(string).is_err() {
                self.signal_invalid_loop("protect_speculative_string");
                return;
            }
            if opnum == OpCode::Strlen {
                return;
            }
            arraylength = self
                .cpu
                .bh_strlen(string)
                .expect("bh_strlen must succeed after protect_speculative_string");
        } else if matches!(opnum, OpCode::Unicodegetitem | OpCode::Unicodelen) {
            // optimizer.py:850-855 unicode branch.
            let unicode = match self
                .get_constant_box(&op.arg(0).get_box_replacement(false))
                .expect("protect_speculative_operation: unicode arg0 must be Const")
            {
                Value::Ref(r) => r,
                v => unreachable!(
                    "UNICODEGETITEM / UNICODELEN arg0 must be a gcref; got {:?}",
                    v
                ),
            };
            if self.cpu.protect_speculative_unicode(unicode).is_err() {
                self.signal_invalid_loop("protect_speculative_unicode");
                return;
            }
            if opnum == OpCode::Unicodelen {
                return;
            }
            arraylength = self
                .cpu
                .bh_unicodelen(unicode)
                .expect("bh_unicodelen must succeed after protect_speculative_unicode");
        } else {
            // optimizer.py:857-858 else: return — nothing to validate.
            return;
        }

        // optimizer.py:860-862 shared bounds check:
        //   index = self.get_constant_box(op.getarg(1)).getint()
        //   if not (0 <= index < arraylength): raise SpeculativeError
        let index = match self
            .get_constant_box(&op.arg(1).get_box_replacement(false))
            .expect("protect_speculative_operation: arg1 must be Const")
        {
            Value::Int(i) => i,
            v => unreachable!(
                "GETARRAYITEM / STRGETITEM / UNICODEGETITEM arg1 must be an int index; got {:?}",
                v
            ),
        };
        if !(0 <= index && index < arraylength) {
            self.signal_invalid_loop("index out of bounds for constant-fold");
        }
    }

    /// Look up the producing `Op` for an OpRef in `new_operations`.
    /// Returns `None` for inputargs, constants, and OpRefs not yet emitted.
    ///
    /// RPython equivalent: holding a reference to the producing `Box`
    /// itself (every Box is a Python object, so identity lookup is the
    /// `is` operator — O(1)). pyre's flat `OpRef(u32)` cannot mirror
    /// that in O(1) without an auxiliary index; mutation patterns on
    /// `new_operations` (in-place replace at `optimizer.rs:3391`,
    /// `rewrite.rs:1579/1674`, plus `remove(jump_idx)` at
    /// `optimizer.rs:2605`) make a maintained `pos_to_index` brittle.
    /// unifies on this single API. A maintained index would need to be kept
    /// in sync with every in-place replacement and removal site.
    pub fn op_at(&self, opref: OpRef) -> Option<&Op> {
        if let Some(op) = self
            .new_operations
            .iter()
            .rev()
            .find(|op| op.pos.get() == opref)
        {
            return Some(op);
        }
        // Phase 1 emit-op fallback (history.py:220 box.type parity for
        // cross-phase OpRefs). Reverse scan mirrors `new_operations` so a
        // later replacement of the same `pos` wins. Returned `&Op` is
        // safe to read for `.type_` and other intrinsic attributes; arg
        // / descr fields refer to Phase 1's namespace and should not be
        // dereferenced through this path (Phase 2 callers only consume
        // `op.type_` via `opref_type`).
        self.phase1_emit_ops
            .iter()
            .rev()
            .find(|op| op.pos.get() == opref)
            .map(|rc| rc.as_ref())
    }

    /// optimizer.py: clear_newoperations()
    /// Clear the output operation list (used when restarting optimization).
    pub fn clear_newoperations(&mut self) {
        self.new_operations.clear();
        // Reset next_pos to the iteration's first fresh OpRef position
        // (right after the inputarg slice in the OpRef namespace), but
        // never below the prior iteration's watermark. The context
        // survives across iterations (e.g. Phase 2 final_ctx reused as
        // `jump_ctx` for short-preamble inlining); `reserve_pos` only
        // skips constants, so the previous iteration's typed slots must
        // remain reserved to keep `alloc_op_position` from handing back
        // a pos that already names an emitted op.
        let base = self.inputarg_base + self.num_inputs;
        self.next_pos = self.next_pos.max(base);
        self.const_infos.clear();
    }

    /// resoperation.py: `op.type` parity. The typed OpRef enum
    /// encodes `box.type` (`AbstractValue.type` ∈ {`'i'`, `'r'`, `'f'`,
    /// `'v'`}) directly in the variant tag, so reading the tag is the
    /// line-by-line equivalent of upstream `box.type`. The fall-through
    /// arms cover residual cases — ops that have not yet been emitted,
    /// inputarg slots, and PtrInfo-derived Ref typing. Raw-pointer
    /// `ConstInt` Boxes keep `op.type == 'i'` and become `ConstPtrInfo`
    /// through `getrawptrinfo` per `info.py:870-871`.
    ///
    /// Returns `None` only when none of the above sources have type
    /// information for the OpRef. Callers must treat `None` like
    /// RPython's "unknown type" path and avoid making structural
    /// assumptions about it.
    pub fn opref_type(&self, opref: OpRef) -> Option<majit_ir::Type> {
        let resolved = self.get_replacement_opref(opref);
        // 0. Inputarg slot (recorder-side `InputArg{Int,Ref,Float}.tp`,
        //    history.py:220 parity per resoperation.py:719/727/739).
        //    `inputarg_types[idx]` is the canonical Box.type source
        //    for slot positions — a cross-phase caller that minted the
        //    OpRef with `input_arg_int(idx)` for a Ref-typed slot would
        //    mismatch the variant tag against the recorder's actual
        //    type, so consult the side-table first for inputarg
        //    positions.  Returns `None` for non-inputarg OpRefs, which
        //    falls through to the variant-tag step.
        if let Some(tp) = self.inputarg_type(resolved) {
            return Some(tp);
        }
        // 1. RPython `AbstractValue.type` (resoperation.py:29) parity. The
        //    OpRef enum encodes `box.type` directly in the variant tag
        //    (`ConstInt`/`InputArgInt`/`IntOp` → Int, etc.), so reading
        //    the tag is the line-by-line equivalent of upstream `box.type`.
        //    `OpRef::None` returns `None` here and falls through.
        if let Some(tp) = resolved.ty() {
            return Some(tp);
        }
        // 2. Producing op's intrinsic `type_` (resoperation.py:1693
        //    `res.type = result_type` in `create_class_for_op`, i.e.
        //    `opclasses[opnum].type`). `Op::new` populates this at
        //    construction; this is the primary fast path after allocation.
        //    Reached only when `resolved` is `None`/`TempVar` (every other
        //    variant is typed by step 1); inline-Const collapsed the old
        //    pool-indexed "seeded constant without a typed variant" case
        //    into step 1, so no separate const-value fall-through remains.
        if let Some(op) = self.op_at(resolved) {
            if op.type_ != majit_ir::Type::Void {
                return Some(op.type_);
            }
        }
        // 3. PtrInfo-derived type (box.type parity for
        //    virtual heads across phase boundaries). Phase 1 virtualizes
        //    NewWithVtable / New / NewArray etc. by returning
        //    `OptimizationResult::Remove` from the relevant `optimize_*`
        //    method (virtualize.py:208-225) — the op never lands in
        //    `new_operations` and is therefore absent from
        //    `phase1_emit_ops`. Phase 2 imports the virtual head's
        //    `PtrInfo` via `setinfo_from_preamble` (unroll.py:55-64) but
        //    starts with a fresh `value_types` map, so the prior four
        //    sources all miss. RPython preserves the type intrinsically
        //    on the Box object; pyre recovers it from the PtrInfo variant
        //    because every variant maps to a unique RPython box type.
        //    Ref-typed: Virtual / VirtualArray / VirtualStruct /
        //    VirtualArrayStruct / Array / Struct / Instance / Constant /
        //    NonNull / Virtualizable / Str (RPython instances of
        //    AbstractStructPtrInfo / ArrayPtrInfo / StrPtrInfo carry 'r').
        //    Int-typed: VirtualRawBuffer / VirtualRawSlice
        //    (info.py:865 RawBufferPtrInfo + getrawptrinfo() — these
        //    describe raw pointers stored in 'i' Boxes).
        let resolved_box = self.get_box_replacement_operand_opt(opref);
        if let Some(info) = resolved_box.as_ref().and_then(|b| self.peek_ptr_info(b)) {
            return Some(match info {
                crate::optimizeopt::info::PtrInfo::VirtualRawBuffer(_)
                | crate::optimizeopt::info::PtrInfo::VirtualRawSlice(_) => majit_ir::Type::Int,
                _ => majit_ir::Type::Ref,
            });
        }
        None
    }

    /// Read-only `box.type` lookup for inputarg-slot OpRefs.  Returns
    /// `Some(tp)` when `opref` falls in either the current context's
    /// own inputarg range `[inputarg_base, inputarg_base + num_inputs)`
    /// (RPython invariant) or the shared low range `[0, num_inputs)`
    /// (Phase 1's inputarg slot OpRefs referenced from Phase 2 via
    /// `imported_label_args`).  Returns `None` for constants, sentinels,
    /// out-of-range OpRefs, Void-typed slots, or empty `inputarg_types`
    /// (test contexts that bypass `setup_optimizations`).
    ///
    /// `[0, num_inputs)` fallback: in RPython each `InputArgInt`/
    /// `InputArgRef`/`InputArgFloat` (resoperation.py:719/727/739) Box
    /// carries `.type` intrinsically, so Phase 2 reads the same
    /// `box.type` regardless of which iteration's TraceIterator produced
    /// the box.  Pyre's flat `OpRef(u32)` namespace separates Phase 1
    /// inputargs (at `[0, num_inputs)`) from Phase 2 inputargs (at
    /// `[phase2_inputarg_base, phase2_inputarg_base + num_inputs)`), but
    /// `Optimizer.trace_inputargs` is identical between phases
    /// (single recorder source).  Indexing the same `inputarg_types`
    /// Vec by raw position recovers Phase 1 slot types from Phase 2
    /// without a separate side-table (history.py:220 parity).
    ///
    /// TODO: pyre stores the InputArg type on a
    /// graph-level side-table instead of a per-Box `BoxKind::InputArg`
    /// variant tag because the Box layout splits ResOp / InputArg /
    /// Const at construction time only.  Retiring this helper requires
    /// stamping the type onto `BoxKind::InputArg` so a `BoxRef.type_()`
    /// read on an existing Box is sufficient.  Until then this lookup
    /// is the read-only counterpart of `resoperation.py:719/727/739
    /// InputArg{Int,Ref,Float}.type` — it must not materialize a fresh
    /// Box, because the materialization path keys the new Box's type
    /// off the OpRef variant tag (mod.rs:3791) and a Phase 2 context
    /// referencing a Phase 1 low slot can mismatch that tag against
    /// the canonical `inputarg_types[idx]`.
    pub fn inputarg_type(&self, opref: OpRef) -> Option<majit_ir::Type> {
        if opref.is_none() || opref.is_constant() {
            return None;
        }
        let raw = opref.raw();
        let ni = self.num_inputs as usize;
        let idx = if raw >= self.inputarg_base && (raw - self.inputarg_base) < self.num_inputs {
            (raw - self.inputarg_base) as usize
        } else if self.inputarg_base > 0 && raw < self.num_inputs {
            // Phase 1 inputarg slot accessed from a non-Phase-1 context
            // (Phase 2 / bridge).  RPython resolves these through Box
            // identity; flat-OpRef pyre uses the shared `inputarg_types`
            // Vec — no materialization, just the recorder-seeded type
            // table.
            raw as usize
        } else {
            return None;
        };
        if idx >= ni {
            return None;
        }
        let opref = *self.inputargs.get(idx)?;
        let tp = opref.ty()?;
        if tp == majit_ir::Type::Void {
            None
        } else {
            Some(tp)
        }
    }

    /// Look up the declared type of inputarg slot `idx` (zero-based) from
    /// the `inputargs` Vec seeded by `setup_optimizations`. Returns
    /// `None` if the slot is out of range, the type is `Void`, or the Vec
    /// has not been populated. Reads `inputargs[idx].ty()` — each entry
    /// is a typed `OpRef::input_arg_*` (optimizer.py:34 parity).
    pub fn inputarg_type_at(&self, idx: usize) -> Option<majit_ir::Type> {
        let opref = *self.inputargs.get(idx)?;
        let tp = opref.ty()?;
        if tp == majit_ir::Type::Void {
            None
        } else {
            Some(tp)
        }
    }

    /// Strict counterpart to `inputarg_type_at`. Panics when the slot is
    /// out of range, the variant yields `Void`, or `inputargs` was not
    /// populated by `setup_optimizations`. Mirrors RPython's
    /// `box.type` invariant (history.py:220).
    pub fn inputarg_type_at_strict(&self, idx: usize) -> majit_ir::Type {
        match self.inputargs.get(idx).and_then(|o| o.ty()) {
            Some(majit_ir::Type::Void) => panic!(
                "inputarg_type_at_strict: slot {idx} is Void; \
                 RPython invariant violated (history.py:220 box.type)"
            ),
            Some(tp) => tp,
            None => panic!(
                "inputarg_type_at_strict: slot {idx} out of range \
                 (inputargs.len() = {}); setup_optimizations did not \
                 seed the inputarg list",
                self.inputargs.len()
            ),
        }
    }

    /// Read-only access to the inputarg slot's typed `OpRef` (variant tag
    /// is `box.type`). Returns `None` when the slot is out of range.
    /// Mirrors PyPy `self.inputargs[idx]` (optimizer.py:34).
    pub fn inputarg_at(&self, idx: usize) -> Option<majit_ir::OpRef> {
        self.inputargs.get(idx).copied()
    }

    /// info.py:865-878 `getrawptrinfo(op)` parity (line-by-line port).
    ///
    /// ```python
    /// def getrawptrinfo(op):
    ///     from rpython.jit.metainterp.optimizeopt.intutils import IntBound
    ///     assert op.type == 'i'
    ///     op = op.get_box_replacement()
    ///     assert op.type == 'i'
    ///     if isinstance(op, ConstInt):
    ///         return ConstPtrInfo(op)
    ///     fw = op.get_forwarded()
    ///     if isinstance(fw, IntBound):
    ///         return None
    ///     if fw is not None:
    ///         assert isinstance(fw, AbstractRawPtrInfo)
    ///         return fw
    ///     return None
    /// ```
    ///
    /// info.py:865-878 `getrawptrinfo(op)` parity — orthodox return
    /// shape that preserves RPython `_forwarded` object identity.
    /// `PtrInfoHandle::Const(_)` for the `isinstance(op, ConstInt)`
    /// fresh `ConstPtrInfo` arm; `PtrInfoHandle::Live(rc)` for the
    /// `return fw` arm carrying the live `Rc<RefCell<PtrInfo>>` cell
    /// from the chain terminal's `_forwarded` slot.
    ///
    /// Callers that need an owned snapshot can call `.snapshot()`;
    /// callers that need identity/value parity (`same_info`, in-place
    /// mutation) use `.same_info()` / `.borrow()` / `.borrow_mut()`.
    pub fn getrawptrinfo_handle(&self, op: &Operand) -> Option<PtrInfoHandle> {
        use crate::optimizeopt::info::OpInfo;
        use majit_ir::box_ref::Forwarded;
        // info.py:867 — `assert op.type == 'i'`.
        debug_assert_eq!(
            op.type_(),
            majit_ir::Type::Int,
            "getrawptrinfo_handle: expected 'i'-typed BoxRef"
        );
        // info.py:868 — `op = op.get_box_replacement()`.
        let terminal = op.get_box_replacement(false);
        // info.py:869 — `assert op.type == 'i'`.
        debug_assert_eq!(
            terminal.type_(),
            majit_ir::Type::Int,
            "getrawptrinfo_handle: terminal expected 'i'-typed BoxRef"
        );
        // info.py:870-871 — `if isinstance(op, ConstInt): return ConstPtrInfo(op)`.
        if let Some(Value::Int(bits)) = terminal.const_value() {
            return Some(PtrInfoHandle::Const(PtrInfo::Constant(majit_ir::GcRef(
                bits as usize,
            ))));
        }
        // info.py:872-878 line-by-line dispatch on the forwarded slot:
        //     fw = op.get_forwarded()
        //     if isinstance(fw, IntBound): return None
        //     if fw is not None:
        //         assert isinstance(fw, AbstractRawPtrInfo)
        //         return fw
        //     return None
        match &terminal.get_forwarded() {
            Forwarded::None => None,
            Forwarded::Info(OpInfo::IntBound(_)) => None,
            Forwarded::Info(OpInfo::Ptr(rc)) => Some(PtrInfoHandle::Live(std::rc::Rc::clone(rc))),
            // info.py:876 `assert isinstance(fw, AbstractRawPtrInfo)` —
            // a non-Ptr, non-IntBound forwarded on an `'i'`-typed terminal
            // is a structural invariant violation upstream would crash on.
            Forwarded::Info(other) => panic!(
                "getrawptrinfo: forwarded must be IntBound or AbstractRawPtrInfo \
                 (info.py:876), got {:?}",
                std::mem::discriminant(other),
            ),
            // Terminal of `get_box_replacement(false)` can only be `None`
            // or `Info(_)` per the chain walker (box_ref.rs:295-322); a
            // `Forwarded::Const` terminal is materialized inline by the
            // walker into a fresh BoxRef whose own slot is None.
            Forwarded::Const(_) | Forwarded::Op(_) | Forwarded::InputArg(_) => {
                unreachable!(
                    "getrawptrinfo: chain terminal must not carry Forwarded::Const \
                 (box_ref.rs:295 get_box_replacement walker invariant)",
                )
            }
        }
    }

    /// info.py:880-894 `getptrinfo(op)` parity (line-by-line port).
    ///
    /// ```python
    /// def getptrinfo(op):
    ///     if op.type == 'i':
    ///         return getrawptrinfo(op)
    ///     elif op.type == 'f':
    ///         return None
    ///     assert op.type == 'r'
    ///     op = get_box_replacement(op)
    ///     assert op.type == 'r'
    ///     if isinstance(op, ConstPtr):
    ///         return ConstPtrInfo(op)
    ///     fw = op.get_forwarded()
    ///     if fw is not None:
    ///         assert isinstance(fw, PtrInfo)
    ///         return fw
    ///     return None
    /// ```
    /// The Int arm delegates to `getrawptrinfo` per `info.py:881-882`.
    /// The Float arm short-circuits to `None`. The Void arm panics —
    /// `info.py:885 assert op.type == 'r'` rejects Void boxes outright;
    /// no synthetic Void filler box exists that would smuggle a
    /// type-erased pointer through this helper.
    pub fn getptrinfo(&self, op: &Operand) -> Option<PtrInfo> {
        self.getptrinfo_handle(op).map(|h| h.snapshot())
    }

    /// info.py:880-894 `getptrinfo(op)` parity — orthodox return
    /// shape that preserves RPython `_forwarded` object identity.
    /// See `getrawptrinfo_handle` for the variant semantics.
    pub fn getptrinfo_handle(&self, op: &Operand) -> Option<PtrInfoHandle> {
        use crate::optimizeopt::info::OpInfo;
        use majit_ir::box_ref::Forwarded;
        match op.type_() {
            // info.py:881-882 — `if op.type == 'i': return getrawptrinfo(op)`.
            majit_ir::Type::Int => return self.getrawptrinfo_handle(op),
            // info.py:883-884 — `elif op.type == 'f': return None`.
            majit_ir::Type::Float => return None,
            // info.py:885 — `assert op.type == 'r'`.
            majit_ir::Type::Ref => {}
            majit_ir::Type::Void => panic!(
                "getptrinfo_handle: op.type == 'v' (info.py:885 `assert op.type == 'r'`); \
                 caller must guard on a typed box (Int/Ref/Float) — Void boxes \
                 carry no PtrInfo upstream",
            ),
        }
        // info.py:886-893 type 'r' arm:
        //   op = get_box_replacement(op)
        //   if isinstance(op, ConstPtr): return ConstPtrInfo(op)
        //   fw = op.get_forwarded()
        //   if fw is not None:
        //       assert isinstance(fw, PtrInfo)
        //       return fw
        //   return None
        let terminal = op.get_box_replacement(false);
        debug_assert_eq!(
            terminal.type_(),
            majit_ir::Type::Ref,
            "getptrinfo_handle: chain-walked replacement lost Ref type (got {:?})",
            terminal.type_(),
        );
        // info.py:888-889: if isinstance(op, ConstPtr): return ConstPtrInfo(op)
        if let Some(Value::Ref(gcref)) = terminal.const_value() {
            return Some(PtrInfoHandle::Const(PtrInfo::Constant(gcref)));
        }
        match &terminal.get_forwarded() {
            Forwarded::None => None,
            Forwarded::Info(OpInfo::Ptr(rc)) => Some(PtrInfoHandle::Live(std::rc::Rc::clone(rc))),
            // info.py:892 `assert isinstance(fw, PtrInfo)` — a Ref-typed
            // terminal must not forward to IntBound / FloatConst / Unknown.
            Forwarded::Info(other) => panic!(
                "getptrinfo: forwarded must be PtrInfo (info.py:892), got {:?}",
                std::mem::discriminant(other),
            ),
            // Terminal of `get_box_replacement(false)` can only be `None`
            // or `Info(_)` per the chain walker (box_ref.rs:295-322); a
            // `Forwarded::Const` terminal is materialized inline by the
            // walker into a fresh BoxRef whose own slot is None.
            Forwarded::Const(_) | Forwarded::Op(_) | Forwarded::InputArg(_) => {
                unreachable!(
                    "getptrinfo: chain terminal must not carry Forwarded::Const \
                 (box_ref.rs:295 get_box_replacement walker invariant)",
                )
            }
        }
    }

    /// virtualstate.py:48-55 `GenerateGuardState.get_runtime_field(box, descr)`
    /// parity.
    ///
    /// ```python
    /// def get_runtime_field(self, box, descr):
    ///     struct = box.getref_base()
    ///     if descr.is_pointer_field():
    ///         return InputArgRef(self.cpu.bh_getfield_gc_r(struct, descr))
    ///     elif descr.is_float_field():
    ///         return InputArgFloat(self.cpu.bh_getfield_gc_f(struct, descr))
    ///     else:
    ///         return InputArgInt(self.cpu.bh_getfield_gc_i(struct, descr))
    /// ```
    ///
    /// Walks `runtime_box` to its `Value::Ref(gcref)` payload and reads
    /// the typed value at `gcref.raw() + descr.offset()` using
    /// `FieldDescr.field_size()` / `is_field_signed()` (the same
    /// (offset, size, sign) triple `Cpu::bh_getfield_gc_i` consumes on
    /// the backend — compiler.rs:14570). Wraps the read in a freshly
    /// allocated const OpRef matching `InputArg*` parity.
    ///
    /// Concrete-Ref extractor is `runtime_value_of` (mod.rs) which
    /// cascades box-forwarding chain → const_pool → stamped BoxRef
    /// runtime value (the RPython `InputArg*.value` analog).
    /// Returns `None` when the OpRef does not resolve to a concrete
    /// non-null Ref, when the descr is not a FieldDescr, or when the
    /// runtime pointer is null.
    pub fn get_runtime_field(
        &mut self,
        runtime_box: OpRef,
        descr: &majit_ir::descr::DescrRef,
    ) -> Option<OpRef> {
        // virtualstate.py:48-55 `cpu.bh_getfield_gc_*(struct, descr)` reads the
        // field off the *struct* the runtime box points at. RPython's runtime
        // box is always an eagerly allocated object, so `getref_base()` yields
        // a concrete pointer. Under pyre's lazy boxing the runtime box can be
        // an unmaterialized virtual (state.rs NewWithVtable without a W_*Object
        // allocation), so `getref_base()` has no concrete pointer to read. The
        // faithful equivalent is to read the field's tracked box — the
        // `setfield_gc` source the optimizer recorded on the virtual — which is
        // exactly the value `bh_getfield_gc_*` would observe on the eager
        // object. Consult it before the concrete-Ref read; the concrete path is
        // unchanged for already-materialized runtime boxes.
        if let Some(opref) = self.get_virtual_runtime_field(runtime_box, descr) {
            return Some(opref);
        }
        // virtualstate.py:39 `box.getref_base()` — concrete Ref read.
        // `runtime_value_of` cascades const_pool → stamped BoxRef value
        // (RPython `InputArg*.value` analog).
        let raw = match self.runtime_value_of(runtime_box)? {
            Value::Ref(gcref) if !gcref.is_null() => gcref.0 as i64,
            _ => return None,
        };
        let fd = descr.as_field_descr()?;
        let offset = fd.offset() as i64;
        let ptr = (raw as usize).wrapping_add(offset as usize);
        if raw == 0 {
            return None;
        }
        match fd.field_type() {
            Type::Ref => {
                let val = unsafe { (ptr as *const usize).read_unaligned() };
                Some(self.make_constant_ref(majit_ir::GcRef(val)))
            }
            Type::Float => {
                let val = unsafe { (ptr as *const f64).read_unaligned() };
                Some(self.make_constant_float(val))
            }
            Type::Int => {
                let size = fd.field_size();
                let sign = fd.is_field_signed();
                let val = unsafe {
                    match (size, sign) {
                        (1, true) => (ptr as *const i8).read_unaligned() as i64,
                        (1, false) => (ptr as *const u8).read_unaligned() as i64,
                        (2, true) => (ptr as *const i16).read_unaligned() as i64,
                        (2, false) => (ptr as *const u16).read_unaligned() as i64,
                        (4, true) => (ptr as *const i32).read_unaligned() as i64,
                        (4, false) => (ptr as *const u32).read_unaligned() as i64,
                        _ => (ptr as *const i64).read_unaligned(),
                    }
                };
                Some(self.make_constant_int(val))
            }
            _ => None,
        }
    }

    /// Lazy-boxing companion to `get_runtime_field` / `get_runtime_interiorfield`.
    ///
    /// virtualstate.py:48-55 reads `cpu.bh_getfield_gc_*(struct, descr)` off the
    /// eagerly allocated struct the runtime box points at. When `runtime_box`
    /// resolves to an unmaterialized `PtrInfo::Virtual` / `VirtualStruct`
    /// instead of a concrete pointer (pyre keeps W_IntObject counters as
    /// virtuals — state.rs NewWithVtable with no allocation), there is no
    /// struct to read; the field's observed value lives on the virtual's
    /// tracked field box (the recorded `setfield_gc` source). This mirrors what
    /// `bh_getfield_gc_*` would observe on the eager object: look up the field
    /// slot via `info.getfield(index_in_parent)` (the same parent-local slot
    /// `enum_forced_boxes_for_entry` reads, virtualstate.rs:854-865), resolve
    /// its own runtime value (`runtime_value_of`, the InputArg*.value analog),
    /// and wrap it in a fresh const OpRef matching the concrete-read return
    /// shape. Returns `None` when the runtime box is not a virtual, the slot is
    /// unset, or the slot carries no observed value.
    fn get_virtual_runtime_field(
        &mut self,
        runtime_box: OpRef,
        descr: &majit_ir::descr::DescrRef,
    ) -> Option<OpRef> {
        let fd = descr.as_field_descr()?;
        // virtualstate.py:162 `opinfo._fields[descr.get_index()]`: the slot key
        // is the parent-local field index, not the global Descr.index(), and is
        // only meaningful when a parent SizeDescr is bound (descr.rs index_in_parent).
        let slot = fd
            .get_parent_descr()
            .map(|_| fd.index_in_parent() as u32)
            .unwrap_or_else(|| descr.index());
        // virtualstate.py:149-151 `opinfo = getptrinfo(box); assert
        // opinfo.is_virtual()`. Read the field box only off a virtual struct
        // ptrinfo; a concrete/instance ptrinfo falls through to the eager read.
        let b = self.get_box_replacement_box(runtime_box)?;
        let info = self.getptrinfo(&majit_ir::operand::Operand::from_boxref(&b))?;
        let field_opref = match &info {
            crate::optimizeopt::info::PtrInfo::Virtual(_)
            | crate::optimizeopt::info::PtrInfo::VirtualStruct(_) => {
                info.getfield(slot)?.as_opref()?
            }
            _ => return None,
        };
        // virtualstate.py:48-55: the read produces a fresh InputArg* carrying
        // the concrete field value. `runtime_value_of` reads the field box's
        // own observed value (the setfield_gc source's stamped value); wrap it
        // in the matching const OpRef.
        self.const_opref_from_runtime_value(field_opref)
    }

    /// Lazy-boxing companion to `get_runtime_interiorfield`.
    ///
    /// When `runtime_box` resolves to an unmaterialized `VirtualArrayStruct`,
    /// the interior field's observed value lives on the virtual's tracked
    /// element-field box (info.py:663-668 `getinteriorfield_virtual`), not a
    /// concrete struct. Read it the same way `get_virtual_runtime_field` reads a
    /// struct field. Returns `None` when the runtime box is not a virtual
    /// array-struct, the slot is unset, or the slot carries no observed value.
    fn get_virtual_runtime_interiorfield(
        &mut self,
        runtime_box: OpRef,
        descr: &majit_ir::descr::DescrRef,
        i: usize,
    ) -> Option<OpRef> {
        let ifd = descr.as_interior_field_descr()?;
        let fd = ifd.field_descr();
        // virtualstate.py:321-322 reads `opinfo._fields[i][descr.get_index()]`:
        // the parent-local field slot index (descr.rs index_in_parent), with a
        // fallback to the global index when no parent SizeDescr is bound.
        let slot = fd
            .get_parent_descr()
            .map(|_| fd.index_in_parent() as u32)
            .unwrap_or_else(|| descr.index());
        let b = self.get_box_replacement_box(runtime_box)?;
        let info = self.getptrinfo(&majit_ir::operand::Operand::from_boxref(&b))?;
        if !matches!(
            info,
            crate::optimizeopt::info::PtrInfo::VirtualArrayStruct(_)
        ) {
            return None;
        }
        let field_opref = info.getinteriorfield_virtual(i, slot)?;
        self.const_opref_from_runtime_value(field_opref)
    }

    /// virtualstate.py:48-55 tail: wrap the field box's observed runtime value
    /// (`runtime_value_of`, the `InputArg*.value` analog) in a fresh const OpRef
    /// matching the `InputArg{Int,Float,Ref}` the concrete `bh_getfield_gc_*`
    /// read would produce. Returns `None` for an unobserved or void slot.
    fn const_opref_from_runtime_value(&mut self, field_opref: OpRef) -> Option<OpRef> {
        match self.runtime_value_of(field_opref)? {
            Value::Int(v) => Some(self.make_constant_int(v)),
            Value::Float(v) => Some(self.make_constant_float(v)),
            Value::Ref(v) => Some(self.make_constant_ref(v)),
            Value::Void => None,
        }
    }

    /// virtualstate.py:39-47 `GenerateGuardState.get_runtime_item(box, descr, i)`
    /// parity.
    ///
    /// ```python
    /// def get_runtime_item(self, box, descr, i):
    ///     array = box.getref_base()
    ///     if descr.is_array_of_pointers():
    ///         return InputArgRef(self.cpu.bh_getarrayitem_gc_r(array, i, descr))
    ///     elif descr.is_array_of_floats():
    ///         return InputArgFloat(self.cpu.bh_getarrayitem_gc_f(array, i, descr))
    ///     else:
    ///         return InputArgInt(self.cpu.bh_getarrayitem_gc_i(array, i, descr))
    /// ```
    ///
    /// Reads `array_ptr + base_size + i * itemsize` per
    /// `ArrayDescr.base_size()` / `ArrayDescr.itemsize()` matching the
    /// backend `Cpu::bh_getarrayitem_gc_*` (compiler.rs:14611). Wraps
    /// the read in a freshly allocated const OpRef.
    ///
    /// Concrete-Ref extractor routes through `runtime_value_of`; see
    /// the matching note on `get_runtime_field` for the concrete-value
    /// population path.
    pub fn get_runtime_item(
        &mut self,
        runtime_box: OpRef,
        descr: &majit_ir::descr::DescrRef,
        i: usize,
    ) -> Option<OpRef> {
        // virtualstate.py:39 `box.getref_base()` — concrete Ref read.
        // `runtime_value_of` cascades const_pool → stamped BoxRef value
        // (RPython `InputArg*.value` analog).
        let raw = match self.runtime_value_of(runtime_box)? {
            Value::Ref(gcref) if !gcref.is_null() => gcref.0 as i64,
            _ => return None,
        };
        let ad = descr.as_array_descr()?;
        let base_size = ad.base_size() as i64;
        let itemsize = ad.item_size() as i64;
        let offset = base_size + (i as i64) * itemsize;
        let ptr = (raw as usize).wrapping_add(offset as usize);
        match ad.item_type() {
            Type::Ref => {
                let val = unsafe { (ptr as *const usize).read_unaligned() };
                Some(self.make_constant_ref(majit_ir::GcRef(val)))
            }
            Type::Float => {
                let val = unsafe { (ptr as *const f64).read_unaligned() };
                Some(self.make_constant_float(val))
            }
            Type::Int => {
                let size = itemsize as usize;
                let sign = ad.is_item_signed();
                let val = unsafe {
                    match (size, sign) {
                        (1, true) => (ptr as *const i8).read_unaligned() as i64,
                        (1, false) => (ptr as *const u8).read_unaligned() as i64,
                        (2, true) => (ptr as *const i16).read_unaligned() as i64,
                        (2, false) => (ptr as *const u16).read_unaligned() as i64,
                        (4, true) => (ptr as *const i32).read_unaligned() as i64,
                        (4, false) => (ptr as *const u32).read_unaligned() as i64,
                        _ => (ptr as *const i64).read_unaligned(),
                    }
                };
                Some(self.make_constant_int(val))
            }
            _ => None,
        }
    }

    /// virtualstate.py:57-67 `GenerateGuardState.get_runtime_interiorfield(box, descr, i)`
    /// parity.
    ///
    /// ```python
    /// def get_runtime_interiorfield(self, box, descr, i):
    ///     struct = box.getref_base()
    ///     if descr.is_pointer_field():
    ///         return InputArgRef(self.cpu.bh_getinteriorfield_gc_r(struct, i, descr))
    ///     elif descr.is_float_field():
    ///         return InputArgFloat(self.cpu.bh_getinteriorfield_gc_f(struct, i, descr))
    ///     else:
    ///         return InputArgInt(self.cpu.bh_getinteriorfield_gc_i(struct, i, descr))
    /// ```
    ///
    /// Reads at `struct_ptr + array.base_size() + i * array.item_size()
    /// + field.offset()` per `InteriorFieldDescr.array_descr()` +
    /// `field_descr()`. Matches the backend Cpu::bh_getinteriorfield_gc_*
    /// shape (struct + element_index + interior-field).
    ///
    /// Concrete-Ref extractor routes through `runtime_value_of`; see
    /// `get_runtime_field` docstring.
    pub fn get_runtime_interiorfield(
        &mut self,
        runtime_box: OpRef,
        descr: &majit_ir::descr::DescrRef,
        i: usize,
    ) -> Option<OpRef> {
        // Lazy-boxing path: when the runtime box is an unmaterialized
        // `VirtualArrayStruct` the interior field's observed value lives on the
        // virtual's tracked element-field box rather than a concrete struct.
        // See `get_virtual_runtime_field` for the model this mirrors.
        if let Some(opref) = self.get_virtual_runtime_interiorfield(runtime_box, descr, i) {
            return Some(opref);
        }
        // virtualstate.py:39 `box.getref_base()` — concrete Ref read.
        // `runtime_value_of` cascades const_pool → stamped BoxRef value
        // (RPython `InputArg*.value` analog).
        let raw = match self.runtime_value_of(runtime_box)? {
            Value::Ref(gcref) if !gcref.is_null() => gcref.0 as i64,
            _ => return None,
        };
        let ifd = descr.as_interior_field_descr()?;
        let ad = ifd.array_descr();
        let fd = ifd.field_descr();
        let element_offset = (ad.base_size() as i64) + (i as i64) * (ad.item_size() as i64);
        let offset = element_offset + (fd.offset() as i64);
        let ptr = (raw as usize).wrapping_add(offset as usize);
        match fd.field_type() {
            Type::Ref => {
                let val = unsafe { (ptr as *const usize).read_unaligned() };
                Some(self.make_constant_ref(majit_ir::GcRef(val)))
            }
            Type::Float => {
                let val = unsafe { (ptr as *const f64).read_unaligned() };
                Some(self.make_constant_float(val))
            }
            Type::Int => {
                let size = fd.field_size();
                let sign = fd.is_field_signed();
                let val = unsafe {
                    match (size, sign) {
                        (1, true) => (ptr as *const i8).read_unaligned() as i64,
                        (1, false) => (ptr as *const u8).read_unaligned() as i64,
                        (2, true) => (ptr as *const i16).read_unaligned() as i64,
                        (2, false) => (ptr as *const u16).read_unaligned() as i64,
                        (4, true) => (ptr as *const i32).read_unaligned() as i64,
                        (4, false) => (ptr as *const u32).read_unaligned() as i64,
                        _ => (ptr as *const i64).read_unaligned(),
                    }
                };
                Some(self.make_constant_int(val))
            }
            _ => None,
        }
    }

    /// model.py:199-201 `cpu.cls_of_box(box)` parity:
    ///
    /// ```python
    /// def cls_of_box(self, box):
    ///     obj = lltype.cast_opaque_ptr(OBJECTPTR, box.getref_base())
    ///     return ConstInt(ptr2int(obj.typeptr))
    /// ```
    ///
    /// Walks the BoxRef chain to its constant `Value::Ref(gcref)` payload
    /// (`box.getref_base()` parity) and dispatches `cpu.cls_of_box(raw)`
    /// through the `Cpu` trait object stored at `self.cpu`.  Falls back
    /// to the resolved box's per-type mixin slot (`RefOp._resref`,
    /// resoperation.py:612) when the BoxRef chain has no terminal Const
    /// — live `InputArgRef` boxes with a tracer-recorded concrete value
    /// reach the typeptr deref through a synthetic Const wrapper.
    /// Returns `None` when neither path produces a non-null gcref
    /// (`DefaultCpu::cls_of_box` reports both "no Ref" and "null gcref"
    /// as 0).
    ///
    /// Caller shape mirrors `optimizer.cpu.cls_of_box(box)` — every
    /// invocation (`info.rs`, `virtualstate.rs`, `rewrite.rs`,
    /// `bridgeopt.rs`) routes through `ctx.cls_of_box(box)`.  Future
    /// `bh_*` runtime calls will land on the same `Cpu` trait and lose
    /// the `OptContext::cls_of_box` wrapper as that surface fills out.
    pub fn cls_of_box(&self, op: &Operand) -> Option<i64> {
        // model.py:199-201 `cpu.cls_of_box(box)` returns `ConstInt(ptr2int(
        // typeptr))` — the immortal vtable address as a plain integer, never
        // a traced ref. DefaultCpu walks the BoxRef to its Const terminal and
        // dereferences the typeptr-at-offset-0. Returns 0 for non-Ref / null.
        let typeptr = self.cpu.cls_of_box(&op.to_boxref());
        if typeptr != 0 {
            return Some(typeptr);
        }
        // resoperation.py:612-642 `RefOp._resref` fallback — when the
        // BoxRef chain has no Const terminal, read the mixin slot
        // directly off the resolved box.  Wrap as a synthetic Const so
        // the typeptr deref goes through the same `cpu.cls_of_box`
        // path (preserves gcremovetypeptr overrides).
        let resolved = op.get_box_replacement(false);
        if resolved.const_value().is_some() {
            // Already had a Const terminal; cpu reported 0 because
            // non-Ref or null.  No mixin-slot fallback applies.
            return None;
        }
        let value = resolved.get_value()?;
        match value {
            Value::Ref(gcref) if !gcref.is_null() => {}
            _ => return None,
        }
        let synth = majit_ir::box_ref::BoxRef::new_const(value);
        let typeptr = self.cpu.cls_of_box(&synth);
        if typeptr == 0 { None } else { Some(typeptr) }
    }

    /// info.py:880 `getptrinfo(op).get_known_class(cpu)` parity.
    ///
    /// Delegates to `getptrinfo(&BoxRef)` + `PtrInfo::get_known_class` so
    /// constant pointers are handled via `cls_of_box` the same way
    /// `Instance` / `Virtual` read their stored `known_class`.
    pub fn get_known_class(&self, op: &Operand) -> Option<i64> {
        self.getptrinfo(op)?.get_known_class(self.cpu.as_ref())
    }

    /// optimizer.py:127-135 `getnullness(op)` parity (line-by-line port).
    ///
    /// ```python
    /// def getnullness(self, op):
    ///     if op.type == 'r' or self.is_raw_ptr(op):
    ///         ptrinfo = getptrinfo(op)
    ///         if ptrinfo is None:
    ///             return info.INFO_UNKNOWN
    ///         return ptrinfo.getnullness()
    ///     elif op.type == 'i':
    ///         return self.getintbound(op).getnullness()
    ///     assert False
    /// ```
    ///
    /// Returns one of `INFO_NULL` / `INFO_NONNULL` / `INFO_UNKNOWN`
    /// (info.py:13-15) so callers can compare directly against the
    /// upstream constants.
    ///
    /// The `Type::Int` arm inlines `getintbound` (optimizer.py:99-113)
    /// BoxRef-direct, preserving the lazy install of `IntBound.unbounded()`
    /// on first access via `set_forwarded_info` (interior mutability lets
    /// the method take `&self`).
    pub fn getnullness(&self, op: &Operand) -> i8 {
        use crate::optimizeopt::info::OpInfo;
        use majit_ir::box_ref::Forwarded;
        // optimizer.py:128: if op.type == 'r' or self.is_raw_ptr(op):
        //
        // `Box.type` is intrinsic in upstream — never Void. In pyre,
        // `materialize_box_at` lazy-creates `Type::Void` phantom placeholders
        // for OpRefs the recorder has not yet typed; the chain walker
        // hop into the terminal Box (which carries the proper type via
        // `BoxRef::new_const` for Const targets) recovers the
        // RPython-intrinsic type. Read after chain walk so a phantom
        // forwarded to a typed Const still routes via the type arm.
        let resolved = op.get_box_replacement(false);
        let tp = resolved.type_();
        if matches!(tp, majit_ir::Type::Ref) || self.is_raw_ptr(op) {
            // optimizer.py:129-132 with info.py:880-894 `getptrinfo` inlined.
            //
            // info.py:886-893: `r`-typed: walk the chain, synthesize
            // `ConstPtrInfo` for `ConstPtr`, else return the forwarded slot.
            // For `Type::Int` raw-pointer entries (this branch when
            // `is_raw_ptr(op)` returned true), the forwarded slot is a
            // `VirtualRaw{Buffer,Slice}` and `ptr_info()` reads it directly.
            let ptrinfo: Option<PtrInfo> = if let Some(Value::Ref(gcref)) = resolved.const_value() {
                // info.py:888-889: isinstance(op, ConstPtr): ConstPtrInfo(op)
                Some(PtrInfo::Constant(gcref))
            } else {
                // info.py:890-893: fw = op.get_forwarded(); return fw or None
                resolved.ptr_info().map(|r| r.clone())
            };
            // optimizer.py:130-132: if ptrinfo is None: INFO_UNKNOWN; else ptrinfo.getnullness()
            return match ptrinfo {
                None => INFO_UNKNOWN,
                Some(info) => info.getnullness(),
            };
        }
        // optimizer.py:133-134: elif op.type == 'i': return getintbound(op).getnullness()
        //
        // Void phantoms (untyped recorder placeholders) route through the
        // Int arm as the pyre equivalent of RPython's unknown-type tolerance
        // — the inlined `getintbound` side effect (line 110-113) installs
        // `IntBound.unbounded()` so subsequent reads agree.
        if matches!(tp, majit_ir::Type::Int | majit_ir::Type::Void) {
            // optimizer.py:99-113 `getintbound` inlined BoxRef-direct.
            // optimizer.py:101: op = get_box_replacement(op) — already
            // walked above (`resolved` shadows here for parity).
            // optimizer.py:102-103: if isinstance(op, ConstInt): from_constant
            if let Some(Value::Int(v)) = resolved.const_value() {
                return crate::optimizeopt::intutils::IntBound::from_constant(v as i64)
                    .getnullness();
            }
            // optimizer.py:104-109: fw = op.get_forwarded(); branch on type.
            {
                let fw = resolved.get_forwarded();
                match &fw {
                    Forwarded::Info(OpInfo::IntBound(rc)) => return rc.borrow().getnullness(),
                    // optimizer.py:108-109: rare case (fw is RawBufferPtrInfo).
                    // optimizer.py:104-109 reads anything that is not an
                    // IntBound through the same "return unbounded" path.
                    Forwarded::Info(_) => {
                        return crate::optimizeopt::intutils::IntBound::unbounded().getnullness();
                    }
                    Forwarded::Const(_) | Forwarded::Op(_) | Forwarded::InputArg(_) => {
                        unreachable!("chain walker terminal")
                    }
                    Forwarded::None => {}
                }
            }
            // optimizer.py:110-113: intbound = unbounded; op.set_forwarded(intbound); return intbound
            let intbound = crate::optimizeopt::intutils::IntBound::unbounded();
            resolved.set_forwarded_info(OpInfo::int_bound(intbound.clone()));
            return intbound.getnullness();
        }
        // optimizer.py:135: assert False — Float / Void never reaches here in upstream.
        INFO_UNKNOWN
    }

    /// optimizer.py:154-158 `is_raw_ptr(op)` parity (line-by-line port).
    ///
    /// ```python
    /// def is_raw_ptr(self, op):
    ///     fw = get_box_replacement(op).get_forwarded()
    ///     if isinstance(fw, info.AbstractRawPtrInfo):
    ///         return True
    ///     return False
    /// ```
    ///
    /// `AbstractRawPtrInfo` is the upstream base for `RawBufferPtrInfo`,
    /// `RawStructPtrInfo`, `RawSlicePtrInfo` (info.py:374-485). Of these:
    ///
    /// - `RawBufferPtrInfo` ↔ majit `PtrInfo::VirtualRawBuffer` (created
    ///   by `OptVirtualize` from `RAW_MALLOC_VARSIZE_CHAR`).
    /// - `RawSlicePtrInfo` ↔ majit `PtrInfo::VirtualRawSlice` (created
    ///   by `OptVirtualize::optimize_int_add` slice creator,
    ///   virtualize.py:60 make_virtual_raw_slice).
    /// - `RawStructPtrInfo` is defined at info.py:452 but never
    ///   instantiated anywhere in upstream (`grep -rn "RawStructPtrInfo("
    ///   rpython/jit/` returns only the class definition). It is dead
    ///   reservation code, so the absence of a majit variant is not a
    ///   parity gap.
    ///
    /// `ConstPtrInfo` is NOT a subclass of `AbstractRawPtrInfo` in
    /// upstream, so a constant raw-pointer `ConstInt` is `False` here
    /// (matches `isinstance(fw, AbstractRawPtrInfo)` returning `False`
    /// for `ConstPtrInfo`).
    pub fn is_raw_ptr(&self, op: &Operand) -> bool {
        let resolved = op.get_box_replacement(false);
        matches!(
            resolved.ptr_info().as_deref(),
            Some(PtrInfo::VirtualRawBuffer(_) | PtrInfo::VirtualRawSlice(_))
        )
    }

    /// Set PtrInfo without clearing forwarding.
    /// RPython parity: set PtrInfo at the terminal of opref's forwarding chain.
    /// In RPython, `box.set_forwarded(info)` sets info on the Box directly.
    /// `get_box_replacement(box)` then returns the terminal Box which has the info.
    /// In majit, we follow the Op chain to the terminal OpRef and set Info there.
    fn ensure_ptr_info_preserve_forwarding(&mut self, opref: OpRef, info: PtrInfo) {
        use crate::optimizeopt::info::OpInfo;
        let terminal = self.get_replacement_opref(opref);
        if terminal.is_constant() {
            return;
        }
        let b = self.materialize_operand_at(terminal);
        let already_set = !matches!(b.get_forwarded(), majit_ir::box_ref::Forwarded::None);
        if !already_set {
            b.set_forwarded_info(OpInfo::ptr(info));
        }
    }

    /// info.py:718-726 `ConstPtrInfo._get_info(descr, optheap)` parity.
    ///
    /// ```python
    /// def _get_info(self, descr, optheap):
    ///     ref = self._const.getref_base()
    ///     if not ref:
    ///         raise InvalidLoop   # null protection
    ///     info = optheap.const_infos.get(ref, None)
    ///     if info is None:
    ///         info = StructPtrInfo(descr)
    ///         optheap.const_infos[ref] = info
    ///     return info
    /// ```
    ///
    /// majit's port: route through `getptrinfo` (which encapsulates the
    /// RPython `op.type` dispatch + `ConstPtrInfo` synthesis), then read
    /// `_const.getref_base()` from the resulting `PtrInfo::Constant`.
    /// Both `Value::Ref` constants and `Value::Int` constants tagged
    /// with a `Type::Ref` override hash to the same `const_infos` slot
    /// — the upstream invariant that any `ConstPtrInfo._get_info()`
    /// call on the same address returns the same shared
    /// `StructPtrInfo`.
    ///
    /// Returns `None` when `opref` is not a constant pointer at all
    /// (matching PyPy's `getrawptrinfo` returning `None` for non-pointer
    /// boxes — there's no `_get_info` to call), and also when a constant
    /// pointer resolves to a null `gcref`. In the null case it additionally
    /// records a deferred `InvalidLoop` signal on the context, mirroring
    /// `info.py:720-721`:
    ///
    /// ```python
    /// def _get_info(self, descr, optheap):
    ///     ref = self._const.getref_base()
    ///     if not ref:
    ///         raise InvalidLoop   # null protection
    /// ```
    ///
    /// The trace was constant-folding through a null base pointer, which
    /// is an impossible execution path; the driver aborts the trace at its
    /// next barrier so the JIT can retry with a different shape.
    /// Like `get_const_info_mut` but does NOT create an entry on miss.
    /// Returns `None` when:
    /// - `opref` is not a constant pointer
    /// - The constant is null
    /// - No `const_infos` entry has been created yet
    ///
    /// Used by array invalidation paths that only need to clear existing
    /// items, not install new PtrInfo variants.
    pub fn get_const_info_mut_if_exists(
        &mut self,
        opref: OpRef,
    ) -> Option<&mut crate::optimizeopt::info::PtrInfo> {
        // Resolve the position to its canonical box, then delegate to the
        // box-native form. `get_box_replacement_box` returns `None` for an
        // unresolved position (the old `and_then`/`_ => return None` arm).
        let op = self.get_box_replacement_operand_opt(opref)?;
        self.get_const_info_mut_if_exists_box(&op)
    }

    /// info.py:715-726 `ConstPtrInfo._get_info(descr, optheap)` parity.
    ///
    /// `parent_descr` is the parent SizeDescr, passed so that the
    /// vacant-slot case creates `StructPtrInfo(descr)` (info.py:724)
    /// rather than a bare `PtrInfo::instance(None, None)`. Callers
    /// that don't have the parent descr (e.g. the field read path)
    /// extract it from the field descr via
    /// `descr.as_field_descr().get_parent_descr()`.
    pub fn get_const_info_mut(
        &mut self,
        opref: OpRef,
        parent_descr: Option<DescrRef>,
    ) -> Option<&mut crate::optimizeopt::info::PtrInfo> {
        // Resolve the position to its canonical box, then delegate to the
        // box-native form.
        let op = self.get_box_replacement_operand_opt(opref)?;
        self.get_const_info_mut_box(&op, parent_descr)
    }

    /// Box-native form of `get_const_info_mut_if_exists`: the caller already
    /// holds the (canonical) struct box, so `getptrinfo` chain-walks it
    /// directly (info.py:886 `op = get_box_replacement(op)`). Returns `None`
    /// when the box is not a constant pointer, the constant is null, or no
    /// `const_infos` entry exists yet.
    pub fn get_const_info_mut_if_exists_box(
        &mut self,
        op: &Operand,
    ) -> Option<&mut crate::optimizeopt::info::PtrInfo> {
        use crate::optimizeopt::info::PtrInfo;
        let gcref = match self.getptrinfo(op) {
            Some(PtrInfo::Constant(g)) => g,
            _ => return None,
        };
        if gcref.is_null() {
            return None;
        }
        self.const_infos.get_mut(&gcref.0)
    }

    /// Box-native form of `get_const_info_mut` (info.py:715-726
    /// `ConstPtrInfo._get_info`): the caller holds the (canonical) struct
    /// box. Creates a `StructPtrInfo(parent_descr)` on miss; raises
    /// `InvalidLoop` on a null constant base (info.py:720-721).
    pub fn get_const_info_mut_box(
        &mut self,
        op: &Operand,
        parent_descr: Option<DescrRef>,
    ) -> Option<&mut crate::optimizeopt::info::PtrInfo> {
        use crate::optimizeopt::info::PtrInfo;
        // info.py:719: ref = self._const.getref_base()
        let gcref = match self.getptrinfo(op) {
            Some(PtrInfo::Constant(g)) => g,
            _ => return None,
        };
        // info.py:720-721: if not ref: raise InvalidLoop. Recorded as a
        // deferred signal (checked by the driver) and `None` returned so
        // callers take their existing "no info" path until the barrier aborts.
        if gcref.is_null() {
            self.signal_invalid_loop("ConstPtrInfo._get_info: null constant base pointer");
            return None;
        }
        let addr = gcref.0;
        // info.py:722-725: info = optheap.const_infos.get(ref, None)
        //                  if info is None: info = StructPtrInfo(descr)
        //                  optheap.const_infos[ref] = info
        Some(self.const_infos.entry(addr).or_insert_with(|| {
            // info.py:724: StructPtrInfo(descr)
            match parent_descr {
                Some(d) => PtrInfo::struct_ptr(d),
                None => PtrInfo::instance(None, None),
            }
        }))
    }

    /// info.py:728-735 `ConstPtrInfo._get_array_info(descr, optheap)`
    /// parity:
    ///
    /// ```python
    /// def _get_array_info(self, descr, optheap):
    ///     ref = self._const.getref_base()
    ///     if not ref:
    ///         raise InvalidLoop   # null protection
    ///     info = optheap.const_infos.get(ref, None)
    ///     if info is None:
    ///         info = ArrayPtrInfo(descr)
    ///         optheap.const_infos[ref] = info
    ///     return info
    /// ```
    ///
    /// Companion to `get_const_info_mut` for the array path. Both share
    /// the same `const_infos` slot keyed by `gcref` — PyPy's invariant
    /// is that a given constant ref is used as either a struct base or
    /// an array base, never both. The Vacant entry inserts an
    /// `ArrayPtrInfo` (descr + `nonnegative` lenbound) so subsequent
    /// `setitem`/`getitem` calls land on the right variant.
    pub fn get_const_info_array_mut(
        &mut self,
        opref: OpRef,
        descr: DescrRef,
    ) -> Option<&mut crate::optimizeopt::info::PtrInfo> {
        // Resolve the position to its canonical box, then delegate to the
        // box-native form.
        let op = self.get_box_replacement_operand_opt(opref)?;
        self.get_const_info_array_mut_box(&op, descr)
    }

    /// Box-native form of `get_const_info_array_mut` (info.py:728-735
    /// `ConstPtrInfo._get_array_info`): the caller holds the (canonical)
    /// array box, so `getptrinfo` chain-walks it directly. Creates an
    /// `ArrayPtrInfo(descr)` with a `nonnegative` lenbound on miss; raises
    /// `InvalidLoop` on a null constant base (info.py:730-731).
    pub fn get_const_info_array_mut_box(
        &mut self,
        op: &Operand,
        descr: DescrRef,
    ) -> Option<&mut crate::optimizeopt::info::PtrInfo> {
        use crate::optimizeopt::info::PtrInfo;
        // info.py:729: ref = self._const.getref_base()
        let gcref = match self.getptrinfo(op) {
            Some(PtrInfo::Constant(g)) => g,
            _ => return None,
        };
        // info.py:730-731: if not ref: raise InvalidLoop. Deferred signal +
        // `None` return; the driver aborts at the next barrier.
        if gcref.is_null() {
            self.signal_invalid_loop("ConstPtrInfo._get_array_info: null constant base pointer");
            return None;
        }
        let addr = gcref.0;
        Some(self.const_infos.entry(addr).or_insert_with(|| {
            crate::optimizeopt::info::PtrInfo::array(
                descr,
                crate::optimizeopt::intutils::IntBound::nonnegative(),
            )
        }))
    }

    /// info.py:750-752 `ConstPtrInfo.setfield` + info.py:203-211
    /// `AbstractStructPtrInfo.setfield` parity (line-by-line PyPy
    /// `structinfo.setfield(...)` routing).
    ///
    /// ```python
    /// # ConstPtrInfo
    /// def setfield(self, fielddescr, struct, op, optheap=None, cf=None):
    ///     info = self._get_info(fielddescr.get_parent_descr(), optheap)
    ///     info.setfield(fielddescr, struct, op, optheap=optheap, cf=cf)
    ///
    /// # AbstractStructPtrInfo
    /// def setfield(self, fielddescr, struct, op, optheap=None, cf=None):
    ///     self.init_fields(fielddescr.get_parent_descr(),
    ///                      fielddescr.get_index())
    ///     self._fields[fielddescr.get_index()] = op
    /// ```
    ///
    /// The Rust port routes both branches through one helper so heap.rs
    /// callers don't need to special-case the constant arg0 path. The
    /// constant case lands on `const_infos[gcref]`; the regular case
    /// runs `ensure_ptr_info_arg0(op).as_mut().setfield(...)`.
    pub fn structinfo_setfield(&mut self, op: &Op, field_idx: u32, value: OpRef) {
        let value = self.materialize_operand_at(value);
        let arg0 = self.resolve_operand_box(&op.arg(0)).to_opref();
        if arg0.is_constant()
            || self
                .get_box_replacement_box(arg0)
                .and_then(|cb| cb.const_value())
                .is_some()
        {
            let parent_descr = op.with_field_descr(|fd| fd.get_parent_descr()).flatten();
            if let Some(info) = self.get_const_info_mut(arg0, parent_descr) {
                info.setfield(field_idx, value.clone());
            }
            return;
        }
        // info.py:203-211 AbstractStructPtrInfo.setfield: mutate `_fields`
        // in the PtrInfo object stored in the BoxRef's `_forwarded` slot.
        // PyPy has the same single-object behavior via `box._forwarded`.
        self.with_ensured_ptr_info_arg0(op, |mut handle| {
            if let Some(mut pi) = handle.as_mut() {
                pi.setfield(field_idx, value.clone());
            }
        });
    }

    /// info.py:746-748 `ConstPtrInfo.setitem` + info.py: ArrayPtrInfo
    /// `setitem` parity. Same shape as `structinfo_setfield` but routes
    /// through `_get_array_info` (`get_const_info_array_mut`) for the
    /// constant arg0 path so the const_infos slot is created as
    /// `PtrInfo::Array` rather than `PtrInfo::Instance`.
    pub fn arrayinfo_setitem(&mut self, op: &Op, index: usize, value: OpRef) {
        let value = self.materialize_operand_at(value);
        let arg0 = self.resolve_operand_operand(&op.arg(0));
        if arg0.is_constant() || arg0.const_value().is_some() {
            if let Some(descr) = op.getdescr() {
                if let Some(info) = self.get_const_info_array_mut_box(&arg0, descr) {
                    info.setitem(index, value.clone());
                }
            }
            return;
        }
        // info.py: ArrayPtrInfo.setitem: mutate `_items` in the PtrInfo object
        // stored in the BoxRef's `_forwarded` slot.
        self.with_ensured_ptr_info_arg0(op, |mut handle| {
            if let Some(mut pi) = handle.as_mut() {
                pi.setitem(index, value.clone());
            }
        });
    }

    /// optimizer.py:440-451: make_nonnull(op) line-by-line port.
    ///
    /// ```python
    /// def make_nonnull(self, op):
    ///     op = self.get_box_replacement(op)
    ///     if op.is_constant():
    ///         return
    ///     if op.type == 'i':
    ///         # raw pointers
    ///         return
    ///     opinfo = op.get_forwarded()
    ///     if opinfo is not None:
    ///         assert opinfo.is_nonnull()
    ///         return
    ///     op.set_forwarded(info.NonNullPtrInfo())
    /// ```
    pub fn make_nonnull(&self, op: &Operand) {
        use crate::optimizeopt::info::OpInfo;
        // optimizer.py:441: op = self.get_box_replacement(op)
        let op = op.get_box_replacement(false);
        // optimizer.py:442-443: if op.is_constant(): return
        if op.is_constant() {
            return;
        }
        // optimizer.py:444-445: if op.type == 'i': return  (raw pointers)
        if matches!(op.type_(), majit_ir::Type::Int) {
            return;
        }
        // optimizer.py:446-449: opinfo = op.get_forwarded()
        //                       if opinfo is not None: ... return
        // After `get_box_replacement` walks the chain, the terminal box's
        // forwarded slot is either `Forwarded::None` or `Forwarded::Info(_)`
        // (Box variants are consumed during walk). The skip condition maps
        // directly to "Info present".
        if matches!(op.get_forwarded(), majit_ir::box_ref::Forwarded::Info(_)) {
            return;
        }
        // optimizer.py:451: op.set_forwarded(info.NonNullPtrInfo())
        op.set_forwarded_info(OpInfo::ptr(PtrInfo::NonNull { last_guard_pos: -1 }));
    }

    /// optimizer.py:461-499 `ensure_ptr_info_arg0(op)` — direct line-by-line
    /// port that returns the same kind of value as PyPy.
    ///
    /// ```python
    /// def ensure_ptr_info_arg0(self, op):
    ///     from rpython.jit.metainterp.optimizeopt import vstring
    ///     arg0 = self.get_box_replacement(op.getarg(0))
    ///     if arg0.is_constant():
    ///         return info.ConstPtrInfo(arg0)
    ///     opinfo = arg0.get_forwarded()
    ///     if isinstance(opinfo, info.AbstractVirtualPtrInfo):
    ///         return opinfo
    ///     elif opinfo is not None:
    ///         last_guard_pos = opinfo.get_last_guard_pos()
    ///     else:
    ///         last_guard_pos = -1
    ///     assert opinfo is None or opinfo.__class__ is info.NonNullPtrInfo
    ///     opnum = op.opnum
    ///     if (rop.is_getfield(opnum) or opnum == rop.SETFIELD_GC or
    ///         opnum == rop.QUASIIMMUT_FIELD):
    ///         descr = op.getdescr()
    ///         parent_descr = descr.get_parent_descr()
    ///         if parent_descr.is_object():
    ///             opinfo = info.InstancePtrInfo(parent_descr)
    ///         else:
    ///             opinfo = info.StructPtrInfo(parent_descr)
    ///         opinfo.init_fields(parent_descr, descr.get_index())
    ///     elif (rop.is_getarrayitem(opnum) or opnum == rop.SETARRAYITEM_GC or
    ///           opnum == rop.ARRAYLEN_GC):
    ///         opinfo = info.ArrayPtrInfo(op.getdescr())
    ///     elif opnum in (rop.GUARD_CLASS, rop.GUARD_NONNULL_CLASS):
    ///         opinfo = info.InstancePtrInfo()
    ///     elif opnum in (rop.STRLEN,):
    ///         opinfo = vstring.StrPtrInfo(vstring.mode_string)
    ///     elif opnum in (rop.UNICODELEN,):
    ///         opinfo = vstring.StrPtrInfo(vstring.mode_unicode)
    ///     else:
    ///         assert False, "operations %s unsupported" % op
    ///     assert isinstance(opinfo, info.NonNullPtrInfo)
    ///     opinfo.last_guard_pos = last_guard_pos
    ///     arg0.set_forwarded(opinfo)
    ///     return opinfo
    /// ```
    ///
    /// Returns an [`EnsuredPtrInfo`] discriminating the constant arg0 path
    /// (`Constant(GcRef)` ↔ `info.ConstPtrInfo(arg0)`) from the regular
    /// path (`Forwarded(&mut PtrInfo)` ↔ `arg0.set_forwarded(opinfo); return
    /// opinfo`). Callers invoke methods on the return value directly,
    /// matching PyPy's `structinfo.setfield(...)` /
    /// `arrayinfo.getlenbound(...)` patterns.
    pub fn ensure_ptr_info_arg0(&mut self, op: &Op) -> EnsuredPtrInfo {
        // optimizer.py:464: arg0 = self.get_box_replacement(op.getarg(0))
        let arg0 = self.resolve_operand_box(&op.arg(0)).to_opref();
        // optimizer.py:465-466: if arg0.is_constant(): return info.ConstPtrInfo(arg0)
        //
        // PyPy's `info.ConstPtrInfo(arg0)` wraps the constant box itself,
        // which can be either a `ConstPtr` (Ref) or a `ConstInt` (raw
        // pointer). PyPy doesn't reject either at this point — downstream
        // code calls `_const.getref_base()` and raises `InvalidLoop` only
        // when the ref is null. The Rust port matches that permissive
        // contract: extract whatever GcRef we can (Ref → the gcref, raw
        // pointer Int → cast, anything else → null sentinel) and let the
        // downstream user decide whether to act on it.
        let arg0_const = self
            .get_box_replacement_box(arg0)
            .and_then(|cb| cb.const_value());
        if arg0.is_constant() || arg0_const.is_some() {
            let gcref = match arg0_const {
                Some(Value::Ref(g)) => g,
                Some(Value::Int(bits)) => majit_ir::GcRef(bits as usize),
                // Float / Void / no-constant fall back to a null sentinel —
                // PyPy's getref_base would return null and InvalidLoop guard
                // the dereference at the actual use site.
                _ => majit_ir::GcRef(0),
            };
            // info.py:810-822 `ConstPtrInfo.getstrlen1(mode)`: clone the
            // resolver Arc into the EnsuredPtrInfo so subsequent
            // `getlenbound(Some(mode))` calls can ask the runtime for an
            // exact constant string length without re-borrowing self.
            let resolver = self.string_length_resolver.clone();
            return EnsuredPtrInfo::Constant {
                gcref,
                string_length_resolver: resolver,
            };
        }
        // optimizer.py:467-474:
        //     opinfo = arg0.get_forwarded()
        //     if isinstance(opinfo, info.AbstractVirtualPtrInfo):
        //         return opinfo
        //     elif opinfo is not None:
        //         last_guard_pos = opinfo.get_last_guard_pos()
        //     else:
        //         last_guard_pos = -1
        //     assert opinfo is None or opinfo.__class__ is info.NonNullPtrInfo
        //
        // The PyPy class hierarchy that drives the AbstractVirtualPtrInfo
        // early-return:
        //
        //     PtrInfo
        //       NonNullPtrInfo                       ← only this falls through
        //         AbstractVirtualPtrInfo
        //           AbstractStructPtrInfo
        //             InstancePtrInfo                ← Instance / Virtual
        //             StructPtrInfo                  ← Struct / VirtualStruct
        //           AbstractRawPtrInfo
        //             RawBufferPtrInfo               ← VirtualRawBuffer
        //             RawSlicePtrInfo                ← VirtualRawSlice
        //           ArrayPtrInfo                     ← Array / VirtualArray
        //             ArrayStructInfo                ← VirtualArrayStruct
        //         vstring.StrPtrInfo                 ← Str
        //       ConstPtrInfo                         ← Constant (handled before)
        //
        // The early-return path uses a `&'s mut PtrInfo` whose lifetime
        // matches the function return. Once that mutable borrow is taken,
        // the borrow checker conservatively prevents any further write to
        // the same `_forwarded` slot even on the construction branch (which
        // never executes when we early-returned). To stay close to PyPy's
        // single-`opinfo` shape we read the slot immutably with
        // `get_ptr_info` to compute `last_guard_pos`, drop that read, and
        // then either re-borrow mutably for the early return or fall
        // through to the upgrade.
        // BoxRef-routing read. Owned PtrInfo from `peek_ptr_info` is
        // consumed by `matches!` so no borrow is held when the mutable
        // re-borrow of the BoxRef slot runs below for the early return.
        // optimizer.py:467 opinfo = arg0.get_forwarded(): resolve op.arg(0)
        // box-native to the position's canonical (the info-host
        // `find_producer_op` returns). The Phase-1 heal links the operand's
        // input op to that canonical even at a shared position, so the
        // box-native terminal now carries the PtrInfo `heap`/`virtualize` set.
        let arg0_box = self.resolve_operand_box_opt(&op.arg(0));
        if matches!(
            arg0_box
                .as_ref()
                .and_then(|b| self.peek_ptr_info(&Operand::from_boxref(b))),
            Some(
                PtrInfo::Instance(_)
                    | PtrInfo::Virtual(_)
                    | PtrInfo::Struct(_)
                    | PtrInfo::VirtualStruct(_)
                    | PtrInfo::Array(_)
                    | PtrInfo::VirtualArray(_)
                    | PtrInfo::VirtualArrayStruct(_)
                    | PtrInfo::VirtualRawBuffer(_)
                    | PtrInfo::VirtualRawSlice(_)
                    | PtrInfo::Virtualizable(_)
                    | PtrInfo::Str(_)
            )
        ) {
            // optimizer.py:469: return opinfo. The matches! above required
            // arg0_box to carry a virtual/known PtrInfo, so the terminal
            // BoxRef is already resolved — reuse it instead of re-minting.
            let bx = arg0_box.expect("matched PtrInfo implies a resolved arg0 BoxRef");
            return EnsuredPtrInfo::ForwardedBox(bx);
        }
        let last_guard_pos = if let Some(opinfo) = arg0_box
            .as_ref()
            .and_then(|b| self.peek_ptr_info(&Operand::from_boxref(b)))
        {
            // optimizer.py:474:
            //     assert opinfo is None or opinfo.__class__ is info.NonNullPtrInfo
            debug_assert!(
                matches!(opinfo, PtrInfo::NonNull { .. }),
                "ensure_ptr_info_arg0: existing non-virtual PtrInfo must be NonNullPtrInfo before upgrade, got {:?}",
                opinfo
            );
            // optimizer.py:471: last_guard_pos = opinfo.get_last_guard_pos()
            opinfo.last_guard_pos().unwrap_or(-1)
        } else {
            // optimizer.py:472-473: else: last_guard_pos = -1
            -1
        };
        // optimizer.py:475-495: dispatch on opcode to construct the right
        // PtrInfo class. The Rust port reuses PtrInfo factory constructors
        // (`PtrInfo::array`, `PtrInfo::instance`, `PtrInfo::struct_ptr`,
        // and the StrPtrInfo struct literal).
        let mut new_info = if op.opcode.is_getfield()
            || op.opcode == OpCode::SetfieldGc
            || op.opcode == OpCode::QuasiimmutField
        {
            // optimizer.py:476-484:
            //     descr = op.getdescr()
            //     parent_descr = descr.get_parent_descr()
            //     if parent_descr.is_object():
            //         opinfo = info.InstancePtrInfo(parent_descr)
            //     else:
            //         opinfo = info.StructPtrInfo(parent_descr)
            //     opinfo.init_fields(parent_descr, descr.get_index())
            let ensure_field_descr_arc = op
                .getdescr()
                .expect("ensure_ptr_info_arg0: field op without FieldDescr");
            let field_descr = ensure_field_descr_arc
                .as_field_descr()
                .expect("ensure_ptr_info_arg0: field op without FieldDescr");
            if field_descr.is_w_class() {
                // The `w_class`/typeptr header carries class identity, not a
                // value field, so it has no parent SizeDescr. arg0 is just a
                // non-null instance with no known layout — the same PtrInfo
                // GUARD_CLASS builds (optimizer.py:488-489). OptVirtualize folds
                // this read for virtuals (virtualize.rs `is_w_class`); a
                // non-virtual heap object's `__class__` read reaches here.
                PtrInfo::instance(None, None)
            } else {
                // optimizer.py:479-484: parent_descr.is_object() decides Instance vs Struct.
                let parent_descr = field_descr.get_parent_descr().unwrap_or_else(|| {
                    panic!(
                        "ensure_ptr_info_arg0: FieldDescr.get_parent_descr() returned None \
                         for opcode={:?} descr={:?} field_name={:?} index_in_parent={} \
                         offset={} field_type={:?}; the FieldDescr implementation must \
                         override get_parent_descr() for parity with optimizer.py:478",
                        op.opcode,
                        op.getdescr(),
                        field_descr.field_name(),
                        field_descr.index_in_parent(),
                        field_descr.offset(),
                        field_descr.field_type(),
                    )
                });
                let is_object = parent_descr
                    .as_size_descr()
                    .expect(
                        "ensure_ptr_info_arg0: FieldDescr.get_parent_descr() must point at a SizeDescr",
                    )
                    .is_object();
                let mut new_info = if is_object {
                    PtrInfo::instance(Some(parent_descr.clone()), None)
                } else {
                    PtrInfo::struct_ptr(parent_descr.clone())
                };
                // optimizer.py:484: opinfo.init_fields(parent_descr, descr.get_index())
                // info.py:180-188 init_fields(parent_descr, index) sets self.descr
                // and pre-allocates _fields by parent slot count.
                new_info.init_fields(parent_descr, field_descr.index_in_parent());
                new_info
            }
        } else if op.opcode.is_getarrayitem()
            || op.opcode == OpCode::SetarrayitemGc
            || op.opcode == OpCode::ArraylenGc
        {
            // optimizer.py:485-487: getarrayitem / setarrayitem_gc / arraylen_gc
            // → ArrayPtrInfo(op.getdescr())
            let descr = op
                .getdescr()
                .expect("ensure_ptr_info_arg0: array op without descr");
            PtrInfo::array(descr, crate::optimizeopt::intutils::IntBound::nonnegative())
        } else if op.opcode == OpCode::GuardClass || op.opcode == OpCode::GuardNonnullClass {
            // optimizer.py:488-489: guard_class / guard_nonnull_class
            // → InstancePtrInfo()
            PtrInfo::instance(None, None)
        } else if op.opcode == OpCode::Strlen {
            // optimizer.py:490-491: strlen → StrPtrInfo(mode_string)
            PtrInfo::Str(crate::optimizeopt::info::StrPtrInfo {
                lenbound: None,
                lgtop: None,
                mode: 0,
                length: -1,
                variant: crate::optimizeopt::info::VStringVariant::Ptr,
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            })
        } else if op.opcode == OpCode::Unicodelen {
            // optimizer.py:492-493: unicodelen → StrPtrInfo(mode_unicode)
            PtrInfo::Str(crate::optimizeopt::info::StrPtrInfo {
                lenbound: None,
                lgtop: None,
                mode: 1,
                length: -1,
                variant: crate::optimizeopt::info::VStringVariant::Ptr,
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            })
        } else {
            // optimizer.py:494-495: assert False, "operations %s unsupported"
            panic!("ensure_ptr_info_arg0: opcode {:?} unsupported", op.opcode);
        };
        // optimizer.py:496: assert isinstance(opinfo, info.NonNullPtrInfo)
        // — every constructed PtrInfo above is a NonNullPtrInfo subclass.
        // optimizer.py:497: opinfo.last_guard_pos = last_guard_pos
        new_info.set_last_guard_pos(last_guard_pos);
        // optimizer.py:498: arg0.set_forwarded(opinfo)
        let bx = self.materialize_box_at(arg0);
        use crate::optimizeopt::info::OpInfo;
        bx.set_forwarded_info(OpInfo::ptr(new_info));
        // optimizer.py:499: return opinfo — hand back the BoxRef so subsequent
        // mutations land on the authoritative slot.
        EnsuredPtrInfo::ForwardedBox(bx)
    }

    /// optimizer.py:453-462: make_nonnull_str(op, mode) line-by-line port.
    ///
    /// ```python
    /// def make_nonnull_str(self, op, mode):
    ///     from rpython.jit.metainterp.optimizeopt import vstring
    ///     op = self.get_box_replacement(op)
    ///     if op.is_constant():
    ///         return
    ///     opinfo = op.get_forwarded()
    ///     if isinstance(opinfo, vstring.StrPtrInfo):
    ///         return
    ///     op.set_forwarded(vstring.StrPtrInfo(mode))
    /// ```
    pub fn make_nonnull_str(&self, op: &Operand, mode: u8) {
        use crate::optimizeopt::info::OpInfo;
        // optimizer.py:455: op = self.get_box_replacement(op)
        let op = op.get_box_replacement(false);
        // optimizer.py:457: if op.is_constant(): return
        if op.is_constant() {
            return;
        }
        // optimizer.py:459-460: opinfo = op.get_forwarded();
        //                       if isinstance(opinfo, vstring.StrPtrInfo): return
        if matches!(op.ptr_info().as_deref(), Some(PtrInfo::Str(_))) {
            return;
        }
        // optimizer.py:462: op.set_forwarded(vstring.StrPtrInfo(mode))
        op.set_forwarded_info(OpInfo::ptr(PtrInfo::Str(
            crate::optimizeopt::info::StrPtrInfo {
                lenbound: None,
                lgtop: None,
                mode,
                length: -1,
                variant: crate::optimizeopt::info::VStringVariant::Ptr,
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            },
        )));
    }

    /// rewrite.py:434-435: isinstance(old_guard_op.getdescr(),
    /// compile.ResumeAtPositionDescr).
    /// guard_pos is a _newoperations index (info.py:100-103).
    pub fn is_resume_at_position_guard(&self, guard_pos: i32) -> bool {
        if guard_pos < 0 {
            return false;
        }
        self.new_operations
            .get(guard_pos as usize)
            .and_then(|op| op.getdescr())
            .map_or(false, |descr| descr.is_resume_at_position())
    }

    /// Take ownership of PtrInfo, replacing with None.
    /// Used by force_box to mutate info in-place (RPython parity).
    pub fn take_ptr_info(&self, op: &Operand) -> Option<PtrInfo> {
        use crate::optimizeopt::info::OpInfo;
        use majit_ir::box_ref::Forwarded;
        let resolved = op.get_box_replacement(false);
        // Read terminal's `_forwarded` slot; clone the PtrInfo (if any),
        // drop the Ref borrow, then clear the slot via interior
        // mutability. Const targets are no-op-cleared by
        // `BoxRef::clear_forwarded` per AbstractValue invariant.
        let info = {
            let fw = resolved.get_forwarded();
            match &fw {
                Forwarded::Info(OpInfo::Ptr(rc)) => Some(rc.borrow().clone()),
                _ => None,
            }
        };
        if info.is_some() {
            resolved.clear_forwarded();
        }
        info
    }

    pub fn set_ptr_info(&self, op: &Operand, info: PtrInfo) {
        use crate::optimizeopt::info::OpInfo;
        // Walk chain and write through the terminal slot. Const targets
        // (whose chain walker landed on a `BoxKind::Const`) silently
        // no-op via `set_forwarded_info`'s upstream invariant — Const has
        // no _forwarded slot so any write would assert.
        let resolved = op.get_box_replacement(false);
        if resolved.is_constant() {
            return;
        }
        resolved.set_forwarded_info(OpInfo::ptr(info));
    }

    /// optimizer.py: replace_op_with(old, new_op, ctx)
    /// Replace old opref AND emit the new op.
    pub fn replace_op_with(&mut self, old: OpRef, new_op: Op) -> OpRef {
        let new_ref = self.emit(new_op);
        let b_old = self.get_box_replacement_operand(old);
        let b_new = self.get_box_replacement_operand(new_ref);
        self.make_equal_to(&b_old, &b_new);
        new_ref
    }
}

#[cfg(test)]
pub(crate) fn seed_guard_snapshots_with<F>(
    ops: &[Op],
    mut snapshot_for_guard: F,
) -> (Vec<Op>, SnapshotBoxes)
where
    F: FnMut(&Op) -> Vec<OpRef>,
{
    let mut seeded = ops.to_vec();
    let mut snapshots: SnapshotBoxes = Vec::new();
    let mut next_resume_pos = 0i32;
    for op in seeded.iter_mut().filter(|op| op.opcode.is_guard()) {
        let snapshot_boxes = snapshot_for_guard(op);
        let resume_pos = if op.rd_resume_position.get() >= 0
            && !snapshot_contains(&snapshots, op.rd_resume_position.get())
        {
            op.rd_resume_position.get()
        } else {
            while snapshot_contains(&snapshots, next_resume_pos) {
                next_resume_pos += 1;
            }
            let resume_pos = next_resume_pos;
            next_resume_pos += 1;
            resume_pos
        };
        op.rd_resume_position.set(resume_pos);
        snapshot_insert(
            &mut snapshots,
            resume_pos,
            snapshot_boxes.into_iter().map(SnapshotBox::from).collect(),
        );
    }
    (seeded, snapshots)
}

/// Test fixture helper for optimizer tests whose guard resume state is
/// intentionally irrelevant.  RPython would still have a
/// `capture_resumedata()` entry for such a guard; the explicit empty list
/// models an empty active frame snapshot rather than deriving anything from
/// `guard.fail_args`.
#[cfg(test)]
pub(crate) fn seed_empty_guard_snapshots(ops: &[Op]) -> (Vec<Op>, SnapshotBoxes) {
    seed_guard_snapshots_with(ops, |_| Vec::new())
}

#[cfg(test)]
mod input_ops_index_tests {
    use super::*;
    use majit_ir::OpRef;
    use majit_ir::resoperation::{Op, OpCode};
    use std::rc::Rc;

    fn op_at(pos: OpRef) -> majit_ir::OpRc {
        let op = Op::new(OpCode::SameAsI, &[]);
        op.pos.set(pos);
        Rc::new(op)
    }

    /// `input_ops_index` is a derived O(1) acceleration of
    /// `input_ops.iter().rfind(|o| o.pos == pos)`. Its sole invariant is
    /// that when two seeded producers share a position, the LATER one wins —
    /// matching the `rfind` it replaces and `find_producer_op`'s "a later
    /// emission at the same position always wins" priority. The forward
    /// rebuild guarantees this because a later `insert` at the same key
    /// overwrites the earlier.
    #[test]
    fn input_ops_index_last_occurrence_wins() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 0, 0, 0);
        // A position outside any higher-priority store (new_operations,
        // phase1_emit_ops, resop_refs are all empty here) so the lookup
        // falls through to input_ops_index.
        let pos = OpRef::int_op(100);
        let first = op_at(pos);
        let last = op_at(pos);
        ctx.input_ops = vec![first.clone(), last.clone()];
        ctx.rebuild_input_ops_index();

        let producer = ctx
            .find_producer_op(pos)
            .expect("a producer must be found at the seeded position");
        assert!(
            Rc::ptr_eq(&producer, &last),
            "the last occurrence at a shared position must win"
        );
        assert!(
            !Rc::ptr_eq(&producer, &first),
            "the earlier occurrence must be shadowed by the later one"
        );
    }
}

#[cfg(test)]
mod boxref_forwarding_tests {
    //! BoxRef `_forwarded` invariants: the four writers (`set_ptr_info`,
    //! `setintbound`, `make_constant`, `make_equal_to`) install PyPy-style
    //! forwarding state on the authoritative BoxRef slot.
    use super::*;
    use crate::history::test_support::{bound_inputarg_box, bound_resop_box};
    use crate::optimizeopt::info::{OpInfo, PtrInfo};
    use crate::optimizeopt::intutils::IntBound;
    use majit_ir::box_ref::{BoxRef, Forwarded as BoxForwarded};
    use majit_ir::{InputArgRc, OpRef, Type, Value};

    fn ctx_with_two_int_boxes() -> (OptContext, BoxRef, BoxRef, Vec<InputArgRc>) {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 2, 0, 2);
        let (b0, ia0) = bound_inputarg_box(Type::Int, 0);
        let (b1, ia1) = bound_inputarg_box(Type::Int, 1);
        ctx.seed_boxes_canonical(&[b0.clone(), b1.clone()]);
        (ctx, b0, b1, vec![ia0, ia1])
    }

    /// `make_equal_to(old, new)` plants an `InputArg`-target chain step on
    /// `old`'s `_forwarded` slot (`optimizer.py:394 op.set_forwarded(newop)`
    /// — `newop` is an `AbstractInputArg` here), and `get_box_replacement`
    /// (`resoperation.py:57-68`) walks to a BoxRef bound to `new`'s
    /// `AbstractInputArg` identity. The walker materialises a transient
    /// BoxRef wrapping the same `InputArgRc`, so identity is checked via
    /// the bound handle, not outer `Rc<Box>` pointer equality.
    #[test]
    fn h3_1_replace_op_mirrors_box_forward() {
        let (mut ctx, b0, b1, ia_holder) = ctx_with_two_int_boxes();
        ctx.make_equal_to(&Operand::from_boxref(&b0), &Operand::from_boxref(&b1));
        assert!(matches!(b0.get_forwarded(), BoxForwarded::InputArg(_)));
        let walked = b0.get_box_replacement(false);
        assert!(std::rc::Rc::ptr_eq(
            &walked
                .bound_inputarg()
                .expect("walked terminal carries bound InputArg"),
            &ia_holder[1],
        ));
    }

    /// Forward-reference dup-materialization regression (keystone S2): a
    /// consumer that binds its operand to a position's stand-in BEFORE the
    /// producer at that position is emitted must, after the producer emits,
    /// resolve through to the emitted producer — not freeze on the orphaned
    /// stand-in. This is the `test_guard_true_int_lt_enables_add_ovf_removal`
    /// shape reduced to OptContext primitives: GUARD_TRUE (consumer) is
    /// dispatched and arg-bound to the pos-2 stand-in before INT_LT (producer)
    /// emits its `new_operations` clone. Without the `emit` `set_forwarded_op`
    /// link the box-native walk of the consumer arg terminates on the stand-in
    /// while the OpRef path resolves the producer, tripping `resolve_box_box`'s
    /// divergence witness (panic under `cfg(debug_assertions)`).
    #[test]
    fn s2_forward_reference_consumer_follows_emitted_producer() {
        use majit_ir::resoperation::{Op, OpCode};
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 2, 0, 2);
        let (b0, _ia0) = bound_inputarg_box(Type::Int, 0);
        let (b1, _ia1) = bound_inputarg_box(Type::Int, 1);
        ctx.seed_boxes_canonical(&[b0.clone(), b1.clone()]);

        // pos 2 = an INT_LT result not yet emitted; mint the stand-in the
        // recorder / `bind_input_resops` seeds into resop_refs + live_synthetics.
        let pos2 = OpRef::int_op(2);
        let _standin = ctx.mint_synthetic_resop(pos2, Type::Int);

        // A consumer (GUARD_TRUE) dispatched before the producer binds its
        // operand to the stand-in — the forward reference.
        let consumer_arg = ctx.materialize_box_at(pos2);

        // The producer emits at pos 2.
        let mut producer = Op::new(
            OpCode::IntLt,
            &[Operand::from_boxref(&b0), Operand::from_boxref(&b1)],
        );
        producer.pos.set(pos2);
        ctx.emit(producer);

        // The frozen consumer arg must resolve to the emitted producer; the
        // call also exercises resolve_box_box's witness, which would fire on
        // the box-native-vs-OpRef divergence without the emit rebind.
        let resolved = ctx.resolve_box_box(&consumer_arg);
        let producer_b = ctx
            .find_producer_op(pos2)
            .expect("producer emitted at pos 2");
        assert!(
            resolved
                .bound_op()
                .is_some_and(|op| std::rc::Rc::ptr_eq(&op, &producer_b)),
            "consumer must resolve to the emitted producer, not the orphaned stand-in"
        );
    }

    /// `box.clear_forwarded()` resets a previously-set forwarding slot
    /// back to `Forwarded::None`.  PyPy has no `make_equal_to(op, None)`
    /// path; chain reset happens on the box directly.
    #[test]
    fn h3_1_clear_forwarded_resets_box_forward() {
        let (mut ctx, b0, b1, _ia_holder) = ctx_with_two_int_boxes();
        ctx.make_equal_to(&Operand::from_boxref(&b0), &Operand::from_boxref(&b1));
        b0.clear_forwarded();
        assert!(matches!(b0.get_forwarded(), BoxForwarded::None));
    }

    /// `optimizer.py:387-400 make_equal_to` Info transfer parity: when
    /// `old` carries `Forwarded::IntBound(_)` and is forwarded to a
    /// non-constant `new`, the IntBound moves to `new`'s slot.
    #[test]
    fn h3_1_replace_op_transfers_int_bound_to_new() {
        let (mut ctx, b0, b1, ia_holder) = ctx_with_two_int_boxes();
        let bound = IntBound::from_constant(7);
        ctx.setintbound(&Operand::from_boxref(&b0), &bound);
        ctx.make_equal_to(&Operand::from_boxref(&b0), &Operand::from_boxref(&b1));
        // After: old's IntBound transferred to new (PyPy:
        // `newop.set_forwarded(opinfo)`). old now forwards to new.
        match &b1.get_forwarded() {
            BoxForwarded::Info(OpInfo::IntBound(b)) => assert_eq!(b.borrow().lower, 7),
            other => panic!("BoxRef[1] should carry IntBound, got {:?}", other),
        }
        // old's slot now points to new. Bound-InputArg target routes through
        // `set_forwarded_inputarg`, so the slot carries
        // `Forwarded::InputArg(Weak<InputArg>)`; chain walk lands on a
        // transient BoxRef sharing `ia_holder[1]`'s identity.
        assert!(matches!(b0.get_forwarded(), BoxForwarded::InputArg(_)));
        let walked = b0.get_box_replacement(false);
        assert!(std::rc::Rc::ptr_eq(
            &walked
                .bound_inputarg()
                .expect("walked terminal carries bound InputArg"),
            &ia_holder[1],
        ));
    }

    /// `optimizer.py:400` guard: transfer is **skipped** when `new` is
    /// constant. PyPy short-circuits via `not newop.is_constant()`.
    #[test]
    fn h3_1_replace_op_skips_info_transfer_when_new_is_constant() {
        let (mut ctx, b0, _b1, _ia_holder) = ctx_with_two_int_boxes();
        // Seed an IntBound on old.
        let bound = IntBound::from_constant(42);
        ctx.setintbound(&Operand::from_boxref(&b0), &bound);
        // Forward to an inline-Const target — history.py:227 ConstInt.value
        // carries the value on the Box itself, no const_pool seed needed.
        let const_opref = OpRef::const_int(42);
        let b_const = ctx.materialize_box_at(const_opref);
        ctx.make_equal_to(&Operand::from_boxref(&b0), &Operand::from_boxref(&b_const));
        // The IntBound on old is gone (overwritten by Forwarded::Op(const)).
        // Const targets do not carry transferred info — PyPy skips this case.
        match &b0.get_forwarded() {
            BoxForwarded::Const(majit_ir::Const::Int(v)) => assert_eq!(*v, 42),
            other => panic!("expected b0 to forward to Const, got {:?}", other),
        }
    }

    /// `set_ptr_info(opref, info)` mirrors `box.set_forwarded(PtrInfo)`.
    #[test]
    fn h3_1_set_ptr_info_mirrors_box_info() {
        // PtrInfo applies to ref-typed boxes.
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 1, 0, 1);
        let (b, _ia) = bound_inputarg_box(Type::Ref, 0);
        ctx.seed_boxes_canonical(&[b.clone()]);
        let info = PtrInfo::NonNull { last_guard_pos: -1 };
        ctx.set_ptr_info(&Operand::from_boxref(&b), info);
        match &b.get_forwarded() {
            BoxForwarded::Info(OpInfo::Ptr(rc))
                if matches!(&*rc.borrow(), PtrInfo::NonNull { .. }) => {}
            other => panic!("expected Info(Ptr(NonNull)), got {:?}", other),
        }
    }

    /// PyPy optimizer.py:432 parity: after
    /// `make_constant_arg(&box, Value::Ref(_))` writes the constant onto
    /// the InputArg's `_forwarded` slot, a subsequent `make_nonnull(opref)`
    /// MUST NOT overwrite the Const with `OpInfo::Ptr(NonNull)`.
    #[test]
    fn audit_a_make_nonnull_preserves_box_constant_slot() {
        use majit_ir::GcRef;
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 1, 0, 1);
        let (b, _ia) = bound_inputarg_box(Type::Ref, 0);
        ctx.seed_boxes_canonical(&[b.clone()]);
        ctx.make_constant_arg(&Operand::from_boxref(&b), Value::Ref(GcRef(0xdead_beef)));
        match &b.get_forwarded() {
            BoxForwarded::Const(majit_ir::Const::Ref(g)) => {
                assert_eq!(*g, GcRef(0xdead_beef));
            }
            other => panic!(
                "expected Forwarded::Const(Ref) post make_constant, got {:?}",
                other
            ),
        }
        // Use the BoxRef form because `make_nonnull` writes to the box's
        // forwarded slot.
        ctx.make_nonnull(&Operand::from_boxref(&b));
        match &b.get_forwarded() {
            BoxForwarded::Const(majit_ir::Const::Ref(g)) => {
                assert_eq!(
                    *g,
                    GcRef(0xdead_beef),
                    "make_nonnull must not overwrite the Const slot"
                );
            }
            other => panic!("make_nonnull clobbered Const slot — got {:?}", other),
        }
    }

    /// `resoperation.py:57-68 get_box_replacement` + `history.py:188
    /// Const.is_constant()` parity: after the chain walker advances into
    /// a `Forwarded::Const(constval)` target, `is_constant()` on the
    /// terminal box reports True. Covers both encodings of "this slot is
    /// a known constant": (a) Const-namespace OpRef terminus, and (b)
    /// `Forwarded::Const(constval)` produced by `optimizer.py:432
    /// set_forwarded(constbox)` — equivalent to RPython's single
    /// `is_constant()` predicate after `get_box_replacement`.
    #[test]
    fn audit_a_chain_walker_reaches_constant_through_forwarded_const() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 2, 0, 2);
        let (b0, _ia0) = bound_inputarg_box(Type::Int, 0);
        let (b1, _ia1) = bound_inputarg_box(Type::Int, 1);
        ctx.seed_boxes_canonical(&[b0.clone(), b1.clone()]);
        // (a) Const-namespace OpRef terminates at a Const box.
        let const_opref = OpRef::const_int(7);
        let const_box = ctx.materialize_box_at(const_opref);
        assert!(const_box.get_box_replacement(false).is_constant());
        // (b) `Forwarded::Const(constval)` chain on a non-Const-namespace OpRef.
        let b0_iarg = ctx.materialize_operand_at(OpRef::input_arg_int(0));
        ctx.make_equal_to(&b0_iarg, &Operand::from_boxref(&const_box));
        let b0_after = ctx.materialize_box_at(OpRef::input_arg_int(0));
        assert!(b0_after.get_box_replacement(false).is_constant());
        // `Forwarded::Const(constval)` planted directly via set_forwarded_const.
        b1.set_forwarded_const(majit_ir::Const::Int(42));
        assert!(b1.get_box_replacement(false).is_constant());
        // Negative case: BoxRef with no constant forwarding.
        let (nb, _ia_nb) = bound_inputarg_box(Type::Int, 0);
        assert!(!nb.get_box_replacement(false).is_constant());
    }

    /// `make_constant` mirrors PyPy optimizer.py:432
    /// `box.set_forwarded(constbox)` — Const variant.
    #[test]
    fn h3_1_make_constant_mirrors_box_info_constant() {
        let (mut ctx, b0, _b1, _ia_holder) = ctx_with_two_int_boxes();
        ctx.make_constant_arg(&Operand::from_boxref(&b0), Value::Int(42));
        match &b0.get_forwarded() {
            BoxForwarded::Const(majit_ir::Const::Int(v)) => {
                assert_eq!(*v, 42);
            }
            other => panic!("expected Forwarded::Const(Int 42), got {:?}", other),
        }
    }

    /// `setintbound(opref, bound)` mirrors `box.set_forwarded(IntBound)`.
    #[test]
    fn h3_1_setintbound_mirrors_box_info() {
        let (mut ctx, b0, _b1, _ia_holder) = ctx_with_two_int_boxes();
        let bound = IntBound::from_constant(7);
        ctx.setintbound(&Operand::from_boxref(&b0), &bound);
        match &b0.get_forwarded() {
            BoxForwarded::Info(OpInfo::IntBound(b)) => {
                let b = b.borrow();
                assert_eq!(b.lower, 7);
                assert_eq!(b.upper, 7);
            }
            other => panic!("expected Info(IntBound), got {:?}", other),
        }
    }

    /// `make_equal_to(old, ConstX)` mirrors onto `old_box.set_forwarded_const(
    /// const_value)`. Per RPython parity (`optimizer.py:393`,
    /// `history.py:220` ConstInt construction), the const target is built
    /// fresh from `const_pool[const_index]` per call site — no dedup, value
    /// equality via `same_constant`. The mirror must record the same Value
    /// as the seeded constant via `Forwarded::Const`.
    #[test]
    fn h3_4_replace_op_const_target_mirrors_value_box() {
        let (mut ctx, b0, _b1, _ia_holder) = ctx_with_two_int_boxes();
        let const_opref = OpRef::const_int(42);
        let b_const = ctx.materialize_box_at(const_opref);
        ctx.make_equal_to(&Operand::from_boxref(&b0), &Operand::from_boxref(&b_const));
        match &b0.get_forwarded() {
            BoxForwarded::Const(majit_ir::Const::Int(v)) => {
                assert_eq!(*v, 42);
            }
            other => panic!("expected Forwarded::Const(Int 42), got {:?}", other),
        }
    }

    /// resoperation.py:58 get_box_replacement(not_const=True) stops before
    /// stepping into a Const target. This is required for guard fail args:
    /// resume numbering encodes constants as TAGCONST, while backend liveboxes
    /// keep the runtime Box identity.
    #[test]
    fn get_box_replacement_not_const_stops_before_const_target() {
        let (mut ctx, b0, b1, _ia_holder) = ctx_with_two_int_boxes();
        let const_opref = OpRef::const_int(42);
        let b_const = ctx.materialize_box_at(const_opref);
        ctx.make_equal_to(&Operand::from_boxref(&b0), &Operand::from_boxref(&b_const));

        assert_eq!(
            ctx.get_box_replacement(OpRef::input_arg_typed(0, Type::Int))
                .to_opref(),
            const_opref
        );
        assert_eq!(
            ctx.get_box_replacement_impl(OpRef::input_arg_typed(0, Type::Int), true),
            OpRef::input_arg_typed(0, Type::Int)
        );

        ctx.make_equal_to(&Operand::from_boxref(&b1), &Operand::from_boxref(&b0));
        assert_eq!(
            ctx.get_box_replacement(OpRef::input_arg_typed(1, Type::Int))
                .to_opref(),
            const_opref
        );
        assert_eq!(
            ctx.get_box_replacement_impl(OpRef::input_arg_typed(1, Type::Int), true),
            OpRef::input_arg_typed(0, Type::Int)
        );
    }

    /// Body optimization's inherited emit-position region
    /// `[0..phase2_inputarg_base)` carries placeholder resop hosts (preamble
    /// emit ops do NOT appear in the body trace iteration, so the body iter
    /// has no `cls()` allocation for them). Replicates the import_state
    /// pattern at unroll.rs:3105:
    ///
    ///   1. `make_equal_to(source_p2, target_p1)` writes
    ///      `source._forwarded = Box(placeholder_at_target_p1.raw)`.
    ///   2. Body import writes info via `set_ptr_info(target_p1, info)`,
    ///      setting `placeholder._forwarded = Info(info)`.
    ///   3. Reading source via `peek_ptr_info` walks
    ///      `source → placeholder` and sees the placeholder's info.
    ///
    /// PyPy parity is preserved structurally even though the preamble's actual
    /// Box is not shared with the body context: the placeholder absorbs body
    /// import writes the same way the preamble Box would in PyPy.
    #[test]
    fn h3_4_phase2_placeholder_forwarding_yields_consistent_reads() {
        // Layout: indices 0..2 are preamble emit-position placeholders,
        // indices 2..4 are body inputarg BoxRefs. PyPy `box.type`
        // invariant prevents `make_equal_to(Ref, Void)` (cross-type forward),
        // so place Ref-typed boxes on both sides — the test models a
        // preamble RefOp result acting as the import target.
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 4, 0, 4);
        let (placeholder_target, _op_target) = bound_resop_box(Type::Ref, 0);
        let (placeholder_other, _op_other) = bound_resop_box(Type::Ref, 1);
        let (source_box, _ia_source) = bound_inputarg_box(Type::Ref, 2);
        let (other_box, _ia_other) = bound_inputarg_box(Type::Ref, 3);
        ctx.seed_boxes_canonical(&[
            placeholder_target.clone(),
            placeholder_other.clone(),
            source_box.clone(),
            other_box.clone(),
        ]);

        // BoxRef-first chain walker reconstructs the variant tag from
        // `box.type_()`; placeholders and source are both Ref, so use the
        // typed factories that match.
        let target_p1 = OpRef::ref_op(0);
        let source_p2 = OpRef::input_arg_ref(2);

        // Step 1: import_state's `source.set_forwarded(target)` equivalent.
        ctx.make_equal_to(
            &Operand::from_boxref(&source_box),
            &Operand::from_boxref(&placeholder_target),
        );

        // Step 2: setinfo_from_preamble's terminal write.
        // `setinfo_from_preamble(source, info)` first walks the chain via
        // `get_box_replacement` (mod.rs:2538) which returns `target_p1`,
        // then calls `set_ptr_info(target_p1, info)`. Replicate the
        // post-walk write directly.
        let info = PtrInfo::NonNull { last_guard_pos: -1 };
        let target_p1_box = ctx.materialize_operand_at(target_p1);
        ctx.set_ptr_info(&target_p1_box, info.clone());

        // Read via BoxRef-routing path: walk source's chain to placeholder.
        let source_p2_box = ctx.get_box_replacement_operand(source_p2);
        let via_box = ctx
            .peek_ptr_info(&source_p2_box)
            .expect("BoxRef path must see info");
        assert!(matches!(via_box, PtrInfo::NonNull { .. }));

        // Chain walk lands on target_p1.
        let resolved = ctx.get_replacement_opref(source_p2);
        assert_eq!(resolved, target_p1);

        // Placeholder Box absorbed the mirror write, so its _forwarded now
        // carries the info — equivalent to PyPy's preamble Box receiving
        // setinfo_from_preamble.
        match &placeholder_target.get_forwarded() {
            BoxForwarded::Info(OpInfo::Ptr(rc))
                if matches!(&*rc.borrow(), PtrInfo::NonNull { .. }) => {}
            other => panic!(
                "placeholder must carry Info(NonNull) after set_ptr_info, got {:?}",
                other
            ),
        }
    }

    /// Complementary to
    /// `h3_4_phase2_placeholder_forwarding_yields_consistent_reads`. Pre-import
    /// (no `setinfo_from_preamble` call), reading `target_p1` info via either
    /// path returns None — consistent within pyre. PyPy parity here depends on
    /// `ExportedState.exported_infos` (`unroll.py:529` canonical field)
    /// carrying every preamble op info the body needs; the placeholder cannot
    /// fabricate preamble info that wasn't exported. PyPy itself uses the same
    /// serialization map for the import (PyPy's body optimizer reads exported_infos
    /// → setinfo_from_preamble too), so structural narrowness here matches
    /// PyPy's own dispatch.
    #[test]
    fn h3_4_phase2_placeholder_without_import_returns_none_consistently() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 4, 0, 4);
        // Same Ref-typed alignment as the sibling test: forwarding a Ref
        // source to the placeholder requires placeholder type to match
        // (PyPy `box.type` invariant; `make_equal_to` cross-type assertion).
        let (placeholder_target, _op_target) = bound_resop_box(Type::Ref, 0);
        let (placeholder_other, _op_other) = bound_resop_box(Type::Ref, 1);
        let (source_box, _ia_source) = bound_inputarg_box(Type::Ref, 2);
        let (other_box, _ia_other) = bound_inputarg_box(Type::Ref, 3);
        ctx.seed_boxes_canonical(&[
            placeholder_target.clone(),
            placeholder_other.clone(),
            source_box.clone(),
            other_box.clone(),
        ]);

        let target_p1 = OpRef::ref_op(0);
        let source_p2 = OpRef::input_arg_ref(2);

        // import_state's make_equal_to fires, but the body import chose NOT to import
        // info (e.g. exported_infos didn't carry an entry for target_p1).
        ctx.make_equal_to(
            &Operand::from_boxref(&source_box),
            &Operand::from_boxref(&placeholder_target),
        );

        // BoxRef-routing reader: chain walks source → placeholder → None.
        let source_p2_box = ctx.get_box_replacement_operand(source_p2);
        assert!(ctx.peek_ptr_info(&source_p2_box).is_none());

        // Legacy Vec reader: chain walks source → target_p1 → None
        // (the body's fresh Vec has no entry for target_p1).
        let resolved = ctx.get_replacement_opref(source_p2);
        assert_eq!(resolved, target_p1);

        // Placeholder Box was not mutated (no info import fired) — still None.
        assert!(matches!(
            placeholder_target.get_forwarded(),
            BoxForwarded::None
        ));
    }

    /// With the canonical slot seeded and no forwarding, the
    /// BoxRef-returning reader resolves the slot's bound InputArg.
    /// `resoperation.py:57-68` walker terminates on `None` immediately.
    #[test]
    fn h3_2b_get_box_replacement_box_returns_pool_entry_when_no_forward() {
        let (ctx, _b0, _b1, ia_holder) = ctx_with_two_int_boxes();
        let got = ctx
            .get_box_replacement_box(OpRef::input_arg_typed(0, Type::Int))
            .expect("canonical store resolves the slot");
        // No forwarding: the resolver materialises a fresh terminal BoxRef
        // bound to the same `InputArgRc` as the seeded slot.
        assert!(std::rc::Rc::ptr_eq(
            &got.bound_inputarg()
                .expect("resolved terminal carries bound InputArg"),
            &ia_holder[0],
        ));
    }

    /// With a forwarding chain installed via `make_equal_to`, the
    /// BoxRef walker reaches the terminal Box (`b1`). RPython parity:
    /// `optimizer.py:393 box.set_forwarded(newop)` → reader walks until
    /// `Forwarded::None` and returns the last Box. The walker materialises
    /// a transient BoxRef wrapping `b1`'s bound `InputArgRc`, so terminal
    /// identity is checked via the shared `InputArg` handle rather than
    /// outer `Rc<Box>` pointer equality.
    #[test]
    fn h3_2b_get_box_replacement_box_walks_forwarded_chain() {
        let (mut ctx, b0, b1, ia_holder) = ctx_with_two_int_boxes();
        ctx.make_equal_to(&Operand::from_boxref(&b0), &Operand::from_boxref(&b1));
        let got = ctx
            .get_box_replacement_box(OpRef::input_arg_typed(0, Type::Int))
            .expect("bound box resolves");
        assert!(std::rc::Rc::ptr_eq(
            &got.bound_inputarg()
                .expect("walked terminal carries bound InputArg"),
            &ia_holder[1],
        ));
        // b0 itself is not the terminal.
        assert_ne!(got, b0);
    }

    /// With no seeded canonical stores (test/retrace baseline) the
    /// BoxRef-returning reader returns `None`; the OpRef-returning walker
    /// cannot resolve a Box identity without a bound producer either.
    #[test]
    fn h3_2b_get_box_replacement_box_returns_none_when_pool_empty() {
        let ctx = OptContext::with_num_inputs_and_start_pos(0, 2, 0, 2);
        // No seeded producer: no Box identity to resolve.
        assert!(ctx.get_box_replacement_box(OpRef::int_op(0)).is_none());
    }

    /// `OpRef::NONE` sentinel returns `None` — the BoxRef reader
    /// has no Box to root the walk on. The OpRef-returning reader handles
    /// the sentinel independently by returning it unchanged.
    #[test]
    fn h3_2b_get_box_replacement_box_handles_none_sentinel() {
        let (ctx, _b0, _b1, _ia_holder) = ctx_with_two_int_boxes();
        assert!(ctx.get_box_replacement_box(OpRef::NONE).is_none());
    }

    /// When the chain terminates at `Forwarded::Info(_)`, the
    /// walker returns the Box that holds the Info — `box_ref.rs::BoxRef::
    /// get_box_replacement` stops before descending into Info, matching
    /// PyPy `resoperation.py:60 isinstance(next, AbstractInfo)`.
    #[test]
    fn h3_2b_get_box_replacement_box_stops_at_info_terminal() {
        let (mut ctx, b0, _b1, ia_holder) = ctx_with_two_int_boxes();
        ctx.setintbound(&Operand::from_boxref(&b0), &IntBound::from_constant(7));
        let got = ctx
            .get_box_replacement_box(OpRef::input_arg_typed(0, Type::Int))
            .expect("canonical store resolves the slot");
        // Walker terminates at the slot (its `_forwarded` is Info, not a
        // chain step); the resolved BoxRef shares b0's bound InputArg.
        assert!(std::rc::Rc::ptr_eq(
            &got.bound_inputarg()
                .expect("resolved terminal carries bound InputArg"),
            &ia_holder[0],
        ));
    }

    // BoxRef-routing helpers `is_virtual` / `is_nonnull` read the same
    // `_forwarded` slot that PyPy's getptrinfo() inspects.

    fn ctx_with_one_ref_box() -> (OptContext, BoxRef) {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 1, 0, 1);
        let (b, ia) = bound_inputarg_box(Type::Ref, 0);
        ctx.seed_boxes_canonical(&[b.clone()]);
        // Keep the InputArgRc alive in ctx so the Weak<InputArg> in
        // `b.inputarg_handle` upgrades across the test body.
        ctx.inputarg_refs = vec![ia];
        (ctx, b)
    }

    #[derive(Debug)]
    struct DummySizeDescr;
    impl majit_ir::Descr for DummySizeDescr {}

    #[test]
    fn h3_2c_is_virtual_matches_legacy_when_pool_plumbed() {
        let (mut ctx, b) = ctx_with_one_ref_box();
        let info = PtrInfo::Virtual(crate::optimizeopt::info::VirtualInfo {
            descr: std::sync::Arc::new(DummySizeDescr),
            known_class: None,
            ob_type_descr: None,
            fields: Vec::new(),
            last_guard_pos: -1,
            avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
        });
        ctx.set_ptr_info(&Operand::from_boxref(&b), info);
        assert!(
            ctx.peek_ptr_info(&Operand::from_boxref(&b))
                .is_some_and(|i| i.is_virtual())
        );
        assert!(ctx.is_virtual(&Operand::from_boxref(&b)));
    }

    #[test]
    fn h3_2c_is_virtual_returns_false_for_nonnull_only() {
        let (mut ctx, b) = ctx_with_one_ref_box();
        ctx.set_ptr_info(
            &Operand::from_boxref(&b),
            PtrInfo::NonNull { last_guard_pos: -1 },
        );
        assert!(!ctx.is_virtual(&Operand::from_boxref(&b)));
    }

    #[test]
    fn h3_2c_is_virtual_returns_false_for_unset() {
        let (ctx, b) = ctx_with_one_ref_box();
        assert!(!ctx.is_virtual(&Operand::from_boxref(&b)));
    }

    #[test]
    fn h3_2c_is_nonnull_matches_set_info() {
        let (mut ctx, b) = ctx_with_one_ref_box();
        ctx.set_ptr_info(
            &Operand::from_boxref(&b),
            PtrInfo::NonNull { last_guard_pos: -1 },
        );
        assert!(
            ctx.peek_ptr_info(&Operand::from_boxref(&b))
                .is_some_and(|i| i.is_nonnull())
        );
        assert!(ctx.is_nonnull(&Operand::from_boxref(&b)));
    }

    #[test]
    fn h3_2c_is_nonnull_returns_false_for_unset() {
        let (ctx, b) = ctx_with_one_ref_box();
        assert!(!ctx.is_nonnull(&Operand::from_boxref(&b)));
    }

    #[test]
    fn h3_2c_peek_intbound_box_matches_legacy_when_pool_plumbed() {
        let (mut ctx, b0, _b1, _ia_holder) = ctx_with_two_int_boxes();
        ctx.setintbound(&Operand::from_boxref(&b0), &IntBound::from_constant(42));
        let legacy = ctx
            .peek_intbound(OpRef::input_arg_int(0))
            .expect("legacy bound");
        let via_box = ctx
            .peek_intbound_box(&Operand::from_boxref(&b0))
            .expect("box bound");
        assert!(legacy.is_constant());
        assert_eq!(legacy.get_constant_int(), 42);
        assert!(via_box.is_constant());
        assert_eq!(via_box.get_constant_int(), 42);
    }

    #[test]
    fn h3_2c_peek_intbound_box_returns_none_for_unset() {
        let (ctx, b0, _b1, _ia_holder) = ctx_with_two_int_boxes();
        assert!(ctx.peek_intbound_box(&Operand::from_boxref(&b0)).is_none());
    }

    #[test]
    fn h3_2c_last_guard_pos_matches_legacy_when_pool_plumbed() {
        let (mut ctx, b) = ctx_with_one_ref_box();
        ctx.set_ptr_info(
            &Operand::from_boxref(&b),
            PtrInfo::NonNull { last_guard_pos: 5 },
        );
        assert_eq!(ctx.last_guard_pos(&Operand::from_boxref(&b)), Some(5));
        assert_eq!(
            ctx.peek_ptr_info(&Operand::from_boxref(&b))
                .and_then(|i| i.get_last_guard_pos()),
            Some(5)
        );
    }

    #[test]
    fn h3_2c_last_guard_pos_returns_none_for_unset() {
        let (ctx, b) = ctx_with_one_ref_box();
        assert!(ctx.last_guard_pos(&Operand::from_boxref(&b)).is_none());
    }

    #[test]
    fn h3_2c_last_guard_pos_returns_none_when_no_recorded_guard() {
        let (mut ctx, b) = ctx_with_one_ref_box();
        // info.py:91 last_guard_pos == -1 → get_last_guard_pos returns None.
        ctx.set_ptr_info(
            &Operand::from_boxref(&b),
            PtrInfo::NonNull { last_guard_pos: -1 },
        );
        assert!(ctx.last_guard_pos(&Operand::from_boxref(&b)).is_none());
    }

    #[test]
    fn h3_2c_is_virtualizable_via_box_matches_legacy_when_pool_plumbed() {
        let (mut ctx, b) = ctx_with_one_ref_box();
        ctx.set_ptr_info(
            &Operand::from_boxref(&b),
            PtrInfo::Virtualizable(crate::optimizeopt::info::VirtualizableFieldState {
                fields: Vec::new(),
                field_descrs: Vec::new(),
                arrays: Vec::new(),
                last_guard_pos: -1,
            }),
        );
        assert!(ctx.is_virtualizable(&Operand::from_boxref(&b)));
        assert!(matches!(
            ctx.peek_ptr_info(&Operand::from_boxref(&b)),
            Some(PtrInfo::Virtualizable(_))
        ));
    }

    #[test]
    fn h3_2c_is_virtualizable_returns_false_for_nonnull_only() {
        let (mut ctx, b) = ctx_with_one_ref_box();
        ctx.set_ptr_info(
            &Operand::from_boxref(&b),
            PtrInfo::NonNull { last_guard_pos: -1 },
        );
        assert!(!ctx.is_virtualizable(&Operand::from_boxref(&b)));
    }

    #[test]
    fn h3_2c_is_virtualizable_returns_false_for_unset() {
        let (ctx, b) = ctx_with_one_ref_box();
        assert!(!ctx.is_virtualizable(&Operand::from_boxref(&b)));
    }

    #[test]
    fn h3_2c_has_ptr_info_matches_set_info() {
        let (mut ctx, b) = ctx_with_one_ref_box();
        ctx.set_ptr_info(
            &Operand::from_boxref(&b),
            PtrInfo::NonNull { last_guard_pos: -1 },
        );
        assert!(ctx.has_ptr_info(&Operand::from_boxref(&b)));
        assert!(ctx.peek_ptr_info(&Operand::from_boxref(&b)).is_some());
    }

    #[test]
    fn h3_2c_has_ptr_info_returns_false_for_unset() {
        let (ctx, b) = ctx_with_one_ref_box();
        assert!(!ctx.has_ptr_info(&Operand::from_boxref(&b)));
    }

    #[test]
    fn h3_2c_peek_ptr_info_returns_set_info() {
        let (mut ctx, b) = ctx_with_one_ref_box();
        ctx.set_ptr_info(
            &Operand::from_boxref(&b),
            PtrInfo::NonNull { last_guard_pos: 5 },
        );
        let via_box = ctx
            .peek_ptr_info(&Operand::from_boxref(&b))
            .expect("box clone");
        assert!(matches!(via_box, PtrInfo::NonNull { last_guard_pos: 5 }));
    }

    #[test]
    fn h3_2c_peek_ptr_info_returns_none_for_unset() {
        let (ctx, b) = ctx_with_one_ref_box();
        assert!(ctx.peek_ptr_info(&Operand::from_boxref(&b)).is_none());
    }

    // `with_ptr_info_mut(box, |info| ...)` runs a closure against the
    // `&mut PtrInfo` stored on `box._forwarded::Info` so subsequent
    // BoxRef-routing readers (`peek_ptr_info`, `last_guard_pos`) see
    // the mutation.

    #[test]
    fn h3_2c_with_ptr_info_mut_mirrors_after_mutation_when_pool_plumbed() {
        let (mut ctx, b) = ctx_with_one_ref_box();
        ctx.set_ptr_info(
            &Operand::from_boxref(&b),
            PtrInfo::NonNull { last_guard_pos: 0 },
        );
        // Pre-condition: BoxRef snapshot matches legacy at pos 0.
        assert_eq!(ctx.last_guard_pos(&Operand::from_boxref(&b)), Some(0));
        // Mutate inner state via closure.
        let returned = ctx
            .with_ptr_info_mut(&Operand::from_boxref(&b), |info| {
                info.set_last_guard_pos(42);
                "ok"
            })
            .expect("closure runs");
        assert_eq!(returned, "ok");
        // Post-condition: BoxRef snapshot reflects mutation (mirror ran).
        assert_eq!(ctx.last_guard_pos(&Operand::from_boxref(&b)), Some(42));
        assert_eq!(
            ctx.peek_ptr_info(&Operand::from_boxref(&b))
                .and_then(|i| i.get_last_guard_pos()),
            Some(42)
        );
    }

    #[test]
    fn h3_2c_with_ptr_info_mut_returns_none_for_unset() {
        let (ctx, b) = ctx_with_one_ref_box();
        // No PtrInfo installed at OpRef(0).
        let invoked = std::cell::Cell::new(false);
        let result = ctx.with_ptr_info_mut(&Operand::from_boxref(&b), |_info| {
            invoked.set(true);
        });
        assert!(result.is_none());
        assert!(!invoked.get(), "closure must not run when info is absent");
    }

    /// `PtrInfoHandle::Live` preserves RPython `_forwarded` object
    /// identity: two handles cloned from the same `_forwarded` cell
    /// satisfy `same_info` and observe each other's in-place
    /// mutations — Python `is`-equivalent semantics for non-ConstPtrInfo.
    #[test]
    fn ptr_info_handle_live_identity_propagates_mutation() {
        let (mut ctx, b) = ctx_with_one_ref_box();
        ctx.set_ptr_info(
            &Operand::from_boxref(&b),
            PtrInfo::NonNull { last_guard_pos: 0 },
        );
        let h1 = ctx
            .getptrinfo_handle(&Operand::from_boxref(&b))
            .expect("Live handle for installed PtrInfo");
        let h2 = ctx
            .getptrinfo_handle(&Operand::from_boxref(&b))
            .expect("second call must return another clone of the same cell");
        assert!(
            h1.same_info(&h2),
            "two handles into the same _forwarded cell must satisfy same_info"
        );
        // Mutation through h1 visible through h2 — RPython
        // `opinfo._known_class = ...` propagation.
        {
            let mut m = h1.borrow_mut().expect("Live handle borrows mutably");
            m.set_last_guard_pos(99);
        }
        assert_eq!(
            h2.borrow().get_last_guard_pos(),
            Some(99),
            "h2 must observe h1's mutation (shared Rc cell)"
        );
    }

    /// `ConstPtrInfo.same_info` is value-based: RPython overrides the
    /// base identity check and compares `_const.same_constant(other._const)`.
    #[test]
    fn ptr_info_handle_const_arms_use_constptr_same_info() {
        use majit_ir::GcRef;
        let h1 = PtrInfoHandle::Const(PtrInfo::Constant(GcRef(0x1000)));
        let h2 = PtrInfoHandle::Const(PtrInfo::Constant(GcRef(0x1000)));
        let h3 = PtrInfoHandle::Const(PtrInfo::Constant(GcRef(0x2000)));
        assert!(
            h1.same_info(&h2),
            "two ConstPtrInfo handles for the same const must be same_info"
        );
        assert!(!h1.same_info(&h3), "different constants are not same_info");
    }

    /// The ConstPtrInfo override applies even when one side is a live
    /// `_forwarded` cell carrying the ConstPtrInfo-equivalent payload.
    #[test]
    fn ptr_info_handle_const_and_live_constant_use_constptr_same_info() {
        use crate::optimizeopt::info::OpInfo;
        use majit_ir::{GcRef, Type};

        let (b, _op) = bound_resop_box(Type::Ref, 0);
        b.set_forwarded_info(OpInfo::ptr(PtrInfo::Constant(GcRef(0x1000))));
        let live = PtrInfoHandle::Live(
            b.ptr_info_handle()
                .expect("live forwarded ConstPtrInfo-equivalent handle"),
        );
        let same_const = PtrInfoHandle::Const(PtrInfo::Constant(GcRef(0x1000)));
        let different_const = PtrInfoHandle::Const(PtrInfo::Constant(GcRef(0x2000)));

        assert!(same_const.same_info(&live));
        assert!(live.same_info(&same_const));
        assert!(!different_const.same_info(&live));
    }

    /// `getintbound_handle` lazy-installs `IntBound::unbounded` on first
    /// access (mirroring `optimizer.py:110-112 op.set_forwarded(IntBound())`)
    /// and subsequent calls return the same `Live` cell — `Rc::ptr_eq`
    /// holds and mutation through one handle propagates to the other.
    #[test]
    fn int_bound_handle_live_identity_propagates_mutation() {
        use majit_ir::Type;

        let mut ctx = OptContext::with_num_inputs(0, 0);
        let (b, _op) = bound_resop_box(Type::Int, 0);
        let h1 = ctx.getintbound_handle(&Operand::from_boxref(&b));
        let h2 = ctx.getintbound_handle(&Operand::from_boxref(&b));
        assert!(
            h1.ptr_eq(&h2),
            "Live handles for the same box must share the same Rc cell"
        );

        // Mutation through h1 visible through h2 — RPython
        // `getintbound(box).intersect(b)` propagation.
        {
            let mut m = h1.borrow_mut().expect("Live handle accepts mutable borrow");
            let _ = m.make_ge_const(42);
        }
        assert_eq!(
            h2.borrow().lower,
            42,
            "h2 must observe h1's make_ge mutation (shared Rc cell)"
        );
    }

    /// `IntBoundHandle::Const` arms (synthesized from `ConstInt`) are
    /// independent objects — fresh `IntBound::from_constant(v)` each
    /// time, never `Rc::ptr_eq`.
    #[test]
    fn int_bound_handle_const_arms_are_not_ptr_eq() {
        use majit_ir::Value;

        let mut ctx = OptContext::with_num_inputs(0, 0);
        let c = Operand::const_from_value(Value::Int(7));
        let h1 = ctx.getintbound_handle(&c);
        let h2 = ctx.getintbound_handle(&c);
        assert!(
            !h1.ptr_eq(&h2),
            "Const arms must be fresh independent objects, never ptr_eq"
        );
        assert_eq!(h1.borrow().lower, 7);
        assert_eq!(h2.borrow().lower, 7);
    }

    /// `optimizer.py:102-103 return IntBound.from_constant(...)` —
    /// PyPy yields a *mutable* fresh `IntBound` for ConstInt. The Rust
    /// Const arm mirrors that: `borrow_mut()` succeeds and the mutation
    /// is observable through the same handle (and any clones of it),
    /// while a SEPARATE `getintbound_handle` call on the same ConstInt
    /// produces an independent cell that does NOT see the mutation.
    #[test]
    fn int_bound_handle_const_arm_is_locally_mutable() {
        use majit_ir::Value;

        let mut ctx = OptContext::with_num_inputs(0, 0);
        let c = Operand::const_from_value(Value::Int(7));
        let h = ctx.getintbound_handle(&c);
        // Direct field mutation through the RefMut — `make_ge_const`
        // would reject 20 on `from_constant(7)` (empty interval); the
        // parity claim is "borrow_mut succeeds and writes land in the
        // cell", not "any arbitrary IntBound method succeeds".
        {
            let mut m = h
                .borrow_mut()
                .expect("Const arm must accept mutable borrow (optimizer.py:102)");
            m.upper = 20;
        }
        assert_eq!(
            h.borrow().upper,
            20,
            "Const arm mutation must be visible through the same handle"
        );
        // A fresh getintbound_handle call mints an independent cell —
        // mutations on `h` do not leak across calls (PyPy: each
        // `IntBound.from_constant(7)` is a distinct object).
        let h_fresh = ctx.getintbound_handle(&c);
        assert_eq!(
            h_fresh.borrow().upper,
            7,
            "Fresh Const handle must not observe prior handle's mutation"
        );
    }

    /// `resoperation.py:57-68 get_box_replacement` walks the
    /// `_forwarded` chain until it hits a terminal that is not a Box
    /// forward.  After two consecutive `make_equal_to(a, b)` /
    /// `make_equal_to(b, c)` calls, reading `getptrinfo_handle(&a)`
    /// must return a handle to the same Rc cell that
    /// `getptrinfo_handle(&c)` returns — the chain walker resolves
    /// `a → b → c` and the PtrInfo installed earliest on `a` has
    /// transferred through both steps via the OpInfo clone.
    #[test]
    fn chain_walk_preserves_ptr_info_rc_identity_across_two_hops() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 3, 0, 3);
        let (a, _ia_a) = bound_inputarg_box(Type::Ref, 0);
        let (b, _ia_b) = bound_inputarg_box(Type::Ref, 1);
        let (c, _ia_c) = bound_inputarg_box(Type::Ref, 2);
        ctx.seed_boxes_canonical(&[a.clone(), b.clone(), c.clone()]);
        ctx.set_ptr_info(
            &Operand::from_boxref(&a),
            PtrInfo::NonNull { last_guard_pos: 7 },
        );

        ctx.make_equal_to(&Operand::from_boxref(&a), &Operand::from_boxref(&b));
        ctx.make_equal_to(&Operand::from_boxref(&b), &Operand::from_boxref(&c));

        let h_a = ctx
            .getptrinfo_handle(&Operand::from_boxref(&a))
            .expect("chain a -> b -> c must surface c's _forwarded slot");
        let h_c = ctx
            .getptrinfo_handle(&Operand::from_boxref(&c))
            .expect("c carries the transferred PtrInfo");
        assert!(
            h_a.same_info(&h_c),
            "chain walker must land on the same Rc cell that lives on c"
        );

        // Verify the original last_guard_pos survived both transfers.
        assert_eq!(h_c.borrow().get_last_guard_pos(), Some(7));
    }

    /// `optimizer.py:387 make_equal_to` transfers the `_forwarded`
    /// IntBound from `op` to `newop` by writing the same Python object
    /// into `newop.set_forwarded(...)`.  Counterpart of
    /// [`replace_op_preserves_ptr_info_rc_identity`] for the IntBound
    /// cell.
    #[test]
    fn replace_op_preserves_int_bound_rc_identity() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 2, 0, 2);
        let (old_box, _ia_old) = bound_inputarg_box(Type::Int, 0);
        let (new_box, _ia_new) = bound_inputarg_box(Type::Int, 1);
        ctx.seed_boxes_canonical(&[old_box.clone(), new_box.clone()]);
        ctx.setintbound(
            &Operand::from_boxref(&old_box),
            &crate::optimizeopt::intutils::IntBound::unbounded(),
        );

        let old_handle = ctx.getintbound_handle(&Operand::from_boxref(&old_box));
        assert!(matches!(old_handle, IntBoundHandle::Live(_)));

        ctx.make_equal_to(
            &Operand::from_boxref(&old_box),
            &Operand::from_boxref(&new_box),
        );
        let new_handle = ctx.getintbound_handle(&Operand::from_boxref(&new_box));
        assert!(
            old_handle.ptr_eq(&new_handle),
            "make_equal_to must transfer the same Rc cell for IntBound"
        );

        // Mutation through new_handle visible through old_handle.
        let _ = new_handle.borrow_mut().unwrap().make_ge_const(99);
        assert_eq!(
            old_handle.borrow().lower,
            99,
            "old must observe new's make_ge mutation (shared Rc)"
        );
    }

    /// `optimizer.py:387 make_equal_to` transfers the `_forwarded`
    /// PtrInfo from `op` to `newop` by writing the same Python object
    /// into `newop.set_forwarded(...)`.  pyre's `make_equal_to` clones
    /// the `OpInfo` enum, but since `OpInfo::Ptr` holds an `Rc`, the
    /// clone shares the same cell — so after `make_equal_to(old, new)`
    /// the handles obtained from `old` and `new` satisfy `ptr_eq`
    /// and downstream mutation on one is visible through the other.
    #[test]
    fn replace_op_preserves_ptr_info_rc_identity() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 2, 0, 2);
        let (old_box, _ia_old) = bound_inputarg_box(Type::Ref, 0);
        let (new_box, _ia_new) = bound_inputarg_box(Type::Ref, 1);
        ctx.seed_boxes_canonical(&[old_box.clone(), new_box.clone()]);
        ctx.set_ptr_info(
            &Operand::from_boxref(&old_box),
            PtrInfo::NonNull { last_guard_pos: 0 },
        );

        let old_handle = ctx
            .getptrinfo_handle(&Operand::from_boxref(&old_box))
            .expect("install populated _forwarded on old");
        ctx.make_equal_to(
            &Operand::from_boxref(&old_box),
            &Operand::from_boxref(&new_box),
        );
        let new_handle = ctx
            .getptrinfo_handle(&Operand::from_boxref(&new_box))
            .expect("PtrInfo transferred to new via clone of Rc cell");
        assert!(
            old_handle.same_info(&new_handle),
            "make_equal_to must transfer the same Rc cell (RPython _forwarded share)"
        );

        // Mutation through new_handle visible through old_handle —
        // they share the same Rc<RefCell<PtrInfo>>.
        new_handle
            .borrow_mut()
            .expect("Live handle accepts mutation")
            .set_last_guard_pos(123);
        assert_eq!(
            old_handle.borrow().get_last_guard_pos(),
            Some(123),
            "old's view of the transferred info must see new's mutation"
        );
    }

    /// `make_equal_to` (`optimizer.py:390-401`) shares the same
    /// `Rc<RefCell<IntBound>>` identity across `old` → `new` forwarding.
    #[test]
    fn make_equal_to_preserves_int_bound_rc_identity() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 2, 0, 2);
        let (old_box, _ia_old) = bound_inputarg_box(Type::Int, 0);
        let (new_box, _ia_new) = bound_inputarg_box(Type::Int, 1);
        ctx.seed_boxes_canonical(&[old_box.clone(), new_box.clone()]);
        ctx.setintbound(
            &Operand::from_boxref(&old_box),
            &crate::optimizeopt::intutils::IntBound::unbounded(),
        );

        let old_handle = ctx.getintbound_handle(&Operand::from_boxref(&old_box));
        assert!(matches!(old_handle, IntBoundHandle::Live(_)));

        ctx.make_equal_to(
            &Operand::from_boxref(&old_box),
            &Operand::from_boxref(&new_box),
        );
        let new_handle = ctx.getintbound_handle(&Operand::from_boxref(&new_box));
        assert!(
            old_handle.ptr_eq(&new_handle),
            "make_equal_to must transfer the same Rc cell for IntBound"
        );
    }

    /// `make_equal_to` transfers the `PtrInfo` `Rc` cell from `old` to
    /// `new` per `optimizer.py:400`.
    #[test]
    fn make_equal_to_preserves_ptr_info_rc_identity() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 2, 0, 2);
        let (old_box, _ia_old) = bound_inputarg_box(Type::Ref, 0);
        let (new_box, _ia_new) = bound_inputarg_box(Type::Ref, 1);
        ctx.seed_boxes_canonical(&[old_box.clone(), new_box.clone()]);
        ctx.set_ptr_info(
            &Operand::from_boxref(&old_box),
            PtrInfo::NonNull { last_guard_pos: 0 },
        );

        let old_handle = ctx
            .getptrinfo_handle(&Operand::from_boxref(&old_box))
            .expect("populated _forwarded on old");
        ctx.make_equal_to(
            &Operand::from_boxref(&old_box),
            &Operand::from_boxref(&new_box),
        );
        let new_handle = ctx
            .getptrinfo_handle(&Operand::from_boxref(&new_box))
            .expect("PtrInfo transferred to new via clone of Rc cell");
        assert!(
            old_handle.same_info(&new_handle),
            "make_equal_to must transfer the same Rc cell"
        );
    }

    /// `box.clear_forwarded()` resets `_forwarded` directly.  After the
    /// call, `old`'s slot is `None` and any previously-stored IntBound is
    /// unreachable.
    #[test]
    fn clear_forwarded_drops_int_bound() {
        let mut ctx = OptContext::with_num_inputs_and_start_pos(0, 1, 0, 1);
        let (old_box, ia) = bound_inputarg_box(Type::Int, 0);
        ctx.seed_boxes_canonical(&[old_box.clone()]);
        ctx.inputarg_refs = vec![ia];
        ctx.setintbound(
            &Operand::from_boxref(&old_box),
            &crate::optimizeopt::intutils::IntBound::unbounded(),
        );
        assert!(matches!(
            ctx.getintbound_handle(&Operand::from_boxref(&old_box)),
            IntBoundHandle::Live(_),
        ));

        old_box.clear_forwarded();
        assert!(matches!(
            &old_box.get_forwarded(),
            majit_ir::box_ref::Forwarded::None,
        ));
    }

    /// For an emitted ResOp the producing `Op.type_`,
    /// the minted OpRef variant tag, and `opref_type` all derive from the
    /// single source `opcode.result_type()` (resoperation.py:1693
    /// `res.type = result_type` in `create_class_for_op`). This regresses
    /// if a producer mints a variant tag disagreeing with `op.type_`, or if
    /// `opref_type` stops reading the variant tag / producing op — i.e. if
    /// any retired type side-table (`value_types` / `prev_phase_value_types`
    /// / `renamed_inputarg_types`) were reintroduced as a competing source.
    #[test]
    fn slice_0_8_emitted_op_type_is_single_source_of_truth() {
        use majit_ir::{Op, OpCode};
        let mut ctx = OptContext::with_num_inputs(8, 0);
        let cases: &[(OpCode, Type)] = &[
            (OpCode::SameAsI, Type::Int),
            (OpCode::SameAsR, Type::Ref),
            (OpCode::SameAsF, Type::Float),
        ];
        for &(opcode, ty) in cases {
            let op = Op::new(opcode, &[]);
            // `Op::new` seeds `type_` from `opcode.result_type()`.
            assert_eq!(op.type_, ty, "Op.type_ must equal opcode.result_type()");
            let pos: OpRef = ctx.emit(op);
            // `emit` reserves a typed pos, so the variant tag encodes the type.
            assert_eq!(
                pos.ty(),
                Some(ty),
                "minted OpRef variant tag must encode the op result type"
            );
            // The unified reader agrees, proving it reads the variant tag /
            // producing op.type_ and not any retired side-table.
            assert_eq!(
                ctx.opref_type(pos),
                Some(ty),
                "opref_type must agree with the variant tag / op.type_"
            );
            // The producing op recovered by position carries the same type_.
            let producer = ctx.op_at(pos).expect("emitted op must be findable by pos");
            assert_eq!(
                producer.type_, ty,
                "producing op.type_ must match the variant tag / opref_type"
            );
        }
    }
}

#[cfg(test)]
mod constant_ptr_info_tests {
    //! info.py:706-758 + 865-894 ConstPtrInfo / getptrinfo / getrawptrinfo
    //! parity tests for the typed-Int constant override path. RPython
    //! treats `ConstInt` (raw pointer) and `ConstPtr` uniformly via
    //! `_const.getref_base()`; majit must do the same regardless of how
    //! the constant pool stored the bits (`Value::Ref` vs `Value::Int`
    //! with a `Type::Ref` override).
    use super::*;
    use crate::optimizeopt::info::{PtrInfo, RawBufferPtrInfo, RawSlicePtrInfo, VStringVariant};
    use majit_ir::{GcRef, OpRef, Type, Value};
    use std::borrow::Cow;

    /// info.py:880-894 getptrinfo(ConstPtr) → ConstPtrInfo(op).
    /// A `Value::Ref` constant must be wrapped in `PtrInfo::Constant`.
    #[test]
    fn getptrinfo_returns_constant_for_value_ref() {
        let mut ctx = OptContext::new(0);
        let opref = OpRef::ref_op(10_000);
        let b = ctx.materialize_operand_at(opref);
        ctx.seed_constant(&b, Value::Ref(GcRef(0xdead_beef)));
        match ctx.getptrinfo(&b) {
            Some(PtrInfo::Constant(g)) => assert_eq!(g.0, 0xdead_beef),
            other => panic!("expected ConstPtrInfo(0xdeadbeef), got {other:?}"),
        }
    }

    /// info.py:870-871 getrawptrinfo(ConstInt) → ConstPtrInfo(op).
    /// Every ConstInt reaching `getrawptrinfo` is treated as a raw
    /// pointer (the caller has selected the helper because the
    /// `'i'`-typed box is intended as a pointer). The wrapped GcRef
    /// carries the int bits.
    #[test]
    fn getptrinfo_wraps_int_constant_as_const_ptr_info() {
        let mut ctx = OptContext::new(0);
        let opref = OpRef::int_op(10_002);
        let b = ctx.materialize_operand_at(opref);
        ctx.seed_constant(&b, Value::Int(42));
        match ctx.getptrinfo(&b) {
            Some(PtrInfo::Constant(g)) => assert_eq!(g.0, 42),
            other => panic!("expected ConstPtrInfo(42), got {other:?}"),
        }
    }

    /// info.py:718-726 ConstPtrInfo._get_info(descr, optheap) parity:
    /// the same constant must always resolve to the same shared
    /// `const_infos[ref]` slot. Calling `get_const_info_mut` twice on a
    /// `Value::Ref` constant returns identical info — and a mutation
    /// observed via the second call confirms the slot identity.
    #[test]
    fn const_info_mut_returns_same_slot_for_value_ref() {
        let mut ctx = OptContext::new(0);
        let opref = OpRef::ref_op(10_004);
        let seed_box = ctx.materialize_operand_at(opref);
        ctx.seed_constant(&seed_box, Value::Ref(GcRef(0xa5a5_a5a5)));
        // First lookup: install Instance via the Vacant entry path,
        // then mark a known class so the second lookup observes it.
        {
            let info = ctx
                .get_const_info_mut(opref, None)
                .expect("Ref constant should have const_infos slot");
            *info = PtrInfo::known_class(0x1111_2222, true);
        }
        // Second lookup: the slot must contain the previously written
        // PtrInfo, not a freshly minted Instance.
        let info = ctx
            .get_const_info_mut(opref, None)
            .expect("Ref constant should still have const_infos slot");
        match info {
            PtrInfo::Instance(iinfo) => {
                assert_eq!(iinfo.known_class, Some(0x1111_2222));
            }
            other => panic!("expected Instance(known_class=Some) after re-lookup, got {other:?}"),
        }
    }

    /// info.py:719-720 `if not ref: raise InvalidLoop` — null protection.
    /// `get_const_info_mut` records a deferred `InvalidLoop` signal and
    /// returns `None` when the constant pointer resolves to a null `gcref`.
    /// The driver aborts the impossible trace shape at the next barrier so
    /// the JIT can retry; the Rust port mirrors that contract without
    /// unwinding (so it works under `panic=abort`).
    #[test]
    fn const_info_mut_signals_invalid_loop_on_null_value_ref_constant() {
        let mut ctx = OptContext::new(0);
        let ref_null = OpRef::ref_op(10_007);
        let seed_box = ctx.materialize_operand_at(ref_null);
        ctx.seed_constant(&seed_box, Value::Ref(GcRef(0)));
        let info = ctx.get_const_info_mut(ref_null, None);
        assert!(info.is_none(), "null constant base must not yield ptr info");
        let invalid = ctx
            .take_invalid_loop()
            .expect("expected pending InvalidLoop signal");
        assert!(invalid.0.contains("null constant base pointer"));
    }

    /// `Value::Int(0)` reaches `getrawptrinfo` as `ConstPtrInfo(NULL)`
    /// per `info.py:870-871`, then trips the null-constant InvalidLoop
    /// protection at `get_const_info_mut`. Mirrors the `Value::Ref(0)`
    /// case — null-pointer protection is uniform regardless of the
    /// underlying constant tag.
    #[test]
    fn const_info_mut_signals_invalid_loop_on_null_int_constant() {
        let mut ctx = OptContext::new(0);
        let opref = OpRef::int_op(10_010);
        let seed_box = ctx.materialize_operand_at(opref);
        ctx.seed_constant(&seed_box, Value::Int(0));
        let info = ctx.get_const_info_mut(opref, None);
        assert!(info.is_none(), "null constant base must not yield ptr info");
        let invalid = ctx
            .take_invalid_loop()
            .expect("expected pending InvalidLoop signal");
        assert!(invalid.0.contains("null constant base pointer"));
    }

    /// optimizer.py:154-158 `is_raw_ptr(op)` parity for
    /// `info.RawSlicePtrInfo`: once a raw slice PtrInfo is present, it
    /// must be classified as an `AbstractRawPtrInfo` exactly like its
    /// parent raw buffer.
    #[test]
    fn is_raw_ptr_accepts_virtual_raw_slice() {
        let mut ctx = OptContext::new(0);
        let parent = OpRef::ref_op(10_010);
        let slice = OpRef::ref_op(10_011);

        let parent_box = ctx.materialize_operand_at(parent);
        let slice_box = ctx.materialize_operand_at(slice);
        ctx.set_ptr_info(
            &parent_box,
            PtrInfo::VirtualRawBuffer(RawBufferPtrInfo::new(0, 32, None)),
        );
        ctx.set_ptr_info(
            &slice_box,
            PtrInfo::VirtualRawSlice(RawSlicePtrInfo {
                offset: 8,
                parent: parent_box,
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            }),
        );

        let parent_box = ctx.get_box_replacement_operand(parent);
        let slice_box = ctx.get_box_replacement_operand(slice);
        assert!(ctx.is_raw_ptr(&parent_box));
        assert!(ctx.is_raw_ptr(&slice_box));
    }

    /// vstring.py:50 `StrPtrInfo.__init__(mode, is_virtual=False, length=-1)`
    /// parity for non-virtual strings: `make_nonnull_str()` must install
    /// a base `StrPtrInfo`, not one of the virtual subclasses.
    #[test]
    fn make_nonnull_str_initializes_ptr_variant() {
        let mut ctx = OptContext::new(0);
        let opref = OpRef::ref_op(10_012);
        // Synthetic-OpRef test fixture: lazy-allocate the BoxRef so the
        // BoxRef-direct `make_nonnull_str` can write through it. Production
        // callers obtain the box via `get_box_replacement_box`.
        let op_box = ctx.materialize_operand_at(opref);

        ctx.make_nonnull_str(&op_box, 0);

        match ctx.peek_ptr_info(&op_box) {
            Some(PtrInfo::Str(sinfo)) => {
                assert_eq!(sinfo.mode, 0);
                assert_eq!(sinfo.length, -1);
                assert!(sinfo.lenbound.is_none());
                assert!(matches!(sinfo.variant, VStringVariant::Ptr));
            }
            other => panic!("expected base StrPtrInfo, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod ensure_ptr_info_arg0_tests {
    //! optimizer.py:461-499 `ensure_ptr_info_arg0` parity tests.
    //!
    //! Each test mirrors a single PyPy branch in `ensure_ptr_info_arg0`:
    //! the constant arg0 path, the AbstractVirtualPtrInfo early-return path,
    //! the NonNullPtrInfo upgrade path, and the assertion that fires on
    //! unexpected forwarded info shapes.
    use super::*;
    use crate::optimizeopt::info::{ArrayPtrInfo, EnsuredPtrInfo, PtrInfo};
    use crate::optimizeopt::intutils::IntBound;
    use majit_ir::{Descr, DescrRef, GcRef, Op, OpCode, OpRc, OpRef, SizeDescr, Type, Value};
    use std::sync::Arc;

    #[derive(Debug)]
    struct TestSizeDescr {
        index: u32,
        is_object: bool,
    }

    impl Descr for TestSizeDescr {
        fn index(&self) -> u32 {
            self.index
        }
        fn as_size_descr(&self) -> Option<&dyn SizeDescr> {
            Some(self)
        }
    }

    impl SizeDescr for TestSizeDescr {
        fn size(&self) -> usize {
            64
        }
        fn type_id(&self) -> u32 {
            self.index
        }
        fn is_immutable(&self) -> bool {
            false
        }
        fn is_object(&self) -> bool {
            self.is_object
        }
    }

    fn struct_parent_descr() -> DescrRef {
        Arc::new(TestSizeDescr {
            index: 0xFFFF_0000,
            is_object: false,
        })
    }

    fn instance_parent_descr() -> DescrRef {
        Arc::new(TestSizeDescr {
            index: 0xFFFF_0001,
            is_object: true,
        })
    }

    #[derive(Debug)]
    struct TestFieldDescr {
        index: u32,
        parent: DescrRef,
    }

    impl Descr for TestFieldDescr {
        fn index(&self) -> u32 {
            self.index
        }
        fn as_field_descr(&self) -> Option<&dyn majit_ir::FieldDescr> {
            Some(self)
        }
    }

    impl majit_ir::FieldDescr for TestFieldDescr {
        fn offset(&self) -> usize {
            0
        }
        fn field_size(&self) -> usize {
            8
        }
        fn field_type(&self) -> majit_ir::Type {
            majit_ir::Type::Int
        }
        fn index_in_parent(&self) -> usize {
            0
        }
        fn get_parent_descr(&self) -> Option<DescrRef> {
            Some(self.parent.clone())
        }
    }

    fn field_op_with_parent(parent: DescrRef) -> Op {
        // history.py:182 GetfieldGc receiver is a Ref box; arg0 must
        // carry the Ref variant tag (resoperation.py:615 RefOp).
        let descr: DescrRef = Arc::new(TestFieldDescr { index: 0, parent });
        let mut op = Op::with_descr(
            OpCode::GetfieldGcI,
            &[crate::history::test_support::rooted_inputarg_operand(
                Type::Ref,
                0,
            )],
            descr,
        );
        op.pos.set(OpRef::int_op(1));
        op
    }

    fn array_op() -> Op {
        // ArraylenGc receiver is a Ref box.
        let descr: DescrRef = Arc::new(TestSizeDescr {
            index: 7,
            is_object: false,
        });
        let mut op = Op::with_descr(
            OpCode::ArraylenGc,
            &[crate::history::test_support::rooted_inputarg_operand(
                Type::Ref,
                0,
            )],
            descr,
        );
        op.pos.set(OpRef::int_op(1));
        op
    }

    /// optimizer.py:465-466: `if arg0.is_constant(): return info.ConstPtrInfo(arg0)`
    /// Constant `Value::Ref` arg0 → `EnsuredPtrInfo::Constant(gcref)`.
    #[test]
    fn ensure_ptr_info_arg0_returns_constant_for_value_ref() {
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        let seed_box = ctx.materialize_operand_at(OpRef::input_arg_ref(0));
        ctx.seed_constant(&seed_box, Value::Ref(GcRef(0xdead_beef)));
        let op = field_op_with_parent(struct_parent_descr());
        let info = ctx.ensure_ptr_info_arg0(&op);
        match info {
            EnsuredPtrInfo::Constant { gcref, .. } => assert_eq!(gcref.0, 0xdead_beef),
            _ => panic!("expected EnsuredPtrInfo::Constant"),
        }
    }

    /// optimizer.py:465-466 parity for plain `Value::Int` constants — PyPy
    /// returns `info.ConstPtrInfo(arg0)` regardless of the box's exact type.
    /// majit's port mirrors that by returning `Constant(GcRef(bits))`; null
    /// or unsafe pointers are filtered downstream by `_get_info`'s null
    /// protection (info.py:719-720).
    #[test]
    fn ensure_ptr_info_arg0_returns_constant_for_value_int() {
        // optimizer.py:465-466 PyPy parity: even Value::Int seeded at the
        // GetfieldGc receiver slot is interpreted as a ptr (ConstPtrInfo).
        // The Box class is still Ref because the receiver position is Ref;
        // the inner i64 value just happens to be tagged Int by the trace.
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        let seed_box = ctx.materialize_operand_at(OpRef::input_arg_ref(0));
        ctx.seed_constant(&seed_box, Value::Int(1));
        let op = field_op_with_parent(struct_parent_descr());
        let info = ctx.ensure_ptr_info_arg0(&op);
        assert!(matches!(info, EnsuredPtrInfo::Constant { .. }));
    }

    /// info.py:796-822 `ConstPtrInfo.getlenbound(mode_string)` returns
    /// `IntBound.from_constant(length)` when `getstrlen1(mode)` knows the
    /// exact length. The Rust port consults the `string_length_resolver`
    /// hook the host runtime registered on `OptContext`.
    #[test]
    fn ensure_ptr_info_arg0_constant_string_returns_exact_length_via_resolver() {
        use std::sync::Arc;
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        let seed_box = ctx.materialize_operand_at(OpRef::input_arg_ref(0));
        ctx.seed_constant(&seed_box, Value::Ref(GcRef(0xC0FE)));
        // Resolver pretends every constant has byte-string length 5 in
        // mode_string and unicode length 7 in mode_unicode.
        ctx.string_length_resolver = Some(Arc::new(|gcref: GcRef, mode: u8| {
            assert_eq!(gcref.0, 0xC0FE);
            match mode {
                0 => Some(5),
                1 => Some(7),
                _ => None,
            }
        }));
        let op = {
            let descr: DescrRef = Arc::new(TestSizeDescr {
                index: 1,
                is_object: false,
            });
            let mut op = Op::with_descr(
                OpCode::Strlen,
                &[crate::history::test_support::rooted_inputarg_operand(
                    Type::Ref,
                    0,
                )],
                descr,
            );
            op.pos.set(OpRef::int_op(1));
            op
        };
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        let bound = info
            .getlenbound(Some(0))
            .expect("constant string length should resolve");
        assert_eq!(bound.lower, 5);
        assert_eq!(bound.upper, 5);
        let bound = info
            .getlenbound(Some(1))
            .expect("constant unicode length should resolve");
        assert_eq!(bound.lower, 7);
        assert_eq!(bound.upper, 7);
    }

    /// info.py:799-801 `if length < 0: return IntBound.nonnegative()` —
    /// no resolver registered → conservative nonnegative fallback.
    #[test]
    fn ensure_ptr_info_arg0_constant_string_falls_back_to_nonnegative_without_resolver() {
        use std::sync::Arc;
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        let seed_box = ctx.materialize_operand_at(OpRef::input_arg_ref(0));
        ctx.seed_constant(&seed_box, Value::Ref(GcRef(0x1234)));
        let op = {
            let descr: DescrRef = Arc::new(TestSizeDescr {
                index: 1,
                is_object: false,
            });
            let mut op = Op::with_descr(
                OpCode::Strlen,
                &[crate::history::test_support::rooted_inputarg_operand(
                    Type::Ref,
                    0,
                )],
                descr,
            );
            op.pos.set(OpRef::int_op(1));
            op
        };
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        let bound = info
            .getlenbound(Some(0))
            .expect("nonnegative fallback should be Some");
        assert_eq!(bound.lower, IntBound::nonnegative().lower);
        assert!(!bound.is_constant());
    }

    /// optimizer.py:475-484 GETFIELD branch with `parent_descr.is_object() == false`
    /// → `info.StructPtrInfo(parent_descr)`. The Rust port returns the
    /// freshly-installed `PtrInfo::Struct` via `Forwarded(&mut PtrInfo)`.
    #[test]
    fn ensure_ptr_info_arg0_constructs_struct_for_non_object_field() {
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        // Keep a strong reference to the parent alive for the duration
        // of the test: `SimpleFieldDescr::parent_descr` is a
        // `Weak<DescrRef>` (breaks the cycle between SizeDescr.all_fielddescrs
        // and FieldDescr.parent_descr), so the test must hold the parent
        // Arc until `get_parent_descr()` has been called.
        let _parent = struct_parent_descr();
        let op = field_op_with_parent(_parent.clone());
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        let pi = info.as_mut().expect("Forwarded variant expected");
        assert!(matches!(&*pi, PtrInfo::Struct(_)));
    }

    /// optimizer.py:480-484 GETFIELD branch with `parent_descr.is_object() == true`
    /// → `info.InstancePtrInfo(parent_descr)`.
    #[test]
    fn ensure_ptr_info_arg0_constructs_instance_for_object_field() {
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        let _parent = instance_parent_descr();
        let op = field_op_with_parent(_parent.clone());
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        let pi = info.as_mut().expect("Forwarded variant expected");
        assert!(matches!(&*pi, PtrInfo::Instance(_)));
    }

    /// optimizer.py:485-487 ARRAYLEN_GC branch → `info.ArrayPtrInfo(descr)`.
    /// The PyPy primitive returns the same arrayinfo across calls so
    /// callers can read `arrayinfo.getlenbound(None)` directly. The Rust
    /// port mirrors that and the `getlenbound` call resolves to the
    /// pre-installed `nonnegative` lenbound on the freshly-built ArrayPtrInfo.
    #[test]
    fn ensure_ptr_info_arg0_arraylen_returns_array_with_nonnegative_lenbound() {
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        let op = array_op();
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        let bound = info
            .getlenbound(None)
            .expect("ArrayPtrInfo.getlenbound(None) should be Some");
        assert_eq!(bound.lower, IntBound::nonnegative().lower);
    }

    /// info.py:796-802 `ConstPtrInfo.getlenbound(mode)` returns
    /// `IntBound.nonnegative()` whenever `getstrlen1(mode)` produces a
    /// negative length. info.py:823-824 makes `mode is None` (no
    /// vstring mode) one of those cases via the `else: return -1`
    /// branch. The Rust port must therefore answer `Some(nonnegative())`
    /// — not `None` — for `Constant.getlenbound(None)` so the
    /// ARRAYLEN_GC postprocess on a constant array still propagates a
    /// non-negative bound.
    #[test]
    fn ensure_ptr_info_arg0_constant_arraylen_returns_nonnegative() {
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        let seed_box = ctx.materialize_operand_at(OpRef::input_arg_ref(0));
        ctx.seed_constant(&seed_box, Value::Ref(GcRef(0xfeed)));
        let op = array_op();
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        let bound = info
            .getlenbound(None)
            .expect("ConstPtrInfo.getlenbound(None) must mirror PyPy nonnegative fallback");
        assert_eq!(bound.lower, IntBound::nonnegative().lower);
        assert_eq!(bound.upper, IntBound::nonnegative().upper);
    }

    /// optimizer.py:467-469 `if isinstance(opinfo, AbstractVirtualPtrInfo):
    /// return opinfo` parity. A second call must return the SAME PtrInfo
    /// (verified by mutating via the first call and observing the mutation
    /// via the second). PyPy's structinfo identity is the test of record;
    /// the Rust port checks via state preserved across calls.
    #[test]
    fn ensure_ptr_info_arg0_returns_existing_array_unchanged() {
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        let op = array_op();
        // First call constructs the ArrayPtrInfo and tightens the lenbound
        // through the helper.
        {
            let mut info = ctx.ensure_ptr_info_arg0(&op);
            let mut handle = info.as_mut().expect("expected fresh ArrayPtrInfo");
            if let PtrInfo::Array(arr) = &mut *handle {
                let _ = arr.lenbound.make_gt_const(7);
            } else {
                panic!("expected fresh ArrayPtrInfo");
            }
        }
        // Second call returns the same ArrayPtrInfo (lenbound preserved).
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        let mut handle = info.as_mut().expect("second call must still return Array");
        match &mut *handle {
            PtrInfo::Array(ArrayPtrInfo { lenbound, .. }) => {
                assert!(
                    lenbound.lower >= 8,
                    "second call must return the previously-mutated ArrayPtrInfo (lower={})",
                    lenbound.lower
                );
            }
            _ => panic!("second call must still return Array"),
        }
    }

    /// optimizer.py:470-474 `elif opinfo is not None: ...; assert opinfo is
    /// None or opinfo.__class__ is info.NonNullPtrInfo`. A pre-existing
    /// NonNullPtrInfo flows through the upgrade path; its `last_guard_pos`
    /// is preserved on the freshly-installed PtrInfo.
    #[test]
    fn ensure_ptr_info_arg0_upgrades_nonnull_to_struct() {
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        // Pre-install a NonNullPtrInfo with a specific last_guard_pos.
        let pos0_box = ctx.materialize_operand_at(OpRef::input_arg_ref(0));
        ctx.set_ptr_info(&pos0_box, PtrInfo::NonNull { last_guard_pos: 7 });
        let _parent = struct_parent_descr();
        let op = field_op_with_parent(_parent.clone());
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        let mut handle = info.as_mut().expect("expected upgraded Struct, got None");
        match &mut *handle {
            pi @ PtrInfo::Struct(_) => {
                assert_eq!(pi.last_guard_pos(), Some(7));
            }
            other => panic!("expected upgraded Struct, got {other:?}"),
        }
    }

    /// optimizer.py:474 assertion: an unexpected forwarded info shape (e.g.
    /// an op-forwarding redirect that resolved to a non-PtrInfo state)
    /// must NOT silently overwrite. We seed an `Instance` PtrInfo, then
    /// hand it a field op with a different parent — the early-return path
    /// hits, and the existing Instance is returned without overwrite.
    #[test]
    fn ensure_ptr_info_arg0_does_not_overwrite_existing_instance() {
        let mut ctx = OptContext::with_inputarg_types(4, &[Type::Ref]);
        let pos0_box = ctx.materialize_operand_at(OpRef::input_arg_ref(0));
        ctx.set_ptr_info(
            &pos0_box,
            PtrInfo::instance(Some(instance_parent_descr()), Some(0xc0de)),
        );
        let op = field_op_with_parent(struct_parent_descr());
        let mut info = ctx.ensure_ptr_info_arg0(&op);
        let mut handle = info
            .as_mut()
            .expect("expected Instance preserved, got None");
        match &mut *handle {
            PtrInfo::Instance(_) => {} // unchanged
            other => panic!("expected Instance preserved, got {other:?}"),
        }
    }
}

#[cfg(test)]
mod rd_virtual_info_builder_tests {
    use super::*;
    use crate::walkvirtual::VirtualVisitor;
    use majit_ir::{Descr, DescrRef, FieldDescr, SizeDescr, Type};
    use std::sync::Arc;

    #[derive(Debug)]
    struct TestSizeDescr {
        index: u32,
        type_id: u32,
        is_object: bool,
    }

    impl Descr for TestSizeDescr {
        fn index(&self) -> u32 {
            self.index
        }

        fn as_size_descr(&self) -> Option<&dyn SizeDescr> {
            Some(self)
        }
    }

    impl SizeDescr for TestSizeDescr {
        fn size(&self) -> usize {
            16
        }

        fn type_id(&self) -> u32 {
            self.type_id
        }

        fn is_immutable(&self) -> bool {
            false
        }

        fn is_object(&self) -> bool {
            self.is_object
        }
    }

    #[derive(Debug)]
    struct TestFieldDescr {
        index: u32,
        offset: usize,
        field_size: usize,
        field_type: Type,
    }

    impl Descr for TestFieldDescr {
        fn index(&self) -> u32 {
            self.index
        }

        fn as_field_descr(&self) -> Option<&dyn FieldDescr> {
            Some(self)
        }
    }

    impl FieldDescr for TestFieldDescr {
        fn offset(&self) -> usize {
            self.offset
        }

        fn field_size(&self) -> usize {
            self.field_size
        }

        fn field_type(&self) -> Type {
            self.field_type
        }
    }

    #[test]
    fn visit_virtual_preserves_field_descr_indices() {
        let mut builder = RdVirtualInfoBuilder;
        let size_descr: DescrRef = Arc::new(TestSizeDescr {
            index: 0x3000_0001,
            type_id: 7,
            is_object: true,
        });
        let field0: DescrRef = Arc::new(TestFieldDescr {
            index: 0x1000_0123,
            offset: 16,
            field_size: 8,
            field_type: Type::Int,
        });
        let field1: DescrRef = Arc::new(TestFieldDescr {
            index: 0x1000_0456,
            offset: 24,
            field_size: 8,
            field_type: Type::Ref,
        });

        let Some(majit_ir::RdVirtualInfo::VirtualInfo { fielddescrs, .. }) =
            builder.visit_virtual(&size_descr, &[], &[field0.clone(), field1.clone()])
        else {
            panic!("expected VirtualInfo");
        };

        assert_eq!(fielddescrs[0].index, field0.index());
        assert_eq!(fielddescrs[1].index, field1.index());
    }
}

#[cfg(test)]
mod intbound_invariant_tests {
    use super::*;
    use crate::optimizeopt::intutils::IntBound;
    use majit_ir::{GcRef, OpRef, Value};

    #[test]
    #[should_panic]
    fn getintbound_rejects_non_int_boxes() {
        let mut ctx = OptContext::new(0);
        let opref = OpRef::ref_op(20_000);
        let seed_box = ctx.materialize_operand_at(opref);
        ctx.seed_constant(&seed_box, Value::Ref(GcRef(0xdead_beef)));
        let _ = {
            let __mb = ctx.materialize_operand_at(opref);
            ctx.getintbound_handle(&__mb).borrow().clone()
        };
    }

    #[test]
    #[should_panic]
    fn setintbound_rejects_non_int_boxes() {
        let ctx = OptContext::new(0);
        // BoxRef-direct setintbound asserts `op.type_()` is Int/Void per
        // optimizer.py:116. A Ref-typed BoxRef should trigger the panic.
        let ref_box = majit_ir::box_ref::BoxRef::new_inputarg(majit_ir::Type::Ref, 0);
        ctx.setintbound(&Operand::from_boxref(&ref_box), &IntBound::nonnegative());
    }
}

#[cfg(test)]
mod imported_short_preamble_fallback_tests {
    use super::*;
    use majit_ir::{Op, OpCode, OpRc, OpRef};

    #[test]
    fn force_op_from_preamble_replays_pop_without_builder_lookup() {
        // 2 Ref inputargs for the body label — typical loop-body shape.
        let mut ctx =
            OptContext::with_inputarg_types(16, &[majit_ir::Type::Ref, majit_ir::Type::Ref]);
        ctx.initialize_imported_short_preamble_builder(
            &[OpRef::input_arg_ref(0), OpRef::input_arg_ref(1)],
            &[
                majit_ir::box_ref::BoxRef::from_opref(OpRef::int_op(7)),
                majit_ir::box_ref::BoxRef::from_opref(OpRef::int_op(8)),
            ],
            &[],
        );

        let mut replay_op = Op::new(
            OpCode::IntAdd,
            &[
                crate::history::test_support::rooted_resop_operand(majit_ir::Type::Int, 7),
                crate::history::test_support::rooted_resop_operand(majit_ir::Type::Int, 8),
            ],
        );
        replay_op.pos.set(OpRef::int_op(14));
        // shortpreamble.py:120 non-invented PureOp.produce_op: `op = self.res`.
        // pop.op carries the body-visible OpRef directly (no forwarding chain
        // installed for non-invented Pure).
        let pop = crate::optimizeopt::info::PreambleOp {
            op: majit_ir::box_ref::BoxRef::from_opref(OpRef::int_op(41)),
            invented_name: false,
            preamble_op: std::rc::Rc::new(replay_op),
            same_as_source: None,
        };

        let forced = ctx.force_op_from_preamble_op(&pop);
        assert_eq!(forced, OpRef::int_op(41));

        let sp = ctx
            .build_imported_short_preamble()
            .expect("imported short preamble builder should exist");
        assert_eq!(sp.ops.len(), 1);
        assert_eq!(sp.ops[0].op.opcode, OpCode::IntAdd);
        assert_eq!(
            sp.ops[0]
                .op
                .getarglist()
                .iter()
                .map(|a| a.to_opref())
                .collect::<Vec<_>>(),
            vec![OpRef::int_op(7), OpRef::int_op(8)]
        );
        assert_eq!(sp.ops[0].op.pos.get(), OpRef::int_op(14));
    }
}

#[cfg(test)]
mod opt_box_env_tests {
    use super::*;
    use crate::optimizeopt::info::VirtualInfo;
    use majit_ir::{BoxEnv, DescrRef, GcRef, OpRef};
    use std::sync::Arc;

    #[derive(Debug)]
    struct DummySizeDescr;

    impl majit_ir::Descr for DummySizeDescr {
        fn index(&self) -> u32 {
            0
        }

        fn clone_descr(&self) -> Option<DescrRef> {
            Some(Arc::new(DummySizeDescr))
        }

        fn as_size_descr(&self) -> Option<&dyn majit_ir::SizeDescr> {
            Some(self)
        }
    }

    impl majit_ir::SizeDescr for DummySizeDescr {
        fn size(&self) -> usize {
            24
        }

        fn vtable(&self) -> usize {
            0x1234
        }

        fn type_id(&self) -> u32 {
            7
        }

        fn is_immutable(&self) -> bool {
            false
        }
    }

    #[test]
    fn opt_box_env_is_virtual_ref_follows_box_replacement() {
        let mut ctx = OptContext::with_num_inputs(16, 0);
        let source = OpRef::ref_op(12);
        let target = OpRef::ref_op(21);
        let source_box = ctx.materialize_operand_at(source);
        let target_box = ctx.materialize_operand_at(target);
        ctx.make_equal_to(&source_box, &target_box);
        ctx.set_ptr_info(
            &target_box,
            PtrInfo::Virtual(VirtualInfo {
                descr: Arc::new(DummySizeDescr),
                known_class: Some(0x1234),
                ob_type_descr: None,
                fields: Vec::new(),
                last_guard_pos: -1,
                avpi: crate::optimizeopt::info::AbstractVirtualPtrInfo::new(),
            }),
        );

        let env = OptBoxEnv { ctx: &ctx };
        assert!(
            env.is_virtual_ref(source),
            "forwarded snapshot boxes must classify as virtual via replacement"
        );
    }

    #[test]
    fn materialize_box_at_lazy_materialises_inputarg_for_empty_inputarg_slot() {
        // `resoperation.py:699 AbstractInputArg` and
        // `resoperation.py:250 AbstractResOp` are distinct classes
        // upstream — a Box materialised against `OpRef::InputArg*`
        // must be `is_inputarg()` so the chain walker reconstructs
        // the same variant on the round-trip through the
        // forwarding chain.  Prior to the per-variant empty-slot
        // path, materialisation always emitted `new_resop`,
        // silently demoting the inputarg.
        // `with_inputarg_types` would seed the inputarg slots
        // eagerly, defeating the lazy-materialisation check.  Use
        // `with_num_inputs(_, 0)` to get an empty pool then hand an
        // `OpRef::InputArg*` in directly — the regression we are
        // covering is exactly the path where the empty slot must
        // mint a `new_inputarg`.
        let mut ctx = OptContext::with_num_inputs(8, 0);
        let arg = OpRef::input_arg_typed(0, majit_ir::Type::Int);
        let materialised = ctx.materialize_box_at(arg);
        assert!(
            materialised.is_inputarg(),
            "empty InputArg* slot lazy-materialised the wrong BoxKind",
        );
        assert_eq!(materialised.position(), Some(0));
        assert_eq!(materialised.type_(), majit_ir::Type::Int);

        // Re-entering must resolve to the same canonical `_forwarded`
        // host (`resoperation.py:700 AbstractInputArg._forwarded`). The
        // `BoxRef` wrapper carries no state, so two
        // `materialize_box_at` calls return distinct wrappers bound to the
        // same `InputArgRc`; identity lives on that bound host, not the
        // wrapper `Rc`.
        let second = ctx.materialize_box_at(arg);
        assert!(
            std::rc::Rc::ptr_eq(
                &materialised
                    .bound_inputarg()
                    .expect("materialised bound to InputArg"),
                &second.bound_inputarg().expect("second bound to InputArg"),
            ),
            "second materialize_box_at must resolve to the same InputArg host",
        );
    }

    #[test]
    fn materialize_box_at_lazy_materialises_resop_for_empty_resop_slot() {
        // Companion to the InputArg case — `OpRef::int_op(_)` must
        // continue to produce a `new_resop` Box so `is_resop()` holds.
        let mut ctx = OptContext::with_num_inputs(8, 0);
        let result = OpRef::int_op(3);
        let materialised = ctx.materialize_box_at(result);
        assert!(
            materialised.is_resop(),
            "empty ResOp slot lazy-materialised the wrong BoxKind",
        );
    }
}
