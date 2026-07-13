//! Resume data: encodes the mapping from guard fail_args to full interpreter state.
//!
//! When a guard fails, the JIT needs to reconstruct the interpreter's full state
//! (program counter, local variables, stack contents) from the values stored in
//! the DeadFrame. Resume data provides this mapping.
//!
//! This is the RPython equivalent of `rpython/jit/metainterp/resume.py`.

use indexmap::{IndexMap, IndexSet};
use rustc_hash::FxBuildHasher;
use std::cell::UnsafeCell;
use std::sync::Arc;

use majit_backend::{
    ExitFrameLayout, ExitPendingFieldLayout, ExitRecoveryLayout, ExitVirtualLayout,
};
use majit_ir::{Const, GcRef, OpRef, Type};

pub type LiveboxTypeMap = indexmap::IndexMap<majit_ir::OpRef, majit_ir::Type, FxBuildHasher>;

/// resume.py:656-670: element kind from arraydescr.
/// 0=ref (is_array_of_pointers), 1=int, 2=float (is_array_of_floats).
fn array_kind_from_descr(arraydescr: Option<&majit_ir::DescrRef>) -> u8 {
    arraydescr
        .and_then(|d| d.as_array_descr())
        .map(|ad| {
            if ad.is_array_of_pointers() {
                0u8
            } else if ad.is_array_of_floats() {
                2u8
            } else {
                1u8
            }
        })
        .unwrap_or(0)
}

// ═══════════════════════════════════════════════════════════════
// RPython resume.py:96-139 — structural port (i16 tags).
// ═══════════════════════════════════════════════════════════════

// resume.py:96-97
#[derive(Debug)]
pub struct TagOverflow;

// resume.py:99-104
pub fn tag(value: i32, tagbits: u8) -> Result<i16, TagOverflow> {
    debug_assert!(tagbits <= 3);
    let sx = value >> 13;
    if sx != 0 && sx != -1 {
        return Err(TagOverflow);
    }
    Ok(((value << 2) | tagbits as i32) as i16)
}

// resume.py:106-109
pub fn untag(value: i16) -> (i32, u8) {
    let widened = value as i32;
    let tagbits = (widened & TAGMASK as i32) as u8;
    (widened >> 2, tagbits)
}

// resume.py:111-113
#[inline]
pub fn tagged_eq(x: i16, y: i16) -> bool {
    (x as i32) == (y as i32)
}

// resume.py:115-121
pub fn tagged_list_eq(tl1: &[i16], tl2: &[i16]) -> bool {
    if tl1.len() != tl2.len() {
        return false;
    }
    tl1.iter().zip(tl2.iter()).all(|(&a, &b)| tagged_eq(a, b))
}

// resume.py:123-132
pub const TAGCONST: u8 = 0;
pub const TAGINT: u8 = 1;
pub const TAGBOX: u8 = 2;
pub const TAGVIRTUAL: u8 = 3;
const TAGMASK: u8 = 3;

pub const UNASSIGNED: i16 = ((-1i32 << 13) << 2 | TAGBOX as i32) as i16;
pub const UNASSIGNEDVIRTUAL: i16 = ((-1i32 << 13) << 2 | TAGVIRTUAL as i32) as i16;
pub const NULLREF: i16 = ((-1i32 << 2) | TAGCONST as i32) as i16;
pub const UNINITIALIZED_TAG: i16 = ((-2i32 << 2) | TAGCONST as i32) as i16;
pub const TAG_CONST_OFFSET: i32 = 0;

/// Ordered livebox map: canonical box (`Rc::ptr_eq`) → i16 tag.
///
/// resume.py:137/370: RPython uses `dict` keyed by the actual Box object
/// (object `is` identity). In Python 3 that dict is insertion-ordered, and
/// `_number_virtuals` iterates it directly. #160/S11 keys this map by the
/// canonical [`Operand`](majit_ir::operand::Operand) (`Rc::ptr_eq` on the
/// producer = PyPy `box is box`), the faithful port of the dict-by-`is` —
/// `Operand` IS the box object `resume.py liveboxes` stores; the no-HashMap
/// house rule keeps the `IndexMap` backing (linear scan, dict-assignment
/// semantics, preserved insertion order). Two reaches of one logical box
/// resolve to one producer Rc (via `from_bound_op`/`from_bound_inputarg`) and
/// collapse to one key; distinct boxes — e.g. an `InputArg` vs a `ResOp`
/// result — stay distinct, where a raw-position key could have aliased them.
///
/// Invariant: keys are never Const boxes. Per `resume.py:204-205`
/// `_number_boxes`, `isinstance(box, Const)` short-circuits to
/// `self.getconst(box)` and the result is never written into
/// `numb_state.liveboxes`. Only the `else` branch (line 207-223,
/// non-Const Box) reaches `liveboxes[box] = tagged`. `insert` enforces
/// this via `debug_assert!` so a const-keyed insertion fails loudly in
/// debug builds rather than silently producing an out-of-RPython-shape
/// numbering state.
pub struct LiveboxMap {
    entries: indexmap::IndexMap<majit_ir::operand::Operand, i16>,
}

impl LiveboxMap {
    pub fn new() -> Self {
        Self {
            entries: indexmap::IndexMap::new(),
        }
    }

    #[inline(always)]
    pub fn get(&self, b: &majit_ir::operand::Operand) -> Option<i16> {
        self.entries.get(b).copied()
    }

    #[inline(always)]
    pub fn insert(&mut self, b: majit_ir::operand::Operand, value: i16) {
        debug_assert!(
            b.const_value().is_none(),
            "LiveboxMap::insert: Const box {b:?} violates resume.py:204-223 \
             `_number_boxes` invariant — `isinstance(box, Const)` is encoded \
             via `getconst(box)` and never enters numb_state.liveboxes",
        );
        self.entries.insert(b, value);
    }

    #[inline(always)]
    pub fn contains_key(&self, b: &majit_ir::operand::Operand) -> bool {
        self.entries.contains_key(b)
    }

    /// Iterate over all (canonical box, tag) pairs in RPython dict insertion
    /// order (Rc::ptr_eq identity = PyPy `box is box`).
    pub fn iter(&self) -> impl Iterator<Item = (majit_ir::operand::Operand, i16)> + '_ {
        self.entries.iter().map(|(op, v)| (op.clone(), *v))
    }
}

impl Default for LiveboxMap {
    fn default() -> Self {
        Self::new()
    }
}

// resume.py:134-139
pub struct NumberingState {
    pub writer: crate::resumecode::Writer,
    pub liveboxes: LiveboxMap,
    pub num_boxes: i32,
    pub num_virtuals: i32,
    /// RPython Box.type parity: type of each TAGBOX livebox, captured at
    /// numbering time when env.get_type() is called. Eliminates the need
    /// for post-hoc type inference cascades in store_final_boxes_in_guard.
    ///
    /// Keyed by the typed OpRef (resoperation.py:719-739 InputArg{Int,
    /// Ref,Float}, resoperation.py:564-638 *Op mixins) so that
    /// `InputArgRef(0)` and `RefOp(0)` do not collapse onto the same
    /// raw u32 — pyre's flat-OpRef stand-in for PyPy's `box is box`
    /// identity. See `LiveboxMap` (resume.rs:98) for the matching
    /// typed-key convention.
    pub livebox_types: LiveboxTypeMap,
}

impl NumberingState {
    pub fn new(size: usize) -> Self {
        NumberingState {
            writer: crate::resumecode::Writer::new(size),
            liveboxes: LiveboxMap::new(),
            num_boxes: 0,
            num_virtuals: 0,
            livebox_types: indexmap::IndexMap::default(),
        }
    }
    pub fn append_short(&mut self, item: i16) {
        self.writer.append_short(item as i32);
    }
    pub fn append_int(&mut self, item: i64) {
        self.writer.append_int(item);
    }
    pub fn patch_current_size(&mut self, index: usize) {
        self.writer.patch_current_size(index);
    }
    pub fn create_numbering(&self) -> Vec<u8> {
        self.writer.create_numbering()
    }
}

/// RPython snapshot: the state captured at a guard point.
/// Corresponds to RPython's SnapshotIterator output:
/// snapshot_iter.vable_array, snapshot_iter.vref_array, snapshot_iter.framestack.
///
/// NOTE: RPython does not have this struct. It uses `trace.get_snapshot_iter(position)`
/// which returns a lazy iterator over the trace's snapshot data (opencoder.py).
/// We use an eager struct because pyre's tracing records fail_args directly
/// on guard ops rather than using RPython's snapshot log format.
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// Virtualizable field boxes (resume.py:234-241).
    pub vable_array: Vec<SnapshotBox>,
    /// Virtualref pairs (resume.py:243-247). Length must be even.
    pub vref_array: Vec<SnapshotBox>,
    /// Frame chain (resume.py:249-253). Multiple frames for inlined calls.
    pub framestack: Vec<SnapshotFrame>,
}

/// A snapshot entry corresponding to one RPython Box.
///
/// RPython carries `box.type` on the Box object itself. Pyre's typed
/// `OpRef` enum (resoperation.py:719/727/739 InputArg{Int,Float,Ref},
/// resoperation.py:564-638 *Op mixin variants) carries the same type
/// tag intrinsically, so SnapshotBox copies it from the OpRef variant
/// at construction time. The explicit `tp` field stays around so the
/// SnapshotBox API can answer `box.type` without re-decoding the
/// variant on every read.
#[derive(Debug, Clone)]
pub struct SnapshotBox {
    /// The trace-position ref this snapshot slot references. A
    /// `Const{Ptr}` slot carries its gcref inline (history.py:314
    /// `ConstPtr.value`); during compilation the snapshot root walker
    /// (`walk_compile_snapshot_refs`) forwards it in place through a
    /// collected `*mut OpRef` slot address.
    pub opref: majit_ir::OpRef,
    pub tp: Option<majit_ir::Type>,
}

impl SnapshotBox {
    pub fn untyped(opref: majit_ir::OpRef) -> Self {
        SnapshotBox { opref, tp: None }
    }

    pub fn typed(opref: majit_ir::OpRef, tp: majit_ir::Type) -> Self {
        SnapshotBox {
            opref,
            tp: Some(tp),
        }
    }

    /// The trace-position `OpRef` view of this slot.
    pub fn opref(&self) -> majit_ir::OpRef {
        self.opref
    }

    pub fn map_opref(&self, f: impl FnOnce(majit_ir::OpRef) -> majit_ir::OpRef) -> Self {
        SnapshotBox {
            opref: f(self.opref),
            tp: self.tp,
        }
    }
}

impl From<majit_ir::OpRef> for SnapshotBox {
    fn from(opref: majit_ir::OpRef) -> Self {
        SnapshotBox::untyped(opref)
    }
}

/// One frame in a snapshot's frame chain.
#[derive(Debug, Clone)]
pub struct SnapshotFrame {
    /// Index into the jitcode table (resume.py:250 jitcode_index).
    pub jitcode_index: i32,
    /// Bytecode program counter (resume.py:250 pc).  In RPython this
    /// is the JitCode byte offset; pyre stores the Python bytecode PC
    /// here as a deviation (see `[[project-issue73-phase5-design]]`).
    /// Resume readers translate via `PyJitCode::resume_jitcode_pc_for`.
    pub pc: i32,
    /// Direct JitCode resume coordinate, or
    /// [`majit_ir::resumedata::NO_JITCODE_PC`] when this frame resumes
    /// through the Python `pc` → `pc_map` translation.  A branch-guard
    /// kept-stack capture stores the guard's JitCode byte offset here.
    pub jitcode_pc: i32,
    /// Live boxes for this frame's registers (resume.py:253).
    pub boxes: Vec<SnapshotBox>,
}

impl Snapshot {
    /// Create a simple single-frame snapshot (pyre common case).
    ///
    /// `jitcode_index` identifies the code this frame is running — the
    /// index into `METAINTERP_SD.jitcodes`. Required so the decoder's
    /// `frame_value_count_at(jitcode_index, pc)` query resolves the
    /// frame's liveness in the correct jitcode instead of silently
    /// falling through the pc-out-of-range LiveVars path on `jitcodes[0]`.
    pub fn single_frame(jitcode_index: i32, pc: i32, boxes: Vec<majit_ir::OpRef>) -> Self {
        Self::single_frame_boxes(
            jitcode_index,
            pc,
            boxes.into_iter().map(SnapshotBox::from).collect(),
        )
    }

    pub fn single_frame_boxes(jitcode_index: i32, pc: i32, boxes: Vec<SnapshotBox>) -> Self {
        Self::single_frame_boxes_with_jitcode_pc(
            jitcode_index,
            pc,
            majit_ir::resumedata::NO_JITCODE_PC,
            boxes,
        )
    }

    /// Like [`single_frame_boxes`], but carries the guard's JitCode byte
    /// offset (`jitcode_pc`) for kept-stack resume.  Callers that have no
    /// JitCode coordinate pass [`majit_ir::resumedata::NO_JITCODE_PC`].
    pub fn single_frame_boxes_with_jitcode_pc(
        jitcode_index: i32,
        pc: i32,
        jitcode_pc: i32,
        boxes: Vec<SnapshotBox>,
    ) -> Self {
        Snapshot {
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            framestack: vec![SnapshotFrame {
                jitcode_index,
                pc,
                jitcode_pc,
                boxes,
            }],
        }
    }

    /// Create a multi-frame snapshot from per-frame (jitcode_index, pc,
    /// boxes) tuples. Read-side ordering matches upstream: after
    /// `SnapshotIterator.__init__` calls `self.framestack.reverse()`
    /// (`opencoder.py:217`), `framestack[0]` is the outermost/caller
    /// frame and the last element is the innermost/callee — as asserted
    /// by `test_opencoder.py:123-130` (jc_index=2 at `framestack[0]`,
    /// jc_index=4 at `framestack[1]`). Input tuples for this factory
    /// follow the same caller-first order.
    pub fn multi_frame(frames: Vec<(i32, i32, Vec<majit_ir::OpRef>)>) -> Self {
        Self::multi_frame_boxes(
            frames
                .into_iter()
                .map(|(jitcode_index, pc, boxes)| {
                    (
                        jitcode_index,
                        pc,
                        boxes.into_iter().map(SnapshotBox::from).collect(),
                    )
                })
                .collect(),
        )
    }

    pub fn multi_frame_boxes(frames: Vec<(i32, i32, Vec<SnapshotBox>)>) -> Self {
        Self::multi_frame_boxes_with_jitcode_pc(
            frames
                .into_iter()
                .map(|(jitcode_index, pc, boxes)| {
                    (
                        jitcode_index,
                        pc,
                        majit_ir::resumedata::NO_JITCODE_PC,
                        boxes,
                    )
                })
                .collect(),
        )
    }

    /// Like [`multi_frame_boxes`], but each frame tuple carries the
    /// guard's JitCode byte offset (`jitcode_pc`, 3rd element) for
    /// kept-stack resume.  Frames with no JitCode coordinate pass
    /// [`majit_ir::resumedata::NO_JITCODE_PC`].
    pub fn multi_frame_boxes_with_jitcode_pc(
        frames: Vec<(i32, i32, i32, Vec<SnapshotBox>)>,
    ) -> Self {
        Snapshot {
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            framestack: frames
                .into_iter()
                .map(|(jitcode_index, pc, jitcode_pc, boxes)| SnapshotFrame {
                    jitcode_index,
                    pc,
                    jitcode_pc,
                    boxes,
                })
                .collect(),
        }
    }

    /// Estimated encoded size for NumberingState capacity hint.
    pub fn estimated_size(&self) -> usize {
        let frame_size: usize = self.framestack.iter().map(|f| f.boxes.len() + 2).sum();
        self.vable_array.len() + self.vref_array.len() + frame_size + 4
    }
}

/// Re-export BoxEnv from majit-ir.
pub use majit_ir::BoxEnv;

/// Simple BoxEnv implementation backed by constant/type associative containers.
/// Used in tests and for simple snapshot numbering.
pub struct SimpleBoxEnv {
    pub constants: indexmap::IndexMap<u32, (i64, majit_ir::Type)>,
    pub replacements: indexmap::IndexMap<u32, majit_ir::OpRef>,
    pub types: indexmap::IndexMap<u32, majit_ir::Type>,
    pub virtuals: indexmap::IndexSet<u32>,
    pub virtual_fields: indexmap::IndexMap<u32, majit_ir::VirtualFieldsInfo>,
    /// #160/S11: one canonical box `Operand` per replacement-walked OpRef. This
    /// env holds no producer Ops, so it memoizes here to give `Rc::ptr_eq` dedup
    /// parity with production (where `from_bound_op` memoizes on the Op).
    box_cache: std::cell::RefCell<indexmap::IndexMap<majit_ir::OpRef, majit_ir::operand::Operand>>,
}

impl SimpleBoxEnv {
    pub fn new() -> Self {
        SimpleBoxEnv {
            constants: indexmap::IndexMap::new(),
            replacements: indexmap::IndexMap::new(),
            types: indexmap::IndexMap::new(),
            virtuals: indexmap::IndexSet::new(),
            virtual_fields: indexmap::IndexMap::new(),
            box_cache: std::cell::RefCell::new(indexmap::IndexMap::new()),
        }
    }
}

impl BoxEnv for SimpleBoxEnv {
    // resoperation.py:57-68 get_box_replacement walks the chain
    // op -> op._forwarded -> ... until None / AbstractInfo, returning the
    // last item before that. Iterate the replacement map the same way.
    fn get_box_replacement(&self, opref: majit_ir::OpRef) -> majit_ir::OpRef {
        // history.py:227/268/314 inline-Const is its own forwarding root
        // (Const objects never participate in `_forwarded` per
        // resoperation.py:57 default `_forwarded = None`).
        if opref.inline_const_bits().is_some() {
            return opref;
        }
        let mut opref = opref;
        while let Some(next) = self.replacements.get(&opref.raw()).copied() {
            if next == opref {
                return opref;
            }
            opref = next;
        }
        opref
    }

    fn get_box_replacement_operand(&self, opref: majit_ir::OpRef) -> majit_ir::operand::Operand {
        // #160/S11: no producer Ops here, so memoize one Operand per
        // replacement-walked OpRef to mirror production's from_bound_op
        // memoization — two reaches of one logical box share an Rc (ptr_eq).
        let root = self.get_box_replacement(opref);
        if let Some(b) = self.box_cache.borrow().get(&root) {
            return b.clone();
        }
        // This env holds no producer Ops; in tests, synthesize a rooted bound
        // producer so the box sheds to `Operand::Op`/`InputArg` (the Operand-
        // keyed liveboxes/cached maps reject a position-only box). The method
        // is never reached in non-test builds, where `from_opref` is retained.
        #[cfg(test)]
        let b = crate::history::test_support::rooted_operand_from_opref(root);
        #[cfg(not(test))]
        let b = majit_ir::operand::Operand::from_opref(root);
        self.box_cache.borrow_mut().insert(root, b.clone());
        b
    }

    // resoperation.py:64-65 not_const arm: stop one hop before reaching
    // a Const target, returning the predecessor.
    fn get_box_replacement_not_const(&self, opref: majit_ir::OpRef) -> majit_ir::OpRef {
        if opref.inline_const_bits().is_some() {
            return opref;
        }
        let mut opref = opref;
        while let Some(next) = self.replacements.get(&opref.raw()).copied() {
            if next == opref {
                return opref;
            }
            if next.is_constant() {
                return opref;
            }
            // Legacy idx Const sentinel in side table.
            if next.inline_const_bits().is_none() && self.constants.contains_key(&next.raw()) {
                return opref;
            }
            opref = next;
        }
        opref
    }

    fn is_const(&self, opref: majit_ir::OpRef) -> bool {
        // history.py:227/268/314 inline-Const variants are constants by tag.
        if opref.is_constant() {
            return true;
        }
        self.constants.contains_key(&opref.raw())
    }
    fn get_const(&self, opref: majit_ir::OpRef) -> (i64, majit_ir::Type) {
        // history.py:227 ConstInt.value / :268 ConstFloat.value / :314 ConstPtr.value
        // inline on the Box; read directly without side-table.
        if let (Some(bits), Some(tp)) = (opref.inline_const_bits(), opref.ty()) {
            return (bits, tp);
        }
        self.constants
            .get(&opref.raw())
            .copied()
            .unwrap_or((0, majit_ir::Type::Int))
    }
    fn get_type(&self, opref: majit_ir::OpRef) -> majit_ir::Type {
        // resoperation.py:1693 opclasses[opnum].type — every typed OpRef
        // variant pins `.type` (history.py:220/261/307 + resoperation.py:567/589/615).
        if let Some(tp) = opref.ty() {
            return tp;
        }
        self.types
            .get(&opref.raw())
            .copied()
            .unwrap_or(majit_ir::Type::Int)
    }
    fn is_virtual_ref(&self, opref: majit_ir::OpRef) -> bool {
        if opref.inline_const_bits().is_some() {
            return false;
        }
        self.virtuals.contains(&opref.raw())
    }
    fn is_virtual_raw(&self, opref: majit_ir::OpRef) -> bool {
        if opref.inline_const_bits().is_some() {
            return false;
        }
        self.virtuals.contains(&opref.raw())
    }
    fn get_virtual_fields(&self, opref: majit_ir::OpRef) -> Option<majit_ir::VirtualFieldsInfo> {
        if opref.inline_const_bits().is_some() {
            return None;
        }
        self.virtual_fields.get(&opref.raw()).cloned()
    }
}

// resume.py:123-132 — tag constants (i64 widened for rd_numb encoding).
// Same values as the i16 TAGCONST/TAGINT/TAGBOX/TAGVIRTUAL above.
const TAGMASK_I64: i64 = TAGMASK as i64;
// resume.py:130 `NULLREF = tag(-1, TAGCONST)` — pre-shift num for
// `Const::Ref(NULL)`. Shared by the i16 `NULLREF` constant and the
// i64 `getconst_i64`/`decode_box` pair.
const ENCODED_NULLREF: i64 = -1;
const ENCODED_UNINITIALIZED: i64 = -2;
const ENCODED_UNAVAILABLE: i64 = -3;

// Two low bits are reserved for the tag.
const INLINE_TAGGED_MIN: i64 = -(1_i64 << 61);
const INLINE_TAGGED_MAX: i64 = (1_i64 << 61) - 1;

/// compile.py:853-876 `ResumeGuardDescr` storage.
///
/// Canonical, guard-owned resume payload (`storage.rd_numb/rd_consts/
/// rd_virtuals/rd_pendingfields`). Shared via `Arc<ResumeStorage>` so
/// every reader — `StoredExitLayout` (the sole carrier on the trace
/// surrogate after T4.4 retired the parallel `StoredResumeData` side
/// table), bridge retrace, blackhole resume, GC root walker —
/// observes the **same** pool, matching RPython's guard-owned
/// `ResumeGuardDescr` singleton.
///
/// `rd_consts` uses `UnsafeCell` because the GC root walker
/// (framework.py `root_walker.walk_roots` parity) rewrites `Const::Ref`
/// slots in place during minor collection, and the pyre runtime is
/// single-threaded so `Mutex` overhead is unnecessary.
pub struct ResumeStorage {
    /// resume.py:466 `storage.rd_numb` — packed byte stream (NUMBERING
    /// lltype equivalent). Immutable once installed.
    pub rd_numb: Vec<u8>,
    /// resume.py:467 `storage.rd_consts` — shared constant pool.
    ///
    /// Interior mutability: the minor-collection root walker visits
    /// `Const::Ref` slots to update forwarded GCREFs, so the slice
    /// must remain mutable after the `Arc` is shared. `UnsafeCell`
    /// is sound here because pyre is single-threaded and the walker
    /// holds exclusive access for the duration of each GC cycle.
    pub rd_consts: UnsafeCell<Vec<Const>>,
    /// compile.py:858 `storage.rd_virtuals` — live `RdVirtualInfo`
    /// entries describing virtual objects to materialize on resume.
    pub rd_virtuals: Vec<std::rc::Rc<majit_ir::RdVirtualInfo>>,
    /// resume.py:468 `storage.rd_pendingfields` — pending field writes
    /// replayed during blackhole resume.
    pub rd_pendingfields: Vec<majit_ir::GuardPendingFieldEntry>,
}

// Pyre is single-threaded; UnsafeCell prevents auto-Send/Sync so
// provide them explicitly (matches RPython's non-thread-safe
// ResumeGuardDescr).
unsafe impl Send for ResumeStorage {}
unsafe impl Sync for ResumeStorage {}

impl std::fmt::Debug for ResumeStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let consts_len = unsafe { (*self.rd_consts.get()).len() };
        f.debug_struct("ResumeStorage")
            .field("rd_numb_len", &self.rd_numb.len())
            .field("rd_consts_len", &consts_len)
            .field("rd_virtuals_len", &self.rd_virtuals.len())
            .field("rd_pendingfields_len", &self.rd_pendingfields.len())
            .finish()
    }
}

impl ResumeStorage {
    pub fn new(
        rd_numb: Vec<u8>,
        rd_consts: Vec<Const>,
        rd_virtuals: Vec<std::rc::Rc<majit_ir::RdVirtualInfo>>,
        rd_pendingfields: Vec<majit_ir::GuardPendingFieldEntry>,
    ) -> Arc<Self> {
        Arc::new(ResumeStorage {
            rd_numb,
            rd_consts: UnsafeCell::new(rd_consts),
            rd_virtuals,
            rd_pendingfields,
        })
    }

    /// Empty storage (pre-finalization placeholder).
    pub fn empty() -> Arc<Self> {
        Self::new(Vec::new(), Vec::new(), Vec::new(), Vec::new())
    }

    /// Snapshot the constant pool (for readers that need an owned
    /// copy — e.g. legacy `Vec<Const>`-typed APIs before their
    /// migration to the shared storage handle).
    pub fn rd_consts_snapshot(&self) -> Vec<Const> {
        unsafe { (*self.rd_consts.get()).clone() }
    }

    /// Borrow `rd_consts` for reading. Safety: the root walker holds
    /// the only writer and runs during GC, outside of reader scope.
    pub fn rd_consts(&self) -> &[Const] {
        unsafe { &*self.rd_consts.get() }
    }

    /// Internal accessor for the GC root walker. SAFETY: caller must
    /// ensure exclusive access — only the minor-collection walker in
    /// `MetaInterp::walk_rd_consts_refs` uses this.
    pub(crate) unsafe fn rd_consts_mut_for_gc(&self) -> &mut Vec<Const> {
        unsafe { &mut *self.rd_consts.get() }
    }
}

/// resume.py: ResumeGuardDescr storage fields.
///
/// `rd_numb` is a flat encoded numbering section (resume.py:466):
/// 1. items_resume_section (total rd_numb length)
/// 2. count (number of liveboxes, resume.py:921)
/// 3. number of frames
/// 4. per-frame `(pc, slot_count, slot_sources...)`
///
/// Fields match RPython's `ResumeGuardDescr`:
/// - `rd_numb`: encoded numbering (resume.py:466)
/// - `rd_consts`: shared constant pool (resume.py:467)
/// - `rd_virtuals`: live VirtualInfo objects (compile.py:858)
/// - `rd_pendingfields`: pending field writes (resume.py:468)
#[derive(Debug, Clone)]
pub struct EncodedResumeData {
    /// resume.py:466 storage.rd_numb — flat encoded numbering section.
    pub rd_numb: Vec<i64>,
    /// resume.py:467 storage.rd_consts — shared constant pool.
    ///
    /// RPython stores `list[Const]` (history.py:220/261/307). We keep the
    /// same shape so Ref entries stay visible to the minor-collection root
    /// walker (framework.py `root_walker.walk_roots` parity).
    pub rd_consts: Vec<Const>,
    /// resume.py:468 storage.rd_pendingfields — pending field writes.
    pub rd_pendingfields: Vec<EncodedPendingFieldWrite>,
    /// compile.py:858 storage.rd_virtuals — live VirtualInfo objects.
    pub rd_virtuals: Vec<VirtualInfo>,
    /// resume.py:411 liveboxes — compact TAGBOX(n) → original FailArg index.
    /// In RPython, liveboxes[n] is the Box object that was assigned TAGBOX(n).
    /// Here, liveboxes[n] is the original deadframe slot index.
    pub liveboxes: Vec<usize>,
    /// Per-frame slot count — equivalent to jitcode liveness info.
    /// RPython uses jitcode.get_live_vars_info(pc) at decode time;
    /// we store the counts at encode time since this path lacks jitcodes.
    pub frame_sizes: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecodedResumeLayout {
    pub vable_array: Vec<ResumeValueSource>,
    pub vref_array: Vec<ResumeValueSource>,
    pub frames: Vec<FrameInfo>,
    pub virtuals: Vec<VirtualInfo>,
    pub pending_fields: Vec<PendingFieldInfo>,
}

// ResumeValueKind / ResumeValueLayoutSummary moved to
// majit-ir::resumedata (Phase C-1 cascade) so backend codepaths can
// reference the resume-value tag discriminator + summary without a
// metainterp dependency.
pub use majit_ir::resumedata::{ResumeValueKind, ResumeValueLayoutSummary};

// ResumeFrameLayoutSummary moved to majit-ir::resumedata (Phase C-1
// cascade); re-exported for caller compatibility.
pub use majit_ir::resumedata::ResumeFrameLayoutSummary;

// ResumeVirtualKind moved to majit-ir::resumedata (Phase C-1
// cascade); re-exported for caller compatibility.
pub use majit_ir::resumedata::ResumeVirtualKind;

// ResumeVirtualLayoutSummary moved to majit-ir::resumedata (Phase C-1
// cascade); re-exported for caller compatibility.
pub use majit_ir::resumedata::ResumeVirtualLayoutSummary;

// PendingFieldLayoutSummary moved to majit-ir::resumedata along with
// opt_descr_arc_ptr_eq helper (Phase C-1 cascade); re-exported for
// caller compatibility.
pub use majit_ir::resumedata::{PendingFieldLayoutSummary, opt_descr_arc_ptr_eq};

#[derive(Debug, Clone)]
pub struct ResumeLayoutSummary {
    pub num_frames: usize,
    pub frame_pcs: Vec<u64>,
    pub frame_slot_counts: Vec<usize>,
    pub frame_layouts: Vec<ResumeFrameLayoutSummary>,
    pub num_virtuals: usize,
    pub virtual_kinds: Vec<ResumeVirtualKind>,
    pub virtual_layouts: Vec<ResumeVirtualLayoutSummary>,

    pub pending_field_count: usize,
    pub pending_field_layouts: Vec<PendingFieldLayoutSummary>,
    pub const_pool_size: usize,
}

// ResumeValueLayoutSummary inherent impls moved to
// majit-backend::resume_value (Phase C-1 cascade).  The conversion
// methods (raw_fail_arg_position, to_resume_source, to_exit_source)
// are available through the ResumeValueLayoutSummaryExt trait
// re-exported above.

// ResumeFrameLayoutSummary inherent impls extracted as free functions
// (cross-crate orphan rule after the type moved to majit-ir::resumedata).
fn resume_frame_layout_to_frame_info(layout: &ResumeFrameLayoutSummary) -> FrameInfo {
    FrameInfo {
        jitcode_index: layout.jitcode_index,
        pc: layout.pc,
        slot_map: layout
            .slot_layouts
            .iter()
            .map(|slot| slot.to_resume_source())
            .collect(),
    }
}

fn resume_frame_layout_to_exit_frame_layout(
    layout: &ResumeFrameLayoutSummary,
    virtual_offset: usize,
) -> ExitFrameLayout {
    ExitFrameLayout {
        trace_id: layout.trace_id,
        header_pc: layout.header_pc,
        source_guard: layout.source_guard,
        pc: layout.pc,
        jitcode_index: layout.jitcode_index,
        slots: layout
            .slot_layouts
            .iter()
            .map(|slot| slot.to_exit_source(virtual_offset))
            .collect(),
        slot_types: layout.slot_types.clone(),
    }
}

/// Build a `ResumeFrameLayoutSummary` from a backend-origin `ExitFrameLayout`.
///
/// Each `ExitValueSourceLayout` slot is converted to the corresponding
/// `ResumeValueLayoutSummary`, preserving slot types when present.
/// Free function — `ResumeFrameLayoutSummary` lives in `majit-ir`
/// (orphan rule prevents inherent impl outside of that crate).
pub fn resume_frame_layout_from_exit_frame_layout(
    exit_frame: &ExitFrameLayout,
) -> ResumeFrameLayoutSummary {
    let slot_layouts: Vec<ResumeValueLayoutSummary> = exit_frame
        .slots
        .iter()
        .map(majit_backend::resume_value_layout_summary_from_exit_value_source)
        .collect();
    let slot_sources: Vec<ResumeValueKind> = slot_layouts.iter().map(|s| s.kind).collect();

    ResumeFrameLayoutSummary {
        trace_id: exit_frame.trace_id,
        header_pc: exit_frame.header_pc,
        source_guard: exit_frame.source_guard,
        jitcode_index: exit_frame.jitcode_index,
        pc: exit_frame.pc,
        slot_sources,
        slot_layouts,
        slot_types: exit_frame.slot_types.clone(),
    }
}

// `from_exit_value_source` constructor moved to
// majit-backend::resume_value as
// `resume_value_layout_summary_from_exit_value_source` (cross-crate
// orphan rule prevents inherent impl on the foreign type).
// `to_virtual_info` / `to_exit_virtual_layout` extracted as free
// functions after `ResumeVirtualLayoutSummary` moved to
// `majit-ir::resumedata` (cross-crate orphan rule).
fn resume_virtual_layout_to_virtual_info(layout: &ResumeVirtualLayoutSummary) -> VirtualInfo {
    let s = layout;
    {
        match s {
            ResumeVirtualLayoutSummary::Object {
                descr,
                type_id,
                known_class,
                fields,
                fielddescrs,
                descr_size,
            } => VirtualInfo::VirtualObj {
                descr: descr.clone(),
                type_id: *type_id,
                known_class: *known_class,
                fields: fields
                    .iter()
                    .map(|(fd, src)| (*fd, src.to_resume_source()))
                    .collect(),
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            },
            ResumeVirtualLayoutSummary::Struct {
                typedescr,
                type_id,
                fields,
                fielddescrs,
                descr_size,
            } => VirtualInfo::VStruct {
                typedescr: typedescr.clone(),
                type_id: *type_id,
                fields: fields
                    .iter()
                    .map(|(fd, src)| (*fd, src.to_resume_source()))
                    .collect(),
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            },
            ResumeVirtualLayoutSummary::Array {
                arraydescr,
                clear,
                items,
            } => VirtualInfo::VArray {
                arraydescr: arraydescr.clone(),
                clear: *clear,
                items: items
                    .iter()
                    .map(|source| source.to_resume_source())
                    .collect(),
            },
            ResumeVirtualLayoutSummary::ArrayStruct {
                arraydescr,
                fielddescrs,
                element_fields,
                ..
            } => VirtualInfo::VArrayStruct {
                arraydescr: arraydescr.clone(),
                fielddescrs: fielddescrs.clone(),
                element_fields: element_fields
                    .iter()
                    .map(|fields| {
                        fields
                            .iter()
                            .map(|(field_descr, source)| (*field_descr, source.to_resume_source()))
                            .collect()
                    })
                    .collect(),
            },
            ResumeVirtualLayoutSummary::RawBuffer {
                func,
                size,
                offsets,
                descrs,
                values,
            } => VirtualInfo::VRawBuffer {
                func: *func,
                size: *size,
                offsets: offsets.clone(),
                descrs: descrs.clone(),
                values: values
                    .iter()
                    .map(|source| source.to_resume_source())
                    .collect(),
            },
            ResumeVirtualLayoutSummary::RawSlice { offset, parent } => VirtualInfo::VRawSlice {
                offset: *offset,
                parent: parent.to_resume_source(),
            },
            ResumeVirtualLayoutSummary::StrPlain { chars } => VirtualInfo::VStrPlain {
                chars: chars
                    .iter()
                    .map(|source| source.to_resume_source())
                    .collect(),
            },
            ResumeVirtualLayoutSummary::StrConcat { left, right } => VirtualInfo::VStrConcat {
                left: Box::new(left.to_resume_source()),
                right: Box::new(right.to_resume_source()),
            },
            ResumeVirtualLayoutSummary::StrSlice {
                source,
                start,
                length,
            } => VirtualInfo::VStrSlice {
                source: Box::new(source.to_resume_source()),
                start: Box::new(start.to_resume_source()),
                length: Box::new(length.to_resume_source()),
            },
            ResumeVirtualLayoutSummary::UniPlain { chars } => VirtualInfo::VUniPlain {
                chars: chars
                    .iter()
                    .map(|source| source.to_resume_source())
                    .collect(),
            },
            ResumeVirtualLayoutSummary::UniConcat { left, right } => VirtualInfo::VUniConcat {
                left: Box::new(left.to_resume_source()),
                right: Box::new(right.to_resume_source()),
            },
            ResumeVirtualLayoutSummary::UniSlice {
                source,
                start,
                length,
            } => VirtualInfo::VUniSlice {
                source: Box::new(source.to_resume_source()),
                start: Box::new(start.to_resume_source()),
                length: Box::new(length.to_resume_source()),
            },
        }
    }
}

fn resume_virtual_layout_to_exit_virtual_layout(
    layout: &ResumeVirtualLayoutSummary,
    virtual_offset: usize,
) -> ExitVirtualLayout {
    let s = layout;
    {
        match s {
            ResumeVirtualLayoutSummary::Object {
                descr,
                type_id,
                known_class,
                fields,
                fielddescrs,
                descr_size,
            } => ExitVirtualLayout::Object {
                descr: descr.clone(),
                type_id: *type_id,
                known_class: *known_class,
                fields: fields
                    .iter()
                    .map(|(fd, src)| (*fd, src.to_exit_source(virtual_offset)))
                    .collect(),
                target_slot: None,
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            },
            ResumeVirtualLayoutSummary::Struct {
                typedescr,
                type_id,
                fields,
                fielddescrs,
                descr_size,
            } => ExitVirtualLayout::Struct {
                typedescr: typedescr.clone(),
                type_id: *type_id,
                fields: fields
                    .iter()
                    .map(|(field_descr, source)| {
                        (*field_descr, source.to_exit_source(virtual_offset))
                    })
                    .collect(),
                target_slot: None,
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            },
            ResumeVirtualLayoutSummary::Array {
                arraydescr,
                clear,
                items,
            } => ExitVirtualLayout::Array {
                arraydescr: arraydescr.clone(),
                clear: *clear,
                // resume.py:656-670: element type from arraydescr
                kind: array_kind_from_descr(arraydescr.as_ref()),
                items: items
                    .iter()
                    .map(|source| source.to_exit_source(virtual_offset))
                    .collect(),
            },
            ResumeVirtualLayoutSummary::ArrayStruct {
                arraydescr,
                fielddescrs,
                element_fields,
            } => ExitVirtualLayout::ArrayStruct {
                arraydescr: arraydescr.clone(),
                fielddescrs: fielddescrs.clone(),
                element_fields: element_fields
                    .iter()
                    .map(|fields| {
                        fields
                            .iter()
                            .map(|(field_descr, source)| {
                                (*field_descr, source.to_exit_source(virtual_offset))
                            })
                            .collect()
                    })
                    .collect(),
            },
            ResumeVirtualLayoutSummary::RawBuffer {
                func,
                size,
                offsets,
                descrs,
                values,
            } => ExitVirtualLayout::RawBuffer {
                func: *func,
                size: *size,
                offsets: offsets.clone(),
                descrs: descrs.clone(),
                values: values
                    .iter()
                    .map(|source| source.to_exit_source(virtual_offset))
                    .collect(),
            },
            ResumeVirtualLayoutSummary::RawSlice { offset, parent } => {
                ExitVirtualLayout::RawSlice {
                    offset: *offset,
                    base: parent.to_exit_source(virtual_offset),
                }
            }
            ResumeVirtualLayoutSummary::StrPlain { chars } => ExitVirtualLayout::StrPlain {
                is_unicode: false,
                chars: chars
                    .iter()
                    .map(|source| source.to_exit_source(virtual_offset))
                    .collect(),
            },
            ResumeVirtualLayoutSummary::UniPlain { chars } => ExitVirtualLayout::StrPlain {
                is_unicode: true,
                chars: chars
                    .iter()
                    .map(|source| source.to_exit_source(virtual_offset))
                    .collect(),
            },
            // resume.py:781 VStrConcatInfo / resume.py:836 VUniConcatInfo
            // — funcptr/calldescr resolved at materialization via
            // `callinfocollection.funcptr_for_oopspec(OS_STR_CONCAT /
            // OS_UNI_CONCAT)` (resume.py:1467-1468 / 1494-1495), so the
            // exit layout carries no funcptr.
            ResumeVirtualLayoutSummary::StrConcat { left, right } => ExitVirtualLayout::StrConcat {
                is_unicode: false,
                left: left.to_exit_source(virtual_offset),
                right: right.to_exit_source(virtual_offset),
            },
            ResumeVirtualLayoutSummary::UniConcat { left, right } => ExitVirtualLayout::StrConcat {
                is_unicode: true,
                left: left.to_exit_source(virtual_offset),
                right: right.to_exit_source(virtual_offset),
            },
            // resume.py:801 VStrSliceInfo / resume.py:856 VUniSliceInfo
            // — funcptr/calldescr resolved via callinfocollection at
            // materialization (resume.py:1477-1478 / 1504-1505).
            ResumeVirtualLayoutSummary::StrSlice {
                source,
                start,
                length,
            } => ExitVirtualLayout::StrSlice {
                is_unicode: false,
                str_src: source.to_exit_source(virtual_offset),
                start: start.to_exit_source(virtual_offset),
                length: length.to_exit_source(virtual_offset),
            },
            ResumeVirtualLayoutSummary::UniSlice {
                source,
                start,
                length,
            } => ExitVirtualLayout::StrSlice {
                is_unicode: true,
                str_src: source.to_exit_source(virtual_offset),
                start: start.to_exit_source(virtual_offset),
                length: length.to_exit_source(virtual_offset),
            },
        }
    }
}

// PendingFieldLayoutSummary inherent impls extracted as free
// functions (cross-crate orphan rule after the type moved to
// majit-ir::resumedata).
fn pending_field_layout_to_pending_field_info(
    layout: &PendingFieldLayoutSummary,
) -> PendingFieldInfo {
    PendingFieldInfo {
        descr: layout.descr.clone(),
        target: layout.target.to_resume_source(),
        value: layout.value.to_resume_source(),
        item_index: layout.item_index,
    }
}

fn pending_field_layout_to_exit_pending_field_layout(
    layout: &PendingFieldLayoutSummary,
    virtual_offset: usize,
) -> ExitPendingFieldLayout {
    ExitPendingFieldLayout {
        descr: layout.descr.clone(),
        item_index: layout.item_index,
        is_array_item: layout.is_array_item,
        target: layout.target.to_exit_source(virtual_offset),
        value: layout.value.to_exit_source(virtual_offset),
    }
}

impl ResumeLayoutSummary {
    pub fn to_resume_data(&self) -> ResumeData {
        ResumeData {
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            frames: self
                .frame_layouts
                .iter()
                .map(resume_frame_layout_to_frame_info)
                .collect(),
            virtuals: self
                .virtual_layouts
                .iter()
                .map(resume_virtual_layout_to_virtual_info)
                .collect(),
            pending_fields: self
                .pending_field_layouts
                .iter()
                .map(pending_field_layout_to_pending_field_info)
                .collect(),
        }
    }

    pub fn to_exit_recovery_layout(&self) -> ExitRecoveryLayout {
        self.to_exit_recovery_layout_with_caller_prefix(None)
    }

    pub fn to_exit_recovery_layout_with_caller_prefix(
        &self,
        caller_prefix: Option<&ExitRecoveryLayout>,
    ) -> ExitRecoveryLayout {
        if self.frame_layouts.is_empty() {
            return caller_prefix.cloned().unwrap_or(ExitRecoveryLayout {
                vable_array: Vec::new(),
                vref_array: Vec::new(),
                frames: Vec::new(),
                virtual_layouts: Vec::new(),
                pending_field_layouts: Vec::new(),
            });
        }

        let prefix_frame_count = caller_prefix
            .map(|layout| layout.frames.len().saturating_sub(self.frame_layouts.len()))
            .unwrap_or(0);
        let preserve_prefix = prefix_frame_count > 0;

        let mut frames = caller_prefix
            .map(|layout| layout.frames[..prefix_frame_count].to_vec())
            .unwrap_or_default();
        // RPython parity: rd_virtuals is stored once on the guard descriptor
        // and never replaced (compile.py:866, resume.py:492). Always preserve
        // caller_prefix's virtual_layouts — they originate from
        // build_guard_metadata and must not be overwritten.
        let mut virtual_layouts = caller_prefix
            .map(|layout| layout.virtual_layouts.clone())
            .unwrap_or_default();
        let mut pending_field_layouts = if preserve_prefix {
            caller_prefix
                .map(|layout| layout.pending_field_layouts.clone())
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let virtual_offset = virtual_layouts.len();

        frames.extend(
            self.frame_layouts
                .iter()
                .map(|frame| resume_frame_layout_to_exit_frame_layout(frame, virtual_offset)),
        );
        virtual_layouts.extend(
            self.virtual_layouts
                .iter()
                .map(|virt| resume_virtual_layout_to_exit_virtual_layout(virt, virtual_offset)),
        );
        pending_field_layouts.extend(self.pending_field_layouts.iter().map(|pending| {
            pending_field_layout_to_exit_pending_field_layout(pending, virtual_offset)
        }));

        ExitRecoveryLayout {
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            frames,
            virtual_layouts,
            pending_field_layouts,
        }
    }

    pub fn reconstruct_state(&self, fail_values: &[i64]) -> ReconstructedState {
        let resume_data = self.to_resume_data();
        let virtuals = resume_data.materialize_virtuals(fail_values);
        let pending_fields =
            ResumeData::resolve_pending_field_writes(&resume_data.pending_fields, fail_values);
        let frames = self
            .frame_layouts
            .iter()
            .map(|frame| ReconstructedFrame {
                trace_id: frame.trace_id,
                header_pc: frame.header_pc,
                source_guard: frame.source_guard,
                pc: frame.pc,
                jitcode_index: frame.jitcode_index,
                slot_types: frame.slot_types.clone(),
                values: frame
                    .slot_layouts
                    .iter()
                    .map(|slot| {
                        ResumeData::resolve_frame_slot_source(&slot.to_resume_source(), fail_values)
                    })
                    .collect(),
            })
            .collect();
        ReconstructedState {
            frames,
            virtuals,
            pending_fields,
        }
    }

    pub fn reconstruct(&self, fail_values: &[i64]) -> Vec<ReconstructedFrame> {
        self.reconstruct_state(fail_values).frames
    }

    pub fn reconstruct_frame(
        &self,
        frame_index: usize,
        fail_values: &[i64],
    ) -> Option<ReconstructedFrame> {
        let frame = self.frame_layouts.get(frame_index)?;
        Some(ReconstructedFrame {
            trace_id: frame.trace_id,
            header_pc: frame.header_pc,
            source_guard: frame.source_guard,
            pc: frame.pc,
            jitcode_index: frame.jitcode_index,
            slot_types: frame.slot_types.clone(),
            values: frame
                .slot_layouts
                .iter()
                .map(|slot| {
                    ResumeData::resolve_frame_slot_source(&slot.to_resume_source(), fail_values)
                })
                .collect(),
        })
    }

    pub fn materialize_virtuals(&self, fail_values: &[i64]) -> Vec<MaterializedVirtual> {
        self.to_resume_data().materialize_virtuals(fail_values)
    }

    pub fn resolve_pending_field_writes(
        &self,
        fail_values: &[i64],
    ) -> Vec<ResolvedPendingFieldWrite> {
        let resume_data = self.to_resume_data();
        ResumeData::resolve_pending_field_writes(&resume_data.pending_fields, fail_values)
    }
}

fn can_inline_tagged(value: i64) -> bool {
    (INLINE_TAGGED_MIN..=INLINE_TAGGED_MAX).contains(&value)
}

// `encode_tagged_source` has been promoted to a method on
// `ResumeDataLoopMemo` (see `ResumeDataLoopMemo::encode_tagged_source`)
// so it can share `self.consts` with `getconst`/`newconst` — matching
// RPython's single `self.consts: list[Const]` pool (resume.py:147).

/// resume.py:99-104 tag() — i64 widened variant for rd_numb encoding.
fn tag_i64(value: i64, tagbits: u8) -> i64 {
    debug_assert!(tagbits <= 3);
    debug_assert!(
        can_inline_tagged(value),
        "tagged resume value {value} exceeds inline range"
    );
    (value << 2) | tagbits as i64
}

/// resume.py:106-109 untag() — i64 widened variant for rd_numb decoding.
fn untag_i64(encoded: i64) -> (i64, u8) {
    ((encoded >> 2), (encoded & TAGMASK_I64) as u8)
}

fn encode_len(value: usize) -> i64 {
    i64::try_from(value).expect("resume length exceeds i64")
}

fn decode_len(value: i64) -> usize {
    usize::try_from(value).expect("negative or oversized resume length")
}

fn encode_u64(value: u64) -> i64 {
    value as i64
}

fn decode_u64(value: i64) -> u64 {
    value as u64
}

// FrameInfo moved to majit-backend::resume_value (Phase C-1 cascade).
// Re-exported so existing crate::resume::FrameInfo references stay
// resolvable.
pub use majit_backend::FrameInfo;

// ResumeData moved to majit-backend::resume_value (Phase C-1 cascade);
// re-exported for caller compatibility.  Inherent impl methods are
// provided by ResumeDataExt trait declared below.
pub use majit_backend::ResumeData;

// ResumeValueSource moved to majit-backend::resume_value (Phase C-1
// cascade) so the resume-data type chain can live in a
// backend-accessible crate.  Re-export so existing
// crate::resume::ResumeValueSource references stay resolvable.
pub use majit_backend::ResumeValueLayoutSummaryExt;
pub use majit_backend::ResumeValueSource;

// FrameSlotSource type alias re-exported from majit-backend
// (Phase C-1 cascade) alongside FrameInfo.
pub use majit_backend::FrameSlotSource;

// VirtualInfo moved to majit-backend::resume_value (Phase C-1
// cascade) along with its PartialEq + first impl block
// (field_sources / kind / layout_summary).  The second impl block
// (allocate / is_about_raw, line ~5577) stays in metainterp since it
// depends on ResumeDataDirectReader + BlackholeAllocator which are
// metainterp-specific.  Re-exported here for caller compatibility.
pub use majit_backend::VirtualInfo;

// VirtualFieldSource type alias re-exported from majit-backend
// (Phase C-1 cascade), same as FrameSlotSource.
pub use majit_backend::VirtualFieldSource;

/// Convert a tagged fieldnum (i16, resume.py encoding) to a VirtualFieldSource.
///
/// resume.py:1552-1596 decode_int/decode_ref: tagged values encode where
/// each field value comes from at resume time.
///
/// `consts` is the rd_consts array. `count` is the number of fail_args
/// (used for negative TAGBOX indices). `num_virtuals` is the length of
/// rd_virtuals (used for negative TAGVIRTUAL indices — nested virtuals
/// are numbered negatively by `assign_number_to_virtual`,
/// resume.py:278-284, and resolved via Python negative list indexing).
/// All come from the containing ResumeGuardDescr / EncodedResumeData.
pub fn tagged_to_source(
    tagged: i16,
    consts: &[majit_ir::Const],
    count: i32,
    num_virtuals: usize,
) -> VirtualFieldSource {
    if tagged_eq(tagged, UNASSIGNED) {
        return ResumeValueSource::Unavailable;
    }
    if tagged_eq(tagged, UNINITIALIZED_TAG) {
        return ResumeValueSource::Uninitialized;
    }
    if tagged_eq(tagged, NULLREF) {
        // history.py:361 CONST_NULL = ConstPtr(null). resume.py:1589 parity.
        return ResumeValueSource::Constant(majit_ir::Const::Ref(majit_ir::GcRef::NULL));
    }
    let (num, tag_bits) = untag(tagged);
    match tag_bits {
        TAGCONST => {
            let idx = (num - TAG_CONST_OFFSET) as usize;
            if idx < consts.len() {
                // resume.py:1568 self.consts[num - TAG_CONST_OFFSET] — the
                // Const object carries its type (ConstInt/ConstFloat/ConstPtr).
                ResumeValueSource::Constant(consts[idx])
            } else {
                ResumeValueSource::Constant(majit_ir::Const::Int(0))
            }
        }
        // resume.py:1581 ConstInt(num) — always Int type for TAGINT.
        TAGINT => ResumeValueSource::Constant(majit_ir::Const::Int(num as i64)),
        TAGBOX => {
            let mut idx = num;
            if idx < 0 {
                idx += count;
            }
            ResumeValueSource::FailArg(idx as usize)
        }
        TAGVIRTUAL => {
            let mut idx = num;
            if idx < 0 {
                idx += num_virtuals as i32;
            }
            ResumeValueSource::Virtual(idx as usize)
        }
        _ => ResumeValueSource::Unavailable,
    }
}

/// Convert an `RdVirtualInfo` (IR-level, from compile.rs/pyjitpl.rs)
/// to a `VirtualInfo` (resume-level, used by ResumeDataDirectReader).
///
/// `consts`, `count` and `num_virtuals` are needed to decode tagged
/// fieldnums (see [`tagged_to_source`]).
pub fn rd_virtual_to_virtual_info(
    rd: &majit_ir::RdVirtualInfo,
    consts: &[majit_ir::Const],
    count: i32,
    num_virtuals: usize,
) -> VirtualInfo {
    match rd {
        majit_ir::RdVirtualInfo::VirtualInfo {
            descr,
            type_id,
            known_class,
            fielddescrs,
            fieldnums,
            descr_size,
        } => {
            let fields = fielddescrs
                .iter()
                .zip(fieldnums.iter())
                .map(|(fd, &tagged)| {
                    (
                        fd.index,
                        tagged_to_source(tagged, consts, count, num_virtuals),
                    )
                })
                .collect();
            VirtualInfo::VirtualObj {
                descr: descr.clone(),
                type_id: *type_id,
                known_class: *known_class,
                fields,
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            }
        }
        majit_ir::RdVirtualInfo::VStructInfo {
            typedescr,
            type_id,
            fielddescrs,
            fieldnums,
            descr_size,
        } => {
            let fields = fielddescrs
                .iter()
                .zip(fieldnums.iter())
                .map(|(fd, &tagged)| {
                    (
                        fd.index,
                        tagged_to_source(tagged, consts, count, num_virtuals),
                    )
                })
                .collect();
            VirtualInfo::VStruct {
                typedescr: typedescr.clone(),
                type_id: *type_id,
                fields,
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            }
        }
        majit_ir::RdVirtualInfo::VArrayInfoClear {
            arraydescr,
            fieldnums,
            ..
        } => {
            let items = fieldnums
                .iter()
                .map(|&tagged| tagged_to_source(tagged, consts, count, num_virtuals))
                .collect();
            VirtualInfo::VArray {
                arraydescr: arraydescr.clone(),
                clear: true,
                items,
            }
        }
        majit_ir::RdVirtualInfo::VArrayInfoNotClear {
            arraydescr,
            fieldnums,
            ..
        } => {
            let items = fieldnums
                .iter()
                .map(|&tagged| tagged_to_source(tagged, consts, count, num_virtuals))
                .collect();
            VirtualInfo::VArray {
                arraydescr: arraydescr.clone(),
                clear: false,
                items,
            }
        }
        majit_ir::RdVirtualInfo::VArrayStructInfo {
            arraydescr,
            size,
            fielddescrs: rd_fielddescrs,
            fieldnums,
            ..
        } => {
            // resume.py:736-740: VArrayStructInfo(arraydescr, size, fielddescrs)
            // fieldnums is flat: size * len(fielddescrs) entries
            let num_fields = rd_fielddescrs.len().max(1);
            let mut element_fields = Vec::with_capacity(*size);
            for chunk in fieldnums.chunks(num_fields) {
                // resume.py:754: for j in range(len(self.fielddescrs)):
                let elem: Vec<(u32, VirtualFieldSource)> = chunk
                    .iter()
                    .enumerate()
                    .map(|(j, &tagged)| {
                        (
                            j as u32,
                            tagged_to_source(tagged, consts, count, num_virtuals),
                        )
                    })
                    .collect();
                element_fields.push(elem);
            }
            while element_fields.len() < *size {
                element_fields.push(vec![]);
            }
            VirtualInfo::VArrayStruct {
                arraydescr: arraydescr.clone(),
                fielddescrs: rd_fielddescrs.clone(),
                element_fields,
            }
        }
        majit_ir::RdVirtualInfo::VRawBufferInfo {
            func,
            size,
            offsets,
            descrs,
            fieldnums,
        } => {
            assert_eq!(offsets.len(), descrs.len());
            assert_eq!(offsets.len(), fieldnums.len());
            let values = fieldnums
                .iter()
                .map(|&tagged| tagged_to_source(tagged, consts, count, num_virtuals))
                .collect();
            VirtualInfo::VRawBuffer {
                func: *func,
                size: *size,
                offsets: offsets.clone(),
                descrs: descrs.clone(),
                values,
            }
        }
        majit_ir::RdVirtualInfo::VRawSliceInfo { offset, fieldnums } => {
            let parent = fieldnums
                .first()
                .map(|&tagged| tagged_to_source(tagged, consts, count, num_virtuals))
                .unwrap_or(ResumeValueSource::Unavailable);
            VirtualInfo::VRawSlice {
                offset: *offset as i64,
                parent,
            }
        }
        majit_ir::RdVirtualInfo::VStrPlainInfo { fieldnums } => {
            let chars = fieldnums
                .iter()
                .map(|&tagged| tagged_to_source(tagged, consts, count, num_virtuals))
                .collect();
            VirtualInfo::VStrPlain { chars }
        }
        majit_ir::RdVirtualInfo::VStrConcatInfo { fieldnums } => {
            let left = Box::new(tagged_to_source(fieldnums[0], consts, count, num_virtuals));
            let right = Box::new(tagged_to_source(fieldnums[1], consts, count, num_virtuals));
            VirtualInfo::VStrConcat { left, right }
        }
        majit_ir::RdVirtualInfo::VStrSliceInfo { fieldnums } => {
            let source = Box::new(tagged_to_source(fieldnums[0], consts, count, num_virtuals));
            let start = Box::new(tagged_to_source(fieldnums[1], consts, count, num_virtuals));
            let length = Box::new(tagged_to_source(fieldnums[2], consts, count, num_virtuals));
            VirtualInfo::VStrSlice {
                source,
                start,
                length,
            }
        }
        majit_ir::RdVirtualInfo::VUniPlainInfo { fieldnums } => {
            let chars = fieldnums
                .iter()
                .map(|&tagged| tagged_to_source(tagged, consts, count, num_virtuals))
                .collect();
            VirtualInfo::VUniPlain { chars }
        }
        majit_ir::RdVirtualInfo::VUniConcatInfo { fieldnums } => {
            let left = Box::new(tagged_to_source(fieldnums[0], consts, count, num_virtuals));
            let right = Box::new(tagged_to_source(fieldnums[1], consts, count, num_virtuals));
            VirtualInfo::VUniConcat { left, right }
        }
        majit_ir::RdVirtualInfo::VUniSliceInfo { fieldnums } => {
            let source = Box::new(tagged_to_source(fieldnums[0], consts, count, num_virtuals));
            let start = Box::new(tagged_to_source(fieldnums[1], consts, count, num_virtuals));
            let length = Box::new(tagged_to_source(fieldnums[2], consts, count, num_virtuals));
            VirtualInfo::VUniSlice {
                source,
                start,
                length,
            }
        }
        majit_ir::RdVirtualInfo::Empty => VirtualInfo::VirtualObj {
            descr: None,
            type_id: 0,
            known_class: None,
            fields: vec![],
            fielddescrs: vec![],
            descr_size: 0,
        },
    }
}

// PendingFieldInfo moved to majit-backend::resume_value (Phase C-1
// cascade) along with its PartialEq and layout_summary impl.
// Re-exported for caller compatibility.
pub use majit_backend::PendingFieldInfo;

/// Concrete pending heap write reconstructed from resume data.
///
/// `resume.py:1000-1007 _prepare_pendingfields` parity — RPython
/// hands the live `descr` Arc into `setfield` / `setarrayitem` and
/// they dispatch via `descr.is_pointer_field()` /
/// `descr.is_array_of_pointers()` etc.
#[derive(Debug, Clone)]
pub struct ResolvedPendingFieldWrite {
    /// `resume.py:88 lldescr` — the field/array descriptor itself.
    pub descr: Option<majit_ir::DescrRef>,
    /// Concrete object/array pointer.
    pub target: MaterializedValue,
    /// Concrete value to write.
    pub value: MaterializedValue,
    /// Array item index. `None` means a plain field write.
    pub item_index: Option<usize>,
}

impl PartialEq for ResolvedPendingFieldWrite {
    fn eq(&self, other: &Self) -> bool {
        // `history.py:125 id(descr)` parity — descr identity via Arc::ptr_eq.
        majit_ir::resumedata::opt_descr_arc_ptr_eq(&self.descr, &other.descr)
            && self.target == other.target
            && self.value == other.value
            && self.item_index == other.item_index
    }
}
impl Eq for ResolvedPendingFieldWrite {}

/// Encoded pending field write stored alongside an encoded resume snapshot.
///
/// `resume.py:87-92 PENDINGFIELDSTRUCT` parity — the encoded form
/// carries `lldescr` (the descriptor object itself) so decoding can
/// hand back a live `Arc<dyn Descr>` via `descr.clone()` rather than
/// rebuilding it through an index lookup
/// (`resume.py:1000-1001 cast_base_ptr_to_instance`).
#[derive(Debug, Clone)]
pub struct EncodedPendingFieldWrite {
    /// `resume.py:88 lldescr` — the field/array descriptor itself.
    pub descr: Option<majit_ir::DescrRef>,
    pub target: i64,
    pub value: i64,
    pub item_index: Option<usize>,
}

impl PartialEq for EncodedPendingFieldWrite {
    fn eq(&self, other: &Self) -> bool {
        // `history.py:125 id(descr)` parity — descr identity via Arc::ptr_eq.
        majit_ir::resumedata::opt_descr_arc_ptr_eq(&self.descr, &other.descr)
            && self.target == other.target
            && self.value == other.value
            && self.item_index == other.item_index
    }
}
impl Eq for EncodedPendingFieldWrite {}

impl EncodedResumeData {
    pub fn encode(rd: &ResumeData) -> Self {
        Self::from_semantic(
            &rd.vable_array,
            &rd.vref_array,
            &rd.frames,
            &rd.virtuals,
            &rd.pending_fields,
        )
    }

    /// Build the guard-owned storage shape consumed by
    /// `ResumeDataDirectReader`.
    ///
    /// This is used only by the MetaInterp test helper that injects
    /// `ResumeData` after compilation. The production path obtains the
    /// same fields directly from `ResumeDataVirtualAdder::finish`.
    ///
    /// Pending-field replay still comes from `store_final_boxes_in_guard`
    /// in production. The encoded pending-field records do carry their
    /// live descr Arc (`resume.py:88 PENDINGFIELDSTRUCT.lldescr`), but
    /// this helper does not currently rebuild `GuardPendingFieldEntry`
    /// from them — the production path attaches that elsewhere.
    pub fn to_resume_storage(&self) -> Arc<ResumeStorage> {
        fn const_pool_tag(c: &Const, rd_consts: &mut Vec<Const>) -> i16 {
            let idx = rd_consts
                .iter()
                .position(|existing| existing == c)
                .unwrap_or_else(|| {
                    rd_consts.push(*c);
                    rd_consts.len() - 1
                });
            tag((idx as i32) + TAG_CONST_OFFSET, TAGCONST).unwrap_or(UNASSIGNED)
        }

        fn source_tag(
            source: &ResumeValueSource,
            liveboxes: &[usize],
            rd_consts: &mut Vec<Const>,
        ) -> i16 {
            match source {
                ResumeValueSource::FailArg(index) => liveboxes
                    .iter()
                    .position(|live| live == index)
                    .and_then(|compact| tag(compact as i32, TAGBOX).ok())
                    .unwrap_or(UNASSIGNED),
                ResumeValueSource::Constant(Const::Int(value)) => i32::try_from(*value)
                    .ok()
                    .and_then(|v| tag(v, TAGINT).ok())
                    .unwrap_or_else(|| const_pool_tag(&Const::Int(*value), rd_consts)),
                ResumeValueSource::Constant(Const::Ref(gcref)) if gcref.is_null() => NULLREF,
                ResumeValueSource::Constant(c) => const_pool_tag(c, rd_consts),
                ResumeValueSource::Virtual(index) => {
                    tag(*index as i32, TAGVIRTUAL).unwrap_or(UNASSIGNEDVIRTUAL)
                }
                ResumeValueSource::Uninitialized => UNINITIALIZED_TAG,
                ResumeValueSource::Unavailable => UNASSIGNED,
            }
        }

        fn fieldnums(
            sources: impl IntoIterator<Item = VirtualFieldSource>,
            liveboxes: &[usize],
            rd_consts: &mut Vec<Const>,
        ) -> Vec<i16> {
            sources
                .into_iter()
                .map(|source| source_tag(&source, liveboxes, rd_consts))
                .collect()
        }

        fn rd_virtual(
            info: &VirtualInfo,
            liveboxes: &[usize],
            rd_consts: &mut Vec<Const>,
        ) -> std::rc::Rc<majit_ir::RdVirtualInfo> {
            let rd = match info {
                VirtualInfo::VirtualObj {
                    descr,
                    type_id,
                    known_class,
                    fields,
                    fielddescrs,
                    descr_size,
                } => majit_ir::RdVirtualInfo::VirtualInfo {
                    descr: descr.clone(),
                    type_id: *type_id,
                    known_class: *known_class,
                    fielddescrs: fielddescrs.clone(),
                    fieldnums: fieldnums(
                        fields.iter().map(|(_, source)| source.clone()),
                        liveboxes,
                        rd_consts,
                    ),
                    descr_size: *descr_size,
                },
                VirtualInfo::VStruct {
                    typedescr,
                    type_id,
                    fields,
                    fielddescrs,
                    descr_size,
                } => majit_ir::RdVirtualInfo::VStructInfo {
                    typedescr: typedescr.clone(),
                    type_id: *type_id,
                    fielddescrs: fielddescrs.clone(),
                    fieldnums: fieldnums(
                        fields.iter().map(|(_, source)| source.clone()),
                        liveboxes,
                        rd_consts,
                    ),
                    descr_size: *descr_size,
                },
                VirtualInfo::VArray {
                    arraydescr,
                    clear,
                    items,
                } => {
                    let fieldnums = fieldnums(items.iter().cloned(), liveboxes, rd_consts);
                    if *clear {
                        majit_ir::RdVirtualInfo::VArrayInfoClear {
                            arraydescr: arraydescr.clone(),
                            kind: array_kind_from_descr(arraydescr.as_ref()),
                            fieldnums,
                        }
                    } else {
                        majit_ir::RdVirtualInfo::VArrayInfoNotClear {
                            arraydescr: arraydescr.clone(),
                            kind: array_kind_from_descr(arraydescr.as_ref()),
                            fieldnums,
                        }
                    }
                }
                VirtualInfo::VArrayStruct {
                    arraydescr,
                    fielddescrs,
                    element_fields,
                } => {
                    let mut flat = Vec::new();
                    for element in element_fields {
                        flat.extend(fieldnums(
                            element.iter().map(|(_, source)| source.clone()),
                            liveboxes,
                            rd_consts,
                        ));
                    }
                    // resume.py:740 self.fielddescrs — live InteriorFieldDescr
                    // objects expose offset/field_size/field_type via the
                    // FieldDescr trait (descr.py:273 / llmodel.py:648-649).
                    // Recover the per-field metadata from the live Arc rather
                    // than emitting placeholders; PyPy `make_virtual_info`
                    // (resume.py:488) forwards `fielddescrs[j]` to the
                    // VArrayStructInfo materialiser which reads
                    // `is_pointer_field`/`is_float_field`/offset/field_size
                    // through the same accessors at replay time
                    // (resume.py:751-757).
                    let field_types: Vec<u8> = fielddescrs
                        .iter()
                        .map(|fd| match fd.as_field_descr().map(|f| f.field_type()) {
                            Some(majit_ir::Type::Ref) => 0,
                            Some(majit_ir::Type::Float) => 2,
                            _ => 1,
                        })
                        .collect();
                    let field_offsets: Vec<usize> = fielddescrs
                        .iter()
                        .map(|fd| fd.as_field_descr().map(|f| f.offset()).unwrap_or(0))
                        .collect();
                    let field_sizes: Vec<usize> = fielddescrs
                        .iter()
                        .map(|fd| fd.as_field_descr().map(|f| f.field_size()).unwrap_or(8))
                        .collect();
                    majit_ir::RdVirtualInfo::VArrayStructInfo {
                        arraydescr: arraydescr.clone(),
                        size: element_fields.len(),
                        fielddescrs: fielddescrs.clone(),
                        fielddescr_indices: (0..fielddescrs.len()).map(|i| i as u32).collect(),
                        field_types,
                        base_size: arraydescr
                            .as_ref()
                            .and_then(|d| d.as_array_descr())
                            .map(|ad| ad.base_size())
                            .unwrap_or(0),
                        item_size: arraydescr
                            .as_ref()
                            .and_then(|d| d.as_array_descr())
                            .map(|ad| ad.item_size())
                            .unwrap_or(0),
                        field_offsets,
                        field_sizes,
                        fieldnums: flat,
                    }
                }
                VirtualInfo::VRawBuffer {
                    func,
                    size,
                    offsets,
                    descrs,
                    values,
                } => majit_ir::RdVirtualInfo::VRawBufferInfo {
                    func: *func,
                    size: *size,
                    offsets: offsets.clone(),
                    descrs: descrs.clone(),
                    fieldnums: fieldnums(values.iter().cloned(), liveboxes, rd_consts),
                },
                VirtualInfo::VRawSlice { offset, parent } => {
                    majit_ir::RdVirtualInfo::VRawSliceInfo {
                        offset: *offset,
                        fieldnums: fieldnums(std::iter::once(parent.clone()), liveboxes, rd_consts),
                    }
                }
                VirtualInfo::VStrPlain { chars } => majit_ir::RdVirtualInfo::VStrPlainInfo {
                    fieldnums: fieldnums(chars.iter().cloned(), liveboxes, rd_consts),
                },
                VirtualInfo::VStrConcat { left, right, .. } => {
                    majit_ir::RdVirtualInfo::VStrConcatInfo {
                        fieldnums: fieldnums(
                            [left.as_ref().clone(), right.as_ref().clone()],
                            liveboxes,
                            rd_consts,
                        ),
                    }
                }
                VirtualInfo::VStrSlice {
                    source,
                    start,
                    length,
                    ..
                } => majit_ir::RdVirtualInfo::VStrSliceInfo {
                    fieldnums: fieldnums(
                        [
                            source.as_ref().clone(),
                            start.as_ref().clone(),
                            length.as_ref().clone(),
                        ],
                        liveboxes,
                        rd_consts,
                    ),
                },
                VirtualInfo::VUniPlain { chars } => majit_ir::RdVirtualInfo::VUniPlainInfo {
                    fieldnums: fieldnums(chars.iter().cloned(), liveboxes, rd_consts),
                },
                VirtualInfo::VUniConcat { left, right, .. } => {
                    majit_ir::RdVirtualInfo::VUniConcatInfo {
                        fieldnums: fieldnums(
                            [left.as_ref().clone(), right.as_ref().clone()],
                            liveboxes,
                            rd_consts,
                        ),
                    }
                }
                VirtualInfo::VUniSlice {
                    source,
                    start,
                    length,
                    ..
                } => majit_ir::RdVirtualInfo::VUniSliceInfo {
                    fieldnums: fieldnums(
                        [
                            source.as_ref().clone(),
                            start.as_ref().clone(),
                            length.as_ref().clone(),
                        ],
                        liveboxes,
                        rd_consts,
                    ),
                },
            };
            std::rc::Rc::new(rd)
        }

        let mut writer = crate::resumecode::Writer::new(self.rd_numb.len());
        for &item in &self.rd_numb {
            writer.append_int(item);
        }
        let mut rd_consts = self.rd_consts.clone();
        let rd_virtuals = self
            .rd_virtuals
            .iter()
            .map(|info| rd_virtual(info, &self.liveboxes, &mut rd_consts))
            .collect();

        ResumeStorage::new(
            writer.create_numbering(),
            rd_consts,
            rd_virtuals,
            Vec::new(),
        )
    }

    /// resume.py:231-267 number + resume.py:380-468 finish
    ///
    /// Walks all frames via _number_boxes, assigning compact sequential
    /// TAGBOX numbers to unique liveboxes (resume.py:199-226).
    ///
    /// Unlike `ResumeDataLoopMemo::encode_shared`, this is a single-shot
    /// encoder (no cross-guard dedup). It builds a local memo that shares
    /// the same encoding logic so the tagged output is bit-compatible with
    /// the shared path. Used by tests and standalone embedders that don't
    /// carry a memo instance.
    fn from_semantic(
        vable_array: &[ResumeValueSource],
        vref_array: &[ResumeValueSource],
        frames: &[FrameInfo],
        virtuals: &[VirtualInfo],
        pending_fields: &[PendingFieldInfo],
    ) -> Self {
        let mut memo = ResumeDataLoopMemo::new();
        let mut rd_numb = Vec::new();
        // resume.py:138 numb_state.liveboxes — compact TAGBOX numbering state.
        let mut liveboxes: Vec<usize> = Vec::new();
        let mut box_map: indexmap::IndexMap<usize, usize> = indexmap::IndexMap::new();

        // resume.py:234-235: reserve slots for items_resume_section and count.
        rd_numb.push(0); // [0] = items_resume_section (patched later)
        rd_numb.push(0); // [1] = count (patched later)
        rd_numb.push(encode_len(vable_array.len()));
        for source in vable_array {
            let tagged = memo.encode_tagged_source(source, &mut liveboxes, &mut box_map);
            rd_numb.push(tagged);
        }
        // resume.py:243-247: vref_array (pairs).
        assert!(
            vref_array.len() % 2 == 0,
            "vref_array must have even length (pairs)"
        );
        rd_numb.push(encode_len(vref_array.len() / 2));
        for source in vref_array {
            let tagged = memo.encode_tagged_source(source, &mut liveboxes, &mut box_map);
            rd_numb.push(tagged);
        }

        // resume.py:249-253: per-frame encoding via _number_boxes.
        let mut frame_sizes = Vec::with_capacity(frames.len());
        for frame in frames {
            rd_numb.push(frame.jitcode_index as i64);
            rd_numb.push(encode_u64(frame.pc));
            // Per-frame `jitcode_pc` word; the test/embedder path never
            // captures a guard coordinate, so it stays the sentinel.
            rd_numb.push(majit_ir::resumedata::NO_JITCODE_PC as i64);
            // resume.py:253 _number_boxes(snapshot_iter, iter_array(snapshot), numb_state)
            for source in &frame.slot_map {
                let tagged = memo.encode_tagged_source(source, &mut liveboxes, &mut box_map);
                rd_numb.push(tagged);
            }
            frame_sizes.push(frame.slot_map.len());
        }

        // compile.py:858 rd_virtuals — stored as live objects, not serialized.
        let rd_virtuals = virtuals.to_vec();

        // resume.py:412-418: visitor_walk_recursive — register virtual field boxes.
        for vinfo in &rd_virtuals {
            for source in vinfo.field_sources() {
                if let ResumeValueSource::FailArg(index) = source {
                    box_map.entry(*index).or_insert_with(|| {
                        let n = liveboxes.len();
                        liveboxes.push(*index);
                        n
                    });
                }
            }
        }

        // resume.py:420-430: walk pending fields — register + encode.
        let rd_pendingfields: Vec<_> = pending_fields
            .iter()
            .map(|pending| EncodedPendingFieldWrite {
                // resume.py:547 lldescr = cast_instance_to_base_ptr(descr) —
                // the encoded form carries the descr itself, not a handle.
                descr: pending.descr.clone(),
                target: memo.encode_tagged_source(&pending.target, &mut liveboxes, &mut box_map),
                value: memo.encode_tagged_source(&pending.value, &mut liveboxes, &mut box_map),
                item_index: pending.item_index,
            })
            .collect();
        let rd_consts = memo.take_consts();

        // resume.py:260: numb_state.patch_current_size(0) → items_resume_section
        rd_numb[0] = encode_len(rd_numb.len());
        // resume.py:464: numb_state.patch(1, len(liveboxes)) → count
        rd_numb[1] = encode_len(liveboxes.len());

        EncodedResumeData {
            rd_numb,
            rd_consts,
            rd_pendingfields,
            rd_virtuals,
            liveboxes,
            frame_sizes,
        }
    }

    /// resume.py:916-923 AbstractResumeDataReader._init — decode rd_numb.
    fn decode_layout(&self) -> DecodedResumeLayout {
        let mut cursor = 0usize;
        // resume.py:919 items_resume_section
        let items_resume_section = self.next_word(&mut cursor);
        assert_eq!(
            decode_len(items_resume_section),
            self.rd_numb.len(),
            "resume item count mismatch"
        );
        // resume.py:921 self.count — number of liveboxes in the deadframe.
        let _count = decode_len(self.next_word(&mut cursor));

        let vable_count = decode_len(self.next_word(&mut cursor));
        let mut vable_array = Vec::with_capacity(vable_count);
        for _ in 0..vable_count {
            vable_array.push(self.decode_box(self.next_word(&mut cursor)));
        }
        let vref_count = decode_len(self.next_word(&mut cursor));
        let mut vref_array = Vec::with_capacity(vref_count * 2);
        for _ in 0..(vref_count * 2) {
            vref_array.push(self.decode_box(self.next_word(&mut cursor)));
        }
        // resume.py:1049-1055: frame section.
        // Per-frame: jitcode_index, pc, [tagged_values...].
        // RPython uses jitcode.get_live_vars_info(pc) for frame boundary;
        // we use self.frame_sizes[] stored at encode time.
        let items_resume_len = decode_len(items_resume_section);
        let mut frames = Vec::new();
        let mut frame_idx = 0usize;
        while cursor < items_resume_len {
            let jitcode_index = self.next_word(&mut cursor) as i32;
            let pc = decode_u64(self.next_word(&mut cursor));
            // Per-frame `jitcode_pc` word (after `pc`); discarded on this
            // layout-decode path, which carries only `FrameInfo`.
            let _jitcode_pc = self.next_word(&mut cursor);
            let slot_count = if frame_idx < self.frame_sizes.len() {
                self.frame_sizes[frame_idx]
            } else {
                // Single-frame fallback: consume all remaining items.
                items_resume_len - cursor
            };
            let mut slot_map = Vec::with_capacity(slot_count);
            for _ in 0..slot_count {
                slot_map.push(self.decode_box(self.next_word(&mut cursor)));
            }
            frames.push(FrameInfo {
                jitcode_index,
                pc,
                slot_map,
            });
            frame_idx += 1;
        }

        // compile.py:858 rd_virtuals — live objects, not deserialized from rd_numb.
        let virtuals = self.rd_virtuals.clone();

        assert_eq!(
            cursor,
            self.rd_numb.len(),
            "resume decoder left trailing data"
        );
        // resume.py:993-1001 _prepare_pendingfields — lldescr is restored
        // directly from `PENDINGFIELDSTRUCT.lldescr` via
        // `cast_base_ptr_to_instance(AbstractDescr, lldescr)`; pyre keeps
        // the live `Arc<dyn Descr>` on the encoded record, so decoding is
        // a clone.
        let pending_fields = self
            .rd_pendingfields
            .iter()
            .map(|pending| PendingFieldInfo {
                descr: pending.descr.clone(),
                target: self.decode_box(pending.target),
                value: self.decode_box(pending.value),
                item_index: pending.item_index,
            })
            .collect();
        DecodedResumeLayout {
            vable_array,
            vref_array,
            frames,
            virtuals,
            pending_fields,
        }
    }

    /// resume.py:919 resumecodereader.next_item()
    fn next_word(&self, cursor: &mut usize) -> i64 {
        let word = self
            .rd_numb
            .get(*cursor)
            .copied()
            .expect("truncated encoded resume data");
        *cursor += 1;
        word
    }

    /// resume.py:1240-1270 decode_box — decode a tagged value from rd_numb.
    fn decode_box(&self, encoded: i64) -> ResumeValueSource {
        let (value, tag) = untag_i64(encoded);
        match tag {
            // resume.py:1257 ConstInt(num).
            TAGINT => ResumeValueSource::Constant(majit_ir::Const::Int(value)),
            // resume.py:1261 self.liveboxes[num] — compact TAGBOX → original FailArg.
            TAGBOX => {
                let compact_idx = decode_len(value);
                let original_idx = self.liveboxes[compact_idx];
                ResumeValueSource::FailArg(original_idx)
            }
            TAGVIRTUAL => ResumeValueSource::Virtual(decode_len(value)),
            TAGCONST => match value {
                // resume.py:1552-1596 decode_ref: `if tagged_eq(tagged,
                // NULLREF): return CONST_NULL`. The i64 decoder mirrors
                // the i16 `decode_box`'s NULLREF fast-path (resume.rs
                // line 4363) so encoder/decoder stay symmetric.
                ENCODED_NULLREF => {
                    ResumeValueSource::Constant(majit_ir::Const::Ref(majit_ir::GcRef::NULL))
                }
                ENCODED_UNINITIALIZED => ResumeValueSource::Uninitialized,
                ENCODED_UNAVAILABLE => ResumeValueSource::Unavailable,
                index if index >= 0 => {
                    // resume.py:1555/1571/1583 self.consts[num - TAG_CONST_OFFSET]
                    // — the Const carries its own type.
                    let c = *self
                        .rd_consts
                        .get(decode_len(index))
                        .expect("resume const pool index out of bounds");
                    ResumeValueSource::Constant(c)
                }
                other => panic!("unknown CONST-tagged resume sentinel {other}"),
            },
            other => panic!("unknown resume tag {other}"),
        }
    }

    /// Decode this encoded snapshot back into a `ResumeData`.
    pub fn decode(&self) -> ResumeData {
        let layout = self.decode_layout();
        ResumeData {
            vable_array: layout.vable_array,
            vref_array: layout.vref_array,
            frames: layout.frames,
            virtuals: layout.virtuals,
            pending_fields: layout.pending_fields,
        }
    }

    /// Return a compact summary of this snapshot's frame/jitframe layout.
    pub fn layout_summary(&self) -> ResumeLayoutSummary {
        let layout = self.decode_layout();
        ResumeLayoutSummary {
            num_frames: layout.frames.len(),
            frame_pcs: layout.frames.iter().map(|frame| frame.pc).collect(),
            frame_slot_counts: layout
                .frames
                .iter()
                .map(|frame| frame.slot_map.len())
                .collect(),
            frame_layouts: layout
                .frames
                .iter()
                .map(|frame| ResumeFrameLayoutSummary {
                    trace_id: None,
                    header_pc: None,
                    source_guard: None,
                    jitcode_index: frame.jitcode_index,
                    pc: frame.pc,
                    slot_sources: frame.slot_map.iter().map(ResumeValueSource::kind).collect(),
                    slot_layouts: frame
                        .slot_map
                        .iter()
                        .map(|source| source.layout_summary())
                        .collect(),
                    slot_types: None,
                })
                .collect(),
            num_virtuals: layout.virtuals.len(),
            virtual_kinds: layout.virtuals.iter().map(VirtualInfo::kind).collect(),
            virtual_layouts: layout
                .virtuals
                .iter()
                .map(|virt| virt.layout_summary())
                .collect(),

            pending_field_count: layout.pending_fields.len(),
            pending_field_layouts: layout
                .pending_fields
                .iter()
                .map(|pending| pending.layout_summary())
                .collect(),
            const_pool_size: self.rd_consts.len(),
        }
    }

    /// Reconstruct the full interpreter state directly from the encoded snapshot.
    pub fn reconstruct_state(&self, fail_values: &[i64]) -> ReconstructedState {
        let layout = self.decode_layout();
        let virtuals = ResumeData::materialize_virtuals_from_infos(&layout.virtuals, fail_values);
        let pending_fields =
            ResumeData::resolve_pending_field_writes(&layout.pending_fields, fail_values);
        let frames = layout
            .frames
            .iter()
            .map(|frame| ReconstructedFrame {
                trace_id: None,
                header_pc: None,
                source_guard: None,
                pc: frame.pc,
                jitcode_index: frame.jitcode_index,
                slot_types: None,
                values: frame
                    .slot_map
                    .iter()
                    .map(|slot| ResumeData::resolve_frame_slot_source(slot, fail_values))
                    .collect(),
            })
            .collect();
        ReconstructedState {
            frames,
            virtuals,
            pending_fields,
        }
    }

    /// Reconstruct only the interpreter frames from the encoded snapshot.
    pub fn reconstruct(&self, fail_values: &[i64]) -> Vec<ReconstructedFrame> {
        self.reconstruct_state(fail_values).frames
    }

    /// Materialize virtual objects referenced by this encoded snapshot.
    pub fn materialize_virtuals(&self, fail_values: &[i64]) -> Vec<MaterializedVirtual> {
        let layout = self.decode_layout();
        ResumeData::materialize_virtuals_from_infos(&layout.virtuals, fail_values)
    }

    /// Resolve pending heap writes referenced by this encoded snapshot.
    pub fn resolve_pending_field_writes(
        &self,
        fail_values: &[i64],
    ) -> Vec<ResolvedPendingFieldWrite> {
        let layout = self.decode_layout();
        ResumeData::resolve_pending_field_writes(&layout.pending_fields, fail_values)
    }
}

/// Metainterp-side extension trait for `ResumeData` (which lives in
/// `majit-backend::resume_value` after the Phase C-1 cascade).
///
/// All inherent methods on `ResumeData` move here as trait methods
/// (with default impls) — the trait sits in metainterp because the
/// methods reference metainterp-specific types (`EncodedResumeData`,
/// `DecodedResumeLayout`, `ReconstructedState`, `ReconstructedFrame`,
/// `MaterializedVirtual`, `MaterializedValue`, `ReconstructedValue`,
/// `ResolvedPendingFieldWrite`) and the orphan rule forbids inherent
/// impls on a foreign type.
///
/// Callers reach the methods through `use ResumeDataExt;` plus the
/// usual receiver syntax (`data.encode()`, `data.reconstruct_state(...)`,
/// etc.) or the static-method syntax (`ResumeData::simple(...)`,
/// `ResumeData::resolve_field_source(...)`) — both work for trait
/// associated items as long as the trait is in scope.
pub trait ResumeDataExt {
    /// Create a simple ResumeData for a single-frame trace.
    fn simple(pc: u64, num_slots: usize) -> Self
    where
        Self: Sized;

    /// Encode this resume snapshot into a compact RPython-style numbering.
    fn encode(&self) -> EncodedResumeData;

    /// Decode the encoded resume snapshot back into a layout summary.
    fn decode_layout(&self) -> DecodedResumeLayout;

    /// Reconstruct the full resume state from fail_args data.
    fn reconstruct_state(&self, fail_values: &[i64]) -> ReconstructedState;

    /// Reconstruct frame slots from fail_args data.
    fn reconstruct(&self, fail_values: &[i64]) -> Vec<ReconstructedFrame>;

    /// Materialize virtual objects from resume data.
    fn materialize_virtuals(&self, fail_values: &[i64]) -> Vec<MaterializedVirtual>;

    fn materialize_virtuals_from_infos(
        virtuals: &[VirtualInfo],
        fail_values: &[i64],
    ) -> Vec<MaterializedVirtual>;

    /// Resolve pending heap writes into concrete values.
    fn resolve_pending_field_writes(
        pending_fields: &[PendingFieldInfo],
        fail_values: &[i64],
    ) -> Vec<ResolvedPendingFieldWrite>;

    /// Resolve a single VirtualFieldSource to a concrete i64 value.
    fn resolve_field_source(source: &VirtualFieldSource, fail_values: &[i64]) -> i64;

    fn resolve_materialized_source(
        source: &VirtualFieldSource,
        fail_values: &[i64],
    ) -> MaterializedValue;

    /// Resolve a single frame-slot source into a reconstructed value.
    fn resolve_frame_slot_source(
        source: &FrameSlotSource,
        fail_values: &[i64],
    ) -> ReconstructedValue;
}

impl ResumeDataExt for ResumeData {
    fn simple(pc: u64, num_slots: usize) -> Self {
        let slot_map: Vec<FrameSlotSource> = (0..num_slots).map(FrameSlotSource::FailArg).collect();
        ResumeData {
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            frames: vec![FrameInfo {
                jitcode_index: 0,
                pc,
                slot_map,
            }],
            virtuals: Vec::new(),
            pending_fields: Vec::new(),
        }
    }

    fn encode(&self) -> EncodedResumeData {
        EncodedResumeData::from_semantic(
            &self.vable_array,
            &self.vref_array,
            &self.frames,
            &self.virtuals,
            &self.pending_fields,
        )
    }

    fn decode_layout(&self) -> DecodedResumeLayout {
        self.encode().decode_layout()
    }

    fn reconstruct_state(&self, fail_values: &[i64]) -> ReconstructedState {
        let decoded = self.decode_layout();
        let materialized_virtuals = <ResumeData as ResumeDataExt>::materialize_virtuals_from_infos(
            &decoded.virtuals,
            fail_values,
        );
        let frames = decoded
            .frames
            .iter()
            .map(|frame| {
                let values = frame
                    .slot_map
                    .iter()
                    .map(|slot| {
                        <ResumeData as ResumeDataExt>::resolve_frame_slot_source(slot, fail_values)
                    })
                    .collect();
                ReconstructedFrame {
                    trace_id: None,
                    header_pc: None,
                    source_guard: None,
                    pc: frame.pc,
                    jitcode_index: frame.jitcode_index,
                    slot_types: None,
                    values,
                }
            })
            .collect();
        ReconstructedState {
            frames,
            virtuals: materialized_virtuals,
            pending_fields: <ResumeData as ResumeDataExt>::resolve_pending_field_writes(
                &decoded.pending_fields,
                fail_values,
            ),
        }
    }

    fn reconstruct(&self, fail_values: &[i64]) -> Vec<ReconstructedFrame> {
        self.reconstruct_state(fail_values).frames
    }

    fn materialize_virtuals(&self, fail_values: &[i64]) -> Vec<MaterializedVirtual> {
        let decoded = self.decode_layout();
        <ResumeData as ResumeDataExt>::materialize_virtuals_from_infos(
            &decoded.virtuals,
            fail_values,
        )
    }

    fn materialize_virtuals_from_infos(
        virtuals: &[VirtualInfo],
        fail_values: &[i64],
    ) -> Vec<MaterializedVirtual> {
        let mut result = Vec::with_capacity(virtuals.len());
        for vinfo in virtuals {
            result.push(MaterializedVirtual::from_info(vinfo));
        }
        for (i, vinfo) in virtuals.iter().enumerate() {
            result[i].resolve_fields(vinfo, fail_values);
        }
        result
    }

    fn resolve_pending_field_writes(
        pending_fields: &[PendingFieldInfo],
        fail_values: &[i64],
    ) -> Vec<ResolvedPendingFieldWrite> {
        pending_fields
            .iter()
            .map(|pending| ResolvedPendingFieldWrite {
                descr: pending.descr.clone(),
                target: <ResumeData as ResumeDataExt>::resolve_materialized_source(
                    &pending.target,
                    fail_values,
                ),
                value: <ResumeData as ResumeDataExt>::resolve_materialized_source(
                    &pending.value,
                    fail_values,
                ),
                item_index: pending.item_index,
            })
            .collect()
    }

    fn resolve_field_source(source: &VirtualFieldSource, fail_values: &[i64]) -> i64 {
        match <ResumeData as ResumeDataExt>::resolve_materialized_source(source, fail_values) {
            MaterializedValue::Value(value) => value,
            MaterializedValue::VirtualRef(_) => 0,
        }
    }

    fn resolve_materialized_source(
        source: &VirtualFieldSource,
        fail_values: &[i64],
    ) -> MaterializedValue {
        match source {
            ResumeValueSource::FailArg(idx) => {
                MaterializedValue::Value(fail_values.get(*idx).copied().unwrap_or(0))
            }
            ResumeValueSource::Constant(c) => MaterializedValue::Value(c.as_raw_i64()),
            ResumeValueSource::Virtual(idx) => MaterializedValue::VirtualRef(*idx),
            ResumeValueSource::Uninitialized | ResumeValueSource::Unavailable => {
                MaterializedValue::Value(0)
            }
        }
    }

    fn resolve_frame_slot_source(
        source: &FrameSlotSource,
        fail_values: &[i64],
    ) -> ReconstructedValue {
        match source {
            ResumeValueSource::FailArg(idx) => {
                ReconstructedValue::Value(fail_values.get(*idx).copied().unwrap_or(0))
            }
            ResumeValueSource::Constant(c) => ReconstructedValue::Value(c.as_raw_i64()),
            ResumeValueSource::Virtual(idx) => ReconstructedValue::Virtual(*idx),
            ResumeValueSource::Uninitialized => ReconstructedValue::Uninitialized,
            ResumeValueSource::Unavailable => ReconstructedValue::Unavailable,
        }
    }
}

/// A reconstructed interpreter frame from resume data.
#[derive(Debug, Clone)]
pub struct ReconstructedFrame {
    /// Compiled trace identifier for this frame, when known.
    pub trace_id: Option<u64>,
    /// Trace header pc associated with this frame, when known.
    pub header_pc: Option<u64>,
    /// Source guard this frame's trace is attached to, when known.
    pub source_guard: Option<(u64, u32)>,
    /// Program counter for this frame.
    pub pc: u64,
    /// resume.py:1051: jitcode index for CodeObject lookup.
    pub jitcode_index: i32,
    /// Typed layout of the reconstructed slots, when known.
    pub slot_types: Option<Vec<Type>>,
    /// Reconstructed values for each slot.
    pub values: Vec<ReconstructedValue>,
}

impl ReconstructedFrame {
    /// Lossy conversion: extract integer values, dropping virtual/unavailable info.
    pub fn lossy_values(&self) -> Vec<i64> {
        self.values
            .iter()
            .map(ReconstructedValue::lossy_i64)
            .collect()
    }
}

/// Full reconstructed state for a guard recovery.
#[derive(Debug, Clone)]
pub struct ReconstructedState {
    /// Reconstructed interpreter frames, outermost first.
    pub frames: Vec<ReconstructedFrame>,
    /// Materialized virtual objects referenced by frame slots.
    pub virtuals: Vec<MaterializedVirtual>,
    /// Deferred heap writes that the interpreter must replay after reconstruction.
    pub pending_fields: Vec<ResolvedPendingFieldWrite>,
}

/// Reconstructed slot value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReconstructedValue {
    /// Concrete raw value, including ints, refs, and float bits.
    Value(i64),
    /// Reference to a materialized virtual in `ReconstructedState.virtuals`.
    Virtual(usize),
    /// Slot exists but remains uninitialized.
    Uninitialized,
    /// Slot is dead/unavailable at this guard.
    Unavailable,
}

impl ReconstructedValue {
    /// Lossy conversion used by the current integer-only compatibility layer.
    pub fn lossy_i64(&self) -> i64 {
        match self {
            ReconstructedValue::Value(value) => *value,
            ReconstructedValue::Virtual(_)
            | ReconstructedValue::Uninitialized
            | ReconstructedValue::Unavailable => 0,
        }
    }
}

/// A materialized virtual object, ready for the interpreter to allocate.
///
/// After a guard failure, virtual objects must be allocated on the heap
/// and their fields populated from the DeadFrame values. This struct
/// holds the resolved field values for a single virtual object.
///
/// Mirrors RPython's `_materialize_virtual()` in resume.py.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaterializedValue {
    Value(i64),
    VirtualRef(usize),
}

impl MaterializedValue {
    pub fn resolve_with_refs(&self, materialized_refs: &[Option<GcRef>]) -> Option<i64> {
        match self {
            MaterializedValue::Value(value) => Some(*value),
            MaterializedValue::VirtualRef(index) => materialized_refs
                .get(*index)
                .copied()
                .flatten()
                .map(|gc_ref| gc_ref.as_usize() as i64),
        }
    }
}

#[derive(Debug, Clone)]
pub enum MaterializedVirtual {
    /// Object with vtable — resume.py:612 VirtualInfo.
    /// Carries `descr` (resume.py:615 self.descr) so the deopt path can
    /// `allocate_with_vtable(descr=self.descr)` and replay fields generically,
    /// without special-casing the vtable at the JIT-state layer.
    Obj {
        /// SizeDescr for allocation (exposes vtable + obj_size).
        descr: Option<majit_ir::DescrRef>,
        type_id: u32,
        /// (field_descr_index, concrete_value).
        fields: Vec<(u32, MaterializedValue)>,
    },
    /// Plain struct — resume.py:628 VStructInfo.
    Struct {
        /// resume.py:631 self.typedescr.
        descr: Option<majit_ir::DescrRef>,
        type_id: u32,
        fields: Vec<(u32, MaterializedValue)>,
    },
    /// Array — resume.py:646 VArrayInfo*.
    Array {
        /// resume.py:646 self.arraydescr.
        descr: Option<majit_ir::DescrRef>,
        items: Vec<MaterializedValue>,
    },
    /// Array of structs — resume.py:739 VArrayStructInfo.
    ArrayStruct {
        /// resume.py:739 self.arraydescr.
        descr: Option<majit_ir::DescrRef>,
        /// Per-element: Vec<(field_index, value)>.
        elements: Vec<Vec<(u32, MaterializedValue)>>,
    },
    /// Raw buffer.
    RawBuffer {
        func: i64,
        size: usize,
        /// rawbuffer.py:14 stores offsets as RPython unbounded ints.
        offsets: Vec<i64>,
        descrs: Vec<majit_ir::ArrayDescrInfo>,
        values: Vec<MaterializedValue>,
    },
}

impl MaterializedVirtual {
    /// Create an empty shell from a VirtualInfo (forward-reference safe).
    fn from_info(info: &VirtualInfo) -> Self {
        match info {
            VirtualInfo::VirtualObj { descr, type_id, .. } => MaterializedVirtual::Obj {
                descr: descr.clone(),
                type_id: *type_id,
                fields: Vec::new(),
            },
            VirtualInfo::VStruct {
                typedescr, type_id, ..
            } => MaterializedVirtual::Struct {
                descr: typedescr.clone(),
                type_id: *type_id,
                fields: Vec::new(),
            },
            VirtualInfo::VArray {
                arraydescr,
                clear: _,
                items,
                ..
            } => MaterializedVirtual::Array {
                descr: arraydescr.clone(),
                items: vec![MaterializedValue::Value(0); items.len()],
            },
            VirtualInfo::VArrayStruct {
                arraydescr,
                fielddescrs: _,
                element_fields,
                ..
            } => MaterializedVirtual::ArrayStruct {
                descr: arraydescr.clone(),
                elements: vec![Vec::new(); element_fields.len()],
            },
            VirtualInfo::VRawBuffer {
                func,
                size,
                offsets,
                descrs,
                ..
            } => MaterializedVirtual::RawBuffer {
                func: *func,
                size: *size,
                offsets: offsets.clone(),
                descrs: descrs.clone(),
                values: vec![MaterializedValue::Value(0); offsets.len()],
            },
            VirtualInfo::VRawSlice { .. } => MaterializedVirtual::Struct {
                descr: None,
                type_id: 0,
                fields: Vec::new(),
            },
            // resume.py:763-870 VStr/VUni*Info — virtual string shells
            // reserved for future vstring.py port. Represented as struct
            // shells for now (zero fields) so the materializer doesn't
            // walk into them; actual allocate_string / string_setitem /
            // concat_strings / slice_string still live on the roadmap.
            VirtualInfo::VStrPlain { .. }
            | VirtualInfo::VStrConcat { .. }
            | VirtualInfo::VStrSlice { .. }
            | VirtualInfo::VUniPlain { .. }
            | VirtualInfo::VUniConcat { .. }
            | VirtualInfo::VUniSlice { .. } => MaterializedVirtual::Struct {
                descr: None,
                type_id: 0,
                fields: Vec::new(),
            },
        }
    }

    /// Resolve fields from fail_values.
    fn resolve_fields(&mut self, info: &VirtualInfo, fail_values: &[i64]) {
        match (self, info) {
            (
                MaterializedVirtual::Obj { fields, .. },
                VirtualInfo::VirtualObj {
                    fields: src_fields, ..
                },
            )
            | (
                MaterializedVirtual::Struct { fields, .. },
                VirtualInfo::VStruct {
                    fields: src_fields, ..
                },
            ) => {
                *fields = src_fields
                    .iter()
                    .map(|(idx, src)| {
                        (
                            *idx,
                            ResumeData::resolve_materialized_source(src, fail_values),
                        )
                    })
                    .collect();
            }
            (
                MaterializedVirtual::Array { items, .. },
                VirtualInfo::VArray {
                    items: src_items, ..
                },
            ) => {
                *items = src_items
                    .iter()
                    .map(|src| ResumeData::resolve_materialized_source(src, fail_values))
                    .collect();
            }
            (
                MaterializedVirtual::ArrayStruct { elements, .. },
                VirtualInfo::VArrayStruct {
                    element_fields: src_elems,
                    ..
                },
            ) => {
                *elements = src_elems
                    .iter()
                    .map(|elem_fields| {
                        elem_fields
                            .iter()
                            .map(|(idx, src)| {
                                (
                                    *idx,
                                    ResumeData::resolve_materialized_source(src, fail_values),
                                )
                            })
                            .collect()
                    })
                    .collect();
            }
            (
                MaterializedVirtual::RawBuffer { values, .. },
                VirtualInfo::VRawBuffer {
                    values: src_values, ..
                },
            ) => {
                *values = src_values
                    .iter()
                    .map(|src| ResumeData::resolve_materialized_source(src, fail_values))
                    .collect();
            }
            _ => {} // type mismatch — should not happen
        }
    }

    pub fn resolve_with_refs(
        &self,
        materialized_refs: &[Option<GcRef>],
    ) -> Option<MaterializedVirtual> {
        match self {
            MaterializedVirtual::Obj {
                descr,
                type_id,
                fields,
            } => Some(MaterializedVirtual::Obj {
                descr: descr.clone(),
                type_id: *type_id,
                fields: fields
                    .iter()
                    .map(|(idx, value)| {
                        Some((
                            *idx,
                            MaterializedValue::Value(value.resolve_with_refs(materialized_refs)?),
                        ))
                    })
                    .collect::<Option<Vec<_>>>()?,
            }),
            MaterializedVirtual::Struct {
                descr,
                type_id,
                fields,
            } => Some(MaterializedVirtual::Struct {
                descr: descr.clone(),
                type_id: *type_id,
                fields: fields
                    .iter()
                    .map(|(idx, value)| {
                        Some((
                            *idx,
                            MaterializedValue::Value(value.resolve_with_refs(materialized_refs)?),
                        ))
                    })
                    .collect::<Option<Vec<_>>>()?,
            }),
            MaterializedVirtual::Array { descr, items } => Some(MaterializedVirtual::Array {
                descr: descr.clone(),
                items: items
                    .iter()
                    .map(|value| {
                        Some(MaterializedValue::Value(
                            value.resolve_with_refs(materialized_refs)?,
                        ))
                    })
                    .collect::<Option<Vec<_>>>()?,
            }),
            MaterializedVirtual::ArrayStruct { descr, elements } => {
                Some(MaterializedVirtual::ArrayStruct {
                    descr: descr.clone(),
                    elements: elements
                        .iter()
                        .map(|fields| {
                            fields
                                .iter()
                                .map(|(idx, value)| {
                                    Some((
                                        *idx,
                                        MaterializedValue::Value(
                                            value.resolve_with_refs(materialized_refs)?,
                                        ),
                                    ))
                                })
                                .collect::<Option<Vec<_>>>()
                        })
                        .collect::<Option<Vec<_>>>()?,
                })
            }
            MaterializedVirtual::RawBuffer {
                func,
                size,
                offsets,
                descrs,
                values,
            } => Some(MaterializedVirtual::RawBuffer {
                func: *func,
                size: *size,
                offsets: offsets.clone(),
                descrs: descrs.clone(),
                values: values
                    .iter()
                    .map(|value| {
                        Some(MaterializedValue::Value(
                            value.resolve_with_refs(materialized_refs)?,
                        ))
                    })
                    .collect::<Option<Vec<_>>>()?,
            }),
        }
    }
}

/// Builder for constructing ResumeData during trace compilation.
///
/// resume.py:298-493 ResumeDataVirtualAdder.finish() is implemented
/// across two functions in majit:
/// - `store_final_boxes_in_guard` (mod.rs) — numbering + rd_numb/rd_consts
/// - `store_final_boxes_in_guard` (optimizer.rs) — virtual expansion + rd_virtuals
pub struct ResumeDataVirtualAdder {
    vable_array: Vec<ResumeValueSource>,
    vref_array: Vec<ResumeValueSource>,
    frames: Vec<FrameInfoBuilder>,
    virtuals: Vec<VirtualInfo>,
    pending_fields: Vec<PendingFieldInfo>,
}

struct FrameInfoBuilder {
    jitcode_index: i32,
    pc: u64,
    slot_map: Vec<FrameSlotSource>,
}

impl ResumeDataVirtualAdder {
    /// Create a new builder.
    pub fn new() -> Self {
        ResumeDataVirtualAdder {
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            frames: Vec::new(),
            virtuals: Vec::new(),
            pending_fields: Vec::new(),
        }
    }

    pub fn set_vable_array(&mut self, values: Vec<ResumeValueSource>) {
        self.vable_array = values;
    }

    /// Push a new frame onto the stack.
    /// resume.py:249-252: jitcode_index, pc per frame.
    pub fn push_frame(&mut self, jitcode_index: i32, pc: u64) {
        self.frames.push(FrameInfoBuilder {
            jitcode_index,
            pc,
            slot_map: Vec::new(),
        });
    }

    /// Map a slot in the current frame to a fail_arg index.
    pub fn map_slot(&mut self, slot_idx: usize, fail_arg_idx: usize) {
        self.set_slot_source(slot_idx, FrameSlotSource::FailArg(fail_arg_idx));
    }

    /// Set a slot in the current frame to a tagged source.
    pub fn set_slot_source(&mut self, slot_idx: usize, source: FrameSlotSource) {
        let frame = self.frames.last_mut().expect("no frame pushed");
        while frame.slot_map.len() <= slot_idx {
            frame.slot_map.push(FrameSlotSource::Unavailable);
        }
        frame.slot_map[slot_idx] = source;
    }

    /// Set a frame slot to a compile-time constant.
    pub fn set_slot_constant(&mut self, slot_idx: usize, constant: majit_ir::Const) {
        self.set_slot_source(slot_idx, FrameSlotSource::Constant(constant));
    }

    /// Set a frame slot to reference a virtual object.
    pub fn set_slot_virtual(&mut self, slot_idx: usize, virtual_idx: usize) {
        self.set_slot_source(slot_idx, FrameSlotSource::Virtual(virtual_idx));
    }

    /// Mark a frame slot as present but uninitialized.
    pub fn set_slot_uninitialized(&mut self, slot_idx: usize) {
        self.set_slot_source(slot_idx, FrameSlotSource::Uninitialized);
    }

    /// Mark a frame slot as dead/unavailable.
    pub fn set_slot_unavailable(&mut self, slot_idx: usize) {
        self.set_slot_source(slot_idx, FrameSlotSource::Unavailable);
    }

    /// Add a virtual object description. Returns the index in the virtuals array.
    pub fn add_virtual(&mut self, info: VirtualInfo) -> usize {
        let idx = self.virtuals.len();
        self.virtuals.push(info);
        idx
    }

    /// Convenience: add a virtual object (NEW_WITH_VTABLE).
    pub fn add_virtual_obj(
        &mut self,
        descr: Option<majit_ir::DescrRef>,
        type_id: u32,
        known_class: Option<i64>,
        fields: Vec<(u32, VirtualFieldSource)>,
        fielddescrs: Vec<majit_ir::FieldDescrInfo>,
        descr_size: usize,
    ) -> usize {
        self.add_virtual(VirtualInfo::VirtualObj {
            descr,
            type_id,
            known_class,
            fields,
            fielddescrs,
            descr_size,
        })
    }

    /// Convenience: add a virtual struct (NEW).
    pub fn add_virtual_struct(
        &mut self,
        typedescr: Option<majit_ir::DescrRef>,
        type_id: u32,
        fields: Vec<(u32, VirtualFieldSource)>,
        fielddescrs: Vec<majit_ir::FieldDescrInfo>,
        descr_size: usize,
    ) -> usize {
        self.add_virtual(VirtualInfo::VStruct {
            typedescr,
            type_id,
            fields,
            fielddescrs,
            descr_size,
        })
    }

    /// Convenience: add a virtual array (NEW_ARRAY).
    pub fn add_virtual_array(
        &mut self,
        arraydescr: Option<majit_ir::DescrRef>,
        clear: bool,
        items: Vec<VirtualFieldSource>,
    ) -> usize {
        self.add_virtual(VirtualInfo::VArray {
            arraydescr,
            clear,
            items,
        })
    }

    /// resume.py:332: visit_varraystruct(arraydescr, size, fielddescrs)
    ///                 → VArrayStructInfo(arraydescr, size, fielddescrs)
    pub fn add_virtual_array_struct(
        &mut self,
        arraydescr: Option<majit_ir::DescrRef>,
        fielddescrs: Vec<majit_ir::DescrRef>,
        element_fields: Vec<Vec<(u32, VirtualFieldSource)>>,
    ) -> usize {
        self.add_virtual(VirtualInfo::VArrayStruct {
            arraydescr,
            fielddescrs,
            element_fields,
        })
    }

    /// Convenience: add a virtual raw buffer.
    pub fn add_virtual_raw_buffer(
        &mut self,
        func: i64,
        size: usize,
        offsets: Vec<i64>,
        descrs: Vec<majit_ir::ArrayDescrInfo>,
        values: Vec<VirtualFieldSource>,
    ) -> usize {
        self.add_virtual(VirtualInfo::VRawBuffer {
            func,
            size,
            offsets,
            descrs,
            values,
        })
    }

    /// Add a deferred field write to replay on resume.
    ///
    /// `resume.py:88 PENDINGFIELDSTRUCT.lldescr` — RPython always
    /// captures a live descr off the originating SetfieldGc op.
    pub fn add_pending_field_write(
        &mut self,
        descr: Option<majit_ir::DescrRef>,
        target: ResumeValueSource,
        value: ResumeValueSource,
    ) {
        self.pending_fields.push(PendingFieldInfo {
            descr,
            target,
            value,
            item_index: None,
        });
    }

    /// Add a deferred array item write to replay on resume.
    pub fn add_pending_arrayitem_write(
        &mut self,
        descr: Option<majit_ir::DescrRef>,
        target: ResumeValueSource,
        item_index: usize,
        value: ResumeValueSource,
    ) {
        self.pending_fields.push(PendingFieldInfo {
            descr,
            target,
            value,
            item_index: Some(item_index),
        });
    }

    /// resume.py: visit_vrawslice(offset) — add virtual raw slice.
    pub fn add_virtual_raw_slice(&mut self, offset: i64, parent: VirtualFieldSource) -> usize {
        self.add_virtual(VirtualInfo::VRawSlice { offset, parent })
    }

    /// Build the final ResumeData.
    pub fn build(self) -> ResumeData {
        ResumeData {
            vable_array: self.vable_array,
            vref_array: self.vref_array,
            frames: self
                .frames
                .into_iter()
                .map(|f| FrameInfo {
                    jitcode_index: f.jitcode_index,
                    pc: f.pc,
                    slot_map: f.slot_map,
                })
                .collect(),
            virtuals: self.virtuals,
            pending_fields: self.pending_fields,
        }
    }
}

impl Default for ResumeDataVirtualAdder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Fail-arg compression ─────────────────────────────────────────────

/// Shared resume data storage that deduplicates common snapshot sections
/// across multiple guards in the same trace.
///
/// RPython's `ResumeDataLoopMemo` shares constant pools and frame sections
/// across guards. We use a shared `ResumeEncoder` state so that the same
/// large constant only appears once in the pool.
/// RPython resume.py:142 ResumeDataLoopMemo.
/// Shared constant pool + box numbering cache across all guards in a loop.
///
/// NOTE: RPython's ResumeDataLoopMemo also stores `metainterp_sd` and `cpu`
/// (for box allocation during rebuild). pyre doesn't need these: the BoxEnv
/// trait provides box access. The canonical decoder lives in
/// `majit_ir::resumedata::rebuild_from_numbering`.
/// RPython's nvirtuals/nvholes/nvreused stats are kept for monitoring.
pub struct ResumeDataLoopMemo {
    /// resume.py:147 — shared constant pool.
    /// RPython stores Const objects (with type INT/REF/FLOAT).
    /// We store (value, type) pairs to preserve type information.
    consts: Vec<majit_ir::Const>,
    /// resume.py:148 — large integers (outside TAGINT range) → tagged const.
    large_ints: indexmap::IndexMap<i64, i16>,
    /// resume.py:149 — ref pointers → tagged const.
    refs: indexmap::IndexMap<i64, i16>,
    /// resume.py:147 self.consts — constant pool for encode_shared.
    /// Becomes storage.rd_consts (resume.py:467).
    ///
    /// resume.py:150-151 — cached box/virtual numbering.
    pub cached_boxes: indexmap::IndexMap<majit_ir::operand::Operand, i32>,
    pub cached_virtuals: indexmap::IndexMap<majit_ir::operand::Operand, i32>,
    /// resume.py:153-155 — statistics.
    pub nvirtuals: usize,
    pub nvholes: usize,
    pub nvreused: usize,
}

impl ResumeDataLoopMemo {
    pub fn new() -> Self {
        ResumeDataLoopMemo {
            consts: Vec::new(),
            large_ints: indexmap::IndexMap::new(),
            refs: indexmap::IndexMap::new(),
            cached_boxes: indexmap::IndexMap::new(),
            cached_virtuals: indexmap::IndexMap::new(),
            nvirtuals: 0,
            nvholes: 0,
            nvreused: 0,
        }
    }

    /// resume.py:199-226 `_number_boxes` + resume.py:209 `getconst` parity.
    ///
    /// Encode one `ResumeValueSource` into the i64 tagged form written to
    /// `rd_numb`. `liveboxes` / `box_map` track compact TAGBOX numbering
    /// identical to RPython's `numb_state.liveboxes` dict.
    ///
    /// Pushes Ref/Float/large-Int constants through `getconst_i64`, which
    /// shares `self.consts` with the i16 `getconst` path — so there is
    /// exactly one pool per memo (RPython parity: `self.consts: list[Const]`).
    pub fn encode_tagged_source(
        &mut self,
        source: &ResumeValueSource,
        liveboxes: &mut Vec<usize>,
        box_map: &mut indexmap::IndexMap<usize, usize>,
    ) -> i64 {
        match source {
            // resume.py:214-224: new box → liveboxes[box] = tag(num_boxes, TAGBOX)
            ResumeValueSource::FailArg(index) => {
                let compact = *box_map.entry(*index).or_insert_with(|| {
                    let n = liveboxes.len();
                    liveboxes.push(*index);
                    n
                });
                tag_i64(encode_len(compact), TAGBOX)
            }
            // resume.py:209: isinstance(box, Const) → self.getconst(box).
            ResumeValueSource::Constant(c) => self.getconst_i64(c),
            // resume.py:219-221: virtual → tag(num_virtuals, TAGVIRTUAL)
            ResumeValueSource::Virtual(index) => tag_i64(encode_len(*index), TAGVIRTUAL),
            ResumeValueSource::Uninitialized => tag_i64(ENCODED_UNINITIALIZED, TAGCONST),
            ResumeValueSource::Unavailable => tag_i64(ENCODED_UNAVAILABLE, TAGCONST),
        }
    }

    /// resume.py:157-183 getconst(const) — tag a constant value.
    /// Unified entry point matching RPython's getconst(const) which
    /// dispatches on const.type (INT, REF, FLOAT).
    pub fn getconst(&mut self, val: i64, tp: majit_ir::Type) -> i16 {
        match tp {
            majit_ir::Type::Int => self.getconst_int(val),
            majit_ir::Type::Ref => self.getconst_ref(val),
            majit_ir::Type::Float => self.getconst_float(val),
            majit_ir::Type::Void => self.newconst(val, tp),
        }
    }

    /// resume.py:158-172 getconst for INT type.
    pub fn getconst_int(&mut self, val: i64) -> i16 {
        // Try inline TAGINT (-8191..8190 in RPython's i16 range).
        let shifted = val >> 13;
        if shifted == 0 || shifted == -1 {
            return ((val << 2) | TAGINT as i64) as i16;
        }
        // Large int: check cache.
        if let Some(&tagged) = self.large_ints.get(&val) {
            return tagged;
        }
        let tagged = self.newconst(val, majit_ir::Type::Int);
        self.large_ints.insert(val, tagged);
        tagged
    }

    /// resume.py:173-182 getconst for REF type.
    pub fn getconst_ref(&mut self, val: i64) -> i16 {
        if val == 0 {
            return NULLREF;
        }
        if let Some(&tagged) = self.refs.get(&val) {
            return tagged;
        }
        let tagged = self.newconst(val, majit_ir::Type::Ref);
        self.refs.insert(val, tagged);
        tagged
    }

    /// resume.py:183 getconst fallback for FLOAT type.
    pub fn getconst_float(&mut self, val: i64) -> i16 {
        // FLOAT constants always go to the pool (no inline encoding).
        // RPython: return self._newconst(const)
        self.newconst(val, majit_ir::Type::Float)
    }

    /// resume.py:185 _newconst — add to consts pool, return TAGCONST-tagged.
    fn newconst(&mut self, val: i64, tp: majit_ir::Type) -> i16 {
        let index = self.consts.len() as i32 + TAG_CONST_OFFSET;
        self.consts.push(majit_ir::Const::from_raw_i64(val, tp));
        ((index << 2) | TAGCONST as i32) as i16
    }

    /// resume.py:161-188 getconst — i64-sized variant used by the rd_numb
    /// encoder (`encode_shared`). Shares the pool (`self.consts`) with the
    /// i16 variant (`getconst`) so there is exactly one `rd_consts` per
    /// memo, matching RPython's single `self.consts: list[Const]`.
    fn getconst_i64(&mut self, c: &majit_ir::Const) -> i64 {
        match c {
            // resume.py:163-167: try tag(val, TAGINT).
            majit_ir::Const::Int(value) if can_inline_tagged(*value) => tag_i64(*value, TAGINT),
            majit_ir::Const::Int(value) => {
                // resume.py:168-172 large int.
                if let Some(&tagged_i16) = self.large_ints.get(value) {
                    let (num, _) = untag(tagged_i16);
                    return tag_i64(encode_len((num - TAG_CONST_OFFSET) as usize), TAGCONST);
                }
                let index = self.consts.len();
                self.consts.push(majit_ir::Const::Int(*value));
                // Also publish through the i16 cache so that a later
                // `getconst_int(value)` returns the same pool slot
                // (resume.py:171 self.large_ints[val] = tagged).
                let tagged_i16 =
                    ((((index as i32) + TAG_CONST_OFFSET) << 2) | TAGCONST as i32) as i16;
                self.large_ints.insert(*value, tagged_i16);
                tag_i64(encode_len(index), TAGCONST)
            }
            majit_ir::Const::Ref(gcref) => {
                // resume.py:174-176 val = 0 → NULLREF sentinel (no pool
                // entry allocated). `NULLREF = tag(-1, TAGCONST)` —
                // encoder emits `tag_i64(-1, TAGCONST)` and the
                // matching decoder in `decode_box` recognizes
                // `ENCODED_NULLREF` before the positive-index branch.
                let raw = gcref.as_usize() as i64;
                if raw == 0 {
                    return tag_i64(ENCODED_NULLREF, TAGCONST);
                }
                if let Some(&tagged_i16) = self.refs.get(&raw) {
                    let (num, _) = untag(tagged_i16);
                    return tag_i64(encode_len((num - TAG_CONST_OFFSET) as usize), TAGCONST);
                }
                let index = self.consts.len();
                self.consts.push(majit_ir::Const::Ref(*gcref));
                let tagged_i16 =
                    ((((index as i32) + TAG_CONST_OFFSET) << 2) | TAGCONST as i32) as i16;
                self.refs.insert(raw, tagged_i16);
                tag_i64(encode_len(index), TAGCONST)
            }
            majit_ir::Const::Float(v) => {
                // resume.py:183 _newconst (no dedup for floats in RPython).
                let index = self.consts.len();
                self.consts.push(majit_ir::Const::Float(*v));
                tag_i64(encode_len(index), TAGCONST)
            }
        }
    }

    /// resume.py:261-262 num_cached_boxes — length of the box dedup cache.
    pub fn num_cached_boxes(&self) -> usize {
        self.cached_boxes.len()
    }

    /// resume.py:275-276 num_cached_virtuals — length of the virtual dedup cache.
    pub fn num_cached_virtuals(&self) -> usize {
        self.cached_virtuals.len()
    }

    /// resume.py:264 assign_number_to_box — returns a negative number.
    /// resume.py:264-273 assign_number_to_box(box, boxes).
    ///
    /// RPython version mutates `boxes` list:
    /// - cached: `boxes[-num - 1] = box`
    /// - new: `boxes.append(box); num = -len(boxes)`
    pub fn assign_number_to_box(
        &mut self,
        b: &majit_ir::operand::Operand,
        boxes: &mut Vec<OpRef>,
    ) -> i32 {
        if let Some(&num) = self.cached_boxes.get(b) {
            // resume.py:268: boxes[-num - 1] = box
            let idx = (-num - 1) as usize;
            if idx < boxes.len() {
                boxes[idx] = b.to_opref();
            }
            return num;
        }
        // resume.py:270-271: boxes.append(box); num = -len(boxes)
        boxes.push(b.to_opref());
        let num = -(boxes.len() as i32);
        self.cached_boxes.insert(b.clone(), num);
        num
    }

    /// resume.py:264-273 variant for `_number_virtuals`: boxes is `Vec<Option<OpRef>>`.
    /// RPython's `new_liveboxes = [None] * memo.num_cached_boxes()`.
    pub fn assign_number_to_box_opt(
        &mut self,
        b: &majit_ir::operand::Operand,
        boxes: &mut Vec<Option<OpRef>>,
    ) -> i32 {
        if let Some(&num) = self.cached_boxes.get(b) {
            let idx = (-num - 1) as usize;
            if idx < boxes.len() {
                boxes[idx] = Some(b.to_opref());
            }
            return num;
        }
        boxes.push(Some(b.to_opref()));
        let num = -(boxes.len() as i32);
        self.cached_boxes.insert(b.clone(), num);
        num
    }

    /// resume.py:278 assign_number_to_virtual — returns a negative number.
    pub fn assign_number_to_virtual(&mut self, b: &majit_ir::operand::Operand) -> i32 {
        if let Some(&num) = self.cached_virtuals.get(b) {
            return num;
        }
        // resume.py:283: num = self.cached_virtuals[box] = -len(self.cached_virtuals) - 1
        let num = -(self.num_cached_virtuals() as i32) - 1;
        self.cached_virtuals.insert(b.clone(), num);
        num
    }

    /// resume.py:290-293 update_counters(profiler).
    ///
    /// Roll the memo's cumulative NVIRTUALS / NVHOLES / NVREUSED into the
    /// caller-supplied profiler. Called from optimizeopt/optimizer.py:557
    /// once per trace compilation. The caller owns the profiler state;
    /// the memo only exposes its counters.
    pub fn update_counters(&self, profiler: &crate::jitprof::JitProfiler) {
        profiler.count(crate::pyjitpl::counters::NVIRTUALS, self.nvirtuals);
        profiler.count(crate::pyjitpl::counters::NVHOLES, self.nvholes);
        profiler.count(crate::pyjitpl::counters::NVREUSED, self.nvreused);
    }

    /// resume.py:286 clear_box_virtual_numbers.
    pub fn clear_box_virtual_numbers(&mut self) {
        self.cached_boxes.clear();
        self.cached_virtuals.clear();
    }

    /// Access the shared constant pool (value, type) pairs. Parity with
    /// RPython `memo.consts` list access (resume.py:147).
    pub fn consts(&self) -> &[majit_ir::Const] {
        &self.consts
    }

    /// Take ownership of the shared constant pool — used by single-shot
    /// encoders that discard the memo after encoding.
    pub fn take_consts(&mut self) -> Vec<majit_ir::Const> {
        std::mem::take(&mut self.consts)
    }

    /// resume.py:370-374 register_box — add a non-const, non-seen box to
    /// new_liveboxes with `UNASSIGNED`. The virtual classification is
    /// applied separately by `register_virtual_fields` (resume.py:359),
    /// which overwrites the entry with `UNASSIGNEDVIRTUAL` (or a
    /// pre-numbered tag from `liveboxes_from_env`). RPython's
    /// `register_box` does not consult `env.is_virtual` — see
    /// resume.py:370-374:
    ///     if (box is not None and not isinstance(box, Const)
    ///         and box not in self.liveboxes_from_env
    ///         and box not in self.liveboxes):
    ///         self.liveboxes[box] = UNASSIGNED
    fn register_box(
        &self,
        opref: majit_ir::OpRef,
        env: &dyn majit_ir::BoxEnv,
        liveboxes_from_env: &LiveboxMap,
        new_liveboxes: &mut LiveboxMap,
    ) {
        if opref.is_none() {
            return;
        }
        // resume.py:371 — constants are handled by _gettagged
        // (TAGCONST/TAGINT) and don't need livebox slots.
        if env.is_const(opref) {
            return;
        }
        // #160/S11: key by the canonical box (Rc::ptr_eq = PyPy `box is`).
        // `opref` is already replacement-walked by the caller, so re-walking
        // through get_box_replacement_operand is idempotent and yields the one
        // memoized Rc per logical box.
        let b = env.get_box_replacement_operand(opref);
        if liveboxes_from_env.contains_key(&b) || new_liveboxes.contains_key(&b) {
            return;
        }
        new_liveboxes.insert(b, UNASSIGNED);
    }

    /// resume.py:359-368 register_virtual_fields — stamp a virtual
    /// box's livebox tag and queue it for visitor_walk_recursive.
    ///
    /// `tagged = liveboxes_from_env.get(virtualbox, UNASSIGNEDVIRTUAL)`
    /// then `self.liveboxes[virtualbox] = tagged` unconditionally
    /// (overwriting any UNASSIGNED a prior `register_box` may have
    /// installed). The pre-numbered branch lets a virtual that was
    /// already TAGVIRTUAL'd in numbering keep its negative index.
    fn register_virtual_box(
        &self,
        virtualbox: majit_ir::OpRef,
        env: &dyn majit_ir::BoxEnv,
        liveboxes_from_env: &LiveboxMap,
        new_liveboxes: &mut LiveboxMap,
    ) {
        // #160/S11: key by the canonical box (Rc::ptr_eq). virtualbox is
        // replacement-walked by the caller; re-walking is idempotent.
        let b = env.get_box_replacement_operand(virtualbox);
        let tagged = liveboxes_from_env.get(&b).unwrap_or(UNASSIGNEDVIRTUAL);
        new_liveboxes.insert(b, tagged);
    }

    /// resume.py:454-509 `_number_virtuals(liveboxes, num_env_virtuals)`.
    ///
    /// Walks `new_liveboxes` in insertion order, converts UNASSIGNED /
    /// UNASSIGNEDVIRTUAL tags into real negative numbers via
    /// `assign_number_to_box` / `assign_number_to_virtual`, then
    /// materializes each virtual's fieldnums through
    /// `env.make_virtual_info` and stores them into the returned
    /// `rd_virtuals` Vec.
    ///
    /// `liveboxes` is extended in place with the freshly numbered boxes
    /// (resume.py:484 `liveboxes.extend(new_liveboxes)`). Returns
    /// `(rd_virtuals, nholes)` where nholes is used for the
    /// `_invalidation_needed` heuristic check by the caller.
    #[allow(clippy::too_many_arguments)]
    fn _number_virtuals(
        &mut self,
        liveboxes: &mut Vec<Option<majit_ir::OpRef>>,
        new_liveboxes: &mut LiveboxMap,
        virtual_fields: &indexmap::IndexMap<majit_ir::OpRef, majit_ir::VirtualFieldsInfo>,
        num_env_virtuals: usize,
        numb_state: &NumberingState,
        env: &dyn majit_ir::BoxEnv,
    ) -> (Vec<std::rc::Rc<majit_ir::RdVirtualInfo>>, usize) {
        // resume.py:460: new_liveboxes = [None] * memo.num_cached_boxes()
        let mut new_boxes_list: Vec<Option<majit_ir::OpRef>> = vec![None; self.num_cached_boxes()];
        let mut count = 0;
        // Iterate in insertion order (RPython dict iteration = insertion order).
        // resoperation.py:38 same_box parity: keys carry the typed OpRef
        // each entry was inserted with so virtual numbering preserves
        // `box.type` (history.py:220).
        // #160/S11: new_liveboxes.iter() yields the canonical box directly.
        let keys: Vec<(majit_ir::operand::Operand, i16)> = new_liveboxes.iter().collect();
        for (box_id, tagged) in keys {
            let (_, tagbits) = untag(tagged);
            if tagbits == TAGBOX {
                // resume.py:472-473: index = assign_number_to_box; liveboxes[box] = tag(index, TAGBOX)
                let index = self.assign_number_to_box_opt(&box_id, &mut new_boxes_list);
                if let Ok(t) = tag(index, TAGBOX) {
                    new_liveboxes.insert(box_id, t);
                }
                count += 1;
            } else {
                debug_assert_eq!(tagbits, TAGVIRTUAL);
                if tagged_eq(tagged, UNASSIGNEDVIRTUAL) {
                    // resume.py:479-480: index = assign_number_to_virtual; liveboxes[box] = tag(index, TAGVIRTUAL)
                    let index = self.assign_number_to_virtual(&box_id);
                    if let Ok(t) = tag(index, TAGVIRTUAL) {
                        new_liveboxes.insert(box_id, t);
                    }
                }
            }
        }
        // resume.py:483-484: new_liveboxes.reverse(); liveboxes.extend(new_liveboxes)
        new_boxes_list.reverse();
        for box_id in &new_boxes_list {
            liveboxes.push(*box_id);
        }
        let nholes = new_boxes_list.len() - count;

        // resume.py:488-506: create rd_virtuals
        // resume.py:500-501: make_virtual_info(info, fieldnums) via BoxEnv dispatch
        let mut rd_virtuals: Vec<std::rc::Rc<majit_ir::RdVirtualInfo>> = Vec::new();
        if !virtual_fields.is_empty() {
            // resume.py:491: length = num_env_virtuals + memo.num_cached_virtuals()
            let length = num_env_virtuals + self.num_cached_virtuals();
            // TODO: resume.py:492 uses `[None] * length` —
            // holes are represented as Python `None` in the list. Pyre's
            // descr-side `rd_virtuals: Arc<[Rc<RdVirtualInfo>]>` (compile.py:855
            // `_attrs_`) wraps the whole array in Option but the INNER
            // element type is not itself optional.  We use the
            // `RdVirtualInfo::Empty` sentinel variant to mark hole slots;
            // downstream consumers (compile.rs:644, compiler.rs:10952,
            // state.rs:3180, eval.rs:2680, resume.rs:1766) match `Empty`
            // and treat it as `None` equivalent.  Functional parity is
            // preserved; the structural divergence stays isolated to
            // this one type.
            rd_virtuals.resize(length, std::rc::Rc::new(majit_ir::RdVirtualInfo::Empty));
            // resume.py:493-494: memo.nvirtuals += length; memo.nvholes += length - len(vfieldboxes)
            self.nvirtuals += length;
            self.nvholes += length - virtual_fields.len();

            for (&opref_id, vf) in virtual_fields {
                // resume.py:496: num, _ = untag(self.liveboxes[virtualbox])
                // Check both numb_state.liveboxes (env virtuals) and
                // new_liveboxes (nested virtuals discovered via worklist).
                let opref = opref_id;
                // #160/S11: liveboxes is box-keyed; form the canonical box.
                let b = env.get_box_replacement_operand(opref);
                let tagged = numb_state
                    .liveboxes
                    .get(&b)
                    .or_else(|| new_liveboxes.get(&b))
                    .unwrap_or(UNASSIGNEDVIRTUAL);
                let (num, _) = untag(tagged);
                // RPython uses Python negative indexing: virtuals[-1] = virtuals[len-1].
                // Negative nums come from assign_number_to_virtual for nested virtuals.
                let num_idx = if num >= 0 {
                    num as usize
                } else {
                    (rd_virtuals.len() as i32 + num) as usize
                };
                if num_idx < rd_virtuals.len() {
                    // resume.py:500: fieldnums = [self._gettagged(box) for box in fieldboxes]
                    let fieldnums: Vec<i16> = vf
                        .field_oprefs
                        .iter()
                        .map(|&opref| {
                            // resume.py:560-568 _gettagged with pyre-specific fallback
                            // to cached_boxes/cached_virtuals when the local
                            // liveboxes entries are still UNASSIGNED/UNASSIGNEDVIRTUAL.
                            if opref.is_none() {
                                return UNINITIALIZED_TAG;
                            }
                            if env.is_const(opref) {
                                let (val, tp) = env.get_const(opref);
                                return self.getconst(val, tp);
                            }
                            // #160/S11: livebox / cached maps are box-keyed.
                            let b = env.get_box_replacement_operand(opref);
                            if let Some(t) = numb_state.liveboxes.get(&b) {
                                return t;
                            }
                            if let Some(t) = new_liveboxes.get(&b) {
                                if tagged_eq(t, UNASSIGNED) {
                                    if let Some(&num) = self.cached_boxes.get(&b) {
                                        return tag(num, TAGBOX).unwrap_or(UNASSIGNED);
                                    }
                                }
                                if tagged_eq(t, UNASSIGNEDVIRTUAL) {
                                    if let Some(&num) = self.cached_virtuals.get(&b) {
                                        return tag(num, TAGVIRTUAL).unwrap_or(UNASSIGNEDVIRTUAL);
                                    }
                                }
                                return t;
                            }
                            UNASSIGNED
                        })
                        .collect();
                    let reused = env.virtual_info_would_be_reused(opref_id, &fieldnums);
                    // resume.py:501: vinfo = self.make_virtual_info(info, fieldnums)
                    if let Some(rd_virt) = env.make_virtual_info(opref_id, fieldnums) {
                        if reused {
                            // resume.py:504-505: cached `_cached_vinfo` reused.
                            self.nvreused += 1;
                        }
                        rd_virtuals[num_idx] = rd_virt;
                    }
                }
            }
        }
        (rd_virtuals, nholes)
    }

    /// resume.py:520-558 `_add_pending_fields(pending_setfields)`.
    ///
    /// Tags the target/value boxes of each pending SETFIELD_GC/SETARRAYITEM_GC
    /// operation so the resume path can replay them against rehydrated
    /// struct instances. RPython decodes descr/opnum/itemindex from a
    /// `ResOperation` inline; pyre has already split the op into a
    /// `GuardPendingFieldEntry` by the time finish() is called, so this
    /// method only tags the target and value OpRefs in place.
    fn _add_pending_fields(
        &mut self,
        pending_setfields: &mut [majit_ir::GuardPendingFieldEntry],
        env: &dyn majit_ir::BoxEnv,
        liveboxes_from_env: &LiveboxMap,
        new_liveboxes: &LiveboxMap,
    ) {
        for pf in pending_setfields.iter_mut() {
            let target = env.get_box_replacement(pf.target);
            let value = env.get_box_replacement(pf.value);
            // resume.py:548-549 num = self._gettagged(box); fieldnum = self._gettagged(fieldbox)
            pf.target_tagged = self._gettagged(target, env, liveboxes_from_env, new_liveboxes);
            pf.value_tagged = self._gettagged(value, env, liveboxes_from_env, new_liveboxes);
        }
    }

    /// resume.py:570-574 `_add_optimizer_sections(numb_state, liveboxes, liveboxes_from_env)`.
    ///
    /// Delegates to bridgeopt.py:63-122 `serialize_optimizer_knowledge(optimizer,
    /// numb_state, liveboxes, liveboxes_from_env, memo)`. Emits three
    /// serialized sections on every guard (RPython emits zeros when the
    /// optheap/optrewrite caches are empty; the deserializer relies on the
    /// sections always being present):
    ///
    /// 1. known-class bitfield per Ref livebox (bridgeopt.py:74-90)
    /// 2. heap field + array item triples (bridgeopt.py:92-108)
    /// 3. loopinvariant call results (bridgeopt.py:113-122)
    ///
    /// RPython's `memo` is `self`; `numb_state.liveboxes` plays the role of
    /// the caller's `liveboxes_from_env` (the dict-like live-set). Pyre
    /// additionally carries an explicit `new_liveboxes` map so the per-
    /// guard tagged numbers assigned during `_number_virtuals` line up with
    /// the optimizer_knowledge lookup.
    fn _add_optimizer_sections(
        &mut self,
        numb_state: &mut NumberingState,
        liveboxes: &[Option<majit_ir::OpRef>],
        new_liveboxes: &LiveboxMap,
        env: &dyn majit_ir::BoxEnv,
        optimizer_knowledge: Option<&OptimizerKnowledgeForResume>,
    ) {
        // resume.py:572-574: serialize_optimizer_knowledge(
        //     self.optimizer, numb_state, liveboxes, liveboxes_from_env, self.memo)
        crate::optimizeopt::bridgeopt::serialize_optimizer_knowledge(
            self,
            numb_state,
            liveboxes,
            new_liveboxes,
            env,
            optimizer_knowledge,
        );
    }

    /// resume.py:511-518 `_invalidation_needed(nliveboxes, nholes)`.
    ///
    /// Heuristic for when the shared memo's cached-box dedup should be
    /// flushed after a successful resume encoding:
    ///
    /// ```python
    /// def _invalidation_needed(self, nliveboxes, nholes):
    ///     failargs_limit = memo.metainterp_sd.options.failargs_limit
    ///     if nliveboxes > (failargs_limit // 2):
    ///         if nholes > nliveboxes // 3:
    ///             return True
    ///     return False
    /// ```
    ///
    /// pyre uses the IR's compile-time FAILARGS_LIMIT (majit_ir:value.rs:201)
    /// as the metainterp option isn't wired yet. Matches RPython semantics
    /// exactly: "lots of live boxes, many of them holes" → invalidate.
    fn _invalidation_needed(&self, nliveboxes: usize, nholes: usize) -> bool {
        // resume.py:514-517
        let failargs_limit = majit_ir::FAILARGS_LIMIT;
        if nliveboxes > failargs_limit / 2 && nholes > nliveboxes / 3 {
            return true;
        }
        false
    }

    /// resume.py:560-568 _gettagged — resolve an OpRef to its tagged number.
    /// Looks up in liveboxes_from_env first, then new_liveboxes, then constant.
    pub(crate) fn _gettagged(
        &mut self,
        opref: majit_ir::OpRef,
        env: &dyn majit_ir::BoxEnv,
        liveboxes_from_env: &LiveboxMap,
        new_liveboxes: &LiveboxMap,
    ) -> i16 {
        if opref.is_none() {
            return UNINITIALIZED_TAG;
        }
        // resume.py:563-564: isinstance(box, Const) → getconst
        if env.is_const(opref) {
            let (val, tp) = env.get_const(opref);
            return self.getconst(val, tp);
        }
        // #160/S11: key the livebox / cached maps by the canonical box
        // (Rc::ptr_eq). `opref` is already replacement-walked by the caller.
        let b = env.get_box_replacement_operand(opref);
        // resume.py:566-567: liveboxes_from_env → existing tag
        if let Some(tagged) = liveboxes_from_env.get(&b) {
            return tagged;
        }
        if let Some(tagged) = new_liveboxes.get(&b) {
            // Resolve UNASSIGNED to real cached number
            if tagged_eq(tagged, UNASSIGNED) {
                if let Some(&num) = self.cached_boxes.get(&b) {
                    return tag(num, TAGBOX).unwrap_or(UNASSIGNED);
                }
            }
            if tagged_eq(tagged, UNASSIGNEDVIRTUAL) {
                if let Some(&num) = self.cached_virtuals.get(&b) {
                    return tag(num, TAGVIRTUAL).unwrap_or(UNASSIGNEDVIRTUAL);
                }
            }
            return tagged;
        }
        UNASSIGNED
    }

    /// resume.py:192-226 _number_boxes — tag each box in a snapshot section.
    ///
    /// Exact port of RPython's `_number_boxes(self, iter, iterator, numb_state)`.
    ///
    /// `env` provides box access matching RPython's box operations:
    /// - `get_box_replacement(opref)` → forwarded OpRef (resume.py:202)
    /// - `is_const(opref)` → isinstance(box, Const) (resume.py:204)
    /// - `get_const(opref)` → (value, type) for constants
    /// - `get_type(opref)` → box.type ('i', 'r', 'f') (resume.py:211,214)
    /// - `is_virtual_ref(opref)` → getptrinfo(box).is_virtual() (resume.py:212-213)
    /// - `is_virtual_raw(opref)` → getrawptrinfo(box).is_virtual() (resume.py:215-216)
    /// resume.py:192-226 `_number_boxes` — tag each box in a snapshot section.
    pub fn _number_boxes(
        &mut self,
        boxes: &[SnapshotBox],
        numb_state: &mut NumberingState,
        env: &dyn BoxEnv,
    ) -> Result<(), TagOverflow> {
        for snapshot_box in boxes {
            let raw_opref = snapshot_box.opref();
            if raw_opref.is_none() {
                numb_state.append_short(NULLREF);
                continue;
            }
            let opref = env.get_box_replacement(raw_opref);
            if opref.is_none() {
                numb_state.append_short(NULLREF);
                continue;
            }
            // resume.py:204: isinstance(box, Const) → getconst
            if env.is_const(opref) {
                let (val, tp) = env.get_const(opref);
                let tagged = self.getconst(val, tp);
                numb_state.append_short(tagged);
                continue;
            }
            // #160/S11: key liveboxes by the canonical box (Rc::ptr_eq =
            // PyPy `box is`). `opref` is replacement-walked above and non-const
            // here (Const short-circuited via the is_const branch).
            let b = env.get_box_replacement_operand(opref);
            // resume.py:206-208: liveboxes
            if let Some(tagged) = numb_state.liveboxes.get(&b) {
                numb_state.append_short(tagged);
                continue;
            }
            // resume.py:201-212:
            //
            //     box = iter.get(...)
            //     box = box.get_box_replacement()
            //     ...
            //     if box.type == 'r':
            //
            // The type used for virtual classification is the replacement
            // box's type, not the original snapshot slot's fallback type.  A
            // snapshot slot can carry an Int fallback from tracing but forward
            // to a Ref virtual after optimization; keeping the stale fallback
            // would number that virtual as a TAGBOX and the subsequent
            // optimizer.py:681 fail-arg force would materialize it.
            let box_type = opref
                .ty()
                .or(snapshot_box.tp)
                .unwrap_or_else(|| env.get_type(opref));
            let is_virtual = match box_type {
                majit_ir::Type::Ref => env.is_virtual_ref(opref),
                majit_ir::Type::Int => env.is_virtual_raw(opref),
                _ => false,
            };
            let tagged = if is_virtual {
                let t = tag(numb_state.num_virtuals, TAGVIRTUAL)?;
                numb_state.num_virtuals += 1;
                t
            } else {
                // RPython Box.type parity: capture type alongside TAGBOX
                // assignment. This is the equivalent of Box.type being
                // intrinsic — the type is determined once at numbering time.
                //
                // Typed OpRef variants (resoperation.py:719-739
                // InputArg{Int,Ref,Float}, resoperation.py:564-638 *Op
                // mixins) carry the type intrinsically (variant tag IS
                // RPython Box class identity). The `livebox_types`
                // HashMap is a legacy side-table that must agree with
                // `opref.ty()`; a divergence would indicate an
                // encoder/decoder mismatch we want to fail-loud on
                // (epic #171 will retire the side-table).
                if let Some(intrinsic_tp) = opref.ty() {
                    debug_assert_eq!(
                        intrinsic_tp, box_type,
                        "livebox numbering: typed OpRef {:?} intrinsic type {:?} \
                         disagrees with snapshot/env type {:?}",
                        opref, intrinsic_tp, box_type
                    );
                }
                numb_state.livebox_types.insert(opref, box_type);
                let t = tag(numb_state.num_boxes, TAGBOX)?;
                numb_state.num_boxes += 1;
                t
            };
            numb_state.liveboxes.insert(b, tagged);
            numb_state.append_short(tagged);
        }
        Ok(())
    }

    /// resume.py:228-256 number() — serialize a guard's full snapshot.
    ///
    /// Output format (in NumberingState):
    /// ```text
    /// [0]  size (patched later)
    /// [1]  number of failargs (patched later)
    /// [2]  vable_array_length  (0 if no virtualizable)
    ///      [tagged boxes for vable_array]
    /// [n]  vref_array_length   (0 if no virtualrefs)
    ///      [tagged boxes for vref_array]
    /// [m]  frame0_pc frame0_slots...
    /// [m+] frame1_pc frame1_slots...
    /// ...
    /// ```
    ///
    /// `frames` is a list of (pc, fail_args_slice) for each frame.
    /// In pyre (single frame), this is typically one frame.
    /// resume.py:228-256 number() — serialize a guard's full snapshot.
    ///
    /// Exact port of RPython's `number(self, position, trace, ...)`.
    ///
    /// `snapshot` describes the guard's state:
    /// - `vable_array`: virtualizable field boxes
    /// - `vref_array`: virtualref pairs
    /// - `framestack`: list of (jitcode_index, pc, boxes) per frame
    ///
    /// `env` implements BoxEnv for box operations.
    ///
    /// Returns `Err(TagOverflow)` if any box index exceeds the tag range.
    /// RPython: raises TagOverflow → caller does compile.giveup().
    ///
    /// NOTE: Slot 1 (number of failargs) is left as 0 here.
    /// RPython patches it later in ResumeDataVirtualAdder.finish()
    /// (resume.py:433). Callers must call
    /// `numb_state.writer.patch(1, num_liveboxes)` after finish().
    pub fn number(
        &mut self,
        snapshot: &Snapshot,
        env: &dyn BoxEnv,
        minimum_virtualizable_size: i64,
    ) -> Result<NumberingState, TagOverflow> {
        let size_hint = snapshot.estimated_size();
        let mut numb_state = NumberingState::new(size_hint);

        // resume.py:231-232: patch later
        numb_state.append_int(0); // slot 0: size of resume section
        numb_state.append_int(0); // slot 1: number of failargs (patched by finish())

        // resume.py:236-239: if minimum_virtualizable_size != -1: the
        // virtualizable itself is one entry in the array too, so use '>'.
        if minimum_virtualizable_size != -1 {
            debug_assert!(
                snapshot.vable_array.len() as i64 > minimum_virtualizable_size,
                "vable_array length {} not > minimum_virtualizable_size {}",
                snapshot.vable_array.len(),
                minimum_virtualizable_size
            );
        }

        // resume.py:240-241 virtualizable array.
        //
        // Upstream shape is `virtualizable_boxes = read_boxes(...);`
        // `virtualizable_boxes.append(virtualizable_box)` (pyjitpl.py:3302-3306),
        // i.e. payload first, identity last. The snapshot already carries the
        // tracing-time Box identities in that order, so line-by-line parity is
        // to run the whole array through `_number_boxes()` unchanged.
        numb_state.append_int(snapshot.vable_array.len() as i64);
        self._number_boxes(&snapshot.vable_array, &mut numb_state, env)?;

        // resume.py:243-247: virtualref array
        let vref_len = snapshot.vref_array.len();
        debug_assert!(vref_len & 1 == 0, "vref_array length must be even");
        numb_state.append_int((vref_len >> 1) as i64);
        self._number_boxes(&snapshot.vref_array, &mut numb_state, env)?;

        // resume.py:249-253: frame chain.
        // Per-frame: jitcode_index, pc, [tagged_values...].
        // RPython uses jitcode.get_live_vars_info(pc) at decode time
        // to know how many tagged values each frame has.
        for frame in &snapshot.framestack {
            numb_state.append_int(frame.jitcode_index as i64);
            numb_state.append_int(frame.pc as i64);
            // Per-frame `jitcode_pc` word (after `pc`); see
            // `majit_ir::resumedata::NO_JITCODE_PC`.
            numb_state.append_int(frame.jitcode_pc as i64);
            // `PYRE_M369_RESUME_PC_AUDIT`: report frames whose `jitcode_pc` is
            // non-sentinel — the residual kept-stack cases that still block
            // collapsing this frame chain to the 2-word `(jitcode_index, pc)`
            // shape (resume.py:249-253).  `pc_substitutes` reports whether the
            // (flag-stripped) `pc` word already equals `jitcode_pc`, i.e.
            // whether `pc` alone would suffice at this frame.
            if crate::m369_resume_pc_audit_enabled()
                && frame.jitcode_pc != majit_ir::resumedata::NO_JITCODE_PC
            {
                let (py_pc, after_residual_call) = majit_ir::resumedata::decode_resume_pc(frame.pc);
                eprintln!(
                    "[m369-audit] residual jitcode_pc frame: jitcode_index={} \
                     pc_raw={} pc={} after_residual_call={} jitcode_pc={} \
                     pc_substitutes={}",
                    frame.jitcode_index,
                    frame.pc,
                    py_pc,
                    after_residual_call,
                    frame.jitcode_pc,
                    py_pc == frame.jitcode_pc,
                );
            }
            // Inverse probe under the same audit: frames still encoded with
            // the sentinel word — the writers the #73 S5 twin carry has not
            // reached yet (attribution for the decode-side translation
            // fallback).
            if crate::m369_resume_pc_audit_enabled()
                && frame.jitcode_pc == majit_ir::resumedata::NO_JITCODE_PC
            {
                let (py_pc, after_residual_call) = majit_ir::resumedata::decode_resume_pc(frame.pc);
                eprintln!(
                    "[m369-audit] sentinel jitcode_pc frame: jitcode_index={} \
                     pc_raw={} pc={} after_residual_call={} nframes={} frame_pos={}",
                    frame.jitcode_index,
                    frame.pc,
                    py_pc,
                    after_residual_call,
                    snapshot.framestack.len(),
                    snapshot
                        .framestack
                        .iter()
                        .position(|f| std::ptr::eq(f, frame))
                        .unwrap_or(usize::MAX),
                );
            }
            self._number_boxes(&frame.boxes, &mut numb_state, env)?;
        }

        // resume.py:254: patch total size
        numb_state.patch_current_size(0);

        Ok(numb_state)
    }

    /// resume.py:389-452 ResumeDataVirtualAdder.finish() — exact port.
    ///
    /// `numb_state`: output of `number()`
    /// `env`: BoxEnv for resolving box properties (constants, types).
    ///   Virtual fields are discovered via `env.get_virtual_fields()`,
    ///   matching RPython's `visitor_walk_recursive` callback pattern.
    /// `pending_setfields`: resume.py:428-442 register_box + visitor_walk_recursive,
    ///   resume.py:520-558 _add_pending_fields tagging.
    ///   target_tagged/value_tagged are filled in-place.
    /// `optimizer_knowledge`: bridgeopt.py:63 serialize_optimizer_knowledge.
    ///   Heap field triples and known-class info for bridge compilation.
    ///
    /// Returns `(rd_numb, rd_consts, rd_virtuals, liveboxes, livebox_types)`.
    /// `livebox_types` maps typed OpRef → Type, captured at numbering time
    /// (RPython Box.type parity).
    pub fn finish(
        &mut self,
        mut numb_state: NumberingState,
        env: &dyn majit_ir::BoxEnv,
        pending_setfields: &mut [majit_ir::GuardPendingFieldEntry],
        optimizer_knowledge: Option<&OptimizerKnowledgeForResume>,
    ) -> (
        Vec<u8>,
        Vec<majit_ir::Const>,
        Vec<std::rc::Rc<majit_ir::RdVirtualInfo>>,
        Vec<majit_ir::OpRef>,
        LiveboxTypeMap,
    ) {
        let num_env_virtuals = numb_state.num_virtuals;

        // resume.py:410-426: split liveboxes_from_env into TAGBOX/TAGVIRTUAL
        let mut liveboxes: Vec<Option<majit_ir::OpRef>> = vec![None; numb_state.num_boxes as usize];

        // resume.py:413: self.vfieldboxes collected by virtual walk
        // resume.py:408: self.liveboxes — newly discovered boxes from field walk
        let mut new_liveboxes = LiveboxMap::new();

        // resume.py:414-426: iterate liveboxes_from_env, discover virtual
        // fields. RPython walks the dict in insertion order; pyre's
        // `LiveboxMap` is built on `IndexMap` (resume.rs:98) so the
        // `.iter()` sequence already matches that order, which the
        // virtual worklist drain below relies on for byte-identical
        // visitor_walk_recursive sequencing. Sorting by tag would
        // observably re-order virtuals across builds.
        //
        // TAGBOX placement at `liveboxes[i] = opref` uses the
        // tag-derived index (resume.py:417), so it is iteration-order-
        // invariant; only the TAGVIRTUAL worklist push order matters.
        //
        // resoperation.py:38 same_box parity: iter() yields the typed
        // OpRef each entry was inserted with, so consumers can read
        // `box.type` (history.py:220) directly via `opref.ty()`.

        // Collect virtual fields discovered via env.get_virtual_fields()
        // (resume.py:419-426 visitor_walk_recursive pattern). Keyed by
        // typed OpRef so the same_box (resoperation.py:38) identity is
        // preserved end-to-end through the worklist drain.
        let mut virtual_fields: indexmap::IndexMap<majit_ir::OpRef, majit_ir::VirtualFieldsInfo> =
            indexmap::IndexMap::new();

        // resume.py:419-426: visitor_walk_recursive — worklist for nested virtuals.
        let mut virtual_worklist: Vec<majit_ir::OpRef> = Vec::new();

        for (b, tagged) in numb_state.liveboxes.iter() {
            // #160/S11: liveboxes is now box-keyed; the serialized livebox
            // vector + virtual worklist stay OpRef-based (backend positions).
            let opref = b.to_opref();
            let (i, tagbits) = untag(tagged);
            if tagbits == TAGBOX {
                if (i as usize) < liveboxes.len() {
                    liveboxes[i as usize] = Some(opref);
                }
            } else {
                debug_assert_eq!(tagbits, TAGVIRTUAL);
                virtual_worklist.push(opref);
            }
        }

        // Worklist-based recursive virtual discovery (RPython visitor_walk_recursive).
        // Process each virtual: register its field boxes, and if any field is
        // itself a virtual, add it to the worklist for later processing.
        let mut worklist_idx = 0;
        while worklist_idx < virtual_worklist.len() {
            let opref_id = virtual_worklist[worklist_idx];
            worklist_idx += 1;

            if virtual_fields.contains_key(&opref_id) {
                continue; // already_seen_virtual
            }
            let vf_result = env.get_virtual_fields(opref_id);
            if let Some(vf) = vf_result {
                // resume.py:362-368: register_virtual_fields
                for &field_opref in &vf.field_oprefs {
                    // resume.py:370-374: register_box (UNASSIGNED for
                    // non-virtual fields).
                    self.register_box(field_opref, env, &numb_state.liveboxes, &mut new_liveboxes);
                    // resume.py:419-426 visitor_walk_recursive: if the
                    // field is a virtual, register_virtual_fields
                    // overwrites the UNASSIGNED stamp with the env-
                    // pre-numbered tag (or UNASSIGNEDVIRTUAL).
                    let resolved = env.get_box_replacement(field_opref);
                    if !resolved.is_none()
                        && !virtual_fields.contains_key(&resolved)
                        && (env.is_virtual_ref(resolved) || env.is_virtual_raw(resolved))
                    {
                        self.register_virtual_box(
                            resolved,
                            env,
                            &numb_state.liveboxes,
                            &mut new_liveboxes,
                        );
                        virtual_worklist.push(resolved);
                    }
                }
                virtual_fields.insert(opref_id, vf);
            }
        }

        // resume.py:428-442: process pending_setfields — register_box on
        // target and value, then visitor_walk_recursive on virtual fieldbox.
        for pf in pending_setfields.iter() {
            let box_opref = env.get_box_replacement(pf.target);
            let fieldbox = env.get_box_replacement(pf.value);
            // resume.py:438-439: self.register_box(box); self.register_box(fieldbox)
            self.register_box(box_opref, env, &numb_state.liveboxes, &mut new_liveboxes);
            self.register_box(fieldbox, env, &numb_state.liveboxes, &mut new_liveboxes);
            // resume.py:440-442 — info.visitor_walk_recursive requires
            // the fieldbox to be a virtual:
            //     info = getptrinfo(fieldbox)
            //     assert info is not None and info.is_virtual()
            // A non-virtual fieldbox in pending_setfields is an
            // invariant violation in the optheap producer side; the
            // previous silent skip masked it.
            let vf = env.get_virtual_fields(fieldbox).unwrap_or_else(|| {
                panic!(
                    "pending_setfields fieldbox {:?} (target={:?}) is not virtual \
                     (resume.py:441 assert info is not None and info.is_virtual())",
                    fieldbox, box_opref
                )
            });
            // resume.py:359 register_virtual_fields: stamp the virtual fieldbox
            // UNASSIGNEDVIRTUAL (overwriting the UNASSIGNED that register_box
            // installed above) so _number_virtuals numbers it as a virtual
            // rather than a livebox — otherwise it would be force-boxed.
            self.register_virtual_box(fieldbox, env, &numb_state.liveboxes, &mut new_liveboxes);
            for &field_opref in &vf.field_oprefs {
                self.register_box(field_opref, env, &numb_state.liveboxes, &mut new_liveboxes);
                let resolved = env.get_box_replacement(field_opref);
                if !resolved.is_none()
                    && !virtual_fields.contains_key(&resolved)
                    && (env.is_virtual_ref(resolved) || env.is_virtual_raw(resolved))
                {
                    self.register_virtual_box(
                        resolved,
                        env,
                        &numb_state.liveboxes,
                        &mut new_liveboxes,
                    );
                    virtual_worklist.push(resolved);
                }
            }
            virtual_fields.insert(fieldbox, vf);
        }

        // resume.py:440-442 parity: drain worklist for nested virtuals
        // discovered from pending_setfields. RPython's visitor_walk_recursive
        // recursively processes all levels; our worklist pattern resumes here.
        while worklist_idx < virtual_worklist.len() {
            let opref_id = virtual_worklist[worklist_idx];
            worklist_idx += 1;
            if virtual_fields.contains_key(&opref_id) {
                continue;
            }
            if let Some(vf) = env.get_virtual_fields(opref_id) {
                for &field_opref in &vf.field_oprefs {
                    self.register_box(field_opref, env, &numb_state.liveboxes, &mut new_liveboxes);
                    let resolved = env.get_box_replacement(field_opref);
                    if !resolved.is_none()
                        && !virtual_fields.contains_key(&resolved)
                        && (env.is_virtual_ref(resolved) || env.is_virtual_raw(resolved))
                    {
                        self.register_virtual_box(
                            resolved,
                            env,
                            &numb_state.liveboxes,
                            &mut new_liveboxes,
                        );
                        virtual_worklist.push(resolved);
                    }
                }
                virtual_fields.insert(opref_id, vf);
            }
        }

        // resume.py:454-509 self._number_virtuals(liveboxes, num_env_virtuals)
        let (rd_virtuals, nholes) = self._number_virtuals(
            &mut liveboxes,
            &mut new_liveboxes,
            &virtual_fields,
            num_env_virtuals as usize,
            &numb_state,
            env,
        );

        // resume.py:508-509: if self._invalidation_needed(...): memo.clear_box_virtual_numbers()
        if self._invalidation_needed(liveboxes.len(), nholes) {
            self.clear_box_virtual_numbers();
        }

        // resume.py:445 self._add_pending_fields(pending_setfields)
        self._add_pending_fields(
            pending_setfields,
            env,
            &numb_state.liveboxes,
            &new_liveboxes,
        );

        // resume.py:447: numb_state.patch(1, len(liveboxes))
        numb_state.writer.patch(1, liveboxes.len() as i32);

        // resume.py:449: self._add_optimizer_sections(numb_state, liveboxes, liveboxes_from_env)
        self._add_optimizer_sections(
            &mut numb_state,
            &liveboxes,
            &new_liveboxes,
            env,
            optimizer_knowledge,
        );

        // resume.py:450-451: storage.rd_numb, storage.rd_consts
        let rd_numb = numb_state.create_numbering();
        let rd_consts = self.consts.clone();

        // Resolve each livebox through the forwarding chain so the backend
        // sees the final concrete OpRef (not an optimizer-internal alias).
        //
        // `resume.py:finish` invariant: liveboxes contains ONLY non-Const
        // boxes — Const values are encoded inline via TAGCONST at numbering
        // time (`_number_boxes` classifies via `box.is_constant()` before
        // adding to liveboxes). Backend regalloc enforces the same upstream
        // contract (`pyre/regalloc.rs:453 !arg.is_constant()` mirrors
        // `regalloc.py:1204 assert not isinstance(arg, Const)`).
        //
        // The numbering pass that produced this `liveboxes` list already
        // satisfied that invariant. The re-walk below exists for boxes that
        // were further forwarded between numbering and finish (e.g.
        // `make_equal_to` writing a `Forwarded::Op`/`Const` redirect), so the
        // backend sees the final concrete position. It uses
        // get_box_replacement(not_const=True) parity and stops before a Const
        // target; Consts are represented by rd_numb TAGCONST, not backend
        // livebox slots.
        //
        // resume.py:412-417 + regalloc.py:1204: liveboxes contains ONLY
        // non-Const boxes — `_number_boxes` (resume.rs:3755-3826) classifies
        // Const via `is_const(opref)` → TAGCONST inline (lines 3773-3777)
        // before the box ever reaches liveboxes. PyPy `resume.py:finish` has
        // no post-numbering Const→hole step; the invariant is that liveboxes
        // entries stay non-Const through finish(). Hard-assert that
        // `get_box_replacement_not_const` does not produce a
        // constant-namespace OpRef
        // here — if the assert fires, a writer (e.g. a future
        // `make_constant` flip without paired numbering) is racing the
        // numbering snapshot, which would break rd_numb / liveboxes
        // alignment downstream.
        let ordered_liveboxes: Vec<majit_ir::OpRef> = liveboxes
            .into_iter()
            .map(|opt| {
                opt.map(|opref| {
                    let walked = env.get_box_replacement_not_const(opref);
                    debug_assert!(
                        !walked.is_constant(),
                        "resume.py:412-417 invariant: liveboxes entry walked to \
                         constant-namespace OpRef post-numbering ({opref:?} → {walked:?}); \
                         _number_boxes should have classified this as TAGCONST inline"
                    );
                    walked
                })
                .unwrap_or(majit_ir::OpRef::NONE)
            })
            .collect();

        // Merge livebox_types: numbering-time types + types for boxes
        // discovered during virtual field walking.
        let mut all_livebox_types = numb_state.livebox_types;
        for &opref in &ordered_liveboxes {
            if !opref.is_none() && !all_livebox_types.contains_key(&opref) {
                all_livebox_types.insert(opref, env.get_type(opref));
            }
        }
        (
            rd_numb,
            rd_consts,
            rd_virtuals,
            ordered_liveboxes,
            all_livebox_types,
        )
    }

    /// resume.py:452-468 finish (on ResumeDataVirtualAdder) — encode with shared pool.
    pub fn encode_shared(&mut self, rd: &ResumeData) -> EncodedResumeData {
        let mut rd_numb = Vec::new();
        // resume.py:138 compact TAGBOX numbering state.
        let mut liveboxes: Vec<usize> = Vec::new();
        let mut box_map: indexmap::IndexMap<usize, usize> = indexmap::IndexMap::new();

        // resume.py:234-235: reserve slots
        rd_numb.push(0); // [0] = items_resume_section
        rd_numb.push(0); // [1] = count
        rd_numb.push(encode_len(rd.vable_array.len()));
        for source in &rd.vable_array {
            let tagged = self.encode_tagged_source(source, &mut liveboxes, &mut box_map);
            rd_numb.push(tagged);
        }
        // resume.py:243-247: vref_array (pairs).
        assert!(
            rd.vref_array.len() % 2 == 0,
            "vref_array must have even length (pairs)"
        );
        rd_numb.push(encode_len(rd.vref_array.len() / 2));
        for source in &rd.vref_array {
            let tagged = self.encode_tagged_source(source, &mut liveboxes, &mut box_map);
            rd_numb.push(tagged);
        }

        // resume.py:249-253: per-frame: jitcode_index, pc, [tagged_values...].
        let mut frame_sizes = Vec::with_capacity(rd.frames.len());
        for frame in &rd.frames {
            rd_numb.push(frame.jitcode_index as i64);
            rd_numb.push(encode_u64(frame.pc));
            // Per-frame `jitcode_pc` word; sentinel on the test/embedder path.
            rd_numb.push(majit_ir::resumedata::NO_JITCODE_PC as i64);
            for source in &frame.slot_map {
                let tagged = self.encode_tagged_source(source, &mut liveboxes, &mut box_map);
                rd_numb.push(tagged);
            }
            frame_sizes.push(frame.slot_map.len());
        }

        let rd_virtuals = rd.virtuals.clone();

        // resume.py:412-418: register virtual field boxes.
        for vinfo in &rd_virtuals {
            for source in vinfo.field_sources() {
                if let ResumeValueSource::FailArg(index) = source {
                    box_map.entry(*index).or_insert_with(|| {
                        let n = liveboxes.len();
                        liveboxes.push(*index);
                        n
                    });
                }
            }
        }

        // resume.py:420-430: walk pending fields — register + encode.
        // Collect first, then encode — can't hold iter borrow and call
        // `&mut self` method in the same expression.
        let pending_fields_snapshot: Vec<_> = rd.pending_fields.iter().cloned().collect();
        let rd_pendingfields: Vec<_> = pending_fields_snapshot
            .into_iter()
            .map(|pending| EncodedPendingFieldWrite {
                // resume.py:547 lldescr = cast_instance_to_base_ptr(descr) —
                // the encoded form carries the descr itself, not a handle.
                descr: pending.descr.clone(),
                target: self.encode_tagged_source(&pending.target, &mut liveboxes, &mut box_map),
                value: self.encode_tagged_source(&pending.value, &mut liveboxes, &mut box_map),
                item_index: pending.item_index,
            })
            .collect();

        // resume.py:260 patch_current_size, resume.py:464 patch count
        rd_numb[0] = encode_len(rd_numb.len());
        rd_numb[1] = encode_len(liveboxes.len());

        EncodedResumeData {
            rd_numb,
            // resume.py:451 storage.rd_consts = self.memo.consts — single pool.
            rd_consts: self.consts.clone(),
            rd_pendingfields,
            rd_virtuals,
            liveboxes,
            frame_sizes,
        }
    }

    /// Number of entries in the shared constant pool.
    pub fn num_shared_consts(&self) -> usize {
        self.consts.len()
    }
}

impl Default for ResumeDataLoopMemo {
    fn default() -> Self {
        Self::new()
    }
}

/// resume.py: AbstractResumeDataReader — reads resume data to
/// reconstruct interpreter state after a guard failure.
///
/// Two concrete implementations in RPython:
/// - ResumeDataBoxReader: creates boxes (for blackhole interpreter)
/// - ResumeDataDirectReader: reads values directly (for fast path)
pub struct ResumeDataReader<'a> {
    /// The resume data to read from.
    resume_data: &'a ResumeData,
    /// Fail argument values from the guard failure.
    fail_values: &'a [i64],
    /// Materialized virtuals (lazily populated).
    virtuals: Vec<Option<i64>>,
}

impl<'a> ResumeDataReader<'a> {
    /// resume.py: AbstractResumeDataReader.__init__
    pub fn new(resume_data: &'a ResumeData, fail_values: &'a [i64]) -> Self {
        let num_virtuals = resume_data.virtuals.len();
        ResumeDataReader {
            resume_data,
            fail_values,
            virtuals: vec![None; num_virtuals],
        }
    }

    /// resume.py: _decode_box — decode a tagged value reference.
    pub fn decode_frame_slot(&self, source: &FrameSlotSource) -> i64 {
        self.decode_value(source)
    }

    /// Decode a ResumeValueSource to a concrete value.
    pub fn decode_value(&self, source: &ResumeValueSource) -> i64 {
        match source {
            ResumeValueSource::FailArg(idx) => self.fail_values.get(*idx).copied().unwrap_or(0),
            ResumeValueSource::Constant(c) => c.as_raw_i64(),
            ResumeValueSource::Virtual(vidx) => {
                self.virtuals.get(*vidx).copied().flatten().unwrap_or(0)
            }
            ResumeValueSource::Uninitialized | ResumeValueSource::Unavailable => 0,
        }
    }

    /// resume.py: consume_boxes — read all frame slots for one frame.
    pub fn read_frame_slots(&self, frame_idx: usize) -> Vec<i64> {
        if frame_idx >= self.resume_data.frames.len() {
            return vec![];
        }
        let frame = &self.resume_data.frames[frame_idx];
        frame
            .slot_map
            .iter()
            .map(|source| self.decode_frame_slot(source))
            .collect()
    }

    /// Number of frames in the resume data.
    pub fn num_frames(&self) -> usize {
        self.resume_data.frames.len()
    }

    /// PC for a given frame.
    pub fn frame_pc(&self, frame_idx: usize) -> u64 {
        self.resume_data
            .frames
            .get(frame_idx)
            .map(|f| f.pc)
            .unwrap_or(0)
    }
}

/// resume.py:576-728 VirtualInfo parity.
/// Describes a virtual object's fields for materialization.
/// RPython uses a class hierarchy (VirtualInfo, VStructInfo, VArrayInfoClear, etc.).
/// We use a single struct with tagged field values.
#[derive(Debug, Clone, Default)]
pub struct VirtualFieldValues {
    /// Descriptor (type/class) for the virtual object.
    pub descr: Option<majit_ir::DescrRef>,
    /// Known class pointer (ob_type for NewWithVtable).
    pub known_class: Option<i64>,
    /// Tagged field values (i16 tags referencing consts/boxes/other virtuals).
    pub fieldnums: Vec<i16>,
}

/// resume.py:554-557 — tagged pending field entry.
/// RPython stores (lldescr, num, fieldnum, itemindex) where num and fieldnum
/// are tagged references into the numbering system.
#[derive(Debug, Clone)]
pub struct TaggedPendingField {
    pub descr_index: u32,
    pub item_index: i32,
    /// Tagged reference to target box (from _gettagged).
    pub num: i16,
    /// Tagged reference to value box (from _gettagged).
    pub fieldnum: i16,
}

/// bridgeopt.py:63 — optimizer knowledge for resume data encoding.
/// Passed into finish() for _add_optimizer_sections.
pub struct OptimizerKnowledgeForResume {
    /// (obj_opref, descr_index, val_opref) heap field triples.
    /// bridgeopt.py:96-101
    pub heap_fields: Vec<(majit_ir::OpRef, i32, majit_ir::OpRef)>,
    /// (array_opref, index, descr_index, val_opref) heap array item quads.
    /// bridgeopt.py:102-108
    pub heap_arrayitems: Vec<(majit_ir::OpRef, i64, i32, majit_ir::OpRef)>,
    /// (const_func_ptr, result_opref) loop-invariant call results.
    pub loopinvariant_results: Vec<(i64, majit_ir::OpRef)>,
}

impl OptimizerKnowledgeForResume {
    pub fn is_empty(&self) -> bool {
        self.heap_fields.is_empty()
            && self.heap_arrayitems.is_empty()
            && self.loopinvariant_results.is_empty()
    }
}

/// bridgeopt.py:44-61 decode_box return type.
///
/// RPython's decode_box returns actual Const/Box objects. Two Rust
/// variants mirror that: `LiveBox` for `Box` references (TAGBOX) and
/// `Const` for all Const subtypes (TAGINT/TAGCONST/NULLREF), collapsing
/// into a single `majit_ir::Const` that carries its own type.
#[derive(Clone, Debug, PartialEq)]
pub enum DecodedBox {
    /// TAGBOX → liveboxes[num] (bridge inputarg / optimizer box).
    LiveBox(majit_ir::OpRef),
    /// TAGINT / TAGCONST / NULLREF — all Const subtypes.
    Const(majit_ir::Const),
}

/// bridgeopt.py:44-61 decode_box: untag a tagged value from rd_numb.
///
/// Line-by-line port of PyPy's decode_box(). Returns DecodedBox to
/// preserve the Const vs Box distinction that RPython encodes via
/// Python class hierarchy.
pub fn decode_box(
    tagged: i16,
    rd_consts: &[majit_ir::Const],
    liveboxes: &[majit_ir::OpRef],
) -> DecodedBox {
    let (num, tag_type) = untag(tagged);
    // NB: the TAGVIRTUAL case can't happen here, because this code runs after
    // virtuals are already forced again.
    match tag_type {
        TAGCONST => {
            if tagged_eq(tagged, NULLREF) {
                // bridgeopt.py:51: box = CONST_NULL (history.py:361).
                DecodedBox::Const(majit_ir::Const::Ref(majit_ir::GcRef::NULL))
            } else {
                // bridgeopt.py:54: box = resumestorage.rd_consts[num - TAG_CONST_OFFSET]
                // — direct list index, IndexError on out-of-range. A
                // bridgeopt-knowledge stream that names a const slot
                // outside `rd_consts` is corrupt, and silently swapping
                // in CONST_NULL hides the corruption from the optimizer.
                let idx = (num - TAG_CONST_OFFSET) as usize;
                let c = rd_consts.get(idx).copied().unwrap_or_else(|| {
                    panic!(
                        "bridgeopt decode_box TAGCONST out-of-range: idx={} \
                         rd_consts.len()={} (corrupt knowledge stream — see \
                         bridgeopt.py:54)",
                        idx,
                        rd_consts.len()
                    )
                });
                DecodedBox::Const(c)
            }
        }
        // bridgeopt.py:56: box = ConstInt(num)
        TAGINT => DecodedBox::Const(majit_ir::Const::Int(num as i64)),
        TAGBOX => {
            // bridgeopt.py:58: box = liveboxes[num] — direct list index,
            // IndexError on out-of-range. See TAGCONST comment above.
            let idx = num as usize;
            let lb = *liveboxes.get(idx).unwrap_or_else(|| {
                panic!(
                    "bridgeopt decode_box TAGBOX out-of-range: idx={} \
                     liveboxes.len()={} (corrupt knowledge stream — see \
                     bridgeopt.py:58)",
                    idx,
                    liveboxes.len()
                )
            });
            DecodedBox::LiveBox(lb)
        }
        _ => {
            // bridgeopt.py:60: raise AssertionError("unreachable")
            unreachable!("bridgeopt decode_box: unexpected tag type {}", tag_type);
        }
    }
}

// VirtualFieldInfo removed: replaced by majit_ir::VirtualFieldsInfo.
// finish() now discovers virtual fields via env.get_virtual_fields().

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::resumedata::{RebuiltValue, rebuild_from_numbering};

    #[test]
    fn livebox_map_preserves_box_identity_and_insertion_order() {
        let mut liveboxes = LiveboxMap::new();
        // Two distinct logical boxes — an InputArg and a ResOp result, each
        // bound to a rooted producer so they shed to Operand::InputArg / Op.
        // Under ptr_eq keying they stay distinct keys (PyPy `box is box`),
        // never collapsed by a shared raw slot index.
        let input = crate::history::test_support::rooted_operand_from_opref(
            majit_ir::OpRef::input_arg_int(0),
        );
        let op =
            crate::history::test_support::rooted_operand_from_opref(majit_ir::OpRef::int_op(0));

        liveboxes.insert(input.clone(), UNASSIGNED);
        liveboxes.insert(op.clone(), UNASSIGNEDVIRTUAL);

        assert_eq!(liveboxes.get(&input), Some(UNASSIGNED));
        assert_eq!(liveboxes.get(&op), Some(UNASSIGNEDVIRTUAL));
        // iter() yields each key's Operand directly, so compare by the stable
        // (type, position) OpRef identity + order.
        assert_eq!(
            liveboxes
                .iter()
                .map(|(b, t)| (b.to_opref(), t))
                .collect::<Vec<_>>(),
            vec![
                (input.to_opref(), UNASSIGNED),
                (op.to_opref(), UNASSIGNEDVIRTUAL)
            ]
        );
    }

    #[test]
    fn test_simple_resume_data() {
        let rd = ResumeData::simple(42, 3);
        let fail_values = vec![10, 20, 30];
        let frames = rd.reconstruct(&fail_values);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].pc, 42);
        assert_eq!(
            frames[0].values,
            vec![
                ReconstructedValue::Value(10),
                ReconstructedValue::Value(20),
                ReconstructedValue::Value(30),
            ]
        );
    }

    #[test]
    fn test_resume_data_with_gaps() {
        let rd = ResumeData {
            frames: vec![FrameInfo {
                jitcode_index: 0,
                pc: 100,
                slot_map: vec![
                    FrameSlotSource::FailArg(2),
                    FrameSlotSource::Unavailable,
                    FrameSlotSource::FailArg(0),
                ],
            }],
            virtuals: Vec::new(),
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            pending_fields: Vec::new(),
        };
        let fail_values = vec![10, 20, 30];
        let frames = rd.reconstruct(&fail_values);
        assert_eq!(
            frames[0].values,
            vec![
                ReconstructedValue::Value(30),
                ReconstructedValue::Unavailable,
                ReconstructedValue::Value(10),
            ]
        );
        assert_eq!(frames[0].lossy_values(), vec![30, 0, 10]);
    }

    #[test]
    fn test_multi_frame_resume() {
        let rd = ResumeData {
            frames: vec![
                FrameInfo {
                    jitcode_index: 0,
                    pc: 10,
                    slot_map: vec![FrameSlotSource::FailArg(0), FrameSlotSource::FailArg(1)],
                },
                FrameInfo {
                    jitcode_index: 1,
                    pc: 20,
                    slot_map: vec![FrameSlotSource::FailArg(2), FrameSlotSource::FailArg(3)],
                },
            ],
            virtuals: Vec::new(),
            vable_array: Vec::new(),
            vref_array: Vec::new(),
            pending_fields: Vec::new(),
        };
        let fail_values = vec![1, 2, 3, 4];
        let frames = rd.reconstruct(&fail_values);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].pc, 10);
        assert_eq!(
            frames[0].values,
            vec![ReconstructedValue::Value(1), ReconstructedValue::Value(2)]
        );
        assert_eq!(frames[1].pc, 20);
        assert_eq!(
            frames[1].values,
            vec![ReconstructedValue::Value(3), ReconstructedValue::Value(4)]
        );
    }

    #[test]
    fn test_builder() {
        let mut builder = ResumeDataVirtualAdder::new();
        builder.push_frame(0, 42);
        builder.map_slot(0, 0);
        builder.map_slot(2, 1); // gap at slot 1
        let rd = builder.build();

        assert_eq!(rd.frames.len(), 1);
        assert_eq!(rd.frames[0].pc, 42);
        assert_eq!(
            rd.frames[0].slot_map,
            vec![
                FrameSlotSource::FailArg(0),
                FrameSlotSource::Unavailable,
                FrameSlotSource::FailArg(1),
            ]
        );
    }

    #[test]
    fn test_memo_number_simple() {
        use majit_ir::OpRef;
        let mut memo = ResumeDataLoopMemo::new();
        let mut env = SimpleBoxEnv::new();
        let snapshot = Snapshot::single_frame(
            0,
            8,
            vec![OpRef::const_int(42), OpRef::int_op(1), OpRef::int_op(2)],
        );
        let numb_state = memo.number(&snapshot, &env, -1).unwrap();
        // Should have: [size, num_failargs, 0(vable), 0(vref), 0(jitcode), 8(pc), tagged...]
        let items = crate::resumecode::unpack_numbering(&numb_state.create_numbering());
        // items[0] = total size
        assert!(items[0] > 0);
        // items[1] = num_failargs: 0 (not patched yet — RPython patches in finish())
        // After finish: patch(1, numb_state.liveboxes.len()) would set to 2.
        assert_eq!(items[1], 0);
        // items[2] = vable_array_length = 0
        assert_eq!(items[2], 0);
        // items[3] = vref_array_length = 0
        assert_eq!(items[3], 0);
        // items[4] = jitcode_index = 0
        assert_eq!(items[4], 0);
        // items[5] = pc = 8
        assert_eq!(items[5], 8);
        // items[6] = jitcode_pc = NO_JITCODE_PC (sentinel)
        assert_eq!(items[6], majit_ir::resumedata::NO_JITCODE_PC);
        // items[7] = inline-Const(42) tagged as TAGINT(42) since 42 fits in 13 bits
        let (val, tagbits) = untag(items[7] as i16);
        assert_eq!(tagbits, TAGINT);
        assert_eq!(val, 42);
        // items[8] = OpRef::int_op(1) tagged as TAGBOX(0) — first live box
        let (val, tagbits) = untag(items[8] as i16);
        assert_eq!(tagbits, TAGBOX);
        assert_eq!(val, 0);
        // items[9] = OpRef::int_op(2) tagged as TAGBOX(1) — second live box
        let (val, tagbits) = untag(items[9] as i16);
        assert_eq!(tagbits, TAGBOX);
        assert_eq!(val, 1);
    }

    #[test]
    fn test_single_frame_with_jitcode_pc_carries_offset_into_numbering() {
        // #124 Approach B (M2): the production capture factory writes the
        // guard's real JitCode byte offset into the per-frame `jitcode_pc`
        // word of `rd_numb`, where `single_frame` / the legacy
        // `single_frame_boxes` write `NO_JITCODE_PC` (`test_memo_number_simple`
        // pins the sentinel at the same slot, `items[6]`).
        use majit_ir::OpRef;
        let mut memo = ResumeDataLoopMemo::new();
        let env = SimpleBoxEnv::new();
        let snapshot = Snapshot::single_frame_boxes_with_jitcode_pc(
            0,
            8,
            42,
            vec![OpRef::const_int(42).into(), OpRef::int_op(1).into()],
        );
        let numb_state = memo.number(&snapshot, &env, -1).unwrap();
        let items = crate::resumecode::unpack_numbering(&numb_state.create_numbering());
        // Empty vable/vref single-frame header:
        // items[4]=jitcode_index(0) items[5]=pc(8) items[6]=jitcode_pc.
        assert_eq!(items[4], 0); // jitcode_index
        assert_eq!(items[5], 8); // pc
        assert_eq!(items[6], 42); // jitcode_pc carried by the M2 factory
        assert_ne!(items[6], majit_ir::resumedata::NO_JITCODE_PC);
    }

    #[test]
    fn test_multi_frame_with_jitcode_pc_per_frame() {
        // #124 Approach B (M2): the multi-frame factory carries a distinct
        // `jitcode_pc` per frame; a frame with no JitCode coordinate keeps
        // the `NO_JITCODE_PC` sentinel.
        use majit_ir::OpRef;
        let mut memo = ResumeDataLoopMemo::new();
        let env = SimpleBoxEnv::new();
        let snapshot = Snapshot::multi_frame_boxes_with_jitcode_pc(vec![
            (
                0,
                10,
                majit_ir::resumedata::NO_JITCODE_PC,
                vec![OpRef::int_op(1).into()],
            ),
            (1, 20, 55, vec![OpRef::int_op(2).into()]),
        ]);
        let items = crate::resumecode::unpack_numbering(
            &memo.number(&snapshot, &env, -1).unwrap().create_numbering(),
        );
        // Frame 0: items[4]=jitcode(0) items[5]=pc(10) items[6]=jitcode_pc(sentinel) items[7]=box
        assert_eq!(items[4], 0);
        assert_eq!(items[5], 10);
        assert_eq!(items[6], majit_ir::resumedata::NO_JITCODE_PC);
        // Frame 1: items[8]=jitcode(1) items[9]=pc(20) items[10]=jitcode_pc(55) items[11]=box
        assert_eq!(items[8], 1);
        assert_eq!(items[9], 20);
        assert_eq!(items[10], 55);
    }

    #[test]
    fn test_number_rebuild_roundtrip() {
        use majit_ir::OpRef;
        let mut memo = ResumeDataLoopMemo::new();
        let mut env = SimpleBoxEnv::new();
        let snapshot = Snapshot::single_frame(
            0,
            8,
            vec![OpRef::const_int(42), OpRef::int_op(1), OpRef::int_op(2)],
        );
        let mut numb_state = memo.number(&snapshot, &env, -1).unwrap();
        // RPython: ResumeDataVirtualAdder.finish() patches slot 1 with num_boxes.
        numb_state.writer.patch(1, numb_state.num_boxes);
        let rd_numb = numb_state.create_numbering();

        let fail_arg_types = vec![majit_ir::Type::Int, majit_ir::Type::Int];
        let (num_failargs, _vable_values, _vref_values, rebuilt_frames) =
            rebuild_from_numbering(&rd_numb, memo.consts(), &fail_arg_types, None, 0);
        assert_eq!(num_failargs, 2);
        assert_eq!(rebuilt_frames.len(), 1);
        assert_eq!(rebuilt_frames[0].pc, 8);
        assert_eq!(rebuilt_frames[0].values.len(), 3);
        assert_eq!(
            rebuilt_frames[0].values[0],
            RebuiltValue::Const(majit_ir::Const::Int(42))
        );
        assert_eq!(
            rebuilt_frames[0].values[1],
            RebuiltValue::Box(0, majit_ir::Type::Int)
        );
        assert_eq!(
            rebuilt_frames[0].values[2],
            RebuiltValue::Box(1, majit_ir::Type::Int)
        );
    }

    #[test]
    fn test_number_rebuild_with_virtual() {
        use majit_ir::OpRef;
        let mut memo = ResumeDataLoopMemo::new();
        let mut env = SimpleBoxEnv::new();
        env.virtuals.insert(2); // OpRef::ref_op(2) is virtual (Ref type)
        env.types.insert(2, majit_ir::Type::Ref);
        let snapshot = Snapshot::single_frame(
            0,
            10,
            vec![OpRef::int_op(1), OpRef::ref_op(2), OpRef::int_op(3)],
        );
        let mut numb_state = memo.number(&snapshot, &env, -1).unwrap();
        // RPython: finish() patches with len(newboxes) which is num_boxes
        // (not liveboxes which includes virtuals).
        numb_state.writer.patch(1, numb_state.num_boxes);
        let rd_numb = numb_state.create_numbering();

        let fail_arg_types = vec![majit_ir::Type::Int, majit_ir::Type::Int];
        let (num_failargs, _vable_values, _vref_values, rebuilt_frames) =
            rebuild_from_numbering(&rd_numb, memo.consts(), &fail_arg_types, None, 0);
        assert_eq!(num_failargs, 2); // OpRef::int_op(1) and OpRef::int_op(3) are boxes
        assert_eq!(rebuilt_frames[0].values.len(), 3);
        assert_eq!(
            rebuilt_frames[0].values[0],
            RebuiltValue::Box(0, majit_ir::Type::Int)
        );
        assert_eq!(rebuilt_frames[0].values[1], RebuiltValue::Virtual(0));
        assert_eq!(
            rebuilt_frames[0].values[2],
            RebuiltValue::Box(1, majit_ir::Type::Int)
        );
    }

    #[test]
    fn test_memo_number_with_virtual() {
        use majit_ir::OpRef;
        let mut memo = ResumeDataLoopMemo::new();
        let mut env = SimpleBoxEnv::new();
        env.virtuals.insert(2);
        env.types.insert(2, majit_ir::Type::Ref);
        let snapshot = Snapshot::single_frame(
            0,
            10,
            vec![OpRef::int_op(1), OpRef::ref_op(2), OpRef::int_op(3)],
        );
        let numb_state = memo.number(&snapshot, &env, -1).unwrap();
        let items = crate::resumecode::unpack_numbering(&numb_state.create_numbering());
        // items[1] = num_failargs: 0 (not patched — RPython patches in finish())
        assert_eq!(items[1], 0);
        // items[6] = jitcode_pc = NO_JITCODE_PC (sentinel)
        assert_eq!(items[6], majit_ir::resumedata::NO_JITCODE_PC);
        // items[7] = OpRef::int_op(1) → TAGBOX(0)
        let (val, tagbits) = untag(items[7] as i16);
        assert_eq!(tagbits, TAGBOX);
        assert_eq!(val, 0);
        // items[8] = OpRef::ref_op(2) → TAGVIRTUAL(0)
        let (val, tagbits) = untag(items[8] as i16);
        assert_eq!(tagbits, TAGVIRTUAL);
        assert_eq!(val, 0);
        // items[9] = OpRef::int_op(3) → TAGBOX(1)
        let (val, tagbits) = untag(items[9] as i16);
        assert_eq!(tagbits, TAGBOX);
        assert_eq!(val, 1);
    }

    #[test]
    fn test_number_boxes_uses_replacement_type_for_virtual_classification() {
        use majit_ir::OpRef;
        struct RefOnlyVirtualEnv {
            constants: indexmap::IndexMap<u32, (i64, majit_ir::Type)>,
            replacements: indexmap::IndexMap<u32, majit_ir::OpRef>,
            types: indexmap::IndexMap<u32, majit_ir::Type>,
            virtuals: indexmap::IndexSet<u32>,
            virtual_fields: indexmap::IndexMap<u32, majit_ir::VirtualFieldsInfo>,
            box_cache:
                std::cell::RefCell<indexmap::IndexMap<majit_ir::OpRef, majit_ir::operand::Operand>>,
        }

        impl RefOnlyVirtualEnv {
            fn new() -> Self {
                Self {
                    constants: indexmap::IndexMap::new(),
                    replacements: indexmap::IndexMap::new(),
                    types: indexmap::IndexMap::new(),
                    virtuals: indexmap::IndexSet::new(),
                    virtual_fields: indexmap::IndexMap::new(),
                    box_cache: std::cell::RefCell::new(indexmap::IndexMap::new()),
                }
            }
        }

        impl BoxEnv for RefOnlyVirtualEnv {
            fn get_box_replacement(&self, opref: majit_ir::OpRef) -> majit_ir::OpRef {
                self.replacements
                    .get(&opref.raw())
                    .copied()
                    .unwrap_or(opref)
            }

            fn get_box_replacement_operand(
                &self,
                opref: majit_ir::OpRef,
            ) -> majit_ir::operand::Operand {
                // #160/S11: memoize one Operand per replacement-walked OpRef.
                let root = self.get_box_replacement(opref);
                if let Some(b) = self.box_cache.borrow().get(&root) {
                    return b.clone();
                }
                // Synthesize a rooted bound producer so the box sheds to
                // `Operand::Op`/`InputArg` for the Operand-keyed liveboxes map.
                let b = crate::history::test_support::rooted_operand_from_opref(root);
                self.box_cache.borrow_mut().insert(root, b.clone());
                b
            }

            fn get_box_replacement_not_const(&self, opref: majit_ir::OpRef) -> majit_ir::OpRef {
                self.get_box_replacement(opref)
            }

            fn is_const(&self, opref: majit_ir::OpRef) -> bool {
                self.constants.contains_key(&opref.raw())
            }

            fn get_const(&self, opref: majit_ir::OpRef) -> (i64, majit_ir::Type) {
                self.constants
                    .get(&opref.raw())
                    .copied()
                    .unwrap_or((0, majit_ir::Type::Int))
            }

            fn get_type(&self, opref: majit_ir::OpRef) -> majit_ir::Type {
                self.types
                    .get(&opref.raw())
                    .copied()
                    .unwrap_or(majit_ir::Type::Int)
            }

            fn is_virtual_ref(&self, opref: majit_ir::OpRef) -> bool {
                self.virtuals.contains(&opref.raw())
            }

            fn is_virtual_raw(&self, _opref: majit_ir::OpRef) -> bool {
                false
            }

            fn get_virtual_fields(
                &self,
                opref: majit_ir::OpRef,
            ) -> Option<majit_ir::VirtualFieldsInfo> {
                self.virtual_fields.get(&opref.raw()).cloned()
            }
        }

        let mut memo = ResumeDataLoopMemo::new();
        let mut env = RefOnlyVirtualEnv::new();

        // RPython resume.py reads box.type after get_box_replacement().
        // Model a stale Int-typed snapshot slot that now forwards to a Ref
        // virtual, the shape produced by optimized boxed-int locals.
        let source = OpRef::int_op(1);
        let target = OpRef::ref_op(2);
        env.replacements.insert(source.raw(), target);
        env.virtuals.insert(target.raw());
        env.types.insert(target.raw(), majit_ir::Type::Ref);

        let snapshot = Snapshot::single_frame_boxes(
            0,
            10,
            vec![SnapshotBox::typed(source, majit_ir::Type::Int)],
        );
        let numb_state = memo.number(&snapshot, &env, -1).unwrap();
        let items = crate::resumecode::unpack_numbering(&numb_state.create_numbering());

        // items[6] = jitcode_pc (sentinel); items[7] = the frame's box.
        assert_eq!(items[6], majit_ir::resumedata::NO_JITCODE_PC);
        let (val, tagbits) = untag(items[7] as i16);
        assert_eq!(tagbits, TAGVIRTUAL);
        assert_eq!(val, 0);
        assert_eq!(numb_state.num_boxes, 0);
        assert_eq!(numb_state.num_virtuals, 1);
    }

    #[test]
    fn test_multi_frame_snapshot() {
        use majit_ir::OpRef;
        let mut memo = ResumeDataLoopMemo::new();
        let env = SimpleBoxEnv::new();

        let snapshot = Snapshot {
            vable_array: vec![],
            vref_array: vec![],
            framestack: vec![
                SnapshotFrame {
                    jitcode_index: 0,
                    pc: 10,
                    jitcode_pc: majit_ir::resumedata::NO_JITCODE_PC,
                    boxes: vec![OpRef::int_op(1).into(), OpRef::const_int(99).into()],
                },
                SnapshotFrame {
                    jitcode_index: 1,
                    pc: 20,
                    jitcode_pc: majit_ir::resumedata::NO_JITCODE_PC,
                    boxes: vec![OpRef::int_op(2).into(), OpRef::int_op(3).into()],
                },
            ],
        };

        let mut numb_state = memo.number(&snapshot, &env, -1).unwrap();
        numb_state.writer.patch(1, numb_state.num_boxes);
        let rd_numb = numb_state.create_numbering();

        // Multi-frame encoding: no box_count, RPython parity.
        let items = crate::resumecode::unpack_numbering(&rd_numb);
        assert_eq!(items[1], 3); // num_failargs: 3 boxes patched
        // Frame 0: items[4]=jitcode(0), items[5]=pc(10), items[6]=jitcode_pc, items[7..8]=tagged
        assert_eq!(items[4], 0);
        assert_eq!(items[5], 10);
        assert_eq!(items[6], majit_ir::resumedata::NO_JITCODE_PC);
        // Frame 1: items[9]=jitcode(1), items[10]=pc(20), items[11]=jitcode_pc, items[12..13]=tagged
        assert_eq!(items[9], 1);
        assert_eq!(items[10], 20);
        assert_eq!(items[11], majit_ir::resumedata::NO_JITCODE_PC);

        // Roundtrip with liveness-based closure.
        let rd_consts: Vec<majit_ir::Const> = memo.consts().to_vec();
        let frame_count = |jitcode_index: i32, _pc: i32, _jitcode_pc: i32| -> usize {
            match jitcode_index {
                0 => 2, // Frame 0 has 2 boxes
                1 => 2, // Frame 1 has 2 boxes
                _ => 0,
            }
        };
        let fail_arg_types = vec![
            majit_ir::Type::Int,
            majit_ir::Type::Int,
            majit_ir::Type::Int,
        ];
        let (num_failargs, _vable_values, _vref_values, rebuilt_frames) =
            rebuild_from_numbering(&rd_numb, &rd_consts, &fail_arg_types, Some(&frame_count), 0);
        assert_eq!(num_failargs, 3);
        assert_eq!(rebuilt_frames.len(), 2);
        assert_eq!(rebuilt_frames[0].jitcode_index, 0);
        assert_eq!(rebuilt_frames[0].pc, 10);
        assert_eq!(rebuilt_frames[0].values.len(), 2);
        assert_eq!(rebuilt_frames[1].jitcode_index, 1);
        assert_eq!(rebuilt_frames[1].pc, 20);
        assert_eq!(rebuilt_frames[1].values.len(), 2);
    }

    #[test]
    fn test_finish_produces_rd_numb_and_liveboxes() {
        use majit_ir::OpRef;
        let mut memo = ResumeDataLoopMemo::new();
        let mut env = SimpleBoxEnv::new();
        env.virtuals.insert(2);
        env.types.insert(2, majit_ir::Type::Ref);

        let snapshot = Snapshot::single_frame(
            0,
            8,
            vec![
                OpRef::const_int(42),
                OpRef::int_op(1),
                OpRef::ref_op(2),
                OpRef::int_op(3),
            ],
        );
        let numb_state = memo.number(&snapshot, &env, -1).unwrap();
        let (rd_numb, rd_consts, _rd_virtuals, liveboxes, _livebox_types) =
            memo.finish(numb_state, &env, &mut [], None);

        // liveboxes should contain only TAGBOX entries: OpRef::int_op(1) and OpRef::int_op(3)
        assert_eq!(liveboxes.len(), 2);
        assert_eq!(liveboxes[0], OpRef::int_op(1)); // box #0
        assert_eq!(liveboxes[1], OpRef::int_op(3)); // box #1

        // rd_numb should be valid
        let fail_arg_types = vec![majit_ir::Type::Int, majit_ir::Type::Int];
        let (num_failargs, _vable_values, _vref_values, rebuilt_frames) =
            rebuild_from_numbering(&rd_numb, &rd_consts, &fail_arg_types, None, 0);
        assert_eq!(num_failargs, 2);
        assert_eq!(rebuilt_frames.len(), 1);
        assert_eq!(
            rebuilt_frames[0].values[0],
            RebuiltValue::Const(majit_ir::Const::Int(42))
        );
        assert_eq!(
            rebuilt_frames[0].values[1],
            RebuiltValue::Box(0, majit_ir::Type::Int)
        );
        assert_eq!(rebuilt_frames[0].values[2], RebuiltValue::Virtual(0));
        assert_eq!(
            rebuilt_frames[0].values[3],
            RebuiltValue::Box(1, majit_ir::Type::Int)
        );
    }

    #[test]
    fn test_number_virtualizable_array_preserves_payload_then_identity_order() {
        use majit_ir::OpRef;

        let mut memo = ResumeDataLoopMemo::new();
        let mut env = SimpleBoxEnv::new();
        env.types.insert(7, majit_ir::Type::Ref);
        env.types.insert(1, majit_ir::Type::Int);

        let snapshot = Snapshot {
            // pyjitpl.py:3302-3306 parity: payload slots first,
            // virtualizable identity (`virtualizable_boxes[-1]`) last.
            vable_array: vec![OpRef::int_op(1).into(), OpRef::ref_op(7).into()],
            vref_array: vec![],
            framestack: vec![SnapshotFrame {
                jitcode_index: 0,
                pc: 8,
                jitcode_pc: 13,
                boxes: vec![OpRef::int_op(1).into()],
            }],
        };

        let numb_state = memo.number(&snapshot, &env, 0).unwrap();
        let items = crate::resumecode::unpack_numbering(&numb_state.create_numbering());

        assert_eq!(items[2], 2);
        let (val, tagbits) = untag(items[3] as i16);
        assert_eq!(tagbits, TAGBOX);
        assert_eq!(val, 0);

        let (val, tagbits) = untag(items[4] as i16);
        assert_eq!(tagbits, TAGBOX);
        assert_eq!(val, 1);

        assert_eq!(items[5], 0); // vref_array_length
        assert_eq!(items[6], 0); // jitcode_index
        assert_eq!(items[7], 8); // pc
        assert_eq!(items[8], 13); // jitcode_pc

        // The frame slot reuses the payload tag because numbering follows
        // Box identity exactly: upstream dedups only when the same Box object
        // appears twice, and in this test we passed the same OpRef twice.
        let (val, tagbits) = untag(items[9] as i16);
        assert_eq!(tagbits, TAGBOX);
        assert_eq!(val, 0);
    }

    #[test]
    fn blackhole_from_resumedata_accepts_runtime_jitcode_without_canonical_pair() {
        use crate::blackhole::BlackholeInterpBuilder;
        use crate::jitcode::JitCodeBuilder;
        use crate::jitcode::insns::{BC_ABORT, BC_CATCH_EXCEPTION, BC_LIVE, BC_RVMPROF_CODE};

        let mut writer = crate::resumecode::Writer::new(7);
        writer.append_int(0); // items_resume_section (patched below)
        writer.append_int(0); // count: no failargs
        writer.append_int(0); // vable_array length
        writer.append_int(0); // vref_array length
        writer.append_int(0); // jitcode_pos
        writer.append_int(0); // pc
        writer.append_int(majit_ir::resumedata::NO_JITCODE_PC as i64); // jitcode_pc
        writer.patch_current_size(0);
        let rd_numb = writer.create_numbering();

        let mut runtime = JitCodeBuilder::default().finish();
        runtime.body_mut().code = vec![BC_LIVE, 0, 0, BC_ABORT];
        runtime.body_mut().c_num_regs_i = 1;
        runtime.body_mut().constants_i = vec![321];
        // Hand-crafted body bypasses the builder's `start_instr` path,
        // so populate `startpoints` explicitly: BC_LIVE at 0,
        // BC_ABORT at 3.  RPython `jitcode.py:85-90` asserts
        // `pc in self._startpoints`.
        runtime.body_mut().startpoints = Some([0_usize, 3].into_iter().collect());
        let runtime = std::sync::Arc::new(runtime);

        let mut builder = BlackholeInterpBuilder::new();
        builder.setup_cached_control_opcodes(
            BC_LIVE as i32,
            BC_CATCH_EXCEPTION as i32,
            BC_RVMPROF_CODE as i32,
        );
        let resolve_jitcode =
            |_jitcode_pos: i32, _pc: i32, _jitcode_pc: i32| -> Option<ResolvedJitCode> {
                Some(ResolvedJitCode::new(runtime.clone(), 0))
            };

        let all_liveness: Vec<u8> = vec![0, 0, 0];
        let (bh, virtualizable_ptr) = blackhole_from_resumedata(
            &mut builder,
            &resolve_jitcode,
            &rd_numb,
            &[],
            &all_liveness,
            &[],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            &NullAllocator,
        )
        .expect("runtime-only jitcode should still resume");

        assert_eq!(virtualizable_ptr, 0);
        assert!(std::sync::Arc::ptr_eq(&bh.jitcode, &runtime));
        assert_eq!(bh.position, 0);
        assert_eq!(bh.registers_i, vec![0, 321]);
    }

    struct TestVirtualizableInfo;

    impl VirtualizableInfo for TestVirtualizableInfo {
        fn get_total_size(&self, _virtualizable: i64) -> usize {
            0
        }

        fn reset_token_gcref(&self, _virtualizable: i64) {}

        fn write_from_resume_data_partial(
            &self,
            _virtualizable: i64,
            _reader: &mut ResumeDataDirectReader,
        ) {
        }
    }

    #[test]
    #[should_panic(expected = "vable_size > 0")]
    fn test_consume_vref_and_vable_asserts_zero_vable_size_when_vinfo_present() {
        let mut writer = crate::resumecode::Writer::new(3);
        writer.append_int(0); // items_resume_section (patched below)
        writer.append_int(0); // count
        writer.append_int(0); // vable_size
        writer.patch_current_size(0);
        let rd_numb = writer.create_numbering();

        let mut reader =
            ResumeDataDirectReader::new(&rd_numb, &[], &[], &[], None, None, &NullAllocator);
        reader.consume_vref_and_vable(None, Some(&TestVirtualizableInfo), None, None);
    }

    #[test]
    #[should_panic(expected = "load_next_value_of_type: unexpected type Void")]
    fn test_next_value_of_type_rejects_void() {
        let mut reader =
            ResumeDataDirectReader::new(&[0, 0], &[], &[], &[], None, None, &NullAllocator);
        let _ = reader.next_value_of_type(majit_ir::Type::Void);
    }
}

// ═══════════════════════════════════════════════════════════════
// resume.py:901-1039 AbstractResumeDataReader
// resume.py:1354-1601 ResumeDataDirectReader
//
// Direct reader that decodes resume data and fills blackhole
// interpreter registers with concrete values from the deadframe.
// ═══════════════════════════════════════════════════════════════

use crate::blackhole::BlackholeInterpreter;
use crate::resumecode::Reader;

/// RPython virtualref_info interface for resume data consumption.
///
/// Corresponds to `metainterp_sd.virtualref_info` (VirtualRefInfo).
pub trait VRefInfo {
    /// resume.py:1397 vrefinfo.continue_tracing(vref, virtual)
    fn continue_tracing(&self, vref: i64, virtual_ref: i64);
}

/// RPython virtualizable_info interface for resume data consumption.
///
/// Corresponds to `jitdriver_sd.virtualizable_info` (VirtualizableInfo).
pub trait VirtualizableInfo {
    /// resume.py:1406 vinfo.get_total_size(virtualizable)
    fn get_total_size(&self, virtualizable: i64) -> usize;

    /// resume.py:1407 vinfo.reset_token_gcref(virtualizable)
    fn reset_token_gcref(&self, virtualizable: i64);

    /// resume.py:1408 vinfo.write_from_resume_data_partial(virtualizable, self)
    ///
    /// Read fields from the resume reader and write them into the virtualizable.
    fn write_from_resume_data_partial(
        &self,
        virtualizable: i64,
        reader: &mut ResumeDataDirectReader,
    );
}

/// RPython greenfield_info interface for resume data consumption.
///
/// Corresponds to `jitdriver_sd.greenfield_info`.
pub trait GreenfieldInfo {}

/// resume.py:1354 ResumeDataDirectReader
///
/// Reads encoded resume data (rd_numb) and fills blackhole interpreter
/// resume.py:874-899 AbstractVirtualCache / get_VirtualCache_class
///
/// ```text
/// class AbstractVirtualCache(object):
///     pass
///
/// def get_VirtualCache_class(suffix):
///     class VirtualCache(AbstractVirtualCache):
///         def __init__(self, virtuals_ptr_cache, virtuals_int_cache):
///             self.virtuals_ptr_cache = virtuals_ptr_cache
///             self.virtuals_int_cache = virtuals_int_cache
///
///         def get_ptr(self, i):  return self.virtuals_ptr_cache[i]
///         def get_int(self, i):  return self.virtuals_int_cache[i]
///         def set_ptr(self, i, v): self.virtuals_ptr_cache[i] = v
///         def set_int(self, i, v): self.virtuals_int_cache[i] = v
/// ```
///
/// RPython generates two flavours of this class — one for
/// `ResumeDataDirectReader` (raw `i64` slots) and one for the future
/// `ResumeDataBoxReader` (boxed). majit only emits the direct flavour at
/// runtime, so a single struct backs both.
#[derive(Default)]
pub struct VirtualCache {
    pub virtuals_ptr_cache: Vec<i64>,
    pub virtuals_int_cache: Vec<i64>,
}

impl VirtualCache {
    pub fn new() -> Self {
        VirtualCache::default()
    }

    /// resume.py:882-884 __init__
    pub fn from_caches(virtuals_ptr_cache: Vec<i64>, virtuals_int_cache: Vec<i64>) -> Self {
        VirtualCache {
            virtuals_ptr_cache,
            virtuals_int_cache,
        }
    }

    /// resume.py:886-887 get_ptr
    #[inline]
    pub fn get_ptr(&self, i: usize) -> i64 {
        self.virtuals_ptr_cache[i]
    }

    /// resume.py:889-890 get_int
    #[inline]
    pub fn get_int(&self, i: usize) -> i64 {
        self.virtuals_int_cache[i]
    }

    /// resume.py:892-893 set_ptr
    #[inline]
    pub fn set_ptr(&mut self, i: usize, v: i64) {
        self.virtuals_ptr_cache[i] = v;
    }

    /// resume.py:895-896 set_int
    #[inline]
    pub fn set_int(&mut self, i: usize, v: i64) {
        self.virtuals_int_cache[i] = v;
    }

    /// `len(self.virtuals_ptr_cache)` — both halves stay the same length.
    pub fn len(&self) -> usize {
        self.virtuals_ptr_cache.len()
    }

    pub fn is_empty(&self) -> bool {
        self.virtuals_ptr_cache.is_empty()
    }
}

/// registers directly from the deadframe's fail_args values.
///
/// Combines AbstractResumeDataReader (resume.py:901) mixin with
/// ResumeDataDirectReader (resume.py:1354) concrete class.
pub struct ResumeDataDirectReader<'a> {
    // AbstractResumeDataReader fields (resume.py:909-922)
    /// resume.py:918 resumecodereader
    pub resumecodereader: Reader<'a>,
    /// resume.py:919 items_resume_section — total items in resume section
    pub items_resume_section: i32,
    /// resume.py:921 count — number of failargs
    pub count: i32,
    /// resume.py:922 consts — constant pool from rd_consts.
    ///
    /// RPython stores `list[Const]` where each `Const` carries its own type
    /// (history.py:220 ConstInt / :261 ConstFloat / :307 ConstPtr). In the
    /// Rust port we represent each entry as `(raw_i64, Type)` so that
    /// `ConstPtr.getref_base()` parity (returning a GC-tracked pointer) can
    /// be surfaced to the minor-collection root walker — see
    /// `walk_rd_consts_refs` on `MetaInterp`. Raw `Vec<i64>` hid the
    /// Ref-typed entries from the GC and caused nursery use-after-free in
    /// TAGCONST decode paths (resume.py:1557 decode_int / :1566 decode_ref /
    /// :1578 decode_float).
    pub consts: &'a [majit_ir::Const],

    // ResumeDataDirectReader fields (resume.py:1364-1367)
    /// resume.py:1366 deadframe — raw fail_args values
    pub deadframe: &'a [i64],
    /// pyre flat-deadframe adaptation: original type of each deadframe slot.
    /// RPython's CPU exposes typed getters (get_ref_value/get_int_value/...);
    /// pyre passes a flat raw slice and needs slot kinds to emulate
    /// load_box_from_cpu(kind) for TAGBOX decode.
    pub deadframe_types: Option<&'a [majit_ir::Type]>,

    // resume.py:1358 resume_after_guard_not_forced
    //   0: not a GUARD_NOT_FORCED
    //   1: in handle_async_forcing
    //   2: resuming from the GUARD_NOT_FORCED
    pub resume_after_guard_not_forced: u8,

    // resume.py:909 rd_virtuals
    rd_virtuals: Option<&'a [VirtualInfo]>,

    /// resume.py:910 virtuals_cache — lazy-allocated virtual objects.
    /// Wraps both the ptr and int half so callers go through the RPython
    /// `VirtualCache` API (`get_ptr`/`set_ptr`/`get_int`/`set_int`).
    pub virtuals_cache: VirtualCache,

    /// resume.py:1367 — CPU allocation backend.
    /// RPython uses self.cpu (from metainterp_sd.cpu) for allocate_with_vtable etc.
    allocator: &'a dyn BlackholeAllocator,

    /// resume.py:1022 `self.metainterp_sd.liveness_info` — shared
    /// packed `all_liveness` buffer used by `_prepare_next_section` /
    /// `enumerate_vars`. RPython reaches it through `self.metainterp_sd`;
    /// pyre holds the slice directly because `ResumeDataDirectReader`
    /// lives outside the `MetaInterpStaticData` ownership graph.
    pub all_liveness: &'a [u8],

    /// resume.py:1404: virtualizable pointer read by consume_vable_info.
    /// Stored so the caller (blackhole_from_resumedata) can access it
    /// after consume_vref_and_vable completes.
    pub virtualizable_ptr: i64,
}

/// resume.py:1433-1456 CPU allocation interface for virtual materialization.
///
/// ResumeDataDirectReader calls these methods when TAGVIRTUAL values
/// need to be lazily allocated during decode_ref/decode_int.
pub trait BlackholeAllocator {
    /// resume.py:1437-1439 allocate_with_vtable(known_class, descr) →
    ///   exec_new_with_vtable(self.cpu, descr)
    fn allocate_with_vtable(&self, descr: &majit_ir::DescrRef, vtable: usize) -> i64 {
        let _ = (descr, vtable);
        0
    }
    /// resume.py:1442 cpu.bh_new(typedescr)
    fn bh_new(&self, typedescr: &majit_ir::DescrRef) -> i64 {
        let _ = typedescr;
        0
    }
    /// resume.py:1446 cpu.bh_new_array_clear(length, arraydescr)
    fn bh_new_array_clear(&self, length: usize, arraydescr: &majit_ir::DescrRef) -> i64 {
        let _ = (length, arraydescr);
        0
    }
    /// resume.py:1447 cpu.bh_new_array(length, arraydescr)
    fn bh_new_array(&self, length: usize, arraydescr: &majit_ir::DescrRef) -> i64 {
        let _ = (length, arraydescr);
        0
    }
    /// resume.py:1533 cpu.bh_setarrayitem_gc_i(array, i, value, arraydescr)
    fn bh_setarrayitem_gc_i(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        let _ = (array, index, value, descr);
    }
    /// resume.py:1537 cpu.bh_setarrayitem_gc_r(array, i, value, arraydescr)
    fn bh_setarrayitem_gc_r(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        let _ = (array, index, value, descr);
    }
    /// resume.py:1541 cpu.bh_setarrayitem_gc_f(array, i, value, arraydescr)
    fn bh_setarrayitem_gc_f(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        let _ = (array, index, value, descr);
    }
    /// resume.py:1520-1529 setinteriorfield(index, array, fieldnum, descr)
    /// RPython passes the live descr object; backend reads offset/size/type from it.
    fn bh_setinteriorfield_gc_i(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        let _ = (array, index, value, descr);
    }
    fn bh_setinteriorfield_gc_r(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        let _ = (array, index, value, descr);
    }
    fn bh_setinteriorfield_gc_f(
        &self,
        array: i64,
        index: usize,
        value: i64,
        descr: &majit_ir::DescrRef,
    ) {
        let _ = (array, index, value, descr);
    }
    /// resume.py:1449-1450 allocate_string(length) → cpu.bh_newstr(length)
    fn bh_newstr(&self, length: usize) -> i64 {
        let _ = length;
        0
    }
    /// resume.py:1458-1460 string_setitem(str, index, charnum) →
    /// cpu.bh_strsetitem(str, index, char) — `char` is the decoded
    /// integer from the tagged `charnum`.
    fn bh_strsetitem(&self, string: i64, index: usize, char: i64) {
        let _ = (string, index, char);
    }
    /// resume.py:1462-1470 concat_strings(str1, str2) — implementations
    /// look up `OS_STR_CONCAT` via `callinfocollection.funcptr_for_oopspec`
    /// (resume.py:1467-1468) and call it directly.  The variant carries
    /// no funcptr.
    fn os_str_concat(&self, str1: i64, str2: i64) -> i64 {
        let _ = (str1, str2);
        0
    }
    /// resume.py:1472-1480 slice_string(str, start, length) →
    /// `funcptr_for_oopspec(OS_STR_SLICE)(str, start, stop)` where the
    /// caller pre-computes `stop = start + length` (the OS_STR_SLICE
    /// oopspec signature).  Implementations resolve the funcptr via
    /// `callinfocollection`.
    fn os_str_slice(&self, str: i64, start: i64, stop: i64) -> i64 {
        let _ = (str, start, stop);
        0
    }
    /// resume.py:1482-1483 allocate_unicode(length) →
    /// cpu.bh_newunicode(length)
    fn bh_newunicode(&self, length: usize) -> i64 {
        let _ = length;
        0
    }
    /// resume.py:1485-1487 unicode_setitem(str, index, charnum) →
    /// cpu.bh_unicodesetitem(str, index, char)
    fn bh_unicodesetitem(&self, string: i64, index: usize, char: i64) {
        let _ = (string, index, char);
    }
    /// resume.py:1489-1497 concat_unicodes(str1, str2) →
    /// `funcptr_for_oopspec(OS_UNI_CONCAT)(str1, str2)`. Implementations
    /// resolve the funcptr via `callinfocollection`.
    fn os_uni_concat(&self, str1: i64, str2: i64) -> i64 {
        let _ = (str1, str2);
        0
    }
    /// resume.py:1499-1507 slice_unicode(str, start, length) →
    /// `funcptr_for_oopspec(OS_UNI_SLICE)(str, start, stop)` where the
    /// caller pre-computes `stop = start + length`.  Implementations
    /// resolve the funcptr via `callinfocollection`.
    fn os_uni_slice(&self, str: i64, start: i64, stop: i64) -> i64 {
        let _ = (str, start, stop);
        0
    }
    /// resume.py:1452 allocate_raw_buffer(func, size)
    fn allocate_raw_buffer(&self, func: i64, size: usize) -> i64 {
        let _ = (func, size);
        0
    }
    /// resume.py:1547 cpu.bh_raw_store_f(buffer, offset, value, descr) —
    /// float raw store dispatched from setrawbuffer_item when
    /// `descr.is_array_of_floats()`.  `offset` mirrors
    /// `RawBuffer.offsets[i]` and is signed (rawbuffer.py:14).
    fn bh_raw_store_f(
        &self,
        buffer: i64,
        offset: i64,
        value: i64,
        descr: &majit_ir::ArrayDescrInfo,
    ) {
        let _ = (buffer, offset, value, descr);
    }
    /// resume.py:1550 cpu.bh_raw_store_i(buffer, offset, value, descr) —
    /// integer raw store dispatched from setrawbuffer_item (default
    /// branch — descr is not an array of pointers / floats).
    fn bh_raw_store_i(
        &self,
        buffer: i64,
        offset: i64,
        value: i64,
        descr: &majit_ir::ArrayDescrInfo,
    ) {
        let _ = (buffer, offset, value, descr);
    }
    /// `resume.py:1517 cpu.bh_setfield_gc_i(struct, value, descr)` —
    /// integer setfield dispatched from `resume.py:1509-1518 setfield`.
    fn bh_setfield_gc_i(&self, struct_ptr: i64, value: i64, descr_info: &majit_ir::FieldDescrInfo) {
        let _ = (struct_ptr, value, descr_info);
    }
    /// `resume.py:1512 cpu.bh_setfield_gc_r(struct, value, descr)` —
    /// pointer setfield dispatched when `descr.is_pointer_field()`.
    fn bh_setfield_gc_r(&self, struct_ptr: i64, value: i64, descr_info: &majit_ir::FieldDescrInfo) {
        let _ = (struct_ptr, value, descr_info);
    }
    /// `resume.py:1515 cpu.bh_setfield_gc_f(struct, value, descr)` —
    /// float setfield dispatched when `descr.is_float_field()`.
    fn bh_setfield_gc_f(&self, struct_ptr: i64, value: i64, descr_info: &majit_ir::FieldDescrInfo) {
        let _ = (struct_ptr, value, descr_info);
    }
    /// Pyre-specific: box a raw int to a PyObject ref.
    ///
    /// RPython equivalent: cpu.get_ref_value always returns GCREF because
    /// the jitframe stores typed values. Pyre's deadframe is untyped i64;
    /// when a slot typed as Int is read through decode_ref, this method
    /// wraps it into a valid GCREF (W_IntObject).
    fn box_int(&self, value: i64) -> i64 {
        value // default: return raw value (override in pyre allocator)
    }
    /// Pyre-specific: box raw float bits to a PyObject ref.
    fn box_float(&self, value: i64) -> i64 {
        value
    }
}

/// Default no-op allocator.
pub struct NullAllocator;
impl BlackholeAllocator for NullAllocator {}

/// Metainterp-side extension methods on `VirtualInfo` (which lives in
/// majit-backend since the Phase C-1 cascade).  These methods depend
/// on `ResumeDataDirectReader` + `BlackholeAllocator` — both
/// metainterp-specific — so they stay here as a trait extension.
pub trait VirtualInfoBlackholeExt {
    fn is_about_raw(&self) -> bool;
    /// `resume.py:973 if rd_virtual is not None`: detect the
    /// `RdVirtualInfo::Empty` placeholder propagated as a zero-shaped
    /// `VirtualObj` (resume.rs:1446 conversion).  Used by
    /// `force_all_virtuals` to skip slots that PyPy keeps as `None`.
    fn is_empty_placeholder(&self) -> bool;
    fn allocate(
        &self,
        decoder: &mut ResumeDataDirectReader,
        index: usize,
        allocator: &dyn BlackholeAllocator,
    ) -> i64;
    fn allocate_int(
        &self,
        decoder: &mut ResumeDataDirectReader,
        index: usize,
        allocator: &dyn BlackholeAllocator,
    ) -> i64;
}

/// `resume.py:766-775 VStrPlainInfo.allocate` and `resume.py:821-830
/// VUniPlainInfo.allocate` share the same loop body — the only
/// difference is `decoder.allocate_string` vs `allocate_unicode` and
/// `string_setitem` vs `unicode_setitem`.  The pyre helper takes an
/// `is_unicode` flag to dispatch.  `chars` carries one
/// `VirtualFieldSource` per character (the pyre equivalent of RPython's
/// tagged `fieldnums`); `VirtualFieldSource::Uninitialized` matches
/// resume.py's `tagged_eq(charnum, UNINITIALIZED)` skip.
fn vstr_plain_info_allocate(
    decoder: &mut ResumeDataDirectReader,
    index: usize,
    chars: &[VirtualFieldSource],
    is_unicode: bool,
) -> i64 {
    let length = chars.len();
    // resume.py:769 string = decoder.allocate_string(length)
    // resume.py:824 string = decoder.allocate_unicode(length)
    let string = if is_unicode {
        decoder.allocate_unicode(length)
    } else {
        decoder.allocate_string(length)
    };
    // resume.py:770 / 825 decoder.virtuals_cache.set_ptr(index, string)
    decoder.virtuals_cache.set_ptr(index, string);
    for (i, char_source) in chars.iter().enumerate() {
        // resume.py:773 / 828 if not tagged_eq(charnum, UNINITIALIZED)
        if matches!(char_source, VirtualFieldSource::Uninitialized) {
            continue;
        }
        // resume.py:774 / 829 decoder.{string,unicode}_setitem(string, i, charnum)
        if is_unicode {
            decoder.unicode_setitem(string, i, char_source);
        } else {
            decoder.string_setitem(string, i, char_source);
        }
    }
    string
}

/// `resume.py:596-603 AbstractVirtualStructInfo.setfields(decoder, struct)`
/// — iterate fielddescrs/fieldnums and call decoder.setfield per
/// non-UNINITIALIZED entry.  pyre threads the spec-form `FieldDescrInfo`
/// through `bh_setfield_gc_{i,r,f}` directly because the descr Arc
/// is not interned alongside the spec on `VirtualInfo::{VirtualObj,
/// VStruct}.fielddescrs` (a future slice can replace `Vec<FieldDescrInfo>`
/// with `Vec<Arc<dyn FieldDescr>>` so this helper can call
/// `decoder.setfield(struct, num, descr)` byte-for-byte with RPython).
fn abstract_virtual_struct_info_setfields(
    decoder: &mut ResumeDataDirectReader,
    allocator: &dyn BlackholeAllocator,
    index: usize,
    fielddescrs: &[majit_ir::FieldDescrInfo],
    fields: &[(u32, VirtualFieldSource)],
) {
    for (i, (_field_descr, source)) in fields.iter().enumerate() {
        let Some(descr_info) = fielddescrs.get(i) else {
            continue;
        };
        // resume.py:601 if not tagged_eq(num, UNINITIALIZED)
        if matches!(source, VirtualFieldSource::Uninitialized) {
            continue;
        }
        // resume.py:602 decoder.setfield(struct, num, descr)
        // — pyre dispatches by descr_info.field_type because pyre's
        //   fielddescrs collection holds the spec form, not the live
        //   FieldDescr Arc that RPython passes to decoder.setfield.
        //
        // decode_field_source* may materialize a nested virtual, whose
        // allocation can trigger a minor collection that relocates this
        // struct.  RPython's `struct` local is GC-traced and forwarded in
        // place; pyre's is a raw i64, so re-read the forwarded pointer from
        // the rooted virtuals_ptr_cache[index] slot after each decode and
        // before the write (see getvirtual_ptr at the `bad cache` comment).
        match descr_info.field_type {
            majit_ir::Type::Ref => {
                let value = decoder.decode_field_source(source);
                let struct_ptr = decoder.virtuals_cache.get_ptr(index);
                allocator.bh_setfield_gc_r(struct_ptr, value, descr_info);
            }
            majit_ir::Type::Float => {
                let value = decoder.decode_field_source_float(source);
                let struct_ptr = decoder.virtuals_cache.get_ptr(index);
                allocator.bh_setfield_gc_f(struct_ptr, value, descr_info);
            }
            _ => {
                let value = decoder.decode_field_source_int(source);
                let struct_ptr = decoder.virtuals_cache.get_ptr(index);
                allocator.bh_setfield_gc_i(struct_ptr, value, descr_info);
            }
        }
    }
}

impl VirtualInfoBlackholeExt for VirtualInfo {
    /// resume.py:576 kind attribute — REF for object/struct/array/string,
    /// INT for raw buffers.
    fn is_about_raw(&self) -> bool {
        matches!(
            self,
            VirtualInfo::VRawBuffer { .. } | VirtualInfo::VRawSlice { .. }
        )
    }

    fn is_empty_placeholder(&self) -> bool {
        // The `RdVirtualInfo::Empty` → `VirtualObj` conversion at
        // `resume.rs:1446` produces this exact shape.
        matches!(
            self,
            VirtualInfo::VirtualObj {
                descr: None,
                type_id: 0,
                known_class: None,
                fields,
                fielddescrs,
                descr_size: 0,
            } if fields.is_empty() && fielddescrs.is_empty()
        )
    }

    /// resume.py:618/634/650 allocate(decoder, index)
    ///
    /// Allocate a virtual object and fill in its fields from the decoder.
    /// Sets virtuals_cache_ptr[index] before filling fields (for recursive refs).
    fn allocate(
        &self,
        decoder: &mut ResumeDataDirectReader,
        index: usize,
        allocator: &dyn BlackholeAllocator,
    ) -> i64 {
        match self {
            VirtualInfo::VirtualObj {
                fields,
                fielddescrs,
                descr,
                known_class,
                ..
            } => {
                // resume.py:619 struct = decoder.allocate_with_vtable(descr=self.descr)
                let vtable = known_class.unwrap_or(0) as usize;
                let obj = descr
                    .as_ref()
                    .map(|d| decoder.allocate_with_vtable(d, vtable))
                    .unwrap_or(0);
                decoder.virtuals_cache.set_ptr(index, obj);
                // resume.py:621 return self.setfields(decoder, struct)
                abstract_virtual_struct_info_setfields(
                    decoder,
                    allocator,
                    index,
                    fielddescrs,
                    fields,
                );
                // re-read the forwarded pointer (setfields may have triggered
                // a relocating minor collection); return the live cache slot.
                decoder.virtuals_cache.get_ptr(index)
            }
            VirtualInfo::VStruct {
                typedescr,
                fields,
                fielddescrs,
                ..
            } => {
                // resume.py:635 struct = decoder.allocate_struct(self.typedescr)
                let obj = typedescr
                    .as_ref()
                    .map(|d| decoder.allocate_struct(d))
                    .unwrap_or(0);
                decoder.virtuals_cache.set_ptr(index, obj);
                // resume.py:637 return self.setfields(decoder, struct)
                abstract_virtual_struct_info_setfields(
                    decoder,
                    allocator,
                    index,
                    fielddescrs,
                    fields,
                );
                // re-read the forwarded pointer (setfields may have triggered
                // a relocating minor collection); return the live cache slot.
                decoder.virtuals_cache.get_ptr(index)
            }
            VirtualInfo::VArray {
                arraydescr,
                clear,
                items,
                ..
            } => {
                let length = items.len();
                // resume.py:653: array = decoder.allocate_array(length, arraydescr, self.clear)
                let array = arraydescr
                    .as_ref()
                    .map(|d| decoder.allocate_array(length, d, *clear))
                    .unwrap_or(0);
                decoder.virtuals_cache.set_ptr(index, array);
                // resume.py:656-670: dispatch by arraydescr element type
                let is_pointers = arraydescr
                    .as_ref()
                    .and_then(|d| d.as_array_descr())
                    .map_or(false, |ad| ad.is_array_of_pointers());
                let is_floats = arraydescr
                    .as_ref()
                    .and_then(|d| d.as_array_descr())
                    .map_or(false, |ad| ad.is_array_of_floats());
                if let Some(ad) = arraydescr.as_ref() {
                    for (i, source) in items.iter().enumerate() {
                        // decode_field_source may materialize a nested virtual
                        // and relocate this array; re-read the forwarded
                        // pointer from the rooted cache slot before the write
                        // (same hazard as abstract_virtual_struct_info_setfields).
                        if is_pointers {
                            // resume.py:659: decoder.bh_setarrayitem_gc_r(array, i, num, arraydescr)
                            let value = decoder.decode_field_source(source);
                            let array = decoder.virtuals_cache.get_ptr(index);
                            allocator.bh_setarrayitem_gc_r(array, i, value, ad);
                        } else if is_floats {
                            // resume.py:664: decoder.bh_setarrayitem_gc_f(array, i, num, arraydescr)
                            let value = decoder.decode_field_source_float(source);
                            let array = decoder.virtuals_cache.get_ptr(index);
                            allocator.bh_setarrayitem_gc_f(array, i, value, ad);
                        } else {
                            // resume.py:669: decoder.bh_setarrayitem_gc_i(array, i, num, arraydescr)
                            let value = decoder.decode_field_source_int(source);
                            let array = decoder.virtuals_cache.get_ptr(index);
                            allocator.bh_setarrayitem_gc_i(array, i, value, ad);
                        }
                    }
                }
                decoder.virtuals_cache.get_ptr(index)
            }
            // resume.py:748-760: VArrayStructInfo.allocate
            VirtualInfo::VArrayStruct {
                arraydescr,
                fielddescrs,
                element_fields,
                ..
            } => {
                let size = element_fields.len();
                // resume.py:749: array = decoder.allocate_array(self.size, self.arraydescr, clear=True)
                let array = arraydescr
                    .as_ref()
                    .map(|d| decoder.allocate_array(size, d, /* clear */ true))
                    .unwrap_or(0);
                decoder.virtuals_cache.set_ptr(index, array);
                // resume.py:752-759:
                //   for i in range(self.size):
                //       for j in range(len(self.fielddescrs)):
                //           num = self.fieldnums[p]
                //           if not tagged_eq(num, UNINITIALIZED):
                //               decoder.setinteriorfield(i, array, num, self.fielddescrs[j])
                //           p += 1
                for (i, fields) in element_fields.iter().enumerate() {
                    debug_assert_eq!(
                        fields.len(),
                        fielddescrs.len(),
                        "VArrayStruct element_fields[{i}] has {} fields but {} fielddescrs",
                        fields.len(),
                        fielddescrs.len()
                    );
                    for (j, &(_, ref source)) in fields.iter().enumerate() {
                        if matches!(source, VirtualFieldSource::Uninitialized) {
                            continue;
                        }
                        // resume.py:757: decoder.setinteriorfield(i, array, num, self.fielddescrs[j])
                        decoder.setinteriorfield(i, index, source, &fielddescrs[j], allocator);
                    }
                }
                decoder.virtuals_cache.get_ptr(index)
            }
            // resume.py:766-775 VStrPlainInfo.allocate
            VirtualInfo::VStrPlain { chars } => {
                vstr_plain_info_allocate(decoder, index, chars, /* is_unicode */ false)
            }
            // resume.py:821-830 VUniPlainInfo.allocate
            VirtualInfo::VUniPlain { chars } => {
                vstr_plain_info_allocate(decoder, index, chars, /* is_unicode */ true)
            }
            // resume.py:786-793 VStrConcatInfo.allocate
            VirtualInfo::VStrConcat { left, right } => {
                let string = decoder.concat_strings(left, right);
                decoder.virtuals_cache.set_ptr(index, string);
                string
            }
            // resume.py:805-809 VStrSliceInfo.allocate
            VirtualInfo::VStrSlice {
                source,
                start,
                length,
            } => {
                let string = decoder.slice_string(source, start, length);
                decoder.virtuals_cache.set_ptr(index, string);
                string
            }
            // resume.py:841-848 VUniConcatInfo.allocate
            VirtualInfo::VUniConcat { left, right } => {
                let string = decoder.concat_unicodes(left, right);
                decoder.virtuals_cache.set_ptr(index, string);
                string
            }
            // resume.py:860-864 VUniSliceInfo.allocate
            VirtualInfo::VUniSlice {
                source,
                start,
                length,
            } => {
                let string = decoder.slice_unicode(source, start, length);
                decoder.virtuals_cache.set_ptr(index, string);
                string
            }
            _ => {
                decoder.virtuals_cache.set_ptr(index, 0);
                0
            }
        }
    }

    /// resume.py:701 VRawBufferInfo.allocate_int / VRawSliceInfo.allocate_int
    fn allocate_int(
        &self,
        decoder: &mut ResumeDataDirectReader,
        index: usize,
        allocator: &dyn BlackholeAllocator,
    ) -> i64 {
        match self {
            VirtualInfo::VRawBuffer {
                func,
                size,
                offsets,
                descrs,
                values,
            } => {
                assert_eq!(offsets.len(), descrs.len());
                assert_eq!(offsets.len(), values.len());
                // resume.py:703: buffer = decoder.allocate_raw_buffer(self.func, self.size)
                let buffer = decoder.allocate_raw_buffer(*func, *size);
                // resume.py:704
                decoder.virtuals_cache.set_int(index, buffer);
                // resume.py:705-708: for i in range(len(self.offsets)):
                //     offset = self.offsets[i]; descr = self.descrs[i]
                //     decoder.setrawbuffer_item(buffer, fieldnums[i], offset, descr)
                //
                // Pyre stores fieldnums[i] as a tagged
                // VirtualFieldSource (the value lives on the virtual
                // layout rather than in the resume tape), so we encode
                // it back into the i16 charnum the dispatcher accepts
                // via decode_field_source_{float,int} → tag.
                for i in 0..offsets.len() {
                    let descr = &descrs[i];
                    let source = &values[i];
                    if matches!(source, VirtualFieldSource::Uninitialized) {
                        continue;
                    }
                    // pyre extracts the per-entry value from the virtual
                    // layout instead of the tagged fieldnum, then writes
                    // through bh_raw_store_{i,f} per descr kind — same
                    // dispatch as resume.py:1545-1550 setrawbuffer_item.
                    assert!(
                        descr.item_type != 0,
                        "raw buffer entry must not be pointer type"
                    );
                    if descr.item_type == 2 {
                        let value = decoder.decode_field_source_float(source);
                        allocator.bh_raw_store_f(buffer, offsets[i], value, descr);
                    } else {
                        let value = decoder.decode_field_source_int(source);
                        allocator.bh_raw_store_i(buffer, offsets[i], value, descr);
                    }
                }
                buffer
            }
            VirtualInfo::VRawSlice { offset, parent } => {
                // resume.py:723-725 — parent is an INT virtual (raw buffer)
                let parent_val = decoder.decode_field_source_int(parent);
                let result = parent_val + *offset;
                decoder.virtuals_cache.set_int(index, result);
                result
            }
            _ => panic!("allocate_int called on non-raw virtual"),
        }
    }
}

impl<'a> ResumeDataDirectReader<'a> {
    /// resume.py:1364 __init__
    pub fn new(
        rd_numb: &'a [u8],
        rd_consts: &'a [majit_ir::Const],
        all_liveness: &'a [u8],
        deadframe: &'a [i64],
        deadframe_types: Option<&'a [majit_ir::Type]>,
        all_virtuals: Option<(Vec<i64>, Vec<i64>)>,
        allocator: &'a dyn BlackholeAllocator,
    ) -> Self {
        // resume.py:915-922 _init
        let mut resumecodereader = Reader::new(rd_numb);
        let items_resume_section = resumecodereader.next_item();
        let count = resumecodereader.next_item();

        // resume.py:1368-1376
        let (resume_after_guard_not_forced, virtuals_cache) =
            if let Some((ptrs, ints)) = all_virtuals {
                // resume.py:1373-1374: special case for GUARD_NOT_FORCED
                (2, VirtualCache::from_caches(ptrs, ints))
            } else {
                (0, VirtualCache::new())
            };

        ResumeDataDirectReader {
            resumecodereader,
            items_resume_section,
            count,
            consts: rd_consts,
            deadframe,
            deadframe_types,
            resume_after_guard_not_forced,
            rd_virtuals: None,
            virtuals_cache,
            allocator,
            all_liveness,
            virtualizable_ptr: 0,
        }
    }

    /// resume.py:924 _prepare — init virtuals and pending fields.
    pub fn prepare(
        &mut self,
        rd_virtuals: Option<&'a [VirtualInfo]>,
        rd_guard_pendingfields: Option<&[majit_ir::GuardPendingFieldEntry]>,
    ) {
        // resume.py:925
        self.prepare_virtuals(rd_virtuals);
        // resume.py:926
        if let Some(guard_pf) = rd_guard_pendingfields {
            self.prepare_guard_pendingfields(guard_pf);
        }
    }

    /// resume.py:993 _prepare_pendingfields — variant for GuardPendingFieldEntry.
    ///
    /// RPython encodes pendingfield target/value as tagged values (TAGBOX/
    /// TAGCONST/TAGVIRTUAL) via _gettagged in _add_pending_fields
    /// (resume.py:548-549). At restore time, decode_ref(num) resolves
    /// the tagged value to a concrete pointer.
    ///
    /// `target_tagged` / `value_tagged` are always populated by the
    /// time pendingfields reach this method:
    ///
    ///   * The single production constructor at
    ///     `optimizeopt/optimizer.rs:3453` initializes both to
    ///     `UNASSIGNED`, but the entries are immediately fed through
    ///     `memo.finish()` (`optimizeopt/mod.rs:3390`) →
    ///     `_add_pending_fields` (`resume.rs:3554`), which writes
    ///     the tags from `_gettagged`.
    ///   * The sharing path (`_copy_resume_data_from`) routes resume
    ///     reads through `ResumeGuardCopiedDescr.prev` (compile.py:849
    ///     `get_resumestorage(): return prev`); readers reach the
    ///     donor's already-tagged entries via the descr's `prev`
    ///     pointer rather than touching the shared op directly.
    ///
    /// Hence the only branch is the RPython tagged path; an
    /// `UNASSIGNED` entry escaping into restore-time would be a
    /// soundness bug, so this method panics matching RPython's
    /// implicit invariant (`resume.py:1002` indexes the tagged
    /// number with `decode_ref`).
    fn prepare_guard_pendingfields(&mut self, pendingfields: &[majit_ir::GuardPendingFieldEntry]) {
        for pf in pendingfields {
            // resume.py:1000 PENDINGFIELDSTRUCT.lldescr parity:
            // derive (offset, size, type) from the descr (FieldDescr or
            // ArrayDescr) at consume time. RPython always carries
            // lldescr — pyre's producer at optimizer.rs:3389 mirrors
            // this by setting `pf.descr = pf_op.descr.clone()` for
            // every pending field (pf_op is always a Setfield_gc /
            // Setarrayitem_gc op with a descr).
            let descr = pf
                .descr
                .as_ref()
                .expect("resume.py:1000 PENDINGFIELDSTRUCT.lldescr must be set");
            let field_info = if let Some(fd) = descr.as_field_descr() {
                Some((fd.offset(), fd.field_size(), fd.field_type()))
            } else if descr.as_array_descr().is_some() {
                None
            } else {
                panic!(
                    "pending field descr must be FieldDescr or ArrayDescr (descr={:?})",
                    descr,
                );
            };
            // resume.py:1002-1007 tagged path. UNASSIGNED tags must
            // never reach this method; see doc comment above.
            assert!(
                pf.target_tagged != UNASSIGNED && pf.value_tagged != UNASSIGNED,
                "GuardPendingFieldEntry reached prepare_guard_pendingfields with \
                 UNASSIGNED tag (target_tagged={}, value_tagged={}, descr={:?}); \
                 _add_pending_fields must have run before restore time",
                pf.target_tagged,
                pf.value_tagged,
                descr,
            );
            // resume.py:1002: struct = self.decode_ref(num)
            let struct_ptr = self.decode_ref(pf.target_tagged);

            if pf.item_index < 0 {
                let _ = field_info; // setfield dispatcher reads descr directly.
                // resume.py:1005: self.setfield(struct, fieldnum, descr)
                self.setfield(struct_ptr, pf.value_tagged, descr);
            } else {
                // resume.py:1007: self.setarrayitem(struct, itemindex,
                //                  fieldnum, descr).
                let index = pf.item_index as usize;
                self.setarrayitem(struct_ptr, index, pf.value_tagged, descr);
            }
        }
    }

    /// `resume.py:1509-1518 setfield(struct, fieldnum, descr)` dispatcher:
    /// forwards to `bh_setfield_gc_r` / `bh_setfield_gc_f` /
    /// `bh_setfield_gc_i` based on the field's value kind.  `fieldnum` is
    /// the resume.py-tagged value to decode (decode_ref / decode_float /
    /// decode_int per kind).
    ///
    /// Routes on `field_type()` (the value kind the optimizer numbered the
    /// stored box as) rather than `is_pointer_field()`.  In RPython these
    /// coincide — `FLAG_POINTER` is set iff the field is a GC `Ptr`, so
    /// `is_pointer_field()` ⟺ `field_type == Ref`.  Pyre overloads the
    /// flag for the write barrier: `ExecutionContext.sys_exc_value`
    /// (`pyre-jit-trace/src/descr.rs`) carries `field_type = Ref` (so the
    /// optimizer tracks and numbers the value as a GC ref) but a
    /// non-pointer flag (so `rewrite.rs handle_write_barrier_setfield`
    /// emits no barrier into the non-GC EC root).  The resume decode must
    /// match how the value was numbered — by `field_type` — or a Ref value
    /// (here a virtual exception object) is mis-decoded via `decode_int`
    /// → `getvirtual_int` on a non-raw virtual.
    pub fn setfield(&mut self, struct_ptr: i64, fieldnum: i16, descr: &majit_ir::DescrRef) {
        let fd = descr
            .as_field_descr()
            .expect("resume.py:1509 setfield requires FieldDescr");
        let descr_info = majit_ir::FieldDescrInfo {
            index: descr.index(),
            offset: fd.offset(),
            field_type: fd.field_type(),
            field_size: fd.field_size(),
        };
        if fd.field_type() == majit_ir::Type::Ref {
            // resume.py:1511 newvalue = self.decode_ref(fieldnum)
            // resume.py:1512 self.cpu.bh_setfield_gc_r(struct, newvalue, descr)
            let value = self.decode_ref(fieldnum);
            self.allocator
                .bh_setfield_gc_r(struct_ptr, value, &descr_info);
        } else if fd.is_float_field() {
            // resume.py:1514 newvalue = self.decode_float(fieldnum)
            // resume.py:1515 self.cpu.bh_setfield_gc_f(struct, newvalue, descr)
            let value = self.decode_float(fieldnum);
            self.allocator
                .bh_setfield_gc_f(struct_ptr, value, &descr_info);
        } else {
            // resume.py:1517 newvalue = self.decode_int(fieldnum)
            // resume.py:1518 self.cpu.bh_setfield_gc_i(struct, newvalue, descr)
            let value = self.decode_int(fieldnum);
            self.allocator
                .bh_setfield_gc_i(struct_ptr, value, &descr_info);
        }
    }

    /// resume.py:1437-1439 allocate_with_vtable(descr) →
    /// `executor.exec_new_with_vtable(self.cpu, descr)` — pyre's
    /// `vtable` argument is the resolved class pointer carried on the
    /// virtual layout (info.py:318 _known_class).
    pub fn allocate_with_vtable(&self, descr: &majit_ir::DescrRef, vtable: usize) -> i64 {
        self.allocator.allocate_with_vtable(descr, vtable)
    }

    /// resume.py:1441-1442 allocate_struct(typedescr) → cpu.bh_new(typedescr).
    pub fn allocate_struct(&self, typedescr: &majit_ir::DescrRef) -> i64 {
        self.allocator.bh_new(typedescr)
    }

    /// resume.py:1444-1447 allocate_array(length, arraydescr, clear) →
    /// `cpu.bh_new_array_clear` (clear=True) or `cpu.bh_new_array`.
    pub fn allocate_array(
        &self,
        length: usize,
        arraydescr: &majit_ir::DescrRef,
        clear: bool,
    ) -> i64 {
        if clear {
            self.allocator.bh_new_array_clear(length, arraydescr)
        } else {
            self.allocator.bh_new_array(length, arraydescr)
        }
    }

    /// resume.py:1452-1456 allocate_raw_buffer(func, size) → calldescr =
    /// `callinfo_for_oopspec(OS_RAW_MALLOC_VARSIZE_CHAR)`,
    /// `cpu.bh_call_i(func, [size], None, None, calldescr)` — pyre's
    /// `BlackholeAllocator::allocate_raw_buffer` keeps the wrapped form
    /// (the calldescr / cic resolution lives inside the allocator impl
    /// because pyre lacks a callinfocollection on the reader side).
    pub fn allocate_raw_buffer(&self, func: i64, size: usize) -> i64 {
        self.allocator.allocate_raw_buffer(func, size)
    }

    /// resume.py:1449-1450 allocate_string(length) — forward to allocator.
    pub fn allocate_string(&self, length: usize) -> i64 {
        self.allocator.bh_newstr(length)
    }

    /// resume.py:1458-1460 string_setitem(str, index, charnum) — decode
    /// the per-character source and forward to the allocator.  Pyre
    /// threads a `VirtualFieldSource` where resume.py threads a tagged
    /// i16 charnum; the structural shape matches because both decoders
    /// resolve to an integer character value before calling
    /// bh_strsetitem.
    pub fn string_setitem(&mut self, string: i64, index: usize, source: &VirtualFieldSource) {
        let char = self.decode_field_source_int(source);
        self.allocator.bh_strsetitem(string, index, char);
    }

    /// resume.py:1462-1470 concat_strings(str1num, str2num) — decode
    /// the two ref sources and dispatch to OS_STR_CONCAT.  The funcptr
    /// is resolved by the allocator via
    /// `callinfocollection.funcptr_for_oopspec(OS_STR_CONCAT)`
    /// (resume.py:1467-1468); the variant carries no funcptr.
    pub fn concat_strings(
        &mut self,
        str1_source: &VirtualFieldSource,
        str2_source: &VirtualFieldSource,
    ) -> i64 {
        let str1 = self.decode_field_source(str1_source);
        let str2 = self.decode_field_source(str2_source);
        self.allocator.os_str_concat(str1, str2)
    }

    /// resume.py:1472-1480 slice_string(strnum, startnum, lengthnum) →
    /// OS_STR_SLICE funcptr(str, start, start + length).  Funcptr is
    /// resolved by the allocator via `callinfocollection`.
    pub fn slice_string(
        &mut self,
        str_source: &VirtualFieldSource,
        start_source: &VirtualFieldSource,
        length_source: &VirtualFieldSource,
    ) -> i64 {
        let str = self.decode_field_source(str_source);
        let start = self.decode_field_source_int(start_source);
        let length = self.decode_field_source_int(length_source);
        let stop = start.wrapping_add(length);
        self.allocator.os_str_slice(str, start, stop)
    }

    /// resume.py:1482-1483 allocate_unicode(length) → cpu.bh_newunicode.
    pub fn allocate_unicode(&self, length: usize) -> i64 {
        self.allocator.bh_newunicode(length)
    }

    /// resume.py:1485-1487 unicode_setitem(str, index, charnum) — same
    /// shape as string_setitem.
    pub fn unicode_setitem(&mut self, string: i64, index: usize, source: &VirtualFieldSource) {
        let char = self.decode_field_source_int(source);
        self.allocator.bh_unicodesetitem(string, index, char);
    }

    /// resume.py:1489-1497 concat_unicodes(str1num, str2num).  Funcptr
    /// is resolved by the allocator via `callinfocollection`.
    pub fn concat_unicodes(
        &mut self,
        str1_source: &VirtualFieldSource,
        str2_source: &VirtualFieldSource,
    ) -> i64 {
        let str1 = self.decode_field_source(str1_source);
        let str2 = self.decode_field_source(str2_source);
        self.allocator.os_uni_concat(str1, str2)
    }

    /// resume.py:1543-1550 setrawbuffer_item(buffer, fieldnum, offset,
    /// descr) dispatcher: `assert not descr.is_array_of_pointers()`,
    /// then dispatch to `bh_raw_store_f` (float) or `bh_raw_store_i`
    /// (default).
    pub fn setrawbuffer_item(
        &mut self,
        buffer: i64,
        fieldnum: i16,
        offset: i64,
        descr: &majit_ir::ArrayDescrInfo,
    ) {
        // resume.py:1544 assert not descr.is_array_of_pointers()
        assert!(
            descr.item_type != 0,
            "setrawbuffer_item: descr must not be array_of_pointers"
        );
        if descr.item_type == 2 {
            // resume.py:1546-1547 newvalue = self.decode_float(fieldnum)
            //                    self.cpu.bh_raw_store_f(...)
            let value = self.decode_float(fieldnum);
            self.allocator.bh_raw_store_f(buffer, offset, value, descr);
        } else {
            // resume.py:1549-1550 newvalue = self.decode_int(fieldnum)
            //                    self.cpu.bh_raw_store_i(...)
            let value = self.decode_int(fieldnum);
            self.allocator.bh_raw_store_i(buffer, offset, value, descr);
        }
    }

    /// resume.py:1499-1507 slice_unicode(strnum, startnum, lengthnum).
    /// Funcptr resolved by the allocator via `callinfocollection`.
    pub fn slice_unicode(
        &mut self,
        str_source: &VirtualFieldSource,
        start_source: &VirtualFieldSource,
        length_source: &VirtualFieldSource,
    ) -> i64 {
        let str = self.decode_field_source(str_source);
        let start = self.decode_field_source_int(start_source);
        let length = self.decode_field_source_int(length_source);
        let stop = start.wrapping_add(length);
        self.allocator.os_uni_slice(str, start, stop)
    }

    /// `resume.py:1009-1015 setarrayitem(array, index, fieldnum, descr)`
    /// dispatcher: forwards to `setarrayitem_ref` /
    /// `setarrayitem_float` / `setarrayitem_int` based on the live
    /// `arraydescr.is_array_of_pointers()` / `is_array_of_floats()`
    /// methods.  `fieldnum` is the resume.py-tagged value to decode.
    pub fn setarrayitem(
        &mut self,
        array: i64,
        index: usize,
        fieldnum: i16,
        arraydescr: &majit_ir::DescrRef,
    ) {
        let ad = arraydescr
            .as_array_descr()
            .expect("resume.py:1009 setarrayitem requires ArrayDescr");
        if ad.is_array_of_pointers() {
            // resume.py:1011 self.bh_setarrayitem_gc_r(array, index, fieldnum, arraydescr)
            let value = self.decode_ref(fieldnum);
            self.allocator
                .bh_setarrayitem_gc_r(array, index, value, arraydescr);
        } else if ad.is_array_of_floats() {
            // resume.py:1013 self.bh_setarrayitem_gc_f(array, index, fieldnum, arraydescr)
            let value = self.decode_float(fieldnum);
            self.allocator
                .bh_setarrayitem_gc_f(array, index, value, arraydescr);
        } else {
            // resume.py:1015 self.bh_setarrayitem_gc_i(array, index, fieldnum, arraydescr)
            let value = self.decode_int(fieldnum);
            self.allocator
                .bh_setarrayitem_gc_i(array, index, value, arraydescr);
        }
    }

    /// resume.py:1378 handling_async_forcing
    pub fn handling_async_forcing(&mut self) {
        self.resume_after_guard_not_forced = 1;
    }

    // ---- AbstractResumeDataReader methods (resume.py:928-1038) ----

    /// resume.py:928 read_jitcode_pos_pc.  Returns
    /// `(jitcode_pos, pc, jitcode_pc)`; `jitcode_pc` is the direct JitCode
    /// resume coordinate or `NO_JITCODE_PC` (see
    /// `majit_ir::resumedata::NO_JITCODE_PC`).
    pub fn read_jitcode_pos_pc(&mut self) -> (i32, i32, i32) {
        let jitcode_pos = self.resumecodereader.next_item();
        let pc = self.resumecodereader.next_item();
        let jitcode_pc = self.resumecodereader.next_item();
        (jitcode_pos, pc, jitcode_pc)
    }

    /// resume.py:933 next_int
    pub fn next_int(&mut self) -> i64 {
        let tagged = self.resumecodereader.next_item() as i16;
        self.decode_int(tagged)
    }

    /// resume.py:936 next_ref
    pub fn next_ref(&mut self) -> i64 {
        let tagged = self.resumecodereader.next_item() as i16;
        self.decode_ref(tagged)
    }

    /// resume.py:939 next_float
    pub fn next_float(&mut self) -> i64 {
        let tagged = self.resumecodereader.next_item() as i16;
        self.decode_float(tagged)
    }

    /// resume.py:1410-1421 load_next_value_of_type
    pub fn next_value_of_type(&mut self, tp: majit_ir::Type) -> i64 {
        match tp {
            majit_ir::Type::Int => self.next_int(),
            majit_ir::Type::Ref => self.next_ref(),
            majit_ir::Type::Float => self.next_float(),
            other => panic!("load_next_value_of_type: unexpected type {other:?}"),
        }
    }

    /// resume.py:942 done_reading
    pub fn done_reading(&self) -> bool {
        self.resumecodereader.items_read >= self.items_resume_section as usize
    }

    /// resume.py:945 getvirtual_ptr
    ///
    /// Returns the index'th virtual, building it lazily if needed.
    /// Note that this may be called recursively; that's why the
    /// allocate() methods must fill in the cache as soon as they
    /// have the object, before they fill its fields.
    pub fn getvirtual_ptr(&mut self, index: usize) -> i64 {
        // resume.py:950: assert self.virtuals_cache is not None
        assert!(
            !self.virtuals_cache.is_empty(),
            "getvirtual_ptr: virtuals_cache is empty (rd_virtuals not prepared)"
        );
        // resume.py:951-952
        let v = self.virtuals_cache.get_ptr(index);
        if v != 0 {
            return v;
        }
        // resume.py:953-955: lazy allocation
        assert!(self.rd_virtuals.is_some(), "rd_virtuals is None");
        // Safety: rd_virtuals is an immutable slice reference that we need to
        // read while mutating virtuals_cache through self. The slice data is
        // never modified by allocate(), only the cache vectors are written.
        let rd_virtuals_ptr = self.rd_virtuals.unwrap().as_ptr();
        let rd_virtuals_len = self.rd_virtuals.unwrap().len();
        let vinfo = unsafe { &*rd_virtuals_ptr.add(index) };
        debug_assert!(index < rd_virtuals_len);
        let allocator = self.allocator as *const dyn BlackholeAllocator;
        // resume.py:954 `v = self.rd_virtuals[index].allocate(self, index)`.
        // RPython returns `allocate`'s result and asserts `v == cache`,
        // relying on `v` being a GC-traced stack local that a minor
        // collection triggered while `allocate` fills the virtual's fields
        // keeps equal to the (also-rooted) cache slot.  Pyre's `v` is a raw
        // i64 — not a GC root — so such a collection forwards the rooted
        // `virtuals_ptr_cache` slot in place (rooted by
        // `blackhole_from_resumedata`) but leaves `v` pointing into
        // from-space.  Return the live cache slot so callers (e.g. the
        // pending-field `setfield` that publishes the exception into
        // `EC.sys_exc_value`) store the forwarded pointer, not a dangling
        // from-space one.
        vinfo.allocate(self, index, unsafe { &*allocator });
        self.virtuals_cache.get_ptr(index)
    }

    /// resume.py:958 getvirtual_int
    pub fn getvirtual_int(&mut self, index: usize) -> i64 {
        // resume.py:959: assert self.virtuals_cache is not None
        assert!(
            !self.virtuals_cache.is_empty(),
            "getvirtual_int: virtuals_cache is empty (rd_virtuals not prepared)"
        );
        // resume.py:960-961
        let v = self.virtuals_cache.get_int(index);
        if v != 0 {
            return v;
        }
        // resume.py:962-966
        assert!(self.rd_virtuals.is_some(), "rd_virtuals is None");
        let rd_virtuals_ptr = self.rd_virtuals.unwrap().as_ptr();
        let vinfo = unsafe { &*rd_virtuals_ptr.add(index) };
        assert!(vinfo.is_about_raw(), "getvirtual_int: not a raw virtual");
        let allocator = self.allocator as *const dyn BlackholeAllocator;
        let v = vinfo.allocate_int(self, index, unsafe { &*allocator });
        debug_assert_eq!(
            v,
            self.virtuals_cache.get_int(index),
            "resume.py: bad cache"
        );
        v
    }

    /// resume.py:969 force_all_virtuals
    pub fn force_all_virtuals(&mut self) -> (&[i64], &[i64]) {
        if let Some(rd_virtuals) = self.rd_virtuals {
            for i in 0..rd_virtuals.len() {
                let rd_virtual = &rd_virtuals[i];
                // resume.py:973 `if rd_virtual is not None`: skip empty
                // slots (Pyre carries them as the `Empty`-derived
                // placeholder shape).
                if rd_virtual.is_empty_placeholder() {
                    continue;
                }
                if rd_virtual.is_about_raw() {
                    // resume.py:977: kind == INT
                    self.getvirtual_int(i);
                } else {
                    // resume.py:976: kind == REF
                    self.getvirtual_ptr(i);
                }
            }
        }
        (
            &self.virtuals_cache.virtuals_ptr_cache,
            &self.virtuals_cache.virtuals_int_cache,
        )
    }

    /// resume.py:983 _prepare_virtuals
    fn prepare_virtuals(&mut self, virtuals: Option<&'a [VirtualInfo]>) {
        if let Some(v) = virtuals {
            self.rd_virtuals = Some(v);
            // resume.py:990-991
            self.virtuals_cache = VirtualCache::from_caches(vec![0; v.len()], vec![0; v.len()]);
        }
    }

    // ---- ResumeDataDirectReader methods (resume.py:1380-1601) ----

    /// resume.py:1381-1384 `consume_one_section(self, blackholeinterp)`.
    ///
    /// ```python
    /// def consume_one_section(self, blackholeinterp):
    ///     self.blackholeinterp = blackholeinterp
    ///     info = blackholeinterp.get_current_position_info()
    ///     self._prepare_next_section(info)
    /// ```
    pub fn consume_one_section(&mut self, bh: &mut BlackholeInterpreter) {
        // resume.py:1383
        let info = bh.get_current_position_info();
        // resume.py:1384
        self._prepare_next_section(info, bh);
    }

    /// resume.py:1017-1026 `_prepare_next_section(self, info)`.
    ///
    /// ```python
    /// def _prepare_next_section(self, info):
    ///     from rpython.jit.codewriter.jitcode import enumerate_vars
    ///     enumerate_vars(info,
    ///             self.metainterp_sd.liveness_info,
    ///             self._callback_i,
    ///             self._callback_r,
    ///             self._callback_f,
    ///             self.unique_id)
    /// ```
    ///
    /// `self.all_liveness` shadows `self.metainterp_sd.liveness_info` —
    /// the shared packed buffer that `enumerate_vars` indexes with
    /// `info`. The three callbacks still call `next_int`/`next_ref`/
    /// `next_float` on this reader (resume.py:1028-1038), matching
    /// `_callback_i/_callback_r/_callback_f` plus `write_an_int/write_a_ref/
    /// write_a_float` (resume.py:1590-1597).
    fn _prepare_next_section(&mut self, info: usize, bh: &mut BlackholeInterpreter) {
        use majit_translate::liveness::LivenessIterator;

        let all_liveness: &[u8] = self.all_liveness;

        // jitcode.py:149-151 — three length bytes.
        let length_i = all_liveness[info] as u32;
        let length_r = all_liveness[info + 1] as u32;
        let length_f = all_liveness[info + 2] as u32;
        // jitcode.py:152
        let mut offset = info + 3;

        let bh_debug = crate::bh_debug_enabled();
        if bh_debug {
            eprintln!(
                "[bh-section] info={info} length_i={length_i} length_r={length_r} length_f={length_f} \
                 items_read={} items_resume_section={}",
                self.resumecodereader.items_read, self.items_resume_section,
            );
        }
        // resume.py:1028-1030 `_callback_i` / jitcode.py:153-157.
        if length_i != 0 {
            let mut it = LivenessIterator::new(offset, length_i, all_liveness);
            while let Some(reg_idx) = it.next() {
                let value = self.next_int();
                if bh_debug {
                    eprintln!("[bh-seed] i{reg_idx} = {value}");
                }
                // resume.py:1590-1591 `write_an_int`.
                bh.setarg_i(reg_idx as usize, value);
            }
            offset = it.offset;
        }
        // resume.py:1032-1034 `_callback_r` / jitcode.py:158-162.
        if length_r != 0 {
            let mut it = LivenessIterator::new(offset, length_r, all_liveness);
            while let Some(reg_idx) = it.next() {
                let value = self.next_ref();
                if bh_debug {
                    eprintln!("[bh-seed] r{reg_idx} = {value:#x}");
                }
                // resume.py:1593-1594 `write_a_ref`.
                bh.setarg_r(reg_idx as usize, value);
            }
            offset = it.offset;
        }
        // resume.py:1036-1038 `_callback_f` / jitcode.py:163-166.
        if length_f != 0 {
            let mut it = LivenessIterator::new(offset, length_f, all_liveness);
            while let Some(reg_idx) = it.next() {
                let value = self.next_float();
                // resume.py:1596-1597 `write_a_float`.
                bh.setarg_f(reg_idx as usize, value);
            }
            // `offset` is the end of the float section; no further use.
            let _ = offset;
        }
    }

    /// Callback-driven sibling of `_prepare_next_section` — drives the
    /// same `enumerate_vars(info, all_liveness, _callback_i/r/f)`
    /// walk (`resume.py:1017-1026`) but lets the caller decide what to
    /// do with each `(kind, reg_idx, value)` triple.  Three Rust
    /// FnMut closures cannot share `&mut bh` simultaneously (E0524),
    /// so the kind dispatch happens INSIDE the single closure rather
    /// than across three separate ones.  The on-demand cranelift
    /// deopt callback (Slice QQ-2) uses a closure that appends each
    /// value to a flat `Vec<i64>` mirroring the recovery_layout
    /// walker's `rebuilt` output.
    pub fn _prepare_next_section_with(
        &mut self,
        info: usize,
        mut cb: impl FnMut(majit_ir::Type, u32, i64),
    ) {
        use majit_translate::liveness::LivenessIterator;

        // `self.all_liveness` is `&'a [u8]` — copying the reference does
        // not borrow `self`, so the inner `self.next_*` calls below are
        // free to take `&mut self`.
        let all_liveness: &[u8] = self.all_liveness;

        // jitcode.py:149-151 — three length bytes.
        let length_i = all_liveness[info] as u32;
        let length_r = all_liveness[info + 1] as u32;
        let length_f = all_liveness[info + 2] as u32;
        // jitcode.py:152
        let mut offset = info + 3;

        // resume.py:1028-1030 `_callback_i` / jitcode.py:153-157.
        if length_i != 0 {
            let mut it = LivenessIterator::new(offset, length_i, all_liveness);
            while let Some(reg_idx) = it.next() {
                let value = self.next_int();
                cb(majit_ir::Type::Int, reg_idx, value);
            }
            offset = it.offset;
        }
        // resume.py:1032-1034 `_callback_r` / jitcode.py:158-162.
        if length_r != 0 {
            let mut it = LivenessIterator::new(offset, length_r, all_liveness);
            while let Some(reg_idx) = it.next() {
                let value = self.next_ref();
                cb(majit_ir::Type::Ref, reg_idx, value);
            }
            offset = it.offset;
        }
        // resume.py:1036-1038 `_callback_f` / jitcode.py:163-166.
        if length_f != 0 {
            let mut it = LivenessIterator::new(offset, length_f, all_liveness);
            while let Some(reg_idx) = it.next() {
                let value = self.next_float();
                cb(majit_ir::Type::Float, reg_idx, value);
            }
            // `offset` is the end of the float section; no further use.
            let _ = offset;
        }
    }

    /// On-demand variant for cranelift's deopt path: walk the resume
    /// tape section-by-section, append each decoded `(int|ref|float)`
    /// value into a flat `Vec<i64>` (innermost-first concatenation,
    /// matching the existing recovery_layout walker's `rebuilt`
    /// output).  `resolve_jitcode` mirrors `resume.py:1339
    /// jitcode = jitcodes[jitcode_pos]` and returns the per-PC
    /// `op_live` byte that `BlackholeInterpreter::get_current_position_info`
    /// uses to index `all_liveness`.
    ///
    /// Caller is expected to drive `prepare(rd_virtuals,
    /// rd_guard_pendingfields)` + `consume_vref_and_vable` first per
    /// `resume.py:1324-1325 blackhole_from_resumedata`.
    pub fn consume_all_sections_into_vec(
        &mut self,
        resolve_jitcode: &dyn Fn(
            i32,
            i32,
            i32,
        )
            -> Option<(std::sync::Arc<crate::jitcode::JitCode>, usize, u8)>,
        outputs: &mut Vec<i64>,
    ) -> bool {
        while !self.done_reading() {
            // resume.py:1338-1340 read_jitcode_pos_pc.  `#124`: forward the
            // carried direct JitCode pc to the resolver.
            let (jitcode_pos, pc, jitcode_pc) = self.read_jitcode_pos_pc();
            let Some((jitcode, resolved_pc, op_live)) =
                resolve_jitcode(jitcode_pos, pc, jitcode_pc)
            else {
                return false;
            };
            // `blackhole.rs:1435 get_current_position_info` parity —
            // `jitcode.get_live_vars_info(position, op_live)` is the
            // section info offset for the current PC.
            let info = jitcode.get_live_vars_info(resolved_pc, op_live);
            self._prepare_next_section_with(info, |_kind, _reg_idx, value| {
                outputs.push(value);
            });
        }
        true
    }

    /// resume.py:1386 consume_virtualref_info
    pub fn consume_virtualref_info(&mut self, vrefinfo: Option<&dyn VRefInfo>) {
        // resume.py:1389
        let size = self.resumecodereader.next_item();
        // resume.py:1390-1391
        if vrefinfo.is_none() || size == 0 {
            // resume.py:1391: assert size == 0
            assert!(
                size == 0,
                "consume_virtualref_info: vrefinfo is None but size={size} != 0"
            );
            return;
        }
        let vrefinfo = vrefinfo.unwrap();
        // resume.py:1393-1397
        for _i in 0..size {
            let virtual_val = self.next_ref();
            let vref = self.next_ref();
            // resume.py:1397
            vrefinfo.continue_tracing(vref, virtual_val);
        }
    }

    /// resume.py:1399 consume_vable_info
    pub fn consume_vable_info(
        &mut self,
        vinfo: &dyn VirtualizableInfo,
        vable_size: i32,
        identity_override: Option<i64>,
    ) {
        // resume.py:1403
        assert!(vable_size > 0);
        // The vable section is encoded identity-FIRST: the snapshot writer
        // (`_list_of_boxes_virtualizable`, opencoder.py:718-726) reorders
        // `[field0..fieldN, vable]` to `[vable, field0..fieldN]` so the resume
        // reader can pull the virtualizable out before its field payload. So
        // the identity is the first of the `vable_size` items and the field
        // payload the remaining `vable_size - 1`, read sequentially.
        // resume.py:1404 virtualizable = self.next_ref()
        //
        // Consume the encoded identity even when a host supplies an override:
        // it occupies one resume-data item and keeps the reader aligned for
        // the field payload. Heap virtualizables (PyFrame) use this live
        // TAGBOX exactly as RPython does. The state-field macro JIT opts in
        // to `identity_override` because its host-stack `&state` is folded out
        // of backend failargs; at deopt it must use the current call's address,
        // never a trace-time pointer or an unrelated deadframe slot.
        let encoded_identity = self.next_ref();
        let virtualizable = identity_override.unwrap_or(encoded_identity);
        self.virtualizable_ptr = virtualizable;
        // resume.py:1406: assert vinfo.get_total_size(virtualizable) == vable_size - 1
        let expected = vinfo.get_total_size(virtualizable) as i32;
        assert!(
            expected == vable_size - 1,
            "consume_vable_info: vinfo.get_total_size(0x{:x}) = {} != vable_size - 1 = {}",
            virtualizable,
            expected,
            vable_size - 1
        );
        // resume.py:1407
        vinfo.reset_token_gcref(virtualizable);
        // resume.py:1408 write_from_resume_data_partial reads the field
        // payload from the remaining `vable_size - 1` items, leaving the reader
        // positioned just past the vable section for the vref/frame chain.
        vinfo.write_from_resume_data_partial(virtualizable, self);
    }

    /// resume.py:1424 consume_vref_and_vable
    pub fn consume_vref_and_vable(
        &mut self,
        vrefinfo: Option<&dyn VRefInfo>,
        vinfo: Option<&dyn VirtualizableInfo>,
        ginfo: Option<&dyn GreenfieldInfo>,
        identity_override: Option<i64>,
    ) {
        // resume.py:1425
        let vable_size = self.resumecodereader.next_item();

        if self.resume_after_guard_not_forced != 2 {
            // resume.py:1427-1428
            if let Some(vi) = vinfo {
                self.consume_vable_info(vi, vable_size, identity_override);
            }
            // resume.py:1429-1430
            if ginfo.is_some() {
                let _ginfo_item = self.resumecodereader.next_item();
            }
            // resume.py:1431
            self.consume_virtualref_info(vrefinfo);
        } else {
            // resume.py:1433-1435
            self.resumecodereader.jump(vable_size as usize);
            let vref_size = self.resumecodereader.next_item();
            self.resumecodereader.jump(vref_size as usize * 2);
        }
    }

    /// resume.py: TAGVIRTUAL num → rd_virtuals/virtuals_cache index.
    ///
    /// RPython indexes `self.rd_virtuals[num]` / `virtuals_cache.get_*(num)`
    /// directly with a possibly-negative `num` and relies on Python list
    /// negative indexing (cached/nested virtuals get negative nums from
    /// `assign_number_to_virtual`). Rust's getvirtual_* take a `usize`, so
    /// remap here, mirroring `_number_virtuals` (`rd_virtuals.len() + num`).
    fn virtual_index(&self, num: i32) -> usize {
        if num >= 0 {
            num as usize
        } else {
            // On the GUARD_NOT_FORCED path (resume.py:1373-1374) the
            // virtuals arrive preloaded in `virtuals_cache` while
            // `rd_virtuals` stays None; the cache has one slot per
            // virtual, so its length is the same wrap base.
            let len =
                self.rd_virtuals
                    .map_or_else(|| self.virtuals_cache.len(), |v| v.len()) as i32;
            (len + num) as usize
        }
    }

    /// resume.py:1552 decode_int
    pub fn decode_int(&mut self, tagged: i16) -> i64 {
        let (num, tag) = untag(tagged);
        match tag {
            TAGCONST => {
                // resume.py:1555 — ConstInt.getint(): return the i64 value.
                let idx = (num - TAG_CONST_OFFSET) as usize;
                self.consts[idx].getint()
            }
            TAGINT => {
                // resume.py:1557
                num as i64
            }
            TAGVIRTUAL => {
                // resume.py:1559
                let idx = self.virtual_index(num);
                self.getvirtual_int(idx)
            }
            TAGBOX => {
                // resume.py:1561-1564
                let mut idx = num;
                if idx < 0 {
                    idx += self.count;
                }
                self.deadframe[idx as usize]
            }
            _ => unreachable!("bad tag: {tag}"),
        }
    }

    /// resume.py:1566 decode_ref
    pub fn decode_ref(&mut self, tagged: i16) -> i64 {
        let (num, tag) = untag(tagged);
        match tag {
            TAGCONST => {
                // resume.py:1569-1571
                if tagged_eq(tagged, NULLREF) {
                    return 0; // ConstPtr.value (null pointer)
                }
                // resume.py:1571
                let idx = (num - TAG_CONST_OFFSET) as usize;
                // history.py:316 ConstPtr.getref_base() returns the GCREF value.
                self.consts[idx].getref_base().as_usize() as i64
            }
            TAGVIRTUAL => {
                // resume.py:1573
                let idx = self.virtual_index(num);
                self.getvirtual_ptr(idx)
            }
            TAGBOX => {
                // resume.py:1575-1578
                let mut idx = num;
                if idx < 0 {
                    idx += self.count;
                }
                let value = self.deadframe[idx as usize];
                match self
                    .deadframe_types
                    .and_then(|tys| tys.get(idx as usize))
                    .copied()
                    .unwrap_or(majit_ir::Type::Ref)
                {
                    majit_ir::Type::Ref => value,
                    // RPython: decode_ref + TAGBOX always returns a GC
                    // pointer via cpu.get_ref_value(). These Int/Float
                    // branches are needed because the optimizer may
                    // unbox Ref→Int in deadframe slots.
                    majit_ir::Type::Int => self.allocator.box_int(value),
                    majit_ir::Type::Float => self.allocator.box_float(value),
                    majit_ir::Type::Void => value,
                }
            }
            _ => {
                // resume.py:1574 `assert tag == TAGBOX`: in a ref slot
                // only TAGCONST / TAGVIRTUAL / TAGBOX are valid.  TAGINT
                // here means the numbering stage produced an int-tagged
                // entry in a ref position — a producer bug.
                panic!("decode_ref: unexpected tag {tag}")
            }
        }
    }

    /// resume.py:1580 decode_float
    pub fn decode_float(&mut self, tagged: i16) -> i64 {
        let (num, tag) = untag(tagged);
        match tag {
            TAGCONST => {
                // resume.py:1583 — ConstFloat.getfloatstorage(): i64 bits.
                let idx = (num - TAG_CONST_OFFSET) as usize;
                self.consts[idx].getfloatstorage()
            }
            TAGBOX => {
                // resume.py:1585-1588
                let mut idx = num;
                if idx < 0 {
                    idx += self.count;
                }
                self.deadframe[idx as usize]
            }
            _ => {
                // resume.py:1580 — only TAGCONST and TAGBOX valid for floats
                panic!("decode_float: unexpected tag {tag}")
            }
        }
    }

    /// Decode a VirtualFieldSource as a REF value (resume.py:1566 decode_ref).
    ///
    /// Virtual sources go through getvirtual_ptr (REF virtuals).
    pub fn decode_field_source(&mut self, source: &VirtualFieldSource) -> i64 {
        match source {
            ResumeValueSource::FailArg(index) => self.deadframe[*index],
            // resume.py:1568 ConstPtr.getref_base() — the Const carries its type.
            ResumeValueSource::Constant(c) => c.getref_base().as_usize() as i64,
            ResumeValueSource::Virtual(index) => self.getvirtual_ptr(*index),
            ResumeValueSource::Uninitialized => 0,
            ResumeValueSource::Unavailable => 0,
        }
    }

    /// Decode a VirtualFieldSource as an INT value (resume.py:1552 decode_int).
    ///
    /// Virtual sources go through getvirtual_int (INT/raw virtuals).
    pub fn decode_field_source_int(&mut self, source: &VirtualFieldSource) -> i64 {
        match source {
            ResumeValueSource::FailArg(index) => self.deadframe[*index],
            // resume.py:1555 ConstInt.getint().
            ResumeValueSource::Constant(c) => c.getint(),
            ResumeValueSource::Virtual(index) => self.getvirtual_int(*index),
            ResumeValueSource::Uninitialized => 0,
            ResumeValueSource::Unavailable => 0,
        }
    }

    /// Decode a VirtualFieldSource as a FLOAT value (resume.py:1554 decode_float).
    ///
    /// Floats are stored as raw i64 bits. TAGVIRTUAL is invalid for
    /// float fields — virtual floats would route through a different
    /// VirtualInfo variant.
    pub fn decode_field_source_float(&mut self, source: &VirtualFieldSource) -> i64 {
        match source {
            ResumeValueSource::FailArg(index) => self.deadframe[*index],
            // resume.py:1583 ConstFloat.getfloatstorage().
            ResumeValueSource::Constant(c) => c.getfloatstorage(),
            ResumeValueSource::Virtual(_) => {
                panic!("decode_field_source_float: TAGVIRTUAL not valid for float field")
            }
            ResumeValueSource::Uninitialized => 0,
            ResumeValueSource::Unavailable => 0,
        }
    }

    /// resume.py:1520-1529 setinteriorfield(index, array, fieldnum, descr)
    ///
    /// Dispatches by descr.is_pointer_field() / is_float_field() / else.
    pub fn setinteriorfield(
        &mut self,
        index: usize,
        virtual_index: usize,
        source: &VirtualFieldSource,
        descr: &majit_ir::DescrRef,
        allocator: &dyn BlackholeAllocator,
    ) {
        let is_pointer = descr
            .as_interior_field_descr()
            .map_or(false, |ifd| ifd.field_descr().is_pointer_field());
        let is_float = descr
            .as_interior_field_descr()
            .map_or(false, |ifd| ifd.field_descr().is_float_field());
        // decode_field_source* may materialize a nested virtual and relocate
        // the array; re-read the forwarded pointer from the rooted cache slot
        // before the write (same hazard as abstract_virtual_struct_info_setfields).
        if is_pointer {
            let value = self.decode_field_source(source);
            let array = self.virtuals_cache.get_ptr(virtual_index);
            allocator.bh_setinteriorfield_gc_r(array, index, value, descr);
        } else if is_float {
            let value = self.decode_field_source_float(source);
            let array = self.virtuals_cache.get_ptr(virtual_index);
            allocator.bh_setinteriorfield_gc_f(array, index, value, descr);
        } else {
            let value = self.decode_field_source_int(source);
            let array = self.virtuals_cache.get_ptr(virtual_index);
            allocator.bh_setinteriorfield_gc_i(array, index, value, descr);
        }
    }

    /// resume.py:1599 int_add_const
    pub fn int_add_const(&self, base: i64, offset: i64) -> i64 {
        base + offset
    }
}

/// resume.py:1312 blackhole_from_resumedata
///
/// Build a chain of BlackholeInterpreters from encoded resume data.
/// Returns the topmost (innermost) interpreter.
///
/// `resolve_jitcode` corresponds to RPython's `jitcodes[jitcode_pos]` lookup
/// (`resume.py:1339`). Matches upstream's `(jitcode, pc)` tuple result.
pub struct ResolvedJitCode {
    pub jitcode: std::sync::Arc<crate::jitcode::JitCode>,
    pub pc: usize,
    pub virtualizable_stack_base: Option<usize>,
}

impl ResolvedJitCode {
    pub fn new(jitcode: std::sync::Arc<crate::jitcode::JitCode>, pc: usize) -> Self {
        Self {
            jitcode,
            pc,
            virtualizable_stack_base: None,
        }
    }

    pub fn with_virtualizable_stack_base(mut self, stack_base: usize) -> Self {
        self.virtualizable_stack_base = Some(stack_base);
        self
    }
}

/// resume.py:1054 `consume_boxes` liveness split for a generic JitDriver
/// `setup_bridge_sym`: the live register indices of a guard's resume frame,
/// split by the three liveness banks (int / ref / float).  A frame decoded
/// by `rebuild_from_numbering` lays its `values` out in this same bank order
/// (all int-bank values, then all ref-bank, then float), so the i-th `int`
/// index here pairs with the i-th int-bank `RebuiltValue`.
#[derive(Default, Clone, Debug)]
pub struct FrameLivenessRegIndices {
    pub int: Vec<u32>,
    pub ref_: Vec<u32>,
    pub float: Vec<u32>,
}

impl FrameLivenessRegIndices {
    pub fn total_len(&self) -> usize {
        self.int.len() + self.ref_.len() + self.float.len()
    }
}

/// jitcode.py:149-166 `enumerate_vars` parity: read the per-bank live
/// register indices at a resolved JitCode `pc`. `all_liveness` is
/// `metainterp_sd.liveness_info`; `op_live` is `metainterp_sd.op_live`.
/// Returns empty banks when `pc` is not a valid liveness startpoint, so the
/// caller can decline to seed rather than panic in `get_live_vars_info`.
///
/// jitcode.py:82-100 `get_live_vars_info` asserts on a missing startpoint
/// (MissingLiveness); we deliberately soft-decline instead, because a frame
/// resuming through the Python `pc` legitimately has no JitCode liveness at
/// this coordinate.  An empty return can however mask a genuinely bad resume
/// coordinate (the caller then seeds nothing), so each decline is logged
/// under `MAJIT_BRIDGE_DEBUG` rather than being fully silent.
pub fn read_frame_liveness_reg_indices(
    jitcode: &crate::jitcode::JitCode,
    pc: usize,
    op_live: u8,
    all_liveness: &[u8],
) -> FrameLivenessRegIndices {
    use majit_translate::liveness::LivenessIterator;
    if !jitcode.can_decode_live_vars(pc, op_live) {
        if crate::bridge_debug_enabled() {
            eprintln!(
                "[bridgeB] read_frame_liveness_reg_indices: no liveness startpoint at pc={pc} op_live={op_live} — declining (empty banks)"
            );
        }
        return FrameLivenessRegIndices::default();
    }
    let info = jitcode.get_live_vars_info(pc, op_live);
    if info + 2 >= all_liveness.len() {
        if crate::bridge_debug_enabled() {
            eprintln!(
                "[bridgeB] read_frame_liveness_reg_indices: liveness info {info} out of range (len={}) at pc={pc} — declining (empty banks)",
                all_liveness.len()
            );
        }
        return FrameLivenessRegIndices::default();
    }
    // jitcode.py:149-151 — three length bytes; jitcode.py:152 — body offset.
    let length_i = all_liveness[info] as u32;
    let length_r = all_liveness[info + 1] as u32;
    let length_f = all_liveness[info + 2] as u32;
    let mut offset = info + 3;
    fn read_bank(offset: &mut usize, length: u32, all_liveness: &[u8]) -> Vec<u32> {
        if length == 0 {
            return Vec::new();
        }
        let mut it = LivenessIterator::new(*offset, length, all_liveness);
        let mut out = Vec::with_capacity(length as usize);
        while let Some(reg_idx) = it.next() {
            out.push(reg_idx);
        }
        *offset = it.offset;
        out
    }
    let int = read_bank(&mut offset, length_i, all_liveness);
    let ref_ = read_bank(&mut offset, length_r, all_liveness);
    let float = read_bank(&mut offset, length_f, all_liveness);
    FrameLivenessRegIndices { int, ref_, float }
}

/// RAII guard that pops every resume-construction ref-slice root pushed
/// during `blackhole_from_resumedata` back to the depth captured at entry.
/// Drop runs on ordinary return, `?` propagation, and panic unwind, so the
/// `virtuals_cache` / `registers_r` slices never outlive the construction
/// window in the GC root set.
struct ResumeRefRootsScope {
    base_depth: usize,
}

impl ResumeRefRootsScope {
    fn enter() -> Self {
        ResumeRefRootsScope {
            base_depth: majit_gc::shadow_stack::resume_ref_roots_depth(),
        }
    }
}

impl Drop for ResumeRefRootsScope {
    fn drop(&mut self) {
        majit_gc::shadow_stack::pop_resume_ref_roots_to(self.base_depth);
    }
}

pub fn blackhole_from_resumedata<'a>(
    builder: &mut crate::blackhole::BlackholeInterpBuilder,
    resolve_jitcode: &dyn Fn(i32, i32, i32) -> Option<ResolvedJitCode>,
    rd_numb: &'a [u8],
    rd_consts: &'a [majit_ir::Const],
    all_liveness: &'a [u8],
    deadframe: &'a [i64],
    deadframe_types: Option<&'a [majit_ir::Type]>,
    rd_virtuals: Option<&'a [VirtualInfo]>,
    rd_guard_pendingfields: Option<&[majit_ir::GuardPendingFieldEntry]>,
    vrefinfo: Option<&dyn VRefInfo>,
    vinfo: Option<&dyn VirtualizableInfo>,
    ginfo: Option<&dyn GreenfieldInfo>,
    virtualizable_identity_override: Option<i64>,
    allocator: &'a dyn BlackholeAllocator,
) -> Option<(BlackholeInterpreter, i64)> {
    // resume.py:1315-1327 The initialization is stack-critical code: it
    // must not be interrupted by StackOverflow, otherwise the
    // jit_virtual_refs are left in a dangling state.
    //
    // RPython wraps the body in try/finally so _stop() runs on every
    // exit path. The RAII CriticalCodeGuard gives us Drop-based
    // guarantee — ordinary returns, `?` propagation, AND panic unwind
    // all re-enable the report_error flag.
    let _cc_guard = crate::CriticalCodeGuard::enter();
    // resume.py:1317-1321
    let mut resumereader = ResumeDataDirectReader::new(
        rd_numb,
        rd_consts,
        all_liveness,
        deadframe,
        deadframe_types,
        None,
        allocator,
    );

    // resume.py:1324 _prepare = _prepare_virtuals + _prepare_pendingfields.
    //
    // Root the lazily-filled `virtuals_cache` (and, in the loop below, each
    // frame's `registers_r`) for the whole construction window: decoding a
    // virtual target (`getvirtual_ptr` → allocator) materializes a fresh
    // boxed object, and a minor collection triggered by a later
    // materialization relocates the already-built ones.  The raw `Vec`
    // copies are not otherwise forwarded until `run()`'s `push_bh_regs`, so
    // a from-space pointer would survive into the blackhole.  RPython traces
    // these through the GC-managed reader/blackhole objects.
    //
    // The cache must be rooted BEFORE the first materialization.
    // `_prepare_pendingfields` (resume.py:926) already materializes virtual
    // targets via `decode_ref` → `getvirtual_ptr`, so split `_prepare` and
    // register the ref cache between sizing it (`_prepare_virtuals`) and the
    // pending-field application.  The cache buffer is stable after
    // `_prepare_virtuals` (pre-sized; `set_ptr` only indexes), so the
    // captured pointer stays valid for the window.
    let _resume_roots = ResumeRefRootsScope::enter();
    // resume.py:925 _prepare_virtuals
    resumereader.prepare_virtuals(rd_virtuals);
    unsafe {
        majit_gc::shadow_stack::push_resume_ref_roots(
            &mut resumereader.virtuals_cache.virtuals_ptr_cache,
        );
    }
    // resume.py:926 _prepare_pendingfields
    if let Some(guard_pf) = rd_guard_pendingfields {
        resumereader.prepare_guard_pendingfields(guard_pf);
    }

    // resume.py:1325
    resumereader.consume_vref_and_vable(vrefinfo, vinfo, ginfo, virtualizable_identity_override);
    drop(_cc_guard);

    // resume.py:1404: virtualizable pointer read by consume_vable_info.
    // The virtualizable is the frame being resumed; RPython keeps it live in
    // the GC-traced resume reader across the frame-chain build.  pyre has no
    // GC transform, so root the reader's `virtualizable_ptr` slot for the
    // chain-build window below: a multi-frame resume runs `consume_one_section`
    // per caller, which materializes virtuals (allocator) and can trigger a
    // minor collection.  That relocates the young virtualizable frame; an
    // unrooted bare copy would then point at from-space (a freed frame whose
    // `locals_cells_stack_w` reads null).  Registering the slot makes the root
    // walker forward it in place, mirroring `ResumeDeadframeRoots`.
    unsafe {
        majit_gc::shadow_stack::push_resume_ref_roots(std::slice::from_mut(
            &mut resumereader.virtualizable_ptr,
        ));
    }

    // resume.py:1332-1343
    // Build chain bottom-up: first frame acquired is the outermost.
    let mut curbh: Option<Box<BlackholeInterpreter>> = None;

    while !resumereader.done_reading() {
        // resume.py:1334-1336
        let mut nextbh = builder.acquire_interp();
        nextbh.nextblackholeinterp = curbh;

        // resume.py:1338-1340
        let (jitcode_pos, pc, jitcode_pc) = resumereader.read_jitcode_pos_pc();
        // resume.py:1339-1340: jitcode = jitcodes[jitcode_pos]; curbh.setposition(jitcode, pc).
        // `#124`: pass the carried direct JitCode pc so the resolver can
        // prefer it over the lossy `pc_map` translation.
        let resolved = resolve_jitcode(jitcode_pos, pc, jitcode_pc)?;
        if crate::bh_debug_enabled() {
            eprintln!(
                "[bh-frame] jitcode_pos={jitcode_pos} encoded_pc={pc} resolved_pc={}",
                resolved.pc
            );
        }
        nextbh.setposition(resolved.jitcode.clone(), resolved.pc);
        if let Some(stack_base) = resolved.virtualizable_stack_base {
            nextbh.virtualizable_stack_base = stack_base;
        }

        // `setposition` sized this frame's register files; root the ref
        // bank before filling it so a materialization collection during
        // `consume_one_section` forwards the refs already written here (the
        // Vec buffer is stable — `setarg_r` only indexes — and survives the
        // move into the chained `Box`).
        unsafe {
            majit_gc::shadow_stack::push_resume_ref_roots(&mut nextbh.registers_r);
        }

        // resume.py:1341
        resumereader.consume_one_section(&mut nextbh);

        // resume.py:1342
        nextbh.handle_rvmprof_enter();

        curbh = Some(Box::new(nextbh));
    }

    // Read the (possibly forwarded) virtualizable pointer back from the rooted
    // reader slot: a minor collection during the loop above rewrites it in
    // place to the to-space address.
    let virtualizable_ptr = resumereader.virtualizable_ptr;

    curbh.map(|b| (*b, virtualizable_ptr))
}

/// resume.py:1345 force_from_resumedata
///
/// Force all virtuals from resume data without running a blackhole.
/// Used for GUARD_NOT_FORCED handling.
/// Returns (virtuals_cache_ptr, virtuals_cache_int) — RPython VirtualCache parity.
pub fn force_from_resumedata<'a>(
    rd_numb: &'a [u8],
    rd_consts: &'a [majit_ir::Const],
    all_liveness: &'a [u8],
    deadframe: &'a [i64],
    deadframe_types: Option<&'a [majit_ir::Type]>,
    rd_virtuals: Option<&'a [VirtualInfo]>,
    rd_guard_pendingfields: Option<&[majit_ir::GuardPendingFieldEntry]>,
    vrefinfo: Option<&dyn VRefInfo>,
    vinfo: Option<&dyn VirtualizableInfo>,
    ginfo: Option<&dyn GreenfieldInfo>,
    allocator: &'a dyn BlackholeAllocator,
) -> (Vec<i64>, Vec<i64>) {
    // resume.py:1347-1348
    let mut resumereader = ResumeDataDirectReader::new(
        rd_numb,
        rd_consts,
        all_liveness,
        deadframe,
        deadframe_types,
        None,
        allocator,
    );
    // resume.py:1371 common-case __init__ calls self._prepare(storage)
    // before handling_async_forcing() flips the GUARD_NOT_FORCED state.
    resumereader.prepare(rd_virtuals, rd_guard_pendingfields);
    resumereader.handling_async_forcing();
    // resume.py:1350
    resumereader.consume_vref_and_vable(vrefinfo, vinfo, ginfo, None);
    // resume.py:1351: return resumereader.force_all_virtuals()
    let (ptrs, ints) = resumereader.force_all_virtuals();
    (ptrs.to_vec(), ints.to_vec())
}
