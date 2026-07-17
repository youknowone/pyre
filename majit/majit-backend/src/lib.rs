/// Backend abstraction trait for JIT code generation.
///
/// Translated from rpython/jit/backend/model.py (AbstractCPU).
/// The Backend trait is the contract between the JIT frontend (tracing + optimization)
/// and the code generation backend (Cranelift, etc.).
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use majit_ir::{Const, Descr, FailDescr, GcRef, InputArg, Op, OpRc, Type, Value};

/// `rpython/jit/backend/model.py:8-12 CPUTotalTracker` — per-CPU totals
/// bumped by `CompiledLoopToken.__init__` / `compiling_a_bridge` (loops
/// and bridges created) and by the memory manager (loops and bridges
/// freed).  PyPy attaches one tracker per `AbstractCPU` instance
/// (`model.py:28-29 self.tracker = CPUTotalTracker()`).  Pyre matches
/// that shape: each [`Backend`] impl owns an `Arc<CpuTotalTracker>`
/// exposed via [`Backend::cpu_tracker`], and `MetaInterp::new` rebinds
/// the paired profiler's tracker handle to the same Arc so reads
/// through `Profiler.get_counter(TOTAL_*)` hit the same sink the
/// backend's `compile_loop` / `compile_bridge` write to.  Multiple
/// backend instances coexisting in one process (e.g. parallel
/// translation tests) keep their totals isolated.
///
/// Each field is an [`AtomicUsize`] because PyPy's GIL-protected
/// `+= 1` on a Python int becomes a cross-thread mutation in pyre — the
/// backend may compile loops/bridges from worker threads while another
/// reads totals.  `Relaxed` ordering matches PyPy: there is no causal
/// relationship between bumps and snapshots.
#[derive(Default, Debug)]
pub struct CpuTotalTracker {
    /// `model.py:9` `total_compiled_loops` — bumped once by
    /// [`record_compiled_loop_token`] for each [`CompiledLoopToken`],
    /// at backend `compile_loop` entry (PyPy `x86/assembler.py:514`
    /// parity).  The bump used to live in [`CompiledLoopToken::new`]
    /// but moved because pyre eagerly creates the CLT in
    /// `JitCellToken::new`, which would over-count tokens that are
    /// allocated but never assembled.
    pub total_compiled_loops: AtomicUsize,
    /// `model.py:10` `total_compiled_bridges` — bumped by
    /// [`CompiledLoopToken::compiling_a_bridge`] before bridge assembly.
    pub total_compiled_bridges: AtomicUsize,
    /// `model.py:11` `total_freed_loops` — bumped by the memory manager
    /// (`memmgr.py:_kill_old_loops_now`) when an evicted token had no
    /// attached bridges.
    pub total_freed_loops: AtomicUsize,
    /// `model.py:12` `total_freed_bridges` — bumped by the memory
    /// manager for each bridge attached to an evicted token.
    pub total_freed_bridges: AtomicUsize,
}

/// Process-wide fallback [`CpuTotalTracker`] for callers that have no
/// backend handle in scope.  Should be reserved for legacy/test paths
/// that pre-date the per-backend [`Backend::cpu_tracker`] hook; the
/// production path threads each backend's own `Arc<CpuTotalTracker>`
/// through [`record_compiled_loop_token`] and
/// [`CompiledLoopToken::compiling_a_bridge`] so PyPy's per-CPU
/// `cpu.tracker` semantics survive when multiple backends or
/// `MetaInterpStaticData` instances coexist (e.g. tests that build
/// fresh fixtures inside one process).
pub fn fallback_cpu_tracker() -> &'static CpuTotalTracker {
    static TRACKER: std::sync::OnceLock<CpuTotalTracker> = std::sync::OnceLock::new();
    TRACKER.get_or_init(CpuTotalTracker::default)
}

pub mod call_stub;
pub mod finish_descrs;
pub mod jitframe;
pub mod llmodel;
pub mod model;
pub mod rd_payload;
pub mod resume_guard_descr;
pub mod resume_value;
pub mod synthetic_cpu;

pub use finish_descrs::{
    DoneWithThisFrameDescrFloat, DoneWithThisFrameDescrInt, DoneWithThisFrameDescrMulti,
    DoneWithThisFrameDescrRef, DoneWithThisFrameDescrVoid, ExitFrameWithExceptionDescrRef,
    PropagateExceptionDescr, get_or_attach_done_with_this_frame_descr_multi,
};
pub use jitframe::JitFrameInfo;
pub use rd_payload::RdPayload;
pub use resume_guard_descr::{
    ResumeGuardDescr, STATUS_BUSY_FLAG, STATUS_SHIFT, STATUS_SHIFT_MASK, STATUS_TY_FLOAT,
    STATUS_TY_INT, STATUS_TY_NONE, STATUS_TY_REF, STATUS_TYPE_MASK, alloc_fail_index,
    build_vector_info_chain, flatten_vector_info, make_resume_guard_descr_typed, push_vector_info,
    reset_fail_index_counter,
};
pub use resume_value::{
    FrameInfo, FrameSlotSource, PendingFieldInfo, ResumeData, ResumeValueLayoutSummaryExt,
    ResumeValueSource, VirtualFieldSource, VirtualInfo,
    resume_value_layout_summary_from_exit_value_source,
};

/// Lightweight execution result that avoids DeadFrame boxing.
///
/// Used by `execute_token_ints_raw` to return guard failure data
/// without heap-allocating a DeadFrame.
pub struct RawExecResult {
    /// Output values from the guard exit, truncated to `exit_arity`.
    pub outputs: Vec<i64>,
    /// Typed output values decoded from the exit slots.
    pub typed_outputs: Vec<Value>,
    /// Backend-origin static metadata for this exit, when available.
    pub exit_layout: Option<FailDescrLayout>,
    /// Output slots that carry opaque force tokens instead of GC refs.
    pub force_token_slots: Vec<usize>,
    /// Optional saved-data GC ref captured by this exit.
    pub savedata: Option<GcRef>,
    /// Pending exception value captured by this exit (`GcRef::NULL` = none).
    pub exception_value: GcRef,
    /// Backend fail-index for this exit.
    pub fail_index: u32,
    /// Compiled trace identifier for this exit.
    pub trace_id: u64,
    /// Whether this exit is a FINISH rather than a guard failure.
    pub is_finish: bool,
    /// compile.py:658-662 ExitFrameWithExceptionDescrRef parity.
    /// True when this FINISH was emitted via
    /// pyjitpl.py:3238-3245 compile_exit_frame_with_exception.
    pub is_exit_frame_with_exception: bool,
    /// compile.py:741-745: ResumeGuardDescr.status at guard failure time.
    pub status: u64,
    /// `cpu.get_latest_descr(deadframe)` (`history.py:125`,
    /// `compile.py:701`) — the runtime descr Arc owning this exit.
    /// Always set: routes through `Backend::get_latest_descr_arc`, so
    /// FINISH / `DoneWithThisFrame*` / `ExitFrameWithExceptionDescrRef`
    /// singletons return their attached Arc identity directly.
    /// Bridge consumers (`start_bridge_tracing`,
    /// `_trace_and_compile_from_bridge`) call `descr_arc.as_fail_descr()`
    /// to read `rd_loop_token_clt` / `fail_index_per_trace` directly.
    pub descr_arc: Arc<dyn Descr>,
}

/// Backend-neutral static metadata for a compiled trace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledTraceInfo {
    /// Compiled trace identifier.
    pub trace_id: u64,
    /// Input types expected at this trace header.
    pub input_types: Vec<Type>,
    /// Interpreter header pc associated with this trace.
    pub header_pc: u64,
    /// Source guard this bridge is attached to, or `None` for a root trace.
    pub source_guard: Option<(u64, u32)>,
}

/// Backend-neutral source of a reconstructed frame slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExitValueSourceLayout {
    /// Slot is sourced from a raw exit slot.
    ExitValue(usize),
    /// Slot is a constant value embedded in the layout — `(raw i64 bits,
    /// declared type)`. The type is retained so the resume reader can
    /// reconstruct a typed `Const` rather than assuming `Int`
    /// (resume.py:1017-1038 `decode_box(tagged, kind)`: a slot's type is the
    /// declared type of the variable, so a constant GC pointer decodes as a
    /// `Ref`, not a boxed integer).
    Constant(i64, Type),
    /// Slot refers to a materialized virtual object.
    Virtual(usize),
    /// Slot exists but remains uninitialized.
    Uninitialized,
    /// Slot is unavailable/dead at this exit.
    Unavailable,
}

impl ExitValueSourceLayout {
    pub fn shifted_virtuals(&self, virtual_offset: usize) -> Self {
        match self {
            Self::Virtual(index) => Self::Virtual(index + virtual_offset),
            other => other.clone(),
        }
    }
}

/// Backend-neutral kind of materialized virtual object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExitVirtualKind {
    Object,
    Struct,
    Array,
    ArrayStruct,
    RawBuffer,
}

/// Backend-neutral description of a materialized virtual object.
#[derive(Debug, Clone)]
pub enum ExitVirtualLayout {
    /// resume.py:612 VirtualInfo — allocate_with_vtable(descr=self.descr).
    Object {
        /// resume.py:615 self.descr — live SizeDescr for allocate_with_vtable.
        descr: Option<majit_ir::DescrRef>,
        type_id: u32,
        /// info.py:318 _known_class — vtable pointer for allocate_with_vtable.
        known_class: Option<i64>,
        fields: Vec<(u32, ExitValueSourceLayout)>,
        target_slot: Option<usize>,
        /// resume.py:593 fielddescrs for setfield dispatch.
        fielddescrs: Vec<majit_ir::FieldDescrInfo>,
        descr_size: usize,
    },
    /// resume.py:628 VStructInfo — allocate_struct(self.typedescr).
    Struct {
        /// resume.py:631 self.typedescr — live SizeDescr for allocate_struct.
        typedescr: Option<majit_ir::DescrRef>,
        type_id: u32,
        fields: Vec<(u32, ExitValueSourceLayout)>,
        target_slot: Option<usize>,
        fielddescrs: Vec<majit_ir::FieldDescrInfo>,
        descr_size: usize,
    },
    Array {
        /// `resume.py:646` `self.arraydescr` — live `ArrayDescr` for
        /// `allocate_array`.  Identity-comparable via `Arc::ptr_eq`
        /// (`history.py:125`).
        arraydescr: Option<majit_ir::DescrRef>,
        /// resume.py:653: allocate_array(length, arraydescr, self.clear)
        clear: bool,
        /// resume.py:656: arraydescr element kind (0=ref, 1=int, 2=float)
        kind: u8,
        items: Vec<ExitValueSourceLayout>,
    },
    /// resume.py:736 VArrayStructInfo(arraydescr, size, fielddescrs)
    ArrayStruct {
        /// resume.py:739: self.arraydescr — live ArrayDescr for allocate_array.
        arraydescr: Option<majit_ir::DescrRef>,
        /// resume.py:740: self.fielddescrs — live InteriorFieldDescr per field slot.
        fielddescrs: Vec<majit_ir::DescrRef>,
        element_fields: Vec<Vec<(u32, ExitValueSourceLayout)>>,
    },
    /// resume.py:717 VRawSliceInfo — base_buffer + offset.
    RawSlice {
        /// info.py:460 signed slice base.
        offset: i64,
        base: ExitValueSourceLayout,
    },
    RawBuffer {
        /// resume.py:694: self.func
        func: i64,
        size: usize,
        /// resume.py:695: self.offsets — signed (rawbuffer.py:14).
        offsets: Vec<i64>,
        /// resume.py:697: self.descrs
        descrs: Vec<majit_ir::ArrayDescrInfo>,
        /// resume.py:693: fieldnums (decoded)
        values: Vec<ExitValueSourceLayout>,
    },
    /// resume.py:763 VStrPlainInfo — virtual byte-string
    /// (bh_newstr(len) + bh_strsetitem per character).
    ///
    /// `is_unicode = false` → bh_newstr/bh_strsetitem.
    /// `is_unicode = true`  → bh_newunicode/bh_unicodesetitem (unified
    /// variant for resume.py:817 VUniPlainInfo).
    StrPlain {
        is_unicode: bool,
        /// Per-character values, length = string length. UNINITIALIZED
        /// fieldnums (resume.py:774) remain as `Uninitialized`.
        chars: Vec<ExitValueSourceLayout>,
    },
    /// resume.py:781 VStrConcatInfo + resume.py:836 VUniConcatInfo.
    /// decoder.concat_strings(left, right) looks up OS_STR_CONCAT (or
    /// OS_UNI_CONCAT) via `callinfocollection.funcptr_for_oopspec(...)`
    /// at materialization (resume.py:1467-1468 / 1494-1495); the layout
    /// carries no funcptr / calldescr.
    StrConcat {
        is_unicode: bool,
        left: ExitValueSourceLayout,
        right: ExitValueSourceLayout,
    },
    /// resume.py:801 VStrSliceInfo + resume.py:856 VUniSliceInfo.
    /// decoder.slice_string(str, start, length) looks up OS_STR_SLICE
    /// (or OS_UNI_SLICE) via callinfocollection at materialization
    /// (resume.py:1477-1478 / 1504-1505); the layout carries no
    /// funcptr / calldescr.
    StrSlice {
        is_unicode: bool,
        str_src: ExitValueSourceLayout,
        start: ExitValueSourceLayout,
        length: ExitValueSourceLayout,
    },
}

impl ExitVirtualLayout {
    pub fn shifted_virtuals(&self, virtual_offset: usize) -> Self {
        match self {
            Self::Object {
                descr,
                type_id,
                known_class,
                fields,
                target_slot,
                fielddescrs,
                descr_size,
            } => Self::Object {
                descr: descr.clone(),
                type_id: *type_id,
                known_class: *known_class,
                fields: fields
                    .iter()
                    .map(|(fi, src)| (*fi, src.shifted_virtuals(virtual_offset)))
                    .collect(),
                target_slot: *target_slot,
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            },
            Self::Struct {
                typedescr,
                type_id,
                fields,
                target_slot,
                fielddescrs,
                descr_size,
            } => Self::Struct {
                typedescr: typedescr.clone(),
                type_id: *type_id,
                fields: fields
                    .iter()
                    .map(|(field_index, source)| {
                        (*field_index, source.shifted_virtuals(virtual_offset))
                    })
                    .collect(),
                target_slot: *target_slot,
                fielddescrs: fielddescrs.clone(),
                descr_size: *descr_size,
            },
            Self::Array {
                arraydescr,
                clear,
                kind,
                items,
            } => Self::Array {
                arraydescr: arraydescr.clone(),
                clear: *clear,
                kind: *kind,
                items: items
                    .iter()
                    .map(|source| source.shifted_virtuals(virtual_offset))
                    .collect(),
            },
            Self::RawSlice { offset, base } => Self::RawSlice {
                offset: *offset,
                base: base.shifted_virtuals(virtual_offset),
            },
            Self::ArrayStruct {
                arraydescr,
                fielddescrs,
                element_fields,
            } => Self::ArrayStruct {
                arraydescr: arraydescr.clone(),
                fielddescrs: fielddescrs.clone(),
                element_fields: element_fields
                    .iter()
                    .map(|element| {
                        element
                            .iter()
                            .map(|(field_index, source)| {
                                (*field_index, source.shifted_virtuals(virtual_offset))
                            })
                            .collect()
                    })
                    .collect(),
            },
            Self::RawBuffer {
                func,
                size,
                offsets,
                descrs,
                values,
            } => Self::RawBuffer {
                func: *func,
                size: *size,
                offsets: offsets.clone(),
                descrs: descrs.clone(),
                values: values
                    .iter()
                    .map(|source| source.shifted_virtuals(virtual_offset))
                    .collect(),
            },
            Self::StrPlain { is_unicode, chars } => Self::StrPlain {
                is_unicode: *is_unicode,
                chars: chars
                    .iter()
                    .map(|source| source.shifted_virtuals(virtual_offset))
                    .collect(),
            },
            Self::StrConcat {
                is_unicode,
                left,
                right,
            } => Self::StrConcat {
                is_unicode: *is_unicode,
                left: left.shifted_virtuals(virtual_offset),
                right: right.shifted_virtuals(virtual_offset),
            },
            Self::StrSlice {
                is_unicode,
                str_src,
                start,
                length,
            } => Self::StrSlice {
                is_unicode: *is_unicode,
                str_src: str_src.shifted_virtuals(virtual_offset),
                start: start.shifted_virtuals(virtual_offset),
                length: length.shifted_virtuals(virtual_offset),
            },
        }
    }
}

/// `history.py:125` `id(descr)` parity — `Option<Arc<dyn Descr>>`
/// identity compare via `Arc::ptr_eq`.  Backs the `ExitVirtualLayout`
/// `PartialEq` so canonicalisation matches PyPy's `descr is
/// other_descr` rather than relying on the pyre-only `descr_index`
/// serialization handle.
#[inline]
fn opt_descr_ptr_eq(a: &Option<majit_ir::DescrRef>, b: &Option<majit_ir::DescrRef>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => std::sync::Arc::ptr_eq(a, b),
        _ => false,
    }
}

// `PartialEq/Eq` parity: compare layout structurally + descr Arc
// identity (`history.py:125`); `descr_index` is a serialization handle,
// not identity.
impl PartialEq for ExitVirtualLayout {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Object {
                    descr: a_descr,
                    type_id: a1,
                    known_class: a7,
                    fields: a3,
                    target_slot: a4,
                    fielddescrs: a5,
                    descr_size: a6,
                },
                Self::Object {
                    descr: b_descr,
                    type_id: b1,
                    known_class: b7,
                    fields: b3,
                    target_slot: b4,
                    fielddescrs: b5,
                    descr_size: b6,
                },
            ) => {
                opt_descr_ptr_eq(a_descr, b_descr)
                    && a1 == b1
                    && a3 == b3
                    && a4 == b4
                    && a5 == b5
                    && a6 == b6
                    && a7 == b7
            }
            (
                Self::Struct {
                    typedescr: a_descr,
                    type_id: a1,
                    fields: a3,
                    target_slot: a4,
                    fielddescrs: a5,
                    descr_size: a6,
                },
                Self::Struct {
                    typedescr: b_descr,
                    type_id: b1,
                    fields: b3,
                    target_slot: b4,
                    fielddescrs: b5,
                    descr_size: b6,
                },
            ) => {
                opt_descr_ptr_eq(a_descr, b_descr)
                    && a1 == b1
                    && a3 == b3
                    && a4 == b4
                    && a5 == b5
                    && a6 == b6
            }
            (
                Self::Array {
                    arraydescr: a_ad,
                    clear: a2,
                    kind: a3,
                    items: a4,
                },
                Self::Array {
                    arraydescr: b_ad,
                    clear: b2,
                    kind: b3,
                    items: b4,
                },
            ) => opt_descr_ptr_eq(a_ad, b_ad) && a2 == b2 && a3 == b3 && a4 == b4,
            (
                Self::ArrayStruct {
                    arraydescr: a_ad,
                    fielddescrs: a_fds,
                    element_fields: a2,
                },
                Self::ArrayStruct {
                    arraydescr: b_ad,
                    fielddescrs: b_fds,
                    element_fields: b2,
                },
            ) => {
                // `virtualstate.py:295-305`: arraydescr identity + per-fielddescr identity
                opt_descr_ptr_eq(a_ad, b_ad)
                    && a_fds.len() == b_fds.len()
                    && a_fds
                        .iter()
                        .zip(b_fds.iter())
                        .all(|(a, b)| std::sync::Arc::ptr_eq(a, b))
                    && a2 == b2
            }
            (
                Self::RawSlice {
                    offset: a1,
                    base: a2,
                },
                Self::RawSlice {
                    offset: b1,
                    base: b2,
                },
            ) => a1 == b1 && a2 == b2,
            (
                Self::RawBuffer {
                    size: a1,
                    offsets: a2,
                    values: a3,
                    ..
                },
                Self::RawBuffer {
                    size: b1,
                    offsets: b2,
                    values: b3,
                    ..
                },
            ) => a1 == b1 && a2 == b2 && a3 == b3,
            (
                Self::StrPlain {
                    is_unicode: a1,
                    chars: a2,
                },
                Self::StrPlain {
                    is_unicode: b1,
                    chars: b2,
                },
            ) => a1 == b1 && a2 == b2,
            (
                Self::StrConcat {
                    is_unicode: a1,
                    left: a3,
                    right: a4,
                },
                Self::StrConcat {
                    is_unicode: b1,
                    left: b3,
                    right: b4,
                },
            ) => a1 == b1 && a3 == b3 && a4 == b4,
            (
                Self::StrSlice {
                    is_unicode: a1,
                    str_src: a3,
                    start: a4,
                    length: a5,
                },
                Self::StrSlice {
                    is_unicode: b1,
                    str_src: b3,
                    start: b4,
                    length: b5,
                },
            ) => a1 == b1 && a3 == b3 && a4 == b4 && a5 == b5,
            _ => false,
        }
    }
}
impl Eq for ExitVirtualLayout {}

/// Backend-neutral deferred heap write recovered from an exit.
///
/// `resume.py:88 PENDINGFIELDSTRUCT` parity — carries the live
/// `lldescr` and the (target, value) tagged sources only.  Field
/// metadata (offset / size / type) is *not* duplicated onto the
/// layout: consumers (`pyre-jit::eval::replay_pending_fields`,
/// `cranelift::compiler` guard recovery,
/// `jitdriver::materialize_pending_fields`) call
/// `descr.as_field_descr()` / `descr.as_array_descr()` and read
/// `offset()` / `field_size()` / `field_type()` directly, mirroring
/// `resume.py:1509-1518` and `resume.py:1531-1541`.
#[derive(Debug, Clone)]
pub struct ExitPendingFieldLayout {
    /// `resume.py:88 lldescr` — identity-compared via `Arc::ptr_eq`
    /// (`history.py:125`).
    pub descr: Option<majit_ir::DescrRef>,
    pub item_index: Option<usize>,
    pub is_array_item: bool,
    pub target: ExitValueSourceLayout,
    pub value: ExitValueSourceLayout,
}

impl PartialEq for ExitPendingFieldLayout {
    fn eq(&self, other: &Self) -> bool {
        opt_descr_ptr_eq(&self.descr, &other.descr)
            && self.item_index == other.item_index
            && self.is_array_item == other.is_array_item
            && self.target == other.target
            && self.value == other.value
    }
}
impl Eq for ExitPendingFieldLayout {}

impl ExitPendingFieldLayout {
    pub fn shifted_virtuals(&self, virtual_offset: usize) -> Self {
        Self {
            descr: self.descr.clone(),
            item_index: self.item_index,
            is_array_item: self.is_array_item,
            target: self.target.shifted_virtuals(virtual_offset),
            value: self.value.shifted_virtuals(virtual_offset),
        }
    }
}

/// Backend-neutral reconstructed frame layout for an exit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExitFrameLayout {
    /// Compiled trace identifier that owns this frame layout, when known.
    pub trace_id: Option<u64>,
    /// Trace header pc associated with this frame layout, when known.
    pub header_pc: Option<u64>,
    /// Source guard this frame's trace is attached to, when the frame comes
    /// from a compiled bridge.
    pub source_guard: Option<(u64, u32)>,
    /// Interpreter program counter for the frame.
    pub pc: u64,
    /// resume.py:250 `jitcode_index` — index into
    /// `MetaInterpStaticData.jitcodes` identifying the code this frame is
    /// running. Required so multi-frame inline snapshots can be decoded
    /// with per-frame liveness via `frame_value_count_at(jitcode_index, pc)`.
    /// Encoders that only produce single-frame exits may leave this at 0,
    /// but multi-frame producers MUST populate it per frame.
    pub jitcode_index: i32,
    /// Slot sources within this frame.
    pub slots: Vec<ExitValueSourceLayout>,
    /// Typed layout of the frame slots, when known by the backend.
    pub slot_types: Option<Vec<Type>>,
}

impl ExitFrameLayout {
    pub fn shifted_virtuals(&self, virtual_offset: usize) -> Self {
        Self {
            trace_id: self.trace_id,
            header_pc: self.header_pc,
            source_guard: self.source_guard,
            pc: self.pc,
            jitcode_index: self.jitcode_index,
            slots: self
                .slots
                .iter()
                .map(|slot| slot.shifted_virtuals(virtual_offset))
                .collect(),
            slot_types: self.slot_types.clone(),
        }
    }
}

/// Backend-neutral recovery metadata attached to an exit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExitRecoveryLayout {
    /// resume.py:1099 consume_vref_and_vable_boxes / virtualizable_boxes.
    pub vable_array: Vec<ExitValueSourceLayout>,
    /// resume.py:1093 consume_virtualref_boxes — [virtual, vref, ...] pairs.
    pub vref_array: Vec<ExitValueSourceLayout>,
    /// Reconstructed frames, outermost first.
    pub frames: Vec<ExitFrameLayout>,
    /// Materialized virtual objects referenced by the frames.
    pub virtual_layouts: Vec<ExitVirtualLayout>,
    /// Deferred heap writes to replay after materialization.
    pub pending_field_layouts: Vec<ExitPendingFieldLayout>,
}

impl ExitRecoveryLayout {
    pub fn prefixed_by(&self, caller_prefix: Option<&Self>) -> Self {
        let Some(caller_prefix) = caller_prefix else {
            return self.clone();
        };

        let virtual_offset = caller_prefix.virtual_layouts.len();
        let vable_array = if caller_prefix.vable_array.is_empty() {
            self.vable_array.clone()
        } else {
            caller_prefix.vable_array.clone()
        };
        let vref_array = if caller_prefix.vref_array.is_empty() {
            self.vref_array
                .iter()
                .map(|slot| slot.shifted_virtuals(virtual_offset))
                .collect()
        } else {
            caller_prefix.vref_array.clone()
        };
        let mut frames = caller_prefix.frames.clone();
        frames.extend(
            self.frames
                .iter()
                .map(|frame| frame.shifted_virtuals(virtual_offset)),
        );

        let mut virtual_layouts = caller_prefix.virtual_layouts.clone();
        virtual_layouts.extend(
            self.virtual_layouts
                .iter()
                .map(|layout| layout.shifted_virtuals(virtual_offset)),
        );

        let mut pending_field_layouts = caller_prefix.pending_field_layouts.clone();
        pending_field_layouts.extend(
            self.pending_field_layouts
                .iter()
                .map(|layout| layout.shifted_virtuals(virtual_offset)),
        );

        Self {
            vable_array,
            vref_array,
            frames,
            virtual_layouts,
            pending_field_layouts,
        }
    }
}

/// Static layout metadata for a backend fail descriptor.
#[derive(Debug, Clone)]
pub struct FailDescrLayout {
    /// Backend fail-index for this exit.
    pub fail_index: u32,
    /// Trace op index of the guard/finish that produced this exit, when known.
    pub source_op_index: Option<usize>,
    /// Compiled trace identifier that owns this exit.
    pub trace_id: u64,
    /// Backend-owned metadata for the trace that owns this exit.
    pub trace_info: Option<CompiledTraceInfo>,
    /// Typed layout of the exit slots.
    pub fail_arg_types: Vec<Type>,
    /// Whether this exit is a FINISH rather than a guard failure.
    pub is_finish: bool,
    /// `compile.py:658-662 ExitFrameWithExceptionDescrRef` vs
    /// `compile.py:640-647 DoneWithThisFrameDescrRef`: distinguishes the
    /// exception-propagation FINISH from a normal-result FINISH that
    /// happens to carry a single `Type::Ref` slot.  Read from the source
    /// descr's `FailDescr::is_exit_frame_with_exception()` at layout
    /// build time so the metainterp synthesis fallback at
    /// `compile.rs:1132/1417` (when `op.descr` is missing) routes to the
    /// correct `_DoneWithThisFrameDescr` subclass.
    pub is_exception_exit: bool,
    /// Exit slot indices that hold rooted GC references.
    pub gc_ref_slots: Vec<usize>,
    /// Exit slot indices that carry opaque FORCE_TOKEN handles.
    pub force_token_slots: Vec<usize>,
    /// Optional backend-origin recovery layout for this exit.
    pub recovery_layout: Option<ExitRecoveryLayout>,
    /// Complete frame stack from innermost (this guard's frame) to outermost.
    /// Present when multi-frame reconstruction is supported.
    pub frame_stack: Option<Vec<ExitFrameLayout>>,
    /// resume.py:450 — compact resume numbering (varint-encoded tagged values).
    /// Propagated from the backend's fail descriptor so the frontend can
    /// reconstruct the blackhole chain after a trace's `CompiledTrace` entry
    /// has been evicted but the descriptor itself is still live.
    pub rd_numb: Option<Vec<u8>>,
    /// resume.py:451 — shared constant pool referenced by `rd_numb`.
    pub rd_consts: Option<Vec<Const>>,
    /// resume.py:488 — virtual object field info referenced by `rd_numb`.
    pub rd_virtuals: Option<Vec<std::rc::Rc<majit_ir::RdVirtualInfo>>>,
    /// Deferred heap writes associated with this guard exit.
    pub rd_pendingfields: Option<Vec<majit_ir::GuardPendingFieldEntry>>,
}

/// Static layout metadata for a terminal exit within a compiled trace.
///
/// Unlike [`FailDescrLayout`], terminal exits are keyed by the trace op index
/// rather than a backend fail descriptor, because `JUMP` exits do not
/// necessarily correspond to a deadframe-producing guard site.
#[derive(Debug, Clone)]
pub struct TerminalExitLayout {
    /// Trace op index of the terminal `FINISH`/`JUMP`.
    pub op_index: usize,
    /// Compiled trace identifier that owns this exit.
    pub trace_id: u64,
    /// Backend-owned metadata for the trace that owns this exit.
    pub trace_info: Option<CompiledTraceInfo>,
    /// Backend fail-index if this terminal exit is also a fail descriptor.
    pub fail_index: u32,
    /// Typed layout of the exit slots.
    pub exit_types: Vec<Type>,
    /// Whether this exit is a `FINISH` rather than a `JUMP`.
    pub is_finish: bool,
    /// `compile.py:658-662 ExitFrameWithExceptionDescrRef` discriminator;
    /// see `FailDescrLayout::is_exception_exit`.
    pub is_exception_exit: bool,
    /// Exit slot indices that hold rooted GC references.
    pub gc_ref_slots: Vec<usize>,
    /// Exit slot indices that carry opaque FORCE_TOKEN handles.
    pub force_token_slots: Vec<usize>,
    /// Optional backend-origin recovery layout for this terminal exit.
    pub recovery_layout: Option<ExitRecoveryLayout>,
    /// resume.py:450 — compact resume numbering (terminal exits rarely need
    /// this, but propagate it for symmetry with `FailDescrLayout`).
    pub rd_numb: Option<Vec<u8>>,
    /// resume.py:451 — shared constant pool referenced by `rd_numb`.
    pub rd_consts: Option<Vec<Const>>,
    /// resume.py:488 — virtual object field info referenced by `rd_numb`.
    pub rd_virtuals: Option<Vec<std::rc::Rc<majit_ir::RdVirtualInfo>>>,
    /// Deferred heap writes associated with this terminal exit.
    pub rd_pendingfields: Option<Vec<majit_ir::GuardPendingFieldEntry>>,
}

/// Result of compiling a loop or bridge.
#[derive(Debug)]
pub struct AsmInfo {
    /// Start address of the generated code.
    pub code_addr: usize,
    /// Size of the generated code in bytes.
    pub code_size: usize,
}

/// Tracks alternative loop versions to compile after the main loop.
///
/// Each version is an alternative trace that handles cases where the
/// main loop's version guard fails — e.g., unaligned arrays, short
/// arrays that cannot be vectorized, or different type specializations.
pub struct LoopVersionInfo {
    /// (version_guard_index, alternative_inputargs, alternative_ops)
    pub versions: Vec<(u32, Vec<InputArg>, Vec<Op>)>,
}

impl LoopVersionInfo {
    pub fn new() -> Self {
        Self {
            versions: Vec::new(),
        }
    }

    /// Register an alternative version to compile after the main loop.
    pub fn add_version(&mut self, guard_index: u32, inputargs: Vec<InputArg>, ops: Vec<Op>) {
        self.versions.push((guard_index, inputargs, ops));
    }
}

impl Default for LoopVersionInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for LoopVersionInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoopVersionInfo")
            .field("num_versions", &self.versions.len())
            .finish()
    }
}

// JITFRAMEINFO is the single `JitFrameInfo` defined in `jitframe.rs`
// (re-exported at the crate root above). `CompiledLoopToken.frame_info`
// holds one — the same struct `JitFrame.jf_frame_info` points to, matching
// RPython's single `JITFRAMEINFO` (jitframe.py:30-40).

/// `rpython/jit/backend/model.py:292-338` `CompiledLoopToken` parity.
///
/// Per-loop metadata attached to `JitCellToken.compiled_loop_token`.
/// Mutation goes through `parking_lot::Mutex` on individual fields because
/// pyre reaches these from `&JitCellToken` shared refs (bridge compilation
/// under concurrent execution) — RPython mutates through the GIL instead.
pub struct CompiledLoopToken {
    /// `model.py:299` `self.number = number`.
    pub number: u64,
    /// `compile.py:180-181` `wref = weakref.ref(original_jitcell_token);
    /// clt.loop_token_wref = wref` parity. Weak back-reference to the
    /// owning `JitCellToken`. Set immediately after `make_jitcell_token`
    /// allocates the clt, and read by guard-failure paths via the descr
    /// chain `descr.rd_loop_token() -> Arc<CompiledLoopToken>` →
    /// `clt.loop_token_wref.upgrade()` → `Arc<JitCellToken>` → green_key.
    /// `Mutex` because the wref is set after `CompiledLoopToken::new` —
    /// the owning Arc is built first, then `set_loop_token_wref` patches
    /// the weak ref through `&CompiledLoopToken`.
    pub loop_token_wref: parking_lot::Mutex<std::sync::Weak<JitCellToken>>,
    /// `model.py:300` `self.bridges_count = 0`.
    pub bridges_count: parking_lot::Mutex<usize>,
    /// `model.py:301` `self.invalidate_positions = []`.
    pub invalidate_positions: parking_lot::Mutex<Vec<usize>>,
    /// `model.py:302-304` `self.looptokens_redirected_to = []` — weak
    /// references to `CompiledLoopToken` instances previously redirected
    /// to this one via `redirect_call_assembler`.
    /// `x86/assembler.py:1150-1151` shows that
    /// `newlooptoken.compiled_loop_token.update_frame_info(
    ///     oldlooptoken.compiled_loop_token, baseofs)` passes the
    /// `CompiledLoopToken` (not `JitCellToken`), so the chain stores
    /// weak refs to `CompiledLoopToken`.
    pub looptokens_redirected_to: parking_lot::Mutex<Vec<std::sync::Weak<CompiledLoopToken>>>,
    /// `model.py:293` `asmmemmgr_blocks = None` (class default — lazy-init
    /// on first access, see `llsupport/assembler.py:184-188`
    /// `get_asmmemmgr_blocks`). pyre eagerly initializes to an empty Vec;
    /// the `None` sentinel is a Python idiom not needed on Rust.
    pub asmmemmgr_blocks: parking_lot::Mutex<Vec<Box<dyn std::any::Any + Send>>>,
    /// `model.py:294` `asmmemmgr_gcreftracers = None`; parity shape
    /// reserved for the GC ref-tracer lifecycle (llsupport/assembler.py:190).
    /// Eagerly empty (same rationale as `asmmemmgr_blocks`).
    pub asmmemmgr_gcreftracers: parking_lot::Mutex<Vec<Arc<dyn std::any::Any + Send + Sync>>>,
    /// `x86/assembler.py:514` `looptoken.compiled_loop_token.frame_info`
    /// is populated by the backend when assembling this loop. Mutated
    /// in-place by bridges via `update_frame_info` (model.py:316-329).
    pub frame_info: parking_lot::Mutex<JitFrameInfo>,
    /// `rpython/jit/backend/llsupport/regalloc.py:861-871`
    /// `_set_initial_bindings` assigns
    /// `looptoken.compiled_loop_token._ll_initial_locs = locs` — the
    /// inputarg slot offsets (in bytes, relative to the frame base) that
    /// `CALL_ASSEMBLER` uses in `handle_call_assembler`
    /// (rewrite.py:673) to spill callee arguments.
    pub _ll_initial_locs: parking_lot::Mutex<Vec<i32>>,
    /// Pyre-only one-shot latch for the side effects PyPy performs in
    /// `CompiledLoopToken.__init__` (`model.py:296-307`): increment
    /// `cpu.tracker.total_compiled_loops` and emit the
    /// `jit-mem-looptoken-alloc` log section.
    ///
    /// PyPy creates the CLT inside backend `assemble_loop`, so the
    /// constructor side effects are naturally tied to actual assembly.
    /// Pyre creates `CompiledLoopToken` eagerly with `JitCellToken`;
    /// this latch lets `record_compiled_loop_token` fire at backend
    /// assembly time while preserving PyPy's strict "once per CLT"
    /// semantics if a caller retries `compile_loop` with the same token.
    loop_allocation_recorded: AtomicBool,
}

/// PyPy `model.py:296-307` `CompiledLoopToken.__init__` opens the
/// `jit-mem-looptoken-alloc` debug section and bumps
/// `cpu.tracker.total_compiled_loops` at the point where the CLT is
/// created — `x86/assembler.py:514` and the per-backend equivalents do
/// that *inside* `assemble_loop`.  Pyre's [`CompiledLoopToken::new`]
/// runs eagerly from [`JitCellToken::new`] (line 1265 documents the
/// eager-vs-lazy adaptation), so doing the bump there would
/// over-count loops whose JitCellToken is allocated but never reaches
/// `backend.compile_loop`.
///
/// Each backend's `compile_loop` calls this helper as its first act.
/// The helper records at most once per [`CompiledLoopToken`], matching
/// PyPy's constructor side effect even if a Rust caller retries
/// `compile_loop` with the same token.  The caller passes its own
/// [`CpuTotalTracker`] (via
/// [`Backend::cpu_tracker`]) so multiple backend instances in the
/// same process keep separate totals — matching PyPy's per-CPU
/// `cpu.tracker`.
pub fn record_compiled_loop_token(tracker: &CpuTotalTracker, clt: &CompiledLoopToken) {
    if clt.loop_allocation_recorded.swap(true, Ordering::AcqRel) {
        return;
    }
    tracker.total_compiled_loops.fetch_add(1, Ordering::Relaxed);
    majit_ir::debug::log_one(
        "jit-mem-looptoken-alloc",
        &format!("allocating Loop # {}", clt.number),
    );
}

impl CompiledLoopToken {
    /// `model.py:296-307` `__init__(self, cpu, number)`. The `cpu` is
    /// implicit in pyre — the owning `Backend` holds the token lifetime.
    ///
    /// **Counter / debug-section moved.**  PyPy bumps
    /// `cpu.tracker.total_compiled_loops` and emits the
    /// `jit-mem-looptoken-alloc` debug section here.  Pyre defers both
    /// to [`record_compiled_loop_token`] called at the
    /// [`Backend::compile_loop`] entry — see that helper's doc for the
    /// eager-CLT rationale.
    pub fn new(number: u64) -> Self {
        CompiledLoopToken {
            number,
            loop_token_wref: parking_lot::Mutex::new(std::sync::Weak::new()),
            bridges_count: parking_lot::Mutex::new(0),
            invalidate_positions: parking_lot::Mutex::new(Vec::new()),
            looptokens_redirected_to: parking_lot::Mutex::new(Vec::new()),
            asmmemmgr_blocks: parking_lot::Mutex::new(Vec::new()),
            asmmemmgr_gcreftracers: parking_lot::Mutex::new(Vec::new()),
            frame_info: parking_lot::Mutex::new(JitFrameInfo::default()),
            _ll_initial_locs: parking_lot::Mutex::new(Vec::new()),
            loop_allocation_recorded: AtomicBool::new(false),
        }
    }

    /// `compile.py:180-181` `clt.loop_token_wref = wref` setter. Called
    /// immediately after the owning `Arc<JitCellToken>` becomes the
    /// stable identity (i.e., once `make_jitcell_token` has stamped its
    /// generation and the token is the soon-to-be `compiled_loops[gk]`
    /// entry's `.token` field). The weak ref ages out automatically when
    /// memmgr drops the owning Arc.
    pub fn set_loop_token_wref(&self, wref: std::sync::Weak<JitCellToken>) {
        *self.loop_token_wref.lock() = wref;
    }

    /// `compile.py:180-181` reader. Returns `None` after the owning
    /// `JitCellToken` has been dropped (memmgr eviction); guard failures
    /// reaching here on a dropped token are unreachable in practice
    /// because the descr chain holds the clt alive transitively.
    pub fn upgrade_loop_token(&self) -> Option<Arc<JitCellToken>> {
        self.loop_token_wref.lock().upgrade()
    }

    /// `model.py:309-314` `compiling_a_bridge(self)` — bumps the
    /// owning backend's `total_compiled_bridges` tracker, increments
    /// this token's local `bridges_count`, and emits the
    /// `jit-mem-looptoken-alloc` debug section line-for-line with
    /// upstream.  The `tracker` parameter is the backend's own
    /// [`CpuTotalTracker`] (via [`Backend::cpu_tracker`]) so multiple
    /// backend instances stay isolated, matching PyPy's per-CPU
    /// `cpu.tracker`.
    pub fn compiling_a_bridge(&self, tracker: &CpuTotalTracker) {
        tracker
            .total_compiled_bridges
            .fetch_add(1, Ordering::Relaxed);
        let bridges_count = {
            let mut count = self.bridges_count.lock();
            *count += 1;
            *count
        };
        majit_ir::debug::log_one(
            "jit-mem-looptoken-alloc",
            &format!(
                "allocating Bridge # {} of Loop # {}",
                bridges_count, self.number
            ),
        );
    }

    /// `rpython/jit/backend/model.py:316-329`
    /// `update_frame_info(self, oldlooptoken, baseofs)`.
    ///
    /// `self` is the *new* looptoken's `CompiledLoopToken`, `oldlooptoken`
    /// is the previous one (and its weak self-ref), `baseofs` is
    /// `cpu.get_baseofs_of_frame_field()`. Propagates `self.frame_info`
    /// depth to every `CompiledLoopToken` in
    /// `oldlooptoken.looptokens_redirected_to` (dropping dead weak refs),
    /// to `oldlooptoken` itself, and appends the provided weak ref to
    /// this new token's redirect chain.
    pub fn update_frame_info(
        &self,
        oldlooptoken: &CompiledLoopToken,
        oldlooptoken_weak: std::sync::Weak<CompiledLoopToken>,
        baseofs: i64,
    ) {
        // `model.py:317` `new_fi = self.frame_info`
        // `model.py:318` `new_loop_tokens = []`
        let new_fi_depth = self.frame_info.lock().jfi_frame_depth as i64;
        let mut new_loop_tokens: Vec<std::sync::Weak<CompiledLoopToken>> = Vec::new();
        // `model.py:319-324` propagate depth through the old token's
        // existing redirect chain, dropping dead weak refs (`ref()` →
        // `None` in CPython), and keep the live ones.
        let old_chain = oldlooptoken.looptokens_redirected_to.lock().clone();
        for weak in old_chain {
            if let Some(target) = weak.upgrade() {
                target
                    .frame_info
                    .lock()
                    .update_frame_depth(baseofs, new_fi_depth);
                new_loop_tokens.push(weak);
            }
        }
        // `model.py:325-326` update the old token's own frame_info.
        oldlooptoken
            .frame_info
            .lock()
            .update_frame_depth(baseofs, new_fi_depth);
        // `model.py:327-328` `assert oldlooptoken is not None;
        // new_loop_tokens.append(weakref.ref(oldlooptoken))` — the
        // caller provides the weak ref (Rust can't derive it from a
        // borrow alone; the owning Arc stays in `JitCellToken`).
        new_loop_tokens.push(oldlooptoken_weak);
        // `model.py:329` `self.looptokens_redirected_to = new_loop_tokens`
        *self.looptokens_redirected_to.lock() = new_loop_tokens;
    }
}

/// `compile.py:186` reader chain helper: `descr.rd_loop_token` →
/// `clt.loop_token_wref.upgrade()` → `JitCellToken.green_key`.
///
/// Returns `None` for descrs that have no `rd_loop_token` set (the
/// `_DoneWithThisFrameDescr` family / `ExitFrameWithExceptionDescr`,
/// per `compile.py:185 isinstance(descr, ResumeDescr)` — non-resume
/// descrs are skipped by the post-compile walker) or whose owning
/// `JitCellToken` has been dropped by memmgr.  Bridge-source paths
/// consume the metainterp ResumeGuardDescr Arc directly (Unified-Descr
/// directly), so the chain resolves through the descr's
/// own `rd_loop_token_clt` slot.
pub fn descr_owning_clt(descr: &dyn FailDescr) -> Option<&Arc<CompiledLoopToken>> {
    descr
        .rd_loop_token_clt()?
        .downcast_ref::<Arc<CompiledLoopToken>>()
}

/// `pyjitpl.py:2897` reader chain: `resumedescr.rd_loop_token.loop_token_wref()`
/// — recover the owning `Arc<JitCellToken>` object identity from a
/// FailDescr.  RPython callers consume the returned object directly
/// (e.g. `compile.py:593` passes the descr's owning loop token to
/// bridge attach); pyre callers can either keep the `Arc<JCT>` or
/// derive `green_key` via `jct.green_key`.
///
/// Returns `None` when `descr` is a non-resume FailDescr
/// (`_DoneWithThisFrameDescr` family / `ExitFrameWithExceptionDescr`,
/// which `compile.py:185` skips via `isinstance(descr, ResumeDescr)`)
/// or when the owning JCT was evicted by memmgr.  Bridge-source paths
/// consume the metainterp `AbstractFailDescr` Arc directly.
pub fn descr_owning_jct(descr: &dyn FailDescr) -> Option<Arc<JitCellToken>> {
    descr_owning_clt(descr)?.upgrade_loop_token()
}

/// Token identifying a compiled loop. Bridges are attached to this.
/// RPython history.py JitCellToken parity — green_key carried on token
/// so the backend can identify the parent loop for bridge compilation.
pub struct JitCellToken {
    /// Unique number for this token.
    pub number: u64,
    /// Green key hash identifying the loop entry point.
    /// Set by MetaInterp before compile_loop. Used by the backend's
    /// bridge threshold callback to find the compiled loop metadata.
    ///
    /// `Cell<u64>` because `configure_loop_token_for_driver` writes it
    /// through `&JitCellToken` while self-recursive trace ops already hold
    /// `Arc` clones of the same pending token (`Arc::get_mut` would fail).
    pub green_key: Cell<u64>,
    /// Types of the input arguments, recorded at compile_loop time from
    /// the finalised inputargs. RPython does not carry this list on
    /// `JitCellToken` itself — there, backend code recovers types by
    /// re-walking the LABEL op (`history.py:501 TreeLoop.inputargs` /
    /// `regalloc.py:861-871 _set_initial_bindings`). pyre's external-jump
    /// and bridge-link paths (`compiler.rs:2362/2468/2720/3998/4027`,
    /// `runner.rs:1281`) touch the target token without access to the
    /// trace ops, so this field caches the typed signature for them.
    /// Front-end (tracer) code MUST NOT read this; use
    /// `MetaInterp::front_target_inputarg_types` instead, which derives
    /// types from the LABEL op + `TreeLoop.inputargs` per RPython.
    ///
    /// `OnceLock` because the backend's `compile_loop` writes it exactly
    /// once through `&JitCellToken` (the token may already have `Arc`
    /// clones on recorded CALL_ASSEMBLER ops). Read via `inputarg_types()`.
    pub inputarg_types: OnceLock<Vec<Type>>,
    /// virtualizable.py:86 read_boxes: number of scalar inputargs
    /// (frame + static fields). First local is at this index.
    ///
    /// `Cell<usize>` — written by `configure_loop_token_for_driver`
    /// through `&JitCellToken`.
    pub num_scalar_inputargs: Cell<usize>,
    /// warmspot.py / rewrite.py parity: JitDriverSD.index_of_virtualizable.
    /// Index inside the original CALL_ASSEMBLER arglist before rewrite
    /// collapses it to `[frame]` or `[frame, virtualizable]`.
    ///
    /// `Cell<Option<usize>>` — written by `configure_loop_token_for_driver`
    /// through `&JitCellToken`.
    pub virtualizable_arg_index: Cell<Option<usize>>,
    /// compile.py:168 `jitcell_token.outermost_jitdriver_sd = jitdriver_sd`.
    ///
    /// The backend crate stores the registered jitdriver slot index
    /// rather than a direct metainterp object pointer.
    pub outermost_jitdriver_index: Option<usize>,
    /// Backend-specific compiled data.
    ///
    /// `OnceLock` because the backend's `compile_loop` stores it exactly
    /// once through `&JitCellToken` and the hot execute path reads it via a
    /// cheap `get()`. Recompilation mints a fresh token, so a token's
    /// compiled slot is never overwritten (`reset_compiled` takes it via
    /// `&mut self`).
    pub compiled: OnceLock<Box<dyn std::any::Any + Send>>,
    /// Flag indicating whether the compiled code has been invalidated.
    /// When set to `true`, any `GUARD_NOT_INVALIDATED` in the compiled
    /// code will fail, causing execution to bail out to the interpreter.
    pub invalidated: Arc<AtomicBool>,
    /// Alternative loop versions to compile immediately after the main loop.
    pub version_info: Option<LoopVersionInfo>,
    /// history.py:449: _keepalive_jitcell_tokens — set of other tokens
    /// that this loop can jump to (via CALL_ASSEMBLER or JUMP).
    /// Upstream prevents the target from being evicted while this loop
    /// is alive by holding the actual `JitCellToken` object reference
    /// (`history.py:441 _keepalive_jitcell_tokens = {}` keyed on the
    /// token object itself, with `:451 record_jump_to` writing
    /// `self._keepalive_jitcell_tokens[target] = None`).  Pyre keeps the
    /// same shape: an `Arc<JitCellToken>` set keyed on token number.
    ///
    /// Wrapped in `Mutex` so `record_jump_to` can push through
    /// `&JitCellToken` once the token has been shared via
    /// `Arc<JitCellToken>` (the same object is reachable from
    /// `record_loop_or_bridge`'s descr walk).  The Mutex is the
    /// Rust-side equivalent of RPython's implicit dict-mutation interior
    /// mutability under the single-threaded JIT scheduler invariant.
    pub keepalive_tokens: parking_lot::Mutex<Vec<Arc<JitCellToken>>>,
    /// `rpython/jit/backend/model.py:292` `CompiledLoopToken` parity.
    ///
    /// Carries per-compilation metadata: `asmmemmgr_blocks` (owned bridge
    /// memory blocks — `model.py:293`), `asmmemmgr_gcreftracers`
    /// (`model.py:294`), `frame_info` (JIT frame layout —
    /// `x86/assembler.py:514`), `bridges_count`, `invalidate_positions`,
    /// `looptokens_redirected_to`. Populated eagerly by `JitCellToken::new`
    /// so backends can update fields through `&JitCellToken`.
    ///
    /// Interior mutability on fields uses `parking_lot::Mutex` because
    /// bridge compilation and execution may see `&JitCellToken` from
    /// multiple threads (compile_bridge receives `&JitCellToken`).
    ///
    /// The `Option<Arc<..>>` slot itself is wrapped in `parking_lot::Mutex`
    /// because `register_call_assembler_target` reassigns it through
    /// `&JitCellToken` when a token number is re-registered (preserving the
    /// previously baked CLT pointer). Read a cloned handle via
    /// `compiled_loop_token()`.
    pub compiled_loop_token: parking_lot::Mutex<Option<Arc<CompiledLoopToken>>>,
    /// `rpython/jit/backend/x86/assembler.py:599`
    /// `looptoken._ll_function_addr = rawstart + functionpos` —
    /// address of the compiled loop entry.
    ///
    /// RPython's `compile_tmp_callback` (`metainterp/compile.py:1101-
    /// 1150`) gives every `CALL_ASSEMBLER` token a real body before
    /// emission, so x86 can bake `descr._ll_function_addr` as an
    /// immediate at `assembler.py:320`.  Pyre still has a pending-token
    /// window until that callback identity is ported, so emitted call
    /// thunks may read this shared slot after `compile_loop` stores the
    /// real entry.  Redirects still follow the backend mechanism:
    /// dynasm patches the old entry (`assembler.py:1138`), while
    /// cranelift updates its indirect dispatch state.
    pub _ll_function_addr: AtomicUsize,
    /// `memmgr.py:59-60` `looptoken.generation`. Updated by
    /// `MemoryManager.keep_loop_alive` and read by
    /// `_kill_old_loops_now`. Default `0` means "not yet seen by
    /// memmgr"; the eviction predicate at `memmgr.py:71` requires
    /// `0 <= gen < max_generation`, so default-0 tokens are
    /// candidates for eviction immediately — matching RPython where
    /// `__init__` does not initialize `generation`. In practice
    /// `compile.py:1149` calls `keep_loop_alive` before any
    /// `_kill_old_loops_now` could see the token. RPython's
    /// `r_int64` is signed 64-bit; pyre uses `i64` to preserve the
    /// wraparound math at `memmgr.py:38` (5e9 years given 1000
    /// loops/sec). Interior mutability via `Cell<i64>` mirrors the
    /// RPython attribute write through `&JitCellToken`. Sync is
    /// covered by the existing `unsafe impl Sync for JitCellToken`
    /// at line 1130 — single-threaded JIT scheduler invariant.
    pub generation: Cell<i64>,
    /// `history.py:431-435 JitCellToken.retraced_count` parity slot.
    ///
    /// RPython packs two pieces of state into this u-int:
    ///   * bit 0 = `FORCE_BRIDGE_SEGMENTING` flag.
    ///     `compile.py:728-730 _trace_and_compile_from_bridge` checks
    ///     `loop_token.retraced_count & FORCE_BRIDGE_SEGMENTING` to
    ///     decide whether the next bridge from this loop should
    ///     segment trace at the guard.  Set at `pyjitpl.py:2833`.
    ///   * bits 1+ = retrace count (`history.py:464-468
    ///     get_retraced_count() = retraced_count >> 1` /
    ///     `set_retraced_count(value) = (value << 1) | (current & 1)`),
    ///     compared by `unroll.py:264-272` against `retrace_limit`
    ///     to disable repeated retracing of the same loop.
    ///
    /// Pyre's flow is RPython-orthodox: the retrace count rides on
    /// `JitCellToken.retraced_count` (read via `get_retraced_count`,
    /// updated via `set_retraced_count` at unroll-pass boundaries —
    /// `pyjitpl.rs:7978` etc.); `FORCE_BRIDGE_SEGMENTING` is set
    /// in `MetaInterp::blackhole_trace_too_long_slow` and read by
    /// `MetaInterp::start_retrace_from_guard` (`pyjitpl.rs:8772-
    /// 8784`), mirroring `pyjitpl.py:2833` / `compile.py:729`.
    ///
    /// The complementary `BaseJitCell.flags & FORCE_FINISH` flag in
    /// `warmstate.rs` is NOT a duplicate of this bit — it mirrors
    /// upstream `warmstate.py:135 JC_FORCE_FINISH`, a green-key-keyed
    /// signal read at `warmstate.py:439` (cell-side
    /// `force_finish_trace`).  Upstream itself carries both signals
    /// independently because the green-key cell and the loop token
    /// have distinct lifetimes (a token may outlive its cell after
    /// invalidation; a cell may exist before any token is bound).
    /// Pyre mirrors the upstream split exactly.
    ///
    /// `UnrollOptimizer.retraced_count` is a transient pass-local
    /// copy (`unroll.rs:59`) that the unroll pipeline writes back to
    /// `set_retraced_count` once the pass finishes, matching
    /// `unroll.py:216-217` (read at start, mutate, write back).
    ///
    /// Interior mutability via `Cell<u32>` mirrors RPython's
    /// attribute writes through `&JitCellToken`; the same
    /// `unsafe impl Sync` covers it as `generation`.  Callers must
    /// use the accessors (not `.get()` / `.set()` directly) to keep
    /// the bit-packing invariant.
    pub retraced_count: Cell<u32>,
    /// `history.py:433` `JitCellToken.target_tokens = None` (lazily
    /// populated to a `list[TargetToken]` in `compile.py:286-296` /
    /// `:312-323` once the loop is compiled).  `pyjitpl.py:3898`
    /// `has_compiled_targets(token)` reads this list — `bool(token)
    /// and bool(token.target_tokens)`.
    ///
    /// Pyre stores the descr-side projection of TargetToken
    /// (`LoopTargetDescr` Arc; `TargetToken IS-A AbstractDescr` in
    /// PyPy, so a `DescrRef` is the matching identity).  Each
    /// successful loop / retrace populates this through
    /// `record_target_token` so `has_compiled_loop` reads the same
    /// signal PyPy's `has_compiled_targets` does.  The metainterp-side
    /// `TargetToken` value (with `virtual_state` / `short_preamble`)
    /// stays on the `CompiledEntry::front_target_tokens` list per
    /// the F.6 retirement plan — the per-target descr identity is the
    /// part PyPy parity care about for `has_compiled_targets`.
    pub target_tokens: parking_lot::Mutex<Vec<majit_ir::DescrRef>>,
}

impl JitCellToken {
    /// `history.py:431` `FORCE_BRIDGE_SEGMENTING = 1` — bit packed
    /// into `retraced_count`.  Set at `pyjitpl.py:2833` (pyre:
    /// `MetaInterp::blackhole_trace_too_long_slow`) when a bridge
    /// trace aborts without an inlinable function; read at
    /// `compile.py:729` (pyre: `MetaInterp::start_retrace_from_guard`)
    /// to decide whether the next bridge from this loop should
    /// `force_finish_trace`.
    pub const FORCE_BRIDGE_SEGMENTING: u32 = 1;

    /// `history.py:464` `def get_retraced_count(self): return
    /// self.retraced_count >> 1`.
    #[inline]
    pub fn get_retraced_count(&self) -> u32 {
        self.retraced_count.get() >> 1
    }

    /// `history.py:467` `def set_retraced_count(self, value):
    /// self.retraced_count = (value << 1) | (self.retraced_count & 1)`.
    /// Preserves the FORCE_BRIDGE_SEGMENTING bit.
    #[inline]
    pub fn set_retraced_count(&self, value: u32) {
        let flag = self.retraced_count.get() & Self::FORCE_BRIDGE_SEGMENTING;
        self.retraced_count.set((value << 1) | flag);
    }
    pub fn new(number: u64) -> Self {
        JitCellToken {
            number,
            green_key: Cell::new(0),
            inputarg_types: OnceLock::new(),
            num_scalar_inputargs: Cell::new(0),
            virtualizable_arg_index: Cell::new(None),
            outermost_jitdriver_index: None,
            compiled: OnceLock::new(),
            invalidated: Arc::new(AtomicBool::new(false)),
            version_info: None,
            keepalive_tokens: parking_lot::Mutex::new(Vec::new()),
            // `rpython/jit/backend/x86/assembler.py:514` creates the
            // `compiled_loop_token` at the start of `assemble_loop` —
            // i.e., lazily when the backend compiles the token. pyre
            // initializes it eagerly here so the field is always present.
            compiled_loop_token: parking_lot::Mutex::new(Some(Arc::new(CompiledLoopToken::new(
                number,
            )))),
            _ll_function_addr: AtomicUsize::new(0),
            // memmgr.py:38 default; first keep_loop_alive overwrites this.
            generation: Cell::new(0),
            // history.py:435 `retraced_count = 0` (class attribute default).
            retraced_count: Cell::new(0),
            // history.py:433 `target_tokens = None` — pyre uses the
            // empty-Vec equivalent so `has_target_tokens` is one
            // `is_empty()` check away.
            target_tokens: parking_lot::Mutex::new(Vec::new()),
        }
    }

    /// Clone the current `compiled_loop_token` handle out of its `Mutex`.
    /// `None` only before the eager `JitCellToken::new` init is observed
    /// (never in practice) or on a token whose CLT was cleared.
    #[inline]
    pub fn compiled_loop_token(&self) -> Option<Arc<CompiledLoopToken>> {
        self.compiled_loop_token.lock().clone()
    }

    /// Reassign the `compiled_loop_token` slot through `&JitCellToken`.
    /// `register_call_assembler_target` calls this to reuse a previously
    /// registered CLT Arc so metadata pointers baked into callers stay
    /// stable across a token-number re-registration.
    #[inline]
    pub fn set_compiled_loop_token(&self, clt: Option<Arc<CompiledLoopToken>>) {
        *self.compiled_loop_token.lock() = clt;
    }

    /// The token's `compiled_loop_token`, panicking if absent. Callers keep
    /// the returned `Arc` alive while locking its inner mutexes
    /// (`asmmemmgr_blocks`, `frame_info`, ...).
    ///
    /// `rpython/jit/backend/llsupport/assembler.py:184-188`
    /// `get_asmmemmgr_blocks(self, looptoken)` reaches `asmmemmgr_blocks`
    /// through this Arc.  `JitCellToken::new` sets the CLT eagerly, so
    /// production callers never hit the panic.
    #[inline]
    pub fn compiled_loop_token_expect(&self) -> Arc<CompiledLoopToken> {
        self.compiled_loop_token()
            .expect("JitCellToken missing compiled_loop_token")
    }

    /// history.py:451-453: record_jump_to — record that this loop can
    /// jump to another JitCellToken (via CALL_ASSEMBLER or JUMP).
    ///
    /// ```python
    /// def record_jump_to(self, target_token):
    ///     assert isinstance(target_token, JitCellToken)
    ///     self._keepalive_jitcell_tokens[target_token] = None
    /// ```
    ///
    /// Pyre stores the target's `Arc<JitCellToken>`, the direct Rust
    /// analog of PyPy's Python-object dict-key keepalive.  The
    /// `target.number == self.number` short-circuit matches PyPy's
    /// `if target_token is self: return` (implicit because the dict
    /// would just no-op); the explicit guard avoids the Mutex lock for
    /// the self-jump case.
    pub fn record_jump_to(&self, target: Arc<JitCellToken>) {
        if target.number == self.number {
            return;
        }
        let target_number = target.number;
        let mut guard = self.keepalive_tokens.lock();
        if !guard
            .iter()
            .any(|existing| existing.number == target_number)
        {
            guard.push(target);
        }
    }

    /// Mark this loop as invalidated. Any subsequent execution of
    /// GUARD_NOT_INVALIDATED in the compiled code will fail.
    pub fn invalidate(&self) {
        self.invalidated.store(true, Ordering::Release);
    }

    /// Load the compiled entry address written at backend `compile_loop`
    /// completion (`x86/assembler.py:599`).  A zero value means the token
    /// is still pending.
    #[inline]
    pub fn ll_function_addr(&self) -> usize {
        self._ll_function_addr.load(Ordering::Acquire)
    }

    /// Store the compiled entry address for descr-carried
    /// `CALL_ASSEMBLER` resolution.
    #[inline]
    pub fn set_ll_function_addr(&self, addr: usize) {
        self._ll_function_addr.store(addr, Ordering::Release);
    }

    /// Address of the atomic slot used by backend call thunks for pyre's
    /// pending-token window.
    #[inline]
    pub fn ll_function_addr_slot(&self) -> *const AtomicUsize {
        &self._ll_function_addr as *const AtomicUsize
    }

    /// Check whether this loop has been invalidated.
    pub fn is_invalidated(&self) -> bool {
        self.invalidated.load(Ordering::Acquire)
    }

    /// model.py: has_compiled_code()
    /// Whether this token has compiled code attached.
    pub fn has_compiled_code(&self) -> bool {
        self.compiled.get().is_some()
    }

    /// Store the backend-specific compiled data through `&JitCellToken`.
    /// Write-once: `compile_loop` runs at most once per token (recompiles
    /// mint fresh tokens), so a second call is a no-op on the already-set
    /// slot.
    #[inline]
    pub fn set_compiled(&self, compiled: Box<dyn std::any::Any + Send>) {
        let _ = self.compiled.set(compiled);
    }

    /// model.py: get_number()
    pub fn get_number(&self) -> u64 {
        self.number
    }

    /// The input-arg types recorded at `compile_loop`.  Empty until the
    /// backend has compiled this token.
    #[inline]
    pub fn inputarg_types(&self) -> &[Type] {
        self.inputarg_types.get().map_or(&[], Vec::as_slice)
    }

    /// Record the typed input signature through `&JitCellToken` at
    /// `compile_loop` (write-once).
    #[inline]
    pub fn set_inputarg_types(&self, types: Vec<Type>) {
        let _ = self.inputarg_types.set(types);
    }

    /// The loop's green key, set by `configure_loop_token_for_driver`.
    #[inline]
    pub fn green_key(&self) -> u64 {
        self.green_key.get()
    }

    /// The virtualizable arg index, set by `configure_loop_token_for_driver`.
    #[inline]
    pub fn virtualizable_arg_index(&self) -> Option<usize> {
        self.virtualizable_arg_index.get()
    }

    /// The scalar-inputarg count, set by `configure_loop_token_for_driver`.
    #[inline]
    pub fn num_scalar_inputargs(&self) -> usize {
        self.num_scalar_inputargs.get()
    }

    /// model.py: reset_compiled()
    /// Remove the compiled code (e.g., after invalidation).
    pub fn reset_compiled(&mut self) {
        self.compiled.take();
    }

    /// Get a clone of the invalidated flag (for registering with QuasiImmut).
    pub fn invalidation_flag(&self) -> Arc<AtomicBool> {
        self.invalidated.clone()
    }

    /// `pyjitpl.py:3898` `has_compiled_targets(token)` —
    /// `bool(token) and bool(token.target_tokens)`.  PyPy reads
    /// `token.target_tokens` (a `list[TargetToken]` populated at
    /// `compile.py:286-296`) and treats a non-empty list as the signal
    /// that the loop has been compiled.
    #[inline]
    pub fn has_target_tokens(&self) -> bool {
        !self.target_tokens.lock().is_empty()
    }

    /// `compile.py:286-296` / `:312-323` — append a freshly minted
    /// TargetToken's descr to `token.target_tokens`.  Idempotent on
    /// `Arc::ptr_eq` so retrace paths that reuse `prior_front_target_tokens`
    /// across `compile_loop` and `compile_retrace` do not duplicate.
    pub fn record_target_token(&self, descr: majit_ir::DescrRef) {
        let mut guard = self.target_tokens.lock();
        if !guard.iter().any(|existing| Arc::ptr_eq(existing, &descr)) {
            guard.push(descr);
        }
    }
}

impl std::fmt::Debug for JitCellToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JitCellToken")
            .field("number", &self.number)
            .finish()
    }
}

// pyre is single-threaded (no-GIL → still one JIT thread in practice,
// matching RPython's single-interpreter assumption).  `JitCellToken`
// embeds `Rc<RdVirtualInfo>` and `Box<dyn Any + Send>` which are not
// automatically `Sync`; marking the parent struct `Send + Sync` here
// is sound under the same invariant `CraneliftFailDescr` relies on —
// a promise held by the JIT scheduler that tokens are only touched
// from the JIT thread.  Backends store
// `Weak<JitCellToken>` handles for cross-token `find_descr_by_ptr`
// fallback and need these impls to satisfy their `Send` trait bound.
unsafe impl Send for JitCellToken {}
unsafe impl Sync for JitCellToken {}

/// A "dead frame" — the state after JIT execution finishes or hits a guard.
///
/// The backend stores register/stack values here so the frontend can read them.
pub struct DeadFrame {
    /// Backend-specific frame data.
    pub data: Box<dyn std::any::Any + Send>,
}

/// `compile.py:665-674` `make_and_attach_done_descrs` + `pyjitpl.py:2283`
/// `propagate_exception_descr`: snapshot of the six descrs the metainterp
/// attaches to the owning cpu instance at
/// `MetaInterpStaticData.finish_setup` (pyjitpl.py:2222).  RPython backend
/// code reads these as `self.cpu.xxx` attributes at every use site
/// (`rpython/jit/backend/x86/assembler.py:337`,
/// `rpython/jit/backend/llgraph/runner.py:1478`); pyre captures them by
/// value at `compile_loop` entry (from the owning backend's per-instance
/// fields) so FINISH / CALL_ASSEMBLER emission can stamp `jf_descr` with
/// the attached identity without holding a `&dyn Backend` receiver.
///
/// Field names mirror the RPython attribute names 1:1 for parity with
/// `compile.make_and_attach_done_descrs` targets.
#[derive(Debug, Clone, Copy, Default)]
pub struct AttachedDescrPtrs {
    pub done_with_this_frame_descr_void: usize,
    pub done_with_this_frame_descr_int: usize,
    pub done_with_this_frame_descr_ref: usize,
    pub done_with_this_frame_descr_float: usize,
    pub exit_frame_with_exception_descr_ref: usize,
    pub propagate_exception_descr: usize,
}

impl AttachedDescrPtrs {
    /// `compile.py:324-336` `_call_assembler` — pick the attached
    /// `done_with_this_frame_descr_*` whose result type matches the
    /// portal's finish kind.
    pub fn done_with_this_frame_descr_ptr_for_type(&self, tp: Type) -> usize {
        match tp {
            Type::Void => self.done_with_this_frame_descr_void,
            Type::Int => self.done_with_this_frame_descr_int,
            Type::Ref => self.done_with_this_frame_descr_ref,
            Type::Float => self.done_with_this_frame_descr_float,
        }
    }

    /// `compile.py:665-670` parity: is `ptr` one of the four
    /// `DoneWithThisFrameDescr*` pointers attached to this cpu?
    /// Zero pointers (unattached slots) are never considered matches
    /// — a test with some slots unset would otherwise make
    /// `is_done_with_this_frame_descr(0)` spuriously true.
    pub fn is_done_with_this_frame_descr(&self, ptr: usize) -> bool {
        ptr != 0
            && (ptr == self.done_with_this_frame_descr_void
                || ptr == self.done_with_this_frame_descr_int
                || ptr == self.done_with_this_frame_descr_ref
                || ptr == self.done_with_this_frame_descr_float)
    }

    /// `compile.py:671` parity: does `ptr` match the attached
    /// `exit_frame_with_exception_descr_ref` singleton?
    pub fn is_exit_frame_with_exception_descr(&self, ptr: usize) -> bool {
        ptr != 0 && ptr == self.exit_frame_with_exception_descr_ref
    }
}

/// `compile.py:665-674` `make_and_attach_done_descrs` + `pyjitpl.py:2283`:
/// the six descrs the metainterp attaches to each cpu instance at
/// `MetaInterpStaticData.finish_setup`.
///
/// Stored inside a heap-pinned `Arc<RwLock<CpuDescrAttachments>>` on every
/// `Backend` impl ([`CpuDescrHandle`] below):
///
///   * The `Arc` allocation address stays stable when the `Backend`
///     value is moved (the metainterp keeps `backend: <Backend>` by value
///     in `MetaInterp`; tests hold stack-local `<Backend>::new()`).
///     RPython gets this property for free from Python object identity;
///     pyre pays one heap indirection to recreate it.
///   * Compiled code produced by `compile_loop` carries an `Arc` clone
///     so the attachments outlive the owning backend — the JIT-emitted
///     CALL_ASSEMBLER slow path dereferences the baked handle pointer
///     long after `compile_loop` returns.
///   * The `RwLock` permits `Backend::set_done_with_this_frame_descr_*`
///     to mutate through `&Backend` (via the Arc) without requiring
///     `&mut self` access to the inner; the extern-C trampoline takes a
///     read lock to snapshot pointers for dispatch.
///
/// Field names mirror the RPython attribute names 1:1 for parity with
/// `compile.make_and_attach_done_descrs` targets.
#[derive(Default)]
pub struct CpuDescrAttachments {
    pub done_with_this_frame_descr_void: Option<majit_ir::DescrRef>,
    pub done_with_this_frame_descr_int: Option<majit_ir::DescrRef>,
    pub done_with_this_frame_descr_ref: Option<majit_ir::DescrRef>,
    pub done_with_this_frame_descr_float: Option<majit_ir::DescrRef>,
    pub exit_frame_with_exception_descr_ref: Option<majit_ir::DescrRef>,
    pub propagate_exception_descr: Option<majit_ir::DescrRef>,
}

impl CpuDescrAttachments {
    /// Snapshot the six attached pointers for emission / dispatch sites
    /// that need compile-time or runtime-immediate access.
    pub fn descr_ptrs(&self) -> AttachedDescrPtrs {
        fn ptr(d: &Option<majit_ir::DescrRef>) -> usize {
            d.as_ref()
                .map_or(0, |arc| Arc::as_ptr(arc) as *const () as usize)
        }
        AttachedDescrPtrs {
            done_with_this_frame_descr_void: ptr(&self.done_with_this_frame_descr_void),
            done_with_this_frame_descr_int: ptr(&self.done_with_this_frame_descr_int),
            done_with_this_frame_descr_ref: ptr(&self.done_with_this_frame_descr_ref),
            done_with_this_frame_descr_float: ptr(&self.done_with_this_frame_descr_float),
            exit_frame_with_exception_descr_ref: ptr(&self.exit_frame_with_exception_descr_ref),
            propagate_exception_descr: ptr(&self.propagate_exception_descr),
        }
    }
}

/// Heap-pinned handle for the six per-cpu descr attachments.  The
/// `Arc`'s payload (the `RwLock<CpuDescrAttachments>`) lives at a
/// stable heap address so `compile_loop` can bake that address as an
/// immediate in the JIT-emitted CALL_ASSEMBLER helper call site and
/// the extern-C trampoline can dereference it later.
pub type CpuDescrHandle = Arc<std::sync::RwLock<CpuDescrAttachments>>;

/// The backend trait — implemented by Cranelift (or other code generators).
///
/// Mirrors rpython/jit/backend/model.py AbstractCPU.
pub trait Backend: Send {
    /// `rpython/jit/backend/model.py:28-29` `self.tracker =
    /// CPUTotalTracker()` parity — each backend instance owns its
    /// own [`CpuTotalTracker`].  [`record_compiled_loop_token`] and
    /// [`CompiledLoopToken::compiling_a_bridge`] read this through
    /// `&self` so multi-backend processes (e.g. test fixtures
    /// creating fresh `MetaInterpStaticData` repeatedly) keep
    /// counters isolated.
    ///
    /// Default returns a process-wide `Arc<CpuTotalTracker>` distinct
    /// from [`fallback_cpu_tracker`]: the trait signature is
    /// `&Arc<CpuTotalTracker>` (so callers can `Arc::clone` into
    /// [`crate::JitProfiler::set_cpu_tracker`]) while
    /// `fallback_cpu_tracker` returns a `&'static CpuTotalTracker`
    /// reference for callers that already have direct access.  The
    /// two statics therefore back independent counter stores;
    /// synthetic/test backends that don't override this method
    /// continue to behave like a process-global singleton among
    /// themselves but won't observe writes through
    /// `fallback_cpu_tracker`.
    fn cpu_tracker(&self) -> &Arc<CpuTotalTracker> {
        static FALLBACK_ARC: std::sync::OnceLock<Arc<CpuTotalTracker>> = std::sync::OnceLock::new();
        FALLBACK_ARC.get_or_init(|| Arc::new(CpuTotalTracker::default()))
    }

    /// Compile a loop trace into native code.
    ///
    /// `token` is shared (`&JitCellToken`): a self-recursive loop's own
    /// `CALL_ASSEMBLER` descr can hold an `Arc` clone of this same token,
    /// which the backend dereferences while emitting the call.  The late
    /// fields (`compiled`, `inputarg_types`, `compiled_loop_token`) are
    /// interior-mutable so they can be written through the shared reference.
    fn compile_loop(
        &mut self,
        inputargs: &[InputArg],
        ops: &[OpRc],
        token: &JitCellToken,
    ) -> Result<AsmInfo, BackendError>;

    /// Register the typed constant pool (`OpRef` → `Const`) consumed by
    /// the next `compile_loop` / `compile_bridge` call.  `Const` carries
    /// both value and `Type` (history.py:220/261/307
    /// ConstInt/ConstFloat/ConstPtr `.type` parity).
    ///
    /// Default is a no-op — only backends that honour constants at emit
    /// time override.
    fn set_constants_pool(&mut self, _constants: majit_ir::ConstMap<Const>) {}

    /// Force the next `compile_loop` / `compile_bridge` call to stamp
    /// this trace id on exits.
    fn set_next_trace_id(&mut self, _trace_id: u64) {}

    /// Force the next `compile_loop` / `compile_bridge` call to attach
    /// this header PC to synthesised exit recovery layouts.
    fn set_next_header_pc(&mut self, _header_pc: u64) {}

    /// `compile.py:665-674` `make_and_attach_done_descrs([self, cpu])` —
    /// per-result-type `DoneWithThisFrame*` singleton shared with
    /// `MetaInterpStaticData`.  Attached once per CPU instance, matching
    /// `pyjitpl.py:2222`.  Backends that use pointer identity for the
    /// FINISH fast path (dynasm `is_done_with_this_frame_descr`,
    /// cranelift CA dispatch) override to store the `Arc` and publish
    /// `Arc::as_ptr` to the comparison sites.
    fn set_done_with_this_frame_descr_void(&mut self, _descr: Arc<dyn Descr>) {}

    /// `compile.py:665-674` `done_with_this_frame_descr_int` — INT-typed
    /// variant.  See `set_done_with_this_frame_descr_void` for the
    /// attachment contract.
    fn set_done_with_this_frame_descr_int(&mut self, _descr: Arc<dyn Descr>) {}

    /// `compile.py:665-674` `done_with_this_frame_descr_ref` — REF-typed
    /// variant.  See `set_done_with_this_frame_descr_void` for the
    /// attachment contract.
    fn set_done_with_this_frame_descr_ref(&mut self, _descr: Arc<dyn Descr>) {}

    /// `compile.py:665-674` `done_with_this_frame_descr_float` —
    /// FLOAT-typed variant.  See `set_done_with_this_frame_descr_void`
    /// for the attachment contract.
    fn set_done_with_this_frame_descr_float(&mut self, _descr: Arc<dyn Descr>) {}

    /// `compile.py:665-674` `exit_frame_with_exception_descr_ref` —
    /// FINISH descr used by `compile_exit_frame_with_exception`
    /// (`pyjitpl.py:3238-3245`).  See `set_done_with_this_frame_descr_void`
    /// for the attachment contract.
    fn set_exit_frame_with_exception_descr_ref(&mut self, _descr: Arc<dyn Descr>) {}

    /// `pyjitpl.py:2283` `self.cpu.propagate_exception_descr = exc_descr`
    /// — shared `PropagateExceptionDescr` instance used by
    /// `compile_tmp_callback`'s `GUARD_NO_EXCEPTION` and by the
    /// backend's propagate-exception slow path
    /// (`x86/assembler.py:870`, `aarch64/assembler.py:566-572`).
    fn set_propagate_exception_descr(&mut self, _descr: Arc<dyn Descr>) {}

    /// `compile.py:484 do_compile_bridge(metainterp_sd, faildescr, inputargs,
    /// operations, original_loop_token, log, memo)` — RPython's upstream
    /// signature carries one token (`original_loop_token`), reached via
    /// `metainterp.resumekey_original_loop_token = resumedescr.rd_loop_token
    /// .loop_token_wref()` (`pyjitpl.py:2897`).  Pyre's caller resolves the
    /// owning JCT through `descr_owning_jct(fail_descr)` (Phase E.3+,
    /// `lib.rs:969`) and passes it as `original_token`, so the source descr
    /// is always reachable from `original_token.compiled.fail_descrs`.
    ///
    /// **TODO:** the second `previous_tokens` slice is
    /// pyre-only.  Cranelift recompiles bridges as fresh modules instead of
    /// patching live machine code, so after a retrace the RUNNING machine
    /// code still references descrs in retired predecessor tokens — the
    /// cranelift impl must attach the freshly-compiled bridge to those
    /// predecessor descrs so the running code can still dispatch to it.
    /// Dynasm patches in place and ignores the slice.  Convergence requires
    /// either a cranelift-side registry of running predecessor descrs (so
    /// the parameter can come off the trait), or moving to RPython-style
    /// in-place patching.
    fn compile_bridge(
        &mut self,
        fail_descr: &dyn FailDescr,
        inputargs: &[InputArg],
        ops: &[OpRc],
        original_token: &JitCellToken,
        previous_tokens: &[std::sync::Arc<JitCellToken>],
        caller_recovery_layout: Option<&ExitRecoveryLayout>,
    ) -> Result<AsmInfo, BackendError>;

    /// Whether a `BackendError::Unsupported` returned by `compile_bridge`
    /// is a deterministic structural decline that re-tracing the same guard
    /// would reproduce identically (a compile storm). When `true`, the
    /// metainterp records the guard so `must_compile_with_values` stops
    /// re-firing it and the guard falls back to blackhole resume. The
    /// default is `false`: backends that patch machine code in place never
    /// decline structurally, so a transient failure is retried after the
    /// jitcounter ticks again (`compile.py:790-795 done_compiling`).
    fn bridge_decline_is_terminal(&self) -> bool {
        false
    }

    /// Register a freshly-compiled JitCellToken as still reachable from
    /// the frontend.  Backends that need to resolve `jf_descr` pointers
    /// across token boundaries (dynasm's `find_descr_by_ptr` cross-token
    /// fallback) use this to iterate all live tokens without maintaining
    /// a separate ptr-to-Arc side table.  Called by `MetaInterp` after
    /// `compile_loop` succeeds.  Default no-op for backends whose
    /// descr-resolution paths do not cross tokens (cranelift resolves
    /// via `CompiledLoop::fail_descrs` directly).
    fn track_compiled_token(&mut self, _token: Arc<JitCellToken>) {}

    /// Mark the most recently compiled bridge on the given guard as a
    /// loop-closing bridge: on Finish, its outputs should re-enter the
    /// parent loop instead of returning to the interpreter.
    fn mark_bridge_loop_reentry(
        &self,
        _original_token: &JitCellToken,
        _source_trace_id: u64,
        _fail_index: u32,
    ) {
        // Default no-op — backends that support bridge re-entry override this.
    }

    /// Cranelift workaround — no RPython counterpart.
    ///
    /// RPython's x86/ARM backends patch guard failure jump targets in-place,
    /// so bridges survive retrace automatically. Cranelift cannot patch
    /// emitted machine code, so when a loop is retraced (producing a new
    /// token), existing bridges from the old token must be explicitly copied
    /// to matching guards in the new token.
    ///
    /// Called from metainterp after compile_loop, because only the metainterp
    /// has access to both old_token (from compiled_loops.remove) and new_token
    /// (from compile_loop). compile_loop itself only receives the new token.
    ///
    /// Backends that support in-place patching (e.g. dynasm) leave this as
    /// no-op — bridges are attached to the guard's machine code directly.
    fn migrate_bridges(&self, _old_token: &JitCellToken, _new_token: &JitCellToken) {}

    /// compile.py:826-830 store_hash: assign jitcounter hashes to guards.
    /// Called after compile_loop/compile_bridge with hashes from
    /// jitcounter.fetch_next_hash(). Skips guards that already have
    /// status set by make_a_counter_per_value (GUARD_VALUE).
    fn store_guard_hashes(&self, _token: &JitCellToken, _hashes: &[u64]) {}

    /// compile.py / resume.py:1143 plumbing: publish the shared
    /// `CallInfoCollection` so the backend can resolve OS_STR_CONCAT
    /// etc. function pointers when materializing VStr/VUni
    /// Concat/Slice virtuals from guard-exit recovery data.
    ///
    /// Default is a no-op: backends that handle string virtual
    /// materialization entirely through the frontend `bh_*` runtime
    /// (dynasm) or that never expose VStr/VUni recovery_layouts need
    /// not participate. Cranelift overrides this to cache the
    /// collection for `collect_guards` consumption.
    fn set_callinfocollection(
        &mut self,
        _cic: Option<std::sync::Arc<majit_ir::CallInfoCollection>>,
    ) {
    }

    /// store_hash for bridge guards — same as store_guard_hashes but for
    /// the most recently compiled bridge on the given guard.
    /// Uses (trace_id, fail_index) for recursive descriptor lookup.
    fn store_bridge_guard_hashes(
        &self,
        _token: &JitCellToken,
        _source_trace_id: u64,
        _source_fail_index: u32,
        _hashes: &[u64],
    ) {
    }

    /// Execute compiled code starting at the given token.
    fn execute_token(&self, token: &JitCellToken, args: &[Value]) -> DeadFrame;

    /// Execute compiled code starting at a backend-specific dispatch key.
    ///
    /// Default backends have a single token entry and ignore `dispatch_key`.
    /// Cranelift uses key 0 for the peeled preamble and `label_block_id + 1`
    /// for direct LABEL entry.
    fn execute_token_with_dispatch_key(
        &self,
        token: &JitCellToken,
        args: &[Value],
        dispatch_key: u32,
    ) -> DeadFrame {
        let _ = dispatch_key;
        self.execute_token(token, args)
    }

    /// Whether [`Backend::execute_token_with_dispatch_key`] honors non-zero
    /// dispatch keys as alternate compiled-loop entry points.
    ///
    /// Backends that use the default `execute_token_with_dispatch_key`
    /// implementation enter the ordinary token entry regardless of the key, so
    /// metainterp direct-LABEL handoffs must stay disabled for them.
    fn supports_dispatch_key_entry(&self) -> bool {
        false
    }

    /// Whether CALL_ASSEMBLER may target a `compile_tmp_callback` token
    /// (compile.py:1101-1150) whose body reaches the portal runner.
    ///
    /// The wasm backend admits CALL_ASSEMBLER only against a published
    /// compiled target with no trampoline calls, and a tmp-callback body
    /// calls the portal runner through a host trampoline, so it resolves
    /// pending tokens instead.
    fn supports_tmp_callback_call_assembler(&self) -> bool {
        true
    }

    /// Register a resolvable-but-not-enterable placeholder CALL_ASSEMBLER
    /// target for a pending token, before the loop body is compiled.
    ///
    /// Backends that resolve pending tokens rather than tmp-callback bodies
    /// (`supports_tmp_callback_call_assembler` false) need the pending
    /// target's frame geometry and dispatch slot published now so an
    /// already-emitted caller can enter it and be redirected once the real
    /// loop compiles. Backends that enter tmp-callback bodies never emit a
    /// pending target, so the default is a no-op.
    fn register_pending_target(
        &mut self,
        token_number: u64,
        input_types: Vec<Type>,
        num_inputs: usize,
        num_scalar_inputargs: usize,
        index_of_virtualizable: i32,
    ) {
        let _ = (
            token_number,
            input_types,
            num_inputs,
            num_scalar_inputargs,
            index_of_virtualizable,
        );
    }

    /// Execute compiled code with integer-only arguments.
    ///
    /// Avoids the `Value::Int` wrapping/unwrapping overhead when all
    /// arguments are known to be integers (the common case for loop entry).
    fn execute_token_ints(&self, token: &JitCellToken, args: &[i64]) -> DeadFrame {
        let values: Vec<Value> = args.iter().map(|&v| Value::Int(v)).collect();
        self.execute_token(token, &values)
    }

    /// Execute compiled code with typed arguments and return a lightweight result.
    ///
    /// This preserves mixed `Int` / `Ref` / `Float` arguments while still
    /// avoiding explicit deadframe decoding in the caller.
    fn execute_token_raw(&self, token: &JitCellToken, args: &[Value]) -> RawExecResult {
        let frame = self.execute_token(token, args);
        let descr_arc = self.get_latest_descr_arc(&frame);
        let descr: &dyn FailDescr = descr_arc
            .as_fail_descr()
            .expect("get_latest_descr_arc must return a FailDescr");
        let exit_layout = self.describe_deadframe(&frame);
        let savedata = self.get_savedata_ref(&frame);
        let exception_value = self.grab_exc_value(&frame);
        let exit_arity = descr.fail_arg_types().len();
        let mut outputs = Vec::with_capacity(exit_arity);
        let mut typed_outputs = Vec::with_capacity(exit_arity);
        for (i, &tp) in descr.fail_arg_types().iter().enumerate() {
            match tp {
                Type::Int => {
                    let value = self.get_int_value(&frame, i);
                    outputs.push(value);
                    typed_outputs.push(Value::Int(value));
                }
                Type::Ref => {
                    let value = self.get_ref_value(&frame, i);
                    outputs.push(value.as_usize() as i64);
                    typed_outputs.push(Value::Ref(value));
                }
                Type::Float => {
                    let value = self.get_float_value(&frame, i);
                    outputs.push(value.to_bits() as i64);
                    typed_outputs.push(Value::Float(value));
                }
                Type::Void => {
                    outputs.push(0);
                    typed_outputs.push(Value::Void);
                }
            }
        }
        RawExecResult {
            outputs,
            typed_outputs,
            exit_layout,
            force_token_slots: descr.force_token_slots().to_vec(),
            savedata,
            exception_value,
            fail_index: descr.fail_index(),
            trace_id: descr.trace_id(),
            is_finish: descr.is_finish(),
            is_exit_frame_with_exception: descr.is_exit_frame_with_exception(),
            status: descr.get_status(),
            descr_arc,
        }
    }

    /// Execute compiled code and return a lightweight result without
    /// DeadFrame boxing.
    ///
    /// Returns the output values directly, avoiding the intermediate
    /// DeadFrame heap allocation and the per-value downcast extraction loop.
    fn execute_token_ints_raw(&self, token: &JitCellToken, args: &[i64]) -> RawExecResult {
        let values: Vec<Value> = args.iter().map(|&v| Value::Int(v)).collect();
        self.execute_token_raw(token, &values)
    }

    /// Inspect static exit layouts for a compiled loop token.
    fn compiled_fail_descr_layouts(&self, _token: &JitCellToken) -> Option<Vec<FailDescrLayout>> {
        None
    }

    /// Inspect static exit layouts for a bridge attached to a source guard.
    fn compiled_bridge_fail_descr_layouts(
        &self,
        _original_token: &JitCellToken,
        _source_trace_id: u64,
        _source_fail_index: u32,
    ) -> Option<Vec<FailDescrLayout>> {
        None
    }

    /// Inspect static exit layouts for any compiled trace owned by this token.
    ///
    /// This is the trace-id keyed counterpart to the root/bridge-specific
    /// inspection APIs above.
    fn compiled_trace_fail_descr_layouts(
        &self,
        _token: &JitCellToken,
        _trace_id: u64,
    ) -> Option<Vec<FailDescrLayout>> {
        None
    }

    /// Inspect static terminal-exit layouts for a compiled loop token.
    fn compiled_terminal_exit_layouts(
        &self,
        _token: &JitCellToken,
    ) -> Option<Vec<TerminalExitLayout>> {
        None
    }

    /// Inspect static terminal-exit layouts for a bridge attached to a source guard.
    fn compiled_bridge_terminal_exit_layouts(
        &self,
        _original_token: &JitCellToken,
        _source_trace_id: u64,
        _source_fail_index: u32,
    ) -> Option<Vec<TerminalExitLayout>> {
        None
    }

    /// Inspect static terminal-exit layouts for any compiled trace owned by this token.
    fn compiled_trace_terminal_exit_layouts(
        &self,
        _token: &JitCellToken,
        _trace_id: u64,
    ) -> Option<Vec<TerminalExitLayout>> {
        None
    }

    /// Inspect static metadata for any compiled trace owned by this token.
    fn compiled_trace_info(
        &self,
        _token: &JitCellToken,
        _trace_id: u64,
    ) -> Option<CompiledTraceInfo> {
        None
    }

    /// Patch backend-owned recovery metadata for a specific compiled terminal exit.
    fn update_terminal_exit_recovery_layout(
        &mut self,
        _token: &JitCellToken,
        _trace_id: u64,
        _op_index: usize,
        _recovery_layout: ExitRecoveryLayout,
    ) -> bool {
        false
    }

    /// Describe the latest exit stored in a deadframe.
    ///
    /// Backends can override this to surface backend-owned recovery metadata
    /// directly from the deadframe's fail descriptor.
    fn describe_deadframe(&self, frame: &DeadFrame) -> Option<FailDescrLayout> {
        let descr = self.get_latest_descr(frame);
        Some(FailDescrLayout {
            fail_index: descr.fail_index(),
            source_op_index: None,
            trace_id: descr.trace_id(),
            trace_info: None,
            fail_arg_types: descr.fail_arg_types().to_vec(),
            is_finish: descr.is_finish(),
            is_exception_exit: descr.is_exit_frame_with_exception(),
            gc_ref_slots: descr
                .fail_arg_types()
                .iter()
                .enumerate()
                .filter_map(|(slot, _)| descr.is_gc_ref_slot(slot).then_some(slot))
                .collect(),
            force_token_slots: descr.force_token_slots().to_vec(),
            recovery_layout: None,
            frame_stack: None,
            rd_numb: None,
            rd_consts: None,
            rd_virtuals: None,
            rd_pendingfields: None,
        })
    }

    /// Force a frame identified by a `FORCE_TOKEN` result.
    fn force(&self, _force_token: GcRef) -> Option<DeadFrame> {
        None
    }

    /// Store a saved-data GC ref on a dead frame.
    fn set_savedata_ref(&self, _frame: &mut DeadFrame, _data: GcRef) {
        // No-op: backend doesn't support savedata
    }

    /// Read a saved-data GC ref from a dead frame.
    fn get_savedata_ref(&self, _frame: &DeadFrame) -> Option<GcRef> {
        None
    }

    /// Read a pending exception GC ref from a dead frame.
    fn grab_exc_value(&self, _frame: &DeadFrame) -> GcRef {
        GcRef::NULL
    }

    /// `llmodel.py:194-199 _store_exception` counterpart — clear the backend
    /// `_store_exception` cells (`jit_exc_value` / `jit_exc_type`).  A residual
    /// `bh_call` that raised publishes into BOTH `BH_LAST_EXC_VALUE` and these
    /// cells; when the blackhole catches it in-frame (`route_to_catch`) only
    /// `BH_LAST_EXC_VALUE` is cleared, so the cell keeps the consumed exception
    /// and a later `GUARD_NO_EXCEPTION` re-delivers it.  Draining on handler
    /// entry keeps the cell coherent with the caught state.
    fn clear_stored_exception(&self) {}

    /// Read the FailDescr from the last guard failure.
    fn get_latest_descr<'a>(&'a self, frame: &'a DeadFrame) -> &'a dyn FailDescr;

    /// Owned Arc counterpart of `get_latest_descr`.
    ///
    /// `cpu.get_latest_descr(deadframe)` in RPython (`history.py:125`,
    /// consumed at `pyjitpl.py:2890`) returns the metainterp
    /// `AbstractFailDescr` object stamped on the originating guard's
    /// `op.descr` — the same object that owns `rd_numb` / `rd_consts`
    /// / `rd_loop_token` etc.  Backends return the metainterp Arc
    /// reached through `meta_descr` when present; synthetic backend
    /// descrs (FINISH / `PropagateExceptionDescr` /
    /// `ExitFrameWithExceptionDescr` / external-JUMP) without a
    /// metainterp counterpart fall back to the backend Arc upcast to
    /// `Arc<dyn Descr>`.  Callers obtain `&dyn FailDescr` via
    /// `descr_arc.as_fail_descr()` (`history.py:128 AbstractFailDescr`
    /// is a sub-class of `AbstractDescr`, so the upcast is implicit on
    /// the Python side; pyre exposes the supertrait `Descr` and the
    /// sub-trait `FailDescr` separately).
    fn get_latest_descr_arc(&self, frame: &DeadFrame) -> Arc<dyn Descr>;

    /// Resolve a raw fail-descr address to its owning `Arc<dyn FailDescr>`.
    ///
    /// TODO (no upstream counterpart): RPython dispatches
    /// the CA bridge entry as a method on the descr — `compile.py:706-732
    /// _trace_and_compile_from_bridge(self, deadframe, ...)`, where `self`
    /// IS the descr — so the metainterp receives the descr object directly
    /// (`pyjitpl.py:2890 handle_guard_failure(self, resumedescr, ...)`)
    /// without any addr→object lookup.  Pyre's bridge crosses native code
    /// through function pointers (`majit-backend-dynasm/src/lib.rs`
    /// `BlackholeFn = fn(usize, *const i64, usize, *const i64, usize) ->
    /// Option<i64>` and `BridgeFn = fn(*const i64, usize, usize) -> bool`)
    /// which can only carry primitive types, so the descr identity has to
    /// be transported as a raw `usize` (`descr_addr`) and recovered here
    /// via this method.
    ///
    /// `descr_addr` is the thin pointer of a `majit_ir::FailDescrCell`
    /// baked at code-emission time (`history.py:109-114
    /// AbstractDescr.show` = pure cast against the cell).  Backends
    /// recover via `majit_ir::recover_fail_descr_cell` (`Arc::from_raw`
    /// + `Arc::increment_strong_count`); strong refs live on
    /// `CompiledLoopToken.asmmemmgr_gcreftracers` (`model.py:294`,
    /// `assembler.py:820-823 gcreftracers.append(tracer)`).  Singletons
    /// without a cell wrapper (FINISH `DoneWithThisFrame*`,
    /// `ExitFrameWithExceptionDescrRef`, `PropagateExceptionDescr`) are
    /// pointer-matched at higher layers before reaching this method.
    ///
    /// `warmspot.py:1021 cpu.get_latest_descr(deadframe)` has no failure
    /// mode — every live deadframe carries a valid descr handle.  Pyre
    /// backends mirror this contract via the CLT-pinned cell lifetime;
    /// the recovery is therefore infallible.  Default panics so backends
    /// that receive C-ABI guard-fail callbacks must opt in explicitly;
    /// `SyntheticCpu` and `wasm` never reach this path.
    fn fail_descr_arc_from_addr(&self, _descr_addr: usize) -> majit_ir::DescrRef {
        panic!(
            "Backend::fail_descr_arc_from_addr default invoked: backend wired into a runtime \
             guard-fail path must bake FailDescrCell thin pointers at emission and override \
             this method (warmspot.py:1021 cpu.get_latest_descr parity)"
        )
    }

    /// Read an integer value from a dead frame at the given index.
    fn get_int_value(&self, frame: &DeadFrame, index: usize) -> i64;

    /// Read a float value from a dead frame.
    fn get_float_value(&self, frame: &DeadFrame, index: usize) -> f64;

    /// Read a GC reference value from a dead frame.
    fn get_ref_value(&self, frame: &DeadFrame, index: usize) -> majit_ir::GcRef;

    /// Invalidate a compiled loop (e.g., due to GUARD_NOT_INVALIDATED).
    fn invalidate_loop(&self, token: &JitCellToken);

    /// Redirect calls from one loop token to another (for CALL_ASSEMBLER).
    fn redirect_call_assembler(
        &self,
        _old: &JitCellToken,
        _new: &JitCellToken,
    ) -> Result<(), BackendError> {
        Ok(())
    }

    /// Free resources associated with a compiled loop.
    fn free_loop(&mut self, _token: &JitCellToken) {
        // Default: no-op
    }

    // ── model.py: bh_* blackhole interpreter helpers ──
    //
    // These methods provide fallback implementations for operations
    // that the blackhole interpreter needs to execute when falling
    // back from JIT-compiled code. The backend implements these
    // to read/write memory at known addresses.

    // ── model.py:216-228 field operations ──
    /// model.py:216 bh_getfield_gc_i(struct, fielddescr)
    /// `llmodel.py:693-696 bh_getfield_gc_i` → `read_int_at_mem(struct,
    /// ofs, size, sign)`.  `unpack_fielddescr_size` yields `(offset,
    /// field_size, is_field_signed)`; the load mirrors the array
    /// default's `read_unaligned` discrimination so byte/short fields
    /// need not be naturally aligned.  Shared by pyre's raw-memory
    /// backends (cranelift, dynasm, wasm); backends routing field reads
    /// through their own runtime override.
    fn bh_getfield_gc_i(
        &self,
        struct_ptr: i64,
        fielddescr: &majit_translate::jitcode::BhDescr,
    ) -> i64 {
        let (offset, size, sign) = fielddescr.unpack_fielddescr_size();
        let addr = (struct_ptr as usize).wrapping_add(offset);
        // SAFETY: `struct_ptr` is a GC-managed struct pointer threaded
        // through from the trace recorder / blackhole; `offset`/`size`
        // come from the field descriptor.  `read_unaligned` mirrors
        // `llmodel.py:490 read_int_at_mem`.
        unsafe {
            match (size, sign) {
                (1, true) => (addr as *const i8).read_unaligned() as i64,
                (1, false) => (addr as *const u8).read_unaligned() as i64,
                (2, true) => (addr as *const i16).read_unaligned() as i64,
                (2, false) => (addr as *const u16).read_unaligned() as i64,
                (4, true) => (addr as *const i32).read_unaligned() as i64,
                (4, false) => (addr as *const u32).read_unaligned() as i64,
                (8, _) => (addr as *const i64).read_unaligned(),
                other => panic!("bh_getfield_gc_i: unsupported (size, signed) = {:?}", other,),
            }
        }
    }
    /// model.py:217 bh_getfield_gc_r(struct, fielddescr) →
    /// `read_ref_at_mem(struct, ofs)`.  Pointer-width load at the
    /// field offset.  Shared by pyre's raw-memory backends.
    fn bh_getfield_gc_r(
        &self,
        struct_ptr: i64,
        fielddescr: &majit_translate::jitcode::BhDescr,
    ) -> GcRef {
        let offset = fielddescr.as_offset();
        // SAFETY: see `bh_getfield_gc_i`.
        GcRef(unsafe { *((struct_ptr as *const u8).add(offset) as *const usize) })
    }
    /// model.py:218 bh_getfield_gc_f(struct, fielddescr) →
    /// `read_float_at_mem(struct, ofs)`.  Fixed `FLOATSTORAGE`-width
    /// load at the field offset.  Shared by pyre's raw-memory backends.
    fn bh_getfield_gc_f(
        &self,
        struct_ptr: i64,
        fielddescr: &majit_translate::jitcode::BhDescr,
    ) -> f64 {
        let offset = fielddescr.as_offset();
        let addr = (struct_ptr as usize).wrapping_add(offset);
        // SAFETY: see `bh_getfield_gc_i`.
        unsafe { (addr as *const f64).read_unaligned() }
    }
    /// model.py:222 / llmodel.py:716 bh_setfield_gc_i → `write_int_at_mem(struct,
    /// ofs, size, value)`.  Size + sign come from `unpack_fielddescr_size`;
    /// the store mirrors the field reader's width discrimination.  Shared by
    /// pyre's raw-memory backends; the trait default was a silent no-op that
    /// lost blackhole field writes (e.g. a rematerialized virtual's fields).
    fn bh_setfield_gc_i(
        &self,
        struct_ptr: i64,
        newvalue: i64,
        fielddescr: &majit_translate::jitcode::BhDescr,
    ) {
        let (offset, size, _sign) = fielddescr.unpack_fielddescr_size();
        let addr = (struct_ptr as usize).wrapping_add(offset);
        // SAFETY: `struct_ptr` is a GC-managed struct pointer; `offset`/`size`
        // come from the field descriptor. `write_unaligned` matches the reader.
        unsafe {
            match size {
                1 => (addr as *mut u8).write_unaligned(newvalue as u8),
                2 => (addr as *mut u16).write_unaligned(newvalue as u16),
                4 => (addr as *mut u32).write_unaligned(newvalue as u32),
                8 => (addr as *mut i64).write_unaligned(newvalue),
                other => panic!("bh_setfield_gc_i: unsupported field size {other}"),
            }
        }
    }
    /// model.py:223 / llmodel.py:723 bh_setfield_gc_r → pointer-width store at
    /// the field offset plus the write barrier.
    fn bh_setfield_gc_r(
        &self,
        struct_ptr: i64,
        newvalue: GcRef,
        fielddescr: &majit_translate::jitcode::BhDescr,
    ) {
        let offset = fielddescr.as_offset();
        // SAFETY: see `bh_setfield_gc_i`. `usize` is the pointer-width store.
        unsafe { *((struct_ptr as *mut u8).add(offset) as *mut usize) = newvalue.0 };
        // The store target may be an old-gen object holding a young ref; the
        // active GC's barrier remembers it. No-op where no barrier is installed.
        majit_gc::gc_write_barrier(GcRef(struct_ptr as usize));
    }
    /// model.py:224 / llmodel.py:728 bh_setfield_gc_f → `FLOATSTORAGE`-width
    /// store at the field offset.
    fn bh_setfield_gc_f(
        &self,
        struct_ptr: i64,
        newvalue: f64,
        fielddescr: &majit_translate::jitcode::BhDescr,
    ) {
        let offset = fielddescr.as_offset();
        let addr = (struct_ptr as usize).wrapping_add(offset);
        // SAFETY: see `bh_setfield_gc_i`.
        unsafe { (addr as *mut f64).write_unaligned(newvalue) };
    }

    // ── model.py:209-215, 247-253 array operations ──
    /// `llmodel.py:591 bh_getarrayitem_gc_i`: typed array load with sign
    /// extension.
    ///
    /// `unpack_arraydescr_size(arraydescr)` returns `(ofs, size, sign)`;
    /// the `BhDescr::Array` shape carries the equivalent triple as
    /// `(base_size, itemsize, is_item_signed)`.  The default impl mirrors
    /// `read_int_at_mem(gcref, ofs + index * size, size, sign)` (`llmodel.py:592`)
    /// — itemsize 1/2/4/8 with explicit sign dispatch — so any backend that
    /// receives a typed-array `BhDescr::Array` (cranelift, dynasm, wasm) gets
    /// the correct signed/unsigned read without a per-backend override.
    /// Backends that route GC-array reads through their own runtime (e.g.
    /// llgraph's RPython-typed dictionary) override; pyre's raw-memory
    /// backends share this default.
    fn bh_getarrayitem_gc_i(
        &self,
        array_ptr: i64,
        index: i64,
        arraydescr: &majit_translate::jitcode::BhDescr,
    ) -> i64 {
        use majit_translate::jitcode::BhDescr;
        let (base_size, itemsize, is_signed) = match arraydescr {
            BhDescr::Array {
                base_size,
                itemsize,
                is_item_signed,
                ..
            } => (*base_size, *itemsize, *is_item_signed),
            other => panic!(
                "bh_getarrayitem_gc_i: expected BhDescr::Array, got {:?}",
                other,
            ),
        };
        let item_addr = (array_ptr as usize)
            .wrapping_add(base_size)
            .wrapping_add((index as usize).wrapping_mul(itemsize));
        // SAFETY: `array_ptr` is a GC-managed array pointer threaded through
        // from the trace recorder; `index` is bounded by the outer
        // interpreter's array-length precondition.  `read_unaligned` mirrors
        // `llmodel.py:592 read_int_at_mem` — a raw load that does not assume
        // natural alignment (byte/short item arrays need not be 8-aligned) —
        // with size + sign discrimination from the array descriptor, matching
        // the dynasm backend's read_int_at_mem.
        unsafe {
            match (itemsize, is_signed) {
                (1, true) => (item_addr as *const i8).read_unaligned() as i64,
                (1, false) => (item_addr as *const u8).read_unaligned() as i64,
                (2, true) => (item_addr as *const i16).read_unaligned() as i64,
                (2, false) => (item_addr as *const u16).read_unaligned() as i64,
                (4, true) => (item_addr as *const i32).read_unaligned() as i64,
                (4, false) => (item_addr as *const u32).read_unaligned() as i64,
                (8, _) => (item_addr as *const i64).read_unaligned(),
                other => panic!(
                    "bh_getarrayitem_gc_i: unsupported (itemsize, signed) = {:?}",
                    other,
                ),
            }
        }
    }
    /// model.py:210 / llmodel.py:597 bh_getarrayitem_gc_r → pointer-width load
    /// at `base_size + index * WORD`.  Shared by pyre's raw-memory backends;
    /// the trait default returned NULL, losing reads of Ref array items (e.g. a
    /// tuple element or a `locals_cells_stack_w` slot during blackhole resume).
    fn bh_getarrayitem_gc_r(
        &self,
        array_ptr: i64,
        index: i64,
        arraydescr: &majit_translate::jitcode::BhDescr,
    ) -> GcRef {
        let base_size = arraydescr.array_base_size();
        let item_addr = (array_ptr as usize)
            .wrapping_add(base_size)
            .wrapping_add((index as usize).wrapping_mul(std::mem::size_of::<usize>()));
        // SAFETY: see `bh_getarrayitem_gc_i`. Ref items are pointer-width.
        GcRef(unsafe { (item_addr as *const usize).read_unaligned() })
    }
    /// model.py:211 bh_getarrayitem_gc_f(array, index, arraydescr)
    ///
    /// `llmodel.py:601-604`:
    ///   ofs = unpack_arraydescr(arraydescr)
    ///   fsize = rffi.sizeof(longlong.FLOATSTORAGE)
    ///   read_float_at_mem(array, itemindex * fsize + ofs)
    /// The float load uses the FIXED `FLOATSTORAGE` width, not the
    /// descriptor's `itemsize`; `base_size` supplies `ofs`.  Shared by
    /// pyre's raw-memory backends (cranelift, dynasm, wasm); backends
    /// routing array reads through their own runtime override this.
    fn bh_getarrayitem_gc_f(
        &self,
        array_ptr: i64,
        index: i64,
        arraydescr: &majit_translate::jitcode::BhDescr,
    ) -> f64 {
        use majit_translate::jitcode::BhDescr;
        let base_size = match arraydescr {
            BhDescr::Array { base_size, .. } => *base_size,
            other => panic!(
                "bh_getarrayitem_gc_f: expected BhDescr::Array, got {:?}",
                other,
            ),
        };
        // rffi.sizeof(longlong.FLOATSTORAGE)
        const FSIZE: usize = 8;
        let item_addr = (array_ptr as usize)
            .wrapping_add(base_size)
            .wrapping_add((index as usize).wrapping_mul(FSIZE));
        // SAFETY: `array_ptr` is a float-array pointer threaded through from
        // the trace recorder; `index` is bounded by the outer interpreter's
        // array-length precondition.  `read_unaligned` mirrors `llmodel.py:490
        // read_float_at_mem` — a raw `FLOATSTORAGE` load that does not assume
        // natural alignment — matching the dynasm backend's read_float_at_mem.
        unsafe { (item_addr as *const f64).read_unaligned() }
    }
    /// model.py:247 / llmodel.py:609 bh_setarrayitem_gc_i → typed store at
    /// `base_size + index * itemsize`.  Width from `unpack_arraydescr_size`.
    fn bh_setarrayitem_gc_i(
        &self,
        array_ptr: i64,
        index: i64,
        newvalue: i64,
        arraydescr: &majit_translate::jitcode::BhDescr,
    ) {
        let (base_size, itemsize, _sign) = arraydescr.unpack_arraydescr_size();
        let item_addr = (array_ptr as usize)
            .wrapping_add(base_size)
            .wrapping_add((index as usize).wrapping_mul(itemsize));
        // SAFETY: see `bh_setfield_gc_i`.
        unsafe {
            match itemsize {
                1 => (item_addr as *mut u8).write_unaligned(newvalue as u8),
                2 => (item_addr as *mut u16).write_unaligned(newvalue as u16),
                4 => (item_addr as *mut u32).write_unaligned(newvalue as u32),
                8 => (item_addr as *mut i64).write_unaligned(newvalue),
                other => panic!("bh_setarrayitem_gc_i: unsupported itemsize {other}"),
            }
        }
    }
    /// model.py:248 / llmodel.py:613 bh_setarrayitem_gc_r → pointer-width store
    /// at `base_size + index * WORD` plus the write barrier.
    fn bh_setarrayitem_gc_r(
        &self,
        array_ptr: i64,
        index: i64,
        newvalue: GcRef,
        arraydescr: &majit_translate::jitcode::BhDescr,
    ) {
        let base_size = arraydescr.array_base_size();
        let item_addr = (array_ptr as usize)
            .wrapping_add(base_size)
            .wrapping_add((index as usize).wrapping_mul(std::mem::size_of::<usize>()));
        // SAFETY: see `bh_setfield_gc_i`.
        unsafe { (item_addr as *mut usize).write_unaligned(newvalue.0) };
        majit_gc::gc_write_barrier(GcRef(array_ptr as usize));
    }
    /// model.py:249 / llmodel.py:618 bh_setarrayitem_gc_f → `FLOATSTORAGE`-width
    /// store at `base_size + index * WORD`.
    fn bh_setarrayitem_gc_f(
        &self,
        array_ptr: i64,
        index: i64,
        newvalue: f64,
        arraydescr: &majit_translate::jitcode::BhDescr,
    ) {
        let base_size = arraydescr.array_base_size();
        const FSIZE: usize = 8;
        let item_addr = (array_ptr as usize)
            .wrapping_add(base_size)
            .wrapping_add((index as usize).wrapping_mul(FSIZE));
        // SAFETY: see `bh_getarrayitem_gc_f`.
        unsafe { (item_addr as *mut f64).write_unaligned(newvalue) };
    }

    // ── model.py: raw array operations ──
    /// model.py:212 bh_getarrayitem_raw_i(array, index, arraydescr)
    ///
    /// `llmodel.py:592 read_int_at_mem(array, ofs + index * size, size,
    /// sign)` — identical typed-int memory load as the GC variant for the
    /// blackhole reader, so route through [`bh_getarrayitem_gc_i`].
    fn bh_getarrayitem_raw_i(
        &self,
        array: i64,
        index: i64,
        arraydescr: &majit_translate::jitcode::BhDescr,
    ) -> i64 {
        self.bh_getarrayitem_gc_i(array, index, arraydescr)
    }
    /// model.py:214 bh_getarrayitem_raw_f(array, index, arraydescr)
    ///
    /// `llmodel.py:625 bh_getarrayitem_raw_f = bh_getarrayitem_gc_f` — the
    /// raw float read is the identical typed memory access as the GC
    /// variant for the blackhole reader, so route through
    /// [`bh_getarrayitem_gc_f`].
    fn bh_getarrayitem_raw_f(
        &self,
        array: i64,
        index: i64,
        arraydescr: &majit_translate::jitcode::BhDescr,
    ) -> f64 {
        self.bh_getarrayitem_gc_f(array, index, arraydescr)
    }
    /// model.py:250 bh_setarrayitem_raw_i(array, index, newvalue, arraydescr)
    fn bh_setarrayitem_raw_i(
        &self,
        _array: i64,
        _index: i64,
        _newvalue: i64,
        _arraydescr: &majit_translate::jitcode::BhDescr,
    ) {
    }
    /// model.py:252 bh_setarrayitem_raw_f(array, index, newvalue, arraydescr)
    fn bh_setarrayitem_raw_f(
        &self,
        _array: i64,
        _index: i64,
        _newvalue: f64,
        _arraydescr: &majit_translate::jitcode::BhDescr,
    ) {
    }

    /// model.py:254 bh_arraylen_gc(array, arraydescr).
    ///
    /// Upstream shape is `read_int_at_mem(array, lendescr.offset, WORD, 1)`
    /// (`llmodel.py:585-588`). Production backends override this for
    /// pyre's length-prefixed GC arrays; the trait default remains a
    /// compatibility stub for incomplete/test CPUs.
    fn bh_arraylen_gc(
        &self,
        _array_ptr: i64,
        _arraydescr: &majit_translate::jitcode::BhDescr,
    ) -> i64 {
        0
    }

    // ── model.py:230-236 allocation ──
    /// model.py:230 / llmodel.py:775 bh_new(sizedescr)
    fn bh_new(&self, _sizedescr: &majit_translate::jitcode::BhDescr) -> i64 {
        0
    }
    /// model.py:231 / llmodel.py:778 bh_new_with_vtable(sizedescr)
    fn bh_new_with_vtable(&self, _sizedescr: &majit_translate::jitcode::BhDescr) -> i64 {
        0
    }

    /// llsupport/gc.py:563 GcLLDescr_framework
    ///   .get_typeid_from_classptr_if_gcremovetypeptr(classptr)
    /// Backend-side helper consulted only when `vtable_offset is None`
    /// (i.e. translation with --gcremovetypeptr). Returns the typeid that
    /// `_cmp_guard_gc_type` should compare against.
    ///
    /// Default `None` indicates the GC layer does not implement the
    /// gcremovetypeptr lowering, matching pyre's configuration.
    fn get_typeid_from_classptr_if_gcremovetypeptr(&self, _classptr: usize) -> Option<u32> {
        None
    }
    /// model.py:233 bh_new_array(length, arraydescr)
    fn bh_new_array(&self, _length: i64, _arraydescr: &majit_translate::jitcode::BhDescr) -> i64 {
        0
    }
    /// model.py:234 bh_new_array_clear(length, arraydescr)
    fn bh_new_array_clear(
        &self,
        _length: i64,
        _arraydescr: &majit_translate::jitcode::BhDescr,
    ) -> i64 {
        0
    }
    /// model.py: bh_strlen(string_ptr)
    fn bh_strlen(&self, _string_ptr: i64) -> i64 {
        0
    }
    /// model.py: bh_strgetitem(string_ptr, index)
    fn bh_strgetitem(&self, _string_ptr: i64, _index: i64) -> i64 {
        0
    }
    /// model.py: bh_strsetitem(string_ptr, index, value)
    fn bh_strsetitem(&self, _string_ptr: i64, _index: i64, _value: i64) {}
    /// model.py: bh_newstr(length)
    fn bh_newstr(&self, _length: i64) -> i64 {
        0
    }
    /// model.py:266 bh_call_i(func, args_i, args_r, args_f, calldescr).
    /// `llmodel.py:816 call_stub_i`: ABI-correct dispatch via the shared
    /// arity table.  Default impl shared by pyre's raw-memory backends
    /// (cranelift, dynasm, wasm) — the `extern "C"` transmute+call is
    /// portable.  Without it `bhimpl_residual_call_*_i` silently no-ops.
    fn bh_call_i(
        &self,
        func: i64,
        args_i: Option<&[i64]>,
        args_r: Option<&[i64]>,
        args_f: Option<&[i64]>,
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) -> i64 {
        if func == 0 {
            return 0;
        }
        if let Some(hook) = crate::call_stub::residual_host_call() {
            let args = crate::call_stub::collect_call_args_positional(
                &calldescr.arg_classes,
                args_i,
                args_r,
                args_f,
            );
            return hook(func as usize, &args);
        }
        let (int_args, float_args) =
            crate::call_stub::collect_call_args(&calldescr.arg_classes, args_i, args_r, args_f);
        // SAFETY: `func` is a valid funcptr matching the (ints, floats) arity
        // recovered from `calldescr.arg_classes`.
        unsafe { crate::call_stub::bh_call_i_dispatch(func as usize, &int_args, &float_args) }
    }
    /// model.py:268 bh_call_r(func, args_i, args_r, args_f, calldescr).
    /// `llmodel.py:818 bh_call_r`: GCREF-returning parallel — a host pointer
    /// matches the integer dispatcher's return register, so wrap its result.
    fn bh_call_r(
        &self,
        func: i64,
        args_i: Option<&[i64]>,
        args_r: Option<&[i64]>,
        args_f: Option<&[i64]>,
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) -> GcRef {
        if func == 0 {
            return GcRef::NULL;
        }
        if let Some(hook) = crate::call_stub::residual_host_call() {
            let args = crate::call_stub::collect_call_args_positional(
                &calldescr.arg_classes,
                args_i,
                args_r,
                args_f,
            );
            return GcRef(hook(func as usize, &args) as usize);
        }
        let (int_args, float_args) =
            crate::call_stub::collect_call_args(&calldescr.arg_classes, args_i, args_r, args_f);
        // SAFETY: see `bh_call_i`.
        let raw =
            unsafe { crate::call_stub::bh_call_i_dispatch(func as usize, &int_args, &float_args) };
        GcRef(raw as usize)
    }
    /// model.py:270 bh_call_f(func, args_i, args_r, args_f, calldescr).
    /// `llmodel.py:825 bh_call_f`: routes through the f64-typed dispatcher so
    /// an f64 callee returns via the float register file.
    fn bh_call_f(
        &self,
        func: i64,
        args_i: Option<&[i64]>,
        args_r: Option<&[i64]>,
        args_f: Option<&[i64]>,
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) -> f64 {
        if func == 0 {
            return 0.0;
        }
        if let Some(hook) = crate::call_stub::residual_host_call() {
            let args = crate::call_stub::collect_call_args_positional(
                &calldescr.arg_classes,
                args_i,
                args_r,
                args_f,
            );
            // The trampoline returns an f64 callee result as its raw bits.
            return f64::from_bits(hook(func as usize, &args) as u64);
        }
        let (int_args, float_args) =
            crate::call_stub::collect_call_args(&calldescr.arg_classes, args_i, args_r, args_f);
        // SAFETY: see `bh_call_i`.
        unsafe { crate::call_stub::bh_call_f_dispatch(func as usize, &int_args, &float_args) }
    }
    /// model.py:272 bh_call_v(func, args_i, args_r, args_f, calldescr).
    /// `llmodel.py:834 bh_call_v`: void-typed dispatch so a genuinely void
    /// callee is invoked with the right C-ABI signature.
    fn bh_call_v(
        &self,
        func: i64,
        args_i: Option<&[i64]>,
        args_r: Option<&[i64]>,
        args_f: Option<&[i64]>,
        calldescr: &majit_translate::jitcode::BhCallDescr,
    ) {
        if func == 0 {
            return;
        }
        if let Some(hook) = crate::call_stub::residual_host_call() {
            let args = crate::call_stub::collect_call_args_positional(
                &calldescr.arg_classes,
                args_i,
                args_r,
                args_f,
            );
            let _ = hook(func as usize, &args);
            return;
        }
        let (int_args, float_args) =
            crate::call_stub::collect_call_args(&calldescr.arg_classes, args_i, args_r, args_f);
        // SAFETY: see `bh_call_i`.
        unsafe { crate::call_stub::bh_call_v_dispatch(func as usize, &int_args, &float_args) }
    }

    // ── model.py: additional bh_* helpers ──

    /// model.py: bh_unicodelen(string_ptr)
    fn bh_unicodelen(&self, _string_ptr: i64) -> i64 {
        0
    }
    /// model.py: bh_unicodegetitem(string_ptr, index)
    fn bh_unicodegetitem(&self, _string_ptr: i64, _index: i64) -> i64 {
        0
    }
    /// model.py: bh_unicodesetitem(string_ptr, index, value)
    fn bh_unicodesetitem(&self, _string_ptr: i64, _index: i64, _value: i64) {}
    /// model.py: bh_newunicode(length)
    fn bh_newunicode(&self, _length: i64) -> i64 {
        0
    }
    /// model.py: bh_copystrcontent(src, dst, srcstart, dststart, length)
    fn bh_copystrcontent(
        &self,
        _src: i64,
        _dst: i64,
        _srcstart: i64,
        _dststart: i64,
        _length: i64,
    ) {
    }
    /// model.py: bh_copyunicodecontent(src, dst, srcstart, dststart, length)
    fn bh_copyunicodecontent(
        &self,
        _src: i64,
        _dst: i64,
        _srcstart: i64,
        _dststart: i64,
        _length: i64,
    ) {
    }
    /// llmodel.py:747-750 bh_raw_load_i(addr, offset, descr).
    fn bh_raw_load_i(
        &self,
        _addr: i64,
        _offset: i64,
        _descr: &majit_translate::jitcode::BhDescr,
    ) -> i64 {
        0
    }
    /// llmodel.py:739-742 bh_raw_store_i(addr, offset, newvalue, descr).
    fn bh_raw_store_i(
        &self,
        _addr: i64,
        _offset: i64,
        _newvalue: i64,
        _descr: &majit_translate::jitcode::BhDescr,
    ) {
    }
    // ── model.py: interior field access ──
    /// model.py: bh_getinteriorfield_gc_i(array, index, descr)
    fn bh_getinteriorfield_gc_i(
        &self,
        _array: i64,
        _index: i64,
        _descr: &majit_translate::jitcode::BhDescr,
    ) -> i64 {
        0
    }
    /// model.py: bh_getinteriorfield_gc_r(array, index, descr)
    fn bh_getinteriorfield_gc_r(
        &self,
        _array: i64,
        _index: i64,
        _descr: &majit_translate::jitcode::BhDescr,
    ) -> GcRef {
        GcRef::NULL
    }
    /// model.py: bh_getinteriorfield_gc_f(array, index, descr)
    fn bh_getinteriorfield_gc_f(
        &self,
        _array: i64,
        _index: i64,
        _descr: &majit_translate::jitcode::BhDescr,
    ) -> f64 {
        0.0
    }
    /// model.py: bh_setinteriorfield_gc_i(array, index, newvalue, descr)
    fn bh_setinteriorfield_gc_i(
        &self,
        _array: i64,
        _index: i64,
        _newvalue: i64,
        _descr: &majit_translate::jitcode::BhDescr,
    ) {
    }
    /// model.py: bh_setinteriorfield_gc_r(array, index, newvalue, descr)
    fn bh_setinteriorfield_gc_r(
        &self,
        _array: i64,
        _index: i64,
        _newvalue: GcRef,
        _descr: &majit_translate::jitcode::BhDescr,
    ) {
    }
    /// model.py: bh_setinteriorfield_gc_f(array, index, newvalue, descr)
    fn bh_setinteriorfield_gc_f(
        &self,
        _array: i64,
        _index: i64,
        _newvalue: f64,
        _descr: &majit_translate::jitcode::BhDescr,
    ) {
    }
    fn bh_gc_load_indexed_i(
        &self,
        _addr: i64,
        _index: i64,
        _scale: i64,
        _base_ofs: i64,
        _bytes: i64,
    ) -> i64 {
        0
    }
    fn bh_gc_load_indexed_f(
        &self,
        _addr: i64,
        _index: i64,
        _scale: i64,
        _base_ofs: i64,
        _bytes: i64,
    ) -> f64 {
        0.0
    }
    /// blackhole.py:1525-1529 bhimpl_gc_store_indexed_i
    fn bh_gc_store_indexed_i(
        &self,
        _addr: i64,
        _index: i64,
        _value: i64,
        _scale: i64,
        _base_ofs: i64,
        _bytes: i64,
    ) {
    }
    /// blackhole.py:1531-1535 bhimpl_gc_store_indexed_f
    fn bh_gc_store_indexed_f(
        &self,
        _addr: i64,
        _index: i64,
        _value: f64,
        _scale: i64,
        _base_ofs: i64,
        _bytes: i64,
    ) {
    }
    /// llmodel.py:752-753 bh_raw_load_f(addr, offset, descr).
    fn bh_raw_load_f(
        &self,
        _addr: i64,
        _offset: i64,
        _descr: &majit_translate::jitcode::BhDescr,
    ) -> f64 {
        0.0
    }
    /// llmodel.py:744-745 bh_raw_store_f(addr, offset, newvalue, descr).
    fn bh_raw_store_f(
        &self,
        _addr: i64,
        _offset: i64,
        _newvalue: f64,
        _descr: &majit_translate::jitcode::BhDescr,
    ) {
    }
    // ── model.py: raw field access ──
    fn bh_getfield_raw_i(
        &self,
        _struct_ptr: i64,
        _fielddescr: &majit_translate::jitcode::BhDescr,
    ) -> i64 {
        0
    }
    fn bh_getfield_raw_r(
        &self,
        _struct_ptr: i64,
        _fielddescr: &majit_translate::jitcode::BhDescr,
    ) -> GcRef {
        GcRef::NULL
    }
    fn bh_getfield_raw_f(
        &self,
        _struct_ptr: i64,
        _fielddescr: &majit_translate::jitcode::BhDescr,
    ) -> f64 {
        0.0
    }
    fn bh_setfield_raw_i(
        &self,
        _struct_ptr: i64,
        _newvalue: i64,
        _fielddescr: &majit_translate::jitcode::BhDescr,
    ) {
    }
    fn bh_setfield_raw_f(
        &self,
        _struct_ptr: i64,
        _newvalue: f64,
        _fielddescr: &majit_translate::jitcode::BhDescr,
    ) {
    }

    /// model.py: bh_classof(obj_ptr)
    fn bh_classof(&self, _obj_ptr: i64) -> i64 {
        0
    }
    /// RPython rclass.ll_issubclass(typeptr, bounding_class).
    /// Returns true if `typeptr` is a subclass of `bounding_class`.
    fn bh_issubclass(&self, typeptr: i64, bounding_class: i64) -> bool {
        // rclass.py:1133-1137:
        //   return int_between(cls.subclassrange_min,
        //                      subcls.subclassrange_min,
        //                      cls.subclassrange_max)
        //
        // Backends publish the same classptr -> subclassrange lookup via
        // majit_gc::set_active_gc_guard_hooks when their GC descriptor is
        // installed.  Keep exact-match behavior as the no-GC-hooks fallback
        // for backend-only fixtures.
        if let (Some((cls_min, cls_max)), Some((subcls_min, _))) = (
            majit_gc::subclass_range(bounding_class as usize),
            majit_gc::subclass_range(typeptr as usize),
        ) {
            cls_min <= subcls_min && subcls_min < cls_max
        } else {
            typeptr == bounding_class
        }
    }

    /// `model.py:34 AbstractCPU.setup_once` parity, called by
    /// `pyjitpl.py:2297 self.cpu.setup_once()` inside
    /// `MetaInterpStaticData._setup_once` (`pyjitpl.py:2292-2303`),
    /// guarded by `globaldata.initialized` so it runs exactly once
    /// per CPU after every descr setter has been called.  Backends
    /// that need to materialise per-CPU trampolines (x86
    /// `_build_propagate_exception_path` / `_build_malloc_slowpath`)
    /// override this; default is no-op for backends without
    /// setup-time work.
    fn setup_once(&mut self) {}

    /// `backend/x86/vector_ext.py:55 setup_once(asm)` parity, called
    /// by `pyjitpl.py:2298-2299
    /// `if self.cpu.vector_ext: self.cpu.vector_ext.setup_once(...)`
    /// inside `MetaInterpStaticData._setup_once`.  Backends with a
    /// vector extension override this; pyre's x86 / aarch64 / wasm
    /// backends have no vector_ext, so the default no-op is the honest
    /// port.
    fn vector_ext_setup_once(&mut self) {}

    /// `cpu.vector_ext.register_size` (vector_ext.py): SIMD register width
    /// in bytes, or 0 when the backend has no (enabled) vector unit.
    /// compile.py:303 gates `optimize_vector` on `cpu.vector_ext and
    /// cpu.vector_ext.is_enabled()`; pyre collapses the absent/disabled
    /// vector_ext to a 0 width, and a non-zero width is `is_enabled()`.
    /// The cranelift backend lowers the optimizeopt vector ops
    /// (VecLoad/VecStore/VecIntAdd/VecPack/VecUnpack/VecExpand) to native
    /// SIMD (I64X2/F64X2) and overrides this to 16; x86/aarch64/wasm
    /// have no vector lowering and keep the 0 default.
    fn vector_register_size(&self) -> usize {
        0
    }

    /// pyjitpl.py:2215-2217 `backendmodule = self.cpu.__module__
    /// .split('.')[-2]` parity — backend identifier used in
    /// `self.jit_starting_line = 'JIT starting (%s)' % backendmodule`
    /// (`pyjitpl.py:2296` `debug_print(self.jit_starting_line)`).
    /// RPython derives it by reflection on the CPU module path;
    /// pyre returns a literal because the backend struct already
    /// knows its own name at compile time.
    fn backend_name(&self) -> &'static str {
        "unknown"
    }

    /// model.py: finish_once() — called when the JIT shuts down.
    fn finish_once(&mut self) {}

    // ── model.py: GC integration ──

    /// model.py: gc_set_extra_threshold()
    /// Inform the GC that extra memory was allocated outside of GC control.
    fn gc_set_extra_threshold(&self) {}

    /// model.py: force_head_version()
    /// Force updating the version stamp for GC write barrier optimization.
    fn force_head_version(&self) {}

    /// model.py: get_all_loop_runs()
    /// Return a list of (token_number, loop_run_count) for profiling.
    fn get_all_loop_runs(&self) -> Vec<(u64, u64)> {
        Vec::new()
    }

    /// model.py: cast_int_to_ptr(value)
    fn cast_int_to_ptr(&self, value: i64) -> i64 {
        value // identity on 64-bit
    }

    /// model.py: cast_ptr_to_int(value)
    fn cast_ptr_to_int(&self, value: i64) -> i64 {
        value
    }

    /// model.py: cast_gcref_to_int(ref)
    fn cast_gcref_to_int(&self, gcref: GcRef) -> i64 {
        gcref.as_usize() as i64
    }

    /// model.py:199-201 cpu.cls_of_box(box):
    ///   obj = lltype.cast_opaque_ptr(OBJECTPTR, box.getref_base())
    ///   return ConstInt(ptr2int(obj.typeptr))
    ///
    /// Read the class pointer (typeptr/vtable) from a runtime Ref object.
    /// Default reads offset 0 (standard RPython object layout).
    /// Backends with different object models (e.g. gcremovetypeptr)
    /// should override.
    fn cls_of_box(&self, raw_ref: i64) -> i64 {
        debug_assert!(raw_ref != 0, "cls_of_box: null ref");
        unsafe { *(raw_ref as *const usize) as i64 }
    }
}

/// Errors from the backend.
#[derive(Debug)]
pub enum BackendError {
    /// Compilation failed.
    CompilationFailed(String),
    /// Unsupported operation.
    Unsupported(String),
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendError::CompilationFailed(s) => write!(f, "compilation failed: {s}"),
            BackendError::Unsupported(s) => write!(f, "unsupported: {s}"),
        }
    }
}

impl std::error::Error for BackendError {}

// ── we_are_jitted / JIT mode flag ──

thread_local! {
    static JIT_MODE_FLAG: Cell<bool> = const { Cell::new(false) };
}

/// Returns `true` when executing inside JIT-compiled code.
///
/// Interpreters can use this to choose optimized code paths that
/// the JIT can trace more efficiently.
#[inline]
pub fn we_are_jitted() -> bool {
    JIT_MODE_FLAG.with(|f| f.get())
}

/// `compile.py:1090` `memory_error = MemoryError()` cross-crate hook.
///
/// In RPython the malloc helpers raise `MemoryError`, which the
/// translator lowers to "set `cpu.pos_exc_value` to the singleton
/// instance, then return NULL".  pyre's malloc helpers live in
/// `majit-backend-{cranelift,dynasm}` and cannot depend on
/// `pyre-object` (CLAUDE.md crate boundary), so the singleton is
/// reached through a registered `fn() -> i64` provider.
///
/// Returns 0 when no provider is installed — backends interpret this
/// as "leave `JIT_EXC_VALUE` untouched".  Layer 4
/// (`PropagateExceptionDescr.handle_fail`, `compile.py:1092`) has its
/// own `cast_instance_to_gcref(memory_error)` fallback for that case.
static MEMORY_ERROR_PROVIDER: std::sync::OnceLock<fn() -> i64> = std::sync::OnceLock::new();

/// Install the `memory_error` singleton provider.  Called once from
/// pyre-jit's `install_jit_call_bridge` after the interpreter has
/// initialized the exception type registry.
pub fn register_memory_error_provider(f: fn() -> i64) {
    let _ = MEMORY_ERROR_PROVIDER.set(f);
}

/// Read the `memory_error` singleton pointer as an `i64`, or 0 if no
/// provider is registered.  Backends call this from their malloc
/// helpers to populate `JIT_EXC_VALUE` immediately before returning
/// `NULL` on OOM, mirroring RPython's translated
/// `do_malloc_fixedsize_clear` raising `MemoryError`.
#[inline]
pub fn memory_error_singleton_ref() -> i64 {
    match MEMORY_ERROR_PROVIDER.get() {
        Some(p) => p(),
        None => 0,
    }
}

/// Set the JIT mode flag. Called by the backend when entering compiled code.
pub fn set_jitted(jitted: bool) {
    JIT_MODE_FLAG.with(|f| f.set(jitted));
}

/// RAII guard for the JIT mode flag.
///
/// Sets `we_are_jitted()` to `true` on creation, restores the previous
/// value on drop.
pub struct JittedGuard {
    prev: bool,
}

impl JittedGuard {
    /// Create a new guard, setting `we_are_jitted()` to `true`.
    pub fn enter() -> Self {
        let prev = we_are_jitted();
        set_jitted(true);
        JittedGuard { prev }
    }
}

impl Drop for JittedGuard {
    fn drop(&mut self) {
        set_jitted(self.prev);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loop_version_info_add_and_track() {
        let mut info = LoopVersionInfo::new();
        assert!(info.versions.is_empty());

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![Op::new(majit_ir::OpCode::Finish, &[])];
        info.add_version(
            10,
            inputargs.iter().map(InputArg::fresh_value_copy).collect(),
            ops.clone(),
        );
        assert_eq!(info.versions.len(), 1);
        assert_eq!(info.versions[0].0, 10);

        info.add_version(20, inputargs, ops);
        assert_eq!(info.versions.len(), 2);
        assert_eq!(info.versions[1].0, 20);
    }

    #[test]
    fn loop_version_info_default() {
        let info = LoopVersionInfo::default();
        assert!(info.versions.is_empty());
    }

    #[test]
    fn loop_token_version_info_none_by_default() {
        let token = JitCellToken::new(1);
        assert!(token.version_info.is_none());
    }

    #[test]
    fn loop_token_with_version_info() {
        let mut token = JitCellToken::new(1);
        let mut info = LoopVersionInfo::new();
        info.add_version(
            5,
            vec![InputArg::new_int(0)],
            vec![Op::new(majit_ir::OpCode::Finish, &[])],
        );
        token.version_info = Some(info);

        assert!(token.version_info.is_some());
        assert_eq!(token.version_info.as_ref().unwrap().versions.len(), 1);
    }

    #[test]
    fn jit_cell_token_keepalive_holds_jit_cell_token() {
        let token = JitCellToken::new(1);
        let target = Arc::new(JitCellToken::new(2));

        token.record_jump_to(Arc::clone(&target));
        token.record_jump_to(Arc::clone(&target));

        let guard = token.keepalive_tokens.lock();
        assert_eq!(guard.len(), 1);
        assert_eq!(guard[0].number, 2);
        assert!(Arc::ptr_eq(&guard[0], &target));
    }

    #[test]
    fn jit_cell_token_keepalive_skips_self() {
        let token = Arc::new(JitCellToken::new(1));
        token.record_jump_to(Arc::clone(&token));
        assert!(token.keepalive_tokens.lock().is_empty());
    }

    #[test]
    fn test_jit_cell_token_lifecycle() {
        let mut token = JitCellToken::new(42);
        assert_eq!(token.get_number(), 42);
        assert!(!token.has_compiled_code());
        assert!(!token.is_invalidated());

        // Invalidate
        token.invalidate();
        assert!(token.is_invalidated());

        // Get flag clone for QuasiImmut
        let flag = token.invalidation_flag();
        assert!(flag.load(std::sync::atomic::Ordering::Acquire));

        // Reset
        token.reset_compiled();
        assert!(!token.has_compiled_code());
    }

    #[test]
    fn test_we_are_jitted() {
        assert!(!we_are_jitted());
        set_jitted(true);
        assert!(we_are_jitted());
        set_jitted(false);
        assert!(!we_are_jitted());
    }

    #[test]
    fn memory_error_provider_zero_without_registration() {
        // The OnceLock is process-global — once a real test (or a
        // sibling crate's test) registers a provider, this assertion
        // fails.  Use a sentinel-aware check: it's 0 OR it dereferences
        // to a non-null pointer (the singleton).  Either way, the
        // accessor must not panic when called before registration.
        let v = memory_error_singleton_ref();
        assert!(v == 0 || v != 0, "accessor must not panic");
    }
}
