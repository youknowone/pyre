/// Backend abstraction trait for JIT code generation.
///
/// Translated from rpython/jit/backend/model.py (AbstractCPU).
/// The Backend trait is the contract between the JIT frontend (tracing + optimization)
/// and the code generation backend (Cranelift, etc.).
use std::cell::Cell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use majit_ir::{Descr, FailDescr, GcRef, InputArg, Op, Type, Value};

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
    /// compile.py:741-745: ResumeGuardDescr.status at guard failure time.
    pub status: u64,
    /// compile.py:780: current_object_addr_as_int(self) — descriptor pointer.
    pub descr_addr: usize,
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
    /// Slot is a constant value embedded in the layout.
    Constant(i64),
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
        descr_index: u32,
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
        descr_index: u32,
        fields: Vec<(u32, ExitValueSourceLayout)>,
        target_slot: Option<usize>,
        fielddescrs: Vec<majit_ir::FieldDescrInfo>,
        descr_size: usize,
    },
    Array {
        descr_index: u32,
        /// resume.py:653: allocate_array(length, arraydescr, self.clear)
        clear: bool,
        /// resume.py:656: arraydescr element kind (0=ref, 1=int, 2=float)
        kind: u8,
        items: Vec<ExitValueSourceLayout>,
    },
    ArrayStruct {
        descr_index: u32,
        /// Per-field type within each element: 0=ref, 1=int, 2=float.
        field_types: Vec<u8>,
        /// llmodel.py:648: arraydescr.itemsize.
        item_size: usize,
        /// llmodel.py:649: per-field fielddescr.offset.
        field_offsets: Vec<usize>,
        /// llmodel.py:649: per-field fielddescr.field_size.
        field_sizes: Vec<usize>,
        element_fields: Vec<Vec<(u32, ExitValueSourceLayout)>>,
    },
    /// resume.py:717 VRawSliceInfo — base_buffer + offset.
    RawSlice {
        offset: usize,
        base: ExitValueSourceLayout,
    },
    RawBuffer {
        size: usize,
        entries: Vec<(usize, usize, ExitValueSourceLayout)>,
    },
}

impl ExitVirtualLayout {
    pub fn shifted_virtuals(&self, virtual_offset: usize) -> Self {
        match self {
            Self::Object {
                descr,
                type_id,
                descr_index,
                known_class,
                fields,
                target_slot,
                fielddescrs,
                descr_size,
            } => Self::Object {
                descr: descr.clone(),
                type_id: *type_id,
                descr_index: *descr_index,
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
                descr_index,
                fields,
                target_slot,
                fielddescrs,
                descr_size,
            } => Self::Struct {
                typedescr: typedescr.clone(),
                type_id: *type_id,
                descr_index: *descr_index,
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
                descr_index,
                clear,
                kind,
                items,
            } => Self::Array {
                descr_index: *descr_index,
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
                descr_index,
                field_types,
                item_size,
                field_offsets,
                field_sizes,
                element_fields,
            } => Self::ArrayStruct {
                descr_index: *descr_index,
                field_types: field_types.clone(),
                item_size: *item_size,
                field_offsets: field_offsets.clone(),
                field_sizes: field_sizes.clone(),
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
            Self::RawBuffer { size, entries } => Self::RawBuffer {
                size: *size,
                entries: entries
                    .iter()
                    .map(|(offset, entry_size, source)| {
                        (
                            *offset,
                            *entry_size,
                            source.shifted_virtuals(virtual_offset),
                        )
                    })
                    .collect(),
            },
        }
    }
}

// PartialEq/Eq: compare by data fields, skip descr/typedescr (Arc<dyn Descr>).
impl PartialEq for ExitVirtualLayout {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Object {
                    type_id: a1,
                    descr_index: a2,
                    known_class: a7,
                    fields: a3,
                    target_slot: a4,
                    fielddescrs: a5,
                    descr_size: a6,
                    ..
                },
                Self::Object {
                    type_id: b1,
                    descr_index: b2,
                    known_class: b7,
                    fields: b3,
                    target_slot: b4,
                    fielddescrs: b5,
                    descr_size: b6,
                    ..
                },
            ) => a1 == b1 && a2 == b2 && a3 == b3 && a4 == b4 && a5 == b5 && a6 == b6 && a7 == b7,
            (
                Self::Struct {
                    type_id: a1,
                    descr_index: a2,
                    fields: a3,
                    target_slot: a4,
                    fielddescrs: a5,
                    descr_size: a6,
                    ..
                },
                Self::Struct {
                    type_id: b1,
                    descr_index: b2,
                    fields: b3,
                    target_slot: b4,
                    fielddescrs: b5,
                    descr_size: b6,
                    ..
                },
            ) => a1 == b1 && a2 == b2 && a3 == b3 && a4 == b4 && a5 == b5 && a6 == b6,
            (
                Self::Array {
                    descr_index: a1,
                    clear: a2,
                    kind: a3,
                    items: a4,
                },
                Self::Array {
                    descr_index: b1,
                    clear: b2,
                    kind: b3,
                    items: b4,
                },
            ) => a1 == b1 && a2 == b2 && a3 == b3 && a4 == b4,
            (
                Self::ArrayStruct {
                    descr_index: a1,
                    field_types: a2,
                    item_size: a3,
                    field_offsets: a4,
                    field_sizes: a5,
                    element_fields: a6,
                },
                Self::ArrayStruct {
                    descr_index: b1,
                    field_types: b2,
                    item_size: b3,
                    field_offsets: b4,
                    field_sizes: b5,
                    element_fields: b6,
                },
            ) => a1 == b1 && a2 == b2 && a3 == b3 && a4 == b4 && a5 == b5 && a6 == b6,
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
                    entries: a2,
                },
                Self::RawBuffer {
                    size: b1,
                    entries: b2,
                },
            ) => a1 == b1 && a2 == b2,
            _ => false,
        }
    }
}
impl Eq for ExitVirtualLayout {}

/// Backend-neutral deferred heap write recovered from an exit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExitPendingFieldLayout {
    pub descr_index: u32,
    pub item_index: Option<usize>,
    pub is_array_item: bool,
    pub target: ExitValueSourceLayout,
    pub value: ExitValueSourceLayout,
    /// Byte offset from start of struct (from FieldDescr).
    pub field_offset: usize,
    /// Size of the field in bytes.
    pub field_size: usize,
    /// Type of the value being stored.
    pub field_type: majit_ir::Type,
}

impl ExitPendingFieldLayout {
    pub fn shifted_virtuals(&self, virtual_offset: usize) -> Self {
        Self {
            descr_index: self.descr_index,
            item_index: self.item_index,
            is_array_item: self.is_array_item,
            target: self.target.shifted_virtuals(virtual_offset),
            value: self.value.shifted_virtuals(virtual_offset),
            field_offset: self.field_offset,
            field_size: self.field_size,
            field_type: self.field_type,
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
            frames,
            virtual_layouts,
            pending_field_layouts,
        }
    }
}

/// Static layout metadata for a backend fail descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
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
    /// Exit slot indices that hold rooted GC references.
    pub gc_ref_slots: Vec<usize>,
    /// Exit slot indices that carry opaque FORCE_TOKEN handles.
    pub force_token_slots: Vec<usize>,
    /// Optional backend-origin recovery layout for this exit.
    pub recovery_layout: Option<ExitRecoveryLayout>,
    /// Complete frame stack from innermost (this guard's frame) to outermost.
    /// Present when multi-frame reconstruction is supported.
    pub frame_stack: Option<Vec<ExitFrameLayout>>,
}

/// Static layout metadata for a terminal exit within a compiled trace.
///
/// Unlike [`FailDescrLayout`], terminal exits are keyed by the trace op index
/// rather than a backend fail descriptor, because `JUMP` exits do not
/// necessarily correspond to a deadframe-producing guard site.
#[derive(Debug, Clone, PartialEq, Eq)]
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
    /// Exit slot indices that hold rooted GC references.
    pub gc_ref_slots: Vec<usize>,
    /// Exit slot indices that carry opaque FORCE_TOKEN handles.
    pub force_token_slots: Vec<usize>,
    /// Optional backend-origin recovery layout for this terminal exit.
    pub recovery_layout: Option<ExitRecoveryLayout>,
}

/// Result of compiling a loop or bridge.
#[derive(Debug)]
pub struct AsmInfo {
    /// Start address of the generated code.
    pub code_addr: usize,
    /// Size of the generated code in bytes.
    pub code_size: usize,
}

/// Marker descriptor for loop version guards.
///
/// When a guard uses this descriptor, the alternative loop version
/// is compiled immediately after the main loop (not lazily on failure).
/// This enables speculative optimization: vectorized fast path with
/// scalar fallback, type-specialized path with generic fallback, etc.
///
/// Mirrors `CompileLoopVersionDescr` from rpython/jit/metainterp/compile.py.
#[derive(Debug)]
pub struct LoopVersionDescr {
    /// Index identifying this version guard.
    pub version_index: u32,
    /// Fail argument types for the guard.
    pub fail_arg_types: Vec<Type>,
}

impl Descr for LoopVersionDescr {
    fn index(&self) -> u32 {
        self.version_index
    }
    fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
        Some(self)
    }
    fn is_loop_version(&self) -> bool {
        true
    }
}

impl FailDescr for LoopVersionDescr {
    fn fail_index(&self) -> u32 {
        self.version_index
    }
    fn fail_arg_types(&self) -> &[Type] {
        &self.fail_arg_types
    }
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

/// Token identifying a compiled loop. Bridges are attached to this.
/// RPython history.py JitCellToken parity — green_key carried on token
/// so the backend can identify the parent loop for bridge compilation.
pub struct JitCellToken {
    /// Unique number for this token.
    pub number: u64,
    /// Green key hash identifying the loop entry point.
    /// Set by MetaInterp before compile_loop. Used by the backend's
    /// bridge threshold callback to find the compiled loop metadata.
    pub green_key: u64,
    /// Types of the input arguments.
    pub inputarg_types: Vec<Type>,
    /// virtualizable.py:86 read_boxes: number of scalar inputargs
    /// (frame + static fields). First local is at this index.
    pub num_scalar_inputargs: usize,
    /// Backend-specific compiled data.
    pub compiled: Option<Box<dyn std::any::Any + Send>>,
    /// Flag indicating whether the compiled code has been invalidated.
    /// When set to `true`, any `GUARD_NOT_INVALIDATED` in the compiled
    /// code will fail, causing execution to bail out to the interpreter.
    pub invalidated: Arc<AtomicBool>,
    /// Alternative loop versions to compile immediately after the main loop.
    pub version_info: Option<LoopVersionInfo>,
    /// history.py:449: _keepalive_jitcell_tokens — set of other tokens
    /// that this loop can jump to (via CALL_ASSEMBLER or JUMP).
    /// Prevents the target from being evicted while this loop is alive.
    pub keepalive_tokens: Vec<u64>,
    /// llmodel.py:252 `compiled_loop_token.asmmemmgr_blocks` parity.
    ///
    /// Backend-owned memory blocks (bridge ExecutableBuffers, etc.)
    /// that must live as long as this token. Freed when the token is
    /// dropped — mirrors RPython's `free_loop_and_bridges` which
    /// iterates `asmmemmgr_blocks` and returns each block to the
    /// `AsmMemoryManager` free-list.
    ///
    /// ## Why `parking_lot::Mutex`
    ///
    /// `compile_bridge` receives `&JitCellToken` (shared ref) because
    /// the metainterp holds the token in a `HashMap` and may inspect
    /// other fields concurrently.  Bridge compilation must append to
    /// this vec through a shared ref → interior mutability is required.
    ///
    /// `RefCell` is insufficient: under free-threading (no-GIL Python),
    /// multiple OS threads may hold references to the same
    /// `JitCellToken` — one thread executing compiled code while
    /// another compiles a bridge for a guard in the same loop.
    /// `RefCell`'s single-threaded borrow checking would panic.
    ///
    /// `std::sync::Mutex` works but has two drawbacks:
    ///   1. Poisoning on panic makes recovery cumbersome.
    ///   2. On macOS/Linux, `pthread_mutex` is heavier than needed
    ///      for an uncontended fast-path (bridge compilation is rare).
    ///
    /// `parking_lot::Mutex` avoids both: no poisoning semantics, and
    /// the uncontended lock/unlock is a single atomic CAS — ideal for
    /// the typical case where only one thread touches this field at a
    /// time, with the Mutex serving as a correctness guard for the
    /// rare concurrent bridge-compilation scenario.
    pub asmmemmgr_blocks: parking_lot::Mutex<Vec<Box<dyn std::any::Any + Send>>>,
}

impl JitCellToken {
    pub fn new(number: u64) -> Self {
        JitCellToken {
            number,
            green_key: 0,
            inputarg_types: Vec::new(),
            num_scalar_inputargs: 0,
            compiled: None,
            invalidated: Arc::new(AtomicBool::new(false)),
            version_info: None,
            keepalive_tokens: Vec::new(),
            asmmemmgr_blocks: parking_lot::Mutex::new(Vec::new()),
        }
    }

    /// history.py:451-453: record_jump_to — record that this loop can
    /// jump to another JitCellToken (via CALL_ASSEMBLER or JUMP).
    /// Prevents the MemoryManager from evicting the target.
    pub fn record_jump_to(&mut self, target_number: u64) {
        if !self.keepalive_tokens.contains(&target_number) {
            self.keepalive_tokens.push(target_number);
        }
    }

    /// Mark this loop as invalidated. Any subsequent execution of
    /// GUARD_NOT_INVALIDATED in the compiled code will fail.
    pub fn invalidate(&self) {
        self.invalidated.store(true, Ordering::Release);
    }

    /// Check whether this loop has been invalidated.
    pub fn is_invalidated(&self) -> bool {
        self.invalidated.load(Ordering::Acquire)
    }

    /// model.py: has_compiled_code()
    /// Whether this token has compiled code attached.
    pub fn has_compiled_code(&self) -> bool {
        self.compiled.is_some()
    }

    /// model.py: get_number()
    pub fn get_number(&self) -> u64 {
        self.number
    }

    /// model.py: reset_compiled()
    /// Remove the compiled code (e.g., after invalidation).
    pub fn reset_compiled(&mut self) {
        self.compiled = None;
    }

    /// Get a clone of the invalidated flag (for registering with QuasiImmut).
    pub fn invalidation_flag(&self) -> Arc<AtomicBool> {
        self.invalidated.clone()
    }
}

impl std::fmt::Debug for JitCellToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JitCellToken")
            .field("number", &self.number)
            .finish()
    }
}

/// A "dead frame" — the state after JIT execution finishes or hits a guard.
///
/// The backend stores register/stack values here so the frontend can read them.
pub struct DeadFrame {
    /// Backend-specific frame data.
    pub data: Box<dyn std::any::Any + Send>,
}

/// The backend trait — implemented by Cranelift (or other code generators).
///
/// Mirrors rpython/jit/backend/model.py AbstractCPU.
pub trait Backend: Send {
    /// Compile a loop trace into native code.
    fn compile_loop(
        &mut self,
        inputargs: &[InputArg],
        ops: &[Op],
        token: &mut JitCellToken,
    ) -> Result<AsmInfo, BackendError>;

    /// Register a placeholder for a pending token (RPython compile_tmp_callback).
    /// The placeholder has null code_ptr; call_assembler_fast_path detects this
    /// and falls back to force_fn. Replaced by the real target on compile_loop.
    /// Register a placeholder for a pending token (RPython compile_tmp_callback).
    /// `num_scalar_inputargs` = virtualizable.py:86 NUM_SCALAR_INPUTARGS.
    fn register_pending_target(
        &mut self,
        _token_number: u64,
        _input_types: Vec<Type>,
        _num_inputs: usize,
        _num_scalar_inputargs: usize,
    ) {
    }

    /// Compile a bridge (side exit path) and attach it to the loop.
    ///
    /// `previous_tokens` contains old tokens from retraces. Because
    /// Cranelift can't patch existing machine code (unlike RPython's x86
    /// backend), the running machine code may reference fail_descrs from
    /// an older token. The bridge must be attached to ALL matching
    /// fail_descrs across current + previous tokens.
    fn compile_bridge(
        &mut self,
        fail_descr: &dyn FailDescr,
        inputargs: &[InputArg],
        ops: &[Op],
        original_token: &JitCellToken,
        previous_tokens: &[JitCellToken],
    ) -> Result<AsmInfo, BackendError>;

    /// Compile all registered loop versions as bridges.
    ///
    /// Called after `compile_loop()` succeeds. For each version registered
    /// in the token's `version_info`, finds the corresponding version guard
    /// descriptor and compiles the alternative trace as a bridge.
    fn compile_versions(&mut self, token: &JitCellToken) -> Result<Vec<AsmInfo>, BackendError> {
        let versions = match &token.version_info {
            Some(info) => info
                .versions
                .iter()
                .map(|(guard_idx, inputargs, ops)| (*guard_idx, inputargs.clone(), ops.clone()))
                .collect::<Vec<_>>(),
            None => return Ok(Vec::new()),
        };

        let mut results = Vec::new();
        for (guard_idx, inputargs, ops) in &versions {
            let descr = LoopVersionDescr {
                version_index: *guard_idx,
                fail_arg_types: inputargs.iter().map(|ia| ia.tp).collect(),
            };
            let asm = self.compile_bridge(&descr, inputargs, ops, token, &[])?;
            results.push(asm);
        }
        Ok(results)
    }

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

    /// compile.py:741-745: look up (status, descr_addr) for a guard.
    /// Uses (trace_id, fail_index) to find the exact descriptor, including
    /// bridge guards and previous tokens — matching start_guard_compiling's
    /// find_fail_descr_in_fail_descrs pattern.
    fn get_guard_status(
        &self,
        _token: &JitCellToken,
        _trace_id: u64,
        _fail_index: u32,
    ) -> (u64, usize) {
        (0, 0)
    }

    /// compile.py:826-830 store_hash: assign jitcounter hashes to guards.
    /// Called after compile_loop/compile_bridge with hashes from
    /// jitcounter.fetch_next_hash(). Skips guards that already have
    /// status set by make_a_counter_per_value (GUARD_VALUE).
    fn store_guard_hashes(&self, _token: &JitCellToken, _hashes: &[u64]) {}

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

    /// compile.py:741: self.status — read status from the failed descriptor
    /// directly by its raw address (no re-lookup). RPython calls self.status
    /// on the same descriptor object; descr_addr IS that object's identity.
    ///
    /// # Safety
    /// descr_addr must be a valid pointer returned from a guard failure in
    /// the same compilation session. The descriptor must still be alive
    /// (held by compiled_loops or previous_tokens).
    fn read_descr_status(&self, _descr_addr: usize) -> u64 {
        0
    }

    /// compile.py:786-788: `self.start_compiling()`.
    ///
    /// RPython calls `self.status |= ST_BUSY_FLAG` directly on the
    /// descriptor. In Rust, the descriptor is behind a trait object and
    /// Arc, so we pass `descr_addr` (the Arc's raw pointer from the guard
    /// failure result) and let the backend cast it back. This is the Rust
    /// equivalent of RPython's `self` — both identify the exact descriptor.
    fn start_compiling_descr(&self, _descr_addr: usize) {}

    /// compile.py:790-795: `self.done_compiling()`.
    /// Same pattern as start_compiling_descr — see above.
    fn done_compiling_descr(&self, _descr_addr: usize) {}

    /// Execute compiled code starting at the given token.
    fn execute_token(&self, token: &JitCellToken, args: &[Value]) -> DeadFrame;

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
        let descr = self.get_latest_descr(&frame);
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
            status: descr.get_status(),
            descr_addr: descr as *const dyn majit_ir::descr::FailDescr as *const () as usize,
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

    /// Query complete frame-stack layouts for all guards in a compiled loop.
    ///
    /// Returns `(fail_index, frame_stack)` pairs for each guard that has
    /// recovery layout metadata. Backends that populate recovery layouts
    /// at compile time can override this to expose the frame stacks.
    fn compiled_guard_frame_stacks(
        &self,
        _token: &JitCellToken,
    ) -> Option<Vec<(u32, Vec<ExitFrameLayout>)>> {
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
            gc_ref_slots: descr
                .fail_arg_types()
                .iter()
                .enumerate()
                .filter_map(|(slot, _)| descr.is_gc_ref_slot(slot).then_some(slot))
                .collect(),
            force_token_slots: descr.force_token_slots().to_vec(),
            recovery_layout: None,
            frame_stack: None,
        })
    }

    /// Patch backend-owned recovery metadata for a specific compiled exit.
    fn update_fail_descr_recovery_layout(
        &mut self,
        _token: &JitCellToken,
        _trace_id: u64,
        _fail_index: u32,
        _recovery_layout: ExitRecoveryLayout,
    ) -> bool {
        false
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

    /// Read the FailDescr from the last guard failure.
    fn get_latest_descr<'a>(&'a self, frame: &'a DeadFrame) -> &'a dyn FailDescr;

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

    /// model.py: bh_getfield_gc_i(struct_ptr, descr)
    fn bh_getfield_gc_i(&self, _struct_ptr: i64, _offset: usize) -> i64 {
        0
    }
    /// model.py: bh_getfield_gc_r(struct_ptr, descr)
    fn bh_getfield_gc_r(&self, _struct_ptr: i64, _offset: usize) -> GcRef {
        GcRef::NULL
    }
    /// model.py: bh_getfield_gc_f(struct_ptr, descr)
    fn bh_getfield_gc_f(&self, _struct_ptr: i64, _offset: usize) -> f64 {
        0.0
    }
    /// model.py: bh_setfield_gc_i(struct_ptr, value, descr)
    fn bh_setfield_gc_i(&self, _struct_ptr: i64, _offset: usize, _value: i64) {}
    /// model.py: bh_setfield_gc_r(struct_ptr, value, descr)
    fn bh_setfield_gc_r(&self, _struct_ptr: i64, _offset: usize, _value: GcRef) {}
    /// model.py: bh_setfield_gc_f(struct_ptr, value, descr)
    fn bh_setfield_gc_f(&self, _struct_ptr: i64, _offset: usize, _value: f64) {}
    /// model.py: bh_getarrayitem_gc_i(array_ptr, index, descr)
    fn bh_getarrayitem_gc_i(&self, _array_ptr: i64, _index: i64, _item_size: usize) -> i64 {
        0
    }
    /// model.py: bh_getarrayitem_gc_r(array_ptr, index, descr)
    fn bh_getarrayitem_gc_r(&self, _array_ptr: i64, _index: i64, _item_size: usize) -> GcRef {
        GcRef::NULL
    }
    /// model.py: bh_setarrayitem_gc_i(array_ptr, index, value, descr)
    fn bh_setarrayitem_gc_i(&self, _array_ptr: i64, _index: i64, _item_size: usize, _value: i64) {}
    /// model.py: bh_setarrayitem_gc_r(array_ptr, index, value, descr)
    fn bh_setarrayitem_gc_r(&self, _array_ptr: i64, _index: i64, _item_size: usize, _value: GcRef) {
    }
    /// model.py: bh_getarrayitem_gc_f(array_ptr, index, descr)
    fn bh_getarrayitem_gc_f(&self, _array_ptr: i64, _index: i64, _item_size: usize) -> f64 {
        0.0
    }
    /// model.py: bh_setarrayitem_gc_f(array_ptr, index, value, descr)
    fn bh_setarrayitem_gc_f(&self, _array_ptr: i64, _index: i64, _item_size: usize, _value: f64) {}
    /// model.py: bh_getarrayitem_raw_i(array, index, arraydescr)
    fn bh_getarrayitem_raw_i(&self, _array: i64, _index: i64, _item_size: usize) -> i64 {
        0
    }
    /// model.py: bh_getarrayitem_raw_f(array, index, arraydescr)
    fn bh_getarrayitem_raw_f(&self, _array: i64, _index: i64, _item_size: usize) -> f64 {
        0.0
    }
    /// model.py: bh_setarrayitem_raw_i(array, index, newvalue, arraydescr)
    fn bh_setarrayitem_raw_i(&self, _array: i64, _index: i64, _item_size: usize, _value: i64) {}
    /// model.py: bh_setarrayitem_raw_f(array, index, newvalue, arraydescr)
    fn bh_setarrayitem_raw_f(&self, _array: i64, _index: i64, _item_size: usize, _value: f64) {}
    /// model.py: bh_arraylen_gc(array_ptr, descr)
    fn bh_arraylen_gc(&self, _array_ptr: i64, _len_offset: usize) -> i64 {
        0
    }
    /// llmodel.py:775 bh_new(sizedescr).
    fn bh_new(&self, _sizedescr: &dyn majit_ir::SizeDescr) -> i64 {
        0
    }
    /// bh_new from raw size — used when only the struct size is available
    /// (no SizeDescr object). Newlist/newlist_clear/newlist_hint use this.
    fn bh_new_with_size(&self, _size: usize) -> i64 {
        0
    }
    /// llmodel.py:778 bh_new_with_vtable(sizedescr).
    fn bh_new_with_vtable(&self, _sizedescr: &dyn majit_ir::SizeDescr) -> i64 {
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
    /// model.py: bh_new_array(length, descr)
    fn bh_new_array(&self, _length: i64, _item_size: usize, _type_id: u32) -> i64 {
        0
    }
    /// model.py: bh_new_array_clear(length, descr)
    fn bh_new_array_clear(&self, _length: i64, _item_size: usize, _type_id: u32) -> i64 {
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
    /// model.py: bh_call_i(func_ptr, args, calldescr)
    fn bh_call_i(&self, _func_ptr: i64, _args: &[i64]) -> i64 {
        0
    }
    /// model.py: bh_call_r(func_ptr, args, calldescr)
    fn bh_call_r(&self, _func_ptr: i64, _args: &[i64]) -> GcRef {
        GcRef::NULL
    }
    /// model.py: bh_call_f(func_ptr, args, calldescr)
    fn bh_call_f(&self, _func_ptr: i64, _args: &[i64]) -> f64 {
        0.0
    }
    /// model.py: bh_call_v(func_ptr, args, calldescr)
    fn bh_call_v(&self, _func_ptr: i64, _args: &[i64]) {}

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
    /// model.py: bh_raw_load_i(ptr, offset, descr)
    fn bh_raw_load_i(&self, _ptr: i64, _offset: i64) -> i64 {
        0
    }
    /// model.py: bh_raw_store_i(ptr, offset, value, descr)
    fn bh_raw_store_i(&self, _ptr: i64, _offset: i64, _value: i64) {}
    /// model.py: bh_getinteriorfield_gc_i(array, index, descr)
    fn bh_getinteriorfield_gc_i(&self, _array: i64, _index: i64, _offset: usize) -> i64 {
        0
    }
    /// model.py: bh_setinteriorfield_gc_i(array, index, value, descr)
    fn bh_setinteriorfield_gc_i(&self, _array: i64, _index: i64, _offset: usize, _value: i64) {}
    fn bh_getinteriorfield_gc_r(&self, _array: i64, _index: i64, _offset: usize) -> GcRef {
        GcRef::NULL
    }
    fn bh_getinteriorfield_gc_f(&self, _array: i64, _index: i64, _offset: usize) -> f64 {
        0.0
    }
    fn bh_setinteriorfield_gc_r(&self, _array: i64, _index: i64, _offset: usize, _value: GcRef) {}
    fn bh_setinteriorfield_gc_f(&self, _array: i64, _index: i64, _offset: usize, _value: f64) {}
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
    fn bh_raw_load_f(&self, _ptr: i64, _offset: i64) -> f64 {
        0.0
    }
    fn bh_raw_store_f(&self, _ptr: i64, _offset: i64, _value: f64) {}
    // ── model.py: raw field access ──
    fn bh_getfield_raw_i(&self, _struct_ptr: i64, _offset: usize) -> i64 {
        0
    }
    fn bh_getfield_raw_r(&self, _struct_ptr: i64, _offset: usize) -> GcRef {
        GcRef::NULL
    }
    fn bh_getfield_raw_f(&self, _struct_ptr: i64, _offset: usize) -> f64 {
        0.0
    }
    fn bh_setfield_raw_i(&self, _struct_ptr: i64, _offset: usize, _value: i64) {}
    fn bh_setfield_raw_f(&self, _struct_ptr: i64, _offset: usize, _value: f64) {}

    /// model.py: bh_classof(obj_ptr)
    fn bh_classof(&self, _obj_ptr: i64) -> i64 {
        0
    }
    /// RPython rclass.ll_issubclass(typeptr, bounding_class).
    /// Returns true if `typeptr` is a subclass of `bounding_class`.
    fn bh_issubclass(&self, _typeptr: i64, _bounding_class: i64) -> bool {
        _typeptr == _bounding_class // default: exact match only
    }

    /// model.py: setup_once() — called once when the backend is first used.
    fn setup_once(&mut self) {}
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
    use majit_ir::Type;

    #[test]
    fn loop_version_descr_is_loop_version() {
        let descr = LoopVersionDescr {
            version_index: 42,
            fail_arg_types: vec![Type::Int, Type::Int],
        };
        assert!(descr.is_loop_version());
        assert_eq!(descr.index(), 42);
        assert_eq!(descr.fail_index(), 42);
        assert_eq!(descr.fail_arg_types(), &[Type::Int, Type::Int]);
        assert!(!descr.is_finish());
    }

    #[test]
    fn loop_version_descr_as_fail_descr() {
        let descr = LoopVersionDescr {
            version_index: 7,
            fail_arg_types: vec![Type::Int, Type::Ref, Type::Float],
        };
        let fail = descr.as_fail_descr().unwrap();
        assert_eq!(fail.fail_index(), 7);
        assert_eq!(fail.fail_arg_types(), &[Type::Int, Type::Ref, Type::Float]);
    }

    #[test]
    fn regular_descr_is_not_loop_version() {
        #[derive(Debug)]
        struct PlainDescr;
        impl Descr for PlainDescr {}

        let d = PlainDescr;
        assert!(!d.is_loop_version());
    }

    #[test]
    fn loop_version_info_add_and_track() {
        let mut info = LoopVersionInfo::new();
        assert!(info.versions.is_empty());

        let inputargs = vec![InputArg::new_int(0), InputArg::new_int(1)];
        let ops = vec![Op::new(majit_ir::OpCode::Finish, &[])];
        info.add_version(10, inputargs.clone(), ops.clone());
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
}
