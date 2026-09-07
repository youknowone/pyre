/// Trace recorder — records IR operations during interpreter execution.
///
/// The recorder is the bridge between the interpreter and the JIT.
/// When tracing is active, every interpreter operation is fed to the
/// recorder, which builds a linear sequence of IR operations (a trace).
///
/// Upstream counterpart: `rpython/jit/metainterp/opencoder.py class Trace`
/// (raw op storage, `record_op`, `cut_point`, `cut_at`, `set_inputargs`).
/// The higher-level `History` wrapper (history.py:714) that creates
/// FrontendOps and provides `record_nospec` / `record_same_as` lives in
/// `history.rs` as `impl TraceCtx`.  Pyre callers reach the recorder via
/// `MetaInterp.history.record*` mirroring
/// `pyjitpl.py:2455+ self.history.record2(...)`.
use crate::opencoder::{Box as OcBox, TraceRecordBuffer};
use majit_ir::operand::Operand;
use majit_ir::{DescrRef, GcRef, InputArg, InputArgRc, Op, OpCode, OpRc, OpRef, Type, Value};
use std::cell::Cell;

/// Compact stand-in for `history.py *FrontendOp` while the live recorder
/// writes `TraceRecordBuffer` bytes. The 240-byte `Op` is materialized
/// once, at `into_parts` / `get_iter`, matching `opencoder.py TraceIterator.next`
/// (`cls()`). `unique` is the pyre-legacy all-ops `OpRef` callers already
/// store; `box_index` is the opencoder.py `_index` TAGBOX coordinate.
#[derive(Clone, Debug)]
struct FrontendSlot {
    unique: u32,
    #[allow(dead_code)]
    box_index: u32,
    opcode: OpCode,
    descr: Option<DescrRef>,
    resume: Cell<i32>,
    fail_args: Option<Vec<OpRef>>,
    fail_arg_types: Option<Vec<Type>>,
    concrete: Cell<Option<Value>>,
    args: Vec<OpRef>,
}

/// opencoder.py `cut_point()` — RPython 5-tuple
/// `(_pos, _count, _index, len(_snapshot_data), len(_snapshot_array_data))`.
///
/// The byte-stream recorder (`TraceRecordBuffer`) fills in every field
/// from its byte cursor / counter state.  The legacy `Vec<Op>` recorder
/// (`recorder::Trace`, being migrated away) maps `_pos` to
/// the ops-Vec cursor (number of ops currently stored). `_count` mirrors
/// the total number of recorded ops, while `_index` mirrors the number of
/// box-yielding positions (inputargs + non-void ops), matching
/// opencoder.py's split counters even though the legacy `Vec<Op>` recorder
/// still assigns `OpRef` positions in total-op order. Snapshot lens come
/// from the recorder-owned `snapshots` side table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TracePosition {
    /// opencoder.py:475 `self._pos` — byte cursor for TRB, ops-Vec cursor
    /// for `recorder::Trace`.
    pub _pos: usize,
    /// opencoder.py:497 `self._count` — total op count (including voids).
    pub _count: u32,
    /// opencoder.py:498 `self._index` — count of box-yielding (non-void)
    /// ops; equals `_count` in `recorder::Trace`.
    pub _index: u32,
    /// opencoder.py:567 `len(self._snapshot_data)`.
    pub snapshot_data_len: usize,
    /// opencoder.py:567 `len(self._snapshot_array_data)`.
    pub snapshot_array_data_len: usize,
    /// pyre-only: `Trace::guard_count` at the cut point, so
    /// [`Trace::cut`] restores it without rescanning `ops`.  RPython keeps
    /// no guard counter, so this has no slot in opencoder.py's 5-tuple;
    /// `None` marks a position minted by a producer that does not track
    /// one (`TraceRecordBuffer::cut_point`) and makes `cut` recompute.
    pub guard_count: Option<usize>,
}

/// opencoder.py Snapshot parity: per-guard snapshot of the interpreter
/// frame state, encoded as tagged references to boxes.
///
/// RPython stores snapshots inline in the trace byte stream
/// (`_snapshot_data` / `_snapshot_array_data`).  Pyre owns them on
/// `TraceCtx` as a `Vec<Snapshot>` side-table (the pending migration moves
/// this to the byte-stream form already carried by `TraceRecordBuffer`).
/// Each snapshot captures the live variables of each frame in the call
/// stack at the guard point.
#[derive(Clone, Debug)]
pub struct Snapshot {
    /// Frames in the snapshot, outermost first.
    pub frames: Vec<SnapshotFrame>,
    /// Virtualizable box references (tagged).
    pub vable_boxes: Vec<SnapshotTagged>,
    /// VirtualRef box references (tagged).
    pub vref_boxes: Vec<SnapshotTagged>,
}

impl Snapshot {
    /// Forward every inline gcref this snapshot carries.
    ///
    /// `MIFrame.get_list_of_active_snapshot_boxes` copies a constant ref
    /// register's already-forwarded gcref out of its `OpRef::ConstPtr` into a
    /// raw [`SnapshotTagged::Const`] word, and `jitcode.constants_r` entries
    /// reach the same arm. The side table these land in accumulates for the
    /// whole trace and is only handed to the compiler at the end of it, so
    /// without this walk a collection between capture and compile leaves the
    /// resume data naming a pre-move address. RPython has no such copy: its
    /// snapshot holds the `ConstPtr` box itself, whose `value` the GC traces
    /// through the object graph.
    pub fn walk_const_ptr_refs(&mut self, visitor: &mut dyn FnMut(&mut majit_ir::GcRef)) {
        let tagged = self
            .frames
            .iter_mut()
            .flat_map(|f| f.boxes.iter_mut())
            .chain(self.vable_boxes.iter_mut())
            .chain(self.vref_boxes.iter_mut());
        for t in tagged {
            match t {
                SnapshotTagged::Const(bits, Type::Ref) => {
                    let mut gcref = majit_ir::GcRef(*bits as usize);
                    visitor(&mut gcref);
                    *bits = gcref.0 as i64;
                }
                // `build_vable_snapshot_boxes` / `build_vref_snapshot_boxes`
                // tag their entries by `OpRef::ty()` without asking whether the
                // operand is constant, so this arm can hold an inline gcref too.
                SnapshotTagged::Box(OpRef::ConstPtr(gcref), _) => visitor(gcref),
                SnapshotTagged::Const(..) | SnapshotTagged::Box(..) => {}
            }
        }
    }
}

/// `jitcode_index` for a frame the recorder minted with no real coordinate.
///
/// `crate::history::TraceCtx::record_guard_with_snapshot` attaches a
/// one-frame snapshot to the interpreter-side promotes purely to satisfy
/// `resume.py`'s `assert resume_position >= 0`.  That layer is the
/// recorder-side trace buffer and holds no `MIFrameStack`, so it has no
/// position to put in the frame; the dispatch layer re-stamps it with the
/// walker's real coordinate before the guard is finalized.
///
/// The value has to be one no jitcode table can hold. `0` — what this frame
/// carried before — is a live slot, so a *missed* re-stamp reads as a
/// legitimate frame: the decoder sizes the frame from that entry's liveness
/// and then reads that many tagged words, none of which the placeholder wrote.
/// Out of range, `frame_value_count` answers `0` instead, which is what the
/// frame actually holds, and pyre's `frame_value_count_at` reports the missed
/// re-stamp by name rather than as an opaque `jitcode_index=0`.
///
/// The choice of `-2` is forced from both ends. The frame crosses into the
/// `i32`-typed `resume::SnapshotFrame` / `resumedata::RebuiltFrame` as
/// `x as i32` and is then written to `rd_numb` through
/// `resumecode::Writer::append_int`, which asserts the value round-trips
/// through `i16` — so nothing above `32767` survives. Below zero, `-1` is
/// taken: `create_empty_top_snapshot` (`opencoder.rs`) writes it for the
/// snapshot that precedes any entered frame. `-2` is the first free value,
/// and `x as usize` puts it past the end of every jitcode table.
pub const UNSTAMPED_JITCODE_INDEX: u32 = -2i32 as u32;

/// One frame in a snapshot — corresponds to one MIFrame/JitCode position.
#[derive(Clone, Debug)]
pub struct SnapshotFrame {
    /// Index of the jitcode (or 0 for the root portal), or
    /// [`UNSTAMPED_JITCODE_INDEX`] for a frame awaiting the dispatch layer's
    /// re-stamp.
    pub jitcode_index: u32,
    /// Program counter within the jitcode: the JitCode byte offset, as the
    /// MIFrame's `pc` field is upstream (`pyjitpl.py setposition`). Both
    /// writers stamp a JitCode offset — `build_state_field_snapshot` reads
    /// `MIFrame::pc` directly, and `capture_snapshot_for_last_guard_multi_frame`
    /// takes it from the walker's own `build_framestack_snapshot`. There is no
    /// longer a second, Python-keyed coordinate here and no translation table
    /// between the two.
    pub pc: u32,
    /// Forward-carried Python instruction PC for this JitCode position.
    ///
    /// Unlike the `i32`-typed twins this type carries no no-snapshot sentinel:
    /// both fields here are `u32`, no writer stamps one, and "no snapshot" is
    /// the absence of a frame rather than a value inside one. The `-1` sentinel
    /// belongs to `resume::SnapshotFrame` and `resumedata::RebuiltFrame`, whose
    /// signed fields can represent it.
    ///
    /// Upstream derives the Python-level position where it needs one; this
    /// carries it, because the resume decoder that reads it back for
    /// `f_lasti` and traceback reconstruction holds no jitcode metadata and
    /// so has no jitcode→Python inverse available at that point.
    pub py_pc: u32,
    /// Tagged references to the live boxes in this frame.
    pub boxes: Vec<SnapshotTagged>,
}

/// opencoder.py:603 trace-snapshot encode parity: tagged reference to a
/// live value at a recorder snapshot site.  The recorder only sees Box
/// (live in deadframe fail_args) and Const (compile-time constant)
/// payloads; TAGVIRTUAL belongs to the resume-numbering layer
/// (resume.py:_number_boxes) and is synthesized later from
/// PtrInfo::is_virtual on the live Box's OpRef.  Keeping a `Virtual`
/// variant here would let a virtual index reach
/// `translate_trace_iter_opref`'s _cache as a Box position, silently
/// remapping it (recorder.rs has no Box at that integer).  The variant
/// is therefore intentionally absent — adding a virtual-tagged source
/// requires a dedicated resume enum, not this snapshot type.
///
/// TAGBOX(n)    → value lives in `fail_args[n]` (deadframe slot n)
/// TAGCONST(v)  → compile-time constant (i64 value)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnapshotTagged {
    /// Value from deadframe fail_args slot.
    /// RPython: InputArgRef/InputArgInt carry type ('r'/'i'/'f') in
    /// the Box class itself. Pyre stores the typed `OpRef` so the
    /// variant tag (InputArg{Int,Float,Ref} / IntOp/FloatOp/RefOp)
    /// reaches `_number_boxes` (resume.py `box.type == 'r' vs
    /// 'i'`). The trailing `Type` is a redundant lockstep slot kept
    /// for callers that need the type without re-deriving it via
    /// `opref.ty()`.
    Box(majit_ir::OpRef, majit_ir::Type),
    /// Compile-time constant value with type.
    /// RPython resume.py:157: Const boxes carry their type (INT/REF/FLOAT)
    /// for correct TAGINT/TAGCONST encoding in rd_numb.
    Const(i64, majit_ir::Type),
}

pub struct Trace {
    /// Recorded operations. Stored as `Rc<Op>` so a single `Op` identity
    /// flows from the recorder through the `TreeLoop` handoff into the
    /// optimizer (`AbstractValue` object identity).
    ops: Vec<OpRc>,
    /// Input arguments to the trace (live variables at the loop header).
    /// Stored as `InputArgRc` so the recorder's `box_args` bridge can resolve a
    /// canonical, identity-stable `Operand` per input arg (`from_bound_inputarg`
    /// carries the `Rc<InputArg>` directly), and the same `Rc<InputArg>` flows
    /// through `into_parts` / `from_oprc` into `TreeLoop.inputargs` unchanged.
    inputargs: Vec<InputArgRc>,
    /// Positions reserved by `Trace(max_num_inputargs)` that survived
    /// `initialize_state_from_guard_failure`'s hole filtering.
    ///
    /// The backing vector stays dense while recording so an original TAGBOX
    /// position resolves directly. `into_parts` exposes only live FrontendOps,
    /// matching `History.set_inputargs` in RPython.
    inputarg_live: Vec<bool>,
    /// Next OpRef index to assign.
    op_count: u32,
    /// Running count of recorded guards, kept in sync by the two
    /// `record_guard*` entry points (the only producers — `record_op` /
    /// `record_op_with_descr` assert the opcode is not a guard) and
    /// restored by [`Self::cut`] from `TracePosition::guard_count`.
    /// `num_guards` is consulted once per traced vable op
    /// (`walker_capture_inline_nonstandard_vable_guard`), so counting by
    /// scanning `ops` made tracing quadratic in trace length.
    guard_count: usize,
    /// opencoder.py parity: count of box-yielding positions
    /// (inputargs + non-void ops).
    box_count: u32,
    /// Monotonic count of operations appended during this recording session.
    /// Unlike `ops.len()`, this is not rewound by [`Self::cut`].  The bridge
    /// driver uses it to distinguish an abort before the body walk from a
    /// body abort whose speculative operations were cut back to the setup
    /// position (`history.cut` in `pyjitpl.py`).
    recorded_ops_total: usize,
    /// `opencoder.py Trace._refs_dict` / `_cached_const_ptr`: one canonical
    /// constant-ref box per non-null address for the lifetime of this trace.
    /// The byte-stream recorder has the same pool; this copy lives
    /// only at the legacy `Vec<Op>` boundary until that recorder is swapped
    /// out, and prevents every repeated ConstPtr operand from allocating a
    /// fresh `Rc<Cell<Value>>` in the meantime.
    const_ptrs: crate::FxIndexMap<GcRef, Operand>,
    /// Live JIT path: `History.trace` is `opencoder.Trace`. When present,
    /// `record_*` appends bytes and a [`FrontendSlot`] instead of a 240-byte
    /// `Op`. `into_parts` materializes through `ByteTraceIter`, the
    /// `cls()` step. Tests that construct `Trace::new()` keep the `Vec<Op>`
    /// path so they do not need `metainterp_sd`.
    trb: Option<Box<TraceRecordBuffer>>,
    slots: Vec<FrontendSlot>,
    /// unique `OpRef.raw()` → opencoder `_index` for TAGBOX. Inputargs
    /// occupy `[0, n)` as themselves. Void ops store `u32::MAX` and are
    /// never encoded as a value box.
    unique_to_box: Vec<u32>,
}

impl Trace {
    /// Create a new, empty trace recorder.
    ///
    /// opencoder.py Trace.__init__ — trace_limit is enforced at the
    /// MetaInterp / TraceCtx level by consulting warmstate.trace_limit,
    /// not stored on the recorder.
    pub fn new() -> Self {
        Trace {
            ops: Vec::with_capacity(256),
            inputargs: Vec::new(),
            inputarg_live: Vec::new(),
            op_count: 0,
            guard_count: 0,
            box_count: 0,
            recorded_ops_total: 0,
            const_ptrs: crate::FxIndexMap::default(),
            trb: None,
            slots: Vec::new(),
            unique_to_box: Vec::new(),
        }
    }

    /// Create a trace recorder pre-configured for retracing from a guard.
    ///
    /// The recorder starts with `num_inputs` int-typed input args,
    /// matching the guard's fail_args.
    pub fn with_num_inputs(num_inputs: usize) -> Self {
        Self::with_input_types(&vec![Type::Int; num_inputs])
    }

    /// Create a trace recorder pre-configured for retracing from a guard
    /// with explicit input arg types.
    pub fn with_input_types(input_types: &[Type]) -> Self {
        let mut recorder = Self::new();
        for tp in input_types {
            recorder.record_input_arg(*tp);
        }
        recorder
    }

    /// Reserve the full guard failarg coordinate space while exposing only
    /// the live positions when the trace is handed to the optimizer.
    pub fn with_input_layout(input_types: &[Type], live_inputs: &[bool]) -> Self {
        assert_eq!(input_types.len(), live_inputs.len());
        let mut recorder = Self::with_input_types(input_types);
        recorder.inputarg_live.copy_from_slice(live_inputs);
        recorder
    }

    /// Attach `opencoder.Trace` as the live storage. `History.__init__`
    /// (`history.py`) builds that buffer with `metainterp_sd` once the
    /// inputarg cap is known. Called from `TraceCtx::new` after
    /// `record_input_arg` has reserved the frontend boxes, so existing
    /// inputargs are replayed into the buffer and later `record_*` calls
    /// write bytes instead of `Rc<Op>`.
    pub fn attach_byte_buffer(
        &mut self,
        metainterp_sd: std::sync::Arc<crate::MetaInterpStaticData>,
    ) {
        if self.trb.is_some() || !self.ops.is_empty() || !self.slots.is_empty() {
            return;
        }
        let n = self.inputargs.len() as u32;
        let mut trb = TraceRecordBuffer::new(n, metainterp_sd);
        for ia in &self.inputargs {
            trb.record_input_arg(ia.tp);
        }
        self.unique_to_box = (0..n).collect();
        self.trb = Some(Box::new(trb));
    }

    fn byte_mode(&self) -> bool {
        self.trb.is_some()
    }

    fn arg_to_box(&self, r: OpRef) -> OcBox {
        if r.is_constant() {
            let value = r
                .inline_const_to_value()
                .unwrap_or_else(|| panic!("arg_to_box: constant {r:?} not inline-resolvable"));
            return match value {
                Value::Int(v) => OcBox::ConstInt(v),
                Value::Float(f) => OcBox::ConstFloat(f.to_bits()),
                Value::Ref(g) => OcBox::ConstPtr(g.as_usize() as u64),
                Value::Void => panic!("arg_to_box: constant {r:?} has Void type"),
            };
        }
        let raw = r.raw();
        if (raw as usize) < self.inputargs.len() {
            return OcBox::ResOp(raw);
        }
        let mapped = *self
            .unique_to_box
            .get(raw as usize)
            .unwrap_or_else(|| panic!("arg_to_box: OpRef {r:?} has no TAGBOX index"));
        assert!(
            mapped != u32::MAX,
            "arg_to_box: void op {r:?} used as a value box"
        );
        OcBox::ResOp(mapped)
    }

    fn record_bytes(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        descr: Option<DescrRef>,
        fail_args: Option<&[OpRef]>,
    ) -> OpRef {
        let unique = self.op_count;
        let opref = OpRef::op_typed(unique, opcode.result_type());
        let boxes: Vec<OcBox> = args.iter().copied().map(|a| self.arg_to_box(a)).collect();
        let trb = self
            .trb
            .as_mut()
            .expect("record_bytes requires attach_byte_buffer");
        let box_index = trb.record_op(opcode, &boxes, descr.as_ref());
        if opcode.result_type() != Type::Void {
            if self.unique_to_box.len() <= unique as usize {
                self.unique_to_box.resize(unique as usize + 1, u32::MAX);
            }
            self.unique_to_box[unique as usize] = box_index;
            self.box_count += 1;
        } else if self.unique_to_box.len() <= unique as usize {
            self.unique_to_box.resize(unique as usize + 1, u32::MAX);
        }
        if opcode.is_guard() {
            self.guard_count += 1;
        }
        self.slots.push(FrontendSlot {
            unique,
            box_index,
            opcode,
            descr,
            resume: Cell::new(-1),
            fail_args: fail_args.map(|a| a.to_vec()),
            fail_arg_types: None,
            concrete: Cell::new(None),
            args: args.to_vec(),
        });
        self.recorded_ops_total += 1;
        self.op_count += 1;
        opref
    }

    fn slot_by_unique(&self, raw: u32) -> Option<&FrontendSlot> {
        let n = self.inputargs.len() as u32;
        raw.checked_sub(n).and_then(|i| self.slots.get(i as usize))
    }

    fn last_slot_mut(&mut self) -> Option<&mut FrontendSlot> {
        self.slots.last_mut()
    }

    fn materialize_ops(&self) -> Vec<OpRc> {
        let Some(trb) = self.trb.as_ref() else {
            return self.ops.clone();
        };
        // start_fresh=0 so inputargs occupy `[0, n)` and recorded ops
        // occupy `[n, …)` — the same unique numbering `record_*` handed
        // to snapshots. `get_byte_iter` seeds `_fresh` at
        // `max_num_inputargs`, which would shift every box by `n`.
        let ops: Vec<OpRc> =
            crate::opencoder::ByteTraceIter::new(trb, trb._start as usize, trb._pos, 0).collect();
        let n = self.inputargs.len() as u32;
        for op in &ops {
            for i in 0..op.num_args() {
                let arg = op.arg(i);
                if arg.is_inputarg() {
                    if let Some(idx) = arg.position() {
                        if let Some(ours) = self.inputargs.get(idx as usize) {
                            op.setarg(i, Operand::from_bound_inputarg(ours));
                        }
                    }
                }
            }
        }
        for (i, op) in ops.iter().enumerate() {
            let unique = n + i as u32;
            op.pos.set(OpRef::op_typed(unique, op.opcode.result_type()));
            if let Some(slot) = self.slots.get(i) {
                if let Some(v) = slot.concrete.get() {
                    op.set_value(v);
                }
                if let Some(d) = slot.descr.clone() {
                    op.setdescr(d);
                }
                op.rd_resume_position.set(slot.resume.get());
                if let Some(ref types) = slot.fail_arg_types {
                    op.set_fail_arg_types(types.clone());
                }
            }
        }
        for (i, op) in ops.iter().enumerate() {
            let Some(slot) = self.slots.get(i) else {
                continue;
            };
            let Some(ref fail) = slot.fail_args else {
                continue;
            };
            let boxed: Vec<Operand> = fail
                .iter()
                .copied()
                .map(|r| self.operand_from_materialized(r, &ops))
                .collect();
            op.setfailargs(boxed.into());
        }
        ops
    }

    fn operand_from_materialized(&self, r: OpRef, ops: &[OpRc]) -> Operand {
        if r.is_constant() {
            return Operand::from_opref(r);
        }
        let raw = r.raw() as usize;
        let n = self.inputargs.len();
        if raw < n {
            return Operand::from_bound_inputarg(&self.inputargs[raw]);
        }
        let idx = raw - n;
        if let Some(op) = ops.get(idx) {
            return Operand::from_bound_op(op);
        }
        Operand::from_opref(r)
    }

    /// Register an input argument of the given type.
    /// Returns an OpRef that can be used as an argument to subsequent operations.
    /// Input arguments are numbered starting from 0; the OpRef index matches
    /// the input argument index.
    ///
    /// resoperation.py/727/739 — InputArgInt/InputArgFloat/InputArgRef
    /// each pin `type = 'i'/'f'/'r'` at construction.
    pub fn record_input_arg(&mut self, tp: Type) -> OpRef {
        assert!(
            self.ops.is_empty() && self.slots.is_empty(),
            "input args must be registered before any operations"
        );
        let index = self.inputargs.len() as u32;
        self.inputargs.push(InputArg::from_type_rc(tp, index));
        self.inputarg_live.push(true);
        let opref = match tp {
            Type::Int => OpRef::input_arg_int(self.op_count),
            Type::Float => OpRef::input_arg_float(self.op_count),
            Type::Ref => OpRef::input_arg_ref(self.op_count),
            Type::Void => panic!("input args cannot be Void"),
        };
        self.op_count += 1;
        self.box_count += 1;
        opref
    }

    /// Bridge the recorder's OpRef-operand API to the `Operand`-carrying
    /// `Op.args` / `Op.fail_args` storage. Each operand resolves to its
    /// *canonical* producer `Operand` so the stored args share one producer
    /// `Rc` (`AbstractValue` object identity): `from_bound_op` /
    /// `from_bound_inputarg` carry the producer `Rc<Op>` / `Rc<InputArg>`
    /// directly. Frame registers and the public `record_*` API stay OpRef; the
    /// optimizer bridges back with `Operand::to_opref`, which round-trips to the
    /// same `OpRef` the `from_opref` view produced.
    fn box_args(&mut self, args: &[OpRef]) -> smallvec::SmallVec<[Operand; 3]> {
        args.iter().map(|&a| self.box_for_operand(a)).collect()
    }

    /// Resolve one operand `OpRef` to its canonical `Operand`. Const operands
    /// carry their `Value` inline on the `OpRef` (history.py:227/268/314) and
    /// are minted via `from_opref`; InputArg / ResOp operands resolve to the
    /// producing `InputArg` / `Op` already held in `self.inputargs` / `self.ops`,
    /// which are dense by construction. A non-const operand that does not resolve
    /// to a recorded producer is a recorder invariant violation and panics
    /// (opencoder.py:635/640 assert position rather than mint); the S0/#121 probe
    /// measured 0 hits across the corpus. `#[cfg(test)]` fixtures that build
    /// position-only synthetic operands bind a synthetic producer via
    /// `bound_from_opref` (`to_opref`-identical) rather than panicking.
    /// Deterministic per recorded position: a ResOp / InputArg operand always
    /// binds to the SAME producer `Rc` (`self.ops` / `self.inputargs` are dense
    /// and immutable once recorded), so two calls for the same `OpRef` return
    /// `Operand`s that compare equal under `Operand::eq` (`Rc::ptr_eq`). This is
    /// the real box object — `MIFrame.registers_r` holds these in `pyjitpl.py` —
    /// so consumers can key by box identity instead of flat `OpRef`.
    pub(crate) fn box_for_operand(&mut self, r: OpRef) -> Operand {
        if let OpRef::ConstPtr(gcref) = r {
            if gcref.is_null() {
                return Operand::NullRef;
            }
            if let Some(cached) = self.const_ptrs.get(&gcref) {
                return cached.clone();
            }
            let operand = Operand::from_opref(r);
            self.const_ptrs.insert(gcref, operand.clone());
            return operand;
        }
        if r.is_none() || r.is_constant() {
            return Operand::from_opref(r);
        }
        if let OpRef::InputArgInt(_) | OpRef::InputArgFloat(_) | OpRef::InputArgRef(_) = r {
            if let Some(ia) = self.inputargs.get(r.raw() as usize) {
                return Operand::from_bound_inputarg(ia);
            }
            // S3/#124: inputargs are dense `[0, n)`, so an InputArg operand
            // always resolves above (opencoder.py:635 asserts).
            #[cfg(not(test))]
            panic!(
                "box_for_operand: InputArg operand {r:?} outside dense \
                 inputargs[0,{}) (opencoder.py:635)",
                self.inputargs.len()
            );
            #[cfg(test)]
            return Operand::bound_from_opref(r);
        }
        // ResOp operand: positions are dense in op_count order — inputargs
        // occupy `[0, n)`, recorded ops occupy `[n, ..)` — so the producing op
        // sits at `self.ops[r.raw() - n]`. Confirm `op.pos` matches before
        // binding, falling back to `from_opref` on any mismatch.
        let n = self.inputargs.len();
        if let Some(idx) = (r.raw() as usize).checked_sub(n)
            && let Some(op) = self.ops.get(idx)
            && op.pos.get().raw() == r.raw()
        {
            return Operand::from_bound_op(op);
        }
        // Byte-mode recording has no `Op` yet (`FrontendSlot` only).
        // RPython's FrontendOp *is* the box; `from_opref` is that
        // position-only stand-in until `ByteTraceIter` materializes.
        if self.byte_mode() && self.slot_by_unique(r.raw()).is_some() {
            return Operand::from_opref(r);
        }
        // S3/#124: a ResOp operand always has a dense producer in `ops`
        // (positions are op_count order; opencoder.py:640 asserts).
        #[cfg(not(test))]
        panic!(
            "box_for_operand: ResOp operand {r:?} has no dense producer in \
             ops[0,{}) (opencoder.py:640)",
            self.ops.len()
        );
        #[cfg(test)]
        Operand::bound_from_opref(r)
    }

    /// Record a regular (non-guard) operation.
    /// Returns the OpRef for this operation's result.
    ///
    /// `AbstractResOp` + IntOp/FloatOp/RefOp mixins (resoperation.py)
    /// pin the result type at construction. Void-result ops keep the
    /// `AbstractResOp.type = 'v'` default (resoperation.py).
    pub fn record_op(&mut self, opcode: OpCode, args: &[OpRef]) -> OpRef {
        assert!(!opcode.is_guard(), "use record_guard for guard operations");
        if self.byte_mode() {
            return self.record_bytes(opcode, args, None, None);
        }
        let opref = OpRef::op_typed(self.op_count, opcode.result_type());
        let op = Op::new(opcode, &self.box_args(args));
        op.pos.set(opref);
        self.ops.push(OpRc::new(op));
        self.recorded_ops_total += 1;
        self.op_count += 1;
        if opcode.result_type() != Type::Void {
            self.box_count += 1;
        }
        opref
    }

    /// Record an operation with a descriptor (e.g., field access, call).
    /// Returns the OpRef for this operation's result.
    pub fn record_op_with_descr(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        descr: DescrRef,
    ) -> OpRef {
        assert!(!opcode.is_guard(), "use record_guard for guard operations");
        if self.byte_mode() {
            return self.record_bytes(opcode, args, Some(descr), None);
        }
        let opref = OpRef::op_typed(self.op_count, opcode.result_type());
        let op = Op::with_descr(opcode, &self.box_args(args), descr);
        op.pos.set(opref);
        self.ops.push(OpRc::new(op));
        self.recorded_ops_total += 1;
        self.op_count += 1;
        if opcode.result_type() != Type::Void {
            self.box_count += 1;
        }
        opref
    }

    /// Record a guard operation.
    /// `pyjitpl.py generate_guard()` parity: tracer-stage guards
    /// carry `descr=None`. The optimizer creates the FailDescr later in
    /// `store_final_boxes_in_guard` / `invent_fail_descr_for_op`
    /// (compile.py:722-730 / 924-942). Tests that need a pre-stamped
    /// descr pass `Some(descr)`.
    /// Returns the OpRef for this guard.
    pub fn record_guard(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        descr: Option<DescrRef>,
    ) -> OpRef {
        assert!(opcode.is_guard(), "opcode {:?} is not a guard", opcode);
        if self.byte_mode() {
            return self.record_bytes(opcode, args, descr, None);
        }
        let opref = OpRef::op_typed(self.op_count, opcode.result_type());
        self.guard_count += 1;
        let op = match descr {
            Some(d) => Op::with_descr(opcode, &self.box_args(args), d),
            None => Op::new(opcode, &self.box_args(args)),
        };
        op.pos.set(opref);
        self.ops.push(OpRc::new(op));
        self.recorded_ops_total += 1;
        self.op_count += 1;
        if opcode.result_type() != Type::Void {
            self.box_count += 1;
        }
        opref
    }

    /// Record a guard with explicit fail_args — values stored in the dead
    /// frame on guard failure. Mirrors rpython setfailargs().
    /// See `record_guard` for the descr=None convention.
    pub fn record_guard_with_fail_args(
        &mut self,
        opcode: OpCode,
        args: &[OpRef],
        descr: Option<DescrRef>,
        fail_args: &[OpRef],
    ) -> OpRef {
        assert!(opcode.is_guard(), "opcode {:?} is not a guard", opcode);
        if self.byte_mode() {
            return self.record_bytes(opcode, args, descr, Some(fail_args));
        }
        let opref = OpRef::op_typed(self.op_count, opcode.result_type());
        self.guard_count += 1;
        let op = match descr {
            Some(d) => Op::with_descr(opcode, &self.box_args(args), d),
            None => Op::new(opcode, &self.box_args(args)),
        };
        op.pos.set(opref);
        op.setfailargs(self.box_args(fail_args).iter().cloned().collect());
        self.ops.push(OpRc::new(op));
        self.recorded_ops_total += 1;
        self.op_count += 1;
        if opcode.result_type() != Type::Void {
            self.box_count += 1;
        }
        opref
    }

    /// Set rd_resume_position on the last recorded op.
    /// Called after record_guard* to associate a snapshot.
    pub fn set_last_op_resume_position(&mut self, snapshot_id: i32) {
        if let Some(slot) = self.last_slot_mut() {
            slot.resume.set(snapshot_id);
            return;
        }
        if let Some(op) = self.ops.last() {
            op.rd_resume_position.set(snapshot_id);
        }
    }

    /// Set rd_resume_position on the most-recently recorded *guard* op,
    /// skipping any non-guard ops recorded after it.
    ///
    /// `set_last_op_resume_position` assumes the guard is the last op,
    /// which holds when a guard is captured immediately after recording.
    /// A guard emitted *inside* a helper (e.g. the
    /// `_nonstandard_virtualizable` PTR_EQ promote, after which
    /// `emit_force_virtualizable` records GETFIELD_GC / PTR_NE /
    /// COND_CALL) is not the last op when the caller captures, so the
    /// resume position must target the guard by its guard-ness.
    pub fn set_last_guard_op_resume_position(&mut self, snapshot_id: i32) {
        self.set_guard_op_resume_position_from_end(0, snapshot_id);
    }

    /// Set rd_resume_position on the guard op `from_end` guards back from the
    /// most recently recorded one (`0` == the most recent).
    ///
    /// One helper call can emit more than one guard: a vable array access
    /// promotes the `isstandard` PTR_EQ in `_nonstandard_virtualizable`
    /// (`pyjitpl.py:1135-1138`) and then the index in
    /// `_get_arrayitem_vable_index` (`:1201-1216`).  Upstream captures resume
    /// data inside each `implement_guard_value`, so both are stamped; a caller
    /// that only reaches the guards after the helper returns walks back over
    /// them with this.
    pub fn set_guard_op_resume_position_from_end(&mut self, from_end: usize, snapshot_id: i32) {
        if let Some(slot) = self
            .slots
            .iter()
            .rev()
            .filter(|s| s.opcode.is_guard())
            .nth(from_end)
        {
            slot.resume.set(snapshot_id);
            return;
        }
        if let Some(op) = self
            .ops
            .iter()
            .rev()
            .filter(|op| op.opcode.is_guard())
            .nth(from_end)
        {
            op.rd_resume_position.set(snapshot_id);
        }
    }

    /// Replace the descriptor on the last recorded operation.
    pub fn set_last_op_descr(&mut self, descr: DescrRef) {
        if let Some(slot) = self.last_slot_mut() {
            slot.descr = Some(descr);
            return;
        }
        if let Some(op) = self.ops.last() {
            op.setdescr(descr);
        }
    }

    /// Opcode of the op [`set_last_op_descr`](Self::set_last_op_descr) would
    /// stamp, for a caller that has to mint the descr subtype the opcode
    /// requires.
    pub fn last_op_opcode(&self) -> Option<OpCode> {
        if let Some(slot) = self.slots.last() {
            return Some(slot.opcode);
        }
        self.ops.last().map(|op| op.opcode)
    }

    /// Opcode of the op
    /// [`set_guard_op_descr_from_end`](Self::set_guard_op_descr_from_end)
    /// would stamp, selected by the same walk.
    pub fn guard_op_opcode_from_end(&self, from_end: usize) -> Option<OpCode> {
        if let Some(slot) = self
            .slots
            .iter()
            .rev()
            .filter(|s| s.opcode.is_guard())
            .nth(from_end)
        {
            return Some(slot.opcode);
        }
        self.ops
            .iter()
            .rev()
            .filter(|op| op.opcode.is_guard())
            .nth(from_end)
            .map(|op| op.opcode)
    }

    /// Replace the descriptor on the guard `from_end` guards back from the
    /// most recently recorded one.
    pub fn set_guard_op_descr_from_end(&mut self, from_end: usize, descr: DescrRef) {
        if let Some(slot) = self
            .slots
            .iter_mut()
            .rev()
            .filter(|s| s.opcode.is_guard())
            .nth(from_end)
        {
            slot.descr = Some(descr);
            return;
        }
        if let Some(op) = self
            .ops
            .iter()
            .rev()
            .filter(|op| op.opcode.is_guard())
            .nth(from_end)
        {
            op.setdescr(descr);
        }
    }

    /// Opcode of the most recently recorded guard, if any.  Snapshot
    /// capture keys `after_residual_call` on the guard opcode itself
    /// (`pyjitpl.py generate_guard`).
    pub fn last_guard_opcode(&self) -> Option<OpCode> {
        if let Some(slot) = self.slots.iter().rev().find(|s| s.opcode.is_guard()) {
            return Some(slot.opcode);
        }
        self.ops
            .iter()
            .rev()
            .find(|op| op.opcode.is_guard())
            .map(|op| op.opcode)
    }

    /// Set fail_args on a recorded op identified by `opref`.
    ///
    /// Mirrors RPython's `Op.setfailargs([...])` (`resoperation.py`)
    /// — the post-record mutation API used by tests and any path that
    /// constructs a synthetic guard shape outside the standard
    /// `capture_resumedata` flow.  Production guard recording uses the
    /// snapshot path (`record_guard_typed` + `capture_resumedata` +
    /// `set_last_op_resume_position`); `op.fail_args` is then derived
    /// from the snapshot via `op.store_final_boxes(liveboxes)` in the
    /// optimizer's `store_final_boxes_in_guard`.
    pub fn set_op_fail_args(&mut self, opref: OpRef, fail_args: &[OpRef]) {
        if let Some(slot) = self
            .slots
            .iter_mut()
            .rev()
            .find(|s| s.unique == opref.raw())
        {
            slot.fail_args = Some(fail_args.to_vec());
            return;
        }
        let boxed_fail_args = self.box_args(fail_args).iter().cloned().collect();
        let op = self
            .ops
            .iter()
            .rev()
            .find(|op| op.pos.get() == opref)
            .unwrap_or_else(|| panic!("set_op_fail_args: no op with pos {:?}", opref));
        op.setfailargs(boxed_fail_args);
    }

    /// Set `fail_arg_types` on the last recorded op. Used by
    /// `trace_ctx::record_guard_typed` to record types for fail args
    /// without stamping a tracer-stage descr (codex #3 /
    /// pyjitpl.py generate_guard parity).
    pub fn set_last_op_fail_arg_types(&mut self, types: Vec<Type>) {
        if let Some(slot) = self.last_slot_mut() {
            slot.fail_arg_types = Some(types);
            return;
        }
        if let Some(op) = self.ops.last() {
            op.set_fail_arg_types(types);
        }
    }

    /// Close the loop: add a JUMP operation back to the start.
    /// `jump_args` are the values of the input arguments at the end of the loop.
    pub fn close_loop(&mut self, jump_args: &[OpRef]) {
        self.close_loop_with_descr(jump_args, None);
    }

    /// Close the loop with an explicit JUMP descriptor.
    ///
    /// RPython pyjitpl.py:3188-3190 records the tentative JUMP with
    /// `descr=ptoken` before compile_trace(). Plain loop recording keeps
    /// `descr=None` until optimization rewrites it.
    pub fn close_loop_with_descr(&mut self, jump_args: &[OpRef], descr: Option<DescrRef>) {
        if self.byte_mode() {
            self.record_bytes(OpCode::Jump, jump_args, descr, None);
            return;
        }
        // RPython parity: Jump args may differ from InputArgs count when
        // virtualizable arrays change depth. The optimizer (OptUnroll preamble
        // peeling) bridges the gap by creating a Label with the extended count.
        let opref = OpRef::op_typed(self.op_count, OpCode::Jump.result_type());
        let op = match descr {
            Some(descr) => Op::with_descr(OpCode::Jump, &self.box_args(jump_args), descr),
            None => Op::new(OpCode::Jump, &self.box_args(jump_args)),
        };
        op.pos.set(opref);
        self.ops.push(OpRc::new(op));
        self.recorded_ops_total += 1;
        self.op_count += 1;
        if OpCode::Jump.result_type() != Type::Void {
            self.box_count += 1;
        }
    }

    /// Finish the trace (non-looping): add a FINISH operation.
    /// `finish_args` are the values returned from the trace.
    pub fn finish(&mut self, finish_args: &[OpRef], descr: DescrRef) {
        if self.byte_mode() {
            self.record_bytes(OpCode::Finish, finish_args, Some(descr), None);
            return;
        }
        let opref = OpRef::op_typed(self.op_count, OpCode::Finish.result_type());
        let op = Op::with_descr(OpCode::Finish, &self.box_args(finish_args), descr);
        op.pos.set(opref);
        self.ops.push(OpRc::new(op));
        self.recorded_ops_total += 1;
        self.op_count += 1;
        if OpCode::Finish.result_type() != Type::Void {
            self.box_count += 1;
        }
    }

    /// Consume the recorder and return its parts: (inputargs, ops).
    ///
    /// history.py parity: the recording phase ends and the trace is handed
    /// to the optimizer as a `TreeLoop`. See `TraceCtx::into_tree_loop` for
    /// the snapshot-bearing path.
    pub fn into_parts(self) -> (Vec<InputArgRc>, Vec<OpRc>) {
        let ops = if self.trb.is_some() {
            self.materialize_ops()
        } else {
            self.ops
        };
        let inputargs = self
            .inputargs
            .into_iter()
            .zip(self.inputarg_live)
            .filter_map(|(arg, live)| live.then_some(arg))
            .collect();
        (inputargs, ops)
    }

    /// Materialize the history's live input box list without consuming it.
    ///
    /// `pyjitpl.py initialize_state_from_guard_failure` installs the
    /// hole-filtered list with `History.set_inputargs`; each surviving box
    /// retains its original position in the recorder's reserved coordinate
    /// space.  This is the pre-`into_parts` view used while closing a bridge.
    pub fn live_inputargs_cloned(&self) -> Vec<InputArg> {
        self.inputargs
            .iter()
            .zip(self.inputarg_live.iter())
            .filter_map(|(arg, &live)| live.then(|| arg.fresh_value_copy()))
            .collect()
    }

    /// Convenience: consume the recorder and produce a `TreeLoop`.
    ///
    /// Snapshots are NOT included — callers that need them should use
    /// `TraceCtx::into_tree_loop` instead.
    pub fn get_trace(self) -> crate::history::TreeLoop {
        // `self.ops` is already `Vec<OpRc>`; `from_oprc` preserves that
        // shared identity.
        let (inputargs, ops) = self.into_parts();
        crate::history::TreeLoop::from_oprc(inputargs, ops, Vec::new())
    }

    /// opencoder.py `cut_point()` — the recorder's local slice of
    /// the 5-tuple. `snapshot_data_len` / `snapshot_array_data_len` come
    /// from `TraceCtx` (which owns the pyre-only `Vec<Snapshot>` side
    /// table); callers should use `TraceCtx::get_trace_position` for a
    /// fully-populated position.
    pub fn get_position(&self) -> TracePosition {
        if let Some(trb) = self.trb.as_ref() {
            return TracePosition {
                _pos: trb._pos,
                _count: self.op_count,
                _index: self.box_count,
                snapshot_data_len: 0,
                snapshot_array_data_len: 0,
                guard_count: Some(self.guard_count),
            };
        }
        TracePosition {
            _pos: self.ops.len(),
            _count: self.op_count,
            _index: self.box_count,
            snapshot_data_len: 0,
            snapshot_array_data_len: 0,
            guard_count: Some(self.guard_count),
        }
    }

    /// opencoder.py `cut_at(end)` — restore the recorder to a
    /// previously saved position.
    ///
    /// Discards all operations recorded after `pos`. Used to undo a
    /// tentative JUMP after compile_trace succeeds or fails
    /// (pyjitpl.py finally: `self.history.cut(cut_at)`).
    pub fn cut(&mut self, pos: TracePosition) {
        if let Some(trb) = self.trb.as_mut() {
            let n = self.inputargs.len() as u32;
            trb.cut_at(crate::recorder::TracePosition {
                _pos: pos._pos,
                _count: n + (pos._count.saturating_sub(n)),
                _index: pos._index,
                snapshot_data_len: 0,
                snapshot_array_data_len: 0,
                guard_count: pos.guard_count,
            });
            let nops = pos._count.saturating_sub(n) as usize;
            self.slots.truncate(nops);
            self.unique_to_box.truncate(pos._count as usize);
            self.op_count = pos._count;
            self.box_count = pos._index;
            self.guard_count = pos
                .guard_count
                .unwrap_or_else(|| self.slots.iter().filter(|s| s.opcode.is_guard()).count());
            self.ops.clear();
            return;
        }
        self.ops.truncate(pos._pos);
        self.op_count = pos._count;
        self.box_count = pos._index;
        // `record_result_of_call_pure` cuts back after every speculative
        // pure-call record, so this runs on a hot path — restore from the
        // saved position instead of rescanning `ops`.
        self.guard_count = pos
            .guard_count
            .unwrap_or_else(|| self.ops.iter().filter(|op| op.opcode.is_guard()).count());
    }

    /// history.py `length`: number of non-inputarg ops recorded so far.
    /// Compared against `warmstate.trace_limit` by
    /// `MetaInterp.blackhole_if_trace_too_long` (pyjitpl.py).
    pub fn num_ops(&self) -> usize {
        if self.byte_mode() {
            return self.slots.len();
        }
        self.ops.len()
    }

    /// Number of operations ever appended to this recorder, including
    /// speculative operations subsequently discarded by [`Self::cut`].
    pub fn recorded_ops_total(&self) -> usize {
        self.recorded_ops_total
    }

    /// Number of input arguments registered.
    pub fn num_inputargs(&self) -> usize {
        self.inputargs.len()
    }

    /// Input argument types in loop-header order.
    pub fn inputarg_types(&self) -> Vec<Type> {
        self.inputargs.iter().map(|arg| arg.tp).collect()
    }

    /// Number of guards recorded so far.
    pub fn num_guards(&self) -> usize {
        self.guard_count
    }

    /// Access the recorded operations.
    pub fn ops(&self) -> &[OpRc] {
        &self.ops
    }

    /// Visit each `ConstPtr` box once and re-key `_refs_dict` after a moving
    /// collection. RPython's `new_ref_dict` follows moved keys as part of the
    /// translated GC; Rust's `IndexMap` does not, so the re-key is the minimal
    /// explicit adaptation. Constants inserted by test-only direct-op helpers
    /// are not in the pool and are visited from their operation instead.
    pub(crate) fn walk_const_ptr_refs(&mut self, visitor: &mut dyn FnMut(&mut GcRef)) {
        // `history.py new_ref_dict` uses `rd_hash` / `lltype.identityhash`,
        // which survives movement. Our address-keyed map must remove ALL old
        // keys before inserting any new ones: a destination can equal another
        // object's old address. Removing index zero and immediately reinserting
        // also changes which object swap_remove brings into index zero, so a
        // fixed-count loop can repeatedly visit the same box and skip others.
        // Drain the identities, retaining the map's capacity, then re-key once.
        let operands: smallvec::SmallVec<[Operand; 8]> = self
            .const_ptrs
            .drain(..)
            .map(|(_, operand)| operand)
            .collect();
        for operand in operands {
            operand.walk_const_ptr_refs(visitor);
            let Value::Ref(gcref) = operand
                .const_value()
                .expect("_refs_dict contains only ConstPtr operands")
            else {
                unreachable!("_refs_dict contains only ConstPtr operands")
            };
            self.const_ptrs.insert(gcref, operand);
        }

        let is_pooled_const_ptr = |arg: &Operand| {
            let Some(Value::Ref(gcref)) = arg.const_value() else {
                return false;
            };
            self.const_ptrs
                .get(&gcref)
                .is_some_and(|cached| cached == arg)
        };
        for slot in &self.slots {
            if let Some(Value::Ref(mut gcref)) = slot.concrete.get() {
                visitor(&mut gcref);
                slot.concrete.set(Some(Value::Ref(gcref)));
            }
        }
        for op in &self.ops {
            for arg in op.args.borrow().iter() {
                if !is_pooled_const_ptr(arg) {
                    arg.walk_const_ptr_refs(visitor);
                }
            }
            if let Some(fail_args) = op.fail_args.borrow().as_ref() {
                for arg in fail_args {
                    if !is_pooled_const_ptr(arg) {
                        arg.walk_const_ptr_refs(visitor);
                    }
                }
            }
            if let Some(Value::Ref(mut gcref)) = op.get_value() {
                visitor(&mut gcref);
                op.set_value(Value::Ref(gcref));
            }
        }
    }

    /// Test-only direct append. Production callers go through
    /// `record_op` so `op_count` / `box_count` stay in sync; this helper
    /// is for GC walker unit tests that exercise the op-graph storage
    /// without driving the full record path.
    #[cfg(test)]
    pub fn push_op_for_test(&mut self, op: Op) {
        if op.opcode.is_guard() {
            self.guard_count += 1;
        }
        self.ops.push(OpRc::new(op));
    }

    /// Access the recorded input arguments.
    pub fn inputargs(&self) -> &[InputArgRc] {
        &self.inputargs
    }

    /// Get an operation by its OpRef position.
    pub fn get_op_by_pos(&self, pos: OpRef) -> Option<&Op> {
        self.ops
            .iter()
            .find(|op| op.pos.get() == pos)
            .map(|op| &**op)
    }

    /// Byte-mode stand-in for `get_op_by_raw_pos` on a `GetfieldGcR`
    /// recorded as a [`FrontendSlot`]. Used by `recover_ref_value` while
    /// the 240-byte `Op` does not yet exist.
    pub(crate) fn getfield_gc_r_at(&self, raw: u32) -> Option<(DescrRef, OpRef)> {
        let slot = self.slot_by_unique(raw)?;
        if slot.opcode != OpCode::GetfieldGcR {
            return None;
        }
        let descr = slot.descr.clone()?;
        let obj = slot.args.first().copied()?;
        Some((descr, obj))
    }

    /// Fill `self.ops` from the byte buffer so `&[Op]` readers
    /// (`get_op_by_raw_pos`, tests after `into_recorder`) see the
    /// materialized trace. No-op on the `Vec<Op>` path or if already filled.
    pub fn materialize_into_ops(&mut self) {
        if self.trb.is_some() && self.ops.is_empty() && !self.slots.is_empty() {
            self.ops = self.materialize_ops();
        }
    }

    /// Get an operation by its raw u32 position, ignoring the variant
    /// tag. Use this when iterating a numeric position range without
    /// knowing each op's RPython `box.type` upfront — the typed lookup
    /// `get_op_by_pos` requires the variant to match (variant-aware Eq).
    pub fn get_op_by_raw_pos(&self, raw: u32) -> Option<&Op> {
        self.ops
            .iter()
            .find(|op| op.pos.get().raw() == raw)
            .map(|op| &**op)
    }

    /// Stamp the concrete runtime value on the canonical frontend object
    /// for `position` (`history.py *FrontendOp(pos, value)` — the
    /// value lives on the `InputArg`/`Op` identity, not on a side pool).
    /// `position` indexes the dense recording order: `[0, inputargs.len())`
    /// are inputargs, `[inputargs.len(), ..)` are recorded ops. Returns
    /// `false` if `position` is past the recorded range.
    pub(crate) fn set_concrete_at(&self, position: u32, value: Value) -> bool {
        let pos = position as usize;
        let n = self.inputargs.len();
        if pos < n {
            self.inputargs[pos].set_value(value);
            true
        } else if let Some(slot) = self.slots.get(pos - n) {
            slot.concrete.set(Some(value));
            true
        } else if let Some(op) = self.ops.get(pos - n) {
            op.set_value(value);
            true
        } else {
            false
        }
    }

    /// Read the concrete runtime value stamped on the canonical frontend
    /// object for `position` (`history.py *FrontendOp.getint()`).
    /// `None` when never stamped or out of range.
    pub(crate) fn concrete_at(&self, position: u32) -> Option<Value> {
        let pos = position as usize;
        let n = self.inputargs.len();
        if pos < n {
            self.inputargs[pos].get_value()
        } else if let Some(slot) = self.slots.get(pos - n) {
            slot.concrete.get()
        } else {
            self.ops.get(pos - n).and_then(|op| op.get_value())
        }
    }
}

impl Default for Trace {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_ir::{DescrRef, FailDescr, Type};
    use std::sync::Arc;

    /// The three constraints that pick [`UNSTAMPED_JITCODE_INDEX`], asserted
    /// together so a future edit to the value fails here instead of silently
    /// re-aliasing a coordinate the decoder accepts.
    #[test]
    fn unstamped_jitcode_index_is_reserved_and_encodable() {
        let idx = UNSTAMPED_JITCODE_INDEX as i32;
        // Written to `rd_numb` through `resumecode::Writer::append_int`, which
        // asserts the value round-trips through `i16`.
        assert_eq!(idx as i16 as i32, idx, "must survive the rd_numb i16 write");
        // `-1` is `create_empty_top_snapshot`'s own frame index.
        assert_ne!(idx, -1, "collides with the empty-top-snapshot index");
        // Every `frame_value_count` decoder resolves the frame with
        // `jitcodes.get(jitcode_index as usize)`; the reserved value must miss.
        assert!(
            (idx as usize) > u32::MAX as usize,
            "must be out of range for any jitcode table"
        );
    }

    fn iarg(pos: u32) -> OpRef {
        OpRef::input_arg_int(pos)
    }

    fn farg(pos: u32) -> OpRef {
        OpRef::input_arg_float(pos)
    }

    fn rarg(pos: u32) -> OpRef {
        OpRef::input_arg_ref(pos)
    }

    fn iop(pos: u32) -> OpRef {
        OpRef::int_op(pos)
    }

    fn vop(pos: u32) -> OpRef {
        OpRef::void_op(pos)
    }

    /// A minimal FailDescr implementation for testing.
    #[derive(Debug)]
    struct TestFailDescr {
        index: u32,
    }

    impl majit_ir::Descr for TestFailDescr {
        fn index(&self) -> u32 {
            self.index
        }
        fn as_fail_descr(&self) -> Option<&dyn FailDescr> {
            Some(self)
        }
    }

    impl FailDescr for TestFailDescr {
        fn fail_index(&self) -> u32 {
            self.index
        }
        fn fail_arg_types(&self) -> &[Type] {
            &[]
        }
    }

    fn make_fail_descr(index: u32) -> DescrRef {
        Arc::new(TestFailDescr { index })
    }

    /// `cut(pos)` truncates the recorded ops back to the saved position.
    #[test]
    fn cut_truncates_recorded_ops() {
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let saved = rec.get_position();
        let _add1 = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        let _add2 = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        assert_eq!(rec.num_ops(), 2);
        rec.cut(saved);
        assert_eq!(rec.num_ops(), 0);
        assert_eq!(rec.recorded_ops_total(), 2);
        assert_eq!(rec.num_inputargs(), 1);
    }

    #[test]
    fn box_for_operand_is_deterministic_per_position() {
        // The canonical Operand bridge must be stable per recorded position: the
        // same OpRef always binds to the same producer Rc, so a consumer can key
        // by box identity (Operand::eq = Rc::ptr_eq) instead of flat OpRef and
        // still get the "same value -> same key" behaviour, matching the
        // box-object-keyed dicts in heapcache.py / registers_r in pyjitpl.py.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_op(OpCode::IntAdd, &[i0, i0]);

        // ResOp position: two calls resolve to the SAME producer box.
        let a = rec.box_for_operand(i1);
        let b = rec.box_for_operand(i1);
        assert_eq!(a, b, "same position must yield ptr_eq-equal Operands");
        assert_eq!(a.to_opref(), i1, "Operand round-trips to its OpRef");

        // InputArg position: same invariant.
        let ia_a = rec.box_for_operand(i0);
        let ia_b = rec.box_for_operand(i0);
        assert_eq!(ia_a, ia_b);
        assert_eq!(ia_a.to_opref(), i0);

        // Distinct positions are distinct boxes.
        assert_ne!(rec.box_for_operand(i0), rec.box_for_operand(i1));
    }

    /// `opencoder.py Trace._cached_const_ptr`: repeated non-null pointers use
    /// one box, and a moving collection re-keys the address dictionary.
    #[test]
    fn const_ptr_operands_share_the_trace_ref_pool_and_rekey_after_gc() {
        let mut rec = Trace::new();
        let old = GcRef(0x1000);
        let first = rec.box_for_operand(OpRef::const_ptr(old));
        let second = rec.box_for_operand(OpRef::const_ptr(old));
        assert_eq!(first, second, "same address must reuse one ConstPtr box");

        rec.walk_const_ptr_refs(&mut |gcref| gcref.0 += 0x1000);
        let moved = rec.box_for_operand(OpRef::const_ptr(GcRef(0x2000)));
        assert_eq!(first, moved, "moved address must resolve to the same box");
        assert_eq!(moved.const_value(), Some(Value::Ref(GcRef(0x2000))));

        let stale = rec.box_for_operand(OpRef::const_ptr(old));
        assert_ne!(stale, moved, "the pre-move key must no longer be cached");
    }

    #[test]
    fn ref_pool_gc_visits_each_box_once_and_rekeys_overlapping_addresses() {
        let mut rec = Trace::new();
        let boxes: Vec<_> = (1..=4)
            .map(|i| rec.box_for_operand(OpRef::const_ptr(GcRef(i * 0x1000))))
            .collect();
        let constants: Vec<_> = boxes.iter().map(Operand::to_opref).collect();
        for &constant in &constants {
            rec.record_op(OpCode::SameAsR, &[constant]);
        }
        rec.record_guard_with_fail_args(
            OpCode::GuardTrue,
            &[OpRef::const_int(1)],
            None,
            &constants,
        );
        let mut visited = Vec::new();
        rec.walk_const_ptr_refs(&mut |reference| {
            visited.push(reference.0);
            reference.0 += 0x1000;
        });
        visited.sort_unstable();
        assert_eq!(visited, vec![0x1000, 0x2000, 0x3000, 0x4000]);
        assert_eq!(rec.const_ptrs.len(), 4);
        for (index, original) in boxes.iter().enumerate() {
            let address = GcRef((index + 2) * 0x1000);
            assert_eq!(original.const_value(), Some(Value::Ref(address)));
            assert_eq!(*original, rec.box_for_operand(OpRef::const_ptr(address)));
        }
    }

    #[test]
    fn test_record_simple_loop() {
        // Trace: i0 -> i1 = int_add(i0, i0) -> jump(i1)
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        assert_eq!(i0, iarg(0));

        let i1 = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        assert_eq!(i1, iop(1));

        rec.close_loop(&[i1]);

        let trace = rec.get_trace();
        assert!(trace.is_loop());
        assert_eq!(trace.num_inputargs(), 1);
        assert_eq!(trace.num_ops(), 2); // IntAdd + Jump
        assert_eq!(trace.ops[0].opcode, OpCode::IntAdd);
        assert_eq!(trace.ops[1].opcode, OpCode::Jump);
        assert_eq!(trace.ops[1].arg(0).to_opref(), i1);
    }

    #[test]
    fn test_record_with_guard() {
        // i0 -> guard_true(i0) -> i1 = int_add(i0, i0) -> jump(i1)
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let descr = make_fail_descr(0);
        let _g = rec.record_guard(OpCode::GuardTrue, &[i0], Some(descr));

        let i1 = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        rec.close_loop(&[i1]);

        let trace = rec.get_trace();
        assert_eq!(trace.num_ops(), 3); // GuardTrue + IntAdd + Jump
        assert!(trace.ops[0].opcode.is_guard());
        assert!(trace.ops[0].has_descr());
    }

    #[test]
    fn test_record_finish() {
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_op(OpCode::IntAdd, &[i0, i0]);

        let descr = make_fail_descr(99);
        rec.finish(&[i1], descr);

        let trace = rec.get_trace();
        assert!(trace.is_finished());
        assert!(!trace.is_loop());
    }

    #[test]
    fn test_record_multiple_inputargs() {
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let r0 = rec.record_input_arg(Type::Ref);
        let f0 = rec.record_input_arg(Type::Float);
        assert_eq!(i0, iarg(0));
        assert_eq!(r0, rarg(1));
        assert_eq!(f0, farg(2));
        assert_eq!(rec.num_inputargs(), 3);

        let i1 = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        rec.close_loop(&[i1, r0, f0]);

        let trace = rec.get_trace();
        assert_eq!(trace.num_inputargs(), 3);
        assert_eq!(trace.inputargs[0].tp, Type::Int);
        assert_eq!(trace.inputargs[1].tp, Type::Ref);
        assert_eq!(trace.inputargs[2].tp, Type::Float);
    }

    #[test]
    fn test_opref_assignment() {
        // OpRef indices are monotonically increasing, with input args taking
        // the first slots.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);
        assert_eq!(i0, iarg(0));
        assert_eq!(i1, iarg(1));

        let i2 = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        assert_eq!(i2, iop(2));

        let descr = make_fail_descr(0);
        let g0 = rec.record_guard(OpCode::GuardTrue, &[i2], Some(descr));
        assert_eq!(g0, vop(3));

        let i3 = rec.record_op(OpCode::IntSub, &[i2, i0]);
        assert_eq!(i3, iop(4));

        rec.close_loop(&[i3, i1]);
        let trace = rec.get_trace();

        // The op's .pos should match
        assert_eq!(trace.ops[0].pos.get(), iop(2)); // IntAdd
        assert_eq!(trace.ops[1].pos.get(), vop(3)); // GuardTrue
        assert_eq!(trace.ops[2].pos.get(), iop(4)); // IntSub
        assert_eq!(trace.ops[3].pos.get(), vop(5)); // Jump
    }

    #[test]
    fn byte_buffer_materialize_keeps_unique_positions() {
        // history.py record + opencoder.py get_iter: bytes during
        // record, ResOp objects only at iterate. Unique OpRefs (void
        // and non-void) must survive materialize so snapshots still match.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);
        rec.attach_byte_buffer(std::sync::Arc::new(crate::MetaInterpStaticData::new()));
        let i2 = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        let g0 = rec.record_guard(OpCode::GuardTrue, &[i2], None);
        rec.set_last_op_resume_position(7);
        rec.close_loop(&[i2, i1]);
        assert_eq!(rec.num_ops(), 3);
        let (inputs, ops) = rec.into_parts();
        assert_eq!(inputs.len(), 2);
        assert_eq!(ops.len(), 3);
        assert_eq!(ops[0].opcode, OpCode::IntAdd);
        assert_eq!(ops[0].pos.get(), i2);
        assert_eq!(ops[1].opcode, OpCode::GuardTrue);
        assert_eq!(ops[1].pos.get(), g0);
        assert_eq!(ops[1].rd_resume_position.get(), 7);
        assert_eq!(ops[2].opcode, OpCode::Jump);
    }

    #[test]
    #[should_panic(expected = "use record_guard")]
    fn test_record_op_with_guard_opcode() {
        let mut rec = Trace::new();
        rec.record_input_arg(Type::Int);
        // Should panic: guard opcodes must use record_guard
        rec.record_op(OpCode::GuardTrue, &[iarg(0)]);
    }

    #[test]
    fn test_num_ops_counts_non_inputargs() {
        // history.py length() = trace._count - len(inputargs).
        // In pyre that's ops.len() since inputargs aren't stored in ops.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        assert_eq!(rec.num_ops(), 0);

        rec.record_op(OpCode::IntAdd, &[i0, i0]);
        assert_eq!(rec.num_ops(), 1);
    }

    #[test]
    fn test_record_op_with_descr() {
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let descr = make_fail_descr(42);
        let i1 = rec.record_op_with_descr(OpCode::CallI, &[i0], descr);
        assert_eq!(i1, iop(1));

        // Verify the op has a descriptor
        assert!(rec.ops[0].has_descr());
    }

    #[test]
    fn test_complex_trace() {
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let i2 = rec.record_op(OpCode::IntLt, &[i0, i1]);

        let descr = make_fail_descr(0);
        rec.record_guard(OpCode::GuardTrue, &[i2], Some(descr));

        let i3 = rec.record_op(OpCode::IntAdd, &[i0, i1]);

        rec.close_loop(&[i3, i1]);

        let trace = rec.get_trace();
        assert!(trace.is_loop());
        assert_eq!(trace.num_ops(), 4);

        let guards: Vec<_> = trace.iter_guards().collect();
        assert_eq!(guards.len(), 1);
        assert_eq!(guards[0].opcode, OpCode::GuardTrue);
    }

    // Opencoder parity tests
    // Ported from rpython/jit/metainterp/test/test_opencoder.py

    #[test]
    fn test_simple_iterator() {
        // Parity: test_simple_iterator
        // Record two INT_ADD ops and verify trace structure matches.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let add0 = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        let add1 = rec.record_op(OpCode::IntAdd, &[add0, i0]);

        rec.close_loop(&[add1, i1]);
        let trace = rec.get_trace();

        // Verify the trace has the correct number of ops (2 + Jump).
        assert_eq!(trace.num_ops(), 3);
        assert_eq!(trace.ops[0].opcode, OpCode::IntAdd);
        assert_eq!(trace.ops[1].opcode, OpCode::IntAdd);
        assert_eq!(trace.ops[2].opcode, OpCode::Jump);

        // First add uses input args i0, i1.
        assert_eq!(trace.ops[0].arg(0).to_opref(), i0);
        assert_eq!(trace.ops[0].arg(1).to_opref(), i1);

        // Second add references the result of first add and i0.
        assert_eq!(trace.ops[1].arg(0).to_opref(), add0);
        assert_eq!(trace.ops[1].arg(1).to_opref(), i0);
    }

    #[test]
    fn test_inputargs_preserved() {
        // Parity: Trace([i0, i1], ...) preserves input args.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);
        assert_eq!(rec.num_inputargs(), 2);

        rec.record_op(OpCode::IntAdd, &[i0, i1]);
        let add = rec.record_op(OpCode::IntAdd, &[iop(2), i1]);
        rec.close_loop(&[add, i1]);

        let trace = rec.get_trace();
        assert_eq!(trace.num_inputargs(), 2);
        assert_eq!(trace.inputargs[0].tp, Type::Int);
        assert_eq!(trace.inputargs[1].tp, Type::Int);
    }

    #[test]
    fn test_op_references_chain() {
        // Parity: ops that reference previous ops form correct chains.
        // i0 -> add = int_add(i0, i0) -> sub = int_sub(add, i0) -> jump(sub)
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let add = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        let sub = rec.record_op(OpCode::IntSub, &[add, i0]);

        rec.close_loop(&[sub]);
        let trace = rec.get_trace();

        assert_eq!(trace.ops[0].opcode, OpCode::IntAdd);
        assert_eq!(trace.ops[0].pos.get(), iop(1)); // after 1 inputarg
        assert_eq!(trace.ops[1].opcode, OpCode::IntSub);
        assert_eq!(trace.ops[1].arg(0).to_opref(), add); // references the add result
        assert_eq!(trace.ops[1].arg(1).to_opref(), i0); // references the input arg
    }

    #[test]
    fn test_guard_with_fail_args() {
        // Parity: guards can carry fail_args describing live values at guard.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        let descr = make_fail_descr(0);
        let guard =
            rec.record_guard_with_fail_args(OpCode::GuardTrue, &[add], Some(descr), &[i0, i1, add]);

        let sub = rec.record_op(OpCode::IntSub, &[add, i0]);
        rec.close_loop(&[sub, i1]);

        let trace = rec.get_trace();
        // Find the guard op.
        let guard_op = &trace.ops[1]; // after IntAdd
        assert_eq!(guard_op.pos.get(), guard);
        assert!(guard_op.opcode.is_guard());
        let fail_args = guard_op.getfailargs().unwrap();
        assert_eq!(fail_args.len(), 3);
        assert_eq!(fail_args[0].to_opref(), i0);
        assert_eq!(fail_args[1].to_opref(), i1);
        assert_eq!(fail_args[2].to_opref(), add);
    }

    #[test]
    fn test_multiple_guards() {
        // Parity: multiple guards in one trace.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let descr0 = make_fail_descr(0);
        rec.record_guard_with_fail_args(OpCode::GuardTrue, &[i0], Some(descr0), &[i0, i1]);

        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);

        let descr1 = make_fail_descr(1);
        rec.record_guard_with_fail_args(OpCode::GuardFalse, &[add], Some(descr1), &[i0, add]);

        let sub = rec.record_op(OpCode::IntSub, &[add, i0]);
        rec.close_loop(&[sub, i1]);

        let trace = rec.get_trace();
        let guards: Vec<_> = trace.iter_guards().collect();
        assert_eq!(guards.len(), 2);
        assert_eq!(guards[0].opcode, OpCode::GuardTrue);
        assert_eq!(guards[1].opcode, OpCode::GuardFalse);

        // First guard's fail_args
        let fa0 = guards[0].getfailargs().unwrap();
        assert_eq!(fa0.len(), 2);
        assert_eq!(fa0[0].to_opref(), i0);
        assert_eq!(fa0[1].to_opref(), i1);

        // Second guard's fail_args
        let fa1 = guards[1].getfailargs().unwrap();
        assert_eq!(fa1.len(), 2);
        assert_eq!(fa1[0].to_opref(), i0);
        assert_eq!(fa1[1].to_opref(), add);
    }

    #[test]
    fn test_close_loop_jump_targets_inputargs() {
        // Parity: close_loop produces a JUMP whose args correspond to inputargs.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        rec.close_loop(&[add, i1]);

        let trace = rec.get_trace();
        let jump = trace.ops.last().unwrap();
        assert_eq!(jump.opcode, OpCode::Jump);
        assert_eq!(jump.num_args(), trace.num_inputargs());
        assert_eq!(jump.arg(0).to_opref(), add);
        assert_eq!(jump.arg(1).to_opref(), i1);
    }

    #[test]
    fn test_finish_produces_finish_op() {
        // Parity: finish() produces a FINISH op with the given args.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let add = rec.record_op(OpCode::IntAdd, &[i0, i0]);

        let descr = make_fail_descr(42);
        rec.finish(&[add], descr);

        let trace = rec.get_trace();
        assert!(trace.is_finished());
        assert!(!trace.is_loop());

        let finish_op = trace.ops.last().unwrap();
        assert_eq!(finish_op.opcode, OpCode::Finish);
        assert_eq!(finish_op.num_args(), 1);
        assert_eq!(finish_op.arg(0).to_opref(), add);
        assert!(finish_op.has_descr());
    }

    #[test]
    fn test_trace_length_tracking() {
        // Parity: num_ops() tracks the count accurately.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        assert_eq!(rec.num_ops(), 0);

        rec.record_op(OpCode::IntAdd, &[i0, i0]);
        assert_eq!(rec.num_ops(), 1);

        rec.record_op(OpCode::IntSub, &[iop(1), i0]);
        assert_eq!(rec.num_ops(), 2);

        let descr = make_fail_descr(0);
        rec.record_guard(OpCode::GuardTrue, &[iop(2)], Some(descr));
        assert_eq!(rec.num_ops(), 3);

        rec.close_loop(&[iop(2)]);
        // After close_loop, Jump is added.
        assert_eq!(rec.num_ops(), 4);
    }

    #[test]
    fn test_trace_position_splits_count_and_index_for_void_ops() {
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        let before_guard = rec.get_position();
        assert_eq!(before_guard._count, 2);
        assert_eq!(before_guard._index, 2);

        let descr = make_fail_descr(0);
        rec.record_guard(OpCode::GuardTrue, &[i1], Some(descr));
        rec.close_loop(&[i1]);
        rec.finish(&[i1], make_fail_descr(1));

        let pos = rec.get_position();
        assert_eq!(pos._count, 5);
        assert_eq!(pos._index, 2);
    }

    #[test]
    fn test_with_num_inputs_creates_inputargs() {
        // Parity: Trace::with_num_inputs pre-creates input args.
        let rec = Trace::with_num_inputs(3);
        assert_eq!(rec.num_inputargs(), 3);
        assert_eq!(rec.num_ops(), 0);
    }

    #[test]
    fn guard_failure_history_drops_holes_but_keeps_frontend_positions() {
        let mut rec =
            Trace::with_input_layout(&[Type::Int, Type::Ref, Type::Int], &[true, false, true]);
        let result = rec.record_op(OpCode::IntAdd, &[iarg(0), iarg(2)]);
        assert_eq!(
            result.raw(),
            3,
            "the hole remains reserved in trace positions"
        );

        let live = rec.live_inputargs_cloned();
        assert_eq!(
            live.iter().map(InputArg::opref).collect::<Vec<_>>(),
            vec![OpRef::input_arg_int(0), OpRef::input_arg_int(2)]
        );

        let (inputargs, ops) = rec.into_parts();
        assert_eq!(inputargs.len(), 2);
        assert_eq!(inputargs[0].opref(), iarg(0));
        assert_eq!(inputargs[1].opref(), iarg(2));
        assert_eq!(ops[0].arg(1).to_opref(), iarg(2));
    }

    #[test]
    fn test_with_num_inputs_oprefs() {
        // Input args from with_num_inputs get InputArgInt(0), InputArgInt(1), ...
        let mut rec = Trace::with_num_inputs(3);

        // The input args consumed positions 0..2, so next op gets BoxInt(3).
        let add = rec.record_op(OpCode::IntAdd, &[iarg(0), iarg(1)]);
        assert_eq!(add, iop(3));

        rec.close_loop(&[iarg(0), iarg(1), iarg(2)]);
        let trace = rec.get_trace();
        assert_eq!(trace.num_inputargs(), 3);
        assert_eq!(trace.ops[0].pos.get(), iop(3));
    }

    #[test]
    fn test_guard_descr_preserved() {
        // Parity: guard descriptors are preserved through get_trace().
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let descr = make_fail_descr(77);
        rec.record_guard(OpCode::GuardNoException, &[], Some(descr));

        rec.close_loop(&[i0]);
        let trace = rec.get_trace();
        let guard = &trace.ops[0];
        assert!(guard.has_descr());
        let d = guard.getdescr().unwrap();
        assert_eq!(d.index(), 77);
    }

    #[test]
    fn test_op_with_descr_preserved() {
        // Parity: op descriptors (e.g., for calls) are preserved.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let call_descr = make_fail_descr(55);
        let result = rec.record_op_with_descr(OpCode::CallI, &[i0], call_descr);

        rec.close_loop(&[result]);
        let trace = rec.get_trace();
        let call_op = &trace.ops[0];
        assert_eq!(call_op.opcode, OpCode::CallI);
        assert!(call_op.has_descr());
        assert_eq!(call_op.getdescr().unwrap().index(), 55);
    }

    #[test]
    fn test_empty_fail_args() {
        // Parity: guard with empty fail_args is valid.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let descr = make_fail_descr(0);
        rec.record_guard_with_fail_args(OpCode::GuardTrue, &[i0], Some(descr), &[]);

        rec.close_loop(&[i0]);
        let trace = rec.get_trace();
        let guard = &trace.ops[0];
        let fail_args = guard.getfailargs().unwrap();
        assert!(fail_args.is_empty());
    }

    #[test]
    #[should_panic(expected = "opcode")]
    fn test_record_guard_with_non_guard_opcode() {
        // Parity: record_guard rejects non-guard opcodes.
        let mut rec = Trace::new();
        rec.record_input_arg(Type::Int);
        let descr = make_fail_descr(0);
        rec.record_guard(OpCode::IntAdd, &[iarg(0)], Some(descr));
    }

    #[test]
    #[should_panic(expected = "input args must be registered before any operations")]
    fn test_inputarg_after_ops() {
        // Parity: input args must come before any operations.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        rec.record_op(OpCode::IntAdd, &[i0, i0]);
        // This should panic.
        rec.record_input_arg(Type::Int);
    }

    // Opencoder breadth tests — deeper parity with test_opencoder.py

    #[test]
    fn test_recorder_const_int_via_constant_oprefs() {
        // Constants live in a dedicated pool and keep stable OpRefs.
        // Recording ops that reference constants should preserve them.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        // Simulate a pooled constant reference.
        let const_ref = OpRef::const_int(0);
        let add = rec.record_op(OpCode::IntAdd, &[i0, const_ref]);

        rec.close_loop(&[add]);
        let trace = rec.get_trace();

        assert_eq!(trace.ops[0].arg(1).to_opref(), const_ref);
        assert!(trace.ops[0].arg(1).is_constant());
    }

    #[test]
    fn test_recorder_const_deduplication_by_opref() {
        // Two ops referencing the same constant OpRef share the same value.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let const_ref = OpRef::const_int(1);
        let add1 = rec.record_op(OpCode::IntAdd, &[i0, const_ref]);
        let add2 = rec.record_op(OpCode::IntAdd, &[add1, const_ref]);

        rec.close_loop(&[add2]);
        let trace = rec.get_trace();

        // Both ops reference the same constant
        assert_eq!(
            trace.ops[0].arg(1).to_opref(),
            trace.ops[1].arg(1).to_opref()
        );
        assert_eq!(trace.ops[0].arg(1).to_opref(), const_ref);
    }

    #[test]
    fn test_recorder_descriptors_preserved_on_ops() {
        // Descriptors on non-guard ops should survive through get_trace().
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let descr1 = make_fail_descr(10);
        let descr2 = make_fail_descr(20);

        let call1 = rec.record_op_with_descr(OpCode::CallI, &[i0], descr1);
        let call2 = rec.record_op_with_descr(OpCode::CallI, &[i1], descr2);

        rec.close_loop(&[call1, call2]);
        let trace = rec.get_trace();

        assert_eq!(trace.ops[0].getdescr().unwrap().index(), 10);
        assert_eq!(trace.ops[1].getdescr().unwrap().index(), 20);
    }

    #[test]
    fn test_recorder_100_plus_ops_stress() {
        // Recording 200 ops should work without issue.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let mut prev = i0;
        for _ in 0..200 {
            prev = rec.record_op(OpCode::IntAdd, &[prev, i0]);
        }
        assert_eq!(rec.num_ops(), 200);

        rec.close_loop(&[prev]);
        let trace = rec.get_trace();
        // 200 IntAdd + 1 Jump
        assert_eq!(trace.num_ops(), 201);
        assert!(trace.is_loop());

        // Verify chain: each op references the previous op's result.
        // i0 = input_arg_int(0), first IntAdd = int_op(1), second = int_op(2), ...
        for (i, op) in trace.ops[..200].iter().enumerate() {
            assert_eq!(op.opcode, OpCode::IntAdd);
            if i > 0 {
                // The previous IntAdd produced BoxInt(i) (offset by inputarg).
                assert_eq!(op.arg(0).to_opref(), iop(i as u32));
            }
        }
    }

    #[test]
    fn test_recorder_mixed_type_input_args() {
        // Mixed-type inputs (Int, Float, Ref) produce correct types.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let f0 = rec.record_input_arg(Type::Float);
        let r0 = rec.record_input_arg(Type::Ref);
        let i1 = rec.record_input_arg(Type::Int);

        assert_eq!(i0, iarg(0));
        assert_eq!(f0, farg(1));
        assert_eq!(r0, rarg(2));
        assert_eq!(i1, iarg(3));

        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        rec.close_loop(&[add, f0, r0, i1]);

        let trace = rec.get_trace();
        assert_eq!(trace.inputargs[0].tp, Type::Int);
        assert_eq!(trace.inputargs[1].tp, Type::Float);
        assert_eq!(trace.inputargs[2].tp, Type::Ref);
        assert_eq!(trace.inputargs[3].tp, Type::Int);
    }

    #[test]
    fn test_recorder_guard_with_many_fail_args() {
        // Guard with 10+ fail_args.
        let mut rec = Trace::new();
        let mut inputs = Vec::new();
        for _ in 0..12 {
            inputs.push(rec.record_input_arg(Type::Int));
        }

        // Record some ops
        let add = rec.record_op(OpCode::IntAdd, &[inputs[0], inputs[1]]);

        // Guard with all 12 inputs + 1 computed value as fail_args (13 total)
        let mut fail_args: Vec<OpRef> = inputs.clone();
        fail_args.push(add);

        let descr = make_fail_descr(0);
        let guard =
            rec.record_guard_with_fail_args(OpCode::GuardTrue, &[add], Some(descr), &fail_args);

        rec.close_loop(&inputs);
        let trace = rec.get_trace();

        // Find the guard
        let guard_op = trace.iter_guards().next().unwrap();
        assert_eq!(guard_op.pos.get(), guard);
        let fa = guard_op.getfailargs().unwrap();
        assert_eq!(fa.len(), 13);

        // Verify all fail_args match what we specified
        for (i, &expected) in fail_args.iter().enumerate() {
            assert_eq!(fa[i].to_opref(), expected);
        }
    }

    #[test]
    fn test_recorder_guard_descr_index_survives() {
        // Each guard's descriptor index must survive through get_trace().
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);

        let d0 = make_fail_descr(100);
        let d1 = make_fail_descr(200);
        let d2 = make_fail_descr(300);

        rec.record_guard(OpCode::GuardTrue, &[i0], Some(d0));
        rec.record_guard(OpCode::GuardFalse, &[i0], Some(d1));
        rec.record_guard(OpCode::GuardNoException, &[], Some(d2));

        rec.close_loop(&[i0]);
        let trace = rec.get_trace();

        let guards: Vec<_> = trace.iter_guards().collect();
        assert_eq!(guards.len(), 3);
        assert_eq!(guards[0].getdescr().unwrap().index(), 100);
        assert_eq!(guards[1].getdescr().unwrap().index(), 200);
        assert_eq!(guards[2].getdescr().unwrap().index(), 300);
    }

    #[test]
    fn test_recorder_ops_interleaved_with_guards() {
        // Interleaved ops and guards: verify ordering is correct.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        let add = rec.record_op(OpCode::IntAdd, &[i0, i1]);
        let d0 = make_fail_descr(0);
        rec.record_guard(OpCode::GuardTrue, &[add], Some(d0));
        let sub = rec.record_op(OpCode::IntSub, &[add, i0]);
        let d1 = make_fail_descr(1);
        rec.record_guard(OpCode::GuardFalse, &[sub], Some(d1));
        let mul = rec.record_op(OpCode::IntMul, &[sub, i1]);

        rec.close_loop(&[mul, i1]);
        let trace = rec.get_trace();

        let expected = vec![
            OpCode::IntAdd,
            OpCode::GuardTrue,
            OpCode::IntSub,
            OpCode::GuardFalse,
            OpCode::IntMul,
            OpCode::Jump,
        ];
        let actual: Vec<_> = trace.iter_ops().map(|op| op.opcode).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn test_recorder_finish_with_descr() {
        // finish() records a FINISH op with the given descriptor.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let add = rec.record_op(OpCode::IntAdd, &[i0, i0]);

        let descr = make_fail_descr(999);
        rec.finish(&[add], descr);

        let trace = rec.get_trace();
        assert!(trace.is_finished());

        let finish = trace.ops.last().unwrap();
        assert_eq!(finish.opcode, OpCode::Finish);
        assert_eq!(finish.getdescr().unwrap().index(), 999);
    }

    #[test]
    fn test_recorder_no_ops_just_inputargs_and_jump() {
        // Minimal trace: input args directly jump back.
        let mut rec = Trace::new();
        let i0 = rec.record_input_arg(Type::Int);
        let i1 = rec.record_input_arg(Type::Int);

        rec.close_loop(&[i0, i1]);
        let trace = rec.get_trace();

        assert_eq!(trace.num_ops(), 1); // Only Jump
        assert_eq!(trace.ops[0].opcode, OpCode::Jump);
        assert_eq!(trace.ops[0].arg(0).to_opref(), i0);
        assert_eq!(trace.ops[0].arg(1).to_opref(), i1);
    }

    #[test]
    fn test_get_position_and_cut() {
        let mut rec = Trace::with_num_inputs(2);
        let pos0 = rec.get_position();
        assert_eq!(pos0._pos, 0);
        assert_eq!(pos0._count, 2); // 2 inputargs
        assert_eq!(pos0._index, 2);
        assert_eq!(pos0.snapshot_data_len, 0);
        assert_eq!(pos0.snapshot_array_data_len, 0);

        let _a = rec.record_op(OpCode::IntAdd, &[iarg(0), iarg(1)]);
        let pos1 = rec.get_position();
        assert_eq!(pos1._pos, 1);
        assert_eq!(pos1._count, 3);
        assert_eq!(pos1._index, 3);

        let _b = rec.record_op(OpCode::IntSub, &[iarg(0), iarg(1)]);
        let _c = rec.record_op(OpCode::IntMul, &[iarg(0), iarg(1)]);
        assert_eq!(rec.num_ops(), 3);

        // Cut back to pos1 — should discard IntSub and IntMul
        rec.cut(pos1);
        assert_eq!(rec.num_ops(), 1);
        assert_eq!(rec.get_position(), pos1);

        // Can record more ops after cut
        let d = rec.record_op(OpCode::IntNeg, &[iarg(0)]);
        assert_eq!(d, iop(3)); // continues from pos1._count
        assert_eq!(rec.num_ops(), 2);

        // Cut back to pos0 — should discard everything
        rec.cut(pos0);
        assert_eq!(rec.num_ops(), 0);
    }

    #[test]
    fn test_get_position_tracks_count_and_index_separately() {
        let mut rec = Trace::with_num_inputs(1);
        let i0 = iarg(0);

        let _add = rec.record_op(OpCode::IntAdd, &[i0, i0]);
        let pos_after_add = rec.get_position();
        assert_eq!(pos_after_add._count, 2);
        assert_eq!(pos_after_add._index, 2);

        let descr = make_fail_descr(1);
        rec.record_guard(OpCode::GuardTrue, &[iop(1)], Some(descr));
        let pos_after_guard = rec.get_position();
        assert_eq!(pos_after_guard._pos, 2);
        assert_eq!(pos_after_guard._count, 3);
        assert_eq!(pos_after_guard._index, 2);
    }
}
