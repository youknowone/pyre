/// MIFrame and MIFrameStack — execution frames for jitcode interpretation.
///
/// RPython pyjitpl.py: class MIFrame holds jitcode reference, PC,
/// and three register arrays (int/ref/float). MIFrameStack manages
/// the call stack of nested inline calls.
use std::sync::Arc;

use majit_ir::OpRef;

use crate::jitcode::{JitArgKind, JitCode, read_u8, read_u16};
use crate::opencoder::{Box as OpBox, TraceRecordBuffer};

/// Map an int register (OpRef, concrete value) to an `OpBox`.
/// Constant OpRefs materialize as `ConstInt(value)`; real trace slots
/// materialize as `ResOp(opref.0)`. Mirrors RPython's implicit
/// isinstance dispatch in `_encode(box)` when the caller passes
/// `self.registers_i[index]` (which is already a `ConstInt` /
/// `InputArg` / `AbstractResOp`).
#[inline]
fn register_to_box_int(opref: OpRef, value: i64) -> OpBox {
    if opref.is_constant() {
        OpBox::ConstInt(value)
    } else {
        OpBox::ResOp(opref.0)
    }
}

/// Map a ref register (OpRef, concrete address) to an `OpBox`.
/// `value` stores the raw address of the GC ref (cast to i64 at write
/// time).
#[inline]
fn register_to_box_ref(opref: OpRef, value: i64) -> OpBox {
    if opref.is_constant() {
        OpBox::ConstPtr(value as u64)
    } else {
        OpBox::ResOp(opref.0)
    }
}

/// Map a float register (OpRef, raw bits) to an `OpBox`.
/// `value` stores the bit-casted `f64` payload.
#[inline]
fn register_to_box_float(opref: OpRef, value: i64) -> OpBox {
    if opref.is_constant() {
        OpBox::ConstFloat(value as u64)
    } else {
        OpBox::ResOp(opref.0)
    }
}

/// A single execution frame for jitcode bytecode.
///
/// RPython pyjitpl.py: class MIFrame
///
/// PRE-EXISTING-ADAPTATION: RPython holds `self.metainterp` as a
/// back-pointer to the owning MetaInterp; pyre takes `&mut MetaInterp`
/// as a parameter on call sites instead so that the borrow checker
/// allows `MetaInterp::framestack` to own a `Vec<MIFrame>` without
/// self-referential aliasing.  `jitcode` is held as `Arc<JitCode>` so
/// the frame can outlive any single function-scope borrow — this
/// matches the upstream model where frames live as long as
/// `MetaInterp.framestack` keeps them alive.
pub struct MIFrame {
    pub jitcode: Arc<JitCode>,
    pub pc: usize,
    pub code_cursor: usize,
    pub int_regs: Vec<Option<OpRef>>,
    pub int_values: Vec<Option<i64>>,
    pub ref_regs: Vec<Option<OpRef>>,
    pub ref_values: Vec<Option<i64>>,
    pub float_regs: Vec<Option<OpRef>>,
    pub float_values: Vec<Option<i64>>,
    pub inline_frame: bool,
    pub return_i: Option<(usize, usize)>,
    pub return_r: Option<(usize, usize)>,
    pub return_f: Option<(usize, usize)>,
    /// pyjitpl.py `MIFrame.greenkey` — set when this frame is a
    /// recursive portal call (pyjitpl.py:80).
    pub greenkey: Option<u64>,
    /// pyjitpl.py:91 `self._result_argcode = 'v'`.
    ///
    /// Single-byte argcode of the *previous* opimpl's result type
    /// (`b'i'` / `b'r'` / `b'f'` / `b'v'`).  Updated by call-recording
    /// opimpls before they advance the pc; consulted by `_try_tco`
    /// (pyjitpl.py:1281) to decide whether the call's result type
    /// matches a following `*_return` opcode.  Initialized to `b'v'`
    /// because a fresh frame's "previous opimpl" is the implicit
    /// frame setup (returns void).
    pub _result_argcode: u8,
    /// pyjitpl.py `MIFrame.pushed_box` — the box that the previous
    /// `_opimpl_any_push` instruction parked on the frame.  Reset to
    /// `None` by `cleanup_registers` so a recycled frame does not keep
    /// the parked box alive.
    pub pushed_box: Option<OpRef>,
    /// pyjitpl.py:93 `self.parent_snapshot = -1`.
    ///
    /// Set by `TraceRecordBuffer::_ensure_parent_resumedata` to the
    /// snapshot_index returned by the paired `create_snapshot(back,
    /// is_last)` call — so that a later `capture_resumedata` whose
    /// framestack shares this frame as an ancestor can short-circuit
    /// the parent chain walk by patching with `snapshot_add_prev(
    /// target.parent_snapshot)` instead of re-emitting the snapshot.
    pub parent_snapshot: i64,
}

impl MIFrame {
    pub fn new(jitcode: Arc<JitCode>, pc: usize) -> Self {
        let num_regs_i = jitcode.c_num_regs_i as usize;
        let num_regs_r = jitcode.c_num_regs_r as usize;
        let num_regs_f = jitcode.c_num_regs_f as usize;
        Self {
            jitcode,
            pc,
            code_cursor: 0,
            int_regs: vec![None; num_regs_i],
            int_values: vec![None; num_regs_i],
            ref_regs: vec![None; num_regs_r],
            ref_values: vec![None; num_regs_r],
            float_regs: vec![None; num_regs_f],
            float_values: vec![None; num_regs_f],
            inline_frame: false,
            return_i: None,
            return_r: None,
            return_f: None,
            greenkey: None,
            _result_argcode: b'v',
            pushed_box: None,
            parent_snapshot: -1,
        }
    }

    pub fn next_u8(&mut self) -> u8 {
        read_u8(&self.jitcode.code, &mut self.code_cursor)
    }

    pub fn next_u16(&mut self) -> u16 {
        read_u16(&self.jitcode.code, &mut self.code_cursor)
    }

    pub fn finished(&self) -> bool {
        self.code_cursor >= self.jitcode.code.len()
    }

    /// pyjitpl.py:121-127 `MIFrame.cleanup_registers()`.
    ///
    /// ```python
    /// def cleanup_registers(self):
    ///     for i in range(self.jitcode.num_regs_r()):
    ///         self.registers_r[i] = None
    ///     self.pushed_box = None
    /// ```
    ///
    /// Iterates `0..num_regs_r()` (RPython skips the constants area
    /// that lives past `num_regs_r`); pyre's `ref_regs` is sized
    /// exactly to `num_regs_r` so the loop scans the same slots.
    /// `ref_values` is cleared in lockstep — it is the pyre-only
    /// concrete-value mirror that lives next to each box.
    pub fn cleanup_registers(&mut self) {
        let num_regs_r = self.jitcode.num_regs_r() as usize;
        for i in 0..num_regs_r {
            self.ref_regs[i] = None;
            self.ref_values[i] = None;
        }
        self.pushed_box = None;
    }

    /// pyjitpl.py:1878-1879 `MIFrame.setup_resume_at_op(pc)`.
    pub fn setup_resume_at_op(&mut self, pc: usize) {
        self.pc = pc;
    }

    /// pyjitpl.py:258-275 `MIFrame.make_result_of_lastop(resultbox)`.
    ///
    /// Stores the result of the last opimpl into the typed register at
    /// `target_index`. RPython reads `target_index = ord(self.bytecode[self.pc-1])`
    /// from the bytecode; pyre's call BC encodes `dst` explicitly so
    /// callers pass it directly.
    pub fn make_result_of_lastop(
        &mut self,
        kind: JitArgKind,
        target_index: usize,
        opref: OpRef,
        concrete: i64,
    ) {
        match kind {
            JitArgKind::Int => {
                self.int_regs[target_index] = Some(opref);
                self.int_values[target_index] = Some(concrete);
            }
            JitArgKind::Ref => {
                self.ref_regs[target_index] = Some(opref);
                self.ref_values[target_index] = Some(concrete);
            }
            JitArgKind::Float => {
                self.float_regs[target_index] = Some(opref);
                self.float_values[target_index] = Some(concrete);
            }
        }
    }

    /// pyjitpl.py:1862-1876 `MIFrame.setup_call(argboxes)`.
    ///
    /// Resets `pc` to 0 and copies each argbox into the first slot of
    /// its typed register bank in declaration order. RPython's
    /// `setup_call` consults `box.type`; pyre's `OpRef` does not carry
    /// type info, so the caller passes a typed `(kind, value, concrete)`
    /// tuple per arg.
    ///
    /// Also resets `parent_snapshot = -1`. In RPython this lives in
    /// `MIFrame.setup()` (pyjitpl.py:93) which always precedes
    /// `setup_call`; pyre's `MIFrame::new` already sets it, but any
    /// future frame-recycling path (upstream `free_frames_list`) would
    /// reuse an MIFrame and skip `new()`, so we reset here to match the
    /// RPython "every call-entry clears parent_snapshot" invariant.
    /// pyjitpl.py:177-234 `MIFrame.get_list_of_active_boxes`.
    ///
    /// Reads the LIVE-op liveness header preceding the current pc and
    /// pushes each live register onto the trace's snapshot-array data
    /// in int → ref → float order.  Returns the `_snapshot_array_data`
    /// offset produced by `new_array` (RPython `storage`).
    ///
    /// `op_live` and `all_liveness` are threaded through from
    /// `MetaInterpStaticData` (RPython
    /// `self.metainterp.staticdata.op_live` /
    /// `.liveness_info`) — pyre passes them explicitly so this method
    /// does not depend on `MetaInterpStaticData` structurally.
    ///
    /// `in_a_call=true` branch is not yet wired: it requires
    /// `CONST_FALSE` / `CONST_NULL` / `CONST_FZERO` OpRef seeding
    /// (pyjitpl.py:188-192) which is tracked in the ConstantPool
    /// follow-up. `create_top_snapshot` (the current capture_resumedata
    /// caller for the topmost frame) passes `in_a_call=false`, so this
    /// restriction does not block it.
    pub fn get_list_of_active_boxes(
        &mut self,
        in_a_call: bool,
        trace: &mut TraceRecordBuffer,
        op_live: u8,
        all_liveness: &[u8],
        after_residual_call: bool,
    ) -> i64 {
        const SIZE_LIVE_OP: usize = majit_translate::liveness::OFFSET_SIZE + 1;
        use majit_translate::liveness::{LivenessIterator, decode_offset};

        // pyjitpl.py:180-193 — result_argcode clear when this frame is
        // not the topmost frame.  Still unsupported; see docstring.
        assert!(
            !in_a_call,
            "MIFrame::get_list_of_active_boxes: in_a_call=true branch \
             (pyjitpl.py:180-193 result_argcode clear) needs CONST_FALSE / \
             CONST_NULL / CONST_FZERO OpRef seeding; currently only the \
             topmost-frame (in_a_call=false) path is wired."
        );

        // pyjitpl.py:194-198 — pick the pc of the preceding LIVE op.
        let pc = if in_a_call || after_residual_call {
            self.pc
        } else {
            self.pc - SIZE_LIVE_OP
        };

        // pyjitpl.py:199 `assert ord(self.jitcode.code[pc]) == op_live`.
        debug_assert_eq!(self.jitcode.code[pc], op_live);

        // pyjitpl.py:202-207 — decode offset + per-type lengths.
        let mut offset = decode_offset(&self.jitcode.code, pc + 1);
        let length_i = all_liveness[offset] as u32;
        let length_r = all_liveness[offset + 1] as u32;
        let length_f = all_liveness[offset + 2] as u32;
        offset += 3;

        // pyjitpl.py:209-214 — pre-allocate the storage array.
        let total = (length_i + length_r + length_f) as usize;
        let storage = trace.new_array(total);

        // pyjitpl.py:216-221 — push live int registers.
        if length_i > 0 {
            let mut it = LivenessIterator::new(offset, length_i, all_liveness);
            while let Some(index) = it.next() {
                let idx = index as usize;
                let opref = self.int_regs[idx]
                    .expect("get_list_of_active_boxes: int register uninitialized");
                let value = self.int_values[idx]
                    .expect("get_list_of_active_boxes: int value uninitialized");
                let b = register_to_box_int(opref, value);
                trace._add_box_to_storage_box(b);
            }
            offset = it.offset;
        }

        // pyjitpl.py:222-227 — push live ref registers.
        if length_r > 0 {
            let mut it = LivenessIterator::new(offset, length_r, all_liveness);
            while let Some(index) = it.next() {
                let idx = index as usize;
                let opref = self.ref_regs[idx]
                    .expect("get_list_of_active_boxes: ref register uninitialized");
                let value = self.ref_values[idx]
                    .expect("get_list_of_active_boxes: ref value uninitialized");
                let b = register_to_box_ref(opref, value);
                trace._add_box_to_storage_box(b);
            }
            offset = it.offset;
        }

        // pyjitpl.py:228-233 — push live float registers.
        if length_f > 0 {
            let mut it = LivenessIterator::new(offset, length_f, all_liveness);
            while let Some(index) = it.next() {
                let idx = index as usize;
                let opref = self.float_regs[idx]
                    .expect("get_list_of_active_boxes: float register uninitialized");
                let value = self.float_values[idx]
                    .expect("get_list_of_active_boxes: float value uninitialized");
                let b = register_to_box_float(opref, value);
                trace._add_box_to_storage_box(b);
            }
            let _ = it.offset; // offset no longer read after the last bank
        }

        storage
    }

    pub fn setup_call(&mut self, argboxes: &[(JitArgKind, OpRef, i64)]) {
        self.pc = 0;
        self.parent_snapshot = -1;
        let mut count_i = 0;
        let mut count_r = 0;
        let mut count_f = 0;
        for (kind, value, concrete) in argboxes {
            match kind {
                JitArgKind::Int => {
                    self.int_regs[count_i] = Some(*value);
                    self.int_values[count_i] = Some(*concrete);
                    count_i += 1;
                }
                JitArgKind::Ref => {
                    self.ref_regs[count_r] = Some(*value);
                    self.ref_values[count_r] = Some(*concrete);
                    count_r += 1;
                }
                JitArgKind::Float => {
                    self.float_regs[count_f] = Some(*value);
                    self.float_values[count_f] = Some(*concrete);
                    count_f += 1;
                }
            }
        }
    }
}

/// RPython pyjitpl.py: MetaInterp.framestack
#[derive(Default)]
pub struct MIFrameStack {
    pub frames: Vec<MIFrame>,
}

impl MIFrameStack {
    /// Empty framestack.  Mirrors `self.framestack = []` in
    /// `MetaInterp.initialize_state_from_start` (pyjitpl.py:3269) and
    /// `rebuild_state_after_failure` (pyjitpl.py:3403).
    pub fn empty() -> Self {
        Self { frames: Vec::new() }
    }

    /// Build a stack pre-seeded with one root frame.  Pyre's
    /// `JitCodeMachine` always opens with a single root frame, so this
    /// constructor mirrors `MIFrameStack::new(root)` from before the
    /// `Arc<JitCode>` migration.
    pub fn new(root: MIFrame) -> Self {
        Self { frames: vec![root] }
    }

    pub fn current_mut(&mut self) -> &mut MIFrame {
        self.frames.last_mut().expect("empty JitCode frame stack")
    }

    pub fn push(&mut self, frame: MIFrame) {
        self.frames.push(frame);
    }

    pub fn pop(&mut self) -> Option<MIFrame> {
        self.frames.pop()
    }

    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jitcode::JitCodeBuilder;
    use majit_ir::OpRef;

    fn make_jitcode_with_regs(num_i: u16, num_r: u16, num_f: u16) -> Arc<JitCode> {
        let mut builder = JitCodeBuilder::new();
        for i in 0..num_i {
            builder.load_const_i_value(i, 0);
        }
        for i in 0..num_r {
            builder.load_const_r_value(i, 0);
        }
        for i in 0..num_f {
            builder.load_const_f_value(i, 0);
        }
        Arc::new(builder.finish())
    }

    #[test]
    fn setup_call_distributes_argboxes_by_kind_in_declaration_order() {
        let jitcode = make_jitcode_with_regs(2, 2, 1);
        let mut frame = MIFrame::new(jitcode.clone(), 5);
        frame.pc = 99;
        frame.setup_call(&[
            (JitArgKind::Int, OpRef(10), 100),
            (JitArgKind::Ref, OpRef(20), 200),
            (JitArgKind::Int, OpRef(11), 101),
            (JitArgKind::Float, OpRef(30), 300),
            (JitArgKind::Ref, OpRef(21), 201),
        ]);

        assert_eq!(frame.pc, 0);
        assert_eq!(frame.int_regs[0], Some(OpRef(10)));
        assert_eq!(frame.int_values[0], Some(100));
        assert_eq!(frame.int_regs[1], Some(OpRef(11)));
        assert_eq!(frame.int_values[1], Some(101));
        assert_eq!(frame.ref_regs[0], Some(OpRef(20)));
        assert_eq!(frame.ref_values[0], Some(200));
        assert_eq!(frame.ref_regs[1], Some(OpRef(21)));
        assert_eq!(frame.ref_values[1], Some(201));
        assert_eq!(frame.float_regs[0], Some(OpRef(30)));
        assert_eq!(frame.float_values[0], Some(300));
    }

    /// Step 1b: `MIFrame::get_list_of_active_boxes` (non-in_a_call
    /// path) pushes live int / ref / float registers onto the trace's
    /// `_snapshot_array_data` in declaration order, returning the
    /// `new_array` offset. Tests with `after_residual_call=true` so the
    /// LIVE op sits exactly at `self.pc` (pyjitpl.py:194-195).
    #[test]
    fn get_list_of_active_boxes_populates_trace_snapshot_array() {
        use crate::opencoder::{Box as OpBox, TraceRecordBuffer};
        use majit_ir::OpRef;
        use std::sync::Arc;

        // 1 int reg + 1 ref reg + 0 float regs live at pc=0.
        let mut builder = JitCodeBuilder::new();
        let mut jitcode = builder.finish();
        jitcode.c_num_regs_i = 1;
        jitcode.c_num_regs_r = 1;
        jitcode.c_num_regs_f = 0;
        const LIVE_OP: u8 = 0x42;
        // bytecode: [LIVE_OP, offset_lo, offset_hi] at pc 0..3.
        // decode_offset reads 2 bytes → liveness info at offset 0.
        jitcode.code = vec![LIVE_OP, 0x00, 0x00];
        let jitcode = Arc::new(jitcode);

        // all_liveness at offset 0:
        //   len_i=1 len_r=1 len_f=0  →  3 header bytes
        //   int bitmask byte: bit 0 = register 0 live
        //   ref bitmask byte: bit 0 = register 0 live
        let all_liveness: Vec<u8> = vec![1, 1, 0, 0b0000_0001, 0b0000_0001];

        let mut frame = MIFrame::new(jitcode, 0);
        // int_regs[0] non-constant OpRef(5) → Box::ResOp(5).
        frame.int_regs[0] = Some(OpRef(5));
        frame.int_values[0] = Some(0);
        // ref_regs[0] constant pointer addr=0xdead_beef → Box::ConstPtr.
        frame.ref_regs[0] = Some(OpRef::from_const(7));
        frame.ref_values[0] = Some(0xdead_beef);

        let sd = Arc::new(crate::MetaInterpStaticData::new());
        let mut trace = TraceRecordBuffer::new(16, sd);
        let storage = frame.get_list_of_active_boxes(
            /* in_a_call */ false,
            &mut trace,
            LIVE_OP,
            &all_liveness,
            /* after_residual_call */ true,
        );

        // Two boxes pushed → array length prefix + 2 varints.
        // new_array(2) returns the offset before the length prefix.
        assert!(
            storage > 0,
            "storage offset must be non-zero for non-empty array"
        );

        // Verify the encoded tagged values match what we expect:
        // int: ResOp(5) encodes via _encode_box_position(5).
        // ref: ConstPtr(0xdead_beef) encodes via _encode_ptr.
        // Decode directly from _snapshot_array_data at `storage`.
        let (length, consumed) =
            crate::opencoder::decode_varint_signed(&trace._snapshot_array_data[storage as usize..]);
        assert_eq!(length, 2, "array length prefix");
        let p0 = storage as usize + consumed;
        let (tag0, c0) = crate::opencoder::decode_varint_signed(&trace._snapshot_array_data[p0..]);
        assert_eq!(
            tag0,
            TraceRecordBuffer::_encode_box_position(5),
            "int register → TAGBOX(5)"
        );
        let p1 = p0 + c0;
        let (tag1, _) = crate::opencoder::decode_varint_signed(&trace._snapshot_array_data[p1..]);
        assert!(tag1 != 0, "ref register should encode to non-zero tag");
    }

    #[test]
    #[should_panic(expected = "in_a_call=true branch")]
    fn get_list_of_active_boxes_in_a_call_is_not_yet_wired() {
        use crate::opencoder::TraceRecordBuffer;
        use std::sync::Arc;

        let jitcode = make_jitcode_with_regs(1, 0, 0);
        let mut frame = MIFrame::new(jitcode, 0);
        frame.int_regs[0] = Some(OpRef(0));
        frame.int_values[0] = Some(0);

        let sd = Arc::new(crate::MetaInterpStaticData::new());
        let mut trace = TraceRecordBuffer::new(1, sd);
        // Triggers the in_a_call guard — see docstring.
        let _ = frame.get_list_of_active_boxes(
            /* in_a_call */ true,
            &mut trace,
            /* op_live */ 0,
            /* all_liveness */ &[],
            /* after_residual_call */ false,
        );
    }

    #[test]
    fn cleanup_registers_clears_ref_slots_and_pushed_box() {
        let jitcode = make_jitcode_with_regs(2, 2, 1);
        let mut frame = MIFrame::new(jitcode.clone(), 0);
        frame.int_regs[0] = Some(OpRef(1));
        frame.int_values[0] = Some(11);
        frame.ref_regs[0] = Some(OpRef(2));
        frame.ref_values[0] = Some(22);
        frame.float_regs[0] = Some(OpRef(3));
        frame.float_values[0] = Some(33);
        frame.pushed_box = Some(OpRef(99));

        frame.cleanup_registers();

        // pyjitpl.py:121-127: int and float slots are untouched.
        assert_eq!(frame.int_regs[0], Some(OpRef(1)));
        assert_eq!(frame.int_values[0], Some(11));
        assert_eq!(frame.float_regs[0], Some(OpRef(3)));
        assert_eq!(frame.float_values[0], Some(33));
        // pyjitpl.py:124-126: ref slots [0, num_regs_r()) are cleared.
        assert!(frame.ref_regs.iter().all(|r| r.is_none()));
        assert!(frame.ref_values.iter().all(|v| v.is_none()));
        // pyjitpl.py:127: pushed_box is reset to None.
        assert_eq!(frame.pushed_box, None);
    }

    #[test]
    fn make_result_of_lastop_stores_into_typed_slot() {
        let jitcode = make_jitcode_with_regs(2, 2, 1);
        let mut frame = MIFrame::new(jitcode.clone(), 0);

        frame.make_result_of_lastop(JitArgKind::Int, 1, OpRef(7), 77);
        assert_eq!(frame.int_regs[1], Some(OpRef(7)));
        assert_eq!(frame.int_values[1], Some(77));

        frame.make_result_of_lastop(JitArgKind::Ref, 0, OpRef(8), 88);
        assert_eq!(frame.ref_regs[0], Some(OpRef(8)));
        assert_eq!(frame.ref_values[0], Some(88));

        frame.make_result_of_lastop(JitArgKind::Float, 0, OpRef(9), 99);
        assert_eq!(frame.float_regs[0], Some(OpRef(9)));
        assert_eq!(frame.float_values[0], Some(99));
    }

    #[test]
    fn setup_resume_at_op_assigns_pc() {
        let jitcode = make_jitcode_with_regs(0, 0, 0);
        let mut frame = MIFrame::new(jitcode.clone(), 0);
        frame.setup_resume_at_op(123);
        assert_eq!(frame.pc, 123);
    }

    #[test]
    fn setup_call_with_empty_argboxes_only_resets_pc() {
        let jitcode = make_jitcode_with_regs(1, 1, 1);
        let mut frame = MIFrame::new(jitcode.clone(), 5);
        frame.pc = 42;
        frame.setup_call(&[]);
        assert_eq!(frame.pc, 0);
        assert!(frame.int_regs.iter().all(|r| r.is_none()));
        assert!(frame.ref_regs.iter().all(|r| r.is_none()));
        assert!(frame.float_regs.iter().all(|r| r.is_none()));
    }
}
