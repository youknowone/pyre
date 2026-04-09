//! JitState implementation for pyre.
//!
//! `PyreJitState` bridges the interpreter's `PyFrame` with majit's JIT
//! infrastructure. It extracts live values from the frame, restores them
//! after compiled code runs, and provides the meta/sym types for tracing.

use majit_ir::{DescrRef, OpCode, OpRef, Type, Value};
use majit_metainterp::virtualizable::VirtualizableInfo;
use majit_metainterp::{
    JitDriverStaticData, JitState, ResidualVirtualizableSync, TraceAction, TraceCtx,
};

use pyre_interpreter::bytecode::{BinaryOperator, CodeObject, ComparisonOperator, Instruction};
use pyre_interpreter::pyframe::PendingInlineResult;
use pyre_interpreter::truth_value as objspace_truth_value;
use pyre_object::PyObjectRef;
use pyre_object::boolobject::w_bool_get_value;
use pyre_object::listobject::w_list_getitem;
use pyre_object::pyobject::{
    BOOL_TYPE, DICT_TYPE, FLOAT_TYPE, INT_TYPE, LIST_TYPE, NONE_TYPE, PyType, TUPLE_TYPE, is_bool,
    is_dict, is_float, is_int, is_int_or_long, is_list, is_long, is_none, is_tuple,
};
use pyre_object::rangeobject::RANGE_ITER_TYPE;
use pyre_object::strobject::is_str;
use pyre_object::tupleobject::w_tuple_getitem;
use pyre_object::{
    PY_NULL, w_bool_from, w_float_get_value, w_int_get_value, w_int_new,
    w_list_can_append_without_realloc, w_list_is_inline_storage, w_list_len, w_list_new,
    w_list_uses_float_storage, w_list_uses_int_storage, w_list_uses_object_storage,
    w_str_get_value, w_tuple_len,
};

/// jitcode.py:9-21 / codewriter.py:68: JitCode — compiled bytecode unit.
/// RPython creates JitCode objects with codewriter; pyre wraps
/// CodeObject pointers with the same sequential index.
// SAFETY: JitCode is only written once (during creation) and then
// read-only. The code pointer is stable for the program lifetime.
unsafe impl Sync for JitCode {}

pub(crate) struct JitCode {
    /// Pointer to the Code object (W_CodeObject).
    /// Matches frame.code and getcode(func).
    pub code: *const (),
    /// codewriter.py:68: jitcode.index = len(all_jitcodes).
    pub index: i32,
    /// RPython parity: pointer to majit JitCode (liveness, bytecodes).
    /// Set by codewriter via set_majit_jitcode(). Used by
    /// get_list_of_active_boxes to access the same LivenessInfo
    /// that consume_one_section uses (all_liveness parity).
    pub majit_jitcode: *const majit_metainterp::jitcode::JitCode,
}

impl JitCode {
    /// Extract raw CodeObject from the W_CodeObject stored in this JitCode.
    #[inline]
    pub unsafe fn raw_code(&self) -> *const CodeObject {
        if self.code.is_null() {
            return std::ptr::null();
        }
        pyre_interpreter::w_code_get_ptr(self.code as pyre_object::PyObjectRef) as *const CodeObject
    }
}

/// warmspot.py:148-282: MetaInterpStaticData — per-driver compile-time data.
///
/// RPython: created by WarmRunnerDesc, holds jitcodes list populated
/// by codewriter.make_jitcodes(). Accessed as MetaInterp.staticdata.
///
/// pyre: per-thread equivalent (no-GIL runtime). Indices assigned
/// lazily at trace-time (no codewriter phase). Portal gets 0,
/// inline callees get 1, 2, ….
struct MetaInterpStaticData {
    /// codewriter.py:80: CodeObject* → index into jitcodes vec.
    by_code: std::collections::HashMap<usize, usize>,
    /// warmspot.py:282: self.metainterp_sd.jitcodes = jitcodes.
    /// Box<JitCode> for address stability across vec growth.
    jitcodes: Vec<Box<JitCode>>,
}

impl MetaInterpStaticData {
    fn new() -> Self {
        Self {
            by_code: std::collections::HashMap::new(),
            jitcodes: Vec::new(),
        }
    }

    /// codewriter.py:68: get or create JitCode for a CodeObject.
    /// Returns a stable pointer (Box ensures no reallocation moves).
    fn jitcode_for(&mut self, code: *const ()) -> *const JitCode {
        let key = code as usize;
        if let Some(&idx) = self.by_code.get(&key) {
            return &*self.jitcodes[idx] as *const JitCode;
        }
        let index = self.jitcodes.len() as i32;
        let jitcode = Box::new(JitCode {
            code,
            index,
            majit_jitcode: std::ptr::null(),
        });
        let ptr = &*jitcode as *const JitCode;
        self.by_code.insert(key, self.jitcodes.len());
        self.jitcodes.push(jitcode);
        ptr
    }
}

use std::cell::RefCell;

thread_local! {
    /// warmspot.py:282: MetaInterp.staticdata (per-thread for no-GIL).
    static METAINTERP_SD: RefCell<MetaInterpStaticData> =
        RefCell::new(MetaInterpStaticData::new());
}

/// pyjitpl.py:74: frame.jitcode — get or create JitCode for CodeObject.
/// RPython: MetaInterp.staticdata.jitcodes[idx]; pyre: METAINTERP_SD.
pub(crate) fn jitcode_for(code: *const ()) -> *const JitCode {
    // Register global frame_value_count callback on first use.
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        majit_ir::resumedata::set_frame_value_count_fn(frame_value_count_at);
    });
    METAINTERP_SD.with(|r| r.borrow_mut().jitcode_for(code))
}

/// RPython parity: link state::JitCode to its majit JitCode.
/// Called by codewriter after PyJitCode compilation so that
/// get_list_of_active_boxes can look up LivenessInfo from the same
/// data source as consume_one_section (RPython all_liveness parity).
pub fn set_majit_jitcode(
    code: *const (),
    majit_jitcode: *const majit_metainterp::jitcode::JitCode,
) {
    METAINTERP_SD.with(|r| {
        let mut sd = r.borrow_mut();
        // Ensure the JitCode entry exists.
        let _ = sd.jitcode_for(code);
        let key = code as usize;
        if let Some(&idx) = sd.by_code.get(&key) {
            // SAFETY: jitcode is Box'd and never removed from the vec.
            let jc = &mut *sd.jitcodes[idx];
            jc.majit_jitcode = majit_jitcode;
        }
    });
}

/// warmspot.py:282 metainterp_sd.jitcodes[jitcode_index]:
/// Resolve jitcode_index (sequential int from snapshot numbering)
/// to the corresponding CodeObject pointer.
pub fn code_for_jitcode_index(jitcode_index: i32) -> Option<*const ()> {
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        let idx = jitcode_index as usize;
        sd.jitcodes.get(idx).map(|jc| jc.code)
    })
}

/// resume.py:1049 consume_one_section → enumerate_vars parity:
/// Returns the number of tagged values encoded for a frame at
/// (jitcode_index, pc). Uses JitCode liveness (same data as
/// get_list_of_active_boxes) for RPython-parity multi-frame decode.
/// Falls back to LiveVars when JitCode liveness is unavailable.
pub fn frame_value_count_at(jitcode_index: i32, pc: i32) -> usize {
    METAINTERP_SD.with(|r| {
        let sd = r.borrow();
        let idx = jitcode_index as usize;
        let jc = match sd.jitcodes.get(idx) {
            Some(jc) => jc,
            None => return 0,
        };
        // Primary: JitCode liveness (same path as get_list_of_active_boxes).
        if !jc.majit_jitcode.is_null() {
            let mjc = unsafe { &*jc.majit_jitcode };
            if let Some(&jit_pc) = mjc.py_to_jit_pc.get(pc as usize) {
                if let Some(info) = mjc.liveness.iter().find(|i| i.pc as usize == jit_pc) {
                    // pyjitpl.py:212: total = length_i + length_r + length_f
                    return info.total_live();
                }
            }
        }
        // Fallback: LiveVars from CodeObject (backward-compat).
        // Mirror trace_opcode.rs:get_list_of_active_boxes LiveVars path:
        // both iterate locals and stack values filtering by
        // is_local_live / is_stack_live so the encoder box count and
        // the decoder value count agree even for inlined-function
        // frames whose majit_jitcode has not been built at trace time.
        if !jc.code.is_null() {
            let raw = unsafe { jc.raw_code() };
            let live = crate::liveness::liveness_for(raw);
            let code_ref = unsafe { &*raw };
            let nlocals = code_ref.varnames.len();
            let live_locals = (0..nlocals)
                .filter(|&i| live.is_local_live(pc as usize, i))
                .count();
            let stack_depth = live.stack_depth_at(pc as usize);
            let live_stack = (0..stack_depth)
                .filter(|&i| live.is_stack_live(pc as usize, i))
                .count();
            return live_locals + live_stack;
        }
        0
    })
}

/// Sentinel null JitCode for uninitialized PyreSym.
static NULL_JITCODE: JitCode = JitCode {
    code: std::ptr::null(),
    index: -1,
    majit_jitcode: std::ptr::null(),
};

/// Traced value — RPython `FrontendOp(position, _resint/_resref/_resfloat)` parity.
///
/// Carries both the symbolic IR reference (OpRef) and the concrete
/// execution value (ConcreteValue). Created by opcode handlers that
/// compute concrete results alongside IR recording.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrontendOp {
    pub opref: OpRef,
    pub concrete: ConcreteValue,
}

impl FrontendOp {
    pub fn new(opref: OpRef, concrete: ConcreteValue) -> Self {
        Self { opref, concrete }
    }

    pub fn opref_only(opref: OpRef) -> Self {
        Self {
            opref,
            concrete: ConcreteValue::Null,
        }
    }
}

/// Typed concrete value — RPython `FrontendOp._resint/_resref/_resfloat` parity.
///
/// Python bytecode uses untyped locals, so we use a tagged enum instead of
/// RPython's separate `registers_i/r/f` arrays. Each variant corresponds to
/// one of RPython's Box types: `BoxInt`, `BoxPtr`, `BoxFloat`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConcreteValue {
    Int(i64),
    Float(f64),
    Ref(PyObjectRef),
    Null,
}

/// Convert a frame slot value to ConcreteValue, preserving null pointers
/// as Ref(PY_NULL) instead of ConcreteValue::Null. Frame slots always
/// contain known values — null means "uninitialized local", not "untracked".
pub(crate) fn concrete_value_from_slot(obj: PyObjectRef) -> ConcreteValue {
    if obj.is_null() {
        return ConcreteValue::Ref(pyre_object::PY_NULL);
    }
    ConcreteValue::from_pyobj(obj)
}

impl ConcreteValue {
    /// Convert from PyObjectRef (unbox if possible).
    /// Null pointers become ConcreteValue::Null ("untracked").
    pub fn from_pyobj(obj: PyObjectRef) -> Self {
        if obj.is_null() {
            return ConcreteValue::Null;
        }
        unsafe {
            if is_int(obj) {
                ConcreteValue::Int(w_int_get_value(obj))
            } else if is_float(obj) {
                ConcreteValue::Float(w_float_get_value(obj))
            } else {
                ConcreteValue::Ref(obj)
            }
        }
    }

    /// Convert to PyObjectRef (box if needed).
    pub fn to_pyobj(self) -> PyObjectRef {
        match self {
            ConcreteValue::Int(v) => w_int_new(v),
            ConcreteValue::Float(v) => pyre_object::w_float_new(v),
            ConcreteValue::Ref(obj) => obj,
            ConcreteValue::Null => PY_NULL,
        }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, ConcreteValue::Null)
    }

    /// RPython box.getint() parity.
    pub fn getint(&self) -> Option<i64> {
        match self {
            ConcreteValue::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// RPython box.getfloatstorage() parity.
    pub fn getfloat(&self) -> Option<f64> {
        match self {
            ConcreteValue::Float(v) => Some(*v),
            ConcreteValue::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// RPython box.getref_base() parity.
    pub fn getref(&self) -> PyObjectRef {
        self.to_pyobj()
    }

    /// Convert to majit IR Type.
    pub fn ir_type(&self) -> Type {
        match self {
            ConcreteValue::Int(_) => Type::Int,
            ConcreteValue::Float(_) => Type::Float,
            ConcreteValue::Ref(_) => Type::Ref,
            ConcreteValue::Null => Type::Ref,
        }
    }

    /// Truth value (RPython box.getint() != 0 for goto_if_not).
    pub fn is_truthy(&self) -> bool {
        match self {
            ConcreteValue::Int(v) => *v != 0,
            ConcreteValue::Float(v) => *v != 0.0,
            ConcreteValue::Ref(obj) => objspace_truth_value(*obj),
            ConcreteValue::Null => false,
        }
    }
}

/// Convert a bytecode constant to ConcreteValue.
pub fn load_const_concrete(constant: &pyre_interpreter::bytecode::ConstantData) -> ConcreteValue {
    use pyre_interpreter::bytecode::ConstantData;
    match constant {
        ConstantData::Integer { value } => match i64::try_from(value).ok() {
            Some(v) => ConcreteValue::Int(v),
            None => ConcreteValue::Ref(pyre_object::w_long_new(value.clone())),
        },
        ConstantData::Float { value } => ConcreteValue::Float(*value),
        ConstantData::Boolean { value } => ConcreteValue::Int(*value as i64),
        ConstantData::Str { value } => ConcreteValue::Ref(pyre_object::w_str_new(
            value.as_str().expect("non-UTF-8 string constant"),
        )),
        ConstantData::None => ConcreteValue::Ref(pyre_object::w_none()),
        _ => ConcreteValue::Null,
    }
}

use pyre_interpreter::{
    ArithmeticOpcodeHandler, BranchOpcodeHandler, ConstantOpcodeHandler, ControlFlowOpcodeHandler,
    IterOpcodeHandler, LocalOpcodeHandler, NamespaceOpcodeHandler, OpcodeStepExecutor, PyBigInt,
    PyError, PyNamespace, PyObjectArray, SharedOpcodeHandler, StackOpcodeHandler,
    TruthOpcodeHandler, builtin_code_name, decode_instruction_at, execute_opcode_step,
    function_get_code, function_get_globals, is_builtin_code, is_function, range_iter_continues,
};

use crate::descr::{
    bool_boolval_descr, dict_len_descr, float_floatval_descr, int_intval_descr,
    list_float_items_heap_cap_descr, list_float_items_len_descr, list_float_items_ptr_descr,
    list_int_items_heap_cap_descr, list_int_items_len_descr, list_int_items_ptr_descr,
    list_items_heap_cap_descr, list_items_len_descr, list_items_ptr_descr, list_strategy_descr,
    make_array_descr, make_size_descr, namespace_values_len_descr, namespace_values_ptr_descr,
    ob_type_descr, range_iter_current_descr, range_iter_step_descr, range_iter_stop_descr,
    str_len_descr, tuple_items_len_descr, tuple_items_ptr_descr, w_float_size_descr,
    w_int_size_descr,
};
use crate::frame_layout::{
    PYFRAME_CODE_OFFSET, PYFRAME_LOCALS_CELLS_STACK_OFFSET, PYFRAME_NAMESPACE_OFFSET,
    PYFRAME_NEXT_INSTR_OFFSET, PYFRAME_VALUESTACKDEPTH_OFFSET,
};
use crate::helpers::{TraceHelperAccess, emit_box_float_inline, emit_trace_bool_value_from_truth};

// Re-export liveness items so downstream `pyre_jit_trace::state::*` keeps working.
pub use crate::liveness::{LiveVars, expand_compact_to_dense, liveness_for};

/// Interpreter state exposed to the JIT framework.
///
/// Built from `PyFrame` before calling `back_edge`, and synced back
/// after compiled code runs.
/// Heap is the single source of truth (RPython parity).
/// next_instr / valuestackdepth live on the PyFrame heap object
/// and are accessed via read_frame_usize / write_frame_usize.
#[derive(majit_macros::VirtualizableState)]
pub struct PyreJitState {
    #[vable(frame)]
    pub frame: usize,
    /// blackhole.py:337 parity: liveness PC from rd_numb (setposition PC).
    /// When set, `restore_guard_failure_values` uses this instead of
    /// next_instr for liveness lookup — matching RPython's pattern where
    /// `blackholeinterp.setposition(jitcode, pc)` is called before
    /// `consume_one_section`.
    pub resume_pc: Option<usize>,
}

/// Meta information for a trace — describes the shape of the code being traced.
#[derive(Clone, majit_macros::VirtualizableMeta)]
pub struct PyreMeta {
    #[vable(num_locals)]
    pub num_locals: usize,
    pub ns_keys: Vec<String>,
    #[vable(valuestackdepth)]
    pub valuestackdepth: usize,
    pub has_virtualizable: bool,
    #[vable(slot_types)]
    pub slot_types: Vec<Type>,
}

/// Symbolic state during tracing.
///
/// `frame` maps to a live IR `OpRef`. Symbolic frame field tracking
/// (locals, stack, valuestackdepth, next_instr) persists across instructions.
/// Locals and stack are virtualized (carried through JUMP args);
/// only next_instr and valuestackdepth are synced before guards / loop close.
#[derive(Clone, majit_macros::VirtualizableSym)]
pub struct PyreSym {
    /// OpRef for the owning PyFrame pointer.
    #[vable(frame)]
    pub frame: OpRef,
    // ── Persistent symbolic frame field tracking ──
    #[vable(locals)]
    pub(crate) symbolic_locals: Vec<OpRef>,
    #[vable(stack)]
    pub symbolic_stack: Vec<OpRef>,
    #[vable(local_types)]
    pub(crate) symbolic_local_types: Vec<Type>,
    #[vable(stack_types)]
    pub symbolic_stack_types: Vec<Type>,
    pub pending_next_instr: Option<usize>,
    pub(crate) locals_cells_stack_array_ref: OpRef,
    #[vable(valuestackdepth)]
    pub(crate) valuestackdepth: usize,
    #[vable(nlocals)]
    pub(crate) nlocals: usize,
    pub(crate) symbolic_initialized: bool,
    /// Bridge-specific override for symbolic_locals.
    /// resume.py:1042 parity: when set, init_symbolic uses these OpRefs
    /// (mapped from RebuiltValue::Box(n) in rebuild_from_resumedata) instead
    /// of the vable_array_base-based layout. This ensures bridge traces see
    /// frame locals as symbolic InputArgs, not concrete values.
    pub(crate) bridge_local_oprefs: Option<Vec<OpRef>>,
    /// Bridge-specific override for symbolic_stack.
    /// resume.py:1042 parity: consume_boxes fills both local and stack
    /// registers for the resumed frame.
    pub(crate) bridge_stack_oprefs: Option<Vec<OpRef>>,
    /// Bridge-specific override for symbolic_local_types.
    /// resume.py:1245 decode_box parity: each slot's `.type` is fixed by
    /// which `_callback_i/_callback_r/_callback_f` the encoder dispatched
    /// through `enumerate_vars(info, liveness_info, ...)`. RPython's
    /// `IntFrontendOp/RefFrontendOp/FloatFrontendOp` carries this kind on
    /// `box.type`. pyre's `setup_bridge_sym` walks `live_i_regs +
    /// live_r_regs + live_f_regs` and records `Type::Int/Ref/Float` per
    /// slot here. Without this, init_symbolic falls into the
    /// `concrete_slot_types` path which always returns `Type::Ref` for
    /// W_IntObject locals, and `RETURN_VALUE` then takes the
    /// `trace_guarded_int_payload` unbox path that the optimizer
    /// constant-folds — producing `Finish(constant 0)` bridges.
    pub(crate) bridge_local_types: Option<Vec<Type>>,
    /// Bridge-specific override for symbolic_stack_types.
    pub(crate) bridge_stack_types: Option<Vec<Type>>,
    // virtualizable.py:86-93: ALL static fields in declared order.
    // RPython's unroll_static_fields includes every field from
    // _virtualizable_; ALL must be inputarg (not info_only).
    #[vable(inputarg)]
    pub(crate) vable_next_instr: OpRef,
    #[vable(inputarg)]
    pub(crate) vable_code: OpRef,
    #[vable(inputarg)]
    pub(crate) vable_valuestackdepth: OpRef,
    #[vable(inputarg)]
    pub(crate) vable_namespace: OpRef,
    pub(crate) symbolic_namespace_slots: std::collections::HashMap<usize, OpRef>,
    #[vable(array_base)]
    pub(crate) vable_array_base: Option<u32>,
    /// RPython goto_if_not fusion: cached raw truth OpRef from the
    /// most recent int/float comparison. POP_JUMP_IF_FALSE/TRUE
    /// consumes this to emit GUARD_TRUE/GUARD_FALSE directly,
    /// bypassing bool object creation.
    pub(crate) last_comparison_truth: Option<OpRef>,
    /// Concrete truth value paired with `last_comparison_truth`.
    /// RPython pyjitpl.py passes the computed condbox directly into
    /// goto_if_not and reads `box.getint()` there, instead of reloading a
    /// stack value later. Keep the concrete branch direction alongside the
    /// raw traced truth so POP_JUMP_IF can consume the same pair.
    pub(crate) last_comparison_concrete_truth: Option<bool>,
    /// Popped branch source currently being converted to truth.
    /// Generic guards emitted during `truth_value(value)` must still capture
    /// the pre-pop stack shape, matching RPython's goto_if_not(box).
    pub(crate) pending_branch_value: Option<OpRef>,
    /// RPython goto_if_not parity: the branch target NOT taken during tracing.
    /// On guard failure, the interpreter jumps to this PC instead of
    /// re-executing the branch instruction (stack machine safety).
    pub(crate) pending_branch_other_target: Option<usize>,
    pub transient_value_types: std::collections::HashMap<OpRef, Type>,
    // ── MIFrame concrete Box tracking (RPython registers_i/r/f parity) ──
    // Concrete Python object values for locals and stack, tracked in
    // parallel with symbolic_locals/symbolic_stack. Each opcode handler
    // updates these alongside the symbolic OpRefs so that guard decisions,
    // branch directions, and call results use internally tracked values
    // instead of reading from an external PyFrame snapshot.
    pub(crate) concrete_locals: Vec<ConcreteValue>,
    pub concrete_stack: Vec<ConcreteValue>,
    /// pyjitpl.py:74: frame.jitcode — JitCode reference.
    /// Provides both .code (CodeObject*) and .index (snapshot encoding).
    pub(crate) jitcode: *const JitCode,
    /// Namespace for global lookups.
    pub(crate) concrete_namespace: *mut pyre_interpreter::PyNamespace,
    /// Execution context pointer (for creating callee frames).
    pub(crate) concrete_execution_context: *const pyre_interpreter::PyExecutionContext,
    /// Virtualizable object pointer (PyFrame).
    /// RPython MetaInterp stores the virtualizable separately from MIFrame.
    pub(crate) concrete_vable_ptr: *mut u8,
    /// Function-entry traces use typed locals (RPython MIFrame parity).
    pub(crate) is_function_entry_trace: bool,
    /// RPython capture_resumedata(resumepc=orgpc) parity: pre-opcode
    /// snapshot of valuestackdepth + symbolic_stack so guards capture
    /// the state at opcode START. On guard failure the interpreter
    /// re-executes the opcode from orgpc.
    pub(crate) pre_opcode_vsd: Option<usize>,
    pub(crate) pre_opcode_stack: Option<Vec<OpRef>>,
    pub(crate) pre_opcode_stack_types: Option<Vec<Type>>,
    /// RPython generate_guard parity (pyjitpl.py:2558-2570): when a
    /// COMPARE_OP fast-path emits IntLt/FloatLt directly, save the PC
    /// and pre-opcode stack snapshot of the COMPARE_OP itself. The
    /// following PopJumpIf*'s record_branch_guard uses these so the
    /// guard's resume PC is the COMPARE_OP's orgpc and the operands
    /// are still on the symbolic stack — matching what blackhole sees
    /// when it re-executes the COMPARE_OP from orgpc.
    pub(crate) last_comparison_orgpc: Option<usize>,
    pub(crate) last_comparison_pre_vsd: Option<usize>,
    pub(crate) last_comparison_pre_stack: Option<Vec<OpRef>>,
    pub(crate) last_comparison_pre_stack_types: Option<Vec<Type>>,
    /// RPython MetaInterp.last_exc_value (pyjitpl.py:2745): concrete
    /// exception object pending during tracing. Set by execute_ll_raised
    /// (raise_varargs), consumed by handle_possible_exception.
    pub(crate) last_exc_value: pyre_object::PyObjectRef,
    /// RPython MetaInterp.class_of_last_exc_is_const (pyjitpl.py:2754):
    /// True after GUARD_EXCEPTION or GUARD_CLASS on the exception.
    pub(crate) class_of_last_exc_is_const: bool,
    /// RPython MetaInterp.last_exc_box (pyjitpl.py:3386): symbolic OpRef
    /// for the exception value. Set by handle_possible_exception after
    /// GUARD_EXCEPTION, consumed by finishframe_exception for stack push.
    pub(crate) last_exc_box: OpRef,
    /// pyjitpl.py:2597 virtualref_boxes: pairs of (jit_virtual, real_vref).
    /// Each pair: (symbolic OpRef, concrete pointer).
    /// resume.py:1093 restores virtual references on guard failure.
    /// Pairs stored flat: [virt_sym, virt_ptr, real_sym, real_ptr, ...].
    pub(crate) virtualref_boxes: Vec<(OpRef, usize)>,
}

/// Trace-time view over the virtualizable `PyFrame`.
///
/// Per-instruction wrapper that borrows persistent symbolic state from
/// `PyreSym` via raw pointer. The symbolic tracking (locals, stack,
/// valuestackdepth, next_instr) lives in PyreSym and survives across
/// instructions; this struct provides the per-instruction context
/// (ctx, fallthrough_pc).
pub struct MIFrame {
    pub(crate) ctx: *mut TraceCtx,
    pub(crate) sym: *mut PyreSym,
    pub(crate) ob_type_fd: DescrRef,
    pub(crate) fallthrough_pc: usize,
    /// Concrete PyFrame address for exception table lookup.
    pub(crate) concrete_frame_addr: usize,
    /// RPython pyjitpl.py orgpc parity: the PC at the START of the current
    /// opcode. All guards within one opcode capture this as their resume PC
    /// so that guard failure re-executes the opcode from the beginning.
    pub(crate) orgpc: usize,
    /// PyPy capture_resumedata: parent frame chain for multi-frame guards.
    /// Each element is (fail_args, fail_arg_types, resumepc, jitcode_index).
    /// opencoder.py:819-834: walks framestack to build parent snapshot chain.
    pub parent_frames: Vec<(Vec<OpRef>, Vec<Type>, usize, i32)>,
    pub pending_inline_frame: Option<PendingInlineFrame>,
}

pub(crate) fn code_has_backward_jump(code: &CodeObject) -> bool {
    for pc in 0..code.instructions.len() {
        let Some((instruction, _)) = decode_instruction_at(code, pc) else {
            continue;
        };
        if matches!(
            instruction,
            Instruction::JumpBackward { .. } | Instruction::JumpBackwardNoInterrupt { .. }
        ) {
            return true;
        }
    }
    false
}

pub(crate) fn instruction_consumes_comparison_truth(instruction: Instruction) -> bool {
    matches!(
        instruction,
        Instruction::PopJumpIfFalse { .. } | Instruction::PopJumpIfTrue { .. }
    )
}

pub(crate) fn instruction_is_trivia_between_compare_and_branch(instruction: Instruction) -> bool {
    matches!(
        instruction,
        Instruction::ExtendedArg
            | Instruction::Resume { .. }
            | Instruction::Nop
            | Instruction::Cache
            | Instruction::NotTaken
            | Instruction::ToBool
    )
}

/// RPython exc=True parity: instructions that correspond to JitCode ops
/// with exc=True. Only external calls and operations that invoke arbitrary
/// Python code need GUARD_NO_EXCEPTION. Arithmetic, comparisons, and
/// local variable access are lowered to primitive IR ops (exc=False) in
/// RPython and protected by type-specific guards instead.
pub(crate) fn instruction_may_raise(instruction: Instruction) -> bool {
    matches!(
        instruction,
        // RPython exc=True: external calls and attribute access that
        // may invoke arbitrary Python code (__getattr__, descriptors).
        Instruction::Call { .. }
            | Instruction::CallKw { .. }
            | Instruction::CallFunctionEx { .. }
            | Instruction::StoreAttr { .. }
            | Instruction::DeleteAttr { .. }
            | Instruction::StoreSubscr
            | Instruction::DeleteSubscr
            | Instruction::ImportName { .. }
            | Instruction::ImportFrom { .. }
            // RPython opimpl_raise/opimpl_reraise: always Err, needs
            // GUARD_EXCEPTION + finishframe_exception for exception-path tracing.
            | Instruction::RaiseVarargs { .. }
            | Instruction::Reraise { .. }
    )
}

/// Environment context — currently unused.
pub struct PyreEnv;

pub(crate) fn pyobject_array_descr() -> DescrRef {
    make_array_descr(0, 8, Type::Ref, false)
}

pub(crate) fn int_array_descr() -> DescrRef {
    make_array_descr(0, 8, Type::Int, true)
}

pub(crate) fn float_array_descr() -> DescrRef {
    make_array_descr(0, 8, Type::Float, false)
}

/// resume.py:656 arraydescr kind dispatch for virtual array materialization.
/// kind: 0=ref (is_array_of_pointers), 1=int, 2=float (is_array_of_floats).
pub(crate) fn array_descr_for_kind(kind: u8, _descr_index: u32) -> DescrRef {
    match kind {
        0 => pyobject_array_descr(),
        2 => float_array_descr(),
        _ => int_array_descr(),
    }
}

/// `descr.py SizeDescr` for the host `PyFrame` virtualizable struct.
///
/// All `PyFrame` field descriptors point at this SizeDescr via
/// `FieldDescr.parent_descr` so the optimizer's `ensure_ptr_info_arg0`
/// (`optimizer.py:478-484`) can dispatch the GETFIELD/SETFIELD branch
/// to `InstancePtrInfo` / `StructPtrInfo`. Also handed to
/// `VirtualizableInfo::set_parent_descr` so virtualizable field
/// descriptors share the same parent.
pub fn pyframe_size_descr() -> DescrRef {
    make_size_descr(std::mem::size_of::<pyre_interpreter::pyframe::PyFrame>())
}

pub(crate) fn frame_locals_cells_stack_descr() -> DescrRef {
    crate::descr::pyframe_locals_cells_stack_descr()
}

pub(crate) fn frame_stack_depth_descr() -> DescrRef {
    crate::descr::pyframe_stack_depth_descr()
}

pub(crate) fn frame_next_instr_descr() -> DescrRef {
    crate::descr::pyframe_next_instr_descr()
}

pub(crate) fn frame_code_descr() -> DescrRef {
    crate::descr::pyframe_code_descr()
}

pub(crate) fn frame_namespace_descr() -> DescrRef {
    crate::descr::pyframe_namespace_descr()
}

pub(crate) fn trace_ob_type_descr() -> DescrRef {
    ob_type_descr()
}

pub(crate) fn wrapint(ctx: &mut TraceCtx, value: OpRef) -> OpRef {
    crate::helpers::emit_box_int_inline(ctx, value, w_int_size_descr(), int_intval_descr())
}

pub(crate) fn note_inline_trace_too_long(
    callee_key: u64,
    caller_function_key: u64,
    root_trace_key: u64,
    err: &PyError,
) {
    if err.message != "inline trace aborted" {
        return;
    }
    let (driver, _) = crate::driver::driver_pair();
    let warm_state = driver.meta_interp_mut().warm_state_mut();
    warm_state.disable_noninlinable_function(callee_key);
    if callee_key == caller_function_key {
        // RPython converges the next portal entry quickly after marking the
        // huge function non-inlinable. Mirror that by priming the warmstate-
        // owned function-entry counter instead of only bumping a backedge
        // hot counter.
        warm_state.boost_function_entry(caller_function_key);
    } else {
        warm_state.trace_next_iteration(root_trace_key);
    }
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][trace-through] disable_noninlinable_function key={} caller_function_key={} root_trace_key={} same_key={}",
            callee_key,
            caller_function_key,
            root_trace_key,
            callee_key == caller_function_key
        );
    }
}

pub(crate) fn current_trace_green_key(state: &mut MIFrame) -> u64 {
    state.with_ctx(|_, ctx| ctx.green_key())
}

pub(crate) fn root_trace_green_key(state: &mut MIFrame) -> u64 {
    state.with_ctx(|_, ctx| ctx.root_green_key())
}

/// pyjitpl.py:3514 find_biggest_function
pub(crate) fn biggest_inline_trace_key(state: &mut MIFrame) -> Option<u64> {
    state.with_ctx(|_, ctx| ctx.find_biggest_function())
}

pub(crate) fn note_root_trace_too_long(green_key: u64) {
    let (driver, _) = crate::driver::driver_pair();
    let warm_state = driver.meta_interp_mut().warm_state_mut();
    warm_state.trace_next_iteration(green_key);
    warm_state.mark_force_finish_tracing(green_key);
    if majit_metainterp::majit_log_enabled() {
        eprintln!(
            "[jit][trace-too-long] trace_next_iteration + mark_force_finish_tracing key={}",
            green_key
        );
    }
}

pub(crate) fn wrapfloat(ctx: &mut TraceCtx, value: OpRef) -> OpRef {
    emit_box_float_inline(ctx, value, w_float_size_descr(), float_floatval_descr())
}

pub(crate) fn ensure_boxed_for_ca(ctx: &mut TraceCtx, state: &MIFrame, value: OpRef) -> OpRef {
    match state.value_type(value) {
        Type::Int => wrapint(ctx, value),
        Type::Float => wrapfloat(ctx, value),
        Type::Ref | Type::Void => value,
    }
}

pub(crate) fn box_value_for_python_helper(
    state: &mut MIFrame,
    ctx: &mut TraceCtx,
    value: OpRef,
) -> OpRef {
    match state.value_type(value) {
        Type::Int => wrapint(ctx, value),
        Type::Float => wrapfloat(ctx, value),
        Type::Ref | Type::Void => value,
    }
}

pub(crate) fn box_args_for_python_helper(
    state: &mut MIFrame,
    ctx: &mut TraceCtx,
    args: &[OpRef],
) -> Vec<OpRef> {
    args.iter()
        .map(|&arg| box_value_for_python_helper(state, ctx, arg))
        .collect()
}

// RPython parity note: pyjitpl.py (tracer) records GETFIELD_GC ops WITHOUT
// any constant folding. Folding happens exclusively in the optimizer's
// `optimize_GETFIELD_GC_I` (heap.py:639-646), which delegates to
// `optimizer.constant_fold(op)` → `_execute_arglist` → `do_getfield_gc_*`.
// pyre's `OptContext::constant_fold` in optimizeopt/mod.rs is the exact
// port of that path — it handles Int/Float/Ref via `execute_nonspec_const`
// dispatched on `field_type()` and `field_size()`.
//
// The previous tracer-level `try_trace_const_pure_int_field` helper was a
// pyre-specific pre-optimization that duplicated (and mistyped) the
// optimizer logic. It has been removed for structural parity with RPython.

pub(crate) fn try_trace_const_boxed_int(
    ctx: &mut TraceCtx,
    value: OpRef,
    concrete_value: PyObjectRef,
) -> Option<OpRef> {
    if ctx.const_value(value) != Some(concrete_value as i64) {
        return None;
    }
    unsafe {
        if is_int(concrete_value) {
            return Some(ctx.const_int(w_int_get_value(concrete_value)));
        }
        if is_bool(concrete_value) {
            return Some(ctx.const_int(if w_bool_get_value(concrete_value) {
                1
            } else {
                0
            }));
        }
    }
    None
}

pub(crate) fn try_trace_const_boxed_float(
    ctx: &mut TraceCtx,
    value: OpRef,
    concrete_value: PyObjectRef,
) -> Option<OpRef> {
    if ctx.const_value(value) != Some(concrete_value as i64) {
        return None;
    }
    unsafe {
        // The result is a raw f64 bit pattern. Use const_float so the
        // constant pool tags it as Float, mirroring ConstFloat parity.
        is_float(concrete_value)
            .then(|| ctx.const_float(w_float_get_value(concrete_value).to_bits() as i64))
    }
}

/// pyjitpl.py:750-758: read container length.
///
/// RPython's `arraylen_gc` reads the GC array header — there is exactly one
/// length per array, so RPython keeps a per-box `heapc_deps[0]` slot. pyre
/// stores list/bytes/tuple lengths as plain struct fields, so the cached
/// value lives in the regular field cache (`heap_cache.field_cache`).
/// `opimpl_getfield_gc_i` already does that lookup, so this helper is now
/// just a thin alias kept for source-stability with the call sites.
pub(crate) fn trace_arraylen_gc(ctx: &mut TraceCtx, obj: OpRef, descr: DescrRef) -> OpRef {
    opimpl_getfield_gc_i(ctx, obj, descr)
}

pub(crate) fn opimpl_getfield_gc_i(ctx: &mut TraceCtx, obj: OpRef, descr: DescrRef) -> OpRef {
    // pyjitpl.py:opimpl_getfield_gc_i parity: the tracer does NOT fold
    // pure field reads on constant objects. Folding happens in the
    // optimizer (heap.py:optimize_GETFIELD_GC_I → optimizer.constant_fold),
    // which pyre ports in OptContext::execute_nonspec_const with correct
    // type dispatch (Int/Float/Ref). The tracer only records the GC op.
    //
    // heapcache.py: check if this field was already read/written in this trace
    let field_index = descr.index();
    if let Some(cached) = ctx.heap_cache().getfield_cached(obj, field_index) {
        return cached;
    }
    // pyjitpl.py:1074-1089: quasi-immutable field handling.
    // Record the field as quasi-immut known so subsequent reads skip
    // the QUASIIMMUT_FIELD op. Emit GUARD_NOT_INVALIDATED if needed.
    // NOTE: GuardNotInvalidated is NOT emitted here — it requires
    // PyreSym.generate_guard for proper snapshot/fail_args (pyjitpl.py:1087
    // generate_guard parity). Instead, set a flag on ctx so the caller
    // (PyreSym with_ctx block) can emit it with full resume data.
    if descr.is_quasi_immutable() && !ctx.heap_cache().is_quasi_immut_known(obj, field_index) {
        ctx.heap_cache_mut().quasi_immut_now_known(obj, field_index);
        ctx.record_op_with_descr(OpCode::QuasiimmutField, &[obj], descr.clone());
        if ctx.heap_cache_mut().check_and_clear_guard_not_invalidated() {
            ctx.set_pending_guard_not_invalidated(Some(ctx.last_traced_pc));
        }
    }
    let opcode = if descr.is_always_pure() {
        OpCode::GetfieldGcPureI
    } else {
        OpCode::GetfieldGcI
    };
    let result = ctx.record_op_with_descr(opcode, &[obj], descr);
    ctx.heap_cache_mut()
        .getfield_now_known(obj, field_index, result);
    result
}

pub(crate) fn trace_gc_object_type_field(ctx: &mut TraceCtx, obj: OpRef, descr: DescrRef) -> OpRef {
    opimpl_getfield_gc_i(ctx, obj, descr)
}

// Note: pyre does not currently route GetfieldGcF/GetfieldGcPureF through
// state.rs. Float field unboxing goes via the codewriter-generated
// `getfield_gc_f_pureornot` (majit-codewriter/src/codegen.rs),
// which — matching RPython's pyjitpl.py opimpl_getfield_gc_f — records
// the GC op without folding. The optimizer's `optimize_GETFIELD_GC_F`
// (= `optimize_GETFIELD_GC_I` via RPython's alias) handles folding.

/// Unbox int with proper GuardClass resume data via MIFrame::generate_guard.
pub(crate) fn trace_unbox_int_with_resume(
    frame: &mut MIFrame,
    ctx: &mut TraceCtx,
    obj: OpRef,
    int_type_addr: i64,
) -> OpRef {
    trace_unbox_int_with_resume_descr(
        frame,
        ctx,
        obj,
        int_type_addr,
        crate::descr::int_intval_descr(),
    )
}

pub(crate) fn trace_unbox_int_with_resume_descr(
    frame: &mut MIFrame,
    ctx: &mut TraceCtx,
    obj: OpRef,
    type_addr: i64,
    intval_descr: majit_ir::DescrRef,
) -> OpRef {
    // pyjitpl.py GUARD_CLASS(box, cls): guard takes object box directly,
    // backend loads typeptr at offset 0.
    if !ctx.heap_cache().is_class_known(obj) {
        let type_const = ctx.const_int(type_addr);
        frame.generate_guard(ctx, OpCode::GuardClass, &[obj, type_const]);
        ctx.heap_cache_mut()
            .class_now_known(obj, majit_ir::GcRef(type_addr as usize));
    }
    crate::generated::trace_unbox_int(
        ctx,
        obj,
        type_addr,
        crate::descr::ob_type_descr(),
        intval_descr,
        &[],
    )
}

/// Unbox float with proper GuardClass resume data via MIFrame::generate_guard.
pub(crate) fn trace_unbox_float_with_resume(
    frame: &mut MIFrame,
    ctx: &mut TraceCtx,
    obj: OpRef,
    float_type_addr: i64,
) -> OpRef {
    if !ctx.heap_cache().is_class_known(obj) {
        let type_const = ctx.const_int(float_type_addr);
        frame.generate_guard(ctx, OpCode::GuardClass, &[obj, type_const]);
        ctx.heap_cache_mut()
            .class_now_known(obj, majit_ir::GcRef(float_type_addr as usize));
    }
    crate::generated::trace_unbox_float(
        ctx,
        obj,
        float_type_addr,
        crate::descr::ob_type_descr(),
        crate::descr::float_floatval_descr(),
        &[],
    )
}

pub(crate) unsafe fn objspace_compare_ints(
    lhs_obj: PyObjectRef,
    rhs_obj: PyObjectRef,
    op: ComparisonOperator,
) -> bool {
    let lhs = w_int_get_value(lhs_obj);
    let rhs = w_int_get_value(rhs_obj);
    match op {
        ComparisonOperator::Less => lhs < rhs,
        ComparisonOperator::LessOrEqual => lhs <= rhs,
        ComparisonOperator::Greater => lhs > rhs,
        ComparisonOperator::GreaterOrEqual => lhs >= rhs,
        ComparisonOperator::Equal => lhs == rhs,
        ComparisonOperator::NotEqual => lhs != rhs,
    }
}

/// baseobjspace as_float: coerce int|float → f64.
/// Called only for int/float operands in the tracing fast path.
/// Long operands are handled by residual fallback, not this function.
unsafe fn as_float_for_trace(obj: PyObjectRef) -> f64 {
    if is_float(obj) {
        w_float_get_value(obj)
    } else if is_int(obj) {
        w_int_get_value(obj) as f64
    } else {
        0.0 // unreachable in trace fast path — long triggers residual
    }
}

/// Compare two numeric values as floats. Handles float_pair (int+float)
/// via as_float coercion matching baseobjspace::float_lt/le/gt/ge/eq/ne.
/// Long operands don't reach here — they trigger residual fallback.
pub(crate) unsafe fn objspace_compare_floats(
    lhs_obj: PyObjectRef,
    rhs_obj: PyObjectRef,
    op: ComparisonOperator,
) -> bool {
    let lhs = as_float_for_trace(lhs_obj);
    let rhs = as_float_for_trace(rhs_obj);
    match op {
        ComparisonOperator::Less => lhs < rhs,
        ComparisonOperator::LessOrEqual => lhs <= rhs,
        ComparisonOperator::Greater => lhs > rhs,
        ComparisonOperator::GreaterOrEqual => lhs >= rhs,
        ComparisonOperator::Equal => lhs == rhs,
        ComparisonOperator::NotEqual => lhs != rhs,
    }
}

pub(crate) fn frame_locals_cells_stack_array(ctx: &mut TraceCtx, frame: OpRef) -> OpRef {
    ctx.record_op_with_descr(
        OpCode::GetfieldRawI,
        &[frame],
        frame_locals_cells_stack_descr(),
    )
}

/// Read from frame's locals_cells_stack_w array.
/// Uses GcR (Ref-typed) to match RPython's GETARRAYITEM_GC_R,
/// ensuring the optimizer knows these are boxed pointers.
pub(crate) fn trace_array_getitem_value(ctx: &mut TraceCtx, array: OpRef, index: OpRef) -> OpRef {
    let descr = pyobject_array_descr();
    let descr_idx = descr.index();
    if let Some(cached) = ctx.heap_cache().getarrayitem_cache(array, index, descr_idx) {
        return cached;
    }
    let result = ctx.record_op_with_descr(OpCode::GetarrayitemGcR, &[array, index], descr);
    ctx.heap_cache_mut()
        .getarrayitem_now_known(array, index, descr_idx, result);
    result
}

/// Read from frame's locals_cells_stack_w — namespace access path.
pub(crate) fn trace_raw_array_getitem_value(
    ctx: &mut TraceCtx,
    array: OpRef,
    index: OpRef,
) -> OpRef {
    let descr = pyobject_array_descr();
    let descr_idx = descr.index();
    if let Some(cached) = ctx.heap_cache().getarrayitem_cache(array, index, descr_idx) {
        return cached;
    }
    let result = ctx.record_op_with_descr(OpCode::GetarrayitemGcR, &[array, index], descr);
    ctx.heap_cache_mut()
        .getarrayitem_now_known(array, index, descr_idx, result);
    result
}

pub(crate) fn trace_raw_int_array_getitem_value(
    ctx: &mut TraceCtx,
    array: OpRef,
    index: OpRef,
) -> OpRef {
    ctx.record_op_with_descr(OpCode::GetarrayitemRawI, &[array, index], int_array_descr())
}

pub(crate) fn trace_raw_float_array_getitem_value(
    ctx: &mut TraceCtx,
    array: OpRef,
    index: OpRef,
) -> OpRef {
    ctx.record_op_with_descr(
        OpCode::GetarrayitemRawF,
        &[array, index],
        float_array_descr(),
    )
}

/// Write to frame's locals_cells_stack_w array.
/// Uses Gc (GC-typed) to match RPython's SETARRAYITEM_GC.
pub(crate) fn trace_raw_array_setitem_value(
    ctx: &mut TraceCtx,
    array: OpRef,
    index: OpRef,
    value: OpRef,
) {
    let descr = pyobject_array_descr();
    let descr_idx = descr.index();
    ctx.record_op_with_descr(OpCode::SetarrayitemGc, &[array, index, value], descr);
    ctx.heap_cache_mut()
        .setarrayitem_cache(array, index, descr_idx, value);
}

pub(crate) fn trace_raw_int_array_setitem_value(
    ctx: &mut TraceCtx,
    array: OpRef,
    index: OpRef,
    value: OpRef,
) {
    ctx.record_op_with_descr(
        OpCode::SetarrayitemRaw,
        &[array, index, value],
        int_array_descr(),
    );
}

pub(crate) fn trace_raw_float_array_setitem_value(
    ctx: &mut TraceCtx,
    array: OpRef,
    index: OpRef,
    value: OpRef,
) {
    ctx.record_op_with_descr(
        OpCode::SetarrayitemRaw,
        &[array, index, value],
        float_array_descr(),
    );
}

pub(crate) fn is_boxed_int_value(concrete_value: PyObjectRef) -> bool {
    !concrete_value.is_null() && unsafe { is_int(concrete_value) }
}

pub(crate) fn frame_get_next_instr(ctx: &mut TraceCtx, frame: OpRef) -> OpRef {
    ctx.record_op_with_descr(OpCode::GetfieldRawI, &[frame], frame_next_instr_descr())
}

pub(crate) fn frame_get_code(ctx: &mut TraceCtx, frame: OpRef) -> OpRef {
    ctx.record_op_with_descr(OpCode::GetfieldGcR, &[frame], frame_code_descr())
}

pub(crate) fn frame_get_stack_depth(ctx: &mut TraceCtx, frame: OpRef) -> OpRef {
    ctx.record_op_with_descr(OpCode::GetfieldRawI, &[frame], frame_stack_depth_descr())
}

pub(crate) fn frame_get_namespace(ctx: &mut TraceCtx, frame: OpRef) -> OpRef {
    ctx.record_op_with_descr(OpCode::GetfieldGcR, &[frame], frame_namespace_descr())
}

/// Read a value from the unified `locals_cells_stack_w` at the given absolute index.
/// Get the concrete top-of-stack value for the RETURN_VALUE opcode.
pub(crate) fn concrete_return_value(frame: usize) -> Option<PyObjectRef> {
    let frame_ptr = (frame != 0).then_some(frame as *const u8)?;
    let vsd = unsafe {
        *(frame_ptr.add(crate::frame_layout::PYFRAME_VALUESTACKDEPTH_OFFSET) as *const usize)
    };
    if vsd == 0 {
        return None;
    }
    concrete_stack_value(frame, vsd - 1)
}

/// Read a value from the unified `locals_cells_stack_w` at the given absolute index.
pub fn concrete_stack_value(frame: usize, abs_idx: usize) -> Option<PyObjectRef> {
    let frame_ptr = (frame != 0).then_some(frame as *const u8)?;
    let arr =
        unsafe { &*(frame_ptr.add(PYFRAME_LOCALS_CELLS_STACK_OFFSET) as *const PyObjectArray) };
    arr.as_slice().get(abs_idx).copied()
}

/// Return nlocals for the given frame (from the unified array's total length and code object).
pub(crate) fn concrete_nlocals(frame: usize) -> Option<usize> {
    let frame_ptr = (frame != 0).then_some(frame as *const u8)?;
    let w_code =
        unsafe { *(frame_ptr.add(crate::frame_layout::PYFRAME_CODE_OFFSET) as *const *const ()) };
    if w_code.is_null() {
        return None;
    }
    let raw_code = unsafe {
        pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
            as *const pyre_interpreter::CodeObject
    };
    Some(unsafe { (&(*raw_code).varnames).len() })
}

/// Return nlocals as the "locals_len" for virtualizable.
pub(crate) fn concrete_locals_len(frame: usize) -> Option<usize> {
    concrete_nlocals(frame)
}

/// Return the absolute valuestackdepth.
pub(crate) fn concrete_stack_depth(frame: usize) -> Option<usize> {
    let frame_ptr = (frame != 0).then_some(frame as *const u8)?;
    Some(unsafe { *(frame_ptr.add(PYFRAME_VALUESTACKDEPTH_OFFSET) as *const usize) })
}

pub(crate) fn concrete_namespace_slot(frame: usize, name: &str) -> Option<usize> {
    let frame_ptr = (frame != 0).then_some(frame as *const u8)?;
    let namespace_ptr =
        unsafe { *(frame_ptr.add(PYFRAME_NAMESPACE_OFFSET) as *const *mut PyNamespace) };
    let namespace = (!namespace_ptr.is_null()).then_some(unsafe { &*namespace_ptr })?;
    namespace.slot_of(name)
}

pub(crate) fn concrete_namespace_value(frame: usize, idx: usize) -> Option<PyObjectRef> {
    let frame_ptr = (frame != 0).then_some(frame as *const u8)?;
    let namespace_ptr =
        unsafe { *(frame_ptr.add(PYFRAME_NAMESPACE_OFFSET) as *const *mut PyNamespace) };
    let namespace = (!namespace_ptr.is_null()).then_some(unsafe { &*namespace_ptr })?;
    namespace.get_slot(idx)
}

pub(crate) fn namespace_slot_direct(ns: *mut PyNamespace, name: &str) -> Option<usize> {
    if ns.is_null() {
        return None;
    }
    unsafe { &*ns }.slot_of(name)
}

pub(crate) fn namespace_value_direct(ns: *mut PyNamespace, idx: usize) -> Option<PyObjectRef> {
    if ns.is_null() {
        return None;
    }
    unsafe { &*ns }.get_slot(idx)
}

pub(crate) fn record_current_state_guard(
    ctx: &mut TraceCtx,
    frame: OpRef,
    next_instr: OpRef,
    code: OpRef,
    stack_depth: OpRef,
    namespace: OpRef,
    locals: &[OpRef],
    stack: &[OpRef],
    opcode: OpCode,
    args: &[OpRef],
) {
    let mut fail_args = vec![frame, next_instr, code, stack_depth, namespace];
    fail_args.extend_from_slice(locals);
    fail_args.extend_from_slice(stack);
    let num_slots = fail_args.len() - crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
    let fail_arg_types = crate::virtualizable_gen::virt_live_value_types(num_slots);
    ctx.record_guard_typed_with_fail_args(opcode, args, fail_arg_types, &fail_args);
}

pub(crate) fn concrete_value_type(value: PyObjectRef) -> Type {
    if value.is_null() {
        return Type::Ref;
    }
    if !looks_like_heap_ref(value) {
        // Non-heap value (e.g. PY_NULL sentinel or leaked raw int)
        return Type::Int;
    }
    if unsafe { is_int(value) } {
        return Type::Int;
    }
    if unsafe { is_float(value) } {
        return Type::Float;
    }
    Type::Ref
}

/// pyre slots are always GCREF (Ref) at the concrete frame level.
/// Even W_IntObject values are stored as Ref pointers — the trace
/// unboxes via GetfieldGcPureI. adapt_live_values_to_trace_types
/// converts Ref→Int at compiled entry to match trace-internal types.
pub(crate) fn concrete_virtualizable_slot_type(_value: PyObjectRef) -> Type {
    Type::Ref
}

pub(crate) fn looks_like_heap_ref(value: PyObjectRef) -> bool {
    let addr = value as usize;
    let word_align = std::mem::align_of::<usize>() - 1;
    addr >= 0x1_0000 && addr < ((1u64 << 56) as usize) && (addr & word_align) == 0
}

pub(crate) fn extract_concrete_typed_value(slot_type: Type, value: PyObjectRef) -> Value {
    match slot_type {
        Type::Int => {
            if value.is_null() {
                Value::Int(0)
            } else if looks_like_heap_ref(value) && unsafe { is_int(value) } {
                Value::Int(unsafe { w_int_get_value(value) })
            } else {
                Value::Int(value as i64)
            }
        }
        Type::Float => {
            if value.is_null() {
                Value::Float(0.0)
            } else if looks_like_heap_ref(value) && unsafe { is_float(value) } {
                Value::Float(unsafe { pyre_object::floatobject::w_float_get_value(value) })
            } else {
                Value::Float(f64::from_bits(value as u64))
            }
        }
        Type::Ref | Type::Void => Value::Ref(majit_ir::GcRef(value as usize)),
    }
}

pub(crate) fn concrete_slot_types(
    frame: usize,
    num_locals: usize,
    valuestackdepth: usize,
) -> Vec<Type> {
    let stack_only = valuestackdepth.saturating_sub(num_locals);
    let mut types = Vec::with_capacity(num_locals + stack_only);
    for idx in 0..num_locals {
        types.push(
            concrete_stack_value(frame, idx)
                .map(concrete_virtualizable_slot_type)
                .unwrap_or(Type::Ref),
        );
    }
    for stack_idx in 0..stack_only {
        types.push(
            concrete_stack_value(frame, num_locals + stack_idx)
                .map(concrete_virtualizable_slot_type)
                .unwrap_or(Type::Ref),
        );
    }
    types
}

pub(crate) fn boxed_slot_i64_for_type(slot_type: Type, raw: i64) -> PyObjectRef {
    match slot_type {
        Type::Int => w_int_new(raw),
        Type::Float => pyre_object::floatobject::w_float_new(f64::from_bits(raw as u64)),
        Type::Ref | Type::Void => raw as PyObjectRef,
    }
}

pub(crate) fn boxed_slot_value_for_type(slot_type: Type, value: &Value) -> PyObjectRef {
    match (slot_type, value) {
        (Type::Int, Value::Int(v)) => w_int_new(*v),
        (Type::Float, Value::Float(v)) => pyre_object::floatobject::w_float_new(*v),
        (_, Value::Ref(r)) => r.as_usize() as PyObjectRef,
        (_, Value::Int(v)) => *v as PyObjectRef,
        (_, Value::Float(v)) => v.to_bits() as PyObjectRef,
        (_, Value::Void) => PY_NULL,
    }
}

/// RPython parity: virtualizable array slots are always GCREF.
/// When restoring from guard failure, treat ALL values as Ref (PyObjectRef).
/// Int values that look like heap pointers are used directly; otherwise box.
/// RPython parity: virtualizable array slots are always GCREF.
/// When restoring from guard failure, treat ALL values as Ref (PyObjectRef).
/// Int values that look like heap pointers are used directly (they are
/// PyObjectRef that the optimizer typed as Int during unboxing). Zero/null
/// values become PY_NULL. Only true small ints (non-zero, non-pointer)
/// need boxing.
pub(crate) fn boxed_slot_value_as_ref(value: &Value) -> PyObjectRef {
    match value {
        Value::Ref(r) => r.as_usize() as PyObjectRef,
        Value::Int(v) => {
            let addr = *v as usize;
            if addr == 0 {
                PY_NULL
            } else if addr >= 0x1_0000 && addr < ((1u64 << 56) as usize) && (addr & 7) == 0 {
                // Heap pointer — use as PyObjectRef directly
                *v as PyObjectRef
            } else {
                // Small int — box it
                w_int_new(*v)
            }
        }
        Value::Float(v) => pyre_object::floatobject::w_float_new(*v),
        Value::Void => PY_NULL,
    }
}

pub(crate) fn boxed_slot_value_from_runtime_kind(value: &Value) -> PyObjectRef {
    match value {
        Value::Int(v) => w_int_new(*v),
        Value::Float(v) => pyre_object::floatobject::w_float_new(*v),
        Value::Ref(r) => r.as_usize() as PyObjectRef,
        Value::Void => PY_NULL,
    }
}

/// virtualizable.py:126/139 parity: box value for frame array slot.
/// Frame array items (locals_cells_stack_w[*]) are declared as GCREF
/// (interp_jit.py:25). The optimizer may unbox ints/floats in fail_args;
/// this function re-boxes them for the frame. Ref values pass through.
pub(crate) fn virtualizable_box_value(value: &Value) -> PyObjectRef {
    match value {
        Value::Ref(r) => r.as_usize() as PyObjectRef,
        Value::Int(v) => w_int_new(*v),
        Value::Float(v) => pyre_object::floatobject::w_float_new(*v),
        Value::Void => PY_NULL,
    }
}

pub(crate) fn fail_arg_opref_for_typed_value(ctx: &mut TraceCtx, value: Value) -> OpRef {
    match value {
        Value::Int(v) => ctx.const_int(v),
        Value::Float(v) => ctx.const_int(v.to_bits() as i64),
        Value::Ref(r) => ctx.const_ref(r.as_usize() as i64),
        Value::Void => ctx.const_ref(PY_NULL as i64),
    }
}

pub fn pending_inline_result_from_concrete(
    result_type: Type,
    concrete_result: PyObjectRef,
) -> PendingInlineResult {
    match result_type {
        Type::Int => PendingInlineResult::Int(unsafe { w_int_get_value(concrete_result) }),
        Type::Float => PendingInlineResult::Float(unsafe {
            pyre_object::floatobject::w_float_get_value(concrete_result)
        }),
        Type::Ref | Type::Void => PendingInlineResult::Ref(concrete_result),
    }
}

pub fn materialize_pending_inline_result(result: PendingInlineResult) -> PyObjectRef {
    match result {
        PendingInlineResult::Ref(result) => result,
        PendingInlineResult::Int(value) => w_int_new(value),
        PendingInlineResult::Float(value) => pyre_object::floatobject::w_float_new(value),
    }
}

pub(crate) fn frame_callable_arg_types(nargs: usize) -> Vec<Type> {
    let mut types = Vec::with_capacity(2 + nargs);
    types.push(Type::Ref);
    types.push(Type::Ref);
    for _ in 0..nargs {
        types.push(Type::Ref);
    }
    types
}

pub(crate) fn one_arg_callee_frame_helper(
    arg_type: Type,
    is_self_recursive: bool,
) -> (*const (), Vec<Type>) {
    match (is_self_recursive, arg_type) {
        (true, Type::Int) => (
            crate::callbacks::get().jit_create_self_recursive_callee_frame_1_raw_int,
            vec![Type::Ref, Type::Int],
        ),
        (true, _) => (
            crate::callbacks::get().jit_create_self_recursive_callee_frame_1,
            vec![Type::Ref, Type::Ref],
        ),
        (false, Type::Int) => (
            crate::callbacks::get().jit_create_callee_frame_1_raw_int,
            vec![Type::Ref, Type::Ref, Type::Int],
        ),
        (false, _) => (
            crate::callbacks::get().jit_create_callee_frame_1,
            vec![Type::Ref, Type::Ref, Type::Ref],
        ),
    }
}

pub(crate) fn fail_arg_types_for_virtualizable_state(len: usize) -> Vec<Type> {
    let n = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
    crate::virtualizable_gen::virt_live_value_types(len.saturating_sub(n))
}

pub(crate) fn frame_entry_arg_types_from_slot_types(slot_types: &[Type]) -> Vec<Type> {
    if slot_types.is_empty() {
        vec![Type::Ref]
    } else {
        let mut types = crate::virtualizable_gen::virt_live_value_types(0);
        types.extend(slot_types.iter().copied());
        types
    }
}

pub(crate) fn pending_entry_slot_types_from_args(
    arg_types: &[Type],
    callee_nlocals: usize,
    callee_stack_only: usize,
) -> Vec<Type> {
    let mut slot_types = Vec::with_capacity(callee_nlocals + callee_stack_only);
    slot_types.extend(arg_types.iter().copied().take(callee_nlocals));
    while slot_types.len() < callee_nlocals {
        slot_types.push(Type::Ref);
    }
    while slot_types.len() < callee_nlocals + callee_stack_only {
        slot_types.push(Type::Ref);
    }
    slot_types
}

pub(crate) fn synthesize_fresh_callee_entry_args(
    ctx: &mut TraceCtx,
    callee_frame: OpRef,
    args: &[OpRef],
    callee_nlocals: usize,
    callee_code_ptr: usize,
    callee_namespace_ptr: usize,
) -> Vec<OpRef> {
    // Fresh entry: next_instr=0, valuestackdepth=nlocals (empty stack).
    // virtualizable.py:86: read_boxes reads ALL static fields from the heap.
    // code and namespace are immutable for the callee — use concrete values.
    let mut ca_args = vec![
        callee_frame,
        ctx.const_int(0),                           // next_instr
        ctx.const_ref(callee_code_ptr as i64),      // code
        ctx.const_int(callee_nlocals as i64),       // valuestackdepth
        ctx.const_ref(callee_namespace_ptr as i64), // namespace
    ];
    ca_args.extend(args.iter().copied().take(callee_nlocals));
    let null = ctx.const_ref(PY_NULL as i64);
    while ca_args.len() < crate::virtualizable_gen::NUM_SCALAR_INPUTARGS + callee_nlocals {
        ca_args.push(null);
    }
    ca_args
}

impl PyreSym {
    pub(crate) fn new_uninit(frame: OpRef) -> Self {
        Self {
            frame,
            symbolic_locals: Vec::new(),
            symbolic_stack: Vec::new(),
            symbolic_local_types: Vec::new(),
            symbolic_stack_types: Vec::new(),
            pending_next_instr: None,
            locals_cells_stack_array_ref: OpRef::NONE,
            valuestackdepth: 0,
            nlocals: 0,
            symbolic_initialized: false,
            bridge_local_oprefs: None,
            bridge_stack_oprefs: None,
            bridge_local_types: None,
            bridge_stack_types: None,
            vable_next_instr: OpRef::NONE,
            vable_code: OpRef::NONE,
            vable_valuestackdepth: OpRef::NONE,
            vable_namespace: OpRef::NONE,
            symbolic_namespace_slots: std::collections::HashMap::new(),
            vable_array_base: None,
            last_comparison_truth: None,
            last_comparison_concrete_truth: None,
            pending_branch_value: None,
            pending_branch_other_target: None,
            transient_value_types: std::collections::HashMap::new(),
            concrete_locals: Vec::new(),
            concrete_stack: Vec::new(),
            // jitcode and concrete_namespace initialized below
            jitcode: &NULL_JITCODE as *const JitCode,
            concrete_namespace: std::ptr::null_mut(),
            is_function_entry_trace: false,
            concrete_execution_context: std::ptr::null(),
            concrete_vable_ptr: std::ptr::null_mut(),
            pre_opcode_vsd: None,
            pre_opcode_stack: None,
            pre_opcode_stack_types: None,
            last_comparison_orgpc: None,
            last_comparison_pre_vsd: None,
            last_comparison_pre_stack: None,
            last_comparison_pre_stack_types: None,
            last_exc_value: std::ptr::null_mut(),
            class_of_last_exc_is_const: false,
            last_exc_box: OpRef::NONE,
            virtualref_boxes: Vec::new(),
        }
    }

    /// Initialize symbolic tracking state on first trace instruction.
    /// Subsequent calls are no-ops (state persists across instructions).
    pub(crate) fn init_symbolic(&mut self, ctx: &mut TraceCtx, concrete_frame: usize) {
        if self.symbolic_initialized {
            return;
        }
        self.is_function_entry_trace = ctx.header_pc == 0;
        let nlocals = concrete_nlocals(concrete_frame).unwrap_or(0);
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][init-sym] concrete_frame={:#x} nlocals={} vable_base={:?} header_pc={} func_entry={}",
                concrete_frame,
                nlocals,
                self.vable_array_base,
                ctx.header_pc,
                self.is_function_entry_trace
            );
        }
        let valuestackdepth = concrete_stack_depth(concrete_frame).unwrap_or(nlocals);
        let stack_only_depth = valuestackdepth.saturating_sub(nlocals);
        self.nlocals = nlocals;
        self.locals_cells_stack_array_ref = if self.vable_array_base.is_some() {
            OpRef::NONE
        } else {
            frame_locals_cells_stack_array(ctx, self.frame)
        };
        self.symbolic_locals = if let Some(ref overrides) = self.bridge_local_oprefs {
            // resume.py:1042 parity: bridge trace uses OpRefs derived from
            // rebuild_from_resumedata (Box(n) → bridge InputArg OpRef(n)).
            let mut locals = overrides.clone();
            locals.resize(nlocals, OpRef::NONE);
            locals
        } else if let Some(base) = self.vable_array_base {
            (0..nlocals).map(|i| OpRef(base + i as u32)).collect()
        } else {
            vec![OpRef::NONE; nlocals]
        };
        let inputarg_slot_types = self.vable_array_base.map(|base| {
            let inputarg_types = ctx.inputarg_types();
            let locals: Vec<Type> = (0..nlocals)
                .map(|i| {
                    inputarg_types
                        .get(base as usize + i)
                        .copied()
                        .unwrap_or(Type::Ref)
                })
                .collect();
            let stack: Vec<Type> = (0..stack_only_depth)
                .map(|i| {
                    inputarg_types
                        .get(base as usize + nlocals + i)
                        .copied()
                        .unwrap_or(Type::Ref)
                })
                .collect();
            (locals, stack)
        });
        if self.is_function_entry_trace {
            // RPython MIFrame parity: function-entry traces use concrete
            // value types (W_IntObject → Int). Always override.
            self.symbolic_local_types = (0..nlocals)
                .map(|i| {
                    concrete_stack_value(concrete_frame, i)
                        .map(concrete_value_type)
                        .unwrap_or(Type::Ref)
                })
                .collect();
        } else if let Some(ref overrides) = self.bridge_local_types {
            // resume.py:1245 decode_box parity: bridge inputarg types are
            // determined by which jitcode liveness list (live_i_regs /
            // live_r_regs / live_f_regs) the slot belongs to at the resume
            // PC. setup_bridge_sym walks these lists in encoder order and
            // populates `bridge_local_types[reg_idx] = Type::Int/Ref/Float`.
            // Use that override here so RETURN_VALUE sees the correct
            // type and skips the trace_guarded_int_payload unbox path.
            let mut types = overrides.clone();
            types.resize(nlocals, Type::Ref);
            self.symbolic_local_types = types;
        } else if let Some((ref local_types, _)) = inputarg_slot_types {
            // Bridge/root traces resume from the JIT inputarg contract.
            // Re-deriving slot kinds from the concrete frame would turn boxed
            // W_Int/W_Float locals back into raw Int/Float and break raw-array
            // stores that expect an explicit unbox step.
            self.symbolic_local_types = local_types.clone();
        } else if self.symbolic_local_types.len() != nlocals {
            self.symbolic_local_types = concrete_slot_types(concrete_frame, nlocals, nlocals);
        }
        self.symbolic_stack = if let Some(ref overrides) = self.bridge_stack_oprefs {
            let mut stack = overrides.clone();
            stack.resize(stack_only_depth, OpRef::NONE);
            stack
        } else if let Some(base) = self.vable_array_base {
            let stack_base = base + nlocals as u32;
            (0..stack_only_depth)
                .map(|i| OpRef(stack_base + i as u32))
                .collect()
        } else {
            vec![OpRef::NONE; stack_only_depth]
        };
        if let Some(ref overrides) = self.bridge_stack_types {
            let mut types = overrides.clone();
            types.resize(stack_only_depth, Type::Ref);
            self.symbolic_stack_types = types;
        } else if let Some((_, ref stack_types)) = inputarg_slot_types {
            self.symbolic_stack_types = stack_types.clone();
        } else if self.symbolic_stack_types.len() != stack_only_depth {
            self.symbolic_stack_types =
                concrete_slot_types(concrete_frame, nlocals, valuestackdepth)
                    .into_iter()
                    .skip(nlocals)
                    .collect();
        }
        self.pending_next_instr = None;
        self.valuestackdepth = valuestackdepth;
        // MIFrame concrete Box tracking: populate concrete value arrays
        // from the concrete frame snapshot (RPython MIFrame.setup_call parity).
        // Use concrete_value_from_slot to distinguish "real null pointer"
        // (Ref(PY_NULL)) from "untracked" (ConcreteValue::Null).
        self.concrete_locals = (0..nlocals)
            .map(|i| {
                let obj = concrete_stack_value(concrete_frame, i).unwrap_or(PY_NULL);
                concrete_value_from_slot(obj)
            })
            .collect();
        self.concrete_stack = (0..stack_only_depth)
            .map(|i| {
                let obj = concrete_stack_value(concrete_frame, nlocals + i).unwrap_or(PY_NULL);
                concrete_value_from_slot(obj)
            })
            .collect();
        // Extract frame metadata pointers for use without concrete_frame
        if concrete_frame != 0 {
            let frame = unsafe { &*(concrete_frame as *const pyre_interpreter::pyframe::PyFrame) };
            self.jitcode = jitcode_for(frame.code);
            self.concrete_namespace = frame.namespace;
            self.concrete_execution_context = frame.execution_context;
            self.concrete_vable_ptr = concrete_frame as *mut u8;
        }
        self.symbolic_initialized = true;
    }

    /// Stack-only depth (number of values on the operand stack).
    #[inline]
    pub fn stack_only_depth(&self) -> usize {
        self.valuestackdepth.saturating_sub(self.nlocals)
    }

    pub(crate) fn value_type_of(&self, value: OpRef) -> Type {
        if value.is_none() {
            return Type::Ref;
        }
        if let Some(value_type) = self.transient_value_types.get(&value).copied() {
            return value_type;
        }
        if let Some(idx) = self.symbolic_locals.iter().position(|&slot| slot == value) {
            return self
                .symbolic_local_types
                .get(idx)
                .copied()
                .unwrap_or(Type::Ref);
        }
        let stack_only = self.stack_only_depth().min(self.symbolic_stack.len());
        if let Some(idx) = self.symbolic_stack[..stack_only]
            .iter()
            .position(|&slot| slot == value)
        {
            return self
                .symbolic_stack_types
                .get(idx)
                .copied()
                .unwrap_or(Type::Ref);
        }
        Type::Ref
    }

    /// Read a concrete value from the Box arrays using an absolute
    /// unified-array index (0..nlocals = locals, nlocals.. = stack).
    pub(crate) fn concrete_value_at(&self, abs_idx: usize) -> ConcreteValue {
        if abs_idx < self.nlocals {
            self.concrete_locals
                .get(abs_idx)
                .copied()
                .unwrap_or(ConcreteValue::Null)
        } else {
            let stack_idx = abs_idx - self.nlocals;
            self.concrete_stack
                .get(stack_idx)
                .copied()
                .unwrap_or(ConcreteValue::Null)
        }
    }

    pub(crate) fn concrete_pyobj_at(&self, abs_idx: usize) -> PyObjectRef {
        self.concrete_value_at(abs_idx).to_pyobj()
    }
}

/// pyjitpl.py:1789-1814 opimpl_virtual_ref parity.
/// Creates a concrete JitVirtualRef via virtual_ref_during_tracing(),
/// records VIRTUAL_REF(box, cindex), and pushes
/// [virtualbox, vrefbox] onto virtualref_boxes.
///
/// Called from metainterp push_inline_frame (executioncontext.enter parity).
pub(crate) fn opimpl_virtual_ref(
    ctx: &mut TraceCtx,
    sym: &mut PyreSym,
    virtual_obj: OpRef,
    virtual_obj_ptr: usize,
) -> OpRef {
    // pyjitpl.py:1804: virtual_ref_during_tracing(virtual_obj)
    let vref_info = majit_metainterp::virtualref::VirtualRefInfo::new();
    let vref_ptr = vref_info.virtual_ref_during_tracing(virtual_obj_ptr as *mut u8);
    // pyjitpl.py:1805: cindex = ConstInt(len(virtualref_boxes) // 2)
    let cindex = ctx.const_int((sym.virtualref_boxes.len() / 2) as i64);
    // pyjitpl.py:1806: record VIRTUAL_REF(box, cindex)
    let vref = ctx.record_op(OpCode::VirtualRefR, &[virtual_obj, cindex]);
    // pyjitpl.py:1807: heapcache.new(resbox)
    ctx.heap_cache_mut().new_box(vref);
    // pyjitpl.py:1814: virtualref_boxes += [virtualbox, vrefbox]
    sym.virtualref_boxes.push((virtual_obj, virtual_obj_ptr));
    sym.virtualref_boxes.push((vref, vref_ptr as usize));
    vref
}

/// pyjitpl.py:1819-1831 opimpl_virtual_ref_finish parity.
/// Pops vrefbox and lastbox from virtualref_boxes (LIFO),
/// asserts `box == lastbox`, records VIRTUAL_REF_FINISH if still virtual.
///
/// Called from metainterp finishframe_inline/exception (executioncontext.leave parity).
pub(crate) fn opimpl_virtual_ref_finish(ctx: &mut TraceCtx, sym: &mut PyreSym, virtual_obj: OpRef) {
    if sym.virtualref_boxes.len() < 2 {
        return;
    }
    // pyjitpl.py:1821: vrefbox = virtualref_boxes.pop()
    let (vref_opref, vref_ptr) = sym.virtualref_boxes.pop().unwrap();
    // pyjitpl.py:1822: lastbox = virtualref_boxes.pop()
    let (lastbox_opref, _lastbox_ptr) = sym.virtualref_boxes.pop().unwrap();
    // pyjitpl.py:1823: assert box.getref_base() == lastbox.getref_base()
    debug_assert_eq!(
        virtual_obj, lastbox_opref,
        "opimpl_virtual_ref_finish: leaving frame box != top virtualref box"
    );
    // pyjitpl.py:1831: if is_virtual_ref(vref) → record VIRTUAL_REF_FINISH
    let vref_info = majit_metainterp::virtualref::VirtualRefInfo::new();
    let is_vref = vref_ptr != 0 && unsafe { vref_info.is_virtual_ref(vref_ptr as *const u8) };
    if is_vref {
        // pyjitpl.py:1832: VIRTUAL_REF_FINISH(vrefbox, nullbox)
        let null = ctx.const_ref(0);
        let _ = ctx.record_op(OpCode::VirtualRefFinish, &[vref_opref, null]);
    }
}

impl PyreJitState {
    /// RPython resume.py parity: restore virtualizable frame state directly
    /// from raw output buffer values, bypassing exit_layout type decoding.
    ///
    /// The optimizer may type virtualizable slots as Int (after unboxing),
    /// but the raw values in the output buffer are always the actual Python
    /// objects (PyObjectRef pointers). RPython uses FieldDescr at decode time
    /// to determine types; pyre uses the virtualizable layout to know that
    /// all array slots (locals + stack) are always Ref (GCREF).
    pub fn restore_virtualizable_from_raw(&mut self, raw_values: &[i64]) -> bool {
        if raw_values.is_empty() {
            return false;
        }
        // Fail_args layout: [frame, scalars..., active_locals..., active_stack...]
        // This is the guard failure path — raw_values carries only ACTIVE
        // slots, not the full backing array. RPython's full-array restore is
        // write_from_resume_data_partial (virtualizable.py:126-137), which
        // corresponds to import_virtualizable_state, not this function.
        let mut idx = crate::virtualizable_gen::virt_restore_scalars_raw(self, raw_values);

        let nlocals = self.local_count();
        let stack_only = self.valuestackdepth().saturating_sub(nlocals);
        for local_idx in 0..nlocals {
            if idx < raw_values.len() {
                let _ = self.set_local_at(local_idx, raw_values[idx] as PyObjectRef);
            }
            idx += 1;
        }
        for stack_idx in 0..stack_only {
            if idx < raw_values.len() {
                let _ = self.set_stack_at(stack_idx, raw_values[idx] as PyObjectRef);
            }
            idx += 1;
        }
        true
    }

    /// Returns true if the optimizer virtualizable mechanism is active.
    fn has_virtualizable_info(&self) -> bool {
        // pyre always uses virtualizable (JitDriverStaticData::with_virtualizable)
        true
    }

    fn frame_ptr(&self) -> Option<*mut u8> {
        (self.frame != 0).then_some(self.frame as *mut u8)
    }

    fn frame_array(&self, offset: usize) -> Option<&PyObjectArray> {
        let frame_ptr = self.frame_ptr()?;
        Some(unsafe { &*(frame_ptr.add(offset) as *const PyObjectArray) })
    }

    fn frame_array_mut(&mut self, offset: usize) -> Option<&mut PyObjectArray> {
        let frame_ptr = self.frame_ptr()?;
        Some(unsafe { &mut *(frame_ptr.add(offset) as *mut PyObjectArray) })
    }

    fn read_frame_usize(&self, offset: usize) -> Option<usize> {
        let frame_ptr = self.frame_ptr()?;
        Some(unsafe { *(frame_ptr.add(offset) as *const usize) })
    }

    fn write_frame_usize(&mut self, offset: usize, value: usize) -> bool {
        let Some(frame_ptr) = self.frame_ptr() else {
            return false;
        };
        unsafe {
            *(frame_ptr.add(offset) as *mut usize) = value;
        }
        true
    }

    fn locals_cells_stack_array(&self) -> Option<&PyObjectArray> {
        self.frame_array(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
    }

    fn locals_cells_stack_array_mut(&mut self) -> Option<&mut PyObjectArray> {
        self.frame_array_mut(PYFRAME_LOCALS_CELLS_STACK_OFFSET)
    }

    fn namespace_ptr(&self) -> Option<*mut PyNamespace> {
        let frame_ptr = self.frame_ptr()?;
        let namespace_ptr =
            unsafe { *(frame_ptr.add(PYFRAME_NAMESPACE_OFFSET) as *const *mut PyNamespace) };
        (!namespace_ptr.is_null()).then_some(namespace_ptr)
    }

    fn namespace_len(&self) -> usize {
        let Some(namespace_ptr) = self.namespace_ptr() else {
            return 0;
        };
        unsafe { (*namespace_ptr).len() }
    }

    fn namespace_keys(&self) -> Vec<String> {
        let Some(namespace_ptr) = self.namespace_ptr() else {
            return Vec::new();
        };
        let namespace = unsafe { &*namespace_ptr };
        let mut keys: Vec<String> = namespace.keys().cloned().collect();
        keys.sort();
        keys
    }

    fn restore_single_frame(&mut self, meta: &PyreMeta, values: &[i64]) {
        let Some(&frame) = values.first() else {
            return;
        };
        self.frame = frame as usize;
        if values.len() == 1 {
            return;
        }
        if meta.has_virtualizable {
            self.restore_virtualizable_i64(values);
        } else {
            let nlocals = self.local_count();
            let stack_only = self.valuestackdepth().saturating_sub(nlocals);
            let mut idx = 1;
            for local_idx in 0..nlocals {
                if idx < values.len() {
                    let slot_type = meta.slot_types.get(local_idx).copied().unwrap_or(Type::Ref);
                    let _ = self
                        .set_local_at(local_idx, boxed_slot_i64_for_type(slot_type, values[idx]));
                }
                idx += 1;
            }
            for i in 0..stack_only {
                if idx < values.len() {
                    let slot_type = meta
                        .slot_types
                        .get(nlocals + i)
                        .copied()
                        .unwrap_or(Type::Ref);
                    let _ = self.set_stack_at(i, boxed_slot_i64_for_type(slot_type, values[idx]));
                }
                idx += 1;
            }
        }
    }

    pub fn local_at(&self, idx: usize) -> Option<PyObjectRef> {
        self.locals_cells_stack_array()
            .and_then(|arr| arr.as_slice().get(idx).copied())
    }

    /// Number of local variable slots.
    pub fn local_count(&self) -> usize {
        concrete_nlocals(self.frame).unwrap_or(0)
    }

    pub fn set_local_at(&mut self, idx: usize, value: PyObjectRef) -> bool {
        let Some(arr) = self.locals_cells_stack_array_mut() else {
            return false;
        };
        let Some(slot) = arr.as_mut_slice().get_mut(idx) else {
            return false;
        };
        *slot = value;
        true
    }

    /// Read a stack value at stack-relative index `idx` (0-based from stack bottom).
    pub fn stack_at(&self, idx: usize) -> Option<PyObjectRef> {
        let nlocals = self.local_count();
        self.locals_cells_stack_array()
            .and_then(|arr| arr.as_slice().get(nlocals + idx).copied())
    }

    /// Total capacity of the unified array.
    pub fn array_capacity(&self) -> usize {
        self.locals_cells_stack_array()
            .map(PyObjectArray::len)
            .unwrap_or(0)
    }

    /// Set a stack value at stack-relative index `idx`.
    pub fn set_stack_at(&mut self, idx: usize, value: PyObjectRef) -> bool {
        let nlocals = self.local_count();
        let Some(arr) = self.locals_cells_stack_array_mut() else {
            return false;
        };
        let Some(slot) = arr.as_mut_slice().get_mut(nlocals + idx) else {
            return false;
        };
        *slot = value;
        true
    }

    // ── Heap accessors: single source of truth (RPython parity) ──
    // RPython's virtualizable IS the heap object — getattr/setattr go
    // directly to the heap.  These accessors do the same via frame_ptr.

    pub fn next_instr(&self) -> usize {
        self.read_frame_usize(PYFRAME_NEXT_INSTR_OFFSET)
            .expect("PyreJitState.frame must point to a valid PyFrame")
    }

    pub fn set_next_instr(&mut self, value: usize) {
        assert!(
            self.write_frame_usize(PYFRAME_NEXT_INSTR_OFFSET, value),
            "PyreJitState.frame must point to a valid PyFrame"
        );
    }

    pub fn valuestackdepth(&self) -> usize {
        self.read_frame_usize(PYFRAME_VALUESTACKDEPTH_OFFSET)
            .expect("PyreJitState.frame must point to a valid PyFrame")
    }

    pub fn set_valuestackdepth(&mut self, value: usize) {
        assert!(
            self.write_frame_usize(PYFRAME_VALUESTACKDEPTH_OFFSET, value),
            "PyreJitState.frame must point to a valid PyFrame"
        );
    }

    /// Read the code pointer (pycode) from the heap frame.
    pub fn code_as_usize(&self) -> usize {
        self.read_frame_usize(PYFRAME_CODE_OFFSET)
            .expect("PyreJitState.frame must point to a valid PyFrame")
    }

    /// Read the namespace pointer from the heap frame.
    pub fn namespace_as_usize(&self) -> usize {
        self.read_frame_usize(PYFRAME_NAMESPACE_OFFSET)
            .expect("PyreJitState.frame must point to a valid PyFrame")
    }

    /// Write the code pointer to the heap frame.
    /// virtualizable.py:101-107 write_boxes: ALL static fields written.
    pub fn set_code(&mut self, value: usize) {
        assert!(
            self.write_frame_usize(PYFRAME_CODE_OFFSET, value),
            "PyreJitState.frame must point to a valid PyFrame"
        );
    }

    /// Write the namespace pointer to the heap frame.
    pub fn set_namespace(&mut self, value: usize) {
        assert!(
            self.write_frame_usize(PYFRAME_NAMESPACE_OFFSET, value),
            "PyreJitState.frame must point to a valid PyFrame"
        );
    }

    /// Validate that the frame pointer is usable (fields readable, array present).
    fn validate_frame(&self) -> bool {
        self.read_frame_usize(PYFRAME_NEXT_INSTR_OFFSET).is_some()
            && self
                .read_frame_usize(PYFRAME_VALUESTACKDEPTH_OFFSET)
                .is_some()
            && self.locals_cells_stack_array().is_some()
    }

    /// Restore from virtualizable fail_args format:
    ///   [frame, scalars..., active_locals..., active_stack...]
    ///
    /// This is the guard failure / fail_args consumer path.
    /// Carries only ACTIVE slots from current_fail_args(), not the full
    /// backing array. The full-array restore path is
    /// import_virtualizable_state (virtualizable.py:126-137 parity).
    fn restore_virtualizable_i64(&mut self, values: &[i64]) {
        let mut idx = crate::virtualizable_gen::virt_restore_scalars_raw(self, values);

        let nlocals = self.local_count();
        for i in 0..nlocals {
            if idx < values.len() {
                let _ = self.set_local_at(i, values[idx] as PyObjectRef);
            }
            idx += 1;
        }

        let stack_only = self.valuestackdepth().saturating_sub(nlocals);
        for i in 0..stack_only {
            if idx < values.len() {
                let _ = self.set_stack_at(i, values[idx] as PyObjectRef);
            }
            idx += 1;
        }
    }

    fn import_virtualizable_state(
        &mut self,
        static_boxes: &[i64],
        array_boxes: &[Vec<i64>],
    ) -> bool {
        // virtualizable.py:126-137 write_from_resume_data_partial parity:
        // write ALL static fields to heap via VirtualizableInfo.
        let info = crate::frame_layout::build_pyframe_virtualizable_info();
        if !self.virt_import_static_boxes(&info, static_boxes) {
            return false;
        }

        // virtualizable.py:134-137: write array items to heap.
        // Validate array structure matches VirtualizableInfo.
        if array_boxes.len() != info.array_fields.len() {
            return false;
        }
        let Some(unified) = array_boxes.first() else {
            return info.array_fields.is_empty();
        };
        let Some(frame_arr) = self.locals_cells_stack_array_mut() else {
            return false;
        };
        if frame_arr.len() != unified.len() {
            return false;
        }
        for (dst, &src) in frame_arr.as_mut_slice().iter_mut().zip(unified) {
            *dst = src as PyObjectRef;
        }
        true
    }

    fn export_virtualizable_state(&self) -> (Vec<i64>, Vec<Vec<i64>>) {
        let info = crate::frame_layout::build_pyframe_virtualizable_info();
        self.virt_export_all(&info)
    }

    pub fn sync_from_virtualizable(&mut self, info: &VirtualizableInfo) -> bool {
        let _ = info;
        // Heap IS the source of truth. Just validate the frame is usable.
        self.validate_frame()
    }

    pub fn sync_to_virtualizable(&self, info: &VirtualizableInfo) -> bool {
        let Some(frame_ptr) = self.frame_ptr() else {
            return false;
        };
        // Heap is the single source of truth — no state-backed fields to
        // flush.  Only the vable_token needs resetting (virtualizable.py:218
        // force_now: set vable_token to TOKEN_NONE).
        unsafe {
            info.reset_vable_token(frame_ptr);
        }
        true
    }
}

/// resume.py:945-956 getvirtual_ptr parity, trace-time variant.
///
/// `materialize_virtual_from_rd` (eval.rs) does the same job at *runtime*
/// for the blackhole resume path: walks the `RdVirtualInfo` for `vidx`
/// and allocates a real heap object. This function does the same walk
/// at *trace* time, emitting `NEW_WITH_VTABLE` + `SETFIELD_GC` ops into
/// the bridge's trace via `ctx`. Returns the OpRef of the materialized
/// virtual.
///
/// Mirrors RPython's `ResumeDataBoxReader.consume_boxes` →
/// `rd_virtuals[i].allocate(decoder, i)` where `decoder.allocate_with_vtable`
/// is `metainterp.execute_new_with_vtable` (resume.py:1111-1112). The
/// recorded ops appear at the start of the bridge trace, before any
/// python interpreter opcodes are recorded — so when the bridge tracer
/// encounters the first `LOAD_FAST` of a previously-virtual local, it
/// sees the materialized OpRef in `bridge_local_oprefs` instead of
/// falling through to a stale vable-array read.
///
/// Currently unused: setup_bridge_sym still routes Virtual entries via
/// the existing vable scalar override path. The helper is added now so
/// future setup_bridge_sym work can drop the OpRef::NONE fallback for
/// RebuiltValue::Virtual without re-deriving the materialization logic.
#[allow(dead_code)]
fn materialize_bridge_virtual(
    ctx: &mut majit_metainterp::TraceCtx,
    vidx: usize,
    rd_virtuals: Option<&[majit_ir::RdVirtualInfo]>,
    resume_data: &majit_metainterp::ResumeDataResult,
    cache: &mut std::collections::HashMap<usize, OpRef>,
) -> OpRef {
    use majit_ir::OpCode;
    use majit_ir::resumedata::{TAG_CONST_OFFSET, TAGBOX, TAGCONST, TAGINT, TAGVIRTUAL, untag};

    // resume.py:951 virtuals_cache.get_ptr(index): hit → return cached.
    if let Some(&cached) = cache.get(&vidx) {
        return cached;
    }

    let Some(virtuals) = rd_virtuals else {
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][bridge-virtual] missing rd_virtuals (vidx={}), abort materialization",
                vidx
            );
        }
        return OpRef::NONE;
    };
    let Some(entry) = virtuals.get(vidx) else {
        return OpRef::NONE;
    };

    // resume.py:1556-1564 decode_box parity for fieldnums (i16 tagged).
    fn decode_fieldnum(
        ctx: &mut majit_metainterp::TraceCtx,
        tagged: i16,
        rd_virtuals: Option<&[majit_ir::RdVirtualInfo]>,
        resume_data: &majit_metainterp::ResumeDataResult,
        cache: &mut std::collections::HashMap<usize, OpRef>,
    ) -> OpRef {
        if tagged == majit_ir::resumedata::UNINITIALIZED_TAG {
            return OpRef::NONE;
        }
        let (val, tagbits) = untag(tagged);
        match tagbits {
            TAGBOX => {
                // resume.py:1556-1564 decode_box parity:
                //   if num < 0: num += len(liveboxes)
                //   return liveboxes[num]
                // Negative `val` is Python-style indexing into the parent
                // guard's liveboxes array (`num_failargs` long). For
                // bridges, the boxes are inputargs at OpRef(0..n_inputargs).
                let idx = if val < 0 {
                    val + resume_data.num_failargs
                } else {
                    val
                };
                if idx < 0 {
                    OpRef::NONE
                } else {
                    OpRef(idx as u32)
                }
            }
            TAGINT => ctx.const_int(val as i64),
            TAGCONST => {
                // resume.py:1247-1251 decode_box parity:
                //   if tag == TAGCONST:
                //       if tagged_eq(tagged, NULLREF):
                //           box = CONST_NULL
                //       else:
                //           box = self.consts[num - TAG_CONST_OFFSET]
                if tagged == majit_ir::resumedata::NULLREF {
                    return ctx.const_null();
                }
                let ci = (val - TAG_CONST_OFFSET) as usize;
                // resume_data.constants is the rebuilt mirror of `self.consts`,
                // keyed by the OpRef the encoder assigned at numbering time
                // (`OpRef::from_const(ci)`). Look it up by that key.
                let opref = majit_ir::OpRef::from_const(ci as u32);
                if let Some((_, raw, tp)) = resume_data
                    .constants
                    .iter()
                    .find(|(idx, _, _)| *idx == opref.0)
                {
                    match tp {
                        majit_ir::Type::Ref => ctx.const_ref(*raw),
                        majit_ir::Type::Float => ctx.const_float(*raw),
                        _ => ctx.const_int(*raw),
                    }
                } else {
                    opref
                }
            }
            TAGVIRTUAL => {
                materialize_bridge_virtual(ctx, val as usize, rd_virtuals, resume_data, cache)
            }
            _ => OpRef::NONE,
        }
    }

    // resume.py:612-760 dispatch by virtual kind.
    // RPython: rd_virtuals[index].allocate(self, index) — polymorphic on
    // the AbstractVirtualInfo subclass. Rust equivalent: match on
    // RdVirtualInfo enum variant.

    /// resume.py:591-603 AbstractVirtualStructInfo.setfields helper.
    /// Walks fielddescrs in lock-step with fieldnums, decoding each
    /// fieldnum and emitting SETFIELD_GC.
    fn setfields(
        ctx: &mut majit_metainterp::TraceCtx,
        struct_op: OpRef,
        fielddescrs: &[majit_ir::FieldDescrInfo],
        fieldnums: &[i16],
        parent_descr: majit_ir::DescrRef,
        rd_virtuals: Option<&[majit_ir::RdVirtualInfo]>,
        resume_data: &majit_metainterp::ResumeDataResult,
        cache: &mut std::collections::HashMap<usize, OpRef>,
    ) {
        for (fd_info, &fnum) in fielddescrs.iter().zip(fieldnums.iter()) {
            if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                continue;
            }
            let value = decode_fieldnum(ctx, fnum, rd_virtuals, resume_data, cache);
            if value.is_none() {
                continue;
            }
            let signed = matches!(
                fd_info.field_type,
                majit_ir::Type::Int | majit_ir::Type::Float
            );
            // RPython: decoder.setfield(struct, fieldnum, fielddescr).
            // fielddescr carries parent_descr so descr_index() resolves
            // to index_in_parent (small sequential) not stable_field_index (268M hash).
            let field_descr = crate::descr::make_field_descr_with_parent(
                parent_descr.clone(),
                fd_info.offset,
                fd_info.field_size,
                fd_info.field_type,
                signed,
            );
            ctx.record_op_with_descr(OpCode::SetfieldGc, &[struct_op, value], field_descr.clone());
            ctx.heap_cache_mut()
                .setfield_cached(struct_op, fd_info.index, value);
        }
    }

    match entry {
        // resume.py:612-621 VirtualInfo.allocate
        majit_ir::RdVirtualInfo::VirtualInfo {
            descr,
            fielddescrs,
            fieldnums,
            ..
        } => {
            let Some(size_descr) = descr.clone() else {
                return OpRef::NONE;
            };
            // resume.py:619 decoder.allocate_with_vtable(descr=self.descr)
            let new_op = ctx.record_op_with_descr(OpCode::NewWithVtable, &[], size_descr.clone());
            ctx.heap_cache_mut().new_object(new_op);
            // resume.py:620 decoder.virtuals_cache.set_ptr(index, struct)
            cache.insert(vidx, new_op);
            // resume.py:621 self.setfields(decoder, struct)
            setfields(
                ctx,
                new_op,
                fielddescrs,
                fieldnums,
                size_descr,
                rd_virtuals,
                resume_data,
                cache,
            );
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} VirtualInfo → OpRef({})",
                    vidx, new_op.0,
                );
            }
            new_op
        }
        // resume.py:628-637 VStructInfo.allocate
        majit_ir::RdVirtualInfo::VStructInfo {
            typedescr,
            fielddescrs,
            fieldnums,
            ..
        } => {
            let Some(struct_descr) = typedescr.clone() else {
                return OpRef::NONE;
            };
            // resume.py:635 decoder.allocate_struct(self.typedescr)
            let new_op = ctx.record_op_with_descr(OpCode::New, &[], struct_descr.clone());
            ctx.heap_cache_mut().new_object(new_op);
            // resume.py:636 decoder.virtuals_cache.set_ptr(index, struct)
            cache.insert(vidx, new_op);
            // resume.py:637 self.setfields(decoder, struct)
            setfields(
                ctx,
                new_op,
                fielddescrs,
                fieldnums,
                struct_descr,
                rd_virtuals,
                resume_data,
                cache,
            );
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} VStructInfo → OpRef({})",
                    vidx, new_op.0,
                );
            }
            new_op
        }
        // resume.py:649-671 AbstractVArrayInfo.allocate (clear=True or False)
        majit_ir::RdVirtualInfo::VArrayInfoClear {
            fieldnums,
            kind,
            descr_index,
            ..
        }
        | majit_ir::RdVirtualInfo::VArrayInfoNotClear {
            fieldnums,
            kind,
            descr_index,
            ..
        } => {
            let clear = matches!(entry, majit_ir::RdVirtualInfo::VArrayInfoClear { .. });
            let kind = *kind;
            let descr_index = *descr_index;
            let length = fieldnums.len();
            let len_ref = ctx.const_int(length as i64);
            // resume.py:653 decoder.allocate_array(length, arraydescr, self.clear)
            let alloc_opcode = if clear {
                OpCode::NewArrayClear
            } else {
                OpCode::NewArray
            };
            let array_descr = array_descr_for_kind(kind, descr_index);
            let new_op = ctx.record_op_with_descr(alloc_opcode, &[len_ref], array_descr.clone());
            ctx.heap_cache_mut().new_object(new_op);
            // resume.py:654 decoder.virtuals_cache.set_ptr(index, array)
            cache.insert(vidx, new_op);
            // resume.py:656-670 element loop: dispatch by arraydescr kind
            // NB. the check for the kind of array elements is moved out of the loop
            let set_opcode = match kind {
                0 => OpCode::SetarrayitemGc, // arraydescr.is_array_of_pointers()
                2 => OpCode::SetarrayitemGc, // arraydescr.is_array_of_floats() — TODO: SetarrayitemRaw/Float
                _ => OpCode::SetarrayitemGc, // int
            };
            for (i, &fnum) in fieldnums.iter().enumerate() {
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                let value = decode_fieldnum(ctx, fnum, rd_virtuals, resume_data, cache);
                if value.is_none() {
                    continue;
                }
                let idx_ref = ctx.const_int(i as i64);
                // resume.py:660/665/670 setarrayitem_{ref,float,int}
                ctx.record_op_with_descr(
                    set_opcode,
                    &[new_op, idx_ref, value],
                    array_descr.clone(),
                );
            }
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} VArrayInfo(clear={}) → OpRef({})",
                    vidx, clear, new_op.0,
                );
            }
            new_op
        }
        // resume.py:747-760 VArrayStructInfo.allocate
        majit_ir::RdVirtualInfo::VArrayStructInfo {
            descr_index,
            size,
            fielddescr_indices,
            field_types,
            item_size,
            field_offsets,
            field_sizes,
            fieldnums,
        } => {
            let len_ref = ctx.const_int(*size as i64);
            // resume.py:749 decoder.allocate_array(self.size, self.arraydescr, clear=True)
            let array_descr = array_descr_for_kind(0, *descr_index); // array-of-structs = ref-ptr array
            let new_op =
                ctx.record_op_with_descr(OpCode::NewArrayClear, &[len_ref], array_descr.clone());
            ctx.heap_cache_mut().new_object(new_op);
            // resume.py:751 decoder.virtuals_cache.set_ptr(index, array)
            cache.insert(vidx, new_op);
            // resume.py:752-759 nested (element, field) loop with setinteriorfield
            let num_fields = fielddescr_indices.len();
            let mut p = 0;
            for i in 0..*size {
                for j in 0..num_fields {
                    let fnum = fieldnums[p];
                    p += 1;
                    if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                        continue;
                    }
                    // resume.py:1200-1209 setinteriorfield: dispatch by field type
                    let value = decode_fieldnum(ctx, fnum, rd_virtuals, resume_data, cache);
                    if value.is_none() {
                        continue;
                    }
                    let idx_ref = ctx.const_int(i as i64);
                    // resume.py:1208 execute_setinteriorfield_gc(descr, array, ConstInt(index), fieldbox)
                    let field_descr = crate::descr::make_interior_field_descr(
                        *descr_index,
                        *item_size,
                        field_offsets.get(j).copied().unwrap_or(0),
                        field_sizes.get(j).copied().unwrap_or(8),
                        field_types.get(j).copied().unwrap_or(1),
                        fielddescr_indices.get(j).copied().unwrap_or(0),
                    );
                    ctx.record_op_with_descr(
                        OpCode::SetinteriorfieldGc,
                        &[new_op, idx_ref, value],
                        field_descr,
                    );
                }
            }
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} VArrayStructInfo(size={}, fields={}) → OpRef({})",
                    vidx, size, num_fields, new_op.0,
                );
            }
            new_op
        }
        // resume.py:700-709 VRawBufferInfo.allocate_int
        majit_ir::RdVirtualInfo::VRawBufferInfo {
            func,
            size,
            offsets,
            entry_sizes,
            entry_types,
            fieldnums,
        } => {
            // resume.py:703: buffer = decoder.allocate_raw_buffer(self.func, self.size)
            // resume.py:1124-1132: allocate_raw_buffer →
            //   calldescr = callinfo_for_oopspec(OS_RAW_MALLOC_VARSIZE_CHAR)
            //   execute_and_record_varargs(CALL_I, [ConstInt(func), ConstInt(size)], calldescr)
            let func_ref = ctx.const_int(*func);
            let size_ref = ctx.const_int(*size as i64);
            let calldescr = crate::descr::make_call_descr_int();
            let buffer = ctx.record_op_with_descr(OpCode::CallI, &[func_ref, size_ref], calldescr);
            // resume.py:704: decoder.virtuals_cache.set_int(index, buffer)
            cache.insert(vidx, buffer);
            // resume.py:705-708: for i in range(len(self.offsets)):
            //     offset = self.offsets[i]; descr = self.descrs[i]
            //     decoder.setrawbuffer_item(buffer, self.fieldnums[i], offset, descr)
            for (i, (&off, &fnum)) in offsets.iter().zip(fieldnums.iter()).enumerate() {
                if fnum == majit_ir::resumedata::UNINITIALIZED_TAG {
                    continue;
                }
                let item = decode_fieldnum(ctx, fnum, rd_virtuals, resume_data, cache);
                if item.is_none() {
                    continue;
                }
                // resume.py:1225-1234 setrawbuffer_item: dispatch by arraydescr kind
                let entry_type = entry_types.get(i).copied().unwrap_or(1);
                let item_size = entry_sizes.get(i).copied().unwrap_or(8);
                let tp = match entry_type {
                    0 => majit_ir::Type::Ref,
                    2 => majit_ir::Type::Float,
                    _ => majit_ir::Type::Int,
                };
                let store_descr =
                    crate::descr::make_array_descr(0, item_size, tp, tp == majit_ir::Type::Int);
                let offset_ref = ctx.const_int(off as i64);
                // resume.py:1233: execute_raw_store(arraydescr, buffer, ConstInt(offset), itembox)
                ctx.record_op_with_descr(
                    OpCode::RawStore,
                    &[buffer, offset_ref, item],
                    store_descr,
                );
            }
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} VRawBufferInfo(func={:#x}, size={}) → OpRef({})",
                    vidx, func, size, buffer.0,
                );
            }
            buffer
        }
        // resume.py:722-728 VRawSliceInfo.allocate_int
        majit_ir::RdVirtualInfo::VRawSliceInfo { offset, fieldnums } => {
            // resume.py:724: assert len(self.fieldnums) == 1
            assert!(
                fieldnums.len() == 1,
                "VRawSliceInfo must have exactly 1 fieldnum"
            );
            // resume.py:725: base_buffer = decoder.decode_int(self.fieldnums[0])
            let base_buffer = decode_fieldnum(ctx, fieldnums[0], rd_virtuals, resume_data, cache);
            // resume.py:726: buffer = decoder.int_add_const(base_buffer, self.offset)
            let offset_ref = ctx.const_int(*offset as i64);
            let buffer = ctx.record_op(OpCode::IntAdd, &[base_buffer, offset_ref]);
            // resume.py:727: decoder.virtuals_cache.set_int(index, buffer)
            cache.insert(vidx, buffer);
            if majit_metainterp::majit_log_enabled() {
                eprintln!(
                    "[jit][bridge-virtual] vidx={} VRawSliceInfo(offset={}) → OpRef({})",
                    vidx, offset, buffer.0,
                );
            }
            buffer
        }
        majit_ir::RdVirtualInfo::Empty => OpRef::NONE,
    }
}

impl JitState for PyreJitState {
    type Meta = PyreMeta;
    type Sym = PyreSym;
    type Env = PyreEnv;

    fn build_meta(&self, header_pc: usize, _env: &Self::Env) -> Self::Meta {
        let num_locals = self.local_count();
        let vsd = self.valuestackdepth();
        let slot_types = concrete_slot_types(self.frame, num_locals, vsd);
        PyreMeta {
            num_locals,
            ns_keys: self.namespace_keys(),
            valuestackdepth: vsd,
            has_virtualizable: self.has_virtualizable_info(),
            slot_types,
        }
    }

    fn extract_live(&self, _meta: &Self::Meta) -> Vec<i64> {
        self.extract_live_values(_meta)
            .into_iter()
            .map(|value| match value {
                Value::Int(v) => v,
                Value::Float(v) => v.to_bits() as i64,
                Value::Ref(r) => r.as_usize() as i64,
                Value::Void => 0,
            })
            .collect()
    }

    fn extract_live_values(&self, meta: &Self::Meta) -> Vec<Value> {
        crate::virtualizable_gen::virt_extract_live_values(
            self.frame,
            self.next_instr(),
            self.code_as_usize(),
            self.valuestackdepth(),
            self.namespace_as_usize(),
            meta.num_locals,
            meta.valuestackdepth,
            |i| self.local_at(i).unwrap_or(PY_NULL) as usize,
            |i| self.stack_at(i).unwrap_or(PY_NULL) as usize,
        )
    }

    /// history.py:_make_op parity: pyre pre-unboxes Python locals at
    /// the JIT entry boundary for function-entry traces, so the
    /// recorder records each inputarg with the post-unbox kind exactly
    /// as RPython's `wrap` produces typed FrontendOps from the start.
    /// Loop traces still call `extract_live_values` directly because
    /// the loop preamble peeling assumes boxed locals at the loop
    /// header (the recorder emits `guard_class` + `getfield_gc_pure_*`
    /// inside the trace).
    fn extract_live_values_for_entry(&self, meta: &Self::Meta) -> Vec<Value> {
        let mut values = self.extract_live_values(meta);
        let scalar_count = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        let total_slots = meta.valuestackdepth;
        for slot_idx in 0..total_slots {
            let value_idx = scalar_count + slot_idx;
            let Some(value) = values.get_mut(value_idx) else {
                break;
            };
            let raw = match value {
                Value::Ref(r) => r.as_usize() as PyObjectRef,
                _ => continue,
            };
            let slot_type = concrete_value_type(raw);
            *value = extract_concrete_typed_value(slot_type, raw);
        }
        values
    }

    fn live_value_types(&self, meta: &Self::Meta) -> Vec<Type> {
        crate::virtualizable_gen::virt_live_value_types(meta.slot_types.len())
    }

    fn create_sym(_meta: &Self::Meta, _header_pc: usize) -> Self::Sym {
        let mut sym = PyreSym::new_uninit(OpRef(0));
        sym.init_vable_indices();
        sym.nlocals = _meta.num_locals;
        sym.valuestackdepth = _meta.valuestackdepth;
        // RPython parity: all Python locals/stack are Ref (PyObjectRef).
        sym.symbolic_local_types = vec![Type::Ref; _meta.num_locals.min(_meta.slot_types.len())];
        sym.symbolic_stack_types =
            vec![Type::Ref; _meta.slot_types.len().saturating_sub(_meta.num_locals)];
        let stack_only = _meta.vable_stack_only_depth();
        sym.symbolic_stack = vec![OpRef::NONE; stack_only];
        sym.concrete_stack = vec![ConcreteValue::Null; stack_only];
        sym
    }

    fn driver_descriptor(&self, _meta: &Self::Meta) -> Option<JitDriverStaticData> {
        None
    }

    fn is_compatible(&self, meta: &Self::Meta) -> bool {
        // warmstate.py:503-511: RPython enters assembler unconditionally
        // when procedure_token exists. No next_instr check —
        // the compiled code's preamble handles entry from any PC.
        // Shape checks (nlocals, namespace) ensure frame layout matches.
        self.local_count() == meta.num_locals && self.namespace_len() == meta.ns_keys.len()
    }

    fn update_meta_for_bridge(meta: &mut Self::Meta, fail_arg_types: &[Type]) {
        meta.vable_update_vsd_from_len(
            fail_arg_types.len(),
            crate::virtualizable_gen::NUM_SCALAR_INPUTARGS,
        );
    }

    fn setup_bridge_sym(
        sym: &mut Self::Sym,
        ctx: &mut majit_metainterp::TraceCtx,
        resume_data: &majit_metainterp::ResumeDataResult,
        rd_virtuals: Option<&[majit_ir::RdVirtualInfo]>,
    ) {
        use majit_ir::resumedata::RebuiltValue;

        if resume_data.frames.is_empty() {
            return;
        }

        // resume.py:874-899 VirtualCache parity: per-bridge cache so shared
        // / recursive virtuals materialize exactly once.
        let mut virtuals_cache: std::collections::HashMap<usize, OpRef> =
            std::collections::HashMap::new();

        // Resolve a `RebuiltValue` to the OpRef the bridge tracer should
        // see. Box(n) → parent inputarg OpRef(n). Int/Const → constant
        // pool OpRef (cursor advances per call). Virtual(vidx) →
        // materialize the virtual at trace start via
        // `materialize_bridge_virtual`, mirroring resume.py:945-956
        // getvirtual_ptr → rd_virtuals[i].allocate →
        // metainterp.execute_new_with_vtable.
        let resolve = |ctx: &mut majit_metainterp::TraceCtx,
                       cursor: &mut usize,
                       cache: &mut std::collections::HashMap<usize, OpRef>,
                       v: &RebuiltValue|
         -> OpRef {
            match v {
                RebuiltValue::Box(n) => OpRef(*n as u32),
                RebuiltValue::Int(_) | RebuiltValue::Const(..) => {
                    let opref = majit_ir::OpRef::from_const(*cursor as u32);
                    *cursor += 1;
                    opref
                }
                RebuiltValue::Virtual(vidx) => {
                    materialize_bridge_virtual(ctx, *vidx as usize, rd_virtuals, resume_data, cache)
                }
                _ => OpRef::NONE,
            }
        };

        let nlocals = sym.nlocals;
        let mut bridge_locals = vec![OpRef::NONE; nlocals];
        // resume.py:1245 decode_box parity: each slot's `.type` is fixed
        // by which _callback_i/_callback_r/_callback_f the encoder dispatched.
        let mut bridge_local_types = vec![Type::Ref; nlocals];
        let parent_types = &resume_data.fail_arg_types;
        let type_for_value = |v: &RebuiltValue| -> Type {
            match v {
                RebuiltValue::Box(n) => parent_types.get(*n as usize).copied().unwrap_or(Type::Ref),
                RebuiltValue::Int(_) => Type::Int,
                RebuiltValue::Const(_, tp) => *tp,
                _ => Type::Ref,
            }
        };

        // pyjitpl.py:3281-3288 + virtualizable.py:126-137 parity: source
        // both vable scalar fields, frame locals AND stack temps from
        // `virtualizable_values` (the canonical layout
        // [frame_ptr, ni, code, vsd, ns, locals..., stack...]). Walking
        // vable_values with a single cursor keeps the constant-pool OpRefs
        // in lock-step with rebuild_from_resumedata's encoding order, and
        // ensures dead-at-PC locals (which liveness-driven `frame.values`
        // omits but the parent loop's LABEL still expects as inputargs)
        // are filled in. resume.py:945-956 getvirtual_ptr equivalent:
        // RebuiltValue::Virtual entries flow through
        // materialize_bridge_virtual via the shared virtuals_cache so
        // shared virtuals are emitted exactly once.
        let vvals = &resume_data.virtualizable_values;
        let mut vable_cursor: usize = 0;
        let num_scalars = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        for (i, v) in vvals.iter().enumerate() {
            let resolved = resolve(ctx, &mut vable_cursor, &mut virtuals_cache, v);
            match i {
                0 => {} // frame_ptr — implicit, no slot
                1 => sym.vable_next_instr = resolved,
                2 => sym.vable_code = resolved,
                3 => sym.vable_valuestackdepth = resolved,
                4 => sym.vable_namespace = resolved,
                _ => {
                    let off = i - num_scalars;
                    if off < nlocals {
                        bridge_locals[off] = resolved;
                        bridge_local_types[off] = type_for_value(v);
                    }
                    // Stack values beyond nlocals are intentionally dropped:
                    // the target loop header has stack_only=0.
                }
            }
        }
        if majit_metainterp::majit_log_enabled() {
            eprintln!(
                "[jit][bridge-sym] vable_values → bridge_locals={:?} types={:?}",
                bridge_locals, bridge_local_types,
            );
        }
        // Override sym.symbolic_locals AND sym.symbolic_local_types so
        // subsequent LOAD_FAST / RETURN_VALUE see the bridge inputarg
        // OpRefs and their resume-data-derived types, not the parent's
        // vable_array_base+i OpRefs / concrete_slot_types fallback that
        // init_symbolic seeded before setup_bridge_sym ran.
        //
        // pyre's start_bridge_tracing calls initialize_sym() (which runs
        // init_symbolic) BEFORE setup_bridge_sym, so init_symbolic sees
        // bridge_local_oprefs == None and falls into the vable_array_base
        // branch (init_vable_indices hard-codes vable_array_base = 5 for
        // pyre's 5-slot virtualizable header). That branch produces
        // OpRef(base+i) values from the PARENT trace's namespace, leaving
        // stale parent OpRefs in symbolic_locals after we set
        // bridge_local_oprefs here.
        //
        // The TYPES override is critical: without it, the bridge's
        // RETURN_VALUE handler sees `value_type(OpRef(1)) == Type::Ref`,
        // calls `trace_guarded_int_payload`, and emits a guard_class +
        // getfield_gc_pure_i sequence whose result the optimizer
        // constant-folds — producing a bridge that always returns
        // Finish(constant 0) instead of Finish(n).
        sym.symbolic_locals = {
            let mut locals = bridge_locals.clone();
            locals.resize(sym.nlocals, OpRef::NONE);
            locals
        };
        sym.symbolic_local_types = {
            let mut types = bridge_local_types.clone();
            types.resize(sym.nlocals, Type::Ref);
            types
        };
        // The bridge inputs do NOT have the 5-slot scalar header that
        // init_vable_indices assumes. Clear vable_array_base so any later
        // LOAD_FAST falling through to the vable_array_base branch uses
        // the heap-array path instead of synthesizing parent OpRefs.
        sym.vable_array_base = None;
        // Bridge stack: the target loop header has stack_only=0.
        sym.symbolic_stack = Vec::new();
        sym.symbolic_stack_types = Vec::new();
        sym.valuestackdepth = sym.nlocals;
        sym.bridge_local_oprefs = Some(bridge_locals);
        sym.bridge_stack_oprefs = None;
        sym.bridge_local_types = Some(bridge_local_types);
        sym.bridge_stack_types = None;
    }

    /// resume.py:1042-1057 rebuild_from_resumedata parity.
    ///
    /// Decodes rd_numb via `majit_ir::resumedata::rebuild_from_numbering`.
    /// Frame box counts come from jitcode liveness (jitcode.position_info)
    /// at the frame's resume pc — the same data the encoder uses via
    /// `get_list_of_active_boxes`.
    fn rebuild_from_resumedata(
        _meta: &mut Self::Meta,
        fail_arg_types: &[Type],
        rd_numb: Option<&[u8]>,
        rd_consts: Option<&[(i64, Type)]>,
    ) -> Option<majit_metainterp::ResumeDataResult> {
        use majit_ir::resumedata::{RebuiltValue, rebuild_from_numbering};

        let rd_numb = rd_numb?;
        let rd_consts = rd_consts.unwrap_or(&[]);

        // resume.py:1049-1055 parity: consume_boxes(f.get_current_position_info())
        // RPython uses jitcode liveness via get_current_position_info; majit
        // routes the same lookup through `frame_value_count_at`.
        let cb = crate::state::frame_value_count_at;
        let (num_failargs, vable_values, vref_values, frames) =
            rebuild_from_numbering(rd_numb, rd_consts, Some(&cb));

        if frames.is_empty() {
            return None;
        }

        // resume.py:1245 decode_box parity: TAGCONST and TAGINT entries
        // become ConstPtr/ConstInt boxes in RPython. Rust needs explicit
        // constant pool entries for the optimizer to see them.
        // All sections (vable, vref, frame) are treated uniformly.
        let mut constants = Vec::new();
        let mut const_idx: u32 = 0;
        for value in vable_values.iter().chain(vref_values.iter()) {
            match value {
                RebuiltValue::Const(raw, tp) => {
                    constants.push((majit_ir::OpRef::from_const(const_idx).0, *raw, *tp));
                    const_idx += 1;
                }
                RebuiltValue::Int(v) => {
                    constants.push((
                        majit_ir::OpRef::from_const(const_idx).0,
                        *v as i64,
                        Type::Int,
                    ));
                    const_idx += 1;
                }
                _ => {}
            }
        }
        // resume.py:1245 decode_box parity: TAGINT → ConstInt(num) regardless
        // of whether it came from vable, vref, or a frame slot.
        for frame in &frames {
            for value in &frame.values {
                match value {
                    RebuiltValue::Const(raw, tp) => {
                        constants.push((majit_ir::OpRef::from_const(const_idx).0, *raw, *tp));
                        const_idx += 1;
                    }
                    RebuiltValue::Int(v) => {
                        constants.push((
                            majit_ir::OpRef::from_const(const_idx).0,
                            *v as i64,
                            Type::Int,
                        ));
                        const_idx += 1;
                    }
                    _ => {}
                }
            }
        }

        Some(majit_metainterp::ResumeDataResult {
            frames,
            virtualizable_values: vable_values,
            virtualref_values: vref_values,
            constants,
            // resume.py:1245 decode_box parity: propagate parent guard's
            // fail_arg_types so setup_bridge_sym can type bridge inputarg
            // slots correctly.
            fail_arg_types: fail_arg_types.to_vec(),
            // resume.py:1042 num_failargs from rd_numb header. Used by
            // bridge virtual materialization (resume.py:1556-1564 decode_box
            // negative-index normalization: `num + len(liveboxes)`).
            num_failargs,
        })
    }

    /// pyjitpl.py:2982 get_procedure_token: compute green key for a PC.
    fn green_key_for_pc(&self, pc: usize) -> Option<u64> {
        let frame_ptr = self.frame as *const pyre_interpreter::pyframe::PyFrame;
        if frame_ptr.is_null() {
            return None;
        }
        let code = unsafe { (*frame_ptr).code };
        Some(crate::driver::make_green_key(code, pc))
    }

    fn code_ptr(&self) -> usize {
        let frame_ptr = self.frame as *const pyre_interpreter::pyframe::PyFrame;
        if frame_ptr.is_null() {
            return 0;
        }
        unsafe { (*frame_ptr).code as usize }
    }

    fn update_meta_for_cut(meta: &mut Self::Meta, header_pc: usize, original_box_types: &[Type]) {
        // Update valuestackdepth from the merge point's box layout.
        // Layout: [Ref(frame), Int(ni), Ref(code), Int(vsd), Ref(ns), locals..., stack...]
        // PyreMeta.valuestackdepth is ABSOLUTE (nlocals + stack_items).
        use crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        if original_box_types.len() >= NUM_SCALAR_INPUTARGS {
            let new_vsd = original_box_types.len() - NUM_SCALAR_INPUTARGS;
            if new_vsd < meta.valuestackdepth && meta.slot_types.len() > new_vsd {
                meta.slot_types.truncate(new_vsd);
            } else if new_vsd > meta.valuestackdepth && meta.slot_types.len() < new_vsd {
                meta.slot_types.resize(new_vsd, Type::Ref);
            }
            meta.valuestackdepth = new_vsd;
        }
    }

    fn build_meta_from_merge_point(
        provisional: &PyreMeta,
        header_pc: usize,
        original_box_types: &[Type],
    ) -> PyreMeta {
        use crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        // RPython parity: Python locals/stack are always Ref.
        let slot_types = if original_box_types.len() >= NUM_SCALAR_INPUTARGS {
            vec![Type::Ref; original_box_types.len() - NUM_SCALAR_INPUTARGS]
        } else {
            Vec::new()
        };
        let vsd = if original_box_types.len() >= NUM_SCALAR_INPUTARGS {
            original_box_types.len() - NUM_SCALAR_INPUTARGS
        } else {
            provisional.valuestackdepth
        };
        PyreMeta {
            num_locals: provisional.num_locals,
            ns_keys: provisional.ns_keys.clone(),
            valuestackdepth: vsd,
            has_virtualizable: provisional.has_virtualizable,
            slot_types,
        }
    }

    fn restore(&mut self, meta: &Self::Meta, values: &[i64]) {
        if values.is_empty() {
            return;
        }

        // Multi-frame format: [num_frames, size_0, data_0..., size_1, data_1...]
        // Detect: values[0] is a small number (1-10) = frame count
        // Legacy: values[0] is a large pointer
        let first = values[0];
        if first >= 1 && first <= 10 && values.len() > 2 {
            let _num_frames = first as usize;
            let outer_size = values[1] as usize;
            if outer_size > 0 && 2 + outer_size <= values.len() {
                // Restore outermost frame only.
                // Inner frame guard failure → interpreter re-executes callee call.
                self.restore_single_frame(meta, &values[2..2 + outer_size]);
                return;
            }
        }

        // Legacy single-frame format
        self.restore_single_frame(meta, values);
    }

    fn restore_values(&mut self, meta: &Self::Meta, values: &[Value]) {
        let Some(frame) = values.first() else {
            return;
        };
        self.frame = value_to_usize(frame);
        if majit_metainterp::majit_log_enabled() {
            let arg0 = self.local_at(0).and_then(|value| {
                if value.is_null() || !unsafe { pyre_object::pyobject::is_int(value) } {
                    return None;
                }
                Some(unsafe { pyre_object::intobject::w_int_get_value(value) })
            });
            eprintln!(
                "[jit][restore_values] before arg0={:?} meta.vsd={} has_vable={} values={:?}",
                arg0, meta.valuestackdepth, meta.has_virtualizable, values
            );
        }
        if values.len() == 1 {
            return;
        }

        if meta.has_virtualizable {
            // next_instr is already synced to the PyFrame heap by the
            // compiled code's virtualizable sync before JUMP.
            self.set_valuestackdepth(meta.valuestackdepth);
            let nlocals = self.local_count();
            let stack_only = meta.valuestackdepth.saturating_sub(nlocals);
            let mut idx = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
            for local_idx in 0..nlocals {
                if let Some(value) = values.get(idx) {
                    let slot_type = meta.slot_types.get(local_idx).copied().unwrap_or(Type::Ref);
                    let _ =
                        self.set_local_at(local_idx, boxed_slot_value_for_type(slot_type, value));
                }
                idx += 1;
            }
            for i in 0..stack_only {
                if let Some(value) = values.get(idx) {
                    let slot_type = meta
                        .slot_types
                        .get(nlocals + i)
                        .copied()
                        .unwrap_or(Type::Ref);
                    let _ = self.set_stack_at(i, boxed_slot_value_for_type(slot_type, value));
                }
                idx += 1;
            }
        } else {
            let nlocals = self.local_count();
            let stack_only_depth = meta.valuestackdepth.saturating_sub(nlocals);
            let mut idx = 1;
            for local_idx in 0..nlocals {
                let slot_type = meta.slot_types.get(local_idx).copied().unwrap_or(Type::Ref);
                let _ = self.set_local_at(
                    local_idx,
                    boxed_slot_value_for_type(slot_type, &values[idx]),
                );
                idx += 1;
            }
            for i in 0..stack_only_depth {
                let slot_type = meta
                    .slot_types
                    .get(nlocals + i)
                    .copied()
                    .unwrap_or(Type::Ref);
                let _ = self.set_stack_at(i, boxed_slot_value_for_type(slot_type, &values[idx]));
                idx += 1;
            }
            self.set_valuestackdepth(meta.valuestackdepth);
        }
        if majit_metainterp::majit_log_enabled() {
            let arg0 = self.local_at(0).and_then(|value| {
                if value.is_null() || !unsafe { pyre_object::pyobject::is_int(value) } {
                    return None;
                }
                Some(unsafe { pyre_object::intobject::w_int_get_value(value) })
            });
            eprintln!(
                "[jit][restore_values] after arg0={:?} ni={} vsd={}",
                arg0,
                self.next_instr(),
                self.valuestackdepth()
            );
        }
    }

    fn restore_guard_failure_values(
        &mut self,
        meta: &Self::Meta,
        values: &[Value],
        _exception: &majit_metainterp::blackhole::ExceptionState,
    ) -> bool {
        if !meta.has_virtualizable {
            self.restore_values(meta, values);
            return true;
        }

        let Some(frame) = values.first() else {
            return false;
        };
        self.frame = value_to_usize(frame);
        if values.len() == 1 {
            return self.validate_frame();
        }

        // virtualizable.py:126-137 write_from_resume_data_partial:
        // ALL static fields in unroll_static_fields order.
        if let Some(ni) = values
            .get(crate::virtualizable_gen::SYM_NEXT_INSTR_IDX as usize)
            .map(value_to_usize)
        {
            self.set_next_instr(ni);
        }
        if let Some(code) = values
            .get(crate::virtualizable_gen::SYM_CODE_IDX as usize)
            .map(value_to_usize)
        {
            self.set_code(code);
        }
        if let Some(vsd) = values
            .get(crate::virtualizable_gen::SYM_VALUESTACKDEPTH_IDX as usize)
            .map(value_to_usize)
        {
            // Sanity check: vsd must not exceed the frame's total capacity
            // (nlocals + stacksize). A bad vsd from stale guard recovery
            // values can corrupt the frame and crash in as_mut_slice.
            let max_vsd = self
                .locals_cells_stack_array()
                .map(|arr| arr.len())
                .unwrap_or(0);
            let safe_vsd = vsd.min(max_vsd);
            self.set_valuestackdepth(safe_vsd);
        }
        if let Some(ns) = values
            .get(crate::virtualizable_gen::SYM_NAMESPACE_IDX as usize)
            .map(value_to_usize)
        {
            self.set_namespace(ns);
        }

        let nlocals = self.local_count();
        let stack_only = self.valuestackdepth().saturating_sub(nlocals);
        // resume.py:1077 consume_boxes(info, boxes_i, boxes_r, boxes_f) parity:
        // RPython's consume_boxes uses position_info (liveness at resume PC)
        // to map compact active_boxes back to register indices. Dead registers
        // are skipped in the compact array — only live registers advance the
        // value index.
        //
        // values[3..] = compact active_boxes from get_list_of_active_boxes,
        // which filters by liveness. Use the same liveness table to restore.
        let raw_code_ptr = if self.frame != 0 {
            let w_code = unsafe {
                *((self.frame as *const u8).add(crate::frame_layout::PYFRAME_CODE_OFFSET)
                    as *const *const ())
            };
            if !w_code.is_null() {
                unsafe {
                    pyre_interpreter::w_code_get_ptr(w_code as pyre_object::PyObjectRef)
                        as *const pyre_interpreter::CodeObject
                }
            } else {
                std::ptr::null()
            }
        } else {
            std::ptr::null()
        };
        let live = if !raw_code_ptr.is_null() {
            Some(liveness_for(raw_code_ptr))
        } else {
            None
        };
        // resume.py:1383: info = blackholeinterp.get_current_position_info()
        // blackhole.py:337: position was set by setposition(jitcode, pc) where
        // pc comes from rd_numb — the same orgpc used by get_list_of_active_boxes.
        // next_instr = orgpc + 1 + caches, which may have different liveness.
        let live_pc = self.resume_pc.take().unwrap_or_else(|| self.next_instr());
        let mut idx = crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
        for local_idx in 0..nlocals {
            // resume.py:1077: only live registers consume a value from the
            // compact array. Dead registers keep their previous contents.
            let is_live = live.map_or(true, |lv| lv.is_local_live(live_pc, local_idx));
            if is_live {
                if let Some(value) = values.get(idx) {
                    let boxed = virtualizable_box_value(value);
                    let _ = self.set_local_at(local_idx, boxed);
                }
                idx += 1;
            }
        }
        for stack_idx in 0..stack_only {
            let is_live = live.map_or(true, |lv| lv.is_stack_live(live_pc, stack_idx));
            if is_live {
                if let Some(value) = values.get(idx) {
                    let boxed = virtualizable_box_value(value);
                    let _ = self.set_stack_at(stack_idx, boxed);
                }
                idx += 1;
            }
        }

        // Clear stale slots beyond valuestackdepth (blackhole fresh frame parity).
        let vsd = self.valuestackdepth();
        if let Some(arr) = self.locals_cells_stack_array_mut() {
            for i in vsd..arr.len() {
                arr[i] = pyre_object::PY_NULL;
            }
        }
        true
    }

    /// resume.py:1077 consume_boxes(info, boxes_i, boxes_r, boxes_f) parity:
    /// Return the type of each slot in the resumed frame section.
    /// In pyre, all frame slots are PyObjectRef (GCREF), so every slot
    /// is Ref. RPython uses typed registers (boxes_i/r/f) but pyre's
    /// virtualizable array is uniformly Ref.
    fn reconstructed_frame_value_types(
        &self,
        meta: &Self::Meta,
        _frame_index: usize,
        _total_frames: usize,
        _frame_pc: u64,
    ) -> Option<Vec<Type>> {
        // resume.py:1077: consume_boxes fills boxes_i/boxes_r/boxes_f.
        // pyre frame slots (locals_cells_stack_w) are all GCREF (Ref).
        let nlocals = meta.num_locals;
        let stack_only = self.valuestackdepth().saturating_sub(nlocals);
        // Header [frame_ptr=Ref, ni=Int, code=Ref, vsd=Int, ns=Ref] + all locals/stack as Ref.
        Some(crate::virtualizable_gen::virt_live_value_types(
            nlocals + stack_only,
        ))
    }

    /// resume.py:1049 parity: restore frame register state from decoded values.
    /// resume.py:1077 consume_boxes → _prepare_next_section → enumerate_vars:
    /// each callback_r writes a ref value to the register at the given index.
    /// In pyre, this writes values to the PyFrame's locals/stack via the
    /// virtualizable mechanism (restore_virtualizable_state handles the
    /// full [frame, ni, code, vsd, ns, locals..., stack...] layout).
    fn restore_reconstructed_frame_values(
        &mut self,
        meta: &Self::Meta,
        _frame_index: usize,
        _total_frames: usize,
        _frame_pc: u64,
        values: &[Value],
        exception: &majit_metainterp::blackhole::ExceptionState,
    ) -> bool {
        // resume.py:1077 consume_boxes parity: write values to the frame.
        // blackhole.py:337: setposition(jitcode, pc) before consume_one_section —
        // frame_pc from rd_numb is the liveness PC (orgpc).
        self.resume_pc = Some(_frame_pc as usize);
        self.restore_guard_failure_values(meta, values, exception)
    }

    /// blackhole.py:1800 parity: multi-frame support.
    fn supports_multi_frame_restore(&self) -> bool {
        true
    }

    /// blackhole.py:1333 parity: push outer frame for chain.
    /// Multi-frame recovery handled by blackhole chain in call_jit.rs
    /// which receives all frame sections in the typed vector.
    fn push_caller_frame(
        &mut self,
        _meta: &Self::Meta,
        _frame_index: usize,
        _total_frames: usize,
        _values: &[Value],
        _pc: u64,
        _jitcode_index: i32,
    ) -> bool {
        true
    }

    /// blackhole.py:1760 parity: frame transition via chain.
    fn pop_to_caller_frame(&mut self, _meta: &Self::Meta) -> bool {
        false // Blackhole chain handles this directly.
    }

    fn virtualizable_heap_ptr(
        &self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
    ) -> Option<*mut u8> {
        crate::virtualizable_gen::virt_heap_ptr(self, _virtualizable)
    }

    fn virtualizable_array_lengths(
        &self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
    ) -> Option<Vec<usize>> {
        crate::virtualizable_gen::virt_array_lengths(self, _virtualizable, _info)
    }

    fn sync_virtualizable_before_jit(
        &mut self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        info: &VirtualizableInfo,
    ) -> bool {
        // Heap is source of truth — just validate the frame is usable.
        if !self.validate_frame() {
            return false;
        }
        // virtualizable.py:170 force_token_before_residual_call parity:
        // clear vable_token so the JIT knows the virtualizable is synced.
        if let Some(frame_ptr) = self.frame_ptr() {
            unsafe { info.reset_vable_token(frame_ptr) };
        }
        true
    }

    fn sync_virtualizable_after_jit(
        &mut self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        info: &VirtualizableInfo,
    ) {
        let Some(frame_ptr) = self.frame_ptr() else {
            return;
        };
        // Heap is source of truth — nothing to sync. Just reset token.
        unsafe {
            info.reset_vable_token(frame_ptr);
        }
    }

    fn sync_virtualizable_before_residual_call(&self, ctx: &mut TraceCtx) {
        crate::virtualizable_gen::virt_sync_before_residual(self, ctx)
    }

    fn sync_virtualizable_after_residual_call(
        &self,
        _ctx: &mut TraceCtx,
    ) -> ResidualVirtualizableSync {
        crate::virtualizable_gen::virt_sync_after_residual(self, _ctx)
    }

    fn import_virtualizable_boxes(
        &mut self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
        static_boxes: &[i64],
        array_boxes: &[Vec<i64>],
    ) -> bool {
        self.import_virtualizable_state(static_boxes, array_boxes)
    }

    fn export_virtualizable_boxes(
        &self,
        _meta: &Self::Meta,
        _virtualizable: &str,
        _info: &VirtualizableInfo,
    ) -> Option<(Vec<i64>, Vec<Vec<i64>>)> {
        Some(self.export_virtualizable_state())
    }

    fn collect_jump_args(sym: &Self::Sym) -> Vec<OpRef> {
        sym.vable_collect_jump_args()
    }

    fn collect_typed_jump_args(sym: &Self::Sym) -> Vec<(OpRef, Type)> {
        sym.vable_collect_typed_jump_args()
    }

    fn validate_close(sym: &Self::Sym, meta: &Self::Meta) -> bool {
        let _ = (sym, meta);
        true
    }

    fn validate_close_with_jump_args(
        _sym: &Self::Sym,
        meta: &Self::Meta,
        jump_args: &[OpRef],
    ) -> bool {
        let _ = meta;
        // pyre does not close loops by re-reading symbolic stack state from
        // the trace header.  Instead it materializes explicit jump args from
        // the concrete frame-backed virtualizable state at the merge point:
        //   [frame, next_instr, valuestackdepth, locals..., stack...]
        //
        // Retraces/bridges may start from a transient stackful state
        // (e.g. direct-call trace-through in progress) and still legally
        // jump back to a target loop whose merge-point stack is smaller.
        // RPython closes those traces against the target token contract, not
        // the retrace entry state's stack depth.  So for pyre's explicit
        // jump-arg model, the trace-start `meta.valuestackdepth` is not a
        // sound validator here.
        jump_args.len() >= crate::virtualizable_gen::NUM_SCALAR_INPUTARGS
    }

    /// RPython resume.py: materialize a virtual object from resume data.
    ///
    /// Called during guard failure recovery when the optimizer kept an
    /// object virtual (New + SetfieldGc eliminated). The resume mechanism
    /// reconstructs the object so the interpreter can use it.
    fn materialize_virtual_ref(
        &mut self,
        _meta: &Self::Meta,
        _virtual_index: usize,
        materialized: &majit_metainterp::resume::MaterializedVirtual,
    ) -> Option<majit_ir::GcRef> {
        self.materialize_virtual_ref_from_layout(materialized, &[])
    }

    fn materialize_virtual_ref_with_refs(
        &mut self,
        _meta: &Self::Meta,
        _virtual_index: usize,
        materialized: &majit_metainterp::resume::MaterializedVirtual,
        materialized_refs: &[Option<majit_ir::GcRef>],
    ) -> Option<majit_ir::GcRef> {
        self.materialize_virtual_ref_from_layout(materialized, materialized_refs)
    }
}

impl PyreJitState {
    fn materialize_virtual_ref_from_layout(
        &mut self,
        materialized: &majit_metainterp::resume::MaterializedVirtual,
        materialized_refs: &[Option<majit_ir::GcRef>],
    ) -> Option<majit_ir::GcRef> {
        use majit_metainterp::resume::MaterializedVirtual;

        match materialized {
            MaterializedVirtual::Struct { fields, .. }
            | MaterializedVirtual::Obj { fields, .. } => {
                self.materialize_virtual_object(fields, materialized_refs)
            }
            MaterializedVirtual::RawBuffer { size, entries } => {
                materialize_virtual_raw_buffer(*size, entries, materialized_refs)
            }
            _ => None,
        }
    }

    /// resume.py:597-602 AbstractVirtualStructInfo.setfields parity:
    /// ALL traced fields are restored, including w_class.
    fn materialize_virtual_object(
        &mut self,
        fields: &[(u32, majit_metainterp::resume::MaterializedValue)],
        materialized_refs: &[Option<majit_ir::GcRef>],
    ) -> Option<majit_ir::GcRef> {
        let mut ob_type: usize = 0;
        let mut w_class: usize = 0;
        let mut int_payload: i64 = 0;
        let mut list_items_ptr: usize = 0;
        let mut list_items_len: usize = 0;

        use pyre_object::pyobject::{OB_TYPE_OFFSET, W_CLASS_OFFSET};
        // PyObject layout: [ob_type @ 0, w_class @ 8]
        // Payload fields start at sizeof(PyObject) = 16.
        const PAYLOAD_0: usize = std::mem::size_of::<pyre_object::pyobject::PyObject>();
        const PAYLOAD_1: usize = PAYLOAD_0 + 8;

        for (field_idx, value) in fields {
            let offset = extract_pyre_field_offset(*field_idx);
            let concrete = value.resolve_with_refs(materialized_refs)?;
            match offset {
                Some(o) if o == OB_TYPE_OFFSET => ob_type = concrete as usize,
                // resume.py:598-602: restore ALL fields, including w_class.
                Some(o) if o == W_CLASS_OFFSET => w_class = concrete as usize,
                Some(o) if o == PAYLOAD_0 => {
                    if ob_type == &LIST_TYPE as *const _ as usize {
                        list_items_ptr = concrete as usize;
                    } else {
                        int_payload = concrete;
                    }
                }
                Some(o) if o == PAYLOAD_1 && ob_type == &LIST_TYPE as *const _ as usize => {
                    list_items_len = concrete as usize;
                }
                _ => {}
            }
        }

        let ptr = if ob_type == &INT_TYPE as *const _ as usize {
            pyre_object::intobject::w_int_new(int_payload)
        } else if ob_type == &FLOAT_TYPE as *const _ as usize {
            pyre_object::floatobject::w_float_new(f64::from_bits(int_payload as u64))
        } else if ob_type == &LIST_TYPE as *const _ as usize {
            let items = if list_items_len == 0 {
                Vec::new()
            } else {
                if list_items_ptr == 0 {
                    return None;
                }
                unsafe {
                    std::slice::from_raw_parts(list_items_ptr as *const PyObjectRef, list_items_len)
                        .to_vec()
                }
            };
            w_list_new(items)
        } else {
            return None;
        };

        // resume.py:602 setfields parity: restore traced w_class if recorded.
        // Without this, builtin subclasses or __class__-mutated objects would
        // revert to the default builtin type after guard failure.
        if w_class != 0 {
            unsafe {
                (*(ptr as *mut pyre_object::pyobject::PyObject)).w_class =
                    w_class as *mut pyre_object::pyobject::PyObject;
            }
        }

        Some(majit_ir::GcRef(ptr as usize))
    }
}

fn materialize_virtual_raw_buffer(
    size: usize,
    entries: &[(usize, usize, majit_metainterp::resume::MaterializedValue)],
    materialized_refs: &[Option<majit_ir::GcRef>],
) -> Option<majit_ir::GcRef> {
    use std::alloc::{Layout, alloc_zeroed};

    let layout = Layout::from_size_align(size.max(1), 8).ok()?;
    let ptr = unsafe { alloc_zeroed(layout) };
    if ptr.is_null() {
        return None;
    }

    for (offset, length, value) in entries {
        let concrete = value.resolve_with_refs(materialized_refs)?;
        unsafe {
            let slot = ptr.add(*offset);
            match *length {
                1 => slot.write(concrete as u8),
                2 => (slot as *mut u16).write_unaligned(concrete as u16),
                4 => (slot as *mut u32).write_unaligned(concrete as u32),
                8 => (slot as *mut u64).write_unaligned(concrete as u64),
                _ => return None,
            }
        }
    }

    Some(majit_ir::GcRef(ptr as usize))
}

fn value_to_ptr(value: &Value) -> PyObjectRef {
    match value {
        Value::Ref(gc_ref) => gc_ref.0 as PyObjectRef,
        Value::Int(n) => *n as PyObjectRef,
        _ => std::ptr::null_mut(),
    }
}

fn value_to_raw(value: &Value) -> usize {
    match value {
        Value::Ref(gc_ref) => gc_ref.0,
        Value::Int(n) => *n as usize,
        Value::Float(f) => f.to_bits() as usize,
        Value::Void => 0,
    }
}

fn value_to_usize(value: &Value) -> usize {
    match value {
        Value::Ref(gc_ref) => gc_ref.0,
        Value::Int(n) => *n as usize,
        _ => 0,
    }
}

/// Extract byte offset from a pyre field descriptor index.
/// Mirrors `extract_field_offset` in majit-opt/virtualize.rs.
fn extract_pyre_field_offset(descr_idx: u32) -> Option<usize> {
    const FIELD_DESCR_TAG: u32 = 0x1000_0000;
    if descr_idx & 0xF000_0000 != FIELD_DESCR_TAG {
        return None;
    }
    Some(((descr_idx >> 4) & 0x000f_ffff) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use majit_metainterp::JitState;
    use majit_metainterp::resume::{MaterializedValue, MaterializedVirtual};
    use pyre_object::OB_TYPE_OFFSET;
    use pyre_object::floatobject::w_float_get_value;
    use pyre_object::listobject::w_list_getitem;
    use std::cell::{Cell, UnsafeCell};

    thread_local! {
        static TEST_CALLBACKS_INIT: Cell<bool> = const { Cell::new(false) };
        static TEST_JIT_DRIVER: UnsafeCell<crate::driver::JitDriverPair> = UnsafeCell::new({
            let info = crate::frame_layout::build_pyframe_virtualizable_info();
            let mut driver = majit_metainterp::JitDriver::new(1);
            driver.set_virtualizable_info(info.clone());
            driver.meta_interp_mut().num_scalar_inputargs =
                crate::virtualizable_gen::NUM_SCALAR_INPUTARGS;
            (driver, info)
        });
    }

    fn ensure_test_callbacks() {
        TEST_CALLBACKS_INIT.with(|init| {
            if init.get() {
                return;
            }
            init.set(true);
            let cb = Box::leak(Box::new(crate::callbacks::CallJitCallbacks {
                callee_frame_helper: |_| None,
                callable_prefers_function_entry: |_| false,
                recursive_force_cache_safe: |_| false,
                jit_drop_callee_frame: std::ptr::null(),
                jit_force_callee_frame: std::ptr::null(),
                jit_force_recursive_call_1: std::ptr::null(),
                jit_force_recursive_call_argraw_boxed_1: std::ptr::null(),
                jit_force_self_recursive_call_argraw_boxed_1: std::ptr::null(),
                jit_create_callee_frame_1: std::ptr::null(),
                jit_create_callee_frame_1_raw_int: std::ptr::null(),
                jit_create_self_recursive_callee_frame_1: std::ptr::null(),
                jit_create_self_recursive_callee_frame_1_raw_int: std::ptr::null(),
                driver_pair: || TEST_JIT_DRIVER.with(|cell| cell.get() as *mut u8),
                ensure_majit_jitcode: |_, _| {},
            }));
            crate::callbacks::init(cb);
        });
    }

    fn empty_meta() -> PyreMeta {
        PyreMeta {
            num_locals: 0,
            ns_keys: Vec::new(),
            valuestackdepth: 0,
            slot_types: Vec::new(),
            has_virtualizable: false,
        }
    }

    fn empty_state() -> PyreJitState {
        PyreJitState {
            frame: 0,
            resume_pc: None,
        }
    }

    #[test]
    fn test_is_boxed_int_value_ignores_py_null_sentinel() {
        assert!(!is_boxed_int_value(PY_NULL));
        let boxed = w_int_new(7);
        assert!(is_boxed_int_value(boxed));
    }

    #[test]
    fn test_trace_ob_type_descr_uses_immutable_header_field_descr() {
        let descr = trace_ob_type_descr();
        let field = descr
            .as_field_descr()
            .expect("ob_type descr must be a field descr");
        assert_eq!(field.offset(), OB_TYPE_OFFSET);
        assert_eq!(field.field_type(), Type::Int);
        assert!(descr.is_always_pure());
        assert!(field.is_immutable());
    }

    #[test]
    fn test_trace_gc_object_type_field_uses_pure_getfield_for_immutable_header() {
        let mut ctx = TraceCtx::for_test(1);
        let obj = OpRef(0);
        let _ = trace_gc_object_type_field(&mut ctx, obj, trace_ob_type_descr());

        let recorder = ctx.into_recorder();
        let op = recorder.last_op().expect("getfield op should be present");
        assert_eq!(op.opcode, OpCode::GetfieldGcPureI);
        assert_eq!(op.args.as_slice(), &[obj]);
    }

    #[test]
    fn test_guard_class_uses_guard_nonnull_class() {
        let mut ctx = TraceCtx::for_test(1);
        let obj = OpRef(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_locals = vec![obj];
        sym.symbolic_local_types = vec![Type::Ref];
        sym.nlocals = 1;
        sym.symbolic_initialized = true;

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        state.with_ctx(|this, ctx| {
            this.guard_class(ctx, obj, &INT_TYPE as *const PyType);
        });

        let recorder = ctx.into_recorder();
        let op = recorder.last_op().expect("guard op should be present");
        assert_eq!(op.opcode, OpCode::GuardNonnullClass);
        assert_eq!(op.args[0], obj);
    }

    #[test]
    fn test_trace_guarded_int_payload_uses_guard_nonnull_class_and_pure_payload() {
        let mut ctx = TraceCtx::for_test(1);
        let int_obj = OpRef(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_locals = vec![int_obj];
        sym.symbolic_local_types = vec![Type::Ref];
        sym.nlocals = 1;
        sym.symbolic_initialized = true;

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        let _ = state.with_ctx(|this, ctx| this.trace_guarded_int_payload(ctx, int_obj));

        let recorder = ctx.into_recorder();
        let mut saw_guard_nonnull_class = false;
        let mut saw_pure_payload = false;
        for pos in 1..(1 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_pos(OpRef(pos)) else {
                continue;
            };
            if op.opcode == OpCode::GuardNonnullClass {
                saw_guard_nonnull_class = true;
            }
            if op.opcode == OpCode::GetfieldGcPureI && op.args.as_slice() == &[int_obj] {
                saw_pure_payload = true;
            }
        }
        assert!(
            saw_guard_nonnull_class,
            "int payload fast path should guard object class via GuardNonnullClass"
        );
        assert!(
            saw_pure_payload,
            "int payload fast path should read the immutable payload with GetfieldGcPureI"
        );
    }

    #[test]
    fn test_init_symbolic_skips_heap_array_read_for_standard_virtualizable() {
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("1 + 2").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(1);
        let mut sym = PyreSym::new_uninit(OpRef(0));
        sym.init_vable_indices();

        sym.init_symbolic(&mut ctx, frame_ptr);

        assert_eq!(sym.locals_cells_stack_array_ref, OpRef::NONE);
        let recorder = ctx.into_recorder();
        for pos in 1..(1 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_pos(OpRef(pos)) else {
                continue;
            };
            assert_ne!(
                op.opcode,
                OpCode::GetfieldRawI,
                "standard virtualizable init should not read locals array from heap"
            );
        }
    }

    #[test]
    fn test_materialize_virtual_ref_reconstructs_float_object() {
        let mut state = empty_state();
        let meta = empty_meta();
        let value = 3.25f64;
        let materialized = MaterializedVirtual::Obj {
            type_id: 0,
            descr_index: crate::descr::w_float_size_descr().index(),
            fields: vec![
                (
                    crate::descr::ob_type_descr().index(),
                    MaterializedValue::Value(&FLOAT_TYPE as *const PyType as usize as i64),
                ),
                (
                    crate::descr::float_floatval_descr().index(),
                    MaterializedValue::Value(value.to_bits() as i64),
                ),
            ],
        };

        let ptr = <PyreJitState as JitState>::materialize_virtual_ref_with_refs(
            &mut state,
            &meta,
            0,
            &materialized,
            &[],
        )
        .expect("float virtual should materialize");

        unsafe {
            assert!(is_float(ptr.0 as PyObjectRef));
            assert_eq!(w_float_get_value(ptr.0 as PyObjectRef), value);
        }
    }

    #[test]
    fn test_materialize_virtual_ref_reconstructs_list_from_raw_buffer_ref() {
        let mut state = empty_state();
        let meta = empty_meta();
        let first = w_int_new(2);
        let second = w_int_new(4);

        let raw_items = MaterializedVirtual::RawBuffer {
            size: 16,
            entries: vec![
                (0, 8, MaterializedValue::Value(first as i64)),
                (8, 8, MaterializedValue::Value(second as i64)),
            ],
        };
        let raw_ptr = <PyreJitState as JitState>::materialize_virtual_ref_with_refs(
            &mut state,
            &meta,
            0,
            &raw_items,
            &[],
        )
        .expect("raw items buffer should materialize");

        let list_virtual = MaterializedVirtual::Obj {
            type_id: 0,
            descr_index: 0,
            fields: vec![
                (
                    crate::descr::ob_type_descr().index(),
                    MaterializedValue::Value(&LIST_TYPE as *const PyType as usize as i64),
                ),
                (
                    crate::descr::list_items_ptr_descr().index(),
                    MaterializedValue::VirtualRef(0),
                ),
                (
                    crate::descr::list_items_len_descr().index(),
                    MaterializedValue::Value(2),
                ),
                (
                    crate::descr::list_items_heap_cap_descr().index(),
                    MaterializedValue::Value(2),
                ),
            ],
        };

        let list_ptr = <PyreJitState as JitState>::materialize_virtual_ref_with_refs(
            &mut state,
            &meta,
            1,
            &list_virtual,
            &[Some(raw_ptr)],
        )
        .expect("list virtual should materialize");

        unsafe {
            assert!(is_list(list_ptr.0 as PyObjectRef));
            assert_eq!(
                w_int_get_value(w_list_getitem(list_ptr.0 as PyObjectRef, 0).unwrap()),
                2
            );
            assert_eq!(
                w_int_get_value(w_list_getitem(list_ptr.0 as PyObjectRef, 1).unwrap()),
                4
            );
        }
    }

    #[test]
    fn test_virtualizable_array_lengths_uses_full_array() {
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("x = 1").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        frame.fix_array_ptrs();
        let full_len = frame.locals_cells_stack_w.len();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut state = empty_state();
        state.frame = frame_ptr;
        state.set_valuestackdepth(2);
        let info = crate::frame_layout::build_pyframe_virtualizable_info();

        // virtualizable.py:86 parity: full array length, not valuestackdepth.
        assert_eq!(
            <PyreJitState as JitState>::virtualizable_array_lengths(
                &state,
                &empty_meta(),
                "frame",
                &info,
            ),
            Some(vec![full_len])
        );
    }

    #[test]
    fn test_restore_guard_failure_uses_runtime_value_kinds_for_virtualizable_locals() {
        use majit_ir::GcRef;
        use pyre_interpreter::pyframe::PyFrame;
        use pyre_interpreter::{ConstantData, compile_exec};

        let module = compile_exec("def f(a, b, c):\n    i = 0\n    return i\nf(1, 2, 3)\n")
            .expect("test code should compile");
        let code = module
            .constants
            .iter()
            .find_map(|constant| match constant {
                ConstantData::Code { code } if code.obj_name.as_str() == "f" => {
                    Some((**code).clone())
                }
                _ => None,
            })
            .expect("test source should contain function code");

        let mut frame = Box::new(PyFrame::new(code));
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut state = PyreJitState {
            frame: frame_ptr,
            resume_pc: None,
        };
        state.set_next_instr(0);
        state.set_valuestackdepth(4);
        let meta = PyreMeta {
            num_locals: 4,
            ns_keys: Vec::new(),
            valuestackdepth: 4,
            has_virtualizable: true,
            // Stale trace-entry types: all locals looked boxed refs when the
            // trace started, but guard failure can still carry a raw int in
            // slot `i`.
            slot_types: vec![Type::Ref, Type::Ref, Type::Ref, Type::Ref],
        };
        let values = vec![
            Value::Ref(GcRef(frame_ptr)),             // frame
            Value::Int(9),                            // next_instr
            Value::Ref(GcRef(frame.code as usize)),   // code
            Value::Int(4),                            // valuestackdepth
            Value::Ref(GcRef(0)),                     // namespace
            Value::Ref(GcRef(w_int_new(1) as usize)), // local a
            Value::Ref(GcRef(w_int_new(2) as usize)), // local b
            Value::Ref(GcRef(w_int_new(3) as usize)), // local c
            Value::Int(7),                            // local i
        ];

        assert!(<PyreJitState as JitState>::restore_guard_failure_values(
            &mut state,
            &meta,
            &values,
            &majit_metainterp::blackhole::ExceptionState::default(),
        ));

        assert_eq!(state.next_instr(), 9);
        assert_eq!(state.valuestackdepth(), 4);
        let restored_i = state.local_at(3).expect("local i should be restored");
        assert!(unsafe { is_int(restored_i) });
        assert_eq!(unsafe { w_int_get_value(restored_i) }, 7);
    }

    #[test]
    fn test_load_local_checked_value_skips_nonnull_guard_for_typed_int_local() {
        let mut ctx = TraceCtx::for_test(1);
        let local = OpRef(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_locals = vec![local];
        sym.symbolic_local_types = vec![Type::Int];
        sym.nlocals = 1;
        sym.symbolic_initialized = true;

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        let loaded = <MIFrame as LocalOpcodeHandler>::load_local_checked_value(&mut state, 0, "j")
            .expect("typed int local should load without guard");
        assert_eq!(loaded.opref, local);

        let recorder = ctx.into_recorder();
        assert_eq!(recorder.num_ops(), 0);
    }

    #[test]
    fn test_load_local_checked_value_keeps_nonnull_guard_for_ref_local() {
        let mut ctx = TraceCtx::for_test(1);
        let local = OpRef(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_locals = vec![local];
        sym.symbolic_local_types = vec![Type::Ref];
        sym.nlocals = 1;
        sym.symbolic_initialized = true;

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        let loaded = <MIFrame as LocalOpcodeHandler>::load_local_checked_value(&mut state, 0, "b")
            .expect("ref local should load with guard");
        assert_eq!(loaded.opref, local);

        let recorder = ctx.into_recorder();
        assert_eq!(recorder.num_ops(), 1);
        assert_eq!(
            recorder.last_op().map(|op| op.opcode),
            Some(OpCode::GuardNonnull)
        );
    }

    #[test]
    fn test_store_local_value_preserves_traced_raw_int_type_for_ref_slot() {
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("x = 1").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        frame.locals_cells_stack_w[0] = w_int_new(41);
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(1);
        let raw = OpRef(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_locals = vec![OpRef::NONE];
        sym.symbolic_local_types = vec![Type::Ref];
        sym.nlocals = 1;
        sym.symbolic_initialized = true;
        sym.transient_value_types.insert(raw, Type::Int);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        state
            .with_ctx(|this, ctx| this.store_local_value(ctx, 0, raw))
            .expect("store of raw traced int should succeed");
        assert_eq!(state.sym().symbolic_locals[0], raw);
        assert_eq!(state.sym().symbolic_local_types[0], Type::Int);

        let loaded = <MIFrame as LocalOpcodeHandler>::load_local_checked_value(&mut state, 0, "x")
            .expect("typed raw int local should reload without object guards");
        assert_eq!(loaded.opref, raw);

        let recorder = ctx.into_recorder();
        assert_eq!(recorder.num_ops(), 0);
    }

    #[test]
    fn test_trace_dynamic_list_index_typed_int_skips_object_unbox() {
        let mut ctx = TraceCtx::for_test(2);
        let key = OpRef(0);
        let len = OpRef(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_locals = vec![key];
        sym.symbolic_local_types = vec![Type::Int];
        sym.nlocals = 1;
        sym.symbolic_initialized = true;

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        let raw_index = state.with_ctx(|this, ctx| this.trace_dynamic_list_index(ctx, key, len, 2));
        assert_eq!(raw_index, key);

        let recorder = ctx.into_recorder();
        assert_eq!(recorder.num_ops(), 4);
        assert_eq!(recorder.num_guards(), 2);
    }

    #[test]
    fn test_trace_binary_value_boxes_typed_raw_operands_for_python_helper() {
        let mut ctx = TraceCtx::for_test(2);
        let lhs = OpRef(0);
        let rhs = OpRef(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_locals = vec![lhs, rhs];
        sym.symbolic_local_types = vec![Type::Float, Type::Int];
        sym.nlocals = 2;
        sym.symbolic_initialized = true;

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        let _ = <MIFrame as TraceHelperAccess>::trace_binary_value(
            &mut state,
            lhs,
            rhs,
            BinaryOperator::Power,
        )
        .expect("generic helper call should box raw operands first");

        let recorder = ctx.into_recorder();
        let call = recorder.last_op().expect("call op should be present");
        assert!(matches!(
            call.opcode,
            OpCode::CallI | OpCode::CallR | OpCode::CallF | OpCode::CallN
        ));
        assert_ne!(call.args[0], lhs);
        assert_ne!(call.args[1], rhs);
    }

    #[test]
    fn test_trace_known_builtin_call_boxes_typed_raw_args_for_python_helper_boundary() {
        let mut ctx = TraceCtx::for_test(2);
        let callable = OpRef(0);
        let arg = OpRef(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_locals = vec![callable, arg];
        sym.symbolic_local_types = vec![Type::Ref, Type::Int];
        sym.nlocals = 2;
        sym.symbolic_initialized = true;

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        let _ = state
            .trace_known_builtin_call(callable, &[arg])
            .expect("known builtin helper boundary should box raw int args");

        let recorder = ctx.into_recorder();
        let call = recorder.last_op().expect("call op should be present");
        assert!(matches!(
            call.opcode,
            OpCode::CallI | OpCode::CallR | OpCode::CallF | OpCode::CallN
        ));
        assert_ne!(call.args.last().copied(), Some(arg));
    }

    #[test]
    fn test_compare_value_direct_keeps_raw_truth_for_immediate_branch_consumer() {
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("if 1 < 2:\n    x = 3\n").expect("test code should compile");
        let code_ref =
            pyre_interpreter::w_code_new(Box::into_raw(Box::new(code.clone())) as *const ())
                as *const ();
        let compare_pc = (0..code.instructions.len())
            .find(|&pc| {
                matches!(
                    decode_instruction_at(&code, pc),
                    Some((Instruction::CompareOp { .. }, _))
                )
            })
            .expect("test bytecode should contain COMPARE_OP");
        let branch_pc = ((compare_pc + 1)..code.instructions.len())
            .find(|&pc| {
                decode_instruction_at(&code, pc)
                    .map(|(instruction, _)| instruction_consumes_comparison_truth(instruction))
                    .unwrap_or(false)
            })
            .expect("test bytecode should contain POP_JUMP_IF after COMPARE_OP");

        let mut frame = Box::new(PyFrame::new(code));
        frame.fix_array_ptrs();
        frame.push(w_int_new(1));
        frame.push(w_int_new(2));
        let _frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(2);
        let lhs = OpRef(0);
        let rhs = OpRef(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_initialized = true;
        sym.valuestackdepth = 0;
        sym.jitcode = jitcode_for(code_ref);
        sym.transient_value_types.insert(lhs, Type::Int);
        sym.transient_value_types.insert(rhs, Type::Int);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: branch_pc,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        let concrete_lhs = w_int_new(10);
        let concrete_rhs = w_int_new(20);
        let result = state
            .compare_value_direct(
                lhs,
                rhs,
                ComparisonOperator::Less,
                concrete_lhs,
                concrete_rhs,
            )
            .expect("int comparison should trace");
        let truth = state
            .truth_value_direct(result, PY_NULL)
            .expect("immediate POP_JUMP consumer should reuse raw truth");

        assert_eq!(result, truth);

        let recorder = ctx.into_recorder();
        let mut saw_cmp = false;
        let mut saw_bool_call = false;
        let mut saw_bool_unbox = false;
        for pos in 2..(2 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_pos(OpRef(pos)) else {
                continue;
            };
            if op.opcode == OpCode::IntLt {
                saw_cmp = true;
            }
            if op.opcode == OpCode::CallR {
                saw_bool_call = true;
            }
            if op.opcode == OpCode::GetfieldGcI {
                saw_bool_unbox = true;
            }
        }
        assert!(
            saw_cmp,
            "branch compare should still emit raw int comparison"
        );
        assert!(
            !saw_bool_call,
            "immediate branch consumer should not allocate a bool object"
        );
        assert!(
            !saw_bool_unbox,
            "immediate branch consumer should not unbox a transient bool object"
        );
    }

    #[test]
    fn test_compare_value_direct_boxes_bool_when_not_immediately_consumed_by_branch() {
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("x = 1").expect("test code should compile");
        let code_ref =
            pyre_interpreter::w_code_new(Box::into_raw(Box::new(code.clone())) as *const ())
                as *const ();
        let mut frame = Box::new(PyFrame::new(code));
        frame.fix_array_ptrs();
        frame.push(w_int_new(1));
        frame.push(w_int_new(2));
        let _frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(2);
        let lhs = OpRef(0);
        let rhs = OpRef(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_initialized = true;
        sym.valuestackdepth = 0;
        sym.jitcode = jitcode_for(code_ref);
        sym.transient_value_types.insert(lhs, Type::Int);
        sym.transient_value_types.insert(rhs, Type::Int);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        let concrete_lhs = w_int_new(10);
        let concrete_rhs = w_int_new(20);
        let _ = state
            .compare_value_direct(
                lhs,
                rhs,
                ComparisonOperator::Less,
                concrete_lhs,
                concrete_rhs,
            )
            .expect("non-branch compare should trace");

        let recorder = ctx.into_recorder();
        let mut saw_bool_call = false;
        for pos in 2..(2 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_pos(OpRef(pos)) else {
                continue;
            };
            if op.opcode == OpCode::CallR {
                saw_bool_call = true;
            }
        }
        assert!(
            saw_bool_call,
            "non-branch compare should continue to materialize a Python bool result"
        );
    }

    #[test]
    fn test_next_instruction_consumes_comparison_truth_skips_extended_arg_trivia() {
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::PyFrame;

        let mut source = String::from("def f(x, y):\n    if x < y:\n");
        for i in 0..400 {
            source.push_str(&format!("        z{i} = {i}\n"));
        }
        source.push_str("    return 0\n");
        source.push_str("f(1, 2)\n");

        let module = compile_exec(&source).expect("test code should compile");
        let code = module
            .constants
            .iter()
            .find_map(|constant| match constant {
                pyre_interpreter::ConstantData::Code { code } if code.obj_name.as_str() == "f" => {
                    Some((**code).clone())
                }
                _ => None,
            })
            .expect("test source should contain function code");

        let compare_pc = (0..code.instructions.len())
            .find(|&pc| {
                matches!(
                    decode_instruction_at(&code, pc),
                    Some((Instruction::CompareOp { .. }, _))
                )
            })
            .expect("test bytecode should contain COMPARE_OP");

        let first_after_compare = decode_instruction_at(&code, compare_pc + 1)
            .map(|(instruction, _)| instruction)
            .expect("bytecode should continue after COMPARE_OP");
        assert!(
            instruction_is_trivia_between_compare_and_branch(first_after_compare),
            "test source should force trivia between COMPARE_OP and POP_JUMP_IF"
        );

        let code_ref =
            pyre_interpreter::w_code_new(Box::into_raw(Box::new(code.clone())) as *const ())
                as *const ();
        let mut frame = Box::new(PyFrame::new(code));
        frame.fix_array_ptrs();
        let _frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(2);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_initialized = true;
        sym.valuestackdepth = 0;
        sym.jitcode = jitcode_for(code_ref);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: compare_pc + 1,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        assert!(
            state.next_instruction_consumes_comparison_truth(),
            "branch fusion should survive EXTENDED_ARG/other trivia before the branch"
        );
    }

    #[test]
    fn test_trace_code_step_preserves_comparison_truth_across_extended_arg_trivia() {
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::PyFrame;

        let mut source = String::from("def f(x, y):\n    if x < y:\n");
        for i in 0..400 {
            source.push_str(&format!("        z{i} = {i}\n"));
        }
        source.push_str("    return 0\n");
        source.push_str("f(1, 2)\n");

        let module = compile_exec(&source).expect("test code should compile");
        let code = module
            .constants
            .iter()
            .find_map(|constant| match constant {
                pyre_interpreter::ConstantData::Code { code } if code.obj_name.as_str() == "f" => {
                    Some((**code).clone())
                }
                _ => None,
            })
            .expect("test source should contain function code");

        let compare_pc = (0..code.instructions.len())
            .find(|&pc| {
                matches!(
                    decode_instruction_at(&code, pc),
                    Some((Instruction::CompareOp { .. }, _))
                )
            })
            .expect("test bytecode should contain COMPARE_OP");

        let first_after_compare = decode_instruction_at(&code, compare_pc + 1)
            .map(|(instruction, _)| instruction)
            .expect("bytecode should continue after COMPARE_OP");
        assert!(
            instruction_is_trivia_between_compare_and_branch(first_after_compare),
            "test source should force trivia between COMPARE_OP and POP_JUMP_IF"
        );

        let mut frame = Box::new(PyFrame::new(code.clone()));
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_initialized = true;
        sym.last_comparison_truth = Some(OpRef(123));

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: compare_pc + 2,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        let action = state.trace_code_step(&code, compare_pc + 1);
        assert!(matches!(action, TraceAction::Continue));
        assert_eq!(
            state.sym().last_comparison_truth,
            Some(OpRef(123)),
            "comparison truth cache should survive trivia until the real branch consumer"
        );
    }

    // Tests for concrete_popped_value, concrete_binary_operands,
    // concrete_store_subscr_operands removed: these stack-based concrete
    // read methods were replaced by direct FrontendOp.concrete parameter passing.

    // test_concrete_branch_truth_reads_last_popped_slot removed:
    // concrete_branch_truth now requires explicit concrete parameter.

    #[test]
    fn test_concrete_branch_truth_uses_cached_comparison_truth_without_stack_value() {
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("1 + 2").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_initialized = true;
        sym.nlocals = frame.nlocals();
        sym.valuestackdepth = frame.valuestackdepth;
        sym.last_comparison_truth = Some(OpRef(321));
        sym.last_comparison_concrete_truth = Some(false);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        assert!(
            !state
                .concrete_branch_truth()
                .expect("cached comparison truth should avoid concrete stack lookup")
        );
        assert_eq!(state.sym().last_comparison_concrete_truth, None);
    }

    #[test]
    fn test_truth_value_direct_caches_concrete_truth_for_raw_int_branch_consumer() {
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("1 + 2").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        frame.push(w_int_new(7));
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_initialized = true;
        sym.nlocals = frame.nlocals();
        sym.valuestackdepth = frame.valuestackdepth - 1;
        sym.transient_value_types.insert(OpRef(77), Type::Int);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        let truth = state
            .truth_value_direct(OpRef(77), w_int_new(7))
            .expect("raw int truth should trace");
        assert!(!truth.is_none());

        state.sym_mut().valuestackdepth = frame.valuestackdepth;
        assert!(
            state
                .concrete_branch_truth()
                .expect("cached raw-int truth should avoid concrete stack lookup")
        );
        assert_eq!(state.sym().last_comparison_concrete_truth, None);
    }

    #[test]
    fn test_record_branch_guard_uses_branch_pc_and_pre_pop_stack_shape() {
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("1 + 2").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        frame.next_instr = 456;
        let lower_value = w_int_new(0);
        let branch_value = w_int_new(1);
        frame.push(lower_value);
        frame.push(branch_value);
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(3);
        let expected_pc = ctx.const_int(456);
        let expected_vsd = ctx.const_int(2);
        let expected_branch_value = ctx.const_int(branch_value as i64);
        let frame_ref = OpRef(0);
        let lower_stack = OpRef(1);
        let truth = OpRef(2);

        let mut sym = PyreSym::new_uninit(frame_ref);
        sym.symbolic_initialized = true;
        sym.nlocals = frame.nlocals();
        sym.valuestackdepth = frame.valuestackdepth - 1;
        sym.symbolic_stack = vec![lower_stack];
        sym.symbolic_stack_types = vec![Type::Ref];

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 459,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        state.with_ctx(|this, ctx| {
            this.record_branch_guard(ctx, OpRef::NONE, truth, true);
        });

        let recorder = ctx.into_recorder();
        let guard = recorder.last_op().expect("branch guard should be recorded");
        let fail_args = guard
            .fail_args
            .as_ref()
            .expect("branch guard should carry explicit fail args");
        assert_eq!(guard.opcode, OpCode::GuardTrue);
        // fail_args: [frame, pc_const, vsd_const, lower_stack, ...]
        // fail_args: [frame, ni, code, vsd, ns, lower_stack, ...]
        assert!(fail_args.len() >= 6);
        assert_eq!(fail_args[0], frame_ref);
        assert_eq!(fail_args[5], lower_stack);
    }

    #[test]
    fn test_generic_guard_during_branch_truth_uses_pre_pop_stack_shape() {
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("1 + 2").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        frame.next_instr = 456;
        let lower_value = w_int_new(0);
        let branch_value = w_int_new(1);
        frame.push(lower_value);
        frame.push(branch_value);
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(3);
        let expected_pc = ctx.const_int(456);
        let expected_vsd = ctx.const_int(2);
        let expected_branch_value = ctx.const_int(branch_value as i64);
        let frame_ref = OpRef(0);
        let lower_stack = OpRef(1);
        let truth = OpRef(2);

        let mut sym = PyreSym::new_uninit(frame_ref);
        sym.symbolic_initialized = true;
        sym.nlocals = frame.nlocals();
        sym.valuestackdepth = frame.valuestackdepth - 1;
        sym.symbolic_stack = vec![lower_stack];
        sym.symbolic_stack_types = vec![Type::Ref];
        sym.pending_branch_value = Some(OpRef::NONE);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 459,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        state.sym_mut().pending_branch_value = Some(OpRef::NONE);
        state.with_ctx(|this, ctx| {
            this.sym_mut().pending_branch_value = Some(OpRef::NONE);
            this.generate_guard(ctx, OpCode::GuardTrue, &[truth]);
        });

        let recorder = ctx.into_recorder();
        let guard = recorder.last_op().expect("branch guard should be recorded");
        let fail_args = guard
            .fail_args
            .as_ref()
            .expect("branch guard should carry explicit fail args");
        assert_eq!(guard.opcode, OpCode::GuardTrue);
        // fail_args: [frame, ni, code, vsd, ns, lower_stack, ...]
        assert!(fail_args.len() >= 6);
        assert_eq!(fail_args[0], frame_ref);
    }

    #[test]
    fn test_branch_truth_uses_concrete_parameter() {
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("1 + 2").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        frame.push(w_int_new(1));
        frame.fix_array_ptrs();

        let mut ctx = TraceCtx::for_test(2);
        let frame_ref = OpRef(0);
        let truth = OpRef(1);
        let mut sym = PyreSym::new_uninit(frame_ref);
        sym.symbolic_initialized = true;
        sym.nlocals = frame.nlocals();
        sym.valuestackdepth = frame.valuestackdepth;
        sym.symbolic_stack = vec![truth];
        sym.symbolic_stack_types = vec![Type::Int];
        sym.pending_branch_value = Some(truth);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        state.with_ctx(|this, ctx| {
            this.generate_guard(ctx, OpCode::GuardTrue, &[truth]);
        });
        // concrete_branch_truth_for_value now takes concrete value as parameter
        assert_eq!(
            state
                .concrete_branch_truth_for_value(truth, w_int_new(1))
                .unwrap(),
            true
        );
        <MIFrame as BranchOpcodeHandler>::leave_branch_truth(&mut state).unwrap();
    }

    #[test]
    fn test_current_fail_args_flushes_virtualizable_header_before_capture() {
        let mut ctx = TraceCtx::for_test(2);
        let frame_ref = OpRef(0);
        let local0 = OpRef(1);
        let stack0 = OpRef(2);
        let stack1 = OpRef(3);

        let mut sym = PyreSym::new_uninit(frame_ref);
        sym.symbolic_initialized = true;
        sym.nlocals = 1;
        sym.valuestackdepth = 3;
        sym.vable_next_instr = ctx.const_int(12);
        sym.vable_valuestackdepth = ctx.const_int(1);
        sym.symbolic_locals = vec![local0];
        sym.symbolic_local_types = vec![Type::Ref];
        sym.symbolic_stack = vec![stack0, stack1];
        sym.symbolic_stack_types = vec![Type::Ref, Type::Ref];

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        let fail_args = state.with_ctx(|this, ctx| this.current_fail_args(ctx));

        // fail_args: [frame, next_instr_const, vsd_const, local0, stack0, stack1]
        // fail_args: [frame, ni, code, vsd, ns, local0, stack0, stack1]
        assert_eq!(fail_args.len(), 8);
        assert_eq!(fail_args[0], frame_ref);
        assert_eq!(fail_args[5], local0);
        assert_eq!(fail_args[6], stack0);
        assert_eq!(fail_args[7], stack1);
    }

    #[test]
    fn test_current_fail_args_materializes_symbolic_holes_from_concrete_frame() {
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("len(x)").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        let local_ref = w_int_new(41);
        let stack_int = w_int_new(7);
        frame.locals_cells_stack_w[0] = local_ref;
        frame.locals_cells_stack_w[1] = stack_int;
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(1);
        let frame_ref = OpRef(0);
        let code_ref = frame.code;
        let mut sym = PyreSym::new_uninit(frame_ref);
        sym.symbolic_initialized = true;
        sym.nlocals = 1;
        sym.valuestackdepth = 2;
        sym.vable_next_instr = ctx.const_int(33);
        sym.vable_valuestackdepth = ctx.const_int(0);
        sym.symbolic_locals = vec![OpRef::NONE];
        sym.symbolic_local_types = vec![Type::Ref];
        sym.symbolic_stack = vec![OpRef::NONE];
        sym.symbolic_stack_types = vec![Type::Int];
        sym.jitcode = jitcode_for(code_ref);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: frame_ptr,
        };

        let fail_args = state.with_ctx(|this, ctx| this.current_fail_args(ctx));

        // fail_args: [frame, pc_const, vsd_const, live_slots...]
        // Liveness-based: only slots live at orgpc are included.
        assert!(
            fail_args.len() >= crate::virtualizable_gen::NUM_SCALAR_INPUTARGS,
            "must have frame + ni + code + vsd + ns header"
        );
        assert_eq!(fail_args[0], frame_ref);
        assert!(
            fail_args.iter().all(|arg| !arg.is_none()),
            "materialized fail args should not contain OpRef::NONE holes"
        );
    }

    #[test]
    fn test_close_loop_args_at_target_pc_uses_locals_only_jump_contract() {
        ensure_test_callbacks();
        let mut ctx = TraceCtx::for_test(0);
        let frame_ref = ctx.const_ref(0x1000);
        let code_ref = ctx.const_ref(0x2000);
        let namespace_ref = ctx.const_ref(0x3000);
        let local0 = ctx.const_ref(0x4000);
        let stack0 = ctx.const_ref(0x5000);
        let stack1 = ctx.const_ref(0x6000);

        let mut sym = PyreSym::new_uninit(frame_ref);
        sym.symbolic_initialized = true;
        sym.nlocals = 1;
        sym.valuestackdepth = 3;
        sym.vable_next_instr = ctx.const_int(33);
        sym.vable_code = code_ref;
        sym.vable_valuestackdepth = ctx.const_int(3);
        sym.vable_namespace = namespace_ref;
        sym.symbolic_locals = vec![local0];
        sym.symbolic_local_types = vec![Type::Ref];
        sym.symbolic_stack = vec![stack0, stack1];
        sym.symbolic_stack_types = vec![Type::Ref, Type::Ref];
        sym.concrete_stack = vec![ConcreteValue::Null, ConcreteValue::Null];

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        let jump_args = state.with_ctx(|this, ctx| this.close_loop_args_at(ctx, Some(77)));

        assert_eq!(
            jump_args.len(),
            crate::virtualizable_gen::NUM_SCALAR_INPUTARGS + 1,
            "target loop header should receive frame header plus locals only"
        );
        assert_eq!(state.sym().valuestackdepth, state.sym().nlocals);
        assert!(state.sym().symbolic_stack.is_empty());
        assert!(state.sym().symbolic_stack_types.is_empty());
    }

    #[test]
    fn test_direct_len_value_returns_typed_raw_len_for_integer_list() {
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::PyFrame;

        let code = compile_exec("len(x)").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        let list = w_list_new(vec![w_int_new(1), w_int_new(2), w_int_new(3)]);
        unsafe {
            assert!(w_list_uses_int_storage(list));
        }
        let arg_idx = frame.stack_base() + 2;
        frame.locals_cells_stack_w[arg_idx] = list;
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(2);
        let value = OpRef(0);
        let callable = OpRef(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_locals = vec![value, callable];
        sym.symbolic_local_types = vec![Type::Ref, Type::Ref];
        sym.nlocals = 2;
        sym.valuestackdepth = frame.stack_base();
        sym.symbolic_initialized = true;

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        let len = state
            .direct_len_value(callable, value, list)
            .expect("integer-list len fast path should trace");
        assert!(
            state.sym().transient_value_types.contains_key(&len),
            "len fast path should remember the typed raw result"
        );
        let len_type = state.value_type(len);

        let recorder = ctx.into_recorder();
        assert_ne!(
            recorder.last_op().map(|op| op.opcode),
            Some(OpCode::CallI),
            "len(list) should not fall back to the builtin helper call path"
        );
        let mut saw_len_field = false;
        let mut saw_new = false;
        for pos in 2..(2 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_pos(OpRef(pos)) else {
                continue;
            };
            if op.opcode == OpCode::New {
                saw_new = true;
            }
            if op.opcode == OpCode::GetfieldGcI
                && op.descr.as_ref().map(|d| d.index()) == Some(list_int_items_len_descr().index())
            {
                saw_len_field = true;
            }
        }
        assert_eq!(len_type, Type::Int);
        assert!(
            saw_len_field,
            "list len fast path should read the integer-list length field"
        );
        assert!(
            !saw_new,
            "len(list) should not allocate a W_Int box in the trace"
        );
    }

    #[test]
    fn test_trace_direct_float_list_getitem_uses_gc_field_loads_for_list_object() {
        let mut ctx = TraceCtx::for_test(2);
        let list = OpRef(0);
        let key = OpRef(1);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_locals = vec![list, key];
        sym.symbolic_local_types = vec![Type::Ref, Type::Int];
        sym.nlocals = 2;
        sym.symbolic_initialized = true;

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        let result = state.with_ctx(|this, ctx| {
            crate::generated_list_getitem_by_strategy(this, ctx, list, key, 2, 2)
        });
        assert_eq!(state.value_type(result), Type::Float);

        let recorder = ctx.into_recorder();
        let mut saw_gc_field = false;
        let mut saw_raw_field = false;
        let mut saw_raw_array = false;
        for pos in 2..(2 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_pos(OpRef(pos)) else {
                continue;
            };
            match op.opcode {
                OpCode::GetfieldGcI => saw_gc_field = true,
                OpCode::GetfieldRawI => saw_raw_field = true,
                OpCode::GetarrayitemRawF => saw_raw_array = true,
                _ => {}
            }
        }
        assert!(saw_gc_field, "list object fields should use GetfieldGcI");
        assert!(
            !saw_raw_field,
            "float-list fast path should not use raw field loads on GC list objects"
        );
        assert!(saw_raw_array, "payload array access stays raw");
    }

    #[test]
    fn test_list_append_value_uses_raw_int_storage_fast_path() {
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::PyFrame;

        let mut ctx = TraceCtx::for_test(2);
        let list = OpRef(0);
        let value = OpRef(1);
        let concrete_list = w_list_new(vec![w_int_new(1), w_int_new(2)]);
        let code = compile_exec("x = 1").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;
        unsafe {
            assert!(w_list_uses_int_storage(concrete_list));
            assert!(w_list_can_append_without_realloc(concrete_list));
        }

        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_locals = vec![list, value];
        sym.symbolic_local_types = vec![Type::Ref, Type::Int];
        sym.nlocals = 2;
        sym.symbolic_initialized = true;
        sym.transient_value_types.insert(value, Type::Int);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        let concrete_value = w_int_new(42);
        state
            .list_append_value(list, value, concrete_list, concrete_value)
            .expect("integer-list append fast path should trace");

        let recorder = ctx.into_recorder();
        let mut saw_raw_setitem = false;
        let mut saw_len_update = false;
        let mut saw_call = false;
        let mut saw_new = false;
        for pos in 2..(2 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_pos(OpRef(pos)) else {
                continue;
            };
            if matches!(
                op.opcode,
                OpCode::CallI | OpCode::CallN | OpCode::CallR | OpCode::CallF
            ) {
                saw_call = true;
            }
            if op.opcode == OpCode::New {
                saw_new = true;
            }
            if op.opcode == OpCode::SetarrayitemRaw
                && op.descr.as_ref().map(|d| d.index()) == Some(int_array_descr().index())
            {
                saw_raw_setitem = true;
            }
            if op.opcode == OpCode::SetfieldGc
                && op.descr.as_ref().map(|d| d.index()) == Some(list_int_items_len_descr().index())
            {
                saw_len_update = true;
            }
        }

        assert!(
            saw_raw_setitem,
            "integer-list append should write through the raw int array"
        );
        assert!(
            saw_len_update,
            "integer-list append should update the integer-list length field"
        );
        assert!(
            !saw_call,
            "integer-list append should not fall back to jit_list_append"
        );
        assert!(
            !saw_new,
            "integer-list append should not allocate a W_Int box in the trace"
        );
    }

    #[test]
    fn test_list_append_value_uses_raw_float_storage_fast_path() {
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::PyFrame;

        let mut ctx = TraceCtx::for_test(2);
        let list = OpRef(0);
        let value = OpRef(1);
        let concrete_list = w_list_new(vec![
            pyre_object::floatobject::w_float_new(1.5),
            pyre_object::floatobject::w_float_new(2.5),
        ]);
        let code = compile_exec("x = 1").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        frame.fix_array_ptrs();
        let frame_ptr = (&mut *frame) as *mut PyFrame as usize;
        unsafe {
            assert!(w_list_uses_float_storage(concrete_list));
            assert!(w_list_can_append_without_realloc(concrete_list));
        }

        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_locals = vec![list, value];
        sym.symbolic_local_types = vec![Type::Ref, Type::Float];
        sym.nlocals = 2;
        sym.symbolic_initialized = true;
        sym.transient_value_types.insert(value, Type::Float);

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        let concrete_value = pyre_object::w_float_new(3.14);
        state
            .list_append_value(list, value, concrete_list, concrete_value)
            .expect("float-list append fast path should trace");

        let recorder = ctx.into_recorder();
        let mut saw_raw_setitem = false;
        let mut saw_len_update = false;
        let mut saw_call = false;
        for pos in 2..(2 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_pos(OpRef(pos)) else {
                continue;
            };
            if matches!(
                op.opcode,
                OpCode::CallI | OpCode::CallN | OpCode::CallR | OpCode::CallF
            ) {
                saw_call = true;
            }
            if op.opcode == OpCode::SetarrayitemRaw
                && op.descr.as_ref().map(|d| d.index()) == Some(float_array_descr().index())
            {
                saw_raw_setitem = true;
            }
            if op.opcode == OpCode::SetfieldGc
                && op.descr.as_ref().map(|d| d.index())
                    == Some(list_float_items_len_descr().index())
            {
                saw_len_update = true;
            }
        }

        assert!(
            saw_raw_setitem,
            "float-list append should write through the raw float array"
        );
        assert!(
            saw_len_update,
            "float-list append should update the float-list length field"
        );
        assert!(
            !saw_call,
            "float-list append should not fall back to jit_list_append"
        );
    }

    #[test]
    fn test_iter_next_value_for_range_iterator_uses_gc_fields_and_returns_raw_int() {
        use pyre_interpreter::compile_exec;
        use pyre_interpreter::pyframe::PyFrame;
        use pyre_object::w_range_iter_new;

        let code = compile_exec("x = 1").expect("test code should compile");
        let mut frame = Box::new(PyFrame::new(code));
        frame.fix_array_ptrs();
        let range_iter = w_range_iter_new(0, 2, 1);
        frame.push(range_iter);
        let _frame_ptr = (&mut *frame) as *mut PyFrame as usize;

        let mut ctx = TraceCtx::for_test(1);
        let iter = OpRef(0);
        let mut sym = PyreSym::new_uninit(OpRef::NONE);
        sym.symbolic_stack = vec![iter];
        sym.symbolic_stack_types = vec![Type::Ref];
        sym.concrete_stack = vec![ConcreteValue::Ref(range_iter)];
        sym.valuestackdepth = 1;
        sym.symbolic_initialized = true;

        let mut state = MIFrame {
            ctx: &mut ctx,
            sym: &mut sym,
            ob_type_fd: trace_ob_type_descr(),
            fallthrough_pc: 0,
            parent_frames: Vec::new(),
            pending_inline_frame: None,
            orgpc: 0,
            concrete_frame_addr: 0,
        };

        let next = MIFrame::iter_next_value(&mut state, iter, range_iter)
            .expect("range iterator fast path should trace");
        assert_eq!(state.value_type(next.opref), Type::Int);
        <MIFrame as IterOpcodeHandler>::guard_optional_value(&mut state, next, true)
            .expect("typed range next should not need optional guard");

        let recorder = ctx.into_recorder();
        let mut saw_getfield_gc = false;
        let mut saw_setfield_gc = false;
        let mut saw_setfield_raw = false;
        let mut saw_getfield_raw = false;
        let mut saw_new = false;
        let mut saw_optional_guard = false;
        for pos in 1..(1 + recorder.num_ops() as u32) {
            let Some(op) = recorder.get_op_by_pos(OpRef(pos)) else {
                continue;
            };
            match op.opcode {
                OpCode::GetfieldGcI => saw_getfield_gc = true,
                OpCode::SetfieldGc => saw_setfield_gc = true,
                OpCode::SetfieldRaw => saw_setfield_raw = true,
                OpCode::GetfieldRawI => saw_getfield_raw = true,
                OpCode::New => saw_new = true,
                OpCode::GuardNonnull | OpCode::GuardIsnull => saw_optional_guard = true,
                _ => {}
            }
        }
        assert!(
            saw_getfield_gc,
            "range iterator fields should use GetfieldGcI"
        );
        assert!(
            saw_setfield_gc,
            "range iterator current update should use SetfieldGc"
        );
        assert!(
            !saw_setfield_raw,
            "range iterator fast path should not use raw stores on GC iterator objects"
        );
        assert!(
            !saw_getfield_raw,
            "range iterator fast path should not use raw field loads on GC iterator objects"
        );
        assert!(
            !saw_new,
            "range iterator fast path should not box the yielded int"
        );
        assert!(
            !saw_optional_guard,
            "raw-int range iteration should not emit optional-value guards"
        );
    }
}

// ── Virtualizable configuration ──────────────────────────────────────
//
// PyPy's `pypy/interpreter/pyframe.py` declares:
//
//     _virtualizable_ = ['locals_stack_w[*]', 'valuestackdepth',
//                         'last_instr', ...]
//
// Our Rust equivalent uses explicit byte offsets instead of name-based
// introspection. The JIT optimizer's Virtualize pass uses this info
// to keep frame fields in CPU registers, eliminating heap accesses
// for LoadFast/StoreFast and stack push/pop during compiled code.
//
// The shared frame layout contract now also lives in `pyre-jit/src/frame_layout.rs`
// so the tracer can compute the same offsets without depending on
// `pyre-interpreter`. Driver registration still happens in `pyre-jit/src/eval.rs`.

pub struct PendingInlineFrame {
    pub sym: PyreSym,
    pub concrete_frame: pyre_interpreter::pyframe::PyFrame,
    pub drop_frame_opref: Option<OpRef>,
    pub green_key: u64,
    /// opencoder.py:819-834: accumulated parent frame chain.
    /// Each: (fail_args, types, resumepc, jitcode_index).
    pub parent_frames: Vec<(Vec<OpRef>, Vec<Type>, usize, i32)>,
    pub nargs: usize,
    pub caller_result_stack_idx: Option<usize>,
}

pub enum InlineTraceStepAction {
    Trace(TraceAction),
    PushFrame(PendingInlineFrame),
}

pub fn execute_inline_residual_call(
    frame: &mut pyre_interpreter::pyframe::PyFrame,
    nargs: usize,
) -> Result<(), pyre_interpreter::PyError> {
    let required = nargs + 2; // callable + null/self + args
    if frame.valuestackdepth < frame.stack_base() + required {
        return Err(pyre_interpreter::PyError::type_error(
            "inline residual call stack underflow",
        ));
    }

    let mut args = Vec::with_capacity(nargs);
    for _ in 0..nargs {
        args.push(frame.pop());
    }
    args.reverse();
    let _null_or_self = frame.pop();
    let callable = frame.pop();
    let result = pyre_interpreter::call::call_callable_inline_residual(frame, callable, &args)?;
    frame.push(result);
    Ok(())
}

// inline_trace_and_execute removed — replaced by PyreMetaInterp.interpret()
// which uses a single framestack for both root and inline frames.
// trace_through_callee removed — replaced by build_pending_inline_frame +
// MetaInterp.push_inline_frame (RPython perform_call parity).

/// listobject.rs:241-249 parity: int strategy only preserves identity for
/// canonical cached ints. Unique small ints (from w_int_new_unique) trigger
/// de-specialization to object strategy.
///
/// For large ints (outside small cache range), the strategy always keeps them
/// as raw i64 values regardless of pointer identity.
pub unsafe fn int_strategy_preserves_identity(value: pyre_object::PyObjectRef) -> bool {
    let v = pyre_object::w_int_get_value(value);
    if pyre_object::w_int_small_cached(v) {
        // Small cached range: only canonical pointer preserves int strategy.
        std::ptr::eq(value, pyre_object::w_int_new(v))
    } else {
        // Large ints are always stored as raw i64 in int strategy.
        true
    }
}
