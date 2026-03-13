//! PyFrame — execution frame for Python bytecode.
//!
//! Each function call creates a new frame with its own value stack,
//! local variables, and instruction pointer. The JIT virtualizes
//! these fields so they live in registers instead of memory.

use std::rc::Rc;

use pyre_bytecode::CodeObject;
use pyre_bytecode::ConstantData;
use pyre_object::*;
use pyre_runtime::{PyExecutionContext, PyNamespace, w_code_new};

/// Execution frame for a single Python code block.
///
/// The JIT's Virtualize pass keeps `locals_w` and `value_stack_w`
/// in CPU registers during compiled code execution, eliminating
/// heap reads/writes for the hottest interpreter state.
///
/// The `vable_token` field coordinates ownership: when JIT code is
/// running, the token is nonzero and the canonical field values live
/// in registers. A "force" flushes them back to the heap.
pub struct PyFrame {
    /// Shared interpreter-wide execution context.
    pub execution_context: Rc<PyExecutionContext>,
    /// The code object being executed (shared via Rc for borrow-safe access).
    pub code: Rc<CodeObject>,
    /// Local variables (fast locals), indexed by varname slot.
    pub locals_w: Vec<PyObjectRef>,
    /// Operand stack for bytecode execution.
    pub value_stack_w: Vec<PyObjectRef>,
    /// Current stack depth (number of live values on value_stack_w).
    pub stack_depth: usize,
    /// Index of the next instruction to execute.
    pub next_instr: usize,
    /// Name-based local/global namespace (for module-level code).
    pub namespace: PyNamespace,
    /// Virtualizable token — set by JIT when this frame is virtualized.
    /// 0 = not virtualized, nonzero = pointer to JIT state.
    pub vable_token: usize,
}

// ── Virtualizable field offsets ───────────────────────────────────────
//
// These constants tell the JIT where each virtualizable field lives
// inside a PyFrame, so it can read/write them via raw pointer arithmetic.
// Equivalent to PyPy's `_virtualizable_` descriptor on pyframe.py.

/// Byte offset of `vable_token` in `PyFrame`.
pub const PYFRAME_VABLE_TOKEN_OFFSET: usize = std::mem::offset_of!(PyFrame, vable_token);

/// Byte offset of `next_instr` in `PyFrame`.
pub const PYFRAME_NEXT_INSTR_OFFSET: usize = std::mem::offset_of!(PyFrame, next_instr);

/// Byte offset of `stack_depth` in `PyFrame`.
pub const PYFRAME_STACK_DEPTH_OFFSET: usize = std::mem::offset_of!(PyFrame, stack_depth);

/// Byte offset of `locals_w` in `PyFrame`.
pub const PYFRAME_LOCALS_OFFSET: usize = std::mem::offset_of!(PyFrame, locals_w);

/// Byte offset of `value_stack_w` in `PyFrame`.
pub const PYFRAME_STACK_OFFSET: usize = std::mem::offset_of!(PyFrame, value_stack_w);

impl PyFrame {
    /// Create a new frame for executing a code object with a fresh execution context.
    pub fn new(code: CodeObject) -> Self {
        Self::new_with_context(code, Rc::new(PyExecutionContext::default()))
    }

    /// Create a new frame for executing a code object in the given context.
    pub fn new_with_context(code: CodeObject, execution_context: Rc<PyExecutionContext>) -> Self {
        let namespace = execution_context.fresh_namespace();
        Self::new_with_namespace(code, execution_context, namespace)
    }

    /// Create a new frame with an explicitly provided namespace.
    pub fn new_with_namespace(
        code: CodeObject,
        execution_context: Rc<PyExecutionContext>,
        namespace: PyNamespace,
    ) -> Self {
        let num_locals = code.varnames.len();
        let max_stack = code.max_stackdepth as usize;

        PyFrame {
            execution_context,
            code: Rc::new(code),
            locals_w: vec![PY_NULL; num_locals],
            value_stack_w: vec![PY_NULL; max_stack],
            stack_depth: 0,
            next_instr: 0,
            namespace,
            vable_token: 0,
        }
    }

    // ── Stack operations ──────────────────────────────────────────────

    #[inline]
    pub fn push(&mut self, value: PyObjectRef) {
        self.value_stack_w[self.stack_depth] = value;
        self.stack_depth += 1;
    }

    #[inline]
    pub fn pop(&mut self) -> PyObjectRef {
        self.stack_depth -= 1;
        self.value_stack_w[self.stack_depth]
    }

    #[inline]
    pub fn peek(&self) -> PyObjectRef {
        self.value_stack_w[self.stack_depth - 1]
    }

    #[inline]
    #[allow(dead_code)]
    pub fn peek_at(&self, depth: usize) -> PyObjectRef {
        self.value_stack_w[self.stack_depth - 1 - depth]
    }

    // ── Constant loading ──────────────────────────────────────────────

    /// Convert a RustPython ConstantData to a PyObjectRef.
    pub fn load_const(constant: &ConstantData) -> PyObjectRef {
        match constant {
            ConstantData::Integer { value } => {
                use num_traits::ToPrimitive;
                match value.to_i64() {
                    Some(v) => w_int_new(v),
                    None => w_long_new(value.clone()),
                }
            }
            ConstantData::Float { value } => w_float_new(*value),
            ConstantData::Boolean { value } => w_bool_from(*value),
            ConstantData::Str { value } => {
                w_str_new(value.as_str().expect("non-UTF-8 string constant"))
            }
            ConstantData::Tuple { elements } => {
                let items: Vec<PyObjectRef> =
                    elements.iter().map(|e| Self::load_const(e)).collect();
                w_tuple_new(items)
            }
            ConstantData::Code { code } => {
                // Clone the CodeObject, leak it, and wrap as W_CodeObject.
                // The W_CodeObject stores an opaque pointer for MakeFunction.
                let code_clone = Box::new(code.as_ref().clone());
                let code_ptr = Box::into_raw(code_clone) as *const ();
                w_code_new(code_ptr)
            }
            ConstantData::None => w_none(),
            _ => {
                // Other constant types not yet supported
                w_none()
            }
        }
    }

    /// Create a new frame for a function call.
    ///
    /// Binds positional arguments to fast locals and copies
    /// globals from the caller's namespace.
    pub fn new_for_call(
        code: CodeObject,
        args: &[PyObjectRef],
        caller_namespace: &PyNamespace,
        execution_context: Rc<PyExecutionContext>,
    ) -> Self {
        let num_locals = code.varnames.len();
        let max_stack = code.max_stackdepth as usize;

        let namespace = execution_context.inherit_namespace(caller_namespace);

        let mut locals = vec![PY_NULL; num_locals];

        // Bind positional arguments to local variable slots
        let nargs = args.len().min(num_locals);
        locals[..nargs].copy_from_slice(&args[..nargs]);

        PyFrame {
            execution_context,
            code: Rc::new(code),
            locals_w: locals,
            value_stack_w: vec![PY_NULL; max_stack],
            stack_depth: 0,
            next_instr: 0,
            namespace,
            vable_token: 0,
        }
    }
}

// ── Virtualizable configuration ──────────────────────────────────────

use majit_ir::Type;
use majit_meta::virtualizable::VirtualizableInfo;

/// Build a `VirtualizableInfo` describing PyFrame's layout.
///
/// Corresponds to PyPy's `_virtualizable_` declaration on pyframe.py:
///
/// ```text
///     _virtualizable_ = ['locals_stack_w[*]', 'valuestackdepth',
///                         'last_instr', ...]
/// ```
///
/// We use explicit byte offsets instead of name-based introspection.
/// The JIT optimizer's Virtualize pass uses this to keep frame fields
/// in CPU registers, eliminating heap accesses for LoadFast/StoreFast
/// and stack push/pop during compiled code.
///
/// Static integer fields:
/// - `next_instr` — instruction pointer
/// - `stack_depth` — number of live stack values
///
/// Array fields:
/// - `locals_w` — fast local variables
/// - `value_stack_w` — operand stack
pub fn build_pyframe_virtualizable_info() -> VirtualizableInfo {
    let mut info = VirtualizableInfo::new(PYFRAME_VABLE_TOKEN_OFFSET);

    // Static fields (scalars kept in registers)
    info.add_field("next_instr", Type::Int, PYFRAME_NEXT_INSTR_OFFSET);
    info.add_field("stack_depth", Type::Int, PYFRAME_STACK_DEPTH_OFFSET);

    // Array fields (element-wise virtualization)
    info.add_array_field("locals_w", Type::Ref, PYFRAME_LOCALS_OFFSET);
    info.add_array_field("value_stack_w", Type::Ref, PYFRAME_STACK_OFFSET);

    info
}
