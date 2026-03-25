//! PyFrame — execution frame for Python bytecode.
//!
//! Each function call creates a new frame with its own value stack,
//! local variables, and instruction pointer. The JIT virtualizes
//! these fields so they live in registers instead of memory.

use std::collections::VecDeque;
use std::rc::Rc;

use crate::{PyExecutionContext, PyNamespace, PyObjectArray};
use pyre_bytecode::CodeObject;
use pyre_object::*;

// Ensure *const PyExecutionContext and Rc<PyExecutionContext> have the same
// size so that PyFrame field offsets are preserved after the switch.
const _: () = assert!(
    std::mem::size_of::<*const PyExecutionContext>()
        == std::mem::size_of::<Rc<PyExecutionContext>>()
);

#[derive(Debug, Clone, Copy)]
pub enum PendingInlineResult {
    Ref(PyObjectRef),
    Int(i64),
    Float(f64),
}

/// Execution frame for a single Python code block.
///
/// Unified `locals_cells_stack_w` array layout:
///   - indices `0..nlocals` — local variables
///   - indices `nlocals..nlocals+ncells` — cell/free variable slots
///   - indices `nlocals+ncells..` — operand stack
///
/// `valuestackdepth` is the absolute index into this array; it starts at
/// `nlocals + ncells` (empty stack) and grows upward on push.
///
/// The JIT's Virtualize pass keeps `locals_cells_stack_w` slots in CPU
/// registers during compiled code execution, eliminating heap reads/writes
/// for the hottest interpreter state.
///
/// The `vable_token` field coordinates ownership: when JIT code is
/// running, the token is nonzero and the canonical field values live
/// in registers. A "force" flushes them back to the heap.
#[repr(C)]
pub struct PyFrame {
    /// Raw pointer to the shared execution context.
    /// The top-level frame leaks the Rc via `Rc::into_raw`.
    /// Callee frames just copy the pointer (no atomic refcount ops).
    pub execution_context: *const PyExecutionContext,
    /// Raw pointer to the code object (shared, not owned — the CodeObject
    /// is leaked via `Box::into_raw` at creation time and lives forever).
    pub code: *const CodeObject,
    /// Unified locals + cells + operand stack array.
    pub locals_cells_stack_w: PyObjectArray,
    /// Absolute index into `locals_cells_stack_w` marking the top of the
    /// operand stack. Starts at `nlocals + ncells` (empty stack), grows upward.
    pub valuestackdepth: usize,
    /// Index of the next instruction to execute.
    pub next_instr: usize,
    /// Raw pointer to the shared globals namespace object.
    /// All frames in the same module share the same globals.
    pub namespace: *mut PyNamespace,
    /// Virtualizable token — set by JIT when this frame is virtualized.
    /// 0 = not virtualized, nonzero = pointer to JIT state.
    pub vable_token: usize,
    /// Exception handler block stack (PyPy: lastblock linked list).
    /// NOT repr(C)-visible to JIT — stored after vable_token.
    pub block_stack: Vec<Block>,
    /// Concrete inline-trace replay results owned by this frame.
    ///
    /// PyPy's `finishframe()` writes each child result into the parent in
    /// bytecode order. Tracing can run ahead of concrete execution and queue
    /// multiple inline-handled CALL results on the same caller frame before
    /// the interpreter replays the first CALL opcode, so this must preserve
    /// ordering instead of using a single overwrite-prone slot.
    pub pending_inline_results: VecDeque<PendingInlineResult>,
    /// Outermost inline trace-through resumed this frame past the CALL at the
    /// recorded pc.
    ///
    /// Nested inline frames already follow the MIFrame-owned
    /// make_result_of_lastop path directly. This marker narrows the same
    /// protocol to the outermost interpreter loop without conflating it with
    /// unrelated next_instr changes.
    pub pending_inline_resume_pc: Option<usize>,
    /// Optional class-body local namespace.
    ///
    /// PyPy equivalent: pyframe.py has separate `w_locals` and `w_globals`.
    /// When set (non-null), STORE_NAME writes here instead of `namespace`,
    /// and LOAD_NAME checks here first before falling back to `namespace`.
    /// Used for class body execution where locals ≠ globals.
    pub class_locals: *mut PyNamespace,
}

/// Exception handler block — pushed by SETUP_FINALLY/SETUP_EXCEPT, popped by POP_BLOCK.
/// PyPy equivalent: pyframe.py Block classes (ExceptBlock, FinallyBlock).
#[derive(Debug, Clone, Copy)]
pub struct Block {
    pub handler: usize,
    pub level: usize,
}

// ── Virtualizable field offsets ───────────────────────────────────────
//
// These constants tell the JIT where each virtualizable field lives
// inside a PyFrame, so it can read/write them via raw pointer arithmetic.
// Equivalent to PyPy's `_virtualizable_` descriptor on pyframe.py.

/// Byte offset of `code` in `PyFrame`.
pub const PYFRAME_CODE_OFFSET: usize = std::mem::offset_of!(PyFrame, code);

/// Byte offset of `vable_token` in `PyFrame`.
pub const PYFRAME_VABLE_TOKEN_OFFSET: usize = std::mem::offset_of!(PyFrame, vable_token);

/// Byte offset of `next_instr` in `PyFrame`.
pub const PYFRAME_NEXT_INSTR_OFFSET: usize = std::mem::offset_of!(PyFrame, next_instr);

/// Byte offset of `valuestackdepth` in `PyFrame`.
pub const PYFRAME_VALUESTACKDEPTH_OFFSET: usize = std::mem::offset_of!(PyFrame, valuestackdepth);

/// Byte offset of `locals_cells_stack_w` in `PyFrame`.
pub const PYFRAME_LOCALS_CELLS_STACK_OFFSET: usize =
    std::mem::offset_of!(PyFrame, locals_cells_stack_w);

// Backward-compat aliases used by JIT code.
pub const PYFRAME_STACK_DEPTH_OFFSET: usize = PYFRAME_VALUESTACKDEPTH_OFFSET;
pub const PYFRAME_LOCALS_OFFSET: usize = PYFRAME_LOCALS_CELLS_STACK_OFFSET;

/// Number of cell + free variable slots for a code object.
#[inline]
fn ncells(code: &CodeObject) -> usize {
    code.cellvars.len() + code.freevars.len()
}

impl PyFrame {
    /// Create a minimal frame stub for passing to call dispatch.
    /// Used by MIFrame Box tracking when concrete_frame is unavailable.
    pub fn new_minimal(
        code: *const CodeObject,
        namespace: *mut crate::PyNamespace,
        execution_context: *const PyExecutionContext,
    ) -> Self {
        let nlocals = unsafe { (&*code).varnames.len() };
        let ncells = unsafe { (&*code).cellvars.len() + (&*code).freevars.len() };
        let size = nlocals + ncells + 16; // small stack
        PyFrame {
            execution_context,
            code,
            locals_cells_stack_w: crate::PyObjectArray::from_vec(vec![pyre_object::PY_NULL; size]),
            valuestackdepth: nlocals + ncells,
            next_instr: 0,
            namespace,
            vable_token: 0,
            block_stack: Vec::new(),
            pending_inline_results: std::collections::VecDeque::new(),
            pending_inline_resume_pc: None,
            class_locals: std::ptr::null_mut(),
        }
    }

    /// Create a new frame for executing a code object with a fresh execution context.
    pub fn new(code: CodeObject) -> Self {
        Self::new_with_context(code, Rc::new(PyExecutionContext::default()))
    }

    /// Create a new frame for executing a code object in the given context.
    ///
    /// The `Rc` is leaked via `Rc::into_raw` — consistent with pyre's
    /// memory model where code objects and namespaces are also leaked.
    pub fn new_with_context(code: CodeObject, execution_context: Rc<PyExecutionContext>) -> Self {
        let mut namespace = Box::new(execution_context.fresh_namespace());
        namespace.fix_ptr();
        // Set __name__ — PyPy: Module.__init__ sets __name__ in w_dict
        crate::namespace_store(
            &mut namespace,
            "__name__",
            pyre_object::w_str_new("__main__"),
        );
        let namespace = Box::into_raw(namespace);
        let code_ptr = Box::into_raw(Box::new(code));
        let ctx_ptr = Rc::into_raw(execution_context);
        Self::new_with_namespace(code_ptr, ctx_ptr, namespace)
    }

    /// Create a new frame with an explicitly provided namespace pointer.
    pub fn new_with_namespace(
        code: *const CodeObject,
        execution_context: *const PyExecutionContext,
        namespace: *mut PyNamespace,
    ) -> Self {
        let code_ref = unsafe { &*code };
        let num_locals = code_ref.varnames.len();
        let num_cells = ncells(code_ref);
        let max_stack = code_ref.max_stackdepth as usize;

        PyFrame {
            execution_context,
            code,
            locals_cells_stack_w: PyObjectArray::filled(
                num_locals + num_cells + max_stack,
                PY_NULL,
            ),
            valuestackdepth: num_locals + num_cells,
            next_instr: 0,
            namespace,
            vable_token: 0,
            block_stack: Vec::new(),
            pending_inline_results: VecDeque::new(),
            pending_inline_resume_pc: None,
            class_locals: std::ptr::null_mut(),
        }
    }

    /// RPython MetaInterp traces against its own MIFrame stack instead of
    /// mutating the live interpreter frame in place. pyre still executes
    /// bytecodes concretely during tracing, so use an owned snapshot when
    /// recording a trace to keep the real frame state unchanged until the
    /// interpreter actually executes the same path.
    pub fn snapshot_for_tracing(&self) -> Self {
        let mut frame = PyFrame {
            execution_context: self.execution_context,
            code: self.code,
            locals_cells_stack_w: PyObjectArray::from_vec(self.locals_cells_stack_w.to_vec()),
            valuestackdepth: self.valuestackdepth,
            next_instr: self.next_instr,
            namespace: self.namespace,
            vable_token: self.vable_token,
            block_stack: self.block_stack.clone(),
            pending_inline_results: self.pending_inline_results.clone(),
            pending_inline_resume_pc: self.pending_inline_resume_pc,
            class_locals: self.class_locals,
        };
        frame.fix_array_ptrs();
        frame
    }

    /// Number of local variable slots (from code object).
    #[inline]
    pub fn nlocals(&self) -> usize {
        unsafe { (&(*self.code).varnames).len() }
    }

    /// Number of cell + free variable slots.
    #[inline]
    pub fn ncells(&self) -> usize {
        unsafe { ncells(&*self.code) }
    }

    /// First index of the operand stack (after locals and cells).
    #[inline]
    pub fn stack_base(&self) -> usize {
        self.nlocals() + self.ncells()
    }

    // ── Stack operations ──────────────────────────────────────────────

    #[inline]
    pub fn push(&mut self, value: PyObjectRef) {
        self.locals_cells_stack_w[self.valuestackdepth] = value;
        self.valuestackdepth += 1;
    }

    #[inline]
    pub fn pop(&mut self) -> PyObjectRef {
        self.valuestackdepth -= 1;
        self.locals_cells_stack_w[self.valuestackdepth]
    }

    #[inline]
    pub fn peek(&self) -> PyObjectRef {
        self.locals_cells_stack_w[self.valuestackdepth - 1]
    }

    #[inline]
    #[allow(dead_code)]
    pub fn peek_at(&self, depth: usize) -> PyObjectRef {
        self.locals_cells_stack_w[self.valuestackdepth - 1 - depth]
    }

    /// Create a new frame for a function call.
    ///
    /// The `globals` pointer is shared from the function object -- no clone.
    /// The `code` pointer is shared from the function object -- no clone.
    /// `closure` is a tuple of cell objects from the enclosing scope,
    /// or PY_NULL if the function has no free variables.
    pub fn new_for_call(
        code: *const CodeObject,
        args: &[PyObjectRef],
        globals: *mut PyNamespace,
        execution_context: *const PyExecutionContext,
    ) -> Self {
        Self::new_for_call_with_closure(code, args, globals, execution_context, PY_NULL)
    }

    /// Create a new frame for a function call with a closure.
    pub fn new_for_call_with_closure(
        code: *const CodeObject,
        args: &[PyObjectRef],
        globals: *mut PyNamespace,
        execution_context: *const PyExecutionContext,
        closure: PyObjectRef,
    ) -> Self {
        let code_ref = unsafe { &*code };
        let num_locals = code_ref.varnames.len();
        let num_cells = ncells(code_ref);
        let max_stack = code_ref.max_stackdepth as usize;

        let mut locals_cells_stack_w =
            PyObjectArray::filled(num_locals + num_cells + max_stack, PY_NULL);

        // Bind positional arguments directly -- no intermediate Vec.
        let nargs = args.len().min(num_locals);
        for i in 0..nargs {
            locals_cells_stack_w[i] = args[i];
        }

        // Copy free variables from closure tuple into the frame's cell slots.
        // Free vars go after cell vars: indices nlocals+ncellvars..nlocals+ncells
        if !closure.is_null() {
            let n_cellvars = code_ref.cellvars.len();
            let n_freevars = code_ref.freevars.len();
            for i in 0..n_freevars {
                let cell = unsafe { w_tuple_getitem(closure, i as i64).unwrap() };
                locals_cells_stack_w[num_locals + n_cellvars + i] = cell;
            }
        }

        PyFrame {
            execution_context,
            code,
            locals_cells_stack_w,
            valuestackdepth: num_locals + num_cells,
            next_instr: 0,
            namespace: globals,
            vable_token: 0,
            block_stack: Vec::new(),
            pending_inline_results: VecDeque::new(),
            pending_inline_resume_pc: None,
            class_locals: std::ptr::null_mut(),
        }
    }

    /// Borrow the shared code object.
    #[inline]
    pub fn code(&self) -> &CodeObject {
        unsafe { &*self.code }
    }

    /// Repoint internal array pointers after a struct move.
    ///
    /// `PyObjectArray` with small-buffer optimization stores an inline
    /// buffer whose address changes on move. Call this once after the
    /// frame is at its final stack location.
    #[inline]
    pub fn fix_array_ptrs(&mut self) {
        self.locals_cells_stack_w.fix_ptr();
    }
}

// Virtualizable configuration is in jit/frame_layout.rs
