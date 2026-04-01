//! PyFrame — execution frame for Python bytecode.
//!
//! Each function call creates a new frame with its own value stack,
//! local variables, and instruction pointer. The JIT virtualizes
//! these fields so they live in registers instead of memory.

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use crate::CodeObject;
use crate::{PyExecutionContext, PyNamespace, PyObjectArray};
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

#[derive(Clone, Copy)]
pub struct FrameDebugData {
    w_locals: *mut PyNamespace,
    w_globals: *mut PyNamespace,
    w_f_trace: PyObjectRef,
    is_being_profiled: bool,
    is_in_line_tracing: bool,
    instr_lb: usize,
    instr_ub: usize,
    instr_prev_plus_one: usize,
    f_lineno: usize,
    escaped: bool,
}

impl Default for FrameDebugData {
    fn default() -> Self {
        Self {
            w_locals: std::ptr::null_mut(),
            w_globals: std::ptr::null_mut(),
            w_f_trace: pyre_object::PY_NULL,
            is_being_profiled: false,
            is_in_line_tracing: false,
            instr_lb: 0,
            instr_ub: 0,
            instr_prev_plus_one: 0,
            f_lineno: 0,
            escaped: false,
        }
    }
}

thread_local! {
    static FRAME_DEBUG: RefCell<HashMap<usize, FrameDebugData>> = RefCell::new(HashMap::new());
}

/// Exception handler block — pushed by SETUP_FINALLY/SETUP_EXCEPT, popped by POP_BLOCK.
/// PyPy equivalent: pyframe.py Block classes (ExceptBlock, FinallyBlock).
#[derive(Debug, Clone, Copy)]
pub struct Block {
    pub handler: usize,
    pub level: usize,
}

#[inline]
pub fn get_block_class(opname: &str) -> &'static str {
    match opname {
        "SETUP_LOOP" | "SETUP_EXCEPT" | "SETUP_FINALLY" | "SETUP_WITH" => "Block",
        _ => "Block",
    }
}

#[inline]
pub fn unpickle_block(_space: PyObjectRef, w_tup: PyObjectRef) -> Block {
    let _ = _space;
    let handler = unsafe {
        w_tuple_getitem(w_tup, 0).and_then(|v| {
            if is_int(v) {
                Some(w_int_get_value(v) as usize)
            } else {
                None
            }
        })
    }
    .unwrap_or(0);
    let level = unsafe {
        w_tuple_getitem(w_tup, 2).and_then(|v| {
            if is_int(v) {
                Some(w_int_get_value(v) as usize)
            } else {
                None
            }
        })
    }
    .unwrap_or(0);
    Block { handler, level }
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
/// Count cell+free variable slots, excluding cellvars already in varnames.
/// CPython 3.13+ unified indexing: cellvars that overlap with varnames
/// share the same slot. Only cellvar-only variables get extra slots.
pub fn ncells(code: &CodeObject) -> usize {
    let cellvars_only = code
        .cellvars
        .iter()
        .filter(|cv| {
            let cv_name: &str = cv;
            !code.varnames.iter().any(|v| {
                let v_name: &str = v;
                v_name == cv_name
            })
        })
        .count();
    cellvars_only + code.freevars.len()
}

impl PyFrame {
    #[inline]
    fn debug_key(&self) -> usize {
        self as *const Self as usize
    }

    #[inline]
    fn with_debug_data<R>(
        &self,
        create: bool,
        op: impl FnOnce(&mut FrameDebugData) -> R,
    ) -> Option<R> {
        let key = self.debug_key();
        FRAME_DEBUG.with(|cell| {
            let mut table = cell.borrow_mut();
            let entry = if create {
                table.entry(key).or_insert(FrameDebugData {
                    w_locals: self.namespace,
                    w_globals: self.namespace,
                    f_lineno: 0,
                    w_f_trace: pyre_object::PY_NULL,
                    is_being_profiled: false,
                    is_in_line_tracing: false,
                    instr_lb: 0,
                    instr_ub: 0,
                    instr_prev_plus_one: 0,
                    escaped: false,
                })
            } else {
                table.get_mut(&key)?
            };
            if entry.w_locals.is_null() {
                entry.w_locals = self.class_locals;
            }
            if entry.w_locals.is_null() {
                entry.w_locals = self.namespace;
            }
            if entry.w_globals.is_null() {
                entry.w_globals = self.namespace;
            }
            Some(op(entry))
        })
    }

    #[inline]
    fn getdebug_data(&self) -> Option<FrameDebugData> {
        self.with_debug_data(false, |data| *data)
    }

    #[inline]
    fn getorcreate_debug_data(&self) -> FrameDebugData {
        self.with_debug_data(true, |data| *data).unwrap_or_default()
    }

    /// PyPy-compatible `getdebug()`.
    #[inline]
    pub fn getdebug(&self) -> Option<FrameDebugData> {
        self.getdebug_data()
    }

    /// PyPy-compatible `getorcreatedebug()`.
    #[inline]
    pub fn getorcreatedebug(&mut self) -> FrameDebugData {
        self.getorcreate_debug_data()
    }

    /// PyPy-compatible alias for `code()`.
    #[inline]
    pub fn getcode(&self) -> &CodeObject {
        self.code()
    }

    /// PyPy-compatible `fget_code`.
    #[inline]
    pub fn fget_code(&self) -> &CodeObject {
        self.code()
    }

    /// PyPy-compatible frame stack alias for locals/cell locals conversion.
    #[inline]
    pub fn get_w_globals(&self) -> *mut PyNamespace {
        self.getdebug_data()
            .map_or(self.namespace, |data| data.w_globals)
    }

    /// PyPy-compatible `get_w_f_trace()`.
    #[inline]
    pub fn get_w_f_trace(&self) -> PyObjectRef {
        self.getdebug_data()
            .and_then(|data| {
                if data.w_f_trace.is_null() {
                    None
                } else {
                    Some(data.w_f_trace)
                }
            })
            .unwrap_or(pyre_object::PY_NULL)
    }

    /// PyPy-compatible `get_is_being_profiled()`.
    #[inline]
    pub fn get_is_being_profiled(&self) -> bool {
        self.getdebug_data()
            .is_some_and(|data| data.is_being_profiled)
    }

    /// PyPy-compatible `get_w_locals()`.
    #[inline]
    pub fn get_w_locals(&self) -> *mut PyNamespace {
        self.getdebug_data()
            .map_or(std::ptr::null_mut(), |data| data.w_locals)
    }

    /// PyPy-compatible `getdictscope`.
    #[inline]
    pub fn getdictscope(&mut self) -> *mut PyNamespace {
        let namespace = self.get_w_locals();
        if namespace.is_null() {
            self.getorcreate_debug_data().w_locals = self.namespace;
            self.namespace
        } else {
            namespace
        }
    }

    /// PyPy-compatible `__init__` hook.
    #[inline]
    pub fn __init__(
        &mut self,
        code: *const CodeObject,
        namespace: *mut PyNamespace,
        outer_func: PyObjectRef,
    ) {
        let _ = outer_func;
        self.code = code;
        self.namespace = namespace;
        self.locals_cells_stack_w = PyObjectArray::filled(
            unsafe {
                (&*code).varnames.len()
                    + ncells(unsafe { &*code })
                    + unsafe { (&*code).max_stackdepth as usize }
            },
            PY_NULL,
        );
        self.valuestackdepth = unsafe { (&*code).varnames.len() + ncells(unsafe { &*code }) };
        self.next_instr = 0;
        self.block_stack.clear();
        self.pending_inline_results.clear();
        self.pending_inline_resume_pc = None;
        self.class_locals = std::ptr::null_mut();
        self.initialize_frame_scopes(outer_func, code);
    }

    /// PyPy-compatible `__repr__`.
    #[inline]
    pub fn __repr__(&self) -> String {
        format!("<{}>", self.get_last_lineno())
    }

    /// PyPy-compatible `fget_getdictscope`.
    #[inline]
    pub fn fget_getdictscope(&mut self) -> *mut PyNamespace {
        self.getdictscope()
    }

    /// PyPy-compatible `fget_w_globals`.
    #[inline]
    pub fn fget_w_globals(&self) -> *mut PyNamespace {
        self.get_w_globals()
    }

    /// PyPy-compatible `_getcell`.
    #[inline]
    pub fn _getcell(&self, varindex: usize) -> PyObjectRef {
        self.locals_cells_stack_w
            .as_slice()
            .get(self.nlocals() + varindex)
            .copied()
            .unwrap_or(PY_NULL)
    }

    /// PyPy-compatible `getclosure`.
    #[inline]
    pub fn getclosure(&self) -> PyObjectRef {
        PY_NULL
    }

    /// PyPy-compatible `initialize_frame_scopes`.
    #[inline]
    pub fn initialize_frame_scopes(&mut self, _outer_func: PyObjectRef, _code: *const CodeObject) {
        let _ = _outer_func;
        let _ = _code;
    }

    /// PyPy-compatible `setdictscope`.
    #[inline]
    pub fn setdictscope(&mut self, w_locals: *mut PyNamespace) {
        self.getorcreate_debug_data().w_locals = w_locals;
    }

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
        self.assert_stack_index(self.valuestackdepth);
        self.locals_cells_stack_w[self.valuestackdepth] = value;
        self.valuestackdepth += 1;
    }

    #[inline]
    pub fn pop(&mut self) -> PyObjectRef {
        assert!(self.valuestackdepth > self.stack_base());
        let depth = self.valuestackdepth - 1;
        let value = self.locals_cells_stack_w[depth];
        self.locals_cells_stack_w[depth] = PY_NULL;
        self.valuestackdepth = depth;
        value
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

    /// PyPy-compatible stack operation aliases.
    #[inline]
    pub fn pushvalue(&mut self, value: PyObjectRef) {
        self.push(value)
    }

    /// PyPy-compatible `pushvalue(None)` helper.
    #[inline]
    pub fn pushvalue_none(&mut self) {
        self.push(w_none())
    }

    /// PyPy-compatible stack index guard.
    #[inline]
    pub fn assert_stack_index(&self, index: usize) {
        debug_assert!(self._check_stack_index(index));
    }

    /// PyPy-compatible stack index validator.
    #[inline]
    pub fn _check_stack_index(&self, index: usize) -> bool {
        index >= self.stack_base()
    }

    /// PyPy-compatible `popvalue()` alias.
    #[inline]
    pub fn popvalue(&mut self) -> PyObjectRef {
        let value = self.pop();
        assert!(!value.is_null(), "popvalue on empty value stack");
        value
    }

    /// PyPy-compatible nullable pop path.
    #[inline]
    pub fn popvalue_maybe_none(&mut self) -> PyObjectRef {
        if self.valuestackdepth <= self.stack_base() {
            return PY_NULL;
        }
        self.pop()
    }

    /// PyPy `PyFrame._new_popvalues` factory.
    #[inline]
    pub fn _new_popvalues() -> fn(&mut Self, usize) -> Vec<PyObjectRef> {
        Self::popvalues
    }

    /// PyPy-compatible pop-values helper.
    #[inline]
    pub fn popvalues(&mut self, n: usize) -> Vec<PyObjectRef> {
        let mut out = vec![PY_NULL; n];
        let mut idx = n;
        while idx > 0 {
            idx -= 1;
            out[idx] = self.popvalue();
        }
        out
    }

    /// PyPy-compatible `popvalues_mutable`.
    #[inline]
    pub fn popvalues_mutable(&mut self, n: usize) -> Vec<PyObjectRef> {
        self.popvalues(n)
    }

    /// PyPy-compatible stack peek helper.
    #[inline]
    pub fn peekvalues(&self, n: usize) -> Vec<PyObjectRef> {
        let base = self.valuestackdepth.saturating_sub(n);
        self.assert_stack_index(base);
        let mut values = Vec::with_capacity(n);
        for i in base..self.valuestackdepth {
            values.push(self.locals_cells_stack_w[i]);
        }
        values
    }

    /// PyPy-compatible `dropvalues`.
    #[inline]
    pub fn dropvalues(&mut self, n: usize) {
        let finaldepth = self.valuestackdepth.saturating_sub(n);
        self.assert_stack_index(finaldepth);
        while self.valuestackdepth > finaldepth {
            self.locals_cells_stack_w[self.valuestackdepth - 1] = PY_NULL;
            self.valuestackdepth -= 1;
        }
    }

    /// PyPy-compatible `pushrevvalues`.
    #[inline]
    pub fn pushrevvalues(&mut self, _n: usize, values_w: &[PyObjectRef]) {
        let n = if _n == 0 { values_w.len() } else { _n };
        assert!(n <= values_w.len());
        let mut idx = n;
        while idx > 0 {
            idx -= 1;
            self.push(values_w[idx]);
        }
    }

    /// PyPy-compatible `dupvalues`.
    #[inline]
    pub fn dupvalues(&mut self, n: usize) {
        let values = self.peekvalues(n);
        for value in values {
            self.push(value);
        }
    }

    /// PyPy-compatible `peekvalue()`.
    #[inline]
    pub fn peekvalue(&self, index_from_top: usize) -> PyObjectRef {
        self.peek_at(index_from_top)
    }

    /// PyPy-compatible `peekvalue_maybe_none()`.
    #[inline]
    pub fn peekvalue_maybe_none(&self, index_from_top: usize) -> PyObjectRef {
        let index = self
            .valuestackdepth
            .checked_sub(index_from_top + 1)
            .unwrap_or(usize::MAX);
        if index == usize::MAX || index < self.stack_base() {
            return PY_NULL;
        }
        self.locals_cells_stack_w[index]
    }

    /// PyPy-compatible `settopvalue()`.
    #[inline]
    pub fn settopvalue(&mut self, value: PyObjectRef, index_from_top: usize) {
        let index = self
            .valuestackdepth
            .checked_sub(index_from_top + 1)
            .unwrap_or(0);
        self.assert_stack_index(index);
        assert!(index < self.valuestackdepth);
        self.locals_cells_stack_w[index] = value;
    }

    /// PyPy-compatible `dropvaluesuntil()`.
    #[inline]
    pub fn dropvaluesuntil(&mut self, finaldepth: usize) {
        self.assert_stack_index(finaldepth);
        while self.valuestackdepth > finaldepth {
            self.locals_cells_stack_w[self.valuestackdepth - 1] = PY_NULL;
            self.valuestackdepth -= 1;
        }
    }

    /// PyPy-compatible block-stack helpers.
    #[inline]
    pub fn append_block(&mut self, block: Block) {
        self.block_stack.push(block);
    }

    /// PyPy-compatible block pop helper.
    #[inline]
    pub fn pop_block(&mut self) -> Option<Block> {
        self.block_stack.pop()
    }

    /// PyPy-compatible block list check.
    #[inline]
    pub fn blockstack_non_empty(&self) -> bool {
        !self.block_stack.is_empty()
    }

    /// PyPy-compatible exception-info unwind helper.
    #[inline]
    pub fn _exc_info_unroll(&self, _for_hidden: bool) -> PyObjectRef {
        let _ = _for_hidden;
        pyre_object::w_none()
    }

    /// PyPy-compatible unexpected-exception converter.
    #[inline]
    pub fn _convert_unexpected_exception(&self, _e: PyObjectRef) -> PyObjectRef {
        let _ = _e;
        pyre_object::w_none()
    }

    /// PyPy-compatible pickle state helper.
    #[inline]
    pub fn _reduce_state(&self) -> PyObjectRef {
        pyre_object::w_tuple_new(vec![
            pyre_object::w_none(),
            pyre_object::w_none(),
            pyre_object::w_none(),
            pyre_object::w_int_new(self.next_instr as i64),
            pyre_object::w_int_new(self.valuestackdepth as i64),
        ])
    }

    /// PyPy-compatible `descr__reduce__`.
    #[inline]
    pub fn descr__reduce__(&self) -> PyObjectRef {
        pyre_object::w_tuple_new(vec![
            pyre_object::w_none(),
            pyre_object::w_none(),
            self._reduce_state(),
        ])
    }

    /// PyPy-compatible `descr__setstate__`.
    #[inline]
    pub fn descr__setstate__(&mut self, _state: PyObjectRef) {
        let _ = _state;
    }

    /// PyPy-compatible materialized block list.
    #[inline]
    pub fn get_blocklist(&self) -> Vec<Block> {
        self.block_stack.iter().rev().copied().collect()
    }

    /// PyPy-compatible block list restore.
    #[inline]
    pub fn set_blocklist(&mut self, lst: &[Block]) {
        self.block_stack = lst.iter().rev().copied().collect();
    }

    /// PyPy-compatible execution entrypoint.
    #[inline]
    pub fn run(&mut self) -> crate::PyResult {
        crate::eval::eval_frame_plain(self)
    }

    /// PyPy-compatible execution entrypoint with optional inbound values.
    #[inline]
    #[allow(unused_variables)]
    pub fn execute_frame(
        &mut self,
        _w_inputvalue: Option<PyObjectRef>,
        _operr: Option<PyObjectRef>,
    ) -> crate::PyResult {
        self.run()
    }

    /// PyPy-compatible `hide`.
    #[inline]
    pub fn hide(&self) -> bool {
        false
    }

    /// PyPy-compatible `mark_as_escaped`.
    #[inline]
    pub fn mark_as_escaped(&mut self) {
        self.getorcreate_debug_data().escaped = true;
    }

    /// PyPy-compatible `get_builtin`.
    #[inline]
    pub fn get_builtin(&self) -> PyObjectRef {
        pyre_object::PY_NULL
    }

    /// PyPy-compatible `get_f_back`.
    #[inline]
    pub fn get_f_back(&self) -> *mut PyFrame {
        std::ptr::null_mut()
    }

    /// PyPy-compatible `fget_f_builtins`.
    #[inline]
    pub fn fget_f_builtins(&self) -> PyObjectRef {
        self.get_builtin()
    }

    /// PyPy-compatible `fget_f_back`.
    #[inline]
    pub fn fget_f_back(&self) -> *mut PyFrame {
        self.get_f_back()
    }

    /// PyPy-compatible `fget_f_lasti`.
    #[inline]
    pub fn fget_f_lasti(&self) -> usize {
        self.next_instr
    }

    /// PyPy-compatible `fget_f_trace`.
    #[inline]
    pub fn fget_f_trace(&self) -> PyObjectRef {
        self.get_w_f_trace()
    }

    /// PyPy-compatible `fset_f_trace`.
    #[inline]
    pub fn fset_f_trace(&mut self, w_trace: PyObjectRef) {
        self.getorcreate_debug_data().w_f_trace = w_trace;
    }

    /// PyPy-compatible `fdel_f_trace`.
    #[inline]
    pub fn fdel_f_trace(&mut self) {
        self.getorcreate_debug_data().w_f_trace = pyre_object::PY_NULL;
    }

    /// PyPy-compatible `fget_f_exc_type`.
    #[inline]
    pub fn fget_f_exc_type(&self) -> PyObjectRef {
        pyre_object::PY_NULL
    }

    /// PyPy-compatible `fget_f_exc_value`.
    #[inline]
    pub fn fget_f_exc_value(&self) -> PyObjectRef {
        pyre_object::PY_NULL
    }

    /// PyPy-compatible `fget_f_exc_traceback`.
    #[inline]
    pub fn fget_f_exc_traceback(&self) -> PyObjectRef {
        pyre_object::PY_NULL
    }

    /// PyPy-compatible `fget_f_restricted`.
    #[inline]
    pub fn fget_f_restricted(&self) -> bool {
        false
    }

    /// PyPy-compatible `get_f_lineno`.
    #[inline]
    pub fn get_last_lineno(&self) -> usize {
        self.next_instr
    }

    /// PyPy-compatible `fget_f_lineno`.
    #[inline]
    pub fn fget_f_lineno(&self) -> usize {
        if self.get_w_f_trace().is_null() {
            self.get_last_lineno()
        } else {
            self.getorcreate_debug_data().f_lineno
        }
    }

    /// PyPy-compatible `fset_f_lineno`.
    #[inline]
    pub fn fset_f_lineno(&mut self, new_f_lineno: usize) {
        self.getorcreate_debug_data().f_lineno = new_f_lineno;
        self.next_instr = new_f_lineno;
    }

    /// PyPy-compatible `setfastscope`.
    #[inline]
    pub fn setfastscope(&mut self, scope_w: &[PyObjectRef]) {
        assert!(scope_w.len() <= self.nlocals());
        for (index, value) in scope_w.iter().copied().enumerate() {
            self.locals_cells_stack_w[index] = value;
        }
        // In this port, cell initialization is performed as part of scope load.
        self.init_cells();
    }

    /// PyPy-compatible `locals2fast`.
    #[inline]
    pub fn locals2fast(&mut self) {
        let namespace = self.get_w_locals();
        assert!(!namespace.is_null());
        let namespace = unsafe { &*namespace };
        let varnames = self.code().varnames.clone();
        let mut fast_slots = vec![PY_NULL; self.nlocals()];
        for (i, name) in varnames.iter().enumerate() {
            fast_slots[i] = namespace.get(name).copied().unwrap_or(PY_NULL);
        }
        for (i, value) in fast_slots.iter().enumerate() {
            self.locals_cells_stack_w[i] = *value;
        }
    }

    /// PyPy-compatible `init_cells`.
    #[inline]
    pub fn init_cells(&mut self) {
        let ncellvars = self.code().cellvars.len();
        let num_locals = self.code().varnames.len();
        let base = num_locals;
        for i in 0..ncellvars {
            if base + i >= self.locals_cells_stack_w.len() {
                break;
            }
            self.locals_cells_stack_w[base + i] = self.locals_cells_stack_w[i];
        }
    }

    /// PyPy-compatible `fast2locals`.
    #[inline]
    pub fn fast2locals(&mut self) {
        let namespace = match self.getdictscope() {
            namespace if namespace.is_null() => return,
            namespace => unsafe { &mut *namespace },
        };
        namespace.clear();
        let locals = self.locals_cells_stack_w.as_slice();
        let code = self.code();

        for (name, value) in code.varnames.iter().zip(locals.iter()) {
            if !value.is_null() {
                namespace.insert(name.to_string(), *value);
            }
        }

        let num_locals = code.varnames.len();
        let cellvars_only = code
            .cellvars
            .iter()
            .filter(|cv| !code.varnames.iter().any(|v| v == *cv))
            .count();

        for (slot, name) in code
            .cellvars
            .iter()
            .filter(|cv| !code.varnames.iter().any(|v| v == *cv))
            .enumerate()
        {
            let idx = num_locals + slot;
            if let Some(value) = locals.get(idx).copied() {
                if !value.is_null() {
                    namespace.insert((*name).to_string(), value);
                }
            }
        }

        for (slot, name) in code.freevars.iter().enumerate() {
            let idx = num_locals + cellvars_only + slot;
            if let Some(value) = locals.get(idx).copied() {
                if !value.is_null() {
                    namespace.insert(name.to_string(), value);
                }
            }
        }
    }

    /// PyPy-compatible `setdictscope` and locals conversion.
    #[inline]
    pub fn setdictscope_and_fast(&mut self, w_locals: *mut PyNamespace) {
        self.setdictscope(w_locals);
        self.fast2locals();
    }

    /// PyPy-compatible `make_arguments`.
    #[inline]
    pub fn make_arguments(&self, nargs: usize, _methodcall: bool) -> Vec<PyObjectRef> {
        self.peekvalues(nargs)
    }

    /// PyPy-compatible argument list builder.
    #[inline]
    #[allow(unused_variables)]
    pub fn argument_factory(
        &self,
        _arguments: &[PyObjectRef],
        _keywords: &[PyObjectRef],
        _keywords_w: &[PyObjectRef],
        _w_star: PyObjectRef,
        _w_starstar: PyObjectRef,
        _methodcall: bool,
    ) -> Vec<PyObjectRef> {
        let mut args = Vec::new();
        args.extend_from_slice(_arguments);
        args.extend_from_slice(_keywords);
        args.extend_from_slice(_keywords_w);
        if !_w_star.is_null() {
            args.push(_w_star);
        }
        if !_w_starstar.is_null() {
            args.push(_w_starstar);
        }
        args
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
        // Freevars go after cellvar-only slots: indices nlocals+ncellvars_only..
        if !closure.is_null() {
            let n_cellvars_only = num_cells - code_ref.freevars.len();
            let n_freevars = code_ref.freevars.len();
            for i in 0..n_freevars {
                let cell = unsafe { w_tuple_getitem(closure, i as i64).unwrap() };
                locals_cells_stack_w[num_locals + n_cellvars_only + i] = cell;
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

    /// Load a constant from the code object by raw index.
    /// Used by the blackhole interpreter's bh_load_const_fn.
    pub fn load_const_pyobj(&self, idx: usize) -> PyObjectRef {
        use crate::bytecode::ConstantData;
        use num_traits::ToPrimitive;
        let code = self.code();
        // RPython: constants are in JitCode.constants_r. In pyre, we resolve
        // from the CodeObject's constant table at runtime.
        let constants: &[ConstantData] = unsafe {
            std::slice::from_raw_parts(
                code.constants.as_ptr() as *const ConstantData,
                code.constants.len(),
            )
        };
        if idx >= constants.len() {
            return pyre_object::w_none();
        }
        match &constants[idx] {
            ConstantData::Integer { value } => {
                pyre_object::intobject::w_int_new(value.to_i64().unwrap_or(0))
            }
            ConstantData::Float { value } => pyre_object::floatobject::w_float_new(*value),
            ConstantData::Boolean { value } => {
                pyre_object::intobject::w_int_new(if *value { 1 } else { 0 })
            }
            ConstantData::None => pyre_object::w_none(),
            _ => pyre_object::w_none(),
        }
    }
}

/// Load a constant from a CodeObject without a PyFrame.
/// Used by the blackhole's bh_load_const_fn when the code pointer
/// comes from a virtualizable field read.
pub fn load_const_from_code(code: &CodeObject, idx: usize) -> PyObjectRef {
    use crate::bytecode::ConstantData;
    use num_traits::ToPrimitive;
    let constants: &[ConstantData] = unsafe {
        std::slice::from_raw_parts(
            code.constants.as_ptr() as *const ConstantData,
            code.constants.len(),
        )
    };
    if idx >= constants.len() {
        return pyre_object::w_none();
    }
    match &constants[idx] {
        ConstantData::Integer { value } => {
            pyre_object::intobject::w_int_new(value.to_i64().unwrap_or(0))
        }
        ConstantData::Float { value } => pyre_object::floatobject::w_float_new(*value),
        ConstantData::Boolean { value } => {
            pyre_object::intobject::w_int_new(if *value { 1 } else { 0 })
        }
        ConstantData::None => pyre_object::w_none(),
        _ => pyre_object::w_none(),
    }
}

// Virtualizable configuration is in jit/frame_layout.rs
