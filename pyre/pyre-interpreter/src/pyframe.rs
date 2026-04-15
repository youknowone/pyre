//! PyFrame — execution frame for Python bytecode.
//!
//! Each function call creates a new frame with its own value stack,
//! local variables, and instruction pointer. The JIT virtualizes
//! these fields so they live in registers instead of memory.

use std::collections::VecDeque;
use std::rc::Rc;

use crate::CodeObject;
use crate::{PyExecutionContext, PyNamespace};
use pyre_object::FixedObjectArray;
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
    /// Pointer to the Code object (W_CodeObject).
    ///
    /// PyPy: pyframe.py `self.pycode = code` — stores the PyCode instance.
    /// Same pointer as `func.getcode()`, so `getcode(func) == frame.code`.
    pub code: *const (),
    /// pypy/interpreter/pyframe.py:84,110-112 locals_cells_stack_w
    /// `[None] * size; make_sure_not_resized(...)` → fixed-length GcArray.
    pub locals_cells_stack_w: *mut FixedObjectArray,
    /// Absolute index into `locals_cells_stack_w` marking the top of the
    /// operand stack. Starts at `nlocals + ncells` (empty stack), grows upward.
    pub valuestackdepth: usize,
    /// Index of the next instruction to execute.
    pub next_instr: usize,
    /// Raw pointer to the shared globals namespace object.
    /// All frames in the same module share the same globals.
    pub namespace: *mut PyNamespace,
    /// pyframe.py:82 debugdata — pointer to FrameDebugData or null.
    /// Virtualizable static field (interp_jit.py:28).
    pub debugdata: usize,
    /// pyframe.py:86 lastblock — pointer to top Block or null.
    /// Virtualizable static field (interp_jit.py:29).
    pub lastblock: usize,
    /// Virtualizable token — set by JIT when this frame is virtualized.
    /// 0 = not virtualized, nonzero = pointer to JIT state.
    pub vable_token: usize,
    /// pyframe.py:80 escaped — see mark_as_escaped()
    pub escaped: bool,
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

/// GC header size in bytes.  Matches `majit_gc::header::GcHeader::SIZE`.
/// Every PyObjectArray and PyFrame allocation prepends this many zero bytes
/// so that RPython-style write barriers (`*(obj + wb_byteofs) & mask`) read a
/// valid header with `TRACK_YOUNG_PTRS=0` and skip the slow path.
pub const GC_HEADER_SIZE: usize = 8;

/// Allocate a `FixedObjectArray` with a zeroed GC header prepended.
///
/// incminimark write barrier reads at `obj - HEADER_SIZE`;
/// zeroed header ⇒ `TRACK_YOUNG_PTRS` clear ⇒ barrier fast-path skips.
pub unsafe fn alloc_array_with_gc_header(array: FixedObjectArray) -> *mut FixedObjectArray {
    let total = GC_HEADER_SIZE + std::mem::size_of::<FixedObjectArray>();
    let layout = std::alloc::Layout::from_size_align(total, 8).unwrap();
    let raw = std::alloc::alloc_zeroed(layout);
    if raw.is_null() {
        std::alloc::handle_alloc_error(layout);
    }
    let ptr = raw.add(GC_HEADER_SIZE) as *mut FixedObjectArray;
    std::ptr::write(ptr, array);
    ptr
}

/// Deallocate a `FixedObjectArray` allocated with [`alloc_array_with_gc_header`].
pub unsafe fn dealloc_array_with_gc_header(ptr: *mut FixedObjectArray) {
    std::ptr::drop_in_place(ptr);
    let raw = (ptr as *mut u8).sub(GC_HEADER_SIZE);
    let total = GC_HEADER_SIZE + std::mem::size_of::<FixedObjectArray>();
    let layout = std::alloc::Layout::from_size_align(total, 8).unwrap();
    std::alloc::dealloc(raw, layout);
}

impl Drop for PyFrame {
    fn drop(&mut self) {
        if !self.locals_cells_stack_w.is_null() {
            unsafe { dealloc_array_with_gc_header(self.locals_cells_stack_w) };
            self.locals_cells_stack_w = std::ptr::null_mut();
        }
        self.clear_blocks();
        if self.debugdata != 0 {
            unsafe { drop(Box::from_raw(self.debugdata as *mut FrameDebugData)) };
            self.debugdata = 0;
        }
    }
}

impl PyFrame {
    /// Free all blocks in the linked list.
    pub fn clear_blocks(&mut self) {
        while self.lastblock != 0 {
            let node = unsafe { Box::from_raw(self.lastblock as *mut Block) };
            self.lastblock = node.previous;
        }
    }

    /// Deep-clone debugdata for snapshot/generator.
    pub fn clone_debugdata(&self) -> usize {
        if self.debugdata == 0 {
            return 0;
        }
        let src = unsafe { &*(self.debugdata as *const FrameDebugData) };
        let copy = Box::new(*src);
        Box::into_raw(copy) as usize
    }

    /// Deep-clone the block linked list for snapshot/generator.
    pub fn clone_blocks(&self) -> usize {
        let list = self.get_blocklist();
        let mut head: usize = 0;
        // Rebuild in reverse (get_blocklist returns head-first)
        for block in list.iter().rev() {
            let node = Box::new(Block {
                handler: block.handler,
                level: block.level,
                previous: head,
            });
            head = Box::into_raw(node) as usize;
        }
        head
    }
}

impl PyFrame {
    /// Access locals_cells_stack_w (deref the pointer).
    #[inline]
    pub fn locals_w(&self) -> &FixedObjectArray {
        unsafe { &*self.locals_cells_stack_w }
    }

    /// Mutably access locals_cells_stack_w.
    #[inline]
    pub fn locals_w_mut(&mut self) -> &mut FixedObjectArray {
        unsafe { &mut *self.locals_cells_stack_w }
    }
}

/// Extract raw CodeObject from frame's W_CodeObject.
///
/// PyPy: `frame.pycode` gives `PyCode` which IS the code object.
/// pyre: W_CodeObject wraps a raw CodeObject — this extracts it.
#[inline]
pub unsafe fn pyframe_get_pycode(frame: &PyFrame) -> *const CodeObject {
    crate::w_code_get_ptr(frame.code as pyre_object::PyObjectRef) as *const CodeObject
}

/// pyframe.py:34-49 FrameDebugData
#[derive(Clone, Copy)]
pub struct FrameDebugData {
    /// pyframe.py:37
    pub w_f_trace: PyObjectRef,
    /// pyframe.py:38
    pub instr_prev_plus_one: usize,
    /// pyframe.py:39
    pub f_lineno: usize,
    /// pyframe.py:40
    pub is_being_profiled: bool,
    /// pyframe.py:41
    pub is_in_line_tracing: bool,
    /// pyframe.py:42
    pub f_trace_lines: bool,
    /// pyframe.py:43
    pub f_trace_opcodes: bool,
    /// pyframe.py:44
    pub w_locals: *mut PyNamespace,
    /// pyframe.py:45
    pub hidden_operationerr: PyObjectRef,
    /// pyframe.py:49 — set in __init__ from pycode.w_globals
    pub w_globals: *mut PyNamespace,
}

impl Default for FrameDebugData {
    fn default() -> Self {
        Self {
            w_f_trace: pyre_object::PY_NULL,
            instr_prev_plus_one: 0,
            f_lineno: 0,
            is_being_profiled: false,
            is_in_line_tracing: false,
            f_trace_lines: true,
            f_trace_opcodes: false,
            w_locals: std::ptr::null_mut(),
            hidden_operationerr: pyre_object::PY_NULL,
            w_globals: std::ptr::null_mut(),
        }
    }
}

/// pyopcode.py:1875-1897 FrameBlock — linked list node for the block stack.
/// `previous` forms a singly-linked list; `lastblock` in PyFrame is the head.
#[derive(Debug, Clone, Copy)]
pub struct Block {
    /// pyopcode.py:1882 handlerposition
    pub handler: usize,
    /// pyopcode.py:1881 valuestackdepth
    pub level: usize,
    /// pyopcode.py:1884 previous — raw pointer to the previous Block (0 = None).
    pub previous: usize,
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
    Block {
        handler,
        level,
        previous: 0,
    }
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

/// Byte offset of `debugdata` in `PyFrame`.
pub const PYFRAME_DEBUGDATA_OFFSET: usize = std::mem::offset_of!(PyFrame, debugdata);

/// Byte offset of `lastblock` in `PyFrame`.
pub const PYFRAME_LASTBLOCK_OFFSET: usize = std::mem::offset_of!(PyFrame, lastblock);

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
    /// pyframe.py:121 getdebug → self.debugdata
    #[inline]
    pub fn getdebug(&self) -> *mut FrameDebugData {
        self.debugdata as *mut FrameDebugData
    }

    /// pyframe.py:124 getorcreatedebug
    #[inline]
    pub fn getorcreatedebug(&mut self) -> &mut FrameDebugData {
        if self.debugdata == 0 {
            // pyframe.py:124-127: FrameDebugData(self.pycode, init_lineno=-1)
            let mut dd = Box::new(FrameDebugData::default());
            dd.w_globals = self.namespace; // pyframe.py:49
            self.debugdata = Box::into_raw(dd) as usize;
        }
        unsafe { &mut *(self.debugdata as *mut FrameDebugData) }
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

    /// pyframe.py:129 get_w_globals
    #[inline]
    pub fn get_w_globals(&self) -> *mut PyNamespace {
        let dd = self.getdebug();
        if !dd.is_null() {
            let w = unsafe { (*dd).w_globals };
            if !w.is_null() {
                return w;
            }
        }
        self.namespace
    }

    /// pyframe.py:135 get_w_f_trace
    #[inline]
    pub fn get_w_f_trace(&self) -> PyObjectRef {
        let dd = self.getdebug();
        if !dd.is_null() {
            return unsafe { (*dd).w_f_trace };
        }
        pyre_object::PY_NULL
    }

    /// pyframe.py:141 get_is_being_profiled
    #[inline]
    pub fn get_is_being_profiled(&self) -> bool {
        let dd = self.getdebug();
        if !dd.is_null() {
            return unsafe { (*dd).is_being_profiled };
        }
        false
    }

    /// pyframe.py:147 get_w_locals
    #[inline]
    pub fn get_w_locals(&self) -> *mut PyNamespace {
        let dd = self.getdebug();
        if !dd.is_null() {
            return unsafe { (*dd).w_locals };
        }
        std::ptr::null_mut()
    }

    /// pyframe.py:228 getdictscope
    #[inline]
    pub fn getdictscope(&mut self) -> *mut PyNamespace {
        let w_locals = self.get_w_locals();
        if !w_locals.is_null() {
            return w_locals;
        }
        self.getorcreatedebug().w_locals = self.namespace;
        self.namespace
    }

    /// PyPy-compatible `__init__` hook.
    #[inline]
    pub fn __init__(
        &mut self,
        code: *const (),
        namespace: *mut PyNamespace,
        outer_func: PyObjectRef,
    ) {
        let _ = outer_func;
        self.code = code;
        let raw =
            unsafe { crate::w_code_get_ptr(code as pyre_object::PyObjectRef) as *const CodeObject };
        self.namespace = namespace;
        unsafe { dealloc_array_with_gc_header(self.locals_cells_stack_w) };
        self.locals_cells_stack_w = unsafe {
            alloc_array_with_gc_header(FixedObjectArray::filled(
                (&*raw).varnames.len() + ncells(&*raw) + (&*raw).max_stackdepth as usize,
                PY_NULL,
            ))
        };
        self.valuestackdepth = unsafe { (&*raw).varnames.len() + ncells(unsafe { &*raw }) };
        self.next_instr = 0;
        if self.debugdata != 0 {
            unsafe { drop(Box::from_raw(self.debugdata as *mut FrameDebugData)) };
            self.debugdata = 0;
        }
        self.escaped = false;
        self.clear_blocks();
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
        self.locals_w()
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
    pub fn initialize_frame_scopes(&mut self, _outer_func: PyObjectRef, _code: *const ()) {
        let _ = _outer_func;
        let _ = _code;
    }

    /// pyframe.py:231 setdictscope
    #[inline]
    pub fn setdictscope(&mut self, w_locals: *mut PyNamespace) {
        self.getorcreatedebug().w_locals = w_locals;
    }

    /// Create a minimal frame stub for passing to call dispatch.
    /// Used by MIFrame Box tracking when concrete_frame is unavailable.
    pub fn new_minimal(
        code: *const (),
        namespace: *mut crate::PyNamespace,
        execution_context: *const PyExecutionContext,
    ) -> Self {
        let raw =
            unsafe { crate::w_code_get_ptr(code as pyre_object::PyObjectRef) as *const CodeObject };
        let nlocals = unsafe { (&*raw).varnames.len() };
        let ncells = unsafe { (&*raw).cellvars.len() + (&*raw).freevars.len() };
        let size = nlocals + ncells + 16; // small stack
        PyFrame {
            execution_context,
            code,
            locals_cells_stack_w: unsafe {
                alloc_array_with_gc_header(FixedObjectArray::from_vec(vec![
                    pyre_object::PY_NULL;
                    size
                ]))
            },
            valuestackdepth: nlocals + ncells,
            next_instr: 0,
            namespace,
            debugdata: 0,
            lastblock: 0,
            vable_token: 0,
            escaped: false,
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
        let w_code = crate::w_code_new(code_ptr as *const ());
        let ctx_ptr = Rc::into_raw(execution_context);
        Self::new_with_namespace(w_code as *const (), ctx_ptr, namespace)
    }

    /// Create a new frame with an explicitly provided namespace pointer.
    pub fn new_with_namespace(
        code: *const (),
        execution_context: *const PyExecutionContext,
        namespace: *mut PyNamespace,
    ) -> Self {
        let raw =
            unsafe { crate::w_code_get_ptr(code as pyre_object::PyObjectRef) as *const CodeObject };
        let code_ref = unsafe { &*raw };
        let num_locals = code_ref.varnames.len();
        let num_cells = ncells(code_ref);
        let max_stack = code_ref.max_stackdepth as usize;

        PyFrame {
            execution_context,
            code,
            locals_cells_stack_w: unsafe {
                alloc_array_with_gc_header(FixedObjectArray::filled(
                    num_locals + num_cells + max_stack,
                    PY_NULL,
                ))
            },
            valuestackdepth: num_locals + num_cells,
            next_instr: 0,
            namespace,
            debugdata: 0,
            lastblock: 0,
            vable_token: 0,
            escaped: false,
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
    pub fn snapshot_for_tracing(&self) -> Box<Self> {
        let mut frame = Box::new(PyFrame {
            execution_context: self.execution_context,
            code: self.code,
            locals_cells_stack_w: unsafe {
                alloc_array_with_gc_header(FixedObjectArray::from_vec(self.locals_w().to_vec()))
            },
            valuestackdepth: self.valuestackdepth,
            next_instr: self.next_instr,
            namespace: self.namespace,
            debugdata: self.clone_debugdata(),
            lastblock: self.clone_blocks(),
            vable_token: self.vable_token,
            escaped: self.escaped,
            pending_inline_results: self.pending_inline_results.clone(),
            pending_inline_resume_pc: self.pending_inline_resume_pc,
            class_locals: self.class_locals,
        });
        // fix_array_ptrs AFTER Box allocation: inline_buf ptr must
        // point to the heap-allocated frame, not a stale stack address.
        frame.fix_array_ptrs();
        frame
    }

    /// Number of local variable slots (from code object).
    #[inline]
    pub fn nlocals(&self) -> usize {
        unsafe { (&*pyframe_get_pycode(self)).varnames.len() }
    }

    /// Number of cell + free variable slots.
    #[inline]
    pub fn ncells(&self) -> usize {
        unsafe { ncells(&*pyframe_get_pycode(self)) }
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
        let idx = self.valuestackdepth;
        self.locals_w_mut()[idx] = value;
        self.valuestackdepth += 1;
    }

    #[inline]
    pub fn pop(&mut self) -> PyObjectRef {
        assert!(self.valuestackdepth > self.stack_base());
        let depth = self.valuestackdepth - 1;
        let value = self.locals_w()[depth];
        self.locals_w_mut()[depth] = PY_NULL;
        self.valuestackdepth = depth;
        value
    }

    #[inline]
    pub fn peek(&self) -> PyObjectRef {
        self.locals_w()[self.valuestackdepth - 1]
    }

    #[inline]
    #[allow(dead_code)]
    pub fn peek_at(&self, depth: usize) -> PyObjectRef {
        self.locals_w()[self.valuestackdepth - 1 - depth]
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
            values.push(self.locals_w()[i]);
        }
        values
    }

    /// PyPy-compatible `dropvalues`.
    #[inline]
    pub fn dropvalues(&mut self, n: usize) {
        let finaldepth = self.valuestackdepth.saturating_sub(n);
        self.assert_stack_index(finaldepth);
        while self.valuestackdepth > finaldepth {
            let idx = self.valuestackdepth - 1;
            self.locals_w_mut()[idx] = PY_NULL;
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
        self.locals_w()[index]
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
        self.locals_w_mut()[index] = value;
    }

    /// PyPy-compatible `dropvaluesuntil()`.
    #[inline]
    pub fn dropvaluesuntil(&mut self, finaldepth: usize) {
        self.assert_stack_index(finaldepth);
        while self.valuestackdepth > finaldepth {
            let idx = self.valuestackdepth - 1;
            self.locals_w_mut()[idx] = PY_NULL;
            self.valuestackdepth -= 1;
        }
    }

    /// pyframe.py:186 append_block
    #[inline]
    pub fn append_block(&mut self, block: Block) {
        let node = Box::new(Block {
            handler: block.handler,
            level: block.level,
            previous: self.lastblock,
        });
        self.lastblock = Box::into_raw(node) as usize;
    }

    /// pyframe.py:190 pop_block
    #[inline]
    pub fn pop_block(&mut self) -> Option<Block> {
        if self.lastblock == 0 {
            return None;
        }
        let node = unsafe { Box::from_raw(self.lastblock as *mut Block) };
        self.lastblock = node.previous;
        Some(*node)
    }

    /// pyframe.py:195 blockstack_non_empty
    #[inline]
    pub fn blockstack_non_empty(&self) -> bool {
        self.lastblock != 0
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

    /// pyframe.py:198 get_blocklist — walk linked list, return in reverse order.
    #[inline]
    pub fn get_blocklist(&self) -> Vec<Block> {
        let mut lst = Vec::new();
        let mut ptr = self.lastblock;
        while ptr != 0 {
            let block = unsafe { &*(ptr as *const Block) };
            lst.push(*block);
            ptr = block.previous;
        }
        lst
    }

    /// pyframe.py:207 set_blocklist — rebuild linked list from slice.
    #[inline]
    pub fn set_blocklist(&mut self, lst: &[Block]) {
        self.clear_blocks();
        for block in lst.iter().rev() {
            self.append_block(*block);
        }
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

    /// pyframe.py:183 mark_as_escaped
    #[inline]
    pub fn mark_as_escaped(&mut self) {
        self.escaped = true;
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

    /// pyframe.py fset_f_trace
    #[inline]
    pub fn fset_f_trace(&mut self, w_trace: PyObjectRef) {
        self.getorcreatedebug().w_f_trace = w_trace;
    }

    /// pyframe.py fdel_f_trace
    #[inline]
    pub fn fdel_f_trace(&mut self) {
        self.getorcreatedebug().w_f_trace = pyre_object::PY_NULL;
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

    /// pyframe.py fget_f_lineno
    #[inline]
    pub fn fget_f_lineno(&self) -> usize {
        if self.get_w_f_trace().is_null() {
            self.get_last_lineno()
        } else {
            let dd = self.getdebug();
            if !dd.is_null() {
                unsafe { (*dd).f_lineno }
            } else {
                0
            }
        }
    }

    /// pyframe.py fset_f_lineno
    #[inline]
    pub fn fset_f_lineno(&mut self, new_f_lineno: usize) {
        self.getorcreatedebug().f_lineno = new_f_lineno;
        self.next_instr = new_f_lineno;
    }

    /// PyPy-compatible `setfastscope`.
    #[inline]
    pub fn setfastscope(&mut self, scope_w: &[PyObjectRef]) {
        assert!(scope_w.len() <= self.nlocals());
        for (index, value) in scope_w.iter().copied().enumerate() {
            self.locals_w_mut()[index] = value;
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
            self.locals_w_mut()[i] = *value;
        }
    }

    /// PyPy-compatible `init_cells`.
    #[inline]
    pub fn init_cells(&mut self) {
        let ncellvars = self.code().cellvars.len();
        let num_locals = self.code().varnames.len();
        let base = num_locals;
        for i in 0..ncellvars {
            if base + i >= self.locals_w().len() {
                break;
            }
            let val = self.locals_w()[i];
            self.locals_w_mut()[base + i] = val;
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
        let locals = self.locals_w().as_slice();
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
        code: *const (),
        args: &[PyObjectRef],
        globals: *mut PyNamespace,
        execution_context: *const PyExecutionContext,
    ) -> Self {
        Self::new_for_call_with_closure(code, args, globals, execution_context, PY_NULL)
    }

    /// Create a new frame for a function call with a closure.
    pub fn new_for_call_with_closure(
        code: *const (),
        args: &[PyObjectRef],
        globals: *mut PyNamespace,
        execution_context: *const PyExecutionContext,
        closure: PyObjectRef,
    ) -> Self {
        let code_ref = unsafe {
            &*(crate::w_code_get_ptr(code as pyre_object::PyObjectRef) as *const CodeObject)
        };
        let num_locals = code_ref.varnames.len();
        let num_cells = ncells(code_ref);
        let max_stack = code_ref.max_stackdepth as usize;

        let mut locals_cells_stack_w_arr =
            FixedObjectArray::filled(num_locals + num_cells + max_stack, PY_NULL);

        // Bind positional arguments directly -- no intermediate Vec.
        let nargs = args.len().min(num_locals);
        for i in 0..nargs {
            locals_cells_stack_w_arr[i] = args[i];
        }

        // Copy free variables from closure tuple into the frame's cell slots.
        // Freevars go after cellvar-only slots: indices nlocals+ncellvars_only..
        if !closure.is_null() {
            let n_cellvars_only = num_cells - code_ref.freevars.len();
            let n_freevars = code_ref.freevars.len();
            for i in 0..n_freevars {
                let cell = unsafe { w_tuple_getitem(closure, i as i64).unwrap() };
                locals_cells_stack_w_arr[num_locals + n_cellvars_only + i] = cell;
            }
        }

        PyFrame {
            execution_context,
            code,
            locals_cells_stack_w: unsafe { alloc_array_with_gc_header(locals_cells_stack_w_arr) },
            valuestackdepth: num_locals + num_cells,
            next_instr: 0,
            namespace: globals,
            debugdata: 0,
            lastblock: 0,
            vable_token: 0,
            escaped: false,
            pending_inline_results: VecDeque::new(),
            pending_inline_resume_pc: None,
            class_locals: std::ptr::null_mut(),
        }
    }

    /// Borrow the raw CodeObject.
    #[inline]
    pub fn code(&self) -> &CodeObject {
        unsafe { &*pyframe_get_pycode(self) }
    }

    /// Repoint internal array pointers after a struct move.
    ///
    /// `PyObjectArray` with small-buffer optimization stores an inline
    /// buffer whose address changes on move. Call this once after the
    /// frame is at its final stack location.
    #[inline]
    pub fn fix_array_ptrs(&mut self) {
        // locals_cells_stack_w is now a heap-allocated pointer;
        // fix_ptr on PyObjectArray is a no-op.
    }

    /// Load a constant from the code object by raw index.
    /// Used by the blackhole interpreter's bh_load_const_fn.
    pub fn load_const_pyobj(&self, idx: usize) -> PyObjectRef {
        let code = self.code();
        // RPython: constants are in JitCode.constants_r. In pyre, we resolve
        // from the CodeObject's constant table at runtime.
        let constants = code_constants(code);
        if idx >= constants.len() {
            return pyre_object::w_none();
        }
        pyobject_from_constant(&constants[idx])
    }
}

/// Load a constant from a CodeObject without a PyFrame.
/// Used by the blackhole's bh_load_const_fn when the code pointer
/// comes from a virtualizable field read.
pub fn load_const_from_code(code: &CodeObject, idx: usize) -> PyObjectRef {
    let constants = code_constants(code);
    if idx >= constants.len() {
        return pyre_object::w_none();
    }
    pyobject_from_constant(&constants[idx])
}

fn code_constants(code: &CodeObject) -> &[crate::bytecode::ConstantData] {
    unsafe {
        std::slice::from_raw_parts(
            code.constants.as_ptr() as *const crate::bytecode::ConstantData,
            code.constants.len(),
        )
    }
}

fn pyobject_from_constant(constant: &crate::bytecode::ConstantData) -> PyObjectRef {
    use crate::bytecode::ConstantData;
    use num_traits::ToPrimitive;
    match constant {
        ConstantData::Integer { value } => {
            pyre_object::intobject::w_int_new(value.to_i64().unwrap_or(0))
        }
        ConstantData::Float { value } => pyre_object::floatobject::w_float_new(*value),
        ConstantData::Boolean { value } => {
            pyre_object::intobject::w_int_new(if *value { 1 } else { 0 })
        }
        ConstantData::None => pyre_object::w_none(),
        ConstantData::Ellipsis => pyre_object::noneobject::w_ellipsis(),
        _ => pyre_object::w_none(),
    }
}

// Virtualizable configuration is in jit/frame_layout.rs

#[cfg(test)]
mod tests {
    use super::load_const_from_code;

    #[test]
    fn load_const_from_code_returns_ellipsis_singleton() {
        let code = crate::compile_eval("...");
        let code = code.expect("compile should succeed");
        let ellipsis_index = super::code_constants(&code)
            .iter()
            .position(|constant| matches!(constant, crate::bytecode::ConstantData::Ellipsis))
            .expect("compiled code should contain an Ellipsis constant");

        let loaded = load_const_from_code(&code, ellipsis_index);
        assert_eq!(loaded, pyre_object::noneobject::w_ellipsis());
    }
}
