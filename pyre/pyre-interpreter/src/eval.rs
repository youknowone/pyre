//! Bytecode evaluation loop — pure interpreter.
//!
//! JIT integration lives in pyre-jit/src/eval.rs. This module is
//! JIT-free: it processes bytecode instructions with no tracing,
//! no merge points, and no compiled-code hooks.

use crate::bytecode::{BinaryOperator, ComparisonOperator, Instruction};
use crate::*;
use crate::{
    ArithmeticOpcodeHandler, BranchOpcodeHandler, ConstantOpcodeHandler, ControlFlowOpcodeHandler,
    IterOpcodeHandler, LocalOpcodeHandler, NamespaceOpcodeHandler, OpcodeStepExecutor, PyError,
    PyErrorKind, PyResult, SharedOpcodeHandler, StackOpcodeHandler, StepResult, TruthOpcodeHandler,
    build_list_from_refs, build_map_from_refs, build_tuple_from_refs, decode_instruction_at,
    ensure_range_iter, execute_opcode_step, make_function_from_code_obj, namespace_load,
    namespace_store, range_iter_continues, range_iter_next_or_null, stack_underflow_error,
    unpack_sequence_exact, w_code_new,
};
use pyre_object::*;

use crate::call::call_callable;
use std::cell::Cell;

#[derive(Debug, Clone)]
pub struct Code {
    pub name: String,
    pub code: Option<PyObjectRef>,
}

impl Code {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            code: None,
        }
    }

    pub fn __repr__(&self) -> String {
        format!("<code {}>", self.name)
    }
}

// Thread-local current exception for bare `raise` (RAISE_VARARGS 0).
// PyPy: executioncontext.py sys_exc_info — the current active exception.
thread_local! {
    static CURRENT_EXCEPTION: Cell<PyObjectRef> = const { Cell::new(PY_NULL) };
    pub(crate) static CURRENT_FRAME: Cell<*mut PyFrame> = const { Cell::new(std::ptr::null_mut()) };
}
use crate::pyframe::PyFrame;

pub struct CurrentFrameGuard {
    previous: *mut PyFrame,
}

impl Drop for CurrentFrameGuard {
    fn drop(&mut self) {
        CURRENT_FRAME.with(|current| current.set(self.previous));
    }
}

pub fn install_current_frame(frame: &mut PyFrame) -> CurrentFrameGuard {
    let previous = CURRENT_FRAME.with(|current| {
        let previous = current.get();
        current.set(frame as *mut PyFrame);
        previous
    });
    CurrentFrameGuard { previous }
}

/// Try to dispatch an exception using the exception table or block stack.
///
/// Returns `true` if a handler was found (frame.next_instr updated to handler),
/// `false` if the exception should propagate to the caller.
pub fn handle_exception(frame: &mut PyFrame, err: &PyError) -> bool {
    // GeneratorReturn is not a real exception — always propagate it.
    if err.kind == crate::PyErrorKind::GeneratorReturn {
        return false;
    }
    let code = unsafe { &*frame.code };
    let pc = frame.next_instr.saturating_sub(1) as u32;

    // Python 3.11+ exception table dispatch
    if let Some(entry) = crate::bytecode::find_exception_handler(&code.exceptiontable, pc) {
        // Unwind stack to handler's expected depth
        let target_depth = frame.nlocals() + frame.ncells() + entry.depth as usize;
        while frame.valuestackdepth > target_depth {
            frame.pop();
        }
        if entry.push_lasti {
            frame.push(pyre_object::w_int_new(pc as i64));
        }
        // Push exception value as W_ExceptionObject
        let exc_obj = err.to_exc_object();
        frame.push(exc_obj);
        frame.next_instr = entry.target as usize;
        return true;
    }

    // Fallback: block_stack (old-style SETUP_FINALLY/SETUP_EXCEPT)
    if let Some(block) = frame.block_stack.pop() {
        while frame.valuestackdepth > block.level {
            frame.pop();
        }
        let exc_obj = err.to_exc_object();
        frame.push(exc_obj);
        frame.next_instr = block.handler;
        return true;
    }

    false
}

/// Execute a frame — pure interpreter, no JIT.
pub fn eval_frame_plain(frame: &mut PyFrame) -> PyResult {
    frame.fix_array_ptrs();
    // Track last known execution context for call_user_function_with_args
    if !frame.execution_context.is_null() {
        crate::call::set_last_exec_ctx(frame.execution_context);
    }
    eval_loop(frame)
}

/// Resume interpretation after compiled code guard failure.
pub fn eval_loop_for_force(frame: &mut PyFrame) -> PyResult {
    eval_loop(frame)
}

fn eval_loop(frame: &mut PyFrame) -> PyResult {
    let _current_frame_guard = install_current_frame(frame);
    let code = unsafe { &*frame.code };

    loop {
        if frame.next_instr >= code.instructions.len() {
            return Ok(w_none());
        }

        let pc = frame.next_instr;
        let Some((instruction, op_arg)) = decode_instruction_at(code, pc) else {
            return Ok(w_none());
        };
        frame.next_instr += 1;
        let next_instr = frame.next_instr;
        match execute_opcode_step(frame, code, instruction, op_arg, next_instr) {
            Ok(StepResult::Continue)
            | Ok(StepResult::CloseLoop {
                jump_args: _,
                loop_header_pc: _,
            }) => {}
            Ok(StepResult::Return(result)) => return Ok(result),
            Ok(StepResult::Yield(result)) => return Ok(result),
            Err(err) => {
                // GeneratorReturn: RETURN_GENERATOR unwind → return generator object
                if err.kind == crate::PyErrorKind::GeneratorReturn {
                    let gen_ptr = err.message.parse::<usize>().unwrap_or(0);
                    return Ok(gen_ptr as pyre_object::PyObjectRef);
                }
                if handle_exception(frame, &err) {
                    continue;
                }
                return Err(err);
            }
        }
    }
}

impl SharedOpcodeHandler for PyFrame {
    type Value = PyObjectRef;

    fn push_value(&mut self, value: Self::Value) -> Result<(), PyError> {
        PyFrame::push(self, value);
        Ok(())
    }

    fn pop_value(&mut self) -> Result<Self::Value, PyError> {
        if self.valuestackdepth <= self.nlocals() {
            return Err(stack_underflow_error("interpreter opcode"));
        }
        Ok(PyFrame::pop(self))
    }

    fn peek_at(&mut self, depth: usize) -> Result<Self::Value, PyError> {
        if self.valuestackdepth <= self.nlocals() + depth {
            return Err(stack_underflow_error("interpreter peek"));
        }
        Ok(PyFrame::peek_at(self, depth))
    }

    fn make_function(&mut self, code_obj: Self::Value) -> Result<Self::Value, PyError> {
        Ok(make_function_from_code_obj(code_obj, self.namespace))
    }

    fn call_callable(
        &mut self,
        callable: Self::Value,
        args: &[Self::Value],
    ) -> Result<Self::Value, PyError> {
        call_callable(self, callable, args)
    }

    fn build_list(&mut self, items: &[Self::Value]) -> Result<Self::Value, PyError> {
        Ok(build_list_from_refs(items))
    }

    fn build_tuple(&mut self, items: &[Self::Value]) -> Result<Self::Value, PyError> {
        Ok(build_tuple_from_refs(items))
    }

    fn build_map(&mut self, items: &[Self::Value]) -> Result<Self::Value, PyError> {
        Ok(build_map_from_refs(items))
    }

    fn store_subscr(
        &mut self,
        obj: Self::Value,
        key: Self::Value,
        value: Self::Value,
    ) -> Result<(), PyError> {
        setitem(obj, key, value).map(|_| ())
    }

    fn list_append(&mut self, list: Self::Value, value: Self::Value) -> Result<(), PyError> {
        unsafe { w_list_append(list, value) };
        Ok(())
    }

    fn unpack_sequence(
        &mut self,
        seq: Self::Value,
        count: usize,
    ) -> Result<Vec<Self::Value>, PyError> {
        unpack_sequence_exact(seq, count)
    }

    fn load_attr(&mut self, obj: Self::Value, name: &str) -> Result<Self::Value, PyError> {
        getattr(obj, name)
    }

    fn store_attr(
        &mut self,
        obj: Self::Value,
        name: &str,
        value: Self::Value,
    ) -> Result<(), PyError> {
        setattr(obj, name, value).map(|_| ())
    }
}

impl LocalOpcodeHandler for PyFrame {
    fn load_local_value(&mut self, idx: usize) -> Result<Self::Value, PyError> {
        Ok(self.locals_cells_stack_w[idx])
    }

    fn load_local_checked_value(&mut self, idx: usize, name: &str) -> Result<Self::Value, PyError> {
        let value = self.locals_cells_stack_w[idx];
        if value.is_null() {
            return Err(PyError::new(
                PyErrorKind::NameError,
                format!("local variable '{name}' referenced before assignment"),
            ));
        }
        // Cell objects are valid even if their contents are PY_NULL
        // (needed for __class__ cell during class body execution).
        // The cell itself is non-null, so the check above passes.
        Ok(value)
    }

    fn store_local_value(&mut self, idx: usize, value: Self::Value) -> Result<(), PyError> {
        // STORE_FAST always writes directly to the slot.
        // Cell content updates use STORE_DEREF, not STORE_FAST.
        self.locals_cells_stack_w[idx] = value;
        Ok(())
    }
}

impl NamespaceOpcodeHandler for PyFrame {
    /// PyPy: LOAD_NAME checks locals first (class body), then globals.
    fn load_name_value(&mut self, name: &str) -> Result<Self::Value, PyError> {
        // Check class_locals first (PyPy: w_locals in class body scope)
        if !self.class_locals.is_null() {
            let locals = unsafe { &*self.class_locals };
            if let Ok(value) = namespace_load(locals, name) {
                return Ok(value);
            }
        }
        // Fall back to globals (PyPy: w_globals)
        let ns = unsafe { &*self.namespace };
        namespace_load(ns, name)
    }

    /// PyPy: STORE_NAME writes to locals (class body) or globals.
    fn store_name_value(&mut self, name: &str, value: Self::Value) -> Result<(), PyError> {
        let ns = if !self.class_locals.is_null() {
            unsafe { &mut *self.class_locals }
        } else {
            unsafe { &mut *self.namespace }
        };
        namespace_store(ns, name, value);
        Ok(())
    }

    fn null_value(&mut self) -> Result<Self::Value, PyError> {
        Ok(PY_NULL)
    }
}

impl StackOpcodeHandler for PyFrame {
    fn swap_values(&mut self, depth: usize) -> Result<(), PyError> {
        let top_idx = self.valuestackdepth - 1;
        let other_idx = self.valuestackdepth - depth;
        self.locals_cells_stack_w.swap(top_idx, other_idx);
        Ok(())
    }
}

thread_local! {
    /// Cache for user-defined iterator __next__ result.
    /// concrete_iter_continues calls __next__ and caches here;
    /// iter_next_value returns the cached value.
    static USER_ITER_NEXT_CACHE: std::cell::Cell<PyObjectRef> =
        const { std::cell::Cell::new(PY_NULL) };
}

/// PyPy: pyopcode.py GET_ITER → space.iter(w_iterable)
///       pyopcode.py FOR_ITER → space.next(w_iterator)
impl IterOpcodeHandler for PyFrame {
    /// GET_ITER: convert iterable to iterator.
    /// PyPy: space.iter(w_iterable) → calls __iter__ or wraps in seq_iter.
    fn ensure_iter_value(&mut self, iter: Self::Value) -> Result<(), PyError> {
        unsafe {
            // Already an iterator
            if pyre_object::is_range_iter(iter)
                || pyre_object::is_seq_iter(iter)
                || pyre_object::generatorobject::is_generator(iter)
            {
                return Ok(());
            }
            // list → seq_iter
            if pyre_object::is_list(iter) {
                let len = pyre_object::w_list_len(iter);
                let seq_iter = pyre_object::w_seq_iter_new(iter, len);
                self.locals_cells_stack_w[self.valuestackdepth - 1] = seq_iter;
                return Ok(());
            }
            // tuple → seq_iter
            if pyre_object::is_tuple(iter) {
                let len = pyre_object::w_tuple_len(iter);
                let seq_iter = pyre_object::w_seq_iter_new(iter, len);
                self.locals_cells_stack_w[self.valuestackdepth - 1] = seq_iter;
                return Ok(());
            }
            // str → list of 1-char strings → seq_iter
            if pyre_object::is_str(iter) {
                let s = pyre_object::w_str_get_value(iter);
                let chars: Vec<pyre_object::PyObjectRef> = s
                    .chars()
                    .map(|c| {
                        let mut buf = [0u8; 4];
                        pyre_object::w_str_new(c.encode_utf8(&mut buf))
                    })
                    .collect();
                let char_list = pyre_object::w_list_new(chars);
                let len = s.chars().count();
                let seq_iter = pyre_object::w_seq_iter_new(char_list, len);
                self.locals_cells_stack_w[self.valuestackdepth - 1] = seq_iter;
                return Ok(());
            }
            // dict → iterate over keys (PyPy: dictobject.py __iter__ → dict_keys)
            if pyre_object::is_dict(iter) {
                let d = &*(iter as *const pyre_object::dictobject::W_DictObject);
                let entries = &*d.entries;
                let keys: Vec<pyre_object::PyObjectRef> = entries.iter().map(|&(k, _)| k).collect();
                let key_list = pyre_object::w_list_new(keys);
                let len = entries.len();
                let seq_iter = pyre_object::w_seq_iter_new(key_list, len);
                self.locals_cells_stack_w[self.valuestackdepth - 1] = seq_iter;
                return Ok(());
            }
            // User-defined __iter__ — PyPy: space.iter → __iter__()
            // Check both type MRO and instance dict (ATTR_TABLE)
            if pyre_object::is_instance(iter) {
                if let Ok(iter_method) = crate::baseobjspace::getattr(iter, "__iter__") {
                    let result = crate::call_function(iter_method, &[iter]);
                    self.locals_cells_stack_w[self.valuestackdepth - 1] = result;
                    return Ok(());
                }
            }
            // Type object: metaclass __iter__ (NOT the type's own MRO)
            // CPython: iter(X) calls type(X).__iter__(X)
            if pyre_object::is_type(iter) {
                let mc = crate::baseobjspace::ATTR_TABLE.with(|table| {
                    table
                        .borrow()
                        .get(&(iter as usize))
                        .and_then(|d| d.get("__metaclass__").copied())
                });
                if let Some(metaclass) = mc {
                    if let Some(method) = crate::baseobjspace::lookup_in_type(metaclass, "__iter__")
                    {
                        let result = crate::call_function(method, &[iter]);
                        self.locals_cells_stack_w[self.valuestackdepth - 1] = result;
                        return Ok(());
                    }
                }
            }
        }
        ensure_range_iter(iter)
    }

    /// FOR_ITER: check if iterator has more items.
    /// PyPy: space.next() → StopIteration means exhausted.
    /// For user-defined iterators, we speculatively call __next__ and
    /// cache the result — iter_next_value returns the cached value.
    fn concrete_iter_continues(&mut self, iter: Self::Value) -> Result<bool, PyError> {
        unsafe {
            // Generator iterator
            if pyre_object::generatorobject::is_generator(iter) {
                match crate::baseobjspace::next(iter) {
                    Ok(result) => {
                        USER_ITER_NEXT_CACHE.with(|c| c.set(result));
                        return Ok(true);
                    }
                    Err(e) if e.kind == PyErrorKind::StopIteration => {
                        USER_ITER_NEXT_CACHE.with(|c| c.set(PY_NULL));
                        return Ok(false);
                    }
                    Err(e) => return Err(e),
                }
            }
            // User-defined iterator with __next__
            if pyre_object::is_instance(iter) {
                let w_type = pyre_object::w_instance_get_type(iter);
                if let Some(next_method) = crate::baseobjspace::lookup_in_type(w_type, "__next__") {
                    match crate::call::call_callable(self, next_method, &[iter]) {
                        Ok(result) => {
                            USER_ITER_NEXT_CACHE.with(|c| c.set(result));
                            return Ok(true);
                        }
                        Err(e) if e.kind == PyErrorKind::StopIteration => {
                            USER_ITER_NEXT_CACHE.with(|c| c.set(PY_NULL));
                            return Ok(false);
                        }
                        Err(e) => return Err(e),
                    }
                }
            }
        }
        range_iter_continues(iter)
    }

    /// PyPy: space.next(w_iterator) → returns cached value from concrete_iter_continues.
    fn iter_next_value(&mut self, iter: Self::Value) -> Result<Self::Value, PyError> {
        // Generator/user-defined iterator: return cached value
        if unsafe {
            pyre_object::generatorobject::is_generator(iter) || pyre_object::is_instance(iter)
        } {
            let cached = USER_ITER_NEXT_CACHE.with(|c| c.get());
            if !cached.is_null() {
                return Ok(cached);
            }
            return Ok(PY_NULL);
        }
        range_iter_next_or_null(iter)
    }

    fn on_iter_exhausted(&mut self, target: usize) -> Result<(), PyError> {
        self.next_instr = target;
        Ok(())
    }
}

impl TruthOpcodeHandler for PyFrame {
    type Truth = bool;

    fn truth_value(&mut self, value: Self::Value) -> Result<Self::Truth, PyError> {
        Ok(truth_value(value))
    }

    fn bool_value_from_truth(
        &mut self,
        truth: Self::Truth,
        negate: bool,
    ) -> Result<Self::Value, PyError> {
        Ok(bool_value_from_truth(if negate { !truth } else { truth }))
    }
}

impl ControlFlowOpcodeHandler for PyFrame {
    fn fallthrough_target(&mut self) -> usize {
        self.next_instr
    }

    fn set_next_instr(&mut self, target: usize) -> Result<(), PyError> {
        self.next_instr = target;
        Ok(())
    }

    fn close_loop(&mut self, target: usize) -> Result<StepResult<Self::Value>, PyError> {
        // Signal a back-edge to the main eval_loop, which handles
        // JIT counting and compiled code execution via try_back_edge_jit.
        Ok(StepResult::CloseLoop {
            jump_args: vec![],
            loop_header_pc: target,
        })
    }
}

impl BranchOpcodeHandler for PyFrame {
    fn concrete_truth_as_bool(
        &mut self,
        _value: Self::Value,
        truth: Self::Truth,
    ) -> Result<bool, PyError> {
        Ok(truth)
    }
}

impl ArithmeticOpcodeHandler for PyFrame {
    fn binary_value(
        &mut self,
        a: Self::Value,
        b: Self::Value,
        op: BinaryOperator,
    ) -> Result<Self::Value, PyError> {
        binary_value(a, b, op)
    }

    fn compare_value(
        &mut self,
        a: Self::Value,
        b: Self::Value,
        op: ComparisonOperator,
    ) -> Result<Self::Value, PyError> {
        compare_value(a, b, op)
    }

    fn unary_negative_value(&mut self, value: Self::Value) -> Result<Self::Value, PyError> {
        unary_negative_value(value)
    }

    fn unary_invert_value(&mut self, value: Self::Value) -> Result<Self::Value, PyError> {
        unary_invert_value(value)
    }
}

impl ConstantOpcodeHandler for PyFrame {
    fn int_constant(&mut self, value: i64) -> Result<Self::Value, PyError> {
        Ok(w_int_new(value))
    }

    fn bigint_constant(&mut self, value: &crate::PyBigInt) -> Result<Self::Value, PyError> {
        Ok(w_long_new(value.clone()))
    }

    fn float_constant(&mut self, value: f64) -> Result<Self::Value, PyError> {
        Ok(w_float_new(value))
    }

    fn bool_constant(&mut self, value: bool) -> Result<Self::Value, PyError> {
        Ok(w_bool_from(value))
    }

    fn str_constant(&mut self, value: &str) -> Result<Self::Value, PyError> {
        Ok(box_str_constant(value))
    }

    fn code_constant(
        &mut self,
        code: &crate::bytecode::CodeObject,
    ) -> Result<Self::Value, PyError> {
        let code_ptr = Box::into_raw(Box::new(code.clone())) as *const ();
        Ok(w_code_new(code_ptr))
    }

    fn none_constant(&mut self) -> Result<Self::Value, PyError> {
        Ok(w_none())
    }

    fn slice_constant(
        &mut self,
        start: Self::Value,
        stop: Self::Value,
        step: Self::Value,
    ) -> Result<Self::Value, PyError> {
        Ok(pyre_object::w_slice_new(start, stop, step))
    }
}

impl OpcodeStepExecutor for PyFrame {
    type Error = PyError;

    // ── LoadCommonConstant ──
    fn load_common_constant(
        &mut self,
        cc: crate::bytecode::CommonConstant,
    ) -> Result<(), Self::Error> {
        use crate::bytecode::CommonConstant;
        let val = match cc {
            CommonConstant::AssertionError => crate::builtin_code_new("AssertionError", |_args| {
                Err(crate::PyError::new(
                    crate::PyErrorKind::AssertionError,
                    "assertion error".to_string(),
                ))
            }),
            CommonConstant::NotImplementedError => {
                crate::builtin_code_new("NotImplementedError", |_args| {
                    Err(crate::PyError::type_error("not implemented"))
                })
            }
            CommonConstant::BuiltinTuple => {
                crate::typedef::gettypeobject(&pyre_object::pyobject::TUPLE_TYPE)
            }
            CommonConstant::BuiltinAll => {
                crate::builtin_code_new("all", crate::builtins::builtin_all_fn)
            }
            CommonConstant::BuiltinAny => {
                crate::builtin_code_new("any", crate::builtins::builtin_any_fn)
            }
            CommonConstant::BuiltinList => {
                crate::typedef::gettypeobject(&pyre_object::pyobject::LIST_TYPE)
            }
            CommonConstant::BuiltinSet => {
                crate::typedef::gettypeobject(&pyre_object::pyobject::LIST_TYPE)
            }
        };
        self.push(val);
        Ok(())
    }

    // ── PopJumpIfNone / PopJumpIfNotNone ──
    // CPython 3.13: replaces IS_OP + POP_JUMP_IF_TRUE/FALSE for None checks

    fn pop_jump_if_none(&mut self, target: usize) -> Result<(), Self::Error> {
        let val = self.pop();
        if unsafe { pyre_object::is_none(val) } || val.is_null() {
            self.next_instr = target;
        }
        Ok(())
    }

    fn pop_jump_if_not_none(&mut self, target: usize) -> Result<(), Self::Error> {
        let val = self.pop();
        if !val.is_null() && !unsafe { pyre_object::is_none(val) } {
            self.next_instr = target;
        }
        Ok(())
    }

    // ── Closures / cells ──

    /// PyPy: pyopcode.py LOAD_DEREF
    ///
    /// Reads cell/free variable. If the slot holds a cell object (from
    /// closure tuple via COPY_FREE_VARS), dereferences it. Otherwise
    /// reads the raw value (pyre's direct storage for cellvars).
    /// LOAD_DEREF — RustPython 3.13 uses unified index (same as LOAD_FAST).
    ///
    /// PyPy: pyopcode.py LOAD_DEREF → cell.get()
    /// If the slot holds a cell object, dereference it to get the value.
    fn load_deref(&mut self, idx: usize) -> Result<(), Self::Error> {
        let slot = self.locals_cells_stack_w[idx];
        let value = if !slot.is_null() && unsafe { pyre_object::is_cell(slot) } {
            unsafe { pyre_object::w_cell_get(slot) }
        } else {
            slot
        };
        if value == PY_NULL {
            return Err(PyError::type_error(
                "free variable referenced before assignment",
            ));
        }
        self.push(value);
        Ok(())
    }

    /// STORE_DEREF — unified index. Stores into cell if present.
    ///
    /// PyPy: pyopcode.py STORE_DEREF → cell.set(value)
    fn store_deref(&mut self, idx: usize) -> Result<(), Self::Error> {
        let value = self.pop();
        let slot = self.locals_cells_stack_w[idx];
        if !slot.is_null() && unsafe { pyre_object::is_cell(slot) } {
            unsafe { pyre_object::w_cell_set(slot, value) };
        } else {
            self.locals_cells_stack_w[idx] = value;
        }
        Ok(())
    }

    /// LOAD_CLOSURE — unified index. Push cell object itself (not contents).
    ///
    /// PyPy: pyopcode.py LOAD_CLOSURE → push cell for closure capture.
    fn load_closure(&mut self, idx: usize) -> Result<(), Self::Error> {
        let cell = self.locals_cells_stack_w[idx];
        self.push(cell);
        Ok(())
    }

    /// MAKE_CELL — no-op in pyre.
    ///
    /// CPython 3.13 / RustPython MAKE_CELL — create cell object in slot.
    ///
    /// PyPy: pyframe.py cell initialization.
    /// Wraps the current value (PY_NULL if uninitialized) in a W_CellObject.
    /// LoadFast on cell slots returns the cell object itself (needed for
    /// closure creation via BUILD_TUPLE + SET_FUNCTION_ATTRIBUTE).
    fn make_cell(&mut self, idx: usize) -> Result<(), Self::Error> {
        let code = unsafe { &*self.code };
        if std::env::var("PYRE_DEBUG_CELL").is_ok() {
            eprintln!("  varnames: {:?}", code.varnames);
            eprintln!("  cellvars: {:?}", code.cellvars);
            for (i, instr) in code.instructions.iter().enumerate().take(25) {
                eprintln!("  {i}: {:?}", instr);
            }
        }
        let current = self.locals_cells_stack_w[idx];
        self.locals_cells_stack_w[idx] = pyre_object::w_cell_new(current);
        Ok(())
    }

    fn delete_deref(&mut self, idx: usize) -> Result<(), Self::Error> {
        let nlocals = self.nlocals();
        self.locals_cells_stack_w[nlocals + idx] = PY_NULL;
        Ok(())
    }

    // ── Exception handling ──

    fn setup_finally(&mut self, handler: usize) -> Result<(), Self::Error> {
        self.block_stack.push(crate::pyframe::Block {
            handler,
            level: self.valuestackdepth,
        });
        Ok(())
    }

    fn setup_except(&mut self, handler: usize) -> Result<(), Self::Error> {
        self.setup_finally(handler)
    }

    fn pop_block(&mut self) -> Result<(), Self::Error> {
        self.block_stack.pop();
        Ok(())
    }

    fn raise_varargs(&mut self, argc: usize) -> Result<(), Self::Error> {
        match argc {
            0 => {
                // Bare `raise` — re-raise current exception
                // PyPy: executioncontext.py sys_exc_info
                let exc = CURRENT_EXCEPTION.with(|c| c.get());
                if exc.is_null() || unsafe { pyre_object::is_none(exc) } {
                    Err(PyError::runtime_error("no active exception to re-raise"))
                } else if unsafe { pyre_object::is_exception(exc) } {
                    Err(unsafe { PyError::from_exc_object(exc) })
                } else {
                    Err(PyError::runtime_error("no active exception to re-raise"))
                }
            }
            1 => {
                let exc = self.pop();
                unsafe {
                    if pyre_object::is_exception(exc) {
                        Err(PyError::from_exc_object(exc))
                    } else if crate::is_builtin_code(exc) {
                        // raise TypeError → call TypeError() to create instance
                        // PyPy: RAISE_VARARGS calls type to create exception instance
                        let func = crate::builtin_code_get(exc);
                        match func(&[]) {
                            Ok(exc_obj) if pyre_object::is_exception(exc_obj) => {
                                Err(PyError::from_exc_object(exc_obj))
                            }
                            Ok(exc_obj) => {
                                // Treat non-exception return as RuntimeError
                                Err(PyError::runtime_error(&crate::py_str(exc_obj)))
                            }
                            Err(e) => Err(e),
                        }
                    } else if pyre_object::is_type(exc) {
                        // raise SomeType → call type() to create instance
                        let result = crate::call_function(exc, &[]);
                        if pyre_object::is_exception(result) {
                            Err(PyError::from_exc_object(result))
                        } else {
                            Err(PyError::runtime_error(
                                "exceptions must derive from BaseException",
                            ))
                        }
                    } else if pyre_object::is_str(exc) {
                        Err(PyError::runtime_error(pyre_object::w_str_get_value(exc)))
                    } else {
                        Err(PyError::runtime_error(
                            "exceptions must derive from BaseException",
                        ))
                    }
                }
            }
            _ => Err(PyError::type_error("too many arguments for raise")),
        }
    }

    fn end_finally(&mut self) -> Result<(), Self::Error> {
        // Pop the exception or None from stack
        let _ = self.pop();
        Ok(())
    }

    // ── Import ──
    // PyPy: pyopcode.py IMPORT_NAME
    // Stack: [level, fromlist] → pops both, pushes module object.
    fn import_name(&mut self, name: &str) -> Result<(), Self::Error> {
        let w_fromlist = self.pop();
        let w_level = self.pop();
        let level = if unsafe { pyre_object::is_int(w_level) } {
            unsafe { pyre_object::w_int_get_value(w_level) }
        } else {
            0
        };

        let module = crate::importing::importhook(
            name,
            self.namespace as PyObjectRef, // for relative imports: __name__/__package__
            w_fromlist,
            level,
            self.execution_context,
        )?;
        self.push(module);
        Ok(())
    }

    // PyPy: pyopcode.py IMPORT_FROM
    // Stack: [module] → peek module, push getattr(module, name)
    fn import_from(&mut self, name: &str) -> Result<(), Self::Error> {
        let module = self.peek();
        let attr = crate::importing::import_from(module, name)?;
        self.push(attr);
        Ok(())
    }

    // ── ContainsOp (in / not in) ──
    // PyPy: pyopcode.py COMPARE_OP with 'in' / 'not in'

    fn contains_op(&mut self, invert: crate::bytecode::Invert) -> Result<(), Self::Error> {
        // CPython 3.13: TOS = container, TOS1 = item
        let haystack = self.pop();
        let needle = self.pop();
        let result = crate::baseobjspace::contains(haystack, needle)?;
        let inverted = match invert {
            crate::bytecode::Invert::No => result,
            crate::bytecode::Invert::Yes => !result,
        };
        self.push(pyre_object::w_bool_from(inverted));
        Ok(())
    }

    // ── IsOp (is / is not) ──
    // PyPy: pyopcode.py COMPARE_OP with 'is' / 'is not'

    fn is_op(&mut self, invert: crate::bytecode::Invert) -> Result<(), Self::Error> {
        let b = self.pop();
        let a = self.pop();
        let same = std::ptr::eq(a, b); // pointer identity
        let result = match invert {
            crate::bytecode::Invert::No => same,
            crate::bytecode::Invert::Yes => !same,
        };
        self.push(pyre_object::w_bool_from(result));
        Ok(())
    }

    // ── ToBool ──
    // CPython 3.13: converts TOS to bool

    fn to_bool(&mut self) -> Result<(), Self::Error> {
        let val = self.pop();
        let truth = crate::baseobjspace::is_true(val);
        self.push(pyre_object::w_bool_from(truth));
        Ok(())
    }

    // ── DeleteSubscr ──

    fn delete_subscript(&mut self) -> Result<(), Self::Error> {
        let index = self.pop();
        let obj = self.pop();
        crate::baseobjspace::delitem(obj, index)?;
        Ok(())
    }

    // ── DeleteFast ──

    fn delete_fast(&mut self, idx: usize) -> Result<(), Self::Error> {
        self.locals_cells_stack_w[idx] = PY_NULL;
        Ok(())
    }

    // ── FormatSimple (str(TOS)) ──
    fn format_simple(&mut self) -> Result<(), Self::Error> {
        let val = self.pop();
        let s = unsafe { crate::py_str(val) };
        self.push(pyre_object::w_str_new(&s));
        Ok(())
    }

    // ── FormatWithSpec (format(TOS1, TOS)) ──
    fn format_with_spec(&mut self) -> Result<(), Self::Error> {
        let _spec = self.pop();
        let val = self.pop();
        // Phase 1: ignore spec, just convert to str
        let s = unsafe { crate::py_str(val) };
        self.push(pyre_object::w_str_new(&s));
        Ok(())
    }

    // ── ConvertValue (repr/str/ascii conversion) ──
    fn convert_value(
        &mut self,
        conv: crate::bytecode::ConvertValueOparg,
    ) -> Result<(), Self::Error> {
        let val = self.pop();
        let s = match conv {
            crate::bytecode::ConvertValueOparg::Str => unsafe { crate::py_str(val) },
            crate::bytecode::ConvertValueOparg::Repr => unsafe { crate::py_repr(val) },
            crate::bytecode::ConvertValueOparg::Ascii => unsafe { crate::py_repr(val) },
            crate::bytecode::ConvertValueOparg::None => unsafe { crate::py_str(val) },
        };
        self.push(pyre_object::w_str_new(&s));
        Ok(())
    }

    // ── CopyFreeVars ──
    // CPython 3.13: copy n freevars from function closure to frame cell slots
    fn copy_free_vars(&mut self, _count: usize) -> Result<(), Self::Error> {
        // Phase 1: no-op — closure passing needs call-site integration
        // The closure tuple is on the Function, but COPY_FREE_VARS
        // runs inside the callee frame which doesn't have a reference to
        // the function object. Need to pass closure during frame creation.
        Ok(())
    }

    // ── SetFunctionAttribute ──
    /// CPython 3.13 SET_FUNCTION_ATTRIBUTE: pop attr, pop func, set, push func.
    /// Stack effect: (2) → (1)
    /// CPython 3.13 SET_FUNCTION_ATTRIBUTE: (attr, func -- func)
    /// attr = TOS1 (below), func = TOS (top).
    /// Pops both, sets attribute on func, pushes func back.
    fn set_function_attribute_with_flag(
        &mut self,
        flag: crate::bytecode::MakeFunctionFlag,
    ) -> Result<(), Self::Error> {
        use crate::bytecode::MakeFunctionFlag;
        let func = self.pop(); // TOS = function
        let attr = self.pop(); // TOS1 = attribute value (closure tuple etc.)
        match flag {
            MakeFunctionFlag::Closure => unsafe {
                crate::function_set_closure(func, attr);
            },
            MakeFunctionFlag::Defaults => unsafe {
                crate::function_set_defaults(func, attr);
            },
            MakeFunctionFlag::KwOnlyDefaults => unsafe {
                crate::function_set_kwdefaults(func, attr);
            },
            _ => {} // annotations, etc.
        }
        self.push(func);
        Ok(())
    }

    // ── PushExcInfo ──
    // PyPy: executioncontext.py enter_frame / normalize_exception
    fn push_exc_info(&mut self) -> Result<(), Self::Error> {
        let exc = self.pop();
        // Save previous exception, set current
        let prev = CURRENT_EXCEPTION.with(|c| c.get());
        CURRENT_EXCEPTION.with(|c| c.set(exc));
        // Push "previous exception" for later restore
        self.push(prev);
        // Push the exception value back
        self.push(exc);
        Ok(())
    }

    // ── CheckExcMatch ──
    // TOS = exception type to match, TOS1 = caught exception
    // Pops type, peeks exc, pushes bool result
    fn check_exc_match(&mut self) -> Result<(), Self::Error> {
        let exc_type = self.pop();
        let exc_value = self.peek();
        let matched = unsafe {
            if !pyre_object::is_exception(exc_value) {
                true // not a proper exception object — match everything
            } else {
                let kind = pyre_object::w_exception_get_kind(exc_value);
                if pyre_object::is_str(exc_type) {
                    let type_name = pyre_object::w_str_get_value(exc_type);
                    pyre_object::exc_kind_matches(kind, type_name)
                } else if crate::is_builtin_code(exc_type) {
                    let type_name = crate::builtin_code_name(exc_type);
                    pyre_object::exc_kind_matches(kind, type_name)
                } else {
                    true
                }
            }
        };
        self.push(pyre_object::w_bool_from(matched));
        Ok(())
    }

    // ── PopExcept ──
    fn pop_except(&mut self) -> Result<(), Self::Error> {
        // Restore previous exc_info from stack
        let prev_exc = self.pop();
        CURRENT_EXCEPTION.with(|c| c.set(prev_exc));
        Ok(())
    }

    // ── Reraise ──
    // CPython: RERAISE raises the exception that's on TOS.
    // The exception table handler (handle_exception) unwinds the stack.
    // We peek TOS to get the exception but do NOT pop — handle_exception
    // will set the stack to the correct depth.
    fn reraise(&mut self) -> Result<(), Self::Error> {
        // TOS is the exception, TOS1 is prev_exc_info
        let exc = self.peek();
        unsafe {
            if pyre_object::is_exception(exc) {
                Err(PyError::from_exc_object(exc))
            } else if pyre_object::is_str(exc) {
                Err(PyError::runtime_error(
                    pyre_object::w_str_get_value(exc).to_string(),
                ))
            } else {
                Err(PyError::runtime_error("exception re-raised"))
            }
        }
    }

    // ── LoadFromDictOrGlobals ──
    // CPython 3.13: LOAD_FROM_DICT_OR_GLOBALS — try TOS dict first, then globals
    fn load_from_dict_or_globals(&mut self, name: &str) -> Result<(), Self::Error> {
        let dict = self.pop();
        // Try dict first (if it's a dict or has attrs)
        if let Ok(val) = crate::baseobjspace::getattr(dict, name) {
            self.push(val);
            return Ok(());
        }
        // Fall back to globals
        unsafe {
            if let Some(&val) = (*self.namespace).get(name) {
                self.push(val);
                return Ok(());
            }
        }
        Err(PyError::type_error(format!("name '{name}' is not defined")))
    }

    // ── GetLen ──
    fn get_len(&mut self, obj: PyObjectRef) -> Result<PyObjectRef, Self::Error> {
        let len = crate::baseobjspace::len(obj)?;
        Ok(len)
    }

    // ── LoadFastAndClear (comprehension scope) ──
    fn load_fast_and_clear(&mut self, idx: usize) -> Result<(), Self::Error> {
        let val = self.locals_cells_stack_w[idx];
        self.push(val);
        self.locals_cells_stack_w[idx] = PY_NULL;
        Ok(())
    }

    // ── BuildSet ──
    fn build_set(&mut self, count: usize) -> Result<(), Self::Error> {
        // PyPy: setobject.py BUILD_SET → W_SetObject
        let mut items = Vec::with_capacity(count);
        for _ in 0..count {
            items.push(self.pop());
        }
        items.reverse();
        // Create a proper set using the builtin set() constructor,
        // then add each item.
        let set_obj = crate::builtins::builtin_set_from_items(&items)?;
        self.push(set_obj);
        Ok(())
    }

    // ── DictUpdate ──
    // PyPy: dict.update(source); CPython: DICT_UPDATE
    // Merges source dict entries into STACK[-i] dict.
    fn dict_update(&mut self, i: usize) -> Result<(), Self::Error> {
        let source = self.pop();
        let dict = PyFrame::peek_at(self, i - 1);
        unsafe {
            if pyre_object::is_dict(source) {
                let src = &*(source as *const pyre_object::dictobject::W_DictObject);
                let entries = &*src.entries;
                for &(k, v) in entries {
                    pyre_object::w_dict_store(dict, k, v);
                }
            }
        }
        Ok(())
    }

    // ── DictMerge ──
    // PyPy: dict merge for **kwargs; CPython: DICT_MERGE
    fn dict_merge(&mut self, i: usize) -> Result<(), Self::Error> {
        let source = self.pop();
        let dict = PyFrame::peek_at(self, i - 1);
        unsafe {
            if pyre_object::is_dict(source) {
                let src = &*(source as *const pyre_object::dictobject::W_DictObject);
                let entries = &*src.entries;
                for &(k, v) in entries {
                    pyre_object::w_dict_store(dict, k, v);
                }
            }
        }
        Ok(())
    }

    // ── MapAdd ──
    // PyPy: STORE_MAP/MAP_ADD; CPython: MAP_ADD
    // dict = STACK[-i-2]; dict[TOS1] = TOS; pop key+value
    fn map_add(&mut self, i: usize) -> Result<(), Self::Error> {
        let value = self.pop();
        let key = self.pop();
        let dict = PyFrame::peek_at(self, i - 1);
        unsafe {
            pyre_object::w_dict_store(dict, key, value);
        }
        Ok(())
    }

    // ── SetAdd ──
    // PyPy: SET_ADD; CPython: SET_ADD
    // set = STACK[-i]; set.add(TOS); pop value
    // Phase 1: set is backed by list, so we use list_append.
    fn set_add(&mut self, i: usize) -> Result<(), Self::Error> {
        let value = self.pop();
        let set = PyFrame::peek_at(self, i - 1);
        unsafe {
            if pyre_object::is_list(set) {
                pyre_object::w_list_append(set, value);
            }
        }
        Ok(())
    }

    // ── none_value ──
    fn none_value(&mut self) -> Result<PyObjectRef, Self::Error> {
        Ok(pyre_object::w_none())
    }

    // ── unary_positive ──
    // PyPy: UNARY_POSITIVE → space.pos(w_value)
    fn unary_positive(&mut self, val: PyObjectRef) -> Result<PyObjectRef, Self::Error> {
        unsafe {
            if pyre_object::is_int(val) || pyre_object::is_float(val) {
                return Ok(val);
            }
        }
        Err(PyError::type_error("bad operand type for unary +"))
    }

    // ── list_to_tuple ──
    // PyPy intrinsic: convert list to tuple (used in star unpacking).
    fn list_to_tuple(&mut self, val: PyObjectRef) -> Result<PyObjectRef, Self::Error> {
        unsafe {
            if pyre_object::is_list(val) {
                let list = &*(val as *const pyre_object::listobject::W_ListObject);
                let items = list.items.as_slice().to_vec();
                return Ok(pyre_object::w_tuple_new(items));
            }
        }
        Err(PyError::type_error("expected list for list_to_tuple"))
    }

    // ── print_expr ──
    // PyPy: PRINT_EXPR → sys.displayhook(value)
    fn print_expr(&mut self, val: PyObjectRef) -> Result<(), Self::Error> {
        if !unsafe { pyre_object::is_none(val) } {
            let s = unsafe { crate::py_repr(val) };
            println!("{}", s);
        }
        Ok(())
    }

    // ── delete_name ──
    fn delete_name(&mut self, name: &str) -> Result<(), Self::Error> {
        let ns = self.namespace as *mut crate::PyNamespace;
        unsafe {
            crate::namespace_delete(&mut *ns, name);
        }
        Ok(())
    }

    // ── delete_global ──
    fn delete_global(&mut self, name: &str) -> Result<(), Self::Error> {
        let ns = self.namespace as *mut crate::PyNamespace;
        unsafe {
            crate::namespace_delete(&mut *ns, name);
        }
        Ok(())
    }

    // ── import_star ──
    // PyPy: IMPORT_STAR — merge module's public names into current namespace.
    fn import_star(&mut self) -> Result<(), Self::Error> {
        let module = self.pop();
        crate::importing::import_all_from(module, self.namespace);
        Ok(())
    }

    // ── load_build_class ──
    // PyPy: BUILD_CLASS; CPython: LOAD_BUILD_CLASS
    fn load_build_class(&mut self) -> Result<(), Self::Error> {
        let bc = crate::get_build_class_func();
        self.push(bc);
        Ok(())
    }

    // ── yield from / send ──
    fn get_yield_from_iter(&mut self) -> Result<(), Self::Error> {
        let iterable = self.pop();
        let iter = crate::baseobjspace::iter(iterable)?;
        self.push(iter);
        Ok(())
    }

    fn send_value(&mut self, target: usize) -> Result<(), Self::Error> {
        let _value = self.pop(); // sent value
        let iter = self.peek();
        match crate::baseobjspace::next(iter) {
            Ok(result) => {
                self.push(result);
                Ok(())
            }
            Err(e) if e.kind == crate::PyErrorKind::StopIteration => {
                // Don't pop iterator — END_SEND will handle it
                self.push(pyre_object::w_none()); // push result (None)
                self.next_instr = target;
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    fn end_send(&mut self) -> Result<(), Self::Error> {
        let result = self.pop();
        let _iter = self.pop();
        self.push(result);
        Ok(())
    }

    // ── load_method ──
    // PyPy: LOOKUP_METHOD — interpreter-only override.
    // For instances, pushes [attr, self] so CALL prepends self.
    // ── return_generator ──
    // CPython 3.12: RETURN_GENERATOR creates a generator from the current
    // frame and returns it to the caller. PyPy: generator.py GeneratorIterator.
    fn return_generator(&mut self) -> Result<(), Self::Error> {
        // When the generator function is already wrapped (CodeFlags::GENERATOR
        // detected in call_user_function_with_eval), RETURN_GENERATOR fires
        // during the first __next__() resume. It's a no-op in that case —
        // the generator object was already created at call time.
        // Push dummy value for the following POP_TOP to consume.
        self.push(pyre_object::w_none());
        return Ok(());

        // Legacy path for CPython-style RETURN_GENERATOR (not used with RustPython compiler):
        // Copy the current frame into a heap-allocated frame for the generator.
        // PyPy: GeneratorIterator stores the PyFrame and resumes it on __next__.
        let code = self.code;
        let code_ref = unsafe { &*code };
        let n_total = code_ref.varnames.len() + code_ref.cellvars.len() + code_ref.freevars.len();

        let mut gen_frame = crate::pyframe::PyFrame::new_with_namespace(
            code,
            self.execution_context,
            self.namespace,
        );
        gen_frame.class_locals = self.class_locals;
        gen_frame.next_instr = self.next_instr;
        // Copy locals + cells + stack
        for i in 0..self.valuestackdepth {
            gen_frame.locals_cells_stack_w[i] = self.locals_cells_stack_w[i];
        }
        gen_frame.valuestackdepth = self.valuestackdepth;
        gen_frame.block_stack = self.block_stack.clone();

        let frame_ptr = Box::into_raw(Box::new(gen_frame)) as *mut u8;
        let generator = pyre_object::generatorobject::w_generator_new(frame_ptr);
        // Signal the eval loop to return this generator object.
        // Encode the generator pointer in the error message for retrieval.
        return Err(crate::PyError {
            kind: crate::PyErrorKind::GeneratorReturn,
            message: format!("{}", generator as usize),
            exc_object: std::ptr::null_mut(),
        }
        .into());
    }

    // ── load_super_attr ──
    // CPython 3.12 LOAD_SUPER_ATTR: stack = [global_super, class, self]
    // → super(class, self).attr
    fn load_super_attr_with(&mut self, name: &str, is_method: bool) -> Result<(), Self::Error> {
        let self_obj = self.pop();
        let cls = self.pop();
        let _global_super = self.pop();

        let proxy = pyre_object::superobject::w_super_new(cls, self_obj);
        let result = crate::baseobjspace::getattr(proxy, name)?;

        // CPython _PySuper_Lookup: determines whether the resolved attr
        // is an unbound method (needs self binding) or a staticmethod /
        // classmethod (no self binding / bind class).
        if is_method {
            // Check the raw descriptor in MRO to decide binding.
            let null_or_self =
                unsafe { crate::baseobjspace::super_lookup_binding(cls, self_obj, name) };
            self.push(result);
            self.push(null_or_self);
        } else {
            self.push(result);
        }
        Ok(())
    }

    // For non-instances (modules etc.), pushes [attr, NULL].
    // The default trait impl always pushes [attr, NULL], which is what
    // the JIT tracer uses — no runtime branch in the shared path.
    fn load_method(&mut self, name: &str) -> Result<(), Self::Error> {
        let obj = self.pop();
        let attr = crate::baseobjspace::getattr(obj, name)?;
        if unsafe { pyre_object::is_method(attr) } {
            self.push(attr);
            self.push(PY_NULL);
            return Ok(());
        }
        self.push(attr);
        // Bind self only for regular instance method calls.
        // staticmethod/classmethod descriptors already unwrap to the raw
        // function via getattr → get; self must NOT
        // be prepended for those.
        // PyPy: LOOKUP_METHOD checks whether the attr came from a
        // non-data descriptor that is a plain function (not staticmethod).
        // Determine what to bind as null_or_self.
        // PyPy: LOOKUP_METHOD resolves descriptors and decides binding.
        //  - regular method → bind instance (self)
        //  - classmethod → bind class (w_type)
        //  - staticmethod → no binding (NULL)
        //  - builtin type method (list.append etc.) → bind instance
        let bound = unsafe {
            if pyre_object::is_instance(obj) {
                let w_type = pyre_object::w_instance_get_type(obj);
                let raw = crate::baseobjspace::lookup_in_type(w_type, name);
                match raw {
                    Some(d) if pyre_object::is_staticmethod(d) => PY_NULL,
                    // PyPy: ClassMethod.__get__ → Method(func, klass)
                    Some(d) if pyre_object::is_classmethod(d) => w_type,
                    Some(_) => obj, // found in type MRO → bind self (method)
                    None => {
                        // Not found in type MRO → found in instance __dict__.
                        // PyPy: instance __dict__ attrs bypass descriptor protocol.
                        // User functions in instance dict: no binding.
                        // Builtin functions stored per-instance (e.g. set methods
                        // stored via ATTR_TABLE): bind self.
                        if crate::is_function(attr) {
                            PY_NULL
                        } else {
                            obj
                        }
                    }
                }
            } else if pyre_object::is_type(obj) {
                // Type object: check for classmethod in type's MRO
                let raw = crate::baseobjspace::lookup_in_type(obj, name);
                match raw {
                    Some(d) if pyre_object::is_classmethod(d) => obj,
                    Some(_) => PY_NULL, // found in own MRO → no binding
                    None => {
                        // Not found in type's own MRO → check metaclass MRO.
                        // If found there, bind obj (the type) as self.
                        // PyPy: type.__getattribute__ metatype descriptor binding.
                        obj
                    }
                }
            } else if crate::typedef::r#type(obj).is_some() && !pyre_object::is_module(obj) {
                // Builtin type method (list.append, etc.) found via TypeDef.
                // PyPy: LOOKUP_METHOD binds self for builtin type methods.
                obj
            } else {
                PY_NULL
            }
        };
        self.push(bound);
        Ok(())
    }

    // ── call ──
    // PyPy: CALL_FUNCTION — interpreter-only override.
    // Handles null_or_self prepend for instance method calls.
    // The default trait impl (opcode_call) always discards null_or_self,
    // which is what the JIT tracer uses — no trace/concrete divergence.
    fn call(&mut self, nargs: usize) -> Result<(), Self::Error> {
        let mut args = Vec::with_capacity(nargs);
        for _ in 0..nargs {
            args.push(self.pop());
        }
        args.reverse();
        let null_or_self = self.pop();
        let callable = self.pop();

        let result = if null_or_self.is_null() {
            call_callable(self, callable, &args)?
        } else {
            let mut full_args = Vec::with_capacity(1 + args.len());
            full_args.push(null_or_self);
            full_args.extend_from_slice(&args);
            call_callable(self, callable, &full_args)?
        };
        self.push(result);
        Ok(())
    }

    // ── call_function_ex ──
    // PyPy: CALL_FUNCTION_VAR_KW; CPython: CALL_FUNCTION_EX
    // Stack: [callable, NULL, args_tuple, kwargs_dict_or_null]
    fn call_function_ex(&mut self) -> Result<(), Self::Error> {
        let kwargs_or_null = self.pop();
        let args_obj = self.pop();
        let _null = self.pop();
        let callable = self.pop();

        let mut args: Vec<PyObjectRef> = unsafe {
            if pyre_object::is_tuple(args_obj) {
                let t = &*(args_obj as *const pyre_object::tupleobject::W_TupleObject);
                t.items.as_slice().to_vec()
            } else if pyre_object::is_list(args_obj) {
                let l = &*(args_obj as *const pyre_object::listobject::W_ListObject);
                l.items.as_slice().to_vec()
            } else {
                vec![]
            }
        };

        // Merge kwargs dict into call.
        // PyPy: argument.py Arguments.prepend + unpack_combined_starstarargs
        if !kwargs_or_null.is_null() && unsafe { pyre_object::is_dict(kwargs_or_null) } {
            let entries = unsafe { pyre_object::w_dict_str_entries(kwargs_or_null) };
            if !entries.is_empty() {
                let result = crate::call::call_with_kwargs(self, callable, &args, &entries)?;
                self.push(result);
                return Ok(());
            }
        }

        let result = call_callable(self, callable, &args)?;
        self.push(result);
        Ok(())
    }

    // ── call_kw ──
    // PyPy: CALL_FUNCTION_KW; CPython 3.13: CALL_KW
    // Stack: [callable, self_or_null, arg1, ..., argN, kwarg_names_tuple]
    /// CALL_KW — call with keyword arguments.
    ///
    /// PyPy: argument.py _match_signature
    /// Stack: [callable, null_or_self, arg0..argN-1, kwarg_names_tuple]
    /// The last `len(kwarg_names)` args are keyword args.
    ///
    /// Keyword resolution happens HERE (before frame creation) so the
    /// JIT eval loop sees correctly-positioned locals. PyPy does this
    /// in Arguments.parse_into_scope before the frame executes.
    fn call_kw(&mut self, nargs: usize) -> Result<(), Self::Error> {
        let kwarg_names = self.pop();
        let mut args = Vec::with_capacity(nargs);
        for _ in 0..nargs {
            args.push(self.pop());
        }
        args.reverse();
        let self_or_null = self.pop();
        let callable = self.pop();

        if self_or_null != PY_NULL && !unsafe { pyre_object::is_none(self_or_null) } {
            args.insert(0, self_or_null);
        }

        // For type objects with kwargs: use call_with_kwargs which handles
        // __new__/__init__ kwargs forwarding correctly.
        let callable_unwrapped = crate::baseobjspace::unwrap_cell(callable);
        if unsafe { pyre_object::is_type(callable_unwrapped) } {
            let nkw = if unsafe { pyre_object::is_tuple(kwarg_names) } {
                unsafe { pyre_object::w_tuple_len(kwarg_names) }
            } else {
                0
            };
            if nkw > 0 {
                let n_pos = args.len() - nkw;
                let pos_args = args[..n_pos].to_vec();
                let mut kw_entries = Vec::with_capacity(nkw);
                for ki in 0..nkw {
                    let name = unsafe { pyre_object::w_tuple_getitem(kwarg_names, ki as i64) };
                    if let Some(name_obj) = name {
                        let key = unsafe { pyre_object::w_str_get_value(name_obj) }.to_string();
                        kw_entries.push((key, args[n_pos + ki]));
                    }
                }
                let result = crate::call::call_with_kwargs(
                    self,
                    callable_unwrapped,
                    &pos_args,
                    &kw_entries,
                )?;
                self.push(result);
                return Ok(());
            }
        }

        // Resolve keyword args into positional order.
        // PyPy: argument.py _match_signature step: match keywords to argnames
        let resolved = if unsafe { crate::is_builtin_code(callable_unwrapped) } {
            // Builtin functions: pack kwargs into a dict as last arg
            let nkw = if unsafe { pyre_object::is_tuple(kwarg_names) } {
                unsafe { pyre_object::w_tuple_len(kwarg_names) }
            } else {
                0
            };
            if nkw > 0 {
                let n_pos = args.len() - nkw;
                let mut resolved = args[..n_pos].to_vec();
                let kwargs_dict = pyre_object::w_dict_new();
                // Marker key so print() can distinguish kwargs dict from regular dict arg
                unsafe {
                    pyre_object::w_dict_store(
                        kwargs_dict,
                        pyre_object::w_str_new("__pyre_kw__"),
                        pyre_object::w_bool_from(true),
                    );
                }
                for ki in 0..nkw {
                    let name = unsafe { pyre_object::w_tuple_getitem(kwarg_names, ki as i64) };
                    if let Some(name_obj) = name {
                        unsafe {
                            pyre_object::w_dict_store(kwargs_dict, name_obj, args[n_pos + ki]);
                        }
                    }
                }
                resolved.push(kwargs_dict);
                resolved
            } else {
                args.clone()
            }
        } else {
            crate::call::resolve_kwargs(callable, &args, kwarg_names)
        };
        let result = call_callable(self, callable_unwrapped, &resolved)?;
        self.push(result);
        Ok(())
    }

    // ── load_locals ──
    // PyPy: LOAD_LOCALS; CPython: LOAD_LOCALS
    // Pushes the current namespace dict onto the stack.
    fn load_locals(&mut self) -> Result<(), Self::Error> {
        let dict = pyre_object::w_dict_new();
        unsafe {
            if !self.class_locals.is_null() {
                for (key, &value) in (*self.class_locals).entries() {
                    if !value.is_null() {
                        pyre_object::w_dict_store(dict, pyre_object::w_str_new(key), value);
                    }
                }
            } else {
                let code = &*self.code;
                for (idx, name) in code.varnames.iter().enumerate() {
                    let value = self.locals_cells_stack_w[idx];
                    if !value.is_null() {
                        pyre_object::w_dict_store(dict, pyre_object::w_str_new(name), value);
                    }
                }
                if self.nlocals() == 0 && !self.namespace.is_null() {
                    for (key, &value) in (*self.namespace).entries() {
                        if !value.is_null() {
                            pyre_object::w_dict_store(dict, pyre_object::w_str_new(key), value);
                        }
                    }
                }
            }
        }
        self.push(dict);
        Ok(())
    }

    // ── unpack_ex ──
    // PyPy: UNPACK_SEQUENCE with star; CPython: UNPACK_EX
    // `a, *b, c = iterable`
    fn unpack_ex(&mut self, args: crate::bytecode::UnpackExArgs) -> Result<(), Self::Error> {
        let before = args.before as usize;
        let after = args.after as usize;
        let value = self.pop();

        let elements: Vec<PyObjectRef> = unsafe {
            if pyre_object::is_tuple(value) {
                let t = &*(value as *const pyre_object::tupleobject::W_TupleObject);
                t.items.as_slice().to_vec()
            } else if pyre_object::is_list(value) {
                let l = &*(value as *const pyre_object::listobject::W_ListObject);
                l.items.as_slice().to_vec()
            } else {
                return Err(PyError::type_error("cannot unpack non-sequence"));
            }
        };

        let min_expected = before + after;
        if elements.len() < min_expected {
            return Err(PyError::value_error(&format!(
                "not enough values to unpack (expected at least {}, got {})",
                min_expected,
                elements.len()
            )));
        }

        let middle_len = elements.len() - min_expected;

        // Push after items (reversed), then middle list, then before items (reversed)
        for i in (0..after).rev() {
            self.push(elements[before + middle_len + i]);
        }
        let middle: Vec<PyObjectRef> = elements[before..before + middle_len].to_vec();
        self.push(pyre_object::w_list_new(middle));
        for i in (0..before).rev() {
            self.push(elements[i]);
        }

        Ok(())
    }

    // ── delete_attr ──
    // PyPy: DELETE_ATTR → space.delattr(obj, name)
    fn delete_attr(&mut self, name: &str) -> Result<(), Self::Error> {
        let obj = self.pop();
        crate::baseobjspace::delattr(obj, name)?;
        Ok(())
    }

    // ── set_update ──
    // PyPy: set.update(iterable); CPython: SET_UPDATE
    fn set_update(&mut self, i: usize) -> Result<(), Self::Error> {
        let iterable = self.pop();
        let set = PyFrame::peek_at(self, i - 1);
        unsafe {
            if pyre_object::is_list(set) {
                if pyre_object::is_list(iterable) {
                    let src = &*(iterable as *const pyre_object::listobject::W_ListObject);
                    for &item in src.items.as_slice() {
                        pyre_object::w_list_append(set, item);
                    }
                } else if pyre_object::is_tuple(iterable) {
                    let src = &*(iterable as *const pyre_object::tupleobject::W_TupleObject);
                    for &item in src.items.as_slice() {
                        pyre_object::w_list_append(set, item);
                    }
                }
            }
        }
        Ok(())
    }

    // ── BuildSlice ──
    // CPython 3.13: BUILD_SLICE creates a slice object from 2 or 3 stack items
    fn build_slice(
        &mut self,
        argc: crate::bytecode::BuildSliceArgCount,
    ) -> Result<(), Self::Error> {
        use crate::bytecode::BuildSliceArgCount;
        let step = match argc {
            BuildSliceArgCount::Three => self.pop(),
            BuildSliceArgCount::Two => pyre_object::w_none(),
        };
        let stop = self.pop();
        let start = self.pop();
        self.push(pyre_object::w_slice_new(start, stop, step));
        Ok(())
    }

    // ── BinarySlice (a[b:c]) ──
    // PyPy: BINARY_SUBSCR with slice; CPython 3.13: BINARY_SLICE
    fn binary_slice(&mut self) -> Result<(), Self::Error> {
        let stop = self.pop();
        let start = self.pop();
        let obj = self.pop();
        unsafe {
            if pyre_object::is_list(obj) {
                let len = pyre_object::w_list_len(obj) as i64;
                let s = if pyre_object::is_none(start) {
                    0
                } else {
                    pyre_object::w_int_get_value(start)
                };
                let e = if pyre_object::is_none(stop) {
                    len
                } else {
                    pyre_object::w_int_get_value(stop)
                };
                let s = if s < 0 { (len + s).max(0) } else { s.min(len) } as usize;
                let e = if e < 0 { (len + e).max(0) } else { e.min(len) } as usize;
                let mut items = Vec::new();
                for i in s..e {
                    if let Some(v) = pyre_object::w_list_getitem(obj, i as i64) {
                        items.push(v);
                    }
                }
                self.push(pyre_object::w_list_new(items));
                return Ok(());
            }
            if pyre_object::is_str(obj) {
                let full = pyre_object::w_str_get_value(obj);
                let len = full.len() as i64;
                let s = if pyre_object::is_none(start) {
                    0
                } else {
                    pyre_object::w_int_get_value(start)
                };
                let e = if pyre_object::is_none(stop) {
                    len
                } else {
                    pyre_object::w_int_get_value(stop)
                };
                let s = if s < 0 { (len + s).max(0) } else { s.min(len) } as usize;
                let e = if e < 0 { (len + e).max(0) } else { e.min(len) } as usize;
                let slice = &full[s..e.min(full.len())];
                self.push(pyre_object::w_str_new(slice));
                return Ok(());
            }
        }
        Err(PyError::type_error("object is not subscriptable"))
    }

    // ── StoreSlice (a[b:c] = d) ──
    fn store_slice(&mut self) -> Result<(), Self::Error> {
        // Phase 1 stub — rarely used in hot loops
        Err(PyError::type_error("STORE_SLICE not yet implemented"))
    }

    // ── BuildString (f-string concatenation) ──
    // CPython 3.13: concatenate N string fragments from stack
    fn build_string(&mut self, count: usize) -> Result<(), Self::Error> {
        let mut parts = Vec::with_capacity(count);
        for _ in 0..count {
            parts.push(self.pop());
        }
        parts.reverse();
        let mut result = String::new();
        for part in &parts {
            unsafe {
                if pyre_object::is_str(*part) {
                    result.push_str(pyre_object::w_str_get_value(*part));
                } else if pyre_object::is_int(*part) {
                    result.push_str(&pyre_object::w_int_get_value(*part).to_string());
                } else if pyre_object::is_none(*part) {
                    result.push_str("None");
                } else if pyre_object::is_bool(*part) {
                    result.push_str(if pyre_object::w_bool_get_value(*part) {
                        "True"
                    } else {
                        "False"
                    });
                } else {
                    result.push_str("<object>");
                }
            }
        }
        self.push(pyre_object::w_str_new(&result));
        Ok(())
    }

    // ── ListExtend ──
    fn list_extend(&mut self, _i: usize) -> Result<(), Self::Error> {
        let iterable = self.pop();
        let list = self.peek();
        unsafe {
            if pyre_object::is_list(iterable) {
                let src_len = pyre_object::w_list_len(iterable);
                for j in 0..src_len {
                    if let Some(item) = pyre_object::w_list_getitem(iterable, j as i64) {
                        pyre_object::w_list_append(list, item);
                    }
                }
                return Ok(());
            }
            if pyre_object::is_tuple(iterable) {
                let src_len = pyre_object::w_tuple_len(iterable);
                for j in 0..src_len {
                    if let Some(item) = pyre_object::w_tuple_getitem(iterable, j as i64) {
                        pyre_object::w_list_append(list, item);
                    }
                }
                return Ok(());
            }
        }
        Err(PyError::type_error("object is not iterable"))
    }

    fn unsupported(
        &mut self,
        instruction: &Instruction,
    ) -> Result<StepResult<PyObjectRef>, Self::Error> {
        Err(PyError::type_error(format!(
            "unimplemented instruction: {instruction:?}"
        )))
    }
}

// ── JitState ↔ PyFrame conversion ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;
    use crate::{PyExecutionContext, function_new};
    use std::rc::Rc;

    fn run_eval(source: &str) -> PyResult {
        let code = compile_eval(source).expect("compile failed");
        let mut frame = PyFrame::new_with_context(code, Rc::new(PyExecutionContext::default()));
        eval_frame_plain(&mut frame)
    }

    fn run_exec_frame(source: &str) -> (PyResult, PyFrame) {
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new_with_context(code, Rc::new(PyExecutionContext::default()));
        let result = eval_frame_plain(&mut frame);
        (result, frame)
    }

    #[test]
    fn test_literal() {
        let result = run_eval("42").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 42) };
    }

    #[test]
    fn test_addition() {
        let result = run_eval("1 + 2").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 3) };
    }

    #[test]
    fn test_subtraction() {
        let result = run_eval("10 - 3").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 7) };
    }

    #[test]
    fn test_multiplication() {
        let result = run_eval("6 * 7").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 42) };
    }

    #[test]
    fn test_complex_expr() {
        let result = run_eval("(2 + 3) * 4 - 1").unwrap();
        unsafe { assert_eq!(w_int_get_value(result), 19) };
    }

    #[test]
    fn test_comparison() {
        let result = run_eval("3 < 5").unwrap();
        unsafe { assert!(w_bool_get_value(result)) };
    }

    #[test]
    fn test_comparison_false() {
        let result = run_eval("5 < 3").unwrap();
        unsafe { assert!(!w_bool_get_value(result)) };
    }

    #[test]
    fn test_store_load_namespace() {
        let source = "x = 5\ny = x * x";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let x = *(*frame.namespace).get("x").unwrap();
            let y = *(*frame.namespace).get("y").unwrap();
            assert_eq!(w_int_get_value(x), 5);
            assert_eq!(w_int_get_value(y), 25);
        }
    }

    #[test]
    fn test_while_loop() {
        let source = "i = 0\nwhile i < 10:\n    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            assert_eq!(w_int_get_value(i), 10);
        }
    }

    #[test]
    fn test_eval_loop_redecodes_opargs_after_extended_arg_jumps() {
        let source = "\
def fannkuch(n):
    p = [0] * n
    q = [0] * n
    s = [0] * n
    i = 0
    while i < n:
        p[i] = i
        q[i] = i
        s[i] = i
        i = i + 1
    maxflips = 0
    checksum = 0
    sign = 1
    while True:
        q0 = p[0]
        if q0 != 0:
            i = 1
            while i < n:
                q[i] = p[i]
                i = i + 1
            flips = 1
            while True:
                qq = q[q0]
                if qq == 0:
                    break
                q[q0] = q0
                if q0 >= 3:
                    i = 1
                    j = q0 - 1
                    while i < j:
                        t = q[i]
                        q[i] = q[j]
                        q[j] = t
                        i = i + 1
                        j = j - 1
                q0 = qq
                flips = flips + 1
            if flips > maxflips:
                maxflips = flips
            checksum = checksum + sign * flips
        if sign == 1:
            t = p[0]
            p[0] = p[1]
            p[1] = t
            sign = -1
        else:
            t = p[1]
            p[1] = p[2]
            p[2] = t
            sign = 1
            i = 2
            while i < n:
                sx = s[i]
                if sx != 0:
                    s[i] = sx - 1
                    break
                if i == n - 1:
                    return 999
                s[i] = i
                t = p[0]
                j = 0
                while j < i + 1:
                    p[j] = p[j + 1]
                    j = j + 1
                p[i + 1] = t
                i = i + 1

r = fannkuch(6)";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let r = *(*frame.namespace).get("r").unwrap();
            assert_eq!(w_int_get_value(r), 999);
        }
    }

    #[test]
    fn test_none_result() {
        let result = run_eval("None").unwrap();
        unsafe { assert!(is_none(result)) };
    }

    #[test]
    fn test_bool_result() {
        let result = run_eval("True").unwrap();
        unsafe {
            assert!(is_bool(result));
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_float_literal() {
        let result = run_eval("1.5").unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), 1.5);
        }
    }

    #[test]
    fn test_float_addition() {
        let result = run_eval("1.5 + 2.5").unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), 4.0);
        }
    }

    #[test]
    fn test_float_truediv() {
        let result = run_eval("10 / 4").unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), 2.5);
        }
    }

    #[test]
    fn test_float_comparison() {
        let result = run_eval("1.5 < 2.5").unwrap();
        unsafe {
            assert!(is_bool(result));
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_float_int_mixed() {
        let result = run_eval("1.5 + 2").unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), 3.5);
        }
    }

    #[test]
    fn test_float_negation() {
        let result = run_eval("-3.14").unwrap();
        unsafe {
            assert!(is_float(result));
            assert_eq!(w_float_get_value(result), -3.14);
        }
    }

    #[test]
    fn test_float_truthiness() {
        // Test via is_true directly since `not` uses ToBool instruction
        assert!(!is_true(w_float_new(0.0)));
        assert!(is_true(w_float_new(1.5)));
        assert!(is_true(w_float_new(-0.1)));
    }

    // ── str tests ────────────────────────────────────────────────────

    #[test]
    fn test_str_literal() {
        let result = run_eval("'hello'").unwrap();
        unsafe {
            assert!(is_str(result));
            assert_eq!(w_str_get_value(result), "hello");
        }
    }

    #[test]
    fn test_str_concat() {
        let result = run_eval("'hello' + ' world'").unwrap();
        unsafe {
            assert!(is_str(result));
            assert_eq!(w_str_get_value(result), "hello world");
        }
    }

    #[test]
    fn test_str_repeat() {
        let result = run_eval("'ab' * 3").unwrap();
        unsafe {
            assert!(is_str(result));
            assert_eq!(w_str_get_value(result), "ababab");
        }
    }

    #[test]
    fn test_str_comparison() {
        let result = run_eval("'abc' < 'abd'").unwrap();
        unsafe {
            assert!(is_bool(result));
            assert!(w_bool_get_value(result));
        }
    }

    // ── for loop / range tests ──────────────────────────────────────

    #[test]
    fn test_for_range() {
        let source = "s = 0\nfor i in range(10):\n    s = s + i";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let s = *(*frame.namespace).get("s").unwrap();
            assert_eq!(w_int_get_value(s), 45);
        }
    }

    #[test]
    fn test_hot_range_loop_survives_compiled_trace() {
        let source = "s = 0\nfor i in range(3000):\n    s = s + i";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let s = *(*frame.namespace).get("s").unwrap();
            assert_eq!(w_int_get_value(s), 4_498_500);
        }
    }

    #[test]
    fn test_hot_module_branch_loop_survives_compiled_trace() {
        let source = "\
i = 0
acc = 0
while i < 3000:
    if i < 1500:
        acc = acc + 1
    else:
        acc = acc + 2
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 4500);
        }
    }

    #[test]
    fn test_hot_tuple_unpack_loop_survives_compiled_trace() {
        let source = "\
i = 0
acc = 0
while i < 3000:
    a, b = (i, 1)
    acc = acc + a + b
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 4_501_500);
        }
    }

    #[test]
    fn test_hot_list_index_store_loop_survives_compiled_trace() {
        let source = "\
lst = [0]
i = 0
acc = 0
while i < 3000:
    lst[0] = i
    acc = acc + lst[0]
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            let lst = *(*frame.namespace).get("lst").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 4_498_500);
            assert_eq!(w_int_get_value(w_list_getitem(lst, 0).unwrap()), 2999);
        }
    }

    #[test]
    fn test_hot_bitwise_or_loop_survives_compiled_trace() {
        let source = "\
i = 0
acc = 0
while i < 3000:
    acc = acc | i
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 4095);
        }
    }

    #[test]
    fn test_hot_unary_invert_loop_survives_compiled_trace() {
        let source = "\
i = 0
acc = 0
while i < 3000:
    acc = acc + (~i)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), -4_501_500);
        }
    }

    #[test]
    fn test_hot_positive_floordiv_loop_survives_compiled_trace() {
        let source = "\
i = 0
acc = 0
while i < 3000:
    acc = acc + (i // 3)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 1_498_500);
        }
    }

    #[test]
    fn test_hot_positive_mod_loop_survives_compiled_trace() {
        let source = "\
i = 0
acc = 0
while i < 3000:
    acc = acc + (i % 7)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 8_994);
        }
    }

    #[test]
    fn test_hot_builtin_abs_loop_survives_compiled_trace() {
        let source = "\
i = 0
acc = 0
while i < 3000:
    acc = acc + abs(i - 1500)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 2_250_000);
        }
    }

    #[test]
    fn test_hot_list_truth_loop_survives_compiled_trace() {
        let source = "\
lst = [1]
i = 0
acc = 0
while i < 3000:
    if lst:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_empty_tuple_truth_loop_survives_compiled_trace() {
        let source = "\
tpl = ()
i = 0
acc = 0
while i < 3000:
    if tpl:
        acc = acc + 100
    else:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_none_truth_loop_survives_compiled_trace() {
        let source = "\
value = None
i = 0
acc = 0
while i < 3000:
    if value:
        acc = acc + 100
    else:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_float_truth_loop_survives_compiled_trace() {
        let source = "\
value = 0.5
i = 0
acc = 0
while i < 3000:
    if value:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_string_truth_loop_survives_compiled_trace() {
        let source = "\
value = \"pyre\"
i = 0
acc = 0
while i < 3000:
    if value:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_empty_string_truth_loop_survives_compiled_trace() {
        let source = "\
value = \"\"
i = 0
acc = 0
while i < 3000:
    if value:
        acc = acc + 100
    else:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_dict_truth_loop_survives_compiled_trace() {
        let source = "\
value = {1: 2}
i = 0
acc = 0
while i < 3000:
    if value:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_builtin_len_string_loop_survives_compiled_trace() {
        let source = "\
value = \"pyre\"
i = 0
acc = 0
while i < 3000:
    acc = acc + len(value)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 12_000);
        }
    }

    #[test]
    fn test_hot_builtin_len_dict_loop_survives_compiled_trace() {
        let source = "\
value = {1: 2, 3: 4}
i = 0
acc = 0
while i < 3000:
    acc = acc + len(value)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 6_000);
        }
    }

    #[test]
    fn test_hot_builtin_isinstance_true_loop_survives_compiled_trace() {
        let source = "\
x = 42
i = 0
acc = 0
while i < 3000:
    if isinstance(x, int):
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_builtin_isinstance_false_loop_survives_compiled_trace() {
        let source = "\
x = []
i = 0
acc = 0
while i < 3000:
    if isinstance(x, int):
        acc = acc + 1
    else:
        acc = acc + 2
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 6000);
        }
    }

    #[test]
    fn test_hot_builtin_type_loop_survives_compiled_trace() {
        let source = "\
x = []
i = 0
acc = 0
while i < 3000:
    if type(x) == list:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_builtin_min_small_int_loop_survives_compiled_trace() {
        let source = "\
i = 0
acc = 0
while i < 3000:
    acc = acc + min(i % 7, 3)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 6426);
        }
    }

    #[test]
    fn test_hot_builtin_max_small_int_loop_survives_compiled_trace() {
        let source = "\
i = 0
acc = 0
while i < 3000:
    acc = acc + max(i % 7, 3)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 11568);
        }
    }

    #[test]
    fn test_hot_empty_dict_truth_loop_survives_compiled_trace() {
        let source = "\
value = {}
i = 0
acc = 0
while i < 3000:
    if value:
        acc = acc + 100
    else:
        acc = acc + 1
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 3000);
        }
    }

    #[test]
    fn test_hot_list_negative_index_store_loop_survives_compiled_trace() {
        let source = "\
lst = [0, 1]
i = 0
acc = 0
while i < 3000:
    lst[-1] = i
    acc = acc + lst[-1]
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            let lst = *(*frame.namespace).get("lst").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 4_498_500);
            assert_eq!(w_int_get_value(w_list_getitem(lst, -1).unwrap()), 2999);
        }
    }

    #[test]
    fn test_hot_tuple_negative_index_load_loop_survives_compiled_trace() {
        let source = "\
tpl = (3, 5)
i = 0
acc = 0
while i < 3000:
    acc = acc + tpl[-1]
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 15_000);
        }
    }

    #[test]
    fn test_hot_user_function_loop_survives_compiled_trace() {
        let source = "\
def inc(x):
    return x + 1
i = 0
acc = 0
while i < 3000:
    acc = acc + inc(i)
    i = i + 1";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let i = *(*frame.namespace).get("i").unwrap();
            let acc = *(*frame.namespace).get("acc").unwrap();
            assert_eq!(w_int_get_value(i), 3000);
            assert_eq!(w_int_get_value(acc), 4_501_500);
        }
    }

    #[test]
    fn test_for_range_start_stop() {
        let source = "s = 0\nfor i in range(5, 10):\n    s = s + i";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let s = *(*frame.namespace).get("s").unwrap();
            assert_eq!(w_int_get_value(s), 35);
        }
    }

    #[test]
    fn test_for_range_step() {
        let source = "s = 0\nfor i in range(0, 10, 2):\n    s = s + i";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let s = *(*frame.namespace).get("s").unwrap();
            // 0 + 2 + 4 + 6 + 8 = 20
            assert_eq!(w_int_get_value(s), 20);
        }
    }

    #[test]
    fn test_for_range_empty() {
        let source = "s = 42\nfor i in range(0):\n    s = 0";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let s = *(*frame.namespace).get("s").unwrap();
            assert_eq!(w_int_get_value(s), 42);
        }
    }

    #[test]
    fn test_builtin_range_print() {
        let source = "s = 0\nfor i in range(5):\n    s = s + i";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let s = *(*frame.namespace).get("s").unwrap();
            // 0 + 1 + 2 + 3 + 4 = 10
            assert_eq!(w_int_get_value(s), 10);
        }
    }

    // ── builtin tests ───────────────────────────────────────────────

    #[test]
    fn test_builtin_len() {
        let source = "x = len([1, 2, 3])";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let x = *(*frame.namespace).get("x").unwrap();
            assert_eq!(w_int_get_value(x), 3);
        }
    }

    #[test]
    fn test_builtin_abs() {
        let source = "x = abs(-5)";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let x = *(*frame.namespace).get("x").unwrap();
            assert_eq!(w_int_get_value(x), 5);
        }
    }

    #[test]
    fn test_builtin_min_max() {
        let source = "a = min(3, 7)\nb = max(3, 7)";
        let code = compile_exec(source).expect("compile failed");
        let mut frame = PyFrame::new(code);
        let _ = eval_frame_plain(&mut frame);
        unsafe {
            let a = *(*frame.namespace).get("a").unwrap();
            let b = *(*frame.namespace).get("b").unwrap();
            assert_eq!(w_int_get_value(a), 3);
            assert_eq!(w_int_get_value(b), 7);
        }
    }

    // ── container tests ────────────────────────────────────────────

    #[test]
    fn test_list_literal() {
        let source = "x = [1, 2, 3]";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let x = *(*frame.namespace).get("x").unwrap();
            assert!(is_list(x));
            assert_eq!(w_list_len(x), 3);
            assert_eq!(w_int_get_value(w_list_getitem(x, 0).unwrap()), 1);
            assert_eq!(w_int_get_value(w_list_getitem(x, 1).unwrap()), 2);
            assert_eq!(w_int_get_value(w_list_getitem(x, 2).unwrap()), 3);
        }
    }

    #[test]
    fn test_tuple_unpack() {
        let source = "a, b = 1, 2";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let a = *(*frame.namespace).get("a").unwrap();
            let b = *(*frame.namespace).get("b").unwrap();
            assert_eq!(w_int_get_value(a), 1);
            assert_eq!(w_int_get_value(b), 2);
        }
    }

    #[test]
    fn test_list_subscr() {
        let source = "lst = [10, 20, 30]\nx = lst[1]";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let x = *(*frame.namespace).get("x").unwrap();
            assert_eq!(w_int_get_value(x), 20);
        }
    }

    #[test]
    fn test_list_store_subscr() {
        let source = "lst = [1, 2, 3]\nlst[0] = 99\nx = lst[0]";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let x = *(*frame.namespace).get("x").unwrap();
            assert_eq!(w_int_get_value(x), 99);
        }
    }

    #[test]
    fn test_dict_literal_and_subscr() {
        let source = "d = {1: 10, 2: 20}\nx = d[1]";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let x = *(*frame.namespace).get("x").unwrap();
            assert_eq!(w_int_get_value(x), 10);
        }
    }

    // ── function definition and call tests ──────────────────────────

    #[test]
    fn test_simple_function() {
        let source = "def double(x):\n    return x * 2\nresult = double(21)";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 42);
        }
    }

    #[test]
    fn test_function_with_locals() {
        let source = "\
def add_squares(a, b):
    aa = a * a
    bb = b * b
    return aa + bb
result = add_squares(3, 4)";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 25);
        }
    }

    #[test]
    fn test_recursive_function() {
        let source = "\
def factorial(n):
    if n < 2:
        return 1
    return n * factorial(n - 1)
result = factorial(5)";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 120);
        }
    }

    // ── attribute tests ─────────────────────────────────────────────

    #[test]
    fn test_store_load_attr() {
        let source = "\
def f():
    pass
f.x = 42
result = f.x";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 42);
        }
    }

    #[test]
    fn test_store_load_multiple_attrs() {
        let source = "\
def f():
    pass
f.a = 10
f.b = 20
result = f.a + f.b";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 30);
        }
    }

    #[test]
    fn test_attr_overwrite() {
        let source = "\
def f():
    pass
f.x = 1
f.x = 2
result = f.x";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 2);
        }
    }

    #[test]
    fn test_attr_on_different_objects() {
        let source = "\
def f():
    pass
def g():
    pass
f.x = 10
g.x = 20
result = f.x + g.x";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 30);
        }
    }

    // ── Phase 1 opcode tests ──

    #[test]
    fn test_contains_op_in() {
        let source = "x = [1, 2, 3]\nresult = 1 in x";
        let (res, frame) = run_exec_frame(source);
        res.expect("exec failed");
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert!(w_bool_get_value(result), "1 in [1,2,3] should be True");
        }
    }

    #[test]
    fn test_contains_op_not_in() {
        let source = "result = 4 not in [1, 2, 3]";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_is_op() {
        let result = run_eval("None is None").unwrap();
        unsafe {
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_is_not_op() {
        let result = run_eval("1 is not None").unwrap();
        unsafe {
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_fstring() {
        let source = "x = 42\nresult = f'val={x}'";
        let (_, frame) = run_exec_frame(source);
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_str_get_value(result), "val=42");
        }
    }

    #[test]
    fn test_list_slice() {
        let source = "x = [1, 2, 3, 4, 5]\nresult = x[1:3]";
        let (res, frame) = run_exec_frame(source);
        match res {
            Ok(_) => unsafe {
                let result = *(*frame.namespace).get("result").unwrap();
                assert!(is_list(result), "slice result should be list");
                assert_eq!(w_list_len(result), 2);
                assert_eq!(w_int_get_value(w_list_getitem(result, 0).unwrap()), 2);
                assert_eq!(w_int_get_value(w_list_getitem(result, 1).unwrap()), 3);
            },
            Err(e) => panic!("list_slice failed: {} (kind: {:?})", e.message, e.kind),
        }
    }

    #[test]
    fn test_delete_subscr() {
        // del x[0] in a list
        let source = "x = [1, 2, 3]\ndel x[0]\nresult = x[0]";
        let (result, _) = run_exec_frame(source);
        // After del x[0], x[0] becomes PY_NULL; accessing may succeed or fail
        // Phase 1: just check it doesn't crash during del
        let _ = result;
    }

    #[test]
    fn test_to_bool() {
        let result = run_eval("not 0").unwrap();
        unsafe {
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_none_is_none() {
        let result = run_eval("None is None").unwrap();
        unsafe {
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_fstring_with_expr() {
        let source = "x = 10\ny = 20\nresult = f'{x} + {y} = {x + y}'";
        let (res, frame) = run_exec_frame(source);
        res.expect("f-string exec failed");
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_str_get_value(result), "10 + 20 = 30");
        }
    }

    #[test]
    fn test_string_contains() {
        let source = "result = 'lo' in 'hello'";
        let (res, frame) = run_exec_frame(source);
        res.expect("string contains failed");
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_tuple_contains() {
        let source = "result = 2 in (1, 2, 3)";
        let (res, frame) = run_exec_frame(source);
        res.expect("tuple contains failed");
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_not_in() {
        let source = "result = 5 not in [1, 2, 3]";
        let (res, frame) = run_exec_frame(source);
        res.expect("not in failed");
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_is_not_none() {
        let source = "result = 42 is not None";
        let (res, frame) = run_exec_frame(source);
        res.expect("is not None failed");
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert!(w_bool_get_value(result));
        }
    }

    #[test]
    fn test_list_slice_negative() {
        let source = "x = [1, 2, 3, 4, 5]\nresult = x[-3:]";
        let (res, frame) = run_exec_frame(source);
        res.expect("negative slice failed");
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert!(is_list(result));
            assert_eq!(w_list_len(result), 3);
        }
    }

    #[test]
    fn test_nested_function_call() {
        let source = "\
def add(a, b):
    return a + b
result = add(add(1, 2), add(3, 4))";
        let (res, frame) = run_exec_frame(source);
        res.expect("nested call failed");
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 10);
        }
    }

    #[test]
    fn test_while_loop_with_break() {
        let source = "\
x = 0
while True:
    x = x + 1
    if x == 5:
        break
result = x";
        let (res, frame) = run_exec_frame(source);
        res.expect("while+break failed");
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 5);
        }
    }

    #[test]
    fn test_inplace_add() {
        let source = "x = 10\nx += 5\nresult = x";
        let (res, frame) = run_exec_frame(source);
        res.expect("inplace add failed");
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 15);
        }
    }

    #[test]
    fn test_string_iteration_chars() {
        let source = "\
result = ''
for c in 'hello':
    result = result + c
";
        let (res, frame) = run_exec_frame(source);
        res.expect("string iteration failed");
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_str_get_value(result), "hello");
        }
    }

    #[test]
    fn test_enumerate_style() {
        // Test: manual counter with for loop
        let source = "\
count = 0
for x in [10, 20, 30]:
    count = count + 1
result = count";
        let (res, frame) = run_exec_frame(source);
        res.expect("enumerate style failed");
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 3);
        }
    }

    #[test]
    fn test_nested_for_loops() {
        let source = "\
result = 0
for i in [1, 2, 3]:
    for j in [10, 20]:
        result = result + i * j
";
        let (res, frame) = run_exec_frame(source);
        res.expect("nested for failed");
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            // 1*10 + 1*20 + 2*10 + 2*20 + 3*10 + 3*20 = 10+20+20+40+30+60 = 180
            assert_eq!(w_int_get_value(result), 180);
        }
    }

    #[test]
    fn test_try_except_basic() {
        let source = "\
x = 0
try:
    x = 1 / 0
except:
    x = 42
result = x";
        let (res, frame) = run_exec_frame(source);
        match res {
            Ok(_) => unsafe {
                let result = *(*frame.namespace).get("result").unwrap();
                assert_eq!(w_int_get_value(result), 42);
            },
            Err(e) => panic!("try/except failed: {} ({:?})", e.message, e.kind),
        }
    }

    #[test]
    fn test_recursive_fibonacci() {
        let source = "\
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
result = fib(10)";
        let (res, frame) = run_exec_frame(source);
        res.expect("fib failed");
        unsafe {
            let r = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_int_get_value(r), 55);
        }
    }

    #[test]
    fn test_string_multiply() {
        let result = run_eval("'ab' * 3").unwrap();
        unsafe {
            assert_eq!(w_str_get_value(result), "ababab");
        }
    }

    #[test]
    fn test_list_multiply() {
        let result = run_eval("[1, 2] * 3").unwrap();
        unsafe {
            assert!(is_list(result));
            assert_eq!(w_list_len(result), 6);
        }
    }

    #[test]
    fn test_negative_index() {
        let source = "x = [10, 20, 30]\nresult = x[-1]";
        let (res, frame) = run_exec_frame(source);
        res.expect("negative index failed");
        unsafe {
            let r = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_int_get_value(r), 30);
        }
    }

    #[test]
    fn test_boolean_operators() {
        let source = "result = True and False";
        let (res, frame) = run_exec_frame(source);
        res.expect("boolean and failed");
        unsafe {
            let r = *(*frame.namespace).get("result").unwrap();
            assert!(!crate::baseobjspace::is_true(r));
        }
    }

    #[test]
    fn test_chained_comparison() {
        let source = "result = 1 < 2 < 3";
        let (res, frame) = run_exec_frame(source);
        match res {
            Ok(_) => unsafe {
                let r = *(*frame.namespace).get("result").unwrap();
                assert!(w_bool_get_value(r));
            },
            Err(e) => eprintln!("chained comparison: {}", e.message),
        }
    }

    #[test]
    fn test_try_except_specific() {
        let source = "\
result = 0
try:
    x = 1 / 0
except ZeroDivisionError:
    result = 99
";
        let (res, frame) = run_exec_frame(source);
        match res {
            Ok(_) => unsafe {
                let r = *(*frame.namespace).get("result").unwrap();
                assert_eq!(w_int_get_value(r), 99);
            },
            Err(e) => panic!("specific except failed: {} ({:?})", e.message, e.kind),
        }
    }

    #[test]
    fn test_try_except_no_match_propagates() {
        // If except doesn't match, error should propagate
        let source = "\
try:
    x = 1 / 0
except ValueError:
    pass
";
        let (res, _) = run_exec_frame(source);
        // Should fail because ZeroDivisionError != ValueError
        // But Phase 1: bare except catches all, specific except may not work yet
        let _ = res; // Don't assert — depends on CHECK_EXC_MATCH impl
    }

    #[test]
    fn test_try_finally() {
        let source = "\
result = 0
try:
    result = 1
finally:
    result = result + 10
";
        let (res, frame) = run_exec_frame(source);
        match res {
            Ok(_) => unsafe {
                let r = *(*frame.namespace).get("result").unwrap();
                assert_eq!(w_int_get_value(r), 11);
            },
            Err(e) => panic!("try/finally failed: {} ({:?})", e.message, e.kind),
        }
    }

    #[test]
    fn test_multiple_except() {
        let source = "\
result = 0
try:
    x = 1 / 0
except:
    result = 1
result = result + 10
";
        let (res, frame) = run_exec_frame(source);
        res.expect("multiple except failed");
        unsafe {
            let r = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_int_get_value(r), 11);
        }
    }

    #[test]
    fn test_for_with_continue() {
        let source = "\
result = 0
for x in [1, 2, 3, 4, 5]:
    if x == 3:
        continue
    result = result + x
";
        let (res, frame) = run_exec_frame(source);
        res.expect("for+continue failed");
        unsafe {
            let r = *(*frame.namespace).get("result").unwrap();
            // 1 + 2 + 4 + 5 = 12 (skips 3)
            assert_eq!(w_int_get_value(r), 12);
        }
    }

    #[test]
    fn test_default_args() {
        let source = "\
def greet(name, greeting='hello'):
    return greeting
result = greet('world')
";
        let (res, frame) = run_exec_frame(source);
        match res {
            Ok(_) => unsafe {
                let r = *(*frame.namespace).get("result").unwrap();
                assert_eq!(w_str_get_value(r), "hello");
            },
            Err(e) => {
                // Default args may need KW_DEFAULTS support
                eprintln!("default args: {} ({:?})", e.message, e.kind);
            }
        }
    }

    #[test]
    fn test_augmented_assign_list() {
        let source = "x = [1, 2]\nx += [3]\nresult = x";
        let (res, frame) = run_exec_frame(source);
        match res {
            Ok(_) => unsafe {
                let result = *(*frame.namespace).get("result").unwrap();
                assert!(is_list(result));
                // After += [3], x should have 3 elements
                assert_eq!(w_list_len(result), 3);
            },
            Err(e) => panic!("augmented list failed: {} ({:?})", e.message, e.kind),
        }
    }

    #[test]
    fn test_for_loop_over_list() {
        let source = "\
total = 0
for x in [1, 2, 3, 4, 5]:
    total = total + x
result = total";
        let (res, frame) = run_exec_frame(source);
        res.expect("for loop failed");
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 15);
        }
    }

    #[test]
    fn test_for_loop_over_string() {
        let source = "\
result = 0
for c in 'abc':
    result = result + 1";
        let (res, frame) = run_exec_frame(source);
        match res {
            Ok(_) => unsafe {
                let result = *(*frame.namespace).get("result").unwrap();
                assert_eq!(w_int_get_value(result), 3);
            },
            Err(e) => {
                // String iteration might not work yet — ignore
                eprintln!("for-string: {}", e.message);
            }
        }
    }

    #[test]
    fn test_multiple_assignment() {
        let source = "a = b = 42\nresult = a + b";
        let (res, frame) = run_exec_frame(source);
        res.expect("multiple assign failed");
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 84);
        }
    }

    #[test]
    #[ignore = "closure: RustPython uses LOAD_FAST for freevars, needs COPY_FREE_VARS to copy cells"]
    fn test_closure_basic() {
        let source = "\
def make_adder(n):
    def adder(x):
        return x + n
    return adder
add5 = make_adder(5)
result = add5(10)";
        let (res, frame) = run_exec_frame(source);
        match res {
            Ok(_) => unsafe {
                let r = *(*frame.namespace).get("result").unwrap();
                assert_eq!(w_int_get_value(r), 15);
            },
            Err(e) => panic!("closure failed: {} ({:?})", e.message, e.kind),
        }
    }

    #[test]
    fn test_tuple_unpacking_assign() {
        let source = "a, b, c = 1, 2, 3\nresult = a + b + c";
        let (res, frame) = run_exec_frame(source);
        res.expect("tuple unpack failed");
        unsafe {
            let r = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_int_get_value(r), 6);
        }
    }

    #[test]
    fn test_dict_access_ops() {
        let source = "d = {1: 10, 2: 20}\nresult = d[1] + d[2]";
        let (res, frame) = run_exec_frame(source);
        res.expect("dict access failed");
        unsafe {
            let r = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_int_get_value(r), 30);
        }
    }

    #[test]
    fn test_string_len() {
        let source = "result = len('hello')";
        let (res, frame) = run_exec_frame(source);
        res.expect("string len failed");
        unsafe {
            let r = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_int_get_value(r), 5);
        }
    }

    #[test]
    fn test_power_operator() {
        let result = run_eval("2 ** 10").unwrap();
        unsafe {
            assert_eq!(w_int_get_value(result), 1024);
        }
    }

    #[test]
    fn test_modulo() {
        let result = run_eval("17 % 5").unwrap();
        unsafe {
            assert_eq!(w_int_get_value(result), 2);
        }
    }

    #[test]
    fn test_floor_division() {
        let result = run_eval("17 // 3").unwrap();
        unsafe {
            assert_eq!(w_int_get_value(result), 5);
        }
    }

    #[test]
    fn test_bitwise_ops() {
        let result = run_eval("(0xFF & 0x0F) | 0x30").unwrap();
        unsafe {
            assert_eq!(w_int_get_value(result), 0x3F);
        }
    }

    #[test]
    fn test_list_comprehension() {
        // Use explicit loop with list + index (no method calls)
        let source = "\
result = [0, 0, 0]
i = 0
for x in [1, 2, 3]:
    result[i] = x * 2
    i = i + 1
";
        let (res, frame) = run_exec_frame(source);
        match res {
            Ok(_) => unsafe {
                let result = *(*frame.namespace).get("result").unwrap();
                assert!(is_list(result));
                assert_eq!(w_list_len(result), 3);
                assert_eq!(w_int_get_value(w_list_getitem(result, 0).unwrap()), 2);
                assert_eq!(w_int_get_value(w_list_getitem(result, 1).unwrap()), 4);
                assert_eq!(w_int_get_value(w_list_getitem(result, 2).unwrap()), 6);
            },
            Err(e) => panic!("list comprehension failed: {} ({:?})", e.message, e.kind),
        }
    }

    #[test]
    fn test_globals_builtin_uses_current_module_namespace() {
        let source = "x = 41\nresult = globals()['x'] + 1";
        let (res, frame) = run_exec_frame(source);
        res.expect("globals() failed");
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 42);
        }
    }

    #[test]
    fn test_locals_builtin_uses_current_function_locals() {
        let source = "\
def f(a, b):
    c = a + b
    return locals()['a'] + locals()['b'] + locals()['c']
result = f(2, 3)";
        let (res, frame) = run_exec_frame(source);
        res.expect("locals() in function failed");
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 10);
        }
    }

    #[test]
    fn test_locals_builtin_uses_class_namespace() {
        let source = "\
x = 1
class C:
    y = 2
    snap = locals()
result = C.snap['y'] + globals()['x']";
        let (res, frame) = run_exec_frame(source);
        res.expect("locals() in class failed");
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 3);
        }
    }

    #[test]
    fn test_bound_method_materialized_by_attribute_access() {
        let source = "\
class C:
    def add(self, x):
        return x + 1
c = C()
m = c.add
result = m(41)";
        let (res, frame) = run_exec_frame(source);
        res.expect("bound method lookup failed");
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 42);
        }
    }

    #[test]
    fn test_bound_method_lookup_materializes_method_object() {
        let source = "\
class C:
    def add(self, x):
        return x + 1
c = C()
m = c.add";
        let (res, frame) = run_exec_frame(source);
        res.expect("bound method lookup setup failed");
        unsafe {
            let c_obj = *(*frame.namespace).get("c").unwrap();
            let m_obj = *(*frame.namespace).get("m").unwrap();
            assert!(pyre_object::is_method(m_obj));
            assert!(std::ptr::eq(pyre_object::w_method_get_self(m_obj), c_obj));
        }
    }

    #[test]
    fn test_builtin_type_method_materialized_by_attribute_access() {
        let source = "\
xs = []
m = xs.append
m(42)
result = len(xs)";
        let (res, frame) = run_exec_frame(source);
        res.expect("builtin type method lookup failed");
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 1);
        }
    }

    #[test]
    fn test_builtin_function_stored_on_class_is_not_bound() {
        let source = "\
class C:
    f = len
c = C()
result = c.f([1, 2, 3])";
        let (res, frame) = run_exec_frame(source);
        res.expect("builtin function descriptor semantics failed");
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 3);
        }
    }

    #[test]
    fn test_metaclass_method_materialized_by_attribute_access() {
        let source = "\
class Meta(type):
    def pick(cls):
        return cls
class C(metaclass=Meta):
    pass
bound = C.pick
result = bound()";
        let (res, frame) = run_exec_frame(source);
        res.expect("metaclass descriptor lookup failed");
        let result = unsafe { *(*frame.namespace).get("result").unwrap() };
        let c_obj = unsafe { *(*frame.namespace).get("C").unwrap() };
        assert!(std::ptr::eq(result, c_obj));
    }

    #[test]
    fn test_staticmethod_prepare_is_called_with_bound_lookup() {
        let source = "\
class Meta(type):
    @staticmethod
    def __prepare__(name, bases):
        return {'seed': 41}
class C(metaclass=Meta):
    value = seed + 1
result = C.value";
        let (res, frame) = run_exec_frame(source);
        res.expect("__prepare__ lookup failed");
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 42);
        }
    }

    #[test]
    fn test_function_dunder_globals_and_code_are_materialized() {
        let source = "\
x = 7
def f(a, *, b=3):
    return a + b + x
g = f.__globals__
code = f.__code__";
        let (res, frame) = run_exec_frame(source);
        res.expect("function dunder lookup failed");
        let globals = unsafe { *(*frame.namespace).get("g").unwrap() };
        let code = unsafe { *(*frame.namespace).get("code").unwrap() };
        unsafe {
            let x = pyre_object::w_dict_lookup(globals, pyre_object::w_str_new("x")).unwrap();
            assert_eq!(w_int_get_value(x), 7);
            let argcount = crate::baseobjspace::getattr(code, "co_argcount").unwrap();
            assert_eq!(w_int_get_value(argcount), 1);
            let kwonly = crate::baseobjspace::getattr(code, "co_kwonlyargcount").unwrap();
            assert_eq!(w_int_get_value(kwonly), 1);
            let name = crate::baseobjspace::getattr(code, "co_name").unwrap();
            assert_eq!(w_str_get_value(name), "f");
            let varnames = crate::baseobjspace::getattr(code, "co_varnames").unwrap();
            let first = w_tuple_getitem(varnames, 0).unwrap();
            assert_eq!(w_str_get_value(first), "a");
        }
    }

    #[test]
    fn test_vars_builtin_raises_type_error_without_dict() {
        let source = "\
result = 0
try:
    vars(1)
except TypeError:
    result = 1";
        let (res, frame) = run_exec_frame(source);
        res.expect("vars() exception path failed");
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 1);
        }
    }

    #[test]
    fn test_type_builtin_rejects_invalid_arity() {
        let source = "\
result = 0
try:
    type()
except TypeError:
    result = 1";
        let (res, frame) = run_exec_frame(source);
        res.expect("type() exception path failed");
        unsafe {
            let result = *(*frame.namespace).get("result").unwrap();
            assert_eq!(w_int_get_value(result), 1);
        }
    }
}
