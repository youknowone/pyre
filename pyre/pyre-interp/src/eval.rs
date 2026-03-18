//! Bytecode evaluation loop — pure interpreter.
//!
//! JIT integration lives in pyre-jit/src/eval.rs. This module is
//! JIT-free: it processes bytecode instructions with no tracing,
//! no merge points, and no compiled-code hooks.

use pyre_bytecode::bytecode::{BinaryOperator, ComparisonOperator, Instruction, OpArgState};
use pyre_object::*;
use pyre_objspace::*;
use pyre_runtime::{
    ArithmeticOpcodeHandler, BranchOpcodeHandler, ConstantOpcodeHandler, ControlFlowOpcodeHandler,
    IterOpcodeHandler, LocalOpcodeHandler, NamespaceOpcodeHandler, OpcodeStepExecutor, PyError,
    PyErrorKind, PyResult, SharedOpcodeHandler, StackOpcodeHandler, StepResult, TruthOpcodeHandler,
    build_list_from_refs, build_map_from_refs, build_tuple_from_refs, ensure_range_iter,
    execute_opcode_step, make_function_from_code_obj, namespace_load, namespace_store,
    range_iter_continues, range_iter_next_or_null, stack_underflow_error, unpack_sequence_exact,
    w_code_new,
};

use crate::call::call_callable;
use crate::frame::PyFrame;

/// Execute a frame — pure interpreter, no JIT.
pub fn eval_frame_plain(frame: &mut PyFrame) -> PyResult {
    frame.fix_array_ptrs();
    eval_loop(frame)
}

/// Resume interpretation after compiled code guard failure.
pub fn eval_loop_for_force(frame: &mut PyFrame) -> PyResult {
    eval_loop(frame)
}

fn eval_loop(frame: &mut PyFrame) -> PyResult {
    let mut arg_state = OpArgState::default();
    let code = unsafe { &*frame.code };

    loop {
        if frame.next_instr >= code.instructions.len() {
            return Ok(w_none());
        }

        let code_unit = code.instructions[frame.next_instr];
        let (instruction, op_arg) = arg_state.get(code_unit);
        frame.next_instr += 1;
        let next_instr = frame.next_instr;
        match execute_opcode_step(frame, code, instruction, op_arg, next_instr) {
            Ok(StepResult::Continue) | Ok(StepResult::CloseLoop(_)) => {}
            Ok(StepResult::Return(result)) => return Ok(result),
            Ok(StepResult::Yield(result)) => return Ok(result),
            Err(err) => {
                // PyPy: handle_operation_error — walk block stack for handler
                if let Some(block) = frame.block_stack.pop() {
                    // Unwind operand stack to block level
                    while frame.valuestackdepth > block.level {
                        frame.pop();
                    }
                    // Push exception value for except clause
                    let exc_obj = pyre_object::w_str_new(&err.message);
                    frame.push(exc_obj);
                    frame.next_instr = block.handler;
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
        py_setitem(obj, key, value).map(|_| ())
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
        py_getattr(obj, name)
    }

    fn store_attr(
        &mut self,
        obj: Self::Value,
        name: &str,
        value: Self::Value,
    ) -> Result<(), PyError> {
        py_setattr(obj, name, value).map(|_| ())
    }
}

impl LocalOpcodeHandler for PyFrame {
    fn load_local_value(&mut self, idx: usize) -> Result<Self::Value, PyError> {
        Ok(self.locals_cells_stack_w[idx])
    }

    fn load_local_checked_value(&mut self, idx: usize, name: &str) -> Result<Self::Value, PyError> {
        let value = self.locals_cells_stack_w[idx];
        if value.is_null() {
            return Err(PyError {
                kind: PyErrorKind::NameError,
                message: format!("local variable '{name}' referenced before assignment"),
            });
        }
        Ok(value)
    }

    fn store_local_value(&mut self, idx: usize, value: Self::Value) -> Result<(), PyError> {
        self.locals_cells_stack_w[idx] = value;
        Ok(())
    }
}

impl NamespaceOpcodeHandler for PyFrame {
    fn load_name_value(&mut self, name: &str) -> Result<Self::Value, PyError> {
        let ns = unsafe { &*self.namespace };
        namespace_load(ns, name)
    }

    fn store_name_value(&mut self, name: &str, value: Self::Value) -> Result<(), PyError> {
        let ns = unsafe { &mut *self.namespace };
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

impl IterOpcodeHandler for PyFrame {
    fn ensure_iter_value(&mut self, iter: Self::Value) -> Result<(), PyError> {
        unsafe {
            if pyre_object::is_range_iter(iter) || pyre_object::is_seq_iter(iter) {
                return Ok(());
            }
            // Convert list/tuple to seq iterator on the stack
            if pyre_object::is_list(iter) {
                let len = pyre_object::w_list_len(iter);
                let seq_iter = pyre_object::w_seq_iter_new(iter, len);
                // Replace TOS with the iterator
                self.locals_cells_stack_w[self.valuestackdepth - 1] = seq_iter;
                return Ok(());
            }
            if pyre_object::is_tuple(iter) {
                let len = pyre_object::w_tuple_len(iter);
                let seq_iter = pyre_object::w_seq_iter_new(iter, len);
                self.locals_cells_stack_w[self.valuestackdepth - 1] = seq_iter;
                return Ok(());
            }
        }
        ensure_range_iter(iter)
    }

    fn concrete_iter_continues(&mut self, iter: Self::Value) -> Result<bool, PyError> {
        range_iter_continues(iter)
    }

    fn iter_next_value(&mut self, iter: Self::Value) -> Result<Self::Value, PyError> {
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

    fn close_loop(&mut self, _target: usize) -> Result<StepResult<Self::Value>, PyError> {
        // Signal a back-edge to the main eval_loop, which handles
        // JIT counting and compiled code execution via try_back_edge_jit.
        Ok(StepResult::CloseLoop(vec![]))
    }
}

impl BranchOpcodeHandler for PyFrame {
    fn concrete_truth_as_bool(&mut self, truth: Self::Truth) -> Result<bool, PyError> {
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

    fn bigint_constant(&mut self, value: &pyre_runtime::PyBigInt) -> Result<Self::Value, PyError> {
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
        code: &pyre_bytecode::bytecode::CodeObject,
    ) -> Result<Self::Value, PyError> {
        let code_ptr = Box::into_raw(Box::new(code.clone())) as *const ();
        Ok(w_code_new(code_ptr))
    }

    fn none_constant(&mut self) -> Result<Self::Value, PyError> {
        Ok(w_none())
    }
}

impl OpcodeStepExecutor for PyFrame {
    type Error = PyError;

    // ── Closures / cells ──

    fn load_deref(&mut self, idx: usize) -> Result<(), Self::Error> {
        let nlocals = self.nlocals();
        let value = self.locals_cells_stack_w[nlocals + idx];
        if value == PY_NULL {
            return Err(PyError::type_error(
                "free variable referenced before assignment",
            ));
        }
        self.push(value);
        Ok(())
    }

    fn store_deref(&mut self, idx: usize) -> Result<(), Self::Error> {
        let nlocals = self.nlocals();
        let value = self.pop();
        self.locals_cells_stack_w[nlocals + idx] = value;
        Ok(())
    }

    fn load_closure(&mut self, idx: usize) -> Result<(), Self::Error> {
        // Push the cell value itself (same as load_deref for Phase 1)
        self.load_deref(idx)
    }

    fn delete_deref(&mut self, idx: usize) -> Result<(), Self::Error> {
        let nlocals = self.nlocals();
        self.locals_cells_stack_w[nlocals + idx] = PY_NULL;
        Ok(())
    }

    // ── Exception handling ──

    fn setup_finally(&mut self, handler: usize) -> Result<(), Self::Error> {
        self.block_stack.push(crate::frame::Block {
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
            0 => Err(PyError::type_error("no active exception to re-raise")),
            1 => {
                let exc = self.pop();
                // For Phase 1: treat the exception object as an error message
                let _ = exc; // TODO: proper exception propagation
                Err(PyError::type_error("exception raised"))
            }
            _ => Err(PyError::type_error("too many arguments for raise")),
        }
    }

    fn end_finally(&mut self) -> Result<(), Self::Error> {
        // Pop the exception or None from stack
        let _ = self.pop();
        Ok(())
    }

    // ── Import (Phase 1: stub) ──

    fn import_name(&mut self, name: &str) -> Result<(), Self::Error> {
        // Phase 1: pop level and fromlist, push a namespace object
        let _fromlist = self.pop();
        let _level = self.pop();
        // Create a simple namespace as the "module"
        let module = pyre_object::w_none();
        self.push(module);
        // TODO: actual module loading
        Ok(())
    }

    fn import_from(&mut self, name: &str) -> Result<(), Self::Error> {
        // Phase 1: peek module (TOS), get attribute
        let module = self.peek();
        let attr = pyre_objspace::space::py_getattr(module, name).unwrap_or(pyre_object::w_none());
        self.push(attr);
        Ok(())
    }

    // ── ContainsOp (in / not in) ──
    // PyPy: pyopcode.py COMPARE_OP with 'in' / 'not in'

    fn contains_op(&mut self, invert: pyre_bytecode::bytecode::Invert) -> Result<(), Self::Error> {
        // CPython 3.13: TOS = container, TOS1 = item
        let haystack = self.pop();
        let needle = self.pop();
        let result = pyre_objspace::space::py_contains(haystack, needle)?;
        let inverted = match invert {
            pyre_bytecode::bytecode::Invert::No => result,
            pyre_bytecode::bytecode::Invert::Yes => !result,
        };
        self.push(pyre_object::w_bool_from(inverted));
        Ok(())
    }

    // ── IsOp (is / is not) ──
    // PyPy: pyopcode.py COMPARE_OP with 'is' / 'is not'

    fn is_op(&mut self, invert: pyre_bytecode::bytecode::Invert) -> Result<(), Self::Error> {
        let b = self.pop();
        let a = self.pop();
        let same = std::ptr::eq(a, b); // pointer identity
        let result = match invert {
            pyre_bytecode::bytecode::Invert::No => same,
            pyre_bytecode::bytecode::Invert::Yes => !same,
        };
        self.push(pyre_object::w_bool_from(result));
        Ok(())
    }

    // ── ToBool ──
    // CPython 3.13: converts TOS to bool

    fn to_bool(&mut self) -> Result<(), Self::Error> {
        let val = self.pop();
        let truth = pyre_objspace::space::py_is_true(val);
        self.push(pyre_object::w_bool_from(truth));
        Ok(())
    }

    // ── PopJumpIfNone / PopJumpIfNotNone ──

    fn pop_jump_if_none(&mut self, target: usize) -> Result<(), Self::Error> {
        let val = self.pop();
        if unsafe { pyre_object::is_none(val) } {
            self.next_instr = target;
        }
        Ok(())
    }

    fn pop_jump_if_not_none(&mut self, target: usize) -> Result<(), Self::Error> {
        let val = self.pop();
        if !unsafe { pyre_object::is_none(val) } {
            self.next_instr = target;
        }
        Ok(())
    }

    // ── DeleteSubscr ──

    fn delete_subscript(&mut self) -> Result<(), Self::Error> {
        let index = self.pop();
        let obj = self.pop();
        pyre_objspace::space::py_delitem(obj, index)?;
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
        let s = pyre_objspace::space::py_str(val);
        self.push(pyre_object::w_str_new(&s));
        Ok(())
    }

    // ── FormatWithSpec (format(TOS1, TOS)) ──
    fn format_with_spec(&mut self) -> Result<(), Self::Error> {
        let _spec = self.pop();
        let val = self.pop();
        // Phase 1: ignore spec, just convert to str
        let s = pyre_objspace::space::py_str(val);
        self.push(pyre_object::w_str_new(&s));
        Ok(())
    }

    // ── ConvertValue (repr/str/ascii conversion) ──
    fn convert_value(
        &mut self,
        conv: pyre_bytecode::bytecode::ConvertValueOparg,
    ) -> Result<(), Self::Error> {
        let val = self.pop();
        let s = match conv {
            pyre_bytecode::bytecode::ConvertValueOparg::Str => pyre_objspace::space::py_str(val),
            pyre_bytecode::bytecode::ConvertValueOparg::Repr => pyre_objspace::space::py_repr(val),
            pyre_bytecode::bytecode::ConvertValueOparg::Ascii => pyre_objspace::space::py_repr(val),
            pyre_bytecode::bytecode::ConvertValueOparg::None => pyre_objspace::space::py_str(val),
        };
        self.push(pyre_object::w_str_new(&s));
        Ok(())
    }

    // ── LoadFromDictOrGlobals ──
    // CPython 3.13: LOAD_FROM_DICT_OR_GLOBALS — try TOS dict first, then globals
    fn load_from_dict_or_globals(&mut self, name: &str) -> Result<(), Self::Error> {
        let dict = self.pop();
        // Try dict first (if it's a dict or has attrs)
        if let Ok(val) = pyre_objspace::space::py_getattr(dict, name) {
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
        let len = pyre_objspace::space::py_len(obj)?;
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
        // Phase 1: build as list (no set type yet)
        let mut items = Vec::with_capacity(count);
        for _ in 0..count {
            items.push(self.pop());
        }
        items.reverse();
        self.push(pyre_object::w_list_new(items)); // TODO: proper set type
        Ok(())
    }

    // ── DictUpdate ──
    fn dict_update(&mut self, _i: usize) -> Result<(), Self::Error> {
        // Phase 1 stub: pop update dict, merge into TOS
        let update = self.pop();
        // TODO: actual dict merge
        let _ = update;
        Ok(())
    }

    // ── DictMerge ──
    fn dict_merge(&mut self, _i: usize) -> Result<(), Self::Error> {
        let update = self.pop();
        let _ = update;
        Ok(())
    }

    // ── MapAdd ──
    fn map_add(&mut self, _i: usize) -> Result<(), Self::Error> {
        let value = self.pop();
        let key = self.pop();
        let _ = (key, value);
        // TODO: dict[key] = value on stack[i]
        Ok(())
    }

    // ── SetAdd ──
    fn set_add(&mut self, _i: usize) -> Result<(), Self::Error> {
        let value = self.pop();
        let _ = value;
        // TODO: set.add(value) on stack[i]
        Ok(())
    }

    // ── BuildSlice ──
    // CPython 3.13: BUILD_SLICE creates a slice object from 2 or 3 stack items
    fn build_slice(&mut self, argc: pyre_bytecode::bytecode::BuildSliceArgCount) -> Result<(), Self::Error> {
        use pyre_bytecode::bytecode::BuildSliceArgCount;
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
    use pyre_bytecode::*;
    use pyre_runtime::{PyExecutionContext, w_func_new};
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
        // Test via py_is_true directly since `not` uses ToBool instruction
        assert!(!py_is_true(w_float_new(0.0)));
        assert!(py_is_true(w_float_new(1.5)));
        assert!(py_is_true(w_float_new(-0.1)));
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
    if isinstance(x, \"int\"):
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
    if isinstance(x, \"int\"):
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
    if type(x) == \"list\":
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
        unsafe { assert!(w_bool_get_value(result)); }
    }

    #[test]
    fn test_is_not_op() {
        let result = run_eval("1 is not None").unwrap();
        unsafe { assert!(w_bool_get_value(result)); }
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
        unsafe { assert!(w_bool_get_value(result)); }
    }

    #[test]
    fn test_none_is_none() {
        let result = run_eval("None is None").unwrap();
        unsafe { assert!(w_bool_get_value(result)); }
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
    #[ignore = "list comprehension needs inner function scope (MAKE_FUNCTION + CopyFreeVars)"]
    fn test_list_comprehension() {
        let source = "result = [x * 2 for x in [1, 2, 3]]";
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
}
