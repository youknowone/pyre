//! `_pickle.Unpickler` — `interp_pickle.py W_Unpickler` (atom + container subset).

use pyre_object::PyObjectRef;

use crate::PyError;

use super::{
    HIGHEST_PROTOCOL, call_fn, call_meth, decode_long, getattribute_dotted, import_module, op,
    parse_int_text, read_int_le, str_from_utf8, unpickling_error,
};

#[crate::pyre_class("_pickle.Unpickler")]
pub struct W_Unpickler {
    w_file_read: PyObjectRef,
    w_file_readline: PyObjectRef,
    /// Result stack — a Python `list` (GC-managed across `read` allocs).
    w_stack: PyObjectRef,
    /// Saved stacks for the MARK machinery — a Python `list` of lists.
    w_metastack: PyObjectRef,
    /// Memo — a Python `dict` keyed by integer index.
    w_memo: PyObjectRef,
    /// Next free memo slot (`_memo_append` target).
    memo_index: i64,
    /// Active frame bytes (`bytes`) or None.
    w_frame: PyObjectRef,
    frame_index: i64,
    proto: i64,
    /// Apply the `_compat_pickle` py2→py3 name remap at protocol < 3.
    fix_imports: bool,
    /// Out-of-band `buffers` iterator (proto 5), or None.
    w_buffers: PyObjectRef,
}

#[crate::pyre_methods(doc = "Unpickler(file) -> unpickler reading from file.")]
impl W_Unpickler {
    #[staticmethod]
    fn __new__(_cls: PyObjectRef) -> PyObjectRef {
        W_Unpickler::allocate(W_Unpickler {
            ob: pyre_object::PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_file_read: pyre_object::w_none(),
            w_file_readline: pyre_object::w_none(),
            w_stack: pyre_object::w_none(),
            w_metastack: pyre_object::w_none(),
            w_memo: pyre_object::w_none(),
            memo_index: 0,
            w_frame: pyre_object::w_none(),
            frame_index: 0,
            proto: 0,
            fix_imports: true,
            w_buffers: pyre_object::w_none(),
        })
    }

    fn __init__(
        &mut self,
        file: PyObjectRef,
        #[default(pyre_object::boolobject::w_bool_from(true))] fix_imports: PyObjectRef,
        #[default(pyre_object::w_none())] encoding: PyObjectRef,
        #[default(pyre_object::w_none())] errors: PyObjectRef,
        #[default(pyre_object::w_none())] buffers: PyObjectRef,
    ) -> Result<(), PyError> {
        // `encoding` / `errors` govern the legacy py2 byte string decode path;
        // pyre stores unicode natively, so they are accepted for signature
        // compatibility. `fix_imports` gates the proto-< 3 py2→py3 name remap.
        let _ = (encoding, errors);
        self.fix_imports = crate::baseobjspace::is_true(fix_imports);
        self.w_file_read = crate::baseobjspace::getattr_str(file, "read")?;
        self.w_file_readline = crate::baseobjspace::getattr_str(file, "readline")?;
        self.w_stack = pyre_object::w_none();
        self.w_metastack = pyre_object::w_none();
        // The memo persists across `load` calls (a multi-object stream may
        // back-reference an object memoized by an earlier load).
        self.w_memo = pyre_object::dictmultiobject::w_dict_new();
        self.memo_index = 0;
        self.w_frame = pyre_object::w_none();
        self.frame_index = 0;
        self.proto = 0;
        // A non-None `buffers` is consumed as an iterator by NEXT_BUFFER.
        self.w_buffers = if unsafe { pyre_object::is_none(buffers) } {
            pyre_object::w_none()
        } else {
            crate::baseobjspace::iter(buffers)?
        };
        Ok(())
    }

    fn load(&mut self) -> Result<PyObjectRef, PyError> {
        // Fresh stack each load; the memo persists across `load` calls so a
        // later object can back-reference one memoized by an earlier load
        // (lazily created when the unpickler was built only via `__new__`).
        self.w_stack = pyre_object::listobject::w_list_new(Vec::new());
        self.w_metastack = pyre_object::listobject::w_list_new(Vec::new());
        if unsafe { pyre_object::is_none(self.w_memo) } {
            self.w_memo = pyre_object::dictmultiobject::w_dict_new();
            self.memo_index = 0;
        }
        self.w_frame = pyre_object::w_none();
        self.frame_index = 0;
        self.proto = 0;

        let self_ptr = self as *mut W_Unpickler as PyObjectRef;
        let _roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(self_ptr);
        let slot = pyre_object::gc_roots::shadow_stack_len() - 1;

        loop {
            let opcode = read1(slot)?;
            if opcode == op::STOP {
                let me = cur(slot);
                return unsafe { pyre_object::listobject::w_list_pop_end(me.w_stack) }
                    .ok_or_else(|| unpickling_error("STOP with empty stack"));
            }
            dispatch(slot, opcode)?;
        }
    }
}

/// Re-read the (possibly relocated) unpickler from the pinned shadow slot.
#[inline]
fn cur(slot: usize) -> &'static mut W_Unpickler {
    unsafe { &mut *(pyre_object::gc_roots::shadow_stack_get(slot) as *mut W_Unpickler) }
}

// ── stack / metastack helpers ────────────────────────────────────────

fn push(slot: usize, obj: PyObjectRef) {
    let me = cur(slot);
    unsafe { pyre_object::listobject::w_list_append(me.w_stack, obj) };
}

/// `data_pop` — pop the top of the current stack.
fn pop(slot: usize) -> Result<PyObjectRef, PyError> {
    let me = cur(slot);
    unsafe { pyre_object::listobject::w_list_pop_end(me.w_stack) }
        .ok_or_else(|| unpickling_error("unpickling stack underflow"))
}

/// `_stack_top` — the top of the current stack without removing it.
fn top(slot: usize, opcode_name: &str) -> Result<PyObjectRef, PyError> {
    let me = cur(slot);
    let n = unsafe { pyre_object::listobject::w_list_len(me.w_stack) };
    if n < 1 {
        return Err(unpickling_error(&format!("stack empty in {opcode_name}")));
    }
    Ok(unsafe { pyre_object::listobject::w_list_getitem(me.w_stack, (n - 1) as i64).unwrap() })
}

/// `load_mark` — save the current stack and start a fresh one.
fn mark(slot: usize) {
    let me = cur(slot);
    unsafe { pyre_object::listobject::w_list_append(me.w_metastack, me.w_stack) };
    let new_stack = pyre_object::listobject::w_list_new(Vec::new());
    cur(slot).w_stack = new_stack;
}

/// `pop_mark` — return the items pushed since the last MARK and restore the
/// previous stack.
fn pop_mark(slot: usize) -> Result<PyObjectRef, PyError> {
    let me = cur(slot);
    let items = me.w_stack;
    let prev = unsafe { pyre_object::listobject::w_list_pop_end(me.w_metastack) }
        .ok_or_else(|| unpickling_error("no items on stack"))?;
    cur(slot).w_stack = prev;
    Ok(items)
}

// ── out-of-band buffers ──────────────────────────────────────────────

/// `load_next_buffer` (NEXT_BUFFER) — push the next buffer from the
/// `buffers` iterator given at construction.
fn load_next_buffer(slot: usize) -> Result<(), PyError> {
    let w_buffers = cur(slot).w_buffers;
    if unsafe { pyre_object::is_none(w_buffers) } {
        return Err(unpickling_error(
            "pickle stream refers to out-of-band data but no *buffers* argument was given",
        ));
    }
    let w_buf = match crate::baseobjspace::next(w_buffers) {
        Ok(b) => b,
        Err(e) if e.kind == crate::PyErrorKind::StopIteration => {
            return Err(unpickling_error("not enough out-of-band buffers"));
        }
        Err(e) => return Err(e),
    };
    push(slot, w_buf);
    Ok(())
}

/// `load_readonly_buffer` (READONLY_BUFFER) — replace the top buffer with a
/// read-only memoryview onto it.
fn load_readonly_buffer(slot: usize) -> Result<(), PyError> {
    let w_buf = top(slot, "READONLY_BUFFER")?;
    let w_mv = call_fn(memoryview_type()?, &[w_buf])?;
    let w_readonly = call_meth(w_mv, "toreadonly", &[])?;
    // Replace the top of the stack (`stack[-1] = w_readonly`).
    pop(slot)?;
    push(slot, w_readonly);
    Ok(())
}

/// The `memoryview` builtin type via the live execution context.
fn memoryview_type() -> Result<PyObjectRef, PyError> {
    let frame = crate::eval::CURRENT_FRAME.with(|f| f.get());
    let ec = if frame.is_null() {
        std::ptr::null()
    } else {
        unsafe { (*frame).execution_context }
    };
    if ec.is_null() {
        return Err(unpickling_error("memoryview type unavailable"));
    }
    unsafe { (*ec).lookup_builtin("memoryview") }
        .ok_or_else(|| unpickling_error("memoryview type unavailable"))
}

// ── memo helpers ─────────────────────────────────────────────────────

/// `_memo_put` — store `w_val` at index `i`, advancing the next-free slot.
fn memo_put(slot: usize, i: i64, w_val: PyObjectRef) {
    let me = cur(slot);
    unsafe { pyre_object::dictmultiobject::w_dict_setitem(me.w_memo, i, w_val) };
    let me = cur(slot);
    if i >= me.memo_index {
        me.memo_index = i + 1;
    }
}

/// `_memo_append` — store `w_val` at the next free slot.
fn memo_append(slot: usize, w_val: PyObjectRef) {
    let i = cur(slot).memo_index;
    memo_put(slot, i, w_val);
}

fn memo_get(slot: usize, i: i64) -> Result<PyObjectRef, PyError> {
    let me = cur(slot);
    unsafe { pyre_object::dictmultiobject::w_dict_getitem(me.w_memo, i) }
        .ok_or_else(|| unpickling_error(&format!("Memo value not found at index {i}")))
}

// ── reading ──────────────────────────────────────────────────────────

/// Read one opcode byte (from the active frame, else the file).
fn read1(slot: usize) -> Result<u8, PyError> {
    let me = cur(slot);
    if !unsafe { pyre_object::is_none(me.w_frame) } {
        let frame = unsafe { pyre_object::bytesobject::w_bytes_data(me.w_frame) };
        let idx = me.frame_index as usize;
        if idx < frame.len() {
            me.frame_index += 1;
            return Ok(frame[idx]);
        }
    }
    let v = read(slot, 1)?;
    Ok(v[0])
}

/// Read `n` bytes (from the active frame, else the file). Returns an owned
/// copy so the result survives later allocations.
fn read(slot: usize, n: usize) -> Result<Vec<u8>, PyError> {
    let me = cur(slot);
    if !unsafe { pyre_object::is_none(me.w_frame) } {
        let frame = unsafe { pyre_object::bytesobject::w_bytes_data(me.w_frame) };
        let idx = me.frame_index as usize;
        if idx + n <= frame.len() {
            let out = frame[idx..idx + n].to_vec();
            me.frame_index += n as i64;
            return Ok(out);
        }
        // Frame exhausted — fall through to the file.
        me.w_frame = pyre_object::w_none();
        me.frame_index = 0;
    }
    let w_n = pyre_object::w_int_new(n as i64);
    let read_fn = cur(slot).w_file_read;
    let w_res = call_fn(read_fn, &[w_n])?;
    let data = unsafe { pyre_object::bytesobject::w_bytes_data(w_res) };
    if data.len() < n {
        return Err(unpickling_error("pickle data was truncated"));
    }
    Ok(data[..n].to_vec())
}

fn dispatch(slot: usize, opcode: u8) -> Result<(), PyError> {
    match opcode {
        x if x == op::PROTO => {
            let p = read1(slot)? as i64;
            if !(0..=HIGHEST_PROTOCOL).contains(&p) {
                return Err(PyError::value_error("unsupported pickle protocol"));
            }
            cur(slot).proto = p;
        }
        x if x == op::FRAME => {
            let sz = read(slot, 8)?;
            let frame_size = read_int_le(&sz) as usize;
            // Load the frame body from the file.
            let w_n = pyre_object::w_int_new(frame_size as i64);
            let read_fn = cur(slot).w_file_read;
            let w_res = call_fn(read_fn, &[w_n])?;
            let data = unsafe { pyre_object::bytesobject::w_bytes_data(w_res) };
            if data.len() < frame_size {
                return Err(unpickling_error("pickle data was truncated"));
            }
            let w_frame = pyre_object::w_bytes_from_bytes(&data[..frame_size]);
            let me = cur(slot);
            me.w_frame = w_frame;
            me.frame_index = 0;
        }
        x if x == op::NONE => push(slot, pyre_object::w_none()),
        x if x == op::NEWTRUE => push(slot, pyre_object::w_bool_from(true)),
        x if x == op::NEWFALSE => push(slot, pyre_object::w_bool_from(false)),
        x if x == op::BININT => {
            let d = read(slot, 4)?;
            let v = i32::from_le_bytes([d[0], d[1], d[2], d[3]]) as i64;
            push(slot, pyre_object::w_int_new(v));
        }
        x if x == op::BININT1 => {
            let d = read(slot, 1)?;
            push(slot, pyre_object::w_int_new(d[0] as i64));
        }
        x if x == op::BININT2 => {
            let d = read(slot, 2)?;
            push(
                slot,
                pyre_object::w_int_new(u16::from_le_bytes([d[0], d[1]]) as i64),
            );
        }
        x if x == op::LONG1 => {
            let n = read(slot, 1)?[0] as usize;
            let d = read(slot, n)?;
            push(slot, decode_long(&d));
        }
        x if x == op::LONG4 => {
            let nb = read(slot, 4)?;
            let n = i32::from_le_bytes([nb[0], nb[1], nb[2], nb[3]]) as usize;
            let d = read(slot, n)?;
            push(slot, decode_long(&d));
        }
        x if x == op::BINFLOAT => {
            let d = read(slot, 8)?;
            let f = f64::from_be_bytes([d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]]);
            push(slot, pyre_object::w_float_new(f));
        }
        x if x == op::INT => {
            let s = read_line(slot)?;
            let w = match s.as_str() {
                "00" => pyre_object::w_bool_from(false),
                "01" => pyre_object::w_bool_from(true),
                _ => parse_int_text(&s)?,
            };
            push(slot, w);
        }
        x if x == op::LONG => {
            let mut s = read_line(slot)?;
            // strip the Python 2 'L' suffix, if present.
            if s.ends_with('L') {
                s.pop();
            }
            push(slot, parse_int_text(&s)?);
        }
        x if x == op::FLOAT => {
            let s = read_line(slot)?;
            let f = s
                .trim()
                .parse::<f64>()
                .map_err(|_| PyError::value_error("could not convert string to float"))?;
            push(slot, pyre_object::w_float_new(f));
        }
        x if x == op::UNICODE => {
            // raw-unicode-escape over the line's raw bytes.
            let data = read_line_bytes(slot)?;
            let w_bytes = pyre_object::w_bytes_from_bytes(&data);
            let w_uni = call_meth(
                w_bytes,
                "decode",
                &[pyre_object::w_str_new("raw-unicode-escape")],
            )?;
            push(slot, w_uni);
        }
        x if x == op::SHORT_BINUNICODE => {
            let n = read(slot, 1)?[0] as usize;
            let d = read(slot, n)?;
            push(slot, str_from_utf8(&d)?);
        }
        x if x == op::BINUNICODE => {
            let nb = read(slot, 4)?;
            let n = u32::from_le_bytes([nb[0], nb[1], nb[2], nb[3]]) as usize;
            let d = read(slot, n)?;
            push(slot, str_from_utf8(&d)?);
        }
        x if x == op::BINUNICODE8 => {
            let nb = read(slot, 8)?;
            let n = read_int_le(&nb) as usize;
            let d = read(slot, n)?;
            push(slot, str_from_utf8(&d)?);
        }
        x if x == op::SHORT_BINBYTES => {
            let n = read(slot, 1)?[0] as usize;
            let d = read(slot, n)?;
            push(slot, pyre_object::w_bytes_from_bytes(&d));
        }
        x if x == op::BINBYTES => {
            let nb = read(slot, 4)?;
            let n = u32::from_le_bytes([nb[0], nb[1], nb[2], nb[3]]) as usize;
            let d = read(slot, n)?;
            push(slot, pyre_object::w_bytes_from_bytes(&d));
        }
        x if x == op::BINBYTES8 => {
            let nb = read(slot, 8)?;
            let n = read_int_le(&nb) as usize;
            let d = read(slot, n)?;
            push(slot, pyre_object::w_bytes_from_bytes(&d));
        }
        // ── stack ────────────────────────────────────────────────────
        x if x == op::MARK => mark(slot),
        x if x == op::POP => {
            // Pop a stack item, or discard the topmost MARK group.
            let me = cur(slot);
            let n = unsafe { pyre_object::listobject::w_list_len(me.w_stack) };
            if n > 0 {
                pop(slot)?;
            } else {
                pop_mark(slot)?;
            }
        }
        x if x == op::POP_MARK => {
            pop_mark(slot)?;
        }
        // ── tuple ─────────────────────────────────────────────────────
        x if x == op::EMPTY_TUPLE => push(slot, pyre_object::tupleobject::w_tuple_new(Vec::new())),
        x if x == op::TUPLE => {
            let items = pop_mark(slot)?;
            push(slot, list_to_tuple(items));
        }
        x if x == op::TUPLE1 => {
            let a = pop(slot)?;
            push(slot, pyre_object::tupleobject::w_tuple_new(vec![a]));
        }
        x if x == op::TUPLE2 => {
            let b = pop(slot)?;
            let a = pop(slot)?;
            push(slot, pyre_object::tupleobject::w_tuple_new(vec![a, b]));
        }
        x if x == op::TUPLE3 => {
            let c = pop(slot)?;
            let b = pop(slot)?;
            let a = pop(slot)?;
            push(slot, pyre_object::tupleobject::w_tuple_new(vec![a, b, c]));
        }
        // ── list ──────────────────────────────────────────────────────
        x if x == op::EMPTY_LIST => push(slot, pyre_object::listobject::w_list_new(Vec::new())),
        x if x == op::LIST => {
            let items = pop_mark(slot)?;
            push(slot, list_copy(items));
        }
        x if x == op::APPEND => {
            let value = pop(slot)?;
            let w_list = top(slot, "APPEND")?;
            call_meth(w_list, "append", &[value])?;
        }
        x if x == op::APPENDS => {
            let items = pop_mark(slot)?;
            let w_list = top(slot, "APPENDS")?;
            call_meth(w_list, "extend", &[items])?;
        }
        // ── dict ──────────────────────────────────────────────────────
        x if x == op::EMPTY_DICT => push(slot, pyre_object::dictmultiobject::w_dict_new()),
        x if x == op::DICT => {
            let items = pop_mark(slot)?;
            let w_dict = pyre_object::dictmultiobject::w_dict_new();
            dict_update_from_pairs(w_dict, items)?;
            push(slot, w_dict);
        }
        x if x == op::SETITEM => {
            let value = pop(slot)?;
            let key = pop(slot)?;
            let w_dict = top(slot, "SETITEM")?;
            crate::baseobjspace::setitem(w_dict, key, value)?;
        }
        x if x == op::SETITEMS => {
            let items = pop_mark(slot)?;
            let w_dict = top(slot, "SETITEMS")?;
            dict_update_from_pairs(w_dict, items)?;
        }
        // ── set / frozenset ───────────────────────────────────────────
        x if x == op::EMPTY_SET => push(slot, pyre_object::setobject::w_set_new()),
        x if x == op::FROZENSET => {
            let items = pop_mark(slot)?;
            push(slot, list_to_frozenset(items));
        }
        x if x == op::ADDITEMS => {
            let items = pop_mark(slot)?;
            let w_set = top(slot, "ADDITEMS")?;
            let n = unsafe { pyre_object::listobject::w_list_len(items) };
            for i in 0..n {
                let item =
                    unsafe { pyre_object::listobject::w_list_getitem(items, i as i64).unwrap() };
                unsafe { pyre_object::setobject::w_set_add(w_set, item) };
            }
        }
        // ── bytearray ─────────────────────────────────────────────────
        x if x == op::BYTEARRAY8 => {
            let nb = read(slot, 8)?;
            let n = read_int_le(&nb) as usize;
            let d = read(slot, n)?;
            push(
                slot,
                pyre_object::bytearrayobject::w_bytearray_from_bytes(&d),
            );
        }
        // ── proto-5 out-of-band buffers ───────────────────────────────
        x if x == op::NEXT_BUFFER => load_next_buffer(slot)?,
        x if x == op::READONLY_BUFFER => load_readonly_buffer(slot)?,
        // ── global / reduce / build ───────────────────────────────────
        x if x == op::GLOBAL => {
            let module = read_line(slot)?;
            let name = read_line(slot)?;
            let proto = cur(slot).proto;
            let fix_imports = cur(slot).fix_imports;
            push(slot, find_class(&module, &name, proto, fix_imports)?);
        }
        x if x == op::STACK_GLOBAL => {
            let w_name = pop(slot)?;
            let w_module = pop(slot)?;
            if !unsafe { pyre_object::is_str(w_name) } || !unsafe { pyre_object::is_str(w_module) }
            {
                return Err(unpickling_error("STACK_GLOBAL requires str"));
            }
            let name = unsafe { pyre_object::strobject::w_str_get_value(w_name) }.to_string();
            let module = unsafe { pyre_object::strobject::w_str_get_value(w_module) }.to_string();
            let proto = cur(slot).proto;
            let fix_imports = cur(slot).fix_imports;
            push(slot, find_class(&module, &name, proto, fix_imports)?);
        }
        x if x == op::REDUCE => {
            let w_args = pop(slot)?;
            let w_func = pop(slot)?;
            let args = tuple_items(w_args);
            let w_obj = call_fn(w_func, &args)?;
            push(slot, w_obj);
        }
        x if x == op::NEWOBJ => {
            let w_args = pop(slot)?;
            let w_cls = pop(slot)?;
            let w_obj = new_instance(w_cls, &tuple_items(w_args))?;
            push(slot, w_obj);
        }
        x if x == op::NEWOBJ_EX => {
            let w_kwargs = pop(slot)?;
            let w_args = pop(slot)?;
            let w_cls = pop(slot)?;
            let kw_items = unsafe { pyre_object::dictmultiobject::w_dict_items(w_kwargs) };
            let args = tuple_items(w_args);
            let w_obj = if kw_items.is_empty() {
                new_instance(w_cls, &args)?
            } else {
                new_instance_kw(w_cls, &args, &kw_items)?
            };
            push(slot, w_obj);
        }
        x if x == op::BUILD => {
            let w_state = pop(slot)?;
            let w_inst = top(slot, "BUILD")?;
            build_instance(w_inst, w_state)?;
        }
        // ── memo ──────────────────────────────────────────────────────
        x if x == op::MEMOIZE => {
            let v = top(slot, "MEMOIZE")?;
            memo_append(slot, v);
        }
        x if x == op::PUT => {
            let i = read_line_int(slot)?;
            if i < 0 {
                return Err(PyError::value_error("negative PUT argument"));
            }
            let v = top(slot, "PUT")?;
            memo_put(slot, i, v);
        }
        x if x == op::BINPUT => {
            let i = read(slot, 1)?[0] as i64;
            let v = top(slot, "BINPUT")?;
            memo_put(slot, i, v);
        }
        x if x == op::LONG_BINPUT => {
            let d = read(slot, 4)?;
            let i = u32::from_le_bytes([d[0], d[1], d[2], d[3]]) as i64;
            let v = top(slot, "LONG_BINPUT")?;
            memo_put(slot, i, v);
        }
        x if x == op::GET => {
            let i = read_line_int(slot)?;
            let v = memo_get(slot, i)?;
            push(slot, v);
        }
        x if x == op::BINGET => {
            let i = read(slot, 1)?[0] as i64;
            let v = memo_get(slot, i)?;
            push(slot, v);
        }
        x if x == op::LONG_BINGET => {
            let d = read(slot, 4)?;
            let i = u32::from_le_bytes([d[0], d[1], d[2], d[3]]) as i64;
            let v = memo_get(slot, i)?;
            push(slot, v);
        }
        x if x == op::PERSID => {
            let pid = read_line_bytes(slot)?;
            if !pid.is_ascii() {
                return Err(unpickling_error(
                    "persistent IDs in protocol 0 must be ASCII strings",
                ));
            }
            let w_pid = str_from_utf8(&pid)?;
            let v = persistent_load(slot, w_pid)?;
            push(slot, v);
        }
        x if x == op::BINPERSID => {
            let w_pid = pop(slot)?;
            let v = persistent_load(slot, w_pid)?;
            push(slot, v);
        }
        x if x == op::INST => {
            let module = read_line(slot)?;
            let name = read_line(slot)?;
            let w_cls = find_class(&module, &name, cur(slot).proto, cur(slot).fix_imports)?;
            let w_args = pop_mark(slot)?;
            let v = instantiate(w_cls, w_args)?;
            push(slot, v);
        }
        x if x == op::OBJ => {
            let args = pop_mark(slot)?;
            let n = unsafe { pyre_object::listobject::w_list_len(args) };
            if n == 0 {
                return Err(unpickling_error("OBJ opcode with empty stack"));
            }
            let w_cls = unsafe { pyre_object::listobject::w_list_getitem(args, 0).unwrap() };
            let rest: Vec<PyObjectRef> = (1..n)
                .map(|i| unsafe {
                    pyre_object::listobject::w_list_getitem(args, i as i64).unwrap()
                })
                .collect();
            let v = instantiate(w_cls, pyre_object::listobject::w_list_new(rest))?;
            push(slot, v);
        }
        _ => {
            return Err(unpickling_error("unsupported opcode in this build"));
        }
    }
    Ok(())
}

/// Build a tuple from the items of a (popped) stack list.
fn list_to_tuple(items: PyObjectRef) -> PyObjectRef {
    let n = unsafe { pyre_object::listobject::w_list_len(items) };
    let v: Vec<PyObjectRef> = (0..n)
        .map(|i| unsafe { pyre_object::listobject::w_list_getitem(items, i as i64).unwrap() })
        .collect();
    pyre_object::tupleobject::w_tuple_new(v)
}

/// Build a frozenset from the items of a (popped) stack list.
fn list_to_frozenset(items: PyObjectRef) -> PyObjectRef {
    let n = unsafe { pyre_object::listobject::w_list_len(items) };
    let v: Vec<PyObjectRef> = (0..n)
        .map(|i| unsafe { pyre_object::listobject::w_list_getitem(items, i as i64).unwrap() })
        .collect();
    pyre_object::setobject::w_frozenset_from_items(&v)
}

/// Copy a (popped) stack list into a fresh list.
fn list_copy(items: PyObjectRef) -> PyObjectRef {
    let n = unsafe { pyre_object::listobject::w_list_len(items) };
    let v: Vec<PyObjectRef> = (0..n)
        .map(|i| unsafe { pyre_object::listobject::w_list_getitem(items, i as i64).unwrap() })
        .collect();
    pyre_object::listobject::w_list_new(v)
}

/// Set `dict[items[2k]] = items[2k+1]` for each pair in a (popped) stack list.
fn dict_update_from_pairs(w_dict: PyObjectRef, items: PyObjectRef) -> Result<(), PyError> {
    let n = unsafe { pyre_object::listobject::w_list_len(items) };
    if n % 2 != 0 {
        return Err(unpickling_error("odd number of items for DICT"));
    }
    let mut i = 0;
    while i < n {
        let k = unsafe { pyre_object::listobject::w_list_getitem(items, i as i64).unwrap() };
        let v = unsafe { pyre_object::listobject::w_list_getitem(items, (i + 1) as i64).unwrap() };
        crate::baseobjspace::setitem(w_dict, k, v)?;
        i += 2;
    }
    Ok(())
}

/// Read a newline-terminated line (without the trailing newline).
fn read_line_bytes(slot: usize) -> Result<Vec<u8>, PyError> {
    let mut bytes: Vec<u8> = Vec::new();
    loop {
        let b = read1(slot)?;
        if b == b'\n' {
            break;
        }
        bytes.push(b);
    }
    Ok(bytes)
}

fn read_line(slot: usize) -> Result<String, PyError> {
    let bytes = read_line_bytes(slot)?;
    String::from_utf8(bytes).map_err(|_| unpickling_error("invalid utf-8 in pickle line"))
}

/// Read a newline-terminated decimal integer argument (GET / PUT in the
/// text protocols).
fn read_line_int(slot: usize) -> Result<i64, PyError> {
    let s = read_line(slot)?;
    s.trim()
        .parse::<i64>()
        .map_err(|_| PyError::value_error("invalid int literal"))
}

/// `find_class` — import `module_name` and resolve `name` against it.
/// Builtin names resolve through the execution context's `lookup_builtin`
/// (the `LOAD_GLOBAL` path); the module-object `getattr` does not see
/// builtins installed on the underlying storage. Other non-dotted names
/// resolve through the module's `__dict__` (dict subscript), and dotted
/// (protocol >= 4 nested) names fall back to the attribute walk.
fn find_class(
    module_name: &str,
    name: &str,
    proto: i64,
    fix_imports: bool,
) -> Result<PyObjectRef, PyError> {
    // protocol < 3 with `fix_imports` applies the py2 → py3 `_compat_pickle`
    // forward map before resolution; otherwise the name is resolved literally.
    let (module_name, name) = if proto < 3 && fix_imports {
        crate::module::_pickle::compat_map(module_name, name, false)
    } else {
        (module_name.to_string(), name.to_string())
    };
    let module_name = module_name.as_str();
    let name = name.as_str();
    if module_name == "builtins" && !name.contains('.') {
        if let Some(obj) = crate::module::_pickle::lookup_builtin(name) {
            return Ok(obj);
        }
    }
    let module = import_module(module_name)?;
    if name.contains('.') {
        return Ok(getattribute_dotted(module, name)?.0);
    }
    let w_dict = crate::baseobjspace::getattr_str(module, "__dict__")?;
    crate::baseobjspace::getitem(w_dict, pyre_object::w_str_new(name))
}

/// Resolve and invoke `self.persistent_load(pid)` (PERSID / BINPERSID).
fn persistent_load(slot: usize, w_pid: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let self_obj = pyre_object::gc_roots::shadow_stack_get(slot);
    match crate::baseobjspace::findattr(self_obj, "persistent_load") {
        Some(f) if !unsafe { pyre_object::is_none(f) } => call_fn(f, &[w_pid]),
        _ => Err(unpickling_error("unsupported persistent id encountered")),
    }
}

/// `_instantiate` — build an old-style INST / OBJ instance. With args, or a
/// non-type class, or a `__getinitargs__`, call the class; otherwise build
/// via `__new__` without invoking `__init__`.
fn instantiate(w_cls: PyObjectRef, w_args: PyObjectRef) -> Result<PyObjectRef, PyError> {
    let n = unsafe { pyre_object::listobject::w_list_len(w_args) };
    let has_getinitargs = crate::baseobjspace::findattr(w_cls, "__getinitargs__").is_some();
    let is_type = unsafe { pyre_object::typeobject::is_type(w_cls) };
    if n > 0 || !is_type || has_getinitargs {
        let args: Vec<PyObjectRef> = (0..n)
            .map(|i| unsafe { pyre_object::listobject::w_list_getitem(w_args, i as i64).unwrap() })
            .collect();
        call_fn(w_cls, &args)
    } else {
        new_instance(w_cls, &[])
    }
}

/// `cls.__new__(cls, *args)`.
fn new_instance(w_cls: PyObjectRef, args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let w_new = crate::baseobjspace::getattr_str(w_cls, "__new__")?;
    let mut call_args = vec![w_cls];
    call_args.extend_from_slice(args);
    call_fn(w_new, &call_args)
}

/// `cls.__new__(cls, *args, **kwargs)` — NEWOBJ_EX with the keyword
/// arguments returned by `__getnewargs_ex__`. Keyword delivery to a
/// user `__new__` needs the frame-based call path (`call_with_kwargs`);
/// the flat-slice path binds every argument positionally.
fn new_instance_kw(
    w_cls: PyObjectRef,
    args: &[PyObjectRef],
    kw_items: &[(PyObjectRef, PyObjectRef)],
) -> Result<PyObjectRef, PyError> {
    let w_new = crate::baseobjspace::getattr_str(w_cls, "__new__")?;
    let mut call_args = Vec::with_capacity(1 + args.len());
    call_args.push(w_cls);
    call_args.extend_from_slice(args);
    let mut kwargs = Vec::with_capacity(kw_items.len());
    for &(k, v) in kw_items {
        if !unsafe { pyre_object::is_str(k) } {
            return Err(unpickling_error("keyword arguments must be strings"));
        }
        let name = unsafe { pyre_object::strobject::w_str_get_wtf8(k) }.to_owned();
        kwargs.push((name, v));
    }
    let ec = crate::call::getexecutioncontext();
    if ec.is_null() {
        return Err(unpickling_error("no execution context for NEWOBJ_EX"));
    }
    let frame = unsafe { (*ec).gettopframe() };
    if frame.is_null() {
        return Err(unpickling_error("no frame for NEWOBJ_EX with kwargs"));
    }
    crate::call::call_with_kwargs(unsafe { &mut *frame }, w_new, &call_args, &kwargs)
}

/// `load_build` — apply pickled state to a freshly created instance.
fn build_instance(w_inst: PyObjectRef, w_state: PyObjectRef) -> Result<(), PyError> {
    // __setstate__ takes precedence.
    if let Some(setstate) = crate::baseobjspace::findattr(w_inst, "__setstate__") {
        if !unsafe { pyre_object::is_none(setstate) } {
            call_fn(setstate, &[w_state])?;
            return Ok(());
        }
    }

    // state may be a (dict-state, slot-state) pair.
    let (w_dict_state, w_slot_state) = if unsafe { pyre_object::is_tuple(w_state) }
        && unsafe { pyre_object::tupleobject::w_tuple_len(w_state) } == 2
    {
        (
            unsafe { pyre_object::tupleobject::w_tuple_getitem(w_state, 0).unwrap() },
            unsafe { pyre_object::tupleobject::w_tuple_getitem(w_state, 1).unwrap() },
        )
    } else {
        (w_state, pyre_object::w_none())
    };

    if !unsafe { pyre_object::is_none(w_dict_state) } {
        let w_inst_dict = crate::baseobjspace::getattr_str(w_inst, "__dict__")?;
        call_meth(w_inst_dict, "update", &[w_dict_state])?;
    }
    if !unsafe { pyre_object::is_none(w_slot_state) } {
        for (k, v) in unsafe { pyre_object::dictmultiobject::w_dict_items(w_slot_state) } {
            crate::baseobjspace::setattr(w_inst, k, v)?;
        }
    }
    Ok(())
}

fn tuple_items(w_tuple: PyObjectRef) -> Vec<PyObjectRef> {
    let n = unsafe { pyre_object::tupleobject::w_tuple_len(w_tuple) };
    (0..n)
        .map(|i| unsafe { pyre_object::tupleobject::w_tuple_getitem(w_tuple, i as i64).unwrap() })
        .collect()
}
