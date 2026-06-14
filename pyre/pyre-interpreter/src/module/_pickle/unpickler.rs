//! `_pickle.Unpickler` — `interp_pickle.py W_Unpickler` (atom + container subset).

use pyre_object::PyObjectRef;

use crate::PyError;

use super::{
    HIGHEST_PROTOCOL, call_fn, call_meth, decode_long, op, read_int_le, str_from_utf8,
    unpickling_error,
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
        })
    }

    fn __init__(&mut self, w_file: PyObjectRef) -> Result<(), PyError> {
        self.w_file_read = crate::baseobjspace::getattr_str(w_file, "read")?;
        self.w_file_readline = crate::baseobjspace::getattr_str(w_file, "readline")?;
        self.w_stack = pyre_object::w_none();
        self.w_metastack = pyre_object::w_none();
        self.w_memo = pyre_object::w_none();
        self.memo_index = 0;
        self.w_frame = pyre_object::w_none();
        self.frame_index = 0;
        self.proto = 0;
        Ok(())
    }

    fn load(&mut self) -> Result<PyObjectRef, PyError> {
        // Fresh stack / metastack / memo each load.
        self.w_stack = pyre_object::listobject::w_list_new(Vec::new());
        self.w_metastack = pyre_object::listobject::w_list_new(Vec::new());
        self.w_memo = pyre_object::dictmultiobject::w_dict_new();
        self.memo_index = 0;
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

/// Read a newline-terminated decimal integer argument (GET / PUT in the
/// text protocols).
fn read_line_int(slot: usize) -> Result<i64, PyError> {
    let mut digits: Vec<u8> = Vec::new();
    loop {
        let b = read1(slot)?;
        if b == b'\n' {
            break;
        }
        digits.push(b);
    }
    let s = std::str::from_utf8(&digits)
        .map_err(|_| PyError::value_error("invalid int literal"))?;
    s.trim()
        .parse::<i64>()
        .map_err(|_| PyError::value_error("invalid int literal"))
}
