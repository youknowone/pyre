//! `_pickle.Unpickler` — `interp_pickle.py W_Unpickler` (atom subset).

use pyre_object::PyObjectRef;

use crate::PyError;

use super::{HIGHEST_PROTOCOL, call_fn, decode_long, op, read_int_le, str_from_utf8, unpickling_error};

#[crate::pyre_class("_pickle.Unpickler")]
pub struct W_Unpickler {
    w_file_read: PyObjectRef,
    w_file_readline: PyObjectRef,
    /// Result stack — a Python `list` (GC-managed across `read` allocs).
    w_stack: PyObjectRef,
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
            w_frame: pyre_object::w_none(),
            frame_index: 0,
            proto: 0,
        })
    }

    fn __init__(&mut self, w_file: PyObjectRef) -> Result<(), PyError> {
        self.w_file_read = crate::baseobjspace::getattr_str(w_file, "read")?;
        self.w_file_readline = crate::baseobjspace::getattr_str(w_file, "readline")?;
        self.w_stack = pyre_object::w_none();
        self.w_frame = pyre_object::w_none();
        self.frame_index = 0;
        self.proto = 0;
        Ok(())
    }

    fn load(&mut self) -> Result<PyObjectRef, PyError> {
        // Fresh stack each load.
        self.w_stack = pyre_object::listobject::w_list_new(Vec::new());
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
                    .ok_or_else(|| unpickling_error("unexpected MARK found"));
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

fn push(slot: usize, obj: PyObjectRef) {
    let me = cur(slot);
    unsafe { pyre_object::listobject::w_list_append(me.w_stack, obj) };
}

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
        // Memo put opcodes: consume the argument; dedup arrives in inc 2.
        x if x == op::MEMOIZE => {}
        x if x == op::BINPUT => {
            read(slot, 1)?;
        }
        x if x == op::LONG_BINPUT => {
            read(slot, 4)?;
        }
        _ => {
            return Err(unpickling_error("unsupported opcode in this build"));
        }
    }
    Ok(())
}
