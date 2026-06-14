//! `_pickle.Pickler` — `interp_pickle.py W_Pickler` (atom subset).

use malachite_bigint::BigInt;
use pyre_object::PyObjectRef;

use crate::PyError;

use super::{
    DEFAULT_PROTOCOL, FRAME_SIZE_MIN, HIGHEST_PROTOCOL, call_meth, encode_long, op,
};

#[crate::pyre_class("_pickle.Pickler")]
pub struct W_Pickler {
    /// Output file (has a `write` method).
    w_file: PyObjectRef,
    proto: i64,
    bin: bool,
    framing: bool,
}

/// Per-`dump` pickling context. `memo_size` drives the MEMOIZE counter;
/// value dedup (GET back-references) arrives in increment 2.
struct PickleCtx {
    proto: i64,
    bin: bool,
    memo_size: usize,
}

#[crate::pyre_methods(doc = "Pickler(file, protocol=None) -> pickler writing to file.")]
impl W_Pickler {
    #[staticmethod]
    fn __new__(_cls: PyObjectRef) -> PyObjectRef {
        W_Pickler::allocate(W_Pickler {
            ob: pyre_object::PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            w_file: pyre_object::w_none(),
            proto: 0,
            bin: false,
            framing: false,
        })
    }

    fn __init__(
        &mut self,
        w_file: PyObjectRef,
        #[default(pyre_object::w_none())] w_protocol: PyObjectRef,
    ) -> Result<(), PyError> {
        let proto = if unsafe { pyre_object::is_none(w_protocol) } {
            DEFAULT_PROTOCOL
        } else {
            let p = crate::baseobjspace::int_w(w_protocol)?;
            if p < 0 {
                HIGHEST_PROTOCOL
            } else if p > HIGHEST_PROTOCOL {
                return Err(PyError::value_error("pickle protocol must be <= 5"));
            } else {
                p
            }
        };
        // `file must have a 'write' attribute` (interp_pickle.py:557).
        if crate::baseobjspace::findattr(w_file, "write").is_none() {
            return Err(PyError::type_error("file must have a 'write' attribute"));
        }
        self.w_file = w_file;
        self.proto = proto;
        self.bin = proto >= 1;
        self.framing = proto >= 4;
        Ok(())
    }

    fn dump(&mut self, w_obj: PyObjectRef) -> Result<(), PyError> {
        // Read every field before any allocation can relocate `self`.
        let proto = self.proto;
        let bin = self.bin;
        let framing = self.framing;
        let w_file = self.w_file;

        let _roots = pyre_object::gc_roots::push_roots();
        pyre_object::gc_roots::pin_root(w_file);
        let file_slot = pyre_object::gc_roots::shadow_stack_len() - 1;
        pyre_object::gc_roots::pin_root(w_obj);

        let mut ctx = PickleCtx {
            proto,
            bin,
            memo_size: 0,
        };
        let mut body: Vec<u8> = Vec::new();
        save(&mut ctx, &mut body, w_obj)?;
        body.push(op::STOP);

        let mut out: Vec<u8> = Vec::new();
        if proto >= 2 {
            out.push(op::PROTO);
            out.push(proto as u8);
        }
        if framing && body.len() >= FRAME_SIZE_MIN {
            out.push(op::FRAME);
            out.extend_from_slice(&(body.len() as u64).to_le_bytes());
        }
        out.extend_from_slice(&body);

        let w_bytes = pyre_object::w_bytes_from_bytes(&out);
        // `w_file` may have moved during the build; re-read from the pin.
        let w_file = pyre_object::gc_roots::shadow_stack_get(file_slot);
        call_meth(w_file, "write", &[w_bytes])?;
        Ok(())
    }
}

/// `interp_pickle.py W_Pickler.save` — atom subset. Exact-type dispatch
/// via the `is_*` predicates (bool is checked before int because a bool
/// is not an int here, and `is_int_or_long` also covers big integers).
fn save(ctx: &mut PickleCtx, buf: &mut Vec<u8>, w_obj: PyObjectRef) -> Result<(), PyError> {
    // Atoms — never memoized.
    if unsafe { pyre_object::is_none(w_obj) } {
        buf.push(op::NONE);
        return Ok(());
    }
    if unsafe { pyre_object::is_bool(w_obj) } {
        save_bool(ctx, buf, w_obj);
        return Ok(());
    }
    if unsafe { pyre_object::is_int_or_long(w_obj) } {
        save_long(ctx, buf, w_obj)?;
        return Ok(());
    }
    if unsafe { pyre_object::is_float(w_obj) } {
        save_float(ctx, buf, w_obj)?;
        return Ok(());
    }
    // Memoized atoms — write the value, then a put opcode.
    if unsafe { pyre_object::is_bytes(w_obj) } {
        save_bytes(ctx, buf, w_obj);
        memoize(ctx, buf);
        return Ok(());
    }
    if unsafe { pyre_object::is_str(w_obj) } {
        save_str(ctx, buf, w_obj);
        memoize(ctx, buf);
        return Ok(());
    }
    Err(PyError::not_implemented(
        "_pickle: only atoms are supported in this build",
    ))
}

fn save_bool(ctx: &PickleCtx, buf: &mut Vec<u8>, w_obj: PyObjectRef) {
    let truthy = crate::baseobjspace::is_true(w_obj);
    if ctx.proto >= 2 {
        buf.push(if truthy { op::NEWTRUE } else { op::NEWFALSE });
    } else {
        // I00\n / I01\n
        buf.extend_from_slice(if truthy { b"I01\n" } else { b"I00\n" });
    }
}

fn save_long(ctx: &PickleCtx, buf: &mut Vec<u8>, w_obj: PyObjectRef) -> Result<(), PyError> {
    match crate::baseobjspace::int_w(w_obj) {
        Ok(v) => {
            if ctx.bin {
                if v >= 0 {
                    if v <= 0xff {
                        buf.push(op::BININT1);
                        buf.push(v as u8);
                        return Ok(());
                    }
                    if v <= 0xffff {
                        buf.push(op::BININT2);
                        buf.extend_from_slice(&(v as u16).to_le_bytes());
                        return Ok(());
                    }
                }
                if (-0x8000_0000..=0x7fff_ffff).contains(&v) {
                    buf.push(op::BININT);
                    buf.extend_from_slice(&(v as i32).to_le_bytes());
                    return Ok(());
                }
            }
            write_long(buf, &encode_long(&BigInt::from(v)));
            Ok(())
        }
        Err(_) => {
            // Beyond i64 — a true big integer.
            let big = unsafe { crate::builtins::obj_to_bigint(w_obj) };
            write_long(buf, &encode_long(&big));
            Ok(())
        }
    }
}

fn write_long(buf: &mut Vec<u8>, enc: &[u8]) {
    let n = enc.len();
    if n < 256 {
        buf.push(op::LONG1);
        buf.push(n as u8);
    } else {
        buf.push(op::LONG4);
        buf.extend_from_slice(&(n as i32).to_le_bytes());
    }
    buf.extend_from_slice(enc);
}

fn save_float(_ctx: &PickleCtx, buf: &mut Vec<u8>, w_obj: PyObjectRef) -> Result<(), PyError> {
    let f = crate::baseobjspace::float_w(w_obj)?;
    // BINFLOAT — 8-byte big-endian IEEE 754.
    buf.push(op::BINFLOAT);
    buf.extend_from_slice(&f.to_be_bytes());
    Ok(())
}

fn save_bytes(ctx: &PickleCtx, buf: &mut Vec<u8>, w_obj: PyObjectRef) {
    // proto < 3 (interp_pickle.py:1349) emits a `codecs.encode(latin1)` /
    // `bytes()` reduce instead of a BINBYTES opcode. That needs `save_reduce`
    // (increment 5); until then proto-2 bytes use the modern SHORT_BINBYTES
    // form — valid and round-trippable, but not byte-identical to CPython at
    // protocol 2.
    let data = unsafe { pyre_object::bytesobject::w_bytes_data(w_obj) };
    let n = data.len();
    if n <= 0xff {
        buf.push(op::SHORT_BINBYTES);
        buf.push(n as u8);
    } else if n > 0xffff_ffff && ctx.proto >= 4 {
        buf.push(op::BINBYTES8);
        buf.extend_from_slice(&(n as u64).to_le_bytes());
    } else {
        buf.push(op::BINBYTES);
        buf.extend_from_slice(&(n as u32).to_le_bytes());
    }
    buf.extend_from_slice(data);
}

fn save_str(ctx: &PickleCtx, buf: &mut Vec<u8>, w_obj: PyObjectRef) {
    let s = unsafe { pyre_object::strobject::w_str_get_value(w_obj) };
    let data = s.as_bytes();
    let n = data.len();
    if n <= 0xff && ctx.proto >= 4 {
        buf.push(op::SHORT_BINUNICODE);
        buf.push(n as u8);
    } else if n > 0xffff_ffff && ctx.proto >= 4 {
        buf.push(op::BINUNICODE8);
        buf.extend_from_slice(&(n as u64).to_le_bytes());
    } else {
        buf.push(op::BINUNICODE);
        buf.extend_from_slice(&(n as u32).to_le_bytes());
    }
    buf.extend_from_slice(data);
}

/// `interp_pickle.py memoize` — write the put opcode + bump the counter.
/// Increment 1 records nothing (no dedup); the put opcode is still emitted
/// for wire-format parity.
fn memoize(ctx: &mut PickleCtx, buf: &mut Vec<u8>) {
    let idx = ctx.memo_size;
    ctx.memo_size += 1;
    if ctx.proto >= 4 {
        buf.push(op::MEMOIZE);
    } else if ctx.bin {
        if idx < 256 {
            buf.push(op::BINPUT);
            buf.push(idx as u8);
        } else {
            buf.push(op::LONG_BINPUT);
            buf.extend_from_slice(&(idx as u32).to_le_bytes());
        }
    } else {
        buf.push(op::PUT);
        buf.extend_from_slice(format!("{idx}\n").as_bytes());
    }
}
