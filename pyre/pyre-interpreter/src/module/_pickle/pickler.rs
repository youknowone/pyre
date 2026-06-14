//! `_pickle.Pickler` — `interp_pickle.py W_Pickler` (atom + container subset).

use std::collections::HashMap;

use malachite_bigint::BigInt;
use pyre_object::PyObjectRef;

use crate::PyError;

use super::{
    BATCHSIZE, DEFAULT_PROTOCOL, FRAME_SIZE_MIN, HIGHEST_PROTOCOL, call_meth, encode_long, op,
};

#[crate::pyre_class("_pickle.Pickler")]
pub struct W_Pickler {
    /// Output file (has a `write` method).
    w_file: PyObjectRef,
    proto: i64,
    bin: bool,
    framing: bool,
}

/// Per-`dump` pickling context. The identity memo maps an already-saved
/// object (by address — pyre `id()` is address-based and interpreter
/// objects never move) to its memo index; `memo_size` is the next index
/// and the put-opcode counter.
struct PickleCtx {
    proto: i64,
    bin: bool,
    memo: HashMap<usize, usize>,
    memo_size: usize,
}

impl PickleCtx {
    fn memo_get(&self, w_obj: PyObjectRef) -> Option<usize> {
        self.memo.get(&(w_obj as usize)).copied()
    }
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
            memo: HashMap::new(),
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

/// `interp_pickle.py W_Pickler.save`. Exact-type dispatch via the `is_*`
/// predicates (bool is checked before int because a bool is not an int
/// here, and `is_int_or_long` also covers big integers). Atoms are never
/// memoized; everything else is checked against the identity memo for a
/// GET back-reference before being saved.
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

    // Identity memo — a repeated reference becomes a GET back-reference.
    if let Some(idx) = ctx.memo_get(w_obj) {
        write_get(ctx, buf, idx);
        return Ok(());
    }

    if unsafe { pyre_object::is_bytes(w_obj) } {
        save_bytes(ctx, buf, w_obj);
        return Ok(());
    }
    if unsafe { pyre_object::is_str(w_obj) } {
        save_str(ctx, buf, w_obj);
        return Ok(());
    }
    if unsafe { pyre_object::is_dict(w_obj) } {
        return save_dict(ctx, buf, w_obj);
    }
    if unsafe { pyre_object::is_list(w_obj) } {
        return save_list(ctx, buf, w_obj);
    }
    if unsafe { pyre_object::is_tuple(w_obj) } {
        return save_tuple(ctx, buf, w_obj);
    }
    Err(PyError::not_implemented(
        "_pickle: this object type is not supported in this build",
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

fn save_bytes(ctx: &mut PickleCtx, buf: &mut Vec<u8>, w_obj: PyObjectRef) {
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
    memoize(ctx, buf, w_obj);
}

fn save_str(ctx: &mut PickleCtx, buf: &mut Vec<u8>, w_obj: PyObjectRef) {
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
    memoize(ctx, buf, w_obj);
}

/// `interp_pickle.py save_tuple`.
fn save_tuple(ctx: &mut PickleCtx, buf: &mut Vec<u8>, w_obj: PyObjectRef) -> Result<(), PyError> {
    let n = unsafe { pyre_object::tupleobject::w_tuple_len(w_obj) };
    if n == 0 {
        if ctx.bin {
            buf.push(op::EMPTY_TUPLE);
        } else {
            buf.push(op::MARK);
            buf.push(op::TUPLE);
        }
        return Ok(());
    }

    // Snapshot the elements before recursing (interpreter objects do not
    // move, so the captured references stay valid across the saves below).
    let elems: Vec<PyObjectRef> = (0..n)
        .map(|i| unsafe { pyre_object::tupleobject::w_tuple_getitem(w_obj, i as i64).unwrap() })
        .collect();

    if n <= 3 && ctx.proto >= 2 {
        for &e in &elems {
            save(ctx, buf, e)?;
        }
        // Subtle: saving the elements may have memoized this very tuple
        // (a recursive tuple). If so, discard the elements and GET it.
        if let Some(idx) = ctx.memo_get(w_obj) {
            for _ in 0..n {
                buf.push(op::POP);
            }
            write_get(ctx, buf, idx);
        } else {
            buf.push(op::TUPLESIZE2CODE[n]);
            memoize(ctx, buf, w_obj);
        }
        return Ok(());
    }

    buf.push(op::MARK);
    for &e in &elems {
        save(ctx, buf, e)?;
    }
    if let Some(idx) = ctx.memo_get(w_obj) {
        // Recursive tuple: throw away the stack contents and GET it.
        if ctx.bin {
            buf.push(op::POP_MARK);
        } else {
            for _ in 0..(n + 1) {
                buf.push(op::POP);
            }
        }
        write_get(ctx, buf, idx);
        return Ok(());
    }
    buf.push(op::TUPLE);
    memoize(ctx, buf, w_obj);
    Ok(())
}

/// `interp_pickle.py save_list`. The PyPy ascii/bytes-list fast paths are
/// gated on `pypy_extensions` (off here) so the generic path is used,
/// matching CPython's wire format.
fn save_list(ctx: &mut PickleCtx, buf: &mut Vec<u8>, w_obj: PyObjectRef) -> Result<(), PyError> {
    if ctx.bin {
        buf.push(op::EMPTY_LIST);
    } else {
        buf.push(op::MARK);
        buf.push(op::LIST);
    }
    memoize(ctx, buf, w_obj);

    let n = unsafe { pyre_object::listobject::w_list_len(w_obj) };
    let items: Vec<PyObjectRef> = (0..n)
        .map(|i| unsafe { pyre_object::listobject::w_list_getitem(w_obj, i as i64).unwrap() })
        .collect();
    batch_appends(ctx, buf, &items)
}

/// `interp_pickle.py save_dict`.
fn save_dict(ctx: &mut PickleCtx, buf: &mut Vec<u8>, w_obj: PyObjectRef) -> Result<(), PyError> {
    if ctx.bin {
        buf.push(op::EMPTY_DICT);
    } else {
        buf.push(op::MARK);
        buf.push(op::DICT);
    }
    memoize(ctx, buf, w_obj);

    let items = unsafe { pyre_object::dictmultiobject::w_dict_items(w_obj) };
    batch_setitems(ctx, buf, &items)
}

/// `interp_pickle.py _batch_appends` (generic bin path). Single item →
/// APPEND; otherwise MARK … APPENDS in batches of `BATCHSIZE`.
fn batch_appends(
    ctx: &mut PickleCtx,
    buf: &mut Vec<u8>,
    items: &[PyObjectRef],
) -> Result<(), PyError> {
    let n = items.len();
    let mut i = 0;
    while i < n {
        if i + 1 == n {
            // Exactly one item left.
            save(ctx, buf, items[i])?;
            buf.push(op::APPEND);
            return Ok(());
        }
        buf.push(op::MARK);
        let mut cnt = 0;
        while i < n && cnt < BATCHSIZE {
            save(ctx, buf, items[i])?;
            i += 1;
            cnt += 1;
        }
        buf.push(op::APPENDS);
    }
    Ok(())
}

/// `interp_pickle.py _batch_setitems` (bin path). Single pair → SETITEM;
/// otherwise MARK … SETITEMS in batches of `BATCHSIZE`.
fn batch_setitems(
    ctx: &mut PickleCtx,
    buf: &mut Vec<u8>,
    items: &[(PyObjectRef, PyObjectRef)],
) -> Result<(), PyError> {
    let n = items.len();
    let mut i = 0;
    while i < n {
        if i + 1 == n {
            // Exactly one pair left.
            save(ctx, buf, items[i].0)?;
            save(ctx, buf, items[i].1)?;
            buf.push(op::SETITEM);
            return Ok(());
        }
        buf.push(op::MARK);
        let mut cnt = 0;
        while i < n && cnt < BATCHSIZE {
            save(ctx, buf, items[i].0)?;
            save(ctx, buf, items[i].1)?;
            i += 1;
            cnt += 1;
        }
        buf.push(op::SETITEMS);
    }
    Ok(())
}

/// `interp_pickle.py W_Pickler.write_get` — emit a GET back-reference.
fn write_get(ctx: &PickleCtx, buf: &mut Vec<u8>, idx: usize) {
    if ctx.bin {
        if idx < 256 {
            buf.push(op::BINGET);
            buf.push(idx as u8);
        } else {
            buf.push(op::LONG_BINGET);
            buf.extend_from_slice(&(idx as u32).to_le_bytes());
        }
    } else {
        buf.push(op::GET);
        buf.extend_from_slice(format!("{idx}\n").as_bytes());
    }
}

/// `interp_pickle.py memoize` — record the object's identity and write the
/// put opcode.
fn memoize(ctx: &mut PickleCtx, buf: &mut Vec<u8>, w_obj: PyObjectRef) {
    let idx = ctx.memo_size;
    ctx.memo_size += 1;
    ctx.memo.insert(w_obj as usize, idx);
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
