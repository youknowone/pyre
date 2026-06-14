//! `_pickle.Pickler` — `interp_pickle.py W_Pickler` (atom + container subset).

use std::collections::HashMap;

use malachite_bigint::BigInt;
use pyre_object::PyObjectRef;

use crate::PyError;

use super::{
    BATCHSIZE, DEFAULT_PROTOCOL, FRAME_SIZE_MIN, HIGHEST_PROTOCOL, call_fn, call_meth, encode_long,
    getattribute_dotted, import_module, op, pickling_error,
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
        return save_bytes(ctx, buf, w_obj);
    }
    if unsafe { pyre_object::is_str(w_obj) } {
        save_str(ctx, buf, w_obj);
        return Ok(());
    }
    if unsafe { pyre_object::is_dict(w_obj) } {
        return save_dict(ctx, buf, w_obj);
    }
    if unsafe { pyre_object::is_set(w_obj) } {
        return save_set(ctx, buf, w_obj);
    }
    if unsafe { pyre_object::is_frozenset(w_obj) } {
        return save_frozenset(ctx, buf, w_obj);
    }
    if unsafe { pyre_object::is_list(w_obj) } {
        return save_list(ctx, buf, w_obj);
    }
    if unsafe { pyre_object::is_tuple(w_obj) } {
        return save_tuple(ctx, buf, w_obj);
    }
    if unsafe { pyre_object::is_bytearray(w_obj) } {
        return save_bytearray(ctx, buf, w_obj);
    }

    // Classes and functions are saved by reference.
    if unsafe { pyre_object::typeobject::is_type(w_obj) }
        || unsafe { crate::function::is_function(w_obj) }
    {
        return save_global(ctx, buf, w_obj, None);
    }

    // Everything else goes through the reduce protocol.
    let w_rv = match crate::baseobjspace::findattr(w_obj, "__reduce_ex__") {
        Some(reduce_ex) => call_fn(reduce_ex, &[pyre_object::w_int_new(ctx.proto)])?,
        None => match crate::baseobjspace::findattr(w_obj, "__reduce__") {
            Some(reduce) => call_fn(reduce, &[])?,
            None => return Err(pickling_error("Can't pickle object: no __reduce_ex__")),
        },
    };
    if unsafe { pyre_object::is_str(w_rv) } {
        return save_global(ctx, buf, w_obj, Some(w_rv));
    }
    if unsafe { pyre_object::is_tuple(w_rv) } {
        let n = unsafe { pyre_object::tupleobject::w_tuple_len(w_rv) };
        if !(2..=6).contains(&n) {
            return Err(pickling_error(
                "Tuple returned by __reduce__ must have two to six elements",
            ));
        }
        let rv: Vec<PyObjectRef> = (0..n)
            .map(|i| unsafe { pyre_object::tupleobject::w_tuple_getitem(w_rv, i as i64).unwrap() })
            .collect();
        return save_reduce(ctx, buf, &rv, Some(w_obj));
    }
    Err(pickling_error(
        "__reduce__ must return string or tuple",
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

fn save_bytes(ctx: &mut PickleCtx, buf: &mut Vec<u8>, w_obj: PyObjectRef) -> Result<(), PyError> {
    // proto < 3 emits a `codecs.encode(s, 'latin1')` / `bytes()` reduce
    // instead of a BINBYTES opcode (interp_pickle.py:1349).
    if ctx.proto < 3 {
        let data = unsafe { pyre_object::bytesobject::w_bytes_data(w_obj) };
        if data.is_empty() {
            let w_bytes = crate::typedef::gettypeobject(&pyre_object::bytesobject::BYTES_TYPE);
            let w_args = pyre_object::tupleobject::w_tuple_new(Vec::new());
            return save_reduce(ctx, buf, &[w_bytes, w_args], Some(w_obj));
        }
        let codecs = import_module("codecs")?;
        let w_encode = crate::baseobjspace::getattr_str(codecs, "encode")?;
        let w_decoded = call_meth(w_obj, "decode", &[pyre_object::w_str_new("latin1")])?;
        let w_args = pyre_object::tupleobject::w_tuple_new(vec![
            w_decoded,
            pyre_object::w_str_new("latin1"),
        ]);
        return save_reduce(ctx, buf, &[w_encode, w_args], Some(w_obj));
    }
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
    Ok(())
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

/// `interp_pickle.py save_set`. Sets are unordered, so the wire bytes are
/// not byte-identical to CPython, but the encoding round-trips. The
/// protocol < 4 reduce fallback arrives with `save_reduce`.
fn save_set(ctx: &mut PickleCtx, buf: &mut Vec<u8>, w_obj: PyObjectRef) -> Result<(), PyError> {
    if ctx.proto < 4 {
        // save_reduce(set, (list(obj),)).
        let items = unsafe { pyre_object::setobject::w_set_items(w_obj) };
        let w_list = pyre_object::listobject::w_list_new(items);
        let w_args = pyre_object::tupleobject::w_tuple_new(vec![w_list]);
        let w_set_type = crate::typedef::gettypeobject(&pyre_object::setobject::SET_TYPE);
        return save_reduce(ctx, buf, &[w_set_type, w_args], Some(w_obj));
    }
    buf.push(op::EMPTY_SET);
    memoize(ctx, buf, w_obj);

    let items = unsafe { pyre_object::setobject::w_set_items(w_obj) };
    let length = items.len();
    if length == 0 {
        return Ok(());
    }
    buf.push(op::MARK);
    save(ctx, buf, items[0])?;
    let mut i = 1;
    while i + 1 < length {
        if i % BATCHSIZE == 0 {
            buf.push(op::ADDITEMS);
            buf.push(op::MARK);
        }
        save(ctx, buf, items[i])?;
        i += 1;
    }
    if length > 1 {
        save(ctx, buf, items[length - 1])?;
    }
    buf.push(op::ADDITEMS);
    Ok(())
}

/// `interp_pickle.py save_frozenset`. Protocol < 4 reduces to
/// `frozenset(list(obj))`; protocol >= 4 uses the FROZENSET opcode.
/// Unordered, so not byte-identical to CPython.
fn save_frozenset(
    ctx: &mut PickleCtx,
    buf: &mut Vec<u8>,
    w_obj: PyObjectRef,
) -> Result<(), PyError> {
    if ctx.proto < 4 {
        // save_reduce(frozenset, (list(obj),)).
        let items = unsafe { pyre_object::setobject::w_set_items(w_obj) };
        let w_list = pyre_object::listobject::w_list_new(items);
        let w_args = pyre_object::tupleobject::w_tuple_new(vec![w_list]);
        let w_frozenset_type =
            crate::typedef::gettypeobject(&pyre_object::setobject::FROZENSET_TYPE);
        return save_reduce(ctx, buf, &[w_frozenset_type, w_args], Some(w_obj));
    }
    buf.push(op::MARK);
    let items = unsafe { pyre_object::setobject::w_set_items(w_obj) };
    for &e in &items {
        save(ctx, buf, e)?;
    }
    if let Some(idx) = ctx.memo_get(w_obj) {
        buf.push(op::POP);
        write_get(ctx, buf, idx);
    } else {
        buf.push(op::FROZENSET);
        memoize(ctx, buf, w_obj);
    }
    Ok(())
}

/// `interp_pickle.py save_bytearray` (proto >= 5 raw form; lower protocols
/// reach the generic reduce path).
fn save_bytearray(
    ctx: &mut PickleCtx,
    buf: &mut Vec<u8>,
    w_obj: PyObjectRef,
) -> Result<(), PyError> {
    if ctx.proto < 5 {
        // save_reduce(bytearray, ()) for empty, else save_reduce(bytearray, (bytes,)).
        let data = unsafe { pyre_object::bytearrayobject::w_bytearray_data(w_obj) };
        let w_bytearray_type =
            crate::typedef::gettypeobject(&pyre_object::bytearrayobject::BYTEARRAY_TYPE);
        let w_args = if data.is_empty() {
            pyre_object::tupleobject::w_tuple_new(Vec::new())
        } else {
            let w_bytes = pyre_object::w_bytes_from_bytes(data);
            pyre_object::tupleobject::w_tuple_new(vec![w_bytes])
        };
        return save_reduce(ctx, buf, &[w_bytearray_type, w_args], Some(w_obj));
    }
    let data = unsafe { pyre_object::bytearrayobject::w_bytearray_data(w_obj) };
    buf.push(op::BYTEARRAY8);
    buf.extend_from_slice(&(data.len() as u64).to_le_bytes());
    buf.extend_from_slice(data);
    memoize(ctx, buf, w_obj);
    Ok(())
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

// ── reduce / global ──────────────────────────────────────────────────

/// `interp_pickle.py whichmodule` — the module an object belongs to.
/// `__module__` takes precedence; otherwise scan `sys.modules` for the
/// module that exposes `name` resolving back to `w_obj`, skipping
/// `__main__` / `__mp_main__` / `None`, and default to `"__main__"`.
fn whichmodule(w_obj: PyObjectRef, name: &str) -> Result<PyObjectRef, PyError> {
    if let Some(m) = crate::baseobjspace::findattr(w_obj, "__module__") {
        if !unsafe { pyre_object::is_none(m) } {
            return Ok(m);
        }
    }
    let modules = crate::importing::sys_modules_dict();
    if !modules.is_null() {
        for (w_modname, w_module) in unsafe { pyre_object::dictmultiobject::w_dict_items(modules) } {
            if !unsafe { pyre_object::is_str(w_modname) } || unsafe { pyre_object::is_none(w_module) }
            {
                continue;
            }
            let modname = unsafe { pyre_object::strobject::w_str_get_value(w_modname) };
            if modname == "__main__" || modname == "__mp_main__" {
                continue;
            }
            if let Ok((resolved, _)) = getattribute_dotted(w_module, name) {
                if crate::baseobjspace::is_w(resolved, w_obj) {
                    return Ok(w_modname);
                }
            }
        }
    }
    Ok(pyre_object::w_str_new("__main__"))
}

/// `interp_pickle.py save_global` / `save_global2` — save an object by
/// qualified reference. `w_name_opt` carries the name when a `__reduce__`
/// returned a string; otherwise it is derived from `__qualname__`.
fn save_global(
    ctx: &mut PickleCtx,
    buf: &mut Vec<u8>,
    w_obj: PyObjectRef,
    w_name_opt: Option<PyObjectRef>,
) -> Result<(), PyError> {
    let w_name = match w_name_opt {
        Some(n) => n,
        None => crate::baseobjspace::findattr(w_obj, "__qualname__")
            .or_else(|| crate::baseobjspace::findattr(w_obj, "__name__"))
            .ok_or_else(|| pickling_error("Can't pickle object: no __qualname__ / __name__"))?,
    };
    let name = unsafe { pyre_object::strobject::w_str_get_value(w_name) }.to_string();
    let w_module_name = whichmodule(w_obj, &name)?;
    let module_name = unsafe { pyre_object::strobject::w_str_get_value(w_module_name) }.to_string();

    // The unpickler resolves `module_name.name` at load time via `find_class`.
    // CPython additionally verifies the name resolves back to this exact
    // object at dump time, but pyre's `getattr` on the `builtins` module is
    // unreliable here (it can return a non-canonical object and corrupt
    // builtin state), so that round-trip check is skipped. Nested-ness is
    // derived from the qualname instead of an attribute walk.
    let nested = name.contains('.');

    if ctx.proto >= 4 {
        save(ctx, buf, w_module_name)?;
        save(ctx, buf, w_name)?;
        buf.push(op::STACK_GLOBAL);
    } else if nested {
        // Nested object at protocol < 4: reduce to getattr(parent, lastname).
        let module = import_module(&module_name)?;
        let dot = name.rfind('.').unwrap();
        let (parent, _) = getattribute_dotted(module, &name[..dot])?;
        let lastname = &name[dot + 1..];
        let w_getattr = builtin_attr("getattr")?;
        let w_args =
            pyre_object::tupleobject::w_tuple_new(vec![parent, pyre_object::w_str_new(lastname)]);
        save_reduce(ctx, buf, &[w_getattr, w_args], None)?;
    } else {
        buf.push(op::GLOBAL);
        buf.extend_from_slice(module_name.as_bytes());
        buf.push(b'\n');
        buf.extend_from_slice(name.as_bytes());
        buf.push(b'\n');
    }
    memoize(ctx, buf, w_obj);
    Ok(())
}

/// `__builtins__.<name>` (e.g. `getattr`), via the execution context's
/// `lookup_builtin` (the `LOAD_GLOBAL` path). Only used on the rare nested
/// protocol < 4 path.
fn builtin_attr(name: &str) -> Result<PyObjectRef, PyError> {
    crate::module::_pickle::lookup_builtin(name)
        .ok_or_else(|| pickling_error(format!("Can't resolve builtin {name:?}")))
}

/// `interp_pickle.py save_reduce`. `rv` is the 2-to-6 element reduce tuple
/// `(func, args[, state[, listitems[, dictitems[, state_setter]]]])`.
fn save_reduce(
    ctx: &mut PickleCtx,
    buf: &mut Vec<u8>,
    rv: &[PyObjectRef],
    w_obj_opt: Option<PyObjectRef>,
) -> Result<(), PyError> {
    let w_func = rv[0];
    let w_args = rv[1];
    let nonnull = |i: usize| rv.get(i).copied().filter(|&x| !unsafe { pyre_object::is_none(x) });
    let w_state = nonnull(2);
    let w_listitems = nonnull(3);
    let w_dictitems = nonnull(4);
    let w_state_setter = nonnull(5);

    if !unsafe { pyre_object::is_tuple(w_args) } {
        return Err(pickling_error("args from save_reduce() must be a tuple"));
    }
    if !crate::baseobjspace::callable_w(w_func) {
        return Err(pickling_error("func from save_reduce() must be callable"));
    }

    let func_name = func_name_str(w_func);
    let args_w = tuple_items(w_args);

    if ctx.proto >= 2 && func_name.as_deref() == Some("__newobj_ex__") {
        if args_w.len() != 3 {
            return Err(pickling_error("__newobj_ex__ requires three args"));
        }
        let (w_cls, w_a, w_kw) = (args_w[0], args_w[1], args_w[2]);
        if crate::baseobjspace::findattr(w_cls, "__new__").is_none() {
            return Err(pickling_error("args[0] from __newobj_ex__ args has no __new__"));
        }
        if ctx.proto >= 4 {
            save(ctx, buf, w_cls)?;
            save(ctx, buf, w_a)?;
            save(ctx, buf, w_kw)?;
            buf.push(op::NEWOBJ_EX);
        } else {
            // protocol 2/3 with keyword args needs functools.partial.
            let kw_len = unsafe { pyre_object::dictmultiobject::w_dict_items(w_kw) }.len();
            if kw_len > 0 {
                return Err(PyError::not_implemented(
                    "_pickle: __newobj_ex__ with kwargs at protocol < 4",
                ));
            }
            let w_new = crate::baseobjspace::getattr_str(w_cls, "__new__")?;
            let mut full = vec![w_cls];
            full.extend(tuple_items(w_a));
            save(ctx, buf, w_new)?;
            save(ctx, buf, pyre_object::tupleobject::w_tuple_new(full))?;
            buf.push(op::REDUCE);
        }
    } else if ctx.proto >= 2 && func_name.as_deref() == Some("__newobj__") {
        if args_w.is_empty() {
            return Err(pickling_error("__newobj__ requires at least one arg"));
        }
        let w_cls = args_w[0];
        if crate::baseobjspace::findattr(w_cls, "__new__").is_none() {
            return Err(pickling_error("args[0] from __newobj__ args has no __new__"));
        }
        let w_newargs = pyre_object::tupleobject::w_tuple_new(args_w[1..].to_vec());
        save(ctx, buf, w_cls)?;
        save(ctx, buf, w_newargs)?;
        buf.push(op::NEWOBJ);
    } else {
        save(ctx, buf, w_func)?;
        save(ctx, buf, w_args)?;
        buf.push(op::REDUCE);
    }

    if let Some(w_obj) = w_obj_opt {
        if let Some(idx) = ctx.memo_get(w_obj) {
            buf.push(op::POP);
            write_get(ctx, buf, idx);
        } else {
            memoize(ctx, buf, w_obj);
        }
    }

    if let Some(li) = w_listitems {
        let items = drain_iter(li)?;
        batch_appends(ctx, buf, &items)?;
    }
    if let Some(di) = w_dictitems {
        let pairs = drain_iter_pairs(di)?;
        batch_setitems(ctx, buf, &pairs)?;
    }
    if let Some(st) = w_state {
        if let Some(setter) = w_state_setter {
            save(ctx, buf, setter)?;
            save(ctx, buf, w_obj_opt.unwrap())?;
            save(ctx, buf, st)?;
            buf.push(op::TUPLE2);
            buf.push(op::REDUCE);
            buf.push(op::POP);
        } else {
            save(ctx, buf, st)?;
            buf.push(op::BUILD);
        }
    }
    Ok(())
}

/// The `__name__` of a callable as an owned `String`, if it is a str.
fn func_name_str(w_func: PyObjectRef) -> Option<String> {
    let w_name = crate::baseobjspace::findattr(w_func, "__name__")?;
    if unsafe { pyre_object::is_str(w_name) } {
        Some(unsafe { pyre_object::strobject::w_str_get_value(w_name) }.to_string())
    } else {
        None
    }
}

fn tuple_items(w_tuple: PyObjectRef) -> Vec<PyObjectRef> {
    let n = unsafe { pyre_object::tupleobject::w_tuple_len(w_tuple) };
    (0..n)
        .map(|i| unsafe { pyre_object::tupleobject::w_tuple_getitem(w_tuple, i as i64).unwrap() })
        .collect()
}

/// Drain an iterable into a `Vec`.
fn drain_iter(w_iterable: PyObjectRef) -> Result<Vec<PyObjectRef>, PyError> {
    let w_iter = crate::baseobjspace::iter(w_iterable)?;
    let mut out = Vec::new();
    loop {
        match crate::baseobjspace::next(w_iter) {
            Ok(item) => out.push(item),
            Err(e) if e.kind == crate::PyErrorKind::StopIteration => break,
            Err(e) => return Err(e),
        }
    }
    Ok(out)
}

/// Drain an iterable of `(key, value)` pairs into a `Vec`.
fn drain_iter_pairs(w_iterable: PyObjectRef) -> Result<Vec<(PyObjectRef, PyObjectRef)>, PyError> {
    let items = drain_iter(w_iterable)?;
    let mut out = Vec::with_capacity(items.len());
    for it in items {
        if !unsafe { pyre_object::is_tuple(it) }
            || unsafe { pyre_object::tupleobject::w_tuple_len(it) } != 2
        {
            return Err(pickling_error("dictitems must yield (key, value) pairs"));
        }
        let k = unsafe { pyre_object::tupleobject::w_tuple_getitem(it, 0).unwrap() };
        let v = unsafe { pyre_object::tupleobject::w_tuple_getitem(it, 1).unwrap() };
        out.push((k, v));
    }
    Ok(out)
}
