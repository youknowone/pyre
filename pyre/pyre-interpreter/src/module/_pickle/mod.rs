//! `_pickle` — interp-level accelerator for the `pickle` module.
//!
//! Port of `pypy/module/_pickle/interp_pickle.py` (`W_Pickler` /
//! `W_Unpickler`). Targets the CPython 3.14 wire format.
//!
//! Current scope: protocol 2-5 atoms — None / bool / int / float / str /
//! bytes — plus PROTO / FRAME / STOP framing, the memo (MEMOIZE/BINPUT/PUT
//! writes + GET/BINGET/LONG_BINGET back-references), and the built-in
//! containers tuple / list / dict (with APPENDS / SETITEMS batching and
//! recursive-structure handling). Sets, global/reduce, the legacy
//! protocol-0/1 text opcodes, and out-of-band buffers land in later
//! increments. The module deliberately exports only `Pickler` /
//! `Unpickler`; `pickle.py`'s `from _pickle import (...)` keeps falling
//! back to the pure-Python path until the full surface is ready.

use malachite_bigint::{BigInt, Sign};
use pyre_object::PyObjectRef;

use crate::PyError;

mod pickler;
mod unpickler;

pub use pickler::W_Pickler;
pub use unpickler::W_Unpickler;

pub(crate) const HIGHEST_PROTOCOL: i64 = 5;
pub(crate) const DEFAULT_PROTOCOL: i64 = 5;
pub(crate) const FRAME_SIZE_MIN: usize = 4;

/// `interp_pickle.py Opcodes`.
pub(crate) mod op {
    pub const PROTO: u8 = 0x80;
    pub const FRAME: u8 = 0x95;
    pub const STOP: u8 = b'.';
    pub const NONE: u8 = b'N';
    pub const NEWTRUE: u8 = 0x88;
    pub const NEWFALSE: u8 = 0x89;
    pub const BININT: u8 = b'J';
    pub const BININT1: u8 = b'K';
    pub const BININT2: u8 = b'M';
    pub const LONG1: u8 = 0x8a;
    pub const LONG4: u8 = 0x8b;
    pub const BINFLOAT: u8 = b'G';
    pub const SHORT_BINUNICODE: u8 = 0x8c;
    pub const BINUNICODE: u8 = b'X';
    pub const BINUNICODE8: u8 = 0x8d;
    pub const SHORT_BINBYTES: u8 = b'C';
    pub const BINBYTES: u8 = b'B';
    pub const BINBYTES8: u8 = 0x8e;
    // memo
    pub const MEMOIZE: u8 = 0x94;
    pub const BINPUT: u8 = b'q';
    pub const LONG_BINPUT: u8 = b'r';
    pub const PUT: u8 = b'p';
    pub const GET: u8 = b'g';
    pub const BINGET: u8 = b'h';
    pub const LONG_BINGET: u8 = b'j';
    // stack
    pub const MARK: u8 = b'(';
    pub const POP: u8 = b'0';
    pub const POP_MARK: u8 = b'1';
    // tuple
    pub const EMPTY_TUPLE: u8 = b')';
    pub const TUPLE: u8 = b't';
    pub const TUPLE1: u8 = 0x85;
    pub const TUPLE2: u8 = 0x86;
    pub const TUPLE3: u8 = 0x87;
    // list
    pub const EMPTY_LIST: u8 = b']';
    pub const LIST: u8 = b'l';
    pub const APPEND: u8 = b'a';
    pub const APPENDS: u8 = b'e';
    // dict
    pub const EMPTY_DICT: u8 = b'}';
    pub const DICT: u8 = b'd';
    pub const SETITEM: u8 = b's';
    pub const SETITEMS: u8 = b'u';
    // set / frozenset
    pub const EMPTY_SET: u8 = 0x8f;
    pub const FROZENSET: u8 = 0x91;
    pub const ADDITEMS: u8 = 0x90;
    // bytearray
    pub const BYTEARRAY8: u8 = 0x96;

    /// `_tuplesize2code` — TUPLE1/2/3 indexed by element count (1..=3).
    pub const TUPLESIZE2CODE: [u8; 4] = [EMPTY_TUPLE, TUPLE1, TUPLE2, TUPLE3];
}

/// `interp_pickle.py W_Pickler._BATCHSIZE`.
pub(crate) const BATCHSIZE: usize = 1000;

// ── shared call helpers ──────────────────────────────────────────────
// `call_function` / `call_method` return PY_NULL on failure and stash the
// error through `call::set_call_error`; surface it as a Rust `Result`.

pub(crate) fn call_fn(callable: PyObjectRef, args: &[PyObjectRef]) -> Result<PyObjectRef, PyError> {
    let r = crate::baseobjspace::call_function(callable, args);
    if r.is_null() {
        return Err(crate::call::take_call_error()
            .unwrap_or_else(|| PyError::runtime_error("call failed")));
    }
    Ok(r)
}

pub(crate) fn call_meth(
    obj: PyObjectRef,
    name: &str,
    args: &[PyObjectRef],
) -> Result<PyObjectRef, PyError> {
    let r = crate::baseobjspace::call_method(obj, name, args);
    if r.is_null() {
        return Err(crate::call::take_call_error()
            .unwrap_or_else(|| PyError::runtime_error("method call failed")));
    }
    Ok(r)
}

// TODO(inc8): raise the app-level `_pickle.PicklingError` / `UnpicklingError`
// once those classes are registered; until then carry the faithful message
// text on a generic error.
pub(crate) fn unpickling_error(msg: &str) -> PyError {
    PyError::value_error(msg)
}

// ── encode_long / decode_long — two's-complement little-endian ───────
// `interp_pickle.py encode_long` / CPython `pickle.encode_long`.

pub(crate) fn encode_long(big: &BigInt) -> Vec<u8> {
    let sign = big.sign();
    if sign == Sign::NoSign {
        return Vec::new(); // 0 -> b''
    }
    // magnitude, little-endian
    let (_s, digits) = big.to_u32_digits();
    let mut mag: Vec<u8> = Vec::with_capacity(digits.len() * 4);
    for d in &digits {
        mag.extend_from_slice(&d.to_le_bytes());
    }
    while mag.len() > 1 && *mag.last().unwrap() == 0 {
        mag.pop();
    }
    // reserve a byte for the sign bit when the top magnitude bit is set
    if mag.last().map_or(true, |&b| b & 0x80 != 0) {
        mag.push(0x00);
    }
    if sign == Sign::Plus {
        return mag;
    }
    // negative: two's complement (invert + 1)
    let mut carry: u16 = 1;
    for b in mag.iter_mut() {
        let v = (!*b as u16) + carry;
        *b = (v & 0xff) as u8;
        carry = v >> 8;
    }
    // trim a redundant 0xff sign byte (encode_long minimal form)
    while mag.len() > 1 && mag[mag.len() - 1] == 0xff && (mag[mag.len() - 2] & 0x80) != 0 {
        mag.pop();
    }
    mag
}

pub(crate) fn decode_long(data: &[u8]) -> PyObjectRef {
    if data.is_empty() {
        return pyre_object::w_int_new(0);
    }
    let negative = data[data.len() - 1] & 0x80 != 0;
    let unsigned = BigInt::from_bytes_le(Sign::Plus, data);
    let value = if negative {
        // subtract 2**(8*len): little-endian bytes are `len` zeros then 0x01
        let mut pow = vec![0u8; data.len()];
        pow.push(1);
        unsigned - BigInt::from_bytes_le(Sign::Plus, &pow)
    } else {
        unsigned
    };
    int_from_bigint(value)
}

/// Demote to a small int when it fits, mirroring the int/long unification.
pub(crate) fn int_from_bigint(value: BigInt) -> PyObjectRef {
    match i64::try_from(&value) {
        Ok(v) => pyre_object::w_int_new(v),
        Err(_) => pyre_object::w_long_new(value),
    }
}

pub(crate) fn read_int_le(data: &[u8]) -> i64 {
    let mut v: i64 = 0;
    for (i, &b) in data.iter().enumerate() {
        v |= (b as i64) << (8 * i);
    }
    v
}

pub(crate) fn str_from_utf8(data: &[u8]) -> Result<PyObjectRef, PyError> {
    let s = std::str::from_utf8(data).map_err(|_| unpickling_error("invalid utf-8 in pickle"))?;
    Ok(pyre_object::w_str_new(s))
}

crate::py_module! {
    "_pickle",
    interpleveldefs: {
        "Pickler" => pickler::type_object(),
        "Unpickler" => unpickler::type_object(),
    },
}
