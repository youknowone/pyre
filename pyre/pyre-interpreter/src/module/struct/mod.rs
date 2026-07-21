//! _struct module — PyPy: `pypy/module/struct/`.
//!
//! Ports the format interpreter (`rpython/rlib/rstruct/formatiterator.py`
//! `FormatIterator.interpret`) plus the standard and native format tables
//! (`standardfmttable.py`, `nativefmttable.py`): byte-order/size/alignment
//! decoding from the leading char, repetition counts, `needcount` codes
//! (`x`/`s`/`p`), per-code range checks, and the `s`/`c`/`x`/`p`/`?`/`e`
//! string, char, pad, pascal, bool and half-float codes.  `struct.error`
//! is `space.new_exception_class("struct.error", space.w_Exception)`
//! (`interp_struct.py:20 Cache`).

use malachite_bigint::BigInt;
use num_traits::ToPrimitive;
use pyre_object::*;

/// True on a big-endian build; the native (`@` / no-prefix) byte order.
const fn native_is_bigendian() -> bool {
    cfg!(target_endian = "big")
}

/// `interp_struct.py:23 get_error` — raise an instance of `struct.error`
/// (registered in the exc-class registry under its qualified name) carrying
/// `msg`.
fn struct_error(msg: impl Into<String>) -> crate::PyError {
    let msg = msg.into();
    let cls = crate::builtins::lookup_exc_class("struct.error")
        .or_else(|| crate::builtins::lookup_exc_class("Exception"))
        .expect("Exception must be installed before _struct is used");
    let exc = crate::builtins::exc_exception_new(&[cls, w_str_new(&msg)])
        .expect("exc_exception_new is infallible for str args");
    let mut err = crate::PyError::new(crate::PyErrorKind::ValueError, msg);
    err.exc_object = exc;
    err
}

/// `StructOverflowError` → `space.w_OverflowError` (`interp_struct.py:58`).
fn struct_overflow(msg: &str) -> crate::PyError {
    crate::PyError::new(crate::PyErrorKind::OverflowError, msg.to_string())
}

/// Accept a str or bytes-like format spec (`interp_struct.py:38
/// text_or_bytes_w`).  CPython's `_struct` boundary encodes text formats as
/// strict ASCII, so non-ASCII text (including lone surrogates) raises
/// `UnicodeEncodeError`; bytes are preserved byte-for-byte and rejected later
/// as `struct.error` when they are not format codes.
fn format_to_string(obj: PyObjectRef) -> Result<String, crate::PyError> {
    unsafe {
        if is_str(obj) {
            let bytes = crate::type_methods::encode_object(obj, "ascii", "strict")?;
            Ok(String::from_utf8(bytes).expect("ASCII encoding is valid UTF-8"))
        } else if bytesobject::is_bytes_like(obj) {
            Ok(bytesobject::bytes_like_data(obj)
                .iter()
                .map(|byte| char::from(*byte))
                .collect())
        } else {
            Err(crate::PyError::type_error(
                "Struct() argument 1 must be str or bytes, not object",
            ))
        }
    }
}

// ── format table ─────────────────────────────────────────────────────

#[derive(Clone, Copy)]
enum Code {
    Pad,
    Char,
    Str,
    Pascal,
    Bool,
    HalfFloat,
    Float,
    Double,
    Int { signed: bool },
}

/// One entry of the standard/native format table (`FmtDesc`).
#[derive(Clone, Copy)]
struct Fmt {
    code: Code,
    /// Element size in bytes (int/float width; 1 for `x`/`c`/`s`/`p`).
    size: usize,
    /// Native alignment (`FmtDesc.alignment`); always 1 in standard mode.
    alignment: usize,
    /// `x`/`s`/`p` consume the repetition as a byte count, not a repeat.
    needcount: bool,
    fmtchar: char,
}

/// `standard_fmttable` / `native_fmttable` lookup.  `native` selects the
/// native table (`@` or no prefix), where `l`/`L`/`n`/`N`/`P` take C
/// `long`/`ssize_t`/`size_t`/pointer sizes and every scalar is naturally
/// aligned; the standard table has no `n`/`N`/`P` and no alignment.
fn lookup_fmt(c: char, native: bool) -> Option<Fmt> {
    // Shared string / pad / char / pascal rows (both tables, size 1).
    match c {
        'x' => {
            return Some(Fmt {
                code: Code::Pad,
                size: 1,
                alignment: 1,
                needcount: true,
                fmtchar: c,
            });
        }
        'c' => {
            return Some(Fmt {
                code: Code::Char,
                size: 1,
                alignment: 1,
                needcount: false,
                fmtchar: c,
            });
        }
        's' => {
            return Some(Fmt {
                code: Code::Str,
                size: 1,
                alignment: 1,
                needcount: true,
                fmtchar: c,
            });
        }
        'p' => {
            return Some(Fmt {
                code: Code::Pascal,
                size: 1,
                alignment: 1,
                needcount: true,
                fmtchar: c,
            });
        }
        _ => {}
    }

    let (code, size) = if native {
        let c_long = std::mem::size_of::<std::os::raw::c_long>();
        let sz_t = std::mem::size_of::<usize>();
        match c {
            'b' => (Code::Int { signed: true }, 1),
            'B' => (Code::Int { signed: false }, 1),
            'h' => (Code::Int { signed: true }, 2),
            'H' => (Code::Int { signed: false }, 2),
            'i' => (Code::Int { signed: true }, 4),
            'I' => (Code::Int { signed: false }, 4),
            'l' => (Code::Int { signed: true }, c_long),
            'L' => (Code::Int { signed: false }, c_long),
            'q' => (Code::Int { signed: true }, 8),
            'Q' => (Code::Int { signed: false }, 8),
            'n' => (Code::Int { signed: true }, sz_t),
            'N' => (Code::Int { signed: false }, sz_t),
            'P' => (Code::Int { signed: false }, sz_t),
            '?' => (Code::Bool, 1),
            'e' => (Code::HalfFloat, 2),
            'f' => (Code::Float, 4),
            'd' => (Code::Double, 8),
            _ => return None,
        }
    } else {
        match c {
            'b' => (Code::Int { signed: true }, 1),
            'B' => (Code::Int { signed: false }, 1),
            'h' => (Code::Int { signed: true }, 2),
            'H' => (Code::Int { signed: false }, 2),
            'i' | 'l' => (Code::Int { signed: true }, 4),
            'I' | 'L' => (Code::Int { signed: false }, 4),
            'q' => (Code::Int { signed: true }, 8),
            'Q' => (Code::Int { signed: false }, 8),
            '?' => (Code::Bool, 1),
            'e' => (Code::HalfFloat, 2),
            'f' => (Code::Float, 4),
            'd' => (Code::Double, 8),
            // `n`/`N`/`P` do not exist in standard sizes.
            _ => return None,
        }
    };
    // Scalar types are naturally aligned on the targets pyre builds for;
    // in standard mode all alignments collapse to 1.
    let alignment = if native { size.max(1) } else { 1 };
    Some(Fmt {
        code,
        size,
        alignment,
        needcount: false,
        fmtchar: c,
    })
}

/// A fully parsed format string: byte order plus the sequence of
/// `(descriptor, repetitions)` units.
struct Parsed {
    bigendian: bool,
    units: Vec<(Fmt, usize)>,
}

impl Parsed {
    /// Number of values a `pack` call must supply: pad consumes none, the
    /// `needcount` codes consume one, every other code consumes one per
    /// repetition.
    fn expected_args(&self) -> usize {
        self.units
            .iter()
            .map(|(fmt, rep)| match fmt.code {
                Code::Pad => 0,
                _ if fmt.needcount => 1,
                _ => *rep,
            })
            .sum()
    }

    /// `CalcSizeFormatIterator` — total packed size including native
    /// alignment padding.
    fn calcsize(&self) -> Result<i64, crate::PyError> {
        let mut total: usize = 0;
        for (fmt, rep) in &self.units {
            if fmt.alignment > 1 {
                total = align_up(total, fmt.alignment);
            }
            let unit = fmt
                .size
                .checked_mul(*rep)
                .ok_or_else(|| struct_error("total struct size too long"))?;
            total = total
                .checked_add(unit)
                .ok_or_else(|| struct_error("total struct size too long"))?;
        }
        i64::try_from(total).map_err(|_| struct_error("total struct size too long"))
    }
}

fn align_up(pos: usize, alignment: usize) -> usize {
    (pos + alignment - 1) & !(alignment - 1)
}

/// `FormatIterator.interpret` — decode the leading byte-order char then
/// each format unit, validating repetition counts and codes.
fn parse_format(format: &str) -> Result<Parsed, crate::PyError> {
    let chars: Vec<char> = format.chars().collect();
    let mut native = true;
    let mut bigendian = native_is_bigendian();
    let mut index = 0;
    if let Some(&first) = chars.first() {
        match first {
            '@' => index = 1,
            '=' => {
                native = false;
                index = 1;
            }
            '<' => {
                native = false;
                bigendian = false;
                index = 1;
            }
            '>' | '!' => {
                native = false;
                bigendian = true;
                index = 1;
            }
            _ => index = 0,
        }
    }

    let mut units = Vec::new();
    while index < chars.len() {
        let mut c = chars[index];
        index += 1;
        if c.is_whitespace() {
            continue;
        }
        let repetitions = if c.is_ascii_digit() {
            let mut rep: i64 = (c as i64) - ('0' as i64);
            loop {
                if index == chars.len() {
                    return Err(struct_error("repeat count given without format specifier"));
                }
                c = chars[index];
                index += 1;
                if !c.is_ascii_digit() {
                    break;
                }
                rep = rep
                    .checked_mul(10)
                    .and_then(|v| v.checked_add((c as i64) - ('0' as i64)))
                    .ok_or_else(|| struct_error("overflow in item count"))?;
            }
            rep as usize
        } else {
            1
        };

        let fmt = match lookup_fmt(c, native) {
            Some(f) => f,
            None => {
                if c == '\0' {
                    return Err(struct_error("embedded null character"));
                }
                return Err(struct_error("bad char in struct format"));
            }
        };
        units.push((fmt, repetitions));
    }
    Ok(Parsed { bigendian, units })
}

// ── argument acceptance ──────────────────────────────────────────────

/// `PackFormatIterator._accept_integral` — an int / bool / `__index__`
/// object as a `BigInt` for range checking.
unsafe fn accept_int(arg: PyObjectRef) -> Result<BigInt, crate::PyError> {
    unsafe {
        if is_int(arg) || is_long(arg) {
            return Ok(crate::builtins::obj_to_bigint(arg));
        }
        if is_bool(arg) {
            return Ok(BigInt::from(if w_bool_get_value(arg) { 1 } else { 0 }));
        }
        if let Some(tp) = crate::typedef::r#type(arg) {
            if let Some(index_fn) = crate::baseobjspace::lookup_in_type(tp, "__index__") {
                let r = crate::call::call_function_impl_result(index_fn, &[arg])?;
                if is_int(r) || is_long(r) {
                    return Ok(crate::builtins::obj_to_bigint(r));
                }
                if is_bool(r) {
                    return Ok(BigInt::from(if w_bool_get_value(r) { 1 } else { 0 }));
                }
            }
        }
        Err(struct_error("required argument is not an integer"))
    }
}

/// `PackFormatIterator.accept_float_arg` — a float / int / `__float__`
/// object as an `f64`.
unsafe fn accept_float(arg: PyObjectRef) -> Result<f64, crate::PyError> {
    unsafe {
        if is_float(arg) {
            return Ok(w_float_get_value(arg));
        }
        if is_int(arg) {
            return Ok(w_int_get_value(arg) as f64);
        }
        if is_long(arg) {
            // A value that does not fit a C double fails the conversion;
            // `np_float` / `np_double` reformat any such failure to
            // "required argument is not a float".
            let f = longobject::jit_bigint_to_f64_or_inf(longobject::w_long_get_value(arg));
            if !f.is_finite() {
                return Err(struct_error("required argument is not a float"));
            }
            return Ok(f);
        }
        if is_bool(arg) {
            return Ok(if w_bool_get_value(arg) { 1.0 } else { 0.0 });
        }
        if let Some(tp) = crate::typedef::r#type(arg) {
            if let Some(float_fn) = crate::baseobjspace::lookup_in_type(tp, "__float__") {
                let r = crate::call::call_function_impl_result(float_fn, &[arg])?;
                if is_float(r) {
                    return Ok(w_float_get_value(r));
                }
            }
        }
        Err(struct_error("required argument is not a float"))
    }
}

/// A bytes / bytearray argument's raw bytes (`accept_str_arg` =
/// `space.bytes_w`).  `role` names the code for the error message.
unsafe fn accept_bytes<'a>(arg: PyObjectRef, msg: &str) -> Result<&'a [u8], crate::PyError> {
    unsafe {
        if bytesobject::is_bytes_like(arg) {
            Ok(bytesobject::bytes_like_data(arg))
        } else {
            Err(struct_error(msg.to_string()))
        }
    }
}

// ── per-code packing ─────────────────────────────────────────────────

/// `make_int_packer.pack_int` — range-check then write `size` low bytes.
unsafe fn pack_int(
    out: &mut Vec<u8>,
    arg: PyObjectRef,
    size: usize,
    signed: bool,
    fmtchar: char,
    bigendian: bool,
) -> Result<(), crate::PyError> {
    let value = unsafe { accept_int(arg)? };
    // Largest unsigned value representable in `size` bytes (`ulargest` in
    // `_range_error`); the signed bounds are `ulargest >> 1` and its
    // bitwise complement.
    let ulargest = u64::MAX >> ((8 - size) * 8);

    // The valid range always fits i64 (signed) or u64 (unsigned, 8-byte),
    // so `to_i64` / `to_u64` double as the in-range test.
    let le8: [u8; 8] = if signed {
        let max = (ulargest >> 1) as i64;
        let min = !max;
        if longobject::jit_bigint_to_i64_fits(&value) != 0 {
            let v = longobject::jit_bigint_to_i64_value(&value);
            if v >= min && v <= max {
                v.to_le_bytes()
            } else {
                return Err(range_error(fmtchar, size, signed));
            }
        } else {
            return Err(range_error(fmtchar, size, signed));
        }
    } else if size < 8 {
        if longobject::jit_bigint_to_i64_fits(&value) != 0 {
            let v = longobject::jit_bigint_to_i64_value(&value);
            if v >= 0 && (v as u64) <= ulargest {
                (v as u64).to_le_bytes()
            } else {
                return Err(range_error(fmtchar, size, signed));
            }
        } else {
            return Err(range_error(fmtchar, size, signed));
        }
    } else {
        match value.to_u64() {
            Some(u) => u.to_le_bytes(),
            None => return Err(range_error(fmtchar, size, signed)),
        }
    };

    if bigendian {
        for i in (0..size).rev() {
            out.push(le8[i]);
        }
    } else {
        out.extend_from_slice(&le8[0..size]);
    }
    Ok(())
}

fn range_error(fmtchar: char, size: usize, signed: bool) -> crate::PyError {
    let ulargest = u64::MAX >> ((8 - size) * 8);
    let msg = if signed {
        let max = (ulargest >> 1) as i64;
        let min = !max;
        format!("'{fmtchar}' format requires {min} <= number <= {max}")
    } else {
        format!("'{fmtchar}' format requires 0 <= number <= {ulargest}")
    };
    struct_error(msg)
}

unsafe fn pack_float_code(
    out: &mut Vec<u8>,
    arg: PyObjectRef,
    code: Code,
    bigendian: bool,
) -> Result<(), crate::PyError> {
    let d = unsafe { accept_float(arg)? };
    match code {
        Code::Float => {
            let f = d as f32;
            if d.is_finite() && f.is_infinite() {
                return Err(struct_overflow("float too large to pack with f format"));
            }
            let b = f.to_bits().to_le_bytes();
            push_endian(out, &b, bigendian);
        }
        Code::Double => {
            let b = d.to_bits().to_le_bytes();
            push_endian(out, &b, bigendian);
        }
        Code::HalfFloat => {
            let bits = pack_half(d)?;
            push_endian(out, &bits.to_le_bytes(), bigendian);
        }
        _ => unreachable!(),
    }
    Ok(())
}

fn push_endian(out: &mut Vec<u8>, le: &[u8], bigendian: bool) {
    if bigendian {
        out.extend(le.iter().rev());
    } else {
        out.extend_from_slice(le);
    }
}

/// `_pack_string` — write `min(len, count)` bytes then zero-pad to `count`.
fn pack_string_bytes(out: &mut Vec<u8>, data: &[u8], count: usize) {
    if data.len() < count {
        out.extend_from_slice(data);
        out.extend(std::iter::repeat_n(0u8, count - data.len()));
    } else {
        out.extend_from_slice(&data[..count]);
    }
}

// ── packing driver ───────────────────────────────────────────────────

/// `do_pack` — pack `values` according to `format`.
fn do_pack(format: &str, values: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let parsed = parse_format(format)?;
    Ok(w_bytes_from_bytes(&pack_values(&parsed, values)?))
}

/// `_pack` — the packed bytes of `values` according to a parsed format.
fn pack_values(parsed: &Parsed, values: &[PyObjectRef]) -> Result<Vec<u8>, crate::PyError> {
    let expected = parsed.expected_args();
    if values.len() != expected {
        return Err(struct_error(format!(
            "pack expected {expected} items for packing (got {})",
            values.len()
        )));
    }

    let mut out: Vec<u8> = Vec::new();
    let mut ai = 0usize;
    for (fmt, rep) in &parsed.units {
        if fmt.alignment > 1 {
            let pad = align_up(out.len(), fmt.alignment) - out.len();
            out.extend(std::iter::repeat_n(0u8, pad));
        }
        let rep = *rep;
        match fmt.code {
            Code::Pad => out.extend(std::iter::repeat_n(0u8, rep)),
            Code::Char => {
                for _ in 0..rep {
                    let arg = values[ai];
                    ai += 1;
                    let data = unsafe {
                        accept_bytes(arg, "char format requires a bytes object of length 1")?
                    };
                    if data.len() != 1 {
                        return Err(struct_error(
                            "char format requires a bytes object of length 1",
                        ));
                    }
                    out.push(data[0]);
                }
            }
            Code::Str => {
                let arg = values[ai];
                ai += 1;
                let data = unsafe { accept_bytes(arg, "argument for 's' must be a bytes object")? };
                pack_string_bytes(&mut out, data, rep);
            }
            Code::Pascal => {
                let arg = values[ai];
                ai += 1;
                let data = unsafe { accept_bytes(arg, "argument for 'p' must be a bytes object")? };
                // `pack_pascal` — length prefix byte (clamped to count-1 and
                // 255) then the string, padded to `count` bytes total.  A
                // `0p` field has no room for even the length byte.
                let mut prefix = data.len();
                if prefix >= rep {
                    if rep == 0 {
                        return Err(struct_error("bad '0p' in struct format"));
                    }
                    prefix = rep - 1;
                }
                if prefix > 255 {
                    prefix = 255;
                }
                out.push(prefix as u8);
                pack_string_bytes(&mut out, &data[..prefix], rep - 1);
            }
            Code::Bool => {
                for _ in 0..rep {
                    let arg = values[ai];
                    ai += 1;
                    out.push(if crate::baseobjspace::is_true(arg)? {
                        1
                    } else {
                        0
                    });
                }
            }
            Code::Int { signed } => {
                for _ in 0..rep {
                    let arg = values[ai];
                    ai += 1;
                    unsafe {
                        pack_int(
                            &mut out,
                            arg,
                            fmt.size,
                            signed,
                            fmt.fmtchar,
                            parsed.bigendian,
                        )?
                    };
                }
            }
            Code::Float | Code::Double | Code::HalfFloat => {
                for _ in 0..rep {
                    let arg = values[ai];
                    ai += 1;
                    unsafe { pack_float_code(&mut out, arg, fmt.code, parsed.bigendian)? };
                }
            }
        }
    }
    Ok(out)
}

/// `space.writebuf_w` — a writable byte slice from a `bytearray` or a
/// contiguous read-write `memoryview` over one.
unsafe fn writebuf<'a>(obj: PyObjectRef) -> Result<&'a mut [u8], crate::PyError> {
    unsafe {
        if bytearrayobject::is_bytearray(obj) {
            return Ok(bytearrayobject::w_bytearray_data_mut(obj));
        }
        if memoryview::is_w_memoryview(obj) {
            crate::builtins::memoryview_check_released(obj)?;
            if memoryview::w_memoryview_readonly(obj) {
                return Err(crate::PyError::type_error(
                    "a read-write bytes-like object is required",
                ));
            }
            let view = memoryview::w_memoryview_view(obj);
            if crate::builtins::memoryview_contiguity(obj).0 {
                let off = view.offset() as usize;
                let len = memoryview::w_memoryview_length(obj) as usize;
                if let Some(full) = view.backing().as_bytes_mut() {
                    if off <= full.len() && len <= full.len() - off {
                        return Ok(&mut full[off..off + len]);
                    }
                }
            }
        }
        Err(crate::PyError::type_error(
            "argument must be read-write bytes-like object",
        ))
    }
}

/// `do_pack_into` — pack `values` into `buffer` starting at `offset`,
/// resolving a negative offset against the buffer end.
fn do_pack_into(
    format: &str,
    buffer: PyObjectRef,
    offset: i64,
    values: &[PyObjectRef],
) -> Result<PyObjectRef, crate::PyError> {
    let parsed = parse_format(format)?;
    let size = parsed.calcsize()?;
    let buf = unsafe { writebuf(buffer)? };
    let buflen = buf.len() as i64;
    let mut offset = offset;
    if offset < 0 {
        if offset + size > 0 {
            return Err(struct_error(format!(
                "no space to pack {size} bytes at offset {offset}"
            )));
        }
        if offset + buflen < 0 {
            return Err(struct_error(format!(
                "offset {offset} out of range for {buflen}-byte buffer"
            )));
        }
        offset += buflen;
    }
    if buflen - offset < size {
        return Err(struct_error(format!(
            "pack_into requires a buffer of at least {} bytes for packing {} bytes at offset {} (actual buffer size is {})",
            size + offset,
            size,
            offset,
            buflen
        )));
    }
    let packed = pack_values(&parsed, values)?;
    let start = offset as usize;
    buf[start..start + packed.len()].copy_from_slice(&packed);
    Ok(w_none())
}

// ── per-code unpacking ───────────────────────────────────────────────

/// `make_int_unpacker.unpack_int` — read `size` bytes and box the value,
/// promoting an unsigned value above `i64::MAX` to a `W_LongObject`.
fn unpack_int(raw: &[u8], size: usize, signed: bool, bigendian: bool) -> PyObjectRef {
    // Normalise to little-endian ordering.
    let mut le = [0u8; 8];
    for i in 0..size {
        le[i] = if bigendian { raw[size - 1 - i] } else { raw[i] };
    }
    if signed {
        // Sign-extend from the top byte of the field.
        let fill = if le[size - 1] & 0x80 != 0 { 0xff } else { 0x00 };
        for b in le.iter_mut().skip(size) {
            *b = fill;
        }
        w_int_new(i64::from_le_bytes(le))
    } else {
        let u = u64::from_le_bytes(le);
        if u <= i64::MAX as u64 {
            w_int_new(u as i64)
        } else {
            longobject::w_long_new(BigInt::from(u))
        }
    }
}

fn unpack_float(raw: &[u8], code: Code, bigendian: bool) -> PyObjectRef {
    match code {
        Code::Float => {
            let mut b = [0u8; 4];
            copy_endian(raw, &mut b, bigendian);
            w_float_new(f32::from_bits(u32::from_le_bytes(b)) as f64)
        }
        Code::Double => {
            let mut b = [0u8; 8];
            copy_endian(raw, &mut b, bigendian);
            w_float_new(f64::from_bits(u64::from_le_bytes(b)))
        }
        Code::HalfFloat => {
            let mut b = [0u8; 2];
            copy_endian(raw, &mut b, bigendian);
            w_float_new(unpack_half(u16::from_le_bytes(b)))
        }
        _ => unreachable!(),
    }
}

fn copy_endian(raw: &[u8], out: &mut [u8], bigendian: bool) {
    let n = out.len();
    for i in 0..n {
        out[i] = if bigendian { raw[n - 1 - i] } else { raw[i] };
    }
}

// ── unpacking driver ─────────────────────────────────────────────────

/// `do_unpack` / `_unpack` — unpack the whole of `buf` (which must be
/// exactly `calcsize(format)` bytes) according to `format`.
fn do_unpack(format: &str, buf: &[u8]) -> Result<PyObjectRef, crate::PyError> {
    let parsed = parse_format(format)?;
    let size = parsed.calcsize()? as usize;
    if buf.len() != size {
        return Err(struct_error(format!(
            "unpack requires a buffer of {size} bytes"
        )));
    }
    unpack_units(&parsed, buf)
}

fn unpack_units(parsed: &Parsed, buf: &[u8]) -> Result<PyObjectRef, crate::PyError> {
    let mut out: Vec<PyObjectRef> = Vec::new();
    let mut pos = 0usize;
    for (fmt, rep) in &parsed.units {
        if fmt.alignment > 1 {
            pos = align_up(pos, fmt.alignment);
        }
        let rep = *rep;
        match fmt.code {
            Code::Pad => pos += rep,
            Code::Char => {
                for _ in 0..rep {
                    out.push(w_bytes_from_bytes(&buf[pos..pos + 1]));
                    pos += 1;
                }
            }
            Code::Str => {
                out.push(w_bytes_from_bytes(&buf[pos..pos + rep]));
                pos += rep;
            }
            Code::Pascal => {
                // `unpack_pascal` — first byte is the length, clamped to the
                // field width; the value is the following bytes.
                if rep == 0 {
                    return Err(struct_error("bad '0p' in struct format"));
                }
                let data = &buf[pos..pos + rep];
                let end = (1 + data[0] as usize).min(rep);
                out.push(w_bytes_from_bytes(&data[1..end]));
                pos += rep;
            }
            Code::Bool => {
                for _ in 0..rep {
                    out.push(boolobject::w_bool_from(buf[pos] != 0));
                    pos += 1;
                }
            }
            Code::Int { signed } => {
                for _ in 0..rep {
                    out.push(unpack_int(
                        &buf[pos..pos + fmt.size],
                        fmt.size,
                        signed,
                        parsed.bigendian,
                    ));
                    pos += fmt.size;
                }
            }
            Code::Float | Code::Double | Code::HalfFloat => {
                for _ in 0..rep {
                    out.push(unpack_float(
                        &buf[pos..pos + fmt.size],
                        fmt.code,
                        parsed.bigendian,
                    ));
                    pos += fmt.size;
                }
            }
        }
    }
    Ok(w_tuple_new(out))
}

/// `do_unpack_from` — unpack `calcsize(format)` bytes of `buf` starting at
/// `offset`, resolving a negative offset against the buffer end.
fn do_unpack_from(format: &str, buf: &[u8], offset: i64) -> Result<PyObjectRef, crate::PyError> {
    let parsed = parse_format(format)?;
    let size = parsed.calcsize()? as usize;
    let buflen = buf.len() as i64;
    let mut offset = offset;
    if offset < 0 {
        if offset + size as i64 > 0 {
            return Err(struct_error(format!(
                "not enough data to unpack {size} bytes at offset {offset}"
            )));
        }
        if offset + buflen < 0 {
            return Err(struct_error(format!(
                "offset {offset} out of range for {buflen}-byte buffer"
            )));
        }
        offset += buflen;
    }
    if buflen - offset < size as i64 {
        return Err(struct_error(format!(
            "unpack_from requires a buffer of at least {} bytes for unpacking {} bytes at offset {} (actual buffer size is {})",
            size as i64 + offset,
            size,
            offset,
            buflen
        )));
    }
    let start = offset as usize;
    unpack_units(&parsed, &buf[start..start + size])
}

// ── half-float (IEEE 754 binary16) ───────────────────────────────────

/// `frexp` for a finite non-zero positive `x`: returns `(m, e)` with
/// `x == m * 2**e` and `m` in `[0.5, 1.0)`.
fn frexp(x: f64) -> (f64, i32) {
    let bits = x.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    if biased == 0 {
        // Subnormal: scale up by 2**53 to normalise, then correct exponent.
        let norm = x * 9007199254740992.0; // 2**53
        let nbits = norm.to_bits();
        let e = (((nbits >> 52) & 0x7ff) as i32) - 1022 - 53;
        let m = f64::from_bits((nbits & !(0x7ffu64 << 52)) | (1022u64 << 52));
        return (m, e);
    }
    let e = biased - 1022;
    let m = f64::from_bits((bits & !(0x7ffu64 << 52)) | (1022u64 << 52));
    (m, e)
}

fn ldexp(x: f64, e: i32) -> f64 {
    x * (2.0f64).powi(e)
}

/// `_PyFloat_Pack2` — round `x` to IEEE binary16, raising `OverflowError`
/// for a finite value too large to represent.
pub(crate) fn pack_half(x: f64) -> Result<u16, crate::PyError> {
    let sign: u16;
    let e: i32;
    let bits: u16;
    if x == 0.0 {
        sign = if x.is_sign_negative() { 1 } else { 0 };
        e = 0;
        bits = 0;
    } else if x.is_infinite() {
        sign = if x < 0.0 { 1 } else { 0 };
        e = 0x1f;
        bits = 0;
    } else if x.is_nan() {
        sign = if x.is_sign_negative() { 1 } else { 0 };
        e = 0x1f;
        bits = 512;
    } else {
        sign = if x < 0.0 { 1 } else { 0 };
        let ax = x.abs();
        let (mut f, mut ex) = frexp(ax);
        // Normalize f to [1.0, 2.0).
        f *= 2.0;
        ex -= 1;
        if ex >= 16 {
            return Err(struct_overflow("float too large to pack with e format"));
        } else if ex < -25 {
            // |x| < 2**-25: underflow to zero.
            f = 0.0;
            ex = 0;
        } else if ex < -14 {
            // Gradual underflow.
            f = ldexp(f, 14 + ex);
            ex = 0;
        } else {
            ex += 15;
            f -= 1.0; // Discard the leading 1.
        }
        f *= 1024.0; // 2**10
        let mut b = f as u16; // truncation
        let frac = f - (b as f64);
        if frac > 0.5 || (frac == 0.5 && (b & 1) == 1) {
            b += 1;
            if b == 1024 {
                // Carry propagated out of ten 1 bits.
                b = 0;
                ex += 1;
                if ex == 31 {
                    return Err(struct_overflow("float too large to pack with e format"));
                }
            }
        }
        e = ex;
        bits = b;
    }
    Ok(bits | ((e as u16) << 10) | (sign << 15))
}

/// `_PyFloat_Unpack2` — decode an IEEE binary16 to a double.
pub(crate) fn unpack_half(bits: u16) -> f64 {
    let sign = (bits >> 15) & 1 == 1;
    let e = ((bits >> 10) & 0x1f) as i32;
    let f = (bits & 0x3ff) as u32;
    let val = if e == 0x1f {
        if f == 0 { f64::INFINITY } else { f64::NAN }
    } else {
        let mut x = f as f64 / 1024.0;
        let ee = if e == 0 {
            -14
        } else {
            x += 1.0;
            e - 15
        };
        ldexp(x, ee)
    };
    if sign { -val } else { val }
}

// ── W_Struct ─────────────────────────────────────────────────────────

/// `interp_struct.py:213 W_Struct` — a compiled struct object holding its
/// format string and precomputed size.
#[crate::pyre_class("_struct.Struct")]
pub struct W_Struct {
    /// Format string object (`text_or_bytes_w` of the constructor arg),
    /// promoted by value before each pack/unpack.  `_immutable_fields_ =
    /// ["format", "size"]`.
    format: PyObjectRef,
    size: i64,
}

#[crate::pyre_methods(doc = "Struct(fmt) --> compiled struct object")]
impl W_Struct {
    /// `interp_struct.py:256 descr__new__` — the format string is consumed by
    /// `__init__`, so the allocator accepts and discards the trailing
    /// `__args__` instead of failing the gateway arity check.
    #[staticmethod]
    fn __new__(_cls: PyObjectRef, _args: &[PyObjectRef]) -> PyObjectRef {
        W_Struct::allocate_stable(W_Struct {
            ob: pyre_object::PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            format: w_str_new(""),
            size: -1,
        })
    }

    /// `interp_struct.py:222 descr__init__` — store the (normalized)
    /// format string and its `_calcsize`.
    fn __init__(&mut self, w_format: PyObjectRef) -> Result<(), crate::PyError> {
        let format = format_to_string(w_format)?;
        self.size = parse_format(&format)?.calcsize()?;
        self.format = w_str_new(&format);
        pyre_object::gc_hook::try_gc_write_barrier(self as *mut Self as *mut u8);
        Ok(())
    }

    #[getter]
    fn format(&self) -> PyObjectRef {
        self.format
    }

    #[getter]
    fn size(&self) -> i64 {
        self.size
    }

    /// `interp_struct.py:227 descr_pack` —
    /// `do_pack(space, jit.promote_string(self.format), args_w)`.
    /// The whole-args-slice ABI hands `args[0]` = self; the packed values
    /// are `args[1..]`.
    fn pack(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let format = majit_metainterp::jit::promote_string(self.format);
        let fmt = unsafe { w_str_get_value(format) };
        do_pack(fmt, &args[1..])
    }

    /// `interp_struct.py:234 descr_unpack` —
    /// `do_unpack(space, jit.promote_string(self.format), w_str)`.
    fn unpack(&self, w_str: PyObjectRef) -> Result<PyObjectRef, crate::PyError> {
        let format = majit_metainterp::jit::promote_string(self.format);
        let fmt = unsafe { w_str_get_value(format) };
        let buf = unsafe { readbuf(w_str)? };
        do_unpack(fmt, buf)
    }

    /// `interp_struct.py:231 descr_pack_into` —
    /// `do_pack_into(space, jit.promote_string(self.format), buffer, offset, args_w)`.
    /// Whole-args ABI: `args[0]` = self, `args[1]` = buffer, `args[2]` =
    /// offset, `args[3..]` = the packed values.
    fn pack_into(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let format = majit_metainterp::jit::promote_string(self.format);
        let fmt = unsafe { w_str_get_value(format) };
        let (pos, _) = crate::builtins::split_builtin_kwargs(&args[1..]);
        if pos.len() < 2 {
            return Err(crate::PyError::type_error(
                "pack_into() missing buffer or offset argument",
            ));
        }
        let offset = unsafe { crate::builtins::space_index_w(pos[1])? };
        do_pack_into(fmt, pos[0], offset, &pos[2..])
    }

    /// `interp_struct.py:238 descr_unpack_from` —
    /// `do_unpack_from(space, jit.promote_string(self.format), buffer, offset)`.
    /// `buffer` / `offset` are accepted positionally or by keyword.
    fn unpack_from(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let format = majit_metainterp::jit::promote_string(self.format);
        let fmt = unsafe { w_str_get_value(format) };
        let (buffer, offset) = resolve_buffer_offset(&args[1..])?;
        let buf = unsafe { readbuf(buffer)? };
        do_unpack_from(fmt, buf, offset)
    }

    /// `interp_struct.py:241 descr_iter_unpack` — a `W_UnpackIter` over
    /// `buffer`.  Whole-args ABI: `args[0]` = self, `args[1]` = buffer.
    fn iter_unpack(&self, args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
        let buffer = args
            .get(1)
            .copied()
            .ok_or_else(|| crate::PyError::type_error("iter_unpack() missing buffer argument"))?;
        make_unpack_iter(self.format, self.size, buffer)
    }

    /// `_struct.Struct.__repr__` — `Struct('<format>')`.
    fn __repr__(&self) -> PyObjectRef {
        let fmt = unsafe { w_str_get_value(self.format) };
        w_str_new(&format!("Struct('{fmt}')"))
    }
}

/// Resolve `(buffer, offset=0)` from a positional/keyword argument slice
/// (`unpack_from` accepts both `buffer=` and `offset=`).
fn resolve_buffer_offset(args: &[PyObjectRef]) -> Result<(PyObjectRef, i64), crate::PyError> {
    let (pos, kwargs) = crate::builtins::split_builtin_kwargs(args);
    let buffer = pos
        .first()
        .copied()
        .or_else(|| crate::builtins::kwarg_get(kwargs, "buffer"))
        .ok_or_else(|| {
            crate::PyError::type_error("unpack_from() missing required argument 'buffer'")
        })?;
    let offset = match pos
        .get(1)
        .copied()
        .or_else(|| crate::builtins::kwarg_get(kwargs, "offset"))
    {
        Some(o) => unsafe { crate::builtins::space_index_w(o)? },
        None => 0,
    };
    Ok((buffer, offset))
}

// ── W_UnpackIter ─────────────────────────────────────────────────────

/// `interp_struct.py:177 W_UnpackIter` — an iterator yielding the
/// successive fixed-size records of a buffer.  Kept in its own module
/// because each `#[pyre_class]` emits a module-scoped `type_object()`.
pub mod unpack_iter {
    use super::{do_unpack, readbuf, struct_error};
    use pyre_object::*;

    #[crate::pyre_class("_struct.unpack_iterator")]
    pub struct W_UnpackIter {
        format: PyObjectRef,
        buffer: PyObjectRef,
        size: i64,
        index: i64,
    }

    #[crate::pyre_methods]
    impl W_UnpackIter {
        fn __iter__(&self, args: &[PyObjectRef]) -> PyObjectRef {
            args[0]
        }

        /// `descr_next` — unpack the next record, raising `StopIteration`
        /// at the end of the buffer.
        fn __next__(&mut self) -> Result<PyObjectRef, crate::PyError> {
            let buf = unsafe { readbuf(self.buffer)? };
            if self.index >= buf.len() as i64 {
                return Err(crate::PyError::new(crate::PyErrorKind::StopIteration, ""));
            }
            let start = self.index as usize;
            let end = start + self.size as usize;
            let fmt = unsafe { w_str_get_value(self.format) };
            let res = do_unpack(fmt, &buf[start..end])?;
            self.index += self.size;
            Ok(res)
        }

        /// `descr_length_hint` — records remaining.
        fn __length_hint__(&self) -> PyObjectRef {
            let remaining = match unsafe { readbuf(self.buffer) } {
                Ok(buf) => (buf.len() as i64 - self.index) / self.size,
                Err(_) => 0,
            };
            w_int_new(remaining)
        }
    }

    /// `interp_struct.py:178 W_UnpackIter.__init__` — build an unpack
    /// iterator, rejecting a zero-length struct or a buffer whose length is
    /// not a whole multiple of the record size.
    pub fn make_unpack_iter(
        format: PyObjectRef,
        size: i64,
        buffer: PyObjectRef,
    ) -> Result<PyObjectRef, crate::PyError> {
        // Force the type's method table to be built before allocating (the
        // `unpack_iterator` type is not exposed as a module attribute, so
        // nothing else triggers `type_object()`).
        let _ = type_object();
        let buf = unsafe { readbuf(buffer)? };
        if size <= 0 {
            return Err(struct_error(format!(
                "cannot iteratively unpack with a struct of length {size}"
            )));
        }
        if buf.len() as i64 % size != 0 {
            return Err(struct_error(format!(
                "iterative unpacking requires a buffer of a multiple of {size} bytes"
            )));
        }
        Ok(W_UnpackIter::allocate_stable(W_UnpackIter {
            ob: pyre_object::PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            format,
            buffer,
            size,
            index: 0,
        }))
    }
}

use unpack_iter::make_unpack_iter;

/// True if `obj` is an `unpack_iterator` (the `iter_unpack` result).  Used
/// by the `iter()` / `next()` dispatch to route the iteration protocol,
/// mirroring the other builtin iterators (`is_sre_scanner`, …).
pub fn is_unpack_iter(obj: PyObjectRef) -> bool {
    unpack_iter::W_UnpackIter::from_obj(obj).is_some()
}

/// `space.readbuf_w` — a read-only byte slice from a bytes-like object.
unsafe fn readbuf<'a>(obj: PyObjectRef) -> Result<&'a [u8], crate::PyError> {
    unsafe {
        if bytesobject::is_bytes_like(obj) {
            return Ok(bytesobject::bytes_like_data(obj));
        }
        if interp_array::is_array(obj) {
            return Ok(interp_array::w_array_bytes(obj));
        }
        if memoryview::is_w_memoryview(obj) {
            crate::builtins::memoryview_check_released(obj)?;
            if crate::builtins::memoryview_contiguity(obj).0 {
                let view = memoryview::w_memoryview_view(obj);
                let full = view.backing().as_bytes();
                let off = view.offset() as usize;
                let len = memoryview::w_memoryview_length(obj) as usize;
                if off <= full.len() && len <= full.len() - off {
                    return Ok(&full[off..off + len]);
                }
            }
        }
        Err(crate::PyError::type_error(
            "a bytes-like object is required",
        ))
    }
}

crate::py_module! {
    "_struct",
    interpleveldefs: {
        "Struct" => type_object(),
    },
    inline_functions: {
        fn _clearcache() {}
        fn calcsize(fmt_obj: PyObjectRef) -> Result<i64, crate::PyError> {
            let fmt = format_to_string(fmt_obj)?;
            parse_format(&fmt)?.calcsize()
        }
        fn unpack(fmt_obj: PyObjectRef, buf: &[u8]) -> Result<PyObjectRef, crate::PyError> {
            let fmt = format_to_string(fmt_obj)?;
            do_unpack(&fmt, buf)
        }
    },
    functions: {
        // `pack(fmt, *args)` — variadic positional after fmt; route
        // through the args slice (typed varargs are not supported by
        // inline_functions arity inference).
        "pack" / * = |args| {
            if args.is_empty() {
                return Err(crate::PyError::type_error(
                    "pack() missing 1 required positional argument: 'fmt'",
                ));
            }
            let fmt = format_to_string(args[0])?;
            do_pack(&fmt, &args[1..])
        },
        // `pack_into(fmt, buffer, offset, *args)` — write the packed bytes
        // into a writable buffer at `offset`.
        "pack_into" / * = |args| {
            let (pos, _) = crate::builtins::split_builtin_kwargs(args);
            if pos.len() < 3 {
                return Err(crate::PyError::type_error(
                    "pack_into() missing format, buffer or offset argument",
                ));
            }
            let fmt = format_to_string(pos[0])?;
            let offset = unsafe { crate::builtins::space_index_w(pos[2])? };
            do_pack_into(&fmt, pos[1], offset, &pos[3..])
        },
        // `unpack_from(fmt, /, buffer, offset=0)` — `buffer` / `offset`
        // are accepted positionally or by keyword.
        "unpack_from" / * = |args| {
            let (pos, _) = crate::builtins::split_builtin_kwargs(args);
            if pos.is_empty() {
                return Err(crate::PyError::type_error(
                    "unpack_from() missing required argument 'format'",
                ));
            }
            let fmt = format_to_string(pos[0])?;
            let (buffer, offset) = resolve_buffer_offset(&args[1..])?;
            let buf = unsafe { readbuf(buffer)? };
            do_unpack_from(&fmt, buf, offset)
        },
        // `iter_unpack(fmt, buffer)` — an iterator over the records.
        "iter_unpack" / 2 = |args| {
            let fmt = format_to_string(args[0])?;
            let size = parse_format(&fmt)?.calcsize()?;
            make_unpack_iter(w_str_new(&fmt), size, args[1])
        },
    },
    extra_init: |ns| {
        // `interp_struct.py:20 Cache` —
        // `space.new_exception_class("struct.error", space.w_Exception)`.
        let base = crate::builtins::lookup_exc_class("Exception")
            .expect("Exception must be installed before _struct init");
        let error = crate::builtins::make_exc_type(
            "struct.error",
            crate::builtins::exc_exception_new,
            base,
        );
        crate::module_ns_store(ns, "error", error);
    },
}
