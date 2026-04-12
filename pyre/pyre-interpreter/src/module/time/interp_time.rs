//! Time function implementations.
//!
//! PyPy equivalent: pypy/module/time/interp_time.py

use pyre_object::*;
use std::time::{SystemTime, UNIX_EPOCH};

/// time.time() → float (seconds since epoch)
pub fn time(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _ = args;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    Ok(floatobject::w_float_new(now.as_secs_f64()))
}

/// time.time_ns() → int (nanoseconds since epoch)
pub fn time_ns(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _ = args;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    Ok(w_int_new(now.as_nanos() as i64))
}

/// time.monotonic() → float
pub fn monotonic(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _ = args;
    // Simplified: use system time as monotonic approximation
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    Ok(floatobject::w_float_new(now.as_secs_f64()))
}

/// time.sleep(seconds)
pub fn sleep(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "sleep() takes exactly one argument");
    let secs = unsafe {
        if is_int(args[0]) {
            w_int_get_value(args[0]) as f64
        } else if is_float(args[0]) {
            floatobject::w_float_get_value(args[0])
        } else {
            0.0
        }
    };
    #[cfg(feature = "host_env")]
    std::thread::sleep(std::time::Duration::from_secs_f64(secs));
    Ok(w_none())
}

/// time.perf_counter() → float
pub fn perf_counter(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _ = args;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    Ok(floatobject::w_float_new(now.as_secs_f64()))
}

// ── libc tm helpers ──────────────────────────────────────────────────

/// Convert epoch seconds → libc `struct tm` via `gmtime_r`.
fn _c_gmtime(seconds: libc::time_t) -> Option<libc::tm> {
    unsafe {
        let mut tm: libc::tm = std::mem::zeroed();
        let p = libc::gmtime_r(&seconds, &mut tm);
        if p.is_null() { None } else { Some(tm) }
    }
}

/// Convert epoch seconds → libc `struct tm` via `localtime_r`.
fn _c_localtime(seconds: libc::time_t) -> Option<libc::tm> {
    unsafe {
        let mut tm: libc::tm = std::mem::zeroed();
        let p = libc::localtime_r(&seconds, &mut tm);
        if p.is_null() { None } else { Some(tm) }
    }
}

/// Build a Python time tuple from a libc `struct tm`.
fn _tm_to_tuple(tm: &libc::tm) -> PyObjectRef {
    w_tuple_new(vec![
        w_int_new((tm.tm_year + 1900) as i64),
        w_int_new((tm.tm_mon + 1) as i64),
        w_int_new(tm.tm_mday as i64),
        w_int_new(tm.tm_hour as i64),
        w_int_new(tm.tm_min as i64),
        w_int_new(tm.tm_sec as i64),
        w_int_new(((tm.tm_wday + 6) % 7) as i64), // Monday=0
        w_int_new((tm.tm_yday + 1) as i64),
        w_int_new(tm.tm_isdst as i64),
    ])
}

/// Extract epoch seconds from an optional argument (int, float, or None/absent → now).
fn _get_seconds(args: &[PyObjectRef]) -> libc::time_t {
    if let Some(&arg) = args.first() {
        unsafe {
            if !is_none(arg) {
                if is_int(arg) {
                    return w_int_get_value(arg) as libc::time_t;
                }
                if is_float(arg) {
                    return floatobject::w_float_get_value(arg) as libc::time_t;
                }
            }
        }
    }
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as libc::time_t)
        .unwrap_or(0)
}

/// Extract a `struct tm` from a Python time tuple argument.
/// interp_time.py: _gettmarg
fn _gettmarg(args: &[PyObjectRef], default_now: bool) -> Result<libc::tm, crate::PyError> {
    let tup = if let Some(&arg) = args.first() {
        if unsafe { is_none(arg) } {
            if default_now {
                return _c_localtime(_get_seconds(&[]))
                    .ok_or_else(|| crate::PyError::value_error("unconvertible time"));
            }
            return Err(crate::PyError::type_error(
                "Tuple or struct_time argument required",
            ));
        }
        arg
    } else if default_now {
        return _c_localtime(_get_seconds(&[]))
            .ok_or_else(|| crate::PyError::value_error("unconvertible time"));
    } else {
        return Err(crate::PyError::type_error(
            "Tuple or struct_time argument required",
        ));
    };

    unsafe {
        let len = w_tuple_len(tup);
        if len != 9 {
            return Err(crate::PyError::type_error(
                "time.struct_time() takes a sequence of length 9",
            ));
        }
        let get = |i: usize| -> i32 {
            let item = w_tuple_getitem(tup, i as i64).unwrap();
            if is_int(item) {
                w_int_get_value(item) as i32
            } else if is_float(item) {
                floatobject::w_float_get_value(item) as i32
            } else {
                0
            }
        };
        let mut tm: libc::tm = std::mem::zeroed();
        tm.tm_year = get(0) - 1900;
        tm.tm_mon = get(1) - 1;
        tm.tm_mday = get(2);
        tm.tm_hour = get(3);
        tm.tm_min = get(4);
        tm.tm_sec = get(5);
        tm.tm_wday = (get(6) + 1) % 7; // Python Monday=0 → C Sunday=0
        tm.tm_yday = get(7) - 1;
        tm.tm_isdst = get(8);
        Ok(tm)
    }
}

/// time.localtime([seconds]) — interp_time.localtime
pub fn localtime(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let seconds = _get_seconds(args);
    let tm =
        _c_localtime(seconds).ok_or_else(|| crate::PyError::value_error("unconvertible time"))?;
    Ok(_tm_to_tuple(&tm))
}

/// time.gmtime([seconds]) — interp_time.gmtime
pub fn gmtime(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let seconds = _get_seconds(args);
    let tm = _c_gmtime(seconds).ok_or_else(|| crate::PyError::value_error("unconvertible time"))?;
    Ok(_tm_to_tuple(&tm))
}

/// time.strftime(format[, tuple]) — interp_time.strftime
pub fn strftime(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let fmt = args
        .first()
        .copied()
        .ok_or_else(|| crate::PyError::type_error("strftime() requires at least one argument"))?;

    let tm = _gettmarg(&args[1..], true)?;

    let fmt_str = unsafe {
        if !is_str(fmt) {
            return Err(crate::PyError::type_error(
                "strftime() argument 1 must be str",
            ));
        }
        w_str_get_value(fmt)
    };

    let c_fmt = std::ffi::CString::new(fmt_str)
        .map_err(|_| crate::PyError::value_error("embedded null in format string"))?;

    let mut buf = vec![0u8; 256];
    unsafe {
        loop {
            let n = libc::strftime(
                buf.as_mut_ptr() as *mut libc::c_char,
                buf.len(),
                c_fmt.as_ptr(),
                &tm,
            );
            if n != 0 {
                let s = std::str::from_utf8_unchecked(&buf[..n]);
                return Ok(w_str_new(s));
            }
            if buf.len() > 16384 {
                return Ok(w_str_new(""));
            }
            buf.resize(buf.len() * 2, 0);
        }
    }
}

/// time.mktime(tuple) — interp_time.mktime
pub fn mktime(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mut tm = _gettmarg(args, false)?;
    tm.tm_wday = -1;
    let tt = unsafe { libc::mktime(&mut tm) };
    if tt == -1 && tm.tm_wday == -1 {
        return Err(crate::PyError::overflow_error(
            "mktime argument out of range",
        ));
    }
    Ok(floatobject::w_float_new(tt as f64))
}

/// time.asctime([tuple]) — interp_time.asctime
pub fn asctime(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let tm = _gettmarg(args, true)?;
    let p = unsafe { libc::asctime(&tm) };
    if p.is_null() {
        return Err(crate::PyError::value_error("unconvertible time"));
    }
    let s = unsafe { std::ffi::CStr::from_ptr(p) }
        .to_str()
        .unwrap_or("")
        .trim_end_matches('\n');
    Ok(w_str_new(s))
}

/// time.ctime([seconds]) — interp_time.ctime
pub fn ctime(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let seconds = _get_seconds(args);
    let mut t = seconds;
    let p = unsafe { libc::ctime(&mut t) };
    if p.is_null() {
        return Err(crate::PyError::value_error("unconvertible time"));
    }
    let s = unsafe { std::ffi::CStr::from_ptr(p) }
        .to_str()
        .unwrap_or("")
        .trim_end_matches('\n');
    Ok(w_str_new(s))
}
