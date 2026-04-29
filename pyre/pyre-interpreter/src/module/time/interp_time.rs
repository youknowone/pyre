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

// On Windows, libc::tm does not exist, and the time functions have
// different signatures.  We define a portable Tm struct and platform
// shims so the rest of the module stays identical.

/// Portable `struct tm` representation used across platforms.
#[repr(C)]
#[derive(Clone, Copy)]
#[allow(non_camel_case_types)]
struct c_tm {
    pub tm_sec: i32,
    pub tm_min: i32,
    pub tm_hour: i32,
    pub tm_mday: i32,
    pub tm_mon: i32,
    pub tm_year: i32,
    pub tm_wday: i32,
    pub tm_yday: i32,
    pub tm_isdst: i32,
}

/// Portable time_t alias.
#[allow(non_camel_case_types)]
type time_t = i64;

#[cfg(unix)]
fn _c_gmtime(seconds: time_t) -> Option<c_tm> {
    unsafe {
        let secs = seconds as libc::time_t;
        let mut tm: libc::tm = std::mem::zeroed();
        let p = libc::gmtime_r(&secs, &mut tm);
        if p.is_null() {
            None
        } else {
            Some(libc_tm_to_c_tm(&tm))
        }
    }
}

#[cfg(windows)]
fn _c_gmtime(seconds: time_t) -> Option<c_tm> {
    // Windows MSVC CRT: _gmtime64_s(struct tm *result, const __time64_t *time)
    // Returns 0 on success.  The libc crate on Windows does not expose
    // gmtime_r, but the MSVC CRT provides _gmtime64_s via an extern.
    unsafe extern "C" {
        fn _gmtime64_s(result: *mut MsvcTm, time: *const i64) -> i32;
    }
    unsafe {
        let mut tm: MsvcTm = std::mem::zeroed();
        let ret = _gmtime64_s(&mut tm, &seconds);
        if ret != 0 {
            None
        } else {
            Some(msvc_tm_to_c_tm(&tm))
        }
    }
}

#[cfg(unix)]
fn _c_localtime(seconds: time_t) -> Option<c_tm> {
    unsafe {
        let secs = seconds as libc::time_t;
        let mut tm: libc::tm = std::mem::zeroed();
        let p = libc::localtime_r(&secs, &mut tm);
        if p.is_null() {
            None
        } else {
            Some(libc_tm_to_c_tm(&tm))
        }
    }
}

#[cfg(windows)]
fn _c_localtime(seconds: time_t) -> Option<c_tm> {
    unsafe extern "C" {
        fn _localtime64_s(result: *mut MsvcTm, time: *const i64) -> i32;
    }
    unsafe {
        let mut tm: MsvcTm = std::mem::zeroed();
        let ret = _localtime64_s(&mut tm, &seconds);
        if ret != 0 {
            None
        } else {
            Some(msvc_tm_to_c_tm(&tm))
        }
    }
}

// ── Unix helpers ────────────────────────────────────────────────────

#[cfg(unix)]
fn libc_tm_to_c_tm(tm: &libc::tm) -> c_tm {
    c_tm {
        tm_sec: tm.tm_sec,
        tm_min: tm.tm_min,
        tm_hour: tm.tm_hour,
        tm_mday: tm.tm_mday,
        tm_mon: tm.tm_mon,
        tm_year: tm.tm_year,
        tm_wday: tm.tm_wday,
        tm_yday: tm.tm_yday,
        tm_isdst: tm.tm_isdst,
    }
}

#[cfg(unix)]
fn c_tm_to_libc_tm(tm: &c_tm) -> libc::tm {
    unsafe {
        let mut out: libc::tm = std::mem::zeroed();
        out.tm_sec = tm.tm_sec;
        out.tm_min = tm.tm_min;
        out.tm_hour = tm.tm_hour;
        out.tm_mday = tm.tm_mday;
        out.tm_mon = tm.tm_mon;
        out.tm_year = tm.tm_year;
        out.tm_wday = tm.tm_wday;
        out.tm_yday = tm.tm_yday;
        out.tm_isdst = tm.tm_isdst;
        out
    }
}

// ── Windows helpers ─────────────────────────────────────────────────

#[cfg(windows)]
#[repr(C)]
#[allow(non_camel_case_types)]
struct MsvcTm {
    tm_sec: i32,
    tm_min: i32,
    tm_hour: i32,
    tm_mday: i32,
    tm_mon: i32,
    tm_year: i32,
    tm_wday: i32,
    tm_yday: i32,
    tm_isdst: i32,
}

#[cfg(windows)]
fn msvc_tm_to_c_tm(tm: &MsvcTm) -> c_tm {
    c_tm {
        tm_sec: tm.tm_sec,
        tm_min: tm.tm_min,
        tm_hour: tm.tm_hour,
        tm_mday: tm.tm_mday,
        tm_mon: tm.tm_mon,
        tm_year: tm.tm_year,
        tm_wday: tm.tm_wday,
        tm_yday: tm.tm_yday,
        tm_isdst: tm.tm_isdst,
    }
}

#[cfg(windows)]
fn c_tm_to_msvc_tm(tm: &c_tm) -> MsvcTm {
    MsvcTm {
        tm_sec: tm.tm_sec,
        tm_min: tm.tm_min,
        tm_hour: tm.tm_hour,
        tm_mday: tm.tm_mday,
        tm_mon: tm.tm_mon,
        tm_year: tm.tm_year,
        tm_wday: tm.tm_wday,
        tm_yday: tm.tm_yday,
        tm_isdst: tm.tm_isdst,
    }
}

/// Build a Python time tuple from our portable `c_tm`.
fn _tm_to_tuple(tm: &c_tm) -> PyObjectRef {
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
fn _get_seconds(args: &[PyObjectRef]) -> time_t {
    if let Some(&arg) = args.first() {
        unsafe {
            if !is_none(arg) {
                if is_int(arg) {
                    return w_int_get_value(arg) as time_t;
                }
                if is_float(arg) {
                    return floatobject::w_float_get_value(arg) as time_t;
                }
            }
        }
    }
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as time_t)
        .unwrap_or(0)
}

/// Extract a `c_tm` from a Python time tuple argument.
/// interp_time.py: _gettmarg
fn _gettmarg(args: &[PyObjectRef], default_now: bool) -> Result<c_tm, crate::PyError> {
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
        let mut tm = c_tm {
            tm_sec: 0,
            tm_min: 0,
            tm_hour: 0,
            tm_mday: 0,
            tm_mon: 0,
            tm_year: 0,
            tm_wday: 0,
            tm_yday: 0,
            tm_isdst: 0,
        };
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

    // strftime is available on both Unix and Windows CRT.
    #[cfg(unix)]
    {
        let libc_tm = c_tm_to_libc_tm(&tm);
        let mut buf = vec![0u8; 256];
        unsafe {
            loop {
                let n = libc::strftime(
                    buf.as_mut_ptr() as *mut libc::c_char,
                    buf.len(),
                    c_fmt.as_ptr(),
                    &libc_tm,
                );
                if n != 0 {
                    let s = String::from_utf8_lossy(&buf[..n]);
                    return Ok(w_str_new(&s));
                }
                if buf.len() > 16384 {
                    return Ok(w_str_new(""));
                }
                buf.resize(buf.len() * 2, 0);
            }
        }
    }
    #[cfg(windows)]
    {
        unsafe extern "C" {
            fn strftime(
                buf: *mut libc::c_char,
                maxsize: usize,
                format: *const libc::c_char,
                timeptr: *const MsvcTm,
            ) -> usize;
        }
        let msvc_tm = c_tm_to_msvc_tm(&tm);
        let mut buf = vec![0u8; 256];
        unsafe {
            loop {
                let n = strftime(
                    buf.as_mut_ptr() as *mut libc::c_char,
                    buf.len(),
                    c_fmt.as_ptr(),
                    &msvc_tm,
                );
                if n != 0 {
                    let s = String::from_utf8_lossy(&buf[..n]);
                    return Ok(w_str_new(&s));
                }
                if buf.len() > 16384 {
                    return Ok(w_str_new(""));
                }
                buf.resize(buf.len() * 2, 0);
            }
        }
    }
}

/// time.mktime(tuple) — interp_time.mktime
pub fn mktime(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let mut tm = _gettmarg(args, false)?;
    tm.tm_wday = -1;

    #[cfg(unix)]
    let tt = {
        let mut libc_tm = c_tm_to_libc_tm(&tm);
        let result = unsafe { libc::mktime(&mut libc_tm) };
        tm.tm_wday = libc_tm.tm_wday;
        result as i64
    };
    #[cfg(windows)]
    let tt = {
        unsafe extern "C" {
            fn _mktime64(timeptr: *mut MsvcTm) -> i64;
        }
        let mut msvc_tm = c_tm_to_msvc_tm(&tm);
        let result = unsafe { _mktime64(&mut msvc_tm) };
        tm.tm_wday = msvc_tm.tm_wday;
        result
    };

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
    _asctime_from_tm(&tm)
}

fn _asctime_from_tm(tm: &c_tm) -> Result<PyObjectRef, crate::PyError> {
    #[cfg(unix)]
    {
        let libc_tm = c_tm_to_libc_tm(&tm);
        let mut buf = [0 as libc::c_char; 26];
        let p = unsafe { libc::asctime_r(&libc_tm, buf.as_mut_ptr()) };
        if p.is_null() {
            return Err(crate::PyError::value_error("unconvertible time"));
        }
        let lossy =
            unsafe { std::ffi::CStr::from_ptr(p as *const libc::c_char) }.to_string_lossy();
        let s = lossy.trim_end_matches('\n');
        Ok(w_str_new(s))
    }
    #[cfg(windows)]
    {
        unsafe extern "C" {
            fn asctime(timeptr: *const MsvcTm) -> *const libc::c_char;
        }
        let msvc_tm = c_tm_to_msvc_tm(&tm);
        let p = unsafe { asctime(&msvc_tm) };
        if p.is_null() {
            return Err(crate::PyError::value_error("unconvertible time"));
        }
        let lossy = unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy();
        let s = lossy.trim_end_matches('\n');
        Ok(w_str_new(s))
    }
}

/// time.ctime([seconds]) — interp_time.ctime
pub fn ctime(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let seconds = _get_seconds(args);

    #[cfg(unix)]
    {
        let tm = _c_localtime(seconds)
            .ok_or_else(|| crate::PyError::value_error("unconvertible time"))?;
        _asctime_from_tm(&tm)
    }
    #[cfg(windows)]
    {
        unsafe extern "C" {
            fn _ctime64(time: *const i64) -> *const libc::c_char;
        }
        let t = seconds;
        let p = unsafe { _ctime64(&t) };
        if p.is_null() {
            return Err(crate::PyError::value_error("unconvertible time"));
        }
        let lossy = unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy();
        let s = lossy.trim_end_matches('\n');
        Ok(w_str_new(s))
    }
}
