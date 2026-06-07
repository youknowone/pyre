//! Time function implementations.
//!
//! PyPy equivalent: pypy/module/time/interp_time.py

use pyre_object::*;

#[cfg(feature = "host_env")]
use rustpython_host_env::time as host_time;
use std::sync::OnceLock;
use std::time::Instant;
#[cfg(not(feature = "host_env"))]
use std::time::{SystemTime, UNIX_EPOCH};

/// Number of fields in `time.struct_time` as seen by C extensions.
/// `interp_time.py:290` uses 9, raised to 11 when `struct tm` carries
/// `tm_zone`/`tm_gmtoff` — true on every Unix target pyre builds for.
#[cfg(unix)]
pub const STRUCT_TM_ITEMS: i64 = 11;
#[cfg(not(unix))]
pub const STRUCT_TM_ITEMS: i64 = 9;

/// Process-start `Instant` used as the monotonic-clock origin whenever
/// `clock_gettime(CLOCK_MONOTONIC)` is unavailable.  Keeping a single
/// baseline preserves `time.monotonic()`'s non-decreasing guarantee.
fn monotonic_baseline() -> Instant {
    static BASELINE: OnceLock<Instant> = OnceLock::new();
    *BASELINE.get_or_init(Instant::now)
}

fn monotonic_seconds() -> f64 {
    #[cfg(all(unix, feature = "host_env"))]
    {
        if let Ok(d) = host_time::clock_gettime(host_time::ClockId::CLOCK_MONOTONIC) {
            return d.as_secs_f64();
        }
    }
    monotonic_baseline().elapsed().as_secs_f64()
}

/// Wall-clock seconds since the unix epoch, falling back to 0 on
/// `SystemTimeError`.  Routes through `host_env::time` when enabled.
fn duration_since_epoch() -> std::time::Duration {
    #[cfg(feature = "host_env")]
    {
        host_time::duration_since_system_now().unwrap_or_default()
    }
    #[cfg(not(feature = "host_env"))]
    {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
    }
}

/// time.time() → float (seconds since epoch)
pub fn time(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _ = args;
    Ok(floatobject::w_float_new(
        duration_since_epoch().as_secs_f64(),
    ))
}

/// time.time_ns() → int (nanoseconds since epoch)
pub fn time_ns(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _ = args;
    Ok(w_int_new(duration_since_epoch().as_nanos() as i64))
}

/// time.monotonic() → float
///
/// Prefers `clock_gettime(CLOCK_MONOTONIC)` on Unix; otherwise reads
/// the process-start `Instant` baseline.  `std::time::Instant` is the
/// platform's monotonic clock on every Rust target, so the fallback
/// preserves `time.monotonic`'s non-decreasing guarantee even when the
/// syscall is unavailable.
pub fn monotonic(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _ = args;
    Ok(floatobject::w_float_new(monotonic_seconds()))
}

/// time.sleep(seconds)
///
/// Routes through `host_env::time::nanosleep` on Unix when available so
/// that signal-driven wakeups propagate; falls back to
/// `std::thread::sleep` otherwise.
pub fn sleep(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "sleep() takes exactly one argument");
    // `timeutils.py:20-38 timestamp_w` — a float rejects NaN; everything
    // else goes through `space.bigint_w`, so bools are accepted (int
    // subclass) and any other type raises TypeError.
    let secs = unsafe {
        if is_float(args[0]) {
            let s = floatobject::w_float_get_value(args[0]);
            if s.is_nan() {
                return Err(crate::PyError::value_error("timestamp is nan"));
            }
            s
        } else if is_int(args[0]) {
            w_int_get_value(args[0]) as f64
        } else if is_bool(args[0]) {
            if boolobject::w_bool_get_value(args[0]) {
                1.0
            } else {
                0.0
            }
        } else {
            let name = crate::typedef::r#type(args[0])
                .map(|tp| w_type_get_name(tp).to_string())
                .unwrap_or_else(|| "object".to_string());
            return Err(crate::PyError::type_error(format!(
                "'{name}' object cannot be interpreted as an integer"
            )));
        }
    };
    if secs < 0.0 {
        return Err(crate::PyError::value_error(
            "sleep length must be non-negative",
        ));
    }
    if secs == 0.0 {
        return Ok(w_none());
    }
    let dur = std::time::Duration::from_secs_f64(secs);
    #[cfg(all(unix, feature = "host_env"))]
    {
        // `interp_time.py:622-710 time_sleep` — sleep toward a monotonic
        // deadline; on EINTR deliver any pending signal and retry with the
        // remaining time, breaking once the deadline has passed.
        let deadline = std::time::Instant::now() + dur;
        let mut remaining = dur;
        loop {
            match host_time::nanosleep(remaining) {
                Ok(()) => return Ok(w_none()),
                Err(e) if e.raw_os_error() == Some(libc::EINTR) => {
                    crate::module::_signal::interp_signal::checksignals_now()?;
                    let now = std::time::Instant::now();
                    if now >= deadline {
                        return Ok(w_none());
                    }
                    remaining = deadline - now;
                }
                Err(e) => {
                    return Err(crate::PyError::os_error_with_errno(
                        e.raw_os_error().unwrap_or(0),
                        format!("sleep: {e}"),
                    ));
                }
            }
        }
    }
    #[cfg(not(all(unix, feature = "host_env")))]
    {
        std::thread::sleep(dur);
        Ok(w_none())
    }
}

/// time.perf_counter() → float
///
/// Shares `monotonic_seconds()` with `time.monotonic`; the two clocks
/// are nominally separate but pyre exposes the same monotonic source so
/// that `perf_counter` is also non-decreasing across calls.
pub fn perf_counter(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _ = args;
    Ok(floatobject::w_float_new(monotonic_seconds()))
}

/// Monotonic clock as integer nanoseconds.
fn monotonic_nanos() -> i128 {
    #[cfg(all(unix, feature = "host_env"))]
    {
        if let Ok(d) = host_time::clock_gettime(host_time::ClockId::CLOCK_MONOTONIC) {
            return d.as_nanos() as i128;
        }
    }
    monotonic_baseline().elapsed().as_nanos() as i128
}

/// time.monotonic_ns() → int
pub fn monotonic_ns(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _ = args;
    Ok(w_int_new(monotonic_nanos() as i64))
}

/// time.perf_counter_ns() → int
///
/// Shares the monotonic nanosecond source with `time.perf_counter`.
pub fn perf_counter_ns(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _ = args;
    Ok(w_int_new(monotonic_nanos() as i64))
}

/// Process CPU time (kernel + user) as nanoseconds.
///
/// Prefers `clock_gettime(CLOCK_PROCESS_CPUTIME_ID)`; falls back to
/// `getrusage(RUSAGE_SELF)` summing `ru_utime` and `ru_stime`.  With no
/// usable process clock, `_clock_impl` raises RuntimeError rather than
/// reporting a bogus zero.
#[cfg(all(unix, feature = "host_env"))]
fn process_time_nanos() -> Result<i128, crate::PyError> {
    if let Ok(d) = host_time::clock_gettime(host_time::ClockId::CLOCK_PROCESS_CPUTIME_ID) {
        return Ok(d.as_nanos() as i128);
    }
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } == 0 {
        let tv_ns = |tv: &libc::timeval| -> i128 {
            tv.tv_sec as i128 * 1_000_000_000 + tv.tv_usec as i128 * 1_000
        };
        return Ok(tv_ns(&usage.ru_utime) + tv_ns(&usage.ru_stime));
    }
    Err(crate::PyError::runtime_error(
        "the processor time used is not available or its value cannot be represented",
    ))
}

#[cfg(not(all(unix, feature = "host_env")))]
fn process_time_nanos() -> Result<i128, crate::PyError> {
    // No host clock available; fall back to the monotonic baseline so
    // the value is still non-decreasing.
    Ok(monotonic_baseline().elapsed().as_nanos() as i128)
}

/// time.process_time() → float
///
/// Process time for profiling: sum of the kernel and user-space CPU time.
pub fn process_time(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _ = args;
    Ok(floatobject::w_float_new(
        process_time_nanos()? as f64 * 1e-9,
    ))
}

/// time.process_time_ns() → int
pub fn process_time_ns(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let _ = args;
    Ok(w_int_new(process_time_nanos()? as i64))
}

/// time.clock_gettime(clk_id) → float seconds
#[cfg(all(unix, feature = "host_env"))]
pub fn clock_gettime(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "clock_gettime() missing argument",
        ));
    }
    if !unsafe { is_int(args[0]) } {
        return Err(crate::PyError::type_error("clock id must be an integer"));
    }
    let id = unsafe { w_int_get_value(args[0]) } as libc::clockid_t;
    let d = host_time::clock_gettime(host_time::ClockId::from_raw(id)).map_err(|e| {
        crate::PyError::os_error_with_errno(
            e.raw_os_error().unwrap_or(0),
            format!("clock_gettime: {e}"),
        )
    })?;
    Ok(floatobject::w_float_new(d.as_secs_f64()))
}

/// time.clock_gettime_ns(clk_id) → int nanoseconds
#[cfg(all(unix, feature = "host_env"))]
pub fn clock_gettime_ns(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "clock_gettime_ns() missing argument",
        ));
    }
    if !unsafe { is_int(args[0]) } {
        return Err(crate::PyError::type_error("clock id must be an integer"));
    }
    let id = unsafe { w_int_get_value(args[0]) } as libc::clockid_t;
    let d = host_time::clock_gettime(host_time::ClockId::from_raw(id)).map_err(|e| {
        crate::PyError::os_error_with_errno(
            e.raw_os_error().unwrap_or(0),
            format!("clock_gettime_ns: {e}"),
        )
    })?;
    Ok(w_int_new(d.as_nanos() as i64))
}

/// time.clock_settime(clk_id, time: float) → None
#[cfg(all(unix, feature = "host_env", not(target_os = "redox")))]
pub fn clock_settime(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Err(crate::PyError::type_error(
            "clock_settime() requires 2 arguments",
        ));
    }
    if !unsafe { is_int(args[0]) } {
        return Err(crate::PyError::type_error("clock id must be an integer"));
    }
    let id = unsafe { w_int_get_value(args[0]) } as libc::clockid_t;
    let secs = unsafe {
        if is_int(args[1]) {
            w_int_get_value(args[1]) as f64
        } else if is_float(args[1]) {
            floatobject::w_float_get_value(args[1])
        } else {
            return Err(crate::PyError::type_error(
                "clock_settime: time must be a real number",
            ));
        }
    };
    // Build a signed timespec: `integer_secs = rffi.cast(c_tv_sec, secs)`
    // truncates toward zero, `frac = secs - integer_secs`.
    let integer_secs = secs.trunc();
    let frac = secs - integer_secs;
    let ts = libc::timespec {
        tv_sec: integer_secs as libc::time_t,
        tv_nsec: (frac * 1e9) as libc::c_long,
    };
    let ret = unsafe { libc::clock_settime(id, &ts) };
    if ret != 0 {
        let e = std::io::Error::last_os_error();
        return Err(crate::PyError::os_error_with_errno(
            e.raw_os_error().unwrap_or(0),
            format!("clock_settime: {e}"),
        ));
    }
    Ok(w_none())
}

/// time.clock_settime_ns(clk_id, time: int) → None
#[cfg(all(unix, feature = "host_env", not(target_os = "redox")))]
pub fn clock_settime_ns(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Err(crate::PyError::type_error(
            "clock_settime_ns() requires 2 arguments",
        ));
    }
    if !unsafe { is_int(args[0]) } || !unsafe { is_int(args[1]) } {
        return Err(crate::PyError::type_error(
            "clock_settime_ns: clock id and time must be integers",
        ));
    }
    let id = unsafe { w_int_get_value(args[0]) } as libc::clockid_t;
    let ns = unsafe { w_int_get_value(args[1]) };
    // `tv_sec = ns // 10**9`, `tv_nsec = ns % 10**9` (Python floor div/mod,
    // so a negative `ns` normalises to a non-negative `tv_nsec`).
    let ts = libc::timespec {
        tv_sec: ns.div_euclid(1_000_000_000) as libc::time_t,
        tv_nsec: ns.rem_euclid(1_000_000_000) as libc::c_long,
    };
    let ret = unsafe { libc::clock_settime(id, &ts) };
    if ret != 0 {
        let e = std::io::Error::last_os_error();
        return Err(crate::PyError::os_error_with_errno(
            e.raw_os_error().unwrap_or(0),
            format!("clock_settime_ns: {e}"),
        ));
    }
    Ok(w_none())
}

/// time.clock_getres(clk_id) → float seconds
#[cfg(all(unix, feature = "host_env", not(target_os = "redox")))]
pub fn clock_getres(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "clock_getres() missing argument",
        ));
    }
    if !unsafe { is_int(args[0]) } {
        return Err(crate::PyError::type_error("clock id must be an integer"));
    }
    let id = unsafe { w_int_get_value(args[0]) } as libc::clockid_t;
    let d = host_time::clock_getres(host_time::ClockId::from_raw(id)).map_err(|e| {
        crate::PyError::os_error_with_errno(
            e.raw_os_error().unwrap_or(0),
            format!("clock_getres: {e}"),
        )
    })?;
    Ok(floatobject::w_float_new(d.as_secs_f64()))
}

// ── libc tm helpers ──────────────────────────────────────────────────

// On Windows, libc::tm does not exist, and the time functions have
// different signatures.  We define a portable Tm struct and platform
// shims so the rest of the module stays identical.

/// Portable `struct tm` representation used across platforms.
#[derive(Clone)]
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
    /// `tm_gmtoff` / `tm_zone` — the `struct tm` fields exposed when
    /// `HAS_TM_ZONE` (every Unix target); default elsewhere.
    pub tm_gmtoff: i64,
    pub tm_zone: String,
}

/// Portable time_t alias.
#[allow(non_camel_case_types)]
type time_t = i64;

#[cfg(feature = "host_env")]
fn _c_gmtime(seconds: time_t) -> Result<c_tm, crate::PyError> {
    host_time::gmtime_from_timestamp(seconds as host_time::TimeT)
        .map(|tm| libc_tm_to_c_tm(&tm))
        .ok_or_else(|| crate::PyError::value_error("unconvertible time"))
}

// `interp_time.py c_gmtime` libc backend, used when the host_env
// abstraction layer is disabled.  Mirrors PyPy's rffi.llexternal call
// to libc gmtime_r (Unix) / _gmtime64_s (Windows CRT).
#[cfg(all(unix, not(feature = "host_env")))]
fn _c_gmtime(seconds: time_t) -> Result<c_tm, crate::PyError> {
    let t = seconds as libc::time_t;
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    let p = unsafe { libc::gmtime_r(&t, &mut tm) };
    if p.is_null() {
        return Err(crate::PyError::value_error("unconvertible time"));
    }
    Ok(libc_tm_to_c_tm(&tm))
}

#[cfg(all(windows, not(feature = "host_env")))]
fn _c_gmtime(seconds: time_t) -> Result<c_tm, crate::PyError> {
    unsafe extern "C" {
        fn _gmtime64_s(out: *mut MsvcTm, t: *const i64) -> i32;
    }
    let t = seconds;
    let mut tm: MsvcTm = unsafe { std::mem::zeroed() };
    let rc = unsafe { _gmtime64_s(&mut tm, &t) };
    if rc != 0 {
        return Err(crate::PyError::value_error("unconvertible time"));
    }
    Ok(msvc_tm_to_c_tm(&tm))
}

#[cfg(feature = "host_env")]
fn _c_localtime(seconds: time_t) -> Result<c_tm, crate::PyError> {
    host_time::localtime_from_timestamp(seconds as host_time::TimeT)
        .map(|tm| libc_tm_to_c_tm(&tm))
        .ok_or_else(|| crate::PyError::value_error("unconvertible time"))
}

#[cfg(all(unix, not(feature = "host_env")))]
fn _c_localtime(seconds: time_t) -> Result<c_tm, crate::PyError> {
    let t = seconds as libc::time_t;
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    let p = unsafe { libc::localtime_r(&t, &mut tm) };
    if p.is_null() {
        return Err(crate::PyError::value_error("unconvertible time"));
    }
    Ok(libc_tm_to_c_tm(&tm))
}

#[cfg(all(windows, not(feature = "host_env")))]
fn _c_localtime(seconds: time_t) -> Result<c_tm, crate::PyError> {
    unsafe extern "C" {
        fn _localtime64_s(out: *mut MsvcTm, t: *const i64) -> i32;
    }
    let t = seconds;
    let mut tm: MsvcTm = unsafe { std::mem::zeroed() };
    let rc = unsafe { _localtime64_s(&mut tm, &t) };
    if rc != 0 {
        return Err(crate::PyError::value_error("unconvertible time"));
    }
    Ok(msvc_tm_to_c_tm(&tm))
}

// ── Unix helpers ────────────────────────────────────────────────────

fn libc_tm_to_c_tm(tm: &libc::tm) -> c_tm {
    // `tm_gmtoff` / `tm_zone` only exist on the Unix `struct tm`.
    #[cfg(unix)]
    let (tm_gmtoff, tm_zone) = (
        tm.tm_gmtoff as i64,
        if tm.tm_zone.is_null() {
            String::new()
        } else {
            unsafe { std::ffi::CStr::from_ptr(tm.tm_zone) }
                .to_string_lossy()
                .into_owned()
        },
    );
    #[cfg(not(unix))]
    let (tm_gmtoff, tm_zone): (i64, String) = (0, String::new());
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
        tm_gmtoff,
        tm_zone,
    }
}

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
        // The MSVC `struct tm` carries no zone fields (HAS_TM_ZONE false).
        tm_gmtoff: 0,
        tm_zone: String::new(),
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

/// `app_time.py:5-23 class struct_time(metaclass=structseqtype)` —
/// process-wide cached subclass-of-tuple type.  The 9-field positional
/// core; on Unix (`HAS_TM_ZONE`) `tm_zone` / `tm_gmtoff` are named-only
/// extras so `n_fields == _STRUCT_TM_ITEMS == 11`.
thread_local! {
    static STRUCT_TIME_TYPE: std::cell::OnceCell<PyObjectRef> =
        const { std::cell::OnceCell::new() };
}

pub(crate) fn struct_time_type() -> PyObjectRef {
    const SEQ: &[&str] = &[
        "tm_year", "tm_mon", "tm_mday", "tm_hour", "tm_min", "tm_sec", "tm_wday", "tm_yday",
        "tm_isdst",
    ];
    STRUCT_TIME_TYPE.with(|c| {
        *c.get_or_init(|| {
            #[cfg(unix)]
            {
                crate::structseq::make_struct_seq_with_extra(
                    "time.struct_time",
                    SEQ,
                    &["tm_zone", "tm_gmtoff"],
                )
            }
            #[cfg(not(unix))]
            {
                crate::structseq::make_struct_seq("time.struct_time", SEQ)
            }
        })
    })
}

/// Build a `time.struct_time` from our portable `c_tm`.
fn _tm_to_tuple(tm: &c_tm) -> PyObjectRef {
    let seq = vec![
        w_int_new((tm.tm_year + 1900) as i64),
        w_int_new((tm.tm_mon + 1) as i64),
        w_int_new(tm.tm_mday as i64),
        w_int_new(tm.tm_hour as i64),
        w_int_new(tm.tm_min as i64),
        w_int_new(tm.tm_sec as i64),
        w_int_new(((tm.tm_wday + 6) % 7) as i64), // Monday=0
        w_int_new((tm.tm_yday + 1) as i64),
        w_int_new(tm.tm_isdst as i64),
    ];
    // `_tm_to_tuple` — on Unix the zone fields are exposed as extras.
    #[cfg(unix)]
    {
        let extras = vec![
            ("tm_zone", pyre_object::w_str_new(&tm.tm_zone)),
            ("tm_gmtoff", w_int_new(tm.tm_gmtoff)),
        ];
        crate::structseq::new_instance_with_extra(struct_time_type(), seq, extras)
    }
    #[cfg(not(unix))]
    {
        crate::structseq::new_instance(struct_time_type(), seq)
    }
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
    duration_since_epoch().as_secs() as time_t
}

/// Extract a `c_tm` from a Python time tuple argument.
/// interp_time.py: _gettmarg
fn _gettmarg(args: &[PyObjectRef], default_now: bool) -> Result<c_tm, crate::PyError> {
    let tup = if let Some(&arg) = args.first() {
        if unsafe { is_none(arg) } {
            if default_now {
                return _c_localtime(_get_seconds(&[]));
            }
            return Err(crate::PyError::type_error(
                "Tuple or struct_time argument required",
            ));
        }
        arg
    } else if default_now {
        return _c_localtime(_get_seconds(&[]));
    } else {
        return Err(crate::PyError::type_error(
            "Tuple or struct_time argument required",
        ));
    };

    unsafe {
        let len = w_tuple_len(tup);
        if len < 9 {
            return Err(crate::PyError::type_error(format!(
                "argument must be sequence of at least length 9, not {len}"
            )));
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
            tm_gmtoff: 0,
            tm_zone: String::new(),
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
        // interp_time.py:830-841 — a sequence of length >=10 supplies
        // `tm_zone` (idx 9) and length >=11 supplies `tm_gmtoff` (idx 10).
        if len >= 10 {
            let item = w_tuple_getitem(tup, 9).unwrap();
            if is_str(item) {
                tm.tm_zone = w_str_get_value(item).to_string();
            }
        }
        if len >= 11 {
            tm.tm_gmtoff = get(10) as i64;
        }
        Ok(tm)
    }
}

/// time.localtime([seconds]) — interp_time.localtime
pub fn localtime(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let seconds = _get_seconds(args);
    let tm = _c_localtime(seconds)?;
    Ok(_tm_to_tuple(&tm))
}

/// time.gmtime([seconds]) — interp_time.gmtime
pub fn gmtime(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let seconds = _get_seconds(args);
    let tm = _c_gmtime(seconds)?;
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

    #[cfg(feature = "host_env")]
    let tt = {
        let mut libc_tm = c_tm_to_libc_tm(&tm);
        let result = host_time::mktime(&mut libc_tm);
        tm.tm_wday = libc_tm.tm_wday;
        result as i64
    };
    #[cfg(all(unix, not(feature = "host_env")))]
    let tt: i64 = {
        let mut libc_tm = c_tm_to_libc_tm(&tm);
        let r = unsafe { libc::mktime(&mut libc_tm) };
        tm.tm_wday = libc_tm.tm_wday;
        r as i64
    };
    #[cfg(all(windows, not(feature = "host_env")))]
    let tt: i64 = {
        unsafe extern "C" {
            fn _mktime64(out: *mut MsvcTm) -> i64;
        }
        let mut msvc_tm = c_tm_to_msvc_tm(&tm);
        let r = unsafe { _mktime64(&mut msvc_tm) };
        tm.tm_wday = msvc_tm.tm_wday;
        r
    };
    #[cfg(not(any(unix, windows)))]
    let tt: i64 = {
        return Err(crate::PyError::not_implemented(
            "time.mktime requires host_env feature on this platform",
        ));
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
        let lossy = unsafe { std::ffi::CStr::from_ptr(p as *const libc::c_char) }.to_string_lossy();
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
        let tm = _c_localtime(seconds)?;
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
