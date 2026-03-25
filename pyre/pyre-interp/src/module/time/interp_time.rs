//! Time function implementations.
//!
//! PyPy equivalent: pypy/module/time/interp_time.py

use pyre_object::*;
use std::time::{SystemTime, UNIX_EPOCH};

/// time.time() → float (seconds since epoch)
pub fn time(args: &[PyObjectRef]) -> PyObjectRef {
    let _ = args;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    floatobject::w_float_new(now.as_secs_f64())
}

/// time.time_ns() → int (nanoseconds since epoch)
pub fn time_ns(args: &[PyObjectRef]) -> PyObjectRef {
    let _ = args;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    w_int_new(now.as_nanos() as i64)
}

/// time.monotonic() → float
pub fn monotonic(args: &[PyObjectRef]) -> PyObjectRef {
    let _ = args;
    // Simplified: use system time as monotonic approximation
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    floatobject::w_float_new(now.as_secs_f64())
}

/// time.sleep(seconds)
pub fn sleep(args: &[PyObjectRef]) -> PyObjectRef {
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
    std::thread::sleep(std::time::Duration::from_secs_f64(secs));
    w_none()
}

/// time.perf_counter() → float
pub fn perf_counter(args: &[PyObjectRef]) -> PyObjectRef {
    let _ = args;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    floatobject::w_float_new(now.as_secs_f64())
}
