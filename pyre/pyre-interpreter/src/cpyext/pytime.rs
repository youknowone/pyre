//! The `PyTime_t` clock API.
//!
//! `PyTime_t` counts nanoseconds in a signed 64-bit integer. The entry points
//! answer 0 and store through `result` on success, -1 otherwise; the `Raw`
//! half differs only in leaving no exception set. pyre's clocks cannot fail,
//! so every one of them answers 0 — and each reads the same source its
//! `time` module counterpart does, which is what keeps `PyTime_Monotonic` and
//! `time.monotonic_ns()` on one timeline.

use std::ffi::{c_double, c_int};

use crate::module::time::interp_time;

/// A count of nanoseconds.
#[allow(non_camel_case_types)]
pub type PyTime_t = i64;

/// Nanoseconds in a second, the divisor `PyTime_AsSecondsDouble` splits on.
const SEC_TO_NS: PyTime_t = 1_000_000_000;

fn monotonic() -> PyTime_t {
    interp_time::monotonic_nanos() as PyTime_t
}

fn wall_clock() -> PyTime_t {
    interp_time::duration_since_epoch().as_nanos() as PyTime_t
}

/// `PyTime_AsSecondsDouble(t)` — nanoseconds as seconds.
///
/// A whole number of seconds is converted from the second count rather than
/// from the nanosecond one, so a timestamp that is exact stays exact instead
/// of picking up the rounding of a division by 1e9.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyTime_AsSecondsDouble(t: PyTime_t) -> c_double {
    if t % SEC_TO_NS == 0 {
        (t / SEC_TO_NS) as c_double
    } else {
        (t as c_double) / 1e9
    }
}

/// `PyTime_Monotonic(result)` — a clock that cannot go backwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyTime_Monotonic(result: *mut PyTime_t) -> c_int {
    unsafe { store(result, monotonic()) }
}

/// `PyTime_MonotonicRaw(result)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyTime_MonotonicRaw(result: *mut PyTime_t) -> c_int {
    unsafe { store(result, monotonic()) }
}

/// `PyTime_PerfCounter(result)` — the highest-resolution clock available,
/// which is the monotonic one here, as `time.perf_counter_ns()` also reads.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyTime_PerfCounter(result: *mut PyTime_t) -> c_int {
    unsafe { store(result, monotonic()) }
}

/// `PyTime_PerfCounterRaw(result)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyTime_PerfCounterRaw(result: *mut PyTime_t) -> c_int {
    unsafe { store(result, monotonic()) }
}

/// `PyTime_Time(result)` — nanoseconds since the epoch.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyTime_Time(result: *mut PyTime_t) -> c_int {
    unsafe { store(result, wall_clock()) }
}

/// `PyTime_TimeRaw(result)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn PyTime_TimeRaw(result: *mut PyTime_t) -> c_int {
    unsafe { store(result, wall_clock()) }
}

/// Write a reading back, answering the 0 every clock here returns.
///
/// A null `result` is the caller's error and reaches no clock: the entry
/// points are declared to take a pointer to write through, and upstream
/// dereferences it unconditionally.
unsafe fn store(result: *mut PyTime_t, value: PyTime_t) -> c_int {
    unsafe { *result = value };
    0
}

pub(super) fn ensure_linked() {
    std::hint::black_box(PyTime_AsSecondsDouble as *const ());
    std::hint::black_box(PyTime_Monotonic as *const ());
    std::hint::black_box(PyTime_MonotonicRaw as *const ());
    std::hint::black_box(PyTime_PerfCounter as *const ());
    std::hint::black_box(PyTime_PerfCounterRaw as *const ());
    std::hint::black_box(PyTime_Time as *const ());
    std::hint::black_box(PyTime_TimeRaw as *const ());
}
