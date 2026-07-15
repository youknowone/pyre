//! time module — PyPy: pypy/module/time/

pub mod interp_time;

use interp_time as t;

crate::py_module! {
    "time",
    interpleveldefs: {
        // `app_time.py:5-23 class struct_time` — exposed as `time.struct_time`.
        "struct_time" => t::struct_time_type(),
        // `interp_time.py:290` — 9 base fields plus tm_zone/tm_gmtoff when
        // the platform's `struct tm` carries them (always on the Unix
        // targets pyre supports).
        "_STRUCT_TM_ITEMS" => pyre_object::w_int_new(t::STRUCT_TM_ITEMS),
        "timezone"    => pyre_object::w_int_new(0),
        "altzone"     => pyre_object::w_int_new(0),
        "daylight"    => pyre_object::w_int_new(0),
        "tzname"      => pyre_object::w_tuple_new(vec![
            pyre_object::w_str_new("UTC"),
            pyre_object::w_str_new("UTC"),
        ]),
    },
    functions: {
        "time"         / 0 = t::time,
        "time_ns"      / 0 = t::time_ns,
        "monotonic"    / 0 = t::monotonic,
        "monotonic_ns" / 0 = t::monotonic_ns,
        "sleep"        / 1 = t::sleep,
        "perf_counter" / 0 = t::perf_counter,
        "perf_counter_ns" / 0 = t::perf_counter_ns,
        "process_time" / 0 = t::process_time,
        "process_time_ns" / 0 = t::process_time_ns,
        "localtime"    / * = t::localtime,
        "gmtime"       / * = t::gmtime,
        "strftime"     / * = t::strftime,
        "mktime"       / 1 = t::mktime,
        "asctime"      / * = t::asctime,
        "ctime"        / * = t::ctime,
    },
    extra_init: |ns| {
        // POSIX clock identifiers + clock_gettime / clock_getres
        // (Unix host_env path only — Windows uses different timers and
        // CPython exposes a different surface there.)
        #[cfg(all(unix, feature = "host_env"))]
        {
            crate::module_ns_store(ns, "clock_gettime",
                crate::make_builtin_function_with_arity("clock_gettime", t::clock_gettime, 1));
            crate::module_ns_store(ns, "clock_gettime_ns",
                crate::make_builtin_function_with_arity("clock_gettime_ns", t::clock_gettime_ns, 1));
            #[cfg(not(target_os = "redox"))]
            {
                crate::module_ns_store(ns, "clock_getres",
                    crate::make_builtin_function_with_arity("clock_getres", t::clock_getres, 1));
                // clock_settime{,_ns} set the system clock (a privileged
                // syscall that escapes mediation); omit them under sandbox.
                #[cfg(not(feature = "sandbox"))]
                {
                    crate::module_ns_store(ns, "clock_settime",
                        crate::make_builtin_function_with_arity("clock_settime", t::clock_settime, 2));
                    crate::module_ns_store(ns, "clock_settime_ns",
                        crate::make_builtin_function_with_arity("clock_settime_ns", t::clock_settime_ns, 2));
                }
            }
            crate::module_ns_store(ns, "CLOCK_REALTIME",
                pyre_object::w_int_new(libc::CLOCK_REALTIME as i64));
            crate::module_ns_store(ns, "CLOCK_MONOTONIC",
                pyre_object::w_int_new(libc::CLOCK_MONOTONIC as i64));
            #[cfg(not(any(
                target_os = "illumos",
                target_os = "netbsd",
                target_os = "solaris",
                target_os = "openbsd",
                target_os = "wasi",
            )))]
            crate::module_ns_store(ns, "CLOCK_PROCESS_CPUTIME_ID",
                pyre_object::w_int_new(libc::CLOCK_PROCESS_CPUTIME_ID as i64));
            #[cfg(not(any(
                target_os = "illumos",
                target_os = "netbsd",
                target_os = "solaris",
                target_os = "openbsd",
                target_os = "redox",
            )))]
            crate::module_ns_store(ns, "CLOCK_THREAD_CPUTIME_ID",
                pyre_object::w_int_new(libc::CLOCK_THREAD_CPUTIME_ID as i64));
        }
        // localtime/mktime/ctime/strftime consult $TZ + /etc/localtime (and
        // the LC_TIME locale DB), reading host state outside the controller;
        // gmtime (UTC) and asctime (fixed C format) stay pure.
        #[cfg(feature = "sandbox")]
        {
            fn tz_unavailable(
                _: &[pyre_object::PyObjectRef],
            ) -> Result<pyre_object::PyObjectRef, crate::PyError> {
                Err(crate::host_seam::stub("this time function"))
            }
            for name in ["localtime", "mktime", "ctime", "strftime"] {
                crate::module_ns_store(
                    ns,
                    name,
                    crate::make_builtin_function(name, tz_unavailable),
                );
            }
        }
        #[cfg(not(all(unix, feature = "host_env")))]
        let _ = ns;
    }
}
