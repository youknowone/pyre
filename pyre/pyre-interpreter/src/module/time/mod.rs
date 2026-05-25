//! time module — PyPy: pypy/module/time/

pub mod interp_time;

use interp_time as t;

crate::py_module! {
    "time",
    interpleveldefs: {
        // `app_time.py:5-23 class struct_time` — exposed as `time.struct_time`.
        "struct_time" => t::struct_time_type(),
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
        "sleep"        / 1 = t::sleep,
        "perf_counter" / 0 = t::perf_counter,
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
            crate::dict_storage_store(ns, "clock_gettime",
                crate::make_builtin_function_with_arity("clock_gettime", t::clock_gettime, 1));
            crate::dict_storage_store(ns, "clock_gettime_ns",
                crate::make_builtin_function_with_arity("clock_gettime_ns", t::clock_gettime_ns, 1));
            #[cfg(not(target_os = "redox"))]
            {
                crate::dict_storage_store(ns, "clock_getres",
                    crate::make_builtin_function_with_arity("clock_getres", t::clock_getres, 1));
                crate::dict_storage_store(ns, "clock_settime",
                    crate::make_builtin_function_with_arity("clock_settime", t::clock_settime, 2));
                crate::dict_storage_store(ns, "clock_settime_ns",
                    crate::make_builtin_function_with_arity("clock_settime_ns", t::clock_settime_ns, 2));
            }
            crate::dict_storage_store(ns, "CLOCK_REALTIME",
                pyre_object::w_int_new(libc::CLOCK_REALTIME as i64));
            crate::dict_storage_store(ns, "CLOCK_MONOTONIC",
                pyre_object::w_int_new(libc::CLOCK_MONOTONIC as i64));
            #[cfg(not(any(
                target_os = "illumos",
                target_os = "netbsd",
                target_os = "solaris",
                target_os = "openbsd",
                target_os = "wasi",
            )))]
            crate::dict_storage_store(ns, "CLOCK_PROCESS_CPUTIME_ID",
                pyre_object::w_int_new(libc::CLOCK_PROCESS_CPUTIME_ID as i64));
            #[cfg(not(any(
                target_os = "illumos",
                target_os = "netbsd",
                target_os = "solaris",
                target_os = "openbsd",
                target_os = "redox",
            )))]
            crate::dict_storage_store(ns, "CLOCK_THREAD_CPUTIME_ID",
                pyre_object::w_int_new(libc::CLOCK_THREAD_CPUTIME_ID as i64));
        }
        #[cfg(not(all(unix, feature = "host_env")))]
        let _ = ns;
    }
}
