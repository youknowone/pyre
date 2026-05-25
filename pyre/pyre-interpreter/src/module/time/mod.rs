//! time module — PyPy: pypy/module/time/

pub mod interp_time;

use interp_time as t;

crate::py_module! {
    "time",
    interpleveldefs: {
        "time"         => crate::make_builtin_function_with_arity("time",         t::time,         0),
        "time_ns"      => crate::make_builtin_function_with_arity("time_ns",      t::time_ns,      0),
        "monotonic"    => crate::make_builtin_function_with_arity("monotonic",    t::monotonic,    0),
        "sleep"        => crate::make_builtin_function_with_arity("sleep",        t::sleep,        1),
        "perf_counter" => crate::make_builtin_function_with_arity("perf_counter", t::perf_counter, 0),
        "localtime"    => crate::make_builtin_function("localtime", t::localtime),
        "gmtime"       => crate::make_builtin_function("gmtime",    t::gmtime),
        "strftime"     => crate::make_builtin_function("strftime",  t::strftime),
        "mktime"       => crate::make_builtin_function_with_arity("mktime", t::mktime, 1),
        "asctime"      => crate::make_builtin_function("asctime",   t::asctime),
        "ctime"        => crate::make_builtin_function("ctime",     t::ctime),

        // `app_time.py:5-23 class struct_time` — exposed as `time.struct_time`.
        "struct_time"  => t::struct_time_type(),
        "timezone"     => pyre_object::w_int_new(0),
        "altzone"      => pyre_object::w_int_new(0),
        "daylight"     => pyre_object::w_int_new(0),
        "tzname"       => pyre_object::w_tuple_new(vec![
            pyre_object::w_str_new("UTC"),
            pyre_object::w_str_new("UTC"),
        ]),
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
