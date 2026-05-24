//! time module definition.
//!
//! PyPy equivalent: pypy/module/time/moduledef.py

use crate::{
    DictStorage, dict_storage_store, make_builtin_function, make_builtin_function_with_arity,
};

use super::interp_time;

pub fn init(ns: &mut DictStorage) {
    dict_storage_store(
        ns,
        "time",
        make_builtin_function_with_arity("time", interp_time::time, 0),
    );
    dict_storage_store(
        ns,
        "time_ns",
        make_builtin_function_with_arity("time_ns", interp_time::time_ns, 0),
    );
    dict_storage_store(
        ns,
        "monotonic",
        make_builtin_function_with_arity("monotonic", interp_time::monotonic, 0),
    );
    dict_storage_store(
        ns,
        "sleep",
        make_builtin_function_with_arity("sleep", interp_time::sleep, 1),
    );
    dict_storage_store(
        ns,
        "perf_counter",
        make_builtin_function_with_arity("perf_counter", interp_time::perf_counter, 0),
    );
    dict_storage_store(
        ns,
        "localtime",
        make_builtin_function("localtime", interp_time::localtime),
    );
    dict_storage_store(
        ns,
        "gmtime",
        make_builtin_function("gmtime", interp_time::gmtime),
    );
    dict_storage_store(
        ns,
        "strftime",
        make_builtin_function("strftime", interp_time::strftime),
    );
    dict_storage_store(
        ns,
        "mktime",
        make_builtin_function_with_arity("mktime", interp_time::mktime, 1),
    );
    dict_storage_store(
        ns,
        "asctime",
        make_builtin_function("asctime", interp_time::asctime),
    );
    dict_storage_store(
        ns,
        "ctime",
        make_builtin_function("ctime", interp_time::ctime),
    );
    dict_storage_store(ns, "timezone", pyre_object::w_int_new(0));
    dict_storage_store(ns, "altzone", pyre_object::w_int_new(0));
    dict_storage_store(ns, "daylight", pyre_object::w_int_new(0));
    dict_storage_store(
        ns,
        "tzname",
        pyre_object::w_tuple_new(vec![
            pyre_object::w_str_new("UTC"),
            pyre_object::w_str_new("UTC"),
        ]),
    );

    // POSIX clock identifiers + clock_gettime / clock_getres
    // (Unix host_env path only — Windows path uses different timers and
    // CPython exposes a different surface there.)
    #[cfg(all(unix, feature = "host_env"))]
    {
        dict_storage_store(
            ns,
            "clock_gettime",
            make_builtin_function_with_arity("clock_gettime", interp_time::clock_gettime, 1),
        );
        dict_storage_store(
            ns,
            "clock_gettime_ns",
            make_builtin_function_with_arity("clock_gettime_ns", interp_time::clock_gettime_ns, 1),
        );
        #[cfg(not(target_os = "redox"))]
        {
            dict_storage_store(
                ns,
                "clock_getres",
                make_builtin_function_with_arity("clock_getres", interp_time::clock_getres, 1),
            );
            dict_storage_store(
                ns,
                "clock_settime",
                make_builtin_function_with_arity("clock_settime", interp_time::clock_settime, 2),
            );
            dict_storage_store(
                ns,
                "clock_settime_ns",
                make_builtin_function_with_arity(
                    "clock_settime_ns",
                    interp_time::clock_settime_ns,
                    2,
                ),
            );
        }
        dict_storage_store(
            ns,
            "CLOCK_REALTIME",
            pyre_object::w_int_new(libc::CLOCK_REALTIME as i64),
        );
        dict_storage_store(
            ns,
            "CLOCK_MONOTONIC",
            pyre_object::w_int_new(libc::CLOCK_MONOTONIC as i64),
        );
        #[cfg(not(any(
            target_os = "illumos",
            target_os = "netbsd",
            target_os = "solaris",
            target_os = "openbsd",
            target_os = "wasi",
        )))]
        dict_storage_store(
            ns,
            "CLOCK_PROCESS_CPUTIME_ID",
            pyre_object::w_int_new(libc::CLOCK_PROCESS_CPUTIME_ID as i64),
        );
        #[cfg(not(any(
            target_os = "illumos",
            target_os = "netbsd",
            target_os = "solaris",
            target_os = "openbsd",
            target_os = "redox",
        )))]
        dict_storage_store(
            ns,
            "CLOCK_THREAD_CPUTIME_ID",
            pyre_object::w_int_new(libc::CLOCK_THREAD_CPUTIME_ID as i64),
        );
    }
}
