//! time module definition.
//!
//! PyPy equivalent: pypy/module/time/moduledef.py

use crate::{PyNamespace, make_builtin_function, namespace_store};

use super::interp_time;

pub fn init(ns: &mut PyNamespace) {
    namespace_store(ns, "time", make_builtin_function("time", interp_time::time));
    namespace_store(
        ns,
        "time_ns",
        make_builtin_function("time_ns", interp_time::time_ns),
    );
    namespace_store(
        ns,
        "monotonic",
        make_builtin_function("monotonic", interp_time::monotonic),
    );
    namespace_store(
        ns,
        "sleep",
        make_builtin_function("sleep", interp_time::sleep),
    );
    namespace_store(
        ns,
        "perf_counter",
        make_builtin_function("perf_counter", interp_time::perf_counter),
    );
    namespace_store(
        ns,
        "localtime",
        make_builtin_function("localtime", interp_time::localtime),
    );
    namespace_store(
        ns,
        "gmtime",
        make_builtin_function("gmtime", interp_time::gmtime),
    );
    namespace_store(
        ns,
        "strftime",
        make_builtin_function("strftime", interp_time::strftime),
    );
    namespace_store(
        ns,
        "mktime",
        make_builtin_function("mktime", interp_time::mktime),
    );
    namespace_store(
        ns,
        "asctime",
        make_builtin_function("asctime", interp_time::asctime),
    );
    namespace_store(
        ns,
        "ctime",
        make_builtin_function("ctime", interp_time::ctime),
    );
    namespace_store(ns, "timezone", pyre_object::w_int_new(0));
    namespace_store(ns, "altzone", pyre_object::w_int_new(0));
    namespace_store(ns, "daylight", pyre_object::w_int_new(0));
    namespace_store(
        ns,
        "tzname",
        pyre_object::w_tuple_new(vec![
            pyre_object::w_str_new("UTC"),
            pyre_object::w_str_new("UTC"),
        ]),
    );
}
