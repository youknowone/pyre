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
}
