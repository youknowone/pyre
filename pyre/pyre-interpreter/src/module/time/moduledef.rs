//! time module definition.
//!
//! PyPy equivalent: pypy/module/time/moduledef.py

use crate::{PyNamespace, namespace_store, w_builtin_func_new};

use super::interp_time;

pub fn init(ns: &mut PyNamespace) {
    namespace_store(ns, "time", w_builtin_func_new("time", interp_time::time));
    namespace_store(
        ns,
        "time_ns",
        w_builtin_func_new("time_ns", interp_time::time_ns),
    );
    namespace_store(
        ns,
        "monotonic",
        w_builtin_func_new("monotonic", interp_time::monotonic),
    );
    namespace_store(ns, "sleep", w_builtin_func_new("sleep", interp_time::sleep));
    namespace_store(
        ns,
        "perf_counter",
        w_builtin_func_new("perf_counter", interp_time::perf_counter),
    );
}
