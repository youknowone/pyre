//! time module definition.
//!
//! PyPy equivalent: pypy/module/time/moduledef.py

use crate::{PyNamespace, builtin_code_new, namespace_store};

use super::interp_time;

pub fn init(ns: &mut PyNamespace) {
    namespace_store(ns, "time", builtin_code_new("time", interp_time::time));
    namespace_store(
        ns,
        "time_ns",
        builtin_code_new("time_ns", interp_time::time_ns),
    );
    namespace_store(
        ns,
        "monotonic",
        builtin_code_new("monotonic", interp_time::monotonic),
    );
    namespace_store(ns, "sleep", builtin_code_new("sleep", interp_time::sleep));
    namespace_store(
        ns,
        "perf_counter",
        builtin_code_new("perf_counter", interp_time::perf_counter),
    );
}
