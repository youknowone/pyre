//! _signal module — PyPy: pypy/module/signal/
//!
//! signal() / getsignal() / set_wakeup_fd() remain stubs because the
//! real implementations need interpreter-side trampolines to invoke
//! Python handlers from a Rust signal context.  alarm / pause /
//! raise_signal / strsignal / valid_signals are full implementations
//! backed by `rustpython_host_env::signal`.

crate::pyre_module_init!(interp_signal);
