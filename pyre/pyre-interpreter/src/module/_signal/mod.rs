//! _signal module — PyPy: pypy/module/signal/
//!
//! `signal()` / `getsignal()` register real handlers via `signalstate`
//! (sigaction + a pending-signal flag) and deliver them through
//! `CheckSignalAction` at the next interpreter checkpoint.  alarm / pause
//! / raise_signal / strsignal / valid_signals are backed by
//! `rustpython_host_env::signal`.  `set_wakeup_fd` records the fd but the
//! handler does not yet write to it.

pub mod signalstate;

crate::pyre_module_init!(interp_signal);
