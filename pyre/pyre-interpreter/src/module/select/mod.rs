//! select module — PyPy: pypy/module/select/
//!
//! Implements `select.select(rlist, wlist, xlist, timeout=None)` via
//! `rustpython_host_env::select`.  poll/epoll/kqueue object types are
//! not implemented yet.

pub mod interp_select;
pub mod moduledef;
