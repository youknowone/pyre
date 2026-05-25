//! grp module definition — PyPy: lib_pypy/grp.py (via cffi).

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    // `interp_grp::register_module` is `#[cfg(unix)]` — the libc
    // getgrgid / getgrnam / getgrall callees do not exist on
    // Windows (PyPy's `lib_pypy/grp.py` is itself Unix-only).
    // Leave the module dict empty on Windows so `import grp` still
    // resolves to the builtin module object.
    #[cfg(unix)]
    super::interp_grp::register_module(ns);
    #[cfg(not(unix))]
    let _ = ns;
}
