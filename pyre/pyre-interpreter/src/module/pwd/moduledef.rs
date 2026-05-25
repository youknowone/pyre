//! pwd module definition — PyPy: pypy/module/pwd/moduledef.py

use crate::DictStorage;

pub fn init(ns: &mut DictStorage) {
    // `interp_pwd::register_module` is `#[cfg(unix)]`; PyPy's pwd
    // module is also Unix-only.  Leave the module dict empty on
    // Windows so `import pwd` still resolves to the builtin
    // module object.
    #[cfg(unix)]
    super::interp_pwd::register_module(ns);
    #[cfg(not(unix))]
    let _ = ns;
}
