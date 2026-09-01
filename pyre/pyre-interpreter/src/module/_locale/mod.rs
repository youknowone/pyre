//! _locale module — PyPy: pypy/module/_locale/
//!
//! Provides the 'C' locale defaults so locale.py's `from _locale import *`
//! succeeds and Lib/locale.py exposes working `localeconv` / `setlocale`.

pub use pyre_native::locale as rlocale;

crate::pyre_module_init!(interp_locale);
