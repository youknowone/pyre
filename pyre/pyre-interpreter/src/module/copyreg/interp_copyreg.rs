//! copyreg implementation — PyPy: pypy/module/copyreg/interp_copyreg.py
//!
//! Verbatim move of the inline block previously in importing.rs.

use crate::DictStorage;

/// copyreg stub — PyPy: pypy/module/copyreg/
pub fn register_module(ns: &mut DictStorage) {
    // copyreg.pickle(type, reduce_func, constructor=None) — register a
    // pickle reducer. Stub: ignore (pyre doesn't support pickle).
    crate::dict_storage_store(
        ns,
        "pickle",
        crate::make_builtin_function_with_arity("pickle", |_| Ok(pyre_object::w_none()), 3),
    );
    crate::dict_storage_store(ns, "dispatch_table", pyre_object::w_dict_new());
}
