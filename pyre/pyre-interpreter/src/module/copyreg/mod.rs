//! copyreg module — PyPy: `pypy/module/copyreg/`.
//!
//! Stub surface — pyre does not support pickle.  `copyreg.pickle` is a
//! no-op that returns None; `copyreg.dispatch_table` is an empty dict
//! so callers' `dispatch_table[type]` lookups raise KeyError as they
//! would on CPython with no registered reducer.

use pyre_object::*;

crate::py_module! {
    "copyreg",
    interpleveldefs: {
        "dispatch_table" => w_dict_new(),
    },
    functions: {
        // `copyreg.pickle(type, reduce_func, constructor=None)` — register
        // a pickle reducer.  `constructor` is optional, so the stub
        // accepts both 2- and 3-argument calls.
        "pickle" / * = |args| {
            if !(2..=3).contains(&args.len()) {
                return Err(crate::PyError::type_error(
                    "pickle() takes 2 or 3 arguments",
                ));
            }
            Ok(w_none())
        },
    },
}
