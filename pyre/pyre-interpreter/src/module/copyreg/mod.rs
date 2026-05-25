//! copyreg module — PyPy: `pypy/module/copyreg/`.
//!
//! Stub surface — pyre does not support pickle.  `copyreg.pickle` is a
//! no-op that returns None; `copyreg.dispatch_table` is an empty dict
//! so callers' `dispatch_table[type]` lookups raise KeyError as they
//! would on CPython with no registered reducer.

use pyre_object::*;

// `copyreg.pickle(type, reduce_func, constructor=None)` — register a
// pickle reducer.  Stub ignores the call.
fn copyreg_pickle(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_none())
}

crate::py_module! {
    "copyreg",
    interpleveldefs: {
        "dispatch_table" => w_dict_new(),
    },
    functions: {
        "pickle" / 3 = copyreg_pickle,
    },
}
