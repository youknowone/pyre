//! unicodedata module — PyPy: `pypy/module/unicodedata/`.
//!
//! Stub providing `normalize` / `category` / `name` / `lookup` /
//! `decimal` / `numeric` — enough to let `import unicodedata` succeed.
//! `category` returns `"Cn"` (unassigned) for every code point;
//! `normalize` is identity; `name` / `decimal` / `numeric` return the
//! caller-supplied default if any, else raise.

use pyre_object::*;

crate::py_module! {
    "unicodedata",
    interpleveldefs: {
        "unidata_version" => w_str_new("15.1.0"),
    },
    functions: {
        "normalize" / 2 = |args| Ok(args.get(1).copied().unwrap_or_else(|| w_str_new(""))),
        "category"  / 1 = |_| Ok(w_str_new("Cn")),
        "name"      / * = |args| args.get(1).copied()
            .ok_or_else(|| crate::PyError::value_error("no such name")),
        "lookup"    / 1 = |_| Err(crate::PyError::key_error("character not found")),
        "decimal"   / * = |args| args.get(1).copied()
            .ok_or_else(|| crate::PyError::value_error("not a decimal")),
        "numeric"   / * = |args| args.get(1).copied()
            .ok_or_else(|| crate::PyError::value_error("not a numeric character")),
    },
}
