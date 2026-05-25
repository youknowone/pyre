//! unicodedata module — PyPy: `pypy/module/unicodedata/`.
//!
//! Stub providing `normalize` / `category` / `name` / `lookup` /
//! `decimal` / `numeric` — enough to let `import unicodedata` succeed.
//! `category` returns `"Cn"` (unassigned) for every code point;
//! `normalize` is identity; `name` / `decimal` / `numeric` return the
//! caller-supplied default if any, else raise.

use pyre_object::*;

fn normalize(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() >= 2 {
        Ok(args[1])
    } else {
        Ok(w_str_new(""))
    }
}

fn category(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(w_str_new("Cn"))
}

fn name(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() >= 2 {
        Ok(args[1])
    } else {
        Err(crate::PyError::value_error("no such name"))
    }
}

fn lookup(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Err(crate::PyError::key_error("character not found"))
}

fn decimal(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() >= 2 {
        Ok(args[1])
    } else {
        Err(crate::PyError::value_error("not a decimal"))
    }
}

fn numeric(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() >= 2 {
        Ok(args[1])
    } else {
        Err(crate::PyError::value_error("not a numeric character"))
    }
}

crate::py_module! {
    "unicodedata",
    interpleveldefs: {
        "unidata_version" => w_str_new("15.1.0"),
    },
    functions: {
        "normalize" / 2 = normalize,
        "category"  / 1 = category,
        "name"      / * = name,
        "lookup"    / 1 = lookup,
        "decimal"   / * = decimal,
        "numeric"   / * = numeric,
    },
}
