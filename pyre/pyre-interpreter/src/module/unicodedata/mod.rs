//! unicodedata module — PyPy: `pypy/module/unicodedata/`.
//!
//! Stub providing `normalize` / `category` / `name` / `lookup` /
//! `decimal` / `numeric` — enough to let `import unicodedata` succeed.
//! `category` returns `"Cn"` (unassigned) for every code point;
//! `normalize` is identity; `name` / `decimal` / `numeric` return the
//! caller-supplied default if any, else raise.

crate::py_module! {
    "unicodedata",
    interpleveldefs: {
        "normalize" => crate::make_builtin_function_with_arity(
            "normalize",
            |args| {
                if args.len() >= 2 {
                    Ok(args[1])
                } else {
                    Ok(pyre_object::w_str_new(""))
                }
            },
            2,
        ),
        "category" => crate::make_builtin_function_with_arity(
            "category", |_| Ok(pyre_object::w_str_new("Cn")), 1),
        "name" => crate::make_builtin_function("name", |args| {
            if args.len() >= 2 {
                Ok(args[1])
            } else {
                Err(crate::PyError::value_error("no such name"))
            }
        }),
        "lookup" => crate::make_builtin_function_with_arity(
            "lookup",
            |_| Err(crate::PyError::key_error("character not found")),
            1,
        ),
        "decimal" => crate::make_builtin_function("decimal", |args| {
            if args.len() >= 2 {
                Ok(args[1])
            } else {
                Err(crate::PyError::value_error("not a decimal"))
            }
        }),
        "numeric" => crate::make_builtin_function("numeric", |args| {
            if args.len() >= 2 {
                Ok(args[1])
            } else {
                Err(crate::PyError::value_error("not a numeric character"))
            }
        }),
        "unidata_version" => pyre_object::w_str_new("15.1.0"),
    }
}
