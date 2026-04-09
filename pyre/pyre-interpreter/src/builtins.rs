use malachite_bigint::BigInt;
use num_traits::ToPrimitive;

use crate::executioncontext::PyNamespace;
use crate::{PyDisplay, make_builtin_function, make_module_builtin_function};
use pyre_object::*;

/// Install the default builtins into a namespace.
pub fn install_default_builtins(namespace: &mut PyNamespace) {
    namespace.get_or_insert_with("print", || {
        make_module_builtin_function("print", builtin_print)
    });
    namespace.get_or_insert_with("range", || {
        make_module_builtin_function("range", builtin_range)
    });
    namespace.get_or_insert_with("len", || make_module_builtin_function("len", builtin_len));
    namespace.get_or_insert_with("abs", || make_module_builtin_function("abs", builtin_abs));
    namespace.get_or_insert_with("min", || make_module_builtin_function("min", builtin_min));
    namespace.get_or_insert_with("max", || make_module_builtin_function("max", builtin_max));
    namespace.get_or_insert_with("type", || crate::typedef::w_type());
    namespace.get_or_insert_with("isinstance", || {
        make_module_builtin_function("isinstance", builtin_isinstance)
    });
    namespace.get_or_insert_with("str", || crate::typedef::gettypeobject(&STR_TYPE));
    namespace.get_or_insert_with("repr", || {
        make_module_builtin_function("repr", builtin_repr)
    });
    namespace.get_or_insert_with("int", || crate::typedef::gettypeobject(&INT_TYPE));
    namespace.get_or_insert_with("float", || crate::typedef::gettypeobject(&FLOAT_TYPE));
    namespace.get_or_insert_with("bool", || crate::typedef::gettypeobject(&BOOL_TYPE));
    namespace.get_or_insert_with("True", || w_bool_from(true));
    namespace.get_or_insert_with("False", || w_bool_from(false));
    namespace.get_or_insert_with("None", || w_none());
    namespace.get_or_insert_with("NotImplemented", || w_not_implemented());
    namespace.get_or_insert_with("hasattr", || {
        make_module_builtin_function("hasattr", builtin_hasattr)
    });
    namespace.get_or_insert_with("getattr", || {
        make_module_builtin_function("getattr", builtin_getattr)
    });
    namespace.get_or_insert_with("setattr", || {
        make_module_builtin_function("setattr", builtin_setattr)
    });
    namespace.get_or_insert_with("delattr", || {
        make_module_builtin_function("delattr", builtin_delattr)
    });
    namespace.get_or_insert_with("tuple", || crate::typedef::gettypeobject(&TUPLE_TYPE));
    namespace.get_or_insert_with("list", || crate::typedef::gettypeobject(&LIST_TYPE));
    namespace.get_or_insert_with("dict", || crate::typedef::gettypeobject(&DICT_TYPE));
    namespace.get_or_insert_with("object", || {
        // `object` is a W_TypeObject, not a builtin function.
        // PyPy: baseobjspace.py w_object = W_TypeObject("object", ...)
        crate::typedef::w_object()
    });
    namespace.get_or_insert_with("super", || {
        make_module_builtin_function("super", builtin_super)
    });
    namespace.get_or_insert_with("id", || make_module_builtin_function("id", builtin_id));
    namespace.get_or_insert_with("hash", || {
        make_module_builtin_function("hash", builtin_hash)
    });
    namespace.get_or_insert_with("ord", || make_module_builtin_function("ord", builtin_ord));
    namespace.get_or_insert_with("chr", || make_module_builtin_function("chr", builtin_chr));
    namespace.get_or_insert_with("map", || make_module_builtin_function("map", builtin_map));
    namespace.get_or_insert_with("zip", || make_module_builtin_function("zip", builtin_zip));
    namespace.get_or_insert_with("enumerate", || {
        make_module_builtin_function("enumerate", builtin_enumerate)
    });
    namespace.get_or_insert_with("reversed", || {
        make_module_builtin_function("reversed", builtin_reversed)
    });
    namespace.get_or_insert_with("sorted", || {
        make_module_builtin_function("sorted", builtin_sorted)
    });
    namespace.get_or_insert_with("iter", || {
        make_module_builtin_function("iter", builtin_iter)
    });
    namespace.get_or_insert_with("next", || {
        make_module_builtin_function("next", builtin_next)
    });
    namespace.get_or_insert_with("callable", || {
        make_module_builtin_function("callable", builtin_callable)
    });
    namespace.get_or_insert_with("vars", || {
        make_module_builtin_function("vars", builtin_vars)
    });
    namespace.get_or_insert_with("dir", || make_module_builtin_function("dir", builtin_dir));
    namespace.get_or_insert_with("__build_class__", || {
        make_module_builtin_function("__build_class__", |args| {
            crate::call::real_build_class(args)
        })
    });
    // bytearrayobject.py W_BytearrayObject — register the real type
    // (callable as a constructor and usable in isinstance(x, bytearray)).
    namespace.get_or_insert_with("bytearray", || {
        crate::typedef::gettypeobject(&pyre_object::bytearrayobject::BYTEARRAY_TYPE)
    });
    // pyre lacks a separate bytes type — alias `bytes` to bytearray so
    // `isinstance(x, bytes)` succeeds for byte literals (which are also
    // materialised as bytearray; see ConstantData::Bytes handling).
    namespace.get_or_insert_with("bytes", || {
        crate::typedef::gettypeobject(&pyre_object::bytearrayobject::BYTEARRAY_TYPE)
    });
    namespace.get_or_insert_with("slice", || {
        // The slice type object, for isinstance(x, slice) checks.
        crate::typedef::gettypefor(&pyre_object::sliceobject::SLICE_TYPE)
            .unwrap_or(pyre_object::PY_NULL)
    });
    namespace.get_or_insert_with("frozenset", || {
        crate::typedef::gettypeobject(&pyre_object::setobject::FROZENSET_TYPE)
    });
    namespace.get_or_insert_with("set", || {
        crate::typedef::gettypeobject(&pyre_object::setobject::SET_TYPE)
    });
    namespace.get_or_insert_with("property", || {
        crate::typedef::gettypeobject(&pyre_object::propertyobject::PROPERTY_TYPE)
    });
    namespace.get_or_insert_with("staticmethod", || {
        crate::typedef::gettypeobject(&pyre_object::propertyobject::STATICMETHOD_TYPE)
    });
    namespace.get_or_insert_with("classmethod", || {
        crate::typedef::gettypeobject(&pyre_object::propertyobject::CLASSMETHOD_TYPE)
    });
    namespace.get_or_insert_with("Ellipsis", || w_none());
    namespace.get_or_insert_with("__debug__", || w_bool_from(true));
    // memoryview stub: pyre doesn't model real buffer protocol, but
    // re._compiler._bytes_to_codes wants `memoryview(b).cast('I').tolist()`.
    // We register `memoryview` as a real type whose __new__ stores the
    // backing bytearray on the instance; .cast('I') and .tolist() do the
    // little-endian unpack inline.
    namespace.get_or_insert_with("memoryview", || {
        crate::typedef::make_builtin_type("memoryview", |ns| {
            crate::namespace_store(
                ns,
                "__new__",
                make_builtin_function("__new__", |args| {
                    // args[0] = cls (memoryview), args[1] = buffer-like
                    let cls = args.get(0).copied().unwrap_or(w_none());
                    let buf = args.get(1).copied().unwrap_or(w_none());
                    let inst = pyre_object::w_instance_new(cls);
                    crate::baseobjspace::setattr(inst, "__pyre_buf__", buf)?;
                    crate::baseobjspace::setattr(inst, "__pyre_fmt__", w_str_new("B"))?;
                    crate::baseobjspace::setattr(inst, "__pyre_itemsize__", w_int_new(1))?;
                    Ok(inst)
                }),
            );
            crate::namespace_store(
                ns,
                "cast",
                make_builtin_function("cast", |args| {
                    let mv = args.get(0).copied().unwrap_or(w_none());
                    let fmt_obj = args.get(1).copied().unwrap_or(w_none());
                    let fmt = if unsafe { pyre_object::is_str(fmt_obj) } {
                        unsafe { pyre_object::w_str_get_value(fmt_obj) }
                    } else {
                        "B"
                    };
                    let itemsize: i64 = match fmt {
                        "I" | "i" | "L" | "l" | "f" => 4,
                        "Q" | "q" | "d" => 8,
                        "H" | "h" => 2,
                        _ => 1,
                    };
                    let buf = crate::baseobjspace::getattr(mv, "__pyre_buf__")?;
                    let cls = crate::typedef::r#type(mv).unwrap_or(pyre_object::PY_NULL);
                    let inst = pyre_object::w_instance_new(cls);
                    crate::baseobjspace::setattr(inst, "__pyre_buf__", buf)?;
                    crate::baseobjspace::setattr(inst, "__pyre_fmt__", w_str_new(fmt))?;
                    crate::baseobjspace::setattr(inst, "__pyre_itemsize__", w_int_new(itemsize))?;
                    Ok(inst)
                }),
            );
            crate::namespace_store(
                ns,
                "tolist",
                make_builtin_function("tolist", |args| {
                    let mv = args.get(0).copied().unwrap_or(w_none());
                    let buf = crate::baseobjspace::getattr(mv, "__pyre_buf__")?;
                    let itemsize_obj = crate::baseobjspace::getattr(mv, "__pyre_itemsize__")?;
                    let itemsize = unsafe { pyre_object::w_int_get_value(itemsize_obj) } as usize;
                    let data = if unsafe { pyre_object::bytearrayobject::is_bytearray(buf) } {
                        unsafe { pyre_object::bytearrayobject::w_bytearray_data(buf) }
                    } else {
                        return Ok(w_list_new(vec![]));
                    };
                    let mut items = Vec::with_capacity(data.len() / itemsize.max(1));
                    let mut i = 0;
                    while i + itemsize <= data.len() {
                        let mut val: i64 = 0;
                        for j in 0..itemsize {
                            val |= (data[i + j] as i64) << (8 * j);
                        }
                        items.push(w_int_new(val));
                        i += itemsize;
                    }
                    Ok(w_list_new(items))
                }),
            );
            crate::namespace_store(
                ns,
                "__len__",
                make_builtin_function("__len__", |args| {
                    let mv = args.get(0).copied().unwrap_or(w_none());
                    let buf = crate::baseobjspace::getattr(mv, "__pyre_buf__")?;
                    let itemsize_obj = crate::baseobjspace::getattr(mv, "__pyre_itemsize__")?;
                    let itemsize = unsafe { pyre_object::w_int_get_value(itemsize_obj) } as usize;
                    let n = if unsafe { pyre_object::bytearrayobject::is_bytearray(buf) } {
                        unsafe { pyre_object::bytearrayobject::w_bytearray_len(buf) }
                    } else {
                        0
                    };
                    Ok(w_int_new((n / itemsize.max(1)) as i64))
                }),
            );
            // memoryview.itemsize attribute — read from the per-instance
            // __pyre_itemsize__ slot via property descriptor.
            crate::namespace_store(
                ns,
                "itemsize",
                pyre_object::w_property_new(
                    make_builtin_function("itemsize", |args| {
                        let mv = args.get(0).copied().unwrap_or(w_none());
                        crate::baseobjspace::getattr(mv, "__pyre_itemsize__")
                    }),
                    pyre_object::PY_NULL,
                    pyre_object::PY_NULL,
                ),
            );
        })
    });
    namespace.get_or_insert_with("globals", || {
        make_module_builtin_function("globals", builtin_globals)
    });
    namespace.get_or_insert_with("locals", || {
        make_module_builtin_function("locals", builtin_locals)
    });
    namespace.get_or_insert_with("exec", || {
        make_module_builtin_function("exec", builtin_exec)
    });
    namespace.get_or_insert_with("eval", || {
        make_module_builtin_function("eval", builtin_eval)
    });
    namespace.get_or_insert_with("compile", || {
        make_module_builtin_function("compile", builtin_compile)
    });
    namespace.get_or_insert_with("complex", || {
        make_module_builtin_function("complex", |_| Ok(w_none()))
    });
    namespace.get_or_insert_with("filter", || {
        make_module_builtin_function("filter", |args| {
            if args.len() < 2 {
                return Ok(w_list_new(vec![]));
            }
            let func = args[0];
            let items = collect_iterable(args[1])?;
            let mut out = Vec::new();
            let func_is_none = unsafe { pyre_object::is_none(func) };
            for item in items {
                let keep = if func_is_none {
                    crate::baseobjspace::is_true(item)
                } else {
                    let result = crate::call_function(func, &[item]);
                    if result.is_null() {
                        false
                    } else {
                        crate::baseobjspace::is_true(result)
                    }
                };
                if keep {
                    out.push(item);
                }
            }
            Ok(w_list_new(out))
        })
    });
    namespace.get_or_insert_with("input", || {
        make_module_builtin_function("input", |_| Ok(pyre_object::w_str_new("")))
    });
    namespace.get_or_insert_with("open", || {
        make_module_builtin_function("open", |_| {
            Err(crate::PyError::type_error("open() not implemented"))
        })
    });
    // Exception hierarchy — exceptions are real types so they can be
    // subclassed (`class FrozenInstanceError(AttributeError): pass`).
    // Built in dependency order: each subclass refers to its already-built
    // parent. PyPy: each typedef.py W_<Exception>.typedef registers a real
    // W_TypeObject in space.builtin.
    let base_exc = make_exc_type(
        "BaseException",
        exc_base_exception_new,
        crate::typedef::w_object(),
    );
    crate::namespace_store(namespace, "BaseException", base_exc);

    let exception = make_exc_type("Exception", exc_exception_new, base_exc);
    crate::namespace_store(namespace, "Exception", exception);

    let arithmetic = make_exc_type("ArithmeticError", exc_arithmetic_error_new, exception);
    crate::namespace_store(namespace, "ArithmeticError", arithmetic);
    crate::namespace_store(
        namespace,
        "ZeroDivisionError",
        make_exc_type("ZeroDivisionError", exc_zero_division_new, arithmetic),
    );
    crate::namespace_store(
        namespace,
        "OverflowError",
        make_exc_type("OverflowError", exc_overflow_error_new, arithmetic),
    );
    crate::namespace_store(
        namespace,
        "FloatingPointError",
        make_exc_type("FloatingPointError", exc_arithmetic_error_new, arithmetic),
    );

    let lookup_error = make_exc_type("LookupError", exc_exception_new, exception);
    crate::namespace_store(namespace, "LookupError", lookup_error);
    crate::namespace_store(
        namespace,
        "IndexError",
        make_exc_type("IndexError", exc_index_error_new, lookup_error),
    );
    crate::namespace_store(
        namespace,
        "KeyError",
        make_exc_type("KeyError", exc_key_error_new, lookup_error),
    );

    crate::namespace_store(
        namespace,
        "AttributeError",
        make_exc_type("AttributeError", exc_attribute_error_new, exception),
    );
    crate::namespace_store(
        namespace,
        "TypeError",
        make_exc_type("TypeError", exc_type_error_new, exception),
    );
    crate::namespace_store(
        namespace,
        "ValueError",
        make_exc_type("ValueError", exc_value_error_new, exception),
    );
    crate::namespace_store(
        namespace,
        "NameError",
        make_exc_type("NameError", exc_name_error_new, exception),
    );

    let runtime_error = make_exc_type("RuntimeError", exc_runtime_error_new, exception);
    crate::namespace_store(namespace, "RuntimeError", runtime_error);
    crate::namespace_store(
        namespace,
        "NotImplementedError",
        make_exc_type(
            "NotImplementedError",
            exc_not_implemented_error_new,
            runtime_error,
        ),
    );
    crate::namespace_store(
        namespace,
        "RecursionError",
        make_exc_type("RecursionError", exc_runtime_error_new, runtime_error),
    );

    crate::namespace_store(
        namespace,
        "StopIteration",
        make_exc_type("StopIteration", exc_stop_iteration_new, exception),
    );
    crate::namespace_store(
        namespace,
        "StopAsyncIteration",
        make_exc_type("StopAsyncIteration", exc_exception_new, exception),
    );
    crate::namespace_store(
        namespace,
        "GeneratorExit",
        make_exc_type("GeneratorExit", exc_base_exception_new, base_exc),
    );
    crate::namespace_store(
        namespace,
        "SystemExit",
        make_exc_type("SystemExit", exc_base_exception_new, base_exc),
    );
    crate::namespace_store(
        namespace,
        "KeyboardInterrupt",
        make_exc_type("KeyboardInterrupt", exc_base_exception_new, base_exc),
    );

    crate::namespace_store(
        namespace,
        "ImportError",
        make_exc_type("ImportError", exc_import_error_new, exception),
    );
    crate::namespace_store(
        namespace,
        "ModuleNotFoundError",
        make_exc_type("ModuleNotFoundError", exc_import_error_new, exception),
    );
    crate::namespace_store(
        namespace,
        "AssertionError",
        make_exc_type("AssertionError", exc_assertion_error_new, exception),
    );

    let os_error = make_exc_type("OSError", exc_exception_new, exception);
    crate::namespace_store(namespace, "OSError", os_error);
    crate::namespace_store(namespace, "IOError", os_error);
    crate::namespace_store(
        namespace,
        "FileNotFoundError",
        make_exc_type("FileNotFoundError", exc_exception_new, os_error),
    );
    crate::namespace_store(
        namespace,
        "FileExistsError",
        make_exc_type("FileExistsError", exc_exception_new, os_error),
    );
    crate::namespace_store(
        namespace,
        "PermissionError",
        make_exc_type("PermissionError", exc_exception_new, os_error),
    );
    crate::namespace_store(
        namespace,
        "NotADirectoryError",
        make_exc_type("NotADirectoryError", exc_exception_new, os_error),
    );
    crate::namespace_store(
        namespace,
        "IsADirectoryError",
        make_exc_type("IsADirectoryError", exc_exception_new, os_error),
    );

    let warning = make_exc_type("Warning", exc_exception_new, exception);
    crate::namespace_store(namespace, "Warning", warning);
    for warn_name in [
        "UserWarning",
        "DeprecationWarning",
        "PendingDeprecationWarning",
        "RuntimeWarning",
        "FutureWarning",
        "ImportWarning",
        "UnicodeWarning",
        "BytesWarning",
        "ResourceWarning",
        "SyntaxWarning",
        "EncodingWarning",
    ] {
        crate::namespace_store(
            namespace,
            warn_name,
            make_exc_type(warn_name, exc_exception_new, warning),
        );
    }

    crate::namespace_store(
        namespace,
        "UnicodeError",
        make_exc_type("UnicodeError", exc_value_error_new, exception),
    );
    crate::namespace_store(
        namespace,
        "UnicodeDecodeError",
        make_exc_type("UnicodeDecodeError", exc_value_error_new, exception),
    );
    crate::namespace_store(
        namespace,
        "UnicodeEncodeError",
        make_exc_type("UnicodeEncodeError", exc_value_error_new, exception),
    );
    crate::namespace_store(
        namespace,
        "UnicodeTranslateError",
        make_exc_type("UnicodeTranslateError", exc_value_error_new, exception),
    );

    crate::namespace_store(
        namespace,
        "BufferError",
        make_exc_type("BufferError", exc_exception_new, exception),
    );
    crate::namespace_store(
        namespace,
        "MemoryError",
        make_exc_type("MemoryError", exc_exception_new, exception),
    );
    crate::namespace_store(
        namespace,
        "ReferenceError",
        make_exc_type("ReferenceError", exc_exception_new, exception),
    );
    crate::namespace_store(
        namespace,
        "SystemError",
        make_exc_type("SystemError", exc_exception_new, exception),
    );
    crate::namespace_store(
        namespace,
        "EOFError",
        make_exc_type("EOFError", exc_exception_new, exception),
    );
    crate::namespace_store(
        namespace,
        "SyntaxError",
        make_exc_type("SyntaxError", exc_exception_new, exception),
    );
    crate::namespace_store(
        namespace,
        "IndentationError",
        make_exc_type("IndentationError", exc_exception_new, exception),
    );
    crate::namespace_store(
        namespace,
        "TabError",
        make_exc_type("TabError", exc_exception_new, exception),
    );
    crate::namespace_store(
        namespace,
        "BlockingIOError",
        make_exc_type("BlockingIOError", exc_exception_new, os_error),
    );
    crate::namespace_store(
        namespace,
        "ChildProcessError",
        make_exc_type("ChildProcessError", exc_exception_new, os_error),
    );
    crate::namespace_store(
        namespace,
        "ConnectionError",
        make_exc_type("ConnectionError", exc_exception_new, os_error),
    );
    crate::namespace_store(
        namespace,
        "BrokenPipeError",
        make_exc_type("BrokenPipeError", exc_exception_new, os_error),
    );
    crate::namespace_store(
        namespace,
        "ConnectionAbortedError",
        make_exc_type("ConnectionAbortedError", exc_exception_new, os_error),
    );
    crate::namespace_store(
        namespace,
        "ConnectionRefusedError",
        make_exc_type("ConnectionRefusedError", exc_exception_new, os_error),
    );
    crate::namespace_store(
        namespace,
        "ConnectionResetError",
        make_exc_type("ConnectionResetError", exc_exception_new, os_error),
    );
    crate::namespace_store(
        namespace,
        "InterruptedError",
        make_exc_type("InterruptedError", exc_exception_new, os_error),
    );
    crate::namespace_store(
        namespace,
        "ProcessLookupError",
        make_exc_type("ProcessLookupError", exc_exception_new, os_error),
    );
    crate::namespace_store(
        namespace,
        "TimeoutError",
        make_exc_type("TimeoutError", exc_exception_new, os_error),
    );
    crate::namespace_store(
        namespace,
        "BaseExceptionGroup",
        make_exc_type("BaseExceptionGroup", exc_base_exception_new, base_exc),
    );
    crate::namespace_store(
        namespace,
        "ExceptionGroup",
        make_exc_type("ExceptionGroup", exc_exception_new, exception),
    );
    crate::namespace_store(
        namespace,
        "PythonFinalizationError",
        make_exc_type(
            "PythonFinalizationError",
            exc_runtime_error_new,
            runtime_error,
        ),
    );
    namespace.get_or_insert_with("any", || make_module_builtin_function("any", builtin_any));
    namespace.get_or_insert_with("all", || make_module_builtin_function("all", builtin_all));
    namespace.get_or_insert_with("sum", || make_module_builtin_function("sum", builtin_sum));
    namespace.get_or_insert_with("round", || {
        make_module_builtin_function("round", builtin_round)
    });
    namespace.get_or_insert_with("divmod", || {
        make_module_builtin_function("divmod", builtin_divmod)
    });
    namespace.get_or_insert_with("pow", || make_module_builtin_function("pow", builtin_pow));
    namespace.get_or_insert_with("hex", || make_module_builtin_function("hex", builtin_hex));
    namespace.get_or_insert_with("oct", || make_module_builtin_function("oct", builtin_oct));
    namespace.get_or_insert_with("bin", || make_module_builtin_function("bin", builtin_bin));
    namespace.get_or_insert_with("format", || {
        make_module_builtin_function("format", builtin_format)
    });
    namespace.get_or_insert_with("issubclass", || {
        make_module_builtin_function("issubclass", builtin_issubclass)
    });
    namespace.get_or_insert_with("__import__", || {
        make_module_builtin_function("__import__", builtin_import_stub)
    });

    // Descriptor types
    namespace.get_or_insert_with("property", || {
        crate::typedef::gettypeobject(&pyre_object::propertyobject::PROPERTY_TYPE)
    });
    // staticmethod/classmethod registered as types for isinstance() support.
    // The type's __new__ creates the descriptor wrapper.
    namespace.get_or_insert_with("staticmethod", || {
        crate::typedef::gettypeobject(&pyre_object::propertyobject::STATICMETHOD_TYPE)
    });
    namespace.get_or_insert_with("classmethod", || {
        crate::typedef::gettypeobject(&pyre_object::propertyobject::CLASSMETHOD_TYPE)
    });
}

/// Create a fresh namespace seeded with the default builtins.
pub fn new_builtin_namespace() -> PyNamespace {
    crate::typedef::init_typeobjects();
    let mut namespace = PyNamespace::new();
    install_default_builtins(&mut namespace);
    namespace
}

/// `print(*args)` — write space-separated str representations to stdout.
fn builtin_print(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // Check if last arg is a kwargs dict (from CALL_KW builtin dispatch).
    // Distinguished from regular dict args by __pyre_kw__ marker key.
    let is_kwargs = !args.is_empty()
        && unsafe {
            let last = *args.last().unwrap();
            is_dict(last) && pyre_object::w_dict_lookup(last, w_str_new("__pyre_kw__")).is_some()
        };
    let (positional, end, sep) = if is_kwargs {
        let kwargs = *args.last().unwrap();
        let end_key = w_str_new("end");
        let sep_key = w_str_new("sep");
        let end_val = unsafe { pyre_object::w_dict_lookup(kwargs, end_key) };
        let sep_val = unsafe { pyre_object::w_dict_lookup(kwargs, sep_key) };
        let end_str = end_val
            .map(|v| unsafe { crate::py_str(v) })
            .unwrap_or_else(|| "\n".to_string());
        let sep_str = sep_val
            .map(|v| unsafe { crate::py_str(v) })
            .unwrap_or_else(|| " ".to_string());
        (&args[..args.len() - 1], end_str, sep_str)
    } else {
        (args, "\n".to_string(), " ".to_string())
    };

    let parts: Vec<String> = positional
        .iter()
        .map(|&obj| format!("{}", PyDisplay(obj)))
        .collect();
    crate::print_output(&format!("{}{}", parts.join(&sep), end));
    Ok(w_none())
}

/// Extract an i64 from an int or long object. Panics if the long value
/// exceeds i64 range (range() cannot iterate over such large spans).
unsafe fn range_arg_to_i64(obj: PyObjectRef) -> i64 {
    unsafe {
        if is_int(obj) {
            return w_int_get_value(obj);
        }
        if is_long(obj) {
            let val = w_long_get_value(obj);
            return val.to_i64().unwrap_or(i64::MAX);
        }
        return 0; // non-integer argument fallback
    }
}

/// `range(stop)` or `range(start, stop)` or `range(start, stop, step)`.
///
/// Returns a range iterator directly (simplified: no separate range object).
fn builtin_range(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    match args.len() {
        1 => {
            let stop = unsafe { range_arg_to_i64(args[0]) };
            Ok(w_range_iter_new(0, stop, 1))
        }
        2 => {
            let start = unsafe { range_arg_to_i64(args[0]) };
            let stop = unsafe { range_arg_to_i64(args[1]) };
            Ok(w_range_iter_new(start, stop, 1))
        }
        3 => {
            let start = unsafe { range_arg_to_i64(args[0]) };
            let stop = unsafe { range_arg_to_i64(args[1]) };
            let step = unsafe { range_arg_to_i64(args[2]) };
            Ok(w_range_iter_new(start, stop, step))
        }
        _ => panic!("range() takes 1 to 3 arguments"),
    }
}

/// `len(obj)` — return the length of an object.
/// `len(obj)` — PyPy: operation.py len → space.len_w
fn builtin_len(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "len() takes exactly one argument");
    crate::baseobjspace::len(args[0])
}

/// `abs(x)` — return the absolute value of a number.
fn builtin_abs(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "abs() takes exactly one argument");
    let obj = args[0];
    unsafe {
        if is_int(obj) {
            let v = w_int_get_value(obj);
            // i64::MIN.abs() overflows; promote to long
            return Ok(match v.checked_abs() {
                Some(r) => w_int_new(r),
                None => w_long_new(-BigInt::from(v)),
            });
        }
        if is_long(obj) {
            let val = w_long_get_value(obj).clone();
            return Ok(w_long_new(if val < BigInt::from(0) { -val } else { val }));
        }
        if is_float(obj) {
            return Ok(w_float_new(w_float_get_value(obj).abs()));
        }
    }
    // Instance __abs__ — PyPy: baseobjspace.py abs
    unsafe {
        if pyre_object::is_instance(obj) {
            let w_type = pyre_object::w_instance_get_type(obj);
            if let Some(method) = crate::baseobjspace::lookup_in_type(w_type, "__abs__") {
                return Ok(crate::call_function(method, &[obj]));
            }
        }
    }
    panic!("bad operand type for abs()")
}

/// Convert an int or long object to BigInt for comparison.
pub(crate) unsafe fn obj_to_bigint(obj: PyObjectRef) -> BigInt {
    unsafe {
        if is_int(obj) {
            BigInt::from(w_int_get_value(obj))
        } else {
            w_long_get_value(obj).clone()
        }
    }
}

/// `min(a, b)` — return the smaller of two values.
fn builtin_min(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2, "min() takes exactly two arguments");
    let a = args[0];
    let b = args[1];
    unsafe {
        if is_int(a) && is_int(b) {
            let va = w_int_get_value(a);
            let vb = w_int_get_value(b);
            return Ok(if va <= vb { a } else { b });
        }
        if is_int_or_long(a) && is_int_or_long(b) {
            let va = obj_to_bigint(a);
            let vb = obj_to_bigint(b);
            return Ok(if va <= vb { a } else { b });
        }
    }
    panic!("min() not supported for these types")
}

/// `max(a, b)` / `max(iterable)` — return the largest of two values or an iterable.
/// PyPy: operation.py max_w
fn builtin_max(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty(), "max() takes at least one argument");
    let items: Vec<PyObjectRef> = if args.len() == 1 {
        collect_iterable(args[0])?
    } else {
        args.to_vec()
    };
    if items.is_empty() {
        return Err(crate::PyError::new(
            crate::PyErrorKind::ValueError,
            "max() iterable argument is empty",
        ));
    }
    let mut best = items[0];
    for &item in &items[1..] {
        unsafe {
            let keep_best = if is_int(item) && is_int(best) {
                w_int_get_value(best) >= w_int_get_value(item)
            } else if is_int_or_long(item) && is_int_or_long(best) {
                obj_to_bigint(best) >= obj_to_bigint(item)
            } else if is_str(item) && is_str(best) {
                w_str_get_value(best) >= w_str_get_value(item)
            } else {
                true
            };
            if !keep_best {
                best = item;
            }
        }
    }
    Ok(best)
}

/// `type(obj)` — return the type name as a string (simplified).
/// `type(obj)` — return the type of an object as a W_TypeObject.
///
/// PyPy: `space.type(w_obj)` → W_TypeObject
pub(crate) fn type_descr_new(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // type.__new__(metatype, name, bases, dict)
    // May be called with extra self-binding from super():
    //   [self, metatype, name, bases, dict] — 5 args
    //   [metatype, name, bases, dict] — 4 args
    //   [metatype, obj] — 2 args (type(obj))
    // Find the (name, bases, dict) triple by scanning for the first str arg.
    // Also extract the metatype (first type arg before the name str).
    let mut w_metaclass = pyre_object::PY_NULL;
    for i in 0..args.len() {
        if unsafe { pyre_object::is_str(args[i]) } && i + 2 < args.len() {
            // Extract metatype from preceding args
            for j in 0..i {
                if unsafe { pyre_object::is_type(args[j]) } {
                    w_metaclass = args[j];
                }
            }
            return type_descr_new_with_metaclass(&args[i..], w_metaclass);
        }
    }
    if args.len() == 1 && unsafe { pyre_object::is_type(args[0]) } {
        return Err(crate::PyError::type_error("type() takes 1 or 3 arguments"));
    }
    if args.len() == 1 {
        return type_descr_new_without_metaclass(args);
    }
    if args.len() == 2 {
        return type_descr_new_without_metaclass(&args[1..]);
    }
    Err(crate::PyError::type_error("type() takes 1 or 3 arguments"))
}
fn type_descr_new_without_metaclass(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    type_descr_new_with_metaclass(args, pyre_object::PY_NULL)
}

fn type_descr_new_with_metaclass(
    args: &[PyObjectRef],
    w_metaclass: PyObjectRef,
) -> Result<PyObjectRef, crate::PyError> {
    if args.len() != 1 && args.len() != 3 {
        return Err(crate::PyError::type_error("type() takes 1 or 3 arguments"));
    }
    // type(name, bases, dict) — 3-arg form creates a new type
    // PyPy: typeobject.py type.__new__(metatype, name, bases, dict)
    if args.len() == 3 {
        let name_obj = args[0];
        let bases = args[1];
        let w_namespace_dict = args[2];
        let name = unsafe { pyre_object::w_str_get_value(name_obj) };

        // CPython: calculate_metaclass — if bases have a custom metaclass,
        // delegate to that metaclass instead of using type.__new__ directly.
        if w_metaclass.is_null() && !bases.is_null() && unsafe { is_tuple(bases) } {
            let n = unsafe { w_tuple_len(bases) };
            for i in 0..n {
                if let Some(base) = unsafe { pyre_object::w_tuple_getitem(bases, i as i64) } {
                    if unsafe { pyre_object::is_type(base) } {
                        // baseobjspace.py:76 — metaclass from w_class
                        let w_metaclass = unsafe {
                            let w_class = (*base).w_class;
                            let w_type_type = crate::typedef::w_type();
                            if !w_class.is_null() && !std::ptr::eq(w_class, w_type_type) {
                                Some(w_class)
                            } else {
                                None
                            }
                        };
                        if let Some(w_metaclass) = w_metaclass {
                            // Delegate: call metaclass(name, bases, dict, **kwds)
                            // Pass extra args from the original call
                            let mut metaclass_args = vec![name_obj, bases, w_namespace_dict];
                            if args.len() > 3 {
                                metaclass_args.extend_from_slice(&args[3..]);
                            }
                            return Ok(crate::call_function(w_metaclass, &metaclass_args));
                        }
                    }
                }
            }
        }

        // Convert dict to PyNamespace
        let mut class_ns = Box::new(crate::PyNamespace::new());
        class_ns.fix_ptr();
        if unsafe { is_dict(w_namespace_dict) } {
            let d = unsafe { &*(w_namespace_dict as *const pyre_object::dictobject::W_DictObject) };
            for &(k, v) in unsafe { &*d.entries } {
                if unsafe { is_str(k) } {
                    let key = unsafe { pyre_object::w_str_get_value(k) };
                    crate::namespace_store(&mut class_ns, key, v);
                }
            }
        }
        let ns_ptr = Box::into_raw(class_ns);

        // Default bases to (object,) if empty
        let w_effective_bases =
            if bases.is_null() || !unsafe { is_tuple(bases) } || unsafe { w_tuple_len(bases) } == 0
            {
                let w_object = crate::typedef::w_object();
                if !w_object.is_null() {
                    pyre_object::w_tuple_new(vec![w_object])
                } else {
                    bases
                }
            } else {
                bases
            };

        // CPython: calculate_metaclass — delegate to winner if different
        let default_meta = if w_metaclass.is_null() {
            crate::typedef::w_type()
        } else {
            w_metaclass
        };
        let w_winner = crate::call::calculate_metaclass(default_meta, w_effective_bases)
            .unwrap_or(default_meta);
        if !std::ptr::eq(w_winner, default_meta) {
            // Winner is a different metaclass — delegate to its __new__
            if let Some(w_metaclass_new) =
                unsafe { crate::baseobjspace::lookup_in_type(w_winner, "__new__") }
            {
                let mut new_args = vec![w_winner, name_obj, bases, w_namespace_dict];
                if args.len() > 3 {
                    new_args.extend_from_slice(&args[3..]);
                }
                drop(unsafe { Box::from_raw(ns_ptr) });
                return Ok(crate::call_function(w_metaclass_new, &new_args));
            }
        }
        let w_metaclass = w_winner;

        let w_type = pyre_object::w_type_new(name, w_effective_bases, ns_ptr as *mut u8);
        // typeobject.py:1143-1204 create_all_slots parity.
        unsafe {
            let ns = &*ns_ptr;
            crate::call::create_all_slots(w_type, ns, w_effective_bases)?;
        }
        // rclass.py:739-743 — set w_class (typeptr) at allocation time.
        // For type objects, w_class is the metaclass (type(C) → Meta).
        // baseobjspace.py:76 getclass() returns the metatype.
        unsafe {
            (*w_type).w_class = w_metaclass;
        }
        let mro = unsafe { crate::baseobjspace::compute_default_mro(w_type) };
        unsafe { pyre_object::w_type_set_mro(w_type, mro) };

        // __set_name__ protocol — CPython: type_new_set_names
        // PyPy: typeobject.py type_new → call __set_name__(owner, name) on each descriptor
        if unsafe { is_dict(w_namespace_dict) } {
            let d = unsafe { &*(w_namespace_dict as *const pyre_object::dictobject::W_DictObject) };
            let entries: Vec<(PyObjectRef, PyObjectRef)> = unsafe { (*d.entries).clone() };
            for (k, v) in entries {
                if unsafe { is_str(k) } {
                    if let Ok(set_name) = crate::baseobjspace::getattr(v, "__set_name__") {
                        // getattr returns a bound method, so self is already bound.
                        // Call: bound_set_name(owner, name)
                        let _ = crate::call_function(set_name, &[w_type, k]);
                    }
                }
            }
        }

        return Ok(w_type);
    }

    // type(obj) — 1-arg form returns the type
    // PyPy objspace.py:400: space.type(w_obj) → w_obj.getclass(space)
    // typedef::type() respects __class__ override for all object kinds.
    let obj = args[0];
    if let Some(tp) = crate::typedef::r#type(obj) {
        return Ok(tp);
    }
    if obj.is_null() {
        return Ok(crate::typedef::gettypeobject(
            &pyre_object::pyobject::NONE_TYPE,
        ));
    }
    let name = unsafe { (*(*obj).ob_type).name };
    Ok(box_str_constant(name))
}

/// `isinstance(obj, cls)` — pypy/module/__builtin__/abstractinst.py
/// `app_isinstance` → `abstract_isinstance_w(allow_override=True)`.
fn builtin_isinstance(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2, "isinstance() takes exactly two arguments");
    Ok(w_bool_from(crate::baseobjspace::isinstance(
        args[0], args[1],
    )?))
}

/// isinstance(obj, cls) for JIT fast path.
///
/// Returns Some(bool) if the check can be resolved, None if cls format
/// is not supported for the fast path (e.g. tuple of types).
/// Uses the same MRO-based `issubtype_w` as the full dispatch.
pub fn call_isinstance(obj: PyObjectRef, cls: PyObjectRef) -> Option<bool> {
    unsafe {
        if is_type(cls) {
            return Some(crate::baseobjspace::isinstance_w(obj, cls));
        }
    }
    None
}

/// `issubclass(cls, classinfo)` — pypy/module/__builtin__/abstractinst.py
/// `app_issubclass` → `abstract_issubclass_w(allow_override=True)`.
fn builtin_issubclass(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2, "issubclass() takes exactly two arguments");
    Ok(w_bool_from(crate::baseobjspace::issubclass(
        args[0], args[1],
    )?))
}

// Descroperation helpers (lookup_type_special, should_try_reverse_first,
// try_dispatch_binary_special, try_dispatch_ternary_special,
// try_int_long_pow_with_modulo, binary_builtin_type_error,
// box_bigint_result, issubtype_w) live in `crate::baseobjspace` because
// they are space-level semantics shared between the builtin module,
// weakproxy wrappers, and any future opcode dispatch.

/// Exception type constructor — called as e.g. `ValueError("msg")`.
/// Extracts the message from the first argument and creates a W_ExceptionObject.
macro_rules! exc_constructor {
    ($fn_name:ident, $kind:expr) => {
        fn $fn_name(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            let msg = if args.is_empty() {
                ""
            } else if unsafe { is_str(args[0]) } {
                unsafe { w_str_get_value(args[0]) }
            } else {
                ""
            };
            Ok(pyre_object::excobject::w_exception_new($kind, msg))
        }
    };
}

exc_constructor!(
    exc_base_exception,
    pyre_object::excobject::ExcKind::BaseException
);
exc_constructor!(exc_exception, pyre_object::excobject::ExcKind::Exception);
exc_constructor!(
    exc_arithmetic_error,
    pyre_object::excobject::ExcKind::ArithmeticError
);
exc_constructor!(
    exc_zero_division,
    pyre_object::excobject::ExcKind::ZeroDivisionError
);
exc_constructor!(exc_type_error, pyre_object::excobject::ExcKind::TypeError);
exc_constructor!(exc_value_error, pyre_object::excobject::ExcKind::ValueError);
exc_constructor!(exc_key_error, pyre_object::excobject::ExcKind::KeyError);
exc_constructor!(exc_index_error, pyre_object::excobject::ExcKind::IndexError);
exc_constructor!(
    exc_attribute_error,
    pyre_object::excobject::ExcKind::AttributeError
);
exc_constructor!(exc_name_error, pyre_object::excobject::ExcKind::NameError);
exc_constructor!(
    exc_runtime_error,
    pyre_object::excobject::ExcKind::RuntimeError
);
exc_constructor!(
    exc_stop_iteration,
    pyre_object::excobject::ExcKind::StopIteration
);
exc_constructor!(
    exc_overflow_error,
    pyre_object::excobject::ExcKind::OverflowError
);
exc_constructor!(
    exc_import_error,
    pyre_object::excobject::ExcKind::ImportError
);
exc_constructor!(
    exc_not_implemented_error,
    pyre_object::excobject::ExcKind::NotImplementedError
);
exc_constructor!(
    exc_assertion_error,
    pyre_object::excobject::ExcKind::AssertionError
);

/// `cls.__new__` wrapper that strips `cls` and calls an exception constructor.
/// PyPy: each exception type's descr__new__ creates a W_<Kind>Object.
macro_rules! exc_new_wrapper {
    ($wrapper:ident, $ctor:ident) => {
        fn $wrapper(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
            let rest: &[PyObjectRef] = if args.is_empty() { args } else { &args[1..] };
            $ctor(rest)
        }
    };
}

exc_new_wrapper!(exc_base_exception_new, exc_base_exception);
exc_new_wrapper!(exc_exception_new, exc_exception);
exc_new_wrapper!(exc_arithmetic_error_new, exc_arithmetic_error);
exc_new_wrapper!(exc_zero_division_new, exc_zero_division);
exc_new_wrapper!(exc_type_error_new, exc_type_error);
exc_new_wrapper!(exc_value_error_new, exc_value_error);
exc_new_wrapper!(exc_key_error_new, exc_key_error);
exc_new_wrapper!(exc_index_error_new, exc_index_error);
exc_new_wrapper!(exc_attribute_error_new, exc_attribute_error);
exc_new_wrapper!(exc_name_error_new, exc_name_error);
exc_new_wrapper!(exc_runtime_error_new, exc_runtime_error);
exc_new_wrapper!(exc_stop_iteration_new, exc_stop_iteration);
exc_new_wrapper!(exc_overflow_error_new, exc_overflow_error);
exc_new_wrapper!(exc_import_error_new, exc_import_error);
exc_new_wrapper!(exc_not_implemented_error_new, exc_not_implemented_error);
exc_new_wrapper!(exc_assertion_error_new, exc_assertion_error);

/// Build a builtin exception type with the given name, base, and __new__ wrapper.
fn make_exc_type(
    name: &'static str,
    new_fn: crate::gateway::BuiltinCodeFn,
    base: PyObjectRef,
) -> PyObjectRef {
    crate::typedef::make_builtin_type_with_base(
        name,
        move |ns| {
            crate::namespace_store(ns, "__new__", make_builtin_function("__new__", new_fn));
        },
        base,
    )
}

/// `__build_class__(body, name, *bases)` — class creation.
///
/// PyPy equivalent: pyopcode.py BUILD_CLASS
/// Direct call to call::real_build_class (no callback needed —
/// interpreter and runtime are in the same crate).
fn builtin_build_class(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    crate::call::real_build_class(args)
}

/// Get a reference to the `__build_class__` builtin function.
pub fn get_build_class_func() -> PyObjectRef {
    make_builtin_function("__build_class__", builtin_build_class)
}

/// `property(fget=None, fset=None, fdel=None, doc=None)` → W_PropertyObject
///
/// PyPy: descriptor.py W_Property
fn builtin_property(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let fget = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    let fset = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    let fdel = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
    Ok(pyre_object::w_property_new(fget, fset, fdel))
}

/// `staticmethod(func)` → W_StaticMethodObject
///
/// PyPy: function.py StaticMethod — __get__ returns wrapped func as-is.
fn builtin_staticmethod(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(
        !args.is_empty(),
        "staticmethod requires a callable argument"
    );
    Ok(pyre_object::w_staticmethod_new(args[0]))
}

/// `classmethod(func)` → W_ClassMethodObject
///
/// PyPy: function.py ClassMethod — __get__ binds the class as first arg.
fn builtin_classmethod(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty(), "classmethod requires a callable argument");
    Ok(pyre_object::w_classmethod_new(args[0]))
}

/// `str(obj)` → convert to string
pub(crate) fn builtin_str(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(w_str_new(""));
    }
    let obj = args[0];
    unsafe {
        if is_str(obj) {
            return Ok(obj);
        }
    }
    let s = unsafe { crate::py_str(obj) };
    Ok(w_str_new(&s))
}

/// `repr(obj)` → string representation
fn builtin_repr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "repr() takes exactly one argument");
    let s = unsafe { crate::py_repr(args[0]) };
    Ok(w_str_new(&s))
}

/// `int(obj)` → convert to int
pub(crate) fn builtin_int(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(w_int_new(0));
    }
    let obj = args[0];
    unsafe {
        // int(int_value) → return as-is. base argument is meaningful only
        // for string conversion; ignore it for int/float/bool inputs.
        if is_int(obj) {
            return Ok(obj);
        }
        if is_float(obj) {
            return Ok(w_int_new(floatobject::w_float_get_value(obj) as i64));
        }
        if is_bool(obj) {
            return Ok(w_int_new(if w_bool_get_value(obj) { 1 } else { 0 }));
        }
        // int(string, base) — parse string in given base.
        let base: u32 = if args.len() > 1 && is_int(args[1]) {
            w_int_get_value(args[1]) as u32
        } else {
            10
        };
        // bytearray/bytes input is treated like a string of ASCII digits.
        if pyre_object::bytearrayobject::is_bytearray(obj) {
            let data = pyre_object::bytearrayobject::w_bytearray_data(obj);
            let s_owned = String::from_utf8_lossy(data).into_owned();
            let s = s_owned.trim();
            let cleaned: String = s.chars().filter(|&c| c != '_').collect();
            if let Ok(v) = i64::from_str_radix(&cleaned, base) {
                return Ok(w_int_new(v));
            }
            return Err(crate::PyError::new(
                crate::PyErrorKind::ValueError,
                format!("invalid literal for int() with base {base}: '{s}'"),
            ));
        }
        if is_str(obj) {
            let s = w_str_get_value(obj).trim();
            // Strip optional 0b/0o/0x prefix when base is 0 or matches.
            let (sign, rest) = if let Some(r) = s.strip_prefix('-') {
                (-1i64, r)
            } else if let Some(r) = s.strip_prefix('+') {
                (1i64, r)
            } else {
                (1i64, s)
            };
            let (radix, digits) = if base == 0 {
                if let Some(r) = rest.strip_prefix("0x").or(rest.strip_prefix("0X")) {
                    (16u32, r)
                } else if let Some(r) = rest.strip_prefix("0b").or(rest.strip_prefix("0B")) {
                    (2u32, r)
                } else if let Some(r) = rest.strip_prefix("0o").or(rest.strip_prefix("0O")) {
                    (8u32, r)
                } else {
                    (10u32, rest)
                }
            } else {
                let stripped = match base {
                    16 => rest
                        .strip_prefix("0x")
                        .or(rest.strip_prefix("0X"))
                        .unwrap_or(rest),
                    2 => rest
                        .strip_prefix("0b")
                        .or(rest.strip_prefix("0B"))
                        .unwrap_or(rest),
                    8 => rest
                        .strip_prefix("0o")
                        .or(rest.strip_prefix("0O"))
                        .unwrap_or(rest),
                    _ => rest,
                };
                (base, stripped)
            };
            // Skip underscores allowed in numeric literals.
            let cleaned: String = digits.chars().filter(|&c| c != '_').collect();
            if let Ok(v) = i64::from_str_radix(&cleaned, radix) {
                return Ok(w_int_new(sign * v));
            }
            return Err(crate::PyError::new(
                crate::PyErrorKind::ValueError,
                format!("invalid literal for int() with base {base}: '{s}'"),
            ));
        }
    }
    // __int__ protocol
    if let Ok(int_fn) = crate::baseobjspace::getattr(obj, "__int__") {
        return Ok(crate::call_function(int_fn, &[obj]));
    }
    Ok(w_int_new(0))
}

/// `float(obj)` → convert to float
pub(crate) fn builtin_float(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(floatobject::w_float_new(0.0));
    }
    let obj = args[0];
    unsafe {
        if is_float(obj) {
            return Ok(obj);
        }
        if is_int(obj) {
            return Ok(floatobject::w_float_new(w_int_get_value(obj) as f64));
        }
        if is_str(obj) {
            let s = w_str_get_value(obj);
            if let Ok(v) = s.trim().parse::<f64>() {
                return Ok(floatobject::w_float_new(v));
            }
        }
    }
    Ok(floatobject::w_float_new(0.0))
}

/// `bool(obj)` — PyPy: operation.py bool → space.is_true
pub(crate) fn builtin_bool(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(w_bool_from(false));
    }
    Ok(w_bool_from(crate::baseobjspace::is_true(args[0])))
}

/// `hasattr(obj, name)` → bool — direct call (no callback needed after merge)
fn builtin_hasattr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2, "hasattr() takes exactly two arguments");
    let obj = args[0];
    let name = unsafe { w_str_get_value(args[1]) };
    Ok(w_bool_from(crate::baseobjspace::getattr(obj, name).is_ok()))
}

/// `getattr(obj, name[, default])` → value — direct call
fn builtin_getattr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2, "getattr() takes at least two arguments");
    let obj = args[0];
    let name = unsafe { w_str_get_value(args[1]) };
    match crate::baseobjspace::getattr(obj, name) {
        Ok(val) => Ok(val),
        Err(e) => {
            if args.len() > 2 {
                Ok(args[2]) // default value
            } else {
                Err(e) // propagate AttributeError
            }
        }
    }
}

/// `setattr(obj, name, value)` — direct call
fn builtin_setattr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 3, "setattr() takes exactly three arguments");
    let obj = args[0];
    let name = unsafe { w_str_get_value(args[1]) };
    let _ = crate::baseobjspace::setattr(obj, name, args[2]);
    Ok(w_none())
}

/// `delattr(obj, name)` — PyPy: baseobjspace.py delattr
fn builtin_delattr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2, "delattr() takes exactly 2 arguments");
    let obj = args[0];
    let name = unsafe { w_str_get_value(args[1]) };
    crate::baseobjspace::delattr(obj, name)?;
    Ok(w_none())
}

pub(crate) fn builtin_tuple(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(w_tuple_new(vec![]));
    }
    let obj = args[0];
    unsafe {
        if is_tuple(obj) {
            return Ok(obj);
        }
        if is_list(obj) {
            let n = w_list_len(obj);
            let items: Vec<_> = (0..n)
                .filter_map(|i| w_list_getitem(obj, i as i64))
                .collect();
            return Ok(w_tuple_new(items));
        }
    }
    Ok(w_tuple_new(collect_iterable(obj)?))
}

pub(crate) fn builtin_list_ctor(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(w_list_new(vec![]));
    }
    let obj = args[0];
    unsafe {
        if is_list(obj) {
            // Copy the list
            let n = w_list_len(obj);
            let items: Vec<_> = (0..n)
                .filter_map(|i| w_list_getitem(obj, i as i64))
                .collect();
            return Ok(w_list_new(items));
        }
        if is_tuple(obj) {
            let n = w_tuple_len(obj);
            let items: Vec<_> = (0..n)
                .filter_map(|i| w_tuple_getitem(obj, i as i64))
                .collect();
            return Ok(w_list_new(items));
        }
    }
    // Consume iterator — PyPy: listobject.py W_ListObject(iterable)
    Ok(w_list_new(collect_iterable(obj)?))
}

pub fn collect_iterable(obj: PyObjectRef) -> Result<Vec<PyObjectRef>, crate::PyError> {
    let it = crate::baseobjspace::iter(obj)?;
    let mut items = Vec::new();
    loop {
        match crate::baseobjspace::next(it) {
            Ok(v) => items.push(v),
            Err(e) if e.kind == crate::PyErrorKind::StopIteration => break,
            Err(e) => return Err(e),
        }
    }
    Ok(items)
}

/// Create a `set` from a slice of elements.
///
/// PyPy: `setobject.py` W_SetObject.descr_init → `_initialize_set`.
pub fn builtin_set_from_items(items: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    Ok(pyre_object::w_set_from_items(items))
}

/// `set([iterable])` — PyPy: setobject.py W_SetObject.descr_new + descr_init.
pub(crate) fn builtin_set_ctor(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return builtin_set_from_items(&[]);
    }
    let items = collect_iterable(args[0])?;
    builtin_set_from_items(&items)
}

/// `frozenset([iterable])` — PyPy: setobject.py W_FrozensetObject.descr_new.
pub(crate) fn builtin_frozenset_ctor(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(pyre_object::w_frozenset_new());
    }
    let items = collect_iterable(args[0])?;
    Ok(pyre_object::w_frozenset_from_items(&items))
}

/// `dict()` — PyPy: dictobject.py W_DictMultiObject.descr_init
pub(crate) fn builtin_dict_ctor(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(w_dict_new());
    }
    unsafe {
        if is_dict(args[0]) {
            // PyPy: descr_init → shallow copy when first arg is a dict
            let src = args[0];
            let dict = w_dict_new();
            for (k, v) in pyre_object::w_dict_items(src) {
                w_dict_store(dict, k, v);
            }
            return Ok(dict);
        }
    }
    // Construct from iterable of (key, value) pairs.
    // PyPy: dictobject.py W_DictMultiObject.descr_init → update1_seq
    let src = args[0];
    let dict = w_dict_new();
    let items = collect_iterable(src)?;
    for pair in items {
        let (k, v) = unsafe {
            if is_tuple(pair) && w_tuple_len(pair) == 2 {
                (
                    w_tuple_getitem(pair, 0).unwrap(),
                    w_tuple_getitem(pair, 1).unwrap(),
                )
            } else if is_list(pair) && w_list_len(pair) == 2 {
                (
                    w_list_getitem(pair, 0).unwrap(),
                    w_list_getitem(pair, 1).unwrap(),
                )
            } else {
                return Err(crate::PyError::type_error(
                    "dict update sequence element is not a 2-element sequence",
                ));
            }
        };
        unsafe { w_dict_store(dict, k, v) };
    }
    Ok(dict)
}

/// `object()` — PyPy: objectobject.py descr__new__
fn builtin_object(_args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    // PyPy: objectobject.py descr__new__ → allocate bare object
    // In Python, object() returns a featureless object instance.
    // Use PY_NULL as type since there's no builtin 'object' W_TypeObject yet.
    Ok(pyre_object::w_instance_new(pyre_object::PY_NULL))
    // Full implementation requires a base object type in TypeDef.
}

/// `super()` — PyPy: descriptor.py W_Super
/// `super(cls, obj)` — PyPy: superobject.py W_Super
///
/// Returns a proxy that looks up methods in cls's MRO starting after cls.
/// `getattr` handles the super proxy via `is_super` check.
///
/// Zero-arg super() finds __class__ and self from the calling frame.
/// CPython: Objects/typeobject.c super_init
fn builtin_super(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() >= 2 {
        let cls = args[0];
        let obj = args[1];
        return Ok(pyre_object::superobject::w_super_new(cls, obj));
    }
    // Zero-arg super(): find __class__ cell and first arg from calling frame
    //
    // IMPORTANT: CURRENT_FRAME points to the frame that is currently
    // executing the `super()` CALL.  For zero-arg super the __class__
    // cell lives in the *caller* of super(), which IS the current frame
    // (super is a builtin, not a user function that gets its own frame).
    crate::eval::CURRENT_FRAME.with(|current| {
        let frame_ptr = current.get();
        if frame_ptr.is_null() {
            return Err(crate::PyError::runtime_error("super(): no current frame"));
        }
        let frame = unsafe { &*frame_ptr };
        let code = frame.code();

        // Find __class__ in freevars (it's a cell variable from the enclosing class scope)
        let num_locals = code.varnames.len();
        let cellvars_only = code
            .cellvars
            .iter()
            .filter(|c| !code.varnames.contains(c))
            .count();
        let locals = frame.locals_cells_stack_w.as_slice();

        let mut w_class = pyre_object::PY_NULL;

        // Check freevars for __class__
        for (slot, name) in code.freevars.iter().enumerate() {
            if name == "__class__" {
                let idx = num_locals + cellvars_only + slot;
                if idx < locals.len() {
                    let cell = locals[idx];
                    if !cell.is_null() {
                        if unsafe { pyre_object::is_cell(cell) } {
                            w_class = unsafe { pyre_object::w_cell_get(cell) };
                        } else {
                            w_class = cell;
                        }
                    }
                }
                break;
            }
        }

        // Also check cellvars for __class__
        if w_class.is_null() {
            for (slot, name) in code.cellvars.iter().enumerate() {
                if name == "__class__" {
                    let idx = if code.varnames.iter().any(|v| v == name) {
                        code.varnames.iter().position(|v| v == name).unwrap()
                    } else {
                        num_locals + slot
                    };
                    if idx < locals.len() {
                        let cell = locals[idx];
                        if !cell.is_null() {
                            if unsafe { pyre_object::is_cell(cell) } {
                                w_class = unsafe { pyre_object::w_cell_get(cell) };
                            } else {
                                w_class = cell;
                            }
                        }
                    }
                    break;
                }
            }
        }

        if w_class.is_null() {
            return Err(crate::PyError::runtime_error(
                "super(): __class__ cell not found",
            ));
        }

        // First argument is self/cls/mcs (locals[0])
        let w_self = if locals.is_empty() {
            pyre_object::PY_NULL
        } else {
            locals[0]
        };

        if w_self.is_null() {
            return Err(crate::PyError::runtime_error(
                "super(): no first argument found",
            ));
        }

        Ok(pyre_object::superobject::w_super_new(w_class, w_self))
    })
}

/// `iter(obj)` — PyPy: baseobjspace.py iter
fn builtin_iter(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "iter() requires at least one argument",
        ));
    }
    crate::baseobjspace::iter(args[0])
}

/// `next(iterator[, default])` — PyPy: baseobjspace.py next
fn builtin_next(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "next() requires at least one argument",
        ));
    }
    match crate::baseobjspace::next(args[0]) {
        Ok(v) => Ok(v),
        Err(e) if e.kind == crate::PyErrorKind::StopIteration && args.len() > 1 => {
            Ok(args[1]) // default value
        }
        Err(e) => Err(e),
    }
}

/// `callable(obj)` — PyPy: baseobjspace.py callable
fn builtin_callable(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let obj = args[0];
    let is_callable = unsafe {
        crate::is_function(obj)
            || pyre_object::is_type(obj)
            || (pyre_object::is_instance(obj)
                && crate::baseobjspace::lookup_in_type(
                    pyre_object::w_instance_get_type(obj),
                    "__call__",
                )
                .is_some())
    };
    Ok(w_bool_from(is_callable))
}

/// `compile(source, filename, mode, ...)` — PyPy: pyopcode.py builtin_compile
///
/// Compiles a Python string to a code object. Only `source`, `filename` and
/// `mode` are honoured; flags / dont_inherit / optimize are accepted but
/// ignored, matching the minimal stub PyPy uses for shim modules.
fn builtin_compile(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 3 {
        return Err(crate::PyError::type_error(
            "compile() requires source, filename, mode",
        ));
    }
    let source = args[0];
    let filename_obj = args[1];
    let mode_obj = args[2];
    let source_str = unsafe {
        if pyre_object::is_str(source) {
            pyre_object::w_str_get_value(source).to_string()
        } else if pyre_object::bytearrayobject::is_bytearray(source) {
            String::from_utf8_lossy(pyre_object::bytearrayobject::w_bytearray_data(source))
                .into_owned()
        } else {
            return Err(crate::PyError::type_error(
                "compile() arg 1 must be a string or bytes",
            ));
        }
    };
    let filename = unsafe {
        if pyre_object::is_str(filename_obj) {
            pyre_object::w_str_get_value(filename_obj).to_string()
        } else {
            "<string>".to_string()
        }
    };
    let mode = unsafe {
        if pyre_object::is_str(mode_obj) {
            pyre_object::w_str_get_value(mode_obj).to_string()
        } else {
            "exec".to_string()
        }
    };
    let mode = match mode.as_str() {
        "exec" => crate::compile::Mode::Exec,
        "eval" => crate::compile::Mode::Eval,
        "single" => crate::compile::Mode::Single,
        other => {
            return Err(crate::PyError::new(
                crate::PyErrorKind::ValueError,
                format!("compile() mode must be 'exec', 'eval' or 'single', not {other:?}"),
            ));
        }
    };
    let code = crate::compile::compile_source_with_filename(&source_str, mode, &filename)
        .map_err(|e| crate::PyError::new(crate::PyErrorKind::ValueError, e))?;
    let code_ptr = Box::into_raw(Box::new(code)) as *const ();
    Ok(crate::w_code_new(code_ptr))
}

/// `exec(source_or_code, globals=None, locals=None)` — PyPy:
/// pyopcode.py builtin_exec.
///
/// Compiles `source` if necessary, then runs the resulting code object in
/// the supplied namespaces.  When the namespaces are dicts, pyre converts
/// them into `PyNamespace`s before invocation and copies the post-run
/// namespace contents back so that callers see the new bindings.
fn builtin_exec(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error("exec() requires source"));
    }
    let source = args[0];
    let globals_arg = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    let locals_arg = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
    exec_or_eval(source, globals_arg, locals_arg, false)
}

/// `eval(source_or_code, globals=None, locals=None)` — same plumbing as
/// exec but returns the value of the expression.
fn builtin_eval(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error("eval() requires source"));
    }
    let source = args[0];
    let globals_arg = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    let locals_arg = args.get(2).copied().unwrap_or(pyre_object::PY_NULL);
    exec_or_eval(source, globals_arg, locals_arg, true)
}

fn exec_or_eval(
    source: PyObjectRef,
    globals_arg: PyObjectRef,
    locals_arg: PyObjectRef,
    is_eval: bool,
) -> Result<PyObjectRef, crate::PyError> {
    // Resolve a runnable code object: accept a precompiled W_Code or
    // compile a str on the fly.
    let code_obj_ref = unsafe {
        if pyre_object::is_str(source) {
            let s = pyre_object::w_str_get_value(source).to_string();
            let mode = if is_eval {
                crate::compile::Mode::Eval
            } else {
                crate::compile::Mode::Exec
            };
            let code = crate::compile::compile_source(&s, mode)
                .map_err(|e| crate::PyError::new(crate::PyErrorKind::ValueError, e))?;
            let code_ptr = Box::into_raw(Box::new(code)) as *const ();
            crate::w_code_new(code_ptr)
        } else if !source.is_null() && crate::is_code(source) {
            source
        } else {
            return Err(crate::PyError::type_error(
                "exec() / eval() expects str or code",
            ));
        }
    };
    let raw_code = unsafe {
        crate::w_code_get_ptr(code_obj_ref as pyre_object::PyObjectRef) as *const crate::CodeObject
    };

    // Build a PyNamespace from the supplied globals dict (or fall back to
    // the caller's frame globals when None / missing). Mutations made by
    // the executed code propagate back to the original dict via the
    // dict-namespace sync helpers.
    let (ns_box, sync_back_dict) =
        if !globals_arg.is_null() && unsafe { pyre_object::is_dict(globals_arg) } {
            let mut ns = Box::new(crate::PyNamespace::new());
            unsafe {
                for (key, value) in pyre_object::w_dict_items(globals_arg) {
                    if !value.is_null() && pyre_object::is_str(key) {
                        crate::namespace_store(&mut ns, pyre_object::w_str_get_value(key), value);
                    }
                }
            }
            ns.fix_ptr();
            (ns, Some(globals_arg))
        } else {
            // Inherit caller globals.
            let mut ns = Box::new(crate::PyNamespace::new());
            crate::eval::CURRENT_FRAME.with(|current| {
                let frame = current.get();
                if !frame.is_null() {
                    let parent_ns = unsafe { (*frame).namespace };
                    if !parent_ns.is_null() {
                        for (k, &v) in unsafe { &*parent_ns }.entries() {
                            if !v.is_null() {
                                crate::namespace_store(&mut ns, k, v);
                            }
                        }
                    }
                }
            });
            ns.fix_ptr();
            (ns, None)
        };
    // If a separate `locals` dict was passed, layer its bindings on top
    // of the globals namespace. Module-level execs in dataclasses pass
    // the same dict for both, so this also covers `exec(src, g, l)` where
    // `l is g`. Track which dict to write back into.
    let mut ns_box = ns_box;
    let locals_sync = if !locals_arg.is_null()
        && unsafe { pyre_object::is_dict(locals_arg) }
        && !std::ptr::eq(locals_arg, globals_arg)
    {
        unsafe {
            for (key, value) in pyre_object::w_dict_items(locals_arg) {
                if !value.is_null() && pyre_object::is_str(key) {
                    crate::namespace_store(&mut ns_box, pyre_object::w_str_get_value(key), value);
                }
            }
        }
        Some(locals_arg)
    } else {
        None
    };
    let ns_ptr = Box::into_raw(ns_box);

    let exec_ctx = crate::eval::CURRENT_FRAME.with(|current| {
        let frame = current.get();
        if frame.is_null() {
            std::ptr::null::<crate::PyExecutionContext>()
        } else {
            unsafe { (*frame).execution_context }
        }
    });
    let mut frame =
        crate::pyframe::PyFrame::new_with_namespace(code_obj_ref as *const (), exec_ctx, ns_ptr);
    frame.fix_array_ptrs();
    let result = crate::eval::eval_frame_plain(&mut frame);

    // Drain the namespace back into the dict the caller passed in so the
    // exec'd module-level bindings are visible to subsequent code.
    let writeback_target = locals_sync.or(sync_back_dict);
    if let Some(target) = writeback_target {
        unsafe {
            for (k, &v) in (*ns_ptr).entries() {
                if !v.is_null() {
                    pyre_object::w_dict_store(target, pyre_object::w_str_new(k), v);
                }
            }
        }
    }

    let _ = raw_code; // keep raw_code alive until after exec for safety.
    let _ = unsafe { Box::from_raw(ns_ptr) };
    match result {
        Ok(v) if is_eval => Ok(v),
        Ok(_) => Ok(pyre_object::w_none()),
        Err(e) => Err(e),
    }
}

fn builtin_globals(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if !args.is_empty() {
        return Err(crate::PyError::type_error("globals() takes no arguments"));
    }
    crate::eval::CURRENT_FRAME.with(|current| {
        let frame = current.get();
        if frame.is_null() {
            return Err(crate::PyError::runtime_error(
                "globals() requires an active frame",
            ));
        }
        let namespace = unsafe { (*frame).namespace };
        if namespace.is_null() {
            return Err(crate::PyError::runtime_error(
                "globals() requires an active frame",
            ));
        }
        // Create a dict backed by the live namespace. Mutations (update,
        // __setitem__) are synced back to the namespace so that patterns
        // like `globals().update({...})` work correctly.
        let dict = pyre_object::w_dict_new_with_namespace(namespace as *mut u8);
        for (k, &v) in unsafe { &*namespace }.entries() {
            unsafe { pyre_object::w_dict_store(dict, pyre_object::w_str_new(k), v) };
        }
        Ok(dict)
    })
}

fn builtin_locals(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if !args.is_empty() {
        return Err(crate::PyError::type_error("locals() takes no arguments"));
    }
    crate::eval::CURRENT_FRAME.with(|current| {
        let frame = current.get();
        if frame.is_null() {
            return Err(crate::PyError::runtime_error(
                "locals() requires an active frame",
            ));
        }
        let frame = unsafe { &*frame };
        if !frame.class_locals.is_null() {
            let dict = pyre_object::w_dict_new();
            for (k, &v) in unsafe { &*frame.class_locals }.entries() {
                unsafe { pyre_object::w_dict_store(dict, pyre_object::w_str_new(k), v) };
            }
            return Ok(dict);
        }
        let dict = pyre_object::w_dict_new();
        let code = unsafe { &*crate::pyframe_get_pycode(frame) };
        for (idx, name) in code.varnames.iter().enumerate() {
            let value = frame.locals_cells_stack_w[idx];
            if !value.is_null() {
                unsafe { pyre_object::w_dict_store(dict, pyre_object::w_str_new(name), value) };
            }
        }
        if frame.nlocals() == 0 && !frame.namespace.is_null() {
            for (k, &v) in unsafe { &*frame.namespace }.entries() {
                unsafe { pyre_object::w_dict_store(dict, pyre_object::w_str_new(k), v) };
            }
        }
        Ok(dict)
    })
}

fn builtin_vars(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return builtin_locals(args);
    }
    if args.len() != 1 {
        return Err(crate::PyError::type_error(
            "vars() takes at most 1 argument.",
        ));
    }
    let obj = args[0];
    let has_dict = unsafe {
        pyre_object::is_instance(obj) || crate::is_function(obj) || pyre_object::is_module(obj)
    } || crate::baseobjspace::ATTR_TABLE
        .with(|table| table.borrow().contains_key(&(obj as usize)));
    if !has_dict {
        return Err(crate::PyError::type_error(
            "vars() argument must have __dict__ attribute",
        ));
    }
    let dict = crate::baseobjspace::getattr(obj, "__dict__")
        .map_err(|_| crate::PyError::type_error("vars() argument must have __dict__ attribute"))?;
    if dict.is_null() || unsafe { pyre_object::is_none(dict) } {
        return Err(crate::PyError::type_error(
            "vars() argument must have __dict__ attribute",
        ));
    }
    Ok(dict)
}

/// `dir([obj])` — PyPy: pypy/module/__builtin__/interp_classobj.py descr_dir
///
/// Without argument: names in the current local scope (not supported).
/// With argument: sorted list of attribute names from obj.__dict__ plus
/// type MRO. Modules expose their namespace via w_module_get_namespace.
fn builtin_dir(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        // Return empty list — pyre doesn't currently expose locals here.
        return Ok(w_list_new(vec![]));
    }
    let obj = args[0];
    let mut names: Vec<String> = Vec::new();
    unsafe {
        if pyre_object::is_module(obj) {
            let ns_ptr = pyre_object::moduleobject::w_module_get_dict_ptr(obj);
            if !ns_ptr.is_null() {
                let ns = &*(ns_ptr as *const PyNamespace);
                for (name, _) in ns.entries() {
                    names.push(name.to_string());
                }
            }
        } else if pyre_object::is_type(obj) {
            let ns_ptr = pyre_object::typeobject::w_type_get_dict_ptr(obj);
            if !ns_ptr.is_null() {
                let ns = &*(ns_ptr as *const PyNamespace);
                for (name, _) in ns.entries() {
                    names.push(name.to_string());
                }
            }
        } else if pyre_object::is_instance(obj) {
            // typeobject.py:1247 type_dir collects names from the instance
            // dict and the type's MRO. The instance dict for hasdict objects
            // is the live W_DictObject returned by w_obj.getdict(space).
            let w_dict = crate::baseobjspace::getdict(obj);
            if !w_dict.is_null() {
                for (k, _) in pyre_object::w_dict_items(w_dict) {
                    if pyre_object::is_str(k) {
                        names.push(pyre_object::w_str_get_value(k).to_string());
                    }
                }
            }
            // Plus any legacy ATTR_TABLE entries (slot values stored via Member
            // descriptors before the live-dict path existed).
            crate::baseobjspace::ATTR_TABLE.with(|table| {
                if let Some(attrs) = table.borrow().get(&(obj as usize)) {
                    for (name, _) in attrs {
                        names.push(name.clone());
                    }
                }
            });
            // Plus the type's own namespace.
            let w_type = pyre_object::w_instance_get_type(obj);
            if !w_type.is_null() && pyre_object::is_type(w_type) {
                let ns_ptr = pyre_object::typeobject::w_type_get_dict_ptr(w_type);
                if !ns_ptr.is_null() {
                    let ns = &*(ns_ptr as *const PyNamespace);
                    for (name, _) in ns.entries() {
                        names.push(name.to_string());
                    }
                }
            }
        } else if pyre_object::is_dict(obj) {
            for (k, _) in pyre_object::w_dict_items(obj) {
                if pyre_object::is_str(k) {
                    names.push(pyre_object::w_str_get_value(k).to_string());
                }
            }
        }
    }
    names.sort();
    names.dedup();
    let items: Vec<_> = names.into_iter().map(|s| w_str_new(&s)).collect();
    Ok(w_list_new(items))
}

/// `id(obj)` — PyPy: baseobjspace.py id → object identity as int
fn builtin_id(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty(), "id() takes exactly one argument");
    Ok(w_int_new(args[0] as i64))
}

/// `hash(obj)` — PyPy: baseobjspace.py hash → identity for now
fn builtin_hash(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty(), "hash() takes exactly one argument");
    unsafe {
        if is_int(args[0]) {
            return Ok(args[0]); // int hashes to itself
        }
        if is_str(args[0]) {
            // Simplified string hash — deterministic within one run
            let s = w_str_get_value(args[0]);
            let mut h: i64 = 0;
            for b in s.bytes() {
                h = h.wrapping_mul(1000003).wrapping_add(b as i64);
            }
            return Ok(w_int_new(h));
        }
    }
    // Instance __hash__ — PyPy: baseobjspace.py hash_w
    unsafe {
        if pyre_object::is_instance(args[0]) {
            let w_type = pyre_object::w_instance_get_type(args[0]);
            if let Some(method) = crate::baseobjspace::lookup_in_type(w_type, "__hash__") {
                return Ok(crate::call_function(method, &[args[0]]));
            }
        }
    }
    Ok(w_int_new(args[0] as i64)) // identity hash fallback
}

/// `ord(c)` — PyPy: operation.py ord
fn builtin_ord(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "ord() takes exactly one argument");
    let s = unsafe { w_str_get_value(args[0]) };
    assert!(s.len() == 1, "ord() expected a character");
    Ok(w_int_new(s.chars().next().unwrap() as i64))
}

/// `chr(i)` — PyPy: operation.py chr
fn builtin_chr(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "chr() takes exactly one argument");
    let obj = args[0];
    let val = if unsafe { is_int(obj) } {
        unsafe { w_int_get_value(obj) }
    } else if unsafe { is_str(obj) } {
        // pyre treats `b'...'` literals as str; iterating yields 1-char str.
        // Accept single-char str and use its codepoint as the value.
        let s = unsafe { w_str_get_value(obj) };
        match s.chars().next() {
            Some(c) => c as i64,
            None => {
                return Err(crate::PyError::type_error(
                    "chr() arg must be a non-empty string",
                ));
            }
        }
    } else {
        // int subclass instance — check __int_value__ via builtin_int
        match builtin_int(args) {
            Ok(v) if unsafe { is_int(v) } => unsafe { w_int_get_value(v) },
            _ => {
                return Err(crate::PyError::type_error(
                    "an integer is required (got type non-int)",
                ));
            }
        }
    };
    if val < 0 || val > 0x10ffff {
        return Err(crate::PyError::type_error(&format!(
            "chr() arg not in range(0x110000): {val}"
        )));
    }
    match char::from_u32(val as u32) {
        Some(c) => Ok(w_str_new(&c.to_string())),
        None => Err(crate::PyError::type_error(&format!(
            "chr() arg not in range(0x110000): {val}"
        ))),
    }
}

/// `map()` — PyPy: functional.py W_Map (returns iterator)
fn builtin_map(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.len() < 2 {
        return Err(crate::PyError::type_error(
            "map() requires at least 2 arguments",
        ));
    }
    let func = args[0];
    let iterable = args[1];
    let items = collect_iterable(iterable)?;
    let mut results = Vec::with_capacity(items.len());
    for item in items {
        let result = crate::call_function(func, &[item]);
        results.push(result);
    }
    let n = results.len();
    let list = pyre_object::w_list_new(results);
    Ok(pyre_object::w_seq_iter_new(list, n))
}

/// `zip(*iterables)` — PyPy: functional.py W_Zip
fn builtin_zip(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Ok(pyre_object::w_seq_iter_new(
            pyre_object::w_list_new(vec![]),
            0,
        ));
    }
    // Collect all iterables into lists, zip them
    let mut iters: Vec<Vec<PyObjectRef>> = Vec::new();
    for &arg in args {
        let mut items = Vec::new();
        unsafe {
            if pyre_object::is_list(arg) {
                let n = pyre_object::w_list_len(arg);
                for i in 0..n {
                    if let Some(v) = pyre_object::w_list_getitem(arg, i as i64) {
                        items.push(v);
                    }
                }
            } else if pyre_object::is_tuple(arg) {
                let n = pyre_object::w_tuple_len(arg);
                for i in 0..n {
                    if let Some(v) = pyre_object::w_tuple_getitem(arg, i as i64) {
                        items.push(v);
                    }
                }
            } else {
                // Use iter/next protocol
                let it = crate::baseobjspace::iter(arg)?;
                loop {
                    match crate::baseobjspace::next(it) {
                        Ok(v) => items.push(v),
                        Err(e) if e.kind == crate::PyErrorKind::StopIteration => break,
                        Err(e) => return Err(e),
                    }
                }
            }
        }
        iters.push(items);
    }
    let min_len = iters.iter().map(|v| v.len()).min().unwrap_or(0);
    let mut result = Vec::with_capacity(min_len);
    for i in 0..min_len {
        let tuple_items: Vec<_> = iters.iter().map(|v| v[i]).collect();
        result.push(pyre_object::w_tuple_new(tuple_items));
    }
    let list = pyre_object::w_list_new(result);
    Ok(pyre_object::w_seq_iter_new(list, min_len))
}

/// `enumerate(iterable, start=0)` — PyPy: functional.py W_Enumerate
fn builtin_enumerate(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "enumerate() requires at least one argument",
        ));
    }
    let start = if args.len() > 1 {
        unsafe {
            if pyre_object::is_int(args[1]) {
                pyre_object::w_int_get_value(args[1])
            } else {
                0
            }
        }
    } else {
        0
    };
    // Collect iterable, pair with indices
    let mut items = Vec::new();
    unsafe {
        let obj = args[0];
        if pyre_object::is_list(obj) {
            let n = pyre_object::w_list_len(obj);
            for i in 0..n {
                if let Some(v) = pyre_object::w_list_getitem(obj, i as i64) {
                    items.push(pyre_object::w_tuple_new(vec![
                        pyre_object::w_int_new(start + i as i64),
                        v,
                    ]));
                }
            }
        } else {
            let it = crate::baseobjspace::iter(obj)?;
            let mut idx = start;
            loop {
                match crate::baseobjspace::next(it) {
                    Ok(v) => {
                        items.push(pyre_object::w_tuple_new(vec![
                            pyre_object::w_int_new(idx),
                            v,
                        ]));
                        idx += 1;
                    }
                    Err(e) if e.kind == crate::PyErrorKind::StopIteration => break,
                    Err(e) => return Err(e),
                }
            }
        }
    }
    let n = items.len();
    let list = pyre_object::w_list_new(items);
    Ok(pyre_object::w_seq_iter_new(list, n))
}

/// `reversed()` — PyPy: functional.py W_ReversedIterator
fn builtin_reversed(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    if args.is_empty() {
        return Err(crate::PyError::type_error(
            "reversed() requires one argument",
        ));
    }
    let obj = args[0];
    unsafe {
        // List: reverse a copy
        if pyre_object::is_list(obj) {
            let n = pyre_object::w_list_len(obj);
            let mut items = Vec::with_capacity(n);
            for i in (0..n as i64).rev() {
                if let Some(v) = pyre_object::w_list_getitem(obj, i) {
                    items.push(v);
                }
            }
            return Ok(pyre_object::w_seq_iter_new(
                pyre_object::w_list_new(items),
                n,
            ));
        }
        // Tuple: reverse
        if pyre_object::is_tuple(obj) {
            let n = pyre_object::w_tuple_len(obj);
            let mut items = Vec::with_capacity(n);
            for i in (0..n as i64).rev() {
                if let Some(v) = pyre_object::w_tuple_getitem(obj, i) {
                    items.push(v);
                }
            }
            let t = pyre_object::w_tuple_new(items);
            return Ok(pyre_object::w_seq_iter_new(t, n));
        }
        // Instance __reversed__
        if pyre_object::is_instance(obj) {
            let w_type = pyre_object::w_instance_get_type(obj);
            if let Some(method) = crate::baseobjspace::lookup_in_type(w_type, "__reversed__") {
                return Ok(crate::call_function(method, &[obj]));
            }
        }
    }
    Err(crate::PyError::type_error(
        "argument to reversed() must be a sequence",
    ))
}

/// `sorted(iterable)` — PyPy: listobject.py listsort
///
/// Returns a new sorted list. Simplified: sorts by int value.
fn builtin_sorted(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty(), "sorted() takes at least one argument");
    let iterable = args[0];
    let mut items = collect_iterable(iterable)?;
    unsafe {
        items.sort_by(|a, b| {
            if is_int(*a) && is_int(*b) {
                w_int_get_value(*a).cmp(&w_int_get_value(*b))
            } else if is_str(*a) && is_str(*b) {
                w_str_get_value(*a).cmp(w_str_get_value(*b))
            } else {
                std::cmp::Ordering::Equal
            }
        });
    }
    Ok(w_list_new(items))
}

/// `any(iterable)` — PyPy: operation.py any
/// `any(iterable)` — PyPy: baseobjspace.py any_w
pub fn builtin_any_fn(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    builtin_any(args)
}
fn builtin_any(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty(), "any() takes exactly one argument");
    let items = collect_iterable(args[0])?;
    for item in items {
        if crate::baseobjspace::is_true(item) {
            return Ok(w_bool_from(true));
        }
    }
    Ok(w_bool_from(false))
}

/// `all(iterable)` — PyPy: operation.py all
/// `all(iterable)` — PyPy: baseobjspace.py all_w
pub fn builtin_all_fn(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    builtin_all(args)
}
fn builtin_all(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty(), "all() takes exactly one argument");
    let items = collect_iterable(args[0])?;
    for item in items {
        if !crate::baseobjspace::is_true(item) {
            return Ok(w_bool_from(false));
        }
    }
    Ok(w_bool_from(true))
}

/// `sum(iterable, start=0)` — PyPy: operation.py sum
fn builtin_sum(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty(), "sum() takes at least one argument");
    let iterable = args[0];
    let start = args.get(1).copied().unwrap_or_else(|| w_int_new(0));
    let mut acc = start;
    unsafe {
        if is_list(iterable) {
            let n = w_list_len(iterable);
            for i in 0..n {
                if let Some(item) = w_list_getitem(iterable, i as i64) {
                    acc = crate::baseobjspace::add(acc, item).expect("sum: unsupported type");
                }
            }
            return Ok(acc);
        }
        if is_tuple(iterable) {
            let n = w_tuple_len(iterable);
            for i in 0..n {
                if let Some(item) = w_tuple_getitem(iterable, i as i64) {
                    acc = crate::baseobjspace::add(acc, item).expect("sum: unsupported type");
                }
            }
            return Ok(acc);
        }
    }
    panic!("sum() argument must be list or tuple");
}

/// `round(number, ndigits=None)` — PyPy: operation.py round
fn builtin_round(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty(), "round() takes at least one argument");
    let obj = args[0];
    let ndigits = args.get(1);
    unsafe {
        if is_float(obj) {
            let v = floatobject::w_float_get_value(obj);
            return Ok(match ndigits {
                Some(nd) if is_int(*nd) => {
                    let n = w_int_get_value(*nd);
                    let factor = 10f64.powi(n as i32);
                    floatobject::w_float_new((v * factor).round() / factor)
                }
                _ => w_int_new(v.round() as i64),
            });
        }
        if is_int(obj) {
            return Ok(obj);
        }
    }
    panic!("round() not supported for this type");
}

/// `divmod(a, b)` — pypy/interpreter/baseobjspace.py:2159 divmod row.
fn builtin_divmod(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 2, "divmod() takes exactly two arguments");
    crate::baseobjspace::divmod(args[0], args[1])
}

/// `pow(base, exp[, mod])` — pypy/interpreter/baseobjspace.py:2160 pow row.
fn builtin_pow(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() >= 2, "pow() takes at least two arguments");
    if args.len() >= 3 && !unsafe { is_none(args[2]) } {
        crate::baseobjspace::pow3(args[0], args[1], args[2])
    } else {
        crate::baseobjspace::pow(args[0], args[1])
    }
}

/// `hex(x)` — PyPy: operation.py hex
fn builtin_hex(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "hex() takes exactly one argument");
    let v = unsafe { w_int_get_value(args[0]) };
    let s = if v < 0 {
        format!("-0x{:x}", -v)
    } else {
        format!("0x{v:x}")
    };
    Ok(w_str_new(&s))
}

/// `oct(x)` — PyPy: operation.py oct
fn builtin_oct(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "oct() takes exactly one argument");
    let v = unsafe { w_int_get_value(args[0]) };
    let s = if v < 0 {
        format!("-0o{:o}", -v)
    } else {
        format!("0o{v:o}")
    };
    Ok(w_str_new(&s))
}

/// `bin(x)` — PyPy: operation.py bin
fn builtin_bin(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(args.len() == 1, "bin() takes exactly one argument");
    let v = unsafe { w_int_get_value(args[0]) };
    let s = if v < 0 {
        format!("-0b{:b}", -v)
    } else {
        format!("0b{v:b}")
    };
    Ok(w_str_new(&s))
}

/// `format(value, format_spec='')` — PyPy: operation.py format
fn builtin_format(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    assert!(!args.is_empty(), "format() takes at least one argument");
    // Simplified: format without format_spec returns str(value)
    let s = unsafe { crate::py_str(args[0]) };
    Ok(w_str_new(&s))
}

/// `__import__(name, globals=None, locals=None, fromlist=(), level=0)`
/// — PyPy: pypy/module/__builtin__/interp_import.importhook.
fn builtin_import_stub(args: &[PyObjectRef]) -> Result<PyObjectRef, crate::PyError> {
    let name_obj = args.first().copied().unwrap_or(pyre_object::PY_NULL);
    let name = if !name_obj.is_null() && unsafe { pyre_object::is_str(name_obj) } {
        unsafe { pyre_object::w_str_get_value(name_obj) }
    } else {
        ""
    };
    let globals = args.get(1).copied().unwrap_or(pyre_object::PY_NULL);
    let fromlist = args.get(3).copied().unwrap_or(pyre_object::PY_NULL);
    let level = args
        .get(4)
        .copied()
        .filter(|&a| unsafe { pyre_object::is_int(a) })
        .map(|a| unsafe { pyre_object::w_int_get_value(a) })
        .unwrap_or(0);
    let exec_ctx = crate::eval::CURRENT_FRAME.with(|current| {
        let frame = current.get();
        if frame.is_null() {
            std::ptr::null::<crate::PyExecutionContext>()
        } else {
            unsafe { (*frame).execution_context }
        }
    });
    crate::importing::importhook(name, globals, fromlist, level, exec_ctx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builtin_divmod_delegates_through_proxy() {
        crate::typedef::init_typeobjects();
        let proxy = crate::module::_weakref::interp_weakref::W_Proxy_new(w_int_new(5), PY_NULL);
        let result = builtin_divmod(&[proxy, w_int_new(3)]).unwrap();
        assert_eq!(
            unsafe { w_int_get_value(w_tuple_getitem(result, 0).unwrap()) },
            1
        );
        assert_eq!(
            unsafe { w_int_get_value(w_tuple_getitem(result, 1).unwrap()) },
            2
        );
    }

    #[test]
    fn test_builtin_divmod_allows_lhs_dunder_before_dead_proxy_rhs() {
        crate::typedef::init_typeobjects();
        let user_type = crate::typedef::make_builtin_type("DivmodLhs", |ns| {
            crate::namespace_store(
                ns,
                "__divmod__",
                make_builtin_function("__divmod__", |_| {
                    Ok(w_tuple_new(vec![w_int_new(41), w_int_new(1)]))
                }),
            );
        });
        let lhs = pyre_object::instanceobject::w_instance_new(user_type);
        let dead_proxy = crate::module::_weakref::interp_weakref::W_Proxy_new(w_none(), PY_NULL);
        let result = builtin_divmod(&[lhs, dead_proxy]).unwrap();
        assert_eq!(
            unsafe { w_int_get_value(w_tuple_getitem(result, 0).unwrap()) },
            41
        );
        assert_eq!(
            unsafe { w_int_get_value(w_tuple_getitem(result, 1).unwrap()) },
            1
        );
    }

    #[test]
    fn test_builtin_pow_three_arg_delegates_through_proxy() {
        crate::typedef::init_typeobjects();
        let proxy = crate::module::_weakref::interp_weakref::W_Proxy_new(w_int_new(5), PY_NULL);
        let result = builtin_pow(&[proxy, w_int_new(3), w_int_new(13)]).unwrap();
        assert_eq!(unsafe { w_int_get_value(result) }, 8);
    }

    #[test]
    fn test_builtin_pow_two_arg_delegates_through_proxy() {
        crate::typedef::init_typeobjects();
        let proxy = crate::module::_weakref::interp_weakref::W_Proxy_new(w_int_new(5), PY_NULL);
        let result = builtin_pow(&[proxy, w_int_new(3)]).unwrap();
        assert_eq!(unsafe { w_int_get_value(result) }, 125);
    }

    #[test]
    fn test_builtin_pow_three_arg_allows_lhs_dunder_before_dead_proxy_exp() {
        crate::typedef::init_typeobjects();
        let user_type = crate::typedef::make_builtin_type("PowLhs", |ns| {
            crate::namespace_store(
                ns,
                "__pow__",
                make_builtin_function("__pow__", |_| Ok(w_int_new(99))),
            );
        });
        let lhs = pyre_object::instanceobject::w_instance_new(user_type);
        let dead_proxy = crate::module::_weakref::interp_weakref::W_Proxy_new(w_none(), PY_NULL);
        let result = builtin_pow(&[lhs, dead_proxy, w_int_new(7)]).unwrap();
        assert_eq!(unsafe { w_int_get_value(result) }, 99);
    }

    #[test]
    fn test_builtin_pow_three_arg_rejects_negative_exponent() {
        crate::typedef::init_typeobjects();
        let err = builtin_pow(&[w_int_new(5), w_int_new(-1), w_int_new(13)]).unwrap_err();
        assert_eq!(err.kind, crate::PyErrorKind::TypeError);
        assert_eq!(
            err.message,
            "pow() 2nd argument cannot be negative when 3rd argument specified"
        );
    }
}
