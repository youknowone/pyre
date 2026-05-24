//! _collections implementation — PyPy: pypy/module/_collections/interp_deque.py
//!
//! Verbatim move of the inline block previously in importing.rs.  The
//! deque and defaultdict type initialisers stay private; `init_collections_c`
//! is renamed to `register_module`.

use crate::DictStorage;

/// _collections C-extension stub — PyPy: pypy/module/_collections/
/// Provides the C-accelerated deque/defaultdict/OrderedDict types.
/// Our stubs are backed by lists/dicts, which is correct semantically
/// but not performant. PyPy's W_Deque is a doubly-linked block list.
pub fn register_module(ns: &mut DictStorage) {
    // deque(iterable=(), maxlen=None) — returns a list that we alias as deque.
    // Sufficient for collections.py's MutableSequence.register(deque).
    let deque_type = crate::typedef::make_builtin_type("deque", init_deque_type);
    crate::dict_storage_store(ns, "deque", deque_type);
    // _deque_iterator — reuse object (just a type sentinel)
    crate::dict_storage_store(ns, "_deque_iterator", crate::typedef::w_object());
    // defaultdict — returns a dict-like instance
    let defaultdict_type = crate::typedef::make_builtin_type("defaultdict", init_defaultdict_type);
    crate::dict_storage_store(ns, "defaultdict", defaultdict_type);
    // OrderedDict — same as dict for our purposes
    crate::dict_storage_store(ns, "OrderedDict", crate::typedef::w_type());
}

/// deque methods — PyPy: pypy/module/_collections/interp_deque.py W_Deque
fn init_deque_type(ns: &mut DictStorage) {
    // __init__(self, iterable=(), maxlen=None) — store items as __data__ list
    crate::dict_storage_store(
        ns,
        "__init__",
        crate::make_builtin_function("__init__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            let self_obj = args[0];
            let items: Vec<_> = if args.len() >= 2 {
                crate::builtins::collect_iterable(args[1]).unwrap_or_default()
            } else {
                Vec::new()
            };
            let list = pyre_object::w_list_new(items);
            let _ = crate::baseobjspace::setattr(self_obj, "__data__", list);
            let _ = crate::baseobjspace::setattr(
                self_obj,
                "maxlen",
                if args.len() >= 3 {
                    args[2]
                } else {
                    pyre_object::w_none()
                },
            );
            Ok(pyre_object::w_none())
        }),
    );
    crate::dict_storage_store(
        ns,
        "append",
        crate::make_builtin_function_with_arity(
            "append",
            |args| {
                if args.len() >= 2 {
                    if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                        unsafe { pyre_object::w_list_append(data, args[1]) };
                    }
                }
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );
    crate::dict_storage_store(
        ns,
        "appendleft",
        crate::make_builtin_function_with_arity(
            "appendleft",
            |args| {
                if args.len() >= 2 {
                    if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                        unsafe {
                            let n = pyre_object::w_list_len(data);
                            let mut items: Vec<_> = (0..n)
                                .filter_map(|i| pyre_object::w_list_getitem(data, i as i64))
                                .collect();
                            items.insert(0, args[1]);
                            let new_list = pyre_object::w_list_new(items);
                            let _ = crate::baseobjspace::setattr(args[0], "__data__", new_list);
                        }
                    }
                }
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );
    crate::dict_storage_store(
        ns,
        "pop",
        crate::make_builtin_function_with_arity(
            "pop",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_none());
                }
                if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                    unsafe {
                        let n = pyre_object::w_list_len(data);
                        if n > 0 {
                            let item = pyre_object::w_list_getitem(data, (n - 1) as i64)
                                .unwrap_or(pyre_object::w_none());
                            let items: Vec<_> = (0..n - 1)
                                .filter_map(|i| pyre_object::w_list_getitem(data, i as i64))
                                .collect();
                            let new_list = pyre_object::w_list_new(items);
                            let _ = crate::baseobjspace::setattr(args[0], "__data__", new_list);
                            return Ok(item);
                        }
                    }
                }
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "popleft",
        crate::make_builtin_function_with_arity(
            "popleft",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_none());
                }
                if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                    unsafe {
                        let n = pyre_object::w_list_len(data);
                        if n > 0 {
                            let item = pyre_object::w_list_getitem(data, 0)
                                .unwrap_or(pyre_object::w_none());
                            let items: Vec<_> = (1..n)
                                .filter_map(|i| pyre_object::w_list_getitem(data, i as i64))
                                .collect();
                            let new_list = pyre_object::w_list_new(items);
                            let _ = crate::baseobjspace::setattr(args[0], "__data__", new_list);
                            return Ok(item);
                        }
                    }
                }
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "clear",
        crate::make_builtin_function_with_arity(
            "clear",
            |args| {
                if !args.is_empty() {
                    let _ = crate::baseobjspace::setattr(
                        args[0],
                        "__data__",
                        pyre_object::w_list_new(vec![]),
                    );
                }
                Ok(pyre_object::w_none())
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "extend",
        crate::make_builtin_function_with_arity(
            "extend",
            |args| {
                if args.len() >= 2 {
                    let items = crate::builtins::collect_iterable(args[1])?;
                    if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                        for item in items {
                            unsafe { pyre_object::w_list_append(data, item) };
                        }
                    }
                }
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );
    crate::dict_storage_store(
        ns,
        "__len__",
        crate::make_builtin_function_with_arity(
            "__len__",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_int_new(0));
                }
                if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                    return Ok(pyre_object::w_int_new(
                        unsafe { pyre_object::w_list_len(data) } as i64,
                    ));
                }
                Ok(pyre_object::w_int_new(0))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "__iter__",
        crate::make_builtin_function_with_arity(
            "__iter__",
            |args| {
                if args.is_empty() {
                    return Ok(pyre_object::w_seq_iter_new(
                        pyre_object::w_list_new(vec![]),
                        0,
                    ));
                }
                if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                    return crate::baseobjspace::iter(data);
                }
                Ok(pyre_object::w_seq_iter_new(
                    pyre_object::w_list_new(vec![]),
                    0,
                ))
            },
            1,
        ),
    );
    crate::dict_storage_store(
        ns,
        "__getitem__",
        crate::make_builtin_function_with_arity(
            "__getitem__",
            |args| {
                if args.len() < 2 {
                    return Ok(pyre_object::w_none());
                }
                if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                    return crate::baseobjspace::getitem(data, args[1]);
                }
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );
}

/// defaultdict — PyPy: pypy/module/_collections/interp_defaultdict.py
fn init_defaultdict_type(ns: &mut DictStorage) {
    crate::dict_storage_store(
        ns,
        "__init__",
        crate::make_builtin_function("__init__", |args| {
            if args.is_empty() {
                return Ok(pyre_object::w_none());
            }
            let self_obj = args[0];
            let factory = if args.len() >= 2 {
                args[1]
            } else {
                pyre_object::w_none()
            };
            let _ = crate::baseobjspace::setattr(self_obj, "default_factory", factory);
            let _ = crate::baseobjspace::setattr(self_obj, "__data__", pyre_object::w_dict_new());
            Ok(pyre_object::w_none())
        }),
    );
    crate::dict_storage_store(
        ns,
        "__getitem__",
        crate::make_builtin_function_with_arity(
            "__getitem__",
            |args| {
                if args.len() < 2 {
                    return Ok(pyre_object::w_none());
                }
                let self_obj = args[0];
                let key = args[1];
                if let Ok(data) = crate::baseobjspace::getattr(self_obj, "__data__") {
                    unsafe {
                        if let Some(v) = pyre_object::w_dict_lookup(data, key) {
                            return Ok(v);
                        }
                    }
                    // Not present — try factory
                    if let Ok(factory) = crate::baseobjspace::getattr(self_obj, "default_factory") {
                        if !factory.is_null() && !unsafe { pyre_object::is_none(factory) } {
                            // Can't easily call factory without frame — return None.
                            let default = pyre_object::w_none();
                            unsafe { pyre_object::w_dict_store(data, key, default) };
                            return Ok(default);
                        }
                    }
                }
                Ok(pyre_object::w_none())
            },
            2,
        ),
    );
    crate::dict_storage_store(
        ns,
        "__setitem__",
        crate::make_builtin_function_with_arity(
            "__setitem__",
            |args| {
                if args.len() >= 3 {
                    if let Ok(data) = crate::baseobjspace::getattr(args[0], "__data__") {
                        unsafe { pyre_object::w_dict_store(data, args[1], args[2]) };
                    }
                }
                Ok(pyre_object::w_none())
            },
            3,
        ),
    );
}
