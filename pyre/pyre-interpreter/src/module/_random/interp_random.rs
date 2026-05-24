//! _random implementation — PyPy: pypy/module/_random/interp_random.py
//!
//! Verbatim move of the inline block previously in importing.rs.

use crate::DictStorage;
use pyre_object::PyObjectRef;

/// `_random` C-extension stub — PyPy: pypy/module/_random/interp_random.py.
///
/// Provides a minimal `Random` class that wraps a very small linear
/// congruential generator. Good enough for `random.py` to construct a
/// `random._inst` at module import time; real tests can then use the
/// Python `random.Random` subclass as a drop-in.
pub fn register_module(ns: &mut DictStorage) {
    fn random_type() -> PyObjectRef {
        thread_local! {
            static RANDOM_TYPE: std::cell::OnceCell<PyObjectRef> = const { std::cell::OnceCell::new() };
        }
        RANDOM_TYPE.with(|c| {
            *c.get_or_init(|| {
                let tp = crate::typedef::make_builtin_type("_random.Random", |ns| {
                    // random_method_* are defined in importing.rs; routing
                    // through make_builtin_function binds them as unbound
                    // methods so `rand.random()` calls pass `self` as args[0].
                    crate::dict_storage_store(
                        ns,
                        "__init__",
                        crate::make_builtin_function("__init__", |args| {
                            let seed = if args.len() >= 2 {
                                unsafe {
                                    if pyre_object::is_int(args[1]) {
                                        pyre_object::w_int_get_value(args[1]) as u64
                                    } else {
                                        0x1234_5678
                                    }
                                }
                            } else {
                                0x1234_5678
                            };
                            let _ = crate::baseobjspace::setattr(
                                args[0],
                                "__rand_state__",
                                pyre_object::w_int_new(seed as i64),
                            );
                            Ok(pyre_object::w_none())
                        }),
                    );
                    crate::dict_storage_store(
                        ns,
                        "seed",
                        crate::make_builtin_function("seed", |args| {
                            let seed = if args.len() >= 2 {
                                unsafe {
                                    if pyre_object::is_int(args[1]) {
                                        pyre_object::w_int_get_value(args[1]) as u64
                                    } else {
                                        0x1234_5678
                                    }
                                }
                            } else {
                                0x1234_5678
                            };
                            let _ = crate::baseobjspace::setattr(
                                args[0],
                                "__rand_state__",
                                pyre_object::w_int_new(seed as i64),
                            );
                            Ok(pyre_object::w_none())
                        }),
                    );
                    crate::dict_storage_store(
                        ns,
                        "random",
                        crate::make_builtin_function_with_arity(
                            "random",
                            |args| {
                                // Tiny xorshift PRNG — ok for import-time construction.
                                let self_obj = args[0];
                                let state =
                                    crate::baseobjspace::getattr(self_obj, "__rand_state__")
                                        .ok()
                                        .map(|v| unsafe { pyre_object::w_int_get_value(v) as u64 })
                                        .unwrap_or(0x1234_5678);
                                let mut x = state;
                                x ^= x << 13;
                                x ^= x >> 7;
                                x ^= x << 17;
                                let _ = crate::baseobjspace::setattr(
                                    self_obj,
                                    "__rand_state__",
                                    pyre_object::w_int_new(x as i64),
                                );
                                Ok(pyre_object::w_float_new((x as f64) / (u64::MAX as f64)))
                            },
                            1,
                        ),
                    );
                    crate::dict_storage_store(
                        ns,
                        "getrandbits",
                        crate::make_builtin_function("getrandbits", |args| {
                            let k = if args.len() >= 2 {
                                unsafe { pyre_object::w_int_get_value(args[1]) as u32 }
                            } else {
                                32
                            };
                            let state = crate::baseobjspace::getattr(args[0], "__rand_state__")
                                .ok()
                                .map(|v| unsafe { pyre_object::w_int_get_value(v) as u64 })
                                .unwrap_or(0x1234_5678);
                            let mut x = state;
                            x ^= x << 13;
                            x ^= x >> 7;
                            x ^= x << 17;
                            let _ = crate::baseobjspace::setattr(
                                args[0],
                                "__rand_state__",
                                pyre_object::w_int_new(x as i64),
                            );
                            let mask = if k >= 64 { u64::MAX } else { (1u64 << k) - 1 };
                            Ok(pyre_object::w_int_new((x & mask) as i64))
                        }),
                    );
                    crate::dict_storage_store(
                        ns,
                        "getstate",
                        crate::make_builtin_function_with_arity(
                            "getstate",
                            |args| {
                                let state = crate::baseobjspace::getattr(args[0], "__rand_state__")
                                    .unwrap_or_else(|_| pyre_object::w_int_new(0));
                                Ok(pyre_object::w_tuple_new(vec![state]))
                            },
                            1,
                        ),
                    );
                    crate::dict_storage_store(
                        ns,
                        "setstate",
                        crate::make_builtin_function_with_arity(
                            "setstate",
                            |args| {
                                if args.len() >= 2 {
                                    unsafe {
                                        if pyre_object::is_tuple(args[1])
                                            && pyre_object::w_tuple_len(args[1]) >= 1
                                        {
                                            if let Some(state) =
                                                pyre_object::w_tuple_getitem(args[1], 0)
                                            {
                                                let _ = crate::baseobjspace::setattr(
                                                    args[0],
                                                    "__rand_state__",
                                                    state,
                                                );
                                            }
                                        }
                                    }
                                }
                                Ok(pyre_object::w_none())
                            },
                            2,
                        ),
                    );
                });
                unsafe { pyre_object::typeobject::w_type_set_hasdict(tp, true) };
                tp
            })
        })
    }
    crate::dict_storage_store(ns, "Random", random_type());
}
