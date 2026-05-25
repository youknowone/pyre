//! _random module — PyPy: `pypy/module/_random/`.
//!
//! Minimal `Random` class backed by a small xorshift PRNG — enough for
//! `random.py` to construct its `_inst` at module import time.  Real
//! tests can then subclass `random.Random` as a drop-in.

use pyre_object::*;

const DEFAULT_SEED: u64 = 0x1234_5678;

fn read_state(self_obj: PyObjectRef) -> u64 {
    crate::baseobjspace::getattr(self_obj, "__rand_state__")
        .ok()
        .map(|v| unsafe { w_int_get_value(v) as u64 })
        .unwrap_or(DEFAULT_SEED)
}

fn write_state(self_obj: PyObjectRef, state: u64) {
    let _ = crate::baseobjspace::setattr(
        self_obj,
        "__rand_state__",
        w_int_new(state as i64),
    );
}

fn xorshift(state: u64) -> u64 {
    let mut x = state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

crate::py_class! {
    "_random.Random",
    methods: {
        fn __init__(self_obj: PyObjectRef, seed: Option<i64>) {
            write_state(self_obj, seed.unwrap_or(DEFAULT_SEED as i64) as u64);
        }
        fn seed(self_obj: PyObjectRef, s: Option<i64>) {
            write_state(self_obj, s.unwrap_or(DEFAULT_SEED as i64) as u64);
        }
        fn random(self_obj: PyObjectRef) -> f64 {
            let x = xorshift(read_state(self_obj));
            write_state(self_obj, x);
            (x as f64) / (u64::MAX as f64)
        }
        fn getrandbits(self_obj: PyObjectRef, k: Option<u32>) -> i64 {
            let k = k.unwrap_or(32);
            let x = xorshift(read_state(self_obj));
            write_state(self_obj, x);
            let mask = if k >= 64 { u64::MAX } else { (1u64 << k) - 1 };
            (x & mask) as i64
        }
        fn getstate(self_obj: PyObjectRef) -> PyObjectRef {
            let state = crate::baseobjspace::getattr(self_obj, "__rand_state__")
                .unwrap_or_else(|_| w_int_new(0));
            w_tuple_new(vec![state])
        }
        fn setstate(self_obj: PyObjectRef, state_tuple: PyObjectRef) {
            unsafe {
                if is_tuple(state_tuple) && w_tuple_len(state_tuple) >= 1 {
                    if let Some(state) = w_tuple_getitem(state_tuple, 0) {
                        let _ = crate::baseobjspace::setattr(
                            self_obj, "__rand_state__", state);
                    }
                }
            }
        }
    }
}

crate::py_module! {
    "_random",
    interpleveldefs: {
        "Random" => type_object(),
    },
}
