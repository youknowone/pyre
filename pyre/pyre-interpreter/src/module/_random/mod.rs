//! _random module — PyPy: `pypy/module/_random/`.
//!
//! Minimal `Random` class backed by a small xorshift PRNG — enough for
//! `random.py` to construct its `_inst` at module import time.  Real
//! tests can then subclass `random.Random` as a drop-in.
//!
//! `W_Random` is the typed-payload demo for the `#[pyre_class]` /
//! `#[pyre_methods]` pipeline: the state lives inline as a `u64` field
//! on the Rust struct and methods receive `&mut self` directly,
//! matching `pypy/module/_random/interp_random.py W_Random`'s
//! `self._rnd.random()` style.

use pyre_object::*;

const DEFAULT_SEED: u64 = 0x1234_5678;

#[crate::pyre_class("_random.Random", type_id = 53)]
pub struct W_Random {
    pub state: u64,
}

fn xorshift(state: u64) -> u64 {
    let mut x = state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

#[crate::pyre_methods]
impl W_Random {
    #[staticmethod]
    fn __new__(_cls: PyObjectRef) -> PyObjectRef {
        W_Random::allocate(W_Random {
            ob: PyObject {
                ob_type: std::ptr::null(),
                w_class: std::ptr::null_mut(),
            },
            state: DEFAULT_SEED,
        })
    }
    fn __init__(&mut self, #[default(DEFAULT_SEED as i64)] seed: i64) {
        self.state = seed as u64;
    }
    fn seed(&mut self, #[default(DEFAULT_SEED as i64)] s: i64) {
        self.state = s as u64;
    }
    fn random(&mut self) -> f64 {
        self.state = xorshift(self.state);
        (self.state as f64) / (u64::MAX as f64)
    }
    fn getrandbits(&mut self, #[default(32u32)] k: u32) -> i64 {
        self.state = xorshift(self.state);
        let mask = if k >= 64 { u64::MAX } else { (1u64 << k) - 1 };
        (self.state & mask) as i64
    }
    fn getstate(&self) -> PyObjectRef {
        w_tuple_new(vec![w_int_new(self.state as i64)])
    }
    fn setstate(&mut self, state_tuple: PyObjectRef) {
        unsafe {
            if is_tuple(state_tuple) && w_tuple_len(state_tuple) >= 1
                && let Some(state) = w_tuple_getitem(state_tuple, 0)
                && is_int(state)
            {
                self.state = w_int_get_value(state) as u64;
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
