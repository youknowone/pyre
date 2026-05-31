//! _random module — PyPy `pypy/module/_random/`.
//!
//! Provides the `Random` class backed by a small xorshift PRNG, enough
//! for `random.py` to construct its module-level instance at import time.
//! Real programs subclass `random.Random` instead.
//!
//! `W_Random` stores its state inline as a `u64` field and methods take
//! `&mut self` directly, matching
//! `pypy/module/_random/interp_random.py W_Random`'s `self._rnd.random()`
//! style.

use pyre_object::*;

const DEFAULT_SEED: u64 = 0x1234_5678;

#[crate::pyre_class("_random.Random")]
#[derive(Default)]
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

#[crate::pyre_methods(
    doc = "Random() -> create a random number generator.\n\nNot for security or cryptographic use.",
    weakrefable
)]
impl W_Random {
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
    fn getrandbits(&mut self, #[default(32i64)] k: PyIndex) -> Result<i64, crate::PyError> {
        if k < 0 {
            crate::bail_value_error!("number of bits must be non-negative");
        }
        self.state = xorshift(self.state);
        let k = k as u32;
        let mask = if k >= 64 { u64::MAX } else { (1u64 << k) - 1 };
        Ok((self.state & mask) as i64)
    }
    fn getstate(&self) -> PyObjectRef {
        crate::pytuple![self.state as i64]
    }
    fn setstate(&mut self, state_tuple: PyTuple) -> Result<(), crate::PyError> {
        unsafe {
            if w_tuple_len(state_tuple) < 1 {
                crate::bail_value_error!("setstate: tuple must have at least 1 element");
            }
            let Some(state) = w_tuple_getitem(state_tuple, 0) else {
                crate::bail_value_error!("setstate: missing state element");
            };
            if !is_int(state) {
                crate::bail_type_error!("setstate: element 0 must be int");
            }
            self.state = w_int_get_value(state) as u64;
        }
        Ok(())
    }
}

crate::py_module! {
    "_random",
    interpleveldefs: {
        "Random" => type_object(),
    },
}

#[cfg(test)]
mod macro_smoke;
