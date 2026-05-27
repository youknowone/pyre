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
    #[getter]
    fn raw_state(&self) -> i64 {
        self.state as i64
    }
    #[setter]
    fn set_raw_state(&mut self, v: i64) {
        self.state = v as u64;
    }
}

/// `_seed_from_path(path)` — PyPath alias smoke: accepts str, bytes, or
/// any `os.PathLike` and converts via `space.fsencode_w` parity.
#[crate::pyre_function]
fn _seed_from_path(path: PyPath) -> i64 {
    path.bytes().map(|b| b as i64).sum()
}

/// `_path_bytes(path)` — `Vec<i64>` return auto-wraps to a list per
/// PyPy `space.newlist([space.newint(b) for b in bytes])`.
#[crate::pyre_function]
fn _path_bytes(path: PyPath) -> Vec<i64> {
    path.bytes().map(|b| b as i64).collect()
}

crate::py_module! {
    "_random",
    interpleveldefs: {
        "Random" => type_object(),
        "_seed_from_path" => crate::make_builtin_function("_seed_from_path", _seed_from_path),
        "_path_bytes" => crate::make_builtin_function("_path_bytes", _path_bytes),
    },
    appleveldefs: {
        "app_random.py" => ["_ascii_seed"],
    },
    inline_app: {
        "def _is_even(n):\n    return n % 2 == 0\n" => ["_is_even"],
    },
}
