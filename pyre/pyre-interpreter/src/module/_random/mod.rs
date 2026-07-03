//! _random module — PyPy `pypy/module/_random/`.
//!
//! `W_Random` wraps a Mersenne Twister generator (`rrandom.Random`) and
//! exposes `random`/`seed`/`getstate`/`setstate`/`getrandbits`, matching
//! `pypy/module/_random/interp_random.py W_Random`.

use malachite_bigint::{BigInt, Sign};
use pyre_object::*;

/// Mersenne Twister state — `rpython/rlib/rrandom.py Random`.
const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908b0df;
const UPPER_MASK: u32 = 0x80000000;
const LOWER_MASK: u32 = 0x7fffffff;
const TEMPERING_MASK_A: u32 = 0x9d2c5680;
const TEMPERING_MASK_B: u32 = 0xefc60000;
const MAGIC_CONSTANT_A: u32 = 1812433253;
const MAGIC_CONSTANT_B: u32 = 19650218;
const MAGIC_CONSTANT_C: u32 = 1664525;
const MAGIC_CONSTANT_D: u32 = 1566083941;

struct Random {
    state: [u32; N],
    index: usize,
}

impl Default for Random {
    fn default() -> Self {
        let mut rnd = Random {
            state: [0; N],
            index: 0,
        };
        rnd.init_genrand(0);
        rnd
    }
}

impl Random {
    fn init_genrand(&mut self, s: u32) {
        let mt = &mut self.state;
        mt[0] = s;
        for mti in 1..N {
            mt[mti] = MAGIC_CONSTANT_A
                .wrapping_mul(mt[mti - 1] ^ (mt[mti - 1] >> 30))
                .wrapping_add(mti as u32);
        }
        self.index = N;
    }

    fn init_by_array(&mut self, init_key: &[u32]) {
        let key_length = init_key.len();
        self.init_genrand(MAGIC_CONSTANT_B);
        let mut i = 1usize;
        let mut j = 0usize;
        let max_k = if N > key_length { N } else { key_length };
        for _ in 0..max_k {
            let mt = &mut self.state;
            mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(MAGIC_CONSTANT_C))
                .wrapping_add(init_key[j])
                .wrapping_add(j as u32); // non linear
            i += 1;
            j += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
            if j >= key_length {
                j = 0;
            }
        }
        for _ in (1..N).rev() {
            let mt = &mut self.state;
            mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(MAGIC_CONSTANT_D))
                .wrapping_sub(i as u32); // non linear
            i += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
        }
        self.state[0] = UPPER_MASK;
    }

    fn conditionally_apply(val: u32, y: u32) -> u32 {
        if y & 1 != 0 { val ^ MATRIX_A } else { val }
    }

    fn genrand32(&mut self) -> u32 {
        if self.index >= N {
            let mt = &mut self.state;
            for kk in 0..(N - M) {
                let y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
                mt[kk] = Random::conditionally_apply(mt[kk + M] ^ (y >> 1), y);
            }
            for kk in (N - M)..(N - 1) {
                let y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
                mt[kk] = Random::conditionally_apply(mt[kk + M - N] ^ (y >> 1), y);
            }
            let y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
            mt[N - 1] = Random::conditionally_apply(mt[M - 1] ^ (y >> 1), y);
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & TEMPERING_MASK_A;
        y ^= (y << 15) & TEMPERING_MASK_B;
        y ^= y >> 18;
        y
    }

    fn random(&mut self) -> f64 {
        let a = (self.genrand32() >> 5) as f64;
        let b = (self.genrand32() >> 6) as f64;
        (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)
    }
}

#[crate::pyre_class("_random.Random")]
#[derive(Default)]
pub struct W_Random {
    rnd: Random,
}

#[crate::pyre_methods(
    doc = "Random() -> create a random number generator.\n\nNot for security or cryptographic use.",
    weakrefable
)]
impl W_Random {
    fn __init__(
        &mut self,
        #[default(pyre_object::w_none())] w_anything: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        self.seed(w_anything)
    }

    fn random(&mut self) -> f64 {
        self.rnd.random()
    }

    fn seed(
        &mut self,
        #[default(pyre_object::w_none())] w_n: PyObjectRef,
    ) -> Result<(), crate::PyError> {
        // None: seed from os.urandom(8); fall back to a time-based int only
        // when urandom raises (interp_random.py:28). Under sandbox the entropy
        // comes from the trusted controller, not host getrandom.
        let w_n = if unsafe { is_none(w_n) } {
            #[cfg(not(feature = "sandbox"))]
            let entropy = crate::importing::host::os::urandom(8).ok();
            #[cfg(feature = "sandbox")]
            let entropy = crate::host_seam::ops::urandom(8).ok();
            match entropy {
                Some(buf) => w_bytes_from_bytes(&buf),
                None => w_int_new(seed_from_time() as i64),
            }
        } else {
            w_n
        };
        let n = unsafe {
            if is_int_or_long(w_n) {
                // space.abs(w_n)
                let v = crate::builtins::obj_to_bigint(w_n);
                if v.sign() == Sign::Minus { -v } else { v }
            } else {
                // n = space.hash_w(w_n); w_n = space.newint(r_uint(n))
                BigInt::from(crate::baseobjspace::hash_w_strict(w_n)? as u64)
            }
        };
        // Split into little-endian 32-bit chunks.
        let (_sign, key) = n.to_u32_digits();
        let key = if key.is_empty() { vec![0u32] } else { key };
        self.rnd.init_by_array(&key);
        Ok(())
    }

    fn getstate(&self) -> PyObjectRef {
        let mut state: Vec<PyObjectRef> = Vec::with_capacity(N + 1);
        for i in 0..N {
            state.push(w_int_new(self.rnd.state[i] as i64));
        }
        state.push(w_int_new(self.rnd.index as i64));
        w_tuple_new(state)
    }

    fn setstate(&mut self, w_state: PyTuple) -> Result<(), crate::PyError> {
        unsafe {
            if w_tuple_len(w_state) != N + 1 {
                crate::bail_value_error!("state vector is the wrong size");
            }
            let mut new_state = [0u32; N];
            for i in 0..N {
                let Some(item) = w_tuple_getitem(w_state, i as i64) else {
                    crate::bail_value_error!("state vector is the wrong size");
                };
                if !is_int_or_long(item) {
                    crate::bail_type_error!("state vector must contain ints");
                }
                let mut v = crate::builtins::obj_to_bigint(item);
                if v.sign() == Sign::Minus {
                    v += BigInt::from(1u64 << 32);
                }
                // space.uint_w: every word must fit an unsigned 32-bit int.
                if v.sign() == Sign::Minus {
                    crate::bail_overflow_error!("cannot convert negative integer to unsigned int");
                }
                if v >= BigInt::from(1u64 << 32) {
                    crate::bail_overflow_error!("int too large to convert to unsigned int");
                }
                let (_s, digits) = v.to_u32_digits();
                new_state[i] = digits.first().copied().unwrap_or(0);
            }
            let Some(item) = w_tuple_getitem(w_state, N as i64) else {
                crate::bail_value_error!("state vector is the wrong size");
            };
            // space.int_w: handles long overflow / non-int (TypeError).
            let index = crate::baseobjspace::int_w(item)?;
            if index < 0 || index > N as i64 {
                crate::bail_value_error!("invalid state");
            }
            self.rnd.state = new_state;
            self.rnd.index = index as usize;
        }
        Ok(())
    }

    fn getrandbits(&mut self, k: PyIndex) -> Result<PyObjectRef, crate::PyError> {
        let mut k = k as i64;
        if k < 0 {
            crate::bail_value_error!("number of bits must be non-negative");
        }
        if k == 0 {
            return Ok(w_int_new(0));
        }
        if k < 32 {
            // fits an int, skip the bytes-to-long dance
            let r = self.rnd.genrand32() >> (32 - k);
            return Ok(w_int_new(r as i64));
        }
        let nbytes = (((k - 1) / 32 + 1) * 4) as usize;
        let mut bytes = Vec::with_capacity(nbytes);
        while k > 0 {
            let mut r = self.rnd.genrand32();
            if k < 32 {
                r >>= 32 - k;
            }
            bytes.push((r & 0xff) as u8);
            bytes.push(((r >> 8) & 0xff) as u8);
            bytes.push(((r >> 16) & 0xff) as u8);
            bytes.push(((r >> 24) & 0xff) as u8);
            k -= 32;
        }
        // little endian order to match the byte append order
        let result = BigInt::from_bytes_le(Sign::Plus, &bytes);
        Ok(w_long_new(result))
    }
}

/// Time-based fallback seed — `int(time.time() * 256)`.
fn seed_from_time() -> u64 {
    // Under sandbox the clock is read through the trusted controller, not the
    // host directly.
    #[cfg(feature = "sandbox")]
    let secs = crate::host_seam::ops::time().unwrap_or(0.0);
    #[cfg(not(feature = "sandbox"))]
    let secs = {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0)
    };
    (secs * 256.0) as u64
}

crate::py_module! {
    "_random",
    interpleveldefs: {
        "Random" => type_object(),
    },
}

#[cfg(test)]
mod macro_smoke;
