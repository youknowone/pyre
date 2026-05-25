//! _random module — PyPy: pypy/module/_random/
//!
//! Minimal `Random` class backed by a small linear congruential
//! generator — enough for `random.py` to construct its `_inst` at
//! module import time.

crate::pyre_module_init!(interp_random);
