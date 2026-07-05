//! `annotator` — Rust port of `rpython/annotator/`.
//!
//! Builds the
//! type-lattice (`SomeValue` hierarchy) + forward-propagation driver
//! (`RPythonAnnotator`) that feeds the rtyper.
//!
//! RPython upstream lives at `rpython/annotator/`.
//! Upstream `__init__.py` is 4 LOC; this `mod.rs` only declares the
//! submodules that have been landed.
//!
//! Module → upstream file mapping (populated incrementally per roadmap):
//!
//! | majit                         | rpython/annotator/              |
//! |-------------------------------|---------------------------------|
//! | `model`                       | `model.py`                      |
//! | `annrpython`                   | `annrpython.py`                 |
//! | `argument`                     | `argument.py`                   |
//! | `binaryop`                     | `binaryop.py`                   |
//! | `unaryop`                      | `unaryop.py`                    |
//! | `bookkeeper`                   | `bookkeeper.py`                 |
//! | `builtin`                      | `builtin.py`                    |
//! | `description`                  | `description.py`                |
//! | `classdesc`                    | `classdesc.py`                  |
//! | `dictdef`                      | `dictdef.py`                    |
//! | `listdef`                      | `listdef.py`                    |
//! | `exception`                    | `exception.py`                  |
//! | `policy`                       | `policy.py`                     |
//! | `signature`                    | `signature.py`                  |
//! | `specialize`                   | `specialize.py`                 |

pub mod annrpython;
pub mod argument;
pub mod binaryop;
pub mod bookkeeper;
pub mod builtin;
pub mod classdesc;
pub mod description;
pub mod dictdef;
pub mod exception;
pub mod listdef;
pub mod model;
pub mod policy;
pub(crate) mod repr_guard;
pub mod signature;
pub mod specialize;
pub mod unaryop;
