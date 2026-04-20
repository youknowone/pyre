//! `annotator` — Rust port of `rpython/annotator/`.
//!
//! Phase 4-5 of the five-year roadmap (see
//! `.claude/plans/majestic-forging-meteor.md`). Builds the
//! type-lattice (`SomeValue` hierarchy) + forward-propagation driver
//! (`RPythonAnnotator`) that feeds the rtyper.
//!
//! RPython upstream lives at `/Users/al03219714/Projects/pypy/rpython/annotator/`.
//! Upstream `__init__.py` is 4 LOC; this `mod.rs` only declares the
//! submodules that have been landed.
//!
//! Module → upstream file mapping (populated incrementally per roadmap):
//!
//! | majit                         | rpython/annotator/              |
//! |-------------------------------|---------------------------------|
//! | `model`                       | `model.py`                      |
//! | `annrpython`      (Phase 5)   | `annrpython.py`                 |
//! | `argument`        (Phase 5)   | `argument.py`                   |
//! | `binaryop`        (Phase 5)   | `binaryop.py`                   |
//! | `unaryop`         (Phase 5)   | `unaryop.py`                    |
//! | `bookkeeper`      (Phase 5)   | `bookkeeper.py`                 |
//! | `builtin`         (Phase 5)   | `builtin.py`                    |
//! | `description`     (Phase 5)   | `description.py`                |
//! | `classdesc`       (Phase 5)   | `classdesc.py`                  |
//! | `dictdef`         (Phase 5)   | `dictdef.py`                    |
//! | `listdef`         (Phase 5)   | `listdef.py`                    |
//! | `exception`       (Phase 5)   | `exception.py`                  |
//! | `policy`          (Phase 5)   | `policy.py`                     |
//! | `signature`       (Phase 5)   | `signature.py`                  |
//! | `specialize`      (Phase 5)   | `specialize.py`                 |

pub mod dictdef;
pub mod exception;
pub mod listdef;
pub mod model;
