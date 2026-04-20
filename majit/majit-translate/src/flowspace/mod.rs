//! `flowspace` — Rust port of `rpython/flowspace/`.
//!
//! Phase 1 of the five-year roadmap landing a line-by-line port of
//! `rpython/{flowspace,annotator,rtyper}/` into majit-translate. See
//! `.claude/plans/majestic-forging-meteor.md` for the full plan.
//!
//! RPython upstream lives at
//! `/Users/al03219714/Projects/pypy/rpython/flowspace/`. The
//! corresponding `__init__.py` is empty, so this module's `mod.rs`
//! only declares the submodules that exist.
//!
//! Module → upstream file mapping (populated incrementally as each
//! roadmap sub-phase lands):
//!
//! | majit                        | rpython/flowspace/              |
//! |------------------------------|---------------------------------|
//! | `model`                      | `model.py`                      |
//! | `bytecode`                   | `bytecode.py`                   |
//! | `framestate`                 | `framestate.py`                 |
//! | `operation`                  | `operation.py`                  |
//! | `flowcontext`                 | `flowcontext.py`                |
//! | `flowcontext_py314` (Phase 3)| PYRE-ONLY, no RPython basis     |
//! | `pygraph`                    | `pygraph.py`                    |
//! | `generator`                  | `generator.py`                  |
//! | `argument`                   | `argument.py`                   |
//! | `objspace`                   | `objspace.py`                   |
//! | `specialcase`                | `specialcase.py`                |

pub mod argument;
pub mod bytecode;
pub mod flowcontext;
pub mod framestate;
pub mod generator;
pub mod model;
pub mod objspace;
pub mod operation;
pub mod pygraph;
pub mod specialcase;
