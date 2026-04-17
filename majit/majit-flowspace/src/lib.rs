//! majit-flowspace — Rust port of `rpython/flowspace/`.
//!
//! Phase 1 of the five-year roadmap landing a line-by-line port of
//! `rpython/{flowspace,annotator,rtyper}/` into majit. See
//! `.claude/plans/majestic-forging-meteor.md` for the full plan.
//!
//! RPython upstream lives at
//! `/Users/al03219714/Projects/pypy/rpython/flowspace/`. The
//! corresponding `__init__.py` is empty, so this crate's `lib.rs`
//! only declares the submodules that exist.
//!
//! Module → upstream file mapping (populated incrementally as each
//! roadmap sub-phase lands):
//!
//! | majit                        | rpython/flowspace/              |
//! |------------------------------|---------------------------------|
//! | `model`                      | `model.py`                      |
//! | `bytecode`        (Phase 2)  | `bytecode.py`                   |
//! | `framestate`      (Phase 3)  | `framestate.py`                 |
//! | `operation`       (Phase 3)  | `operation.py`                  |
//! | `flowcontext`     (Phase 3)  | `flowcontext.py`                |
//! | `flowcontext_py314` (Phase 3)| PYRE-ONLY, no RPython basis     |
//! | `pygraph`         (Phase 3)  | `pygraph.py`                    |
//! | `generator`       (Phase 3)  | `generator.py`                  |
//! | `argument`        (Phase 3)  | `argument.py`                   |
//! | `objspace`        (Phase 3)  | `objspace.py`                   |
//! | `specialcase`     (Phase 3)  | `specialcase.py`                |

pub mod model;
