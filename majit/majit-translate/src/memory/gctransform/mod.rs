//! `rpython/memory/gctransform/` — the GC root transformer.
//!
//! Upstream inserts the shadow-stack save/reload bracket automatically:
//! `framework.py` brackets every collecting operation, `shadowstack.py`
//! emits the `gc_push_roots` / `gc_pop_roots` pair, `shadowcolor.py` colours
//! the saves like registers, and `postprocess_double_check` verifies the
//! placement.  Application code therefore carries no rooting at all —
//! `pypy/objspace/std/listobject.py descr_index` names its receiver and its
//! needle as plain locals.
//!
//! pyre's interpreter is compiled by rustc, so there is no flow graph for a
//! `hop.genop` to write into and the bracket is written by hand
//! (`pyre_object::gc_roots`).  What this module ports is therefore the
//! *analysis* half: the same questions, answered over the ULLBC charon emits
//! for those crates, so a hand-written bracket can be checked instead of
//! trusted.
pub mod framework;
pub mod liveness;
