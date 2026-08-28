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
//! [`shadowstack`] now owns the same graph-wide postprocessing hook as PyPy:
//! the backend invokes it once over the translator's complete graph set, so
//! application functions carry no per-function attribute.  It applies
//! [`shadowcolor`] after the framework transformer has emitted
//! `gc_push_roots` / `gc_pop_roots` markers.
//!
//! The marker-insertion half of `framework.py` has not landed yet.  Native
//! interpreter paths therefore retain their existing `push_roots` brackets,
//! and [`liveness`] checks those brackets over the extracted ULLBC.  That is a
//! current port boundary, not an implementation-language exception: once the
//! framework handlers are ported, they feed the same automatic graph stage.
pub mod framework;
pub mod liveness;
pub mod shadowcolor;
pub mod shadowstack;
