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
//! pyre's interpreter is compiled by rustc, so its hand-written bracket is
//! checked over ULLBC by [`liveness`].  The generated translator graph is a
//! separate owner: [`shadowcolor`] is the strict flowspace-graph port used
//! after `gc_push_roots` / `gc_pop_roots` have been emitted there.
//!
//! # Why not the insertion half
//!
//! Upstream's transformer runs inside the translator, between the RTyper and
//! the backend, and rewrites the graphs it is about to emit.  This crate has
//! no such position over the interpreter: `majit-translate` lowers LLBC for
//! the *JIT*, and the interpreter's machine code comes out of rustc, which
//! this pipeline never sees.  Inserting the bracket automatically would mean
//! either a rustc MIR pass or LLVM statepoints — a different project with a
//! different toolchain, not a port of `gctransform`.  The charon artefact is
//! read-only: it is emitted from the same source rustc compiles, so it can
//! *answer* `framework.py`'s question about that code, but nothing written
//! back into it would reach the binary.
//!
//! So the hand-written brackets stay, and `liveness::scan` reports the calls
//! that can collect with a GC pointer live across them and no bracket.  This
//! does not replace the ordinary translator-side transformation: that remains
//! the flow-graph algorithm in `shadowcolor.py`, ported by [`shadowcolor`].
pub mod framework;
pub mod liveness;
pub mod shadowcolor;
