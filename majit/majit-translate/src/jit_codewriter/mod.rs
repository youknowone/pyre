//! Line-by-line port of `rpython/jit/codewriter/`.
//!
//! Contains the low-level codewriter stage that converts an rtyped graph
//! into `JitCode` consumed by `majit-metainterp`. Sibling modules in this
//! directory mirror `rpython/jit/codewriter/*.py` one-to-one.

pub mod annotation_state;
pub mod assembler;
pub mod call;
pub mod codewriter;
pub mod effectinfo;
pub mod flatten;
pub mod format;
pub mod heaptracker;
pub mod insns;
pub mod jitcode;
pub mod jtransform;
pub mod longlong;
// Opname-dispatch transducer ("Spine B"): lowers rtyper low-level helper
// graphs (opname `SpaceOperation`s) to rich-`OpKind` graphs that re-enter the
// shared flatten/regalloc/assembler tail. Port of `jtransform.py`'s
// `_rewrite_ops[op.opname]` dispatch; see the module docs.
pub mod jtransform_opname;
// No upstream sibling: an inert, env-gated diagnostic that gauges how much
// of the rtyped flowspace graph an opname-dispatching jtransform would
// already accept. Never on the production path; see the module docs.
pub mod jtransform_shadow;
pub mod liveness;
pub mod policy;
pub mod regalloc;
pub mod support;
pub mod transform_profile;
pub mod type_state;
