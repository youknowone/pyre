//! Semantic graph rewrite passes.
//!
//! RPython equivalents:
//! - `annotate`: annotator/annrpython.py (type propagation to fixpoint)
//! - `jtransform`: jit/codewriter/jtransform.py (JIT-specific graph rewriting)

pub mod annotate;
pub mod flatten;
pub mod rtype;
mod jtransform;

pub use annotate::{AnnotationState, annotate as annotate_graph};
pub use flatten::{FlatOp, FlattenedFunction, Label, flatten};
pub use rtype::{ConcreteType, TypeResolutionState, resolve_types};
pub use jtransform::{
    GraphTransformConfig, GraphTransformNote, GraphTransformResult, rewrite_graph,
};
