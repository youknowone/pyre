//! Semantic graph rewrite passes.
//!
//! RPython equivalents:
//! - `annotate`: annotator/annrpython.py (type propagation to fixpoint)
//! - `jtransform`: jit/codewriter/jtransform.py (JIT-specific graph rewriting)

pub mod annotate;
pub mod flatten;
mod jtransform;
pub mod pipeline;
pub mod rtype;

pub use annotate::{AnnotationState, annotate as annotate_graph};
pub use flatten::{FlatOp, Label, RegKind, SSARepr, flatten, flatten_with_types};
pub use jtransform::{
    CallEffectKind, CallEffectOverride, GraphTransformConfig, GraphTransformNote,
    GraphTransformResult, Transformer, VirtualizableFieldDescriptor, rewrite_graph,
};
pub use pipeline::{
    PipelineConfig, PipelineOpcodeArm, PipelineResult, ProgramPipelineResult, analyze_function,
    analyze_program,
};
pub use rtype::{ConcreteType, TypeResolutionState, resolve_types};
