//! Front-end scaffolding for semantic graph construction.

pub mod ast;

pub use ast::{
    AstGraphOptions, SemanticFunction, SemanticProgram, StructFieldRegistry,
    build_semantic_program, build_semantic_program_from_parsed_files,
};
