//! The label carrier of `rpython/jit/codewriter/flatten.py`, which the
//! assembler's switch descrs keep until labels are resolved.

use serde::{Deserialize, Serialize};

/// A label in the flattened instruction stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Label(pub usize);

/// `flatten.py class TLabel`.
///
/// Rust encodes the definition-vs-target distinction in `FlatOp` variants
/// (`Label` vs jump targets), so the target wrapper shares the same numeric
/// label carrier.
pub type TLabel = Label;
