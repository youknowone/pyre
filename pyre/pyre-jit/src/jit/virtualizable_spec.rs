//! Canonical virtualizable field/array specification for `PyFrame`.
//!
//! This module is intentionally data-only so both runtime code and
//! `build.rs` can share the same translator-facing layout contract.

/// Virtualizable scalar fields in canonical index order.
pub const PYFRAME_VABLE_FIELDS: &[(&str, usize)] = &[("next_instr", 0), ("valuestackdepth", 1)];

/// Virtualizable array fields in canonical index order.
pub const PYFRAME_VABLE_ARRAYS: &[(&str, usize)] = &[("locals_cells_stack_w", 0)];
