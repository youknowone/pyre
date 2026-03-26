//! Canonical virtualizable field/array specification for `PyFrame`.
//!
//! This module is intentionally data-only so both runtime code and
//! `build.rs` can share the same translator-facing layout contract.

pub const PYFRAME_VABLE_OWNER_ROOT: &str = "PyFrame";

#[derive(Clone, Copy)]
pub enum FieldPatternRole {
    LocalArray,
    InstructionPosition,
    ConstantPool,
}

#[derive(Clone, Copy)]
pub struct FieldRoleSpec {
    pub name: &'static str,
    pub owner_root: &'static str,
    pub role: FieldPatternRole,
}

/// Virtualizable scalar fields in canonical index order.
pub const PYFRAME_VABLE_FIELDS: &[(&str, usize)] = &[("next_instr", 0), ("valuestackdepth", 1)];

/// Virtualizable array fields in canonical index order.
pub const PYFRAME_VABLE_ARRAYS: &[(&str, usize)] = &[("locals_cells_stack_w", 0)];

/// Canonical field-role descriptors used by trace-pattern classification.
pub const PYFRAME_FIELD_ROLES: &[FieldRoleSpec] = &[
    FieldRoleSpec {
        name: "locals_cells_stack_w",
        owner_root: PYFRAME_VABLE_OWNER_ROOT,
        role: FieldPatternRole::LocalArray,
    },
    FieldRoleSpec {
        name: "next_instr",
        owner_root: PYFRAME_VABLE_OWNER_ROOT,
        role: FieldPatternRole::InstructionPosition,
    },
    FieldRoleSpec {
        name: "last_instr",
        owner_root: PYFRAME_VABLE_OWNER_ROOT,
        role: FieldPatternRole::InstructionPosition,
    },
    FieldRoleSpec {
        name: "pc",
        owner_root: PYFRAME_VABLE_OWNER_ROOT,
        role: FieldPatternRole::InstructionPosition,
    },
    FieldRoleSpec {
        name: "co_consts",
        owner_root: "PyCode",
        role: FieldPatternRole::ConstantPool,
    },
    FieldRoleSpec {
        name: "constants",
        owner_root: "Code",
        role: FieldPatternRole::ConstantPool,
    },
];
