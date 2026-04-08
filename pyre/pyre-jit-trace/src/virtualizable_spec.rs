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
/// interp_jit.py:25-32: last_instr, pycode, valuestackdepth, ..., w_globals
/// pyre maps: last_instr → next_instr, pycode → code, w_globals → namespace
pub const PYFRAME_VABLE_FIELDS: &[(&str, usize)] = &[
    ("next_instr", 0),      // interp_jit.py:25 last_instr
    ("code", 1),            // interp_jit.py:25 pycode
    ("valuestackdepth", 2), // interp_jit.py:26 valuestackdepth
    ("namespace", 3),       // interp_jit.py:31 w_globals
];

/// Virtualizable array fields in canonical index order.
pub const PYFRAME_VABLE_ARRAYS: &[(&str, usize)] = &[("locals_cells_stack_w", 0)];

/// Canonical vable-array index for `locals_cells_stack_w`.
///
/// PyFrame's unified locals+cells+stack vector is the first (and currently
/// only) virtualizable array. Portal `LOAD_FAST`/`STORE_FAST` in the
/// codewriter use this constant with the Python `var_num` as item index to
/// emit `GETARRAYITEM_VABLE_R` / `SETARRAYITEM_VABLE_R`.
///
/// Compile-time invariants guarded below: the entry must be present at
/// index 0 and named `"locals_cells_stack_w"`.
pub const LOCALS_CELLS_STACK_W_VABLE_ARRAY_INDEX: usize = 0;

const _: () = {
    assert!(
        !PYFRAME_VABLE_ARRAYS.is_empty(),
        "PYFRAME_VABLE_ARRAYS must contain locals_cells_stack_w"
    );
    assert!(
        PYFRAME_VABLE_ARRAYS[LOCALS_CELLS_STACK_W_VABLE_ARRAY_INDEX].1
            == LOCALS_CELLS_STACK_W_VABLE_ARRAY_INDEX,
        "locals_cells_stack_w must be registered at the expected vable array index"
    );
    // Verify the name bytewise — no `str::eq` in const context.
    let name = PYFRAME_VABLE_ARRAYS[LOCALS_CELLS_STACK_W_VABLE_ARRAY_INDEX]
        .0
        .as_bytes();
    let expected = b"locals_cells_stack_w";
    assert!(
        name.len() == expected.len(),
        "PYFRAME_VABLE_ARRAYS[0] name mismatch"
    );
    let mut i = 0;
    while i < expected.len() {
        assert!(
            name[i] == expected[i],
            "PYFRAME_VABLE_ARRAYS[0] name mismatch"
        );
        i += 1;
    }
};

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
