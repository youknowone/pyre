//! Canonical virtualizable field/array specification for `PyFrame`.
//!
//! This module is intentionally data-only so both runtime code and
//! `build.rs` can share the same translator-facing layout contract.

pub const PYFRAME_VABLE_OWNER_ROOT: &str = "PyFrame";

/// Virtualizable scalar fields in canonical PyPy index order.
/// interp_jit.py:25-31: last_instr, pycode, valuestackdepth,
/// debugdata, lastblock, w_globals
pub const PYFRAME_VABLE_FIELDS: &[(&str, usize)] = &[
    ("last_instr", 0),
    ("pycode", 1),
    ("valuestackdepth", 2), // interp_jit.py:26 valuestackdepth
    ("debugdata", 3),       // interp_jit.py:28 debugdata
    ("lastblock", 4),       // interp_jit.py:29 lastblock
    ("w_globals", 5),       // interp_jit.py:31 w_globals
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
