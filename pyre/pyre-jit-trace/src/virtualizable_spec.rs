//! Canonical virtualizable field/array specification for `PyFrame`.
//!
//! This module is intentionally data-only so both runtime code and
//! `build.rs` can share the same translator-facing layout contract.

pub const PYFRAME_VABLE_OWNER_ROOT: &str = "PyFrame";

/// Virtualizable scalar fields.
///
/// This table is the scalar subset of
/// `pypy/module/pypyjit/interp_jit.py`'s `_virtualizable_` list,
/// in declaration order. `PyFrame.lastblock` is deliberately absent:
/// the frame model tracked by this tree has no block stack. Unwind uses
/// the `co_exceptiontable` lookup at
/// `pypy/interpreter/pyopcode.py lookup_exceptiontable`, and pyre's
/// 3.14 bytecode emits no `SETUP_*` / `POP_BLOCK`, so nothing mutates
/// the field inside a trace. It remains an ordinary heap field with a
/// plain `FieldDescr` and a GC root slot. If block opcodes are ever
/// reintroduced, re-add it here and emit `_opimpl_setfield_vable` from
/// their handlers; a layout slot with no setfield is not tracking.
pub const PYFRAME_VABLE_FIELDS: &[(&str, usize)] = &[
    ("last_instr", 0),      // interp_jit.py:25 last_instr
    ("pycode", 1),          // interp_jit.py:25 pycode
    ("valuestackdepth", 2), // interp_jit.py:26 valuestackdepth
    ("debugdata", 3),       // interp_jit.py:28 debugdata
    ("w_globals", 4),       // interp_jit.py:29 w_globals
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

/// Canonical vable-field index for `debugdata`.
///
/// A static field's flat shadow index IS its position in
/// [`PYFRAME_VABLE_FIELDS`] (`initialize_virtualizable` lays the statics out
/// first, then the arrays, then the identity slot), so this doubles as the
/// `virtualizable_entry_at` index a reader of `getorcreatedebug()` uses.
pub const DEBUGDATA_VABLE_FIELD_INDEX: usize = 3;

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

    assert!(
        PYFRAME_VABLE_FIELDS[DEBUGDATA_VABLE_FIELD_INDEX].1 == DEBUGDATA_VABLE_FIELD_INDEX,
        "debugdata must be registered at the expected vable field index"
    );
    let name = PYFRAME_VABLE_FIELDS[DEBUGDATA_VABLE_FIELD_INDEX]
        .0
        .as_bytes();
    let expected = b"debugdata";
    assert!(
        name.len() == expected.len(),
        "PYFRAME_VABLE_FIELDS[3] name mismatch"
    );
    let mut i = 0;
    while i < expected.len() {
        assert!(
            name[i] == expected[i],
            "PYFRAME_VABLE_FIELDS[3] name mismatch"
        );
        i += 1;
    }
};
