//! Byte layout of `rustpython_compiler_core::bytecode::CodeObject`.
//!
//! Charon's `field_offsets` for this struct stay unresolved (`chosen`
//! null), so `layout_for_target` records nothing and the heuristic sizes
//! every `Box<_>` as one word. The real layout is `repr(Rust)`.
//! `offset_of` measures it when this crate is compiled for the target
//! (`target_word_size` equals `size_of::<usize>()`). A wasm32 prepass
//! still runs on the host, where that equality fails; the table below is
//! a wasm32-wasip1 measurement of the same compiler-core revision and is
//! used only then.

use std::collections::HashMap;

use crate::front::semantic::ExactLayout;

pub fn exact_layout_for(canonical_name: &str) -> Option<ExactLayout> {
    if canonical_name != "bytecode::CodeObject" {
        return None;
    }
    let word = crate::layout::target_word_size();
    let host_word = std::mem::size_of::<usize>();
    if word == host_word {
        Some(host_layout())
    } else if word == 4 && host_word == 8 {
        wasm32_layout()
    } else {
        None
    }
}

fn host_layout() -> ExactLayout {
    use rustpython_compiler_core::bytecode::CodeObject;
    use std::mem::{align_of, offset_of, size_of};
    layout_from(
        size_of::<CodeObject>() as u64,
        align_of::<CodeObject>() as u64,
        &[
            ("instructions", offset_of!(CodeObject, instructions) as u64),
            ("locations", offset_of!(CodeObject, locations) as u64),
            ("flags", offset_of!(CodeObject, flags) as u64),
            (
                "posonlyarg_count",
                offset_of!(CodeObject, posonlyarg_count) as u64,
            ),
            ("arg_count", offset_of!(CodeObject, arg_count) as u64),
            (
                "kwonlyarg_count",
                offset_of!(CodeObject, kwonlyarg_count) as u64,
            ),
            ("source_path", offset_of!(CodeObject, source_path) as u64),
            (
                "first_line_number",
                offset_of!(CodeObject, first_line_number) as u64,
            ),
            (
                "max_stackdepth",
                offset_of!(CodeObject, max_stackdepth) as u64,
            ),
            ("obj_name", offset_of!(CodeObject, obj_name) as u64),
            ("qualname", offset_of!(CodeObject, qualname) as u64),
            ("constants", offset_of!(CodeObject, constants) as u64),
            ("names", offset_of!(CodeObject, names) as u64),
            ("varnames", offset_of!(CodeObject, varnames) as u64),
            ("cellvars", offset_of!(CodeObject, cellvars) as u64),
            ("freevars", offset_of!(CodeObject, freevars) as u64),
            (
                "localspluskinds",
                offset_of!(CodeObject, localspluskinds) as u64,
            ),
            ("linetable", offset_of!(CodeObject, linetable) as u64),
            (
                "exceptiontable",
                offset_of!(CodeObject, exceptiontable) as u64,
            ),
        ],
    )
}

/// wasm32-wasip1 `offset_of` of `CodeObject` at compiler-core
/// `86c7fa20c4dd`. Used only when this process's pointer is wider than
/// the target. `first_line_number` follows the `u32` counts here and
/// precedes `flags` on a 64-bit host, so the table is not a scaled copy.
fn wasm32_layout() -> Option<ExactLayout> {
    const FIELDS: &[(&str, u64)] = &[
        ("instructions", 36),
        ("locations", 60),
        ("flags", 132),
        ("posonlyarg_count", 136),
        ("arg_count", 140),
        ("kwonlyarg_count", 144),
        ("source_path", 0),
        ("first_line_number", 148),
        ("max_stackdepth", 152),
        ("obj_name", 12),
        ("qualname", 24),
        ("constants", 68),
        ("names", 76),
        ("varnames", 84),
        ("cellvars", 92),
        ("freevars", 100),
        ("localspluskinds", 108),
        ("linetable", 116),
        ("exceptiontable", 124),
    ];
    const SIZE: u64 = 156;
    const ALIGN: u64 = 4;
    Some(layout_from(SIZE, ALIGN, FIELDS))
}

fn layout_from(size: u64, align: u64, fields: &[(&str, u64)]) -> ExactLayout {
    let mut field_offsets = HashMap::with_capacity(fields.len());
    for (name, offset) in fields {
        field_offsets.insert((*name).to_string(), *offset);
    }
    ExactLayout {
        size: Some(size),
        align: Some(align),
        field_offsets,
        host: None,
    }
}

#[cfg(test)]
mod tests {
    use super::exact_layout_for;
    use rustpython_compiler_core::bytecode::CodeObject;
    use std::mem::{offset_of, size_of};

    #[test]
    fn host_codeobject_offsets_match_offset_of() {
        let Some(layout) = exact_layout_for("bytecode::CodeObject") else {
            panic!("native word must record CodeObject");
        };
        assert_eq!(layout.size, Some(size_of::<CodeObject>() as u64));
        let varnames = layout.field_offsets["varnames"];
        assert_eq!(varnames, offset_of!(CodeObject, varnames) as u64);
        assert_eq!(
            layout.field_offsets["localspluskinds"],
            offset_of!(CodeObject, localspluskinds) as u64
        );
        assert_eq!(
            layout.field_offsets["freevars"],
            offset_of!(CodeObject, freevars) as u64
        );
        let size = layout.size.unwrap();
        let mut seen = std::collections::HashSet::new();
        for (name, offset) in &layout.field_offsets {
            assert!(*offset < size, "{name} {offset} >= {size}");
            assert!(seen.insert(*offset), "duplicate offset {offset} ({name})");
        }
        // The one-word heuristic places `varnames` at 0x60, which is a
        // length word inside `CodeUnits` (instruction count), not this field.
        if size_of::<usize>() == 8 {
            assert_ne!(varnames, 0x60);
        }
    }

    #[test]
    fn wasm32_codeobject_table_is_the_measured_layout() {
        let layout = super::wasm32_layout().expect("wasm32 table");
        assert_eq!(layout.size, Some(156));
        assert_eq!(layout.align, Some(4));
        assert_eq!(layout.field_offsets.len(), 19);
        assert_eq!(layout.field_offsets["varnames"], 84);
        assert_eq!(layout.field_offsets["localspluskinds"], 108);
        assert_eq!(layout.field_offsets["freevars"], 100);
        assert_eq!(layout.field_offsets["source_path"], 0);
        assert_eq!(layout.field_offsets["first_line_number"], 148);
        assert_eq!(layout.field_offsets["flags"], 132);
        let mut seen = std::collections::HashSet::new();
        for (name, offset) in &layout.field_offsets {
            assert!(*offset < 156, "{name} {offset}");
            assert!(seen.insert(*offset), "duplicate offset {offset} ({name})");
        }
        if size_of::<usize>() == 8 {
            let host_line = offset_of!(CodeObject, first_line_number) as u64;
            assert_ne!(
                layout.field_offsets["first_line_number"],
                host_line / 2,
                "wasm field order is not the host order scaled by word size"
            );
        }
    }

    #[test]
    fn other_structs_are_not_this_probe() {
        assert!(exact_layout_for("CodeObject").is_none());
        assert!(exact_layout_for("bytecode::CodeFlags").is_none());
    }
}
