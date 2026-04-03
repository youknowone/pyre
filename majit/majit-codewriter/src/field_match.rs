//! Canonical field descriptor matching for the graph/pipeline translator.
//!
//! This centralizes the active field vocabulary so classification and rewrite
//! passes do not each carry their own ad-hoc field-name checks.

use crate::graph::FieldDescriptor;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum KnownFieldKind {
    LocalArray,
    InstructionPosition,
    ConstantPool,
}

fn owner_root_matches(owner_root: Option<&str>, allowed: &[&str]) -> bool {
    owner_root.is_some_and(|root| {
        allowed.iter().any(|candidate| root == *candidate)
            // "self" / "Self" in method bodies match any allowed type
            || root == "self"
            || root == "Self"
            // Single-letter generic type parameter (e.g. "H", "E")
            || crate::call_match::is_generic_receiver(root)
    })
}

pub fn describe_known_field(field: &FieldDescriptor) -> Option<KnownFieldKind> {
    match field.name.as_str() {
        "locals_w" | "locals_cells_stack_w"
            if owner_root_matches(field.owner_root.as_deref(), &["Frame", "PyFrame"]) =>
        {
            Some(KnownFieldKind::LocalArray)
        }
        "next_instr" | "pc" | "last_instr"
            if owner_root_matches(field.owner_root.as_deref(), &["Frame", "PyFrame"]) =>
        {
            Some(KnownFieldKind::InstructionPosition)
        }
        "constants" | "co_consts"
            if owner_root_matches(
                field.owner_root.as_deref(),
                &["Frame", "PyFrame", "Code", "PyCode"],
            ) =>
        {
            Some(KnownFieldKind::ConstantPool)
        }
        _ => None,
    }
}

pub fn is_local_array_field(field: &FieldDescriptor) -> bool {
    describe_known_field(field) == Some(KnownFieldKind::LocalArray)
}

pub fn is_instruction_position_field(field: &FieldDescriptor) -> bool {
    describe_known_field(field) == Some(KnownFieldKind::InstructionPosition)
}

pub fn is_constant_pool_field(field: &FieldDescriptor) -> bool {
    describe_known_field(field) == Some(KnownFieldKind::ConstantPool)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_array_requires_frame_owner() {
        assert!(is_local_array_field(&FieldDescriptor::new(
            "locals_w",
            Some("Frame".into())
        )));
        assert!(!is_local_array_field(&FieldDescriptor::new(
            "locals_w",
            Some("Other".into())
        )));
    }

    #[test]
    fn instruction_position_requires_frame_owner() {
        assert!(is_instruction_position_field(&FieldDescriptor::new(
            "next_instr",
            Some("PyFrame".into())
        )));
        assert!(!is_instruction_position_field(&FieldDescriptor::new(
            "next_instr",
            Some("Instruction".into())
        )));
    }

    #[test]
    fn constant_pool_requires_frame_or_code_owner() {
        assert!(is_constant_pool_field(&FieldDescriptor::new(
            "co_consts",
            Some("PyCode".into())
        )));
        assert!(!is_constant_pool_field(&FieldDescriptor::new(
            "co_consts",
            Some("PyFrameObject".into())
        )));
    }
}
