//! Helper function classification.
//!
//! Determines how each function should be treated during tracing:
//! - Elidable: pure function, can be constant-folded
//! - Residual: opaque call, emit CALL_I/CALL_R
//! - FieldAccess: direct struct field read/write → GetfieldRawI/SetfieldRaw
//! - DontLookInside: never trace into

use serde::{Deserialize, Serialize};

use crate::parse::ParsedInterpreter;

/// Classification of a helper function for JIT purposes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HelperClassification {
    /// Pure function — can be constant-folded if all args are constant.
    Elidable,
    /// Opaque call — emit residual CALL during tracing.
    Residual,
    /// Direct field read — maps to GetfieldRawI.
    FieldRead { struct_name: String, field_name: String },
    /// Direct field write — maps to SetfieldRaw.
    FieldWrite { struct_name: String, field_name: String },
    /// Never trace into this function.
    DontLookInside,
    /// Unclassified — needs manual annotation.
    Unknown,
}

/// Information about a function found in the source.
#[derive(Debug, Serialize, Deserialize)]
pub struct FunctionInfo {
    pub name: String,
    pub classification: HelperClassification,
    pub has_elidable_attr: bool,
    pub has_dont_look_inside_attr: bool,
}

/// Classify all functions in the parsed source.
pub fn classify_functions(parsed: &ParsedInterpreter) -> Vec<FunctionInfo> {
    let mut functions = Vec::new();

    for item in &parsed.file.items {
        if let syn::Item::Fn(func) = item {
            let name = func.sig.ident.to_string();
            let has_elidable = has_attribute(&func.attrs, "elidable");
            let has_dli = has_attribute(&func.attrs, "dont_look_inside");

            let classification = if has_elidable {
                HelperClassification::Elidable
            } else if has_dli {
                HelperClassification::DontLookInside
            } else {
                classify_function_body(func)
            };

            functions.push(FunctionInfo {
                name,
                classification,
                has_elidable_attr: has_elidable,
                has_dont_look_inside_attr: has_dli,
            });
        }
    }

    functions
}

/// Check if a function has a specific attribute.
fn has_attribute(attrs: &[syn::Attribute], name: &str) -> bool {
    attrs.iter().any(|attr| {
        attr.path()
            .segments
            .last()
            .map_or(false, |seg| seg.ident == name)
    })
}

/// Classify a function based on its body structure.
fn classify_function_body(func: &syn::ItemFn) -> HelperClassification {
    let body = quote::quote!(#func.block).to_string();

    // Simple heuristics for common patterns
    if body.contains("unsafe") && body.contains("as *const") && body.contains(").") {
        // Pattern: unsafe { (*(obj as *const T)).field }
        // This is a field read
        return HelperClassification::FieldRead {
            struct_name: "unknown".into(),
            field_name: "unknown".into(),
        };
    }

    HelperClassification::Unknown
}
