//! Helper function classification.
//!
//! Determines how each function should be treated during tracing:
//! - Elidable: pure function, can be constant-folded
//! - Residual: opaque call, emit CALL_I/CALL_R
//! - FieldAccess: direct struct field read/write → GetfieldRawI/SetfieldRaw
//! - Constructor: allocates a new object → New + SetfieldRaw
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
    FieldRead {
        struct_name: String,
        field_name: String,
    },
    /// Direct field write — maps to SetfieldRaw.
    FieldWrite {
        struct_name: String,
        field_name: String,
    },
    /// Object constructor — maps to New + SetfieldRaw sequence.
    Constructor { struct_name: String },
    /// Type check — maps to GetfieldRawI(ob_type) + GuardClass.
    TypeCheck { type_name: String },
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
                classify_by_name_and_body(&name, func)
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

fn has_attribute(attrs: &[syn::Attribute], name: &str) -> bool {
    attrs.iter().any(|attr| {
        attr.path()
            .segments
            .last()
            .map_or(false, |seg| seg.ident == name)
    })
}

/// Classify based on function name patterns and body analysis.
fn classify_by_name_and_body(name: &str, func: &syn::ItemFn) -> HelperClassification {
    let body = {
        let block = &func.block;
        quote::quote!(#block).to_string()
    };

    // ── Name-based patterns ──

    // w_*_get_value / w_*_get_* → field read (elidable)
    if name.starts_with("w_") && name.contains("get_value") {
        let type_name = extract_type_from_helper_name(name);
        return HelperClassification::FieldRead {
            struct_name: type_name,
            field_name: "value".into(),
        };
    }

    // w_*_len → field read
    if name.starts_with("w_") && name.ends_with("_len") {
        let type_name = extract_type_from_helper_name(name);
        return HelperClassification::FieldRead {
            struct_name: type_name,
            field_name: "len".into(),
        };
    }

    // w_*_new → constructor
    if name.starts_with("w_") && name.ends_with("_new") {
        let type_name = extract_type_from_helper_name(name);
        return HelperClassification::Constructor {
            struct_name: type_name,
        };
    }

    // is_* → type check
    if name.starts_with("is_") && !name.contains("compatible") {
        let type_name = name.strip_prefix("is_").unwrap_or(name).to_string();
        return HelperClassification::TypeCheck { type_name };
    }

    // jit_* → JIT helper (residual or elidable)
    if name.starts_with("jit_") {
        return HelperClassification::Residual;
    }

    // ── Body-based patterns ──

    // unsafe field access: *(obj as *const T).field
    if body.contains("as *const") && body.contains("unsafe") {
        if body.contains(".*") || body.contains(").") {
            let struct_name = extract_struct_from_cast(&body);
            let field_name = extract_field_from_access(&body);
            return HelperClassification::FieldRead {
                struct_name,
                field_name,
            };
        }
    }

    // Box::new or alloc pattern
    if body.contains("Box :: new") || body.contains("Box::new") {
        return HelperClassification::Residual;
    }

    HelperClassification::Unknown
}

/// Extract type name from helper function name.
/// e.g., "w_int_get_value" → "W_IntObject", "w_float_new" → "W_FloatObject"
fn extract_type_from_helper_name(name: &str) -> String {
    let parts: Vec<&str> = name.split('_').collect();
    if parts.len() >= 2 {
        // w_int_get_value → int → W_IntObject
        let type_part = parts[1];
        format!("W_{}Object", capitalize(type_part))
    } else {
        "Unknown".into()
    }
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

/// Extract struct name from a cast expression in function body.
fn extract_struct_from_cast(body: &str) -> String {
    // Look for "as *const W_Something" or "as * const W_Something"
    if let Some(pos) = body.find("as *const ") {
        let after = &body[pos + 10..];
        let end = after
            .find(|c: char| !c.is_alphanumeric() && c != '_')
            .unwrap_or(after.len());
        return after[..end].trim().to_string();
    }
    if let Some(pos) = body.find("as * const") {
        let after = &body[pos + 10..];
        let trimmed = after.trim_start();
        let end = trimmed
            .find(|c: char| !c.is_alphanumeric() && c != '_')
            .unwrap_or(trimmed.len());
        return trimmed[..end].to_string();
    }
    "Unknown".into()
}

/// Extract field name from a field access in function body.
fn extract_field_from_access(body: &str) -> String {
    // Look for ").field_name" pattern
    if let Some(pos) = body.rfind(").") {
        let after = &body[pos + 2..];
        let end = after
            .find(|c: char| !c.is_alphanumeric() && c != '_')
            .unwrap_or(after.len());
        if end > 0 {
            return after[..end].to_string();
        }
    }
    "unknown".into()
}
