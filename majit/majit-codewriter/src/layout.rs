//! Layout provider — RPython `symbolic.get_field_token` / `symbolic.get_size`.
//!
//! RPython uses `llmemory.offsetof()` and `llmemory.sizeof()` (backed by C-level
//! layout via ll2ctypes) to compute exact field offsets and struct sizes. The JIT
//! codewriter calls `cpu.fielddescrof(STRUCT, fieldname)` which delegates to
//! `symbolic.get_field_token(STRUCT, fieldname, translate_support_code)`.
//!
//! In majit, the `LayoutProvider` trait serves the same purpose: it supplies
//! real struct layouts from the host runtime. The default `HeuristicLayoutProvider`
//! approximates `#[repr(C)]` layout from parsed type strings; production runtimes
//! (e.g. pyre) should override with layouts from `std::mem::offset_of!()` /
//! `std::mem::size_of::<T>()`.

use std::collections::{HashMap, HashSet};

use crate::call::StructLayout;

/// RPython: `symbolic.get_field_token` + `symbolic.get_size` provider.
///
/// Supplies struct layouts for the codewriter pipeline. Each struct is
/// identified by name (RPython uses LLTYPE identity).
pub trait LayoutProvider {
    /// Return the layout for a struct, or None to fall back to heuristic.
    fn get_struct_layout(&self, struct_name: &str) -> Option<StructLayout>;
}

/// Default provider using type-string heuristic.
///
/// Approximates `#[repr(C)]` layout by parsing field type strings and
/// applying alignment rules. Equivalent to the pre-LayoutProvider code path.
///
/// Production runtimes should provide a concrete `LayoutProvider` with
/// real offsets from `std::mem::offset_of!()` instead.
pub struct HeuristicLayoutProvider {
    known_struct_names: HashSet<String>,
    known_struct_sizes: HashMap<String, usize>,
    fields_by_struct: HashMap<String, Vec<(String, String)>>,
}

impl HeuristicLayoutProvider {
    /// Build the heuristic provider from struct field definitions.
    ///
    /// Runs fixed-point iteration to resolve nested struct sizes, matching
    /// the convergence loop that `symbolic.get_size()` doesn't need (because
    /// RPython queries the C compiler directly).
    pub fn from_struct_fields(
        fields_by_struct: &HashMap<String, Vec<(String, String)>>,
        known_struct_names: &HashSet<String>,
    ) -> Self {
        let mut known_sizes: HashMap<String, usize> = HashMap::new();
        loop {
            let mut changed = false;
            for (struct_name, fields) in fields_by_struct {
                let layout = StructLayout::from_type_strings(fields, &HashSet::new(), &known_sizes);
                if known_sizes.get(struct_name) != Some(&layout.size) {
                    known_sizes.insert(struct_name.clone(), layout.size);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        Self {
            known_struct_names: known_struct_names.clone(),
            known_struct_sizes: known_sizes,
            fields_by_struct: fields_by_struct.clone(),
        }
    }
}

impl LayoutProvider for HeuristicLayoutProvider {
    fn get_struct_layout(&self, struct_name: &str) -> Option<StructLayout> {
        let fields = self.fields_by_struct.get(struct_name)?;
        Some(StructLayout::from_type_strings(
            fields,
            &self.known_struct_names,
            &self.known_struct_sizes,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heuristic_provider_basic_struct() {
        let mut fields = HashMap::new();
        fields.insert(
            "MyStruct".to_string(),
            vec![
                ("x".to_string(), "i64".to_string()),
                ("y".to_string(), "i64".to_string()),
            ],
        );
        let provider = HeuristicLayoutProvider::from_struct_fields(&fields, &HashSet::new());
        let layout = provider.get_struct_layout("MyStruct").unwrap();
        assert_eq!(layout.size, 16);
        assert_eq!(layout.fields.len(), 2);
        assert_eq!(layout.fields[0].name, "x");
        assert_eq!(layout.fields[0].offset, 0);
        assert_eq!(layout.fields[1].name, "y");
        assert_eq!(layout.fields[1].offset, 8);
    }

    #[test]
    fn test_heuristic_provider_unknown_struct() {
        let fields = HashMap::new();
        let provider = HeuristicLayoutProvider::from_struct_fields(&fields, &HashSet::new());
        assert!(provider.get_struct_layout("NoSuchStruct").is_none());
    }
}
