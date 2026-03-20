//! Green field support for virtualizables.
//!
//! Mirrors RPython's `greenfield.py` GreenFieldInfo: tracks which fields
//! of a virtualizable are "green" (loop-invariant) and can be promoted
//! to constants during tracing.

/// Information about green (loop-invariant) fields of a virtualizable.
///
/// greenfield.py: GreenFieldInfo
pub struct GreenFieldInfo {
    /// Descriptor indices of fields that are green (constant within a loop).
    pub green_field_descrs: Vec<u32>,
}

impl GreenFieldInfo {
    pub fn new() -> Self {
        GreenFieldInfo {
            green_field_descrs: Vec::new(),
        }
    }

    /// Register a field as green by its descriptor index.
    pub fn add_green_field(&mut self, descr_index: u32) {
        if !self.green_field_descrs.contains(&descr_index) {
            self.green_field_descrs.push(descr_index);
        }
    }

    /// Check if a field is green.
    pub fn is_green_field(&self, descr_index: u32) -> bool {
        self.green_field_descrs.contains(&descr_index)
    }

    /// Number of green fields.
    pub fn num_green_fields(&self) -> usize {
        self.green_field_descrs.len()
    }
}
