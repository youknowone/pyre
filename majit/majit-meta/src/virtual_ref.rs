//! Virtual references: lazy materialization of virtualized objects.
//!
//! When JIT-compiled code virtualizes an object (e.g., an interpreter frame),
//! external code may still hold a reference to it. A virtual reference is a
//! lightweight wrapper that defers materialization until someone actually
//! accesses (forces) it.
//!
//! The `JitVirtualRef` struct has two fields:
//! - `virtual_token`: while the JIT is running, this is the force_token
//!   (i.e., the JIT frame address). Set to TOKEN_NONE when forced or
//!   when the frame is no longer active.
//! - `forced`: once forced, this points to the materialized object.
//!
//! Mirrors `rpython/jit/metainterp/virtualref.py`.

/// Token value indicating no JIT frame is active.
pub const TOKEN_NONE: i64 = 0;

/// Token value used during tracing when a residual call is in progress.
pub const TOKEN_TRACING_RESCALL: i64 = -1;

/// Well-known field descriptor indices for JitVirtualRef fields.
/// These are used by the optimizer to track virtual_token and forced
/// fields when a VirtualRef becomes a virtual struct.
pub const VREF_FIELD_VIRTUAL_TOKEN: u32 = 0x7F00;
pub const VREF_FIELD_FORCED: u32 = 0x7F01;

/// Descriptor indices for the virtual ref struct fields.
///
/// The optimizer uses these to create and manipulate virtual JitVirtualRef
/// structures without emitting actual allocations.
pub mod descr {
    /// Field descriptor index for `virtual_token`.
    pub const VIRTUAL_TOKEN: u32 = 0x7F00;
    /// Field descriptor index for `forced`.
    pub const FORCED: u32 = 0x7F01;
    /// Size descriptor index for the JitVirtualRef struct itself.
    pub const VREF_SIZE: u32 = 0x7F10;
}

/// Virtual reference state for a single reference.
///
/// A virtual reference wraps a virtualizable object during JIT execution.
/// When code outside the JIT tries to access the virtual object, the
/// reference is "forced" -- the virtual object is materialized on the heap.
#[derive(Debug, Clone)]
pub struct VirtualRefInfo {
    /// Field descriptor index for the `virtual_token` field.
    pub descr_virtual_token: u32,
    /// Field descriptor index for the `forced` field.
    pub descr_forced: u32,
    /// Size descriptor index for the JitVirtualRef struct.
    pub descr_size: u32,
}

impl Default for VirtualRefInfo {
    fn default() -> Self {
        Self::new()
    }
}

impl VirtualRefInfo {
    /// Create a VirtualRefInfo with the standard descriptor indices.
    pub fn new() -> Self {
        VirtualRefInfo {
            descr_virtual_token: descr::VIRTUAL_TOKEN,
            descr_forced: descr::FORCED,
            descr_size: descr::VREF_SIZE,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_constants() {
        assert_eq!(TOKEN_NONE, 0);
        assert_eq!(TOKEN_TRACING_RESCALL, -1);
    }

    #[test]
    fn test_vref_info_default() {
        let info = VirtualRefInfo::new();
        assert_eq!(info.descr_virtual_token, descr::VIRTUAL_TOKEN);
        assert_eq!(info.descr_forced, descr::FORCED);
        assert_eq!(info.descr_size, descr::VREF_SIZE);
    }
}
