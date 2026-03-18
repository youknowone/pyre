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

    /// Force a virtual reference: materialize the virtual object.
    ///
    /// RPython virtualref.py: `force_virtual()`
    ///
    /// Called when non-JIT code accesses a virtual reference. The JIT
    /// frame is forced (flushing register values to heap), and the
    /// virtual_token is set to TOKEN_NONE.
    ///
    /// # Safety
    /// `vref_ptr` must point to a valid JitVirtualRef object.
    pub unsafe fn force_virtual(
        &self,
        vref_ptr: *mut u8,
        force_fn: impl FnOnce(i64) -> *mut u8,
    ) -> *mut u8 {
        let token_ptr = vref_ptr.add(self.descr_virtual_token as usize * 8) as *mut i64;
        let forced_ptr = vref_ptr.add(self.descr_forced as usize * 8) as *mut *mut u8;
        let token = *token_ptr;

        if token == TOKEN_NONE {
            // Already forced or not in JIT
            return *forced_ptr;
        }

        if token == TOKEN_TRACING_RESCALL {
            // In tracing — just clear
            *token_ptr = TOKEN_NONE;
            return *forced_ptr;
        }

        // Active JIT frame — call force_fn to materialize
        let materialized = force_fn(token);
        *forced_ptr = materialized;
        *token_ptr = TOKEN_NONE;
        materialized
    }

    /// Create a virtual reference during tracing.
    ///
    /// virtualref.py: `virtual_ref_during_tracing()`
    ///
    /// Returns (virtual_token_opref, forced_opref) that the tracer should
    /// record in the virtual ref's fields.
    pub fn virtual_ref_during_tracing(
        &self,
        force_token: i64,
    ) -> (i64, i64) {
        (force_token, 0) // (active token, null forced)
    }

    /// Mark a virtual ref as "in residual call" before a non-JIT call.
    ///
    /// virtualref.py: `tracing_before_residual_call(vref)`
    /// Sets token to TOKEN_TRACING_RESCALL so that if the callee
    /// forces the vref, we detect it after the call.
    ///
    /// # Safety
    /// `vref_ptr` must point to a valid JitVirtualRef object.
    pub unsafe fn tracing_before_residual_call(&self, vref_ptr: *mut u8) {
        let token_ptr = vref_ptr.add(self.descr_virtual_token as usize * 8) as *mut i64;
        *token_ptr = TOKEN_TRACING_RESCALL;
    }

    /// Check and restore a virtual ref after a residual call.
    ///
    /// virtualref.py: `tracing_after_residual_call(vref)`
    /// Returns true if the vref was forced during the residual call
    /// (token was cleared by the callee).
    ///
    /// # Safety
    /// `vref_ptr` must point to a valid JitVirtualRef object.
    pub unsafe fn tracing_after_residual_call(
        &self,
        vref_ptr: *mut u8,
        original_token: i64,
    ) -> bool {
        let token_ptr = vref_ptr.add(self.descr_virtual_token as usize * 8) as *mut i64;
        let current_token = *token_ptr;

        if current_token != TOKEN_TRACING_RESCALL {
            // Token was modified during the call — vref was forced
            return true;
        }

        // Not forced: restore original token
        *token_ptr = original_token;
        false
    }

    /// Continue tracing after a residual call that forced a vref.
    ///
    /// virtualref.py: `continue_tracing(vref, real_object)`
    /// Updates the forced field and clears the token.
    ///
    /// # Safety
    /// `vref_ptr` must point to a valid JitVirtualRef object.
    pub unsafe fn continue_tracing(
        &self,
        vref_ptr: *mut u8,
        real_object: *mut u8,
    ) {
        let token_ptr = vref_ptr.add(self.descr_virtual_token as usize * 8) as *mut i64;
        let forced_ptr = vref_ptr.add(self.descr_forced as usize * 8) as *mut *mut u8;
        *forced_ptr = real_object;
        *token_ptr = TOKEN_NONE;
    }

    /// Check if a virtual reference is currently active (has a JIT frame token).
    ///
    /// virtualref.py: token != TOKEN_NONE and token != TOKEN_TRACING_RESCALL
    pub fn is_active(token: i64) -> bool {
        token != TOKEN_NONE && token != TOKEN_TRACING_RESCALL
    }

    /// Check if a virtual reference is forced (token == TOKEN_NONE).
    pub fn is_forced(token: i64) -> bool {
        token == TOKEN_NONE
    }

    /// Check if a virtual reference is in a residual call.
    pub fn is_in_residual_call(token: i64) -> bool {
        token == TOKEN_TRACING_RESCALL
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
