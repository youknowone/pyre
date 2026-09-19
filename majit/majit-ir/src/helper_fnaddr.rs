//! Process-wide registry of macro-emitted residual-call trampolines.
//!
//! `#[dont_look_inside]` / `#[elidable]` (and the rest of that family) emit
//! an `extern "C"` trampoline next to the annotated function. This slice
//! publishes `(path, address, arity)` for each trampoline whose ABI matches
//! what the codewriter records, so a residual callee has a real address
//! without a hand-maintained list.
//!
//! The address is captured in the defining crate (`fn as *const ()` in that
//! crate's static or constructor). Taking `fn as usize` in another crate can
//! produce a second wasm32 table slot; reading the stored pointer bits does
//! not.
//!
//! `distributed_slice` is not implemented for wasm32; that target fills
//! [`WASM_HELPER_FNADDRS`] from a constructor instead.

/// One residual-call trampoline the macros published.
#[derive(Clone, Copy)]
pub struct HelperFnAddr {
    /// `concat!(module_path!(), "::", stringify!(name))` of the annotated
    /// function — the `FunctionPath` a residual call names.
    pub path: &'static str,
    /// Trampoline pointer captured in the defining crate.
    addr: *const (),
    /// Argument count in source order, matching the trampoline.
    pub arity: u8,
}

// Safety: `path` is `'static` and `addr` is a process-global function
// pointer; sharing across threads is sound.
unsafe impl Sync for HelperFnAddr {}
unsafe impl Send for HelperFnAddr {}

impl HelperFnAddr {
    /// Build a registry row. Called from a `const` static initializer in the
    /// defining crate, where `fn as *const ()` is a valid initializer.
    pub const fn new(path: &'static str, addr: *const (), arity: u8) -> Self {
        Self { path, addr, arity }
    }

    /// Address captured in the defining crate, as a `usize`.
    pub fn get(&self) -> usize {
        self.addr as usize
    }
}

/// Link-time registry of every macro-published residual trampoline.
///
/// `distributed_slice` rejects wasm32, which carries the same set in
/// [`WASM_HELPER_FNADDRS`]; read both through [`for_each_helper_fnaddr`].
#[cfg(not(target_arch = "wasm32"))]
#[::linkme::distributed_slice]
pub static HELPER_FNADDRS: [HelperFnAddr] = [..];

/// The same registry on wasm32, populated at constructor time.
#[cfg(target_arch = "wasm32")]
pub static WASM_HELPER_FNADDRS: std::sync::Mutex<Vec<HelperFnAddr>> =
    std::sync::Mutex::new(Vec::new());

/// Append one trampoline to [`WASM_HELPER_FNADDRS`].
///
/// Called only from the constructor the macros emit. An entry appended after
/// the address table has been read is not published.
#[cfg(target_arch = "wasm32")]
pub fn register(path: &'static str, addr: *const (), arity: u8) {
    WASM_HELPER_FNADDRS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .push(HelperFnAddr::new(path, addr, arity));
}

/// Visit every registered trampoline, whichever population the target carries.
pub fn for_each_helper_fnaddr(mut visit: impl FnMut(&HelperFnAddr)) {
    #[cfg(not(target_arch = "wasm32"))]
    for desc in HELPER_FNADDRS {
        visit(&desc);
    }
    #[cfg(target_arch = "wasm32")]
    {
        let guard = WASM_HELPER_FNADDRS
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for desc in guard.iter() {
            visit(desc);
        }
    }
}
