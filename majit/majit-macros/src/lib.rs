/// Proc macros for the majit JIT framework.
///
/// Provides:
/// - #[jit_driver]: Annotate an interpreter's main dispatch loop
/// - jit_merge_point!(): Mark the loop header for tracing
/// - #[elidable]: Mark a function as pure (constant-foldable)
/// - #[dont_look_inside]: Prevent tracing into a function
///
/// This is a stub — full implementation in Stream F.

use proc_macro::TokenStream;

/// Mark a struct as a JIT driver configuration.
///
/// Usage:
/// ```ignore
/// #[majit::jit_driver(
///     greens = [next_instr, pycode],
///     reds = [frame, ec],
/// )]
/// struct MyJitDriver;
/// ```
#[proc_macro_attribute]
pub fn jit_driver(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Stub: pass through unchanged
    item
}

/// Mark a function as elidable (pure / constant-foldable).
///
/// The JIT can eliminate calls to this function when all arguments are constants.
#[proc_macro_attribute]
pub fn elidable(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}

/// Mark a function as opaque to the tracer.
///
/// The JIT will not trace into this function; it will be called as a black box.
#[proc_macro_attribute]
pub fn dont_look_inside(_attr: TokenStream, item: TokenStream) -> TokenStream {
    item
}
