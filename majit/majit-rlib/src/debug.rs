//! `rpython/rlib/debug.py` — the translation-time assertion helpers.
//!
//! Each is an identity at runtime; the content is in the
//! `ExtRegistryEntry.compute_result_annotation` that runs while the
//! annotator is deriving the argument's annotation. `majit-translate` owns
//! that half, in `annotator/builtin.rs`, keyed on this module's paths.

/// `check_not_access_directly(arg)` — "check that arg does not have the
/// `access_directly=True` hint set".
///
/// ```python
/// class Entry(ExtRegistryEntry):
///     _about_ = check_not_access_directly
///
///     def compute_result_annotation(self, s_arg):
///         assert not s_arg.flags.get('access_directly', False)
///         return s_arg
/// ```
///
/// Upstream calls it from `baseobjspace.py W_Root.getclass`, whose comment
/// gives the reason: annotating that method with `access_directly` set
/// would specialize it, "otherwise every call to getclass (and other
/// methods) has an extra indirection due to a much more complicated
/// function set". It is the practical enforcement that the flag stays
/// confined to the graphs the virtualizable protocol means it for —
/// `warmspot.py check_access_directly_sanity` is the other, coarser one.
///
/// The body is the identity, and `specialize_call` is
/// `hop.inputarg(hop.args_r[0], arg=0)` — the same identity after rtyping.
#[inline(always)]
pub fn check_not_access_directly<T>(arg: T) -> T {
    arg
}
