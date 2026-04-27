//! Port of `rpython/translator/c/support.py`.
//!
//! Hosts the smallest helpers the C-backend needs: `cdecl` (declaration
//! template substitution) and `CNameManager` (the `pypy_`-prefixed
//! reserved-keyword-aware [`super::super::gensupp::NameManager`] used
//! by [`super::database::LowLevelDatabase::namespace`]).
//!
//! Pending: `forward_cdecl`, `barebonearray`, `c_string_constant`,
//! `c_char_array_constant`, `gen_assignments`, the `log` AnsiLogger
//! handle. They land alongside the consumers (`node.py` rendering,
//! `funcgen.py`).
//!
//! The `USESLOTS` flag at upstream `:8` is a Python `__slots__`
//! optimisation — Rust structs already have a fixed shape, so the flag
//! is not ported.

use crate::translator::gensupp::NameManager;

/// RPython `support.cdecl(ctype, cname, is_thread_local=False)`
/// (`support.py:20-31`).
///
/// Replaces the `@` placeholder in `ctype` with `cname`, with the
/// special case `(@)` → `@` so that function declarations don't end
/// up with redundant parentheses around the function name.
pub fn cdecl(ctype: &str, cname: &str, is_thread_local: bool) -> String {
    let prefix = if is_thread_local { "__thread " } else { "" };
    let cleaned = ctype.replace("(@)", "@");
    let with_name = cleaned.replace('@', cname);
    format!("{prefix}{}", with_name.trim())
}

/// RPython `support.CNameManager` (`support.py:74-89`).
///
/// A [`NameManager`] preloaded with the C99 keywords so generated
/// names cannot collide with reserved C identifiers, and seeded with
/// the `pypy_` global prefix used everywhere across the C backend.
#[derive(Debug, Clone)]
pub struct CNameManager {
    inner: NameManager,
}

impl CNameManager {
    /// RPython `CNameManager.__init__(global_prefix='pypy_')`
    /// (`support.py:75-89`).
    pub fn new() -> Self {
        Self::with_prefix("pypy_")
    }

    /// Constructor variant matching upstream's optional kwarg.
    pub fn with_prefix(global_prefix: impl Into<String>) -> Self {
        let mut inner = NameManager::new(global_prefix, "_");
        // Upstream `:78-89` — the C99 draft keyword list, copied
        // verbatim. The whitespace layout is preserved so a textual
        // diff against upstream highlights any divergence.
        inner.make_reserved_names(
            "auto      enum      restrict  unsigned \
             break     extern    return    void \
             case      float     short     volatile \
             char      for       signed    while \
             const     goto      sizeof    _Bool \
             continue  if        static    _Complex \
             default   inline    struct    _Imaginary \
             do        int       switch \
             double    long      typedef \
             else      register  union",
        );
        Self { inner }
    }

    /// Mirror of [`NameManager::uniquename`] for the wrapped manager.
    pub fn uniquename(
        &mut self,
        basename: &str,
        with_number: Option<bool>,
        lenmax: usize,
    ) -> String {
        self.inner.uniquename(basename, with_number, lenmax)
    }

    /// Mirror of [`NameManager::uniquename_bare`].
    pub fn uniquename_bare(
        &mut self,
        basename: &str,
        with_number: Option<bool>,
        lenmax: usize,
    ) -> (String, String) {
        self.inner.uniquename_bare(basename, with_number, lenmax)
    }

    /// Borrow the wrapped manager — exposed so callers that need
    /// `make_reserved_names` or direct field access can reach in
    /// without forcing every method to be re-exposed here.
    pub fn inner(&self) -> &NameManager {
        &self.inner
    }

    /// Mutable counterpart of [`Self::inner`].
    pub fn inner_mut(&mut self) -> &mut NameManager {
        &mut self.inner
    }
}

impl Default for CNameManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cdecl_replaces_at_with_name() {
        assert_eq!(cdecl("Signed @", "x", false), "Signed x");
    }

    #[test]
    fn cdecl_collapses_paren_at_for_function_decls() {
        // `Signed (@)(int a)` is the upstream "function-pointer
        // template": the parens around `@` exist only for the
        // pointer-typedef shape and must collapse for the plain
        // function-name case.
        assert_eq!(cdecl("Signed (@)(int a)", "fn", false), "Signed fn(int a)");
    }

    #[test]
    fn cdecl_keeps_explicit_pointer_parens() {
        // For the function-pointer case the template is `Signed
        // (*@)(int a)` — `(*@)` must NOT collapse, because the
        // upstream collapse rule keys on the literal `(@)`. The
        // resulting decl keeps the asterisk and the wrapping parens.
        assert_eq!(
            cdecl("Signed (*@)(int a)", "fn", false),
            "Signed (*fn)(int a)"
        );
    }

    #[test]
    fn cdecl_prepends_thread_local_qualifier() {
        assert_eq!(cdecl("Signed @", "x", true), "__thread Signed x");
    }

    #[test]
    fn cname_manager_reserves_c_keywords() {
        let mut nm = CNameManager::new();
        // Reserved keyword `auto` collides with the preloaded set —
        // first claim must yield a unique name, not the keyword.
        let name = nm.uniquename("auto", Some(false), 50);
        assert_ne!(name, "pypy_auto", "should not return reserved keyword");
    }

    #[test]
    fn cname_manager_uses_pypy_prefix_by_default() {
        let mut nm = CNameManager::new();
        assert_eq!(nm.uniquename("foo", Some(false), 50), "pypy_foo");
    }

    #[test]
    fn cname_manager_with_prefix_overrides_default() {
        let mut nm = CNameManager::with_prefix("custom_");
        assert_eq!(nm.uniquename("foo", Some(false), 50), "custom_foo");
    }
}
