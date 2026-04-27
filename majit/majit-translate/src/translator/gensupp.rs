//! Port of `rpython/translator/gensupp.py`.
//!
//! Only `NameManager` and the `C_IDENTIFIER` translation are ported in
//! this slice; `uniquemodulename` and `_LocalScope` follow once their
//! callers (`genc.py` source generation, `FuncNode` local-name
//! mangling) land.

use std::collections::HashMap;

/// RPython `gensupp.C_IDENTIFIER` (`gensupp.py:18-21`). Maps any byte
/// not in `[0-9A-Za-z]` to `_`.
fn c_identifier_byte(b: u8) -> u8 {
    if b.is_ascii_alphanumeric() { b } else { b'_' }
}

/// Apply the upstream `str.translate(C_IDENTIFIER)` to a UTF-8 input.
/// Non-ASCII bytes (none expected for C-identifier purposes) are
/// passed through; upstream Python operates on bytes directly.
fn translate_c_identifier(s: &str) -> String {
    s.bytes().map(c_identifier_byte).map(char::from).collect()
}

/// RPython `class NameManager(object)` (`gensupp.py:28-72`).
///
/// Tracks every global identifier the generated C source uses so that
/// each new name claim returns a string that doesn't collide with a
/// previously claimed one. The `_LocalScope` helper that upstream
/// hangs off `scopelist` is not yet ported because no caller in the
/// current slice exercises it; keep the field present so reserved
/// names accumulate the same way as upstream.
#[derive(Debug, Clone)]
pub struct NameManager {
    /// `seennames`: every name encountered (reserved or claimed),
    /// mapped to the next dedup counter. Upstream uses a Python dict;
    /// the value is the count of times the name has been seen so the
    /// `_ensure_unique` recursion produces `name`, `name_1`, `name_2`,
    /// etc.
    pub seennames: HashMap<String, usize>,
    /// `scope`: top-level scope id. Upstream initialises to 0 and
    /// `localScope()` increments per nesting level; preserved here so
    /// the field surface matches upstream even though nothing reads it
    /// in this slice.
    pub scope: usize,
    /// `scopelist`: per-scope name lists. Empty until `localScope()`
    /// is ported.
    pub scopelist: Vec<HashMap<String, Vec<String>>>,
    pub global_prefix: String,
    pub number_sep: String,
}

impl NameManager {
    /// RPython `NameManager.__init__(global_prefix='', number_sep='_')`.
    pub fn new(global_prefix: impl Into<String>, number_sep: impl Into<String>) -> Self {
        Self {
            seennames: HashMap::new(),
            scope: 0,
            scopelist: Vec::new(),
            global_prefix: global_prefix.into(),
            number_sep: number_sep.into(),
        }
    }

    /// RPython `NameManager.make_reserved_names(txt)` (`gensupp.py:36-43`).
    ///
    /// Upstream raises `NameError(...)` (with a non-formatted error
    /// message that bubbles up via Python's exception path) when a
    /// name is already in `seennames`. The Rust port panics with the
    /// same message, since this is a programming error: all calls
    /// must run before any `uniquename` (so a duplicate genuinely
    /// signals two reserved-name calls overlapping).
    pub fn make_reserved_names(&mut self, txt: &str) {
        for name in txt.split_ascii_whitespace() {
            if self.seennames.contains_key(name) {
                panic!("{name} has already been seen!");
            }
            self.seennames.insert(name.to_string(), 1);
        }
    }

    /// RPython `NameManager._ensure_unique(basename)` (`gensupp.py:45-50`).
    fn ensure_unique(&mut self, basename: String) -> String {
        let n = *self.seennames.get(&basename).unwrap_or(&0);
        self.seennames.insert(basename.clone(), n + 1);
        if n != 0 {
            self.ensure_unique(format!("{basename}_{n}"))
        } else {
            basename
        }
    }

    /// RPython `NameManager.uniquename(basename, with_number=None,
    /// bare=False, lenmax=50)` (`gensupp.py:52-66`).
    ///
    /// Returns the prefixed unique name (upstream `bare=False`).
    /// Callers needing the bare/prefixed pair use
    /// [`Self::uniquename_bare`].
    pub fn uniquename(
        &mut self,
        basename: &str,
        with_number: Option<bool>,
        lenmax: usize,
    ) -> String {
        self.uniquename_bare(basename, with_number, lenmax).1
    }

    /// RPython `NameManager.uniquename(..., bare=True)`. Returns
    /// `(basename, global_prefix + basename)`.
    pub fn uniquename_bare(
        &mut self,
        basename: &str,
        with_number: Option<bool>,
        lenmax: usize,
    ) -> (String, String) {
        // Upstream `basename = basename[:lenmax].translate(C_IDENTIFIER)`.
        let truncated: String = basename.chars().take(lenmax).collect();
        let mut basename = translate_c_identifier(&truncated);

        let n = *self.seennames.get(&basename).unwrap_or(&0);
        self.seennames.insert(basename.clone(), n + 1);

        // `with_number=None` defaults to True for the canonical
        // generated-variable prefixes.
        let with_number = with_number.unwrap_or_else(|| basename == "v" || basename == "w_");

        // `fmt = '%%s%s%%d' % self.number_sep` — pick the format based
        // on whether the basename's last character is a digit.
        let last_is_digit = basename
            .chars()
            .last()
            .map(|c| c.is_ascii_digit())
            .unwrap_or(false);
        let use_plain_fmt = with_number && !last_is_digit;

        if n != 0 || with_number {
            let candidate = if use_plain_fmt {
                format!("{basename}{n}")
            } else {
                format!("{basename}{}{n}", self.number_sep)
            };
            basename = self.ensure_unique(candidate);
        }

        let prefixed = format!("{}{}", self.global_prefix, basename);
        (basename, prefixed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c_identifier_passes_alnum_and_underscores_others() {
        assert_eq!(translate_c_identifier("abc-9.X"), "abc_9_X");
    }

    #[test]
    fn make_reserved_names_marks_each_token() {
        let mut nm = NameManager::new("", "_");
        nm.make_reserved_names("auto break case");
        assert!(nm.seennames.contains_key("auto"));
        assert!(nm.seennames.contains_key("break"));
        assert!(nm.seennames.contains_key("case"));
    }

    #[test]
    #[should_panic(expected = "already been seen")]
    fn make_reserved_names_panics_on_duplicate() {
        let mut nm = NameManager::new("", "_");
        nm.make_reserved_names("auto");
        nm.make_reserved_names("auto");
    }

    /// First claim of a non-reserved basename returns the prefixed
    /// form unchanged; subsequent claims dedup with `_<n>`.
    #[test]
    fn uniquename_first_claim_returns_prefixed_basename() {
        let mut nm = NameManager::new("pypy_", "_");
        assert_eq!(nm.uniquename("foo", Some(false), 50), "pypy_foo");
        assert_eq!(nm.uniquename("foo", Some(false), 50), "pypy_foo_1");
        assert_eq!(nm.uniquename("foo", Some(false), 50), "pypy_foo_2");
    }

    /// `with_number=None` defaults to True for `v` / `w_`. The
    /// dedup format drops the `number_sep` because the basename
    /// already ends in a non-digit char (`v`).
    #[test]
    fn uniquename_v_basename_appends_digit_without_separator() {
        let mut nm = NameManager::new("", "_");
        assert_eq!(nm.uniquename("v", None, 50), "v0");
        assert_eq!(nm.uniquename("v", None, 50), "v1");
    }

    /// Reserved names do not get returned (they collide with reserved
    /// → ensure_unique walks to `<reserved>_<n>` until clean).
    #[test]
    fn uniquename_avoids_reserved_names() {
        let mut nm = NameManager::new("", "_");
        nm.make_reserved_names("auto");
        // First claim of `auto` sees seennames[auto] == 1; with
        // with_number=Some(false) and last char non-digit, dedup
        // format is `{base}_{n}` → "auto_1".
        assert_eq!(nm.uniquename("auto", Some(false), 50), "auto_1");
    }

    /// `bare=true` returns both the bare and prefixed forms.
    #[test]
    fn uniquename_bare_returns_both_forms() {
        let mut nm = NameManager::new("pypy_", "_");
        let (bare, prefixed) = nm.uniquename_bare("foo", Some(false), 50);
        assert_eq!(bare, "foo");
        assert_eq!(prefixed, "pypy_foo");
    }

    /// `lenmax` truncates the input *before* C-identifier translation.
    #[test]
    fn uniquename_truncates_to_lenmax() {
        let mut nm = NameManager::new("", "_");
        let long = "abcdefghij".repeat(20);
        let out = nm.uniquename(&long, Some(false), 5);
        assert_eq!(out, "abcde");
    }

    /// Non-identifier characters fold to `_` per the C_IDENTIFIER
    /// translation table.
    #[test]
    fn uniquename_translates_non_identifier_chars_to_underscore() {
        let mut nm = NameManager::new("", "_");
        assert_eq!(nm.uniquename("foo-bar.baz", Some(false), 50), "foo_bar_baz");
    }
}
