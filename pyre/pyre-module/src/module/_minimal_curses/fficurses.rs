//! `fficurses.py` — `setupterm`, `rpy_curses_tigetstr`, `rpy_curses_tparm`.
//!
//! `term.h` renames ordinary identifiers, so the bodies live in
//! `fficurses.c`. `guess_eci` picks the libraries; this ECI stays empty
//! and does not emit a second `#[link]`.

use majit_rlib::rffi::{CCHARP, INT};

majit_rlib::rffi::external_compilation_info! {
    const ECI = {};
}

majit_rlib::rffi::llexternal!(
    pub(super) setupterm = "rpy_curses_setupterm",
    [CCHARP, INT, *mut INT],
    INT,
    compilation_info = ECI
);

majit_rlib::rffi::llexternal!(
    pub(super) rpy_curses_tigetstr = "rpy_curses_tigetstr",
    [CCHARP],
    CCHARP,
    compilation_info = ECI
);

majit_rlib::rffi::llexternal!(
    pub(super) rpy_curses_tparm = "rpy_curses_tparm",
    [CCHARP, INT, INT, INT, INT, INT, INT, INT, INT, INT],
    CCHARP,
    compilation_info = ECI
);

/// Uppercase `int` constants `moduledef.py` copies from `_curses`.
/// The build probe writes each value the headers define, as one `&[ ... ]`
/// expression. `include!` rejects a bare list of elements.
pub static CURSES_INTS: &[(&str, i64)] = include!(concat!(env!("OUT_DIR"), "/curses_constants.rs"));

#[cfg(test)]
mod tests {
    use super::CURSES_INTS;

    #[test]
    fn err_and_ok_match_the_curses_sentinels() {
        let value = |name: &str| {
            CURSES_INTS
                .iter()
                .find(|(n, _)| *n == name)
                .map(|(_, v)| *v)
        };
        assert_eq!(value("ERR"), Some(-1));
        assert_eq!(value("OK"), Some(0));
    }
}
