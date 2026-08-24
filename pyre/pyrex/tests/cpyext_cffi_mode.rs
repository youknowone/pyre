//! Which of its two forms a module cffi generated compiles to.

#![cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]

mod cpyext_fixture;

use cpyext_fixture::Fixtures;

const SCRIPT: &str = r#"
import sys
import cpyext_cffi_mode as m

saw_version, saw_version_num, init_prefix, hexversion = m.chosen()
# `_CFFI_` is set, so neither macro is: the module builds its `_cffi_exports`
# side and imports `_cffi_backend` from `PyInit_`.
assert (saw_version, saw_version_num) == (0, 0), (saw_version, saw_version_num)
assert init_prefix == 'PyInit_', init_prefix
# The marker decides that one question and nothing else.
assert hexversion == sys.hexversion, (hex(hexversion), hex(sys.hexversion))

print('cpyext-cffi-mode-ok')
"#;

#[test]
fn the_form_a_cffi_module_compiles_to() {
    let fixtures = Fixtures::new("cpyext-cffi-mode");
    fixtures.compile("cpyext_cffi_mode");
    fixtures.expect_ok(SCRIPT, &[], "cpyext-cffi-mode-ok");
}
