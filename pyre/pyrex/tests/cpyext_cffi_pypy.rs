//! PyPy generated-CFFI extension loading.

#![cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]

mod cpyext_fixture;

use cpyext_fixture::Fixtures;

const SCRIPT: &str = r#"
import sys
import cpyext_cffi_pypy as module

assert type(module.ffi).__module__ == '_cffi_backend'
assert type(module.ffi).__name__ == 'FFI'
assert type(module.lib).__module__ == '_cffi_backend'
assert type(module.lib).__name__ == 'Lib'
assert sys.modules['cpyext_cffi_pypy.lib'] is module.lib
assert module.__file__.endswith('.so')
print('cpyext-cffi-pypy-ok')
"#;

#[test]
fn generated_pypy_cffi_module_uses_interpreter_backend() {
    let fixtures = Fixtures::new("cpyext-cffi-pypy");
    fixtures.compile("cpyext_cffi_pypy");
    fixtures.expect_ok(SCRIPT, &[], "cpyext-cffi-pypy-ok");
}
