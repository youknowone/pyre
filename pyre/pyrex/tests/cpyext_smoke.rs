//! End-to-end check for single-phase extension loading.
//!
//! The fixture is compiled against pyre's own 3.14 ABI header, discovered by
//! the normal import path, loaded through `PyInit_*`, and created by the C call
//! back into the executable's exported `PyModule_Create2`.

#![cfg(all(
    feature = "cpyext",
    not(feature = "sandbox"),
    any(target_os = "macos", target_os = "linux")
))]

mod cpyext_fixture;

use std::path::Path;

use cpyext_fixture::{Fixtures, extension_suffix};

#[test]
fn imports_single_phase_module_created_through_c_api() {
    let fixtures = Fixtures::new("cpyext-smoke");
    fixtures.compile("cpyext_smoke");
    let init_marker = fixtures.join("initialized");
    let unload_marker = fixtures.join("unloaded");

    let code = r#"
import _imp
import _sysconfig
import sys
expected_suffix = %SUFFIX%
suffixes = _imp.extension_suffixes()
assert suffixes[0] == expected_suffix
assert '.abi3.so' in suffixes
assert '.so' not in suffixes
config = _sysconfig.config_vars()
assert config['EXT_SUFFIX'] == expected_suffix
assert expected_suffix == '.' + config['SOABI'] + '.so'
old_dlopenflags = sys.getdlopenflags()
sys.setdlopenflags(0)
try:
    import cpyext_smoke
finally:
    sys.setdlopenflags(old_dlopenflags)
again = __import__('cpyext_smoke')
assert cpyext_smoke is again
assert cpyext_smoke.__name__ == 'cpyext_smoke'
assert cpyext_smoke.__doc__ == 'pyre cpyext smoke module'
assert cpyext_smoke.__file__.endswith(_imp.extension_suffixes()[0])
first = cpyext_smoke
first.__doc__ = 'mutated'
first.runtime_only = 1
del sys.modules['cpyext_smoke']
del cpyext_smoke
reloaded = __import__('cpyext_smoke')
assert reloaded is not first
assert reloaded.__name__ == 'cpyext_smoke'
assert reloaded.__doc__ == 'pyre cpyext smoke module'
assert not hasattr(reloaded, 'runtime_only')
try:
    with open(%UNLOAD_MARKER%, 'rb'):
        pass
except FileNotFoundError:
    pass
else:
    raise AssertionError('extension unloaded during cache restore')
print('cpyext-smoke-ok')
"#
    .replace("%SUFFIX%", &format!("{:?}", extension_suffix()))
    .replace(
        "%UNLOAD_MARKER%",
        &format!("{:?}", unload_marker.to_string_lossy()),
    );

    fixtures.expect_ok(
        &code,
        &[
            ("PYRE_CPYEXT_INIT_MARKER", init_marker.as_path()),
            ("PYRE_CPYEXT_UNLOAD_MARKER", unload_marker.as_path()),
        ],
        "cpyext-smoke-ok",
    );
}

#[test]
fn imports_abi3_suffix_module() {
    let fixtures = Fixtures::new("cpyext-abi3");
    fixtures.compile_with_suffix("cpyext_smoke", ".abi3.so");

    let code = r#"
import _imp
import cpyext_smoke
assert '.abi3.so' in _imp.extension_suffixes()
assert cpyext_smoke.__name__ == 'cpyext_smoke'
assert cpyext_smoke.__file__.endswith('.abi3.so'), cpyext_smoke.__file__
print('cpyext-abi3-ok')
"#;

    fixtures.expect_ok(&code, &[], "cpyext-abi3-ok");
}

#[test]
fn abi3_tags_finder_installs_when_loader_exists() {
    let fixtures = Fixtures::new("cpyext-abi3-tags");
    let lib_pypy = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("pyrex manifest sits two levels below the repo root")
        .join("lib_pypy");
    let code = format!(
        r#"
import sys
sys.path.insert(0, {lib_pypy})
import _imp
import _pypy_abi3_tags
assert '.abi3.so' in _imp.extension_suffixes(), _imp.extension_suffixes()
assert any(isinstance(f, _pypy_abi3_tags._Abi3TagsFinder) for f in sys.meta_path)
print('cpyext-abi3-finder-ok')
"#,
        lib_pypy = format!("{:?}", lib_pypy.to_string_lossy())
    );

    fixtures.expect_ok(&code, &[], "cpyext-abi3-finder-ok");
}
