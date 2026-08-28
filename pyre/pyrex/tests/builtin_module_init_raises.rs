//! A builtin module whose app-level body cannot import reaches the program as
//! an exception, not as an abort.
//!
//! `mixedmodule.py MixedModule.get` runs a module's sibling `app_*.py` lazily,
//! through the ordinary import machinery, so whatever that source raises
//! unwinds out of the import that asked for the module. Pyre installs those
//! sources eagerly from the module initializer instead -- the registry stores
//! `name -> init` and `load_builtin_module` runs it at first import -- and the
//! initializer used to have no way to report a failure, so an app source that
//! raised took the process with it.
//!
//! The imports in such a source resolve through the running `sys.modules`, and
//! `sys.modules[name] = None` is the documented way for a program to stop
//! everything downstream from importing that name. `_collections`'s app source
//! opens with three of them (`__pypy__`, `_collections_abc`, `_operator`), so a
//! program that blocks any one and then imports `_collections` decides whether
//! the initializer can fail.
//!
//! Where such a source is answered is the same question once removed. A body
//! that binds the module (`import operator as _operator`) takes whatever the
//! program left under the name and reports it much later and somewhere else,
//! so these sources name what they need (`from operator import index`) and the
//! import is where a blocked name comes back.
//!
//! Only an end-to-end run sees this: the failure is a process abort, which no
//! in-process test can observe, and the path runs a whole app-level module body
//! from inside a builtin module's initializer.
//!
//! Do not invoke Cargo from this test: the parent `cargo test` holds the target
//! directory lock. `CARGO_BIN_EXE_pyre-dynasm` supplies the binary Cargo already
//! built for this integration test.

// `pyre-dynasm` is `required-features = ["dynasm"]`, so under
// `--no-default-features` the binary does not exist and `CARGO_BIN_EXE_…` is
// unset — which would be a compile error rather than a skip. Compile this file
// to nothing in that configuration instead.
#![cfg(feature = "dynasm")]

use std::path::PathBuf;
use std::process::Command;

/// The dynasm-backend `pyre` binary cargo built for this test.
const PYRE: &str = env!("CARGO_BIN_EXE_pyre-dynasm");

/// Blocks the three names `app_odict.py` imports, then asks for the module
/// that installs it. The `except` is what makes the result observable: an
/// initializer that cannot fail never reaches it.
const PROGRAM: &str = "\
import sys
sys.modules['__pypy__'] = None
sys.modules['_collections_abc'] = None
sys.modules['_operator'] = None
try:
    import _collections
except ImportError:
    print('raised')
else:
    print('imported')
";

/// Blocks the name `_blake2_app.py` imports, then asks for the module that
/// installs it. A source that binds whatever the program left under the name
/// instead reports it much later and somewhere else -- `_operator.index` on an
/// `int` raised an `AttributeError` from inside `blake2b` -- so the `except`
/// here is what says the import is where it is answered.
const BINDS_JUNK_PROGRAM: &str = "\
import sys
sys.modules['operator'] = 42
try:
    import _blake2
except ImportError:
    print('raised')
else:
    print('imported')
";

/// Run `program` as a file and return its trimmed stdout, failing the test on
/// a panic or a non-zero exit with the tail of stderr attached.
fn stdout_of(name: &str, program: &str) -> String {
    let mut path = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    path.push(name);
    std::fs::write(&path, program).expect("write the program under test");

    let out = Command::new(PYRE)
        .arg(&path)
        .output()
        .expect("spawn pyre-dynasm");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let report = format!(
        "exited {:?}\n--- stdout ---\n{stdout}\n--- stderr ---\n{}",
        out.status,
        stderr.lines().rev().take(20).collect::<Vec<_>>().join("\n"),
    );

    assert!(!stderr.contains("panicked at"), "{report}");
    assert!(out.status.success(), "{report}");
    stdout.trim().to_string()
}

#[test]
fn a_blocked_import_in_a_builtins_app_source_raises_instead_of_aborting() {
    // The blocked name is what the source's own `from __pypy__ import ...`
    // hits, so the import that asked for `_collections` is the one that fails.
    assert_eq!(
        stdout_of("builtin_module_init_raises.py", PROGRAM),
        "raised"
    );
}

#[test]
fn a_builtins_app_source_names_what_it_imports_instead_of_binding_it() {
    assert_eq!(
        stdout_of("builtin_module_init_binds_junk.py", BINDS_JUNK_PROGRAM),
        "raised"
    );
}
