//! `clibffi` and `dup3` (`rposix.py` `HAVE_DUP3`) probes go through
//! `majit_eci::rffi_platform`, and `majit_eci::build::compile_crate_ecis`
//! compiles each ECI's C sources (`cbuild.py` `compile_shared_lib`).
#![allow(clippy::disallowed_methods, clippy::disallowed_types)]

use majit_eci::ExternalCompilationInfo;
use majit_eci::rffi_platform::{CConfig, CConfigEntry, CConfigSingleEntry, ConfigValue, configure};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let have_dup3 = have_dup3();
    majit_eci::build::compile_crate_ecis(&ExternalCompilationInfo {
        pre_include_bits: vec![format!("#define HAVE_DUP3 {}", have_dup3 as i32)],
        ..Default::default()
    });

    // `clibffi.py configure_simple_type('ffi_type_longdouble')` reads the
    // symbol's size/alignment/type. libffi defines that symbol only where its
    // configure saw `sizeof(long double) != sizeof(double)`; elsewhere
    // `ffi.h` makes `ffi_type_longdouble` a macro for `ffi_type_double`, and
    // naming the absent symbol is a link error. Ask the target's compiler the
    // same question libffi's configure asked.
    println!("cargo:rustc-check-cfg=cfg(clibffi_longdouble)");
    if target_has_clibffi() && long_double_is_its_own_type() {
        println!("cargo:rustc-cfg=clibffi_longdouble");
    }
}

fn target_has_clibffi() -> bool {
    std::env::var("CARGO_CFG_UNIX").is_ok() || std::env::var("CARGO_CFG_WINDOWS").is_ok()
}

/// Whether the target's `long double` is wider than its `double`, which is
/// what libffi's `HAVE_LONG_DOUBLE` records. Asks `rffi_platform` for both
/// sizes.
fn long_double_is_its_own_type() -> bool {
    let cconfig = CConfig {
        compilation_info: ExternalCompilationInfo::default(),
        entries: vec![
            (
                "LONGDOUBLE".into(),
                CConfigEntry::SizeOf("long double".into()),
            ),
            ("DOUBLE".into(), CConfigEntry::SizeOf("double".into())),
        ],
        single_entries: vec![],
    };
    let results = configure(&cconfig).unwrap_or_else(|err| panic!("{err}"));
    let mut long_double = None;
    let mut double = None;
    for (name, value) in results {
        match (name.as_str(), value) {
            ("LONGDOUBLE", ConfigValue::Size(n)) => long_double = Some(n),
            ("DOUBLE", ConfigValue::Size(n)) => double = Some(n),
            _ => panic!("{name}"),
        }
    }
    long_double.expect("LONGDOUBLE") != double.expect("DOUBLE")
}

/// `rposix.py` `HAVE_DUP3 = rffi_platform.Has('dup3')`. `Has` builds a program
/// that names `dup3`; it fails when the headers do not declare it or the C
/// library does not provide it.
fn have_dup3() -> bool {
    if std::env::var("CARGO_CFG_UNIX").is_err() {
        return false;
    }
    let cconfig = CConfig {
        compilation_info: ExternalCompilationInfo {
            includes: vec!["unistd.h".into(), "fcntl.h".into()],
            ..Default::default()
        },
        entries: vec![],
        single_entries: vec![("HAVE_DUP3".into(), CConfigSingleEntry::Has("dup3".into()))],
    };
    let results = configure(&cconfig).unwrap_or_else(|err| panic!("{err}"));
    match results.as_slice() {
        [(_, ConfigValue::Bool(have))] => *have,
        _ => panic!("{results:?}"),
    }
}
