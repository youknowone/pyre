//! Build the native-only CFFI declaration parser outside pyre-interpreter.

use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let target = std::env::var("TARGET").unwrap_or_default();
    if target.starts_with("wasm32-") || std::env::var_os("CARGO_FEATURE_SANDBOX").is_some() {
        return;
    }

    // PyPy `parse_c_type.py` compiles this exact translation unit: the opcode
    // stream is the ABI consumed by CFFI extension metadata, not interpreter
    // behavior. `longdouble.c` likewise asks the target C compiler about the
    // representation Rust must not guess.
    let root = Path::new("src/cffi_backend");
    let sources = [
        "src/cffi_backend/src/parse_c_type.c",
        "src/cffi_backend/src/longdouble.c",
    ];
    for source in sources {
        println!("cargo:rerun-if-changed={source}");
    }
    for header in [
        "src/precommondefs.h",
        "src/parse_c_type.h",
        "src/commontypes.c",
    ] {
        println!("cargo:rerun-if-changed={}", root.join(header).display());
    }
    cc::Build::new()
        .files(sources)
        .include(root)
        .warnings(false)
        .compile("pyre_cffi_parse_c_type");
}
