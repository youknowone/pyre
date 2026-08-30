//! Build native-only library assets outside `pyre-interpreter`.

use std::path::Path;

/// Pure-Python files reachable from `import re`.  The C-level dependencies
/// (`_sre`, `_abc`, `_weakref`, `itertools`, `_collections`, `_thread`,
/// `operator`) are builtin modules and are not embedded.
#[cfg(feature = "wasm_vfs")]
const RE_CLOSURE: &[&str] = &[
    "_collections_abc.py",
    "abc.py",
    "collections/__init__.py",
    "copyreg.py",
    "enum.py",
    "functools.py",
    "keyword.py",
    "re/__init__.py",
    "re/_casefix.py",
    "re/_compiler.py",
    "re/_constants.py",
    "re/_parser.py",
    "reprlib.py",
    "types.py",
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let target = std::env::var("TARGET").unwrap_or_default();
    if !target.starts_with("wasm32-") && std::env::var_os("CARGO_FEATURE_SANDBOX").is_none() {
        build_cffi_parser();
    }
    #[cfg(feature = "wasm_vfs")]
    if std::env::var_os("CARGO_FEATURE_WASM_VFS").is_some() {
        build_stdlib_vfs();
    }
}

fn build_cffi_parser() {
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

/// Pack the browser wasm stdlib closure beside the native decoder that owns
/// its format.  `pyre-interpreter` only embeds this finished asset and parses
/// its Python-facing VFS records; neither LZ4 nor the source-file walk belongs
/// to the frequently re-extracted runtime crate.
#[cfg(feature = "wasm_vfs")]
fn build_stdlib_vfs() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let stdlib_root = Path::new(&manifest_dir).join("../../lib-python/3");
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR");

    // [count: u32 LE], followed by repeated
    // [name_len: u32 LE][name][src_len: u32 LE][source].
    let mut raw = Vec::new();
    raw.extend_from_slice(&(RE_CLOSURE.len() as u32).to_le_bytes());
    for rel in RE_CLOSURE {
        let path = stdlib_root.join(rel);
        println!("cargo:rerun-if-changed={}", path.display());
        let source = std::fs::read(&path)
            .unwrap_or_else(|err| panic!("wasm_vfs: cannot read {}: {err}", path.display()));
        let name = rel.as_bytes();
        raw.extend_from_slice(&(name.len() as u32).to_le_bytes());
        raw.extend_from_slice(name);
        raw.extend_from_slice(&(source.len() as u32).to_le_bytes());
        raw.extend_from_slice(&source);
    }

    let compressed = lz4_flex::block::compress_prepend_size(&raw);
    let blob_path = Path::new(&out_dir).join("stdlib_vfs.lz4");
    std::fs::write(&blob_path, compressed)
        .unwrap_or_else(|err| panic!("wasm_vfs: cannot write {}: {err}", blob_path.display()));
}
