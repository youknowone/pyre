//! Crate `build.rs` entry that compiles every ECI this crate declares.

use std::path::{Path, PathBuf};

use syn::parse::Parser;
use syn::punctuated::Punctuated;
use syn::{Attribute, Expr, Item, Lit, Meta, Token};

use crate::ExternalCompilationInfo;
use crate::parse::EciInput;

/// ll2ctypes' C step for every ECI this crate declares: `compile_shared_lib`
/// per ECI (after `convert_sources_to_files`), plus the link directives genc
/// hands the linker. `extra` is merged into every ECI first; it carries what
/// upstream computes from the platform at import time (e.g.
/// `platform.include_dirs_for_libffi()`).
pub fn compile_crate_ecis(extra: &ExternalCompilationInfo) {
    println!("cargo:rustc-check-cfg=cfg(majit_eci_build)");

    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let lib = manifest.join("src/lib.rs");
    let mut collected = Vec::new();
    let submodule_dir = submodule_dir_for_file(&lib);
    walk_file(&lib, &submodule_dir, &[], &mut collected);

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    let mut compiled_any = false;
    for item in collected {
        let e = item.eci.merge(&[extra]);
        for dir in &e.library_dirs {
            println!("cargo:rustc-link-search=native={dir}");
        }
        for arg in e.link_extra.iter().chain(&e.link_files) {
            println!("cargo:rustc-link-arg={arg}");
        }
        if e.separate_module_sources.is_empty() && e.separate_module_files.is_empty() {
            continue;
        }
        let libname = eci_libname(&item.module_path, &item.name);
        let cache = out_dir.join("eci").join(&libname);
        if let Err(err) = std::fs::create_dir_all(&cache) {
            panic!("{}: {err}", cache.display());
        }
        let e = e.convert_sources_to_files(&cache);
        let mut build = cc::Build::new();
        for file in &e.separate_module_files {
            let path = resolve_against(&manifest, file);
            println!("cargo:rerun-if-changed={}", path.display());
            build.file(&path);
        }
        for dir in &e.include_dirs {
            build.include(resolve_against(&manifest, dir));
        }
        for flag in &e.compile_extra {
            build.flag(flag);
        }
        build.cpp(e.use_cpp_linker);
        build.warnings(false);
        build.compile(&libname);
        compiled_any = true;
    }
    if compiled_any {
        println!("cargo:rustc-cfg=majit_eci_build");
    }
}

struct CollectedEci {
    module_path: Vec<String>,
    name: String,
    eci: ExternalCompilationInfo,
}

fn eci_libname(module_path: &[String], const_name: &str) -> String {
    let mut parts: Vec<&str> = module_path.iter().map(String::as_str).collect();
    parts.push(const_name);
    let joined = parts.join("_");
    let mut out = String::from("eci_");
    for c in joined.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else {
            out.push('_');
        }
    }
    out
}

fn resolve_against(manifest: &Path, p: &str) -> PathBuf {
    let path = Path::new(p);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        manifest.join(path)
    }
}

fn submodule_dir_for_file(path: &Path) -> PathBuf {
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned());
    match name.as_deref() {
        Some("lib.rs") | Some("mod.rs") | Some("main.rs") => path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf(),
        _ => path.with_extension(""),
    }
}

fn walk_file(
    file_path: &Path,
    submodule_dir: &Path,
    mod_path: &[String],
    out: &mut Vec<CollectedEci>,
) {
    println!("cargo:rerun-if-changed={}", file_path.display());
    let src = std::fs::read_to_string(file_path).unwrap_or_else(|err| {
        panic!("{}: {err}", file_path.display());
    });
    let file = syn::parse_file(&src).unwrap_or_else(|err| {
        panic!("{}: {err}", file_path.display());
    });
    walk_items(&file.items, file_path, submodule_dir, mod_path, out);
}

fn walk_items(
    items: &[Item],
    file_path: &Path,
    submodule_dir: &Path,
    mod_path: &[String],
    out: &mut Vec<CollectedEci>,
) {
    for item in items {
        if !item_cfgs_hold(item, file_path) {
            continue;
        }
        match item {
            Item::Macro(mac) => {
                let is_eci = mac
                    .mac
                    .path
                    .segments
                    .last()
                    .is_some_and(|seg| seg.ident == "external_compilation_info");
                if !is_eci {
                    continue;
                }
                let parsed: EciInput = syn::parse2(mac.mac.tokens.clone()).unwrap_or_else(|err| {
                    panic!("{}: {err}", file_path.display());
                });
                let eci = parsed.to_eci().unwrap_or_else(|err| {
                    panic!("{}: {err}", file_path.display());
                });
                out.push(CollectedEci {
                    module_path: mod_path.to_vec(),
                    name: parsed.name.to_string(),
                    eci,
                });
            }
            Item::Mod(module) => {
                let name = module.ident.to_string();
                let mut child_path = mod_path.to_vec();
                child_path.push(name.clone());
                match &module.content {
                    Some((_, items)) => {
                        let nested_dir = submodule_dir.join(&name);
                        walk_items(items, file_path, &nested_dir, &child_path, out);
                    }
                    None => {
                        let child_file = resolve_mod_file(
                            file_path,
                            submodule_dir,
                            &name,
                            path_attr(&module.attrs),
                        );
                        let child_dir = submodule_dir_for_file(&child_file);
                        walk_file(&child_file, &child_dir, &child_path, out);
                    }
                }
            }
            _ => {}
        }
    }
}

fn resolve_mod_file(
    current_file: &Path,
    submodule_dir: &Path,
    name: &str,
    path_attr: Option<String>,
) -> PathBuf {
    if let Some(p) = path_attr {
        let path = current_file
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(p);
        if !path.exists() {
            panic!("{}", path.display());
        }
        return path;
    }
    let rs = submodule_dir.join(format!("{name}.rs"));
    if rs.exists() {
        return rs;
    }
    let mod_rs = submodule_dir.join(name).join("mod.rs");
    if mod_rs.exists() {
        return mod_rs;
    }
    panic!("{}", rs.display());
}

fn path_attr(attrs: &[Attribute]) -> Option<String> {
    for attr in attrs {
        if !attr.path().is_ident("path") {
            continue;
        }
        let Meta::NameValue(nv) = &attr.meta else {
            continue;
        };
        if let Expr::Lit(expr_lit) = &nv.value
            && let Lit::Str(s) = &expr_lit.lit
        {
            return Some(s.value());
        }
    }
    None
}

fn item_attrs(item: &Item) -> &[Attribute] {
    match item {
        Item::Const(item) => &item.attrs,
        Item::Enum(item) => &item.attrs,
        Item::ExternCrate(item) => &item.attrs,
        Item::Fn(item) => &item.attrs,
        Item::ForeignMod(item) => &item.attrs,
        Item::Impl(item) => &item.attrs,
        Item::Macro(item) => &item.attrs,
        Item::Mod(item) => &item.attrs,
        Item::Static(item) => &item.attrs,
        Item::Struct(item) => &item.attrs,
        Item::Trait(item) => &item.attrs,
        Item::TraitAlias(item) => &item.attrs,
        Item::Type(item) => &item.attrs,
        Item::Union(item) => &item.attrs,
        Item::Use(item) => &item.attrs,
        _ => &[],
    }
}

fn item_cfgs_hold(item: &Item, file_path: &Path) -> bool {
    for attr in item_attrs(item) {
        if !attr.path().is_ident("cfg") {
            continue;
        }
        let meta = attr.parse_args::<Meta>().unwrap_or_else(|err| {
            panic!("{}: {err}", file_path.display());
        });
        if !cfg_holds(&meta) {
            return false;
        }
    }
    true
}

fn cfg_holds(meta: &Meta) -> bool {
    match meta {
        Meta::Path(path) => {
            let Some(ident) = path.get_ident() else {
                return false;
            };
            let name = ident.to_string();
            if name == "test" || name == "doc" {
                return false;
            }
            std::env::var_os(format!("CARGO_CFG_{}", env_key(&name))).is_some()
        }
        Meta::NameValue(nv) => {
            let Some(ident) = nv.path.get_ident() else {
                return false;
            };
            let key = ident.to_string();
            let Expr::Lit(expr_lit) = &nv.value else {
                return false;
            };
            let Lit::Str(value) = &expr_lit.lit else {
                return false;
            };
            let value = value.value();
            if key == "feature" {
                return std::env::var_os(format!("CARGO_FEATURE_{}", env_key(&value))).is_some();
            }
            match std::env::var(format!("CARGO_CFG_{}", env_key(&key))) {
                Ok(v) => v.split(',').any(|item| item == value),
                Err(_) => false,
            }
        }
        Meta::List(list) => {
            let Some(ident) = list.path.get_ident() else {
                return false;
            };
            let nested =
                match Punctuated::<Meta, Token![,]>::parse_terminated.parse2(list.tokens.clone()) {
                    Ok(p) => p,
                    Err(_) => return false,
                };
            match ident.to_string().as_str() {
                "all" => nested.iter().all(cfg_holds),
                "any" => nested.iter().any(cfg_holds),
                "not" => nested.len() == 1 && !cfg_holds(&nested[0]),
                _ => false,
            }
        }
    }
}

fn env_key(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c == '-' {
                '_'
            } else {
                c.to_ascii_uppercase()
            }
        })
        .collect()
}
