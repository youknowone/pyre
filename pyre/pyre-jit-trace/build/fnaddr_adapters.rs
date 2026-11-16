//! Build-time word-ABI adapters for symbolic residual callees.
//!
//! RPython analog: `rtyper.py` `getfunctionptr(graph)` — one compiled C
//! function per nameable graph. Each selected `SymbolicFnaddrTarget` becomes
//! an `extern "C"` trampoline that rebuilds typed arguments from registers
//! (`JitWordArg`) and converts the result back (`JitWordRet`).

use std::fmt::Write as _;

use majit_translate::pipeline::{SymbolicFnaddrTarget, SymbolicTargetDecl, TyShape};

/// Charon `item_meta.name_path()` for these ADTs, taken from the live
/// TypeDecl name arrays in `target/llbc/{pyre-object,pyre-interpreter}.ullbc`.
const RESULT_PATH: &str = "core::result::Result";
const OPTION_PATH: &str = "core::option::Option";
const NONNULL_PATH: &str = "core::ptr::non_null::NonNull";
const PYERROR_PATH: &str = "pyre_interpreter::error::PyError";

const ALLOWED_CRATES: &[&str] = &[
    "pyre_object",
    "pyre_interpreter",
    "majit_rlib",
    "pyre_module",
];

/// First-failing skip reasons, in check order.
const SKIP_REASONS: &[&str] = &[
    "no-abi",
    "no-decl",
    "not-public",
    "not-local",
    "generic",
    "method",
    "crate",
    "arity",
    "arg-shape",
    "ret-shape",
    "path",
];

/// Strict and reserved keywords (Rust 2024). A path segment that matches
/// one is written as a raw identifier (`type` → `r#type`).
fn is_rust_keyword(seg: &str) -> bool {
    matches!(
        seg,
        "abstract"
            | "as"
            | "async"
            | "await"
            | "become"
            | "box"
            | "break"
            | "const"
            | "continue"
            | "crate"
            | "do"
            | "dyn"
            | "else"
            | "enum"
            | "extern"
            | "false"
            | "final"
            | "fn"
            | "for"
            | "gen"
            | "if"
            | "impl"
            | "in"
            | "let"
            | "loop"
            | "macro"
            | "match"
            | "mod"
            | "move"
            | "mut"
            | "override"
            | "priv"
            | "pub"
            | "ref"
            | "return"
            | "self"
            | "Self"
            | "static"
            | "struct"
            | "super"
            | "trait"
            | "true"
            | "try"
            | "type"
            | "typeof"
            | "unsafe"
            | "unsized"
            | "use"
            | "virtual"
            | "where"
            | "while"
            | "yield"
    )
}

fn skip_index(reason: &str) -> usize {
    SKIP_REASONS
        .iter()
        .position(|&r| r == reason)
        .expect("skip reason is in SKIP_REASONS")
}

/// `calldescr_abi_for_path`: argument chars, `>`, result char (`w`/`f`/`v`).
fn parse_abi(abi: &str) -> Option<(&str, char)> {
    let (args, rest) = abi.split_once('>')?;
    let mut chars = rest.chars();
    let result = chars.next()?;
    if chars.next().is_some() {
        return None;
    }
    if !matches!(result, 'w' | 'f' | 'v') {
        return None;
    }
    if !args.chars().all(|c| matches!(c, 'w' | 'f')) {
        return None;
    }
    Some((args, result))
}

fn is_nonnull_adt(ty: &TyShape) -> bool {
    matches!(ty, TyShape::Adt { path, .. } if path == NONNULL_PATH)
}

fn is_option_nonnull(ty: &TyShape) -> bool {
    match ty {
        TyShape::Adt { path, args } if path == OPTION_PATH && args.len() == 1 => {
            is_nonnull_adt(&args[0])
        }
        _ => false,
    }
}

fn is_result_adt(ty: &TyShape) -> Option<(&TyShape, &TyShape)> {
    match ty {
        TyShape::Adt { path, args } if path == RESULT_PATH && args.len() == 2 => {
            Some((&args[0], &args[1]))
        }
        _ => None,
    }
}

fn is_pyerror_adt(ty: &TyShape) -> bool {
    matches!(ty, TyShape::Adt { path, .. } if path == PYERROR_PATH)
}

fn shape_fits_word(ty: &TyShape) -> bool {
    match ty {
        TyShape::Bool | TyShape::Char | TyShape::Int { .. } | TyShape::RawPtr { .. } => true,
        TyShape::Ref { sized: true, .. } => true,
        _ => is_nonnull_adt(ty) || is_option_nonnull(ty),
    }
}

fn shape_fits_abi_arg(ty: &TyShape, abi: char) -> bool {
    match abi {
        'w' => shape_fits_word(ty),
        'f' => matches!(ty, TyShape::Float { bits: 64 }),
        _ => false,
    }
}

fn shape_fits_abi_ret(ty: &TyShape, abi: char) -> bool {
    match abi {
        'v' => matches!(ty, TyShape::Unit),
        'w' => shape_fits_word(ty),
        'f' => matches!(ty, TyShape::Float { bits: 64 }),
        _ => false,
    }
}

fn crate_head(path: &str) -> Option<&str> {
    let head = path.split("::").next().filter(|s| !s.is_empty())?;
    Some(head)
}

fn rust_path_from_canonical(path: &str) -> Option<String> {
    if path.is_empty() {
        return None;
    }
    let mut out = String::from("::");
    for (i, seg) in path.split("::").enumerate() {
        if seg.contains('<') || seg.contains('{') || seg.contains('#') || seg.contains(' ') {
            return None;
        }
        // Private `mod` segments Charon keeps in `name_path`. The parent
        // `pub use`s them, so `::crate::mod::imp::f` is not nameable from
        // another crate even though `f` is `pub`.
        if matches!(seg, "imp" | "posix" | "winsock") {
            return None;
        }
        if seg.is_empty()
            || !seg
                .chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
            || !seg.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        {
            return None;
        }
        if i > 0 {
            out.push_str("::");
        }
        if is_rust_keyword(seg) {
            out.push_str("r#");
        }
        out.push_str(seg);
    }
    Some(out)
}

struct SelectedAdapter<'a> {
    index: usize,
    path: &'a str,
    rust_path: String,
    arg_abi: &'a str,
    result_abi: char,
    pyre_module: bool,
}

fn select_target<'a>(
    index: usize,
    target: &'a SymbolicFnaddrTarget,
) -> Result<SelectedAdapter<'a>, &'static str> {
    let Some(abi) = target.abi.as_deref() else {
        return Err("no-abi");
    };
    let Some((arg_abi, result_abi)) = parse_abi(abi) else {
        return Err("no-abi");
    };
    let Some(decl) = target.decl.as_ref() else {
        return Err("no-decl");
    };
    if !decl.public {
        return Err("not-public");
    }
    if !decl.is_local {
        return Err("not-local");
    }
    if decl.is_generic {
        return Err("generic");
    }
    if decl.is_method {
        return Err("method");
    }
    let Some(head) = crate_head(&target.path) else {
        return Err("crate");
    };
    if !ALLOWED_CRATES.contains(&head) {
        return Err("crate");
    }
    if decl.inputs.len() != arg_abi.len() {
        return Err("arity");
    }
    for (ty, abi_char) in decl.inputs.iter().zip(arg_abi.chars()) {
        if !shape_fits_abi_arg(ty, abi_char) {
            return Err("arg-shape");
        }
    }
    if !ret_shape_ok(decl, target.exception_transformed, result_abi) {
        return Err("ret-shape");
    }
    let Some(rust_path) = rust_path_from_canonical(&target.path) else {
        return Err("path");
    };
    Ok(SelectedAdapter {
        index,
        path: &target.path,
        rust_path,
        arg_abi,
        result_abi,
        pyre_module: head == "pyre_module",
    })
}

fn ret_shape_ok(decl: &SymbolicTargetDecl, exception_transformed: bool, result_abi: char) -> bool {
    let mut ret = &decl.output;
    if exception_transformed {
        let Some((ok, err)) = is_result_adt(ret) else {
            return false;
        };
        if !is_pyerror_adt(err) {
            return false;
        }
        ret = ok;
    } else if is_result_adt(ret).is_some() {
        return false;
    }
    shape_fits_abi_ret(ret, result_abi)
}

fn emit_adapter(out: &mut String, adapter: &SelectedAdapter<'_>) {
    if adapter.pyre_module {
        out.push_str("#[cfg(feature = \"pyre-module\")]\n");
    }
    out.push_str("#[allow(unused_unsafe, clippy::all)]\n");
    out.push_str("pub extern \"C\" fn __gen_fnaddr_adapter_");
    let _ = write!(out, "{}", adapter.index);
    out.push('(');
    for (i, abi_char) in adapter.arg_abi.chars().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        let ty = match abi_char {
            'f' => "f64",
            _ => "i64",
        };
        let _ = write!(out, "a{i}: {ty}");
    }
    match adapter.result_abi {
        'w' => out.push_str(") -> i64 {\n"),
        'f' => out.push_str(") -> f64 {\n"),
        _ => out.push_str(") {\n"),
    }
    out.push_str(
        "    ::pyre_interpreter::jit_word_adapter::JitWordRet::into_reg(unsafe {\n        ",
    );
    out.push_str(&adapter.rust_path);
    out.push('(');
    if adapter.arg_abi.is_empty() {
        out.push(')');
    } else {
        out.push('\n');
        for i in 0..adapter.arg_abi.len() {
            out.push_str(
                "            ::pyre_interpreter::jit_word_adapter::JitWordArg::from_reg(a",
            );
            let _ = write!(out, "{i}),\n");
        }
        out.push_str("        )");
    }
    out.push_str("\n    })\n}\n\n");
}

fn emit_table(out: &mut String, selected: &[SelectedAdapter<'_>]) {
    out.push_str("pub fn generated_fnaddr_adapters() -> Vec<(&'static str, i64)> {\n");
    out.push_str("    let mut rows = Vec::new();\n");
    for adapter in selected {
        if adapter.pyre_module {
            out.push_str("    #[cfg(feature = \"pyre-module\")]\n");
        }
        out.push_str("    rows.push((\"");
        out.push_str(adapter.path);
        out.push_str("\", __gen_fnaddr_adapter_");
        let _ = write!(out, "{} as usize as i64));\n", adapter.index);
    }
    out.push_str("    rows\n}\n");
}

fn format_skip_counts(counts: &[usize]) -> String {
    let mut parts = Vec::new();
    for (reason, count) in SKIP_REASONS.iter().zip(counts.iter()) {
        if *count > 0 {
            parts.push(format!("{reason}={count}"));
        }
    }
    parts.join(" ")
}

/// Write `OUT_DIR/generated_fnaddr_adapters.rs` and a skip summary.
pub fn write_generated_fnaddr_adapters(out_dir: &str, targets: &[SymbolicFnaddrTarget]) {
    let mut skip_counts = vec![0usize; SKIP_REASONS.len()];
    let mut skipped_lines = Vec::new();
    let mut selected = Vec::new();
    for (index, target) in targets.iter().enumerate() {
        match select_target(index, target) {
            Ok(adapter) => selected.push(adapter),
            Err(reason) => {
                skip_counts[skip_index(reason)] += 1;
                skipped_lines.push(format!("{} {reason}", target.path));
            }
        }
    }

    let mut src = String::new();
    src.push_str("// @generated by pyre-jit-trace/build/fnaddr_adapters.rs. Do not edit.\n\n");
    for adapter in &selected {
        emit_adapter(&mut src, adapter);
    }
    emit_table(&mut src, &selected);

    std::fs::write(format!("{out_dir}/generated_fnaddr_adapters.rs"), src).unwrap();

    let emitted = selected.len();
    let skipped = skipped_lines.len();
    let reasons = format_skip_counts(&skip_counts);
    let summary = format!(
        "[pyre-jit-trace build.rs] generated fnaddr adapters: {emitted} emitted, {skipped} skipped ({reasons})"
    );
    eprintln!("{summary}");

    let mut summary_txt = String::new();
    summary_txt.push_str(&summary);
    summary_txt.push('\n');
    for line in &skipped_lines {
        summary_txt.push_str(line);
        summary_txt.push('\n');
    }
    std::fs::write(
        format!("{out_dir}/generated_fnaddr_adapters.summary.txt"),
        summary_txt,
    )
    .unwrap();
}
