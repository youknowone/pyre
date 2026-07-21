//! Wrapper around RustPython's compiler to parse and compile Python source.

pub use rustpython_compiler::CompileError;
pub use rustpython_compiler::CompileOpts;
pub use rustpython_compiler::Mode;
pub use rustpython_compiler::compile as rp_compile;
pub use rustpython_compiler_core::bytecode::{
    self, BinaryOperator, CodeFlags, CodeObject, ComparisonOperator, ConstantData, Instruction,
    MakeFunctionFlags, OpArg, OpArgState, SpecialMethod,
};

/// Compile Python source code to a RustPython CodeObject.
///
/// The filename is the one `exec`/`eval` report for a str source, surfacing
/// as `co_filename` and as the `SyntaxError.filename` of a failed compile.
///
/// The `CompileError` is returned unflattened so the SyntaxError builders
/// can read its `python_location` / `python_end_location` / `source_path`.
pub fn compile_source(source: &str, mode: Mode) -> Result<CodeObject, CompileError> {
    rp_compile(source, mode, "<string>".into(), Default::default())
}

/// Compile Python source code with a custom filename.
///
/// PyPy equivalent: `parse_source_module(space, pathname, source)` in importing.py
pub fn compile_source_with_filename(
    source: &str,
    mode: Mode,
    filename: &str,
) -> Result<CodeObject, CompileError> {
    compile_source_with_opts(source, mode, filename, Default::default())
}

/// Compile Python source with an explicit `CompileOpts`, carrying the
/// `__future__` feature flags and the `optimize` level that `compile()`
/// resolves from its `flags` / `optimize` arguments (pycompiler.py
/// `PythonAstCompiler.compile`).
pub fn compile_source_with_opts(
    source: &str,
    mode: Mode,
    filename: &str,
    opts: CompileOpts,
) -> Result<CodeObject, CompileError> {
    rp_compile(source, mode, filename.into(), opts)
}

/// Scan the first two lines of `source` for a PEP 263 coding cookie
/// (`# -*- coding: <name> -*-`), returning the normalized encoding name.
///
/// A leading UTF-8 BOM on the first line is ignored for the scan.  Only a line
/// that is a comment (optionally preceded by blanks) may carry a cookie; a
/// first line with non-blank, non-`#` content stops the search.
fn detect_source_encoding(source: &[u8]) -> Option<String> {
    fn find_encoding_in_line(line: &[u8]) -> Option<String> {
        let hash_pos = line.iter().position(|&b| b == b'#')?;
        if !line[..hash_pos]
            .iter()
            .all(|&b| b == b' ' || b == b'\t' || b == b'\x0c' || b == b'\r')
        {
            return None;
        }
        let after_hash = &line[hash_pos..];
        let coding_pos = after_hash.windows(6).position(|w| w == b"coding")?;
        let after_coding = &after_hash[coding_pos + 6..];
        let rest = if after_coding.first() == Some(&b':') || after_coding.first() == Some(&b'=') {
            &after_coding[1..]
        } else {
            return None;
        };
        let name: String = rest
            .iter()
            .copied()
            .skip_while(|&b| b == b' ' || b == b'\t')
            .take_while(|&b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.')
            .map(|b| b as char)
            .collect();
        (!name.is_empty()).then(|| normalize_source_encoding(&name))
    }

    let mut lines = source.splitn(3, |&b| b == b'\n');
    if let Some(first) = lines.next() {
        let first = first.strip_prefix(b"\xef\xbb\xbf").unwrap_or(first);
        if let Some(enc) = find_encoding_in_line(first) {
            return Some(enc);
        }
        let trimmed = first
            .iter()
            .skip_while(|&&b| b == b' ' || b == b'\t' || b == b'\x0c' || b == b'\r')
            .copied()
            .collect::<Vec<_>>();
        if !trimmed.is_empty() && trimmed[0] != b'#' {
            return None;
        }
    }
    lines.next().and_then(find_encoding_in_line)
}

/// `tokenizer.c` `_Py_normalize_encoding` slice used for source cookies: the
/// name is lower-cased (with `_`→`-`) and the `utf-8` / `latin-1` families are
/// canonicalized; everything else is returned unchanged.
fn normalize_source_encoding(name: &str) -> String {
    let mut normalized = String::with_capacity(name.len().min(12));
    for ch in name.chars().take(12) {
        if ch == '_' {
            normalized.push('-');
        } else {
            normalized.push(ch.to_ascii_lowercase());
        }
    }

    if normalized == "utf-8" || normalized.starts_with("utf-8-") {
        "utf-8".to_owned()
    } else if normalized == "latin-1"
        || normalized == "iso-8859-1"
        || normalized == "iso-latin-1"
        || normalized.starts_with("latin-1-")
        || normalized.starts_with("iso-8859-1-")
        || normalized.starts_with("iso-latin-1-")
    {
        "iso-8859-1".to_owned()
    } else {
        name.to_owned()
    }
}

fn is_utf8_encoding(name: &str) -> bool {
    name == "utf-8"
}

/// Decode a bytes-like `compile`/`exec`/`eval` source into text, honoring the
/// PEP 263 coding cookie.
///
/// The first two lines are scanned for a cookie (unless `ignore_cookie`), a
/// leading UTF-8 BOM is stripped, and the bytes are decoded strictly.  An
/// undeclared non-UTF-8 byte raises `SyntaxError` (PEP 263) rather than being
/// lossily replaced.
pub fn decode_source_bytes(
    source: &[u8],
    filename: &str,
    ignore_cookie: bool,
) -> Result<String, crate::PyError> {
    let has_bom = source.starts_with(b"\xef\xbb\xbf");
    let encoding = if ignore_cookie {
        None
    } else {
        detect_source_encoding(source)
    };
    let is_utf8 = encoding.as_deref().is_none_or(is_utf8_encoding);
    if has_bom && !is_utf8 {
        let enc = encoding.as_deref().unwrap_or("utf-8");
        return Err(crate::PyError::syntax_error(format!(
            "encoding problem: {enc} with BOM"
        )));
    }

    if is_utf8 {
        let src = if has_bom { &source[3..] } else { source };
        match core::str::from_utf8(src) {
            Ok(s) => Ok(s.to_owned()),
            Err(e) => {
                let bad_byte = src[e.valid_up_to()];
                let line = src[..e.valid_up_to()]
                    .iter()
                    .filter(|&&b| b == b'\n')
                    .count()
                    + 1;
                Err(crate::PyError::syntax_error(format!(
                    "Non-UTF-8 code starting with '\\x{bad_byte:02x}' \
                     on line {line}, but no encoding declared; \
                     see https://peps.python.org/pep-0263/ for details \
                     ({filename}, line {line})"
                )))
            }
        }
    } else {
        let encoding = encoding.as_deref().unwrap();
        let decoded =
            crate::typedef::decode_bytes_to_wtf8(source, encoding, "strict").map_err(|exc| {
                if exc.kind == crate::PyErrorKind::LookupError {
                    crate::PyError::syntax_error(format!(
                        "unknown encoding for '{filename}': {encoding}"
                    ))
                } else {
                    exc
                }
            })?;
        Ok(decoded.to_string_lossy().into_owned())
    }
}

/// Compile a Python expression.
pub fn compile_eval(source: &str) -> Result<CodeObject, CompileError> {
    compile_source(source, Mode::Eval)
}

/// Compile a Python script (module).
pub fn compile_exec(source: &str) -> Result<CodeObject, CompileError> {
    compile_source(source, Mode::Exec)
}
