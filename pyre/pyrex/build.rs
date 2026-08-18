use std::path::Path;

fn main() {
    let target = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let mut symbols = vec![
        "PyThreadState_SetAsyncExc".to_string(),
        "PyGILState_Ensure".to_string(),
        "PyGILState_Release".to_string(),
    ];
    // The C-API entry points a loaded extension resolves against. They exist
    // only in a `cpyext` build, and the same predicate gates the loader itself
    // (see pyre-interpreter's `cpyext` feature).
    if std::env::var_os("CARGO_FEATURE_CPYEXT").is_some()
        && std::env::var_os("CARGO_FEATURE_SANDBOX").is_none()
        && matches!(target.as_str(), "macos" | "linux")
    {
        symbols.extend(cpyext_symbols());
    }
    for symbol in symbols {
        match target.as_str() {
            "macos" => println!(
                "cargo::rustc-link-arg-bins=-Wl,-exported_symbol,_{}",
                symbol
            ),
            "linux" => println!(
                "cargo::rustc-link-arg-bins=-Wl,--export-dynamic-symbol={}",
                symbol
            ),
            _ => {}
        }
    }
}

/// Every `#[unsafe(no_mangle)]` item the cpyext layer defines.
///
/// Read out of the sources rather than repeated here: the list is the layer's
/// entire public ABI, and a hand-kept copy would silently stop exporting a new
/// entry point — which fails at `dlopen` time in a loaded extension, not at
/// build time.
fn cpyext_symbols() -> Vec<String> {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../pyre-interpreter/src/cpyext")
        .canonicalize()
        .expect("the cpyext sources are part of the workspace");
    println!("cargo::rerun-if-changed={}", directory.display());
    let mut symbols = Vec::new();
    let mut sources: Vec<_> = std::fs::read_dir(&directory)
        .expect("read the cpyext source directory")
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|suffix| suffix == "rs"))
        .collect();
    sources.sort();
    for source in sources {
        println!("cargo::rerun-if-changed={}", source.display());
        let text = std::fs::read_to_string(&source).expect("read a cpyext source file");
        let mut exported = false;
        for line in text.lines() {
            let line = line.trim();
            if line == "#[unsafe(no_mangle)]" {
                exported = true;
                continue;
            }
            if !exported || line.starts_with("///") || line.starts_with("//") {
                continue;
            }
            exported = false;
            if let Some(name) = declared_symbol(line) {
                symbols.push(name);
            }
        }
    }
    // A macro-generated static carries the attribute inside the expansion, so
    // the scan above cannot see it; each such family is named by the table its
    // macro is invoked with.
    symbols.extend(macro_generated_symbols(&directory));
    symbols.sort();
    symbols.dedup();
    symbols
}

/// The symbol a `pub ... fn NAME` / `pub static [mut] NAME` item declares.
fn declared_symbol(line: &str) -> Option<String> {
    let rest = line.strip_prefix("pub ")?;
    let rest = if let Some(rest) = rest.strip_prefix("static mut ") {
        rest
    } else if let Some(rest) = rest.strip_prefix("static ") {
        rest
    } else {
        let mut rest = rest;
        for keyword in ["unsafe ", "extern \"C\" ", "extern \"C-unwind\" "] {
            rest = rest.strip_prefix(keyword).unwrap_or(rest);
        }
        rest.strip_prefix("fn ")?
    };
    let name: String = rest
        .chars()
        .take_while(|character| character.is_alphanumeric() || *character == '_')
        .collect();
    (!name.is_empty()).then_some(name)
}

/// The globals a table-driven macro declares: the `PyExc_*` pointers
/// (`pyerrors.rs`) and the `Py*_Type` blocks (`typeobject.rs`).
fn macro_generated_symbols(directory: &Path) -> Vec<String> {
    const TABLES: [(&str, &str); 2] = [
        ("pyerrors.rs", "exception_mirrors! {"),
        ("typeobject.rs", "type_mirrors! {"),
    ];
    let mut symbols = Vec::new();
    for (source, opening) in TABLES {
        let text = std::fs::read_to_string(directory.join(source))
            .expect("read a table-carrying cpyext source file");
        let Some(table) = text.split(opening).nth(1) else {
            continue;
        };
        let Some(table) = table.split("\n}").next() else {
            continue;
        };
        symbols.extend(
            table
                .lines()
                .filter_map(|line| line.trim().split(" =>").next())
                .filter(|name| {
                    !name.is_empty()
                        && name
                            .chars()
                            .all(|character| character.is_alphanumeric() || character == '_')
                })
                .map(str::to_string),
        );
    }
    symbols
}
