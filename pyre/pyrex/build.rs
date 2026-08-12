fn main() {
    let target = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let mut symbols = vec![
        "PyThreadState_SetAsyncExc",
        "PyGILState_Ensure",
        "PyGILState_Release",
    ];
    // The C-API entry points a loaded extension resolves against. They exist
    // only in a `cpyext` build, and the same predicate gates the loader itself
    // (see pyre-interpreter's `cpyext` feature).
    if std::env::var_os("CARGO_FEATURE_CPYEXT").is_some()
        && std::env::var_os("CARGO_FEATURE_SANDBOX").is_none()
        && matches!(target.as_str(), "macos" | "linux")
    {
        symbols.extend([
            "PyModuleDef_Init",
            "PyModule_Create2",
            "Py_IncRef",
            "Py_DecRef",
        ]);
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
