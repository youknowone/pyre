fn main() {
    let target = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let mut symbols = vec![
        "PyThreadState_SetAsyncExc",
        "PyGILState_Ensure",
        "PyGILState_Release",
    ];
    if std::env::var_os("CARGO_FEATURE_SANDBOX").is_none()
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
