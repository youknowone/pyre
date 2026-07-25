fn main() {
    let target = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    for symbol in [
        "PyThreadState_SetAsyncExc",
        "PyGILState_Ensure",
        "PyGILState_Release",
    ] {
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
