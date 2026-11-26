#[path = "../build/fnaddr_adapters.rs"]
mod fnaddr_adapters;

use majit_translate::pipeline::{SymbolicFnaddrTarget, SymbolicTargetDecl, TyShape};

#[test]
fn target_decl_emits_adapter_and_absent_host_decl_is_skipped() {
    let directory = std::env::temp_dir().join(format!(
        "fnaddr-manifest-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&directory).unwrap();
    let portable = SymbolicFnaddrTarget {
        path: "portable".to_string(),
        abi: Some(">w".to_string()),
        exception_transformed: false,
        decl: Some(SymbolicTargetDecl {
            public: true,
            is_local: true,
            is_generic: false,
            is_method: false,
            is_unsafe: false,
            inputs: Vec::new(),
            output: TyShape::Bool,
        }),
        decl_path: Some("pyre_object::portable".to_string()),
    };
    let absent = SymbolicFnaddrTarget {
        path: "host_only".to_string(),
        abi: Some(">w".to_string()),
        exception_transformed: false,
        decl: None,
        decl_path: None,
    };
    fnaddr_adapters::write_generated_fnaddr_adapters(
        directory.to_str().unwrap(),
        &[portable, absent],
    );
    let generated =
        std::fs::read_to_string(directory.join("generated_fnaddr_adapters.rs")).unwrap();
    let summary =
        std::fs::read_to_string(directory.join("generated_fnaddr_adapters.summary.txt")).unwrap();
    assert!(generated.contains("::pyre_object::portable"));
    assert!(!generated.contains("host_only"));
    assert!(summary.contains("1 emitted, 1 skipped (no-decl=1)"));
    assert!(summary.contains("host_only no-decl"));
    assert!(!summary.contains("cross-target"));
    std::fs::remove_dir_all(&directory).unwrap();
}
