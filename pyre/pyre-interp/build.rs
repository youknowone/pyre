/// Build script for pyre-interp: runs majit-analyze to generate trace
/// helper functions. This allows the manual tracing code in jit/state.rs
/// to be replaced by auto-generated equivalents.
fn main() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let pyre_base = format!("{manifest_dir}/..");

    let source_dirs = [
        format!("{pyre_base}/pyre-object/src"),
        format!("{pyre_base}/pyre-runtime/src"),
        format!("{pyre_base}/pyre-interp/src"),
    ];

    let mut sources = Vec::new();
    let mut source_paths = Vec::new();

    for dir in &source_dirs {
        collect_rs_files(dir, &mut sources, &mut source_paths);
    }

    let source_refs: Vec<&str> = sources.iter().map(|s| s.as_str()).collect();
    let result = majit_analyze::analyze_multiple(&source_refs);
    let code = majit_analyze::generate_trace_code(&result);

    let out_dir = std::env::var("OUT_DIR").unwrap();
    std::fs::write(format!("{out_dir}/jit_trace_gen.rs"), &code).unwrap();

    for path in &source_paths {
        println!("cargo::rerun-if-changed={path}");
    }
}

fn collect_rs_files(dir: &str, sources: &mut Vec<String>, paths: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries {
        let Ok(entry) = entry else { continue };
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files(&path.to_string_lossy(), sources, paths);
        } else if path.extension().map_or(false, |ext| ext == "rs") {
            let path_str = path.to_string_lossy().to_string();
            if let Ok(content) = std::fs::read_to_string(&path) {
                paths.push(path_str);
                sources.push(content);
            }
        }
    }
}
