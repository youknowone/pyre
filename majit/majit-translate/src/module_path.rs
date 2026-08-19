//! Deriving a graph's module path from the source file that declares it.
//!
//! `analyze_multiple_pipeline_with_modules` takes `module_paths: &[&str]`
//! parallel to the source files it is given. That string is not a label:
//!
//! - `front::mir::normalize_module_filter` collects it into the set that
//!   decides which LLBC bodies are lowered at all, so a path that does not
//!   match drops every graph in that file, silently;
//! - `majit_ir::descr::path_hash_stripped_crate` hashes the same spelling
//!   for struct and descr identity, so two spellings of one module mint two
//!   type ids for one struct.
//!
//! Both are invariants of this crate, which is why the derivation lives
//! here rather than in each caller's build script. Upstream has no
//! equivalent step — a graph's module is `graph.func.__module__`, produced
//! by the translator that holds the graph — and this is the same
//! arrangement: the translator names its own graphs.

/// Crate-stripped module path of a source file.
///
/// The `/src/`-relative path with `.rs`, a trailing `/lib` or `/mod`, and
/// the crate segment removed, and `/` rewritten to `::`:
///
/// - `foo-runtime/src/storage/list.rs` → `storage::list`
/// - `foo-runtime/src/storage/mod.rs` → `storage`
/// - `foo-runtime/src/lib.rs` → `""` (crate root, no qualifier)
///
/// Returns `""` when the path contains no `/src/` marker. Callers outside
/// the canonical crate layout — synthesized files, fixtures — keep the
/// simple-name registration that an empty path selects.
pub fn module_path_from_source_file(path: &str) -> String {
    // Windows directory walks yield native paths with `\` separators; the
    // marker search, the `/lib` and `/mod` suffix strips, and the final
    // `/` → `::` rewrite all assume forward slashes. An unnormalised
    // Windows path takes the `rfind` `None` branch instead, so every file
    // ends up with an empty module path — which is not an error anywhere,
    // just a whole crate's graphs missing from the pipeline.
    let normalized_path = path.replace('\\', "/");
    let path = normalized_path.as_str();
    let marker = "/src/";
    let Some(idx) = path.rfind(marker) else {
        return String::new();
    };
    let rest = &path[idx + marker.len()..];
    let stem = rest.strip_suffix(".rs").unwrap_or(rest);
    let normalized = stem
        .strip_suffix("/lib")
        .or_else(|| stem.strip_suffix("/mod"))
        .unwrap_or(stem);
    if normalized == "lib" || normalized == "mod" {
        return String::new();
    }
    normalized.replace('/', "::")
}

/// Every `.rs` file under `dir`, recursively, in a stable order.
///
/// Siblings are visited in file-name order and directories are descended
/// as they are reached, so the result does not depend on the filesystem's
/// directory order.
///
/// The order is load-bearing, not cosmetic. It is the order the analyzer
/// meets type and method definitions in, and an order that varies by host
/// produces failures that reproduce on one platform and not another —
/// `SomeBuiltin.call(): no analyser registered for std.ptr.null_mut` and
/// `SomeInstance.getattr on classdef-less instance` on Linux and Windows
/// while macOS passes. Sorting makes the build reproducible and lets one
/// fix cover every platform.
///
/// Unreadable directories are skipped with a warning on stderr rather than
/// failing the caller's build script: a directory that cannot be listed
/// contributes no graphs, which the module filter already tolerates.
pub fn collect_rs_files(dir: &str) -> Vec<String> {
    let mut paths = Vec::new();
    collect_rs_files_into(std::path::Path::new(dir), &mut paths);
    paths
}

fn collect_rs_files_into(dir: &std::path::Path, paths: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        eprintln!("[majit-translate] warning: cannot read {}", dir.display());
        return;
    };
    let mut entries: Vec<_> = entries.flatten().collect();
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            collect_rs_files_into(&path, paths);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            paths.push(path.to_string_lossy().to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_module_path_joins_with_colons() {
        assert_eq!(
            module_path_from_source_file("crate-name/src/storage/list.rs"),
            "storage::list"
        );
    }

    #[test]
    fn a_module_root_names_the_directory_not_the_file() {
        assert_eq!(
            module_path_from_source_file("crate-name/src/storage/mod.rs"),
            "storage"
        );
    }

    /// The crate root qualifies nothing, so it is the empty path — the
    /// same value a path outside the layout returns.
    #[test]
    fn the_crate_root_and_an_unrecognized_path_are_both_empty() {
        assert_eq!(module_path_from_source_file("crate-name/src/lib.rs"), "");
        assert_eq!(module_path_from_source_file("scratch/generated.rs"), "");
    }

    /// A backslash path must reach the same module as its forward-slash
    /// twin. Without the normalisation it takes the `None` branch and
    /// silently returns the crate root instead.
    #[test]
    fn a_windows_path_reaches_the_same_module() {
        assert_eq!(
            module_path_from_source_file(r"crate-name\src\storage\list.rs"),
            module_path_from_source_file("crate-name/src/storage/list.rs")
        );
    }

    /// Only the last `/src/` counts, so a crate checked out beneath
    /// another crate's `src` still reports its own module.
    #[test]
    fn the_last_src_marker_wins() {
        assert_eq!(
            module_path_from_source_file("outer/src/vendor/inner/src/storage/list.rs"),
            "storage::list"
        );
    }

    #[test]
    fn the_walk_is_ordered_by_name_and_descends_in_place() {
        let root = std::env::temp_dir().join("majit-translate-collect-rs-files");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("b_dir")).unwrap();
        std::fs::create_dir_all(root.join("a_dir")).unwrap();
        for path in [
            root.join("m_file.rs"),
            root.join("a_dir/z.rs"),
            root.join("a_dir/a.rs"),
            root.join("b_dir/a.rs"),
            root.join("skipped.txt"),
        ] {
            std::fs::write(path, "").unwrap();
        }

        let found = collect_rs_files(&root.to_string_lossy());
        let relative: Vec<String> = found
            .iter()
            .map(|p| {
                p.strip_prefix(&*root.to_string_lossy())
                    .unwrap_or(p)
                    .replace('\\', "/")
                    .trim_start_matches('/')
                    .to_string()
            })
            .collect();
        assert_eq!(
            relative,
            ["a_dir/a.rs", "a_dir/z.rs", "b_dir/a.rs", "m_file.rs"],
            "siblings ordered by name, each directory descended where it sorts"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A directory that cannot be listed is skipped, not fatal.
    #[test]
    fn a_missing_directory_yields_nothing() {
        assert!(collect_rs_files("/nonexistent/majit-translate/module-path").is_empty());
    }
}
