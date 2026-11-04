//! `rffi_platform.py`: ask the target's C compiler from a build script.

use std::fmt;
use std::fs;
use std::hash::{DefaultHasher, Hasher};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::ExternalCompilationInfo;

/// `platform.CompilationError`.
#[derive(Debug)]
pub struct CompilationError(pub String);

impl fmt::Display for CompilationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for CompilationError {}

/// `rffi_platform.py` `C_HEADER`.
const C_HEADER: &str = r#"
#include <stdio.h>
#include <stddef.h>   /* for offsetof() */

void dump(char* key, int value) {
    printf("%s: %d\n", key, value);
}
"#;

/// Stored in front of a `gcc_cache` payload; a hash collision is a miss.
const CACHE_SEP: &str = "---- rffi_platform cache ----";

/// The (sign, bits) class, `rffi.size_and_sign`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntType {
    pub signed: bool,
    pub bits: u32,
}

/// `rffi_platform.py` `CConfigEntry`.
pub enum CConfigEntry {
    /// `rffi_platform.py` `SimpleType`. The `ctype_hint` is always an integer
    /// type here, and `fixup_ctype` then lands on the class of the measured
    /// (size, sign), so the hint is not stored.
    SimpleType(String),
    /// `rffi_platform.py` `SizeOf`.
    SizeOf(String),
    /// `rffi_platform.py` `ConstantInteger`.
    ConstantInteger(String),
}

impl CConfigEntry {
    fn prepare_code(&self) -> Vec<String> {
        match self {
            CConfigEntry::SimpleType(name) => vec![
                format!("typedef {name} platcheck_t;"),
                String::new(),
                "platcheck_t x;".to_string(),
                "dump(\"size\",  sizeof(platcheck_t));".to_string(),
                "x = 0; x = ~x;".to_string(),
                "dump(\"unsigned\", x > 0);".to_string(),
            ],
            CConfigEntry::SizeOf(name) => {
                vec![format!("dump(\"size\",  sizeof({name}));")]
            }
            CConfigEntry::ConstantInteger(name) => vec![
                format!("if (({name}) <= 0) {{"),
                format!("    long long x = (long long)({name});"),
                "    printf(\"value: %lld\\n\", x);".to_string(),
                "} else {".to_string(),
                format!("    unsigned long long x = (unsigned long long)({name});"),
                "    printf(\"value: %llu\\n\", x);".to_string(),
                "}".to_string(),
            ],
        }
    }

    fn build_result(&self, info: &[(String, i128)]) -> Result<ConfigValue, CompilationError> {
        let get = |key: &str| {
            info.iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| *v)
                .unwrap_or_else(|| panic!("KeyError: {key}"))
        };
        match self {
            CConfigEntry::SimpleType(name) => {
                let size = get("size");
                let unsigned = get("unsigned") != 0;
                // `ctype_hint` is `rffi.INT` (4-byte signed). Keep it when
                // `(size, sign)` matches; otherwise `fixup_ctype`.
                let ctype = if size == 4 && !unsigned {
                    IntType {
                        signed: true,
                        bits: 32,
                    }
                } else {
                    fixup_ctype(name, size, unsigned)?
                };
                Ok(ConfigValue::Type(ctype))
            }
            CConfigEntry::SizeOf(_) => Ok(ConfigValue::Size(get("size") as u64)),
            CConfigEntry::ConstantInteger(_) => {
                Ok(ConfigValue::Int(expose_value_as_rpython(get("value"))?))
            }
        }
    }
}

/// `rffi_platform.py` `integer_class` / `fixup_ctype`. Members are
/// `SIGNEDCHAR`, `UCHAR`, `CHAR`, `SHORT`, `USHORT`, `INT`, `UINT`,
/// `INT_real`, `UINT_real`, `LONG`, `ULONG`, `LONGLONG`, `ULONGLONG`
/// (1, 2, 4, 8 bytes; no 128-bit type).
fn fixup_ctype(name: &str, size: i128, unsigned: bool) -> Result<IntType, CompilationError> {
    let bits = match size {
        1 => 8,
        2 => 16,
        4 => 32,
        8 => 64,
        _ => {
            return Err(CompilationError(format!(
                "conflicting field type for {name}"
            )));
        }
    };
    Ok(IntType {
        signed: !unsigned,
        bits,
    })
}

/// `rffi_platform.py` `expose_value_as_rpython`.
fn expose_value_as_rpython(value: i128) -> Result<i128, CompilationError> {
    if value >= i64::MIN as i128 && value <= u64::MAX as i128 {
        Ok(value)
    } else {
        Err(CompilationError(format!(
            "value {value} does not fit into any RPython integer type"
        )))
    }
}

/// `rffi_platform.py` `CConfigSingleEntry`.
pub enum CConfigSingleEntry {
    /// `rffi_platform.py` `Has`.
    Has(String),
}

impl CConfigSingleEntry {
    fn question(
        &self,
        ask_gcc: &mut dyn FnMut(&str) -> Result<(), CompilationError>,
    ) -> ConfigValue {
        match self {
            CConfigSingleEntry::Has(name) => {
                ConfigValue::Bool(ask_gcc(&format!("(void){name};")).is_ok())
            }
        }
    }
}

/// `rffi_platform.py` `configure` result values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConfigValue {
    Type(IntType),
    Size(u64),
    Bool(bool),
    Int(i128),
}

/// `rffi_platform.py` `CConfig`. Later attributes of the same name replace
/// earlier ones; `configure` sorts each list as `dir()` does.
pub struct CConfig {
    pub compilation_info: ExternalCompilationInfo,
    pub entries: Vec<(String, CConfigEntry)>,
    pub single_entries: Vec<(String, CConfigSingleEntry)>,
}

/// Collapse duplicate names (later wins) and sort as `dir()` does.
fn unique_sorted_by_name<T>(items: &[(String, T)]) -> Vec<(String, &T)> {
    let mut out: Vec<(String, &T)> = Vec::new();
    for (name, value) in items {
        if let Some(pos) = out.iter().position(|(n, _)| n == name) {
            out[pos].1 = value;
        } else {
            out.push((name.clone(), value));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// `rffi_platform.py` `configure`. The result has one value per name, in
/// `dir()` order.
pub fn configure(cconfig: &CConfig) -> Result<Vec<(String, ConfigValue)>, CompilationError> {
    let entries = unique_sorted_by_name(&cconfig.entries);
    let single_entries = unique_sorted_by_name(&cconfig.single_entries);
    let mut res = Vec::new();
    if !entries.is_empty() {
        let list: Vec<&CConfigEntry> = entries.iter().map(|(_, e)| *e).collect();
        let results = configure_entries(&list, &cconfig.compilation_info)?;
        for ((key, _), result) in entries.iter().zip(results) {
            res.push((key.clone(), result));
        }
    }
    for (key, entry) in &single_entries {
        let mut writer = CWriter::new(cconfig.compilation_info.clone());
        writer.write_header();
        let mut writer = Some(writer);
        let result = entry.question(&mut |q| writer.take().unwrap().ask_gcc(q));
        if let Some(pos) = res.iter().position(|(k, _)| k == key) {
            res[pos].1 = result;
        } else {
            res.push((key.clone(), result));
        }
    }
    Ok(res)
}

/// `rffi_platform.py` `configure_entries`. The `ConfigResult` cache has no
/// second reader here, so call `build_result` directly.
fn configure_entries(
    entries: &[&CConfigEntry],
    eci: &ExternalCompilationInfo,
) -> Result<Vec<ConfigValue>, CompilationError> {
    let mut writer = CWriter::new(eci.clone());
    writer.write_header();
    for (i, entry) in entries.iter().enumerate() {
        writer.write_entry(&i.to_string(), entry);
    }
    writer.start_main();
    for i in 0..entries.len() {
        writer.write_entry_main(&i.to_string());
    }
    writer.close()?;
    let infolist = run_example_code(&writer.path, eci)?;
    assert_eq!(infolist.len(), entries.len());
    entries
        .iter()
        .zip(infolist)
        .map(|(entry, info)| entry.build_result(&info))
        .collect()
}

/// `rffi_platform.py` `_CWriter`.
struct CWriter {
    path: PathBuf,
    text: String,
    eci: ExternalCompilationInfo,
}

impl CWriter {
    fn new(eci: ExternalCompilationInfo) -> Self {
        CWriter {
            path: uniquefilepath(),
            text: String::new(),
            eci,
        }
    }

    fn write_header(&mut self) {
        self.eci.write_c_header(&mut self.text);
        self.text.push_str(C_HEADER);
        self.text.push('\n');
        self.text.push('\n');
    }

    fn write_entry(&mut self, key: &str, entry: &CConfigEntry) {
        self.text
            .push_str(&format!("void dump_section_{key}(void) {{\n"));
        for line in entry.prepare_code() {
            let line = if !line.is_empty() && !line.starts_with('#') {
                format!("\t{line}")
            } else {
                line
            };
            self.text.push_str(&line);
            self.text.push('\n');
        }
        self.text.push_str("}\n");
        self.text.push('\n');
    }

    fn write_entry_main(&mut self, key: &str) {
        self.text
            .push_str(&format!("\tprintf(\"-+- {key}\\n\");\n"));
        self.text.push_str(&format!("\tdump_section_{key}();\n"));
        self.text.push_str("\tprintf(\"---\\n\");\n");
    }

    fn start_main(&mut self) {
        self.text.push_str("int main(int argc, char *argv[]) {\n");
    }

    fn close(&mut self) -> Result<(), CompilationError> {
        self.text.push_str("\treturn 0;\n");
        self.text.push_str("}\n");
        fs::write(&self.path, &self.text).map_err(|err| CompilationError(err.to_string()))?;
        Ok(())
    }

    fn ask_gcc(mut self, question: &str) -> Result<(), CompilationError> {
        self.start_main();
        self.text.push_str(question);
        self.text.push('\n');
        self.close()?;
        try_compile_cache(std::slice::from_ref(&self.path), &self.eci)
    }
}

/// Upstream `udir`.
fn probe_dir() -> PathBuf {
    let dir = match std::env::var_os("OUT_DIR") {
        Some(out) => PathBuf::from(out).join("rffi_platform"),
        None => {
            std::env::temp_dir().join(format!("majit-eci-rffi-platform-{}", std::process::id()))
        }
    };
    if let Err(err) = fs::create_dir_all(&dir) {
        panic!("{}: {err}", dir.display());
    }
    dir
}

/// `rffi_platform.py` `uniquefilepath(LAST=[0])`.
fn uniquefilepath() -> PathBuf {
    static LAST: AtomicUsize = AtomicUsize::new(0);
    let n = LAST.fetch_add(1, Ordering::Relaxed);
    probe_dir().join(format!("platcheck_{n}.c"))
}

/// Triple `cc::Build` needs when `TARGET` is unset (unit tests outside a
/// build script).
fn host_triple() -> String {
    let arch = std::env::consts::ARCH;
    match std::env::consts::OS {
        "macos" => format!("{arch}-apple-darwin"),
        "linux" => format!("{arch}-unknown-linux-gnu"),
        "windows" => format!("{arch}-pc-windows-msvc"),
        os => format!("{arch}-unknown-{os}"),
    }
}

fn target_triple() -> String {
    std::env::var("TARGET").unwrap_or_else(|_| host_triple())
}

/// `CARGO_TARGET_<TRIPLE>_RUNNER`: triple uppercased, `-` and `.` to `_`.
fn cargo_target_runner_var(triple: &str) -> String {
    let ident = triple.replace(['-', '.'], "_").to_ascii_uppercase();
    format!("CARGO_TARGET_{ident}_RUNNER")
}

fn compiler_build(use_cpp_linker: bool) -> cc::Build {
    let mut build = cc::Build::new();
    build.cargo_metadata(false).cargo_warnings(false);
    if use_cpp_linker {
        build.cpp(true);
    }
    if std::env::var_os("TARGET").is_none() {
        let host = host_triple();
        build.target(&host).host(&host).opt_level(0);
    }
    build
}

/// `Platform._all_cfiles`: `files` plus `eci.separate_module_files`, each
/// `std::path::absolute`, first occurrence kept.
fn all_cfiles(
    files: &[PathBuf],
    eci: &ExternalCompilationInfo,
) -> Result<Vec<PathBuf>, CompilationError> {
    let mut result = Vec::new();
    for cfile in files
        .iter()
        .cloned()
        .chain(eci.separate_module_files.iter().map(PathBuf::from))
    {
        let cfile = std::path::absolute(&cfile).map_err(|err| CompilationError(err.to_string()))?;
        if !result.contains(&cfile) {
            result.push(cfile);
        }
    }
    Ok(result)
}

/// `Platform.compile`.
fn build_executable(
    files: &[PathBuf],
    eci: &ExternalCompilationInfo,
) -> Result<PathBuf, CompilationError> {
    let eci = eci.convert_sources_to_files(&probe_dir());
    let all_files = all_cfiles(files, &eci)?;
    let windows = match std::env::var("CARGO_CFG_TARGET_OS") {
        Ok(os) => os == "windows",
        Err(_) => cfg!(windows),
    };
    let exe = if windows {
        files[0].with_extension("exe")
    } else {
        files[0].with_extension("")
    };
    // `_finish_linking`: `eci.use_cpp_linker` selects the C++ driver for the
    // whole command (`cc::Build` with `.cpp(true)`).
    let compiler = compiler_build(eci.use_cpp_linker).get_compiler();
    let mut cmd = compiler.to_command();
    if compiler.is_like_msvc() {
        cmd.arg("/nologo");
        for dir in &eci.include_dirs {
            cmd.arg(format!("/I{dir}"));
        }
        for flag in &eci.compile_extra {
            cmd.arg(flag);
        }
        for file in &all_files {
            cmd.arg(file);
        }
        cmd.arg(format!("/Fe{}", exe.display()));
        cmd.arg(format!("/Fo{}\\", probe_dir().display()));
        cmd.arg("/link");
        for d in &eci.library_dirs {
            cmd.arg(format!("/LIBPATH:{d}"));
        }
        for file in &eci.link_files {
            cmd.arg(file);
        }
        for flag in &eci.link_extra {
            cmd.arg(flag);
        }
        for lib in &eci.libraries {
            cmd.arg(format!("{lib}.lib"));
        }
    } else {
        for dir in &eci.include_dirs {
            cmd.arg(format!("-I{dir}"));
        }
        for flag in &eci.compile_extra {
            cmd.arg(flag);
        }
        for file in &all_files {
            cmd.arg(file);
        }
        cmd.arg("-o");
        cmd.arg(&exe);
        // `Platform._link_args_from_eci` / posix: `library_dirs`,
        // `link_files`, `link_extra`, `libraries` (no `link_flags` /
        // `extra_libs`). Darwin then appends `frameworks`.
        for d in &eci.library_dirs {
            cmd.arg(format!("-L{d}"));
        }
        for file in &eci.link_files {
            cmd.arg(file);
        }
        for flag in &eci.link_extra {
            cmd.arg(flag);
        }
        for lib in &eci.libraries {
            cmd.arg(format!("-l{lib}"));
        }
        for f in &eci.frameworks {
            cmd.arg("-framework");
            cmd.arg(f);
        }
    }
    let output = cmd
        .output()
        .map_err(|err| CompilationError(err.to_string()))?;
    if !output.status.success() {
        let mut msg = String::from_utf8_lossy(&output.stderr).into_owned();
        msg.push_str(&String::from_utf8_lossy(&output.stdout));
        return Err(CompilationError(msg));
    }
    Ok(exe)
}

/// `gcc_cache.cache_file_path` root: `$OUT_DIR` `ancestors().nth(2)` /
/// `rffi_platform_cache/<cachename>`, else `probe_dir()` / `cache/<cachename>`.
fn cache_dir_for(cachename: &str) -> Result<PathBuf, CompilationError> {
    let root = match std::env::var_os("OUT_DIR") {
        Some(out) => Path::new(&out)
            .ancestors()
            .nth(2)
            .map(|p| p.join("rffi_platform_cache"))
            .unwrap_or_else(|| PathBuf::from(&out).join("rffi_platform_cache")),
        None => probe_dir().join("cache"),
    };
    let dir = root.join(cachename);
    fs::create_dir_all(&dir).map_err(|err| CompilationError(err.to_string()))?;
    Ok(dir)
}

fn platform_key(use_cpp_linker: bool) -> String {
    let compiler = compiler_build(use_cpp_linker).get_compiler();
    let path = compiler.path().display();
    let args: Vec<String> = compiler
        .args()
        .iter()
        .map(|a| a.to_string_lossy().into_owned())
        .collect();
    let target = target_triple();
    let runner_var = cargo_target_runner_var(&target);
    let runner = std::env::var(&runner_var).unwrap_or_default();
    format!("{path} {args:?} TARGET={target} {runner_var}={runner}")
}

fn cache_key_text(
    files: &[PathBuf],
    eci: &ExternalCompilationInfo,
) -> Result<String, CompilationError> {
    let converted = eci.convert_sources_to_files(&probe_dir());
    let mut text = String::new();
    for path in all_cfiles(files, &converted)? {
        let contents =
            fs::read_to_string(&path).map_err(|err| CompilationError(err.to_string()))?;
        text.push_str(&contents);
        text.push('\n');
    }
    text.push_str(&format!("{eci:?}\n"));
    text.push_str(&platform_key(eci.use_cpp_linker));
    Ok(text)
}

fn cache_file_path(
    files: &[PathBuf],
    eci: &ExternalCompilationInfo,
    cachename: &str,
) -> Result<(PathBuf, String), CompilationError> {
    let key_text = cache_key_text(files, eci)?;
    let mut hasher = DefaultHasher::new();
    hasher.write(key_text.as_bytes());
    let hash = format!("{:016x}", hasher.finish());
    Ok((cache_dir_for(cachename)?.join(hash), key_text))
}

fn read_cache(path: &Path, key_text: &str) -> Option<String> {
    let data = fs::read_to_string(path).ok()?;
    let (stored_key, payload) = data.split_once(&format!("\n{CACHE_SEP}\n"))?;
    if stored_key == key_text {
        Some(payload.to_string())
    } else {
        None
    }
}

/// `gcc_cache.try_atomic_write`.
fn try_atomic_write(path: &Path, data: &str) -> Result<(), CompilationError> {
    let tmp = PathBuf::from(format!("{}~{}", path.display(), std::process::id()));
    fs::write(&tmp, data).map_err(|err| CompilationError(err.to_string()))?;
    if fs::rename(&tmp, path).is_err() {
        let _ = fs::remove_file(&tmp);
    }
    Ok(())
}

fn write_cache(path: &Path, key_text: &str, payload: &str) -> Result<(), CompilationError> {
    try_atomic_write(path, &format!("{key_text}\n{CACHE_SEP}\n{payload}"))
}

/// `gcc_cache.build_executable_cache`.
fn build_executable_cache(
    files: &[PathBuf],
    eci: &ExternalCompilationInfo,
) -> Result<String, CompilationError> {
    let (path, key_text) = cache_file_path(files, eci, "build_executable_cache")?;
    if let Some(payload) = read_cache(&path, &key_text) {
        return Ok(payload);
    }
    let exe = build_executable(files, eci)?;
    let (stdout, stderr) = execute(&exe)?;
    if !stderr.is_empty() {
        let _ = io::stderr().write_all(stderr.as_bytes());
    }
    if stderr.is_empty() {
        write_cache(&path, &key_text, &stdout)?;
    }
    Ok(stdout)
}

/// `gcc_cache.try_compile_cache`.
fn try_compile_cache(
    files: &[PathBuf],
    eci: &ExternalCompilationInfo,
) -> Result<(), CompilationError> {
    let (path, key_text) = cache_file_path(files, eci, "try_compile_cache")?;
    if let Some(payload) = read_cache(&path, &key_text) {
        if payload == "True" {
            return Ok(());
        }
        return Err(CompilationError(payload));
    }
    match build_executable(files, eci) {
        Ok(_) => {
            write_cache(&path, &key_text, "True")?;
            Ok(())
        }
        Err(err) => {
            if err.0 != "True" {
                write_cache(&path, &key_text, &err.0)?;
            }
            Err(err)
        }
    }
}

/// `Platform.execute`. `CARGO_TARGET_<TRIPLE>_RUNNER` is the per-target
/// wrapper (Cargo's equivalent of scratchbox `sb2`). Spawn errors are
/// `CompilationError`. `build_executable_cache` does not look at the
/// return code: stderr is written through when non-empty, and stdout is
/// parsed.
fn execute(exe: &Path) -> Result<(String, String), CompilationError> {
    let triple = target_triple();
    let runner_var = cargo_target_runner_var(&triple);
    let output = match std::env::var(&runner_var) {
        Ok(runner) => {
            let mut parts = runner.split_whitespace();
            match parts.next() {
                Some(prog) => {
                    let mut cmd = Command::new(prog);
                    for arg in parts {
                        cmd.arg(arg);
                    }
                    cmd.arg(exe);
                    cmd.output()
                }
                None => Command::new(exe).output(),
            }
        }
        Err(_) => Command::new(exe).output(),
    }
    .map_err(|err| CompilationError(err.to_string()))?;
    Ok((
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    ))
}

/// `rffi_platform.py` `run_example_code`. Compiles through
/// `build_executable_cache` and runs the program with `Platform.execute`
/// (`CARGO_TARGET_<TRIPLE>_RUNNER` when set).
fn run_example_code(
    filepath: &Path,
    eci: &ExternalCompilationInfo,
) -> Result<Vec<Vec<(String, i128)>>, CompilationError> {
    let output = build_executable_cache(&[filepath.to_path_buf()], eci)?;
    parse_example_output(&output)
}

/// `rffi_platform.py` `run_example_code` parsing loop.
fn parse_example_output(output: &str) -> Result<Vec<Vec<(String, i128)>>, CompilationError> {
    if !output.starts_with("-+- ") {
        return Err(CompilationError("run_example_code failed!".into()));
    }
    let mut sections = Vec::new();
    let mut section: Option<Vec<(String, i128)>> = None;
    for line in output.lines() {
        let line = line.trim();
        if line.starts_with("-+- ") {
            section = Some(Vec::new());
        } else if line == "---" {
            let done = section.take().expect("---");
            sections.push(done);
        } else if !line.is_empty() {
            let section = section.as_mut().expect("line");
            let (key, value) = line.split_once(": ").unwrap_or_else(|| panic!("{line}"));
            let value: i128 = value.parse().unwrap_or_else(|_| panic!("{value}"));
            if let Some((_, slot)) = section.iter_mut().find(|(k, _)| k == key) {
                *slot = value;
            } else {
                section.push((key.to_string(), value));
            }
        }
    }
    Ok(sections)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rffi_platform_entries() {
        assert_eq!(
            CConfigEntry::SimpleType("mode_t".into()).prepare_code(),
            vec![
                "typedef mode_t platcheck_t;".to_string(),
                String::new(),
                "platcheck_t x;".to_string(),
                "dump(\"size\",  sizeof(platcheck_t));".to_string(),
                "x = 0; x = ~x;".to_string(),
                "dump(\"unsigned\", x > 0);".to_string(),
            ]
        );
        assert_eq!(
            CConfigEntry::SizeOf("long double".into()).prepare_code(),
            vec!["dump(\"size\",  sizeof(long double));".to_string()]
        );
        assert_eq!(
            CConfigEntry::ConstantInteger("EOF".into()).prepare_code(),
            vec![
                "if ((EOF) <= 0) {".to_string(),
                "    long long x = (long long)(EOF);".to_string(),
                "    printf(\"value: %lld\\n\", x);".to_string(),
                "} else {".to_string(),
                "    unsigned long long x = (unsigned long long)(EOF);".to_string(),
                "    printf(\"value: %llu\\n\", x);".to_string(),
                "}".to_string(),
            ]
        );

        let mut writer = CWriter::new(ExternalCompilationInfo::default());
        writer.write_header();
        writer.write_entry("0", &CConfigEntry::SizeOf("int".into()));
        writer.write_entry("1", &CConfigEntry::SizeOf("long".into()));
        writer.start_main();
        writer.write_entry_main("0");
        writer.write_entry_main("1");
        writer.close().unwrap();
        assert!(
            writer.text.contains("void dump_section_0(void) {"),
            "{}",
            writer.text
        );
        assert!(
            writer.text.contains("\tdump_section_1();"),
            "{}",
            writer.text
        );
        assert!(writer.text.ends_with("\treturn 0;\n}\n"), "{}", writer.text);

        let parsed =
            parse_example_output("-+- 0\nsize: 4\nunsigned: 1\n---\n-+- 1\nsize: 16\n---\n")
                .unwrap();
        assert_eq!(
            parsed,
            vec![
                vec![("size".into(), 4), ("unsigned".into(), 1)],
                vec![("size".into(), 16)],
            ]
        );
        let parsed_dup = parse_example_output("-+- 0\nsize: 4\nsize: 8\n---\n").unwrap();
        assert_eq!(parsed_dup, vec![vec![("size".into(), 8)]]);
        assert!(parse_example_output("oops").is_err());

        let st = CConfigEntry::SimpleType("mode_t".into());
        assert_eq!(
            st.build_result(&[("size".into(), 4), ("unsigned".into(), 1)])
                .unwrap(),
            ConfigValue::Type(IntType {
                signed: false,
                bits: 32
            })
        );
        assert_eq!(
            st.build_result(&[("size".into(), 8), ("unsigned".into(), 0)])
                .unwrap(),
            ConfigValue::Type(IntType {
                signed: true,
                bits: 64
            })
        );
        assert!(
            st.build_result(&[("size".into(), 16), ("unsigned".into(), 0)])
                .is_err()
        );

        #[cfg(unix)]
        {
            let cconfig = CConfig {
                compilation_info: ExternalCompilationInfo {
                    includes: vec!["stdio.h".into()],
                    ..ExternalCompilationInfo::default()
                },
                entries: vec![
                    ("SIZE".into(), CConfigEntry::SizeOf("int".into())),
                    (
                        "TYPE".into(),
                        CConfigEntry::SimpleType("unsigned short".into()),
                    ),
                    ("EOF".into(), CConfigEntry::ConstantInteger("EOF".into())),
                ],
                single_entries: vec![
                    ("HAS".into(), CConfigSingleEntry::Has("printf".into())),
                    (
                        "NO".into(),
                        CConfigSingleEntry::Has("no_such_function_xyz".into()),
                    ),
                ],
            };
            let results = configure(&cconfig).unwrap();
            assert_eq!(
                results,
                vec![
                    ("EOF".into(), ConfigValue::Int(-1)),
                    ("SIZE".into(), ConfigValue::Size(4)),
                    (
                        "TYPE".into(),
                        ConfigValue::Type(IntType {
                            signed: false,
                            bits: 16
                        })
                    ),
                    ("HAS".into(), ConfigValue::Bool(true)),
                    ("NO".into(), ConfigValue::Bool(false)),
                ]
            );
        }
    }
}
