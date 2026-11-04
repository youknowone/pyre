//! `rffi_platform.py`: ask the target's C compiler from a build script.

use std::fmt;
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
        }
    }

    fn build_result(&self, info: &[(String, i128)]) -> ConfigValue {
        let get = |key: &str| {
            info.iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| *v)
                .unwrap_or_else(|| panic!("KeyError: {key}"))
        };
        match self {
            CConfigEntry::SimpleType(_) => {
                let size = get("size");
                let unsigned = get("unsigned");
                ConfigValue::Type(IntType {
                    signed: unsigned == 0,
                    bits: (size as u32) * 8,
                })
            }
            CConfigEntry::SizeOf(_) => ConfigValue::Size(get("size") as u64),
        }
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
}

/// Upstream `CConfig` class attributes; declaration order replaces `dir()` order.
pub struct CConfig {
    pub compilation_info: ExternalCompilationInfo,
    pub entries: Vec<(String, CConfigEntry)>,
    pub single_entries: Vec<(String, CConfigSingleEntry)>,
}

/// `rffi_platform.py` `configure`.
pub fn configure(cconfig: &CConfig) -> Result<Vec<(String, ConfigValue)>, CompilationError> {
    let mut res = Vec::new();
    if !cconfig.entries.is_empty() {
        let entries: Vec<&CConfigEntry> = cconfig.entries.iter().map(|(_, e)| e).collect();
        let results = configure_entries(&entries, &cconfig.compilation_info)?;
        for ((key, _), result) in cconfig.entries.iter().zip(results) {
            res.push((key.clone(), result));
        }
    }
    for (key, entry) in &cconfig.single_entries {
        let mut writer = CWriter::new(cconfig.compilation_info.clone());
        writer.write_header();
        let mut writer = Some(writer);
        let result = entry.question(&mut |q| writer.take().unwrap().ask_gcc(q));
        res.push((key.clone(), result));
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
    Ok(entries
        .iter()
        .zip(infolist)
        .map(|(entry, info)| entry.build_result(&info))
        .collect())
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
        std::fs::write(&self.path, &self.text).map_err(|err| CompilationError(err.to_string()))?;
        Ok(())
    }

    fn ask_gcc(mut self, question: &str) -> Result<(), CompilationError> {
        self.start_main();
        self.text.push_str(question);
        self.text.push('\n');
        self.close()?;
        try_compile(std::slice::from_ref(&self.path), &self.eci)
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
    if let Err(err) = std::fs::create_dir_all(&dir) {
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

/// `gcc_cache.build_executable_cache` without the cache.
fn build_executable(
    files: &[PathBuf],
    eci: &ExternalCompilationInfo,
) -> Result<PathBuf, CompilationError> {
    let eci = eci.convert_sources_to_files(&probe_dir());
    let mut all_files: Vec<PathBuf> = files.to_vec();
    for extra in &eci.separate_module_files {
        all_files.push(PathBuf::from(extra));
    }
    let windows = match std::env::var("CARGO_CFG_TARGET_OS") {
        Ok(os) => os == "windows",
        Err(_) => cfg!(windows),
    };
    let exe = if windows {
        files[0].with_extension("exe")
    } else {
        files[0].with_extension("")
    };
    let mut build = cc::Build::new();
    build.cargo_metadata(false).cargo_warnings(false);
    if std::env::var_os("TARGET").is_none() {
        let host = host_triple();
        build.target(&host).host(&host).opt_level(0);
    }
    let compiler = build.get_compiler();
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
        for lib in &eci.libraries {
            cmd.arg(format!("{lib}.lib"));
        }
        for flag in &eci.link_extra {
            cmd.arg(flag);
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
        for d in &eci.library_dirs {
            cmd.arg(format!("-L{d}"));
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

/// `gcc_cache.try_compile_cache`.
fn try_compile(files: &[PathBuf], eci: &ExternalCompilationInfo) -> Result<(), CompilationError> {
    build_executable(files, eci).map(|_| ())
}

/// `rffi_platform.py` `run_example_code`.
fn run_example_code(
    filepath: &Path,
    eci: &ExternalCompilationInfo,
) -> Result<Vec<Vec<(String, i128)>>, CompilationError> {
    let exe = build_executable(&[filepath.to_path_buf()], eci)?;
    match (std::env::var("HOST"), std::env::var("TARGET")) {
        (Ok(host), Ok(target)) if host != target => {
            return Err(CompilationError(format!(
                "cannot run a {target} probe on {host}"
            )));
        }
        _ => {}
    }
    let output = Command::new(&exe)
        .output()
        .map_err(|err| CompilationError(err.to_string()))?;
    if !output.status.success() {
        let mut msg = String::from_utf8_lossy(&output.stderr).into_owned();
        msg.push_str(&String::from_utf8_lossy(&output.stdout));
        return Err(CompilationError(msg));
    }
    parse_example_output(&String::from_utf8_lossy(&output.stdout))
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
            section.push((key.to_string(), value));
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
        assert!(parse_example_output("oops").is_err());

        let st = CConfigEntry::SimpleType("mode_t".into());
        assert_eq!(
            st.build_result(&[("size".into(), 4), ("unsigned".into(), 1)]),
            ConfigValue::Type(IntType {
                signed: false,
                bits: 32
            })
        );
        assert_eq!(
            st.build_result(&[("size".into(), 8), ("unsigned".into(), 0)]),
            ConfigValue::Type(IntType {
                signed: true,
                bits: 64
            })
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
