//! RPython `rpython/rtyper/tool/rffi_platform.py`.
//!
//! Upstream discovers C platform facts by generating a C program, compiling
//! it, running it, and turning each dumped section back into lltype/rffi
//! objects. The current Rust translator does not yet own the
//! `gcc_cache`/platform compiler execution leaf, so this module ports the
//! public entry names, C-source generation, result parsing, and deterministic
//! result builders while returning an explicit error at the compiler-probe
//! boundary.

#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]

use std::fmt;
use std::path::PathBuf;

use crate::flowspace::model::ConstValue;
use crate::translator::c::genc::ExternalCompilationInfo;
use crate::translator::rtyper::lltypesystem::lltype::{LowLevelType, StructType};
use crate::translator::rtyper::lltypesystem::rffi;

pub type Info = majit_ir::VecMap<String, i128>;
pub type ConfigureResult = majit_ir::VecMap<String, ConfiguredValue>;

/// RPython `eci_from_header(c_header_source, include_dirs=None,
/// libraries=None)`.
pub fn eci_from_header(
    c_header_source: impl Into<String>,
    include_dirs: Option<Vec<PathBuf>>,
    libraries: Option<Vec<String>>,
) -> ExternalCompilationInfo {
    ExternalCompilationInfo {
        post_include_bits: vec![c_header_source.into()],
        include_dirs: include_dirs.unwrap_or_default(),
        libraries: libraries.unwrap_or_default(),
        ..ExternalCompilationInfo::default()
    }
}

pub fn getstruct(
    name: &str,
    c_header_source: &str,
    interesting_fields: Vec<(String, LowLevelType)>,
) -> Result<ConfiguredValue, RffiPlatformError> {
    let eci = eci_from_header(c_header_source, None, None);
    let entries = vec![Entry::Struct(Struct::new(name, interesting_fields))];
    configure_entries(&entries, &eci, false).map(|mut values| values.remove(0))
}

pub fn getsimpletype(
    name: &str,
    c_header_source: &str,
    ctype_hint: LowLevelType,
) -> Result<ConfiguredValue, RffiPlatformError> {
    let eci = eci_from_header(c_header_source, None, None);
    let entries = vec![Entry::SimpleType(SimpleType::new(name, ctype_hint))];
    configure_entries(&entries, &eci, false).map(|mut values| values.remove(0))
}

pub fn getconstantinteger(
    name: &str,
    c_header_source: &str,
) -> Result<ConfiguredValue, RffiPlatformError> {
    let eci = eci_from_header(c_header_source, None, None);
    let entries = vec![Entry::ConstantInteger(ConstantInteger::new(name))];
    configure_entries(&entries, &eci, false).map(|mut values| values.remove(0))
}

pub fn getdefined(macro_name: &str, c_header_source: &str) -> Result<bool, RffiPlatformError> {
    let eci = eci_from_header(c_header_source, None, None);
    let entries = vec![Entry::Defined(Defined::new(macro_name))];
    configure_entries(&entries, &eci, false).and_then(|mut values| match values.remove(0) {
        ConfiguredValue::Bool(value) => Ok(value),
        other => Err(RffiPlatformError::type_mismatch("Defined", other)),
    })
}

pub fn getdefineddouble(
    macro_name: &str,
    c_header_source: &str,
) -> Result<ConfiguredValue, RffiPlatformError> {
    let eci = eci_from_header(c_header_source, None, None);
    let entries = vec![Entry::DefinedConstantDouble(DefinedConstantDouble::new(
        macro_name,
    ))];
    configure_entries(&entries, &eci, false).map(|mut values| values.remove(0))
}

pub fn getdefinedinteger(
    macro_name: &str,
    c_header_source: &str,
) -> Result<ConfiguredValue, RffiPlatformError> {
    let eci = eci_from_header(c_header_source, None, None);
    let entries = vec![Entry::DefinedConstantInteger(DefinedConstantInteger::new(
        macro_name,
    ))];
    configure_entries(&entries, &eci, false).map(|mut values| values.remove(0))
}

pub fn getdefinedstring(
    macro_name: &str,
    c_header_source: &str,
) -> Result<ConfiguredValue, RffiPlatformError> {
    let eci = eci_from_header(c_header_source, None, None);
    let entries = vec![Entry::DefinedConstantString(DefinedConstantString::new(
        macro_name, None,
    ))];
    configure_entries(&entries, &eci, false).map(|mut values| values.remove(0))
}

pub fn getintegerfunctionresult(
    function: &str,
    args: Option<Vec<String>>,
    c_header_source: &str,
    includes: &[String],
) -> Result<ConfiguredValue, RffiPlatformError> {
    let mut eci = eci_from_header(c_header_source, None, None);
    eci.includes = includes.to_vec();
    let entries = vec![Entry::IntegerFunctionResult(IntegerFunctionResult::new(
        function, args,
    ))];
    configure_entries(&entries, &eci, false).map(|mut values| values.remove(0))
}

pub fn has(
    name: &str,
    c_header_source: &str,
    include_dirs: Option<Vec<PathBuf>>,
    libraries: Option<Vec<String>>,
) -> Result<bool, RffiPlatformError> {
    let eci = eci_from_header(c_header_source, include_dirs, libraries);
    let entry = Has::new(name);
    entry.question(&eci)
}

pub fn verify_eci(_eci: &ExternalCompilationInfo) -> Result<(), RffiPlatformError> {
    Err(RffiPlatformError::compiler_probe_unavailable("verify_eci"))
}

pub fn checkcompiles(
    expression: &str,
    c_header_source: &str,
    include_dirs: Option<Vec<PathBuf>>,
) -> Result<bool, RffiPlatformError> {
    has(expression, c_header_source, include_dirs, None)
}

pub fn sizeof(
    name: &str,
    eci: &ExternalCompilationInfo,
) -> Result<ConfiguredValue, RffiPlatformError> {
    let entries = vec![Entry::SizeOf(SizeOf::new(name))];
    configure_entries(&entries, eci, false).map(|mut values| values.remove(0))
}

pub fn memory_alignment() -> Result<usize, RffiPlatformError> {
    #[cfg(windows)]
    {
        Ok(usize::BITS as usize / 8)
    }
    #[cfg(not(windows))]
    {
        Err(RffiPlatformError::compiler_probe_unavailable(
            "memory_alignment",
        ))
    }
}

/// Rust carrier for an upstream `CConfig` class.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CConfig {
    pub _compilation_info_: ExternalCompilationInfo,
    pub entries: Vec<(String, Entry)>,
    pub single_entries: Vec<(String, SingleEntry)>,
}

pub fn configure(
    CConfig: &CConfig,
    ignore_errors: bool,
) -> Result<ConfigureResult, RffiPlatformError> {
    let mut res = ConfigureResult::new();
    if !CConfig.entries.is_empty() {
        let entries = CConfig
            .entries
            .iter()
            .map(|(_, entry)| entry.clone())
            .collect::<Vec<_>>();
        let results = configure_entries(&entries, &CConfig._compilation_info_, ignore_errors)?;
        for ((name, _), result) in CConfig.entries.iter().zip(results.into_iter()) {
            res.insert(name.clone(), result);
        }
    }
    for (name, entry) in &CConfig.single_entries {
        res.insert(name.clone(), entry.question(&CConfig._compilation_info_)?);
    }
    Ok(res)
}

/// RPython `configure_entries(entries, eci, ignore_errors=False)`.
pub fn configure_entries(
    entries: &[Entry],
    eci: &ExternalCompilationInfo,
    ignore_errors: bool,
) -> Result<Vec<ConfiguredValue>, RffiPlatformError> {
    let source = make_configure_source(entries, eci);
    Err(RffiPlatformError::CompilerProbeUnavailable {
        function: "configure_entries",
        source,
        ignore_errors,
    })
}

pub fn build_configure_entries_from_info(
    entries: &[Entry],
    eci: &ExternalCompilationInfo,
    infolist: &[Info],
) -> Result<Vec<ConfiguredValue>, RffiPlatformError> {
    if entries.len() != infolist.len() {
        return Err(RffiPlatformError::MalformedOutput(format!(
            "rffi_platform.py: expected {} sections, got {}",
            entries.len(),
            infolist.len()
        )));
    }
    let config_result = ConfigResult::new(eci.clone(), infolist.to_vec());
    entries
        .iter()
        .enumerate()
        .map(|(index, entry)| config_result.get_entry_result(index, entry))
        .collect()
}

pub fn make_configure_source(entries: &[Entry], eci: &ExternalCompilationInfo) -> String {
    let mut writer = _CWriter::new(eci.clone());
    writer.write_header();
    for (index, entry) in entries.iter().enumerate() {
        writer.write_entry(&index.to_string(), entry);
    }
    writer.start_main();
    for index in 0..entries.len() {
        writer.write_entry_main(&index.to_string());
    }
    writer.close();
    writer.finish()
}

/// RPython `ConfigResult`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigResult {
    pub eci: ExternalCompilationInfo,
    pub info: Vec<Info>,
}

impl ConfigResult {
    pub fn new(eci: ExternalCompilationInfo, info: Vec<Info>) -> Self {
        Self { eci, info }
    }

    pub fn get_entry_result(
        &self,
        index: usize,
        entry: &Entry,
    ) -> Result<ConfiguredValue, RffiPlatformError> {
        let info = self.info.get(index).ok_or_else(|| {
            RffiPlatformError::MalformedOutput(format!(
                "rffi_platform.py: missing result section {index}"
            ))
        })?;
        entry.build_result(info, self)
    }
}

/// RPython `_CWriter`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct _CWriter {
    source: String,
    pub eci: ExternalCompilationInfo,
}

impl _CWriter {
    pub fn new(eci: ExternalCompilationInfo) -> Self {
        Self {
            source: String::new(),
            eci,
        }
    }

    pub fn write_header(&mut self) {
        self.write_c_header();
        self.source.push_str(C_HEADER);
        self.source.push('\n');
    }

    pub fn write_entry(&mut self, key: &str, entry: &Entry) {
        self.source
            .push_str(&format!("void dump_section_{key}(void) {{\n"));
        for line in entry.prepare_code() {
            if !line.is_empty() && !line.starts_with('#') {
                self.source.push('\t');
            }
            self.source.push_str(&line);
            self.source.push('\n');
        }
        self.source.push_str("}\n\n");
    }

    pub fn write_entry_main(&mut self, key: &str) {
        self.source
            .push_str(&format!("\tprintf(\"-+- {key}\\n\");\n"));
        self.source.push_str(&format!("\tdump_section_{key}();\n"));
        self.source.push_str("\tprintf(\"---\\n\");\n");
    }

    pub fn start_main(&mut self) {
        self.source.push_str("int main(int argc, char *argv[]) {\n");
    }

    pub fn close(&mut self) {
        self.source.push_str("\treturn 0;\n}\n");
    }

    pub fn ask_gcc(&mut self, question: &str) -> Result<(), RffiPlatformError> {
        self.start_main();
        self.source.push_str(question);
        self.source.push('\n');
        self.close();
        Err(RffiPlatformError::CompilerProbeUnavailable {
            function: "_CWriter.ask_gcc",
            source: self.source.clone(),
            ignore_errors: false,
        })
    }

    pub fn finish(self) -> String {
        self.source
    }

    fn write_c_header(&mut self) {
        for line in &self.eci.pre_include_bits {
            self.source.push_str(line);
            self.source.push('\n');
        }
        for include in &self.eci.includes {
            self.source.push_str(&format!("#include <{include}>\n"));
        }
        for line in &self.eci.post_include_bits {
            self.source.push_str(line);
            self.source.push('\n');
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Entry {
    Struct(Struct),
    SimpleType(SimpleType),
    ConstantInteger(ConstantInteger),
    IntegerFunctionResult(IntegerFunctionResult),
    DefinedConstantInteger(DefinedConstantInteger),
    DefinedConstantDouble(DefinedConstantDouble),
    DefinedConstantString(DefinedConstantString),
    Defined(Defined),
    SizeOf(SizeOf),
    PaddingDropFieldLookup(_PaddingDropFieldLookup),
}

impl Entry {
    pub fn prepare_code(&self) -> Vec<String> {
        match self {
            Entry::Struct(entry) => entry.prepare_code(),
            Entry::SimpleType(entry) => entry.prepare_code(),
            Entry::ConstantInteger(entry) => entry.prepare_code(),
            Entry::IntegerFunctionResult(entry) => entry.prepare_code(),
            Entry::DefinedConstantInteger(entry) => entry.prepare_code(),
            Entry::DefinedConstantDouble(entry) => entry.prepare_code(),
            Entry::DefinedConstantString(entry) => entry.prepare_code(),
            Entry::Defined(entry) => entry.prepare_code(),
            Entry::SizeOf(entry) => entry.prepare_code(),
            Entry::PaddingDropFieldLookup(entry) => entry.prepare_code(),
        }
    }

    pub fn build_result(
        &self,
        info: &Info,
        config_result: &ConfigResult,
    ) -> Result<ConfiguredValue, RffiPlatformError> {
        match self {
            Entry::Struct(entry) => entry.build_result(info, config_result),
            Entry::SimpleType(entry) => entry.build_result(info, config_result),
            Entry::ConstantInteger(entry) => entry.build_result(info, config_result),
            Entry::IntegerFunctionResult(entry) => entry.build_result(info, config_result),
            Entry::DefinedConstantInteger(entry) => entry.build_result(info, config_result),
            Entry::DefinedConstantDouble(entry) => entry.build_result(info, config_result),
            Entry::DefinedConstantString(entry) => entry.build_result(info, config_result),
            Entry::Defined(entry) => entry.build_result(info, config_result),
            Entry::SizeOf(entry) => entry.build_result(info, config_result),
            Entry::PaddingDropFieldLookup(entry) => entry.build_result(info, config_result),
        }
    }
}

/// RPython `class CConfigEntry`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CConfigEntry;

/// RPython `class Struct(CConfigEntry)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Struct {
    pub name: String,
    pub interesting_fields: Vec<(String, LowLevelType)>,
    pub ifdef: Option<String>,
    pub adtmeths: majit_ir::VecMap<String, String>,
}

impl Struct {
    pub fn new(name: impl Into<String>, interesting_fields: Vec<(String, LowLevelType)>) -> Self {
        Self {
            name: name.into(),
            interesting_fields,
            ifdef: None,
            adtmeths: majit_ir::VecMap::new(),
        }
    }

    pub fn prepare_code(&self) -> Vec<String> {
        let mut code = Vec::new();
        if let Some(ifdef) = &self.ifdef {
            code.push(format!("#ifdef {ifdef}"));
        }
        code.push(format!("typedef {} platcheck_t;", self.name));
        code.push("typedef struct {".to_string());
        code.push("    char c;".to_string());
        code.push("    platcheck_t s;".to_string());
        code.push("} platcheck2_t;".to_string());
        code.push(String::new());
        code.push("platcheck_t s;".to_string());
        if self.ifdef.is_some() {
            code.push("dump(\"defined\", 1);".to_string());
        }
        code.push("dump(\"align\", offsetof(platcheck2_t, s));".to_string());
        code.push("dump(\"size\",  sizeof(platcheck_t));".to_string());
        for (fieldname, fieldtype) in &self.interesting_fields {
            code.push(format!(
                "dump(\"fldofs {fieldname}\", offsetof(platcheck_t, {fieldname}));"
            ));
            code.push(format!(
                "dump(\"fldsize {fieldname}\",   sizeof(s.{fieldname}));"
            ));
            if integer_class(fieldtype) {
                code.push(format!(
                    "s.{fieldname} = 0; s.{fieldname} = ~s.{fieldname};"
                ));
                code.push(format!(
                    "dump(\"fldunsigned {fieldname}\", s.{fieldname} > 0);"
                ));
            }
        }
        if self.ifdef.is_some() {
            code.push("#else".to_string());
            code.push("dump(\"defined\", 0);".to_string());
            code.push("#endif".to_string());
        }
        code
    }

    pub fn build_result(
        &self,
        info: &Info,
        config_result: &ConfigResult,
    ) -> Result<ConfiguredValue, RffiPlatformError> {
        if self.ifdef.is_some() && info_get_bool(info, "defined")? == Some(false) {
            return Ok(ConfiguredValue::None);
        }
        let size = info_get_usize(info, "size")?;
        let align = info_get_usize(info, "align")?;
        let mut fields = Vec::new();
        let mut layout: Vec<Option<Field>> = vec![None; size];
        for (fieldname, fieldtype) in &self.interesting_fields {
            let offset = info_get_usize(info, &format!("fldofs {fieldname}"))?;
            let size = info_get_usize(info, &format!("fldsize {fieldname}"))?;
            let unsigned = info_get_bool(info, &format!("fldunsigned {fieldname}"))?;
            let ctype = if is_array_nolength(fieldtype) {
                fieldtype.clone()
            } else {
                fixup_ctype(fieldtype, fieldname, size, unsigned.unwrap_or(false))
            };
            fields.push(StructField {
                name: fieldname.clone(),
                ctype: ctype.clone(),
                offset,
                size,
                unsigned,
            });
            layout_addfield(&mut layout, offset, size, ctype, fieldname);
        }
        let mut padfields = Vec::new();
        let mut pad_index = 0usize;
        for index in 0..layout.len() {
            if layout[index].is_some() {
                continue;
            }
            let name = format!("_pad{pad_index}");
            layout_addfield(&mut layout, index, 1, LowLevelType::Unsigned, &name);
            padfields.push(format!("c_{name}"));
            pad_index += 1;
        }
        let mut seen = Vec::new();
        let mut layout_fields = Vec::new();
        let mut fieldoffsets = Vec::new();
        for (offset, cell) in layout.into_iter().enumerate() {
            let Some(cell) = cell else {
                continue;
            };
            if seen.iter().any(|name| name == &cell.name) {
                continue;
            }
            fieldoffsets.push(offset);
            seen.push(cell.name.clone());
            layout_fields.push((cell.name, cell.ctype));
        }
        let mut lltype_name = self.name.clone();
        let typedef = if let Some(rest) = lltype_name.strip_prefix("struct ") {
            lltype_name = rest.to_string();
            false
        } else {
            true
        };
        let mut hints = vec![
            ("align".into(), ConstValue::Int(align as i64)),
            ("size".into(), ConstValue::Int(size as i64)),
            (
                "fieldoffsets".into(),
                ConstValue::Tuple(
                    fieldoffsets
                        .iter()
                        .map(|offset| ConstValue::Int(*offset as i64))
                        .collect(),
                ),
            ),
            (
                "padding".into(),
                ConstValue::Tuple(
                    padfields
                        .iter()
                        .map(|name| ConstValue::byte_str(name.as_bytes()))
                        .collect(),
                ),
            ),
            (
                "get_padding_drop".into(),
                ConstValue::byte_str(format!("PaddingDrop({})", self.name)),
            ),
        ];
        if typedef {
            hints.push(("typedef".into(), ConstValue::Bool(true)));
        }
        let lowlevel = rffi::CStruct_with_hints(&lltype_name, layout_fields, hints);
        Ok(ConfiguredValue::Struct(StructResult {
            name: self.name.clone(),
            align,
            size,
            fields,
            typedef,
            fieldoffsets,
            padding: padfields.clone(),
            padding_drop: PaddingDrop::new(
                self.name.clone(),
                seen.into_iter().map(|name| format!("c_{name}")).collect(),
                padfields.clone(),
                config_result.eci.clone(),
            ),
            lowlevel,
        }))
    }
}

/// RPython `class SimpleType(CConfigEntry)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SimpleType {
    pub name: String,
    pub ctype_hint: LowLevelType,
    pub ifdef: Option<String>,
}

impl SimpleType {
    pub fn new(name: impl Into<String>, ctype_hint: LowLevelType) -> Self {
        Self {
            name: name.into(),
            ctype_hint,
            ifdef: None,
        }
    }

    pub fn prepare_code(&self) -> Vec<String> {
        let mut code = Vec::new();
        if let Some(ifdef) = &self.ifdef {
            code.push(format!("#ifdef {ifdef}"));
        }
        code.push(format!("typedef {} platcheck_t;", self.name));
        code.push(String::new());
        code.push("platcheck_t x;".to_string());
        if self.ifdef.is_some() {
            code.push("dump(\"defined\", 1);".to_string());
        }
        code.push("dump(\"size\",  sizeof(platcheck_t));".to_string());
        if integer_class(&self.ctype_hint) {
            code.push("x = 0; x = ~x;".to_string());
            code.push("dump(\"unsigned\", x > 0);".to_string());
        }
        if self.ifdef.is_some() {
            code.push("#else".to_string());
            code.push("dump(\"defined\", 0);".to_string());
            code.push("#endif".to_string());
        }
        code
    }

    pub fn build_result(
        &self,
        info: &Info,
        _config_result: &ConfigResult,
    ) -> Result<ConfiguredValue, RffiPlatformError> {
        if self.ifdef.is_some() && info_get_bool(info, "defined")? == Some(false) {
            return Ok(ConfiguredValue::None);
        }
        let size = info_get_usize(info, "size")?;
        let unsigned = info_get_bool(info, "unsigned")?;
        let ctype = fixup_ctype(
            &self.ctype_hint,
            &self.name,
            size,
            unsigned.unwrap_or(false),
        );
        Ok(ConfiguredValue::SimpleType(SimpleTypeResult {
            name: self.name.clone(),
            ctype,
            size,
            unsigned,
        }))
    }
}

/// RPython `class ConstantInteger(CConfigEntry)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConstantInteger {
    pub name: String,
}

impl ConstantInteger {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    pub fn prepare_code(&self) -> Vec<String> {
        vec![
            format!("if (({}) <= 0) {{", self.name),
            format!("    long long x = (long long)({});", self.name),
            "    printf(\"value: %lld\\n\", x);".to_string(),
            "} else {".to_string(),
            format!(
                "    unsigned long long x = (unsigned long long)({});",
                self.name
            ),
            "    printf(\"value: %llu\\n\", x);".to_string(),
            "}".to_string(),
        ]
    }

    pub fn build_result(
        &self,
        info: &Info,
        _config_result: &ConfigResult,
    ) -> Result<ConfiguredValue, RffiPlatformError> {
        Ok(ConfiguredValue::Integer(expose_value_as_rpython(info_get(
            info, "value",
        )?)))
    }
}

/// RPython `class IntegerFunctionResult(CConfigEntry)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntegerFunctionResult {
    pub name: String,
    pub args: Vec<String>,
}

impl IntegerFunctionResult {
    pub fn new(name: impl Into<String>, args: Option<Vec<String>>) -> Self {
        Self {
            name: name.into(),
            args: args.unwrap_or_default(),
        }
    }

    pub fn prepare_code(&self) -> Vec<String> {
        let args = self.args.join(", ");
        vec![
            format!("long int result = {}({args});", self.name),
            "if ((result) <= 0) {".to_string(),
            "    long long x = (long long)(result);".to_string(),
            "    printf(\"value: %lld\\n\", x);".to_string(),
            "} else {".to_string(),
            "    unsigned long long x = (unsigned long long)(result);".to_string(),
            "    printf(\"value: %llu\\n\", x);".to_string(),
            "}".to_string(),
        ]
    }

    pub fn build_result(
        &self,
        info: &Info,
        _config_result: &ConfigResult,
    ) -> Result<ConfiguredValue, RffiPlatformError> {
        Ok(ConfiguredValue::Integer(expose_value_as_rpython(info_get(
            info, "value",
        )?)))
    }
}

/// RPython `class DefinedConstantInteger(CConfigEntry)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DefinedConstantInteger {
    pub name: String,
    pub macro_: String,
}

impl DefinedConstantInteger {
    pub fn new(macro_name: impl Into<String>) -> Self {
        let macro_ = macro_name.into();
        Self {
            name: macro_.clone(),
            macro_,
        }
    }

    pub fn prepare_code(&self) -> Vec<String> {
        let macro_ = &self.macro_;
        vec![
            format!("#ifdef {macro_}"),
            "dump(\"defined\", 1);".to_string(),
            format!("if (({macro_}) <= 0) {{"),
            format!("    long long x = (long long)({macro_});"),
            "    printf(\"value: %lld\\n\", x);".to_string(),
            "} else {".to_string(),
            format!("    unsigned long long x = (unsigned long long)({macro_});"),
            "    printf(\"value: %llu\\n\", x);".to_string(),
            "}".to_string(),
            "#else".to_string(),
            "dump(\"defined\", 0);".to_string(),
            "#endif".to_string(),
        ]
    }

    pub fn build_result(
        &self,
        info: &Info,
        _config_result: &ConfigResult,
    ) -> Result<ConfiguredValue, RffiPlatformError> {
        if info_get_bool(info, "defined")? == Some(true) {
            Ok(ConfiguredValue::Integer(expose_value_as_rpython(info_get(
                info, "value",
            )?)))
        } else {
            Ok(ConfiguredValue::None)
        }
    }
}

/// RPython `class DefinedConstantDouble(CConfigEntry)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DefinedConstantDouble {
    pub name: String,
    pub macro_: String,
}

impl DefinedConstantDouble {
    pub fn new(macro_name: impl Into<String>) -> Self {
        let macro_ = macro_name.into();
        Self {
            name: macro_.clone(),
            macro_,
        }
    }

    pub fn prepare_code(&self) -> Vec<String> {
        let macro_ = &self.macro_;
        vec![
            format!("#ifdef {macro_}"),
            "int i;".to_string(),
            format!("double x = {macro_};"),
            "unsigned char *p = (unsigned char *)&x;".to_string(),
            "dump(\"defined\", 1);".to_string(),
            "for (i = 0; i < 8; i++) {".to_string(),
            " printf(\"value_%d: %d\\n\", i, p[i]);".to_string(),
            "}".to_string(),
            "#else".to_string(),
            "dump(\"defined\", 0);".to_string(),
            "#endif".to_string(),
        ]
    }

    pub fn build_result(
        &self,
        info: &Info,
        _config_result: &ConfigResult,
    ) -> Result<ConfiguredValue, RffiPlatformError> {
        if info_get_bool(info, "defined")? != Some(true) {
            return Ok(ConfiguredValue::None);
        }
        let mut data = [0_u8; 8];
        for (i, byte) in data.iter_mut().enumerate() {
            *byte = info_get(info, &format!("value_{i}"))? as u8;
        }
        Ok(ConfiguredValue::Double(f64::from_ne_bytes(data)))
    }
}

/// RPython `class DefinedConstantString(CConfigEntry)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DefinedConstantString {
    pub macro_: String,
    pub name: String,
}

impl DefinedConstantString {
    pub fn new(macro_name: impl Into<String>, name: Option<String>) -> Self {
        let macro_ = macro_name.into();
        Self {
            name: name.unwrap_or_else(|| macro_.clone()),
            macro_,
        }
    }

    pub fn prepare_code(&self) -> Vec<String> {
        vec![
            format!("#ifdef {}", self.macro_),
            "int i;".to_string(),
            format!("const char *p = {};", self.name),
            "dump(\"defined\", 1);".to_string(),
            "for (i = 0; p[i] != 0; i++ ) {".to_string(),
            "  printf(\"value_%d: %d\\n\", i, (int)(unsigned char)p[i]);".to_string(),
            "}".to_string(),
            "#else".to_string(),
            "dump(\"defined\", 0);".to_string(),
            "#endif".to_string(),
        ]
    }

    pub fn build_result(
        &self,
        info: &Info,
        _config_result: &ConfigResult,
    ) -> Result<ConfiguredValue, RffiPlatformError> {
        if info_get_bool(info, "defined")? != Some(true) {
            return Ok(ConfiguredValue::None);
        }
        let mut bytes = Vec::new();
        for index in 0.. {
            match info.get(&format!("value_{index}")) {
                Some(value) => bytes.push(*value as u8),
                None => break,
            }
        }
        String::from_utf8(bytes)
            .map(ConfiguredValue::String)
            .map_err(|e| RffiPlatformError::MalformedOutput(e.to_string()))
    }
}

/// RPython `class Defined(CConfigEntry)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Defined {
    pub macro_: String,
    pub name: String,
}

impl Defined {
    pub fn new(macro_name: impl Into<String>) -> Self {
        let macro_ = macro_name.into();
        Self {
            name: macro_.clone(),
            macro_,
        }
    }

    pub fn prepare_code(&self) -> Vec<String> {
        vec![
            format!("#ifdef {}", self.macro_),
            "dump(\"defined\", 1);".to_string(),
            "#else".to_string(),
            "dump(\"defined\", 0);".to_string(),
            "#endif".to_string(),
        ]
    }

    pub fn build_result(
        &self,
        info: &Info,
        _config_result: &ConfigResult,
    ) -> Result<ConfiguredValue, RffiPlatformError> {
        Ok(ConfiguredValue::Bool(
            info_get_bool(info, "defined")?.unwrap_or(false),
        ))
    }
}

/// RPython `class CConfigSingleEntry`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CConfigSingleEntry;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SingleEntry {
    Has(Has),
    Works(Works),
}

impl SingleEntry {
    pub fn question(
        &self,
        eci: &ExternalCompilationInfo,
    ) -> Result<ConfiguredValue, RffiPlatformError> {
        match self {
            SingleEntry::Has(entry) => entry.question(eci).map(ConfiguredValue::Bool),
            SingleEntry::Works(entry) => entry.question(eci).map(|()| ConfiguredValue::None),
        }
    }
}

/// RPython `class Has(CConfigSingleEntry)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Has {
    pub name: String,
}

impl Has {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    pub fn question(&self, eci: &ExternalCompilationInfo) -> Result<bool, RffiPlatformError> {
        let mut writer = _CWriter::new(eci.clone());
        writer.write_header();
        match writer.ask_gcc(&format!("(void){};", self.name)) {
            Ok(()) => Ok(true),
            Err(RffiPlatformError::CompilationError(_)) => Ok(false),
            Err(e) => Err(e),
        }
    }
}

/// RPython `class Works(CConfigSingleEntry)`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Works;

impl Works {
    pub fn question(&self, eci: &ExternalCompilationInfo) -> Result<(), RffiPlatformError> {
        let mut writer = _CWriter::new(eci.clone());
        writer.write_header();
        writer.ask_gcc("")
    }
}

/// RPython `class SizeOf(CConfigEntry)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SizeOf {
    pub name: String,
}

impl SizeOf {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    pub fn prepare_code(&self) -> Vec<String> {
        vec![format!("dump(\"size\",  sizeof({}));", self.name)]
    }

    pub fn build_result(
        &self,
        info: &Info,
        _config_result: &ConfigResult,
    ) -> Result<ConfiguredValue, RffiPlatformError> {
        Ok(ConfiguredValue::Integer(RPythonInteger::Int(
            info_get(info, "size")? as i64,
        )))
    }
}

/// RPython `class PaddingDrop`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PaddingDrop {
    pub cache: Option<Vec<String>>,
    pub name: String,
    pub allfields: Vec<String>,
    pub padfields: Vec<String>,
    pub eci: ExternalCompilationInfo,
}

impl PaddingDrop {
    pub fn new(
        name: impl Into<String>,
        allfields: Vec<String>,
        padfields: Vec<String>,
        eci: ExternalCompilationInfo,
    ) -> Self {
        Self {
            cache: None,
            name: name.into(),
            allfields,
            padfields,
            eci,
        }
    }
}

/// RPython `class _PaddingDropFieldLookup(CConfigEntry)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct _PaddingDropFieldLookup {
    pub name: String,
    pub staticfields: Vec<Option<String>>,
    pub fieldname: String,
}

impl _PaddingDropFieldLookup {
    pub fn new(
        name: impl Into<String>,
        staticfields: Vec<Option<String>>,
        fieldname: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            staticfields,
            fieldname: fieldname.into(),
        }
    }

    pub fn prepare_code(&self) -> Vec<String> {
        let mut code = vec![
            format!("typedef {} platcheck_t;", self.name),
            "static platcheck_t s = {".to_string(),
        ];
        for (index, typ) in self.staticfields.iter().enumerate() {
            let value = if index == self.staticfields.len() - 1 {
                -1
            } else {
                0
            };
            if let Some(typ) = typ {
                code.push(format!("\t({typ}){value},"));
            } else {
                code.push(format!("\t{value},"));
            }
        }
        code.push("};".to_string());
        assert!(self.fieldname.starts_with("c_"));
        code.push(format!(
            "dump(\"fieldlookup\", s.{} != 0);",
            &self.fieldname[2..]
        ));
        code
    }

    pub fn build_result(
        &self,
        info: &Info,
        _config_result: &ConfigResult,
    ) -> Result<ConfiguredValue, RffiPlatformError> {
        Ok(ConfiguredValue::Bool(
            info_get_bool(info, "fieldlookup")?.unwrap_or(false),
        ))
    }
}

pub fn uniquefilepath(index: usize) -> PathBuf {
    PathBuf::from(format!("platcheck_{index}.c"))
}

/// RPython `class Field`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Field {
    pub name: String,
    pub ctype: LowLevelType,
}

impl Field {
    pub fn new(name: impl Into<String>, ctype: LowLevelType) -> Self {
        Self {
            name: name.into(),
            ctype,
        }
    }
}

pub fn is_array_nolength(TYPE: &LowLevelType) -> bool {
    matches!(TYPE, LowLevelType::Array(array) if array._hints.get(&"nolength".to_string()).is_some())
}

fn integer_class(TYPE: &LowLevelType) -> bool {
    matches!(
        TYPE,
        LowLevelType::Signed
            | LowLevelType::Unsigned
            | LowLevelType::SignedLongLong
            | LowLevelType::SignedLongLongLong
            | LowLevelType::UnsignedLongLong
            | LowLevelType::UnsignedLongLongLong
            | LowLevelType::Bool
            | LowLevelType::Char
            | LowLevelType::UniChar
    )
}

fn size_and_sign(TYPE: &LowLevelType) -> Option<(usize, bool)> {
    match TYPE {
        LowLevelType::Signed => Some((std::mem::size_of::<isize>(), false)),
        LowLevelType::Unsigned => Some((std::mem::size_of::<usize>(), true)),
        LowLevelType::SignedLongLong => Some((std::mem::size_of::<i64>(), false)),
        LowLevelType::SignedLongLongLong => Some((std::mem::size_of::<i128>(), false)),
        LowLevelType::UnsignedLongLong => Some((std::mem::size_of::<u64>(), true)),
        LowLevelType::UnsignedLongLongLong => Some((std::mem::size_of::<u128>(), true)),
        LowLevelType::Bool => Some((std::mem::size_of::<bool>(), true)),
        LowLevelType::Char => Some((std::mem::size_of::<u8>(), false)),
        LowLevelType::UniChar => Some((std::mem::size_of::<char>(), true)),
        LowLevelType::Float => Some((std::mem::size_of::<f64>(), false)),
        LowLevelType::SingleFloat => Some((std::mem::size_of::<f32>(), false)),
        LowLevelType::LongFloat => Some((std::mem::size_of::<f64>(), false)),
        LowLevelType::Ptr(_) | LowLevelType::Address => Some((std::mem::size_of::<usize>(), true)),
        _ => None,
    }
}

fn fixup_ctype(TYPE: &LowLevelType, _fieldname: &str, size: usize, unsigned: bool) -> LowLevelType {
    if is_array_nolength(TYPE) {
        return TYPE.clone();
    }
    if size_and_sign(TYPE) == Some((size, unsigned)) {
        return TYPE.clone();
    }
    if size == 1 && unsigned {
        return LowLevelType::Unsigned;
    }
    if size == 1 && !unsigned {
        return LowLevelType::Char;
    }
    if size == std::mem::size_of::<usize>() && unsigned {
        return LowLevelType::Unsigned;
    }
    if size == std::mem::size_of::<isize>() && !unsigned {
        return LowLevelType::Signed;
    }
    if size == std::mem::size_of::<u64>() && unsigned {
        return LowLevelType::UnsignedLongLong;
    }
    if size == std::mem::size_of::<i64>() && !unsigned {
        return LowLevelType::SignedLongLong;
    }
    for candidate in [
        LowLevelType::Signed,
        LowLevelType::Unsigned,
        LowLevelType::SignedLongLong,
        LowLevelType::UnsignedLongLong,
        LowLevelType::SignedLongLongLong,
        LowLevelType::UnsignedLongLongLong,
        LowLevelType::Char,
        LowLevelType::UniChar,
    ] {
        if size_and_sign(&candidate) == Some((size, unsigned)) {
            return candidate;
        }
    }
    TYPE.clone()
}

fn layout_addfield(
    layout: &mut [Option<Field>],
    offset: usize,
    size: usize,
    ctype: LowLevelType,
    name: &str,
) {
    for index in offset..offset.saturating_add(size) {
        if index < layout.len() {
            layout[index] = Some(Field::new(name, ctype.clone()));
        }
    }
}

pub fn expose_value_as_rpython(value: i128) -> RPythonInteger {
    if value >= i32::MIN as i128 && value <= i32::MAX as i128 {
        RPythonInteger::Int(value as i64)
    } else if value >= 0 && value <= u32::MAX as i128 {
        RPythonInteger::UInt(value as u64)
    } else if value >= i64::MIN as i128 && value <= i64::MAX as i128 {
        RPythonInteger::LongLong(value as i64)
    } else {
        RPythonInteger::ULongLong(value as u64)
    }
}

pub const C_HEADER: &str = r#"
#include <stdio.h>
#include <stddef.h>   /* for offsetof() */

void dump(char* key, int value) {
    printf("%s: %d\n", key, value);
}
"#;

pub fn run_example_code(
    filepath: &str,
    eci: &ExternalCompilationInfo,
    ignore_errors: bool,
) -> Result<Vec<Info>, RffiPlatformError> {
    let source = format!("{}{}", filepath, make_configure_source(&[], eci));
    Err(RffiPlatformError::CompilerProbeUnavailable {
        function: "run_example_code",
        source,
        ignore_errors,
    })
}

pub fn parse_example_code_output(output: &str) -> Result<Vec<Info>, RffiPlatformError> {
    if !output.starts_with("-+- ") {
        return Err(RffiPlatformError::MalformedOutput(
            "run_example_code failed: output does not start with section marker".to_string(),
        ));
    }
    let mut sections = Vec::new();
    let mut section: Option<Info> = None;
    for raw_line in output.lines() {
        let line = raw_line.trim();
        if line.starts_with("-+- ") {
            section = Some(Info::new());
        } else if line == "---" {
            let current = section.take().ok_or_else(|| {
                RffiPlatformError::MalformedOutput(
                    "rffi_platform.py: section end without section".to_string(),
                )
            })?;
            sections.push(current);
        } else if !line.is_empty() {
            let current = section.as_mut().ok_or_else(|| {
                RffiPlatformError::MalformedOutput(
                    "rffi_platform.py: key/value outside section".to_string(),
                )
            })?;
            let (key, value) = line.split_once(": ").ok_or_else(|| {
                RffiPlatformError::MalformedOutput(format!(
                    "rffi_platform.py: malformed output line {line:?}"
                ))
            })?;
            current.insert(
                key.to_string(),
                value.parse::<i128>().map_err(|e| {
                    RffiPlatformError::MalformedOutput(format!(
                        "rffi_platform.py: invalid integer {value:?}: {e}"
                    ))
                })?,
            );
        }
    }
    Ok(sections)
}

pub fn configure_external_library(
    name: &str,
    eci: &ExternalCompilationInfo,
    _configurations: &[ExternalLibraryConfiguration],
    _symbol: Option<&str>,
) -> Result<ExternalCompilationInfo, RffiPlatformError> {
    verify_eci(eci).map(|()| eci.clone()).map_err(|e| match e {
        RffiPlatformError::CompilerProbeUnavailable { .. } => {
            RffiPlatformError::compiler_probe_unavailable("configure_external_library")
        }
        other => {
            RffiPlatformError::CompilationError(format!("Library {name} is not installed: {other}"))
        }
    })
}

pub fn configure_boehm(
    _platform: Option<String>,
) -> Result<ExternalCompilationInfo, RffiPlatformError> {
    Err(RffiPlatformError::compiler_probe_unavailable(
        "configure_boehm",
    ))
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExternalLibraryConfiguration {
    pub prefix: Option<String>,
    pub include_dir: Option<PathBuf>,
    pub library_dir: Option<PathBuf>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ConfiguredValue {
    None,
    Bool(bool),
    Integer(RPythonInteger),
    Double(f64),
    String(String),
    SimpleType(SimpleTypeResult),
    Struct(StructResult),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RPythonInteger {
    Int(i64),
    UInt(u64),
    LongLong(i64),
    ULongLong(u64),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SimpleTypeResult {
    pub name: String,
    pub ctype: LowLevelType,
    pub size: usize,
    pub unsigned: Option<bool>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct StructResult {
    pub name: String,
    pub align: usize,
    pub size: usize,
    pub fields: Vec<StructField>,
    pub typedef: bool,
    pub fieldoffsets: Vec<usize>,
    pub padding: Vec<String>,
    pub padding_drop: PaddingDrop,
    pub lowlevel: StructType,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructField {
    pub name: String,
    pub ctype: LowLevelType,
    pub offset: usize,
    pub size: usize,
    pub unsigned: Option<bool>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RffiPlatformError {
    CompilerProbeUnavailable {
        function: &'static str,
        source: String,
        ignore_errors: bool,
    },
    CompilationError(String),
    MalformedOutput(String),
    TypeMismatch(String),
}

impl RffiPlatformError {
    fn compiler_probe_unavailable(function: &'static str) -> Self {
        Self::CompilerProbeUnavailable {
            function,
            source: String::new(),
            ignore_errors: false,
        }
    }

    fn type_mismatch(expected: &str, got: ConfiguredValue) -> Self {
        Self::TypeMismatch(format!("expected {expected}, got {got:?}"))
    }
}

impl fmt::Display for RffiPlatformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RffiPlatformError::CompilerProbeUnavailable { function, .. } => {
                write!(
                    f,
                    "rffi_platform.py: {function} needs the C compiler probe leaf"
                )
            }
            RffiPlatformError::CompilationError(message)
            | RffiPlatformError::MalformedOutput(message)
            | RffiPlatformError::TypeMismatch(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for RffiPlatformError {}

fn info_get(info: &Info, key: &str) -> Result<i128, RffiPlatformError> {
    info.get(&key.to_string()).copied().ok_or_else(|| {
        RffiPlatformError::MalformedOutput(format!("rffi_platform.py: missing key {key:?}"))
    })
}

fn info_get_usize(info: &Info, key: &str) -> Result<usize, RffiPlatformError> {
    usize::try_from(info_get(info, key)?).map_err(|e| {
        RffiPlatformError::MalformedOutput(format!(
            "rffi_platform.py: key {key:?} is not usize: {e}"
        ))
    })
}

fn info_get_bool(info: &Info, key: &str) -> Result<Option<bool>, RffiPlatformError> {
    match info.get(&key.to_string()).copied() {
        Some(0) => Ok(Some(false)),
        Some(1) => Ok(Some(true)),
        Some(other) => Err(RffiPlatformError::MalformedOutput(format!(
            "rffi_platform.py: key {key:?} is not bool: {other}"
        ))),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eci_from_header_matches_upstream_fields() {
        let eci = eci_from_header(
            "#include <sys/types.h>",
            Some(vec![PathBuf::from("/tmp/include")]),
            Some(vec!["m".to_string()]),
        );

        assert_eq!(eci.post_include_bits, vec!["#include <sys/types.h>"]);
        assert_eq!(eci.include_dirs, vec![PathBuf::from("/tmp/include")]);
        assert_eq!(eci.libraries, vec!["m"]);
    }

    #[test]
    fn parse_example_code_output_sections() {
        let output = "-+- 0\nsize: 8\nunsigned: 1\n---\n-+- 1\nvalue: -3\n---\n";
        let sections = parse_example_code_output(output).unwrap();

        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].get(&"size".to_string()), Some(&8));
        assert_eq!(sections[0].get(&"unsigned".to_string()), Some(&1));
        assert_eq!(sections[1].get(&"value".to_string()), Some(&-3));
    }

    #[test]
    fn defined_constant_integer_builds_none_when_undefined() {
        let entry = Entry::DefinedConstantInteger(DefinedConstantInteger::new("HAVE_FOO"));
        let mut info = Info::new();
        info.insert("defined".to_string(), 0);
        let result = build_configure_entries_from_info(
            &[entry],
            &ExternalCompilationInfo::default(),
            &[info],
        )
        .unwrap();

        assert_eq!(result, vec![ConfiguredValue::None]);
    }

    #[test]
    fn make_configure_source_keeps_section_shape() {
        let entry = Entry::SizeOf(SizeOf::new("int"));
        let source = make_configure_source(&[entry], &ExternalCompilationInfo::default());

        assert!(source.contains("void dump_section_0(void)"));
        assert!(source.contains("dump(\"size\",  sizeof(int));"));
        assert!(source.contains("printf(\"-+- 0\\n\");"));
        assert!(source.contains("dump_section_0();"));
    }

    #[test]
    fn configure_entries_reports_unavailable_probe_without_success() {
        let entry = Entry::Defined(Defined::new("HAVE_FOO"));
        let err = configure_entries(&[entry], &ExternalCompilationInfo::default(), true)
            .expect_err("compiler leaf is not ported");

        match err {
            RffiPlatformError::CompilerProbeUnavailable {
                function,
                source,
                ignore_errors,
            } => {
                assert_eq!(function, "configure_entries");
                assert!(source.contains("#ifdef HAVE_FOO"));
                assert!(ignore_errors);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn struct_prepare_code_emits_field_unsigned_probes() {
        let entry = Struct::new(
            "struct demo",
            vec![
                ("count".to_string(), LowLevelType::Signed),
                ("name".to_string(), (*rffi::CCHARP).clone()),
            ],
        );
        let code = entry.prepare_code().join("\n");

        assert!(code.contains("dump(\"fldofs count\", offsetof(platcheck_t, count));"));
        assert!(code.contains("dump(\"fldsize count\",   sizeof(s.count));"));
        assert!(code.contains("s.count = 0; s.count = ~s.count;"));
        assert!(code.contains("dump(\"fldunsigned count\", s.count > 0);"));
        assert!(!code.contains("fldunsigned name"));
    }

    #[test]
    fn simple_type_prepare_code_emits_unsigned_probe() {
        let entry = SimpleType::new("uintptr_t", LowLevelType::Signed);
        let code = entry.prepare_code().join("\n");

        assert!(code.contains("dump(\"size\",  sizeof(platcheck_t));"));
        assert!(code.contains("x = 0; x = ~x;"));
        assert!(code.contains("dump(\"unsigned\", x > 0);"));
    }

    #[test]
    fn struct_build_result_reconstructs_padding_and_offsets() {
        let entry = Entry::Struct(Struct::new(
            "struct demo",
            vec![
                ("a".to_string(), LowLevelType::Signed),
                ("b".to_string(), LowLevelType::Char),
            ],
        ));
        let mut info = Info::new();
        info.insert("align".to_string(), 4);
        info.insert("size".to_string(), 8);
        info.insert("fldofs a".to_string(), 0);
        info.insert("fldsize a".to_string(), 4);
        info.insert("fldunsigned a".to_string(), 0);
        info.insert("fldofs b".to_string(), 6);
        info.insert("fldsize b".to_string(), 1);
        info.insert("fldunsigned b".to_string(), 0);

        let mut results = build_configure_entries_from_info(
            &[entry],
            &ExternalCompilationInfo::default(),
            &[info],
        )
        .unwrap();
        let ConfiguredValue::Struct(result) = results.remove(0) else {
            panic!("expected struct result");
        };

        assert_eq!(result.align, 4);
        assert_eq!(result.size, 8);
        assert_eq!(result.fieldoffsets, vec![0, 4, 5, 6, 7]);
        assert_eq!(
            result.padding,
            vec![
                "c__pad0".to_string(),
                "c__pad1".to_string(),
                "c__pad2".to_string()
            ]
        );
        assert_eq!(result.fields[0].ctype, LowLevelType::Signed);
        assert_eq!(result.fields[1].ctype, LowLevelType::Char);
        assert!(result.lowlevel._hints.get("fieldoffsets").is_some());
        assert_eq!(result.padding_drop.padfields, result.padding);
    }

    #[test]
    fn simple_type_build_result_uses_size_and_sign_fixup() {
        let entry = Entry::SimpleType(SimpleType::new("uintptr_t", LowLevelType::Signed));
        let mut info = Info::new();
        info.insert("size".to_string(), std::mem::size_of::<usize>() as i128);
        info.insert("unsigned".to_string(), 1);

        let mut results = build_configure_entries_from_info(
            &[entry],
            &ExternalCompilationInfo::default(),
            &[info],
        )
        .unwrap();
        let ConfiguredValue::SimpleType(result) = results.remove(0) else {
            panic!("expected simple type result");
        };

        assert_eq!(result.ctype, LowLevelType::Unsigned);
        assert_eq!(result.unsigned, Some(true));
    }
}
