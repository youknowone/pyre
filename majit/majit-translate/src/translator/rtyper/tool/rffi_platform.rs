//! RPython `rpython/rtyper/tool/rffi_platform.py` parity helpers.
//!
//! The upstream module compiles generated C probes to discover platform
//! layout.  The compile/execute leaf is still owned by the deferred C
//! backend, but the surrounding API shape is ported here: C probe source
//! generation, section-output parsing, config-entry result construction,
//! `eci_from_header`, and small helper policies.

#![allow(non_camel_case_types, non_upper_case_globals)]

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};

use crate::translator::c::genc::ExternalCompilationInfo;

pub fn eci_from_header(
    c_header_source: &str,
    include_dirs: &[&str],
    libraries: &[&str],
) -> ExternalCompilationInfo {
    ExternalCompilationInfo {
        post_include_bits: vec![c_header_source.to_string()],
        include_dirs: include_dirs.iter().map(PathBuf::from).collect(),
        libraries: libraries.iter().map(|s| s.to_string()).collect(),
        ..ExternalCompilationInfo::default()
    }
}

fn deferred(name: &str) -> RffiPlatformError {
    RffiPlatformError::new(format!(
        "rffi_platform.{name} requires the deferred C compiler probe backend"
    ))
}

pub fn getstruct(
    name: &str,
    _c_header_source: &str,
    interesting_fields: Vec<FieldSpec>,
) -> Result<ConfigValue, RffiPlatformError> {
    let _entry = CConfigEntry::Struct(StructEntry::new(name, interesting_fields));
    Err(deferred("getstruct"))
}

pub fn getsimpletype(
    name: &str,
    _c_header_source: &str,
    ctype_hint: FieldSpec,
) -> Result<ConfigValue, RffiPlatformError> {
    let _entry = CConfigEntry::SimpleType(SimpleTypeEntry::new(name, ctype_hint));
    Err(deferred("getsimpletype"))
}

pub fn getconstantinteger(
    name: &str,
    _c_header_source: &str,
) -> Result<ConfigValue, RffiPlatformError> {
    let _entry = CConfigEntry::ConstantInteger {
        name: name.to_string(),
    };
    Err(deferred("getconstantinteger"))
}

pub fn getdefined(
    macro_name: &str,
    _c_header_source: &str,
) -> Result<ConfigValue, RffiPlatformError> {
    let _entry = CConfigEntry::Defined {
        macro_name: macro_name.to_string(),
    };
    Err(deferred("getdefined"))
}

pub fn getdefineddouble(
    macro_name: &str,
    _c_header_source: &str,
) -> Result<ConfigValue, RffiPlatformError> {
    let _entry = CConfigEntry::DefinedConstantDouble {
        macro_name: macro_name.to_string(),
    };
    Err(deferred("getdefineddouble"))
}

pub fn getdefinedinteger(
    macro_name: &str,
    _c_header_source: &str,
) -> Result<ConfigValue, RffiPlatformError> {
    let _entry = CConfigEntry::DefinedConstantInteger {
        macro_name: macro_name.to_string(),
    };
    Err(deferred("getdefinedinteger"))
}

pub fn getdefinedstring(
    macro_name: &str,
    _c_header_source: &str,
) -> Result<ConfigValue, RffiPlatformError> {
    let _entry = CConfigEntry::DefinedConstantString {
        macro_name: macro_name.to_string(),
        name: macro_name.to_string(),
    };
    Err(deferred("getdefinedstring"))
}

pub fn getintegerfunctionresult(
    function: &str,
    args: &[&str],
    _c_header_source: &str,
) -> Result<ConfigValue, RffiPlatformError> {
    let _entry = CConfigEntry::IntegerFunctionResult {
        name: function.to_string(),
        args: args.iter().map(|arg| (*arg).to_string()).collect(),
    };
    Err(deferred("getintegerfunctionresult"))
}

pub fn has(name: &str, _c_header_source: &str) -> Result<bool, RffiPlatformError> {
    let _entry = CConfigEntry::Has {
        name: name.to_string(),
    };
    Err(deferred("has"))
}

pub fn verify_eci(_eci: &ExternalCompilationInfo) -> Result<(), RffiPlatformError> {
    Err(deferred("verify_eci"))
}

pub fn checkcompiles(expression: &str, c_header_source: &str) -> Result<bool, RffiPlatformError> {
    has(expression, c_header_source).map_err(|_| deferred("checkcompiles"))
}

pub fn sizeof(name: &str, _eci: &ExternalCompilationInfo) -> Result<i64, RffiPlatformError> {
    let _entry = CConfigEntry::SizeOf {
        name: name.to_string(),
    };
    Err(deferred("sizeof"))
}

pub fn memory_alignment_from_probe(
    sys_platform: &str,
    long_bit: i64,
    probed_align: Option<i64>,
) -> Result<i64, RffiPlatformError> {
    if sys_platform == "win32" {
        return Ok(long_bit / 8);
    }
    let result = probed_align.ok_or_else(|| {
        RffiPlatformError::new("rffi_platform.py: memory_alignment needs struct align probe")
    })?;
    if result > 0 && (result & (result - 1)) == 0 {
        Ok(result)
    } else {
        Err(RffiPlatformError::new("not a power of two??"))
    }
}

pub static _memory_alignment: LazyLock<Mutex<Option<i64>>> = LazyLock::new(|| Mutex::new(None));

pub fn memory_alignment() -> Result<i64, RffiPlatformError> {
    let cached = *_memory_alignment
        .lock()
        .expect("_memory_alignment cache poisoned");
    cached.ok_or_else(|| deferred("memory_alignment"))
}

pub const C_HEADER: &str = r#"
#include <stdio.h>
#include <stddef.h>   /* for offsetof() */

void dump(char* key, int value) {
    printf("%s: %d\n", key, value);
}
"#;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct CWriter {
    eci: ExternalCompilationInfo,
    body: Vec<String>,
    main: Vec<String>,
    closed: bool,
}

pub type _CWriter = CWriter;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ConfigResult {
    pub eci: ExternalCompilationInfo,
    pub info: HashMap<CConfigEntry, InfoMap>,
    pub result: HashMap<CConfigEntry, ConfigValue>,
}

impl ConfigResult {
    pub fn new(eci: ExternalCompilationInfo, info: HashMap<CConfigEntry, InfoMap>) -> Self {
        Self {
            eci,
            info,
            result: HashMap::new(),
        }
    }

    pub fn get_entry_result(
        &mut self,
        entry: &CConfigEntry,
    ) -> Result<ConfigValue, RffiPlatformError> {
        if let Some(value) = self.result.get(entry) {
            return Ok(value.clone());
        }
        let info = self
            .info
            .get(entry)
            .ok_or_else(|| RffiPlatformError::new("rffi_platform.py: missing config info"))?;
        let value = entry.build_result(info)?;
        self.result.insert(entry.clone(), value.clone());
        Ok(value)
    }
}

impl CWriter {
    pub fn new(eci: ExternalCompilationInfo) -> Self {
        Self {
            eci,
            ..Self::default()
        }
    }

    pub fn write_header(&mut self) {
        for bit in &self.eci.pre_include_bits {
            self.body.push(bit.clone());
        }
        for include in &self.eci.includes {
            self.body.push(format!("#include <{include}>"));
        }
        for bit in &self.eci.post_include_bits {
            self.body.push(bit.clone());
        }
        self.body.push(C_HEADER.trim_start().to_string());
        self.body.push(String::new());
    }

    pub fn write_entry(&mut self, key: &str, entry: &CConfigEntry) {
        self.body.push(format!("void dump_section_{key}(void) {{"));
        for line in entry.prepare_code() {
            if !line.is_empty() && !line.starts_with('#') {
                self.body.push(format!("\t{line}"));
            } else {
                self.body.push(line);
            }
        }
        self.body.push("}".to_string());
        self.body.push(String::new());
    }

    pub fn write_entry_main(&mut self, key: &str) {
        self.main.push(format!("\tprintf(\"-+- {key}\\n\");"));
        self.main.push(format!("\tdump_section_{key}();"));
        self.main.push("\tprintf(\"---\\n\");".to_string());
    }

    pub fn ask_gcc_source(mut self, question: &str) -> String {
        self.start_main();
        self.main.push(question.to_string());
        self.close();
        self.source()
    }

    pub fn start_main(&mut self) {
        if !self.main.iter().any(|line| line.starts_with("int main(")) {
            self.main
                .insert(0, "int main(int argc, char *argv[]) {".to_string());
        }
    }

    pub fn close(&mut self) {
        if !self.closed {
            self.main.push("\treturn 0;".to_string());
            self.main.push("}".to_string());
            self.closed = true;
        }
    }

    pub fn source(&self) -> String {
        let mut lines = self.body.clone();
        lines.extend(self.main.iter().cloned());
        lines.join("\n")
    }
}

pub fn configure_entries_source(entries: &[CConfigEntry], eci: ExternalCompilationInfo) -> String {
    let mut writer = CWriter::new(eci);
    writer.write_header();
    for (i, entry) in entries.iter().enumerate() {
        writer.write_entry(&i.to_string(), entry);
    }
    writer.start_main();
    for i in 0..entries.len() {
        writer.write_entry_main(&i.to_string());
    }
    writer.close();
    writer.source()
}

pub fn parse_run_example_output(output: &str) -> Result<Vec<InfoMap>, RffiPlatformError> {
    if !output.starts_with("-+- ") {
        return Err(RffiPlatformError::new(format!(
            "run_example_code failed! output = {output:?}"
        )));
    }
    let mut result = Vec::new();
    let mut section: Option<InfoMap> = None;
    for raw_line in output.lines() {
        let line = raw_line.trim();
        if line.starts_with("-+- ") {
            section = Some(BTreeMap::new());
        } else if line == "---" {
            let Some(done) = section.take() else {
                return Err(RffiPlatformError::new(
                    "rffi_platform.py: section end before section start",
                ));
            };
            result.push(done);
        } else if !line.is_empty() {
            let Some(ref mut current) = section else {
                return Err(RffiPlatformError::new(format!(
                    "rffi_platform.py: data outside section: {line:?}"
                )));
            };
            let (key, value) = line.split_once(": ").ok_or_else(|| {
                RffiPlatformError::new(format!("rffi_platform.py: malformed output line {line:?}"))
            })?;
            current.insert(
                key.to_string(),
                value.parse::<i64>().map_err(|e| {
                    RffiPlatformError::new(format!(
                        "rffi_platform.py: invalid integer output {value:?}: {e}"
                    ))
                })?,
            );
        }
    }
    Ok(result)
}

pub type InfoMap = BTreeMap<String, i64>;

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum CTypeClass {
    Integer,
    Float,
    Pointer,
    ArrayNoLength,
    Struct(String),
    Other,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct FieldSpec {
    pub name: String,
    pub ctype: String,
    pub class: CTypeClass,
    pub size: i64,
    pub unsigned: bool,
}

impl FieldSpec {
    pub fn new(name: impl Into<String>, ctype: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ctype: ctype.into(),
            class: CTypeClass::Other,
            size: 0,
            unsigned: false,
        }
    }

    pub fn integer(
        name: impl Into<String>,
        ctype: impl Into<String>,
        size: i64,
        unsigned: bool,
    ) -> Self {
        Self {
            name: name.into(),
            ctype: ctype.into(),
            class: CTypeClass::Integer,
            size,
            unsigned,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct StructEntry {
    pub name: String,
    pub interesting_fields: Vec<FieldSpec>,
    pub ifdef: Option<String>,
}

impl StructEntry {
    pub fn new(name: impl Into<String>, interesting_fields: Vec<FieldSpec>) -> Self {
        Self {
            name: name.into(),
            interesting_fields,
            ifdef: None,
        }
    }

    pub fn with_ifdef(mut self, ifdef: impl Into<String>) -> Self {
        self.ifdef = Some(ifdef.into());
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct SimpleTypeEntry {
    pub name: String,
    pub ctype_hint: FieldSpec,
    pub ifdef: Option<String>,
}

impl SimpleTypeEntry {
    pub fn new(name: impl Into<String>, ctype_hint: FieldSpec) -> Self {
        Self {
            name: name.into(),
            ctype_hint,
            ifdef: None,
        }
    }

    pub fn with_ifdef(mut self, ifdef: impl Into<String>) -> Self {
        self.ifdef = Some(ifdef.into());
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum CConfigEntry {
    Struct(StructEntry),
    SimpleType(SimpleTypeEntry),
    ConstantInteger {
        name: String,
    },
    IntegerFunctionResult {
        name: String,
        args: Vec<String>,
    },
    DefinedConstantInteger {
        macro_name: String,
    },
    DefinedConstantDouble {
        macro_name: String,
    },
    DefinedConstantString {
        macro_name: String,
        name: String,
    },
    Defined {
        macro_name: String,
    },
    Has {
        name: String,
    },
    Works,
    SizeOf {
        name: String,
    },
    PaddingDropFieldLookup {
        name: String,
        staticfields: Vec<Option<String>>,
        fieldname: String,
    },
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct CConfigSingleEntry;

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct _PaddingDropFieldLookup {
    pub name: String,
    pub staticfields: Vec<Option<String>>,
    pub fieldname: String,
}

impl _PaddingDropFieldLookup {
    pub fn into_entry(self) -> CConfigEntry {
        CConfigEntry::PaddingDropFieldLookup {
            name: self.name,
            staticfields: self.staticfields,
            fieldname: self.fieldname,
        }
    }
}

impl CConfigEntry {
    pub fn prepare_code(&self) -> Vec<String> {
        match self {
            CConfigEntry::Struct(entry) => prepare_struct(entry),
            CConfigEntry::SimpleType(entry) => prepare_simple_type(entry),
            CConfigEntry::ConstantInteger { name } => prepare_constant_integer(name),
            CConfigEntry::IntegerFunctionResult { name, args } => {
                prepare_integer_function_result(name, args)
            }
            CConfigEntry::DefinedConstantInteger { macro_name } => {
                prepare_defined_constant_integer(macro_name)
            }
            CConfigEntry::DefinedConstantDouble { macro_name } => {
                prepare_defined_constant_double(macro_name)
            }
            CConfigEntry::DefinedConstantString { macro_name, name } => {
                prepare_defined_constant_string(macro_name, name)
            }
            CConfigEntry::Defined { macro_name } => vec![
                format!("#ifdef {macro_name}"),
                "dump(\"defined\", 1);".to_string(),
                "#else".to_string(),
                "dump(\"defined\", 0);".to_string(),
                "#endif".to_string(),
            ],
            CConfigEntry::Has { name } => vec![format!("(void){name};")],
            CConfigEntry::Works => Vec::new(),
            CConfigEntry::SizeOf { name } => vec![format!("dump(\"size\",  sizeof({name}));")],
            CConfigEntry::PaddingDropFieldLookup {
                name,
                staticfields,
                fieldname,
            } => prepare_padding_drop_field_lookup(name, staticfields, fieldname),
        }
    }

    pub fn build_result(&self, info: &InfoMap) -> Result<ConfigValue, RffiPlatformError> {
        match self {
            CConfigEntry::Struct(entry) => build_struct_result(entry, info),
            CConfigEntry::SimpleType(entry) => build_simple_type_result(entry, info),
            CConfigEntry::ConstantInteger { .. } | CConfigEntry::IntegerFunctionResult { .. } => {
                Ok(ConfigValue::Integer(expose_value_as_rpython(required(
                    info, "value",
                )?)))
            }
            CConfigEntry::DefinedConstantInteger { .. } => {
                if required(info, "defined")? != 0 {
                    Ok(ConfigValue::Integer(expose_value_as_rpython(required(
                        info, "value",
                    )?)))
                } else {
                    Ok(ConfigValue::None)
                }
            }
            CConfigEntry::DefinedConstantDouble { .. } => {
                if required(info, "defined")? != 0 {
                    let mut bytes = [0_u8; 8];
                    for (i, byte) in bytes.iter_mut().enumerate() {
                        *byte = required(info, &format!("value_{i}"))? as u8;
                    }
                    Ok(ConfigValue::Double(f64::from_ne_bytes(bytes)))
                } else {
                    Ok(ConfigValue::None)
                }
            }
            CConfigEntry::DefinedConstantString { .. } => {
                if required(info, "defined")? != 0 {
                    let mut bytes = Vec::new();
                    let mut i = 0;
                    while let Some(value) = info.get(&format!("value_{i}")) {
                        bytes.push(*value as u8);
                        i += 1;
                    }
                    Ok(ConfigValue::String(String::from_utf8(bytes).map_err(
                        |e| {
                            RffiPlatformError::new(format!(
                                "rffi_platform.py: invalid configured string: {e}"
                            ))
                        },
                    )?))
                } else {
                    Ok(ConfigValue::None)
                }
            }
            CConfigEntry::Defined { .. } => Ok(ConfigValue::Bool(required(info, "defined")? != 0)),
            CConfigEntry::Has { .. } | CConfigEntry::Works => Ok(ConfigValue::Bool(true)),
            CConfigEntry::SizeOf { .. } => Ok(ConfigValue::Size(required(info, "size")?)),
            CConfigEntry::PaddingDropFieldLookup { .. } => {
                Ok(ConfigValue::Integer(required(info, "fieldlookup")?))
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ConfigValue {
    Struct(StructLayout),
    CType(FieldSpec),
    Integer(i64),
    Bool(bool),
    Double(f64),
    String(String),
    Size(i64),
    None,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StructLayout {
    pub name: String,
    pub align: i64,
    pub size: i64,
    pub fields: Vec<LayoutField>,
    pub padding: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LayoutField {
    pub name: String,
    pub ctype: String,
    pub offset: i64,
    pub size: i64,
}

fn prepare_struct(entry: &StructEntry) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(ifdef) = &entry.ifdef {
        out.push(format!("#ifdef {ifdef}"));
    }
    out.push(format!("typedef {} platcheck_t;", entry.name));
    out.push("typedef struct {".to_string());
    out.push("    char c;".to_string());
    out.push("    platcheck_t s;".to_string());
    out.push("} platcheck2_t;".to_string());
    out.push(String::new());
    out.push("platcheck_t s;".to_string());
    if entry.ifdef.is_some() {
        out.push("dump(\"defined\", 1);".to_string());
    }
    out.push("dump(\"align\", offsetof(platcheck2_t, s));".to_string());
    out.push("dump(\"size\",  sizeof(platcheck_t));".to_string());
    for field in &entry.interesting_fields {
        out.push(format!(
            "dump(\"fldofs {}\", offsetof(platcheck_t, {}));",
            field.name, field.name
        ));
        out.push(format!(
            "dump(\"fldsize {}\",   sizeof(s.{}));",
            field.name, field.name
        ));
        if field.class == CTypeClass::Integer {
            out.push(format!(
                "s.{} = 0; s.{} = ~s.{};",
                field.name, field.name, field.name
            ));
            out.push(format!(
                "dump(\"fldunsigned {}\", s.{} > 0);",
                field.name, field.name
            ));
        }
    }
    if entry.ifdef.is_some() {
        out.push("#else".to_string());
        out.push("dump(\"defined\", 0);".to_string());
        out.push("#endif".to_string());
    }
    out
}

fn build_struct_result(
    entry: &StructEntry,
    info: &InfoMap,
) -> Result<ConfigValue, RffiPlatformError> {
    if entry.ifdef.is_some() && required(info, "defined")? == 0 {
        return Ok(ConfigValue::None);
    }
    let size = required(info, "size")?;
    let mut layout = vec![None; size as usize];
    for field in &entry.interesting_fields {
        let offset = required(info, &format!("fldofs {}", field.name))?;
        let observed_size = required(info, &format!("fldsize {}", field.name))?;
        let observed_unsigned = info
            .get(&format!("fldunsigned {}", field.name))
            .copied()
            .unwrap_or(0)
            != 0;
        let mut ctype = field.clone();
        if field.class != CTypeClass::ArrayNoLength
            && field.size != 0
            && (observed_size, observed_unsigned) != (field.size, field.unsigned)
        {
            ctype.size = observed_size;
            ctype.unsigned = observed_unsigned;
        }
        layout_addfield(&mut layout, offset, &ctype, &field.name)?;
    }

    let mut padfields = Vec::new();
    let mut pad_n = 0;
    for i in 0..layout.len() {
        if layout[i].is_none() {
            let name = format!("_pad{pad_n}");
            pad_n += 1;
            let pad = FieldSpec::integer(&name, "rffi.UCHAR", 1, true);
            layout_addfield(&mut layout, i as i64, &pad, &name)?;
            padfields.push(format!("c_{name}"));
        }
    }

    let mut fields = Vec::new();
    let mut seen = HashMap::new();
    for (offset, cell) in layout.into_iter().enumerate() {
        let Some(cell) = cell else {
            continue;
        };
        if seen.insert(cell.name.clone(), ()).is_none() {
            fields.push(LayoutField {
                name: format!("c_{}", cell.name),
                ctype: cell.ctype,
                offset: offset as i64,
                size: cell.size,
            });
        }
    }

    Ok(ConfigValue::Struct(StructLayout {
        name: entry
            .name
            .strip_prefix("struct ")
            .unwrap_or(&entry.name)
            .to_string(),
        align: required(info, "align")?,
        size,
        fields,
        padding: padfields,
    }))
}

fn prepare_simple_type(entry: &SimpleTypeEntry) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(ifdef) = &entry.ifdef {
        out.push(format!("#ifdef {ifdef}"));
    }
    out.push(format!("typedef {} platcheck_t;", entry.name));
    out.push(String::new());
    out.push("platcheck_t x;".to_string());
    if entry.ifdef.is_some() {
        out.push("dump(\"defined\", 1);".to_string());
    }
    out.push("dump(\"size\",  sizeof(platcheck_t));".to_string());
    if entry.ctype_hint.class == CTypeClass::Integer {
        out.push("x = 0; x = ~x;".to_string());
        out.push("dump(\"unsigned\", x > 0);".to_string());
    }
    if entry.ifdef.is_some() {
        out.push("#else".to_string());
        out.push("dump(\"defined\", 0);".to_string());
        out.push("#endif".to_string());
    }
    out
}

fn build_simple_type_result(
    entry: &SimpleTypeEntry,
    info: &InfoMap,
) -> Result<ConfigValue, RffiPlatformError> {
    if entry.ifdef.is_some() && required(info, "defined")? == 0 {
        return Ok(ConfigValue::None);
    }
    let mut ctype = entry.ctype_hint.clone();
    ctype.size = required(info, "size")?;
    ctype.unsigned = info.get("unsigned").copied().unwrap_or(0) != 0;
    Ok(ConfigValue::CType(ctype))
}

fn prepare_constant_integer(name: &str) -> Vec<String> {
    vec![
        format!("if (({name}) <= 0) {{"),
        format!("    long long x = (long long)({name});"),
        "    printf(\"value: %lld\\n\", x);".to_string(),
        "} else {".to_string(),
        format!("    unsigned long long x = (unsigned long long)({name});"),
        "    printf(\"value: %llu\\n\", x);".to_string(),
        "}".to_string(),
    ]
}

fn prepare_integer_function_result(name: &str, args: &[String]) -> Vec<String> {
    let call_args = args.join(", ");
    vec![
        format!("long int result = {name}({call_args});"),
        "if ((result) <= 0) {".to_string(),
        "    long long x = (long long)(result);".to_string(),
        "    printf(\"value: %lld\\n\", x);".to_string(),
        "} else {".to_string(),
        "    unsigned long long x = (unsigned long long)(result);".to_string(),
        "    printf(\"value: %llu\\n\", x);".to_string(),
        "}".to_string(),
    ]
}

fn prepare_defined_constant_integer(macro_name: &str) -> Vec<String> {
    let mut out = vec![
        format!("#ifdef {macro_name}"),
        "dump(\"defined\", 1);".to_string(),
    ];
    out.extend(prepare_constant_integer(macro_name));
    out.extend([
        "#else".to_string(),
        "dump(\"defined\", 0);".to_string(),
        "#endif".to_string(),
    ]);
    out
}

fn prepare_defined_constant_double(macro_name: &str) -> Vec<String> {
    vec![
        format!("#ifdef {macro_name}"),
        "int i;".to_string(),
        format!("double x = {macro_name};"),
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

fn prepare_defined_constant_string(macro_name: &str, name: &str) -> Vec<String> {
    vec![
        format!("#ifdef {macro_name}"),
        "int i;".to_string(),
        format!("const char *p = {name};"),
        "dump(\"defined\", 1);".to_string(),
        "for (i = 0; p[i] != 0; i++ ) {".to_string(),
        "  printf(\"value_%d: %d\\n\", i, (int)(unsigned char)p[i]);".to_string(),
        "}".to_string(),
        "#else".to_string(),
        "dump(\"defined\", 0);".to_string(),
        "#endif".to_string(),
    ]
}

fn prepare_padding_drop_field_lookup(
    name: &str,
    staticfields: &[Option<String>],
    fieldname: &str,
) -> Vec<String> {
    let mut out = vec![
        format!("typedef {name} platcheck_t;"),
        "static platcheck_t s = {".to_string(),
    ];
    for (i, ty) in staticfields.iter().enumerate() {
        let value = if i == staticfields.len() - 1 { -1 } else { 0 };
        match ty {
            Some(ty) => out.push(format!("\t({ty}){value},")),
            None => out.push(format!("\t{value},")),
        }
    }
    out.push("};".to_string());
    assert!(fieldname.starts_with("c_"));
    out.push(format!(
        "dump(\"fieldlookup\", s.{} != 0);",
        &fieldname[2..]
    ));
    out
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Field {
    name: String,
    ctype: String,
    size: i64,
}

fn layout_addfield(
    layout: &mut [Option<Field>],
    offset: i64,
    ctype: &FieldSpec,
    prefix: &str,
) -> Result<(), RffiPlatformError> {
    let size = if ctype.class == CTypeClass::ArrayNoLength {
        layout.len() as i64 - offset
    } else {
        ctype.size.max(1)
    };
    let mut name = prefix.to_string();
    let mut i = 0;
    while layout
        .iter()
        .any(|cell| cell.as_ref().is_some_and(|field| field.name == name))
    {
        i += 1;
        name = format!("{prefix}_{i}");
    }
    let field = Field {
        name: name.clone(),
        ctype: ctype.ctype.clone(),
        size,
    };
    for idx in offset..offset + size {
        let slot = layout.get_mut(idx as usize).ok_or_else(|| {
            RffiPlatformError::new(format!(
                "rffi_platform.py: field {name} outside layout at offset {idx}"
            ))
        })?;
        if slot.is_some() {
            return Err(RffiPlatformError::new(format!(
                "{name} overlaps {:?}",
                slot.as_ref()
            )));
        }
        *slot = Some(field.clone());
    }
    Ok(())
}

pub fn expose_value_as_rpython(value: i64) -> i64 {
    value
}

pub fn uniquefilepath() -> PathBuf {
    static LAST: AtomicUsize = AtomicUsize::new(0);
    let i = LAST.fetch_add(1, Ordering::Relaxed);
    PathBuf::from(format!("platcheck_{i}.c"))
}

pub static integer_class: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    vec![
        "SIGNEDCHAR",
        "UCHAR",
        "CHAR",
        "SHORT",
        "USHORT",
        "INT",
        "UINT",
        "INT_real",
        "UINT_real",
        "LONG",
        "ULONG",
        "LONGLONG",
        "ULONGLONG",
    ]
});

pub static float_class: LazyLock<Vec<&'static str>> = LazyLock::new(|| vec!["DOUBLE"]);

pub fn _sizeof(ctype: &FieldSpec) -> i64 {
    ctype.size.max(1)
}

pub fn is_array_nolength(ctype: &FieldSpec) -> bool {
    ctype.class == CTypeClass::ArrayNoLength
}

pub fn fixup_ctype(
    fieldtype: &FieldSpec,
    fieldname: &str,
    expected_size_and_sign: (i64, bool),
) -> Result<FieldSpec, RffiPlatformError> {
    if matches!(fieldtype.class, CTypeClass::Integer | CTypeClass::Float) {
        let mut fixed = fieldtype.clone();
        fixed.size = expected_size_and_sign.0;
        fixed.unsigned = expected_size_and_sign.1;
        return Ok(fixed);
    }
    Err(RffiPlatformError::new(format!(
        "conflict between translating python and compiler field type {fieldtype:?} for symbol {fieldname:?}, expected size+sign {expected_size_and_sign:?}"
    )))
}

pub static PYPY_EXTERNAL_DIR: LazyLock<PathBuf> = LazyLock::new(|| PathBuf::from("../.."));

pub fn configure_external_library(
    name: &str,
    _eci: ExternalCompilationInfo,
    _configurations: &[HashMap<String, String>],
) -> Result<ExternalCompilationInfo, RffiPlatformError> {
    Err(RffiPlatformError::new(format!(
        "Library {name} is not installed or configure_external_library backend is deferred"
    )))
}

pub fn configure_boehm() -> Result<ExternalCompilationInfo, RffiPlatformError> {
    Err(deferred("configure_boehm"))
}

fn required(info: &InfoMap, key: &str) -> Result<i64, RffiPlatformError> {
    info.get(key).copied().ok_or_else(|| {
        RffiPlatformError::new(format!("rffi_platform.py: missing key {key:?} in {info:?}"))
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RffiPlatformError {
    pub message: String,
}

impl RffiPlatformError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for RffiPlatformError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RffiPlatformError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn int_field(name: &str, size: i64, unsigned: bool) -> FieldSpec {
        FieldSpec::integer(name, "rffi.INT", size, unsigned)
    }

    #[test]
    fn eci_from_header_populates_post_include_bits_and_search_paths() {
        let eci = eci_from_header("#include <sys/types.h>", &["/usr/include"], &["m"]);
        assert_eq!(eci.post_include_bits, vec!["#include <sys/types.h>"]);
        assert_eq!(eci.include_dirs, vec![PathBuf::from("/usr/include")]);
        assert_eq!(eci.libraries, vec!["m"]);
    }

    #[test]
    fn struct_prepare_code_matches_upstream_probe_shape() {
        let entry = CConfigEntry::Struct(StructEntry::new(
            "struct foo",
            vec![int_field("bar", 4, false)],
        ));
        let code = entry.prepare_code().join("\n");
        assert!(code.contains("typedef struct foo platcheck_t;"));
        assert!(code.contains("dump(\"align\", offsetof(platcheck2_t, s));"));
        assert!(code.contains("dump(\"fldofs bar\", offsetof(platcheck_t, bar));"));
        assert!(code.contains("s.bar = 0; s.bar = ~s.bar;"));
    }

    #[test]
    fn simple_type_prepare_and_build_result_tracks_size_and_sign() {
        let entry = CConfigEntry::SimpleType(SimpleTypeEntry::new(
            "pid_t",
            FieldSpec::integer("pid_t", "rffi.INT", 4, false),
        ));
        assert!(
            entry
                .prepare_code()
                .join("\n")
                .contains("typedef pid_t platcheck_t;")
        );
        let info = InfoMap::from([("size".to_string(), 8), ("unsigned".to_string(), 1)]);
        assert_eq!(
            entry.build_result(&info).unwrap(),
            ConfigValue::CType(FieldSpec::integer("pid_t", "rffi.INT", 8, true))
        );
    }

    #[test]
    fn parse_run_example_output_splits_sections_like_run_example_code() {
        let parsed =
            parse_run_example_output("-+- 0\nsize: 4\n---\n-+- 1\ndefined: 1\n---\n").unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0]["size"], 4);
        assert_eq!(parsed[1]["defined"], 1);
    }

    #[test]
    fn defined_constant_string_builds_bytes_until_missing_index() {
        let entry = CConfigEntry::DefinedConstantString {
            macro_name: "NAME".to_string(),
            name: "NAME".to_string(),
        };
        let info = InfoMap::from([
            ("defined".to_string(), 1),
            ("value_0".to_string(), b'a' as i64),
            ("value_1".to_string(), b'b' as i64),
        ]);
        assert_eq!(
            entry.build_result(&info).unwrap(),
            ConfigValue::String("ab".to_string())
        );
    }

    #[test]
    fn configure_entries_source_writes_header_sections_and_main() {
        let source = configure_entries_source(
            &[
                CConfigEntry::Defined {
                    macro_name: "FOO".to_string(),
                },
                CConfigEntry::SizeOf {
                    name: "long".to_string(),
                },
            ],
            eci_from_header("#include <limits.h>", &[], &[]),
        );
        assert!(source.contains("#include <limits.h>"));
        assert!(source.contains("void dump_section_0(void)"));
        assert!(source.contains("printf(\"-+- 1\\n\");"));
        assert!(source.contains("dump_section_1();"));
    }

    #[test]
    fn public_shortcut_helpers_expose_upstream_names_as_deferred() {
        assert!(getconstantinteger("FOO", "#define FOO 1").is_err());
        assert!(getdefined("FOO", "#define FOO").is_err());
        assert!(has("FOO", "#define FOO").is_err());
        assert!(verify_eci(&ExternalCompilationInfo::default()).is_err());
        assert!(sizeof("long", &ExternalCompilationInfo::default()).is_err());
        assert!(memory_alignment().is_err());
        assert!(configure_boehm().is_err());

        let mut result = ConfigResult::new(ExternalCompilationInfo::default(), HashMap::new());
        let entry = CConfigEntry::SizeOf {
            name: "long".to_string(),
        };
        assert!(result.get_entry_result(&entry).is_err());

        assert_eq!(
            uniquefilepath().extension().and_then(|s| s.to_str()),
            Some("c")
        );
        assert!(integer_class.contains(&"INT"));
        assert_eq!(&*float_class, &["DOUBLE"]);
    }

    #[test]
    fn memory_alignment_uses_windows_shortcut_and_validates_probe() {
        assert_eq!(memory_alignment_from_probe("win32", 64, None).unwrap(), 8);
        assert_eq!(
            memory_alignment_from_probe("linux", 64, Some(16)).unwrap(),
            16
        );
        assert!(memory_alignment_from_probe("linux", 64, Some(12)).is_err());
    }
}
