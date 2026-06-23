//! RPython `rpython/rtyper/tool/mkrffi.py` parity source generator.
//!
//! Upstream walks live `ctypes` objects.  Rust has no equivalent runtime
//! object model here, so this port keeps the same generator shape over an
//! explicit `CType` tree: primitive mapping, pointer/array formatting,
//! struct emission with forward references, and `rffi.llexternal` source
//! rows.

use std::collections::{HashMap, HashSet};
use std::fmt;

pub fn primitive_pointer_repr(tp_s: &str) -> String {
    format!("lltype.Ptr(lltype.FixedSizeArray({tp_s}, 1))")
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum SimpleCType {
    CUByte,
    CByte,
    CChar,
    CInt8,
    CUShort,
    CShort,
    CUInt16,
    CInt16,
    CInt,
    CUInt,
    CInt32,
    CUInt32,
    CLongLong,
    CULongLong,
    CInt64,
    CUInt64,
    CVoidP,
    Void,
    CCharP,
    CDouble,
}

impl SimpleCType {
    pub fn rffi_repr(&self) -> &'static str {
        match self {
            SimpleCType::CUByte => "rffi.UCHAR",
            SimpleCType::CByte => "rffi.CHAR",
            SimpleCType::CChar => "rffi.CHAR",
            SimpleCType::CInt8 => "rffi.CHAR",
            SimpleCType::CUShort => "rffi.USHORT",
            SimpleCType::CShort => "rffi.SHORT",
            SimpleCType::CUInt16 => "rffi.USHORT",
            SimpleCType::CInt16 => "rffi.SHORT",
            SimpleCType::CInt => "rffi.INT",
            SimpleCType::CUInt => "rffi.UINT",
            SimpleCType::CInt32 => "rffi.INT_real",
            SimpleCType::CUInt32 => "rffi.UINT_real",
            SimpleCType::CLongLong => "rffi.LONGLONG",
            SimpleCType::CULongLong => "rffi.ULONGLONG",
            SimpleCType::CInt64 => "rffi.LONGLONG",
            SimpleCType::CUInt64 => "rffi.ULONGLONG",
            SimpleCType::CVoidP => "rffi.VOIDP",
            SimpleCType::Void => "rffi.lltype.Void",
            SimpleCType::CCharP => "rffi.CCHARP",
            SimpleCType::CDouble => "rffi.lltype.Float",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct StructDecl {
    pub name: String,
    pub fields: Vec<(String, CType)>,
}

impl StructDecl {
    pub fn new(name: impl Into<String>, fields: Vec<(impl Into<String>, CType)>) -> Self {
        Self {
            name: name.into(),
            fields: fields
                .into_iter()
                .map(|(name, tp)| (name.into(), tp))
                .collect(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum CType {
    Simple(SimpleCType),
    Pointer(Box<CType>),
    Array(Box<CType>),
    Struct(StructDecl),
    StructRef(String),
}

impl CType {
    pub fn pointer(inner: CType) -> Self {
        CType::Pointer(Box::new(inner))
    }

    pub fn array(inner: CType) -> Self {
        CType::Array(Box::new(inner))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct FunctionDecl {
    pub name: String,
    pub argtypes: Vec<CType>,
    pub restype: CType,
}

impl FunctionDecl {
    pub fn new(name: impl Into<String>, argtypes: Vec<CType>, restype: CType) -> Self {
        Self {
            name: name.into(),
            argtypes,
            restype,
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct RffiSource {
    structs: HashSet<String>,
    source: Vec<String>,
    extra_args: String,
    seen: HashSet<String>,
    forward_refs: usize,
    forward_refs_to_consider: HashMap<String, String>,
}

impl RffiSource {
    pub fn new(includes: &[&str], libraries: &[&str], include_dirs: &[&str]) -> Self {
        let mut extra_args = String::new();
        if !includes.is_empty() {
            extra_args.push_str(&format!("includes={}, ", py_tuple_repr(includes)));
        }
        if !libraries.is_empty() {
            extra_args.push_str(&format!("libraries={}, ", py_tuple_repr(libraries)));
        }
        if !include_dirs.is_empty() {
            extra_args.push_str(&format!("include_dirs={}, ", py_tuple_repr(include_dirs)));
        }
        Self {
            extra_args,
            ..Self::default()
        }
    }

    pub fn with_source(source: impl Into<String>) -> Self {
        Self {
            source: vec![source.into()],
            ..Self::default()
        }
    }

    pub fn next_forward_reference(&mut self) -> String {
        let name = format!("forward_ref{}", self.forward_refs);
        self.forward_refs += 1;
        name
    }

    pub fn append(&mut self, other: &RffiSource) {
        self.structs.extend(other.structs.iter().cloned());
        self.source.extend(other.source.iter().cloned());
    }

    pub fn proc_struct(&mut self, tp: &StructDecl) -> Result<String, MkrffiError> {
        if !self.structs.contains(&tp.name) {
            self.seen.insert(tp.name.clone());
            let fields = tp
                .fields
                .iter()
                .map(|(field_name, field_tp)| {
                    Ok(format!("('{}', {}), ", field_name, self.proc_tp(field_tp)?))
                })
                .collect::<Result<Vec<_>, MkrffiError>>()?
                .join("");
            self.seen.remove(&tp.name);

            self.structs.insert(tp.name.clone());
            self.source.push(format!(
                "{} = lltype.Struct('{}', {}hints={{'external':'C'}})",
                tp.name, tp.name, fields
            ));
            if let Some(forward_ref) = self.forward_refs_to_consider.get(&tp.name) {
                self.source
                    .push(format!("\n{}.become({})\n", forward_ref, tp.name));
            }
        }
        Ok(tp.name.clone())
    }

    pub fn proc_forward_ref(&mut self, tp_name: &str) -> String {
        if let Some(name) = self.forward_refs_to_consider.get(tp_name) {
            return name.clone();
        }
        let name = self.next_forward_reference();
        self.source
            .push(format!("{name} = lltype.ForwardReference()"));
        self.forward_refs_to_consider
            .insert(tp_name.to_string(), name.clone());
        name
    }

    pub fn proc_tp(&mut self, tp: &CType) -> Result<String, MkrffiError> {
        match tp {
            CType::Simple(simple) => Ok(simple.rffi_repr().to_string()),
            CType::Pointer(inner) => match inner.as_ref() {
                CType::Simple(_) => Ok(format!(
                    "lltype.Ptr(lltype.Array({}, hints={{'nolength': True}}))",
                    self.proc_tp(inner)?
                )),
                _ => Ok(format!("lltype.Ptr({})", self.proc_tp(inner)?)),
            },
            CType::Array(inner) => Ok(format!(
                "lltype.Ptr(lltype.Array({}, hints={{'nolength': True}}))",
                self.proc_tp(inner)?
            )),
            CType::Struct(decl) => {
                if self.seen.contains(&decl.name) {
                    Ok(self.proc_forward_ref(&decl.name))
                } else {
                    self.proc_struct(decl)
                }
            }
            CType::StructRef(name) => {
                if self.seen.contains(name) {
                    Ok(self.proc_forward_ref(name))
                } else {
                    Ok(name.clone())
                }
            }
        }
    }

    pub fn proc_func(&mut self, func: &FunctionDecl) -> Result<(), MkrffiError> {
        let args = func
            .argtypes
            .iter()
            .map(|arg| self.proc_tp(arg))
            .collect::<Result<Vec<_>, _>>()?
            .join(", ");
        let result = self.proc_tp(&func.restype)?;
        let extra_args = if self.extra_args.is_empty() {
            String::new()
        } else {
            format!(", {}", self.extra_args)
        };
        self.source.push(format!(
            "{} = rffi.llexternal('{}', [{}], {}{})",
            func.name, func.name, args, result, extra_args
        ));
        Ok(())
    }

    pub fn proc_namespace<'a>(
        &mut self,
        funcs: impl IntoIterator<Item = &'a FunctionDecl>,
    ) -> Result<(), MkrffiError> {
        for func in funcs {
            self.proc_func(func)?;
        }
        Ok(())
    }

    pub fn source(&self) -> String {
        self.source.join("\n")
    }

    pub fn source_with_imports(&self) -> String {
        let body = self.source();
        if body.is_empty() {
            "from rpython.rtyper.lltypesystem import lltype\nfrom rpython.rtyper.lltypesystem import rffi".to_string()
        } else {
            format!(
                "from rpython.rtyper.lltypesystem import lltype\nfrom rpython.rtyper.lltypesystem import rffi\n{body}"
            )
        }
    }
}

impl fmt::Display for RffiSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.source())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MkrffiError {
    pub message: String,
}

impl MkrffiError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for MkrffiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for MkrffiError {}

pub fn unsupported_mapping(tp: impl fmt::Debug) -> MkrffiError {
    MkrffiError::new(format!("Not implemented mapping for {tp:?}"))
}

fn py_tuple_repr(values: &[&str]) -> String {
    let items = values
        .iter()
        .map(|value| format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'")))
        .collect::<Vec<_>>();
    if items.len() == 1 {
        format!("({},)", items[0])
    } else {
        format!("({})", items.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple(tp: SimpleCType) -> CType {
        CType::Simple(tp)
    }

    #[test]
    fn primitive_pointer_repr_matches_mkrffi_py() {
        assert_eq!(
            primitive_pointer_repr("rffi.INT"),
            "lltype.Ptr(lltype.FixedSizeArray(rffi.INT, 1))"
        );
    }

    #[test]
    fn simple_type_mapping_matches_upstream_strings() {
        assert_eq!(SimpleCType::CUByte.rffi_repr(), "rffi.UCHAR");
        assert_eq!(SimpleCType::CInt8.rffi_repr(), "rffi.CHAR");
        assert_eq!(SimpleCType::CUInt32.rffi_repr(), "rffi.UINT_real");
        assert_eq!(SimpleCType::Void.rffi_repr(), "rffi.lltype.Void");
        assert_eq!(SimpleCType::CDouble.rffi_repr(), "rffi.lltype.Float");
    }

    #[test]
    fn proc_tp_formats_simple_pointer_and_array_like_upstream() {
        let mut src = RffiSource::default();
        assert_eq!(
            src.proc_tp(&CType::pointer(simple(SimpleCType::CInt)))
                .unwrap(),
            "lltype.Ptr(lltype.Array(rffi.INT, hints={'nolength': True}))"
        );
        assert_eq!(
            src.proc_tp(&CType::array(simple(SimpleCType::CDouble)))
                .unwrap(),
            "lltype.Ptr(lltype.Array(rffi.lltype.Float, hints={'nolength': True}))"
        );
    }

    #[test]
    fn proc_struct_emits_external_struct_and_forward_ref_become() {
        let node = StructDecl::new(
            "NODE",
            vec![("next", CType::pointer(CType::StructRef("NODE".to_string())))],
        );
        let mut src = RffiSource::default();
        assert_eq!(src.proc_struct(&node).unwrap(), "NODE");
        assert_eq!(
            src.source(),
            "forward_ref0 = lltype.ForwardReference()\nNODE = lltype.Struct('NODE', ('next', lltype.Ptr(forward_ref0)), hints={'external':'C'})\n\nforward_ref0.become(NODE)\n"
        );
    }

    #[test]
    fn proc_func_emits_llexternal_with_extra_args() {
        let mut src = RffiSource::new(&["math.h"], &["m"], &[]);
        let func = FunctionDecl::new(
            "hypot",
            vec![simple(SimpleCType::CDouble), simple(SimpleCType::CDouble)],
            simple(SimpleCType::CDouble),
        );
        src.proc_func(&func).unwrap();
        assert_eq!(
            src.source(),
            "hypot = rffi.llexternal('hypot', [rffi.lltype.Float, rffi.lltype.Float], rffi.lltype.Float, includes=('math.h',), libraries=('m',), )"
        );
    }

    #[test]
    fn append_combines_sources_and_struct_sets() {
        let mut left = RffiSource::with_source("left = 1");
        let right = RffiSource::with_source("right = 2");
        left.append(&right);
        assert_eq!(left.source(), "left = 1\nright = 2");
    }
}
