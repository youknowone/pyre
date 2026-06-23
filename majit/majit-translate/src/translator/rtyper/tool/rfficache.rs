//! RPython `rpython/rtyper/tool/rfficache.py` bootstrapping helpers.
//!
//! Upstream shells out to the platform C compiler via `ask_gcc`.  The local
//! C backend/platform execution layer is still deferred, so this module ports
//! the deterministic pieces around that boundary: generated C source,
//! `sizeof`/signedness output parsing, and the cached integer type registry.

use std::collections::HashMap;
use std::fmt;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NumberType {
    pub name: String,
    pub rust_class_name: String,
    pub signed: bool,
    pub bits: i64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Platform {
    types: HashMap<String, NumberType>,
    numbertype_to_rclass: HashMap<String, String>,
}

impl Platform {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn inttype_from_size(&mut self, name: &str, signed: bool, size: i64) -> NumberType {
        match self.types.get(name) {
            Some(existing) => existing.clone(),
            None => self.make_type(name, signed, size),
        }
    }

    fn make_type(&mut self, name: &str, signed: bool, size: i64) -> NumberType {
        let number_type = NumberType {
            name: name.to_string(),
            rust_class_name: format!("r_{name}"),
            signed,
            bits: size * 8,
        };
        self.numbertype_to_rclass.insert(
            number_type.name.clone(),
            number_type.rust_class_name.clone(),
        );
        self.types.insert(name.to_string(), number_type.clone());
        number_type
    }

    pub fn populate_inttypes_from_sizes(
        &mut self,
        declarations: &[IntTypeDecl],
        sizes: &[i64],
    ) -> Result<(), RffiCacheError> {
        let missing = declarations
            .iter()
            .filter(|decl| !self.types.contains_key(decl.name))
            .collect::<Vec<_>>();
        if missing.len() != sizes.len() {
            return Err(RffiCacheError::new(format!(
                "rfficache.py: expected {} sizeof answers, got {}",
                missing.len(),
                sizes.len()
            )));
        }
        for (decl, size) in missing.into_iter().zip(sizes.iter().copied()) {
            self.make_type(decl.name, decl.signed, size);
        }
        Ok(())
    }

    pub fn get(&self, name: &str) -> Option<&NumberType> {
        self.types.get(name)
    }

    pub fn numbertype_to_rclass(&self) -> &HashMap<String, String> {
        &self.numbertype_to_rclass
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IntTypeDecl {
    pub name: &'static str,
    pub c_name: &'static str,
    pub signed: bool,
}

impl IntTypeDecl {
    pub const fn new(name: &'static str, c_name: &'static str, signed: bool) -> Self {
        Self {
            name,
            c_name,
            signed,
        }
    }
}

pub fn default_includes(platform_name: &str) -> Vec<&'static str> {
    let mut includes = vec!["stdlib.h", "stdio.h", "sys/types.h"];
    if platform_name != "msvc" {
        includes.extend(["inttypes.h", "stddef.h"]);
    }
    includes
}

pub fn ask_gcc_source(question: &str, add_source: &str, platform_name: &str) -> String {
    let include_string = default_includes(platform_name)
        .into_iter()
        .map(|header| format!("#include <{header}>"))
        .collect::<Vec<_>>()
        .join("\n");
    format!(
        "// includes\n{include_string}\n\n{add_source}\n\n// checking code\nint main(void)\n{{\n   {question}\n   return (0);\n}}\n"
    )
}

pub fn sizeof_question(typenames_c: &[&str]) -> String {
    typenames_c
        .iter()
        .map(|c_typename| {
            format!("printf(\"sizeof {c_typename}=%ld\\n\", (long)sizeof({c_typename}));")
        })
        .collect::<Vec<_>>()
        .join("\n\t")
}

pub fn parse_sizeof_c_types(
    typenames_c: &[&str],
    answer: &str,
) -> Result<Vec<i64>, RffiCacheError> {
    let lines = answer.lines().collect::<Vec<_>>();
    if lines.len() != typenames_c.len() {
        return Err(RffiCacheError::new(format!(
            "rfficache.py: expected {} sizeof lines, got {}",
            typenames_c.len(),
            lines.len()
        )));
    }
    let mut result = Vec::with_capacity(typenames_c.len());
    for (line, c_typename) in lines.iter().zip(typenames_c.iter()) {
        let (label, value) = line
            .split_once('=')
            .ok_or_else(|| RffiCacheError::new(format!("rfficache.py: malformed line {line:?}")))?;
        let expected = format!("sizeof {c_typename}");
        if label != expected {
            return Err(RffiCacheError::new(format!(
                "rfficache.py: expected label {expected:?}, got {label:?}"
            )));
        }
        result.push(value.parse::<i64>().map_err(|e| {
            RffiCacheError::new(format!("rfficache.py: invalid sizeof value {value:?}: {e}"))
        })?);
    }
    Ok(result)
}

pub fn parse_sizeof_c_type(c_typename: &str, answer: &str) -> Result<i64, RffiCacheError> {
    parse_sizeof_c_types(&[c_typename], answer).map(|mut values| values.remove(0))
}

pub fn signof_question(c_typename: &str) -> String {
    format!("printf(\"sign {c_typename}=%d\\n\", (({c_typename}) -1) <= ({c_typename})0);")
}

pub fn parse_signof_c_type(c_typename: &str, answer: &str) -> Result<bool, RffiCacheError> {
    match answer.trim() {
        s if s == format!("sign {c_typename}=0") => Ok(false),
        s if s == format!("sign {c_typename}=1") => Ok(true),
        other => Err(RffiCacheError::new(format!(
            "rfficache.py: unexpected sign answer {other:?}"
        ))),
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RffiCacheError {
    pub message: String,
}

impl RffiCacheError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for RffiCacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RffiCacheError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ask_gcc_source_uses_platform_includes_like_upstream() {
        let unix = ask_gcc_source("printf(\"x\");", "typedef int foo;", "posix");
        assert!(unix.contains("#include <inttypes.h>"));
        assert!(unix.contains("#include <stddef.h>"));
        assert!(unix.contains("typedef int foo;"));
        assert!(unix.contains("printf(\"x\");"));

        let msvc = ask_gcc_source("", "", "msvc");
        assert!(!msvc.contains("#include <inttypes.h>"));
        assert!(!msvc.contains("#include <stddef.h>"));
    }

    #[test]
    fn sizeof_question_and_parser_match_rfficache_output_contract() {
        let question = sizeof_question(&["int", "long"]);
        assert_eq!(
            question,
            "printf(\"sizeof int=%ld\\n\", (long)sizeof(int));\n\tprintf(\"sizeof long=%ld\\n\", (long)sizeof(long));"
        );
        assert_eq!(
            parse_sizeof_c_types(&["int", "long"], "sizeof int=4\nsizeof long=8\n").unwrap(),
            vec![4, 8]
        );
    }

    #[test]
    fn sizeof_parser_rejects_mismatched_labels() {
        let err = parse_sizeof_c_type("int", "sizeof short=2\n").unwrap_err();
        assert!(err.message.contains("expected label"));
    }

    #[test]
    fn signof_question_and_parser_match_rfficache_output_contract() {
        assert_eq!(
            signof_question("char"),
            "printf(\"sign char=%d\\n\", ((char) -1) <= (char)0);"
        );
        assert_eq!(parse_signof_c_type("char", "sign char=1\n").unwrap(), true);
        assert_eq!(parse_signof_c_type("char", "sign char=0\n").unwrap(), false);
    }

    #[test]
    fn platform_populate_inttypes_only_fills_missing_entries() {
        let mut platform = Platform::new();
        platform.inttype_from_size("INT", true, 4);
        platform
            .populate_inttypes_from_sizes(
                &[
                    IntTypeDecl::new("INT", "int", true),
                    IntTypeDecl::new("ULONG", "unsigned long", false),
                ],
                &[8],
            )
            .unwrap();

        assert_eq!(platform.get("INT").unwrap().bits, 32);
        assert_eq!(platform.get("ULONG").unwrap().bits, 64);
        assert_eq!(
            platform.numbertype_to_rclass().get("ULONG").unwrap(),
            "r_ULONG"
        );
    }
}
