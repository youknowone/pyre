//! C spellings `database.py gettype` would give for a Rust type in an
//! `llexternal` signature.

use syn::Type;

/// Struct tags `gettype` spells as `struct <name>`.
const STRUCT_TAGS: &[&str] = &[
    "stat",
    "dirent",
    "timeval",
    "timespec",
    "pollfd",
    "sigaction",
    "tm",
    "sockaddr",
    "sockaddr_in",
    "sockaddr_in6",
    "sockaddr_un",
    "addrinfo",
    "msghdr",
    "iovec",
    "rlimit",
    "rusage",
    "statvfs",
    "utsname",
    "termios",
    "passwd",
    "group",
];

/// The C spelling `database.py gettype` would give for `ty`.
///
/// `aliases` maps a last path segment to a `type X = T` target. Paths whose
/// first segment is `libc`, `std`, `core`, or `ffi` never consult it.
pub fn c_type(ty: &Type, aliases: &dyn Fn(&str) -> Option<Type>) -> Result<String, String> {
    match ty {
        Type::Tuple(tuple) if tuple.elems.is_empty() => Ok("void".into()),
        Type::Ptr(ptr) => {
            let inner = c_type(&ptr.elem, aliases)?;
            if ptr.const_token.is_some() {
                Ok(format!("const {inner} *"))
            } else {
                Ok(format!("{inner} *"))
            }
        }
        Type::Path(type_path) if type_path.qself.is_none() => {
            let segments = &type_path.path.segments;
            let Some(first) = segments.first() else {
                return Err(format!(
                    "unsupported type for generate_macro_wrapper: {}",
                    tokens(ty)
                ));
            };
            let last = &segments.last().expect("path has a first segment").ident;
            let last = last.to_string();
            let first = first.ident.to_string();
            if matches!(first.as_str(), "libc" | "std" | "core" | "ffi") {
                return Ok(last_segment_c_type(&last));
            }
            if let Some(alias) = aliases(&last) {
                return c_type(&alias, aliases);
            }
            Ok(last_segment_c_type(&last))
        }
        Type::Paren(paren) => c_type(&paren.elem, aliases),
        Type::Group(group) => c_type(&group.elem, aliases),
        _ => Err(format!(
            "unsupported type for generate_macro_wrapper: {}",
            tokens(ty)
        )),
    }
}

fn last_segment_c_type(name: &str) -> String {
    match name {
        "i8" => "signed char".into(),
        "u8" => "unsigned char".into(),
        "i16" => "short".into(),
        "u16" => "unsigned short".into(),
        "i32" => "int".into(),
        "u32" => "unsigned int".into(),
        "i64" => "long long".into(),
        "u64" => "unsigned long long".into(),
        "isize" => "Signed".into(),
        "usize" => "Unsigned".into(),
        "f32" => "float".into(),
        "f64" => "double".into(),
        "c_char" => "char".into(),
        "c_schar" => "signed char".into(),
        "c_uchar" => "unsigned char".into(),
        "c_short" => "short".into(),
        "c_ushort" => "unsigned short".into(),
        "c_int" => "int".into(),
        "c_uint" => "unsigned int".into(),
        "c_long" => "long".into(),
        "c_ulong" => "unsigned long".into(),
        "c_longlong" => "long long".into(),
        "c_ulonglong" => "unsigned long long".into(),
        "c_float" => "float".into(),
        "c_double" => "double".into(),
        "c_void" => "void".into(),
        tag if STRUCT_TAGS.contains(&tag) => format!("struct {tag}"),
        other => other.to_string(),
    }
}

fn tokens(ty: &Type) -> String {
    quote::quote!(#ty).to_string()
}
