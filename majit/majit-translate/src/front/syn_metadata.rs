//! Syn-tree metadata harvesters over parsed interpreter source.
//!
//! Each helper walks a `syn` tree to recover file-parse metadata the
//! MIR path still sources from interpreter source: struct origins
//! (`bare → defining-module`) and the per-field register classes the
//! annotator pre-fills.  Consumers are the hybrid pre-passes in
//! `lib.rs` and `flowspace::rust_source::register`.
//!
//! Syn-free string classifiers (type-id / generic-arg projections)
//! live in [`crate::front::typestr`]; they parse no syn tree.

use quote::ToTokens;

/// Extract the declaring trait name from a `dyn T + 'a` bound list:
/// returns the first `T::Trait`-style bound's canonical path.  Used by
/// [`type_root_ident`] to key the indirect-call family.
pub(crate) fn trait_object_root_name(
    bounds: &syn::punctuated::Punctuated<syn::TypeParamBound, syn::Token![+]>,
) -> Option<String> {
    bounds.iter().find_map(|b| match b {
        syn::TypeParamBound::Trait(t) => Some(
            t.path
                .segments
                .iter()
                .map(|seg| seg.ident.to_string())
                .collect::<Vec<_>>()
                .join("::"),
        ),
        _ => None,
    })
}

/// Classify a Rust parameter/return `syn::Type` into one of the
/// RPython `lltype` register classes (`Int`/`Ref`/`Float`/`Bool`/
/// `Unsigned`).  Assigned to `OpKind::Input { ty }` so the annotator
/// + rtyper reach every function parameter with a concrete class.
pub fn classify_fn_arg_ty(ty: &syn::Type) -> crate::model::ValueType {
    use crate::model::ValueType;
    match ty {
        syn::Type::Path(path) => {
            let last = match path.path.segments.last() {
                Some(s) => s,
                None => return ValueType::Ref(None),
            };
            if path.path.segments.len() == 2
                && path.path.segments[0].ident == "Self"
                && path.path.segments[1].ident == "Truth"
            {
                return ValueType::Int;
            }
            let name = last.ident.to_string();
            if matches!(name.as_str(), "Box" | "Rc" | "Arc") {
                if let syn::PathArguments::AngleBracketed(args) = &last.arguments {
                    for arg in &args.args {
                        if let syn::GenericArgument::Type(inner) = arg {
                            return classify_fn_arg_ty(inner);
                        }
                    }
                }
                return ValueType::Ref(type_root_ident(ty));
            }
            match name.as_str() {
                "i8" | "i16" | "i32" | "i64" | "isize" | "char" => ValueType::Int,
                "u8" | "u16" | "u32" | "u64" | "usize" => ValueType::Unsigned,
                "bool" => ValueType::Bool,
                "f32" | "f64" => ValueType::Float,
                // Carry the joined path segments as diagnostic metadata
                // on the legacy tag. Precise typed pointers must be
                // attached by producers that can resolve the actual
                // HostObject/lltype identity; `valuetype_to_someshell`
                // deliberately keeps `Ref(_)` on the classdef-less
                // fallback.
                _ => ValueType::Ref(type_root_ident(ty)),
            }
        }
        syn::Type::Reference(_) => ValueType::Ref(type_root_ident(ty)),
        syn::Type::Ptr(_) => ValueType::Ref(type_root_ident(ty)),
        syn::Type::Paren(paren) => classify_fn_arg_ty(&paren.elem),
        syn::Type::Group(group) => classify_fn_arg_ty(&group.elem),
        syn::Type::TraitObject(_) => ValueType::Ref(type_root_ident(ty)),
        syn::Type::Tuple(_) | syn::Type::Array(_) | syn::Type::Slice(_) => ValueType::Ref(None),
        _ => ValueType::Ref(None),
    }
}

/// RPython lltype.Struct objects have globally unique identities;
/// returning all path segments plus qualifiers keeps `&a::Foo`,
/// `*mut a::Foo`, and `a::Foo<'x, N>` from collapsing to the same key.
pub(crate) fn type_root_ident(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(path) => render_path_type(path),
        syn::Type::Reference(reference) => {
            let inner = type_root_ident(&reference.elem)?;
            let mut out = String::from("&");
            if let Some(lifetime) = &reference.lifetime {
                out.push_str(&lifetime.to_token_stream().to_string());
                out.push(' ');
            }
            if reference.mutability.is_some() {
                out.push_str("mut ");
            }
            out.push_str(&inner);
            Some(out)
        }
        syn::Type::Ptr(ptr) => {
            let inner = type_root_ident(&ptr.elem)?;
            let mutability = if ptr.mutability.is_some() {
                "*mut"
            } else {
                "*const"
            };
            Some(format!("{mutability} {inner}"))
        }
        syn::Type::Paren(paren) => type_root_ident(&paren.elem),
        syn::Type::Group(group) => type_root_ident(&group.elem),
        syn::Type::TraitObject(obj) => {
            trait_object_root_name(&obj.bounds).map(|r| format!("dyn {r}"))
        }
        syn::Type::ImplTrait(_) => None,
        _ => None,
    }
}

fn render_path_type(path: &syn::TypePath) -> Option<String> {
    let segments: Vec<String> = path
        .path
        .segments
        .iter()
        .map(render_path_segment)
        .collect::<Option<_>>()?;
    if segments.is_empty() {
        None
    } else {
        Some(segments.join("::"))
    }
}

fn render_path_segment(seg: &syn::PathSegment) -> Option<String> {
    let ident = seg.ident.to_string();
    match &seg.arguments {
        syn::PathArguments::None => Some(ident),
        syn::PathArguments::AngleBracketed(args) => {
            let rendered_args: Vec<String> = args.args.iter().map(render_generic_arg).collect();
            Some(format!("{ident}<{}>", rendered_args.join(", ")))
        }
        syn::PathArguments::Parenthesized(args) => {
            Some(format!("{ident}{}", args.to_token_stream()))
        }
    }
}

fn render_generic_arg(arg: &syn::GenericArgument) -> String {
    match arg {
        syn::GenericArgument::Lifetime(lifetime) => lifetime.to_token_stream().to_string(),
        syn::GenericArgument::Type(ty) => {
            type_root_ident(ty).unwrap_or_else(|| ty.to_token_stream().to_string())
        }
        syn::GenericArgument::Const(value) => value.to_token_stream().to_string(),
        syn::GenericArgument::AssocType(binding) => {
            let value = type_root_ident(&binding.ty)
                .unwrap_or_else(|| binding.ty.to_token_stream().to_string());
            format!("{} = {value}", binding.ident)
        }
        syn::GenericArgument::AssocConst(binding) => {
            format!("{} = {}", binding.ident, binding.value.to_token_stream())
        }
        syn::GenericArgument::Constraint(constraint) => constraint.to_token_stream().to_string(),
        _ => arg.to_token_stream().to_string(),
    }
}
