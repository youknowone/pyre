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

/// Walk every top-level (and nested `mod`) `Item::Struct` declaration
/// in `items` and record each struct's bare name → defining module
/// path.  Mirrors PyPy `bookkeeper.getdesc(TYPE)` resolution: every
/// observed lltype STRUCT identity has a canonical home module; pyre
/// carries names as strings so this map serves the same role.
///
/// Nested `mod foo { struct Bar; }` extends the prefix to `outer::foo`
/// so the registered origin matches what `path_hash(canonical)` would
/// produce for the qualified key.  First-write-wins on duplicate bare
/// names — callers can disambiguate via use-import alias.
pub fn collect_struct_origins(
    items: &[syn::Item],
    module_prefix: &str,
    origins: &mut std::collections::HashMap<String, String>,
) {
    for item in items {
        match item {
            syn::Item::Struct(s) => {
                let bare = s.ident.to_string();
                origins
                    .entry(bare)
                    .or_insert_with(|| module_prefix.to_string());
            }
            syn::Item::Mod(m) => {
                if let Some((_, ref sub_items)) = m.content {
                    let nested = if module_prefix.is_empty() {
                        m.ident.to_string()
                    } else {
                        format!("{}::{}", module_prefix, m.ident)
                    };
                    collect_struct_origins(sub_items, &nested, origins);
                }
            }
            _ => {}
        }
    }
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
/// returning all path segments ensures `a::Foo` and `b::Foo` don't
/// alias.  Recurses through `Box`/`Rc`/`Arc` so `Box<dyn Trait>`
/// returns the trait root, not the container.
pub(crate) fn type_root_ident(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(path) => {
            if let Some(last) = path.path.segments.last() {
                let wrapper = last.ident.to_string();
                if matches!(wrapper.as_str(), "Box" | "Rc" | "Arc") {
                    if let syn::PathArguments::AngleBracketed(args) = &last.arguments {
                        for arg in &args.args {
                            if let syn::GenericArgument::Type(inner) = arg {
                                if let Some(root) = type_root_ident(inner) {
                                    return Some(root);
                                }
                            }
                        }
                    }
                }
            }
            let segments: Vec<_> = path
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect();
            if segments.is_empty() {
                None
            } else {
                Some(segments.join("::"))
            }
        }
        syn::Type::Reference(reference) => type_root_ident(&reference.elem),
        syn::Type::Ptr(ptr) => type_root_ident(&ptr.elem),
        syn::Type::Paren(paren) => type_root_ident(&paren.elem),
        syn::Type::Group(group) => type_root_ident(&group.elem),
        syn::Type::TraitObject(obj) => {
            trait_object_root_name(&obj.bounds).map(|r| format!("dyn {r}"))
        }
        syn::Type::ImplTrait(_) => None,
        _ => None,
    }
}
