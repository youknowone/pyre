//! Small syn parsing utility helpers.
//!
//! These are pure string-manipulation / attribute-walking helpers over
//! a parsed `syn` tree, used to harvest file-parse metadata (imports,
//! statics, struct/trait origins, return-type identities).  Consumers
//! (`jit_codewriter::call`, hybrid passes in `lib.rs`) depend on this
//! module for that metadata alone.

/// Detect the canonical `Result<T, …>` wrapper and project the inner
/// `T`.  Returns `None` for non-`Result` shapes, for `Result<(), …>`
/// (no transparent type to project), and for malformed inputs.
///
/// The only consumers live in `jit_codewriter::call`.
pub fn transparent_result_ok_type(type_str: &str) -> Option<&str> {
    let trimmed = type_str.trim();
    for prefix in ["Result<", "std::result::Result<", "core::result::Result<"] {
        let Some(inner) = trimmed
            .strip_prefix(prefix)
            .and_then(|rest| rest.strip_suffix('>'))
        else {
            continue;
        };
        let ok_type = first_top_level_generic_arg(inner).map(str::trim)?;
        if ok_type == "()" {
            return None;
        }
        return Some(ok_type);
    }
    None
}

/// Return the first comma-delimited top-level generic argument in
/// `args` (`"A, B<C, D>, E"` → `"A"`).  Tracks bracket depth so a
/// nested generic boundary does not confuse the split.
///
/// Used by [`transparent_result_ok_type`].
pub fn first_top_level_generic_arg(args: &str) -> Option<&str> {
    let mut depth = 0usize;
    for (idx, ch) in args.char_indices() {
        match ch {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => return Some(&args[..idx]),
            _ => {}
        }
    }
    if args.is_empty() { None } else { Some(args) }
}

/// Decide whether a registered `array_type_id` describes a
/// headerless item-run pointee or a length-prefixed wrapper.  Bare
/// pointers to identifier types address `items[0]` (no length word);
/// `Vec<T>` / `GcArray<T>` / `Ptr(GcArray(T))` shapes carry a length
/// header at offset 0 and therefore keep the PyPy default `False`.
///
/// Only `jit_codewriter::call` consumes it.
pub fn nolength_from_array_type_id(array_type_id: Option<&str>) -> bool {
    let Some(s) = array_type_id else {
        return false;
    };
    let mut inner = s.trim();
    loop {
        let stripped = inner
            .strip_prefix("*const ")
            .or_else(|| inner.strip_prefix("*mut "))
            .or_else(|| inner.strip_prefix("&mut "))
            .or_else(|| inner.strip_prefix('&'));
        match stripped {
            Some(rest) => inner = rest.trim_start(),
            None => break,
        }
    }
    if inner.starts_with('[') && inner.ends_with(']') {
        return true;
    }
    // Length-prefixed wrappers carry `<` (generic) or `(` (paren-style
    // lltype spelling such as `Ptr(GcArray(...))`).  Keep the PyPy
    // default `False` for those — a pointer to a wrapper still
    // dereferences a length header.
    if inner.contains('<') || inner.contains('(') {
        return false;
    }
    // Bare identifier pointee (`*const i64`, `*const Point`) means the
    // pointer addresses items[0] of a primitive / struct item type.
    // A bare identifier with NO pointer prefix is a value-type binding
    // (e.g. an `array_type_id` directly naming a struct that contains
    // an embedded array); preserve the PyPy default `False` for that.
    s.trim() != inner
}

// ─────────────────────────────────────────────────────────────────────
// Type-string rendering helpers.
//
// Pure syn-tree projections: given a `syn::Type`, render the canonical
// string identity pyre stores on `SemanticFunction.return_type`,
// `StructFieldRegistry`, and the codewriter's signature validator.  The
// `qualified_full_type_string*` family additionally consults the file's
// `prefix` (module-stripped crate-relative path) and `use_imports` table
// so the rendered identity matches the lexical resolution PyPy
// `bookkeeper.getdesc` performs through `f_globals`
// (`rpython/annotator/bookkeeper.py:353-409`).
// ─────────────────────────────────────────────────────────────────────

/// Extract the declaring trait name from a `dyn T + 'a` bound list:
/// returns the first `T::Trait`-style bound's canonical path.
/// Used by `type_root_ident` / `full_type_string` / `extract_dyn_trait_root`
/// to identify the indirect-call family key.
pub fn trait_object_root_name(
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

/// Promote a bare trait identifier to its qualified `prefix::Bare`
/// form when the qualified name is in `known_trait_names`.  Mirrors
/// the resolution PyPy `bookkeeper.py:353-409 getdesc` performs when
/// a single-frame `f_globals` lookup binds the bare name to a trait
/// declared in the same module.
pub fn qualify_known_trait_name(
    bare: &str,
    prefix: &str,
    known_trait_names: &std::collections::HashSet<String>,
) -> String {
    let qualified = if prefix.is_empty() || bare.contains("::") {
        None
    } else {
        Some(format!("{}::{}", prefix, bare))
    };
    if let Some(qualified) = qualified {
        if known_trait_names.contains(&qualified) {
            qualified
        } else {
            bare.to_string()
        }
    } else {
        bare.to_string()
    }
}

/// `trait_object_root_name` then qualified through
/// [`qualify_known_trait_name`] in one call.
pub fn trait_object_root_name_qualified(
    bounds: &syn::punctuated::Punctuated<syn::TypeParamBound, syn::Token![+]>,
    prefix: &str,
    known_trait_names: &std::collections::HashSet<String>,
) -> Option<String> {
    trait_object_root_name(bounds)
        .map(|name| qualify_known_trait_name(&name, prefix, known_trait_names))
}

/// Canonical type string for a syn::Type.
///
/// Produces a string that includes generic arguments,
/// e.g. `Vec<Point>` → `"Vec<Point>"`, `Point` → `"Point"`.
pub fn full_type_string(ty: &syn::Type) -> Option<String> {
    match ty {
        syn::Type::Path(path) => {
            let segments: Vec<String> = path
                .path
                .segments
                .iter()
                .map(|seg| {
                    let name = seg.ident.to_string();
                    match &seg.arguments {
                        syn::PathArguments::None => name,
                        syn::PathArguments::AngleBracketed(args) => {
                            let inner: Vec<String> = args
                                .args
                                .iter()
                                .filter_map(|arg| match arg {
                                    syn::GenericArgument::Type(t) => full_type_string(t),
                                    _ => None,
                                })
                                .collect();
                            if inner.is_empty() {
                                name
                            } else {
                                format!("{}<{}>", name, inner.join(","))
                            }
                        }
                        syn::PathArguments::Parenthesized(_) => name,
                    }
                })
                .collect();
            Some(segments.join("::"))
        }
        syn::Type::Reference(r) => full_type_string(&r.elem),
        syn::Type::Ptr(p) => {
            let inner = full_type_string(&p.elem)?;
            let mutability = if p.mutability.is_some() {
                "*mut"
            } else {
                "*const"
            };
            Some(format!("{mutability} {inner}"))
        }
        syn::Type::Paren(p) => full_type_string(&p.elem),
        syn::Type::Group(g) => full_type_string(&g.elem),
        syn::Type::Slice(s) => full_type_string(&s.elem).map(|t| format!("[{}]", t)),
        syn::Type::TraitObject(obj) => {
            trait_object_root_name(&obj.bounds).map(|r| format!("dyn {r}"))
        }
        // `impl Trait` is a static opaque type — render as the underlying
        // bound name without the `dyn ` prefix so downstream consumers
        // do not mistake it for a trait object (see `type_root_ident`).
        syn::Type::ImplTrait(obj) => trait_object_root_name(&obj.bounds),
        // RPython: ARRAY identity preserves full type including length.
        // [Point; 4] and [Point; 8] are different ARRAY types.
        syn::Type::Array(a) => {
            let elem = full_type_string(&a.elem)?;
            // Extract length from Expr::Lit if possible.
            let len_str = match &a.len {
                syn::Expr::Lit(lit) => match &lit.lit {
                    syn::Lit::Int(int_lit) => int_lit.base10_digits().to_string(),
                    _ => "N".to_string(),
                },
                _ => "N".to_string(),
            };
            Some(format!("[{};{}]", elem, len_str))
        }
        syn::Type::Tuple(t) if t.elems.is_empty() => Some("()".to_string()),
        syn::Type::Tuple(t) => {
            let elems: Option<Vec<String>> = t.elems.iter().map(full_type_string).collect();
            elems.map(|elems| format!("({})", elems.join(",")))
        }
        _ => None,
    }
}

/// `qualified_full_type_string` variant that consults a per-source
/// `use <path> as alias` table when qualifying single-segment leaf
/// types — keeps struct field / fn return metadata in the same name
/// namespace as `qualify_type_name_with_imports`-driven
/// parameter/local lowering, mirroring PyPy `bookkeeper.getdesc`'s
/// single-frame `f_globals` resolution
/// (`rpython/annotator/bookkeeper.py:353-409`).
///
/// `use_imports` is the per-source map collected by
/// `parse::collect_use_imports`; an empty map reduces this back to
/// `qualified_full_type_string`'s plain `prefix::Bar` /
/// `canonical_struct_name` behaviour.
pub fn qualified_full_type_string_with_imports(
    ty: &syn::Type,
    prefix: &str,
    use_imports: &std::collections::HashMap<String, String>,
    known_struct_names: &std::collections::HashSet<String>,
    known_trait_names: &std::collections::HashSet<String>,
) -> Option<String> {
    // Top-level files (`prefix=""`) still need to walk the match when a
    // per-source `use_imports` table is available; PyPy `bookkeeper.getdesc`
    // resolves bare names through the importing frame's `f_globals` even at
    // module root (`rpython/annotator/bookkeeper.py:353`).  Only fall
    // through to `full_type_string` when both qualification sources are
    // empty.
    if prefix.is_empty() && use_imports.is_empty() {
        return full_type_string(ty);
    }
    match ty {
        syn::Type::Path(path) => {
            let segments: Vec<String> = path
                .path
                .segments
                .iter()
                .map(|seg| {
                    let name = seg.ident.to_string();
                    match &seg.arguments {
                        syn::PathArguments::None => {
                            // Leaf type (no generics).  Qualify when the
                            // single-segment name is a known user struct
                            // (direct match) OR aliases to one via
                            // `use foo::Bar as B` — for the rename case
                            // `B` does not itself appear in
                            // `known_struct_names`, but the resolved
                            // target's leaf name does.  Non-struct
                            // imports (`use foo::helper` for a fn,
                            // `use external_crate::Item` for an external
                            // type) leave the bare name unqualified so
                            // their identity stays distinct from the
                            // file's own struct namespace.  PyPy
                            // `bookkeeper.getdesc(value)` binds the alias
                            // to the original Python object identity.
                            let alias_targets_struct = path.path.segments.len() == 1
                                && use_imports.get(&name).is_some_and(|full| {
                                    let leaf = full
                                        .rsplit_once("::")
                                        .map(|(_, l)| l)
                                        .unwrap_or(full.as_str());
                                    known_struct_names.contains(leaf)
                                });
                            if path.path.segments.len() == 1
                                && (known_struct_names.contains(&name) || alias_targets_struct)
                            {
                                crate::front::semantic::qualify_type_name_with_imports(
                                    &name,
                                    prefix,
                                    use_imports,
                                )
                            } else {
                                name
                            }
                        }
                        syn::PathArguments::AngleBracketed(args) => {
                            // Container<T,...> — qualify inner types, not the container.
                            let inner: Vec<String> = args
                                .args
                                .iter()
                                .filter_map(|arg| match arg {
                                    syn::GenericArgument::Type(t) => {
                                        qualified_full_type_string_with_imports(
                                            t,
                                            prefix,
                                            use_imports,
                                            known_struct_names,
                                            known_trait_names,
                                        )
                                    }
                                    _ => None,
                                })
                                .collect();
                            if inner.is_empty() {
                                name
                            } else {
                                format!("{}<{}>", name, inner.join(","))
                            }
                        }
                        syn::PathArguments::Parenthesized(_) => name,
                    }
                })
                .collect();
            Some(segments.join("::"))
        }
        syn::Type::Reference(r) => qualified_full_type_string_with_imports(
            &r.elem,
            prefix,
            use_imports,
            known_struct_names,
            known_trait_names,
        ),
        syn::Type::Ptr(p) => {
            let inner = qualified_full_type_string_with_imports(
                &p.elem,
                prefix,
                use_imports,
                known_struct_names,
                known_trait_names,
            )?;
            let mutability = if p.mutability.is_some() {
                "*mut"
            } else {
                "*const"
            };
            Some(format!("{mutability} {inner}"))
        }
        syn::Type::Paren(p) => qualified_full_type_string_with_imports(
            &p.elem,
            prefix,
            use_imports,
            known_struct_names,
            known_trait_names,
        ),
        syn::Type::Group(g) => qualified_full_type_string_with_imports(
            &g.elem,
            prefix,
            use_imports,
            known_struct_names,
            known_trait_names,
        ),
        syn::Type::Slice(s) => qualified_full_type_string_with_imports(
            &s.elem,
            prefix,
            use_imports,
            known_struct_names,
            known_trait_names,
        )
        .map(|t| format!("[{}]", t)),
        syn::Type::Array(a) => {
            let elem = qualified_full_type_string_with_imports(
                &a.elem,
                prefix,
                use_imports,
                known_struct_names,
                known_trait_names,
            )?;
            let len_str = match &a.len {
                syn::Expr::Lit(lit) => match &lit.lit {
                    syn::Lit::Int(int_lit) => int_lit.base10_digits().to_string(),
                    _ => "N".to_string(),
                },
                _ => "N".to_string(),
            };
            Some(format!("[{};{}]", elem, len_str))
        }
        syn::Type::Tuple(t) if t.elems.is_empty() => Some("()".to_string()),
        syn::Type::Tuple(t) => {
            let elems: Option<Vec<String>> = t
                .elems
                .iter()
                .map(|elem| {
                    qualified_full_type_string_with_imports(
                        elem,
                        prefix,
                        use_imports,
                        known_struct_names,
                        known_trait_names,
                    )
                })
                .collect();
            elems.map(|elems| format!("({})", elems.join(",")))
        }
        syn::Type::TraitObject(obj) => {
            trait_object_root_name_qualified(&obj.bounds, prefix, known_trait_names)
                .map(|r| format!("dyn {r}"))
        }
        // `impl Trait` is a static opaque — render the bound name without
        // the `dyn ` marker.  See `type_root_ident` for the full rationale.
        syn::Type::ImplTrait(obj) => {
            trait_object_root_name_qualified(&obj.bounds, prefix, known_trait_names)
        }
        _ => None,
    }
}

// ─────────────────────────────────────────────────────────────────────
// Top-level item walkers.
//
// Each helper does a single recursive descent over a `[syn::Item]`
// slice (including nested `mod foo { ... }` content) and accumulates
// one specific projection — struct name set, trait name set,
// `bare → defining-module-path` origin map, or
// `#[jit_immutable_fields]` attribute extraction.  PyPy parity citations
// are inline.
// ─────────────────────────────────────────────────────────────────────

/// `collect_struct_names`'s sibling for trait identifiers.  Walks
/// `Item::Trait` declarations recursively through nested `mod`s and
/// inserts both bare and `prefix::bare` qualified forms into
/// `known_trait_names`.
pub fn collect_trait_names(
    items: &[syn::Item],
    prefix: &str,
    known_trait_names: &mut std::collections::HashSet<String>,
) {
    for item in items {
        match item {
            syn::Item::Trait(trait_def) => {
                let bare_name = trait_def.ident.to_string();
                known_trait_names.insert(bare_name.clone());
                if !prefix.is_empty() {
                    known_trait_names.insert(format!("{}::{}", prefix, bare_name));
                }
            }
            syn::Item::Mod(m) => {
                if let Some((_, ref sub_items)) = m.content {
                    let mod_prefix = if prefix.is_empty() {
                        m.ident.to_string()
                    } else {
                        format!("{}::{}", prefix, m.ident)
                    };
                    collect_trait_names(sub_items, &mod_prefix, known_trait_names);
                }
            }
            _ => {}
        }
    }
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

/// Pyre-side `Class::Variant` unit-variant ctors.  These are valid
/// as bare path-expression values; `flowspace_adapter` pre-folds them
/// to `Hlvalue::Constant(ConstValue::HostObject(prebuilt_instance))`
/// before the rtyper sees a call (mirrors PyPy `rtyper` resolving
/// `SomePBC([InstanceDesc(<unit-variant>)])` to a singleton constant
/// before `jtransform`).  Exposed `pub(crate)` so
/// `translator::rtyper::flowspace_adapter::is_synthetic_unit_variant_call`
/// reads the same allowlist.
pub(crate) fn is_synthetic_unit_variant_path(segments: &[String]) -> bool {
    let path: Vec<&str> = segments.iter().map(String::as_str).collect();
    matches!(
        path.as_slice(),
        ["LoopResult", "Done"]
            | ["LoopResult", "ContinueRunningNormally"]
            | ["JitAction", "Return"]
            | ["JitAction", "Continue"]
            | ["StepResult", "Continue"]
            | ["CompareOp", "Lt"]
            | ["CompareOp", "Le"]
            | ["CompareOp", "Gt"]
            | ["CompareOp", "Ge"]
            | ["CompareOp", "Eq"]
            | ["CompareOp", "Ne"]
    )
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

/// Returns the bare trait root (no `dyn ` prefix) when `ty` denotes
/// a `dyn Trait` / `&dyn Trait` / `Box<dyn Trait>` receiver; `None`
/// otherwise.  Used by method-call lowering to decide whether the
/// call should be modeled as an RPython `indirect_call`.
pub fn extract_dyn_trait_root(ty: &syn::Type) -> Option<String> {
    extract_dyn_trait_root_with_context(ty, "", &std::collections::HashSet::new())
}

pub(crate) fn extract_dyn_trait_root_with_context(
    ty: &syn::Type,
    prefix: &str,
    known_trait_names: &std::collections::HashSet<String>,
) -> Option<String> {
    match ty {
        syn::Type::TraitObject(obj) => {
            trait_object_root_name_qualified(&obj.bounds, prefix, known_trait_names)
        }
        syn::Type::ImplTrait(_) => None,
        syn::Type::Reference(r) => {
            extract_dyn_trait_root_with_context(&r.elem, prefix, known_trait_names)
        }
        syn::Type::Paren(p) => {
            extract_dyn_trait_root_with_context(&p.elem, prefix, known_trait_names)
        }
        syn::Type::Group(g) => {
            extract_dyn_trait_root_with_context(&g.elem, prefix, known_trait_names)
        }
        syn::Type::Path(path) => {
            let last = path.path.segments.last()?;
            if !matches!(last.ident.to_string().as_str(), "Box" | "Rc" | "Arc") {
                return None;
            }
            if let syn::PathArguments::AngleBracketed(args) = &last.arguments {
                for arg in &args.args {
                    if let syn::GenericArgument::Type(inner) = arg {
                        if let Some(r) =
                            extract_dyn_trait_root_with_context(inner, prefix, known_trait_names)
                        {
                            return Some(r);
                        }
                    }
                }
            }
            None
        }
        _ => None,
    }
}
