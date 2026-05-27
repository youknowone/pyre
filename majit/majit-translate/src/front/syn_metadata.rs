//! Small syn-AST utility helpers extracted from `front::ast`.
//!
//! These are pure string-manipulation / attribute-walking helpers
//! that do not depend on `GraphBuildContext` or any other piece of
//! the AST graph builder.  Carving them out is a step toward
//! retiring `front::ast` (issue #97 Step 6.F): consumers outside the
//! graph builder (`jit_codewriter::call`, hybrid passes in `lib.rs`)
//! can depend on this module without pulling in 15k LOC of legacy
//! AST lowering.

/// Detect the canonical `Result<T, …>` wrapper and project the inner
/// `T`.  Returns `None` for non-`Result` shapes, for `Result<(), …>`
/// (no transparent type to project), and for malformed inputs.
///
/// `front::ast::transparent_result_ok_type` was the original home;
/// the only consumers (`jit_codewriter::call`) live outside the AST
/// graph builder so the helper moves with them.
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
/// Lives here because [`transparent_result_ok_type`] needs it; the
/// AST graph builder retains its own copy under
/// `front::ast::first_top_level_generic_arg` for its private callers
/// (`transparent_result_err_type`, `transparent_option_inner_type`).
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
/// Originally `front::ast::nolength_from_array_type_id`; only
/// `jit_codewriter::call` consumes it, and it carries no AST-builder
/// dependency.
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

/// Collect RPython-parity JIT attribute hints from a function's `attrs`
/// list, optionally augmented with `oopspec_argnames` derived from the
/// signature when an `#[oopspec(...)]` attribute is present.
///
/// Originally `front::ast::collect_jit_hints`; pure syn-attribute
/// walking with no AST graph builder dependency, so it moves with its
/// consumers in `lib.rs` / `parse.rs` / `jit_codewriter::call`.
pub fn collect_jit_hints(attrs: &[syn::Attribute], sig: Option<&syn::Signature>) -> Vec<String> {
    let mut hints = Vec::new();
    let mut saw_oopspec = false;
    for attr in attrs {
        if let Some(segment) = attr.path().segments.last() {
            let name = segment.ident.to_string();
            match name.as_str() {
                // RPython-parity names (rlib/jit.py)
                "elidable" | "jit_elidable" => hints.push("elidable".into()),
                // `rlib/jit.py:184-201 elidable_promote` is no longer
                // collapsed onto the user-facing function's hints here.
                // `build_graphs_from_items` synthesizes the
                // (`_orig_<NAME>_unlikely_name`, wrapper) pair before
                // this collector ever runs, attaches the synthetic
                // `#[elidable]` attribute to the original, and strips
                // `#[elidable_promote]` from the wrapper's `attrs`.
                // The orig is what RPython's `elidable(func)` at
                // jit.py:185 marks with `_elidable_function_`; the
                // wrapper (`result` at jit.py:198-201) carries no
                // binary flag.  `synthesize_elidable_promote_pair`
                // always succeeds — unrecognised binder patterns (which
                // Python signatures cannot express anyway) panic with
                // a citation to `jit.py:172-178 _get_args(func)`, so
                // there is no silent single-graph fallback.
                //
                // `call.py:292-299 getcalldescr` runs `_canraise(op)` on
                // every elidable callsite to recover the `EF_ELIDABLE_*`
                // 3-way split.  Pyre's `_canraise` is conservative for
                // callees outside `function_graphs` (Vec::len,
                // pyframe_get_pycode, etc.) — `analyze_external_call`
                // defaults to `True` (`call.rs:3631`), so the analyser
                // alone cannot recover `EF_ELIDABLE_CANNOT_RAISE` /
                // `EF_ELIDABLE_OR_MEMORYERROR` even on callees the user
                // has explicitly annotated.  Preserve the assertion as a
                // distinct hint string alongside the canonical
                // `"elidable"` so `lib.rs` can register it with
                // `mark_cannot_raise_assertion` /
                // `mark_memerror_only_assertion` and
                // `getcalldescr`'s elidable branch can honour the
                // user-asserted shape before falling back to
                // `_canraise`.
                "elidable_cannot_raise" => {
                    hints.push("elidable".into());
                    hints.push("elidable_cannot_raise".into());
                }
                "elidable_or_memerror" => {
                    hints.push("elidable".into());
                    hints.push("elidable_or_memerror".into());
                }
                "dont_look_inside" => hints.push("dont_look_inside".into()),
                "unroll_safe" => hints.push("unroll_safe".into()),
                "loop_invariant" | "jit_loop_invariant" => {
                    hints.push("loopinvariant".into());
                }
                "not_in_trace" => hints.push("not_in_trace".into()),
                // rlib/jit.py:250 — `@oopspec(spec)`: extract spec string.
                "oopspec" => {
                    if let Ok(lit) = attr.parse_args::<syn::LitStr>() {
                        hints.push(format!("oopspec:{}", lit.value()));
                    } else {
                        hints.push("oopspec".into());
                    }
                    saw_oopspec = true;
                }
                // majit-specific
                "jit_close_stack" => hints.push("close_stack".into()),
                "jit_cannot_collect" => hints.push("cannot_collect".into()),
                "jit_gc_effects" => hints.push("gc_effects".into()),
                _ => {}
            }
        }
    }
    // `support.py:705 argnames = ll_func.__code__.co_varnames[:nb_args]`
    // — when `#[oopspec(...)]` is present and the function signature
    // is available, emit a paired `"oopspec_argnames:arg1,arg2,..."`
    // hint so `lib.rs:598-600` can populate
    // `CallControl::mark_oopspec_argnames` alongside `mark_oopspec`.
    //
    // `support.py:713 argname2index = dict(zip(argnames, [Index(n) for n
    // in range(nb_args)]))` requires the declaration-order names.
    // Upstream's `co_varnames[:nb_args]` includes the method's
    // `self` parameter when it's a bound method (Python's
    // `co_varnames[0] = 'self'` convention), so for strict parity the
    // Rust port maps `FnArg::Receiver` to the synthetic name `"self"`.
    // Non-`Pat::Ident` patterns (tuple destructuring, wildcards) have
    // no single name to bind — upstream `co_varnames` would record
    // the destructured locals individually, but pyre's
    // `argname2index` lookup is positional and cannot multiplex one
    // slot to many names.  Emitting a `"_"` placeholder there would
    // be a deviation (the hint would shadow no real upstream
    // identifier and silently mis-bind any oopspec literal that
    // happens to spell `_`).  When such a pattern appears, refuse to
    // emit the argnames hint entirely so `decode_builtin_call` falls
    // back to the positional / bare-name path — the same behaviour
    // upstream gets when `co_varnames` is unavailable.
    if saw_oopspec {
        if let Some(sig) = sig {
            let mut argnames: Vec<String> = Vec::with_capacity(sig.inputs.len());
            let mut skip_hint = false;
            for arg in sig.inputs.iter() {
                match arg {
                    // `co_varnames[0]` for a bound method is `'self'`.
                    syn::FnArg::Receiver(_) => argnames.push("self".to_string()),
                    syn::FnArg::Typed(pat_type) => match &*pat_type.pat {
                        syn::Pat::Ident(ident) => argnames.push(ident.ident.to_string()),
                        _ => {
                            skip_hint = true;
                            break;
                        }
                    },
                }
            }
            if !skip_hint && !argnames.is_empty() {
                hints.push(format!("oopspec_argnames:{}", argnames.join(",")));
            }
        }
    }
    hints
}

/// `support.py:705 argnames = ll_func.__code__.co_varnames[:nb_args]`
/// — companion to `collect_jit_hints` that exposes the function's
/// positional parameter names when `#[oopspec(...)]` is present.
///
/// Used by the walker to populate `CallControl::oopspec_argnames`
/// (`lib.rs:598-600` consumes the `"oopspec_argnames:..."` hint
/// alongside the `"oopspec:..."` hint).  Pyre's `parse_oopspec`
/// (`support.py:701-715` port) needs argnames to resolve identifier
/// slots in the spec's `(...)` pattern to `Index(n)` placeholders.
pub fn collect_jit_hints_with_sig(attrs: &[syn::Attribute], sig: &syn::Signature) -> Vec<String> {
    collect_jit_hints(attrs, Some(sig))
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

/// RPython: lltype identity — `full_type_string` with module-prefix qualification.
///
/// RPython's `T.TO` always returns the actual lltype object.
/// This function qualifies single-segment leaf types that are KNOWN structs
/// (in `known_struct_names`) with the module prefix, so `Bar` in `mod a`
/// becomes `a::Bar`. Uses the actual struct name set, not a heuristic.
pub fn qualified_full_type_string(
    ty: &syn::Type,
    prefix: &str,
    known_struct_names: &std::collections::HashSet<String>,
    known_trait_names: &std::collections::HashSet<String>,
) -> Option<String> {
    qualified_full_type_string_with_imports(
        ty,
        prefix,
        &std::collections::HashMap::new(),
        known_struct_names,
        known_trait_names,
    )
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
        syn::Type::TraitObject(obj) => trait_object_root_name_qualified(
            &obj.bounds,
            prefix,
            known_trait_names,
        )
        .map(|r| format!("dyn {r}")),
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

/// Walk `items` (and nested `mod`s) and add every `Item::Struct`'s
/// bare name plus `prefix::bare` qualified form to `known_struct_names`.
/// Used by `collect_program_metadata_pub` to build the program-wide
/// set of struct identifiers `qualified_full_type_string_with_imports`
/// keys off of.
pub fn collect_struct_names(
    items: &[syn::Item],
    prefix: &str,
    known_struct_names: &mut std::collections::HashSet<String>,
) {
    for item in items {
        match item {
            syn::Item::Struct(s) => {
                let bare_name = s.ident.to_string();
                known_struct_names.insert(bare_name.clone());
                if !prefix.is_empty() {
                    known_struct_names.insert(format!("{}::{}", prefix, bare_name));
                }
            }
            syn::Item::Mod(m) => {
                if let Some((_, ref sub_items)) = m.content {
                    let mod_prefix = if prefix.is_empty() {
                        m.ident.to_string()
                    } else {
                        format!("{}::{}", prefix, m.ident)
                    };
                    collect_struct_names(sub_items, &mod_prefix, known_struct_names);
                }
            }
            _ => {}
        }
    }
}

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

/// Read `#[jit_immutable_fields("a", "b?", "c[*]", "d?[*]")]` attributes
/// off a struct declaration and return the declared field names paired
/// with their [`crate::model::ImmutableRank`].  Bare idents
/// (`#[jit_immutable_fields(a, b)]`) remain accepted as
/// `ImmutableRank::Immutable` for backward compatibility.
///
/// Multiple attributes accumulate; non-recognised tokens are silently
/// skipped (matching `syn::Meta::parse` looseness).  Rank suffix encoding
/// follows RPython `rpython/rtyper/rclass.py:644-678 _parse_field_list`.
pub fn collect_immutable_field_attrs(
    attrs: &[syn::Attribute],
) -> Vec<(String, crate::model::ImmutableRank)> {
    use crate::model::ImmutableRank;
    use syn::punctuated::Punctuated;
    use syn::{Expr, ExprLit, ExprPath, Lit, Token};

    let mut specs = Vec::new();
    for attr in attrs {
        let Some(ident) = attr.path().get_ident() else {
            continue;
        };
        if ident != "jit_immutable_fields" {
            continue;
        }
        let parsed = attr.parse_args_with(Punctuated::<Expr, Token![,]>::parse_terminated);
        let Ok(items) = parsed else {
            continue;
        };
        for item in items {
            match item {
                Expr::Lit(ExprLit {
                    lit: Lit::Str(s), ..
                }) => {
                    specs.push(ImmutableRank::parse(&s.value()));
                }
                Expr::Path(ExprPath { path, .. }) => {
                    if let Some(id) = path.get_ident() {
                        specs.push((id.to_string(), ImmutableRank::Immutable));
                    }
                }
                _ => {}
            }
        }
    }
    specs
}

/// `syn::Path` shape probe — returns `Some(ident)` when the path is a
/// bare single-segment identifier with no leading colon and no generic
/// arguments. Used by call-site canonicalisation to detect bare
/// function-name expressions like `helper` vs qualified `mod::helper`.
pub(crate) fn path_as_single_ident(path: &syn::Path) -> Option<&syn::Ident> {
    if path.leading_colon.is_none() && path.segments.len() == 1 {
        let seg = &path.segments[0];
        if matches!(seg.arguments, syn::PathArguments::None) {
            return Some(&seg.ident);
        }
    }
    None
}

/// Returns `true` when the binary operator is one of Rust's compound
/// assignment forms (`+=`, `-=`, ...). Mirrors the lowering decision
/// to emit a `read-then-write` pair vs a plain `Assign`.
pub(crate) fn is_compound_assign(op: syn::BinOp) -> bool {
    matches!(
        op,
        syn::BinOp::AddAssign(_)
            | syn::BinOp::SubAssign(_)
            | syn::BinOp::MulAssign(_)
            | syn::BinOp::DivAssign(_)
            | syn::BinOp::RemAssign(_)
            | syn::BinOp::BitXorAssign(_)
            | syn::BinOp::BitAndAssign(_)
            | syn::BinOp::BitOrAssign(_)
            | syn::BinOp::ShlAssign(_)
            | syn::BinOp::ShrAssign(_)
    )
}

/// Lowering-side name for a `syn::UnOp`. Returns `"unknown_unary"` for
/// non-exhaustive future syn variants.
pub(crate) fn unary_op_name(op: &syn::UnOp) -> &'static str {
    match op {
        syn::UnOp::Deref(_) => "deref",
        syn::UnOp::Not(_) => "not",
        syn::UnOp::Neg(_) => "neg",
        _ => "unknown_unary",
    }
}

/// Lowering-side name for a `syn::BinOp`. Returns `"unknown_binop"` for
/// non-exhaustive future syn variants.
pub(crate) fn binary_op_name(op: &syn::BinOp) -> &'static str {
    match op {
        syn::BinOp::Add(_) => "add",
        syn::BinOp::Sub(_) => "sub",
        syn::BinOp::Mul(_) => "mul",
        syn::BinOp::Div(_) => "div",
        syn::BinOp::Rem(_) => "mod",
        syn::BinOp::And(_) => "and",
        syn::BinOp::Or(_) => "or",
        syn::BinOp::BitXor(_) => "bitxor",
        syn::BinOp::BitAnd(_) => "bitand",
        syn::BinOp::BitOr(_) => "bitor",
        syn::BinOp::Shl(_) => "lshift",
        syn::BinOp::Shr(_) => "rshift",
        syn::BinOp::Eq(_) => "eq",
        syn::BinOp::Lt(_) => "lt",
        syn::BinOp::Le(_) => "le",
        syn::BinOp::Ne(_) => "ne",
        syn::BinOp::Ge(_) => "ge",
        syn::BinOp::Gt(_) => "gt",
        syn::BinOp::AddAssign(_) => "add_assign",
        syn::BinOp::SubAssign(_) => "sub_assign",
        syn::BinOp::MulAssign(_) => "mul_assign",
        syn::BinOp::DivAssign(_) => "div_assign",
        syn::BinOp::RemAssign(_) => "mod_assign",
        syn::BinOp::BitXorAssign(_) => "bitxor_assign",
        syn::BinOp::BitAndAssign(_) => "bitand_assign",
        syn::BinOp::BitOrAssign(_) => "bitor_assign",
        syn::BinOp::ShlAssign(_) => "lshift_assign",
        syn::BinOp::ShrAssign(_) => "rshift_assign",
        _ => "unknown_binop",
    }
}

/// Field accessor name from a `syn::Member` — `.foo` returns `"foo"`,
/// `.0` returns `"0"`.
pub(crate) fn member_name(member: &syn::Member) -> String {
    match member {
        syn::Member::Named(ident) => ident.to_string(),
        syn::Member::Unnamed(idx) => idx.index.to_string(),
    }
}

/// Splits a macro-argument `TokenStream` at the first top-level
/// `,` punctuation (spacing == Alone), returning `(prefix, suffix)`.
/// Returns `None` when no top-level comma exists. Used by macro
/// recognisers that take a head expression followed by a `pat [if guard]`
/// tail.
pub(crate) fn split_macro_args_at_first_top_comma(
    tokens: proc_macro2::TokenStream,
) -> Option<(proc_macro2::TokenStream, proc_macro2::TokenStream)> {
    let mut prefix: Vec<proc_macro2::TokenTree> = Vec::new();
    let mut iter = tokens.into_iter();
    for tt in iter.by_ref() {
        if let proc_macro2::TokenTree::Punct(ref p) = tt {
            if p.as_char() == ',' && p.spacing() == proc_macro2::Spacing::Alone {
                let suffix: proc_macro2::TokenStream = iter.collect();
                return Some((prefix.into_iter().collect(), suffix));
            }
        }
        prefix.push(tt);
    }
    None
}

// ── `match`-arm pattern classifiers ──────────────────────────────────
//
// Used by the AST lowering of `match`-on-bool and `match`-on-integer
// scrutinees to extract per-arm `ExitCase` data. All helpers below
// inspect `syn::Arm` / `syn::Pat` without touching the graph builder,
// so they live in this module alongside the other syn-walk helpers.

/// `syn::parse::Parser` adapter for the `pat [if guard]` tail of a
/// `matches!` invocation. Mirrors the std `matches!` macro grammar
/// (`std::macros::matches`): a single `Pat`, optionally followed by
/// `if Expr`. The `Pat` parse uses `Pat::parse_multi_with_leading_vert`
/// so top-level `|` alternations (`Some(_) | None`) are accepted.
pub(crate) fn parse_matches_pat_and_guard(
    input: syn::parse::ParseStream,
) -> syn::Result<(syn::Pat, Option<syn::Expr>)> {
    let pat = syn::Pat::parse_multi_with_leading_vert(input)?;
    let guard = if input.peek(syn::Token![if]) {
        let _: syn::Token![if] = input.parse()?;
        Some(input.parse::<syn::Expr>()?)
    } else {
        None
    };
    Ok((pat, guard))
}

/// Classify the arms of a `match` whose scrutinee is bool-typed.
/// Returns `Some(per-arm exit cases)` when the arms exhaustively cover
/// `{false, true}` with no `if`-guards; otherwise `None`.
pub(crate) fn classify_match_bool_exitcases(
    arms: &[syn::Arm],
) -> Option<Vec<Vec<crate::model::ExitCase>>> {
    use crate::model::ExitCase;
    if arms.len() != 2 {
        return None;
    }
    let mut all = Vec::with_capacity(arms.len());
    let mut seen_bools = Vec::<bool>::new();
    for (arm_idx, arm) in arms.iter().enumerate() {
        if arm.guard.is_some() {
            return None;
        }
        let is_last_arm = arm_idx + 1 == arms.len();
        let mut sub_pats = Vec::new();
        flatten_or_pattern(&arm.pat, &mut sub_pats);
        if sub_pats.len() > 1 && sub_pats.iter().any(|pat| matches!(pat, syn::Pat::Wild(_))) {
            return None;
        }
        let mut arm_cases = Vec::with_capacity(sub_pats.len());
        for (sub_idx, sub_pat) in sub_pats.iter().enumerate() {
            let is_last = is_last_arm && sub_idx + 1 == sub_pats.len();
            match classify_bool_pattern(sub_pat, is_last, &seen_bools)? {
                BoolPatternCase::Value(value) => {
                    if seen_bools.contains(&value) {
                        return None;
                    }
                    seen_bools.push(value);
                    arm_cases.push(ExitCase::Bool(value));
                }
                BoolPatternCase::Default(values) => {
                    if values.is_empty() {
                        return None;
                    }
                    for value in values {
                        seen_bools.push(value);
                        arm_cases.push(ExitCase::Bool(value));
                    }
                }
            }
        }
        all.push(arm_cases);
    }
    let mut sorted = seen_bools;
    sorted.sort();
    sorted.dedup();
    if sorted.as_slice() == &[false, true] {
        Some(all)
    } else {
        None
    }
}

pub(crate) enum BoolPatternCase {
    Value(bool),
    Default(Vec<bool>),
}

pub(crate) fn classify_bool_pattern(
    pat: &syn::Pat,
    is_last: bool,
    seen_bools: &[bool],
) -> Option<BoolPatternCase> {
    match pat {
        syn::Pat::Wild(_) if is_last => {
            let mut values = Vec::new();
            if !seen_bools.contains(&false) {
                values.push(false);
            }
            if !seen_bools.contains(&true) {
                values.push(true);
            }
            Some(BoolPatternCase::Default(values))
        }
        syn::Pat::Wild(_) => None,
        syn::Pat::Lit(lit) => match &lit.lit {
            syn::Lit::Bool(value) => Some(BoolPatternCase::Value(value.value)),
            _ => None,
        },
        _ => None,
    }
}

/// Classify the arms of a `match` whose scrutinee is integer-typed.
/// Returns `Some(per-arm exit cases)` when arms are exhaustive (last
/// arm may be `_` wildcard for the default) with no `if`-guards.
pub(crate) fn classify_match_switch_exitcases(
    arms: &[syn::Arm],
) -> Option<Vec<Vec<crate::model::ExitCase>>> {
    use crate::model::ExitCase;
    if arms.len() < 2 {
        return None;
    }
    let mut all = Vec::with_capacity(arms.len());
    let mut seen = Vec::<ExitCase>::new();
    for (arm_idx, arm) in arms.iter().enumerate() {
        if arm.guard.is_some() {
            return None;
        }
        let is_last_arm = arm_idx + 1 == arms.len();
        let mut sub_pats = Vec::new();
        flatten_or_pattern(&arm.pat, &mut sub_pats);
        if sub_pats.len() > 1 && sub_pats.iter().any(|pat| matches!(pat, syn::Pat::Wild(_))) {
            return None;
        }
        let mut arm_cases = Vec::with_capacity(sub_pats.len());
        for (sub_idx, sub_pat) in sub_pats.iter().enumerate() {
            let is_last = is_last_arm && sub_idx + 1 == sub_pats.len();
            let exitcase = classify_switch_pattern(sub_pat, is_last)?;
            if !is_default_exitcase(&exitcase) {
                if seen.iter().any(|existing| existing == &exitcase) {
                    return None;
                }
                seen.push(exitcase.clone());
            }
            arm_cases.push(exitcase);
        }
        all.push(arm_cases);
    }
    Some(all)
}

pub(crate) fn flatten_or_pattern<'a>(pat: &'a syn::Pat, out: &mut Vec<&'a syn::Pat>) {
    match pat {
        syn::Pat::Or(or_pat) => {
            for case in &or_pat.cases {
                flatten_or_pattern(case, out);
            }
        }
        syn::Pat::Paren(paren) => flatten_or_pattern(&paren.pat, out),
        other => out.push(other),
    }
}

pub(crate) fn classify_switch_pattern(
    pat: &syn::Pat,
    is_last: bool,
) -> Option<crate::model::ExitCase> {
    use crate::flowspace::model::ConstValue;
    use crate::model::ExitCase;
    match pat {
        syn::Pat::Wild(_) if is_last => Some(default_exitcase()),
        syn::Pat::Wild(_) => None,
        syn::Pat::Lit(lit) => match &lit.lit {
            syn::Lit::Int(int_lit) => int_lit
                .base10_parse::<i64>()
                .ok()
                .map(|value| ExitCase::Const(ConstValue::Int(value))),
            syn::Lit::Byte(value) => Some(ExitCase::Const(ConstValue::Int(value.value() as i64))),
            syn::Lit::Char(value) => Some(ExitCase::Const(ConstValue::Int(value.value() as i64))),
            _ => None,
        },
        _ => None,
    }
}

pub(crate) fn default_exitcase() -> crate::model::ExitCase {
    crate::model::ExitCase::Const(crate::flowspace::model::ConstValue::byte_str("default"))
}

pub(crate) fn is_default_exitcase(exitcase: &crate::model::ExitCase) -> bool {
    use crate::flowspace::model::ConstValue;
    use crate::model::ExitCase;
    matches!(
        exitcase,
        ExitCase::Const(ConstValue::ByteStr(bytes)) if bytes.as_slice() == b"default"
    )
}

// ── `#[elidable_promote]` synthesis (`rlib/jit.py:180-201`) ─────────
//
// Pure syn-tree transformation: takes a `syn::ItemFn` and expands it
// into the (orig, wrapper) pair when the source carries
// `#[elidable_promote]` / `#[purefunction_promote]`, otherwise returns
// the source unchanged. The graph builder is not involved.

/// `promote_args` selector for `#[elidable_promote(...)]` synthesis.
///
/// Mirrors `rlib/jit.py:189-191` — RPython splits a literal string at
/// `,` when the value is not the special `"all"` marker:
///
/// ```python
/// if promote_args != 'all':
///     args = [args[int(i)] for i in promote_args.split(",")]
/// ```
#[derive(Debug, Clone)]
pub(crate) enum PromoteArgsSelector {
    /// `jit.py:180` default `promote_args='all'` — every non-self
    /// positional arg flows through `hint(..., promote=True)`.
    All,
    /// `jit.py:189-191` index list (`"0,2"` → `[0, 2]`).  Indices are
    /// 0-based and point into the **positional arg list including
    /// `self`** when the decorated function is a method (RPython's
    /// `_get_args(func)` reads the raw co_varnames; pyre mirrors the
    /// same convention by treating `self` as index 0 when present).
    Indices(Vec<usize>),
}

/// `rlib/jit.py:180-191` — parse the literal attribute value attached
/// to `#[elidable_promote(promote_args = "...")]`.  Bare
/// `#[elidable_promote]` defaults to `"all"` per the upstream default
/// argument at jit.py:180.
fn parse_elidable_promote_args(attr: &syn::Attribute) -> syn::Result<PromoteArgsSelector> {
    if matches!(attr.meta, syn::Meta::Path(_)) {
        return Ok(PromoteArgsSelector::All);
    }
    let mut selector = None;
    attr.parse_nested_meta(|meta| {
        if meta.path.is_ident("promote_args") {
            let value = meta.value()?;
            let lit: syn::LitStr = value.parse()?;
            selector = Some(if lit.value() == "all" {
                PromoteArgsSelector::All
            } else {
                let mut indices = Vec::new();
                for piece in lit.value().split(',') {
                    indices.push(piece.trim().parse::<usize>().map_err(|err| {
                        syn::Error::new(lit.span(), format!("promote_args: {err}"))
                    })?);
                }
                PromoteArgsSelector::Indices(indices)
            });
            Ok(())
        } else {
            Err(meta.error("unsupported elidable_promote argument"))
        }
    })?;
    Ok(selector.unwrap_or(PromoteArgsSelector::All))
}

/// Expand a `syn::ItemFn` into `[orig, wrapper]` when it carries
/// `#[elidable_promote]` / `#[purefunction_promote]`, else return the
/// single function unchanged.  Mirrors `rlib/jit.py:184-201`'s "module
/// import installs two callables" semantics at every entry point that
/// hands a `syn::ItemFn` to `build_function_graph*` — free functions
/// (`build_graphs_from_items`), inherent / trait-impl methods
/// (`parse::extract_inherent_impl_methods`,
/// `parse::extract_trait_impls`), and trait default methods
/// (`parse::extract_trait_impls`).  Centralising the expansion here
/// keeps the decorator behaviour uniform across all four lowering
/// surfaces.
///
/// `impl_type` carries the qualified type root (`"S"`, `"a::S"`) when
/// the source item lives inside an `impl` block, so the synthesizer can
/// emit a type-qualified tail call (`a::S::_orig_<name>_unlikely_name(
/// self, args)`) that matches the impl-method registration path built
/// by `parse::CallPath::for_impl_method` at `lib.rs:531-537`.  Free
/// functions and trait-default methods (which have no concrete `Self`
/// type at synthesis time) pass `None` and fall back to the bare-path
/// tail call.
pub fn synthesize_or_passthrough(
    fake_fn: syn::ItemFn,
    impl_type: Option<&str>,
) -> Vec<syn::ItemFn> {
    match extract_elidable_promote_selector(&fake_fn.attrs) {
        Some(selector) => {
            let (orig, wrapper) = synthesize_elidable_promote_pair(&fake_fn, &selector, impl_type);
            vec![orig, wrapper]
        }
        None => vec![fake_fn],
    }
}

/// Locate `#[elidable_promote]` (or its deprecated `#[purefunction_promote]`
/// alias from `rlib/jit.py:203-205`) and return its parsed
/// `promote_args` selector, or `None` if neither attribute is present.
///
/// `rlib/jit.py:189-191` `args[int(i)]` propagates `ValueError` /
/// `IndexError` to the caller — the decorator does not silently drop
/// malformed input.  Pyre mirrors that fail-loud behaviour here: a
/// malformed `promote_args = "..."` literal panics with the
/// underlying `syn::Error` rather than falling through.
fn extract_elidable_promote_selector(attrs: &[syn::Attribute]) -> Option<PromoteArgsSelector> {
    for attr in attrs {
        if let Some(segment) = attr.path().segments.last() {
            let name = segment.ident.to_string();
            if name == "elidable_promote" || name == "purefunction_promote" {
                return Some(parse_elidable_promote_args(attr).unwrap_or_else(|err| {
                    panic!("#[{name}(...)]: {err}");
                }));
            }
        }
    }
    None
}

/// Synthesize the wrapper / original function pair from a single
/// `#[elidable_promote] fn foo(...)` source item.
///
/// Line-by-line port of `rlib/jit.py:184-201`.
fn synthesize_elidable_promote_pair(
    func: &syn::ItemFn,
    selector: &PromoteArgsSelector,
    impl_type: Option<&str>,
) -> (syn::ItemFn, syn::ItemFn) {
    use quote::format_ident;
    // jit.py:186 args = _get_args(func) — positional names, self included.
    let arg_names: Vec<syn::Ident> = func
        .sig
        .inputs
        .iter()
        .map(|arg| match arg {
            syn::FnArg::Typed(pt) => match &*pt.pat {
                syn::Pat::Ident(pi) => pi.ident.clone(),
                _ => panic!(
                    "#[elidable_promote] on `fn {}`: unsupported binder \
                     pattern in arg position — RPython `_get_args(func)` \
                     (jit.py:172-178) reads positional names off \
                     `co_varnames`, which never include destructured \
                     binders.  Rewrite the parameter as a plain `name: \
                     Ty` instead.",
                    func.sig.ident
                ),
            },
            // `&self` / `&mut self` map to a positional name "self" so
            // index 0 in `Indices` continues to address the receiver as
            // RPython does (`_get_args(func)` reads co_varnames raw).
            syn::FnArg::Receiver(_) => format_ident!("self"),
        })
        .collect();

    let orig_ident = format_ident!("_orig_{}_unlikely_name", func.sig.ident);

    // jit.py:184-185 — original keeps the body, gains `_elidable_function_`.
    let orig_attrs: Vec<syn::Attribute> = func
        .attrs
        .iter()
        .filter(|a| {
            let name = a
                .path()
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default();
            name != "elidable_promote" && name != "purefunction_promote"
        })
        .cloned()
        .collect();
    let elidable_attr: syn::Attribute = syn::parse_quote!(#[elidable]);
    let mut orig_fn = func.clone();
    orig_fn.attrs = orig_attrs;
    orig_fn.attrs.push(elidable_attr);
    orig_fn.sig.ident = orig_ident.clone();

    // jit.py:189-191 — promote_args=='all' → all indices; else parsed list.
    // RPython `args[int(i)]` raises `IndexError` on out-of-range
    // indices; pyre fails loudly here for the same reason.
    let promote_indices: Vec<usize> = match selector {
        PromoteArgsSelector::All => (0..arg_names.len()).collect(),
        PromoteArgsSelector::Indices(ix) => {
            for &i in ix {
                if i >= arg_names.len() {
                    panic!(
                        "#[elidable_promote(promote_args = ...)] on `fn {}`: \
                         index {} is out of range for a {}-arg function \
                         (jit.py:191 `args[int(i)]` would IndexError)",
                        func.sig.ident,
                        i,
                        arg_names.len()
                    );
                }
            }
            ix.clone()
        }
    };
    // jit.py:191-194 — `for arg in args: hint(arg, promote=True,
    // promote_string=True)`.
    let promote_self = promote_indices.iter().any(|&i| arg_names[i] == "self");
    let promote_stmts: Vec<syn::Stmt> = promote_indices
        .iter()
        .map(|&i| {
            let id = &arg_names[i];
            if id == "self" {
                syn::parse_quote!(let __self_promoted = hint_promote_or_string(self);)
            } else {
                syn::parse_quote!(let #id = hint_promote_or_string(#id);)
            }
        })
        .collect();

    // jit.py:195 — return _orig_func_unlikely_name(args).
    let call_args = arg_names.iter().map(|id| -> syn::Expr {
        if id == "self" {
            if promote_self {
                syn::parse_quote!(__self_promoted)
            } else {
                syn::parse_quote!(self)
            }
        } else {
            syn::parse_quote!(#id)
        }
    });
    let tail_call: syn::Expr = match impl_type {
        Some(ty_str) => {
            let ty_path: syn::Path = syn::parse_str(ty_str).unwrap_or_else(|err| {
                panic!(
                    "synthesize_elidable_promote_pair: failed to parse impl type `{ty_str}`: {err}"
                )
            });
            syn::parse_quote!(#ty_path::#orig_ident(#(#call_args),*))
        }
        None => syn::parse_quote!(#orig_ident(#(#call_args),*)),
    };
    let tail_stmt = syn::Stmt::Expr(tail_call, None);

    // jit.py:198-201 — wrapper is the user-facing decorated name with
    // the promote+forward body; `#[elidable_promote]` is stripped so
    // `collect_jit_hints` does not register the wrapper itself as
    // elidable (the binary flag belongs on the original alone, per
    // jit.py:185 `elidable(func)`).
    let wrapper_attrs: Vec<syn::Attribute> = func
        .attrs
        .iter()
        .filter(|a| {
            let name = a
                .path()
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default();
            name != "elidable_promote" && name != "purefunction_promote"
        })
        .cloned()
        .collect();
    let mut wrapper_block = syn::Block {
        brace_token: Default::default(),
        stmts: Vec::new(),
    };
    wrapper_block.stmts.extend(promote_stmts);
    wrapper_block.stmts.push(tail_stmt);

    let mut wrapper_fn = func.clone();
    wrapper_fn.attrs = wrapper_attrs;
    wrapper_fn.block = Box::new(wrapper_block);

    (orig_fn, wrapper_fn)
}

// ── Loop-header phi pre-scan ────────────────────────────────────────
//
// Static walk of a `while` / `loop` / `for` body to partition the
// referenced names into reads and rebinds. The loop-header eager phi
// allocator consumes this to pre-allocate inputarg slots in the header
// block. No graph builder state is involved.

/// Pre-scan output: names referenced as reads (`LOAD_FAST` equivalent)
/// and names rebound (`STORE_FAST` equivalent) inside the body.
///
/// RPython parity: there is no single-line counterpart in
/// `flowspace/flowcontext.py` because RPython infers reads/writes from
/// `LOAD_FAST` / `STORE_FAST` bytecodes implicit in the per-bytecode
/// dispatch (`flowcontext.py:780-820`).  This pre-scan is the
/// static-AST analogue, computing the must-merge name set ahead of the
/// loop walk so the eager allocator can replace RPython's iterative
/// `mergeblock` widening at `flowcontext.py:430` with a single pass.
#[derive(Debug, Default, Clone)]
pub(crate) struct LoopBodyLocals {
    pub(crate) read_names: std::collections::HashSet<String>,
    pub(crate) rebound_names: std::collections::HashSet<String>,
}

/// Statically scan `body` and partition every locally-referenced name
/// into `read_names` and `rebound_names` for the eager loop-header
/// phi allocator.  See [`LoopBodyLocals`] for the exact contract.
pub(crate) fn loop_body_locals(body: &syn::Block) -> LoopBodyLocals {
    let mut state = LoopBodyLocals::default();
    visit_block_for_loop_locals(body, &mut state);
    state
}

fn visit_block_for_loop_locals(block: &syn::Block, state: &mut LoopBodyLocals) {
    for stmt in &block.stmts {
        visit_stmt_for_loop_locals(stmt, state);
    }
}

fn visit_stmt_for_loop_locals(stmt: &syn::Stmt, state: &mut LoopBodyLocals) {
    match stmt {
        syn::Stmt::Local(local) => {
            // `let pat [: ty] [= init];` — pattern bindings are
            // rebinds.  Inner-scope-only bindings are filtered by the
            // allocator: a name not present in the pre-loop framestate
            // is not a header phi candidate.
            collect_pat_idents(&local.pat, &mut state.rebound_names);
            if let Some(init) = &local.init {
                visit_expr_for_loop_locals(&init.expr, state);
                if let Some((_, diverge)) = &init.diverge {
                    visit_expr_for_loop_locals(diverge, state);
                }
            }
        }
        syn::Stmt::Expr(expr, _semi) => visit_expr_for_loop_locals(expr, state),
        syn::Stmt::Item(_) => {
            // `fn`, `struct`, `use`, etc. — items live in their own scope.
        }
        syn::Stmt::Macro(_) => {
            // Macro invocations cannot be statically introspected for local
            // references.  Conservatively skip; if a macro mutates a local
            // that later participates in the back-edge, the `Link.__init__`
            // arity check at build time fails loud.
        }
    }
}

fn visit_expr_for_loop_locals(expr: &syn::Expr, state: &mut LoopBodyLocals) {
    match expr {
        syn::Expr::Path(p) => {
            if let Some(ident) = path_as_single_ident(&p.path) {
                state.read_names.insert(ident.to_string());
            }
        }
        syn::Expr::Assign(a) => {
            // Simple LHS `name = rhs` is a pure rebind; complex LHS
            // (e.g. `obj.field = ...`, `arr[i] = ...`) reads the
            // base/index instead.
            if let syn::Expr::Path(p) = a.left.as_ref()
                && let Some(ident) = path_as_single_ident(&p.path)
            {
                state.rebound_names.insert(ident.to_string());
            } else {
                visit_expr_for_loop_locals(&a.left, state);
            }
            visit_expr_for_loop_locals(&a.right, state);
        }
        syn::Expr::Binary(b) => {
            // Compound assign (`a += b`, `a |= b`, …) lowers to
            // `Expr::Binary` with one of the `BinOp::*Assign(_)`
            // variants; a simple-ident LHS is BOTH read and rebound.
            if is_compound_assign(b.op) {
                if let syn::Expr::Path(p) = b.left.as_ref()
                    && let Some(ident) = path_as_single_ident(&p.path)
                {
                    let name = ident.to_string();
                    state.read_names.insert(name.clone());
                    state.rebound_names.insert(name);
                } else {
                    visit_expr_for_loop_locals(&b.left, state);
                }
            } else {
                visit_expr_for_loop_locals(&b.left, state);
            }
            visit_expr_for_loop_locals(&b.right, state);
        }
        syn::Expr::Block(b) => visit_block_for_loop_locals(&b.block, state),
        syn::Expr::Unsafe(u) => visit_block_for_loop_locals(&u.block, state),
        syn::Expr::Async(a) => visit_block_for_loop_locals(&a.block, state),
        syn::Expr::If(e) => {
            visit_expr_for_loop_locals(&e.cond, state);
            visit_block_for_loop_locals(&e.then_branch, state);
            if let Some((_, else_branch)) = &e.else_branch {
                visit_expr_for_loop_locals(else_branch, state);
            }
        }
        syn::Expr::Match(m) => {
            visit_expr_for_loop_locals(&m.expr, state);
            for arm in &m.arms {
                if let Some((_, guard)) = &arm.guard {
                    visit_expr_for_loop_locals(guard, state);
                }
                visit_expr_for_loop_locals(&arm.body, state);
            }
        }
        syn::Expr::While(e) => {
            visit_expr_for_loop_locals(&e.cond, state);
            visit_block_for_loop_locals(&e.body, state);
        }
        syn::Expr::Loop(e) => visit_block_for_loop_locals(&e.body, state),
        syn::Expr::ForLoop(e) => {
            // The for-loop's pat introduces a new inner-scope binding;
            // the allocator filters it out by intersecting against the
            // pre-loop framestate.  Visit iterable for reads + body
            // for any rebinds of *outer* locals.
            visit_expr_for_loop_locals(&e.expr, state);
            visit_block_for_loop_locals(&e.body, state);
        }
        syn::Expr::Let(l) => {
            visit_expr_for_loop_locals(&l.expr, state);
        }
        syn::Expr::Closure(_) => {
            // Skip closure body.  A closure captures enclosing locals via a
            // synthetic capture list; those captures do not flow through the
            // loop's straight-line control path, so bindings inside the
            // closure must not influence the header phi set.
        }
        syn::Expr::Call(c) => {
            visit_expr_for_loop_locals(&c.func, state);
            for arg in &c.args {
                visit_expr_for_loop_locals(arg, state);
            }
        }
        syn::Expr::MethodCall(m) => {
            visit_expr_for_loop_locals(&m.receiver, state);
            for arg in &m.args {
                visit_expr_for_loop_locals(arg, state);
            }
        }
        syn::Expr::Field(f) => visit_expr_for_loop_locals(&f.base, state),
        syn::Expr::Index(i) => {
            visit_expr_for_loop_locals(&i.expr, state);
            visit_expr_for_loop_locals(&i.index, state);
        }
        syn::Expr::Reference(r) => visit_expr_for_loop_locals(&r.expr, state),
        syn::Expr::Unary(u) => visit_expr_for_loop_locals(&u.expr, state),
        syn::Expr::Paren(p) => visit_expr_for_loop_locals(&p.expr, state),
        syn::Expr::Try(t) => visit_expr_for_loop_locals(&t.expr, state),
        syn::Expr::TryBlock(t) => visit_block_for_loop_locals(&t.block, state),
        syn::Expr::Tuple(t) => {
            for elem in &t.elems {
                visit_expr_for_loop_locals(elem, state);
            }
        }
        syn::Expr::Cast(c) => visit_expr_for_loop_locals(&c.expr, state),
        syn::Expr::Array(a) => {
            for elem in &a.elems {
                visit_expr_for_loop_locals(elem, state);
            }
        }
        syn::Expr::Range(r) => {
            if let Some(s) = &r.start {
                visit_expr_for_loop_locals(s, state);
            }
            if let Some(e) = &r.end {
                visit_expr_for_loop_locals(e, state);
            }
        }
        syn::Expr::Return(r) => {
            if let Some(e) = &r.expr {
                visit_expr_for_loop_locals(e, state);
            }
        }
        syn::Expr::Break(b) => {
            if let Some(e) = &b.expr {
                visit_expr_for_loop_locals(e, state);
            }
        }
        syn::Expr::Yield(y) => {
            if let Some(e) = &y.expr {
                visit_expr_for_loop_locals(e, state);
            }
        }
        syn::Expr::Await(a) => visit_expr_for_loop_locals(&a.base, state),
        syn::Expr::Group(g) => visit_expr_for_loop_locals(&g.expr, state),
        syn::Expr::Struct(s) => {
            for field in &s.fields {
                visit_expr_for_loop_locals(&field.expr, state);
            }
            if let Some(rest) = &s.rest {
                visit_expr_for_loop_locals(rest, state);
            }
        }
        syn::Expr::Repeat(r) => {
            visit_expr_for_loop_locals(&r.expr, state);
            visit_expr_for_loop_locals(&r.len, state);
        }
        // Read-free leaves and constructs that cannot be statically
        // introspected for reads.
        _ => {}
    }
}

/// Walk a `syn::Pat` and insert every binding-introducing identifier
/// into `out`.  Used by [`loop_body_locals`] to determine the rebound
/// name set; also useful to any pure-syn caller wanting "what names
/// does this pattern introduce" without a graph walk.
pub(crate) fn collect_pat_idents(pat: &syn::Pat, out: &mut std::collections::HashSet<String>) {
    match pat {
        syn::Pat::Ident(i) => {
            out.insert(i.ident.to_string());
            if let Some((_, sub)) = &i.subpat {
                collect_pat_idents(sub, out);
            }
        }
        syn::Pat::Type(t) => collect_pat_idents(&t.pat, out),
        syn::Pat::Tuple(t) => {
            for elem in &t.elems {
                collect_pat_idents(elem, out);
            }
        }
        syn::Pat::TupleStruct(t) => {
            for elem in &t.elems {
                collect_pat_idents(elem, out);
            }
        }
        syn::Pat::Struct(s) => {
            for field in &s.fields {
                collect_pat_idents(&field.pat, out);
            }
        }
        syn::Pat::Or(o) => {
            for case in &o.cases {
                collect_pat_idents(case, out);
            }
        }
        syn::Pat::Reference(r) => collect_pat_idents(&r.pat, out),
        syn::Pat::Paren(p) => collect_pat_idents(&p.pat, out),
        syn::Pat::Slice(s) => {
            for elem in &s.elems {
                collect_pat_idents(elem, out);
            }
        }
        // Other variants (Lit, Path, Range, Rest, Wild, Macro, …)
        // bind no local names directly.
        _ => {}
    }
}

pub(crate) fn is_primitive_type_string(type_str: &str) -> bool {
    matches!(
        type_str,
        "i8" | "i16"
            | "i32"
            | "i64"
            | "isize"
            | "u8"
            | "u16"
            | "u32"
            | "u64"
            | "usize"
            | "bool"
            | "char"
            | "f32"
            | "f64"
            | "()"
    )
}

pub(crate) fn type_string_to_value_type(type_str: &str) -> crate::model::ValueType {
    use crate::model::ValueType;
    let type_str = type_str.trim();
    match type_str {
        "i8" | "i16" | "i32" | "i64" | "isize" | "u8" | "u16" | "u32" | "u64" | "usize"
        | "char" | "Self::Truth" => ValueType::Int,
        "bool" => ValueType::Bool,
        "f32" | "f64" => ValueType::Float,
        "()" => ValueType::Void,
        _ => ValueType::Ref(None),
    }
}

pub(crate) fn type_root_from_type_string(type_str: &str) -> Option<String> {
    let mut trimmed = type_str.trim();
    loop {
        let stripped = trimmed
            .strip_prefix("*const ")
            .or_else(|| trimmed.strip_prefix("*mut "))
            .or_else(|| trimmed.strip_prefix("&mut "))
            .or_else(|| trimmed.strip_prefix("&"));
        match stripped {
            Some(rest) => trimmed = rest.trim_start(),
            None => break,
        }
    }
    if trimmed.is_empty() || is_primitive_type_string(trimmed) {
        return None;
    }
    let head = trimmed
        .split(['<', ' ', '('])
        .next()
        .unwrap_or(trimmed)
        .trim();
    if head.is_empty() {
        None
    } else {
        Some(head.to_string())
    }
}

pub(crate) fn bare_type_root_from_type_str(s: &str) -> Option<String> {
    let mut trimmed = s.trim();
    while let Some(rest) = trimmed.strip_prefix('&') {
        let rest = rest.trim_start();
        let rest = rest.strip_prefix("mut ").unwrap_or(rest).trim_start();
        let rest = if let Some(after_quote) = rest.strip_prefix('\'') {
            let rest = after_quote
                .split_once(char::is_whitespace)
                .map(|(_lt, tail)| tail.trim_start())
                .unwrap_or("");
            rest
        } else {
            rest
        };
        trimmed = rest;
    }
    if trimmed.starts_with("dyn ") {
        return None;
    }
    for wrapper in ["Box", "Rc", "Arc"] {
        let prefix = format!("{wrapper}<");
        if let Some(rest) = trimmed.strip_prefix(prefix.as_str())
            && let Some(inner) = rest.strip_suffix('>')
        {
            return bare_type_root_from_type_str(inner);
        }
    }
    if trimmed.contains('<') {
        return None;
    }
    let leaf = trimmed.split_whitespace().next()?;
    if leaf.is_empty() {
        None
    } else {
        Some(leaf.to_string())
    }
}

pub(crate) fn dyn_trait_root_from_type_str(s: &str) -> Option<String> {
    let trimmed = s.trim();
    if let Some(rest) = trimmed.strip_prefix("dyn ") {
        let head = rest.split('+').next()?.trim();
        if head.is_empty() {
            return None;
        }
        return Some(head.to_string());
    }
    for wrapper in ["Box", "Rc", "Arc"] {
        let prefix = format!("{wrapper}<");
        if let Some(rest) = trimmed.strip_prefix(prefix.as_str())
            && let Some(inner) = rest.strip_suffix('>')
        {
            return dyn_trait_root_from_type_str(inner);
        }
    }
    None
}

pub(crate) fn transparent_result_err_type(type_str: &str) -> Option<&str> {
    let trimmed = type_str.trim();
    for prefix in ["Result<", "std::result::Result<", "core::result::Result<"] {
        let Some(inner) = trimmed
            .strip_prefix(prefix)
            .and_then(|rest| rest.strip_suffix('>'))
        else {
            continue;
        };
        let err_type = second_top_level_generic_arg(inner).map(str::trim)?;
        if err_type == "()" {
            return None;
        }
        return Some(err_type);
    }
    None
}

pub(crate) fn transparent_option_inner_type(type_str: &str) -> Option<&str> {
    let trimmed = type_str.trim();
    for prefix in ["Option<", "std::option::Option<", "core::option::Option<"] {
        let Some(inner) = trimmed
            .strip_prefix(prefix)
            .and_then(|rest| rest.strip_suffix('>'))
        else {
            continue;
        };
        return first_top_level_generic_arg(inner).map(str::trim);
    }
    None
}

pub(crate) fn second_top_level_generic_arg(args: &str) -> Option<&str> {
    let mut depth = 0usize;
    for (idx, ch) in args.char_indices() {
        match ch {
            '<' | '(' | '[' => depth += 1,
            '>' | ')' | ']' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                let rest = args[idx + 1..].trim();
                return if rest.is_empty() { None } else { Some(rest) };
            }
            _ => {}
        }
    }
    None
}

pub(crate) fn split_tuple_type_elements(type_str: &str) -> Option<Vec<String>> {
    let mut s = type_str.trim();
    loop {
        let stripped = s
            .strip_prefix("*const ")
            .or_else(|| s.strip_prefix("*mut "))
            .or_else(|| s.strip_prefix("&mut "))
            .or_else(|| s.strip_prefix("&"));
        match stripped {
            Some(rest) => s = rest.trim_start(),
            None => break,
        }
    }
    let inner = s.strip_prefix('(')?.strip_suffix(')')?;
    let mut elements: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut depth: i32 = 0;
    for ch in inner.chars() {
        match ch {
            '(' | '<' | '[' => {
                depth += 1;
                current.push(ch);
            }
            ')' | '>' | ']' => {
                depth -= 1;
                current.push(ch);
            }
            ',' if depth == 0 => {
                let trimmed = current.trim();
                if !trimmed.is_empty() {
                    elements.push(trimmed.to_string());
                }
                current.clear();
            }
            _ => current.push(ch),
        }
    }
    let last = current.trim();
    if !last.is_empty() {
        elements.push(last.to_string());
    }
    Some(elements)
}

pub(crate) fn strip_pointer_like_prefixes(type_str: &str) -> &str {
    let mut s = type_str.trim();
    loop {
        let stripped = s
            .strip_prefix("*const ")
            .or_else(|| s.strip_prefix("*mut "))
            .or_else(|| s.strip_prefix("&mut "))
            .or_else(|| s.strip_prefix("&"));
        match stripped {
            Some(rest) => s = rest.trim_start(),
            None => break,
        }
    }
    s
}

/// `arr[idx]` element-type extraction from a stored array type id.
/// `Vec<Point>` → `"Point"`, `[i64]` → `"i64"`, `[Point; 10]` → `"Point"`,
/// `&[Point]` → `"Point"` (pointer-like prefixes are stripped first).
pub(crate) fn extract_element_type_from_str(type_str: &str) -> Option<String> {
    let s = strip_pointer_like_prefixes(type_str);
    if s.starts_with('[') && s.ends_with(']') {
        let inner = &s[1..s.len() - 1];
        let elem = if let Some(semi) = inner.find(';') {
            inner[..semi].trim()
        } else {
            inner.trim()
        };
        if !elem.is_empty() {
            return Some(elem.to_string());
        }
    }
    if let (Some(start), Some(end)) = (s.find('<'), s.rfind('>')) {
        if start < end {
            return first_top_level_generic_arg(&s[start + 1..end])
                .map(str::trim)
                .filter(|elem| !elem.is_empty())
                .map(ToOwned::to_owned);
        }
    }
    None
}

pub(crate) fn outer_generic_inner_type(type_str: &str, wrappers: &[&str]) -> Option<String> {
    let s = strip_pointer_like_prefixes(type_str);
    let start = s.find('<')?;
    let inner = s[start + 1..].strip_suffix('>')?;
    let head = s[..start].trim().rsplit("::").next().unwrap_or("").trim();
    if !wrappers.contains(&head) {
        return None;
    }
    first_top_level_generic_arg(inner)
        .map(str::trim)
        .filter(|inner| !inner.is_empty())
        .map(ToOwned::to_owned)
}

pub(crate) fn method_as_ref_return_type(receiver_ty: &str) -> Option<String> {
    if let Some(inner) = outer_generic_inner_type(receiver_ty, &["Rc", "Arc", "Box", "NonNull"]) {
        return Some(format!("&{}", strip_pointer_like_prefixes(&inner)));
    }
    if let Some(inner) = outer_generic_inner_type(receiver_ty, &["Option"]) {
        return Some(format!("Option<&{}>", strip_pointer_like_prefixes(&inner)));
    }
    extract_element_type_from_str(receiver_ty).map(|elem| format!("&[{}]", elem))
}

pub(crate) fn array_item_value_type_from_array_type_id(
    array_type_id: Option<&str>,
) -> Option<crate::model::ValueType> {
    let elem_type = extract_element_type_from_str(array_type_id?)?;
    Some(type_string_to_value_type(&elem_type))
}

/// Both synthetic-wrapper and unit-variant ctor paths are valid
/// call targets that are not registered in `PyreCallRegistry`; the
/// discriminator is the caller's check at `canonical_call_target`
/// that `registered_function_path` returns false.
pub(crate) fn is_synthetic_ctor_path(segments: &[String]) -> bool {
    is_synthetic_result_option_wrapper_path(segments) || is_synthetic_unit_variant_path(segments)
}

/// `Ok`/`Err`/`Some` and their qualified spellings.  Always one-arg
/// transparent wrappers that `jtransform::is_synthetic_result_option_ctor`
/// elides at the `args.len() == 1` site.  Valid only as call targets.
pub(crate) fn is_synthetic_result_option_wrapper_path(segments: &[String]) -> bool {
    let path: Vec<&str> = segments.iter().map(String::as_str).collect();
    matches!(
        path.as_slice(),
        ["Ok"]
            | ["Err"]
            | ["Some"]
            | ["Result", "Ok"]
            | ["Result", "Err"]
            | ["Option", "Some"]
            | ["result", "Result", "Ok"]
            | ["result", "Result", "Err"]
            | ["option", "Option", "Some"]
            | ["std", "result", "Result", "Ok"]
            | ["std", "result", "Result", "Err"]
            | ["std", "option", "Option", "Some"]
            | ["core", "result", "Result", "Ok"]
            | ["core", "result", "Result", "Err"]
            | ["core", "option", "Option", "Some"]
    )
}

/// Pyre-side `Class::Variant` unit-variant ctors.  These are valid
/// as bare path-expression values; `flowspace_adapter` pre-folds them
/// to `Hlvalue::Constant(ConstValue::HostObject(prebuilt_instance))`
/// before the rtyper sees a call (mirrors PyPy `rtyper` resolving
/// `SomePBC([InstanceDesc(<unit-variant>)])` to a singleton constant
/// before `jtransform`).  Re-exported `pub(crate)` from `front::ast`
/// so `translator::rtyper::flowspace_adapter::is_synthetic_unit_variant_call`
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

/// Path-as-value numeric constants — Rust counterpart of PyPy
/// `flowspace.LOAD_GLOBAL` resolving a statically-known module
/// attribute to a `Constant(value)` node.  Returns the `f64` bit
/// pattern matching the `syn::Lit::Float` arm's `OpKind::ConstFloat`.
pub(crate) fn path_as_value_float_constant(segments: &[String]) -> Option<u64> {
    let path: Vec<&str> = segments.iter().map(String::as_str).collect();
    let leaf = match path.as_slice() {
        ["f64", leaf] => leaf,
        ["std", "f64", leaf] => leaf,
        ["core", "f64", leaf] => leaf,
        _ => return None,
    };
    match *leaf {
        "INFINITY" => Some(f64::INFINITY.to_bits()),
        "NEG_INFINITY" => Some(f64::NEG_INFINITY.to_bits()),
        "NAN" => Some(f64::NAN.to_bits()),
        _ => None,
    }
}

pub(crate) fn intrinsic_call_result_type(segments: &[String]) -> Option<crate::model::ValueType> {
    use crate::model::ValueType;
    let path: Vec<&str> = segments.iter().map(String::as_str).collect();
    match path.as_slice() {
        ["std", "mem", "size_of"] | ["core", "mem", "size_of"] | ["mem", "size_of"] => {
            Some(ValueType::Int)
        }
        ["f64", "copysign"] | ["std", "f64", "copysign"] | ["core", "f64", "copysign"] => {
            Some(ValueType::Float)
        }
        _ => None,
    }
}

pub(crate) fn kind_char_to_value_type(kind: char) -> Option<crate::model::ValueType> {
    use crate::model::ValueType;
    match kind {
        'i' => Some(ValueType::Int),
        'r' => Some(ValueType::Ref(None)),
        'f' => Some(ValueType::Float),
        'v' => Some(ValueType::Void),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnaryNotOperandKind {
    Bool,
    Int,
    Unknown,
}

pub(crate) fn value_type_to_unary_not_kind(ty: &crate::model::ValueType) -> UnaryNotOperandKind {
    use crate::model::ValueType;
    match ty {
        ValueType::Bool => UnaryNotOperandKind::Bool,
        ValueType::Int | ValueType::Unsigned => UnaryNotOperandKind::Int,
        _ => UnaryNotOperandKind::Unknown,
    }
}

pub(crate) fn type_string_to_unary_not_kind(type_str: &str) -> UnaryNotOperandKind {
    let trimmed = type_str.trim();
    // Arbitrary-precision integer types — `BigInt` / `BigUint` — route
    // through `UNARY_INVERT` (bitwise NOT) like primitive integers,
    // even though their lattice lowering is `ValueType::Ref`.  Pyre's
    // `OpKind::UnaryOp.result_ty` is computed from the operand's actual
    // lowered type at the emit site, so this kind only drives the
    // bytecode dispatch decision, not the result-type carrier.
    if matches!(trimmed, "BigInt" | "BigUint") {
        return UnaryNotOperandKind::Int;
    }
    value_type_to_unary_not_kind(&type_string_to_value_type(type_str))
}

/// Strip the outer `Result<_, _>` or `Option<_>` wrapper from a type
/// string and return the inner ok / some payload type.
pub(crate) fn unwrap_result_or_option(type_str: &str) -> Option<&str> {
    let trimmed = type_str.trim();
    let inner = trimmed
        .strip_prefix("Result<")
        .or_else(|| trimmed.strip_prefix("Option<"))?
        .strip_suffix('>')?;
    let mut depth = 0_i32;
    for (i, ch) in inner.char_indices() {
        match ch {
            '<' => depth += 1,
            '>' => depth -= 1,
            ',' if depth == 0 => return Some(inner[..i].trim()),
            _ => {}
        }
    }
    Some(inner.trim())
}

pub(crate) fn binary_result_value_type_inner(
    lhs_ty: Option<crate::model::ValueType>,
    rhs_ty: Option<crate::model::ValueType>,
    op: &str,
) -> crate::model::ValueType {
    use crate::model::ValueType;
    match (lhs_ty, rhs_ty) {
        (Some(ValueType::Float), Some(ValueType::Float))
        | (Some(ValueType::Float), Some(ValueType::Int))
        | (Some(ValueType::Int), Some(ValueType::Float))
            if matches!(
                op,
                "add"
                    | "sub"
                    | "mul"
                    | "div"
                    | "mod"
                    | "add_assign"
                    | "sub_assign"
                    | "mul_assign"
                    | "div_assign"
                    | "mod_assign"
            ) =>
        {
            ValueType::Float
        }
        (Some(ValueType::Int), Some(ValueType::Int))
            if matches!(
                op,
                "add"
                    | "sub"
                    | "mul"
                    | "div"
                    | "mod"
                    | "bitand"
                    | "bitor"
                    | "bitxor"
                    | "lshift"
                    | "rshift"
                    | "add_assign"
                    | "sub_assign"
                    | "mul_assign"
                    | "div_assign"
                    | "mod_assign"
                    | "bitand_assign"
                    | "bitor_assign"
                    | "bitxor_assign"
                    | "lshift_assign"
                    | "rshift_assign"
            ) =>
        {
            ValueType::Int
        }
        _ => ValueType::Unknown,
    }
}

pub(crate) fn const_value_value_type(
    c: &crate::flowspace::model::ConstValue,
) -> Option<crate::model::ValueType> {
    use crate::flowspace::model::ConstValue;
    use crate::model::ValueType;
    match c {
        ConstValue::Int(_) | ConstValue::AddressOffset(_) => Some(ValueType::Int),
        ConstValue::Bool(_) => Some(ValueType::Bool),
        ConstValue::Float(_) => Some(ValueType::Float),
        ConstValue::ByteStr(_)
        | ConstValue::UniStr(_)
        | ConstValue::None
        | ConstValue::Tuple(_)
        | ConstValue::List(_)
        | ConstValue::Dict(_)
        | ConstValue::Code(_)
        | ConstValue::Function(_)
        | ConstValue::Graphs(_)
        | ConstValue::Atom(_)
        | ConstValue::LLPtr(_)
        | ConstValue::HostObject(_) => Some(ValueType::Ref(None)),
        ConstValue::LowLevelType(_) => Some(ValueType::Void),
        ConstValue::LLAddress(_) => None,
        ConstValue::SpecTag(_) => None,
        ConstValue::Placeholder => None,
    }
}

pub(crate) fn op_result_value_type(
    kind: &crate::model::OpKind,
) -> Option<crate::model::ValueType> {
    use crate::model::{OpKind, ValueType};
    match kind {
        OpKind::Input { ty, .. }
        | OpKind::FieldRead { ty, .. }
        | OpKind::VableFieldRead { ty, .. }
        | OpKind::BinOp { result_ty: ty, .. }
        | OpKind::UnaryOp { result_ty: ty, .. }
        | OpKind::Call { result_ty: ty, .. }
        | OpKind::IndirectCall { result_ty: ty, .. } => {
            if *ty == ValueType::Unknown {
                None
            } else {
                Some(ty.clone())
            }
        }
        OpKind::ConstInt(_) | OpKind::VtableMethodPtr { .. } | OpKind::CurrentTraceLength => {
            Some(ValueType::Int)
        }
        OpKind::ConstFloat(_) => Some(ValueType::Float),
        OpKind::ConstBool(_) => Some(ValueType::Bool),
        OpKind::ConstRef(_) | OpKind::ConstRefNull | OpKind::ConstRefAddr(_) => {
            Some(ValueType::Ref(None))
        }
        OpKind::ArrayRead { item_ty, .. }
        | OpKind::InteriorFieldRead { item_ty, .. }
        | OpKind::VableArrayRead { item_ty, .. } => {
            if *item_ty == ValueType::Unknown {
                None
            } else {
                Some(item_ty.clone())
            }
        }
        OpKind::CallElidable { result_kind, .. }
        | OpKind::CallResidual { result_kind, .. }
        | OpKind::CallMayForce { result_kind, .. }
        | OpKind::InlineCall { result_kind, .. }
        | OpKind::RecursiveCall { result_kind, .. } => kind_char_to_value_type(*result_kind),
        OpKind::IsConstant { .. } | OpKind::IsVirtual { .. } => Some(ValueType::Int),
        _ => None,
    }
}

/// RPython equivalent of Rust's `expr as T` for numeric / Bool /
/// pointer casts.  Maps each `(source, target)` pair to the matching
/// callable path — a single `__builtin__` name (one segment) for
/// numeric/Bool casts, or a fully qualified `lltype` / `rarithmetic`
/// path for pointer / Unsigned-bridging casts.  Pairs not covered
/// return `None`, letting the caller emit an identity `same_as` op.
pub(crate) fn cast_builtin_name(
    source_ty: Option<&crate::model::ValueType>,
    target_ty: &crate::model::ValueType,
) -> Option<&'static [&'static str]> {
    use crate::model::ValueType;
    match (source_ty, target_ty) {
        (Some(ValueType::Int), ValueType::Float) => Some(&["float"]),
        (Some(ValueType::Bool), ValueType::Float) => Some(&["float"]),
        (Some(ValueType::Unsigned), ValueType::Float) => Some(&["float"]),
        (Some(ValueType::Float), ValueType::Int) => Some(&["int"]),
        (Some(ValueType::Bool), ValueType::Int) => Some(&["int"]),
        (Some(ValueType::Unsigned), ValueType::Int) => {
            Some(&["rpython", "rlib", "rarithmetic", "intmask"])
        }
        (Some(ValueType::Int), ValueType::Bool) => Some(&["bool"]),
        (Some(ValueType::Float), ValueType::Bool) => Some(&["bool"]),
        (Some(ValueType::Unsigned), ValueType::Bool) => Some(&["bool"]),
        (Some(ValueType::Ref(_)), ValueType::Int) => Some(&[
            "rpython",
            "rtyper",
            "lltypesystem",
            "lltype",
            "cast_ptr_to_int",
        ]),
        (Some(ValueType::Int), ValueType::Ref(_)) => Some(&[
            "rpython",
            "rtyper",
            "lltypesystem",
            "lltype",
            "cast_int_to_ptr",
        ]),
        (Some(ValueType::Int), ValueType::Unsigned)
        | (Some(ValueType::Float), ValueType::Unsigned)
        | (Some(ValueType::Bool), ValueType::Unsigned) => {
            Some(&["rpython", "rlib", "rarithmetic", "r_uint"])
        }
        _ => None,
    }
}
