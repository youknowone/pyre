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

// ── `match`-arm pattern classifiers ──────────────────────────────────
//
// Used by the AST lowering of `match`-on-bool and `match`-on-integer
// scrutinees to extract per-arm `ExitCase` data. All helpers below
// inspect `syn::Arm` / `syn::Pat` without touching the graph builder,
// so they live in this module alongside the other syn-walk helpers.

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

pub(crate) fn canonical_pat_name(pat: &syn::Pat) -> String {
    match pat {
        syn::Pat::Ident(ident) => ident.ident.to_string(),
        syn::Pat::Reference(reference) => canonical_pat_name(&reference.pat),
        syn::Pat::Type(typed) => canonical_pat_name(&typed.pat),
        syn::Pat::TupleStruct(tuple_struct) => tuple_struct
            .path
            .segments
            .iter()
            .map(|seg| seg.ident.to_string())
            .collect::<Vec<_>>()
            .join("::"),
        syn::Pat::Struct(strukt) => strukt
            .path
            .segments
            .iter()
            .map(|seg| seg.ident.to_string())
            .collect::<Vec<_>>()
            .join("::"),
        syn::Pat::Tuple(_) => "tuple_pat".into(),
        syn::Pat::Slice(_) => "slice_pat".into(),
        syn::Pat::Lit(_) => "lit_pat".into(),
        syn::Pat::Path(_) => "path_pat".into(),
        syn::Pat::Wild(_) => "_".into(),
        syn::Pat::Or(_) => "or_pat".into(),
        syn::Pat::Range(_) => "range_pat".into(),
        syn::Pat::Macro(_) => "macro_pat".into(),
        syn::Pat::Paren(paren) => canonical_pat_name(&paren.pat),
        _ => "unsupported_pat".into(),
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
