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
