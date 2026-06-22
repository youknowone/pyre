//! `pyre-macros` — attribute proc macros that put the PyPy
//! `@unwrap_spec`-style argument-conversion plumbing behind the same
//! single Rust attribute the function definition already needs.
//!
//! ```ignore
//! #[pyre_function]
//! fn stack_effect(opcode: i64) -> i64 {
//!     // typed body: takes a real i64, returns a real i64.
//!     0
//! }
//! ```
//!
//! emits a wrapper that the `py_module! { functions: { ... } }` table
//! can reference by name:
//!
//! ```ignore
//! fn stack_effect(args: &[pyre_object::PyObjectRef])
//!     -> Result<pyre_object::PyObjectRef, crate::PyError>
//! {
//!     let opcode: i64 = unsafe { pyre_object::w_int_get_value(args[0]) };
//!     Ok(pyre_object::w_int_new(__stack_effect_user(opcode)))
//! }
//!
//! fn __stack_effect_user(opcode: i64) -> i64 { 0 }
//! ```
//!
//! Supported parameter types (per-position):
//! * `i64` / `i32` / `u32` / `usize` — `w_int_get_value` + cast.
//! * `f64` — `w_float_get_value`.
//! * `bool` — `w_bool_get_value`.
//! * `&str` — `w_str_get_value`.
//! * `pyre_object::PyObjectRef` — passthrough (`args[i]`).
//! * `&[pyre_object::PyObjectRef]` — passthrough of the whole slice (varargs).
//!
//! Supported return types:
//! * `i64` / `i32` / `u32` / `usize` — `w_int_new`.
//! * `f64` — `w_float_new`.
//! * `bool` — `w_bool_from`.
//! * `String` — `w_str_new`.
//! * `pyre_object::PyObjectRef` — passthrough.
//! * `Result<T, crate::PyError>` — `?`-propagated, then `T` wrapped.
//! * `()` — `w_none()`.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Fields, FnArg, ImplItem, ItemFn, ItemImpl, ItemStruct, Pat, PatType, ReturnType, Type,
    parse_macro_input, parse_quote, spanned::Spanned,
};

#[proc_macro_attribute]
pub fn pyre_function(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    match expand_pyre_function(func) {
        Ok(ts) => ts.into(),
        Err(err) => err.to_compile_error().into(),
    }
}

fn expand_pyre_function(func: ItemFn) -> syn::Result<proc_macro2::TokenStream> {
    let vis = &func.vis;
    let user_name = func.sig.ident.clone();
    let user_attrs = &func.attrs;
    let user_body = &func.block;
    let user_sig = &func.sig;
    let inner_name = format_ident!("__{}_impl", user_name);

    // Build typed inner fn — verbatim original body, just renamed.
    let mut inner_sig = user_sig.clone();
    inner_sig.ident = inner_name.clone();

    // Generate unwrap statements for each parameter, and collect the
    // parameter-name / required tables so the wrapper can resolve keyword
    // arguments by name (PyPy gateway `Signature` + `_match_signature`).
    let mut unwrap_stmts = Vec::<proc_macro2::TokenStream>::new();
    let mut call_args = Vec::<proc_macro2::TokenStream>::new();
    let mut param_names = Vec::<String>::new();
    let mut param_required = Vec::<bool>::new();
    let mut has_varargs = false;
    for (idx, arg) in user_sig.inputs.iter().enumerate() {
        let pat_type = match arg {
            FnArg::Typed(pt) => pt,
            FnArg::Receiver(r) => {
                return Err(syn::Error::new(
                    r.span(),
                    "#[pyre_function] cannot wrap methods (no `self` arg)",
                ));
            }
        };
        if is_varargs_param(&pat_type.ty) {
            has_varargs = true;
        }
        param_names.push(param_name(idx, pat_type));
        // Optional iff it has a `#[default(...)]` or is `Option<T>`.
        param_required
            .push(arg_default(pat_type)?.is_none() && option_inner(&pat_type.ty).is_none());
        let (unwrap, ident) = unwrap_arg(idx, pat_type)?;
        unwrap_stmts.push(unwrap);
        call_args.push(quote! { #ident });
    }

    // Keyword-binding preamble: when the call carried a `__pyre_kw__`
    // dict, rebind positional+keyword args into a resolved scope keyed by
    // parameter name; otherwise the positional fast path runs unchanged.
    // Skipped for varargs fns — `&[PyObjectRef]` consumes the whole slice
    // (including any trailing kwargs dict), which the resolved-scope shape
    // cannot express.
    let kwargs_preamble = if has_varargs {
        quote! {}
    } else {
        let name_lits = param_names.iter().map(|n| quote! { #n });
        let req_lits = param_required.iter().map(|b| quote! { #b });
        let fn_name_str = user_name.to_string();
        quote! {
            const __PYRE_PARAM_NAMES: &[&str] = &[ #(#name_lits),* ];
            const __PYRE_PARAM_REQUIRED: &[bool] = &[ #(#req_lits),* ];
            let __pyre_bound_args;
            let args: &[::pyre_object::PyObjectRef] =
                if crate::builtins::has_builtin_kwargs(args) {
                    __pyre_bound_args = crate::builtins::bind_builtin_kwargs(
                        args,
                        __PYRE_PARAM_NAMES,
                        __PYRE_PARAM_REQUIRED,
                        #fn_name_str,
                    )?;
                    &__pyre_bound_args
                } else {
                    args
                };
        }
    };

    let call_inner = quote! { #inner_name( #(#call_args),* ) };
    let body = wrap_return(&user_sig.output, call_inner)?;

    // Strip `#[default(expr)]` arg attributes + rewrite typed-receiver
    // aliases (PyTuple → PyObjectRef) so rustc accepts the emitted inner
    // fn.  The wrapper above has already consumed both transforms.
    let mut stripped = inner_sig.clone();
    for arg in stripped.inputs.iter_mut() {
        if let FnArg::Typed(pt) = arg {
            pt.attrs.retain(|a| {
                !a.path().is_ident("default")
                    && !a.path().is_ident("kwonly")
                    && !a.path().is_ident("kwargs")
            });
        }
    }
    rewrite_alias_args(&mut stripped);
    let inner_fn = quote! {
        #(#user_attrs)*
        #[inline]
        #stripped #user_body
    };

    let wrapper = quote! {
        #vis fn #user_name(
            args: &[::pyre_object::PyObjectRef],
        ) -> ::std::result::Result<::pyre_object::PyObjectRef, crate::PyError> {
            #kwargs_preamble
            #(#unwrap_stmts)*
            #body
        }
    };

    // Companion `<name>_pyre_sig()` — derives a keyword-aware `Signature`
    // from the typed parameter names so the registration path can bind
    // keyword / kw-only / `**kwargs`-style calls by name.  A `#[kwonly]`
    // marker starts the keyword-only tail at that parameter; a `#[kwargs]`
    // marker designates the `**kwargs` catch-all dict.
    //
    // A bare `&[PyObjectRef]` parameter is the raw whole-args-slice
    // passthrough ABI (the wrapper hands it the entire `args` slice), which
    // is incompatible with by-name binding — such a builtin gets no
    // `Signature` (`None`) and keeps the raw positional fast path.
    let sig_fn_name = format_ident!("{}_pyre_sig", user_name);
    let mut sig_stmts = Vec::<proc_macro2::TokenStream>::new();
    let mut kwonly_marked = false;
    let mut raw_slice = false;
    for arg in user_sig.inputs.iter() {
        let FnArg::Typed(pt) = arg else { continue };
        let name_lit = match &*pt.pat {
            Pat::Ident(pi) => pi.ident.to_string(),
            _ => continue,
        };
        if pt.attrs.iter().any(|a| a.path().is_ident("kwargs")) {
            sig_stmts.push(quote! { __b.kwargname = ::std::option::Option::Some(#name_lit); });
            continue;
        }
        // Only a `&[PyObjectRef]` slice is the raw whole-args passthrough
        // that suppresses the signature.  A `&[u8]` (or other element type)
        // is a positioned bytes-like parameter and keeps by-name binding.
        let is_slice = match unwrap_type_group(&pt.ty) {
            Type::Reference(r) => match unwrap_type_group(&r.elem) {
                Type::Slice(s) => type_is_py_object_ref(unwrap_type_group(&s.elem)),
                _ => false,
            },
            _ => false,
        };
        if is_slice {
            raw_slice = true;
            continue;
        }
        if !kwonly_marked && pt.attrs.iter().any(|a| a.path().is_ident("kwonly")) {
            sig_stmts.push(quote! { __b.marker_kwonly(); });
            kwonly_marked = true;
        }
        sig_stmts.push(quote! { __b.append(#name_lit); });
    }
    let sig_body = if raw_slice {
        quote! { ::std::option::Option::None }
    } else {
        quote! {
            let mut __b = crate::SignatureBuilder::default();
            #(#sig_stmts)*
            ::std::option::Option::Some(__b.signature())
        }
    };
    let sig_fn = quote! {
        #[allow(dead_code)]
        #vis fn #sig_fn_name() -> ::std::option::Option<crate::Signature> {
            #sig_body
        }
    };

    Ok(quote! {
        #inner_fn
        #wrapper
        #sig_fn
    })
}

/// Generate `let <ident>: <T> = <unwrap-from-args[idx]>;`.
///
/// `#[default(expr)]` on an arg substitutes `expr` whenever
/// `args.len() <= idx`.  Mirrors PyPy `@unwrap_spec(w_x=WrappedDefault(v))`
/// — when the caller omits a positional, the wrapper synthesises a value
/// in the user's typed coordinate space, not in `PyObjectRef` space.
fn unwrap_arg(idx: usize, pt: &PatType) -> syn::Result<(proc_macro2::TokenStream, syn::Ident)> {
    let ident = match &*pt.pat {
        Pat::Ident(pi) => pi.ident.clone(),
        _ => format_ident!("__pyre_arg{}", idx),
    };
    let ty = &*pt.ty;

    let unwrap = unwrap_expr(ty, idx)?;
    // A `&[PyObjectRef]` whole-slice parameter binds the entire `args`
    // slice — it has no per-slot index to bounds-check.  Other slice
    // element types (e.g. `&[u8]`) are positioned params indexing
    // `args[idx]`, so they keep the missing-argument bounds guard.
    let is_whole_slice = if let Type::Reference(r) = unwrap_type_group(ty) {
        if let Type::Slice(s) = unwrap_type_group(&r.elem) {
            type_is_py_object_ref(unwrap_type_group(&s.elem))
        } else {
            false
        }
    } else {
        false
    };
    let argname = ident.to_string();
    // An `Option<T>` parameter carries its own absence: `unwrap_expr` yields
    // `None` for an out-of-range or PY_NULL slot, so the slot is optional and
    // gets no missing-argument guard.
    let is_optional = option_inner(ty).is_some();
    let expr = match arg_default(pt)? {
        // `bind_kwargs_to_signature` pads `args` to the parameter count
        // with PY_NULL, so the default only applies when the slot is both
        // present and non-null.
        Some(default) => quote! {
            if #idx < args.len() && !args[#idx].is_null() { #unwrap } else { #default }
        },
        None if is_whole_slice || is_optional => unwrap,
        None => quote! {
            {
                if #idx >= args.len() || args[#idx].is_null() {
                    return ::std::result::Result::Err(
                        crate::PyError::type_error(
                            format!("missing required argument: '{}'", #argname)
                        )
                    );
                }
                #unwrap
            }
        },
    };
    // Typed-receiver aliases erase to the binding type declared in
    // `typed_alias` (PyObjectRef for passthrough, String for PyPath).
    // The inner fn signature is rewritten in parallel by
    // `rewrite_alias_args` so user code sees the same type.
    let binding_ty = if let Type::Path(p) = unwrap_type_group(ty) {
        if let Some(seg) = p.path.segments.last() {
            typed_alias_binding_ty(&seg.ident.to_string()).unwrap_or_else(|| quote! { #ty })
        } else {
            quote! { #ty }
        }
    } else {
        quote! { #ty }
    };
    Ok((quote! { let #ident: #binding_ty = #expr; }, ident))
}

/// Substitute typed-receiver aliases in a fn signature with the rust
/// type the alias resolves to.  Applied to inner fns (`#[pyre_function]`)
/// and impl-block methods (`#[pyre_methods]`) so the user-visible alias
/// name (`PyTuple` / `PyPath`) is erased to a real Rust type before the
/// body compiles.
fn rewrite_alias_args(sig: &mut syn::Signature) {
    for arg in sig.inputs.iter_mut() {
        let FnArg::Typed(pt) = arg else { continue };
        let ty = unwrap_type_group(&pt.ty);
        let Type::Path(p) = ty else { continue };
        let Some(seg) = p.path.segments.last() else {
            continue;
        };
        if let Some(binding_ty) = typed_alias_binding_ty(&seg.ident.to_string()) {
            pt.ty = Box::new(syn::parse2(binding_ty).expect("alias binding ty parses"));
        }
    }
}

/// Extract the inner expression from `#[default(expr)]` on a fn arg.
fn arg_default(pt: &PatType) -> syn::Result<Option<proc_macro2::TokenStream>> {
    for a in pt.attrs.iter() {
        if !a.path().is_ident("default") {
            continue;
        }
        let expr: syn::Expr = a.parse_args().map_err(|e| {
            syn::Error::new(
                a.span(),
                format!("#[default(...)]: expected a single expression — {e}"),
            )
        })?;
        return Ok(Some(quote! { #expr }));
    }
    Ok(None)
}

/// Resolve a typed-receiver alias name to its `(binding_type,
/// unwrap_expr_template)` pair.  Two flavours of alias coexist:
///
/// * Passthrough-with-typecheck (`PyTuple`, `PyList`, …) — mirrors PyPy
///   `@unwrap_spec(w_x=W_TupleObject)` which `gateway.py:311-316`
///   lowers to `space.interp_w(W_TupleObject, scope_w[i])`.  Binding
///   type stays `PyObjectRef`; the wrapper just emits an `is_X` check
///   and bails out with a TypeError on mismatch.
///
/// * Convert-to-rust (`PyPath`) — mirrors PyPy
///   `@unwrap_spec(path='fsencode')` (`gateway.py:visit_fsencode` line
///   365) which lowers to `space.fsencode_w(...)`.  Binding type is the
///   Rust value the conversion produces (`String` for PyPath); the
///   inner-fn signature is rewritten the same way and the body sees the
///   converted Rust value directly.
///
/// `__a` and `__idx` are the wrapper-local names the template must
/// bind/refer to so the caller can splice the chosen index in.
fn typed_alias(
    name: &str,
    idx: usize,
) -> Option<(proc_macro2::TokenStream, proc_macro2::TokenStream)> {
    let passthrough = |check: proc_macro2::TokenStream| {
        (
            quote! { ::pyre_object::PyObjectRef },
            quote! {
                {
                    let __a = args[#idx];
                    if !unsafe { #check(__a) } {
                        return ::std::result::Result::Err(
                            crate::PyError::type_error(format!(
                                "argument {} must be {}", #idx, #name
                            )),
                        );
                    }
                    __a
                }
            },
        )
    };
    Some(match name {
        "PyTuple" => passthrough(quote! { ::pyre_object::is_tuple }),
        "PyList" => passthrough(quote! { ::pyre_object::is_list }),
        "PyDict" => passthrough(quote! { ::pyre_object::is_dict }),
        "PyStr" => passthrough(quote! { ::pyre_object::is_str }),
        "PyBytes" => passthrough(quote! { ::pyre_object::is_bytes }),
        "PyByteArray" => passthrough(quote! { ::pyre_object::is_bytearray }),
        "PyInt" => passthrough(quote! { ::pyre_object::is_int }),
        "PyFloat" => passthrough(quote! { ::pyre_object::is_float }),
        "PyBool" => passthrough(quote! { ::pyre_object::is_bool }),
        "PySet" => passthrough(quote! { ::pyre_object::is_set }),
        "PyFrozenSet" => passthrough(quote! { ::pyre_object::is_frozenset }),
        "PyPath" => (
            quote! { ::std::string::String },
            quote! { crate::gateway::fsencode_w(args[#idx])? },
        ),
        "PyIndex" => (
            // Mirrors PyPy `space.getindex_w(w_obj, None)`: consults
            // `__index__` per PEP 357 and returns the underlying i64,
            // clamping to i64::MAX / i64::MIN on overflow.  Raises
            // TypeError when the object has no `__index__` and is not
            // already int-like.
            quote! { i64 },
            quote! { crate::baseobjspace::getindex_w(args[#idx])? },
        ),
        // Integer aliases — route through the `space.gateway_nonnegint_w`
        // / `space.c_*_w` converters in baseobjspace.rs so the range / sign
        // checks and their exception messages live in one place.
        "PyNonNegInt" => (
            quote! { i64 },
            quote! { crate::baseobjspace::gateway_nonnegint_w(args[#idx])? },
        ),
        "PyCInt" => (
            quote! { i32 },
            quote! { crate::baseobjspace::c_int_w(args[#idx])? },
        ),
        "PyCUInt" => (
            quote! { u32 },
            quote! { crate::baseobjspace::c_uint_w(args[#idx])? },
        ),
        "PyCShort" => (
            quote! { i16 },
            quote! { crate::baseobjspace::c_short_w(args[#idx])? },
        ),
        "PyCUShort" => (
            quote! { u16 },
            quote! { crate::baseobjspace::c_ushort_w(args[#idx])? },
        ),
        "PyCUidT" => (
            quote! { u32 },
            quote! { crate::baseobjspace::c_uid_t_w(args[#idx])? },
        ),
        "PyTruncatedInt" => (
            quote! { i64 },
            quote! { crate::baseobjspace::truncatedint_w(args[#idx])? },
        ),
        "PyText0" => (
            quote! { &'static str },
            quote! { crate::baseobjspace::text0_w(args[#idx])? },
        ),
        "PyBytes0" => (
            // space.bytes0_w — bytes_w plus a rejection of embedded NUL.
            quote! { &'static [u8] },
            quote! {
                {
                    if !unsafe { ::pyre_object::bytesobject::is_bytes_like(args[#idx]) } {
                        return ::std::result::Result::Err(
                            crate::PyError::type_error(
                                format!("argument {} must be bytes-like", #idx)
                            )
                        );
                    }
                    let __b = unsafe { ::pyre_object::bytesobject::bytes_like_data(args[#idx]) };
                    if __b.contains(&0) {
                        return ::std::result::Result::Err(
                            crate::PyError::value_error(
                                "embedded null byte".to_string()
                            )
                        );
                    }
                    __b
                }
            },
        ),
        "PyPathOrNone" => (
            // space.fsencode_or_none_w — None passes through, otherwise
            // fsencode_w (same conversion as the `PyPath` alias).
            quote! { ::std::option::Option<::std::string::String> },
            quote! {
                if unsafe { ::pyre_object::is_none(args[#idx]) } {
                    ::std::option::Option::None
                } else {
                    ::std::option::Option::Some(crate::gateway::fsencode_w(args[#idx])?)
                }
            },
        ),
        // space.realunicode_w / space.utf8_w — a str argument as unicode
        // text; both route through baseobjspace and return the borrowed
        // utf8 buffer, differing only in the TypeError message they raise.
        "PyUnicode" => (
            quote! { &'static str },
            quote! { crate::baseobjspace::realunicode_w(args[#idx])? },
        ),
        "PyUtf8" => (
            quote! { &'static str },
            quote! { crate::baseobjspace::utf8_w(args[#idx])? },
        ),
        "PyTextOrNone" => (
            // space.text_or_none_w — None passes through, otherwise text_w.
            quote! { ::std::option::Option<&'static str> },
            quote! {
                if args[#idx].is_null() || unsafe { ::pyre_object::is_none(args[#idx]) } {
                    ::std::option::Option::None
                } else {
                    ::std::option::Option::Some(crate::baseobjspace::text_w(args[#idx])?)
                }
            },
        ),
        "PyText0OrNone" => (
            // space.text0_or_none_w — None passes through, otherwise text0_w.
            quote! { ::std::option::Option<&'static str> },
            quote! {
                if args[#idx].is_null() || unsafe { ::pyre_object::is_none(args[#idx]) } {
                    ::std::option::Option::None
                } else {
                    ::std::option::Option::Some(crate::baseobjspace::text0_w(args[#idx])?)
                }
            },
        ),
        "PyBufferStr" => (
            quote! { &'static [u8] },
            quote! { crate::baseobjspace::charbuf_w(args[#idx])? },
        ),
        "PyCNonNegInt" => (
            quote! { i32 },
            quote! { crate::baseobjspace::c_nonnegint_w(args[#idx])? },
        ),
        _ => return None,
    })
}

/// Helper used by `rewrite_alias_args` — looks up just the binding
/// type produced by `typed_alias`.  The dummy idx `0` is fine because
/// the returned binding type never references `idx`.
fn typed_alias_binding_ty(name: &str) -> Option<proc_macro2::TokenStream> {
    typed_alias(name, 0).map(|(ty, _)| ty)
}

fn unwrap_expr(ty: &Type, idx: usize) -> syn::Result<proc_macro2::TokenStream> {
    let ty = unwrap_type_group(ty);
    // Typed-receiver aliases — `state: PyTuple` becomes a typecheck +
    // PyObjectRef binding; `path: PyPath` becomes an fsencode_w call +
    // `String` binding.  Inner fn signatures get rewritten elsewhere so
    // the body's parameter type matches the binding type the alias
    // resolves to.
    if let Type::Path(p) = ty {
        if let Some(seg) = p.path.segments.last() {
            if let Some((_, expr)) = typed_alias(&seg.ident.to_string(), idx) {
                return Ok(expr);
            }
        }
    }
    // `&[PyObjectRef]` — pass the whole slice (varargs).
    // `&[u8]`        — bytes-like (bytes + bytearray) → `bytes_like_data`,
    //                  with a runtime type check that returns a TypeError
    //                  on non-bytes-like input.
    if let Type::Reference(r) = ty {
        if let Type::Slice(s) = &*r.elem {
            let elem = unwrap_type_group(&s.elem);
            if type_is_py_object_ref(elem) {
                return Ok(quote! { args });
            }
            if let Type::Path(p) = elem {
                if path_is_ident(&p.path, "u8") {
                    return Ok(quote! {
                        {
                            if !unsafe { ::pyre_object::bytesobject::is_bytes_like(args[#idx]) } {
                                return ::std::result::Result::Err(
                                    crate::PyError::type_error(
                                        format!("argument {} must be bytes-like", #idx)
                                    )
                                );
                            }
                            unsafe { ::pyre_object::bytesobject::bytes_like_data(args[#idx]) }
                        }
                    });
                }
            }
        }
        // `&str` — borrow from `w_str_get_value`.
        if let Type::Path(p) = &*r.elem {
            if path_is_ident(&p.path, "str") {
                return Ok(quote! {
                    unsafe { ::pyre_object::w_str_get_value(args[#idx]) }
                });
            }
        }
    }

    if let Type::Path(p) = ty {
        if type_is_py_object_ref(ty) {
            return Ok(quote! { args[#idx] });
        }
        // `Option<T>` — present when the slot is in range and not the
        // `PY_NULL` "argument omitted" marker `bind_builtin_kwargs` writes
        // for an absent optional after keyword resolution.  Mirrors PyPy
        // `@unwrap_spec(s=W_Root)` with `def f(self, space, s=None)`.
        if let Some(inner) = option_inner(ty) {
            let inner_unwrap = unwrap_expr(inner, idx)?;
            return Ok(quote! {
                if #idx < args.len() && !args[#idx].is_null() { Some(#inner_unwrap) } else { None }
            });
        }
        if let Some(seg) = p.path.segments.last() {
            let name = seg.ident.to_string();
            match name.as_str() {
                "i64" => {
                    return Ok(quote! { unsafe { ::pyre_object::w_int_get_value(args[#idx]) } });
                }
                "i32" | "u32" | "usize" | "isize" | "u16" | "i16" | "u8" | "i8" => {
                    // Parenthesised so the cast composes inside `if/else`
                    // / `match` arms when a `#[default(...)]` wrapper sits
                    // around the unwrap.  Without the parens,
                    // `if ... { unsafe{} as u32 } else { ... }` fails to
                    // parse because `as` doesn't accept a block-form LHS.
                    return Ok(quote! {
                        (unsafe { ::pyre_object::w_int_get_value(args[#idx]) } as #ty)
                    });
                }
                "f64" => {
                    return Ok(quote! { unsafe { ::pyre_object::w_float_get_value(args[#idx]) } });
                }
                "bool" => {
                    return Ok(quote! { unsafe { ::pyre_object::w_bool_get_value(args[#idx]) } });
                }
                _ => {}
            }
        }
    }

    Err(syn::Error::new(
        ty.span(),
        format!(
            "#[pyre_function]: unsupported argument type — \
             add a mapping in pyre-macros/src/lib.rs::unwrap_expr"
        ),
    ))
}

/// The keyword name a parameter binds under — its identifier, or a
/// synthetic positional-only placeholder for a non-ident pattern (which
/// no real keyword can match).
fn param_name(idx: usize, pt: &PatType) -> String {
    match &*pt.pat {
        Pat::Ident(pi) => pi.ident.to_string(),
        _ => format!("__pyre_positional_{idx}"),
    }
}

/// `true` for a `&[PyObjectRef]` varargs parameter (binds the whole arg
/// slice).
fn is_varargs_param(ty: &Type) -> bool {
    let ty = unwrap_type_group(ty);
    if let Type::Reference(r) = ty {
        if let Type::Slice(s) = &*r.elem {
            return type_is_py_object_ref(unwrap_type_group(&s.elem));
        }
    }
    false
}

/// `Option<T>` → `Some(&T)`; anything else → None.
fn option_inner(ty: &Type) -> Option<&Type> {
    let ty = unwrap_type_group(ty);
    let Type::Path(p) = ty else { return None };
    let seg = p.path.segments.last()?;
    if seg.ident != "Option" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
        return None;
    };
    let first = args.args.iter().next()?;
    let syn::GenericArgument::Type(t) = first else {
        return None;
    };
    Some(t)
}

fn wrap_return(
    ret: &ReturnType,
    call_inner: proc_macro2::TokenStream,
) -> syn::Result<proc_macro2::TokenStream> {
    let ty = match ret {
        ReturnType::Default => {
            return Ok(quote! { #call_inner; Ok(::pyre_object::w_none()) });
        }
        ReturnType::Type(_, t) => &**t,
    };

    // `Result<T, crate::PyError>` — propagate via `?`, then wrap T.
    if let Some(inner) = result_pyerror_inner(ty) {
        let wrap = wrap_value_expr(inner, quote! { __pyre_v })?;
        return Ok(quote! {
            let __pyre_v = #call_inner ?;
            Ok(#wrap)
        });
    }

    let wrap = wrap_value_expr(ty, call_inner)?;
    Ok(quote! { Ok(#wrap) })
}

/// Wrap a Rust value expression of type `ty` into a `PyObjectRef`.
fn wrap_value_expr(
    ty: &Type,
    value: proc_macro2::TokenStream,
) -> syn::Result<proc_macro2::TokenStream> {
    let ty = unwrap_type_group(ty);
    if type_is_py_object_ref(ty) {
        return Ok(value);
    }
    if let Type::Path(p) = ty {
        if let Some(seg) = p.path.segments.last() {
            let name = seg.ident.to_string();
            match name.as_str() {
                "i64" => return Ok(quote! { ::pyre_object::w_int_new(#value) }),
                "i32" | "u32" | "usize" | "isize" | "u16" | "i16" | "u8" | "i8" => {
                    return Ok(quote! { ::pyre_object::w_int_new((#value) as i64) });
                }
                "f64" => return Ok(quote! { ::pyre_object::w_float_new(#value) }),
                "bool" => return Ok(quote! { ::pyre_object::w_bool_from(#value) }),
                "String" => return Ok(quote! { ::pyre_object::w_str_new(&#value) }),
                _ => {}
            }
            // `Vec<T>` — bytes / list-of-X.
            //   * `Vec<u8>`                        → bytes via w_bytes_from_bytes
            //   * `Vec<PyObjectRef>`               → list passthrough
            //   * `Vec<i64> / <i32> / <f64> / <String> / <bool>` →
            //     wrap each element via `PywrapKind` then `w_list_new`.
            // Mirrors PyPy `space.newlist([space.newint(x) for x in vec])`
            // where the interp2app auto-wraps a Rust `[W_Root]` return
            // through `space.newlist`.
            if seg.ident == "Vec" {
                if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                    if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                        let inner = unwrap_type_group(inner);
                        if let Type::Path(ip) = inner {
                            if path_is_ident(&ip.path, "u8") {
                                return Ok(quote! {
                                    ::pyre_object::bytesobject::w_bytes_from_bytes(&#value)
                                });
                            }
                            if type_is_py_object_ref(inner) {
                                return Ok(quote! { ::pyre_object::w_list_new(#value) });
                            }
                            if let Some(last) = ip.path.segments.last() {
                                let nm = last.ident.to_string();
                                if matches!(
                                    nm.as_str(),
                                    "i64"
                                        | "i32"
                                        | "u32"
                                        | "usize"
                                        | "isize"
                                        | "u16"
                                        | "i16"
                                        | "i8"
                                        | "f64"
                                        | "bool"
                                        | "String"
                                ) {
                                    return Ok(quote! {
                                        ::pyre_object::w_list_new(
                                            (#value).into_iter()
                                                .map(<_ as crate::PywrapKind>::into_py)
                                                .collect()
                                        )
                                    });
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    if let Type::Tuple(t) = ty {
        if t.elems.is_empty() {
            return Ok(quote! { { #value; ::pyre_object::w_none() } });
        }
    }
    use quote::ToTokens;
    Err(syn::Error::new(
        ty.span(),
        format!(
            "#[pyre_function]: unsupported return type `{}` — \
             add a mapping in pyre-macros/src/lib.rs::wrap_value_expr",
            ty.to_token_stream()
        ),
    ))
}

/// Strip `Type::Group` wrappers that `macro_rules!` `:ty` capture adds
/// around emitted type fragments — without this, `wrap_value_expr` and
/// friends never recognize the inner `Type::Path` for `String` /
/// `PyObjectRef` / `i64` / etc. when the fn definition arrives through
/// `py_module!`'s `inline_functions:` arm.
fn unwrap_type_group(ty: &Type) -> &Type {
    let mut t = ty;
    while let Type::Group(g) = t {
        t = &g.elem;
    }
    t
}

fn type_is_py_object_ref(ty: &Type) -> bool {
    let ty = unwrap_type_group(ty);
    let Type::Path(p) = ty else { return false };
    let segs: Vec<_> = p
        .path
        .segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect();
    matches!(
        segs.as_slice(),
        [s] if s == "PyObjectRef"
    ) || matches!(
        segs.as_slice(),
        [a, b] if a == "pyre_object" && b == "PyObjectRef"
    ) || matches!(
        segs.as_slice(),
        [_, a, b] if a == "pyre_object" && b == "PyObjectRef"
    )
}

fn path_is_ident(p: &syn::Path, name: &str) -> bool {
    p.segments.len() == 1 && p.segments[0].ident == name
}

/// If `ty` is `Result<T, crate::PyError>` (or `PyError` short form), return `&T`.
fn result_pyerror_inner(ty: &Type) -> Option<&Type> {
    let ty = unwrap_type_group(ty);
    let Type::Path(p) = ty else { return None };
    let seg = p.path.segments.last()?;
    if seg.ident != "Result" {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else {
        return None;
    };
    let mut iter = args.args.iter();
    let first = iter.next()?;
    let second = iter.next()?;
    let syn::GenericArgument::Type(ok_ty) = first else {
        return None;
    };
    let syn::GenericArgument::Type(err_ty) = second else {
        return None;
    };
    if !type_is_pyerror(err_ty) {
        return None;
    }
    Some(ok_ty)
}

fn type_is_pyerror(ty: &Type) -> bool {
    let ty = unwrap_type_group(ty);
    let Type::Path(p) = ty else { return false };
    let segs: Vec<_> = p
        .path
        .segments
        .iter()
        .map(|s| s.ident.to_string())
        .collect();
    matches!(segs.last().map(String::as_str), Some("PyError"))
}

// Avoid "unused" warnings on parse_quote import in the trimmed build.
#[allow(dead_code)]
fn _unused() {
    let _: syn::Type = parse_quote!(i64);
}

// ──────────────────────────────────────────────────────────────────────
// `#[pyre_class("python.name", type_id = N)]` — PyPy `class W_X(W_Root)`
// equivalent.
//
// Generates the typed-payload boilerplate every existing `W_X` struct
// writes by hand (see `pyre/pyre-object/src/superobject.rs`):
// ──────────────────────────────────────────────────────────────────────
//
// User source:
//   #[pyre_class("_random.Random", type_id = 53)]
//   pub struct W_Random {
//       state: u64,
//   }
//
// Emitted:
//   pub static RANDOM_TYPE: ::pyre_object::PyType =
//       ::pyre_object::pyobject::new_pytype("_random.Random");
//
//   #[repr(C)]
//   pub struct W_Random {
//       pub ob: ::pyre_object::PyObject,   // <- macro-prepended header
//       pub state: u64,
//   }
//
//   pub const W_RANDOM_GC_TYPE_ID: u32 = 53;
//   pub const W_RANDOM_OBJECT_SIZE: usize = ::std::mem::size_of::<W_Random>();
//   pub const W_RANDOM_GC_PTR_OFFSETS: [usize; 0] = [];
//
//   impl ::pyre_object::lltype::GcType for W_Random {
//       const TYPE_ID: u32 = W_RANDOM_GC_TYPE_ID;
//       const SIZE: usize = W_RANDOM_OBJECT_SIZE;
//   }
//
//   impl W_Random {
//       pub unsafe fn from_obj(obj: ::pyre_object::PyObjectRef)
//           -> ::std::option::Option<&'static mut Self>
//       {
//           if unsafe { ::pyre_object::py_type_check(obj, &RANDOM_TYPE) } {
//               Some(unsafe { &mut *(obj as *mut Self) })
//           } else { None }
//       }
//   }
//
// `PTR_OFFSETS` auto-derived from the user's struct: every field whose
// type is `PyObjectRef` becomes one entry via `std::mem::offset_of!`.
// Primitive fields (`u64` / `i32` / etc.) are skipped because the GC
// doesn't need to trace them.
//
// `type_id = N` is required (manual): the GC's `pytype_to_tid` table
// asserts a contiguous monotonic sequence at JIT-init in
// `pyre/pyre-jit/src/eval.rs:1335-1352`.  Reserve a slot, register
// it in eval.rs, and pass the same number here.
//
// The PyType static name is derived from the struct name (snake-case
// uppercased + `_TYPE` suffix): `W_Random` → `RANDOM_TYPE`.
// `W_GetSetProperty` → `GETSETPROPERTY_TYPE`.  Override the suffix
// path is not yet supported — pick struct names whose derived static
// matches the import path callers expect.

#[proc_macro_attribute]
pub fn pyre_class(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attrs = parse_macro_input!(attr as PyreClassAttrs);
    let st = parse_macro_input!(item as ItemStruct);
    match expand_pyre_class(attrs, st) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

struct PyreClassAttrs {
    name: syn::LitStr,
    /// Optional explicit `type_id = N`.  When `Some`, the macro emits
    /// the legacy `pub const W_X_GC_TYPE_ID: u32 = N;` alias and the
    /// runtime cell starts pre-initialized to `N` so the JIT driver
    /// can drift-check it.  When `None`, no const is emitted and the
    /// cell starts at `TypeIdCell::UNASSIGNED`; the JIT driver writes
    /// the next available tid back into it.
    type_id: Option<syn::LitInt>,
    /// Optional override for the upper-case suffix used in derived
    /// static / const names.  Defaults to `strip_prefix("W_")` over
    /// the struct ident, but legacy classes like `W_Super` (which
    /// historically shipped as `SUPER_TYPE` and `W_SUPER_GC_TYPE_ID`,
    /// not `SUPEROBJECT_TYPE`) need to opt into the shorter form.
    static_name: Option<syn::LitStr>,
    /// Optional override for *just* the PyType static identifier.  Most
    /// classes accept the `{static_name}_TYPE` default
    /// (`SUPER_TYPE`, `RANDOM_TYPE`); a few legacy classes ship the
    /// PyType under a name unrelated to the GC consts (e.g.
    /// `GETSET_DESCRIPTOR_TYPE` vs `W_GETSET_PROPERTY_GC_TYPE_ID`).
    /// Specifying this lets the GC consts retain one prefix while the
    /// PyType keeps its historical name.
    pytype_static: Option<syn::LitStr>,
}

impl syn::parse::Parse for PyreClassAttrs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        // `"name.path"[, type_id = N][, static_name = "PREFIX"]`
        let name: syn::LitStr = input.parse()?;
        let mut type_id: Option<syn::LitInt> = None;
        let mut static_name: Option<syn::LitStr> = None;
        let mut pytype_static: Option<syn::LitStr> = None;
        while !input.is_empty() {
            input.parse::<syn::Token![,]>()?;
            if input.is_empty() {
                break;
            }
            let key: syn::Ident = input.parse()?;
            input.parse::<syn::Token![=]>()?;
            match key.to_string().as_str() {
                "type_id" => type_id = Some(input.parse()?),
                "static_name" => static_name = Some(input.parse()?),
                "pytype_static" => pytype_static = Some(input.parse()?),
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!(
                            "unknown `#[pyre_class]` key `{other}` — \
                             expected `type_id` / `static_name` / `pytype_static`",
                        ),
                    ));
                }
            }
        }
        Ok(Self {
            name,
            type_id,
            static_name,
            pytype_static,
        })
    }
}

fn expand_pyre_class(
    attrs: PyreClassAttrs,
    mut st: ItemStruct,
) -> syn::Result<proc_macro2::TokenStream> {
    let st_name = st.ident.clone();
    let st_vis = st.vis.clone();
    let name_lit = attrs.name;

    // Derive static names from the struct name.
    //   W_Random          -> RANDOM_TYPE, W_RANDOM_GC_TYPE_ID,
    //                       W_RANDOM_OBJECT_SIZE, W_RANDOM_GC_PTR_OFFSETS
    //   W_GetSetProperty  -> GETSETPROPERTY_TYPE, W_GETSETPROPERTY_GC_TYPE_ID, …
    let st_str = st_name.to_string();
    let suffix = attrs
        .static_name
        .as_ref()
        .map(|s| s.value())
        .unwrap_or_else(|| st_str.strip_prefix("W_").unwrap_or(&st_str).to_uppercase());
    let pytype_static = match attrs.pytype_static.as_ref() {
        Some(s) => format_ident!("{}", s.value()),
        None => format_ident!("{}_TYPE", suffix),
    };
    let gc_type_id_const = format_ident!("W_{}_GC_TYPE_ID", suffix);
    let gc_type_id_cell = format_ident!("W_{}_GC_TYPE_ID_CELL", suffix);
    let object_size_const = format_ident!("W_{}_OBJECT_SIZE", suffix);
    let ptr_offsets_const = format_ident!("W_{}_GC_PTR_OFFSETS", suffix);
    let descriptor_static = format_ident!("W_{}_PYRE_CLASS_DESCRIPTOR", suffix);

    // When the user declared `type_id = N` we pre-initialize the cell
    // to `N` and additionally emit the legacy `pub const W_X_GC_TYPE_ID:
    // u32 = N;` alias so existing consumers (pyre-jit-trace,
    // drift-detection unit tests) keep compiling.  When the user
    // omitted `type_id`, the cell starts unassigned and the legacy
    // const is not emitted — callers must read the cell at runtime via
    // `<W_X as GcType>::type_id()` (which itself becomes `cell.get()`).
    let (cell_init, legacy_const) = match attrs.type_id.as_ref() {
        Some(n) => (
            quote! { ::pyre_object::lltype::TypeIdCell::with(#n) },
            quote! { #st_vis const #gc_type_id_const: u32 = #n; },
        ),
        None => (
            quote! { ::pyre_object::lltype::TypeIdCell::auto() },
            quote! {},
        ),
    };

    // Enforce `#[repr(C)]` and prepend the PyObject header.
    let already_repr_c = st.attrs.iter().any(|a| {
        a.path().is_ident("repr")
            && a.parse_args::<syn::Ident>()
                .map(|i| i == "C")
                .unwrap_or(false)
    });
    if !already_repr_c {
        st.attrs.push(parse_quote!(#[repr(C)]));
    }

    // Prepend `pub ob: PyObject` if not already present.
    let Fields::Named(ref mut named) = st.fields else {
        return Err(syn::Error::new(
            st.span(),
            "#[pyre_class] requires a struct with named fields",
        ));
    };
    let has_ob = named
        .named
        .iter()
        .any(|f| f.ident.as_ref().map(|i| i == "ob").unwrap_or(false));
    if !has_ob {
        use syn::parse::Parser;
        let ob_field: syn::Field = syn::Field::parse_named
            .parse2(quote! { pub ob: ::pyre_object::PyObject })
            .expect("parse ob field");
        named.named.insert(0, ob_field);
    }

    // Collect `PyObjectRef` fields' offsets for GC tracing.  Skip `ob`
    // because the GC walks the header through the parent (object) tid.
    let mut ptr_field_idents: Vec<syn::Ident> = Vec::new();
    for f in named.named.iter() {
        let Some(ident) = f.ident.clone() else {
            continue;
        };
        if ident == "ob" {
            continue;
        }
        if type_is_py_object_ref(&f.ty) {
            ptr_field_idents.push(ident);
        }
    }
    let ptr_offsets_len = ptr_field_idents.len();
    let ptr_offsets_inits: Vec<proc_macro2::TokenStream> = ptr_field_idents
        .iter()
        .map(|i| quote! { ::std::mem::offset_of!(#st_name, #i) })
        .collect();

    Ok(quote! {
        #st

        #st_vis static #pytype_static: ::pyre_object::PyType =
            ::pyre_object::pyobject::new_pytype(#name_lit);

        /// Runtime-resolved GC tid for this class.  Initialized either
        /// to the explicit `type_id = N` from the attribute (drift-
        /// checked at JIT init) or to `TypeIdCell::UNASSIGNED` for
        /// auto-mode (then stamped by the JIT driver).
        #st_vis static #gc_type_id_cell: ::pyre_object::lltype::TypeIdCell =
            #cell_init;
        #legacy_const
        #st_vis const #object_size_const: usize = ::std::mem::size_of::<#st_name>();
        #st_vis const #ptr_offsets_const: [usize; #ptr_offsets_len] = [
            #(#ptr_offsets_inits),*
        ];

        impl ::pyre_object::lltype::GcType for #st_name {
            #[inline]
            fn type_id() -> u32 {
                #gc_type_id_cell.get()
            }
            const SIZE: usize = #object_size_const;
        }

        /// Compile-time descriptor consumed by `pyre/pyre-jit/src/eval.rs`'s
        /// GC registration loop.  Aggregates the four constants above into
        /// a single `Sync` static the JIT driver iterates over without
        /// per-type knowledge.
        #st_vis static #descriptor_static: ::pyre_object::lltype::PyreClassDescriptor =
            ::pyre_object::lltype::PyreClassDescriptor {
                pytype_ptr: &#pytype_static as *const ::pyre_object::PyType,
                gc_type_id: &#gc_type_id_cell,
                object_size: #object_size_const,
                ptr_offsets: &#ptr_offsets_const,
            };

        impl ::pyre_object::lltype::PyreClassPyTypeOf for #st_name {
            const PYTYPE: *const ::pyre_object::PyType =
                &#pytype_static as *const ::pyre_object::PyType;
            const DESCRIPTOR: &'static ::pyre_object::lltype::PyreClassDescriptor =
                &#descriptor_static;
            const PYNAME: &'static str = #name_lit;
        }

        impl #st_name {
            /// Borrow `obj` as `&mut Self` after verifying its
            /// `ob_type` matches this class's static `PyType`.
            /// Returns `None` if `obj` is the wrong type — callers
            /// must convert that to a Python `TypeError`.
            #[allow(dead_code)]
            #[inline]
            pub fn from_obj(obj: ::pyre_object::PyObjectRef)
                -> ::std::option::Option<&'static mut Self>
            {
                if unsafe { ::pyre_object::py_type_check(obj, &#pytype_static) } {
                    ::std::option::Option::Some(unsafe { &mut *(obj as *mut Self) })
                } else {
                    ::std::option::Option::None
                }
            }

            /// Allocate a fresh instance via `lltype::malloc_typed`,
            /// stamping the PyObject header so the GC and dispatcher
            /// can identify it.  `payload` carries the user-defined
            /// fields; the `ob` header is filled in by this fn.
            #[allow(dead_code)]
            pub fn allocate(payload: Self) -> ::pyre_object::PyObjectRef {
                let _roots = ::pyre_object::gc_roots::push_roots();
                let full = Self {
                    ob: ::pyre_object::PyObject {
                        ob_type: &#pytype_static as *const ::pyre_object::PyType,
                        w_class: ::pyre_object::pyobject::get_instantiate(&#pytype_static),
                    },
                    ..payload
                };
                ::pyre_object::lltype::malloc_typed(full) as ::pyre_object::PyObjectRef
            }
        }
    })
}

// ──────────────────────────────────────────────────────────────────────
// `#[pyre_methods]` — PyPy `TypeDef("...", method=interp2app(W_X.m))`
// equivalent attached to an `impl W_X { ... }` block.
//
// User source:
//   #[pyre_methods]
//   impl W_Random {
//       fn __init__(&mut self, seed: Option<i64>) {
//           self.state = seed.unwrap_or(DEFAULT) as u64;
//       }
//       fn random(&mut self) -> f64 { ... }
//   }
//
// Emitted: one `args: &[PyObjectRef]` wrapper per method (downcasts
// `args[0]` to `&mut Self` via `from_obj`, unwraps the rest through
// the same `unwrap_expr` machinery `#[pyre_function]` uses, calls the
// typed method, wraps the return through `wrap_return`) plus
// `pub fn type_object()` that consumes `<Self as PyreClassPyTypeOf>::
// {PYNAME, PYTYPE}` to drive `make_builtin_type_with_layout` and
// `set_instantiate` exactly like `py_class_typed!` does.
// ──────────────────────────────────────────────────────────────────────

#[proc_macro_attribute]
pub fn pyre_methods(attr: TokenStream, item: TokenStream) -> TokenStream {
    let attrs = parse_macro_input!(attr as PyreMethodsAttrs);
    let imp = parse_macro_input!(item as ItemImpl);
    match expand_pyre_methods(attrs, imp) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// `#[pyre_methods(doc = "...", weakrefable, unhashable)]` —
/// declarative slot keys propagated into the `type_object()`
/// registration closure.  Mirrors `TypeDef(..., __doc__=..., __weakref__=
/// make_weakref_descr(W_X), __hash__=None)` in
/// `pypy/interpreter/typedef.py`.
#[derive(Default)]
struct PyreMethodsAttrs {
    doc: Option<syn::LitStr>,
    weakrefable: bool,
    unhashable: bool,
    /// Optional base class expression — the `__base` of the TypeDef.
    /// Mirrors `TypeDef("X", __base=W_Base.typedef)` in
    /// `pypy/interpreter/typedef.py`.  Defaults to `w_object()` (the
    /// `object` base) when omitted.
    base: Option<syn::Expr>,
}

impl syn::parse::Parse for PyreMethodsAttrs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut out = PyreMethodsAttrs::default();
        if input.is_empty() {
            return Ok(out);
        }
        loop {
            let key: syn::Ident = input.parse()?;
            match key.to_string().as_str() {
                "doc" => {
                    input.parse::<syn::Token![=]>()?;
                    out.doc = Some(input.parse()?);
                }
                "weakrefable" => out.weakrefable = true,
                "unhashable" => out.unhashable = true,
                "base" => {
                    input.parse::<syn::Token![=]>()?;
                    out.base = Some(input.parse()?);
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!(
                            "unknown `#[pyre_methods]` key `{other}` — expected \
                             `doc = \"...\"` / `weakrefable` / `unhashable` / `base = <expr>`",
                        ),
                    ));
                }
            }
            if input.is_empty() {
                break;
            }
            input.parse::<syn::Token![,]>()?;
            if input.is_empty() {
                break;
            }
        }
        Ok(out)
    }
}

// PyPy `interp_attrproperty('state', cls=W_X, wrapfn=newint)`
// (`typedef.py:485`) is the read-only-field exposure helper.  The
// `#[getter]` / `#[setter]` machinery below covers the exact same
// ground in one line per slot — `#[getter] fn state(&self) -> i64
// { self.state as i64 }` — without a separate alias.  Documented
// here so future reviewers don't reintroduce `interp_attrproperty`
// as its own macro: there is no syntactic gap left to close.

#[derive(Clone)]
enum MethodKind {
    Instance,
    Static,
    Class,
    // `(py_name, doc)` — doc is the optional `doc="…"` arg shared by all
    // three accessors, mirroring `GetSetProperty(fget, fset, fdel, doc=)`.
    Getter(String, Option<String>),
    Setter(String, Option<String>),
    Deleter(String, Option<String>),
}

/// One `#[getter(...)]` / `#[setter(...)]` / `#[deleter(...)]` argument:
/// either a positional `"name"` (or `name = "…"`) or `doc = "…"`.
enum GetSetArg {
    Name(String),
    Doc(String),
}

impl syn::parse::Parse for GetSetArg {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        if input.peek(syn::LitStr) {
            let s: syn::LitStr = input.parse()?;
            return Ok(GetSetArg::Name(s.value()));
        }
        let key: syn::Ident = input.parse()?;
        let _: syn::Token![=] = input.parse()?;
        let val: syn::LitStr = input.parse()?;
        match key.to_string().as_str() {
            "name" => Ok(GetSetArg::Name(val.value())),
            "doc" => Ok(GetSetArg::Doc(val.value())),
            _ => Err(syn::Error::new(key.span(), "expected `name` or `doc`")),
        }
    }
}

/// Parse a `#[getter]` / `#[setter]` / `#[deleter]` attribute into its
/// optional `(name, doc)`.  Accepts `#[x]`, `#[x("name")]`,
/// `#[x(doc = "…")]`, and `#[x("name", doc = "…")]`.
fn parse_getset_attr(a: &syn::Attribute) -> syn::Result<(Option<String>, Option<String>)> {
    use syn::Meta;
    match &a.meta {
        Meta::Path(_) => Ok((None, None)),
        Meta::List(_) => {
            let args = a.parse_args_with(
                syn::punctuated::Punctuated::<GetSetArg, syn::Token![,]>::parse_terminated,
            )?;
            let mut name = None;
            let mut doc = None;
            for arg in args {
                match arg {
                    GetSetArg::Name(s) => name = Some(s),
                    GetSetArg::Doc(s) => doc = Some(s),
                }
            }
            Ok((name, doc))
        }
        Meta::NameValue(_) => Err(syn::Error::new(
            a.span(),
            "expected `#[attr]`, `#[attr(\"name\")]`, or `#[attr(doc = \"…\")]`",
        )),
    }
}

/// Inspect `#[staticmethod]` / `#[classmethod]` / `#[getter]` /
/// `#[setter]` markers on an impl-fn.  Mutually exclusive; the
/// unannotated default is `Instance`.
///
/// `#[getter]` / `#[setter]` accept an optional `(py_name)` arg.  When
/// omitted, the py-name is derived from the rust fn name (setters strip
/// a leading `set_` to pair with their getter).  Mirrors PyPy
/// `name = GetSetProperty(W_X.descr_get_name, W_X.descr_set_name)`
/// where both descr handlers share the python-visible `name`.
fn classify_method(m: &syn::ImplItemFn) -> syn::Result<MethodKind> {
    let mut kind = MethodKind::Instance;
    let mut seen = false;
    for a in m.attrs.iter() {
        let new_kind = if a.path().is_ident("staticmethod") {
            Some(MethodKind::Static)
        } else if a.path().is_ident("classmethod") {
            Some(MethodKind::Class)
        } else if a.path().is_ident("getter") {
            let (name, doc) = parse_getset_attr(a)?;
            let py_name = name.unwrap_or_else(|| m.sig.ident.to_string());
            Some(MethodKind::Getter(py_name, doc))
        } else if a.path().is_ident("setter") {
            let (name, doc) = parse_getset_attr(a)?;
            let fn_name = m.sig.ident.to_string();
            let py_name = name
                .or_else(|| fn_name.strip_prefix("set_").map(str::to_owned))
                .unwrap_or(fn_name);
            Some(MethodKind::Setter(py_name, doc))
        } else if a.path().is_ident("deleter") {
            let (name, doc) = parse_getset_attr(a)?;
            let fn_name = m.sig.ident.to_string();
            let py_name = name
                .or_else(|| fn_name.strip_prefix("del_").map(str::to_owned))
                .unwrap_or(fn_name);
            Some(MethodKind::Deleter(py_name, doc))
        } else {
            None
        };
        let Some(new_kind) = new_kind else { continue };
        if seen {
            return Err(syn::Error::new(
                a.span(),
                "#[pyre_methods]: at most one of \
                 `#[classmethod]` / `#[staticmethod]` / `#[getter]` / `#[setter]` / `#[deleter]`",
            ));
        }
        kind = new_kind;
        seen = true;
    }
    Ok(kind)
}

/// One python-visible GetSetProperty being assembled across `#[getter]` /
/// `#[setter]` / `#[deleter]` arms that share a py-name.  Mirrors
/// `GetSetProperty(fget, fset, fdel, doc=)`.
#[derive(Default)]
struct PropEntry {
    name: String,
    fget: Option<syn::Ident>,
    fset: Option<syn::Ident>,
    fdel: Option<syn::Ident>,
    doc: Option<String>,
}

/// Which accessor a `record_property` call contributes.
enum Accessor {
    Get,
    Set,
    Del,
}

/// Insert one accessor wrapper into the `properties` Vec, merging with an
/// existing entry on the same py-name and rejecting duplicate accessors.
/// A `doc` provided by any accessor is adopted; conflicting docs error.
fn record_property(
    properties: &mut Vec<PropEntry>,
    name: String,
    accessor: Accessor,
    wrapper: syn::Ident,
    doc: Option<String>,
    m: &syn::ImplItemFn,
) -> syn::Result<()> {
    let entry = match properties.iter_mut().find(|e| e.name == name) {
        Some(e) => e,
        None => {
            properties.push(PropEntry {
                name: name.clone(),
                ..PropEntry::default()
            });
            properties.last_mut().unwrap()
        }
    };
    let (slot, label) = match accessor {
        Accessor::Get => (&mut entry.fget, "getter"),
        Accessor::Set => (&mut entry.fset, "setter"),
        Accessor::Del => (&mut entry.fdel, "deleter"),
    };
    if slot.is_some() {
        return Err(syn::Error::new(
            m.sig.span(),
            format!("#[{label}]: property `{name}` already has a {label}"),
        ));
    }
    *slot = Some(wrapper);
    if let Some(doc) = doc {
        match &entry.doc {
            Some(existing) if *existing != doc => {
                return Err(syn::Error::new(
                    m.sig.span(),
                    format!("property `{name}` has conflicting `doc` values"),
                ));
            }
            _ => entry.doc = Some(doc),
        }
    }
    Ok(())
}

fn expand_pyre_methods(
    attrs: PyreMethodsAttrs,
    mut imp: ItemImpl,
) -> syn::Result<proc_macro2::TokenStream> {
    if let Some((_, path, _)) = &imp.trait_ {
        return Err(syn::Error::new(
            path.span(),
            "#[pyre_methods] must annotate an inherent impl, not a trait impl",
        ));
    }
    let self_ty = (*imp.self_ty).clone();

    // Auto-synthesize `__new__` when the user wrote `__init__` but no
    // `__new__`.  Mirrors PyPy `TypeDef` behavior where a class without
    // an explicit `__new__` inherits `object.__new__` which allocates a
    // zero-initialized instance.  Requires the user struct to implement
    // `Default` (typically via `#[derive(Default)]` — `PyObject` itself
    // derives `Default` so the auto-derive resolves).
    let has_init = imp
        .items
        .iter()
        .any(|i| matches!(i, ImplItem::Fn(f) if f.sig.ident == "__init__"));
    let has_new = imp
        .items
        .iter()
        .any(|i| matches!(i, ImplItem::Fn(f) if f.sig.ident == "__new__"));
    if has_init && !has_new {
        let synth: ImplItem = parse_quote! {
            #[staticmethod]
            fn __new__(_cls: ::pyre_object::PyObjectRef) -> ::pyre_object::PyObjectRef {
                <#self_ty>::allocate(<#self_ty as ::std::default::Default>::default())
            }
        };
        imp.items.push(synth);
    }

    // Collect (python_name, wrapper_ident) per method as we rewrite the
    // impl block in-place: every `fn name(&self|&mut self, …)` keeps
    // its typed body untouched so users can call it directly from Rust,
    // and we synthesise a sibling `pub fn name__pyre_wrapper(args)` as
    // a free-fn inside an attached `mod _pyre_wrappers_<Self>` module.
    let mut wrappers = Vec::<proc_macro2::TokenStream>::new();
    let mut registrations = Vec::<proc_macro2::TokenStream>::new();
    // `(py_name, fget_wrapper, fset_wrapper)` accumulated across the
    // method-loop.  Each `#[getter]` and `#[setter]` arm writes one
    // slot; after the loop we emit one `w_getset_property_new` per
    // distinct py_name.  Mirrors PyPy `name = GetSetProperty(fget=,
    // fset=)` where both handlers share the python-visible name.
    let mut properties: Vec<PropEntry> = Vec::new();

    for item in imp.items.iter() {
        let ImplItem::Fn(m) = item else { continue };
        if m.sig.asyncness.is_some()
            || m.sig.constness.is_some()
            || m.sig.unsafety.is_some()
            || !m.sig.generics.params.is_empty()
        {
            return Err(syn::Error::new(
                m.sig.span(),
                "#[pyre_methods]: async/const/unsafe/generic methods not supported",
            ));
        }
        let mname = &m.sig.ident;
        let wrapper_name = format_ident!("__pyre_wrap_{}", mname);
        let kind = classify_method(m)?;

        // Build per-kind wrapper preamble (self extraction) + call form,
        // and pick the registration constructor.
        //
        // PyPy `gateway.py`: `interp2app(W_X.fn, as_classmethod=True)` /
        // `as_staticmethod=True` decide whether the descriptor binds
        // `cls` / nothing instead of `self`.  Here the descriptor wrap is
        // applied at `make_builtin_function` time via `w_classmethod_new`
        // / `w_staticmethod_new`; everything else (arg unwrap, return
        // wrap) reuses the regular machinery.
        //
        // Getter / Setter share the Instance preamble — they both bind
        // `self` and look identical to a 0-arg / 1-arg method to the
        // wrapper machinery.  The distinction surfaces only at
        // registration time, where instead of `dict_storage_store(...,
        // make_builtin_function(...))` they participate in a deferred
        // `w_getset_property_new(fget=, fset=)` build keyed by py_name.
        let mut inputs = m.sig.inputs.iter().peekable();
        let (preamble, call_target, first_arg_idx) = match &kind {
            MethodKind::Instance
            | MethodKind::Getter(..)
            | MethodKind::Setter(..)
            | MethodKind::Deleter(..) => {
                let recv = match inputs.next() {
                    Some(FnArg::Receiver(r)) => r,
                    _ => {
                        return Err(syn::Error::new(
                            m.sig.span(),
                            "#[pyre_methods]: instance method needs `&self` or `&mut self` \
                             — add `#[staticmethod]` / `#[classmethod]` to opt out",
                        ));
                    }
                };
                // Method dispatch passes `(self,)`+args; GetSetProperty
                // dispatch passes `(descriptor_self, w_obj,…)` so the
                // actual `self` slides one slot right.  Mirrors
                // `typedef.py:312-325` fget_unwrap_spec.
                let self_idx: usize = match &kind {
                    MethodKind::Getter(..) | MethodKind::Setter(..) | MethodKind::Deleter(..) => 1,
                    _ => 0,
                };
                let from_obj_call = if recv.mutability.is_some() {
                    quote! { <#self_ty>::from_obj(args[#self_idx]) }
                } else {
                    // `from_obj` returns `Option<&mut Self>` even for
                    // `&self` callers — reborrow as `&Self` so the
                    // method's signature matches without a separate
                    // `from_obj_ref` API.
                    quote! { <#self_ty>::from_obj(args[#self_idx]).map(|m| &*m) }
                };
                let needed = self_idx + 1;
                let preamble = quote! {
                    if args.len() < #needed {
                        return ::std::result::Result::Err(
                            crate::PyError::type_error(
                                concat!("descriptor '", stringify!(#mname), "' requires self argument"),
                            ),
                        );
                    }
                    let __pyre_self = match #from_obj_call {
                        ::std::option::Option::Some(s) => s,
                        ::std::option::Option::None => {
                            return ::std::result::Result::Err(
                                crate::PyError::type_error(
                                    concat!("descriptor '", stringify!(#mname), "' got wrong receiver type"),
                                ),
                            );
                        }
                    };
                };
                (preamble, quote! { __pyre_self.#mname }, self_idx + 1)
            }
            MethodKind::Static => {
                if matches!(inputs.peek(), Some(FnArg::Receiver(_))) {
                    return Err(syn::Error::new(
                        m.sig.span(),
                        "#[staticmethod]: must not take `self` / `&self` / `&mut self`",
                    ));
                }
                (quote! {}, quote! { <#self_ty>::#mname }, 0)
            }
            MethodKind::Class => {
                if matches!(inputs.peek(), Some(FnArg::Receiver(_))) {
                    return Err(syn::Error::new(
                        m.sig.span(),
                        "#[classmethod]: first arg must be a typed `cls: PyObjectRef`, \
                         not `&self` / `&mut self`",
                    ));
                }
                (quote! {}, quote! { <#self_ty>::#mname }, 0)
            }
        };

        let mut unwrap_stmts = Vec::<proc_macro2::TokenStream>::new();
        let mut call_args = Vec::<proc_macro2::TokenStream>::new();
        let mut param_names = Vec::<String>::new();
        let mut param_required = Vec::<bool>::new();
        let mut has_varargs = false;
        for (offset, arg) in inputs.enumerate() {
            let FnArg::Typed(pt) = arg else {
                return Err(syn::Error::new(
                    arg.span(),
                    "#[pyre_methods]: unexpected receiver mid-signature",
                ));
            };
            let arg_idx = offset + first_arg_idx;
            if is_varargs_param(&pt.ty) {
                has_varargs = true;
            }
            param_names.push(param_name(offset, pt));
            // Optional iff it has a `#[default(...)]` or is `Option<T>`.
            param_required.push(arg_default(pt)?.is_none() && option_inner(&pt.ty).is_none());
            let (stmt, ident) = unwrap_arg(arg_idx, pt)?;
            unwrap_stmts.push(stmt);
            call_args.push(quote! { #ident });
        }

        // Keyword-binding preamble (mirrors `#[pyre_function]`): when the
        // call carried a trailing `__pyre_kw__` dict, rebind positional +
        // keyword args into a resolved scope keyed by parameter name before
        // the receiver/arg unwraps run; otherwise the positional fast path
        // is untouched. Instance methods prepend a `self` slot — filled
        // positionally — so the receiver index lines up with the bound
        // scope; static/class methods already carry every slot (including
        // `cls`) as a typed parameter. Property getters/setters/deleters use
        // the descriptor calling convention and varargs methods consume the
        // whole slice, so both opt out.
        let kwargs_preamble = {
            let is_property = matches!(
                kind,
                MethodKind::Getter(..) | MethodKind::Setter(..) | MethodKind::Deleter(..)
            );
            if has_varargs || is_property {
                quote! {}
            } else {
                let mut all_names: Vec<String> = Vec::new();
                let mut all_required: Vec<bool> = Vec::new();
                if matches!(kind, MethodKind::Instance) {
                    all_names.push("self".to_string());
                    all_required.push(true);
                }
                all_names.extend(param_names.iter().cloned());
                all_required.extend(param_required.iter().copied());
                let name_lits = all_names.iter().map(|n| quote! { #n });
                let req_lits = all_required.iter().map(|b| quote! { #b });
                let fn_name_str = mname.to_string();
                quote! {
                    const __PYRE_PARAM_NAMES: &[&str] = &[ #(#name_lits),* ];
                    const __PYRE_PARAM_REQUIRED: &[bool] = &[ #(#req_lits),* ];
                    let __pyre_bound_args;
                    let args: &[::pyre_object::PyObjectRef] =
                        if crate::builtins::has_builtin_kwargs(args) {
                            __pyre_bound_args = crate::builtins::bind_builtin_kwargs(
                                args,
                                __PYRE_PARAM_NAMES,
                                __PYRE_PARAM_REQUIRED,
                                #fn_name_str,
                            )?;
                            &__pyre_bound_args
                        } else {
                            args
                        };
                }
            }
        };

        let call_inner = quote! { #call_target( #(#call_args),* ) };
        let body = wrap_return(&m.sig.output, call_inner)?;

        // `__new__` subclass override — PyPy `space.allocate_instance(W_X,
        // w_subtype)` semantics adapted for our static-PyType layout.
        // The user's body allocates via `W_X::allocate(...)` which stamps
        // the static `X_TYPE`; if the caller passed a subclass cls
        // (`class MyR(_random.Random)`), re-point `(*obj).w_class = cls`
        // so `type(obj)` / `isinstance(obj, MyR)` see the subclass while
        // the Rust-side `from_obj` (which keys on the static `ob_type`)
        // still resolves to W_X.  Mirrors `typedef.rs:int_descr_new`
        // (line 905-925) which already applies the same fix-up to
        // builtin int/float subclasses.
        let is_new = mname == "__new__" && matches!(kind, MethodKind::Static);
        let body = if is_new {
            quote! {
                let __pyre_obj: ::pyre_object::PyObjectRef = match { #body } {
                    ::std::result::Result::Ok(o) => o,
                    ::std::result::Result::Err(e) => return ::std::result::Result::Err(e),
                };
                if !__pyre_obj.is_null() {
                    let __pyre_cls = args.first().copied().unwrap_or(::pyre_object::PY_NULL);
                    if !__pyre_cls.is_null() && unsafe { ::pyre_object::is_type(__pyre_cls) } {
                        let __pyre_static_tp = crate::typedef::gettypefor(
                            <#self_ty as ::pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
                        );
                        let __pyre_same_tp = match __pyre_static_tp {
                            ::std::option::Option::Some(t) => __pyre_cls == t,
                            ::std::option::Option::None => false,
                        };
                        if !__pyre_same_tp {
                            unsafe { (*__pyre_obj).w_class = __pyre_cls; }
                        }
                    }
                }
                ::std::result::Result::Ok(__pyre_obj)
            }
        } else {
            body
        };

        let py_name = mname.to_string();
        wrappers.push(quote! {
            #[allow(non_snake_case)]
            pub fn #wrapper_name(
                args: &[::pyre_object::PyObjectRef],
            ) -> ::std::result::Result<::pyre_object::PyObjectRef, crate::PyError> {
                #kwargs_preamble
                #preamble
                #(#unwrap_stmts)*
                #body
            }
        });
        let raw_fn = quote! { crate::make_builtin_function(#py_name, #wrapper_name) };
        match &kind {
            MethodKind::Instance => {
                registrations.push(quote! {
                    crate::dict_storage_store(ns, #py_name, #raw_fn);
                });
            }
            MethodKind::Static => {
                registrations.push(quote! {
                    crate::dict_storage_store(ns, #py_name,
                        ::pyre_object::w_staticmethod_new(#raw_fn));
                });
            }
            MethodKind::Class => {
                registrations.push(quote! {
                    crate::dict_storage_store(ns, #py_name,
                        ::pyre_object::w_classmethod_new(#raw_fn));
                });
            }
            MethodKind::Getter(prop_name, doc) => {
                record_property(
                    &mut properties,
                    prop_name.clone(),
                    Accessor::Get,
                    wrapper_name.clone(),
                    doc.clone(),
                    m,
                )?;
            }
            MethodKind::Setter(prop_name, doc) => {
                record_property(
                    &mut properties,
                    prop_name.clone(),
                    Accessor::Set,
                    wrapper_name.clone(),
                    doc.clone(),
                    m,
                )?;
            }
            MethodKind::Deleter(prop_name, doc) => {
                record_property(
                    &mut properties,
                    prop_name.clone(),
                    Accessor::Del,
                    wrapper_name.clone(),
                    doc.clone(),
                    m,
                )?;
            }
        }
    }

    // Property emission: one `w_getset_property_new` per distinct
    // py-name.  Slots not provided fall back to `PY_NULL` (matching
    // PyPy `GetSetProperty(fget=W_X.descr_get_X, fset=None, fdel=None)`
    // when only a getter is declared).  A `doc="…"` provided on any
    // accessor populates the `doc` slot; otherwise `PY_NULL`.
    for prop in &properties {
        let prop_name = &prop.name;
        let accessor_expr = |slot: &Option<syn::Ident>| match slot {
            Some(id) => quote! { crate::make_builtin_function(#prop_name, #id) },
            None => quote! { ::pyre_object::PY_NULL },
        };
        let fget_expr = accessor_expr(&prop.fget);
        let fset_expr = accessor_expr(&prop.fset);
        let fdel_expr = accessor_expr(&prop.fdel);
        let doc_expr = match &prop.doc {
            Some(doc) => quote! { ::pyre_object::w_str_new(#doc) },
            None => quote! { ::pyre_object::PY_NULL },
        };
        registrations.push(quote! {
            crate::dict_storage_store(
                ns,
                #prop_name,
                ::pyre_object::getsetproperty::w_getset_property_new(
                    #fget_expr,
                    #fset_expr,
                    #fdel_expr,
                    #doc_expr,
                    ::pyre_object::PY_NULL,
                    false,
                    ::pyre_object::w_str_new(#prop_name),
                ),
            );
        });
    }

    // Append declarative-slot registrations after method entries.  Order
    // matches PyPy `TypeDef("X", method=..., __doc__=..., __weakref__=...,
    // __hash__=None)`: methods first, slots last.  `make_weakref_descr`
    // returns the canonical descriptor; the typedef install pass
    // (`make_builtin_type_with_layout`) sees `__weakref__` in `ns` and
    // flips the `weakrefable` bit on the new type via
    // `w_type_set_weakrefable`.
    if let Some(lit) = attrs.doc.as_ref() {
        registrations.push(quote! {
            crate::dict_storage_store(ns, "__doc__", ::pyre_object::w_str_new(#lit));
        });
    }
    if attrs.weakrefable {
        registrations.push(quote! {
            crate::dict_storage_store(
                ns,
                "__weakref__",
                crate::typedef::make_weakref_descr(::pyre_object::PY_NULL),
            );
        });
    }
    if attrs.unhashable {
        registrations.push(quote! {
            crate::dict_storage_store(ns, "__hash__", ::pyre_object::w_none());
        });
    }

    // Strip marker attrs the user placed.  `#[pyre_method]` /
    // `#[pyre_property]` are future-compat no-ops today; `#[classmethod]`
    // / `#[staticmethod]` / `#[default(...)]` were already consumed by
    // the wrapper-generation pass above and would otherwise leak into
    // the emitted impl block and confuse rustc.
    for item in imp.items.iter_mut() {
        let ImplItem::Fn(m) = item else { continue };
        m.attrs.retain(|a| {
            !(a.path().is_ident("pyre_method")
                || a.path().is_ident("pyre_property")
                || a.path().is_ident("classmethod")
                || a.path().is_ident("staticmethod")
                || a.path().is_ident("getter")
                || a.path().is_ident("setter")
                || a.path().is_ident("deleter"))
        });
        for arg in m.sig.inputs.iter_mut() {
            if let FnArg::Typed(pt) = arg {
                pt.attrs.retain(|a| !a.path().is_ident("default"));
            }
        }
        rewrite_alias_args(&mut m.sig);
    }

    // Base class for the TypeDef.  `#[pyre_methods(base = <expr>)]`
    // supplies the parent type; absent it, the class inherits `object`.
    let base_expr = match &attrs.base {
        Some(e) => quote! { #e },
        None => quote! { crate::typedef::w_object() },
    };

    let type_object_fn = quote! {
        pub fn type_object() -> ::pyre_object::PyObjectRef {
            thread_local! {
                static CELL: ::std::cell::OnceCell<::pyre_object::PyObjectRef>
                    = const { ::std::cell::OnceCell::new() };
            }
            CELL.with(|c| {
                *c.get_or_init(|| {
                    let tp = crate::typedef::make_builtin_type_with_layout(
                        <#self_ty as ::pyre_object::lltype::PyreClassPyTypeOf>::PYNAME,
                        |ns| { #(#registrations)* },
                        #base_expr,
                        <#self_ty as ::pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE,
                    );
                    ::pyre_object::pyobject::set_instantiate(
                        unsafe {
                            &*<#self_ty as ::pyre_object::lltype::PyreClassPyTypeOf>::PYTYPE
                        },
                        tp,
                    );
                    tp
                })
            })
        }
    };

    Ok(quote! {
        #imp

        #(#wrappers)*

        #type_object_fn
    })
}
