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
    FnArg, ItemFn, Pat, PatType, ReturnType, Type, parse_macro_input, parse_quote, spanned::Spanned,
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
    let inner_fn = quote! {
        #(#user_attrs)*
        #[inline]
        #inner_sig #user_body
    };

    // Generate unwrap statements for each parameter.
    let mut unwrap_stmts = Vec::<proc_macro2::TokenStream>::new();
    let mut call_args = Vec::<proc_macro2::TokenStream>::new();
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
        let (unwrap, ident) = unwrap_arg(idx, pat_type)?;
        unwrap_stmts.push(unwrap);
        call_args.push(quote! { #ident });
    }

    let call_inner = quote! { #inner_name( #(#call_args),* ) };
    let body = wrap_return(&user_sig.output, call_inner)?;

    let wrapper = quote! {
        #vis fn #user_name(
            args: &[::pyre_object::PyObjectRef],
        ) -> ::std::result::Result<::pyre_object::PyObjectRef, crate::PyError> {
            #(#unwrap_stmts)*
            #body
        }
    };

    Ok(quote! {
        #inner_fn
        #wrapper
    })
}

/// Generate `let <ident>: <T> = <unwrap-from-args[idx]>;`.
fn unwrap_arg(idx: usize, pt: &PatType) -> syn::Result<(proc_macro2::TokenStream, syn::Ident)> {
    let ident = match &*pt.pat {
        Pat::Ident(pi) => pi.ident.clone(),
        _ => format_ident!("__pyre_arg{}", idx),
    };
    let ty = &*pt.ty;

    let expr = unwrap_expr(ty, idx)?;
    Ok((quote! { let #ident: #ty = #expr; }, ident))
}

fn unwrap_expr(ty: &Type, idx: usize) -> syn::Result<proc_macro2::TokenStream> {
    let ty = unwrap_type_group(ty);
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
        // `Option<T>` — `if args.len() > idx { Some(unwrap(args[idx])) } else { None }`.
        // Mirrors PyPy `@unwrap_spec(s=W_Root)` with `def f(self, space, s=None)`.
        if let Some(inner) = option_inner(ty) {
            let inner_unwrap = unwrap_expr(inner, idx)?;
            return Ok(quote! {
                if args.len() > #idx { Some(#inner_unwrap) } else { None }
            });
        }
        if let Some(seg) = p.path.segments.last() {
            let name = seg.ident.to_string();
            match name.as_str() {
                "i64" => {
                    return Ok(quote! { unsafe { ::pyre_object::w_int_get_value(args[#idx]) } });
                }
                "i32" | "u32" | "usize" | "isize" | "u16" | "i16" | "u8" | "i8" => {
                    return Ok(quote! {
                        unsafe { ::pyre_object::w_int_get_value(args[#idx]) } as #ty
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
            // `Vec<u8>` — bytes.  Borrow then wrap via `w_bytes_from_bytes`.
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
