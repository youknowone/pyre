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
    type_id: syn::LitInt,
    /// Optional override for the upper-case suffix used in derived
    /// static / const names.  Defaults to `strip_prefix("W_")` over
    /// the struct ident, but legacy classes like `W_SuperObject` (which
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
        // `"name.path", type_id = N [, static_name = "PREFIX"]`
        let name: syn::LitStr = input.parse()?;
        input.parse::<syn::Token![,]>()?;
        let mut type_id: Option<syn::LitInt> = None;
        let mut static_name: Option<syn::LitStr> = None;
        let mut pytype_static: Option<syn::LitStr> = None;
        loop {
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
            if input.is_empty() {
                break;
            }
            input.parse::<syn::Token![,]>()?;
            if input.is_empty() {
                break;
            }
        }
        let type_id = type_id.ok_or_else(|| {
            syn::Error::new(name.span(), "`#[pyre_class]` requires `type_id = N`")
        })?;
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
    let type_id_lit = attrs.type_id;

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
    let object_size_const = format_ident!("W_{}_OBJECT_SIZE", suffix);
    let ptr_offsets_const = format_ident!("W_{}_GC_PTR_OFFSETS", suffix);
    let descriptor_static = format_ident!("W_{}_PYRE_CLASS_DESCRIPTOR", suffix);

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

        #st_vis const #gc_type_id_const: u32 = #type_id_lit;
        #st_vis const #object_size_const: usize = ::std::mem::size_of::<#st_name>();
        #st_vis const #ptr_offsets_const: [usize; #ptr_offsets_len] = [
            #(#ptr_offsets_inits),*
        ];

        impl ::pyre_object::lltype::GcType for #st_name {
            const TYPE_ID: u32 = #gc_type_id_const;
            const SIZE: usize = #object_size_const;
        }

        /// Compile-time descriptor consumed by `pyre/pyre-jit/src/eval.rs`'s
        /// GC registration loop.  Aggregates the four constants above into
        /// a single `Sync` static the JIT driver iterates over without
        /// per-type knowledge.
        #st_vis static #descriptor_static: ::pyre_object::lltype::PyreClassDescriptor =
            ::pyre_object::lltype::PyreClassDescriptor {
                pytype_ptr: &#pytype_static as *const ::pyre_object::PyType,
                gc_type_id: #gc_type_id_const,
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
pub fn pyre_methods(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let imp = parse_macro_input!(item as ItemImpl);
    match expand_pyre_methods(imp) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn expand_pyre_methods(mut imp: ItemImpl) -> syn::Result<proc_macro2::TokenStream> {
    if let Some((_, path, _)) = &imp.trait_ {
        return Err(syn::Error::new(
            path.span(),
            "#[pyre_methods] must annotate an inherent impl, not a trait impl",
        ));
    }
    let self_ty = (*imp.self_ty).clone();

    // Collect (python_name, wrapper_ident) per method as we rewrite the
    // impl block in-place: every `fn name(&self|&mut self, …)` keeps
    // its typed body untouched so users can call it directly from Rust,
    // and we synthesise a sibling `pub fn name__pyre_wrapper(args)` as
    // a free-fn inside an attached `mod _pyre_wrappers_<Self>` module.
    let mut wrappers = Vec::<proc_macro2::TokenStream>::new();
    let mut registrations = Vec::<proc_macro2::TokenStream>::new();

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

        // Split first arg (must be `&mut self` / `&self`) from the rest.
        let mut inputs = m.sig.inputs.iter();
        let recv = match inputs.next() {
            Some(FnArg::Receiver(r)) => r,
            _ => {
                return Err(syn::Error::new(
                    m.sig.span(),
                    "#[pyre_methods]: every method must take `&self` or `&mut self` \
                     as its first arg — static/class methods are not supported yet",
                ));
            }
        };
        let recv_is_mut = recv.mutability.is_some();
        let from_obj_call = if recv_is_mut {
            quote! { <#self_ty>::from_obj(args[0]) }
        } else {
            // `from_obj` returns `Option<&mut Self>` even for `&self`
            // callers — re-borrow as `&Self` to match the method's
            // receiver kind without forcing a separate `from_obj_ref`.
            quote! { <#self_ty>::from_obj(args[0]).map(|m| &*m) }
        };

        // Unwrap remaining args.  `args[0]` is `self`, so user-arg
        // indices are `1`, `2`, ….
        let mut unwrap_stmts = Vec::<proc_macro2::TokenStream>::new();
        let mut call_args = Vec::<proc_macro2::TokenStream>::new();
        for (offset, arg) in inputs.enumerate() {
            let FnArg::Typed(pt) = arg else {
                return Err(syn::Error::new(
                    arg.span(),
                    "#[pyre_methods]: unexpected receiver mid-signature",
                ));
            };
            let arg_idx = offset + 1;
            let (stmt, ident) = unwrap_arg(arg_idx, pt)?;
            unwrap_stmts.push(stmt);
            call_args.push(quote! { #ident });
        }

        let call_inner = quote! { __pyre_self.#mname( #(#call_args),* ) };
        let body = wrap_return(&m.sig.output, call_inner)?;

        let py_name = mname.to_string();
        wrappers.push(quote! {
            #[allow(non_snake_case)]
            pub fn #wrapper_name(
                args: &[::pyre_object::PyObjectRef],
            ) -> ::std::result::Result<::pyre_object::PyObjectRef, crate::PyError> {
                if args.is_empty() {
                    return ::std::result::Result::Err(
                        crate::PyError::type_error(
                            concat!("descriptor '", #py_name, "' requires self argument"),
                        ),
                    );
                }
                let __pyre_self = match #from_obj_call {
                    ::std::option::Option::Some(s) => s,
                    ::std::option::Option::None => {
                        return ::std::result::Result::Err(
                            crate::PyError::type_error(
                                concat!("descriptor '", #py_name, "' got wrong receiver type"),
                            ),
                        );
                    }
                };
                #(#unwrap_stmts)*
                #body
            }
        });
        registrations.push(quote! {
            crate::dict_storage_store(
                ns,
                #py_name,
                crate::make_builtin_function(#py_name, #wrapper_name),
            );
        });
    }

    // Strip `#[pyre_method]` / `#[pyre_property]` marker attrs the user
    // may have placed (future-compat; ignored today).  Other attrs
    // pass through.
    for item in imp.items.iter_mut() {
        if let ImplItem::Fn(m) = item {
            m.attrs.retain(|a| {
                !(a.path().is_ident("pyre_method") || a.path().is_ident("pyre_property"))
            });
        }
    }

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
                        crate::typedef::w_object(),
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
