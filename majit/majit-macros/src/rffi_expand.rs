//! Expansion of `llexternal` and `external_compilation_info`.
//!
//! `llexternal` (`rffi.py`) builds the funcptr, then `call_external_function`,
//! then the forwarding wrapper. The derivations below follow that function's
//! order: `calling_conv`, callback scan, `releasegil` → `invoke_around_handlers`,
//! `random_effects_on_gcobjs`, then `assert not (elidable_function and
//! random_effects_on_gcobjs)`.

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::ext::IdentExt;
use syn::punctuated::Punctuated;
use syn::{
    Expr, Ident, LitBool, LitInt, LitStr, Token, Type, Visibility, bracketed, parse::Parse,
    parse::ParseStream,
};

pub fn expand_llexternal(input: TokenStream) -> syn::Result<TokenStream> {
    let parsed: LlexternalInput = syn::parse2(input)?;
    Ok(parsed.expand())
}

pub fn expand_external_compilation_info(input: TokenStream) -> syn::Result<TokenStream> {
    let parsed: EciInput = syn::parse2(input)?;
    parsed.expand()
}

struct LlexternalInput {
    vis: Visibility,
    name: Ident,
    c_name: String,
    args: Vec<Type>,
    result: Type,
    sandboxsafe: bool,
    releasegil: Tri,
    nowrapper: bool,
    calling_conv: Option<String>,
    elidable: bool,
    macro_path: Option<syn::Path>,
    random_effects: Tri,
    save_err: Expr,
    natural_arity: i64,
    compilation_info: Option<Expr>,
}

#[derive(Clone, Copy)]
enum Tri {
    Auto,
    Bool(bool),
}

impl Parse for LlexternalInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let vis: Visibility = input.parse()?;
        let name: Ident = input.parse()?;
        input.parse::<Token![=]>()?;
        let c_name: LitStr = input.parse()?;
        input.parse::<Token![,]>()?;
        let args_content;
        bracketed!(args_content in input);
        let args: Punctuated<Type, Token![,]> =
            args_content.parse_terminated(Type::parse, Token![,])?;
        input.parse::<Token![,]>()?;
        let result: Type = input.parse()?;

        let mut sandboxsafe = false;
        let mut releasegil = Tri::Auto;
        let mut nowrapper = false;
        let mut calling_conv = None;
        let mut elidable = false;
        let mut macro_path = None;
        let mut random_effects = Tri::Auto;
        let mut save_err = None;
        let mut natural_arity = -1i64;
        let mut compilation_info = None;

        while input.peek(Token![,]) {
            input.parse::<Token![,]>()?;
            if input.is_empty() {
                break;
            }
            let key = Ident::parse_any(input)?;
            input.parse::<Token![=]>()?;
            match key.to_string().as_str() {
                "sandboxsafe" => sandboxsafe = input.parse::<LitBool>()?.value(),
                "releasegil" => releasegil = parse_tri(input, &key)?,
                "_nowrapper" => nowrapper = input.parse::<LitBool>()?.value(),
                "calling_conv" => {
                    calling_conv = Some(input.parse::<LitStr>()?.value());
                }
                "elidable_function" => elidable = input.parse::<LitBool>()?.value(),
                "macro" => macro_path = Some(input.parse()?),
                "random_effects_on_gcobjs" => random_effects = parse_tri(input, &key)?,
                "save_err" => save_err = Some(input.parse()?),
                "natural_arity" => natural_arity = parse_signed_int(input)?,
                "compilation_info" => compilation_info = Some(input.parse()?),
                "_callable" => {
                    return Err(syn::Error::new(
                        key.span(),
                        "`_callable` is the ll2ctypes stand-in; pass `macro = <path>` for a Rust funcptr",
                    ));
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown llexternal kwarg `{other}`"),
                    ));
                }
            }
        }
        if !input.is_empty() {
            return Err(input.error("unexpected tokens after llexternal arguments"));
        }
        if let Some(conv) = &calling_conv
            && conv != "c"
            && conv != "unknown"
            && conv != "win"
        {
            return Err(syn::Error::new(
                Span::call_site(),
                "calling_conv must be \"c\", \"unknown\", or \"win\"",
            ));
        }
        Ok(LlexternalInput {
            vis,
            name,
            c_name: c_name.value(),
            args: args.into_iter().collect(),
            result,
            sandboxsafe,
            releasegil,
            nowrapper,
            calling_conv,
            elidable,
            macro_path,
            random_effects,
            save_err: save_err
                .unwrap_or_else(|| syn::parse_quote!(::majit_rlib::rffi::RFFI_ERR_NONE)),
            natural_arity,
            compilation_info,
        })
    }
}

fn parse_tri(input: ParseStream, key: &Ident) -> syn::Result<Tri> {
    if input.peek(LitBool) {
        return Ok(Tri::Bool(input.parse::<LitBool>()?.value()));
    }
    let ident: Ident = input.parse()?;
    if ident == "auto" {
        Ok(Tri::Auto)
    } else {
        Err(syn::Error::new(
            ident.span(),
            format!("`{key}` is `auto`, `true`, or `false`"),
        ))
    }
}

fn parse_signed_int(input: ParseStream) -> syn::Result<i64> {
    let neg = if input.peek(Token![-]) {
        input.parse::<Token![-]>()?;
        true
    } else {
        false
    };
    let lit: LitInt = input.parse()?;
    let value = lit.base10_parse::<i64>()?;
    Ok(if neg { -value } else { value })
}

fn type_is_fn(ty: &Type) -> bool {
    match ty {
        Type::BareFn(_) => true,
        Type::Ptr(ptr) => type_is_fn(&ptr.elem),
        Type::Reference(reference) => type_is_fn(&reference.elem),
        Type::Paren(paren) => type_is_fn(&paren.elem),
        Type::Group(group) => type_is_fn(&group.elem),
        _ => false,
    }
}

/// `rffi.py` `RFFI_*` integer values. A path whose last segment is one of
/// these names resolves to that integer; the emitted `assert!(lit == expr)`
/// fails the build if the const drifts from this table.
fn rffi_flag(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::Lit(lit) => match &lit.lit {
            syn::Lit::Int(int) => int.base10_parse().ok(),
            _ => None,
        },
        Expr::Unary(unary) if matches!(unary.op, syn::UnOp::Neg(_)) => {
            rffi_flag(&unary.expr).map(|v| -v)
        }
        Expr::Binary(binary) if matches!(binary.op, syn::BinOp::BitOr(_)) => {
            Some(rffi_flag(&binary.left)? | rffi_flag(&binary.right)?)
        }
        Expr::Path(path) => {
            let name = path.path.segments.last()?.ident.to_string();
            Some(match name.as_str() {
                "RFFI_SAVE_ERRNO" => 1,
                "RFFI_READSAVED_ERRNO" => 2,
                "RFFI_ZERO_ERRNO_BEFORE" => 4,
                "RFFI_FULL_ERRNO" => 1 | 2,
                "RFFI_FULL_ERRNO_ZERO" => 1 | 4,
                "RFFI_SAVE_LASTERROR" => 8,
                "RFFI_READSAVED_LASTERROR" => 16,
                "RFFI_SAVE_WSALASTERROR" => 32,
                "RFFI_FULL_LASTERROR" => 8 | 16,
                "RFFI_ERR_NONE" => 0,
                "RFFI_ERR_ALL" => (1 | 2) | (8 | 16),
                "RFFI_ALT_ERRNO" => 64,
                _ => return None,
            })
        }
        Expr::Group(group) => rffi_flag(&group.expr),
        Expr::Paren(paren) => rffi_flag(&paren.expr),
        _ => None,
    }
}

impl LlexternalInput {
    fn expand(&self) -> TokenStream {
        if self.natural_arity < -1 || self.natural_arity > self.args.len() as i64 {
            return syn::Error::new(
                Span::call_site(),
                "natural_arity is -1, or the number of fixed parameters before `...`",
            )
            .to_compile_error();
        }
        let has_callback = self.args.iter().any(type_is_fn);
        let invoke = match self.releasegil {
            Tri::Bool(value) => value,
            Tri::Auto => !self.sandboxsafe && !self.nowrapper,
        };
        let save_err_value = rffi_flag(&self.save_err);
        let save_err_nonzero = save_err_value.unwrap_or(1) != 0;

        let name = &self.name;
        let deriv = self.derivation_consts(has_callback);
        let eci = self.compilation_info.as_ref().map(|eci| {
            quote! {
                #[allow(dead_code)]
                const _COMPILATION_INFO: ::majit_rlib::rffi::ExternalCompilationInfo = #eci;
            }
        });

        if self.nowrapper {
            let funcptr = self.funcptr_item(true);
            let save_err = &self.save_err;
            return quote! {
                #deriv
                #eci
                const _: () = assert!(#save_err == ::majit_rlib::rffi::RFFI_ERR_NONE);
                #funcptr
            };
        }

        let funcptr = self.funcptr_item(false);
        let call_name = format_ident!("ccall_{}", name);
        let (params, arg_names) = self.params();
        let result = &self.result;
        let call_expr = self.call_expr(&arg_names);
        let save_err = &self.save_err;

        let (around_attr, around_assert) = if invoke
            && self.macro_path.is_none()
            && self.natural_arity == -1
        {
            let Some(lit) = save_err_value else {
                return syn::Error::new_spanned(
                        &self.save_err,
                        "save_err must be an integer literal or an RFFI_* const so call_aroundstate_target can take an integer literal",
                    )
                    .to_compile_error();
            };
            let fp = self.funcptr_path();
            (
                quote! { #[::majit_macros::call_aroundstate_target(funcptr = #fp, save_err = #lit)] },
                quote! { const _: () = assert!(#lit == #save_err); },
            )
        } else {
            (quote! {}, quote! {})
        };

        let body = if invoke {
            quote! {
                // `rgil.release` / `rgil.acquire`: `before_external_block` also leaves the STW RUNNING census.
                let __guard = ::majit_gc::gc_sync::before_external_block();
                if #save_err != 0 {
                    ::majit_rlib::rposix::_errno_before(#save_err);
                }
                let __res = unsafe { #call_expr };
                if #save_err != 0 {
                    ::majit_rlib::rposix::_errno_after(#save_err);
                }
                drop(__guard);
                __res
            }
        } else {
            quote! {
                if #save_err != 0 {
                    ::majit_rlib::rposix::_errno_before(#save_err);
                }
                let __res = unsafe { #call_expr };
                if #save_err != 0 {
                    ::majit_rlib::rposix::_errno_after(#save_err);
                }
                __res
            }
        };

        // The around-handler body is one shape on every target. The
        // non-around body depends on `calling_conv`: the default is `"c"`
        // except on windows when a wrapper exists, where `llexternal` uses
        // `"unknown"` and forces `need_wrapper`.
        let direct_on_c = !invoke && !self.macro_path.is_some() && !save_err_nonzero;
        let specified_not_c = self.calling_conv.as_deref().is_some_and(|c| c != "c");
        let need_wrapper =
            !invoke && (self.macro_path.is_some() || save_err_nonzero || specified_not_c);

        if invoke || need_wrapper {
            let header = if invoke {
                // `jit_close_stack` is innermost so it sees the function;
                // `call_aroundstate_target` then sees that function plus the
                // close-stack marker and adds its own sibling const.
                quote! {
                    #[inline(never)]
                    #around_attr
                    #[::majit_macros::jit_close_stack]
                }
            } else {
                quote! { #[::majit_macros::dont_look_inside] }
            };
            let vis = &self.vis;
            quote! {
                #deriv
                #eci
                #around_assert
                #funcptr
                #header
                #vis unsafe fn #call_name(#(#params),*) -> #result {
                    #body
                }
                #vis unsafe fn #name(#(#params),*) -> #result {
                    unsafe { #call_name(#(#arg_names),*) }
                }
            }
        } else if direct_on_c && self.calling_conv.is_none() {
            // unix: calling_conv "c" and no errno → call the funcptr.
            // windows: calling_conv "unknown" → wrapper (`need_wrapper`).
            // `dont_look_inside` is not applied under `cfg(windows)`: a proc-macro
            // attribute is expanded before cfg-elimination and would emit the
            // wrapper on every target.
            let vis = &self.vis;
            quote! {
                #deriv
                #eci
                #funcptr
                #[cfg(not(windows))]
                #vis unsafe fn #name(#(#params),*) -> #result {
                    unsafe { #call_expr }
                }
                #[cfg(windows)]
                #[inline(never)]
                #vis unsafe fn #call_name(#(#params),*) -> #result {
                    unsafe { #call_expr }
                }
                #[cfg(windows)]
                #vis unsafe fn #name(#(#params),*) -> #result {
                    unsafe { #call_name(#(#arg_names),*) }
                }
            }
        } else {
            let vis = &self.vis;
            quote! {
                #deriv
                #eci
                #funcptr
                #vis unsafe fn #name(#(#params),*) -> #result {
                    unsafe { #call_expr }
                }
            }
        }
    }

    fn derivation_consts(&self, has_callback: bool) -> TokenStream {
        let name = &self.name;
        let sandbox = format_ident!("_SANDBOXSAFE_{}", name);
        let nowrap = format_ident!("_NOWRAPPER_{}", name);
        let conv = format_ident!("_CALLING_CONV_{}", name);
        let callback = format_ident!("_HAS_CALLBACK_{}", name);
        let around = format_ident!("_INVOKE_AROUND_HANDLERS_{}", name);
        let elidable = format_ident!("_ELIDABLE_FUNCTION_{}", name);
        let effects = format_ident!("_RANDOM_EFFECTS_ON_GCOBJS_{}", name);
        let sandbox_v = self.sandboxsafe;
        let nowrap_v = self.nowrapper;
        let elidable_v = self.elidable;
        let callback_v = has_callback;
        let around_v = match self.releasegil {
            Tri::Bool(value) => quote! { #value },
            // releasegil='auto' → not sandboxsafe and not _nowrapper
            Tri::Auto => quote! { !#sandbox && !#nowrap },
        };
        let effects_v = match self.random_effects {
            Tri::Bool(value) => quote! { #value },
            // random_effects_on_gcobjs='auto' → invoke_around_handlers or has_callback
            Tri::Auto => quote! { #around || #callback },
        };
        let conv_expr = if let Some(conv) = &self.calling_conv {
            quote! { #conv }
        } else if self.nowrapper {
            quote! { "c" }
        } else {
            quote! {
                {
                    #[cfg(windows)]
                    { "unknown" }
                    #[cfg(not(windows))]
                    { "c" }
                }
            }
        };
        quote! {
            #[allow(dead_code, non_upper_case_globals)]
            const #sandbox: bool = #sandbox_v;
            #[allow(dead_code, non_upper_case_globals)]
            const #nowrap: bool = #nowrap_v;
            #[allow(dead_code, non_upper_case_globals)]
            const #conv: &str = #conv_expr;
            #[allow(dead_code, non_upper_case_globals)]
            const #callback: bool = #callback_v;
            #[allow(dead_code, non_upper_case_globals)]
            const #around: bool = {
                let _ = #conv;
                #around_v
            };
            #[allow(dead_code, non_upper_case_globals)]
            const #elidable: bool = #elidable_v;
            #[allow(dead_code, non_upper_case_globals)]
            const #effects: bool = #effects_v;
            const _: () = assert!(!(#elidable && #effects));
        }
    }

    fn params(&self) -> (Vec<TokenStream>, Vec<Ident>) {
        self.args
            .iter()
            .enumerate()
            .map(|(i, ty)| {
                let ident = format_ident!("a{i}");
                (quote! { #ident: #ty }, ident)
            })
            .unzip()
    }

    fn funcptr_path(&self) -> TokenStream {
        if let Some(path) = &self.macro_path {
            quote! { #path }
        } else {
            let fp = format_ident!("__rffi_fp_{}", self.name);
            quote! { #fp }
        }
    }

    fn call_expr(&self, arg_names: &[Ident]) -> TokenStream {
        let path = self.funcptr_path();
        quote! { #path(#(#arg_names),*) }
    }

    fn funcptr_item(&self, nowrapper: bool) -> TokenStream {
        if let Some(path) = &self.macro_path {
            let name = &self.name;
            let vis = &self.vis;
            return if nowrapper {
                quote! { #vis use #path as #name; }
            } else {
                quote! {}
            };
        }
        let c_name = &self.c_name;
        let result = &self.result;
        let rust_name = if nowrapper {
            self.name.clone()
        } else {
            format_ident!("__rffi_fp_{}", self.name)
        };
        let (fixed, variadic) = self.extern_params();
        let self_vis = &self.vis;
        let vis = if nowrapper {
            quote! { #self_vis }
        } else {
            quote! {}
        };
        let decl = if variadic && fixed.is_empty() {
            quote! { #vis fn #rust_name(...) -> #result; }
        } else if variadic {
            quote! { #vis fn #rust_name(#(#fixed),*, ...) -> #result; }
        } else {
            quote! { #vis fn #rust_name(#(#fixed),*) -> #result; }
        };
        quote! {
            unsafe extern "C" {
                #[link_name = #c_name]
                #decl
            }
        }
    }

    fn extern_params(&self) -> (Vec<TokenStream>, bool) {
        let variadic = self.natural_arity != -1;
        let fixed_n = if variadic {
            usize::try_from(self.natural_arity).unwrap_or(self.args.len())
        } else {
            self.args.len()
        };
        let n = fixed_n.min(self.args.len());
        let fixed = if variadic {
            self.args
                .iter()
                .take(n)
                .enumerate()
                .map(|(i, ty)| {
                    let ident = format_ident!("a{i}");
                    quote! { #ident: #ty }
                })
                .collect()
        } else {
            self.params().0
        };
        (fixed, variadic)
    }
}

const FORBIDDEN_NONEMPTY: &[&str] = &[
    "library_dirs",
    "link_extra",
    "separate_module_sources",
    "separate_module_files",
    "compile_extra",
];

struct EciInput {
    vis: Visibility,
    name: Ident,
    fields: Vec<(Ident, EciValue)>,
}

enum EciValue {
    Strs(Vec<LitStr>),
    Bool(bool),
}

impl Parse for EciInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let vis = input.parse()?;
        input.parse::<Token![const]>()?;
        let name: Ident = input.parse()?;
        input.parse::<Token![=]>()?;
        let body;
        syn::braced!(body in input);
        let mut fields = Vec::new();
        while !body.is_empty() {
            let key: Ident = body.parse()?;
            body.parse::<Token![:]>()?;
            let value = if body.peek(LitBool) {
                EciValue::Bool(body.parse::<LitBool>()?.value())
            } else {
                let list;
                syn::bracketed!(list in body);
                let mut strs = Vec::new();
                while !list.is_empty() {
                    strs.push(list.parse::<LitStr>()?);
                    if list.is_empty() {
                        break;
                    }
                    list.parse::<Token![,]>()?;
                }
                EciValue::Strs(strs)
            };
            fields.push((key, value));
            let _ = body.parse::<Token![,]>();
        }
        let _ = input.parse::<Token![;]>();
        Ok(EciInput { vis, name, fields })
    }
}

impl EciInput {
    fn expand(&self) -> syn::Result<TokenStream> {
        let mut strings: std::collections::BTreeMap<String, Vec<String>> =
            std::collections::BTreeMap::new();
        let mut use_cpp = false;
        for (key, value) in &self.fields {
            let name = key.to_string();
            match value {
                EciValue::Bool(value) => {
                    if name != "use_cpp_linker" {
                        return Err(syn::Error::new(key.span(), "only use_cpp_linker is a bool"));
                    }
                    use_cpp = *value;
                }
                EciValue::Strs(strs) => {
                    if !ECI_FIELDS.contains(&name.as_str()) {
                        return Err(syn::Error::new(
                            key.span(),
                            format!("unknown ExternalCompilationInfo field `{name}`"),
                        ));
                    }
                    let values: Vec<String> = strs.iter().map(LitStr::value).collect();
                    if FORBIDDEN_NONEMPTY.contains(&name.as_str()) && !values.is_empty() {
                        let message = format!(
                            "`{name}` needs a C build step or a Cargo search path; it must be empty"
                        );
                        return Ok(syn::Error::new(key.span(), message).to_compile_error());
                    }
                    strings.insert(name, values);
                }
            }
        }
        let vis = &self.vis;
        let name = &self.name;
        let field_inits = ECI_FIELDS.iter().map(|field| {
            let ident = Ident::new(field, Span::call_site());
            let values = strings.get(*field).cloned().unwrap_or_default();
            quote! { #ident: &[#(#values),*] }
        });
        let mut links = TokenStream::new();
        for lib in strings.get("libraries").into_iter().flatten() {
            links.extend(quote! {
                #[link(name = #lib)]
                unsafe extern "C" {}
            });
        }
        for framework in strings.get("frameworks").into_iter().flatten() {
            links.extend(quote! {
                #[link(name = #framework, kind = "framework")]
                unsafe extern "C" {}
            });
        }
        Ok(quote! {
            #vis const #name: ::majit_rlib::rffi::ExternalCompilationInfo =
                ::majit_rlib::rffi::ExternalCompilationInfo {
                    #(#field_inits,)*
                    use_cpp_linker: #use_cpp,
                };
            #links
        })
    }
}

pub fn expand_jit_close_stack(item: TokenStream) -> TokenStream {
    match close_stack_marker(item) {
        Ok(tokens) => tokens,
        Err(err) => err.to_compile_error(),
    }
}

fn close_stack_marker(item: TokenStream) -> syn::Result<TokenStream> {
    let file = syn::parse2::<syn::File>(item)?;
    let func = file.items.iter().find_map(|item| match item {
        syn::Item::Fn(func) => Some(func),
        _ => None,
    });
    let Some(func) = func else {
        return Err(syn::Error::new(
            Span::call_site(),
            "#[jit_close_stack] supports free functions",
        ));
    };
    let vis = &func.vis;
    let marker = format_ident!("_gctransformer_hint_close_stack_{}", func.sig.ident);
    Ok(quote! {
        #file
        #[doc(hidden)]
        #[allow(non_upper_case_globals, dead_code)]
        #vis const #marker: bool = true;
    })
}

pub fn expand_call_aroundstate_target(attr: TokenStream, item: TokenStream) -> TokenStream {
    match aroundstate_marker(attr, item) {
        Ok(tokens) => tokens,
        Err(err) => err.to_compile_error(),
    }
}

struct AroundStateAttr {
    funcptr: syn::Path,
    save_err: LitInt,
}

impl Parse for AroundStateAttr {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut funcptr = None;
        let mut save_err = None;
        while !input.is_empty() {
            let key = Ident::parse_any(input)?;
            input.parse::<Token![=]>()?;
            match key.to_string().as_str() {
                "funcptr" => funcptr = Some(input.parse()?),
                "save_err" => save_err = Some(input.parse()?),
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown call_aroundstate_target argument `{other}`"),
                    ));
                }
            }
            let _ = input.parse::<Token![,]>();
        }
        Ok(AroundStateAttr {
            funcptr: funcptr.ok_or_else(|| input.error("missing funcptr"))?,
            save_err: save_err.ok_or_else(|| input.error("missing save_err"))?,
        })
    }
}

fn aroundstate_marker(attr: TokenStream, item: TokenStream) -> syn::Result<TokenStream> {
    let args: AroundStateAttr = syn::parse2(attr)?;
    let file = syn::parse2::<syn::File>(item)?;
    let func = file.items.iter().find_map(|item| match item {
        syn::Item::Fn(func) => Some(func),
        _ => None,
    });
    let Some(func) = func else {
        return Err(syn::Error::new(
            Span::call_site(),
            "#[call_aroundstate_target] supports free functions",
        ));
    };
    let vis = &func.vis;
    let marker = format_ident!("_call_aroundstate_target_{}", func.sig.ident);
    let arg_tys = func.sig.inputs.iter().map(|arg| match arg {
        syn::FnArg::Typed(pat) => Ok(&pat.ty),
        syn::FnArg::Receiver(receiver) => Err(syn::Error::new_spanned(
            receiver,
            "#[call_aroundstate_target] supports free functions",
        )),
    });
    let mut tys = Vec::new();
    for ty in arg_tys {
        tys.push(ty?);
    }
    let ret = match &func.sig.output {
        syn::ReturnType::Default => quote! { () },
        syn::ReturnType::Type(_, ty) => quote! { #ty },
    };
    let funcptr = &args.funcptr;
    let save_err = &args.save_err;
    Ok(quote! {
        #file
        #[doc(hidden)]
        #[allow(non_upper_case_globals, dead_code)]
        #vis const #marker: (unsafe extern "C" fn(#(#tys),*) -> #ret, i64) =
            (#funcptr, #save_err);
    })
}

const ECI_FIELDS: &[&str] = &[
    "pre_include_bits",
    "includes",
    "include_dirs",
    "post_include_bits",
    "libraries",
    "library_dirs",
    "separate_module_sources",
    "separate_module_files",
    "compile_extra",
    "link_extra",
    "frameworks",
    "link_files",
    "testonly_libraries",
];
