/// Proc macros for the majit JIT framework.
///
/// Provides:
/// - #[jit_driver]: Annotate an interpreter's main dispatch loop
/// - #[jit_interp]: Auto-generate trace_instruction and JitState from dispatch
/// - #[jit_inline]: Serialize a simple integer helper into a hidden sub-JitCode
/// - #[elidable]: Mark a function as pure (constant-foldable)
/// - #[dont_look_inside]: Prevent tracing into a function
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse::Parse, parse::ParseStream, parse_macro_input, Ident, ItemFn, Path, ReturnType, Token,
    Type,
};

mod jit_interp;

struct JitInlineArgs {
    calls: Vec<(Path, Option<Ident>)>,
}

impl Parse for JitInlineArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut calls = Vec::new();
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            match key.to_string().as_str() {
                "calls" => {
                    let content;
                    syn::braced!(content in input);
                    while !content.is_empty() {
                        let func: Path = content.parse()?;
                        let kind = if content.peek(Token![=>]) {
                            content.parse::<Token![=>]>()?;
                            let kind: Ident = content.parse()?;
                            match kind.to_string().as_str() {
                                "residual_void" | "residual_int" | "elidable_int"
                                | "inline_int" => {}
                                _ => {
                                    return Err(syn::Error::new(
                                        kind.span(),
                                        "#[jit_inline(calls = { ... })] supports only residual_void, residual_int, elidable_int, inline_int",
                                    ));
                                }
                            }
                            Some(kind)
                        } else {
                            None
                        };
                        calls.push((func, kind));
                        let _ = content.parse::<Token![,]>();
                    }
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown jit_inline parameter: `{other}`"),
                    ));
                }
            }
            let _ = input.parse::<Token![,]>();
        }
        Ok(Self { calls })
    }
}

fn helper_policy_fn_name(path: &Path) -> syn::Result<Ident> {
    let last = path.segments.last().ok_or_else(|| {
        syn::Error::new_spanned(path, "helper path must have at least one path segment")
    })?;
    Ok(format_ident!("__majit_call_policy_{}", last.ident))
}

fn helper_call_target_fn_name(path: &Path) -> syn::Result<Ident> {
    let last = path.segments.last().ok_or_else(|| {
        syn::Error::new_spanned(path, "helper path must have at least one path segment")
    })?;
    Ok(format_ident!("__majit_call_target_{}", last.ident))
}

fn primitive_type_ident(ty: &Type) -> Option<&Ident> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    if type_path.qself.is_some() || type_path.path.segments.len() != 1 {
        return None;
    }
    Some(&type_path.path.segments.last()?.ident)
}

fn is_supported_helper_return_type(ty: &Type) -> bool {
    matches!(
        primitive_type_ident(ty).map(|ident| ident.to_string()).as_deref(),
        Some("i64" | "isize")
    )
}

fn helper_arg_from_i64(arg_ident: &Ident, ty: &Type) -> Option<proc_macro2::TokenStream> {
    let ty_ident = primitive_type_ident(ty)?;
    match ty_ident.to_string().as_str() {
        "i8" | "i16" | "i32" | "isize" | "u8" | "u16" | "u32" | "u64" | "usize" => {
            Some(quote! { (#arg_ident) as #ty })
        }
        "i64" => Some(quote! { #arg_ident }),
        "bool" => Some(quote! { (#arg_ident) != 0 }),
        _ => None,
    }
}

fn helper_return_to_i64(
    value: proc_macro2::TokenStream,
    ty: &Type,
) -> Option<proc_macro2::TokenStream> {
    let ty_ident = primitive_type_ident(ty)?;
    match ty_ident.to_string().as_str() {
        "i64" => Some(quote! { #value }),
        "isize" => Some(quote! { (#value) as i64 }),
        _ => None,
    }
}

fn emit_helper_call_target_fn(
    func: &ItemFn,
) -> syn::Result<Option<(Ident, proc_macro2::TokenStream)>> {
    if !func.sig.generics.params.is_empty() {
        return Ok(None);
    }

    let target_name = helper_call_target_fn_name(&Path::from(func.sig.ident.clone()))?;
    let mut wrapper_params = Vec::new();
    let mut converted_args = Vec::new();
    for (index, arg) in func.sig.inputs.iter().enumerate() {
        let syn::FnArg::Typed(pat_type) = arg else {
            return Ok(None);
        };
        let arg_ident = format_ident!("__majit_arg_{index}");
        wrapper_params.push(quote! { #arg_ident: i64 });
        let Some(converted) = helper_arg_from_i64(&arg_ident, &pat_type.ty) else {
            return Ok(None);
        };
        converted_args.push(converted);
    }

    let helper_name = &func.sig.ident;
    let wrapper = match &func.sig.output {
        ReturnType::Default => quote! {
            #[doc(hidden)]
            pub(crate) extern "C" fn #target_name(#(#wrapper_params),*) {
                #helper_name(#(#converted_args),*);
            }
        },
        ReturnType::Type(_, ty) if is_supported_helper_return_type(ty) => {
            let Some(converted_return) = helper_return_to_i64(quote! {
                #helper_name(#(#converted_args),*)
            }, ty) else {
                return Ok(None);
            };
            quote! {
                #[doc(hidden)]
                pub(crate) extern "C" fn #target_name(#(#wrapper_params),*) -> i64 {
                    #converted_return
                }
            }
        }
        _ => return Ok(None),
    };

    Ok(Some((target_name, wrapper)))
}

fn helper_policy_tokens_for_fn(
    func: &ItemFn,
    attr_name: &str,
    call_target_name: Option<&Ident>,
) -> syn::Result<proc_macro2::TokenStream> {
    let unsupported = quote! { (0u8, std::ptr::null(), std::ptr::null()) };
    let Some(call_target_name) = call_target_name else {
        return Ok(unsupported);
    };
    match &func.sig.output {
        ReturnType::Default => Ok(match attr_name {
            "dont_look_inside" => quote! { (1u8, std::ptr::null(), #call_target_name as *const ()) },
            _ => unsupported,
        }),
        ReturnType::Type(_, ty) if is_supported_helper_return_type(ty) => Ok(match attr_name {
            "elidable" => quote! { (3u8, std::ptr::null(), #call_target_name as *const ()) },
            "dont_look_inside" => quote! { (2u8, std::ptr::null(), #call_target_name as *const ()) },
            _ => unsupported,
        }),
        _ => Ok(unsupported),
    }
}

fn emit_helper_policy_fn(
    path: &Path,
    body: proc_macro2::TokenStream,
) -> syn::Result<proc_macro2::TokenStream> {
    let helper_name = helper_policy_fn_name(path)?;
    Ok(quote! {
        #[doc(hidden)]
        pub(crate) fn #helper_name() -> (u8, *const (), *const ()) {
            #body
        }
    })
}

/// Parsed contents of `#[jit_driver(greens = [...], reds = [...])]`.
struct JitDriverArgs {
    greens: Vec<Ident>,
    reds: Vec<Ident>,
}

/// Parse a bracketed list of identifiers: `[a, b, c]`.
fn parse_ident_list(input: ParseStream) -> syn::Result<Vec<Ident>> {
    let content;
    syn::bracketed!(content in input);
    let idents = content.parse_terminated(Ident::parse, Token![,])?;
    Ok(idents.into_iter().collect())
}

impl Parse for JitDriverArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut greens = None;
        let mut reds = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            match key.to_string().as_str() {
                "greens" => {
                    if greens.is_some() {
                        return Err(syn::Error::new(key.span(), "duplicate `greens`"));
                    }
                    greens = Some(parse_ident_list(input)?);
                }
                "reds" => {
                    if reds.is_some() {
                        return Err(syn::Error::new(key.span(), "duplicate `reds`"));
                    }
                    reds = Some(parse_ident_list(input)?);
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown jit_driver parameter: `{other}`"),
                    ));
                }
            }

            // Consume optional trailing comma between greens and reds
            let _ = input.parse::<Token![,]>();
        }

        let greens = greens
            .ok_or_else(|| syn::Error::new(proc_macro2::Span::call_site(), "missing `greens`"))?;
        let reds =
            reds.ok_or_else(|| syn::Error::new(proc_macro2::Span::call_site(), "missing `reds`"))?;

        Ok(JitDriverArgs { greens, reds })
    }
}

/// Mark a struct as a JIT driver configuration.
///
/// Usage:
/// ```ignore
/// #[majit::jit_driver(
///     greens = [next_instr, pycode],
///     reds = [frame, ec],
/// )]
/// struct MyJitDriver;
/// ```
///
/// Generates an `impl` block with associated constants describing the green
/// and red variable names, their counts, and the total number of JIT
/// variables.
#[proc_macro_attribute]
pub fn jit_driver(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as JitDriverArgs);

    let input: syn::DeriveInput = match syn::parse(item) {
        Ok(v) => v,
        Err(e) => return e.to_compile_error().into(),
    };

    let struct_name = &input.ident;

    let green_strs: Vec<String> = args.greens.iter().map(|id| id.to_string()).collect();
    let red_strs: Vec<String> = args.reds.iter().map(|id| id.to_string()).collect();
    let greens_joined = green_strs.join(", ");
    let reds_joined = red_strs.join(", ");

    let num_greens = green_strs.len();
    let num_reds = red_strs.len();
    let num_vars = num_greens + num_reds;

    let doc = format!("JIT driver: greens=[{greens_joined}], reds=[{reds_joined}]");

    let attrs = &input.attrs;
    let vis = &input.vis;
    let generics = &input.generics;
    let data = &input.data;

    // Re-emit the struct with doc annotation, then add the impl block.
    let struct_token = match data {
        syn::Data::Struct(s) => {
            let fields = &s.fields;
            let semi = &s.semi_token;
            quote! {
                #(#attrs)*
                #[doc = #doc]
                #vis struct #struct_name #generics #fields #semi
            }
        }
        _ => {
            return syn::Error::new_spanned(&input, "jit_driver can only be applied to structs")
                .to_compile_error()
                .into();
        }
    };

    let expanded = quote! {
        #struct_token

        impl #generics #struct_name #generics {
            /// Green variable names.
            pub const GREENS: &'static [&'static str] = &[#(#green_strs),*];
            /// Red variable names.
            pub const REDS: &'static [&'static str] = &[#(#red_strs),*];
            /// Total number of JIT variables.
            pub const NUM_VARS: usize = #num_vars;
            /// Number of green variables.
            pub const NUM_GREENS: usize = #num_greens;
            /// Number of red variables.
            pub const NUM_REDS: usize = #num_reds;
        }
    };

    expanded.into()
}

/// Mark a function as elidable (pure / constant-foldable).
///
/// The JIT can eliminate calls to this function when all arguments are constants.
/// Adds `#[inline(never)]` to prevent inlining, and a hidden `#[majit_elidable]`
/// marker attribute that the tracer can detect at compile time.
#[proc_macro_attribute]
pub fn elidable(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let attrs = &func.attrs;
    let vis = &func.vis;
    let sig = &func.sig;
    let block = &func.block;
    let policy_path = Path::from(sig.ident.clone());
    let (call_target_name, call_target_fn) = match emit_helper_call_target_fn(&func) {
        Ok(Some((name, tokens))) => (Some(name), Some(tokens)),
        Ok(None) => (None, None),
        Err(err) => return err.to_compile_error().into(),
    };
    let policy_fn = match emit_helper_policy_fn(
        &policy_path,
        match helper_policy_tokens_for_fn(&func, "elidable", call_target_name.as_ref()) {
            Ok(tokens) => tokens,
            Err(err) => return err.to_compile_error().into(),
        },
    ) {
        Ok(tokens) => tokens,
        Err(err) => return err.to_compile_error().into(),
    };

    let expanded = quote! {
        #(#attrs)*
        #[inline(never)]
        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        #vis #sig {
            #[doc(hidden)]
            #[allow(dead_code)]
            const _MAJIT_ELIDABLE: bool = true;
            #block
        }

        #call_target_fn
        #policy_fn
    };

    expanded.into()
}

/// Mark a function as opaque to the tracer.
///
/// The JIT will not trace into this function; it will be called as a black box.
/// Adds `#[inline(never)]` to prevent inlining, and a hidden `#[majit_opaque]`
/// marker constant that the tracer can detect at compile time.
#[proc_macro_attribute]
pub fn dont_look_inside(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let func = parse_macro_input!(item as ItemFn);
    let attrs = &func.attrs;
    let vis = &func.vis;
    let sig = &func.sig;
    let block = &func.block;
    let policy_path = Path::from(sig.ident.clone());
    let (call_target_name, call_target_fn) = match emit_helper_call_target_fn(&func) {
        Ok(Some((name, tokens))) => (Some(name), Some(tokens)),
        Ok(None) => (None, None),
        Err(err) => return err.to_compile_error().into(),
    };
    let policy_fn = match emit_helper_policy_fn(
        &policy_path,
        match helper_policy_tokens_for_fn(&func, "dont_look_inside", call_target_name.as_ref()) {
            Ok(tokens) => tokens,
            Err(err) => return err.to_compile_error().into(),
        },
    ) {
        Ok(tokens) => tokens,
        Err(err) => return err.to_compile_error().into(),
    };

    let expanded = quote! {
        #(#attrs)*
        #[inline(never)]
        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        #vis #sig {
            #[doc(hidden)]
            #[allow(dead_code)]
            const _MAJIT_OPAQUE: bool = true;
            #block
        }

        #call_target_fn
        #policy_fn
    };

    expanded.into()
}

/// Serialize a simple integer helper into a hidden `JitCode` builder.
///
/// This is the proc-macro side of RPython's `codewriter.py` helper serialization:
/// the original function stays callable by the interpreter, and the macro also
/// emits a hidden `__majit_inline_jitcode_*()` function that `#[jit_interp]`
/// can use when a call policy maps the helper to `inline_int`.
#[proc_macro_attribute]
pub fn jit_inline(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as JitInlineArgs);
    let func = parse_macro_input!(item as ItemFn);
    let helper = match jit_interp::jitcode_lower::generate_inline_helper_jitcode_with_calls(
        &func,
        &args.calls,
    ) {
        Ok(Some(lowered)) => lowered,
        Ok(None) => {
            return syn::Error::new_spanned(
                &func.block,
                "#[jit_inline] could not lower this helper into JitCode",
            )
            .to_compile_error()
            .into();
        }
        Err(err) => return err.to_compile_error().into(),
    };

    let attrs = &func.attrs;
    let vis = &func.vis;
    let sig = &func.sig;
    let block = &func.block;
    let helper_name = format_ident!("__majit_inline_jitcode_{}", sig.ident);
    let policy_name = format_ident!("__majit_call_policy_{}", sig.ident);
    let helper_body = helper.body;
    let param_count = helper.param_count;
    let return_reg = helper.return_reg;

    let expanded = quote! {
        #(#attrs)*
        #vis #sig {
            #block
        }

        #[doc(hidden)]
        pub(crate) fn #helper_name() -> (majit_meta::JitCode, u16) {
            let mut __builder = majit_meta::JitCodeBuilder::new();
            __builder.ensure_i_regs(#param_count);
            #helper_body
            (__builder.finish(), #return_reg)
        }

        #[doc(hidden)]
        pub(crate) fn #policy_name() -> (u8, *const (), *const ()) {
            (4u8, #helper_name as *const (), std::ptr::null())
        }
    };

    expanded.into()
}

/// Auto-generate trace_instruction and JitState from an interpreter's dispatch loop.
///
/// This is the Rust equivalent of RPython's meta-tracing: the proc macro analyzes
/// the interpreter's opcode dispatch match and generates the tracing code automatically.
///
/// The interpreter author writes ONLY the dispatch loop (like RPython's rpaheui).
/// The macro generates:
/// - `trace_instruction()` function (IR recording for each opcode)
/// - `JitState` impl with Meta/Sym types
/// - Replaces `jit_merge_point!()` / `can_enter_jit!()` markers with JitDriver calls
///
/// # Example
///
/// ```ignore
/// #[jit_interp(
///     state = AheuiState,
///     env = Program,
///     storage = {
///         pool: state.storage,
///         selector: state.selected,
///         untraceable: [VAL_QUEUE, VAL_PORT],
///         scan: find_used_storages,
///     },
///     binops = {
///         add => IntAdd, sub => IntSub, mul => IntMul,
///         div => IntFloorDiv, modulo => IntMod, cmp => IntGe,
///     },
///     io_shims = {
///         aheui_io::write_number => jit_write_number,
///         aheui_io::write_utf8 => jit_write_utf8,
///     },
///     // optional: infer direct helper calls from sidecar metadata
///     auto_calls = true,
///     calls = {
///         helper_compute,
///         helper_opaque,
///         helper_sink,
///         helper_inline,
///         // explicit overrides are still allowed:
///         helper_force_residual => residual_int,
///     },
/// )]
/// pub fn mainloop_jit(program: &Program) -> i64 {
///     // ... setup ...
///     while pc < program.size {
///         jit_merge_point!();
///         match op {
///             OP_ADD => state.storage.get_mut(state.selected).add(),
///             // ...
///         }
///     }
/// }
/// ```
#[proc_macro_attribute]
pub fn jit_interp(attr: TokenStream, item: TokenStream) -> TokenStream {
    let config = parse_macro_input!(attr as jit_interp::JitInterpConfig);
    let func = parse_macro_input!(item as ItemFn);
    jit_interp::transform_jit_interp(config, func).into()
}

#[cfg(test)]
mod tests {
    // Proc macro crates cannot have unit tests that invoke the macros directly.
    // Integration tests and compile-time tests are used instead.
    // The parse logic is validated via the proc macro invocations in dependent crates.
}
