/// Proc macros for the majit JIT framework.
///
/// Provides:
/// - #[jit_driver]: Annotate an interpreter's main dispatch loop
/// - #[elidable]: Mark a function as pure (constant-foldable)
/// - #[dont_look_inside]: Prevent tracing into a function
use proc_macro::TokenStream;
use quote::quote;
use syn::{parse::Parse, parse::ParseStream, parse_macro_input, Ident, ItemFn, Token};

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
    };

    expanded.into()
}

#[cfg(test)]
mod tests {
    // Proc macro crates cannot have unit tests that invoke the macros directly.
    // Integration tests and compile-time tests are used instead.
    // The parse logic is validated via the proc macro invocations in dependent crates.
}
