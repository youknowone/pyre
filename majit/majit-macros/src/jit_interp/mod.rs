//! `#[jit_interp]` proc macro implementation.
//!
//! Transforms an interpreter mainloop function into a JIT-enabled version by:
//! 1. Generating `trace_instruction` from the match dispatch
//! 2. Generating `JitState` types and impl
//! 3. Replacing `jit_merge_point!()` / `can_enter_jit!()` markers

mod classify;
mod codegen_state;
mod codegen_trace;
pub(crate) mod jitcode_lower;

use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    braced, bracketed,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
    Expr, Ident, ItemFn, LitBool, Path, Token,
};

/// Parsed configuration from `#[jit_interp(...)]` attributes.
pub struct JitInterpConfig {
    /// The interpreter state type (e.g., `AheuiState`).
    pub state_type: Ident,
    /// The environment type (e.g., `Program`).
    pub env_type: Ident,
    /// Multi-storage configuration.
    pub storage: StorageConfig,
    /// Method name → IR opcode mapping for binary operations.
    pub binops: Vec<(Ident, Ident)>,
    /// Interpreter I/O function → JIT shim function mapping.
    pub io_shims: Vec<(Path, Ident)>,
    /// Interpreter function call policies for helper calls.
    pub calls: Vec<(Path, Option<Ident>)>,
    /// Whether direct helper calls should be auto-inferred from sidecar metadata.
    pub auto_calls: bool,
}

/// Multi-storage configuration parsed from `storage = { ... }`.
pub struct StorageConfig {
    /// Expression to access the storage pool (e.g., `state.storage`).
    pub pool: Expr,
    /// Type of the storage pool (e.g., `StoragePool`).
    pub pool_type: Path,
    /// Expression to access the selected index (e.g., `state.selected`).
    pub selector: Expr,
    /// Storage indices that cannot be traced (e.g., `[VAL_QUEUE, VAL_PORT]`).
    pub untraceable: Vec<Path>,
    /// Function to scan for used storages (e.g., `find_used_storages`).
    pub scan_fn: Ident,
}

impl Parse for JitInterpConfig {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut state_type = None;
        let mut env_type = None;
        let mut storage = None;
        let mut binops = None;
        let mut io_shims = None;
        let mut calls = None;
        let mut auto_calls = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            match key.to_string().as_str() {
                "state" => {
                    state_type = Some(input.parse::<Ident>()?);
                }
                "env" => {
                    env_type = Some(input.parse::<Ident>()?);
                }
                "storage" => {
                    storage = Some(parse_storage_config(input)?);
                }
                "binops" => {
                    binops = Some(parse_binop_map(input)?);
                }
                "io_shims" => {
                    io_shims = Some(parse_io_shim_map(input)?);
                }
                "calls" => {
                    calls = Some(parse_call_map(input)?);
                }
                "auto_calls" => {
                    auto_calls = Some(input.parse::<LitBool>()?.value);
                }
                other => {
                    return Err(syn::Error::new(
                        key.span(),
                        format!("unknown jit_interp parameter: `{other}`"),
                    ));
                }
            }

            let _ = input.parse::<Token![,]>();
        }

        let state_type =
            state_type.ok_or_else(|| syn::Error::new(input.span(), "missing `state` parameter"))?;
        let env_type =
            env_type.ok_or_else(|| syn::Error::new(input.span(), "missing `env` parameter"))?;
        let storage =
            storage.ok_or_else(|| syn::Error::new(input.span(), "missing `storage` parameter"))?;

        Ok(JitInterpConfig {
            state_type,
            env_type,
            storage,
            binops: binops.unwrap_or_default(),
            io_shims: io_shims.unwrap_or_default(),
            calls: calls.unwrap_or_default(),
            auto_calls: auto_calls.unwrap_or(false),
        })
    }
}

/// Parse `{ pool: EXPR, selector: EXPR, untraceable: [...], scan: IDENT }`.
fn parse_storage_config(input: ParseStream) -> syn::Result<StorageConfig> {
    let content;
    braced!(content in input);

    let mut pool = None;
    let mut pool_type = None;
    let mut selector = None;
    let mut untraceable = Vec::new();
    let mut scan_fn = None;

    while !content.is_empty() {
        let key: Ident = content.parse()?;
        content.parse::<Token![:]>()?;

        match key.to_string().as_str() {
            "pool" => {
                pool = Some(content.parse::<Expr>()?);
            }
            "pool_type" => {
                pool_type = Some(content.parse::<Path>()?);
            }
            "selector" => {
                selector = Some(content.parse::<Expr>()?);
            }
            "untraceable" => {
                let inner;
                bracketed!(inner in content);
                let paths: Punctuated<Path, Token![,]> =
                    inner.parse_terminated(Path::parse, Token![,])?;
                untraceable = paths.into_iter().collect();
            }
            "scan" => {
                scan_fn = Some(content.parse::<Ident>()?);
            }
            other => {
                return Err(syn::Error::new(
                    key.span(),
                    format!("unknown storage parameter: `{other}`"),
                ));
            }
        }

        let _ = content.parse::<Token![,]>();
    }

    let pool =
        pool.ok_or_else(|| syn::Error::new(content.span(), "missing `pool` in storage config"))?;
    let pool_type = pool_type
        .ok_or_else(|| syn::Error::new(content.span(), "missing `pool_type` in storage config"))?;
    let selector = selector
        .ok_or_else(|| syn::Error::new(content.span(), "missing `selector` in storage config"))?;
    let scan_fn = scan_fn
        .ok_or_else(|| syn::Error::new(content.span(), "missing `scan` in storage config"))?;

    Ok(StorageConfig {
        pool,
        pool_type,
        selector,
        untraceable,
        scan_fn,
    })
}

/// Parse `{ method => OpCode, ... }`.
fn parse_binop_map(input: ParseStream) -> syn::Result<Vec<(Ident, Ident)>> {
    let content;
    braced!(content in input);
    let mut map = Vec::new();
    while !content.is_empty() {
        let method: Ident = content.parse()?;
        content.parse::<Token![=>]>()?;
        let opcode: Ident = content.parse()?;
        map.push((method, opcode));
        let _ = content.parse::<Token![,]>();
    }
    Ok(map)
}

/// Parse `{ path::func => jit_func, ... }`.
fn parse_io_shim_map(input: ParseStream) -> syn::Result<Vec<(Path, Ident)>> {
    let content;
    braced!(content in input);
    let mut map = Vec::new();
    while !content.is_empty() {
        let func: Path = content.parse()?;
        content.parse::<Token![=>]>()?;
        let shim: Ident = content.parse()?;
        map.push((func, shim));
        let _ = content.parse::<Token![,]>();
    }
    Ok(map)
}

/// Parse `{ path::func, path::func => residual_int, ... }`.
fn parse_call_map(input: ParseStream) -> syn::Result<Vec<(Path, Option<Ident>)>> {
    let content;
    braced!(content in input);
    let mut map = Vec::new();
    while !content.is_empty() {
        let func: Path = content.parse()?;
        let kind = if content.peek(Token![=>]) {
            content.parse::<Token![=>]>()?;
            let kind: Ident = content.parse()?;
            match kind.to_string().as_str() {
                "residual_void" | "residual_int" | "elidable_int" | "inline_int" => {}
                _ => {
                    return Err(syn::Error::new(
                        kind.span(),
                        "call policy must be one of residual_void, residual_int, elidable_int, inline_int",
                    ));
                }
            }
            Some(kind)
        } else {
            None
        };
        map.push((func, kind));
        let _ = content.parse::<Token![,]>();
    }
    Ok(map)
}

/// Main entry point: transform the function with JIT support.
pub fn transform_jit_interp(config: JitInterpConfig, func: ItemFn) -> TokenStream {
    let trace_fn = codegen_trace::generate_trace_fn(&config, &func);
    let state_impl = codegen_state::generate_jit_state(&config);
    let transformed_fn = transform_function(&config, &func);

    quote! {
        #state_impl
        #trace_fn
        #transformed_fn
    }
}

/// Transform the original function: replace jit_merge_point!() and can_enter_jit!() markers.
fn transform_function(config: &JitInterpConfig, func: &ItemFn) -> TokenStream {
    let vis = &func.vis;
    let sig = &func.sig;
    let attrs = &func.attrs;
    let fn_name = &func.sig.ident;
    let trace_fn_name = quote::format_ident!("__trace_{}", fn_name);

    let pool_expr = &config.storage.pool;
    let sel_expr = &config.storage.selector;

    // Rewrite the function body, replacing marker macros
    let body = rewrite_body(&func.block, &trace_fn_name, pool_expr, sel_expr);

    quote! {
        #(#attrs)*
        #vis #sig {
            #body
        }
    }
}

/// Rewrite function body: replace jit_merge_point!() and can_enter_jit!() calls.
fn rewrite_body(
    block: &syn::Block,
    trace_fn_name: &Ident,
    pool_expr: &Expr,
    sel_expr: &Expr,
) -> TokenStream {
    use syn::visit_mut::VisitMut;

    struct MarkerRewriter {
        trace_fn_name: Ident,
        pool_expr: Expr,
        sel_expr: Expr,
    }

    impl VisitMut for MarkerRewriter {
        fn visit_stmt_mut(&mut self, stmt: &mut syn::Stmt) {
            // First recurse into children
            syn::visit_mut::visit_stmt_mut(self, stmt);

            // Check if this statement is a macro invocation
            if let syn::Stmt::Macro(stmt_macro) = stmt {
                let mac = &stmt_macro.mac;
                let path_str = mac
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect::<Vec<_>>()
                    .join("::");

                if path_str == "jit_merge_point" || path_str.ends_with("::jit_merge_point") {
                    let trace_fn = &self.trace_fn_name;
                    let pool = &self.pool_expr;
                    let sel = &self.sel_expr;
                    let new_tokens: TokenStream = quote! {
                        driver.merge_point(|__ctx, __sym| {
                            #trace_fn(__ctx, __sym, program, pc, &#pool, #sel)
                        });
                    };
                    *stmt =
                        syn::parse2(new_tokens).expect("failed to parse merge_point replacement");
                }
            }
        }

        fn visit_expr_mut(&mut self, expr: &mut syn::Expr) {
            // First recurse
            syn::visit_mut::visit_expr_mut(self, expr);

            // Check for can_enter_jit!(...) macro calls within expressions
            // can_enter_jit! is used as a statement, handled below
        }

        fn visit_block_mut(&mut self, block: &mut syn::Block) {
            // Clone expressions upfront to avoid borrow conflicts
            let pool_expr = self.pool_expr.clone();
            let sel_expr = self.sel_expr.clone();

            let mut new_stmts = Vec::new();
            let mut i = 0;
            while i < block.stmts.len() {
                let stmt = &block.stmts[i];

                // Check if this is can_enter_jit!(driver, target, state, env, pre_run)
                if let syn::Stmt::Macro(stmt_macro) = stmt {
                    let mac = &stmt_macro.mac;
                    let path_str = mac
                        .path
                        .segments
                        .iter()
                        .map(|s| s.ident.to_string())
                        .collect::<Vec<_>>()
                        .join("::");

                    if path_str == "can_enter_jit" || path_str.ends_with("::can_enter_jit") {
                        let tokens = &mac.tokens;
                        if let Ok(args) = syn::parse2::<CanEnterJitArgs>(tokens.clone()) {
                            let driver_ident = &args.driver;
                            let target_expr = &args.target;
                            let state_expr = &args.state;
                            let env_expr = &args.env;
                            let pre_run_expr = &args.pre_run;
                            let pool = &pool_expr;
                            let sel = &sel_expr;
                            let back_edge: TokenStream = quote! {
                                if #driver_ident.back_edge(#target_expr, #state_expr, #env_expr, #pre_run_expr) {
                                    pc = #target_expr;
                                    stacksize = #pool.get(#sel).len() as i32;
                                    continue;
                                }
                            };
                            let parsed: syn::Stmt =
                                syn::parse2(back_edge).expect("failed to parse back_edge");
                            new_stmts.push(parsed);
                            i += 1;
                            continue;
                        }
                    }
                }

                let mut cloned = block.stmts[i].clone();
                self.visit_stmt_mut(&mut cloned);
                new_stmts.push(cloned);
                i += 1;
            }
            block.stmts = new_stmts;
        }
    }

    /// Parsed arguments for `can_enter_jit!(driver, target, &mut state, program, || { ... })`.
    struct CanEnterJitArgs {
        driver: Ident,
        target: Expr,
        state: Expr,
        env: Expr,
        pre_run: Expr,
    }

    impl Parse for CanEnterJitArgs {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            let driver: Ident = input.parse()?;
            input.parse::<Token![,]>()?;
            let target: Expr = input.parse()?;
            input.parse::<Token![,]>()?;
            let state: Expr = input.parse()?;
            input.parse::<Token![,]>()?;
            let env: Expr = input.parse()?;
            input.parse::<Token![,]>()?;
            let pre_run: Expr = input.parse()?;
            Ok(CanEnterJitArgs {
                driver,
                target,
                state,
                env,
                pre_run,
            })
        }
    }

    let mut cloned_block = block.clone();
    let mut rewriter = MarkerRewriter {
        trace_fn_name: trace_fn_name.clone(),
        pool_expr: pool_expr.clone(),
        sel_expr: sel_expr.clone(),
    };
    rewriter.visit_block_mut(&mut cloned_block);

    let stmts = &cloned_block.stmts;
    quote! { #(#stmts)* }
}
