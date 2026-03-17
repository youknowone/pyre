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
    Expr, Ident, ItemFn, LitBool, Path, Token, braced, bracketed,
    parse::{Parse, ParseStream},
    punctuated::Punctuated,
};

/// Parsed configuration from `#[jit_interp(...)]` attributes.
///
/// ## Helper discovery
///
/// Helpers (functions called from traced match arms) can be declared in
/// three ways, from most explicit to most concise:
///
/// 1. **`calls = { helper_a, helper_b => residual_int, ... }`**
///    Brace-delimited list with optional per-helper policy overrides.
///
/// 2. **`helpers = [helper_a, helper_b, helper_c]`**
///    Bracket-delimited shorthand — all helpers use auto-inferred policy.
///    Equivalent to listing each in `calls = { ... }` without a `=>` override.
///    Can be combined with `calls` for helpers that need explicit policies.
///
/// 3. **`auto_calls = true`**
///    Infer helper policies from sidecar `#[elidable]` / `#[dont_look_inside]`
///    / `#[jit_inline]` attributes on every call site in the traced arms.
///
/// ### Module-level discovery
///
/// For automatic helper discovery, use `#[jit_module]` on the enclosing
/// `mod` block. It scans all items for JIT-annotated functions and
/// generates hidden registry constants (`__MAJIT_DISCOVERED_HELPERS`,
/// `__MAJIT_HELPER_POLICIES`). Alternatively, use `helpers` or `calls`
/// to explicitly list the functions that need JIT integration.
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
    /// Populated from both `calls = { ... }` and `helpers = [...]`.
    pub calls: Vec<(Path, Option<Ident>)>,
    /// Whether direct helper calls should be auto-inferred from sidecar metadata.
    pub auto_calls: bool,
    /// Optional structured green-key expressions for marker rewrite.
    pub greens: Vec<Expr>,
    /// Virtualizable frame field declaration.
    ///
    /// RPython equivalent: jtransform.py's `is_virtualizable_getset()`.
    /// When set, the proc macro rewrites field accesses on the virtualizable
    /// variable to use TraceCtx vable_* methods instead of heap operations.
    pub virtualizable_decl: Option<VirtualizableDecl>,
    /// State field declarations for register/tape machines.
    ///
    /// When set, the macro tracks state struct fields as JIT-managed values
    /// instead of requiring a storage pool. Enables `state.field` and
    /// `state.array[index]` patterns in match arms.
    pub state_fields: Option<StateFieldsConfig>,
}

/// Virtualizable frame field declaration for `#[jit_interp]`.
///
/// RPython equivalent: VirtualizableInfo from virtualizable.py, combined
/// with jtransform.py's field-to-descriptor mapping.
///
/// Syntax in attribute:
/// ```ignore
/// virtualizable_fields = {
///     var: frame,
///     token_offset: PYFRAME_VABLE_TOKEN_OFFSET,
///     fields: { next_instr: int @ NEXT_INSTR_OFFSET, ... },
///     arrays: { locals_w: ref @ LOCALS_OFFSET, ... },
/// }
/// ```
pub struct VirtualizableDecl {
    /// Expression for the virtualizable variable in the mainloop body.
    pub var_name: Ident,
    /// Constant path for the vable_token field offset.
    pub token_offset: Path,
    /// Static fields: name, type (int/ref/float), byte offset constant.
    pub fields: Vec<VableFieldDecl>,
    /// Array fields: name, item type (int/ref/float), byte offset constant.
    pub arrays: Vec<VableArrayDecl>,
}

/// A single virtualizable static field declaration.
pub struct VableFieldDecl {
    /// Field name as it appears in the struct.
    pub name: Ident,
    /// Field type: `int`, `ref`, or `float`.
    pub field_type: Ident,
    /// Constant path for the byte offset (e.g., `PYFRAME_NEXT_INSTR_OFFSET`).
    pub offset: Path,
}

/// A single virtualizable array field declaration.
pub struct VableArrayDecl {
    /// Array field name as it appears in the struct.
    pub name: Ident,
    /// Item type: `int`, `ref`, or `float`.
    pub item_type: Ident,
    /// Constant path for the byte offset of the array pointer field.
    pub offset: Path,
}

/// State field declaration for register/tape machines.
///
/// Syntax: `state_fields = { a: int, regs: [int], ... }`
///
/// Current implementation supports only `int` and `[int]`.
pub struct StateFieldsConfig {
    pub fields: Vec<StateFieldDecl>,
}

/// A single state field declaration.
pub struct StateFieldDecl {
    pub name: Ident,
    pub kind: StateFieldKind,
}

/// Whether a state field is a scalar or an array.
pub enum StateFieldKind {
    /// Scalar value (e.g., `a: int`).
    Scalar(Ident),
    /// Array value (e.g., `regs: [int]`).
    Array(Ident),
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
    /// Optional method on StoragePool to check JIT compatibility of all values.
    /// When set, `can_trace` additionally calls `pool.method()`.
    pub can_trace_guard: Option<Ident>,
    /// When true, storage stacks are virtualizable — not flattened into
    /// inputargs. The JIT accesses elements via GETARRAYITEM/SETARRAYITEM,
    /// allowing variable stack depths across loop iterations.
    pub virtualizable: bool,
}

impl Parse for JitInterpConfig {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut state_type = None;
        let mut env_type = None;
        let mut storage = None;
        let mut binops = None;
        let mut io_shims = None;
        let mut calls: Vec<(Path, Option<Ident>)> = Vec::new();
        let mut auto_calls = None;
        let mut greens = None;
        let mut virtualizable_decl = None;
        let mut state_fields = None;

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
                    calls.extend(parse_call_map(input)?);
                }
                "helpers" => {
                    calls.extend(parse_helpers_list(input)?);
                }
                "auto_calls" => {
                    auto_calls = Some(input.parse::<LitBool>()?.value);
                }
                "greens" => {
                    greens = Some(parse_expr_list(input)?);
                }
                "virtualizable_fields" => {
                    virtualizable_decl = Some(parse_virtualizable_decl(input)?);
                }
                "state_fields" => {
                    state_fields = Some(parse_state_fields(input)?);
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

        // storage is required unless state_fields is provided.
        let storage = match (storage, &state_fields) {
            (Some(s), _) => s,
            (None, Some(_)) => {
                // Create a dummy StorageConfig for state_fields mode.
                // __DummyPool is a unit struct auto-generated in codegen output.
                StorageConfig {
                    pool: syn::parse_quote!(__DummyPool),
                    pool_type: syn::parse_quote!(__DummyPool),
                    selector: syn::parse_quote!(0usize),
                    untraceable: Vec::new(),
                    scan_fn: syn::parse_str("__dummy_scan").unwrap(),
                    can_trace_guard: None,
                    virtualizable: false,
                }
            }
            (None, None) => {
                return Err(syn::Error::new(
                    input.span(),
                    "missing `storage` or `state_fields` parameter",
                ));
            }
        };

        Ok(JitInterpConfig {
            state_type,
            env_type,
            storage,
            binops: binops.unwrap_or_default(),
            io_shims: io_shims.unwrap_or_default(),
            calls,
            auto_calls: auto_calls.unwrap_or(false),
            greens: greens.unwrap_or_default(),
            virtualizable_decl,
            state_fields,
        })
    }
}

fn parse_expr_list(input: ParseStream) -> syn::Result<Vec<Expr>> {
    let content;
    bracketed!(content in input);
    let exprs: Punctuated<Expr, Token![,]> = content.parse_terminated(Expr::parse, Token![,])?;
    Ok(exprs.into_iter().collect())
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
    let mut can_trace_guard = None;
    let mut virtualizable = false;

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
            "can_trace_guard" => {
                can_trace_guard = Some(content.parse::<Ident>()?);
            }
            "virtualizable" => {
                virtualizable = content.parse::<LitBool>()?.value;
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
        can_trace_guard,
        virtualizable,
    })
}

/// Parse virtualizable_fields = { var: IDENT, token_offset: PATH, fields: { ... }, arrays: { ... } }
///
/// Parse `state_fields = { name: type, ... }` where type is `int` or `[int]`.
fn parse_state_fields(input: ParseStream) -> syn::Result<StateFieldsConfig> {
    let content;
    braced!(content in input);
    let mut fields = Vec::new();

    while !content.is_empty() {
        let name: Ident = content.parse()?;
        content.parse::<Token![:]>()?;

        let kind = if content.peek(syn::token::Bracket) {
            // Array: [int], [ref], [float]
            let inner;
            bracketed!(inner in content);
            let item_type: Ident = inner.parse()?;
            StateFieldKind::Array(item_type)
        } else {
            // Scalar: int, ref, float
            let field_type: Ident = content.parse()?;
            StateFieldKind::Scalar(field_type)
        };

        fields.push(StateFieldDecl { name, kind });
        let _ = content.parse::<Token![,]>();
    }

    Ok(StateFieldsConfig { fields })
}

/// RPython equivalent: VirtualizableInfo construction from virtualizable.py
/// + jtransform.py's field-to-descriptor mapping.
fn parse_virtualizable_decl(input: ParseStream) -> syn::Result<VirtualizableDecl> {
    let content;
    braced!(content in input);

    let mut var_name = None;
    let mut token_offset = None;
    let mut fields = Vec::new();
    let mut arrays = Vec::new();

    while !content.is_empty() {
        let key: Ident = content.parse()?;
        content.parse::<Token![:]>()?;

        match key.to_string().as_str() {
            "var" => {
                var_name = Some(content.parse::<Ident>()?);
            }
            "token_offset" => {
                token_offset = Some(content.parse::<Path>()?);
            }
            "fields" => {
                let inner;
                braced!(inner in content);
                while !inner.is_empty() {
                    let name: Ident = inner.parse()?;
                    inner.parse::<Token![:]>()?;
                    let field_type: Ident = inner.parse()?;
                    inner.parse::<Token![@]>()?;
                    let offset: Path = inner.parse()?;
                    fields.push(VableFieldDecl {
                        name,
                        field_type,
                        offset,
                    });
                    let _ = inner.parse::<Token![,]>();
                }
            }
            "arrays" => {
                let inner;
                braced!(inner in content);
                while !inner.is_empty() {
                    let name: Ident = inner.parse()?;
                    inner.parse::<Token![:]>()?;
                    let item_type: Ident = inner.parse()?;
                    inner.parse::<Token![@]>()?;
                    let offset: Path = inner.parse()?;
                    arrays.push(VableArrayDecl {
                        name,
                        item_type,
                        offset,
                    });
                    let _ = inner.parse::<Token![,]>();
                }
            }
            other => {
                return Err(syn::Error::new(
                    key.span(),
                    format!("unknown virtualizable_fields parameter: `{other}`"),
                ));
            }
        }
        let _ = content.parse::<Token![,]>();
    }

    let var_name = var_name
        .ok_or_else(|| syn::Error::new(content.span(), "missing `var` in virtualizable_fields"))?;
    let token_offset = token_offset.ok_or_else(|| {
        syn::Error::new(
            content.span(),
            "missing `token_offset` in virtualizable_fields",
        )
    })?;

    Ok(VirtualizableDecl {
        var_name,
        token_offset,
        fields,
        arrays,
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
                "residual_void"
                | "residual_void_wrapped"
                | "may_force_void"
                | "may_force_void_wrapped"
                | "release_gil_void"
                | "release_gil_void_wrapped"
                | "loopinvariant_void"
                | "loopinvariant_void_wrapped"
                | "residual_int"
                | "residual_int_wrapped"
                | "may_force_int"
                | "may_force_int_wrapped"
                | "release_gil_int"
                | "release_gil_int_wrapped"
                | "loopinvariant_int"
                | "loopinvariant_int_wrapped"
                | "elidable_int"
                | "elidable_int_wrapped"
                | "residual_ref_wrapped"
                | "may_force_ref_wrapped"
                | "release_gil_ref_wrapped"
                | "loopinvariant_ref_wrapped"
                | "elidable_ref_wrapped"
                | "residual_float_wrapped"
                | "may_force_float_wrapped"
                | "release_gil_float_wrapped"
                | "loopinvariant_float_wrapped"
                | "elidable_float_wrapped"
                | "inline_int" => {}
                _ => {
                    return Err(syn::Error::new(
                        kind.span(),
                        "call policy must be a supported residual/may_force/release_gil/loopinvariant policy or inline_int",
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

/// Parse `[func_a, func_b, func_c]` — shorthand helper list with auto-inferred policies.
fn parse_helpers_list(input: ParseStream) -> syn::Result<Vec<(Path, Option<Ident>)>> {
    let content;
    bracketed!(content in input);
    let paths: Punctuated<Path, Token![,]> = content.parse_terminated(Path::parse, Token![,])?;
    Ok(paths.into_iter().map(|p| (p, None)).collect())
}

/// Main entry point: transform the function with JIT support.
pub fn transform_jit_interp(config: JitInterpConfig, func: ItemFn) -> TokenStream {
    let trace_fn = codegen_trace::generate_trace_fn(&config, &func);
    let state_impl = codegen_state::generate_jit_state(&config);
    let merge_wrapper = generate_merge_wrapper(&config, &func);
    let transformed_fn = transform_function(&config, &func);

    quote! {
        #state_impl
        #trace_fn
        #merge_wrapper
        #transformed_fn
    }
}

/// Generate a `#[cold]` out-of-line wrapper for the merge_point call.
///
/// This keeps the mainloop hot path thin — only an `is_tracing()` flag check
/// appears inline, while the closure capture and tracing logic live here.
fn generate_merge_wrapper(config: &JitInterpConfig, func: &ItemFn) -> TokenStream {
    let fn_name = &func.sig.ident;
    let merge_fn_name = quote::format_ident!("__merge_{}", fn_name);
    let trace_fn_name = quote::format_ident!("__trace_{}", fn_name);
    let state_type = &config.state_type;
    let env_type = &config.env_type;
    let pool_type = &config.storage.pool_type;

    quote! {
        #[cold]
        #[inline(never)]
        #[allow(non_snake_case)]
        fn #merge_fn_name(
            __driver: &mut majit_meta::JitDriver<#state_type>,
            __env: &#env_type,
            __pc: usize,
            __pool: &#pool_type,
            __sel: usize,
        ) {
            __driver.merge_point(|__ctx, __sym| {
                #trace_fn_name(__ctx, __sym, __env, __pc, __pool, __sel)
            });
        }
    }
}

/// Transform the original function: replace jit_merge_point!() and can_enter_jit!() markers.
fn transform_function(config: &JitInterpConfig, func: &ItemFn) -> TokenStream {
    let vis = &func.vis;
    let sig = &func.sig;
    let attrs = &func.attrs;
    let fn_name = &func.sig.ident;
    let trace_fn_name = quote::format_ident!("__trace_{}", fn_name);
    let merge_fn_name = quote::format_ident!("__merge_{}", fn_name);

    let pool_expr = &config.storage.pool;
    let sel_expr = &config.storage.selector;

    let is_state_fields = config.state_fields.is_some();

    // Rewrite the function body, replacing marker macros
    let body = rewrite_body(
        &func.block,
        &trace_fn_name,
        &merge_fn_name,
        pool_expr,
        sel_expr,
        &config.greens,
        is_state_fields,
    );

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
    merge_fn_name: &Ident,
    pool_expr: &Expr,
    sel_expr: &Expr,
    default_greens: &[Expr],
    is_state_fields: bool,
) -> TokenStream {
    use syn::visit_mut::VisitMut;

    #[derive(Default, Clone)]
    struct MergePointArgs {
        driver: Option<Expr>,
        env: Option<Expr>,
        pc: Option<Expr>,
        greens: Vec<Expr>,
    }

    impl Parse for MergePointArgs {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            if input.is_empty() {
                return Ok(Self::default());
            }
            let driver: Expr = input.parse()?;
            input.parse::<Token![,]>()?;
            let env: Expr = input.parse()?;
            input.parse::<Token![,]>()?;
            let pc: Expr = input.parse()?;
            let mut greens = Vec::new();
            if input.peek(Token![;]) {
                input.parse::<Token![;]>()?;
                let exprs: Punctuated<Expr, Token![,]> =
                    input.parse_terminated(Expr::parse, Token![,])?;
                greens = exprs.into_iter().collect();
            }
            Ok(Self {
                driver: Some(driver),
                env: Some(env),
                pc: Some(pc),
                greens,
            })
        }
    }

    struct CanEnterJitArgs {
        driver: Expr,
        target: Expr,
        state: Expr,
        env: Expr,
        pre_run: Expr,
        pc: Option<Expr>,
        stacksize: Option<Expr>,
        greens: Vec<Expr>,
    }

    impl Parse for CanEnterJitArgs {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            let driver: Expr = input.parse()?;
            input.parse::<Token![,]>()?;
            let target: Expr = input.parse()?;
            input.parse::<Token![,]>()?;
            let state: Expr = input.parse()?;
            input.parse::<Token![,]>()?;
            let env: Expr = input.parse()?;
            input.parse::<Token![,]>()?;
            let pre_run: Expr = input.parse()?;

            let mut pc = None;
            let mut stacksize = None;
            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
                pc = Some(input.parse::<Expr>()?);
                input.parse::<Token![,]>()?;
                stacksize = Some(input.parse::<Expr>()?);
            }

            let mut greens = Vec::new();
            if input.peek(Token![;]) {
                input.parse::<Token![;]>()?;
                let exprs: Punctuated<Expr, Token![,]> =
                    input.parse_terminated(Expr::parse, Token![,])?;
                greens = exprs.into_iter().collect();
            }

            Ok(Self {
                driver,
                target,
                state,
                env,
                pre_run,
                pc,
                stacksize,
                greens,
            })
        }
    }

    fn green_key_expr(target: &Expr, greens: &[Expr]) -> Option<TokenStream> {
        if greens.is_empty() {
            None
        } else {
            Some(quote! {
                majit_ir::GreenKey::new(vec![(#target) as i64, #((#greens) as i64),*])
            })
        }
    }

    struct MarkerRewriter {
        trace_fn_name: Ident,
        merge_fn_name: Ident,
        pool_expr: Expr,
        sel_expr: Expr,
        default_greens: Vec<Expr>,
        is_state_fields: bool,
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
                    let args =
                        syn::parse2::<MergePointArgs>(mac.tokens.clone()).unwrap_or_default();
                    let merge_fn = &self.merge_fn_name;
                    let driver = args.driver.unwrap_or_else(|| syn::parse_quote!(driver));
                    let env = args.env.unwrap_or_else(|| syn::parse_quote!(program));
                    let pc = args.pc.unwrap_or_else(|| syn::parse_quote!(pc));
                    let pool = &self.pool_expr;
                    let sel = &self.sel_expr;
                    let new_tokens: TokenStream = quote! {
                        if #driver.is_tracing() {
                            #merge_fn(&mut #driver, #env, #pc, &#pool, #sel);
                        }
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

                // Check if this is can_enter_jit!(driver, target, state, env, pre_run, ...)
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
                            let driver_expr = &args.driver;
                            let target_expr = &args.target;
                            let state_expr = &args.state;
                            let env_expr = &args.env;
                            let pre_run_expr = &args.pre_run;
                            let pc_expr = args
                                .pc
                                .as_ref()
                                .cloned()
                                .unwrap_or_else(|| syn::parse_quote!(pc));
                            let stacksize_expr = args
                                .stacksize
                                .as_ref()
                                .cloned()
                                .unwrap_or_else(|| syn::parse_quote!(stacksize));
                            let greens = if args.greens.is_empty() {
                                self.default_greens.clone()
                            } else {
                                args.greens.clone()
                            };
                            let pool = &pool_expr;
                            let sel = &sel_expr;
                            let is_sf = self.is_state_fields;
                            let stacksize_update: TokenStream = if is_sf {
                                // state_fields mode: no storage pool
                                quote! { #stacksize_expr = 0i32; }
                            } else {
                                quote! { #stacksize_expr = #pool.get(#sel).len() as i32; }
                            };
                            let back_edge: TokenStream = if let Some(green_key) =
                                green_key_expr(target_expr, &greens)
                            {
                                quote! {
                                    if #driver_expr.back_edge_structured(#green_key, #target_expr, #state_expr, #env_expr, #pre_run_expr) {
                                        #pc_expr = #target_expr;
                                        #stacksize_update
                                        continue;
                                    }
                                }
                            } else {
                                quote! {
                                    if #driver_expr.back_edge(#target_expr, #state_expr, #env_expr, #pre_run_expr) {
                                        #pc_expr = #target_expr;
                                        #stacksize_update
                                        continue;
                                    }
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

    let mut cloned_block = block.clone();
    let mut rewriter = MarkerRewriter {
        trace_fn_name: trace_fn_name.clone(),
        merge_fn_name: merge_fn_name.clone(),
        pool_expr: pool_expr.clone(),
        sel_expr: sel_expr.clone(),
        default_greens: default_greens.to_vec(),
        is_state_fields,
    };
    rewriter.visit_block_mut(&mut cloned_block);

    let stmts = &cloned_block.stmts;
    quote! { #(#stmts)* }
}

#[cfg(test)]
mod tests {
    use super::*;
    use syn::parse_quote;

    #[test]
    fn parse_helpers_list_basic() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            [helper_add, helper_sub, helper_mul]
        };
        let result: Vec<(Path, Option<Ident>)> =
            syn::parse2::<HelpersListWrapper>(tokens).unwrap().0;
        assert_eq!(result.len(), 3);
        assert_eq!(
            result[0].0.segments.last().unwrap().ident.to_string(),
            "helper_add"
        );
        assert!(result[0].1.is_none());
        assert_eq!(
            result[1].0.segments.last().unwrap().ident.to_string(),
            "helper_sub"
        );
        assert_eq!(
            result[2].0.segments.last().unwrap().ident.to_string(),
            "helper_mul"
        );
    }

    #[test]
    fn parse_helpers_list_empty() {
        let tokens: proc_macro2::TokenStream = parse_quote! { [] };
        let result: Vec<(Path, Option<Ident>)> =
            syn::parse2::<HelpersListWrapper>(tokens).unwrap().0;
        assert!(result.is_empty());
    }

    #[test]
    fn parse_helpers_list_with_path() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            [module::helper_a, helper_b]
        };
        let result: Vec<(Path, Option<Ident>)> =
            syn::parse2::<HelpersListWrapper>(tokens).unwrap().0;
        assert_eq!(result.len(), 2);
        // First has two path segments
        assert_eq!(result[0].0.segments.len(), 2);
        assert_eq!(result[0].0.segments[0].ident.to_string(), "module");
        assert_eq!(result[0].0.segments[1].ident.to_string(), "helper_a");
    }

    /// Wrapper to make `parse_helpers_list` testable via `syn::parse2`.
    struct HelpersListWrapper(Vec<(Path, Option<Ident>)>);
    impl Parse for HelpersListWrapper {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            Ok(Self(parse_helpers_list(input)?))
        }
    }
}
