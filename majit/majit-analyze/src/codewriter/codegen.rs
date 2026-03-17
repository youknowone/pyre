//! JIT mainloop scaffolding generator.
//!
//! Combines the roles of RPython's `flatten.py` and `assembler.py` into a
//! single codegen step that emits a Rust `TokenStream`.
//!
//! The generator is **interpreter-agnostic**: all interpreter-specific
//! details (loop structure, state initialisation, return logic) come from
//! [`JitDriverConfig`]. The codewriter only produces:
//!
//! 1. `use` declarations
//! 2. State struct definition
//! 3. `#[jit_interp(...)]`-annotated mainloop function
//! 4. Extra helper code
//!
//! The `#[jit_interp]` proc macro then acts as RPython's
//! translator+codewriter, lowering the match arms to JitCode bytecode.

use crate::interp_extract::BinopMapping;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

/// Interpreter-specific match arm transformer.
///
/// Each interpreter (aheui, pyre) implements this to rewrite opcode
/// match arms for JIT.
pub trait ArmTransformer {
    fn transform_arm(&self, arm: &syn::Arm) -> TokenStream;
}

/// Default transformer: passes arms through unchanged.
pub struct IdentityTransformer;

impl ArmTransformer for IdentityTransformer {
    fn transform_arm(&self, arm: &syn::Arm) -> TokenStream {
        let pat = &arm.pat;
        let body = &arm.body;
        quote! { #pat => #body, }
    }
}

/// Configuration for JIT mainloop generation.
///
/// Rust equivalent of RPython's `JitDriver(greens=..., reds=...)`.
/// The consumer (aheui-jit, pyre-mjit) fills this in.
pub struct JitDriverConfig {
    /// Glob use-paths to import.
    pub use_globs: Vec<String>,

    /// Program/env type for `#[jit_interp(env = ...)]`.
    pub env_type: String,

    /// State struct name for `#[jit_interp(state = ...)]`.
    pub state_name: String,
    /// State struct fields as `(name, type_path)`.
    pub state_fields: Vec<(String, String)>,

    /// Green key fields for `#[jit_interp(greens = [...])]`.
    pub greens: Vec<String>,

    /// Binary operation mappings for `#[jit_interp(binops = { ... })]`.
    pub binops: Vec<BinopMapping>,

    /// Storage pool configuration (None for non-storage interpreters).
    pub storage: Option<StorageConfig>,

    /// I/O shim pairs: `(original_fn_path, shim_fn_name)`.
    pub io_shims: Vec<IoShim>,

    /// Virtualizable frame field declaration.
    ///
    /// RPython equivalent: VirtualizableInfo from virtualizable.py.
    /// When set, the proc macro rewrites field accesses on the virtualizable
    /// variable to use TraceCtx vable_* methods (jtransform.py:832 parity).
    pub virtualizable: Option<VirtualizableCodegenConfig>,

    /// The complete mainloop function body — loop structure, state init,
    /// opcode dispatch, and return logic. Provided verbatim by the
    /// interpreter; the codewriter does NOT generate or modify it.
    pub mainloop_body: String,

    /// Function signature (excluding `fn mainloop`):
    /// e.g. `"(program: &Program, threshold: u32) -> Val"`.
    pub fn_signature: String,

    /// Additional code after the mainloop (helper functions).
    pub extra_code: Vec<String>,
}

/// Virtualizable frame configuration for build-time code generation.
///
/// Passed through JitDriverConfig to generate the `virtualizable_fields = { ... }`
/// block in the `#[jit_interp]` attribute.
pub struct VirtualizableCodegenConfig {
    /// Variable name in the mainloop body (e.g., "frame").
    pub var: String,
    /// Constant path for vable_token offset (e.g., "PYFRAME_VABLE_TOKEN_OFFSET").
    pub token_offset: String,
    /// Static fields: (name, type "int"/"ref"/"float", offset constant path).
    pub fields: Vec<(String, String, String)>,
    /// Array fields: (name, item type, offset constant path).
    pub arrays: Vec<(String, String, String)>,
}

/// A single I/O shim declaration.
pub struct IoShim {
    /// Path of the original I/O function (e.g. `"aheui_io::output_write_number"`).
    pub original: String,
    /// Name of the generated shim (e.g. `"jit_write_number"`).
    pub shim_name: String,
    /// The extern "C" wrapper body as a token stream.
    pub wrapper: String,
}

/// Storage pool configuration for `#[jit_interp(storage = { ... })]`.
pub struct StorageConfig {
    pub pool: String,
    pub pool_type: String,
    pub selector: String,
    pub untraceable: Vec<String>,
    pub scan_fn: String,
    pub can_trace_guard: Option<String>,
    /// When true, storage stacks are virtualizable (variable depth allowed).
    pub virtualizable: bool,
}

/// Generate a complete JIT mainloop module.
///
/// Produces:
/// 1. `use` declarations
/// 2. I/O shim wrappers
/// 3. State struct
/// 4. `#[jit_interp(...)]`-annotated mainloop function
/// 5. Extra helper functions
pub fn generate_jitcode(config: &JitDriverConfig) -> TokenStream {
    // ── use declarations ──
    let use_decls: Vec<TokenStream> = config
        .use_globs
        .iter()
        .filter_map(|p| format!("use {p};").parse().ok())
        .collect();

    // ── state struct ──
    let state_name = format_ident!("{}", config.state_name);
    let state_fields: Vec<TokenStream> = config
        .state_fields
        .iter()
        .map(|(name, ty)| {
            let name = format_ident!("{}", name);
            let ty: TokenStream = ty.parse().unwrap();
            quote! { #name: #ty }
        })
        .collect();

    // ── #[jit_interp(...)] attribute pieces ──
    let env_type: TokenStream = config.env_type.parse().unwrap();

    let greens: Vec<TokenStream> = config
        .greens
        .iter()
        .filter_map(|g| g.parse::<TokenStream>().ok())
        .collect();

    let binop_entries: Vec<TokenStream> = config
        .binops
        .iter()
        .map(|b| {
            let method = format_ident!("{}", b.method);
            let opcode = format_ident!("{}", b.opcode);
            quote! { #method => #opcode }
        })
        .collect();

    let storage_attr = config.storage.as_ref().map(|s| {
        let pool: TokenStream = s.pool.parse().unwrap();
        let pool_type = format_ident!("{}", s.pool_type);
        let selector: TokenStream = s.selector.parse().unwrap();
        let untraceable: Vec<TokenStream> = s
            .untraceable
            .iter()
            .filter_map(|u| u.parse::<TokenStream>().ok())
            .collect();
        let scan = format_ident!("{}", s.scan_fn);
        let guard = s
            .can_trace_guard
            .as_ref()
            .map(|g| {
                let g = format_ident!("{}", g);
                quote! { can_trace_guard: #g, }
            })
            .unwrap_or_default();
        let vable = if s.virtualizable {
            quote! { virtualizable: true, }
        } else {
            quote! {}
        };
        quote! {
            storage = {
                pool: #pool, pool_type: #pool_type,
                selector: #selector,
                untraceable: [#(#untraceable),*],
                scan: #scan,
                #guard
                #vable
            },
        }
    });

    let virtualizable_attr = config.virtualizable.as_ref().map(|v| {
        let var = format_ident!("{}", v.var);
        let token_offset: TokenStream = v.token_offset.parse().unwrap();
        let field_entries: Vec<TokenStream> = v
            .fields
            .iter()
            .map(|(name, tp, offset)| {
                let name = format_ident!("{}", name);
                let tp = format_ident!("{}", tp);
                let offset: TokenStream = offset.parse().unwrap();
                quote! { #name: #tp @ #offset }
            })
            .collect();
        let array_entries: Vec<TokenStream> = v
            .arrays
            .iter()
            .map(|(name, tp, offset)| {
                let name = format_ident!("{}", name);
                let tp = format_ident!("{}", tp);
                let offset: TokenStream = offset.parse().unwrap();
                quote! { #name: #tp @ #offset }
            })
            .collect();
        quote! {
            virtualizable_fields = {
                var: #var,
                token_offset: #token_offset,
                fields: { #(#field_entries,)* },
                arrays: { #(#array_entries,)* },
            },
        }
    });

    let io_shim_attr_entries: Vec<TokenStream> = config
        .io_shims
        .iter()
        .map(|s| {
            let orig: TokenStream = s.original.parse().unwrap();
            let shim = format_ident!("{}", s.shim_name);
            quote! { #orig => #shim }
        })
        .collect();

    // ── I/O shim wrappers ──
    let io_shim_wrappers: Vec<TokenStream> = config
        .io_shims
        .iter()
        .filter_map(|s| s.wrapper.parse().ok())
        .collect();

    // ── mainloop body (verbatim from interpreter) ──
    let mainloop_body: TokenStream = config
        .mainloop_body
        .parse()
        .expect("JitDriverConfig.mainloop_body must be valid Rust");

    let fn_sig: TokenStream = config
        .fn_signature
        .parse()
        .expect("JitDriverConfig.fn_signature must be valid Rust");

    // ── extra code ──
    let extra_code: Vec<TokenStream> = config
        .extra_code
        .iter()
        .filter_map(|s| s.parse().ok())
        .collect();

    // ── assemble ──
    quote! {
        // AUTO-GENERATED by majit-analyze codewriter. Do not edit.

        #(#use_decls)*

        use majit_meta::JitDriver;

        const DEFAULT_THRESHOLD: u32 = 0;

        #(#io_shim_wrappers)*

        struct #state_name { #(#state_fields),* }

        #[majit_macros::jit_interp(
            state = #state_name, env = #env_type,
            greens = [#(#greens),*],
            #storage_attr
            #virtualizable_attr
            binops = { #(#binop_entries),* },
            io_shims = { #(#io_shim_attr_entries),* },
        )]
        pub fn mainloop #fn_sig {
            #mainloop_body
        }

        pub const NO_JIT: u32 = u32::MAX;
        pub const JIT_THRESHOLD: u32 = DEFAULT_THRESHOLD;

        #(#extra_code)*
    }
}
