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

use proc_macro2::{Ident, TokenStream};
use quote::{format_ident, quote};
use syn::{GenericArgument, PathArguments, Type};

/// A mapping from an interpreter helper/binary-op method name to its JIT opcode.
///
/// This is codewriter-owned build-time metadata, analogous to the helper/opcode
/// tables consumed by RPython's translation/codewriter layer.
#[derive(Debug, Clone)]
pub struct BinopMapping {
    pub method: Ident,
    pub opcode: Ident,
}

#[derive(Debug, Clone, Copy)]
pub enum CodegenValueKind {
    Int,
    Ref,
    Float,
}

impl CodegenValueKind {
    fn attr_ident(self) -> Ident {
        match self {
            CodegenValueKind::Int => format_ident!("int"),
            CodegenValueKind::Ref => format_ident!("ref"),
            CodegenValueKind::Float => format_ident!("float"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StateFieldKind {
    Scalar(&'static str),
    Array(&'static str),
}

/// Classify a Rust type into a JIT state field kind.
///
/// `state_fields` lowers to the compact state-field JitCode ops, which are
/// currently integer-only in `majit-meta::jitcode`.
///
/// Keep this intentionally narrow until the canonical virtualizable/state-box
/// path owns typed ref/float state the way RPython does.
fn classify_state_field_type(ty: &Type) -> Option<StateFieldKind> {
    match ty {
        Type::Path(path) => {
            let segment = path.path.segments.last()?;
            let ident = segment.ident.to_string();
            match ident.as_str() {
                "i8" | "i16" | "i32" | "i64" | "isize" | "u8" | "u16" | "u32" | "u64" | "usize"
                | "bool" => Some(StateFieldKind::Scalar("int")),
                "Vec" => {
                    let PathArguments::AngleBracketed(args) = &segment.arguments else {
                        return None;
                    };
                    let inner = args.args.iter().find_map(|arg| {
                        if let GenericArgument::Type(ty) = arg {
                            Some(ty)
                        } else {
                            None
                        }
                    })?;
                    match classify_state_field_type(inner)? {
                        StateFieldKind::Scalar(kind) => Some(StateFieldKind::Array(kind)),
                        StateFieldKind::Array(_) => None,
                    }
                }
                _ => None,
            }
        }
        _ => None,
    }
}

fn infer_state_fields_attr(config: &JitDriverConfig) -> Option<TokenStream> {
    if config.storage.is_some() {
        return None;
    }
    if config.state_fields.is_empty() {
        return None;
    }

    let field_entries: Vec<TokenStream> = config
        .state_fields
        .iter()
        .map(|(name, ty)| {
            let ty_string = ty.to_string();
            let parsed_ty: Type = syn::parse2(ty.clone()).unwrap_or_else(|err| {
                panic!(
                    "state_fields codegen: failed to parse type `{}` for `{name}`: {err}",
                    ty_string
                )
            });
            match classify_state_field_type(&parsed_ty) {
                Some(StateFieldKind::Array(kind)) => {
                    let kind_ident = format_ident!("{}", kind);
                    quote! { #name: [#kind_ident] }
                }
                Some(StateFieldKind::Scalar(kind)) => {
                    let kind_ident = format_ident!("{}", kind);
                    quote! { #name: #kind_ident }
                }
                None => panic!(
                    "state_fields codegen: unsupported type `{}` for `{name}`. \
                     Supported: integer scalars and Vec<integer> arrays.",
                    ty_string
                ),
            }
        })
        .collect();

    Some(quote! {
        state_fields = { #(#field_entries,)* },
    })
}

/// Configuration for JIT mainloop generation.
///
/// Rust equivalent of RPython's `JitDriver(greens=..., reds=...)`.
///
/// This compatibility config is currently used only by the remaining
/// `aheui-jit` proc-macro/codewriter path.
pub struct JitDriverConfig {
    /// Glob use-paths to import.
    pub use_globs: Vec<TokenStream>,

    /// Program/env type for `#[jit_interp(env = ...)]`.
    pub env_type: TokenStream,

    /// State struct name for `#[jit_interp(state = ...)]`.
    pub state_name: Ident,
    /// State struct fields as `(name, type_tokens)`.
    pub state_fields: Vec<(Ident, TokenStream)>,

    /// Green key fields for `#[jit_interp(greens = [...])]`.
    pub greens: Vec<TokenStream>,

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
    pub mainloop_body: TokenStream,

    /// Function signature (excluding `fn mainloop`):
    /// e.g. `"(program: &Program, threshold: u32) -> Val"`.
    pub fn_signature: TokenStream,

    /// Additional code after the mainloop (helper functions).
    pub extra_code: Vec<TokenStream>,
}

/// Virtualizable frame configuration for build-time code generation.
///
/// Passed through JitDriverConfig to generate the `virtualizable_fields = { ... }`
/// block in the `#[jit_interp]` attribute.
pub struct VirtualizableCodegenConfig {
    /// Variable name in the mainloop body (e.g., "frame").
    pub var: Ident,
    /// Constant path for vable_token offset (e.g., "PYFRAME_VABLE_TOKEN_OFFSET").
    pub token_offset: TokenStream,
    /// Static fields: (name, type "int"/"ref"/"float", offset constant path).
    pub fields: Vec<(Ident, CodegenValueKind, TokenStream)>,
    /// Array fields: (name, item type, offset constant path).
    pub arrays: Vec<(Ident, CodegenValueKind, TokenStream)>,
}

/// A single I/O shim declaration.
pub struct IoShim {
    /// Path of the original I/O function (e.g. `"aheui_io::output_write_number"`).
    pub original: TokenStream,
    /// Name of the generated shim (e.g. `"jit_write_number"`).
    pub shim_name: Ident,
    /// The extern "C" wrapper body as a token stream.
    pub wrapper: TokenStream,
}

/// Storage pool configuration for `#[jit_interp(storage = { ... })]`.
pub struct StorageConfig {
    pub pool: TokenStream,
    pub pool_type: TokenStream,
    pub selector: TokenStream,
    pub untraceable: Vec<TokenStream>,
    pub scan_fn: TokenStream,
    pub can_trace_guard: Option<TokenStream>,
    pub compact_live: bool,
    pub compact_encode: Option<TokenStream>,
    pub compact_decode: Option<TokenStream>,
    pub compact_min: Option<TokenStream>,
    pub compact_max: Option<TokenStream>,
    pub compact_ptrs_offset: Option<TokenStream>,
    pub compact_lengths_offset: Option<TokenStream>,
    pub compact_caps_offset: Option<TokenStream>,
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
        .map(|p| quote! { use #p; })
        .collect();

    // ── state struct ──
    let state_name = config.state_name.clone();
    let state_fields: Vec<TokenStream> = config
        .state_fields
        .iter()
        .map(|(name, ty)| quote! { #name: #ty })
        .collect();

    // ── #[jit_interp(...)] attribute pieces ──
    let env_type = config.env_type.clone();

    let greens = config.greens.clone();

    let binop_entries: Vec<TokenStream> = config
        .binops
        .iter()
        .map(|b| {
            let method = b.method.clone();
            let opcode = b.opcode.clone();
            quote! { #method => #opcode }
        })
        .collect();

    let storage_attr = config.storage.as_ref().map(|s| {
        let pool = s.pool.clone();
        let pool_type = s.pool_type.clone();
        let selector = s.selector.clone();
        let untraceable = s.untraceable.clone();
        let scan = s.scan_fn.clone();
        let guard = s
            .can_trace_guard
            .as_ref()
            .map(|g| {
                let g = g.clone();
                quote! { can_trace_guard: #g, }
            })
            .unwrap_or_default();
        let compact_live = s.compact_live;
        let compact_encode = s.compact_encode.as_ref().map(|path| {
            let path = path.clone();
            quote! { compact_encode: #path, }
        });
        let compact_decode = s.compact_decode.as_ref().map(|path| {
            let path = path.clone();
            quote! { compact_decode: #path, }
        });
        let compact_min = s.compact_min.as_ref().map(|expr| {
            let expr = expr.clone();
            quote! { compact_min: #expr, }
        });
        let compact_max = s.compact_max.as_ref().map(|expr| {
            let expr = expr.clone();
            quote! { compact_max: #expr, }
        });
        let compact_ptrs_offset = s.compact_ptrs_offset.as_ref().map(|expr| {
            let expr = expr.clone();
            quote! { compact_ptrs_offset: #expr, }
        });
        let compact_lengths_offset = s.compact_lengths_offset.as_ref().map(|expr| {
            let expr = expr.clone();
            quote! { compact_lengths_offset: #expr, }
        });
        let compact_caps_offset = s.compact_caps_offset.as_ref().map(|expr| {
            let expr = expr.clone();
            quote! { compact_caps_offset: #expr, }
        });
        quote! {
            storage = {
                pool: #pool, pool_type: #pool_type,
                selector: #selector,
                untraceable: [#(#untraceable),*],
                scan: #scan,
                #guard
                compact_live: #compact_live,
                #compact_encode
                #compact_decode
                #compact_min
                #compact_max
                #compact_ptrs_offset
                #compact_lengths_offset
                #compact_caps_offset
            },
        }
    });

    let virtualizable_attr = config.virtualizable.as_ref().map(|v| {
        let var = v.var.clone();
        let token_offset = v.token_offset.clone();
        let field_entries: Vec<TokenStream> = v
            .fields
            .iter()
            .map(|(name, tp, offset)| {
                let name = name.clone();
                let tp = tp.attr_ident();
                let offset = offset.clone();
                quote! { #name: #tp @ #offset }
            })
            .collect();
        let array_entries: Vec<TokenStream> = v
            .arrays
            .iter()
            .map(|(name, tp, offset)| {
                let name = name.clone();
                let tp = tp.attr_ident();
                let offset = offset.clone();
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

    let state_fields_attr = infer_state_fields_attr(config);

    let io_shim_attr_entries: Vec<TokenStream> = config
        .io_shims
        .iter()
        .map(|s| {
            let orig = s.original.clone();
            let shim = s.shim_name.clone();
            quote! { #orig => #shim }
        })
        .collect();

    // ── I/O shim wrappers ──
    let io_shim_wrappers: Vec<TokenStream> = config
        .io_shims
        .iter()
        .map(|shim| shim.wrapper.clone())
        .collect();

    // ── mainloop body (verbatim from interpreter) ──
    let mainloop_body = config.mainloop_body.clone();

    let fn_sig = config.fn_signature.clone();

    // ── extra code ──
    let extra_code = config.extra_code.clone();

    // ── assemble ──
    quote! {
        // AUTO-GENERATED by majit-codewriter codewriter. Do not edit.

        #(#use_decls)*

        use majit_metainterp::JitDriver;

        const DEFAULT_THRESHOLD: u32 = 1039;

        #(#io_shim_wrappers)*

        struct #state_name { #(#state_fields),* }

        #[majit_macros::jit_interp(
            state = #state_name, env = #env_type,
            greens = [#(#greens),*],
            #storage_attr
            #state_fields_attr
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

#[cfg(test)]
mod tests {
    use super::*;

    fn base_config() -> JitDriverConfig {
        JitDriverConfig {
            use_globs: vec![],
            env_type: quote!(Program),
            state_name: format_ident!("State"),
            state_fields: vec![],
            greens: vec![],
            binops: vec![],
            storage: None,
            io_shims: vec![],
            virtualizable: None,
            mainloop_body: quote!({}),
            fn_signature: quote!(()),
            extra_code: vec![],
        }
    }

    #[test]
    fn generate_jitcode_emits_state_fields_attr_for_state_only_interpreters() {
        let mut config = base_config();
        config.state_fields = vec![
            (format_ident!("a"), quote!(i64)),
            (format_ident!("regs"), quote!(Vec<i64>)),
        ];

        let generated = generate_jitcode(&config).to_string();
        assert!(generated.contains("state_fields = { a : int , regs : [int] , }"));
    }

    #[test]
    #[should_panic(expected = "state_fields codegen: unsupported type")]
    fn generate_jitcode_rejects_unsupported_state_field_types() {
        let mut config = base_config();
        config.state_fields = vec![(format_ident!("x"), quote!(HashMap<String, i64>))];
        let _ = generate_jitcode(&config);
    }

    #[test]
    fn generate_jitcode_rejects_float_state_fields() {
        let mut config = base_config();
        config.state_fields = vec![(format_ident!("f"), quote!(f64))];
        let panic = std::panic::catch_unwind(|| generate_jitcode(&config)).unwrap_err();
        let message = panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
            .expect("panic payload should be string");
        assert!(message.contains("unsupported type `f64`"));
    }

    #[test]
    fn generate_jitcode_rejects_ref_state_fields_structurally() {
        let mut config = base_config();
        config.state_fields = vec![
            (format_ident!("obj"), quote!(PyObjectRef)),
            (format_ident!("objs"), quote!(Vec<PyObjectRef>)),
        ];
        let panic = std::panic::catch_unwind(|| generate_jitcode(&config)).unwrap_err();
        let message = panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
            .expect("panic payload should be string");
        assert!(message.contains("unsupported type `PyObjectRef`"));
    }
}
