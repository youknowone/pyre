//! JIT mainloop code generation from interpreter opcode match arms.
//!
//! This is majit's equivalent of RPython's `codewriter.py`: given an
//! interpreter's opcode dispatch match expression and a configuration
//! describing the interpreter's types/layout, it generates a complete
//! JIT-annotated mainloop module.
//!
//! The generator is **interpreter-agnostic** — all interpreter-specific
//! details (crate paths, state struct, storage pool, I/O shims) come
//! from [`JitDriverConfig`], analogous to RPython's `JitDriver` declaration.

use crate::interp_extract::BinopMapping;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

/// Interpreter-specific match arm transformer.
///
/// Each interpreter (aheui, pyre) implements this to rewrite opcode
/// match arms for JIT. The codewriter calls `transform_arm` for each
/// arm in the opcode dispatch match.
///
/// RPython equivalent: the graph transformations in codewriter.py
/// that rewrite the interpreter loop for JIT recording.
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
/// This is the Rust equivalent of RPython's `JitDriver(greens=..., reds=...)`.
/// The consumer (e.g., aheui-mjit or pyre-mjit) fills this in, and
/// [`generate_jitcode`] produces a complete JIT-enabled mainloop module.
pub struct JitDriverConfig {
    /// Use-path for the program/env type (e.g., `"aheui_interp::ahsembler::Program"`)
    pub program_path: String,
    /// Glob use-paths to import (e.g., `["aheui_interp::aheui::*", "aheui_interp::value::*"]`)
    pub use_globs: Vec<String>,
    /// Use-path for the I/O module (e.g., `"aheui_interp::io"`)
    pub io_module: Option<String>,

    /// State struct name (e.g., `"AheuiState"`)
    pub state_name: String,
    /// State struct fields as `(name, type_path)` (e.g., `[("storage", "StoragePool"), ("selected", "usize")]`)
    pub state_fields: Vec<(String, String)>,

    /// Storage pool configuration (None for non-storage interpreters like pyre)
    pub storage: Option<StorageConfig>,

    /// I/O shim pairs: `(original_fn_path, shim_fn_name)`
    pub io_shims: Vec<(String, String)>,

    /// Return type expression (e.g., `"Val"`, `"i64"`)
    pub return_type: String,
    /// Default return expression when stack is empty (e.g., `"val_from_i32(0)"`, `"0"`)
    pub default_return: String,

    /// Green key fields on state (e.g., `["state.selected"]`)
    pub greens: Vec<String>,

    /// Flush function path (e.g., `"aheui_io::output_flush"`)
    pub flush_fn: Option<String>,

    /// Stack-ok computation: `(req_size_expr, stackdel_table, stackadd_table)`
    pub stack_accounting: Option<StackConfig>,

    /// Program size field (e.g., `"program.size"`)
    pub program_size: String,
    /// Op fetch expression (e.g., `"program.get_op(pc)"`)
    pub op_fetch: String,

    /// Extra local variable declarations inside mainloop before the while loop
    /// (e.g., `"let mut input = aheui_io::InputBuffer::new();"`).
    pub extra_locals: Vec<String>,

    /// Additional code to append after the mainloop (helper functions like
    /// `find_used_storages`, `run_jit_back_edge`, `restore_jit_guard_state`).
    /// These are interpreter-specific and cannot be generalized by the framework.
    pub extra_code: Vec<String>,

    /// Loop structure: "pc_while" (aheui: `while pc < size { ... pc += 1 }`)
    /// or "step_result" (pyre: `loop { match step { Continue => {}, CloseLoop => ..., Return => ... } }`)
    /// Defaults to "pc_while" if None.
    pub loop_style: Option<String>,

    /// For step_result loop style: the expression that produces a StepResult
    /// from the opcode match. E.g., "execute_opcode_step(frame, code, instruction, op_arg, next_instr)"
    pub step_expr: Option<String>,
}

/// Storage pool configuration for `#[jit_interp(storage = { ... })]`.
pub struct StorageConfig {
    /// Pool field path (e.g., `"state.storage"`)
    pub pool: String,
    /// Pool type name (e.g., `"StoragePool"`)
    pub pool_type: String,
    /// Selector field path (e.g., `"state.selected"`)
    pub selector: String,
    /// Untraceable storage indices (e.g., `["VAL_QUEUE", "VAL_PORT"]`)
    pub untraceable: Vec<String>,
    /// Scan function name (e.g., `"find_used_storages"`)
    pub scan_fn: String,
    /// Optional can-trace guard (e.g., `"all_jit_compatible"`)
    pub can_trace_guard: Option<String>,
}

/// Stack accounting configuration for interpreters with explicit stack tracking.
pub struct StackConfig {
    pub req_size_expr: String,
    pub stackdel_table: String,
    pub stackadd_table: String,
}

/// Generate a complete JIT mainloop module.
///
/// This is majit's equivalent of RPython's `CodeWriter.transform_graph_to_jitcode()`.
///
/// # Arguments
/// - `opcode_match`: the parsed opcode dispatch match expression from the interpreter
/// - `binops`: binary operation mappings (method name → IR opcode)
/// - `config`: interpreter-specific configuration
///
/// # Returns
/// A `TokenStream` containing a complete Rust module with:
/// - Use declarations
/// - I/O shim wrappers
/// - State struct
/// - `#[jit_interp(...)]` annotated mainloop
/// - Framework helper functions
pub fn generate_jitcode(
    opcode_match: &syn::ExprMatch,
    binops: &[BinopMapping],
    config: &JitDriverConfig,
    arm_transformer: &dyn ArmTransformer,
) -> TokenStream {
    let binop_entries: Vec<TokenStream> = binops
        .iter()
        .map(|b| {
            let method = format_ident!("{}", b.method);
            let opcode = format_ident!("{}", b.opcode);
            quote! { #method => #opcode }
        })
        .collect();

    // Transform match arms via interpreter-specific transformer
    let match_scrutinee = &opcode_match.expr;
    let transformed_arms: Vec<TokenStream> = opcode_match
        .arms
        .iter()
        .map(|arm| arm_transformer.transform_arm(arm))
        .collect();

    // Build use declarations from config
    let use_decls: Vec<TokenStream> = config
        .use_globs
        .iter()
        .map(|p| {
            // Parse the full "use ...;" statement to handle aliases like "crate::io as alias"
            let stmt = format!("use {p};");
            stmt.parse().unwrap_or_else(|_| quote! {})
        })
        .collect();

    // Extract the simple type name from the full path for use in attribute
    // (e.g., "aheui_interp::ahsembler::Program" → "Program")
    let program_simple_name = config
        .program_path
        .rsplit("::")
        .next()
        .unwrap_or(&config.program_path);
    let program_type: TokenStream = program_simple_name.parse().unwrap();

    // Build state struct
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

    // Build I/O shim wrappers
    let io_shim_wrappers: Vec<TokenStream> = config
        .io_shims
        .iter()
        .enumerate()
        .map(|(i, (_, shim_name))| {
            let shim = format_ident!("{}", shim_name);
            // Map standard shim names to majit_meta functions
            match shim_name.as_str() {
                "jit_write_number" => quote! {
                    extern "C" fn #shim(value: i64) {
                        majit_meta::jit_write_number_i64(value);
                    }
                },
                "jit_write_utf8" => quote! {
                    extern "C" fn #shim(value: i64) {
                        majit_meta::jit_write_utf8_codepoint(value);
                    }
                },
                _ => {
                    let fn_name = format_ident!("jit_io_shim_{}", i);
                    quote! { extern "C" fn #fn_name(value: i64) { let _ = value; } }
                }
            }
        })
        .collect();

    // Build io_shims attribute entries
    let io_shim_entries: Vec<TokenStream> = config
        .io_shims
        .iter()
        .map(|(original, shim)| {
            let orig: TokenStream = original.parse().unwrap();
            let shim = format_ident!("{}", shim);
            quote! { #orig => #shim }
        })
        .collect();

    // Build #[jit_interp(...)] attribute
    let greens: Vec<TokenStream> = config
        .greens
        .iter()
        .map(|g| g.parse::<TokenStream>().unwrap())
        .collect();

    let storage_attr = config.storage.as_ref().map(|s| {
        let pool: TokenStream = s.pool.parse().unwrap();
        let pool_type = format_ident!("{}", s.pool_type);
        let selector: TokenStream = s.selector.parse().unwrap();
        let untraceable: Vec<TokenStream> = s
            .untraceable
            .iter()
            .map(|u| u.parse::<TokenStream>().unwrap())
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
        quote! {
            storage = {
                pool: #pool, pool_type: #pool_type,
                selector: #selector,
                untraceable: [#(#untraceable),*],
                scan: #scan,
                #guard
            },
        }
    });

    let return_type: TokenStream = config.return_type.parse().unwrap();
    let default_return: TokenStream = config.default_return.parse().unwrap();
    let program_size: TokenStream = config.program_size.parse().unwrap();
    let op_fetch: TokenStream = config.op_fetch.parse().unwrap();

    let extra_locals: Vec<TokenStream> = config
        .extra_locals
        .iter()
        .filter_map(|s| s.parse().ok())
        .collect();

    let extra_code: Vec<TokenStream> = config
        .extra_code
        .iter()
        .filter_map(|s| s.parse().ok())
        .collect();

    let flush_call = config.flush_fn.as_ref().map(|f| {
        let f: TokenStream = f.parse().unwrap();
        quote! { #f(); }
    });

    let stack_accounting = config
        .stack_accounting
        .as_ref()
        .map(|sa| {
            let req: TokenStream = sa.req_size_expr.parse().unwrap();
            let del: TokenStream = sa.stackdel_table.parse().unwrap();
            let add: TokenStream = sa.stackadd_table.parse().unwrap();
            quote! {
                let stackok = #req as i32 <= stacksize;
                let op = #op_fetch;
                stacksize += -#del[op as usize] + #add[op as usize];
            }
        })
        .unwrap_or_else(|| {
            let fetch: TokenStream = config.op_fetch.parse().unwrap();
            quote! { let op = #fetch; }
        });

    // Build the complete pool init and return logic based on state fields
    let state_init: Vec<TokenStream> = config
        .state_fields
        .iter()
        .map(|(name, ty)| {
            let name = format_ident!("{}", name);
            let default_val: TokenStream = match ty.as_str() {
                "usize" => quote! { 0 },
                _ => {
                    let ty_path: TokenStream = ty.parse().unwrap();
                    quote! { #ty_path::new() }
                }
            };
            quote! { #name: #default_val }
        })
        .collect();

    // Storage-aware return logic
    let return_logic = if config.storage.is_some() {
        let selector: TokenStream = config
            .storage
            .as_ref()
            .unwrap()
            .selector
            .replace("state.", "")
            .parse()
            .unwrap();
        let pool: TokenStream = config
            .storage
            .as_ref()
            .unwrap()
            .pool
            .replace("state.", "")
            .parse()
            .unwrap();
        quote! {
            if !state.#pool.get(state.#selector).is_empty() {
                state.#pool.get_mut(state.#selector).pop()
            } else {
                #default_return
            }
        }
    } else {
        quote! { #default_return }
    };

    let is_step_result = config
        .loop_style
        .as_deref()
        .is_some_and(|s| s == "step_result");

    // Build the mainloop body depending on loop style
    let mainloop_body = if is_step_result {
        // pyre-style: loop { decode(pc); match step { Continue, CloseLoop, Return } }
        let step: TokenStream = config
            .step_expr
            .as_deref()
            .unwrap_or("execute_step()")
            .parse()
            .unwrap();
        quote! {
            loop {
                jit_merge_point!();
                match #step {
                    StepResult::Continue => {}
                    StepResult::CloseLoop(_target) => {
                        can_enter_jit!();
                    }
                    StepResult::Return(result) => return result,
                }
            }
        }
    } else {
        // aheui-style: while pc < size { match op { ... } pc += 1 }
        quote! {
            while pc < #program_size {
                jit_merge_point!();
                #stack_accounting

                match #match_scrutinee {
                    #(#transformed_arms)*
                }
                pc += 1;
            }
        }
    };

    quote! {
        // AUTO-GENERATED by majit-analyze. Do not edit.

        #(#use_decls)*

        use majit_meta::JitDriver;
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

        const DEFAULT_THRESHOLD: u32 = 0;
        static JUST_COMPILED_KEY: AtomicU64 = AtomicU64::new(0);
        static JUST_COMPILED_PENDING: AtomicBool = AtomicBool::new(false);

        #(#io_shim_wrappers)*

        struct #state_name { #(#state_fields),* }

        #[majit_macros::jit_interp(
            state = #state_name, env = #program_type,
            greens = [#(#greens),*],
            #storage_attr
            binops = { #(#binop_entries),* },
            io_shims = { #(#io_shim_entries),* },
        )]
        pub fn mainloop(program: &#program_type, threshold: u32) -> #return_type {
            let mut driver: JitDriver<#state_name> = JitDriver::new(threshold);
            driver.meta_interp_mut().set_bridge_threshold(0);
            driver.meta_interp_mut().set_on_compile_loop(|green_key, _, _| {
                JUST_COMPILED_KEY.store(green_key, Ordering::Relaxed);
                JUST_COMPILED_PENDING.store(true, Ordering::Relaxed);
            });

            let mut pc: usize = 0;
            let mut stacksize: i32 = 0;
            let mut state = #state_name { #(#state_init),* };
            #(#extra_locals)*

            #mainloop_body

            #flush_call

            #return_logic
        }

        pub const NO_JIT: u32 = u32::MAX;
        pub const JIT_THRESHOLD: u32 = DEFAULT_THRESHOLD;

        #(#extra_code)*
    }
}
