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
    ext::IdentExt,
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
    pub storage: Option<StorageConfig>,
    /// Method name → IR opcode mapping for binary operations.
    pub binops: Vec<(Ident, Ident)>,
    /// Interpreter I/O function → JIT shim function mapping.
    pub io_shims: Vec<(Path, Ident)>,
    /// Interpreter function call policies for helper calls.
    /// Populated from both `calls = { ... }` and `helpers = [...]`.
    pub calls: Vec<(Path, Option<CallPolicyKind>)>,
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
///     arrays: {
///         locals_w: ref @ LOCALS_OFFSET,
///         stack: int @ (DATA_PTR_OFFSET + SLOT_OFFSET) {
///             ptr_offset: 0,
///             length_offset: LENGTH_OFFSET_MINUS_DATA_PTR,
///             items_offset: 0,
///         },
///     },
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
    /// Byte offset expression (e.g., `PYFRAME_NEXT_INSTR_OFFSET`,
    /// `STORAGEPOOL_LENGTHS_OFFSET + 8`).
    pub offset: Expr,
}

/// A single virtualizable array field declaration.
pub struct VableArrayDecl {
    /// Array field name as it appears in the struct.
    pub name: Ident,
    /// Item type: `int`, `ref`, or `float`.
    pub item_type: Ident,
    /// Physical layout of the array field inside the virtualizable object.
    pub layout: VableArrayLayoutDecl,
}

/// Layout description for a virtualizable array field.
pub enum VableArrayLayoutDecl {
    /// Direct pointer field: the virtualizable stores a pointer to the array.
    Direct { field_offset: Expr },
    /// Embedded container layout: pointer/length live in sibling fields or
    /// an inline container relative to the declared field offset.
    Embedded {
        field_offset: Expr,
        ptr_offset: Expr,
        length_offset: Expr,
        items_offset: Expr,
    },
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

/// Whether a state field is a scalar, array, or virtualizable array.
pub enum StateFieldKind {
    /// Scalar value (e.g., `a: int`).
    Scalar(Ident),
    /// Array value (e.g., `regs: [int]`) — flattened into inputargs.
    Array(Ident),
    /// Virtualizable array (e.g., `tape: [int; virt]`) — NOT flattened.
    /// Only the data pointer and length are tracked as inputargs.
    /// Element access emits GETARRAYITEM_RAW_I / SETARRAYITEM_RAW IR ops.
    VirtArray(Ident),
}

/// Multi-storage configuration parsed from `storage = { ... }`.
pub struct StorageConfig {
    /// Expression to access the storage pool (e.g., `state.storage`).
    pub pool: Expr,
    /// Optional expression yielding the JIT-visible GC storage object reference.
    ///
    /// Linked-list interpreters can use this to pass a real GC ref as a red
    /// variable instead of leaking a raw host pointer into GETFIELD_GC.
    pub pool_ref: Option<Expr>,
    /// Optional expression yielding the JIT-visible GC selected-storage ref.
    pub selected_ref: Option<Expr>,
    /// Type of the storage pool (e.g., `StoragePool`).
    pub pool_type: Path,
    /// Expression to access the selected index (e.g., `state.selected`).
    pub selector: Expr,
    /// Optional expression to access the selected stack size red variable.
    pub stacksize: Option<Expr>,
    /// Storage indices that cannot be traced (e.g., `[VAL_QUEUE, VAL_PORT]`).
    pub untraceable: Vec<Path>,
    /// Function to scan for used storages (e.g., `find_used_storages`).
    pub scan_fn: Ident,
    /// Optional method on StoragePool to check JIT compatibility of all values.
    /// When set, `can_trace` additionally calls `pool.method()`.
    pub can_trace_guard: Option<Ident>,
    /// Track storage state as compact ptr/len/cap triples instead of flattening contents.
    pub compact_live: bool,
    /// Optional helper to encode a semantic i64 value for raw storage writes.
    pub compact_encode: Option<Path>,
    /// Optional helper to decode a raw storage word to semantic i64.
    pub compact_decode: Option<Path>,
    /// Optional lower bound for values representable in compact storage.
    pub compact_min: Option<Expr>,
    /// Optional upper bound for values representable in compact storage.
    pub compact_max: Option<Expr>,
    /// Byte offset of the compact storage pointer cache array on the pool object.
    pub compact_ptrs_offset: Option<Expr>,
    /// Byte offset of the compact storage length cache array on the pool object.
    pub compact_lengths_offset: Option<Expr>,
    /// Byte offset of the compact storage capacity cache array on the pool object.
    pub compact_caps_offset: Option<Expr>,
    /// Linked list node size in bytes (for New IR emission).
    pub linked_list_node_size: Option<Expr>,
    /// Byte offset of the value field within a linked list node.
    pub linked_list_value_offset: Option<Expr>,
    /// Byte offset of the next pointer within a linked list node.
    pub linked_list_next_offset: Option<Expr>,
    /// Byte offset of the storage refs array on the shadow storage object.
    pub linked_list_storage_offset: Option<Expr>,
    /// Byte offset of the head field on the shadow stack object.
    pub linked_list_stack_head_offset: Option<Expr>,
    /// Byte offset of the size field on the shadow stack object.
    pub linked_list_stack_size_offset: Option<Expr>,
    /// Byte offset of the tail pointer on Queue objects (FIFO push).
    /// When set, enables Queue tracing support.
    pub linked_list_queue_tail_offset: Option<Expr>,
    /// Storage indices that are Queue (FIFO) storages.
    pub linked_list_queue_indices: Vec<Expr>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CallPolicyKind {
    ResidualVoid,
    ResidualVoidWrapped,
    MayForceVoid,
    MayForceVoidWrapped,
    ReleaseGilVoid,
    ReleaseGilVoidWrapped,
    LoopInvariantVoid,
    LoopInvariantVoidWrapped,
    ResidualInt,
    ResidualIntWrapped,
    MayForceInt,
    MayForceIntWrapped,
    ReleaseGilInt,
    ReleaseGilIntWrapped,
    LoopInvariantInt,
    LoopInvariantIntWrapped,
    ElidableInt,
    ElidableIntWrapped,
    ResidualRefWrapped,
    MayForceRefWrapped,
    ReleaseGilRefWrapped,
    LoopInvariantRefWrapped,
    ElidableRefWrapped,
    ResidualFloatWrapped,
    MayForceFloatWrapped,
    ReleaseGilFloatWrapped,
    LoopInvariantFloatWrapped,
    ElidableFloatWrapped,
    InlineInt,
}

pub(crate) fn parse_call_policy_kind(kind: &Ident) -> Option<CallPolicyKind> {
    Some(match kind.to_string().as_str() {
        "residual_void" => CallPolicyKind::ResidualVoid,
        "residual_void_wrapped" => CallPolicyKind::ResidualVoidWrapped,
        "may_force_void" => CallPolicyKind::MayForceVoid,
        "may_force_void_wrapped" => CallPolicyKind::MayForceVoidWrapped,
        "release_gil_void" => CallPolicyKind::ReleaseGilVoid,
        "release_gil_void_wrapped" => CallPolicyKind::ReleaseGilVoidWrapped,
        "loopinvariant_void" => CallPolicyKind::LoopInvariantVoid,
        "loopinvariant_void_wrapped" => CallPolicyKind::LoopInvariantVoidWrapped,
        "residual_int" => CallPolicyKind::ResidualInt,
        "residual_int_wrapped" => CallPolicyKind::ResidualIntWrapped,
        "may_force_int" => CallPolicyKind::MayForceInt,
        "may_force_int_wrapped" => CallPolicyKind::MayForceIntWrapped,
        "release_gil_int" => CallPolicyKind::ReleaseGilInt,
        "release_gil_int_wrapped" => CallPolicyKind::ReleaseGilIntWrapped,
        "loopinvariant_int" => CallPolicyKind::LoopInvariantInt,
        "loopinvariant_int_wrapped" => CallPolicyKind::LoopInvariantIntWrapped,
        "elidable_int" => CallPolicyKind::ElidableInt,
        "elidable_int_wrapped" => CallPolicyKind::ElidableIntWrapped,
        "residual_ref_wrapped" => CallPolicyKind::ResidualRefWrapped,
        "may_force_ref_wrapped" => CallPolicyKind::MayForceRefWrapped,
        "release_gil_ref_wrapped" => CallPolicyKind::ReleaseGilRefWrapped,
        "loopinvariant_ref_wrapped" => CallPolicyKind::LoopInvariantRefWrapped,
        "elidable_ref_wrapped" => CallPolicyKind::ElidableRefWrapped,
        "residual_float_wrapped" => CallPolicyKind::ResidualFloatWrapped,
        "may_force_float_wrapped" => CallPolicyKind::MayForceFloatWrapped,
        "release_gil_float_wrapped" => CallPolicyKind::ReleaseGilFloatWrapped,
        "loopinvariant_float_wrapped" => CallPolicyKind::LoopInvariantFloatWrapped,
        "elidable_float_wrapped" => CallPolicyKind::ElidableFloatWrapped,
        "inline_int" => CallPolicyKind::InlineInt,
        _ => return None,
    })
}

impl Parse for JitInterpConfig {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut state_type = None;
        let mut env_type = None;
        let mut storage = None;
        let mut binops = None;
        let mut io_shims = None;
        let mut calls: Vec<(Path, Option<CallPolicyKind>)> = Vec::new();
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

        if storage.is_none() && state_fields.is_none() {
            return Err(syn::Error::new(
                input.span(),
                "missing `storage` or `state_fields` parameter",
            ));
        }

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
    let mut pool_ref = None;
    let mut selected_ref = None;
    let mut pool_type = None;
    let mut selector = None;
    let mut stacksize = None;
    let mut untraceable = Vec::new();
    let mut scan_fn = None;
    let mut can_trace_guard = None;
    let mut compact_live = false;
    let mut linked_list_node_size = None;
    let mut linked_list_value_offset = None;
    let mut linked_list_next_offset = None;
    let mut linked_list_storage_offset = None;
    let mut linked_list_stack_head_offset = None;
    let mut linked_list_stack_size_offset = None;
    let mut linked_list_queue_tail_offset = None;
    let mut linked_list_queue_indices: Vec<Expr> = Vec::new();
    let mut compact_encode = None;
    let mut compact_decode = None;
    let mut compact_min = None;
    let mut compact_max = None;
    let mut compact_ptrs_offset = None;
    let mut compact_lengths_offset = None;
    let mut compact_caps_offset = None;
    while !content.is_empty() {
        let key: Ident = content.parse()?;
        content.parse::<Token![:]>()?;

        match key.to_string().as_str() {
            "pool" => {
                pool = Some(content.parse::<Expr>()?);
            }
            "pool_ref" => {
                pool_ref = Some(content.parse::<Expr>()?);
            }
            "selected_ref" => {
                selected_ref = Some(content.parse::<Expr>()?);
            }
            "pool_type" => {
                pool_type = Some(content.parse::<Path>()?);
            }
            "selector" => {
                selector = Some(content.parse::<Expr>()?);
            }
            "stacksize" => {
                stacksize = Some(content.parse::<Expr>()?);
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
            "compact_live" => {
                compact_live = content.parse::<LitBool>()?.value;
            }
            "compact_encode" => {
                compact_encode = Some(content.parse::<Path>()?);
            }
            "compact_decode" => {
                compact_decode = Some(content.parse::<Path>()?);
            }
            "compact_min" => {
                compact_min = Some(content.parse::<Expr>()?);
            }
            "compact_max" => {
                compact_max = Some(content.parse::<Expr>()?);
            }
            "compact_ptrs_offset" => {
                compact_ptrs_offset = Some(content.parse::<Expr>()?);
            }
            "compact_lengths_offset" => {
                compact_lengths_offset = Some(content.parse::<Expr>()?);
            }
            "compact_caps_offset" => {
                compact_caps_offset = Some(content.parse::<Expr>()?);
            }
            "linked_list_node_size" => {
                linked_list_node_size = Some(content.parse::<Expr>()?);
            }
            "linked_list_value_offset" => {
                linked_list_value_offset = Some(content.parse::<Expr>()?);
            }
            "linked_list_next_offset" => {
                linked_list_next_offset = Some(content.parse::<Expr>()?);
            }
            "linked_list_storage_offset" => {
                linked_list_storage_offset = Some(content.parse::<Expr>()?);
            }
            "linked_list_stack_head_offset" => {
                linked_list_stack_head_offset = Some(content.parse::<Expr>()?);
            }
            "linked_list_stack_size_offset" => {
                linked_list_stack_size_offset = Some(content.parse::<Expr>()?);
            }
            "linked_list_queue_tail_offset" => {
                linked_list_queue_tail_offset = Some(content.parse::<Expr>()?);
            }
            "linked_list_queue_indices" => {
                let inner;
                bracketed!(inner in content);
                let exprs: Punctuated<Expr, Token![,]> =
                    inner.parse_terminated(Expr::parse, Token![,])?;
                linked_list_queue_indices = exprs.into_iter().collect();
            }
            "virtualizable" => {
                let _: LitBool = content.parse()?;
                return Err(syn::Error::new(
                    key.span(),
                    "storage.virtualizable is no longer supported; use `virtualizable_fields` or `state_fields`",
                ));
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
        pool_ref,
        selected_ref,
        pool_type,
        selector,
        stacksize,
        untraceable,
        scan_fn,
        can_trace_guard,
        compact_live,
        compact_encode,
        compact_decode,
        compact_min,
        compact_max,
        compact_ptrs_offset,
        compact_lengths_offset,
        compact_caps_offset,
        linked_list_node_size,
        linked_list_value_offset,
        linked_list_next_offset,
        linked_list_storage_offset,
        linked_list_stack_head_offset,
        linked_list_stack_size_offset,
        linked_list_queue_tail_offset,
        linked_list_queue_indices,
    })
}

/// Parse virtualizable_fields = { var: IDENT, token_offset: PATH, fields: { ... }, arrays: { ... } }
///
/// Parse `state_fields = { name: type, ... }` where type is `int`, `[int]`,
/// or `[int; virt]`.
fn parse_state_fields(input: ParseStream) -> syn::Result<StateFieldsConfig> {
    let content;
    braced!(content in input);
    let mut fields = Vec::new();

    while !content.is_empty() {
        let name: Ident = content.parse()?;
        content.parse::<Token![:]>()?;

        let kind = if content.peek(syn::token::Bracket) {
            // Array: [int] or virtualizable: [int; virt]
            let inner;
            bracketed!(inner in content);
            let item_type: Ident = inner.parse()?;
            if item_type != "int" {
                return Err(syn::Error::new(
                    item_type.span(),
                    format!(
                        "state_fields array `{name}` uses unsupported item type `{item_type}`; \
                         only `int` is currently supported"
                    ),
                ));
            }
            if inner.peek(Token![;]) {
                inner.parse::<Token![;]>()?;
                let flag: Ident = inner.parse()?;
                if flag == "virt" {
                    StateFieldKind::VirtArray(item_type)
                } else {
                    return Err(syn::Error::new(
                        flag.span(),
                        format!("unknown array modifier `{flag}`, expected `virt`"),
                    ));
                }
            } else {
                StateFieldKind::Array(item_type)
            }
        } else {
            // Scalar: int
            let field_type: Ident = content.parse()?;
            if field_type != "int" {
                return Err(syn::Error::new(
                    field_type.span(),
                    format!(
                        "state_fields scalar `{name}` uses unsupported type `{field_type}`; \
                         only `int` is currently supported"
                    ),
                ));
            }
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
                    let field_type: Ident = inner.call(Ident::parse_any)?;
                    inner.parse::<Token![@]>()?;
                    let offset: Expr = if inner.peek(syn::token::Paren) {
                        let expr_content;
                        syn::parenthesized!(expr_content in inner);
                        expr_content.parse::<Expr>()?
                    } else {
                        inner.parse::<Expr>()?
                    };
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
                    let item_type: Ident = inner.call(Ident::parse_any)?;
                    inner.parse::<Token![@]>()?;
                    let field_offset: Expr = if inner.peek(syn::token::Paren) {
                        let expr_content;
                        syn::parenthesized!(expr_content in inner);
                        expr_content.parse::<Expr>()?
                    } else {
                        inner.parse::<Expr>()?
                    };
                    let layout = if inner.peek(syn::token::Brace) {
                        let layout_content;
                        braced!(layout_content in inner);
                        let mut ptr_offset = None;
                        let mut length_offset = None;
                        let mut items_offset = None;
                        while !layout_content.is_empty() {
                            let layout_key: Ident = layout_content.parse()?;
                            layout_content.parse::<Token![:]>()?;
                            match layout_key.to_string().as_str() {
                                "ptr_offset" => {
                                    ptr_offset = Some(layout_content.parse::<Expr>()?);
                                }
                                "length_offset" => {
                                    length_offset = Some(layout_content.parse::<Expr>()?);
                                }
                                "items_offset" => {
                                    items_offset = Some(layout_content.parse::<Expr>()?);
                                }
                                other => {
                                    return Err(syn::Error::new(
                                        layout_key.span(),
                                        format!(
                                            "unknown virtualizable array layout parameter: `{other}`"
                                        ),
                                    ));
                                }
                            }
                            let _ = layout_content.parse::<Token![,]>();
                        }
                        VableArrayLayoutDecl::Embedded {
                            field_offset,
                            ptr_offset: ptr_offset.ok_or_else(|| {
                                syn::Error::new(
                                    inner.span(),
                                    "missing `ptr_offset` in embedded virtualizable array layout",
                                )
                            })?,
                            length_offset: length_offset.ok_or_else(|| {
                                syn::Error::new(
                                    inner.span(),
                                    "missing `length_offset` in embedded virtualizable array layout",
                                )
                            })?,
                            items_offset: items_offset.ok_or_else(|| {
                                syn::Error::new(
                                    inner.span(),
                                    "missing `items_offset` in embedded virtualizable array layout",
                                )
                            })?,
                        }
                    } else {
                        VableArrayLayoutDecl::Direct { field_offset }
                    };
                    arrays.push(VableArrayDecl {
                        name,
                        item_type,
                        layout,
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
fn parse_call_map(input: ParseStream) -> syn::Result<Vec<(Path, Option<CallPolicyKind>)>> {
    let content;
    braced!(content in input);
    let mut map = Vec::new();
    while !content.is_empty() {
        let func: Path = content.parse()?;
        let kind = if content.peek(Token![=>]) {
            content.parse::<Token![=>]>()?;
            let kind: Ident = content.parse()?;
            Some(parse_call_policy_kind(&kind).ok_or_else(|| {
                syn::Error::new(
                    kind.span(),
                    "call policy must be a supported residual/may_force/release_gil/loopinvariant policy or inline_int",
                )
            })?)
        } else {
            None
        };
        map.push((func, kind));
        let _ = content.parse::<Token![,]>();
    }
    Ok(map)
}

/// Parse `[func_a, func_b, func_c]` — shorthand helper list with auto-inferred policies.
fn parse_helpers_list(input: ParseStream) -> syn::Result<Vec<(Path, Option<CallPolicyKind>)>> {
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
    if config.state_fields.is_some() {
        quote! {
            #[cold]
            #[inline(never)]
            #[allow(non_snake_case)]
            fn #merge_fn_name(
                __driver: &mut majit_meta::JitDriver<#state_type>,
                __env: &#env_type,
                __pc: usize,
            ) {
                __driver.merge_point(|__ctx, __sym| {
                    use majit_meta::JitCodeSym;
                    if __sym.trace_started && __pc == __sym.loop_header_pc() {
                        return majit_meta::TraceAction::CloseLoop;
                    }
                    let __result = #trace_fn_name(__ctx, __sym, __env, __pc);
                    __sym.trace_started = true;
                    __result
                });
            }
        }
    } else {
        let storage = config
            .storage
            .as_ref()
            .expect("storage config required in storage mode");
        let pool_type = &storage.pool_type;
        let close_selected_check = if storage.compact_live {
            quote! {}
        } else {
            quote! { && __sel == __sym.header_selected() }
        };
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
                    use majit_meta::JitCodeSym;
                    if __sym.trace_started
                        && __pc == __sym.loop_header_pc()
                        #close_selected_check
                    {
                        return majit_meta::TraceAction::CloseLoop;
                    }
                    let __result = #trace_fn_name(__ctx, __sym, __env, __pc, __pool, __sel);
                    __sym.trace_started = true;
                    __result
                });
            }
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

    let pool_expr = config.storage.as_ref().map(|s| &s.pool);
    let sel_expr = config.storage.as_ref().map(|s| &s.selector);

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
    pool_expr: Option<&Expr>,
    sel_expr: Option<&Expr>,
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
        pool_expr: Option<Expr>,
        sel_expr: Option<Expr>,
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
                    let new_tokens: TokenStream = if self.is_state_fields {
                        quote! {
                            if #driver.is_tracing() {
                                #merge_fn(&mut #driver, #env, #pc);
                            }
                        }
                    } else {
                        let pool = self.pool_expr.as_ref().expect("storage pool expr");
                        let sel = self.sel_expr.as_ref().expect("storage selector expr");
                        quote! {
                            if #driver.is_tracing() {
                                #merge_fn(&mut #driver, #env, #pc, &#pool, #sel);
                            }
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
                            let is_sf = self.is_state_fields;
                            let stacksize_update: TokenStream = if is_sf {
                                quote! { #stacksize_expr = 0i32; }
                            } else {
                                let pool = pool_expr.as_ref().expect("storage pool expr");
                                let sel = sel_expr.as_ref().expect("storage selector expr");
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
        pool_expr: pool_expr.cloned(),
        sel_expr: sel_expr.cloned(),
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
        let result: Vec<(Path, Option<CallPolicyKind>)> =
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
        let result: Vec<(Path, Option<CallPolicyKind>)> =
            syn::parse2::<HelpersListWrapper>(tokens).unwrap().0;
        assert!(result.is_empty());
    }

    #[test]
    fn parse_helpers_list_with_path() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            [module::helper_a, helper_b]
        };
        let result: Vec<(Path, Option<CallPolicyKind>)> =
            syn::parse2::<HelpersListWrapper>(tokens).unwrap().0;
        assert_eq!(result.len(), 2);
        // First has two path segments
        assert_eq!(result[0].0.segments.len(), 2);
        assert_eq!(result[0].0.segments[0].ident.to_string(), "module");
        assert_eq!(result[0].0.segments[1].ident.to_string(), "helper_a");
    }

    /// Wrapper to make `parse_helpers_list` testable via `syn::parse2`.
    struct HelpersListWrapper(Vec<(Path, Option<CallPolicyKind>)>);
    impl Parse for HelpersListWrapper {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            Ok(Self(parse_helpers_list(input)?))
        }
    }

    struct StorageWrapper(StorageConfig);
    impl Parse for StorageWrapper {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            Ok(Self(parse_storage_config(input)?))
        }
    }

    struct VirtualizableWrapper(VirtualizableDecl);
    impl Parse for VirtualizableWrapper {
        fn parse(input: ParseStream) -> syn::Result<Self> {
            Ok(Self(parse_virtualizable_decl(input)?))
        }
    }

    #[test]
    fn parse_storage_config_rejects_legacy_virtualizable_flag() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            {
                pool: state.storage,
                pool_type: StoragePool,
                selector: state.selected,
                untraceable: [],
                scan: find_used_storages,
                virtualizable: true,
            }
        };
        let err = match syn::parse2::<StorageWrapper>(tokens) {
            Ok(_) => panic!("expected legacy storage.virtualizable to be rejected"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("storage.virtualizable"),
            "unexpected error: {err}",
        );
    }

    #[test]
    fn parse_virtualizable_decl_keeps_direct_array_layout() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            {
                var: frame,
                token_offset: FRAME_TOKEN_OFFSET,
                fields: { next_instr: int @ NEXT_INSTR_OFFSET },
                arrays: { locals_w: ref @ LOCALS_OFFSET },
            }
        };
        let parsed = syn::parse2::<VirtualizableWrapper>(tokens).unwrap().0;
        assert_eq!(parsed.arrays.len(), 1);
        match &parsed.arrays[0].layout {
            VableArrayLayoutDecl::Direct { field_offset } => {
                assert_eq!(quote::quote!(#field_offset).to_string(), "LOCALS_OFFSET");
            }
            VableArrayLayoutDecl::Embedded { .. } => {
                panic!("expected direct array layout");
            }
        }
    }

    #[test]
    fn parse_virtualizable_decl_supports_embedded_array_layout() {
        let tokens: proc_macro2::TokenStream = parse_quote! {
            {
                var: frame,
                token_offset: FRAME_TOKEN_OFFSET,
                fields: {},
                arrays: {
                    stack: int @ (STORAGEPOOL_DATA_PTRS_OFFSET + SLOT) {
                        ptr_offset: 0,
                        length_offset: STORAGEPOOL_LENGTHS_OFFSET - STORAGEPOOL_DATA_PTRS_OFFSET,
                        items_offset: 0,
                    },
                },
            }
        };
        let parsed = syn::parse2::<VirtualizableWrapper>(tokens).unwrap().0;
        assert_eq!(parsed.arrays.len(), 1);
        match &parsed.arrays[0].layout {
            VableArrayLayoutDecl::Embedded {
                field_offset,
                ptr_offset,
                length_offset,
                items_offset,
            } => {
                assert_eq!(
                    quote::quote!(#field_offset).to_string(),
                    "STORAGEPOOL_DATA_PTRS_OFFSET + SLOT"
                );
                assert_eq!(quote::quote!(#ptr_offset).to_string(), "0");
                assert_eq!(
                    quote::quote!(#length_offset).to_string(),
                    "STORAGEPOOL_LENGTHS_OFFSET - STORAGEPOOL_DATA_PTRS_OFFSET"
                );
                assert_eq!(quote::quote!(#items_offset).to_string(), "0");
            }
            VableArrayLayoutDecl::Direct { .. } => {
                panic!("expected embedded array layout");
            }
        }
    }
}
