//! Standalone `virtualizable!` proc macro implementation.
//!
//! Generates `VirtualizableInfo` builder, JitState layout helpers, and
//! virtualizable hook functions from a declarative specification.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Expr, Ident, LitStr, Path, Token, braced,
    ext::IdentExt,
    parse::{Parse, ParseStream},
};

use crate::jit_interp::{
    VableArrayDecl, VableArrayLayoutDecl, VableFieldDecl, VirtualizableDecl, codegen_virtualizable,
};

// ═══════════════════════════════════════════════════════════════
// Input arg field: a scalar field included in extract_live / jump_args.
// ═══════════════════════════════════════════════════════════════

struct InputArgField {
    /// Field name in the state struct (e.g., `next_instr`).
    name: Ident,
    /// IR type: `Int`, `Ref`, or `Float`.
    ir_type: Ident,
}

// ═══════════════════════════════════════════════════════════════
// Parsed macro configuration.
// ═══════════════════════════════════════════════════════════════

struct VirtualizableMacroInput {
    state_type: Ident,
    decl: VirtualizableDecl,
    heap_ptr_expr: Expr,
    /// Frame pointer field name in the state struct (e.g., `frame`).
    frame_field: Option<Ident>,
    /// Scalar fields included in extract_live / jump_args, in order.
    inputargs: Vec<InputArgField>,
    /// IR type for array items (default: Ref).
    array_item_type: Option<Ident>,
}

impl Parse for VirtualizableMacroInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut state_type = None;
        let mut var_name = None;
        let mut token_offset = None;
        let mut fields = Vec::new();
        let mut arrays = Vec::new();
        let mut heap_ptr_expr = None;
        let mut frame_field = None;
        let mut inputargs = Vec::new();
        let mut array_item_type = None;

        while !input.is_empty() {
            let key: Ident = input.parse()?;
            input.parse::<Token![=]>()?;

            match key.to_string().as_str() {
                "state" => {
                    state_type = Some(input.parse::<Ident>()?);
                }
                "name" => {
                    let lit: LitStr = input.parse()?;
                    var_name = Some(Ident::new(&lit.value(), lit.span()));
                }
                "heap_ptr" => {
                    heap_ptr_expr = Some(input.parse::<Expr>()?);
                }
                "token_offset" => {
                    token_offset = Some(input.parse::<Path>()?);
                }
                "frame_field" => {
                    frame_field = Some(input.parse::<Ident>()?);
                }
                "inputargs" => {
                    let inner;
                    braced!(inner in input);
                    while !inner.is_empty() {
                        let name: Ident = inner.parse()?;
                        inner.parse::<Token![:]>()?;
                        let ir_type: Ident = inner.parse()?;
                        inputargs.push(InputArgField { name, ir_type });
                        let _ = inner.parse::<Token![,]>();
                    }
                }
                "array_item_type" => {
                    array_item_type = Some(input.parse::<Ident>()?);
                }
                "fields" => {
                    let inner;
                    braced!(inner in input);
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
                    braced!(inner in input);
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
                            let path: Path = inner.parse()?;
                            syn::parse_quote!(#path)
                        };
                        let layout = if inner.peek(syn::token::Brace) {
                            let layout_content;
                            braced!(layout_content in inner);
                            let mut is_embedded = false;
                            let mut ptr_offset = None;
                            let mut length_offset = None;
                            let mut items_offset = None;
                            while !layout_content.is_empty() {
                                let layout_key: Ident = layout_content.parse()?;
                                match layout_key.to_string().as_str() {
                                    "embedded" => {
                                        is_embedded = true;
                                    }
                                    "ptr_offset" => {
                                        layout_content.parse::<Token![:]>()?;
                                        ptr_offset = Some(layout_content.parse::<Expr>()?);
                                    }
                                    "length_offset" => {
                                        layout_content.parse::<Token![:]>()?;
                                        length_offset = Some(layout_content.parse::<Expr>()?);
                                    }
                                    "items_offset" => {
                                        layout_content.parse::<Token![:]>()?;
                                        items_offset = Some(layout_content.parse::<Expr>()?);
                                    }
                                    other => {
                                        return Err(syn::Error::new(
                                            layout_key.span(),
                                            format!(
                                                "unknown virtualizable array layout key: `{other}`"
                                            ),
                                        ));
                                    }
                                }
                                let _ = layout_content.parse::<Token![,]>();
                            }
                            if is_embedded || ptr_offset.is_some() {
                                VableArrayLayoutDecl::Embedded {
                                    field_offset,
                                    ptr_offset: ptr_offset.ok_or_else(|| {
                                        syn::Error::new(
                                            inner.span(),
                                            "missing `ptr_offset` in embedded array layout",
                                        )
                                    })?,
                                    length_offset: length_offset.ok_or_else(|| {
                                        syn::Error::new(
                                            inner.span(),
                                            "missing `length_offset` in embedded array layout",
                                        )
                                    })?,
                                    items_offset: items_offset.ok_or_else(|| {
                                        syn::Error::new(
                                            inner.span(),
                                            "missing `items_offset` in embedded array layout",
                                        )
                                    })?,
                                }
                            } else {
                                VableArrayLayoutDecl::Direct { field_offset }
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
                        format!("unknown virtualizable parameter: `{other}`"),
                    ));
                }
            }
            let _ = input.parse::<Token![,]>();
        }

        let state_type = state_type.ok_or_else(|| {
            syn::Error::new(input.span(), "missing `state` in virtualizable! macro")
        })?;
        let var_name = var_name.ok_or_else(|| {
            syn::Error::new(input.span(), "missing `name` in virtualizable! macro")
        })?;
        let token_offset = token_offset.ok_or_else(|| {
            syn::Error::new(
                input.span(),
                "missing `token_offset` in virtualizable! macro",
            )
        })?;
        let heap_ptr_expr = heap_ptr_expr.ok_or_else(|| {
            syn::Error::new(input.span(), "missing `heap_ptr` in virtualizable! macro")
        })?;

        Ok(VirtualizableMacroInput {
            state_type,
            decl: VirtualizableDecl {
                var_name,
                token_offset,
                fields,
                arrays,
            },
            heap_ptr_expr,
            frame_field,
            inputargs,
            array_item_type,
        })
    }
}

// ═══════════════════════════════════════════════════════════════
// Code generation
// ═══════════════════════════════════════════════════════════════

fn ir_type_token(ident: &Ident) -> TokenStream {
    match ident.to_string().as_str() {
        "Ref" | "ref" => quote! { majit_ir::Type::Ref },
        "Float" | "float" => quote! { majit_ir::Type::Float },
        _ => quote! { majit_ir::Type::Int },
    }
}

fn ir_value_ctor(ir_type: &Ident, expr: TokenStream) -> TokenStream {
    match ir_type.to_string().as_str() {
        "Ref" | "ref" => quote! { majit_ir::Value::Ref(majit_ir::GcRef(#expr as usize)) },
        "Float" | "float" => quote! { majit_ir::Value::Float(f64::from_bits(#expr as u64)) },
        _ => quote! { majit_ir::Value::Int(#expr as i64) },
    }
}

pub fn expand(input: VirtualizableMacroInput) -> TokenStream {
    let state_type = &input.state_type;
    let decl = &input.decl;
    let heap_ptr_expr = &input.heap_ptr_expr;
    let vable_name = decl.var_name.to_string();

    // 1. VirtualizableInfo builder
    let info_fn = codegen_virtualizable::generate_vable_info_pub_fn(decl);

    // 2. Field/array spec constants
    let specs = codegen_virtualizable::generate_vable_specs(decl);

    // 3. Virtualizable state hooks (heap_ptr, sync, etc.)
    let hooks = generate_standalone_hooks(state_type, &vable_name, heap_ptr_expr);

    // 4. Layout-based JitState helpers (extract, collect, live_types, etc.)
    let layout_helpers = if !input.inputargs.is_empty() {
        generate_layout_helpers(
            state_type,
            &input.frame_field,
            &input.inputargs,
            &input.array_item_type,
        )
    } else {
        quote! {}
    };

    quote! {
        #info_fn
        #specs
        #hooks
        #layout_helpers
    }
}

/// Generate layout-based JitState helper functions.
///
/// Layout: [frame:Ref, inputarg0:T0, inputarg1:T1, ..., locals[]:ItemT, stack[]:ItemT]
fn generate_layout_helpers(
    state_type: &Ident,
    frame_field: &Option<Ident>,
    inputargs: &[InputArgField],
    array_item_type: &Option<Ident>,
) -> TokenStream {
    let frame_ident = frame_field
        .as_ref()
        .map_or_else(|| format_ident!("frame"), |f| f.clone());
    let arr_type = array_item_type
        .as_ref()
        .map_or_else(|| format_ident!("Ref"), |t| t.clone());
    let arr_type_token = ir_type_token(&arr_type);
    let num_scalars = 1 + inputargs.len(); // frame + inputarg fields

    // ── extract_live_values ──
    // Parameters: state field values + closures for array access
    let scalar_params: Vec<TokenStream> = inputargs
        .iter()
        .map(|f| {
            let name = &f.name;
            quote! { #name: usize }
        })
        .collect();
    let scalar_value_pushes: Vec<TokenStream> = inputargs
        .iter()
        .map(|f| {
            let name = &f.name;
            let vc = ir_value_ctor(&f.ir_type, quote! { #name });
            quote! { __vals.push(#vc); }
        })
        .collect();
    let frame_value = ir_value_ctor(&format_ident!("Ref"), quote! { __frame });
    let local_value = ir_value_ctor(&arr_type, quote! { local_at(__i) });
    let stack_value = ir_value_ctor(&arr_type, quote! { stack_at(__i) });

    // ── live_value_types ──
    let scalar_type_pushes: Vec<TokenStream> = {
        let mut v = vec![quote! { __types.push(majit_ir::Type::Ref); }]; // frame
        for f in inputargs {
            let tp = ir_type_token(&f.ir_type);
            v.push(quote! { __types.push(#tp); });
        }
        v
    };

    // ── collect_jump_args parameter names ──
    let sym_scalar_params: Vec<TokenStream> = inputargs
        .iter()
        .map(|f| {
            let sym_name = format_ident!("sym_{}", f.name);
            quote! { #sym_name: majit_ir::OpRef }
        })
        .collect();
    let sym_scalar_pushes: Vec<TokenStream> = inputargs
        .iter()
        .map(|f| {
            let sym_name = format_ident!("sym_{}", f.name);
            quote! { __args.push(#sym_name); }
        })
        .collect();

    // ── collect_typed_jump_args ──
    let sym_typed_scalar_pushes: Vec<TokenStream> = inputargs
        .iter()
        .map(|f| {
            let sym_name = format_ident!("sym_{}", f.name);
            let tp = ir_type_token(&f.ir_type);
            quote! { __args.push((#sym_name, #tp)); }
        })
        .collect();

    // ── OpRef index constants for create_sym ──
    let sym_idx_consts: Vec<TokenStream> = inputargs
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let const_name = format_ident!("SYM_{}_IDX", f.name.to_string().to_uppercase());
            let idx = (i + 1) as u32; // frame is 0
            quote! { pub const #const_name: u32 = #idx; }
        })
        .collect();
    let array_base_val = num_scalars as u32;

    // ── is_compatible helper ──
    let compat_params: Vec<TokenStream> = inputargs
        .iter()
        .map(|f| {
            let state_name = format_ident!("state_{}", f.name);
            let meta_name = format_ident!("meta_{}", f.name);
            quote! { #state_name: usize, #meta_name: usize }
        })
        .collect();
    let compat_checks: Vec<TokenStream> = inputargs
        .iter()
        .map(|f| {
            let state_name = format_ident!("state_{}", f.name);
            let meta_name = format_ident!("meta_{}", f.name);
            quote! { && #state_name == #meta_name }
        })
        .collect();

    // ── restore_values helper ──
    let restore_scalars: Vec<TokenStream> = inputargs
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let name = &f.name;
            let idx = i + 1; // frame is values[0]
            let tp = &f.ir_type;
            let conv = match tp.to_string().as_str() {
                "Ref" | "ref" => quote! { __value_to_usize(&values[#idx]) },
                "Float" | "float" => {
                    quote! { f64::from_bits(__value_to_usize(&values[#idx]) as u64) as usize }
                }
                _ => quote! { __value_to_usize(&values[#idx]) },
            };
            quote! {
                state.#name = #conv;
            }
        })
        .collect();

    quote! {
        // ── Layout constants ──

        /// OpRef index for the frame pointer in the symbolic state.
        pub const SYM_FRAME_IDX: u32 = 0;
        #(#sym_idx_consts)*
        /// OpRef base index for array slots (locals + stack).
        pub const SYM_ARRAY_BASE: u32 = #array_base_val;
        /// Number of scalar inputargs (frame + declared fields).
        pub const NUM_SCALAR_INPUTARGS: usize = #num_scalars;

        // ── extract_live_values ──

        /// Extract live values from state in virtualizable layout order.
        ///
        /// Layout: `[frame:Ref, scalars..., locals[0..nlocals]:ItemT, stack[0..stack_only]:ItemT]`
        ///
        /// `local_at(i)` returns the raw value of local slot `i`.
        /// `stack_at(i)` returns the raw value of stack-only slot `i`.
        pub fn virt_extract_live_values(
            __frame: usize,
            #(#scalar_params,)*
            __num_locals: usize,
            __valuestackdepth: usize,
            local_at: impl Fn(usize) -> usize,
            stack_at: impl Fn(usize) -> usize,
        ) -> Vec<majit_ir::Value> {
            let stack_only = __valuestackdepth.saturating_sub(__num_locals);
            let mut __vals = Vec::with_capacity(#num_scalars + __num_locals + stack_only);
            __vals.push(#frame_value);
            #(#scalar_value_pushes)*
            for __i in 0..__num_locals {
                __vals.push(#local_value);
            }
            for __i in 0..stack_only {
                __vals.push(#stack_value);
            }
            __vals
        }

        // ── live_value_types ──

        /// Return the IR type of each live value in layout order.
        pub fn virt_live_value_types(num_slots: usize) -> Vec<majit_ir::Type> {
            let mut __types = Vec::with_capacity(#num_scalars + num_slots);
            #(#scalar_type_pushes)*
            __types.extend(std::iter::repeat_n(#arr_type_token, num_slots));
            __types
        }

        // ── collect_jump_args ──

        /// Collect symbolic OpRefs for JUMP in layout order.
        pub fn virt_collect_jump_args(
            sym_frame: majit_ir::OpRef,
            #(#sym_scalar_params,)*
            symbolic_locals: &[majit_ir::OpRef],
            symbolic_stack: &[majit_ir::OpRef],
            stack_only: usize,
        ) -> Vec<majit_ir::OpRef> {
            let stack_len = stack_only.min(symbolic_stack.len());
            let mut __args = Vec::with_capacity(#num_scalars + symbolic_locals.len() + stack_len);
            __args.push(sym_frame);
            #(#sym_scalar_pushes)*
            __args.extend_from_slice(symbolic_locals);
            __args.extend_from_slice(&symbolic_stack[..stack_len]);
            __args
        }

        /// Collect typed symbolic OpRefs for JUMP in layout order.
        pub fn virt_collect_typed_jump_args(
            sym_frame: majit_ir::OpRef,
            #(#sym_scalar_params,)*
            symbolic_locals: &[majit_ir::OpRef],
            symbolic_local_types: &[majit_ir::Type],
            symbolic_stack: &[majit_ir::OpRef],
            symbolic_stack_types: &[majit_ir::Type],
            stack_only: usize,
        ) -> Vec<(majit_ir::OpRef, majit_ir::Type)> {
            let stack_len = stack_only.min(symbolic_stack.len());
            let mut __args = Vec::with_capacity(#num_scalars + symbolic_locals.len() + stack_len);
            __args.push((sym_frame, majit_ir::Type::Ref));
            #(#sym_typed_scalar_pushes)*
            for (__i, &__opref) in symbolic_locals.iter().enumerate() {
                let __tp = symbolic_local_types.get(__i).copied().unwrap_or(#arr_type_token);
                __args.push((__opref, __tp));
            }
            for (__i, &__opref) in symbolic_stack[..stack_len].iter().enumerate() {
                let __tp = symbolic_stack_types.get(__i).copied().unwrap_or(#arr_type_token);
                __args.push((__opref, __tp));
            }
            __args
        }

        // ── is_compatible ──

        /// Check if state scalar fields match meta (green key + shape check).
        pub fn virt_is_compatible(
            state_nlocals: usize, meta_nlocals: usize,
            #(#compat_params,)*
        ) -> bool {
            state_nlocals == meta_nlocals
            #(#compat_checks)*
        }

        // ── restore helpers ──

        fn __value_to_usize(v: &majit_ir::Value) -> usize {
            match v {
                majit_ir::Value::Int(i) => *i as usize,
                majit_ir::Value::Ref(r) => r.as_usize(),
                majit_ir::Value::Float(f) => f.to_bits() as usize,
                majit_ir::Value::Void => 0,
            }
        }

        /// Restore scalar fields from a Value slice (virtualizable layout).
        ///
        /// Writes `state.frame`, `state.<inputarg>...` from `values[0..]`.
        /// Returns the index of the first array slot in `values`.
        pub fn virt_restore_scalars(
            state: &mut #state_type,
            values: &[majit_ir::Value],
        ) -> usize {
            if values.is_empty() {
                return 0;
            }
            state.#frame_ident = __value_to_usize(&values[0]);
            #(#restore_scalars)*
            #num_scalars
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// Standalone hooks (heap_ptr, sync, etc.) — unchanged from before
// ═══════════════════════════════════════════════════════════════

fn generate_standalone_hooks(
    state_type: &Ident,
    vable_name: &str,
    heap_ptr_closure: &Expr,
) -> TokenStream {
    quote! {
        fn __heap_ptr(state: &#state_type) -> Option<*mut u8> {
            let ptr = (#heap_ptr_closure)(state);
            if ptr.is_null() { None } else { Some(ptr) }
        }

        pub fn virt_heap_ptr(
            state: &#state_type,
            _virtualizable: &str,
        ) -> Option<*mut u8> {
            if _virtualizable != #vable_name {
                return None;
            }
            __heap_ptr(state)
        }

        pub fn virt_array_lengths(
            state: &#state_type,
            _virtualizable: &str,
            info: &majit_metainterp::virtualizable::VirtualizableInfo,
        ) -> Option<Vec<usize>> {
            if _virtualizable != #vable_name {
                return None;
            }
            let obj_ptr = __heap_ptr(state)?;
            if info.can_read_all_array_lengths_from_heap() {
                Some(unsafe { info.read_array_lengths_from_heap(obj_ptr.cast_const()) })
            } else {
                None
            }
        }

        pub fn virt_export_boxes(
            state: &#state_type,
            _virtualizable: &str,
            info: &majit_metainterp::virtualizable::VirtualizableInfo,
        ) -> Option<(Vec<i64>, Vec<Vec<i64>>)> {
            if _virtualizable != #vable_name {
                return None;
            }
            let obj_ptr = __heap_ptr(state)?.cast_const();
            let lengths = if info.can_read_all_array_lengths_from_heap() {
                unsafe { info.read_array_lengths_from_heap(obj_ptr) }
            } else {
                virt_array_lengths(state, _virtualizable, info)?
            };
            Some(unsafe { info.read_all_boxes(obj_ptr, &lengths) })
        }

        pub fn virt_sync_before_residual(
            state: &#state_type,
            ctx: &mut majit_metainterp::TraceCtx,
        ) {
            let info = build_virtualizable_info();
            let Some(vable_ref) = ctx.standard_virtualizable_box() else {
                return;
            };
            ctx.gen_store_back_in_vable(vable_ref);
            let Some(obj_ptr) = __heap_ptr(state) else {
                return;
            };
            unsafe {
                info.tracing_before_residual_call(obj_ptr);
            }
            let force_token = ctx.force_token();
            ctx.vable_setfield_descr(vable_ref, force_token, info.token_field_descr());
        }

        pub fn virt_sync_after_residual(
            state: &#state_type,
            _ctx: &mut majit_metainterp::TraceCtx,
        ) -> majit_metainterp::ResidualVirtualizableSync {
            let info = build_virtualizable_info();
            let Some(obj_ptr) = __heap_ptr(state) else {
                return majit_metainterp::ResidualVirtualizableSync::default();
            };
            let forced = unsafe { info.tracing_after_residual_call(obj_ptr) };
            majit_metainterp::ResidualVirtualizableSync {
                updated_fields: Vec::new(),
                forced,
            }
        }
    }
}

/// Parse and expand the `virtualizable!` macro input.
pub fn parse_and_expand(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let config = syn::parse_macro_input!(input as VirtualizableMacroInput);
    expand(config).into()
}
