//! Standalone `virtualizable!` proc macro implementation.
//!
//! Generates `VirtualizableInfo` builder and JitState hook helpers
//! from a declarative virtualizable field specification.
//!
//! # Usage
//!
//! ```ignore
//! majit_macros::virtualizable! {
//!     state = MyState,
//!     name = "frame",
//!     heap_ptr = |s| s.frame_ptr(),
//!     token_offset = VABLE_TOKEN_OFFSET,
//!
//!     fields = {
//!         next_instr: int @ NEXT_INSTR_OFFSET,
//!         code: ref @ CODE_OFFSET,
//!     },
//!
//!     arrays = {
//!         stack: ref @ STACK_OFFSET {
//!             ptr_offset: PTR_OFFSET,
//!             length_offset: LEN_OFFSET,
//!             items_offset: 0,
//!         },
//!     },
//! }
//! ```

use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    Expr, Ident, LitStr, Path, Token, braced,
    ext::IdentExt,
    parse::{Parse, ParseStream},
};

use crate::jit_interp::{
    VableArrayDecl, VableArrayLayoutDecl, VableFieldDecl, VirtualizableDecl, codegen_virtualizable,
};

/// Parsed configuration from the `virtualizable! { ... }` macro.
struct VirtualizableMacroInput {
    /// The interpreter state type (e.g., `PyreJitState`).
    state_type: Ident,
    /// Virtualizable declaration (reuses jit_interp types).
    decl: VirtualizableDecl,
    /// Expression to obtain `*mut u8` heap pointer from `&self`.
    /// The closure parameter name is `s` (bound to `&self`).
    heap_ptr_expr: Expr,
}

impl Parse for VirtualizableMacroInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut state_type = None;
        let mut var_name = None;
        let mut token_offset = None;
        let mut fields = Vec::new();
        let mut arrays = Vec::new();
        let mut heap_ptr_expr = None;

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
                    // Parse closure-like expression: |s| s.frame_ptr()
                    // We store the body expression and will substitute `self` for `s`.
                    heap_ptr_expr = Some(input.parse::<Expr>()?);
                }
                "token_offset" => {
                    token_offset = Some(input.parse::<Path>()?);
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
                        // Parse offset as Path to avoid struct-literal ambiguity
                        // when followed by `{ embedded, ptr_offset: ... }`.
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
        })
    }
}

/// Generate output for the `virtualizable!` macro.
pub fn expand(input: VirtualizableMacroInput) -> TokenStream {
    let state_type = &input.state_type;
    let decl = &input.decl;
    let heap_ptr_expr = &input.heap_ptr_expr;
    let vable_name = decl.var_name.to_string();

    // 1. Public build_virtualizable_info() function.
    let info_fn = codegen_virtualizable::generate_vable_info_pub_fn(decl);

    // 2. Field/array spec constants.
    let specs = codegen_virtualizable::generate_vable_specs(decl);

    // 3. Standalone helper functions.
    //    The heap_ptr closure is invoked as `(closure)(state)` where `state`
    //    is the free function parameter.
    let hooks = generate_standalone_hooks(state_type, &vable_name, heap_ptr_expr);

    quote! {
        #info_fn
        #specs
        #hooks
    }
}

/// Generate standalone helper functions for JitState virtualizable hooks.
///
/// These are free functions that take `&State` as the first parameter,
/// allowing the user to delegate from their manual JitState impl without
/// generating a partial impl block.
fn generate_standalone_hooks(
    state_type: &Ident,
    vable_name: &str,
    heap_ptr_closure: &Expr,
) -> TokenStream {
    quote! {
        /// Get the raw heap pointer, returning None if null.
        fn __heap_ptr(state: &#state_type) -> Option<*mut u8> {
            let ptr = (#heap_ptr_closure)(state);
            if ptr.is_null() { None } else { Some(ptr) }
        }

        /// Helper: get virtualizable heap pointer.
        pub fn virt_heap_ptr(
            state: &#state_type,
            _virtualizable: &str,
        ) -> Option<*mut u8> {
            if _virtualizable != #vable_name {
                return None;
            }
            __heap_ptr(state)
        }

        /// Helper: get virtualizable array lengths from heap.
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

        /// Helper: export virtualizable boxes from heap.
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

        /// Helper: sync virtualizable before a residual call during tracing.
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

        /// Helper: sync virtualizable after a residual call during tracing.
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
