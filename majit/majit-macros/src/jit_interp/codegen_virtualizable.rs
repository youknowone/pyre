//! Shared code generation for virtualizable info builder and state hooks.
//!
//! Used by both `#[jit_interp]` (storage-pool mode) and the standalone
//! `virtualizable!` proc macro.

use proc_macro2::TokenStream;
use quote::quote;

use super::{VableArrayLayoutDecl, VirtualizableDecl};

/// Map a field type identifier ("int", "ref", "float") to a `majit_ir::Type` token.
fn type_token(field_type: &syn::Ident) -> TokenStream {
    if field_type == "ref" {
        quote! { majit_ir::Type::Ref }
    } else if field_type == "float" {
        quote! { majit_ir::Type::Float }
    } else {
        quote! { majit_ir::Type::Int }
    }
}

/// Generate the `__build_virtualizable_info()` method body.
///
/// Returns a `TokenStream` for a function that constructs a
/// `majit_metainterp::virtualizable::VirtualizableInfo` from the declaration.
pub fn generate_vable_info_fn(decl: &VirtualizableDecl) -> TokenStream {
    let token_offset = &decl.token_offset;
    let field_adds: Vec<TokenStream> = decl
        .fields
        .iter()
        .map(|f| {
            let name = f.name.to_string();
            let offset = &f.offset;
            let tp = type_token(&f.field_type);
            quote! {
                __info.add_field(#name, #tp, #offset);
            }
        })
        .collect();
    let array_adds: Vec<TokenStream> = decl
        .arrays
        .iter()
        .map(|a| {
            let name = a.name.to_string();
            let tp = type_token(&a.item_type);
            match &a.layout {
                VableArrayLayoutDecl::Direct { field_offset } => quote! {
                    __info.add_array_field(#name, #tp, #field_offset);
                },
                VableArrayLayoutDecl::Embedded {
                    field_offset,
                    ptr_offset,
                    length_offset,
                    items_offset,
                } => quote! {
                    __info.add_embedded_array_field_with_layout(
                        #name,
                        #tp,
                        #field_offset,
                        #ptr_offset,
                        #length_offset,
                        #items_offset,
                    );
                },
            }
        })
        .collect();
    quote! {
        #[allow(non_snake_case)]
        fn __build_virtualizable_info() -> Option<majit_metainterp::virtualizable::VirtualizableInfo> {
            let mut __info = majit_metainterp::virtualizable::VirtualizableInfo::new(#token_offset);
            #(#field_adds)*
            #(#array_adds)*
            Some(__info)
        }
    }
}

/// Generate a standalone `build_virtualizable_info()` function (public, not a trait method).
pub fn generate_vable_info_pub_fn(decl: &VirtualizableDecl) -> TokenStream {
    let token_offset = &decl.token_offset;
    let name_str = decl.var_name.to_string();
    let field_adds: Vec<TokenStream> = decl
        .fields
        .iter()
        .map(|f| {
            let fname = f.name.to_string();
            let offset = &f.offset;
            let tp = type_token(&f.field_type);
            quote! {
                __info.add_field(#fname, #tp, #offset);
            }
        })
        .collect();
    let array_adds: Vec<TokenStream> = decl
        .arrays
        .iter()
        .map(|a| {
            let aname = a.name.to_string();
            let tp = type_token(&a.item_type);
            match &a.layout {
                VableArrayLayoutDecl::Direct { field_offset } => quote! {
                    __info.add_array_field(#aname, #tp, #field_offset);
                },
                VableArrayLayoutDecl::Embedded {
                    field_offset,
                    ptr_offset,
                    length_offset,
                    items_offset,
                } => quote! {
                    __info.add_embedded_array_field_with_layout(
                        #aname,
                        #tp,
                        #field_offset,
                        #ptr_offset,
                        #length_offset,
                        #items_offset,
                    );
                },
            }
        })
        .collect();
    quote! {
        /// Build the `VirtualizableInfo` descriptor for this virtualizable.
        pub fn build_virtualizable_info() -> majit_metainterp::virtualizable::VirtualizableInfo {
            let mut __info = majit_metainterp::virtualizable::VirtualizableInfo::new(#token_offset);
            __info.name = #name_str.to_string();
            #(#field_adds)*
            #(#array_adds)*
            __info
        }
    }
}

/// Generate virtualizable state hooks for a JitState impl block.
///
/// `heap_ptr_expr` is the expression to obtain `*mut u8` from `self`
/// (e.g., `(&self.pool as *const _ as *mut u8)` for storage-pool,
///  or a custom expression for pyre).
///
/// Returns method definitions for: `virtualizable_heap_ptr`,
/// `virtualizable_array_lengths`, `import_virtualizable_boxes`,
/// `export_virtualizable_boxes`, `sync_virtualizable_before_residual_call`,
/// `sync_virtualizable_after_residual_call`.
pub fn generate_vable_state_hooks(
    decl: &VirtualizableDecl,
    heap_ptr_expr: &TokenStream,
) -> TokenStream {
    let virtualizable_name = decl.var_name.to_string();
    quote! {
        fn virtualizable_heap_ptr(
            &self,
            _meta: &Self::Meta,
            virtualizable: &str,
            _info: &majit_metainterp::virtualizable::VirtualizableInfo,
        ) -> Option<*mut u8> {
            if virtualizable == #virtualizable_name {
                Some(#heap_ptr_expr)
            } else {
                None
            }
        }

        fn virtualizable_array_lengths(
            &self,
            _meta: &Self::Meta,
            virtualizable: &str,
            info: &majit_metainterp::virtualizable::VirtualizableInfo,
        ) -> Option<Vec<usize>> {
            if virtualizable != #virtualizable_name {
                return None;
            }
            if info.can_read_all_array_lengths_from_heap() {
                let obj_ptr = (#heap_ptr_expr).cast_const();
                Some(unsafe { info.read_array_lengths_from_heap(obj_ptr) })
            } else {
                None
            }
        }

        fn import_virtualizable_boxes(
            &mut self,
            _meta: &Self::Meta,
            _virtualizable: &str,
            _info: &majit_metainterp::virtualizable::VirtualizableInfo,
            _static_boxes: &[i64],
            _array_boxes: &[Vec<i64>],
        ) -> bool {
            true
        }

        fn export_virtualizable_boxes(
            &self,
            meta: &Self::Meta,
            virtualizable: &str,
            info: &majit_metainterp::virtualizable::VirtualizableInfo,
        ) -> Option<(Vec<i64>, Vec<Vec<i64>>)> {
            if virtualizable != #virtualizable_name {
                return None;
            }
            let obj_ptr = (#heap_ptr_expr).cast_const();
            let lengths = if info.can_read_all_array_lengths_from_heap() {
                unsafe { info.read_array_lengths_from_heap(obj_ptr) }
            } else {
                self.virtualizable_array_lengths(meta, virtualizable, info)?
            };
            Some(unsafe { info.read_all_boxes(obj_ptr, &lengths) })
        }

        fn sync_virtualizable_before_residual_call(
            &self,
            ctx: &mut majit_metainterp::trace_ctx::TraceCtx,
        ) {
            let Some(info) = Self::__build_virtualizable_info() else {
                return;
            };
            let Some(vable_ref) = ctx.standard_virtualizable_box() else {
                return;
            };
            let obj_ptr = #heap_ptr_expr;
            unsafe {
                info.tracing_before_residual_call(obj_ptr);
            }
            let force_token = ctx.force_token();
            ctx.vable_setfield_descr(vable_ref, force_token, info.token_field_descr());
        }

        fn sync_virtualizable_after_residual_call(
            &self,
            _ctx: &mut majit_metainterp::trace_ctx::TraceCtx,
        ) -> majit_metainterp::jit_state::ResidualVirtualizableSync {
            let Some(info) = Self::__build_virtualizable_info() else {
                return majit_metainterp::jit_state::ResidualVirtualizableSync::default();
            };
            let obj_ptr = #heap_ptr_expr;
            let forced = unsafe { info.tracing_after_residual_call(obj_ptr) };
            majit_metainterp::jit_state::ResidualVirtualizableSync {
                updated_fields: Vec::new(),
                forced,
            }
        }
    }
}

/// Generate field/array spec constants.
pub fn generate_vable_specs(decl: &VirtualizableDecl) -> TokenStream {
    let num_fields = decl.fields.len();
    let num_arrays = decl.arrays.len();
    let field_names: Vec<String> = decl.fields.iter().map(|f| f.name.to_string()).collect();
    let field_indices: Vec<usize> = (0..num_fields).collect();
    let array_names: Vec<String> = decl.arrays.iter().map(|a| a.name.to_string()).collect();
    let array_indices: Vec<usize> = (0..num_arrays).collect();
    quote! {
        /// Number of static virtualizable fields.
        pub const NUM_STATIC_FIELDS: usize = #num_fields;
        /// Number of virtualizable array fields.
        pub const NUM_ARRAY_FIELDS: usize = #num_arrays;
        /// Static field specs: (name, index).
        pub const VABLE_FIELD_SPECS: &[(&str, usize)] = &[
            #( (#field_names, #field_indices), )*
        ];
        /// Array field specs: (name, index).
        pub const VABLE_ARRAY_SPECS: &[(&str, usize)] = &[
            #( (#array_names, #array_indices), )*
        ];
    }
}
