//! Shared code generation for standalone `virtualizable!` helpers.

use proc_macro2::TokenStream;
use quote::quote;

use crate::jit_interp::{VableArrayLayoutDecl, VirtualizableDecl};

fn type_token(field_type: &syn::Ident) -> TokenStream {
    if field_type == "ref" {
        quote! { majit_ir::Type::Ref }
    } else if field_type == "float" {
        quote! { majit_ir::Type::Float }
    } else {
        quote! { majit_ir::Type::Int }
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
            let item_size_expr = quote! {
                majit_metainterp::virtualizable::item_size_for_type(#tp)
            };
            match &a.layout {
                VableArrayLayoutDecl::Direct { field_offset } => quote! {
                    __info.add_array_field(
                        #aname, #tp, #field_offset, 0, 0,
                        majit_ir::make_array_descr(0, #item_size_expr, #tp),
                    );
                },
                VableArrayLayoutDecl::Embedded {
                    field_offset,
                    ptr_offset,
                    length_offset,
                    items_offset,
                } => quote! {
                    __info.add_embedded_array_field(
                        #aname,
                        #tp,
                        #field_offset,
                        #ptr_offset,
                        #length_offset,
                        #items_offset,
                        majit_ir::make_array_descr(#items_offset, #item_size_expr, #tp),
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
