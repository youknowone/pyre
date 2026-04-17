//! Generate JitState types (Meta, Sym) and impl from the macro configuration.

use proc_macro2::TokenStream;
use quote::quote;

use super::{JitInterpConfig, StateFieldKind};

/// Generate the JitState types and implementation.
pub fn generate_jit_state(config: &JitInterpConfig) -> TokenStream {
    generate_state_fields_jit_state(config)
}

/// Generate JitState types for state_fields mode (register/tape machines).
///
/// Instead of a storage pool with stacks, individual struct fields are tracked
/// as JIT-managed values. Scalars become single OpRefs, flattened arrays become
/// Vec<OpRef>, and virtualizable arrays (`[int; virt]`) track only a data
/// pointer + length OpRef pair (array stays on heap, accessed via raw memory ops).
fn generate_state_fields_jit_state(config: &JitInterpConfig) -> TokenStream {
    let state_type = &config.state_type;
    let env_type = &config.env_type;
    let sf = config.state_fields.as_ref().unwrap();

    let unsupported_fields: Vec<String> = sf
        .fields
        .iter()
        .filter_map(|f| match &f.kind {
            StateFieldKind::Scalar { ir_type, .. } => {
                let ty = ir_type.to_string();
                if ty == "int" {
                    None
                } else {
                    Some(format!("{}: {}", f.name, ty))
                }
            }
            StateFieldKind::Array(tp) | StateFieldKind::VirtArray(tp) => {
                let ty = tp.to_string();
                if ty == "int" {
                    None
                } else {
                    Some(format!("{}: {}", f.name, ty))
                }
            }
            // RPython parity: opaque(T) fields are pass-through; the JIT
            // does not enumerate them as inputargs, so any T is allowed.
            StateFieldKind::Opaque(_) => None,
        })
        .collect();
    if !unsupported_fields.is_empty() {
        let message = format!(
            "state_fields supports int, [int], [int; virt], and opaque(T); unsupported: {}",
            unsupported_fields.join(", ")
        );
        return quote! {
            compile_error!(#message);
        };
    }

    // Separate scalars, flattened arrays, and virtualizable arrays.
    let scalars: Vec<_> = sf
        .fields
        .iter()
        .enumerate()
        .filter(|(_, f)| matches!(f.kind, StateFieldKind::Scalar { .. }))
        .collect();
    // Helper: per-scalar Rust storage type token (`i64` by default, or
    // the explicit `int(<TypePath>)` override). Used to emit `as <type>`
    // casts at the JIT boundary so user struct fields can stay in their
    // natural Rust types (e.g. `selected: usize`, `stacksize: i32`).
    let scalar_rust_type = |kind: &StateFieldKind| -> TokenStream {
        match kind {
            StateFieldKind::Scalar {
                rust_type: Some(p), ..
            } => quote! { #p },
            _ => quote! { i64 },
        }
    };
    let arrays: Vec<_> = sf
        .fields
        .iter()
        .enumerate()
        .filter(|(_, f)| matches!(f.kind, StateFieldKind::Array(_)))
        .collect();
    let virt_arrays: Vec<_> = sf
        .fields
        .iter()
        .enumerate()
        .filter(|(_, f)| matches!(f.kind, StateFieldKind::VirtArray(_)))
        .collect();

    let num_scalars = scalars.len();
    let num_virt_arrays = virt_arrays.len();

    // ── __JitMeta fields: one `{name}_len: usize` per flattened array ──
    // Virt arrays do NOT store length in meta (it's dynamic, tracked as inputarg).
    let meta_fields: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let len_name = quote::format_ident!("{}_len", f.name);
            quote! { #len_name: usize, }
        })
        .collect();

    // ── __JitSym fields ──
    // scalar → OpRef
    // flattened array → Vec<OpRef>
    // virt array → (OpRef, OpRef) for (data_ptr, len)
    let sym_scalar_fields: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { #fname: majit_ir::OpRef, }
        })
        .collect();
    let sym_scalar_value_fields: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let value_name = quote::format_ident!("{}_value", f.name);
            quote! { #value_name: i64, }
        })
        .collect();
    let sym_array_fields: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { #fname: Vec<majit_ir::OpRef>, }
        })
        .collect();
    let sym_array_value_fields: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let value_name = quote::format_ident!("{}_values", f.name);
            quote! { #value_name: Vec<i64>, }
        })
        .collect();
    let sym_virt_array_fields: Vec<TokenStream> = virt_arrays
        .iter()
        .map(|(_, f)| {
            let ptr_name = quote::format_ident!("{}_ptr", f.name);
            let len_name = quote::format_ident!("{}_len", f.name);
            quote! {
                #ptr_name: majit_ir::OpRef,
                #len_name: majit_ir::OpRef,
            }
        })
        .collect();

    // ── JitCodeSym: total_slots ──
    // num_scalars + sum(flattened array lengths) + 2 * num_virt_arrays
    let total_slots_array_parts: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { + self.#fname.len() }
        })
        .collect();

    // ── JitCodeSym: state_field_ref / set_state_field_ref ──
    let state_field_ref_arms: Vec<TokenStream> = scalars
        .iter()
        .enumerate()
        .map(|(idx, (_, f))| {
            let fname = &f.name;
            let idx_lit = idx;
            // OpRef field (Sym side) — return as-is.
            quote! { #idx_lit => Some(self.#fname), }
        })
        .collect();
    let set_state_field_ref_arms: Vec<TokenStream> = scalars
        .iter()
        .enumerate()
        .map(|(idx, (_, f))| {
            let fname = &f.name;
            let idx_lit = idx;
            // OpRef field on Sym — direct assignment, no cast.
            quote! { #idx_lit => { self.#fname = value; } }
        })
        .collect();
    let state_field_value_arms: Vec<TokenStream> = scalars
        .iter()
        .enumerate()
        .map(|(idx, (_, f))| {
            let value_name = quote::format_ident!("{}_value", f.name);
            let idx_lit = idx;
            quote! { #idx_lit => Some(self.#value_name), }
        })
        .collect();
    let set_state_field_value_arms: Vec<TokenStream> = scalars
        .iter()
        .enumerate()
        .map(|(idx, (_, f))| {
            let value_name = quote::format_ident!("{}_value", f.name);
            let idx_lit = idx;
            quote! { #idx_lit => { self.#value_name = value; } }
        })
        .collect();

    // ── JitCodeSym: state_array_ref / set_state_array_ref (flattened only) ──
    let state_array_ref_arms: Vec<TokenStream> = arrays
        .iter()
        .enumerate()
        .map(|(arr_idx, (_, f))| {
            let fname = &f.name;
            let arr_idx_lit = arr_idx;
            quote! { #arr_idx_lit => self.#fname.get(elem_idx).copied(), }
        })
        .collect();
    let set_state_array_ref_arms: Vec<TokenStream> = arrays
        .iter()
        .enumerate()
        .map(|(arr_idx, (_, f))| {
            let fname = &f.name;
            let arr_idx_lit = arr_idx;
            quote! { #arr_idx_lit => {
                if elem_idx < self.#fname.len() {
                    self.#fname[elem_idx] = value;
                }
            } }
        })
        .collect();
    let state_array_value_arms: Vec<TokenStream> = arrays
        .iter()
        .enumerate()
        .map(|(arr_idx, (_, f))| {
            let value_name = quote::format_ident!("{}_values", f.name);
            let arr_idx_lit = arr_idx;
            quote! { #arr_idx_lit => self.#value_name.get(elem_idx).copied(), }
        })
        .collect();
    let set_state_array_value_arms: Vec<TokenStream> = arrays
        .iter()
        .enumerate()
        .map(|(arr_idx, (_, f))| {
            let value_name = quote::format_ident!("{}_values", f.name);
            let arr_idx_lit = arr_idx;
            quote! { #arr_idx_lit => {
                if elem_idx < self.#value_name.len() {
                    self.#value_name[elem_idx] = value;
                }
            } }
        })
        .collect();

    // ── JitCodeSym: state_varray_ptr / state_varray_len ──
    let state_varray_ptr_arms: Vec<TokenStream> = virt_arrays
        .iter()
        .enumerate()
        .map(|(va_idx, (_, f))| {
            let ptr_name = quote::format_ident!("{}_ptr", f.name);
            let va_idx_lit = va_idx;
            quote! { #va_idx_lit => Some(self.#ptr_name), }
        })
        .collect();
    let state_varray_len_arms: Vec<TokenStream> = virt_arrays
        .iter()
        .enumerate()
        .map(|(va_idx, (_, f))| {
            let len_name = quote::format_ident!("{}_len", f.name);
            let va_idx_lit = va_idx;
            quote! { #va_idx_lit => Some(self.#len_name), }
        })
        .collect();

    // ── collect_jump_args: scalars, then flattened arrays, then virt array ptr+len ──
    let collect_scalar_parts: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { args.push(sym.#fname); }
        })
        .collect();
    let collect_array_parts: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { args.extend_from_slice(&sym.#fname); }
        })
        .collect();
    let collect_virt_array_parts: Vec<TokenStream> = virt_arrays
        .iter()
        .map(|(_, f)| {
            let ptr_name = quote::format_ident!("{}_ptr", f.name);
            let len_name = quote::format_ident!("{}_len", f.name);
            quote! {
                args.push(sym.#ptr_name);
                args.push(sym.#len_name);
            }
        })
        .collect();

    // ── fail_args ──
    let fail_scalar_parts: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { args.push(self.#fname); }
        })
        .collect();
    let fail_array_parts: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { args.extend_from_slice(&self.#fname); }
        })
        .collect();
    let fail_virt_array_parts: Vec<TokenStream> = virt_arrays
        .iter()
        .map(|(_, f)| {
            let ptr_name = quote::format_ident!("{}_ptr", f.name);
            let len_name = quote::format_ident!("{}_len", f.name);
            quote! {
                args.push(self.#ptr_name);
                args.push(self.#len_name);
            }
        })
        .collect();

    // ── build_meta: capture flattened array lengths ──
    let build_meta_fields: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let len_name = quote::format_ident!("{}_len", f.name);
            quote! { #len_name: self.#fname.len(), }
        })
        .collect();

    // ── extract_live: scalars, then flattened array elements, then virt array ptr+len ──
    let extract_scalar_parts: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { values.push(self.#fname as i64); }
        })
        .collect();
    let extract_array_parts: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! {
                for elem in &self.#fname {
                    values.push(*elem as i64);
                }
            }
        })
        .collect();
    let extract_virt_array_parts: Vec<TokenStream> = virt_arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! {
                values.push(self.#fname.as_ptr() as i64);
                values.push(self.#fname.len() as i64);
            }
        })
        .collect();

    // ── create_sym: assign sequential OpRef(0), OpRef(1), ... ──
    let create_sym_scalar_inits: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let value_name = quote::format_ident!("{}_value", f.name);
            quote! {
                let #fname = majit_ir::OpRef(__offset as u32);
                __offset += 1;
                let #value_name = 0i64;
            }
        })
        .collect();
    let create_sym_array_inits: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let len_name = quote::format_ident!("{}_len", f.name);
            let value_name = quote::format_ident!("{}_values", f.name);
            quote! {
                let #fname: Vec<majit_ir::OpRef> = (0..meta.#len_name)
                    .map(|i| {
                        let r = majit_ir::OpRef((__offset + i) as u32);
                        r
                    })
                    .collect();
                let #value_name: Vec<i64> = vec![0; meta.#len_name];
                __offset += meta.#len_name;
            }
        })
        .collect();
    let create_sym_virt_array_inits: Vec<TokenStream> = virt_arrays
        .iter()
        .map(|(_, f)| {
            let ptr_name = quote::format_ident!("{}_ptr", f.name);
            let len_name = quote::format_ident!("{}_len", f.name);
            quote! {
                let #ptr_name = majit_ir::OpRef(__offset as u32);
                __offset += 1;
                let #len_name = majit_ir::OpRef(__offset as u32);
                __offset += 1;
            }
        })
        .collect();
    let create_sym_scalar_names: Vec<&syn::Ident> = scalars.iter().map(|(_, f)| &f.name).collect();
    let create_sym_array_names: Vec<&syn::Ident> = arrays.iter().map(|(_, f)| &f.name).collect();
    let create_sym_scalar_value_names: Vec<syn::Ident> = scalars
        .iter()
        .map(|(_, f)| quote::format_ident!("{}_value", f.name))
        .collect();
    let create_sym_array_value_names: Vec<syn::Ident> = arrays
        .iter()
        .map(|(_, f)| quote::format_ident!("{}_values", f.name))
        .collect();
    let create_sym_virt_array_ptr_names: Vec<syn::Ident> = virt_arrays
        .iter()
        .map(|(_, f)| quote::format_ident!("{}_ptr", f.name))
        .collect();
    let create_sym_virt_array_len_names: Vec<syn::Ident> = virt_arrays
        .iter()
        .map(|(_, f)| quote::format_ident!("{}_len", f.name))
        .collect();

    // ── is_compatible: check flattened array lengths match meta ──
    // Virt arrays always compatible (ptr+len are inputargs, not fixed).
    let compat_checks: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let len_name = quote::format_ident!("{}_len", f.name);
            quote! { && self.#fname.len() == meta.#len_name }
        })
        .collect();

    // ── restore: write values back to state fields ──
    // Virt arrays: restore ptr (ignored, Vec owns it) and skip len.
    // The compiled code writes directly to the heap backing the Vec, so
    // no element-level restore is needed.
    let restore_scalar_parts: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let rust_ty = scalar_rust_type(&f.kind);
            quote! {
                self.#fname = values[__offset] as #rust_ty;
                __offset += 1;
            }
        })
        .collect();
    let restore_array_parts: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! {
                let __arr_len = self.#fname.len();
                for i in 0..__arr_len {
                    self.#fname[i] = values[__offset + i];
                }
                __offset += __arr_len;
            }
        })
        .collect();
    let restore_virt_array_parts: Vec<TokenStream> = virt_arrays
        .iter()
        .map(|_| {
            // Skip the 2 slots (ptr + len) — virt array data lives on heap,
            // already modified in-place by compiled code.
            quote! {
                __offset += 2;
            }
        })
        .collect();
    let initialize_sym_scalar_parts: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let value_name = quote::format_ident!("{}_value", f.name);
            // Cast user's typed field to i64 for the JIT Sym slot.
            quote! {
                sym.#value_name = self.#fname as i64;
            }
        })
        .collect();
    let initialize_sym_array_parts: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let value_name = quote::format_ident!("{}_values", f.name);
            quote! {
                sym.#value_name.clone_from(&self.#fname);
            }
        })
        .collect();

    // ── validate_close: flattened array lengths in sym match meta ──
    // Virt arrays always validate (ptr+len are just OpRefs, not sized).
    let validate_array_checks: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let len_name = quote::format_ident!("{}_len", f.name);
            quote! { && sym.#fname.len() == meta.#len_name }
        })
        .collect();

    quote! {
        /// Compiled loop metadata for state_fields mode: flattened array lengths at trace start.
        #[derive(Clone)]
        #[allow(non_camel_case_types)]
        struct __JitMeta {
            #(#meta_fields)*
        }

        /// Symbolic state during tracing: per-field OpRefs.
        #[allow(non_camel_case_types)]
        struct __JitSym {
            #(#sym_scalar_fields)*
            #(#sym_scalar_value_fields)*
            #(#sym_array_fields)*
            #(#sym_array_value_fields)*
            #(#sym_virt_array_fields)*
            loop_header_pc: usize,
            trace_started: bool,
        }

        impl majit_metainterp::JitCodeSym for __JitSym {
            fn total_slots(&self) -> usize {
                #num_scalars #(#total_slots_array_parts)* + #num_virt_arrays * 2
            }

            fn loop_header_pc(&self) -> usize {
                self.loop_header_pc
            }

            fn state_field_ref(&self, field_idx: usize) -> Option<majit_ir::OpRef> {
                match field_idx {
                    #(#state_field_ref_arms)*
                    _ => None,
                }
            }

            fn set_state_field_ref(&mut self, field_idx: usize, value: majit_ir::OpRef) {
                match field_idx {
                    #(#set_state_field_ref_arms)*
                    _ => {}
                }
            }

            fn state_field_value(&self, field_idx: usize) -> Option<i64> {
                match field_idx {
                    #(#state_field_value_arms)*
                    _ => None,
                }
            }

            fn set_state_field_value(&mut self, field_idx: usize, value: i64) {
                match field_idx {
                    #(#set_state_field_value_arms)*
                    _ => {}
                }
            }

            fn state_array_ref(&self, array_idx: usize, elem_idx: usize) -> Option<majit_ir::OpRef> {
                match array_idx {
                    #(#state_array_ref_arms)*
                    _ => None,
                }
            }

            fn set_state_array_ref(&mut self, array_idx: usize, elem_idx: usize, value: majit_ir::OpRef) {
                match array_idx {
                    #(#set_state_array_ref_arms)*
                    _ => {}
                }
            }

            fn state_array_value(&self, array_idx: usize, elem_idx: usize) -> Option<i64> {
                match array_idx {
                    #(#state_array_value_arms)*
                    _ => None,
                }
            }

            fn set_state_array_value(&mut self, array_idx: usize, elem_idx: usize, value: i64) {
                match array_idx {
                    #(#set_state_array_value_arms)*
                    _ => {}
                }
            }

            fn state_varray_ptr(&self, array_idx: usize) -> Option<majit_ir::OpRef> {
                match array_idx {
                    #(#state_varray_ptr_arms)*
                    _ => None,
                }
            }

            fn state_varray_len(&self, array_idx: usize) -> Option<majit_ir::OpRef> {
                match array_idx {
                    #(#state_varray_len_arms)*
                    _ => None,
                }
            }

            fn fail_args(&self) -> Option<Vec<majit_ir::OpRef>> {
                let mut args = Vec::new();
                #(#fail_scalar_parts)*
                #(#fail_array_parts)*
                #(#fail_virt_array_parts)*
                Some(args)
            }
        }

        impl majit_metainterp::JitState for #state_type {
            type Meta = __JitMeta;
            type Sym = __JitSym;
            type Env = #env_type;

            fn can_trace(&self) -> bool {
                true
            }

            fn build_meta(&self, _header_pc: usize, _program: &#env_type) -> __JitMeta {
                __JitMeta {
                    #(#build_meta_fields)*
                }
            }

            fn extract_live(&self, _meta: &__JitMeta) -> Vec<i64> {
                let mut values = Vec::new();
                #(#extract_scalar_parts)*
                #(#extract_array_parts)*
                #(#extract_virt_array_parts)*
                values
            }

            fn create_sym(meta: &__JitMeta, header_pc: usize) -> __JitSym {
                let mut __offset: usize = 0;
                #(#create_sym_scalar_inits)*
                #(#create_sym_array_inits)*
                #(#create_sym_virt_array_inits)*
                __JitSym {
                    #(#create_sym_scalar_names,)*
                    #(#create_sym_scalar_value_names,)*
                    #(#create_sym_array_names,)*
                    #(#create_sym_array_value_names,)*
                    #(#create_sym_virt_array_ptr_names,)*
                    #(#create_sym_virt_array_len_names,)*
                    loop_header_pc: header_pc,
                    trace_started: false,
                }
            }

            fn initialize_sym(&self, sym: &mut __JitSym, _meta: &__JitMeta) {
                #(#initialize_sym_scalar_parts)*
                #(#initialize_sym_array_parts)*
            }

            fn is_compatible(&self, meta: &__JitMeta) -> bool {
                true #(#compat_checks)*
            }

            fn restore(&mut self, _meta: &__JitMeta, values: &[i64]) {
                let mut __offset: usize = 0;
                #(#restore_scalar_parts)*
                #(#restore_array_parts)*
                #(#restore_virt_array_parts)*
            }

            fn collect_jump_args(sym: &__JitSym) -> Vec<majit_ir::OpRef> {
                let mut args = Vec::new();
                #(#collect_scalar_parts)*
                #(#collect_array_parts)*
                #(#collect_virt_array_parts)*
                args
            }

            fn validate_close(sym: &__JitSym, meta: &__JitMeta) -> bool {
                true #(#validate_array_checks)*
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::generate_jit_state;
    use crate::jit_interp::JitInterpConfig;

    fn render(config: proc_macro2::TokenStream) -> String {
        let parsed = syn::parse2::<JitInterpConfig>(config).expect("valid jit_interp config");
        generate_jit_state(&parsed).to_string()
    }
}
