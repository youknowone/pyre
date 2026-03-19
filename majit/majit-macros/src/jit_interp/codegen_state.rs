//! Generate JitState types (Meta, Sym) and impl from the macro configuration.

use proc_macro2::TokenStream;
use quote::quote;
use syn::Expr;

use super::{JitInterpConfig, StateFieldKind};

/// Extract the member field name from a field access expression.
/// e.g., `state.storage` → `storage`, `state.selected` → `selected`.
fn extract_field_member(expr: &Expr) -> &syn::Member {
    if let Expr::Field(field) = expr {
        &field.member
    } else {
        panic!("expected field access expression like `state.field`, got other expression form")
    }
}

/// Generate the JitState types and implementation.
pub fn generate_jit_state(config: &JitInterpConfig) -> TokenStream {
    if config.state_fields.is_some() {
        return generate_state_fields_jit_state(config);
    }
    generate_storage_pool_jit_state(config)
}

/// Generate JitState types for storage-pool mode (existing behaviour).
fn generate_storage_pool_jit_state(config: &JitInterpConfig) -> TokenStream {
    let state_type = &config.state_type;
    let env_type = &config.env_type;
    let pool_field = extract_field_member(&config.storage.pool);
    let sel_field = extract_field_member(&config.storage.selector);
    let untraceable = &config.storage.untraceable;
    let scan_fn = &config.storage.scan_fn;
    let can_trace_guard = &config.storage.can_trace_guard;
    let virtualizable = config.storage.virtualizable;
    let extra_guard = if let Some(method) = can_trace_guard {
        quote! { && self.#pool_field.#method() }
    } else {
        quote! {}
    };
    // ── Frame virtualizable (VirtualizableDecl) code generation ──
    let vable_info_fn = config.virtualizable_decl.as_ref().map(|decl| {
        let token_offset = &decl.token_offset;
        let field_adds: Vec<TokenStream> = decl
            .fields
            .iter()
            .map(|f| {
                let name = f.name.to_string();
                let offset = &f.offset;
                let tp = match f.field_type.to_string().as_str() {
                    "ref" => quote! { majit_ir::Type::Ref },
                    "float" => quote! { majit_ir::Type::Float },
                    _ => quote! { majit_ir::Type::Int },
                };
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
                let offset = &a.offset;
                let tp = match a.item_type.to_string().as_str() {
                    "ref" => quote! { majit_ir::Type::Ref },
                    "float" => quote! { majit_ir::Type::Float },
                    _ => quote! { majit_ir::Type::Int },
                };
                quote! {
                    __info.add_array_field(#name, #tp, #offset);
                }
            })
            .collect();
        quote! {
            /// Build VirtualizableInfo for the interpreter frame.
            ///
            /// Auto-generated from `virtualizable_fields` declaration.
            /// RPython equivalent: VirtualizableInfo.__init__() in virtualizable.py.
            #[allow(non_snake_case)]
            fn __build_virtualizable_info() -> majit_meta::virtualizable::VirtualizableInfo {
                let mut __info = majit_meta::virtualizable::VirtualizableInfo::new(#token_offset);
                #(#field_adds)*
                #(#array_adds)*
                __info
            }
        }
    });

    quote! {
        /// Compiled loop metadata: which storages and how many slots each had.
        #[derive(Clone)]
        #[allow(non_camel_case_types)]
        struct __JitMeta {
            storage_layout: Vec<(usize, usize)>,
            initial_selected: usize,
        }

        /// Symbolic state during tracing: per-storage symbolic stacks.
        #[allow(non_camel_case_types)]
        struct __JitSym {
            stacks: std::collections::HashMap<usize, majit_meta::SymbolicStack>,
            current_selected: usize,
            current_selected_value: Option<majit_ir::OpRef>,
            storage_layout: Vec<(usize, usize)>,
            loop_header_pc: usize,
            /// RPython parity: initial selected at trace header.
            header_selected: usize,
            /// Virtualizable: data pointer OpRefs (storage_idx → OpRef).
            vable_array_refs: std::collections::HashMap<usize, majit_ir::OpRef>,
            /// Virtualizable: length OpRefs (storage_idx → OpRef).
            vable_len_refs: std::collections::HashMap<usize, majit_ir::OpRef>,
            /// Number of storages from the original meta layout.
            /// Extra storages added during tracing are ignored in Jump args.
            meta_storage_count: usize,
            /// Virtualizable: preamble already emitted for this trace.
            preamble_done: bool,
            /// Virtualizable: actual depths loaded by preamble (storage_idx → depth).
            preamble_depths: std::collections::HashMap<usize, usize>,
        }

        #[allow(non_camel_case_types)]
        impl __JitSym {
            fn total_slots(&self) -> usize {
                self.storage_layout.iter().map(|(_, n)| *n).sum()
            }
        }

        impl majit_meta::JitCodeSym for __JitSym {
            fn current_selected(&self) -> usize {
                self.current_selected
            }

            fn current_selected_value(&self) -> Option<majit_ir::OpRef> {
                self.current_selected_value
            }

            fn set_current_selected(&mut self, selected: usize) {
                self.current_selected = selected;
            }

            fn set_current_selected_value(&mut self, selected: usize, value: majit_ir::OpRef) {
                self.current_selected = selected;
                self.current_selected_value = Some(value);
            }

            fn stack(&self, selected: usize) -> Option<&majit_meta::SymbolicStack> {
                self.stacks.get(&selected)
            }

            fn stack_mut(&mut self, selected: usize) -> Option<&mut majit_meta::SymbolicStack> {
                self.stacks.get_mut(&selected)
            }

            fn total_slots(&self) -> usize {
                self.total_slots()
            }

            fn loop_header_pc(&self) -> usize {
                self.loop_header_pc
            }

            fn header_selected(&self) -> usize {
                self.header_selected
            }

            fn ensure_stack(&mut self, selected: usize, offset: usize, len: usize) {
                self.stacks
                    .entry(selected)
                    .or_insert_with(|| majit_meta::SymbolicStack::from_input_args(offset, len));
                if !self.storage_layout.iter().any(|&(s, _)| s == selected) {
                    self.storage_layout.push((selected, len));
                }
            }

            fn fail_args(&self) -> Option<Vec<majit_ir::OpRef>> {
                if #virtualizable {
                    // Virtualizable: [ptr_0, len_0, ptr_1, len_1, ...]
                    let mut args = Vec::new();
                    for &(sidx, _) in self.storage_layout.iter().take(self.meta_storage_count) {
                        if let (Some(ptr), Some(len)) = (
                            self.vable_array_refs.get(&sidx).copied(),
                            self.vable_len_refs.get(&sidx).copied(),
                        ) {
                            args.push(ptr);
                            args.push(len);
                        }
                    }
                    return Some(args);
                }
                let mut args = Vec::new();
                for &(sidx, _) in &self.storage_layout {
                    args.extend(self.stacks[&sidx].to_jump_args());
                }
                Some(args)
            }

            fn fail_storage_lengths(&self) -> Option<Vec<usize>> {
                if #virtualizable { return None; }
                Some(
                    self.storage_layout
                        .iter()
                        .map(|&(sidx, _)| self.stacks[&sidx].len())
                        .collect(),
                )
            }

            fn is_virtualizable_storage(&self) -> bool {
                !self.vable_array_refs.is_empty()
            }

            fn vable_array_ref(&self, storage_idx: usize) -> Option<majit_ir::OpRef> {
                self.vable_array_refs.get(&storage_idx).copied()
            }

            fn vable_len_ref(&self, storage_idx: usize) -> Option<majit_ir::OpRef> {
                self.vable_len_refs.get(&storage_idx).copied()
            }

            fn set_vable_array_ref(&mut self, storage_idx: usize, opref: majit_ir::OpRef) {
                self.vable_array_refs.insert(storage_idx, opref);
            }

            fn set_vable_len_ref(&mut self, storage_idx: usize, opref: majit_ir::OpRef) {
                self.vable_len_refs.insert(storage_idx, opref);
            }

            fn init_vable_storage(&mut self, storage_idx: usize, array_ref: majit_ir::OpRef, len_ref: majit_ir::OpRef) {
                self.vable_array_refs.insert(storage_idx, array_ref);
                self.vable_len_refs.insert(storage_idx, len_ref);
            }
        }

        impl majit_meta::JitState for #state_type {
            type Meta = __JitMeta;
            type Sym = __JitSym;
            type Env = #env_type;

            fn can_trace(&self) -> bool {
                if !(true #( && self.#sel_field != #untraceable )*) {
                    return false;
                }
                true #extra_guard
            }

            fn build_meta(&self, header_pc: usize, program: &#env_type) -> __JitMeta {
                let storages = #scan_fn(program, header_pc, self.#sel_field);
                let layout = if #virtualizable {
                    // Virtualizable: depth=0 for all. Preamble loads from heap.
                    storages.iter().map(|&s| (s, 0usize)).collect()
                } else {
                    storages.iter()
                        .map(|&s| (s, self.#pool_field.get(s).len()))
                        .collect()
                };
                __JitMeta {
                    storage_layout: layout,
                    initial_selected: self.#sel_field,
                }
            }

            fn extract_live(&self, meta: &__JitMeta) -> Vec<i64> {
                if #virtualizable {
                    // Virtualizable: [ptr_0, len_0, ptr_1, len_1, ...]
                    let mut values = Vec::new();
                    for &(sidx, _) in &meta.storage_layout {
                        let store = self.#pool_field.get(sidx);
                        values.push(store.data_ptr() as i64);
                        values.push(store.len() as i64);
                    }
                    values
                } else {
                    let mut values = Vec::new();
                    for &(sidx, num_slots) in &meta.storage_layout {
                        let store = self.#pool_field.get(sidx);
                        for i in 0..num_slots {
                            values.push(store.peek_at(i));
                        }
                    }
                    values
                }
            }

            fn create_sym(meta: &__JitMeta, header_pc: usize) -> __JitSym {
                let mut stacks = std::collections::HashMap::new();
                let mut vable_len_refs = std::collections::HashMap::new();
                let mut vable_array_refs = std::collections::HashMap::new();
                if #virtualizable {
                    // InputArgs = [ptr_0, len_0, ptr_1, len_1, ...]
                    for (i, &(sidx, _)) in meta.storage_layout.iter().enumerate() {
                        vable_array_refs.insert(sidx, majit_ir::OpRef((i * 2) as u32));
                        vable_len_refs.insert(sidx, majit_ir::OpRef((i * 2 + 1) as u32));
                        stacks.insert(sidx, majit_meta::SymbolicStack::new());
                    }
                } else {
                    let mut offset = 0;
                    for &(sidx, num_slots) in &meta.storage_layout {
                        stacks.insert(
                            sidx,
                            majit_meta::SymbolicStack::from_input_args(offset, num_slots),
                        );
                        offset += num_slots;
                    }
                }
                __JitSym {
                    stacks,
                    current_selected: meta.initial_selected,
                    current_selected_value: None,
                    storage_layout: meta.storage_layout.clone(),
                    loop_header_pc: header_pc,
                    header_selected: meta.initial_selected,
                    vable_array_refs,
                    vable_len_refs,
                    meta_storage_count: meta.storage_layout.len(),
                    preamble_done: false,
                    preamble_depths: std::collections::HashMap::new(),
                }
            }

            fn is_compatible(&self, meta: &__JitMeta) -> bool {
                if #virtualizable {
                    // Virtualizable: only check selected. Depths don't matter
                    // because preamble re-loads from heap each iteration.
                    meta.initial_selected == self.#sel_field
                } else {
                    meta.initial_selected == self.#sel_field
                        && meta.storage_layout.iter().all(|&(sidx, expected_len)| {
                            self.#pool_field.get(sidx).len() == expected_len
                        })
                }
            }

            fn restore(&mut self, meta: &__JitMeta, values: &[i64]) {
                if #virtualizable {
                    // Virtualizable: values = [ptr_0, len_0, ptr_1, len_1, ...]
                    for (i, &(sidx, _)) in meta.storage_layout.iter().enumerate() {
                        if let Some(&len_val) = values.get(i * 2 + 1) {
                            self.#pool_field.get_mut(sidx).force_len(len_val as usize);
                        }
                    }
                    self.#sel_field = meta.initial_selected;
                } else {
                    let mut offset = 0;
                    for &(sidx, num_slots) in &meta.storage_layout {
                        let end = offset + num_slots;
                        {
                            let store = self.#pool_field.get_mut(sidx);
                            store.clear();
                            for &value in &values[offset..end] {
                                store.push(value);
                            }
                        }
                        offset = end;
                    }
                    self.#sel_field = values
                        .get(offset)
                        .copied()
                        .map(|value| value as usize)
                        .unwrap_or(meta.initial_selected);
                }
            }

            fn collect_jump_args(sym: &__JitSym) -> Vec<majit_ir::OpRef> {
                if #virtualizable {
                    // Virtualizable: [ptr_0, len_0, ptr_1, len_1, ...]
                    // sync_virtualizable_to_heap already updated len_refs.
                    let mut args = Vec::new();
                    for &(sidx, _) in sym.storage_layout.iter().take(sym.meta_storage_count) {
                        if let (Some(ptr), Some(len)) = (
                            sym.vable_array_refs.get(&sidx).copied(),
                            sym.vable_len_refs.get(&sidx).copied(),
                        ) {
                            args.push(ptr);
                            args.push(len);
                        }
                    }
                    args
                } else {
                    let mut args = Vec::new();
                    for &(sidx, _) in &sym.storage_layout {
                        args.extend(sym.stacks[&sidx].to_jump_args());
                    }
                    args
                }
            }

            #vable_info_fn

            fn validate_close(sym: &__JitSym, meta: &__JitMeta) -> bool {
                if #virtualizable {
                    // Check selected + preamble depths match + extra storages empty.
                    sym.current_selected == meta.initial_selected
                        && sym.preamble_depths.iter().all(|(sidx, &depth)| {
                            sym.stacks.get(sidx).is_some_and(|s| s.len() == depth)
                        })
                        && sym.stacks.iter().all(|(sidx, stack)| {
                            sym.preamble_depths.contains_key(sidx) || stack.len() == 0
                        })
                } else {
                    sym.current_selected == meta.initial_selected
                        && sym.storage_layout.len() == meta.storage_layout.len()
                        && sym
                            .storage_layout
                            .iter()
                            .zip(meta.storage_layout.iter())
                            .all(|(&(sym_idx, _), &(meta_idx, _))| sym_idx == meta_idx)
                        && meta.storage_layout.iter().all(|&(sidx, initial_depth)| {
                            sym.stacks
                                .get(&sidx)
                                .is_some_and(|stack| stack.len() == initial_depth)
                        })
                }
            }
        }
    }
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
        .filter_map(|f| {
            let ty = match &f.kind {
                StateFieldKind::Scalar(tp)
                | StateFieldKind::Array(tp)
                | StateFieldKind::VirtArray(tp) => tp.to_string(),
            };
            match ty.as_str() {
                "int" | "ref" | "float" => None,
                _ => Some(format!("{}: {}", f.name, ty)),
            }
        })
        .collect();
    if !unsupported_fields.is_empty() {
        let message = format!(
            "state_fields supports int/ref/float and [int]/[ref]/[float]; unsupported: {}",
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
        .filter(|(_, f)| matches!(f.kind, StateFieldKind::Scalar(_)))
        .collect();
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
    let sym_array_fields: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { #fname: Vec<majit_ir::OpRef>, }
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
            quote! { #idx_lit => Some(self.#fname), }
        })
        .collect();
    let set_state_field_ref_arms: Vec<TokenStream> = scalars
        .iter()
        .enumerate()
        .map(|(idx, (_, f))| {
            let fname = &f.name;
            let idx_lit = idx;
            quote! { #idx_lit => { self.#fname = value; } }
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
            quote! {
                let #fname = majit_ir::OpRef(__offset as u32);
                __offset += 1;
            }
        })
        .collect();
    let create_sym_array_inits: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let len_name = quote::format_ident!("{}_len", f.name);
            quote! {
                let #fname: Vec<majit_ir::OpRef> = (0..meta.#len_name)
                    .map(|i| {
                        let r = majit_ir::OpRef((__offset + i) as u32);
                        r
                    })
                    .collect();
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
            quote! {
                self.#fname = values[__offset];
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
        /// Dummy storage pool for state_fields mode.
        /// The trace function signature requires a storage reference, but
        /// state_fields mode uses load/store_state_field ops instead.
        #[allow(non_camel_case_types)]
        struct __DummyPool;

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
            #(#sym_array_fields)*
            #(#sym_virt_array_fields)*
            loop_header_pc: usize,
        }

        impl majit_meta::JitCodeSym for __JitSym {
            fn current_selected(&self) -> usize {
                0
            }

            fn current_selected_value(&self) -> Option<majit_ir::OpRef> {
                None
            }

            fn set_current_selected(&mut self, _selected: usize) {}

            fn set_current_selected_value(&mut self, _selected: usize, _value: majit_ir::OpRef) {}

            fn stack(&self, _selected: usize) -> Option<&majit_meta::SymbolicStack> {
                None
            }

            fn stack_mut(&mut self, _selected: usize) -> Option<&mut majit_meta::SymbolicStack> {
                None
            }

            fn total_slots(&self) -> usize {
                #num_scalars #(#total_slots_array_parts)* + #num_virt_arrays * 2
            }

            fn loop_header_pc(&self) -> usize {
                self.loop_header_pc
            }

            fn ensure_stack(&mut self, _selected: usize, _offset: usize, _len: usize) {}

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

        impl majit_meta::JitState for #state_type {
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
                    #(#create_sym_array_names,)*
                    #(#create_sym_virt_array_ptr_names,)*
                    #(#create_sym_virt_array_len_names,)*
                    loop_header_pc: header_pc,
                }
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
