//! Generate JitState types (Meta, Sym) and impl from the macro configuration.

use proc_macro2::TokenStream;
use quote::quote;
use syn::Expr;

use super::{JitInterpConfig, StateFieldKind, VableArrayLayoutDecl};

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
    let storage = config
        .storage
        .as_ref()
        .expect("storage config required in storage mode");
    let state_type = &config.state_type;
    let env_type = &config.env_type;
    let pool_field = extract_field_member(&storage.pool);
    let pool_ref_field = storage.pool_ref.as_ref().map(extract_field_member);
    let sel_field = extract_field_member(&storage.selector);
    let selected_ref_field = storage.selected_ref.as_ref().map(extract_field_member);
    let stacksize_field = storage.stacksize.as_ref().map(extract_field_member);
    let untraceable = &storage.untraceable;
    let scan_fn = &storage.scan_fn;
    let can_trace_guard = &storage.can_trace_guard;
    let extra_guard = if let Some(method) = can_trace_guard {
        quote! { && self.#pool_field.#method() }
    } else {
        quote! {}
    };
    let vable_info_fn = config.virtualizable_decl.as_ref().map(|decl| {
        let token_offset = &decl.token_offset;
        let field_adds: Vec<TokenStream> = decl
            .fields
            .iter()
            .map(|f| {
                let name = f.name.to_string();
                let offset = &f.offset;
                let tp = if f.field_type == "ref" {
                    quote! { majit_ir::Type::Ref }
                } else if f.field_type == "float" {
                    quote! { majit_ir::Type::Float }
                } else {
                    quote! { majit_ir::Type::Int }
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
                let tp = if a.item_type == "ref" {
                    quote! { majit_ir::Type::Ref }
                } else if a.item_type == "float" {
                    quote! { majit_ir::Type::Float }
                } else {
                    quote! { majit_ir::Type::Int }
                };
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
            fn __build_virtualizable_info() -> Option<majit_meta::virtualizable::VirtualizableInfo> {
                let mut __info = majit_meta::virtualizable::VirtualizableInfo::new(#token_offset);
                #(#field_adds)*
                #(#array_adds)*
                Some(__info)
            }
        }
    });
    let vable_state_hooks = config.virtualizable_decl.as_ref().map(|decl| {
        let virtualizable_name = decl.var_name.to_string();
        quote! {
            fn virtualizable_heap_ptr(
                &self,
                _meta: &Self::Meta,
                virtualizable: &str,
                _info: &majit_meta::virtualizable::VirtualizableInfo,
            ) -> Option<*mut u8> {
                if virtualizable == #virtualizable_name {
                    Some((&self.#pool_field as *const _ as *mut u8).cast::<u8>())
                } else {
                    None
                }
            }

            fn virtualizable_array_lengths(
                &self,
                _meta: &Self::Meta,
                virtualizable: &str,
                info: &majit_meta::virtualizable::VirtualizableInfo,
            ) -> Option<Vec<usize>> {
                if virtualizable != #virtualizable_name {
                    return None;
                }
                if info.can_read_all_array_lengths_from_heap() {
                    let obj_ptr = (&self.#pool_field as *const _).cast::<u8>();
                    Some(unsafe { info.read_array_lengths_from_heap(obj_ptr) })
                } else {
                    None
                }
            }

            fn import_virtualizable_boxes(
                &mut self,
                _meta: &Self::Meta,
                _virtualizable: &str,
                _info: &majit_meta::virtualizable::VirtualizableInfo,
                _static_boxes: &[i64],
                _array_boxes: &[Vec<i64>],
            ) -> bool {
                true
            }

            fn export_virtualizable_boxes(
                &self,
                meta: &Self::Meta,
                virtualizable: &str,
                info: &majit_meta::virtualizable::VirtualizableInfo,
            ) -> Option<(Vec<i64>, Vec<Vec<i64>>)> {
                if virtualizable != #virtualizable_name {
                    return None;
                }
                let obj_ptr = (&self.#pool_field as *const _).cast::<u8>();
                let lengths = if info.can_read_all_array_lengths_from_heap() {
                    unsafe { info.read_array_lengths_from_heap(obj_ptr) }
                } else {
                    self.virtualizable_array_lengths(meta, virtualizable, info)?
                };
                Some(unsafe { info.read_all_boxes(obj_ptr, &lengths) })
            }

            fn sync_virtualizable_before_residual_call(
                &self,
                ctx: &mut majit_meta::trace_ctx::TraceCtx,
            ) {
                let Some(info) = Self::__build_virtualizable_info() else {
                    return;
                };
                let Some(vable_ref) = ctx.standard_virtualizable_box() else {
                    return;
                };
                let obj_ptr = (&self.#pool_field as *const _ as *mut u8).cast::<u8>();
                unsafe {
                    info.tracing_before_residual_call(obj_ptr);
                }
                let force_token = ctx.force_token();
                ctx.vable_setfield_descr(vable_ref, force_token, info.token_field_descr());
            }

            fn sync_virtualizable_after_residual_call(
                &self,
                _ctx: &mut majit_meta::trace_ctx::TraceCtx,
            ) -> majit_meta::jit_state::ResidualVirtualizableSync {
                let Some(info) = Self::__build_virtualizable_info() else {
                    return majit_meta::jit_state::ResidualVirtualizableSync::default();
                };
                let obj_ptr = (&self.#pool_field as *const _ as *mut u8).cast::<u8>();
                let forced = unsafe { info.tracing_after_residual_call(obj_ptr) };
                majit_meta::jit_state::ResidualVirtualizableSync {
                    updated_fields: Vec::new(),
                    forced,
                }
            }
        }
    });
    let compact_encode = storage
        .compact_encode
        .as_ref()
        .map(|path| quote! { #path(ctx, value) })
        .unwrap_or_else(|| quote! { value });
    let compact_decode = storage
        .compact_decode
        .as_ref()
        .map(|path| quote! { #path(ctx, raw) })
        .unwrap_or_else(|| quote! { raw });
    let compact_bounds = match (&storage.compact_min, &storage.compact_max) {
        (Some(min), Some(max)) => quote! { Some(((#min) as i64, (#max) as i64)) },
        _ => quote! { None },
    };

    if storage.compact_live {
        if let (Some(ptrs_offset), Some(lengths_offset), Some(caps_offset)) = (
            &storage.compact_ptrs_offset,
            &storage.compact_lengths_offset,
            &storage.compact_caps_offset,
        ) {
            return quote! {
                /// Compiled loop metadata: traced storages in stable order.
                #[derive(Clone)]
                #[allow(non_camel_case_types)]
                struct __JitMeta {
                    storage_layout: Vec<(usize, usize)>,
                    initial_selected: usize,
                }

                /// Symbolic state during tracing: compact storage is reloaded from the pool object.
                #[allow(non_camel_case_types)]
                struct __JitSym {
                    pool_ref: majit_ir::OpRef,
                    storage_refs: std::collections::HashMap<
                        usize,
                        (
                            majit_ir::OpRef,
                            majit_ir::OpRef,
                            majit_ir::OpRef,
                            majit_ir::OpRef,
                            std::collections::BTreeMap<usize, majit_ir::OpRef>,
                        ),
                    >,
                    pending_storage_refs: std::collections::HashMap<
                        usize,
                        (
                            majit_ir::OpRef,
                            Option<usize>,
                            std::collections::BTreeMap<usize, majit_ir::OpRef>,
                        ),
                    >,
                    committed_selected: usize,
                    committed_selected_value: Option<majit_ir::OpRef>,
                    pending_selected: Option<usize>,
                    pending_selected_value: Option<majit_ir::OpRef>,
                    storage_layout: Vec<(usize, usize)>,
                    loop_header_pc: usize,
                    header_selected: usize,
                    trace_started: bool,
                }

                impl majit_meta::JitCodeSym for __JitSym {
                    fn current_selected(&self) -> usize {
                        self.pending_selected.unwrap_or(self.committed_selected)
                    }

                    fn current_selected_value(&self) -> Option<majit_ir::OpRef> {
                        self.pending_selected_value.or(self.committed_selected_value)
                    }

                    fn guard_selected(&self) -> usize {
                        self.committed_selected
                    }

                    fn guard_selected_value(&self) -> Option<majit_ir::OpRef> {
                        self.committed_selected_value
                    }

                    fn selected_in_fail_args_prefix(&self) -> bool {
                        true
                    }

                    fn close_requires_header_selected(&self) -> bool {
                        false
                    }

                    fn set_current_selected(&mut self, selected: usize) {
                        self.pending_selected = Some(selected);
                        self.pending_selected_value = None;
                    }

                    fn set_current_selected_value(&mut self, selected: usize, value: majit_ir::OpRef) {
                        self.pending_selected = Some(selected);
                        self.pending_selected_value = Some(value);
                    }

                    fn begin_portal_op(&mut self, _pc: usize) {
                        self.pending_storage_refs.clear();
                        self.pending_selected = None;
                        self.pending_selected_value = None;
                    }

                    fn commit_portal_op(&mut self) {
                        for (selected, (len, truncate_to, slots)) in self.pending_storage_refs.drain() {
                            let Some((_, _, committed_len, _, committed_slots)) =
                                self.storage_refs.get_mut(&selected)
                            else {
                                continue;
                            };
                            *committed_len = len;
                            if let Some(truncate_to) = truncate_to {
                                committed_slots.retain(|&index, _| index < truncate_to);
                            }
                            committed_slots.extend(slots);
                        }
                        if let Some(selected) = self.pending_selected.take() {
                            self.committed_selected = selected;
                            self.committed_selected_value = self.pending_selected_value.take();
                        } else {
                            self.pending_selected_value = None;
                        }
                    }

                    fn abort_portal_op(&mut self) {
                        self.pending_storage_refs.clear();
                        self.pending_selected = None;
                        self.pending_selected_value = None;
                    }

                    fn stack(&self, _selected: usize) -> Option<&majit_meta::SymbolicStack> {
                        None
                    }

                    fn stack_mut(&mut self, _selected: usize) -> Option<&mut majit_meta::SymbolicStack> {
                        None
                    }

                    fn total_slots(&self) -> usize {
                        2
                    }

                    fn loop_header_pc(&self) -> usize {
                        self.loop_header_pc
                    }

                    fn header_selected(&self) -> usize {
                        self.header_selected
                    }

                    fn ensure_stack(&mut self, _selected: usize, _offset: usize, _len: usize) {}

                    fn fail_args(&self) -> Option<Vec<majit_ir::OpRef>> {
                        let selected = self
                            .guard_selected_value()
                            .unwrap_or_else(|| majit_ir::OpRef::NONE);
                        Some(vec![self.pool_ref, selected])
                    }

                    fn fail_args_with_ctx(
                        &mut self,
                        ctx: &mut majit_meta::TraceCtx,
                    ) -> Option<Vec<majit_ir::OpRef>> {
                        // Guard failures must start with the same live state
                        // as the parent loop's inputargs so attached bridges
                        // can reuse the guard payload as bridge inputs.
                        let selected = self
                            .guard_selected_value()
                            .unwrap_or_else(|| ctx.const_int(self.guard_selected() as i64));
                        let mut args = vec![self.pool_ref, selected];
                        for &(selected, _) in &self.storage_layout {
                            if let Some((_, _, len, _, slots)) = self.storage_refs.get(&selected) {
                                args.push(*len);
                                args.push(ctx.const_int(slots.len() as i64));
                                for (&index, &raw) in slots {
                                    args.push(ctx.const_int(index as i64));
                                    args.push(raw);
                                }
                            } else {
                                args.push(ctx.const_int(-1));
                                args.push(ctx.const_int(0));
                            }
                        }
                        Some(args)
                    }

                    fn compact_storage_ptr(&self, selected: usize) -> Option<majit_ir::OpRef> {
                        self.storage_refs
                            .get(&selected)
                            .map(|(_, ptr, _, _, _)| *ptr)
                    }

                    fn compact_storage_len(&self, selected: usize) -> Option<majit_ir::OpRef> {
                        if let Some((len, _, _)) = self.pending_storage_refs.get(&selected) {
                            return Some(*len);
                        }
                        self.storage_refs
                            .get(&selected)
                            .map(|(_, _, len, _, _)| *len)
                    }

                    fn compact_storage_cap(&self, selected: usize) -> Option<majit_ir::OpRef> {
                        self.storage_refs
                            .get(&selected)
                            .map(|(_, _, _, cap, _)| *cap)
                    }

                    fn set_compact_storage_len(&mut self, selected: usize, value: majit_ir::OpRef) {
                        let truncate_to = self
                            .pending_storage_refs
                            .get(&selected)
                            .and_then(|(_, truncate_to, _)| *truncate_to);
                        let slots = self
                            .pending_storage_refs
                            .remove(&selected)
                            .map(|(_, _, slots)| slots)
                            .unwrap_or_default();
                        if self.storage_refs.contains_key(&selected) {
                            self.pending_storage_refs
                                .insert(selected, (value, truncate_to, slots));
                        }
                    }

                    fn compact_storage_slot_raw(
                        &self,
                        selected: usize,
                        index: usize,
                    ) -> Option<majit_ir::OpRef> {
                        if let Some((_, _, slots)) = self.pending_storage_refs.get(&selected) {
                            if let Some(raw) = slots.get(&index) {
                                return Some(*raw);
                            }
                        }
                        self.storage_refs
                            .get(&selected)
                            .and_then(|(_, _, _, _, slots)| slots.get(&index).copied())
                    }

                    fn set_compact_storage_slot_raw(
                        &mut self,
                        selected: usize,
                        index: usize,
                        raw: majit_ir::OpRef,
                    ) {
                        let entry = self.pending_storage_refs.entry(selected).or_insert_with(|| {
                            let len = self
                                .storage_refs
                                .get(&selected)
                                .map(|(_, _, len, _, _)| *len)
                                .expect("missing compact storage");
                            (len, None, std::collections::BTreeMap::new())
                        });
                        entry.2.insert(index, raw);
                    }

                    fn truncate_compact_storage_slots(&mut self, selected: usize, len: usize) {
                        let entry = self.pending_storage_refs.entry(selected).or_insert_with(|| {
                            let current_len = self
                                .storage_refs
                                .get(&selected)
                                .map(|(_, _, len, _, _)| *len)
                                .expect("missing compact storage");
                            (current_len, None, std::collections::BTreeMap::new())
                        });
                        entry.1 = Some(entry.1.map_or(len, |current| current.min(len)));
                        entry.2.retain(|&index, _| index < len);
                    }

                    fn ensure_compact_storage_loaded(
                        &mut self,
                        ctx: &mut majit_meta::TraceCtx,
                        selected: usize,
                    ) -> Option<(majit_ir::OpRef, majit_ir::OpRef, majit_ir::OpRef)> {
                        if false #( || selected == #untraceable )* {
                            return None;
                        }
                        if let Some((_, ptr, _, cap, _)) = self.storage_refs.get(&selected) {
                            let len = self.compact_storage_len(selected)?;
                            return Some((*ptr, len, *cap));
                        }

                        let ptr_offset = ((#ptrs_offset) as usize) + selected * 8;
                        let len_offset = ((#lengths_offset) as usize) + selected * 8;
                        let cap_offset = ((#caps_offset) as usize) + selected * 8;
                        let ptr_descr: majit_ir::DescrRef = std::sync::Arc::new(
                            majit_ir::descr::SimpleFieldDescr::new(
                                ((ptr_offset as u32) << 2) | 0,
                                ptr_offset,
                                8,
                                majit_ir::Type::Int,
                                false,
                            )
                            .with_signed(false),
                        );
                        let len_descr: majit_ir::DescrRef = std::sync::Arc::new(
                            majit_ir::descr::SimpleFieldDescr::new(
                                ((len_offset as u32) << 2) | 1,
                                len_offset,
                                8,
                                majit_ir::Type::Int,
                                false,
                            )
                            .with_signed(false),
                        );
                        let cap_descr: majit_ir::DescrRef = std::sync::Arc::new(
                            majit_ir::descr::SimpleFieldDescr::new(
                                ((cap_offset as u32) << 2) | 2,
                                cap_offset,
                                8,
                                majit_ir::Type::Int,
                                false,
                            )
                            .with_signed(false),
                        );

                        let ptr = ctx.record_op_with_descr(
                            majit_ir::OpCode::GetfieldRawI,
                            &[self.pool_ref],
                            ptr_descr,
                        );
                        let len = ctx.record_op_with_descr(
                            majit_ir::OpCode::GetfieldRawI,
                            &[self.pool_ref],
                            len_descr,
                        );
                        let cap = ctx.record_op_with_descr(
                            majit_ir::OpCode::GetfieldRawI,
                            &[self.pool_ref],
                            cap_descr,
                        );
                        let selected_ref = ctx.const_int(selected as i64);
                        self.storage_refs
                            .insert(
                                selected,
                                (
                                    selected_ref,
                                    ptr,
                                    len,
                                    cap,
                                    std::collections::BTreeMap::new(),
                                ),
                            );
                        Some((ptr, len, cap))
                    }

                    fn compact_storage_writeback_len(
                        &mut self,
                        ctx: &mut majit_meta::TraceCtx,
                        selected: usize,
                        new_len: majit_ir::OpRef,
                    ) {
                        let len_offset = ((#lengths_offset) as usize) + selected * 8;
                        let len_descr: majit_ir::DescrRef = std::sync::Arc::new(
                            majit_ir::descr::SimpleFieldDescr::new(
                                ((len_offset as u32) << 2) | 1,
                                len_offset,
                                8,
                                majit_ir::Type::Int,
                                false,
                            )
                            .with_signed(false),
                        );
                        ctx.record_op_with_descr(
                            majit_ir::OpCode::SetfieldRaw,
                            &[self.pool_ref, new_len],
                            len_descr,
                        );
                    }

                    fn compact_storage_bounds(&self) -> Option<(i64, i64)> {
                        #compact_bounds
                    }

                    fn compact_storage_decode(
                        &self,
                        ctx: &mut majit_meta::TraceCtx,
                        raw: majit_ir::OpRef,
                    ) -> majit_ir::OpRef {
                        #compact_decode
                    }

                    fn compact_storage_encode(
                        &self,
                        ctx: &mut majit_meta::TraceCtx,
                        value: majit_ir::OpRef,
                    ) -> majit_ir::OpRef {
                        #compact_encode
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
                        let layout = storages.into_iter().map(|sidx| (sidx, 0)).collect();
                        __JitMeta {
                            storage_layout: layout,
                            initial_selected: self.#sel_field,
                        }
                    }

                    fn extract_live(&self, _meta: &__JitMeta) -> Vec<i64> {
                        vec![
                            (&self.#pool_field as *const _ as usize) as i64,
                            self.#sel_field as i64,
                        ]
                    }

                    fn create_sym(meta: &__JitMeta, header_pc: usize) -> __JitSym {
                        __JitSym {
                            pool_ref: majit_ir::OpRef(0),
                            storage_refs: std::collections::HashMap::new(),
                            pending_storage_refs: std::collections::HashMap::new(),
                            committed_selected: meta.initial_selected,
                            committed_selected_value: Some(majit_ir::OpRef(1)),
                            pending_selected: None,
                            pending_selected_value: None,
                            storage_layout: meta.storage_layout.clone(),
                            loop_header_pc: header_pc,
                            header_selected: meta.initial_selected,
                            trace_started: false,
                        }
                    }

                    fn is_compatible(&self, meta: &__JitMeta) -> bool {
                        !meta.storage_layout.is_empty()
                    }

                    fn restore(&mut self, meta: &__JitMeta, values: &[i64]) {
                        if values.len() >= 4 {
                            let payload_end = values.len().saturating_sub(1);
                            let mut cursor = 2usize;
                            let mut parsed = true;
                            for &(sidx, _) in &meta.storage_layout {
                                if cursor + 1 >= payload_end {
                                    parsed = false;
                                    break;
                                }
                                let len_raw = values[cursor];
                                cursor += 1;
                                let Some(dirty_count) = usize::try_from(values[cursor]).ok() else {
                                    parsed = false;
                                    break;
                                };
                                cursor += 1;
                                if len_raw >= 0 {
                                    let Some(new_len) = usize::try_from(len_raw).ok() else {
                                        parsed = false;
                                        break;
                                    };
                                    let store = self.#pool_field.get_mut(sidx);
                                    store.force_len(new_len);
                                    for _ in 0..dirty_count {
                                        if cursor + 1 >= payload_end {
                                            parsed = false;
                                            break;
                                        }
                                        let Some(index) = usize::try_from(values[cursor]).ok() else {
                                            parsed = false;
                                            break;
                                        };
                                        let raw = values[cursor + 1];
                                        cursor += 2;
                                        store.write_jit_raw_value(index, raw);
                                    }
                                    if !parsed {
                                        break;
                                    }
                                } else if dirty_count != 0 {
                                    parsed = false;
                                    break;
                                }
                            }
                            if !parsed || cursor != payload_end {
                                self.#pool_field.apply_jit_lengths();
                            }
                            self.#sel_field = values
                                .get(1)
                                .copied()
                                .and_then(|value| usize::try_from(value).ok())
                                .unwrap_or(meta.initial_selected);
                            return;
                        }
                        if values.len() == 2 {
                            self.#sel_field = values
                                .get(1)
                                .copied()
                                .and_then(|value| usize::try_from(value).ok())
                                .unwrap_or(meta.initial_selected);
                        } else {
                            self.#pool_field.apply_jit_lengths();
                            self.#sel_field = meta.initial_selected;
                        }
                    }

                    #vable_info_fn
                    #vable_state_hooks

                    fn collect_jump_args(sym: &__JitSym) -> Vec<majit_ir::OpRef> {
                        let selected = majit_meta::JitCodeSym::current_selected_value(sym)
                            .unwrap_or_else(|| majit_ir::OpRef::NONE);
                        vec![sym.pool_ref, selected]
                    }

                    fn validate_close(_sym: &__JitSym, _meta: &__JitMeta) -> bool {
                        true
                    }
                }
            };
        }

        return quote! {
            /// Compiled loop metadata: traced storages in stable order.
            #[derive(Clone)]
            #[allow(non_camel_case_types)]
            struct __JitMeta {
                storage_layout: Vec<(usize, usize)>,
                initial_selected: usize,
            }

            /// Symbolic state during tracing: compact ptr/len/cap triples per storage.
            #[allow(non_camel_case_types)]
            struct __JitSym {
                storage_refs: std::collections::HashMap<
                    usize,
                    (majit_ir::OpRef, majit_ir::OpRef, majit_ir::OpRef),
                >,
                current_selected: usize,
                current_selected_value: Option<majit_ir::OpRef>,
                storage_layout: Vec<(usize, usize)>,
                loop_header_pc: usize,
                header_selected: usize,
                trace_started: bool,
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

                fn stack(&self, _selected: usize) -> Option<&majit_meta::SymbolicStack> {
                    None
                }

                fn stack_mut(&mut self, _selected: usize) -> Option<&mut majit_meta::SymbolicStack> {
                    None
                }

                fn total_slots(&self) -> usize {
                    self.storage_refs.len() * 3
                }

                fn loop_header_pc(&self) -> usize {
                    self.loop_header_pc
                }

                fn header_selected(&self) -> usize {
                    self.header_selected
                }

                fn ensure_stack(&mut self, _selected: usize, _offset: usize, _len: usize) {}

                fn fail_args(&self) -> Option<Vec<majit_ir::OpRef>> {
                    let mut args = Vec::with_capacity(self.storage_layout.len() * 3);
                    for &(sidx, _) in &self.storage_layout {
                        let &(ptr, len, cap) = self
                            .storage_refs
                            .get(&sidx)
                            .expect("missing compact storage refs");
                        args.push(ptr);
                        args.push(len);
                        args.push(cap);
                    }
                    Some(args)
                }

                fn compact_storage_ptr(&self, selected: usize) -> Option<majit_ir::OpRef> {
                    self.storage_refs.get(&selected).map(|&(ptr, _, _)| ptr)
                }

                fn compact_storage_len(&self, selected: usize) -> Option<majit_ir::OpRef> {
                    self.storage_refs.get(&selected).map(|&(_, len, _)| len)
                }

                fn compact_storage_cap(&self, selected: usize) -> Option<majit_ir::OpRef> {
                    self.storage_refs.get(&selected).map(|&(_, _, cap)| cap)
                }

                fn set_compact_storage_len(&mut self, selected: usize, value: majit_ir::OpRef) {
                    if let Some((_, len, _)) = self.storage_refs.get_mut(&selected) {
                        *len = value;
                    }
                }

                fn compact_storage_bounds(&self) -> Option<(i64, i64)> {
                    #compact_bounds
                }

                fn compact_storage_decode(
                    &self,
                    ctx: &mut majit_meta::TraceCtx,
                    raw: majit_ir::OpRef,
                ) -> majit_ir::OpRef {
                    #compact_decode
                }

                fn compact_storage_encode(
                    &self,
                    ctx: &mut majit_meta::TraceCtx,
                    value: majit_ir::OpRef,
                ) -> majit_ir::OpRef {
                    #compact_encode
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
                    let layout = storages.into_iter().map(|sidx| (sidx, 0)).collect();
                    __JitMeta {
                        storage_layout: layout,
                        initial_selected: self.#sel_field,
                    }
                }

                fn extract_live(&self, meta: &__JitMeta) -> Vec<i64> {
                    let mut values = Vec::with_capacity(meta.storage_layout.len() * 3);
                    for &(sidx, _) in &meta.storage_layout {
                        let store = self.#pool_field.get(sidx);
                        values.push(store.data_ptr() as i64);
                        values.push(store.len() as i64);
                        values.push(store.capacity() as i64);
                    }
                    values
                }

                fn create_sym(meta: &__JitMeta, header_pc: usize) -> __JitSym {
                    let mut storage_refs = std::collections::HashMap::new();
                    let mut offset = 0usize;
                    for &(sidx, _) in &meta.storage_layout {
                        let ptr = majit_ir::OpRef(offset as u32);
                        let len = majit_ir::OpRef((offset + 1) as u32);
                        let cap = majit_ir::OpRef((offset + 2) as u32);
                        storage_refs.insert(sidx, (ptr, len, cap));
                        offset += 3;
                    }
                    __JitSym {
                        storage_refs,
                        current_selected: meta.initial_selected,
                        current_selected_value: None,
                        storage_layout: meta.storage_layout.clone(),
                        loop_header_pc: header_pc,
                        header_selected: meta.initial_selected,
                        trace_started: false,
                    }
                }

                fn is_compatible(&self, meta: &__JitMeta) -> bool {
                    meta.initial_selected == self.#sel_field
                        && meta.storage_layout.iter().all(|&(sidx, _)| {
                            let store = self.#pool_field.get(sidx);
                            store.data_ptr() != 0 && store.capacity() != 0
                        })
                }

                fn restore(&mut self, meta: &__JitMeta, values: &[i64]) {
                    let mut offset = 0usize;
                    for &(sidx, _) in &meta.storage_layout {
                        let Some(&new_len_raw) = values.get(offset + 1) else {
                            return;
                        };
                        let Ok(new_len) = usize::try_from(new_len_raw) else {
                            return;
                        };
                        self.#pool_field.get_mut(sidx).force_len(new_len);
                        offset += 3;
                    }
                    self.#sel_field = values
                        .get(offset)
                        .copied()
                        .and_then(|value| usize::try_from(value).ok())
                        .unwrap_or(meta.initial_selected);
                }

                fn collect_jump_args(sym: &__JitSym) -> Vec<majit_ir::OpRef> {
                    let mut args = Vec::with_capacity(sym.storage_layout.len() * 3);
                    for &(sidx, _) in &sym.storage_layout {
                        let &(ptr, len, cap) = sym
                            .storage_refs
                            .get(&sidx)
                            .expect("missing compact storage refs");
                        args.push(ptr);
                        args.push(len);
                        args.push(cap);
                    }
                    args
                }

                fn validate_close(sym: &__JitSym, meta: &__JitMeta) -> bool {
                    sym.current_selected == meta.initial_selected
                        && sym.storage_layout.len() == meta.storage_layout.len()
                        && sym
                            .storage_layout
                            .iter()
                            .zip(meta.storage_layout.iter())
                            .all(|(&(sym_idx, _), &(meta_idx, _))| sym_idx == meta_idx)
                }
            }
        };
    }

    // Generate linked list JitCodeSym methods if node layout is provided
    let linked_list_descr_methods =
        if let (Some(node_size), Some(value_offset), Some(next_offset)) = (
            &storage.linked_list_node_size,
            &storage.linked_list_value_offset,
            &storage.linked_list_next_offset,
        ) {
            quote! {
                fn node_size_descr(&self) -> Option<majit_ir::DescrRef> {
                    Some(majit_ir::descr::make_size_descr(#node_size))
                }

                fn node_value_descr(&self) -> Option<majit_ir::DescrRef> {
                    Some(std::sync::Arc::new(
                        majit_ir::descr::SimpleFieldDescr::new(
                            0x8000_0001, // unique tag for value field
                            #value_offset,
                            8,
                            majit_ir::Type::Int,
                            true, // immutable: once set in New, value doesn't change
                        ),
                    ))
                }

                fn node_next_descr(&self) -> Option<majit_ir::DescrRef> {
                    Some(std::sync::Arc::new(
                        majit_ir::descr::SimpleFieldDescr::new(
                            0x8000_0002, // unique tag for next field
                            #next_offset,
                            8,
                            majit_ir::Type::Ref,
                            false,
                        ),
                    ))
                }
            }
        } else {
            quote! {}
        };

    // ── Queue (FIFO) support ──
    let queue_indices = &storage.linked_list_queue_indices;
    let linked_list_queue_methods =
        if let Some(tail_offset) = &storage.linked_list_queue_tail_offset {
            quote! {
                fn is_queue_storage(&self, selected: usize) -> bool {
                    false #( || selected == #queue_indices )*
                }

                fn linked_list_queue_tail_descr(&self) -> Option<majit_ir::DescrRef> {
                    Some(std::sync::Arc::new(
                        majit_ir::descr::SimpleFieldDescr::new(
                            0x8000_0005, // unique tag for queue tail field
                            #tail_offset,
                            8,
                            majit_ir::Type::Ref,
                            false,
                        ),
                    ))
                }

                fn linked_list_tail(&self, selected: usize) -> Option<majit_ir::OpRef> {
                    self.linked_list_tails.get(&selected).copied()
                }

                fn set_linked_list_tail(&mut self, selected: usize, tail: majit_ir::OpRef) {
                    self.linked_list_tails.insert(selected, tail);
                }

                fn linked_list_writeback_tail(
                    &mut self,
                    ctx: &mut majit_meta::TraceCtx,
                    selected: usize,
                    new_tail: majit_ir::OpRef,
                ) {
                    let Some(stack_ref) = self.ensure_linked_list_stack_ref(ctx, selected) else {
                        return;
                    };
                    let Some(tail_descr) = self.linked_list_queue_tail_descr() else {
                        return;
                    };
                    ctx.record_op_with_descr(
                        majit_ir::OpCode::SetfieldGc,
                        &[stack_ref, new_tail],
                        tail_descr,
                    );
                }

                fn ensure_linked_list_tail(
                    &mut self,
                    ctx: &mut majit_meta::TraceCtx,
                    selected: usize,
                ) -> Option<majit_ir::OpRef> {
                    if let Some(&tail) = self.linked_list_tails.get(&selected) {
                        return Some(tail);
                    }
                    let stack_ref = self.ensure_linked_list_stack_ref(ctx, selected)?;
                    let tail_descr = self.linked_list_queue_tail_descr()?;
                    let tail = ctx.record_op_with_descr(
                        majit_ir::OpCode::GetfieldGcR,
                        &[stack_ref],
                        tail_descr,
                    );
                    self.linked_list_tails.insert(selected, tail);
                    Some(tail)
                }
            }
        } else {
            quote! {}
        };

    // ── Linked list mode ──
    // RPython parity: live state carries stacksize + storage ref + selected.
    // Stack heads and sizes live on GC-managed shadow stack objects.
    if storage.linked_list_node_size.is_some() {
        let pool_ref_field =
            pool_ref_field.expect("linked_list mode requires storage.pool_ref as a GcRef field");
        let selected_ref_field = selected_ref_field
            .expect("linked_list mode requires storage.selected_ref as a GcRef field");
        let stacksize_field = stacksize_field.expect("linked_list mode requires storage.stacksize");
        let storage_offset = storage.linked_list_storage_offset.as_ref().expect(
            "linked_list mode requires linked_list_storage_offset (shadow storage pools offset)",
        );
        let stack_head_offset = storage
            .linked_list_stack_head_offset
            .as_ref()
            .expect("linked_list mode requires linked_list_stack_head_offset");
        let stack_size_offset = storage
            .linked_list_stack_size_offset
            .as_ref()
            .expect("linked_list mode requires linked_list_stack_size_offset");
        let pool_live_raw = quote! { self.#pool_ref_field.as_usize() as i64 };
        let selected_ref_live_raw = quote! { self.#selected_ref_field.as_usize() as i64 };

        return quote! {
            #[derive(Clone)]
            #[allow(non_camel_case_types)]
            struct __JitMeta {
                storage_layout: Vec<(usize, usize)>,
                initial_selected: usize,
            }

            #[allow(non_camel_case_types)]
            struct __JitSym {
                pool_ref: majit_ir::OpRef,
                current_stacksize_value: Option<majit_ir::OpRef>,
                current_selected: usize,
                current_selected_value: Option<majit_ir::OpRef>,
                current_selected_ref: Option<majit_ir::OpRef>,
                storage_layout: Vec<(usize, usize)>,
                loop_header_pc: usize,
                header_selected: usize,
                trace_started: bool,
                meta_storage_count: usize,
                linked_list_stack_refs: std::collections::HashMap<usize, majit_ir::OpRef>,
                linked_list_heads: std::collections::HashMap<usize, majit_ir::OpRef>,
                linked_list_tails: std::collections::HashMap<usize, majit_ir::OpRef>,
                // Keep symbolic stacks for fallback (non-linked-list storages)
                stacks: std::collections::HashMap<usize, majit_meta::SymbolicStack>,
            }

            #[allow(non_camel_case_types)]
            impl __JitSym {
                fn total_slots(&self) -> usize {
                    0 // No flattened slots in linked list mode
                }
            }

            impl majit_meta::JitCodeSym for __JitSym {
                fn current_selected(&self) -> usize {
                    self.current_selected
                }

                fn current_selected_value(&self) -> Option<majit_ir::OpRef> {
                    self.current_selected_value
                }

                fn current_selected_ref(&self) -> Option<majit_ir::OpRef> {
                    self.current_selected_ref
                }

                fn current_stacksize_value(&self) -> Option<majit_ir::OpRef> {
                    self.current_stacksize_value
                }

                fn set_current_selected(&mut self, selected: usize) {
                    self.current_selected = selected;
                }

                fn set_current_selected_value(&mut self, selected: usize, value: majit_ir::OpRef) {
                    self.current_selected = selected;
                    self.current_selected_value = Some(value);
                }

                fn set_current_selected_ref(&mut self, selected: usize, value: majit_ir::OpRef) {
                    self.current_selected = selected;
                    self.current_selected_ref = Some(value);
                    self.linked_list_stack_refs.insert(selected, value);
                }

                fn set_current_stacksize_value(&mut self, value: majit_ir::OpRef) {
                    self.current_stacksize_value = Some(value);
                }

                fn stack(&self, selected: usize) -> Option<&majit_meta::SymbolicStack> {
                    self.stacks.get(&selected)
                }

                fn stack_mut(&mut self, selected: usize) -> Option<&mut majit_meta::SymbolicStack> {
                    self.stacks.get_mut(&selected)
                }

                fn total_slots(&self) -> usize { 0 }
                fn loop_header_pc(&self) -> usize { self.loop_header_pc }
                fn header_selected(&self) -> usize { self.header_selected }

                fn ensure_stack(&mut self, selected: usize, _offset: usize, _len: usize) {
                    if !self.storage_layout.iter().any(|&(s, _)| s == selected) {
                        self.storage_layout.push((selected, 0));
                    }
                }

                fn fail_args(&self) -> Option<Vec<majit_ir::OpRef>> {
                    let stacksize = self.current_stacksize_value
                        .unwrap_or(majit_ir::OpRef::NONE);
                    let selected = self.current_selected_value
                        .unwrap_or(majit_ir::OpRef::NONE);
                    let selected_ref = self.current_selected_ref
                        .unwrap_or(majit_ir::OpRef::NONE);
                    // RPython parity: guard fail state carries the red vars.
                    Some(vec![stacksize, self.pool_ref, selected, selected_ref])
                }

                fn fail_args_types(&self) -> Option<Vec<majit_ir::Type>> {
                    Some(vec![
                        majit_ir::Type::Int,
                        majit_ir::Type::Ref,
                        majit_ir::Type::Int,
                        majit_ir::Type::Ref,
                    ])
                }

                fn selected_in_fail_args_prefix(&self) -> bool {
                    true
                }

                fn close_requires_header_selected(&self) -> bool {
                    false
                }

                // Linked list methods
                fn linked_list_head(&self, selected: usize) -> Option<majit_ir::OpRef> {
                    self.linked_list_heads.get(&selected).copied()
                }

                fn linked_list_stack_ref(&self, selected: usize) -> Option<majit_ir::OpRef> {
                    if selected == self.current_selected {
                        self.current_selected_ref
                    } else {
                        self.linked_list_stack_refs.get(&selected).copied()
                    }
                }

                fn set_linked_list_head(&mut self, selected: usize, head: majit_ir::OpRef) {
                    self.linked_list_heads.insert(selected, head);
                }

                fn set_linked_list_stack_ref(&mut self, selected: usize, stack_ref: majit_ir::OpRef) {
                    self.linked_list_stack_refs.insert(selected, stack_ref);
                    if selected == self.current_selected {
                        self.current_selected_ref = Some(stack_ref);
                    }
                }

                fn ensure_linked_list_stack_ref(
                    &mut self,
                    ctx: &mut majit_meta::TraceCtx,
                    selected: usize,
                ) -> Option<majit_ir::OpRef> {
                    if false #( || selected == #untraceable )* {
                        return None;
                    }
                    if selected == self.current_selected {
                        if let Some(stack_ref) = self.current_selected_ref {
                            return Some(stack_ref);
                        }
                    }
                    if let Some(&stack_ref) = self.linked_list_stack_refs.get(&selected) {
                        return Some(stack_ref);
                    }
                    let stack_descr = self.linked_list_storage_item_descr(selected)?;
                    let stack_ref = ctx.record_op_with_descr(
                        majit_ir::OpCode::GetfieldGcR,
                        &[self.pool_ref],
                        stack_descr,
                    );
                    self.linked_list_stack_refs.insert(selected, stack_ref);
                    if selected == self.current_selected {
                        self.current_selected_ref = Some(stack_ref);
                    }
                    Some(stack_ref)
                }

                fn ensure_linked_list_head(
                    &mut self,
                    ctx: &mut majit_meta::TraceCtx,
                    selected: usize,
                ) -> Option<majit_ir::OpRef> {
                    if let Some(&head) = self.linked_list_heads.get(&selected) {
                        return Some(head);
                    }
                    let stack_ref = self.ensure_linked_list_stack_ref(ctx, selected)?;

                    // Guard that the stack is non-empty before reading head.
                    // Without this, the body loop can dereference a null head
                    // pointer when the stack becomes empty after JUMP back to
                    // the Label.
                    let size_descr = self.linked_list_stack_size_descr()?;
                    let size = ctx.record_op_with_descr(
                        majit_ir::OpCode::GetfieldGcI,
                        &[stack_ref],
                        size_descr,
                    );
                    let one = ctx.const_int(1);
                    let ge = ctx.record_op(majit_ir::OpCode::IntGe, &[size, one]);
                    let stacksize = self.current_stacksize_value
                        .unwrap_or(size);
                    let selected_val = self.current_selected_value
                        .unwrap_or_else(|| ctx.const_int(self.current_selected as i64));
                    let selected_ref = self.current_selected_ref
                        .unwrap_or(stack_ref);
                    // Include resume_pc (loop_header_pc) so restore_jit_guard_state
                    // can return Some(resume_pc) → enables bridge compilation.
                    let resume_pc = ctx.const_int(self.loop_header_pc as i64);
                    let fail_args = vec![stacksize, self.pool_ref, selected_val, selected_ref, resume_pc];
                    let fail_types = vec![
                        majit_ir::Type::Int,
                        majit_ir::Type::Ref,
                        majit_ir::Type::Int,
                        majit_ir::Type::Ref,
                        majit_ir::Type::Int,
                    ];
                    ctx.record_guard_typed_with_fail_args(
                        majit_ir::OpCode::GuardTrue,
                        &[ge],
                        fail_types,
                        &fail_args,
                    );

                    let head_descr = self.linked_list_stack_head_descr()?;
                    let head = ctx.record_op_with_descr(
                        majit_ir::OpCode::GetfieldGcR,
                        &[stack_ref],
                        head_descr,
                    );
                    self.linked_list_heads.insert(selected, head);
                    Some(head)
                }

                fn linked_list_writeback_head(
                    &mut self,
                    ctx: &mut majit_meta::TraceCtx,
                    selected: usize,
                    new_head: majit_ir::OpRef,
                ) {
                    let Some(stack_ref) = self.ensure_linked_list_stack_ref(ctx, selected) else {
                        return;
                    };
                    let Some(head_descr) = self.linked_list_stack_head_descr() else {
                        return;
                    };
                    ctx.record_op_with_descr(
                        majit_ir::OpCode::SetfieldGc,
                        &[stack_ref, new_head],
                        head_descr,
                    );
                }

                fn linked_list_storage_item_descr(&self, selected: usize) -> Option<majit_ir::DescrRef> {
                    let offset = ((#storage_offset) as usize) + selected * 8;
                    Some(std::sync::Arc::new(
                        majit_ir::descr::SimpleFieldDescr::new(
                            ((offset as u32) << 2) | 0x1,
                            offset,
                            8,
                            majit_ir::Type::Ref,
                            true, // immutable: stack_ptrs never change after init
                        ),
                    ))
                }

                fn linked_list_stack_head_descr(&self) -> Option<majit_ir::DescrRef> {
                    // RPython: Stack.head is a plain mutable field (no _virtualizable_).
                    // virtualizable=true here is a workaround for heap cache not properly
                    // exporting/importing field state across JUMP→Label transitions.
                    // TODO: fix heap cache export for linked list fields, then remove this.
                    Some(std::sync::Arc::new(
                        majit_ir::descr::SimpleFieldDescr::new(
                            0x8000_0003,
                            #stack_head_offset,
                            8,
                            majit_ir::Type::Ref,
                            false,
                        ).with_virtualizable(true),
                    ))
                }

                fn linked_list_stack_size_descr(&self) -> Option<majit_ir::DescrRef> {
                    // Same workaround as head — see comment above.
                    Some(std::sync::Arc::new(
                        majit_ir::descr::SimpleFieldDescr::new(
                            0x8000_0004,
                            #stack_size_offset,
                            8,
                            majit_ir::Type::Int,
                            false,
                        ).with_virtualizable(true),
                    ))
                }

                fn linked_list_writeback_size(
                    &mut self,
                    ctx: &mut majit_meta::TraceCtx,
                    selected: usize,
                    new_size: majit_ir::OpRef,
                ) {
                    let Some(stack_ref) = self.ensure_linked_list_stack_ref(ctx, selected) else {
                        return;
                    };
                    let Some(size_descr) = self.linked_list_stack_size_descr() else {
                        return;
                    };
                    ctx.record_op_with_descr(
                        majit_ir::OpCode::SetfieldGc,
                        &[stack_ref, new_size],
                        size_descr,
                    );
                }

                #linked_list_descr_methods
                #linked_list_queue_methods
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
                    let layout = storages.into_iter().map(|sidx| (sidx, 0)).collect();
                    __JitMeta {
                        storage_layout: layout,
                        initial_selected: self.#sel_field,
                    }
                }

                fn extract_live(&self, meta: &__JitMeta) -> Vec<i64> {
                    // RPython parity: root loop inputargs are the red vars.
                    // Concrete stack contents stay on shadow stack objects.
                    let _ = meta;
                    vec![
                        self.#stacksize_field as i64,
                        #pool_live_raw,
                        self.#sel_field as i64,
                        #selected_ref_live_raw,
                    ]
                }

                fn live_value_types(&self, meta: &__JitMeta) -> Vec<majit_ir::Type> {
                    let _ = meta;
                    vec![
                        majit_ir::Type::Int,
                        majit_ir::Type::Ref,
                        majit_ir::Type::Int,
                        majit_ir::Type::Ref,
                    ]
                }

                fn create_sym(meta: &__JitMeta, header_pc: usize) -> __JitSym {
                    __JitSym {
                        pool_ref: majit_ir::OpRef(1),
                        current_stacksize_value: Some(majit_ir::OpRef(0)),
                        current_selected: meta.initial_selected,
                        current_selected_value: Some(majit_ir::OpRef(2)),
                        current_selected_ref: Some(majit_ir::OpRef(3)),
                        storage_layout: meta.storage_layout.clone(),
                        loop_header_pc: header_pc,
                        header_selected: meta.initial_selected,
                        trace_started: false,
                        meta_storage_count: meta.storage_layout.len(),
                        linked_list_stack_refs: std::collections::HashMap::new(),
                        linked_list_heads: std::collections::HashMap::new(),
                        linked_list_tails: std::collections::HashMap::new(),
                        stacks: std::collections::HashMap::new(),
                    }
                }

                fn is_compatible(&self, meta: &__JitMeta) -> bool {
                    meta.initial_selected == self.#sel_field
                }

                fn restore(&mut self, meta: &__JitMeta, values: &[i64]) {
                    self.#stacksize_field = values
                        .first()
                        .copied()
                        .and_then(|v| i32::try_from(v).ok())
                        .unwrap_or_default();
                    self.#pool_ref_field = majit_ir::GcRef(
                        values
                            .get(1)
                            .copied()
                            .and_then(|v| usize::try_from(v).ok())
                            .unwrap_or(0),
                    );
                    self.#sel_field = values
                        .get(2)
                        .copied()
                        .and_then(|v| usize::try_from(v).ok())
                        .unwrap_or(meta.initial_selected);
                    self.#selected_ref_field = majit_ir::GcRef(
                        values
                            .get(3)
                            .copied()
                            .and_then(|v| usize::try_from(v).ok())
                            .unwrap_or(0),
                    );
                }

                fn collect_jump_args(sym: &__JitSym) -> Vec<majit_ir::OpRef> {
                    let stacksize = sym.current_stacksize_value.unwrap_or(majit_ir::OpRef::NONE);
                    let selected = sym.current_selected_value.unwrap_or(majit_ir::OpRef::NONE);
                    let selected_ref = sym.current_selected_ref.unwrap_or(majit_ir::OpRef::NONE);
                    vec![stacksize, sym.pool_ref, selected, selected_ref]
                }

                #vable_info_fn
                #vable_state_hooks

                fn validate_close(sym: &__JitSym, meta: &__JitMeta) -> bool {
                    // RPython parity: only check selected matches.
                    // No depth constraint — linked list allows variable depth.
                    sym.current_selected == meta.initial_selected
                }
            }
        };
    }

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
            /// Skip closing on the first merge_point at the loop header.
            trace_started: bool,
            /// Number of storages from the original meta layout.
            /// Extra storages added during tracing are ignored in Jump args.
            meta_storage_count: usize,
            /// Linked list head OpRef per storage (RPython Node virtualization).
            linked_list_heads: std::collections::HashMap<usize, majit_ir::OpRef>,
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
                let mut args = Vec::new();
                for &(sidx, _) in self.storage_layout.iter().take(self.meta_storage_count) {
                    args.extend(self.stacks[&sidx].to_jump_args());
                }
                Some(args)
            }

            fn fail_storage_lengths(&self) -> Option<Vec<usize>> {
                Some(
                    self.storage_layout
                        .iter()
                        .map(|&(sidx, _)| self.stacks[&sidx].len())
                        .collect(),
                )
            }

            #linked_list_descr_methods
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
                let layout = storages
                    .iter()
                    .map(|&s| (s, self.#pool_field.get(s).len()))
                    .collect();
                __JitMeta {
                    storage_layout: layout,
                    initial_selected: self.#sel_field,
                }
            }

            fn extract_live(&self, meta: &__JitMeta) -> Vec<i64> {
                let mut values = Vec::new();
                for &(sidx, num_slots) in &meta.storage_layout {
                    let store = self.#pool_field.get(sidx);
                    for i in 0..num_slots {
                        values.push(store.peek_at(i));
                    }
                }
                values
            }

            fn create_sym(meta: &__JitMeta, header_pc: usize) -> __JitSym {
                let mut stacks = std::collections::HashMap::new();
                let mut offset = 0;
                for &(sidx, num_slots) in &meta.storage_layout {
                    stacks.insert(
                        sidx,
                        majit_meta::SymbolicStack::from_input_args(offset, num_slots),
                    );
                    offset += num_slots;
                }
                __JitSym {
                    stacks,
                    current_selected: meta.initial_selected,
                    current_selected_value: None,
                    storage_layout: meta.storage_layout.clone(),
                    loop_header_pc: header_pc,
                    header_selected: meta.initial_selected,
                    trace_started: false,
                    meta_storage_count: meta.storage_layout.len(),
                    linked_list_heads: std::collections::HashMap::new(),
                }
            }

            fn is_compatible(&self, meta: &__JitMeta) -> bool {
                meta.initial_selected == self.#sel_field
                    && meta.storage_layout.iter().all(|&(sidx, expected_len)| {
                        self.#pool_field.get(sidx).len() == expected_len
                    })
            }

            fn restore(&mut self, meta: &__JitMeta, values: &[i64]) {
                // Two possible formats:
                // 1. JUMP args (loop finish): [stack_elems...] — length == sum(num_slots)
                // 2. Guard fail_args: [stack_elems..., lengths..., selected, resume_pc]
                let total_meta_slots: usize = meta.storage_layout.iter().map(|&(_, n)| n).sum();
                if values.len() == total_meta_slots {
                    // Format 1: simple JUMP args — restore stacks from meta depths
                    let mut offset = 0;
                    for &(sidx, num_slots) in &meta.storage_layout {
                        let end = offset + num_slots;
                        let store = self.#pool_field.get_mut(sidx);
                        store.clear();
                        for &value in &values[offset..end] {
                            store.push(value);
                        }
                        offset = end;
                    }
                    self.#sel_field = meta.initial_selected;
                    return;
                }
                // Format 2: guard fail_args with suffix
                let n_storages = meta.storage_layout.len();
                let suffix_len = n_storages + 2; // lengths + selected + resume_pc
                if values.len() < suffix_len {
                    self.#sel_field = meta.initial_selected;
                    return;
                }
                let total_stack_elems = values.len() - suffix_len;
                let lengths_offset = total_stack_elems;
                let selected_offset = lengths_offset + n_storages;

                let mut offset = 0;
                for (i, &(sidx, _)) in meta.storage_layout.iter().enumerate() {
                    let current_len = values
                        .get(lengths_offset + i)
                        .copied()
                        .and_then(|v| usize::try_from(v).ok())
                        .unwrap_or(0);
                    let end = offset + current_len;
                    let store = self.#pool_field.get_mut(sidx);
                    store.clear();
                    if end <= total_stack_elems {
                        for &value in &values[offset..end] {
                            store.push(value);
                        }
                    }
                    offset = end;
                }
                self.#sel_field = values
                    .get(selected_offset)
                    .copied()
                    .and_then(|v| usize::try_from(v).ok())
                    .unwrap_or(meta.initial_selected);
            }

            fn collect_jump_args(sym: &__JitSym) -> Vec<majit_ir::OpRef> {
                let mut args = Vec::new();
                // Only include original meta storages. Extra storages from
                // trace-time OP_SEL are excluded — their state stays on the
                // interpreter's heap and is not carried by the compiled loop.
                for &(sidx, _) in sym.storage_layout.iter().take(sym.meta_storage_count) {
                    args.extend(sym.stacks[&sidx].to_jump_args());
                }
                args
            }

            #vable_info_fn
            #vable_state_hooks

            fn validate_close(sym: &__JitSym, meta: &__JitMeta) -> bool {
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
                "int" => None,
                _ => Some(format!("{}: {}", f.name, ty)),
            }
        })
        .collect();
    if !unsupported_fields.is_empty() {
        let message = format!(
            "state_fields supports int, [int], and [int; virt]; unsupported: {}",
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
            trace_started: bool,
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
                    trace_started: false,
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

#[cfg(test)]
mod tests {
    use super::generate_jit_state;
    use crate::jit_interp::JitInterpConfig;
    use quote::quote;

    fn render(config: proc_macro2::TokenStream) -> String {
        let parsed = syn::parse2::<JitInterpConfig>(config).expect("valid jit_interp config");
        generate_jit_state(&parsed).to_string()
    }

    #[test]
    fn storage_mode_generates_virtualizable_sync_hooks() {
        let generated = render(quote! {
            state = State,
            env = Program,
            storage = {
                pool: state.storage,
                pool_type: StoragePool,
                selector: state.selected,
                untraceable: [],
                scan: scan_storages,
            },
            virtualizable_fields = {
                var: frame,
                token_offset: VABLE_TOKEN_OFFSET,
                fields: { next_instr: int @ NEXT_INSTR_OFFSET },
                arrays: { locals_w: ref @ LOCALS_OFFSET },
            }
        });

        assert!(generated.contains("fn __build_virtualizable_info"));
        assert!(generated.contains("__info . add_array_field"));
        assert!(generated.contains("fn virtualizable_heap_ptr"));
        assert!(generated.contains("fn virtualizable_array_lengths"));
        assert!(generated.contains("fn import_virtualizable_boxes"));
        assert!(generated.contains("fn export_virtualizable_boxes"));
        assert!(generated.contains("fn sync_virtualizable_before_residual_call"));
        assert!(generated.contains("fn sync_virtualizable_after_residual_call"));
        assert!(generated.contains("tracing_before_residual_call"));
        assert!(generated.contains("tracing_after_residual_call"));
        assert!(generated.contains("ctx . force_token"));
    }

    #[test]
    fn compact_storage_mode_generates_embedded_array_vable_info_and_hooks() {
        let generated = render(quote! {
            state = State,
            env = Program,
            storage = {
                pool: state.storage,
                pool_type: StoragePool,
                selector: state.selected,
                untraceable: [],
                scan: scan_storages,
                compact_live: true,
                compact_ptrs_offset: STORAGEPOOL_DATA_PTRS_OFFSET,
                compact_lengths_offset: STORAGEPOOL_LENGTHS_OFFSET,
                compact_caps_offset: STORAGEPOOL_CAPS_OFFSET,
            },
            virtualizable_fields = {
                var: storage,
                token_offset: STORAGEPOOL_VABLE_TOKEN_OFFSET,
                fields: {},
                arrays: {
                    stack: int @ (STORAGEPOOL_DATA_PTRS_OFFSET + SLOT_OFFSET) {
                        ptr_offset: 0,
                        length_offset: STORAGEPOOL_LENGTHS_OFFSET - STORAGEPOOL_DATA_PTRS_OFFSET,
                        items_offset: 0,
                    },
                },
            }
        });

        assert!(generated.contains("fn __build_virtualizable_info"));
        assert!(generated.contains("__info . add_embedded_array_field_with_layout"));
        assert!(generated.contains("fn virtualizable_heap_ptr"));
        assert!(generated.contains("fn export_virtualizable_boxes"));
        assert!(generated.contains("fn sync_virtualizable_before_residual_call"));
        assert!(generated.contains("fn sync_virtualizable_after_residual_call"));
    }
}
