//! Generate JitState types (Meta, Sym) and impl from the macro configuration.

use proc_macro2::TokenStream;
use quote::quote;
use syn::Expr;

use super::JitInterpConfig;

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
    let state_type = &config.state_type;
    let env_type = &config.env_type;
    let pool_field = extract_field_member(&config.storage.pool);
    let sel_field = extract_field_member(&config.storage.selector);
    let untraceable = &config.storage.untraceable;
    let scan_fn = &config.storage.scan_fn;

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
            storage_layout: Vec<(usize, usize)>,
            loop_header_pc: usize,
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

            fn set_current_selected(&mut self, selected: usize) {
                self.current_selected = selected;
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

            fn ensure_stack(&mut self, selected: usize, offset: usize, len: usize) {
                self.stacks
                    .entry(selected)
                    .or_insert_with(|| majit_meta::SymbolicStack::from_input_args(offset, len));
                if !self.storage_layout.iter().any(|&(s, _)| s == selected) {
                    self.storage_layout.push((selected, len));
                }
            }
        }

        impl majit_meta::JitState for #state_type {
            type Meta = __JitMeta;
            type Sym = __JitSym;
            type Env = #env_type;

            fn can_trace(&self) -> bool {
                #( self.#sel_field != #untraceable )&&*
            }

            fn build_meta(&self, header_pc: usize, program: &#env_type) -> __JitMeta {
                let layout = #scan_fn(program, header_pc, self.#sel_field)
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
                    storage_layout: meta.storage_layout.clone(),
                    loop_header_pc: header_pc,
                }
            }

            fn is_compatible(&self, meta: &__JitMeta) -> bool {
                meta.initial_selected == self.#sel_field
            }

            fn restore(&mut self, meta: &__JitMeta, values: &[i64]) {
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
            }

            fn collect_jump_args(sym: &__JitSym) -> Vec<majit_ir::OpRef> {
                let mut args = Vec::new();
                for &(sidx, _) in &sym.storage_layout {
                    args.extend(sym.stacks[&sidx].to_jump_args());
                }
                args
            }

            fn validate_close(sym: &__JitSym, meta: &__JitMeta) -> bool {
                meta.storage_layout
                    .iter()
                    .all(|&(sidx, initial_depth)| sym.stacks[&sidx].len() == initial_depth)
            }
        }
    }
}
