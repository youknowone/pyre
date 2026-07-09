//! Generate JitState types (Meta, Sym) and impl from the macro configuration.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::ItemFn;

use super::{JitInterpConfig, StateFieldKind};

/// Generate the JitState types and implementation.
pub fn generate_jit_state(config: &JitInterpConfig, func: &ItemFn) -> TokenStream {
    generate_state_fields_jit_state(config, func)
}

/// Generate JitState types for state_fields mode (register/tape machines).
///
/// Instead of a storage pool with stacks, individual struct fields are tracked
/// as JIT-managed values. Scalars become single OpRefs, flattened arrays become
/// Vec<OpRef>, and virtualizable arrays (`[int; virt]`) track only a data
/// pointer + length OpRef pair (array stays on heap, accessed via raw memory ops).
fn generate_state_fields_jit_state(config: &JitInterpConfig, func: &ItemFn) -> TokenStream {
    let state_type = &config.state_type;
    let env_type = &config.env_type;
    let prebuild_fn_name = format_ident!("__prebuild_jitcode_liveness_{}", func.sig.ident);
    let dispatch_jitcode_fn_name = format_ident!("__dispatch_jitcode_{}", func.sig.ident);
    let declare_schema_fn_name = format_ident!("__declare_jit_schema_{}", func.sig.ident);
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
            // ref(T) is supported — a ref-typed scalar (usize carrier).
            StateFieldKind::Ref(_) => None,
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
    let ref_scalars: Vec<_> = sf
        .fields
        .iter()
        .enumerate()
        .filter(|(_, f)| matches!(f.kind, StateFieldKind::Ref(_)))
        .collect();
    // opaque(T) fields are pass-through carriers the JIT never enumerates as
    // inputargs and never reconstructs.  A fresh recursive-portal frame cannot
    // synthesize an arbitrary `T` generically, so any state shape carrying one
    // is excluded from the fresh-entry helpers below (they fall back to the
    // `None` default and the recursive dispatcher aborts to the interpreter).
    let opaque_fields: Vec<_> = sf
        .fields
        .iter()
        .enumerate()
        .filter(|(_, f)| matches!(f.kind, StateFieldKind::Opaque(_)))
        .collect();

    let num_scalars = scalars.len();
    let num_virt_arrays = virt_arrays.len();
    let num_ref_scalars = ref_scalars.len();
    // First ref-bank register available for ref-scalar identity slots.
    // `MIFrame::setup_call` packs the dispatch JitCode's ref args densely
    // from r0 (`program` at r0, the virtualizable identity at r1 when
    // present — `with_vable_input_ref_reg(1)` in codegen_trace), and the
    // blackhole re-executes ops reading those argument registers, so the
    // identity slots start past them.  Mirrors
    // `LowererConfig::ref_identity_base`; the vable-presence condition
    // matches the lowerer's `vable_var` synthesis (an explicit
    // `virtualizable` decl or any `[int; virt]` state array).
    let ref_identity_base: usize =
        1 + usize::from(config.virtualizable_decl.is_some() || num_virt_arrays > 0);
    let ref_identity_end: usize = ref_identity_base + num_ref_scalars;
    // First int-bank register available for scalar/array identity slots —
    // the int-bank mirror of `ref_identity_base`. The dispatch JitCode's
    // only int argument is `pc` at i0; aliasing it lets the guard-time
    // canonical materialization overwrite the pc register before resume
    // encode. Mirrors `LowererConfig::int_identity_base`.
    let int_identity_base: usize = 1;

    let recover_body: TokenStream = if let Some(ref recover_path) = config.recover {
        quote! { self.#recover_path(); }
    } else {
        quote! {}
    };

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
            let ptr_value_name = quote::format_ident!("{}_ptr_value", f.name);
            let len_value_name = quote::format_ident!("{}_len_value", f.name);
            quote! {
                #ptr_name: majit_ir::OpRef,
                #len_name: majit_ir::OpRef,
                #ptr_value_name: i64,
                #len_value_name: i64,
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
    // ── populate_frame_int_regs: scalars + flattened arrays ──
    // Matches `live_slots_for_state_field_jit` slot order so a
    // `MIFrame::get_list_of_active_boxes` walk against the canonical
    // liveness entry decodes back the same OpRefs / values that
    // `__JitSym` and the macro-emitted `live/<offset>` placeholder
    // refer to.  Virt-array populate is deferred — see
    // the trait-method docstring.
    let populate_scalar_parts: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let value_name = quote::format_ident!("{}_value", f.name);
            quote! {
                if __slot < frame.int_regs.len() {
                    frame.int_regs[__slot] = Some(self.#fname);
                    frame.int_values[__slot] = Some(self.#value_name);
                }
                __slot += 1;
            }
        })
        .collect();
    let populate_array_parts: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let value_name = quote::format_ident!("{}_values", f.name);
            quote! {
                for __i in 0..self.#fname.len() {
                    if __slot + __i < frame.int_regs.len() {
                        frame.int_regs[__slot + __i] = Some(self.#fname[__i]);
                        frame.int_values[__slot + __i] = Some(self.#value_name[__i]);
                    }
                }
                __slot += self.#fname.len();
            }
        })
        .collect();
    // Virt-array populate: two consecutive slots per
    // varray — `<varr>_ptr` (data pointer OpRef) at offset N, then
    // `<varr>_len` at N+1.  Value mirrors come from
    // `<varr>_ptr_value` / `<varr>_len_value` cached at
    // `JitState::initialize_sym` from the user state's
    // `<varr>.as_ptr() as i64` / `<varr>.len() as i64`.
    let populate_virt_array_parts: Vec<TokenStream> = virt_arrays
        .iter()
        .map(|(_, f)| {
            let ptr_name = quote::format_ident!("{}_ptr", f.name);
            let len_name = quote::format_ident!("{}_len", f.name);
            let ptr_value_name = quote::format_ident!("{}_ptr_value", f.name);
            let len_value_name = quote::format_ident!("{}_len_value", f.name);
            quote! {
                // `<varr>_ptr` is the backing storage's raw data pointer (a
                // raw address, not a GC ref) and `<varr>_len` its length: both
                // occupy int slots, matching `live_slots_for_state_field_jit`
                // (which counts `2 * num_virt_arrays` into `live_i`), the
                // `StateFieldLayout::total_slots` decode, and the int-stream
                // resume reader. The vable identity, when the state has one,
                // is a separate `ref(<T>)` scalar in the ref bank — NOT this
                // pointer. Writing the ptr to the ref bank desyncs it from the
                // int `live_i` index the resume reader decodes, leaving
                // `int_regs[slot]` unset when the guard snapshot is collected.
                if __slot < frame.int_regs.len() {
                    frame.int_regs[__slot] = Some(self.#ptr_name);
                    frame.int_values[__slot] = Some(self.#ptr_value_name);
                }
                __slot += 1;
                if __slot < frame.int_regs.len() {
                    frame.int_regs[__slot] = Some(self.#len_name);
                    frame.int_values[__slot] = Some(self.#len_value_name);
                }
                __slot += 1;
            }
        })
        .collect();

    // ── seed_recursive_fresh_frame: fresh state as CONSTANTS ──
    // Same slot layout as `populate_frame_int_regs`, but writes const OpRefs
    // (a fresh inline callee's state is known at the call site) with fresh
    // values: scalars 0, fixed-array cells 0, virt-array ptr 0 (the stack is
    // virtual, its cells live in the vable shadow), virt-array len = caller's
    // captured capacity (the fresh callee re-allocates at that size).
    let seed_scalar_parts: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, _f)| {
            quote! {
                if __slot < frame.int_regs.len() {
                    frame.int_regs[__slot] = Some(majit_ir::OpRef::const_int(0));
                    frame.int_values[__slot] = Some(0);
                }
                __slot += 1;
            }
        })
        .collect();
    let seed_array_parts: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! {
                for __i in 0..self.#fname.len() {
                    if __slot + __i < frame.int_regs.len() {
                        frame.int_regs[__slot + __i] = Some(majit_ir::OpRef::const_int(0));
                        frame.int_values[__slot + __i] = Some(0);
                    }
                }
                __slot += self.#fname.len();
            }
        })
        .collect();
    let seed_virt_array_parts: Vec<TokenStream> = virt_arrays
        .iter()
        .map(|(_, f)| {
            let len_value_name = quote::format_ident!("{}_len_value", f.name);
            quote! {
                if __slot < frame.int_regs.len() {
                    frame.int_regs[__slot] = Some(majit_ir::OpRef::const_int(0));
                    frame.int_values[__slot] = Some(0);
                }
                __slot += 1;
                let __cap = self.#len_value_name;
                if __slot < frame.int_regs.len() {
                    frame.int_regs[__slot] = Some(majit_ir::OpRef::const_int(__cap));
                    frame.int_values[__slot] = Some(__cap);
                }
                __slot += 1;
            }
        })
        .collect();

    // ── snapshot/reset/restore inline scalar+fixed-array sym state ──
    // The sym holds the WORKING scalar (and fixed-array) state read/written by
    // `BC_LOAD/STORE_STATE_FIELD` / `_STATE_ARRAY`.  An inline recursive-portal
    // callee overwrites it in place, so nest it: snapshot → reset fresh → run →
    // restore.  Virt arrays are excluded (their cells live in the vable shadow).
    let snapshot_scalar_parts: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let value_name = quote::format_ident!("{}_value", f.name);
            quote! { __out.push((self.#fname, self.#value_name)); }
        })
        .collect();
    let snapshot_array_parts: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let value_name = quote::format_ident!("{}_values", f.name);
            quote! {
                for __i in 0..self.#fname.len() {
                    __out.push((self.#fname[__i], self.#value_name[__i]));
                }
            }
        })
        .collect();
    let reset_scalar_parts: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let value_name = quote::format_ident!("{}_value", f.name);
            quote! {
                self.#fname = majit_ir::OpRef::const_int(0);
                self.#value_name = 0;
            }
        })
        .collect();
    let reset_array_parts: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let value_name = quote::format_ident!("{}_values", f.name);
            quote! {
                for __i in 0..self.#fname.len() {
                    self.#fname[__i] = majit_ir::OpRef::const_int(0);
                    self.#value_name[__i] = 0;
                }
            }
        })
        .collect();
    let restore_inline_scalar_parts: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let value_name = quote::format_ident!("{}_value", f.name);
            quote! {
                if __k < __snapshot.len() {
                    self.#fname = __snapshot[__k].0;
                    self.#value_name = __snapshot[__k].1;
                    __k += 1;
                }
            }
        })
        .collect();
    let restore_inline_array_parts: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let value_name = quote::format_ident!("{}_values", f.name);
            quote! {
                for __i in 0..self.#fname.len() {
                    if __k < __snapshot.len() {
                        self.#fname[__i] = __snapshot[__k].0;
                        self.#value_name[__i] = __snapshot[__k].1;
                        __k += 1;
                    }
                }
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

    // ── canonical_liveness_slots: array_lens slice expression ──
    // RPython `assembler.py:218-231 get_liveness_info` extracts per-kind
    // liveness for each `-live-` marker.  In flat-state JIT every slot
    // is permanently live, so the canonical entry is just
    // `[0..total_slots]` of int slots.  The `array_lens` slice fed to
    // `live_slots_for_state_field_jit` enumerates the runtime lengths
    // captured in `__JitMeta::<arr>_len` (one per flattened array).
    let canonical_liveness_array_len_refs: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let len_name = quote::format_ident!("{}_len", f.name);
            quote! { self.#len_name }
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
                // The vable base is the virtualizable identity (`&state` ==
                // `virtualizable_heap_ptr`), NOT the array's data pointer:
                // `vable_getarrayitem_*` reaches the element through the
                // RustVec storage from this base. Every virt array shares it.
                values.push(self as *const Self as i64);
                values.push(self.#fname.len() as i64);
            }
        })
        .collect();

    // ── #184 recursive CALL_ASSEMBLER portal entry (JitCodeSym side) ──
    // A recursive callee runs as its own compiled loop with a fresh frame.
    // `recursive_fresh_entry_reds` allocates a fresh `#state_type` (scalars
    // zeroed = empty frame; arrays re-allocated at the caller's live
    // capacity) and emits its reds in `extract_live` order.  Capacities come
    // from this symbolic state: a fixed array's sym field is a `Vec<OpRef>`
    // of the captured length, and a virt array caches its length in
    // `<arr>_len_value` (seeded at `JitState::initialize_sym`).  The whole
    // struct equals `state_fields`, so these inits build a complete fresh
    // `#state_type`.
    let fresh_entry_scalar_inits: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { #fname: 0, }
        })
        .collect();
    let fresh_entry_array_inits: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { #fname: ::std::vec![0i64; self.#fname.len()], }
        })
        .collect();
    let fresh_entry_virt_array_inits: Vec<TokenStream> = virt_arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let len_value_name = quote::format_ident!("{}_len_value", f.name);
            quote! { #fname: ::std::vec![0i64; self.#len_value_name as usize], }
        })
        .collect();
    // Reds in `extract_live` order: int scalars, then flattened fixed-array
    // elements, then per virt array the `&state` identity (Ref) + length
    // (Int).  Mirrors `extract_scalar_parts` / `extract_array_parts` /
    // `extract_virt_array_parts` so the fresh reds match the callee loop's
    // input-arg layout and `live_value_types` routing.
    let fresh_entry_scalar_value_pushes: Vec<TokenStream> = scalars
        .iter()
        .map(|_| quote! { __values.push(majit_ir::Value::Int(0)); })
        .collect();
    let fresh_entry_array_value_pushes: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! {
                for _ in 0..self.#fname.len() {
                    __values.push(majit_ir::Value::Int(0));
                }
            }
        })
        .collect();
    let fresh_entry_virt_array_value_pushes: Vec<TokenStream> = virt_arrays
        .iter()
        .map(|(_, f)| {
            let len_value_name = quote::format_ident!("{}_len_value", f.name);
            quote! {
                __values.push(majit_ir::Value::Ref(majit_ir::GcRef(__base as usize)));
                __values.push(majit_ir::Value::Int(self.#len_value_name));
            }
        })
        .collect();
    // The freshly-boxed `&state` identity feeds the virt-array Ref slots; only
    // bound when there is at least one virt array (fixed-array-only reds carry
    // no pointer).
    let fresh_entry_base_let: TokenStream = if num_virt_arrays > 0 {
        quote! { let __base = &*__fresh as *const #state_type as i64; }
    } else {
        quote! {}
    };
    // Emitted only for state shapes whose whole fresh frame can be synthesized
    // generically: no ref scalars and no opaque(T) carriers (neither has a
    // generic fresh value, and the `#state_type` struct literal below omits
    // opaque fields).  Other shapes fall back to the `JitCodeSym` default
    // (`None`) and the recursive dispatcher aborts to the interpreter.
    let recursive_fresh_entry_reds_override: TokenStream =
        if num_ref_scalars == 0 && opaque_fields.is_empty() {
            quote! {
                fn recursive_fresh_entry_reds(
                    &self,
                ) -> Option<(Vec<majit_ir::Value>, Box<dyn ::core::any::Any>)> {
                    let __fresh: Box<#state_type> = Box::new(#state_type {
                        #(#fresh_entry_scalar_inits)*
                        #(#fresh_entry_array_inits)*
                        #(#fresh_entry_virt_array_inits)*
                    });
                    #fresh_entry_base_let
                    let mut __values: Vec<majit_ir::Value> = Vec::new();
                    #(#fresh_entry_scalar_value_pushes)*
                    #(#fresh_entry_array_value_pushes)*
                    #(#fresh_entry_virt_array_value_pushes)*
                    Some((__values, __fresh as Box<dyn ::core::any::Any>))
                }
            }
        } else {
            quote! {}
        };

    // ── #184 recursive CALL_ASSEMBLER portal entry (host alloc/free) ──
    // The compiled caller loop cannot `New` a host `#state_type` through the
    // IR, so the recursive dispatcher records a residual call to these host
    // helpers: `alloc` returns a fresh `Box::into_raw`-ed `#state_type`
    // (scalars zeroed, the single virt array sized at the caller's live
    // capacity passed in `__cap`), `free` drops it.  Emitted only for the
    // shape the single-capacity allocator supports: zero ref scalars, no
    // opaque carriers, no fixed arrays, exactly one virt array (the `tl`
    // storage shape).  Other shapes leave `recursive_fresh_alloc_free_targets`
    // at its `None` default so the dispatcher aborts.
    let supports_fresh_alloc = num_ref_scalars == 0
        && opaque_fields.is_empty()
        && arrays.is_empty()
        && num_virt_arrays == 1;
    let recursive_fresh_alloc_free_fns: TokenStream = if supports_fresh_alloc {
        let virt_name = &virt_arrays[0].1.name;
        quote! {
            #[doc(hidden)]
            #[allow(non_snake_case)]
            extern "C" fn __majit_recursive_fresh_alloc(__cap: i64) -> i64 {
                let __fresh: ::std::boxed::Box<#state_type> = ::std::boxed::Box::new(#state_type {
                    #(#fresh_entry_scalar_inits)*
                    #virt_name: ::std::vec![0i64; __cap as usize],
                });
                ::std::boxed::Box::into_raw(__fresh) as i64
            }
            #[doc(hidden)]
            #[allow(non_snake_case)]
            extern "C" fn __majit_recursive_fresh_free(__ptr: i64) {
                if __ptr != 0 {
                    unsafe {
                        ::core::mem::drop(::std::boxed::Box::from_raw(__ptr as *mut #state_type));
                    }
                }
            }
        }
    } else {
        quote! {}
    };
    let recursive_fresh_alloc_free_targets_override: TokenStream = if supports_fresh_alloc {
        quote! {
            fn recursive_fresh_alloc_free_targets(&self) -> Option<(*const (), *const ())> {
                Some((
                    __majit_recursive_fresh_alloc as usize as *const (),
                    __majit_recursive_fresh_free as usize as *const (),
                ))
            }
        }
    } else {
        quote! {}
    };

    // ── create_sym: assign sequential OpRef::from_raw(0), OpRef::from_raw(1), ... ──
    let create_sym_scalar_inits: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let value_name = quote::format_ident!("{}_value", f.name);
            quote! {
                let #fname = majit_ir::OpRef::input_arg_int(__offset as u32);
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
                        // Array-typed sym storage is the i64 register
                        // bank; each cell mints `InputArgInt`
                        // (resoperation.py:719) consistent with the
                        // scalar i64 sym above and the typed inputarg
                        // produced by `TraceCtx::new`.
                        majit_ir::OpRef::input_arg_int((__offset + i) as u32)
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
            let ptr_value_name = quote::format_ident!("{}_ptr_value", f.name);
            let len_value_name = quote::format_ident!("{}_len_value", f.name);
            quote! {
                let #ptr_name = majit_ir::OpRef::input_arg_ref(__offset as u32);
                __offset += 1;
                let #len_name = majit_ir::OpRef::input_arg_int(__offset as u32);
                __offset += 1;
                let #ptr_value_name = 0i64;
                let #len_value_name = 0i64;
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
    let create_sym_virt_array_ptr_value_names: Vec<syn::Ident> = virt_arrays
        .iter()
        .map(|(_, f)| quote::format_ident!("{}_ptr_value", f.name))
        .collect();
    let create_sym_virt_array_len_value_names: Vec<syn::Ident> = virt_arrays
        .iter()
        .map(|(_, f)| quote::format_ident!("{}_len_value", f.name))
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
    // D2 per-opcode single-executor: read each scalar state field's concrete
    // value off the walk's persistent sym in scalar index order (mirrors
    // `state_field_value` / the `<field>_value` slots). Captured at the
    // CloseLoop point (while the sym is still live) into
    // `MetaInterp::single_pass_scalar_values`.
    let collect_scalar_values_parts: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let value_name = quote::format_ident!("{}_value", f.name);
            quote! { values.push(sym.#value_name); }
        })
        .collect();
    // D2 per-opcode single-executor: write each captured scalar state-field
    // value back into native state, the inverse of
    // `initialize_sym_scalar_parts`. `values[idx]` is the scalar at
    // state-field index `idx` (the order `collect_scalar_values_parts`
    // produced). recover runs afterwards and overwrites the storage-derived
    // caches, so only scalars recover cannot re-derive (e.g. aheui `selected`)
    // meaningfully carry through.
    let writeback_from_values_parts: Vec<TokenStream> = scalars
        .iter()
        .enumerate()
        .map(|(idx, (_, f))| {
            let fname = &f.name;
            let rust_ty = scalar_rust_type(&f.kind);
            let idx_lit = idx;
            quote! {
                self.#fname = values[#idx_lit] as #rust_ty;
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
    // Single-pass close: write the walk-final virt-array element values
    // (captured off the trace-ctx shadow) into native state, one array at a
    // time in declaration order. Mirrors `restore_array_parts`'s per-element
    // copy, but the source is the element-only flat vector
    // (`collect_virtualizable_element_values`), so there are no ptr/len slots to
    // skip. The user's Vec is fixed-capacity (see the reallocation note on
    // `initialize_sym_virt_array_parts`), so the length matches the walk-final
    // element count.
    let writeback_virt_array_parts: Vec<TokenStream> = virt_arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! {
                let __len = self.#fname.len();
                debug_assert!(
                    values.len() >= __offset + __len,
                    "writeback_virt_array: fewer element values than array length",
                );
                for i in 0..__len {
                    self.#fname[i] = values[__offset + i];
                }
                __offset += __len;
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
    // Virt-array `<varr>_ptr_value` / `<varr>_len_value` mirror the
    // current `<state>.<varr>` ptr/len so `populate_frame_int_regs`
    // can fill the corresponding `MIFrame.int_values` slots without
    // re-reading the live state at guard time
    // TODO:
    // accurate iff the user's varray Vec does not reallocate during
    // tracing — true for the 6 macro examples
    // (`vec![0i64; program.len()]` is fixed-capacity).  Dynamic
    // varrays would need per-mutation refresh hooks.
    let initialize_sym_virt_array_parts: Vec<TokenStream> = virt_arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let ptr_value_name = quote::format_ident!("{}_ptr_value", f.name);
            let len_value_name = quote::format_ident!("{}_len_value", f.name);
            quote! {
                // Vable base identity (`&state`), matching `extract_live` and
                // `standard_virtualizable_concrete` so the traced vable box is
                // recognized as the standard virtualizable (no force).
                sym.#ptr_value_name = self as *const Self as i64;
                sym.#len_value_name = self.#fname.len() as i64;
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

    // ── ref(T) scalars ──
    // A ref scalar mints `InputArgRef(__offset)` in the same flat position
    // space as every other inputarg (the virt-array identity slot above
    // does the same).  The optimizer keys inputarg identity by flat
    // position (`OptContext::inputarg_refs[pos]`, `bind_canonical_inputarg`
    // keyed by `OpRef::raw()`), so a bank-local 0-based index would alias
    // `InputArgInt(0)` and `InputArgRef(0)` onto one forwarding host — a
    // promote on the int slot would const-fold every use of the ref slot.
    // In the value vector the ref scalar is APPENDED LAST (after int
    // scalars/arrays/virt) so the int-bank slot layout is unchanged and the
    // flat offset coincides with the value-vector position;
    // `live_value_types` tags those trailing positions `Type::Ref` so
    // `restore_values` routes them to the ref bank.  Struct storage is a
    // `usize` carrier (raw GcRef / pointer bits).
    let sym_ref_scalar_fields: Vec<TokenStream> = ref_scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { #fname: majit_ir::OpRef, }
        })
        .collect();
    let sym_ref_scalar_value_fields: Vec<TokenStream> = ref_scalars
        .iter()
        .map(|(_, f)| {
            let value_name = quote::format_ident!("{}_value", f.name);
            quote! { #value_name: i64, }
        })
        .collect();
    let create_sym_ref_scalar_inits: Vec<TokenStream> = ref_scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let value_name = quote::format_ident!("{}_value", f.name);
            quote! {
                let #fname = majit_ir::OpRef::input_arg_ref(__offset as u32);
                __offset += 1;
                let #value_name = 0i64;
            }
        })
        .collect();
    let create_sym_ref_scalar_names: Vec<&syn::Ident> =
        ref_scalars.iter().map(|(_, f)| &f.name).collect();
    let create_sym_ref_scalar_value_names: Vec<syn::Ident> = ref_scalars
        .iter()
        .map(|(_, f)| quote::format_ident!("{}_value", f.name))
        .collect();
    let extract_ref_scalar_parts: Vec<TokenStream> = ref_scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { values.push(self.#fname as i64); }
        })
        .collect();
    let restore_ref_scalar_parts: Vec<TokenStream> = ref_scalars
        .iter()
        .enumerate()
        .map(|(ref_idx, (_, f))| {
            let fname = &f.name;
            // `live_value_types` routes one `Ref` per virt array (the shared
            // `&state` identity) into the ref bank ahead of the ref scalars, so
            // ref scalar `j` lives at `ref_values[num_virt_arrays + j]`, not
            // `ref_values[j]`.  Mirrors `populate_ref_scalar_parts`, which skips
            // the same prefix in the register bank via `ref_identity_base`.
            let slot = num_virt_arrays + ref_idx;
            quote! { self.#fname = ref_values[#slot] as usize; }
        })
        .collect();
    let initialize_sym_ref_scalar_parts: Vec<TokenStream> = ref_scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let value_name = quote::format_ident!("{}_value", f.name);
            quote! { sym.#value_name = self.#fname as i64; }
        })
        .collect();
    let collect_ref_scalar_parts: Vec<TokenStream> = ref_scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { args.push(sym.#fname); }
        })
        .collect();
    // Canonical ref-bank identity slots: ref scalar `j` lives at
    // `ref_regs[ref_identity_base + j]` so the guard-time snapshot's
    // live_r decode (`get_list_of_active_boxes`) and the blackhole's
    // `ref_scalar_slot` agree without aliasing the dispatch JitCode's
    // ref arguments.
    let populate_ref_scalar_parts: Vec<TokenStream> = ref_scalars
        .iter()
        .enumerate()
        .map(|(ref_idx, (_, f))| {
            let fname = &f.name;
            let value_name = quote::format_ident!("{}_value", f.name);
            let slot = ref_identity_base + ref_idx;
            quote! {
                if #slot < frame.ref_regs.len() {
                    frame.ref_regs[#slot] = Some(self.#fname);
                    frame.ref_values[#slot] = Some(self.#value_name);
                }
            }
        })
        .collect();
    // Override `JitState::collect_jump_args_with_boxes` when the state has any
    // `[int; virt]` array (tl/tlc/tla/braininterp + multi-array interps). With
    // 0 virt arrays (tlr/tinyframe) the defaulted trait method (scalars + fixed
    // arrays) is already correct, so the `else` is the 0-array case.
    //
    // The JUMP must be slot-for-slot identical to the trace-entry Label, whose
    // virt-array inputargs are all the `<arr>_ptr`/`<arr>_len` headers (minted
    // by `create_sym` advancing `__offset`, declaration order) followed by the
    // element boxes (minted by `initialize_virtualizable` at
    // `num_reds..num_reds+total`). `__boxes` =
    // `TraceCtx::collect_virtualizable_boxes()` =
    // `[arr0_elem0.., arr1_elem0.., .., identity]` (`num_static_extra_boxes==0`
    // for state-field; `initialize_virtualizable` concatenates the arrays in
    // declaration order; identity LAST). So push every header first (loop-
    // invariant identity bases + lengths), then the whole element shadow once,
    // dropping the trailing identity (pyjitpl.py:2982-2989 `live_arg_boxes +=
    // virtualizable_boxes; live_arg_boxes.pop()`). The element block is already
    // in per-array order, so a single contiguous splice reproduces the Label
    // for any number of arrays; pushing it once-per-array (or putting elements
    // before a later array's header) would shift every later slot and break the
    // unroll's Label↔Jump virtual-state match.
    let collect_jump_args_with_boxes_method: TokenStream = if num_virt_arrays >= 1 {
        quote! {
            fn collect_jump_args_with_boxes(
                sym: &__JitSym,
                __boxes: &[majit_ir::OpRef],
            ) -> Vec<majit_ir::OpRef> {
                let mut args = Vec::new();
                #(#collect_scalar_parts)*
                #(#collect_array_parts)*
                #(#collect_virt_array_parts)*
                let __elem_count = __boxes.len().saturating_sub(1);
                for __i in 0..__elem_count {
                    args.push(__boxes[__i]);
                }
                #(#collect_ref_scalar_parts)*
                args
            }
        }
    } else {
        quote! {}
    };
    let fail_ref_scalar_parts: Vec<TokenStream> = ref_scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { args.push(self.#fname); }
        })
        .collect();
    let state_ref_field_ref_arms: Vec<TokenStream> = ref_scalars
        .iter()
        .enumerate()
        .map(|(ref_idx, (_, f))| {
            let fname = &f.name;
            quote! { #ref_idx => Some(self.#fname), }
        })
        .collect();
    let set_state_ref_field_ref_arms: Vec<TokenStream> = ref_scalars
        .iter()
        .enumerate()
        .map(|(ref_idx, (_, f))| {
            let fname = &f.name;
            quote! { #ref_idx => { self.#fname = value; } }
        })
        .collect();
    let state_ref_field_value_arms: Vec<TokenStream> = ref_scalars
        .iter()
        .enumerate()
        .map(|(ref_idx, (_, f))| {
            let value_name = quote::format_ident!("{}_value", f.name);
            quote! { #ref_idx => Some(self.#value_name), }
        })
        .collect();
    let set_state_ref_field_value_arms: Vec<TokenStream> = ref_scalars
        .iter()
        .enumerate()
        .map(|(ref_idx, (_, f))| {
            let value_name = quote::format_ident!("{}_value", f.name);
            quote! { #ref_idx => { self.#value_name = value; } }
        })
        .collect();

    // Optional method overrides — emitted ONLY when ref scalars exist, so
    // interps with none generate a byte-identical token stream (the trait
    // defaults from `JitState` / `JitCodeSym` apply).
    // Per-virt-array value-routing types: the identity pointer slot is Ref,
    // the length slot is Int (in `extract_live` ptr-then-len order).
    let virt_array_type_parts: Vec<TokenStream> = virt_arrays
        .iter()
        .map(|_| {
            quote! {
                types.push(majit_ir::Type::Ref);
                types.push(majit_ir::Type::Int);
            }
        })
        .collect();
    // Per-array value-routing types: one Int per element.
    let array_type_parts: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! {
                for _ in 0..self.#fname.len() {
                    types.push(majit_ir::Type::Int);
                }
            }
        })
        .collect();
    let live_value_types_override: TokenStream = if num_ref_scalars > 0 || num_virt_arrays > 0 {
        quote! {
            fn live_value_types(&self, _meta: &__JitMeta) -> Vec<majit_ir::Type> {
                // Value-routing types in `extract_live` order: int scalars,
                // int array elements, then per virt-array the identity ptr
                // (Ref) + length (Int), then the appended ref scalars (Ref).
                // The ptr slot MUST be Ref so the folded loop-invariant
                // `&state` identity numbers as a ref const (TAGCONST), not an
                // int const — the resume reader decodes it through `decode_ref`
                // in both the vable section and the frame ref-liveness.
                let mut types: Vec<majit_ir::Type> = Vec::new();
                for _ in 0..#num_scalars {
                    types.push(majit_ir::Type::Int);
                }
                #(#array_type_parts)*
                #(#virt_array_type_parts)*
                for _ in 0..#num_ref_scalars {
                    types.push(majit_ir::Type::Ref);
                }
                types
            }
        }
    } else {
        quote! {}
    };
    let restore_banked_override: TokenStream = if num_ref_scalars > 0 {
        quote! {
            fn restore_banked(
                &mut self,
                meta: &__JitMeta,
                int_values: &[i64],
                ref_values: &[i64],
            ) {
                // Int scalars/arrays/virt restore from the int bank exactly
                // as `restore`; ref scalars restore from the ref bank by
                // 0-based ref index.
                self.restore(meta, int_values);
                #(#restore_ref_scalar_parts)*
            }
        }
    } else {
        quote! {}
    };
    // Emit the virt-array write-back override only for consumers that declare a
    // `[.. ; virt]` array; 0-array interps (aheui, tlr, tinyframe, i64env) keep
    // the trait default no-op, leaving their generated impl byte-identical.
    let writeback_virt_array_override: TokenStream = if num_virt_arrays > 0 {
        quote! {
            fn writeback_virt_array_state_fields_from_values(&mut self, values: &[i64]) {
                let mut __offset: usize = 0;
                #(#writeback_virt_array_parts)*
            }
        }
    } else {
        quote! {}
    };
    let ref_field_accessor_overrides: TokenStream = if num_ref_scalars > 0 {
        quote! {
            fn state_ref_field_ref(&self, field_idx: usize) -> Option<majit_ir::OpRef> {
                match field_idx {
                    #(#state_ref_field_ref_arms)*
                    _ => None,
                }
            }
            fn set_state_ref_field_ref(&mut self, field_idx: usize, value: majit_ir::OpRef) {
                match field_idx {
                    #(#set_state_ref_field_ref_arms)*
                    _ => {}
                }
            }
            fn state_ref_field_value(&self, field_idx: usize) -> Option<i64> {
                match field_idx {
                    #(#state_ref_field_value_arms)*
                    _ => None,
                }
            }
            fn set_state_ref_field_value(&mut self, field_idx: usize, value: i64) {
                match field_idx {
                    #(#set_state_ref_field_value_arms)*
                    _ => {}
                }
            }

            fn ref_identity_slots_end(&self) -> usize {
                #ref_identity_end
            }
        }
    } else {
        quote! {}
    };
    let state_field_layout_ctor: TokenStream = if num_ref_scalars > 0 {
        quote! {
            majit_metainterp::blackhole::StateFieldLayout::with_ref_scalars(
                #num_scalars,
                ::std::vec![#(self.#create_sym_array_names.len()),*],
                #num_virt_arrays,
                #num_ref_scalars,
                #ref_identity_base,
                #int_identity_base,
            )
        }
    } else {
        quote! {
            majit_metainterp::blackhole::StateFieldLayout::new(
                #num_scalars,
                ::std::vec![#(self.#create_sym_array_names.len()),*],
                #num_virt_arrays,
                #int_identity_base,
            )
        }
    };

    // ── VirtualizableInfo / heap-ptr overrides for `[int; virt]` arrays ──
    // Each virt array becomes a standard-virtualizable RustVec array field on
    // a zero-static-field vinfo, so `state.<arr>[i]` lowers through the
    // `virtualizable_boxes` devirt path. Scalars stay in the state-field
    // scalar resume mechanism (disjoint from the array restore).
    let build_vinfo_override: TokenStream = if num_virt_arrays > 0 {
        // Per virt array: nested data-ptr/len extractor fns + an
        // `add_rust_vec_array_field` call keyed on the field byte offset.
        let virt_array_field_parts: Vec<TokenStream> = virt_arrays
            .iter()
            .map(|(_, f)| {
                let fname = &f.name;
                let data_ptr_fn = quote::format_ident!("__vinfo_{}_data_ptr", f.name);
                let len_fn = quote::format_ident!("__vinfo_{}_len", f.name);
                let fname_str = f.name.to_string();
                quote! {
                    fn #data_ptr_fn(__p: *mut u8) -> *mut i64 {
                        unsafe { (*(__p as *mut #state_type)).#fname.as_mut_ptr() }
                    }
                    fn #len_fn(__p: *const u8) -> usize {
                        unsafe { (*(__p as *const #state_type)).#fname.len() }
                    }
                    let __descr = majit_ir::descr::make_array_descr(
                        0,
                        ::std::mem::size_of::<i64>(),
                        majit_ir::Type::Int,
                    );
                    __info.add_rust_vec_array_field(
                        #fname_str,
                        majit_ir::Type::Int,
                        ::std::mem::offset_of!(#state_type, #fname),
                        #data_ptr_fn,
                        #len_fn,
                        __descr,
                    );
                }
            })
            .collect();
        // Field idents in `virt_arrays` order — the same order
        // `add_rust_vec_array_field` registers them, which is the order
        // `flatten_virtualizable_values` (jitdriver.rs) reads them back.
        let virt_array_field_idents: Vec<&syn::Ident> =
            virt_arrays.iter().map(|(_, f)| &f.name).collect();
        quote! {
            #[allow(non_snake_case)]
            fn __build_virtualizable_info()
            -> Option<::std::sync::Arc<majit_metainterp::virtualizable::VirtualizableInfo>> {
                use majit_metainterp::virtualizable::VirtualizableInfo;
                // token_offset=0: the stack-local state struct is non-GC and
                // never moved, so the vable_token protocol is inert — the
                // identity value (a `&state` pointer) is recovered straight
                // from the resume snapshot, not via a heap token.
                let mut __info = VirtualizableInfo::new(0);
                __info.name = "state".to_string();
                // The dispatch lowering binds the green ref `program` to ref
                // register 0 (it is the base for `program[pc]` reads) and the
                // `&state` virtualizable identity to ref register 1
                // (`vable_input_ref_reg = 1`, jitcode_lower/mod.rs). Tell
                // `initialize_virtualizable` to mint the standard box at that
                // ref-bank index so it matches the traced vable base (the flat
                // `num_green_args + index_of_virtualizable` ordinal would
                // resolve to 0, the slot the green ref occupies).
                __info.identity_ref_bank_index = Some(1);
                #(#virt_array_field_parts)*
                Some(__info.finalize_arc(
                    majit_ir::descr::make_size_descr(::std::mem::size_of::<#state_type>()),
                ))
            }

            fn virtualizable_heap_ptr(
                &self,
                _meta: &Self::Meta,
                _virtualizable: &str,
                _info: &majit_metainterp::virtualizable::VirtualizableInfo,
            ) -> Option<*mut u8> {
                // The state struct is a stack-allocated mainloop local —
                // stable and non-GC. Use its address as the vable identity
                // heap pointer.
                Some(self as *const Self as *mut u8)
            }

            fn export_virtualizable_boxes(
                &self,
                _meta: &Self::Meta,
                _virtualizable: &str,
                _info: &majit_metainterp::virtualizable::VirtualizableInfo,
            ) -> Option<(::std::vec::Vec<i64>, ::std::vec::Vec<::std::vec::Vec<i64>>)> {
                // warmstate.py:482-511: supply the live virtualizable field
                // values so `extend_compiled_live_values` can grow the entry
                // `live_values` to the compiled loop's full inputarg width.
                // The array elements were seeded as boxes by
                // `initialize_virtualizable` at trace start and carried as
                // loop inputargs; re-entry must re-supply them in
                // `flatten_virtualizable_values` order (statics then arrays,
                // each array ascending). Static boxes is empty: scalars stay
                // in the state-field scalar resume mechanism — only
                // `[int; virt]` arrays are virtualized via the vable.
                let __static_boxes: ::std::vec::Vec<i64> = ::std::vec::Vec::new();
                let __array_boxes: ::std::vec::Vec<::std::vec::Vec<i64>> = ::std::vec![
                    #( self.#virt_array_field_idents.iter().map(|&__x| __x as i64).collect() ),*
                ];
                Some((__static_boxes, __array_boxes))
            }
        }
    } else {
        quote! {}
    };

    quote! {
        /// Compiled loop metadata for state_fields mode: flattened array lengths at trace start.
        #[derive(Clone)]
        #[allow(non_camel_case_types)]
        struct __JitMeta {
            #(#meta_fields)*
        }

        impl __JitMeta {
            /// RPython `assembler.py:218-231 get_liveness_info(insn, kind)`
            /// adapted for flat-state JIT: every state_field slot is
            /// permanently live, so the canonical `(live_i, live_r,
            /// live_f)` triple is `live_i = 0..total_slots` in the int
            /// bank (int scalars, fixed-array elements, virt-array
            /// ptr/len) plus `live_r = 0..num_ref_scalars` for any
            /// `ref(T)` scalars carried in the ref bank.  `live_f` is
            /// always empty (no float state fields).  Used by
            /// `JitCodeBuilder::live` (`assembler.py:148+158`) to
            /// register the canonical entry once per process and emit
            /// a `live/<offset>` prefix on each per-opcode jitcode.
            #[allow(dead_code)]
            fn canonical_liveness_slots(
                &self,
            ) -> (::std::vec::Vec<u8>, ::std::vec::Vec<u8>, ::std::vec::Vec<u8>) {
                let __array_lens: &[usize] = &[#(#canonical_liveness_array_len_refs),*];
                majit_metainterp::live_slots_for_state_field_jit(
                    #num_scalars,
                    __array_lens,
                    #num_virt_arrays,
                    #num_ref_scalars,
                    #ref_identity_base,
                    #int_identity_base,
                )
            }

            /// RPython `warmspot.py:281-289`'s `make_jitcodes() →
            /// finish_setup(codewriter)` lifecycle reduced to the
            /// canonical-entry slice for state-field JIT
            /// (`pyjitpl.py:2264 self.liveness_info = "".join(
            /// asm.all_liveness)`).  Builds a fresh `Assembler`,
            /// registers the canonical
            /// `(live_i, live_r, live_f)` triple via
            /// `Assembler::_encode_liveness` (`assembler.py:235-248`),
            /// then publishes the resulting `all_liveness` payload
            /// through `JitDriver::install_canonical_liveness`.
            ///
            /// Caller pattern:
            /// ```ignore
            /// let meta = state.build_meta(0, &program);
            /// meta.install_canonical_liveness(&mut driver);
            /// ```
            /// Must run before the first trace — the
            /// `Arc::get_mut` invariant on `MetaInterp::staticdata`
            /// (`pyjitpl.rs::install_canonical_liveness`) panics
            /// once any tracing setup has cloned the Arc.
            ///
            /// This only installs the canonical liveness entry and
            /// opcode ids.  Consumers whose macro-emitted per-pc
            /// JitCodes can register additional liveness entries via
            /// `JitCodeBuilder::finalize_liveness(__asm)` must build
            /// those JitCodes before the first trace, then call
            /// `JitDriver::sync_liveness_info_from_shared_asm()`.  That
            /// reproduces RPython's order: all `-live-` entries are in
            /// `asm.all_liveness` before `finish_setup` snapshots
            /// `metainterp_sd.liveness_info`.
            #[allow(dead_code)]
            fn install_canonical_liveness(
                &self,
                driver: &mut majit_metainterp::JitDriver<#state_type>,
            ) {
                // RPython `codewriter.py:23-24` calls `CallControl.__init__`
                // (`call.py:46-47`) before `assemble()` produces the jitcodes
                // that read `jitdriver_sd.index`. Pyre's analog: stamp the
                // descriptor onto the driver before the dispatch JitCode
                // build below reads jdindex via
                // `driver.index().expect(...)`.
                //
                // `ensure_descriptor_registered` mirrors PyPy's `for
                // index, jd in enumerate(jitdrivers_sd): jd.index = index`
                // — when the consumer constructed the driver via
                // `JitDriver::with_descriptor(threshold, jd)`, that jd
                // (carrying `greens`/`reds`/`virtualizable`/result_type
                // info) is registered in place; only when no descriptor
                // was pre-built does an empty stub get registered as a
                // pyre-only fail-soft.  Idempotent: re-entry is a no-op
                // once `driver.index()` returns `Some(_)`.
                //
                // Slice (audit Issue #5) — populate the JitDriver's
                // green / red schema BEFORE
                // `ensure_descriptor_registered` runs, so the
                // descriptor that gets registered carries the real
                // `(name, IR Type)` pairs from the dispatch
                // JitCode body's `BC_JIT_MERGE_POINT` rather than the
                // empty stub.  `green_kind_counts` / `red_kind_counts`
                // then reflect the actual payload partition.
                #declare_schema_fn_name(driver);
                driver.ensure_descriptor_registered();
                // Register canonical entry +
                // canonical opcode ids into the driver-shared
                // `Assembler` (cf. `JitDriver::shared_asm`) so per-pc
                // factory calls dedup against the same
                // `all_liveness_positions` and append into the same
                // `all_liveness` byte stream.
                let __shared_asm = driver.shared_asm();
                {
                    let mut __asm = __shared_asm
                        .lock()
                        .expect("shared_asm poisoned at install_canonical_liveness");
                    let (__live_i, __live_r, __live_f) = self.canonical_liveness_slots();
                    // Stage the canonical "all-live" triple for lazy
                    // registration. The first leading-dummy `BC_LIVE`
                    // patched by `JitCodeBuilder::finalize_liveness`
                    // calls `ensure_canonical_liveness_offset`, which
                    // registers the triple at the END of `all_liveness`
                    // (after the per-marker prebuild has populated the
                    // IR-walk-ordered head).  Matches RPython
                    // `assembler.assemble`'s shape: per-marker `-live-`
                    // entries occupy the early offsets; pyre's canonical
                    // entry lands at the tail as a leading-dummy
                    // affordance.
                    __asm.set_canonical_liveness_triple(
                        __live_i,
                        __live_r,
                        __live_f,
                    );
                    // RPython `assembler.py:222 self.insns[key] = opnum`
                    // records every opcode the assembler emits during
                    // `assemble()`.  pyre's macro path skips
                    // `assembler.assemble()` (the per-arm `JitCodeBuilder`
                    // emits BC_* directly), so the canonical state-field
                    // JIT entries are registered explicitly here.  The
                    // downstream `MetaInterpStaticData::
                    // install_canonical_liveness` then calls
                    // `setup_insns(asm.insns())` (`pyjitpl.py:2227-2243`)
                    // to dynamically resolve `op_live` /
                    // `op_catch_exception` / `op_*_return` instead of a
                    // parallel hardcoded `BC_*` seeding block.
                    __asm.register_insn("live/", majit_metainterp::jitcode::insns::BC_LIVE);
                    __asm.register_insn(
                        "catch_exception/L",
                        majit_metainterp::jitcode::insns::BC_CATCH_EXCEPTION,
                    );
                    __asm.register_insn(
                        "rvmprof_code/ii",
                        majit_metainterp::jitcode::insns::BC_RVMPROF_CODE,
                    );
                    __asm.register_insn(
                        "int_return/i",
                        majit_metainterp::jitcode::insns::BC_INT_RETURN,
                    );
                    __asm.register_insn(
                        "ref_return/r",
                        majit_metainterp::jitcode::insns::BC_REF_RETURN,
                    );
                    __asm.register_insn(
                        "float_return/f",
                        majit_metainterp::jitcode::insns::BC_FLOAT_RETURN,
                    );
                    __asm.register_insn(
                        "void_return/",
                        majit_metainterp::jitcode::insns::BC_VOID_RETURN,
                    );
                    // RPython `pyjitpl.py:2255 finish_setup` builds every
                    // JitCode and stamps every per-marker `-live-` triple
                    // into `asm.all_liveness` *before* snapshotting
                    // `metainterp_sd.liveness_info`. Pyre's lazy factory
                    // can't eagerly build (pc, op) pairs, so the macro-
                    // generated `__prebuild_jitcode_liveness_*` function
                    // pre-registers each lowered arm's per-marker triples
                    // into the same locked shared assembler. After the
                    // snapshot below, trace-time
                    // `JitCodeBuilder::finalize_liveness` only dedups —
                    // the table never grows past this point (asserted in
                    // `__trace_*`).
                    #prebuild_fn_name(&mut __asm);
                    // Build the dispatch JitCode singleton against the
                    // same shared assembler. `__prebuild_jitcode_liveness_*`
                    // registers per-marker triples for both the dispatch
                    // JitCode and every per-arm JitCode, so the
                    // `finalize_liveness` calls inside this factory only
                    // dedup — they do not grow `asm.all_liveness` past the
                    // prebuild snapshot. Mirrors `pyjitpl.py:2264
                    // finish_setup`, where `metainterp_sd.liveness_info`
                    // is snapshotted only after every JitCode has been
                    // built and every `-live-` triple stamped.
                    // Single-phase jdindex resolution (jtransform.py:1704):
                    // `register_descriptor` ran above (line 689 onwards),
                    // unconditionally stamping the index on the driver
                    // before this point. Read it through the now-`Some`
                    // accessor and bake it into the dispatch JitCode body.
                    //
                    // Codex Pre-A.3 review BLOCKER (a) absorption: a fake
                    // `0` index must never end up baked into a registered
                    // JitCode body. With `register_descriptor` ordered
                    // before this read, the `expect()` is a structural
                    // invariant — it can fire only if a future change
                    // accidentally moves the registration after this site.
                    let __jdindex: i64 = driver.index().expect(
                        "register_descriptor must run before install_canonical_liveness — \
                         RPython call.py:46-47 / codewriter.py:23-24 lifecycle invariant"
                    ) as i64;
                    let __dispatch_jc_opt = #dispatch_jitcode_fn_name(&mut __asm, __jdindex);
                    // Safety net: ensure the canonical entry has a
                    // registered offset before the snapshot, even if no
                    // per-pc factory has run a `finalize_liveness` yet
                    // to trigger the lazy registration. Subsequent calls
                    // short-circuit via the cached
                    // `canonical_liveness_offset`.
                    let _ = __asm.ensure_canonical_liveness_offset();
                    driver.install_canonical_liveness(&__asm);
                    // PyPy `make_jitcodes()` / `pyjitpl.py:2255
                    // finish_setup()` only install completed jitcodes —
                    // there is no path where a body that the codewriter
                    // failed to lower lands as a successfully-installed
                    // singleton.  When `lower_dispatch_body` returned
                    // `None` at proc-macro time, the dispatch builder
                    // returns `None` here; skip
                    // `register_dispatch_jitcode` to match that
                    // lifecycle.  Successful builds (`Some(jc)`) install
                    // unconditionally per PyPy
                    // `pypy/module/pypyjit/interp_jit.py:82-94`.
                    if let Some(__dispatch_jc) = __dispatch_jc_opt {
                        driver.register_dispatch_jitcode(__dispatch_jc);
                    }
                }
            }
        }

        #recursive_fresh_alloc_free_fns

        /// Symbolic state during tracing: per-field OpRefs.
        #[allow(non_camel_case_types)]
        struct __JitSym {
            #(#sym_scalar_fields)*
            #(#sym_scalar_value_fields)*
            #(#sym_array_fields)*
            #(#sym_array_value_fields)*
            #(#sym_virt_array_fields)*
            #(#sym_ref_scalar_fields)*
            #(#sym_ref_scalar_value_fields)*
            loop_header_pc: usize,
            trace_started: bool,
        }

        impl majit_metainterp::JitCodeSym for __JitSym {
            fn total_slots(&self) -> usize {
                #num_scalars #(#total_slots_array_parts)* + #num_virt_arrays * 2
            }

            fn int_identity_slots_end(&self) -> usize {
                #int_identity_base + self.total_slots()
            }

            fn int_identity_slots_base(&self) -> usize {
                #int_identity_base
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

            #ref_field_accessor_overrides

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

            fn fail_args(&self) -> Option<Vec<majit_ir::OpRef>> {
                let mut args = Vec::new();
                #(#fail_scalar_parts)*
                #(#fail_array_parts)*
                #(#fail_virt_array_parts)*
                #(#fail_ref_scalar_parts)*
                Some(args)
            }

            #[allow(unused_assignments, unused_variables)]
            fn populate_frame_int_regs(
                &self,
                frame: &mut majit_metainterp::MIFrame,
            ) {
                // Slot layout matches `live_slots_for_state_field_jit`
                // Scalars at `int_identity_base..base+num_scalars`
                // (the base keeps the dispatch JitCode's `pc` argument
                // at i0 out of the seeded range),
                // then flattened arrays, then virt-array (ptr, len)
                // pairs.  Virt-array value mirrors are cached at
                // `JitState::initialize_sym` time
                // from the
                // user state's `<varr>.as_ptr()` / `<varr>.len()`,
                // accurate iff the Vec does not reallocate during
                // tracing.
                let mut __slot: usize = #int_identity_base;
                #(#populate_scalar_parts)*
                #(#populate_array_parts)*
                #(#populate_virt_array_parts)*
                let _ = __slot;
                #(#populate_ref_scalar_parts)*
            }

            #[allow(unused_assignments, unused_variables)]
            fn seed_recursive_fresh_frame(
                &self,
                frame: &mut majit_metainterp::MIFrame,
            ) {
                let mut __slot: usize = #int_identity_base;
                #(#seed_scalar_parts)*
                #(#seed_array_parts)*
                #(#seed_virt_array_parts)*
                let _ = __slot;
            }

            fn snapshot_inline_scalar_state(&self) -> Option<Vec<(majit_ir::OpRef, i64)>> {
                let mut __out: Vec<(majit_ir::OpRef, i64)> = Vec::new();
                #(#snapshot_scalar_parts)*
                #(#snapshot_array_parts)*
                Some(__out)
            }

            fn reset_inline_scalar_state_fresh(&mut self) {
                #(#reset_scalar_parts)*
                #(#reset_array_parts)*
            }

            #[allow(unused_assignments, unused_variables)]
            fn restore_inline_scalar_state(&mut self, __snapshot: Vec<(majit_ir::OpRef, i64)>) {
                let mut __k: usize = 0;
                #(#restore_inline_scalar_parts)*
                #(#restore_inline_array_parts)*
            }

            #recursive_fresh_entry_reds_override

            #recursive_fresh_alloc_free_targets_override
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
                #(#extract_ref_scalar_parts)*
                values
            }

            #live_value_types_override

            fn create_sym(meta: &__JitMeta, header_pc: usize) -> __JitSym {
                let mut __offset: usize = 0;
                #(#create_sym_scalar_inits)*
                #(#create_sym_array_inits)*
                #(#create_sym_virt_array_inits)*
                #(#create_sym_ref_scalar_inits)*
                __JitSym {
                    #(#create_sym_scalar_names,)*
                    #(#create_sym_scalar_value_names,)*
                    #(#create_sym_array_names,)*
                    #(#create_sym_array_value_names,)*
                    #(#create_sym_virt_array_ptr_names,)*
                    #(#create_sym_virt_array_len_names,)*
                    #(#create_sym_virt_array_ptr_value_names,)*
                    #(#create_sym_virt_array_len_value_names,)*
                    #(#create_sym_ref_scalar_names,)*
                    #(#create_sym_ref_scalar_value_names,)*
                    loop_header_pc: header_pc,
                    trace_started: false,
                }
            }

            fn initialize_sym(&self, sym: &mut __JitSym, _meta: &__JitMeta) {
                #(#initialize_sym_scalar_parts)*
                #(#initialize_sym_array_parts)*
                #(#initialize_sym_virt_array_parts)*
                #(#initialize_sym_ref_scalar_parts)*
            }

            // ── Part A (bridge resume-decode). ──
            //
            // resume.py:1042-1057 rebuild_from_resumedata parity for the
            // JitDriver state.  Without this the trait default returns None and
            // `start_bridge_tracing` aborts (jitdriver.rs:3789) so no guard-exit
            // bridge ever forms — a failing loop guard re-enters via
            // ContinueRunningNormally instead of forming a bridge.  Adding it
            // flips `start_bridge_tracing` ok=false→ok=true and bridges form;
            // aheui hello/99bottles/fib stay byte-identical.
            //
            // NOTE: this is general guard-exit bridge-formation infrastructure.
            // It does NOT address the logo `--jit` hang: that was root-caused
            // to a separate optimizer issue (a sel=4 peeled loop constant-folds
            // the stacksize red and drops the stack-size exit guard, looping on
            // a stack-mutating residual) — not a missing or unseeded bridge.
            // See `aheui-logo-spin-observer-replay-rootcause.md`.
            fn rebuild_from_resumedata(
                _meta: &mut __JitMeta,
                fail_arg_types: &[majit_ir::Type],
                storage: Option<&std::sync::Arc<majit_metainterp::resume::ResumeStorage>>,
            ) -> Option<majit_metainterp::ResumeDataResult> {
                // The macro mainloop trace is single-frame (storage/helper calls
                // are residuals, not inlined traced sub-frames), so the generic
                // single-frame fallback (None frame_value_count) recovers the one
                // frame's full register slice from rd_numb.
                //
                // PART B TODO: the single-frame fallback returns the frame in
                // OPENCODER order (greens+reds interleaved); `setup_bridge_sym`
                // must map that to the sym's red slots.  A `frame_value_count`
                // callback (pyre `frame_value_count_at` parity) may be needed
                // here once part B lands.
                let storage = storage?;
                let rd_numb = storage.rd_numb.as_slice();
                let rd_consts = storage.rd_consts();
                let (num_failargs, vable_values, vref_values, frames) =
                    majit_ir::resumedata::rebuild_from_numbering(
                        rd_numb,
                        rd_consts,
                        fail_arg_types,
                        None,
                        storage.rd_virtuals.len(),
                    );
                if frames.is_empty() {
                    return None;
                }
                Some(majit_metainterp::ResumeDataResult {
                    frames,
                    virtualizable_values: vable_values,
                    virtualref_values: vref_values,
                    storage: Some(storage.clone()),
                    num_failargs,
                    fail_arg_types: fail_arg_types.to_vec(),
                })
            }

            // ── Part B (bridge sym seeding) — resume.py:1042/1054 setup_bridge_sym
            //    + consume_boxes parity for the JitDriver state.  Seeds each red
            //    slot's symbolic OpRef + concrete shadow from the guard's decoded
            //    resume frame so the bridge specializes its guards on the real
            //    loop state (without this a guard-exit bridge re-traces
            //    un-seeded). ──
            //
            // STATUS: this seeds int/ref SCALAR slots only.  No available aheui
            // workload has been observed to reach this path (setup_bridge_sym
            // emits no MAJIT_BRIDGE_DEBUG lines for 99bottles/fibonacci/
            // factorial), so part B is present-but-unexercised; treat the
            // seeding as unverified on real traces.  Latent gaps once it IS
            // exercised by a consumer:
            //   * flattened `[int]` arrays + virt-array ptr/len slots are NOT
            //     seeded — they keep `create_sym`'s positional InputArg indices,
            //     which need not equal the decoded failarg index; seed them from
            //     `reg_indices` like the scalars below (consume_boxes fills all
            //     live int/ref/float registers, not just selected scalars);
            //   * a multi-frame resume (inlined sub-frames) is decoded as a
            //     single frame by `rebuild_from_resumedata` (None
            //     frame_value_count) — later frame headers would be read as
            //     values.  Both bite only for non-aheui macro states.
            //
            // `frame.values` is laid out by liveness bank: [int-bank, then
            // ref-bank, then float], greens/loop-invariants decoded as `Const`,
            // live reds as `Box(n)`.  The optimizer renumbers the int bank, so
            // the i-th int Box is NOT necessarily int scalar i — routing by a
            // naive kind-counter mis-binds (e.g. pool_ptr → stacksize slot,
            // deref-crashing on the swapped pointer).  Instead we read the
            // per-bank live REGISTER index of each value from the guard jitcode's
            // liveness (`ctx.bridge_reg_indices()`, stashed by
            // `start_bridge_tracing`) and map register → decl slot via
            // IDENTITY-SLOT MATCHING: each state field is read during tracing
            // by `load_state_field(fi)` (lower_vable.rs) from its FIXED identity
            // register `int_identity_base + fi` (refs: `ref_identity_base + fi`),
            // and that slot is kept live across guards, so the resume frame
            // carries the field's current value at exactly that register.  We
            // therefore locate, for each decl slot k, the frame value whose live
            // register == identity_base + k (NOT a positional/kind-counter zip —
            // the optimizer puts recomputed temps like stacksize's `.size`
            // reload at high working registers, and promotes loop-invariant
            // fields like `selected` to `Const`, so only the identity register
            // reliably names the field).  Box → `input_arg(n)` + `fail_values[n]`;
            // Const → a folded pool constant.  MAJIT_BRIDGE_DEBUG dumps it.
            fn setup_bridge_sym(
                sym: &mut __JitSym,
                ctx: &mut majit_metainterp::TraceCtx,
                resume_data: &majit_metainterp::ResumeDataResult,
                _rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
                fail_values: &[i64],
                fail_types: &[majit_ir::Type],
            ) {
                use majit_ir::resumedata::RebuiltValue;
                use majit_metainterp::JitCodeSym as _;
                let frame = match resume_data.frames.first() {
                    Some(f) => f,
                    None => return,
                };
                let __dbg = std::env::var("MAJIT_BRIDGE_DEBUG").is_ok();
                // Clone so `ctx` is free for `const_int`/`const_ref` below.
                let reg_indices = match ctx.bridge_reg_indices() {
                    Some(r) => r.clone(),
                    None => {
                        if __dbg {
                            eprintln!("[bridgeB] no reg_indices stashed — declining to seed");
                        }
                        return;
                    }
                };
                if __dbg {
                    eprintln!(
                        "[bridgeB] frame jc={} pc={} fail_values={:?} fail_types={:?}",
                        frame.jitcode_index, frame.pc, fail_values, fail_types
                    );
                    eprintln!(
                        "[bridgeB] reg_indices int={:?} ref={:?} float={:?} values.len={} int_base={} ref_base={}",
                        reg_indices.int,
                        reg_indices.ref_,
                        reg_indices.float,
                        frame.values.len(),
                        #int_identity_base,
                        #ref_identity_base
                    );
                }
                if reg_indices.total_len() != frame.values.len() {
                    if __dbg {
                        eprintln!("[bridgeB] reg_indices/frame length mismatch — declining");
                    }
                    return;
                }
                let __ref_off = reg_indices.int.len();
                // int scalars: identity register = int_identity_base + k.
                for __k in 0..#num_scalars {
                    let __target = #int_identity_base + __k;
                    let __pos = reg_indices.int.iter().position(|&r| r as usize == __target);
                    let __pos = match __pos {
                        Some(p) => p,
                        None => continue,
                    };
                    match &frame.values[__pos] {
                        RebuiltValue::Box(n, kind) if matches!(kind, majit_ir::Type::Int) => {
                            let (__op, __shadow) = majit_metainterp::bridge_decode_red(
                                *n, *kind, fail_values, fail_types,
                            );
                            sym.set_state_field_ref(__k, __op);
                            sym.set_state_field_value(__k, __shadow);
                            if __dbg {
                                eprintln!("  int scalar {} <- reg {} Box {} = {}", __k, __target, n, __shadow);
                            }
                        }
                        RebuiltValue::Const(c) => {
                            let __bits = c.as_raw_i64();
                            let __op = ctx.const_int(__bits);
                            sym.set_state_field_ref(__k, __op);
                            sym.set_state_field_value(__k, __bits);
                            if __dbg {
                                eprintln!("  int scalar {} <- reg {} Const {}", __k, __target, __bits);
                            }
                        }
                        _ => {}
                    }
                }
                // ref scalars: identity register = ref_identity_base + j.
                for __j in 0..#num_ref_scalars {
                    let __target = #ref_identity_base + __j;
                    let __pos = reg_indices.ref_.iter().position(|&r| r as usize == __target);
                    let __pos = match __pos {
                        Some(p) => p,
                        None => continue,
                    };
                    match &frame.values[__ref_off + __pos] {
                        RebuiltValue::Box(n, kind) if matches!(kind, majit_ir::Type::Ref) => {
                            let (__op, __shadow) = majit_metainterp::bridge_decode_red(
                                *n, *kind, fail_values, fail_types,
                            );
                            sym.set_state_ref_field_ref(__j, __op);
                            sym.set_state_ref_field_value(__j, __shadow);
                            if __dbg {
                                eprintln!("  ref scalar {} <- reg {} Box {} = {:#x}", __j, __target, n, __shadow);
                            }
                        }
                        RebuiltValue::Const(c) => {
                            let __bits = c.as_raw_i64();
                            let __op = ctx.const_ref(__bits);
                            sym.set_state_ref_field_ref(__j, __op);
                            sym.set_state_ref_field_value(__j, __bits);
                            if __dbg {
                                eprintln!("  ref scalar {} <- reg {} Const {:#x}", __j, __target, __bits);
                            }
                        }
                        _ => {}
                    }
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

            #restore_banked_override

            fn recover_after_compiled_run(&mut self) {
                #recover_body
            }

            fn collect_scalar_state_field_values(sym: &Self::Sym) -> Vec<i64> {
                let mut values = Vec::new();
                #(#collect_scalar_values_parts)*
                values
            }

            fn writeback_scalar_state_fields_from_values(&mut self, values: &[i64]) {
                #(#writeback_from_values_parts)*
            }

            #writeback_virt_array_override

            fn state_field_layout(&self) -> majit_metainterp::blackhole::StateFieldLayout {
                // Flat slot layout for blackhole resume: scalar count is
                // static, each flattened fixed `[int]` array contributes its
                // live length, and each virt array contributes two slots
                // (ptr, len).  Ref scalars add a parallel 0-based ref-bank
                // count.  Mirrors `extract_live` / the canonical
                // `live_slots_for_state_field_jit` ordering.
                #state_field_layout_ctor
            }

            fn collect_jump_args(sym: &__JitSym) -> Vec<majit_ir::OpRef> {
                let mut args = Vec::new();
                #(#collect_scalar_parts)*
                #(#collect_array_parts)*
                #(#collect_virt_array_parts)*
                #(#collect_ref_scalar_parts)*
                args
            }

            #collect_jump_args_with_boxes_method

            fn validate_close(sym: &__JitSym, meta: &__JitMeta) -> bool {
                true #(#validate_array_checks)*
            }

            // State-field JIT
            // override of `JitState::populate_frame_for_guard` so
            // jitdriver-level guard sites (e.g. `force_finish_trace`'s
            // GuardAlwaysFails fallback) get the same snapshot wire-up
            // as the dispatch-level `record_state_guard`
            // (`pyjitpl/dispatch.rs:284`).  Calls the macro-emitted
            // `JitCodeSym::populate_frame_int_regs` to bridge
            // `__JitSym` slots onto `MIFrame.int_regs`, then builds a
            // single-frame snapshot via the canonical helper.
            fn populate_frame_for_guard(
                sym: &__JitSym,
                frames: &mut majit_metainterp::MIFrameStack,
                __op_live: u8,
                __all_liveness: &[u8],
                __virtualizable_boxes: &[majit_ir::OpRef],
                __virtualref_boxes: &[(majit_ir::OpRef, usize)],
                __identity_const: Option<i64>,
            ) -> Option<majit_metainterp::recorder::Snapshot> {
                use majit_metainterp::JitCodeSym as _;
                if frames.frames.is_empty() {
                    return None;
                }
                let __root = &mut frames.frames[0];
                let __n = sym.int_identity_slots_end().min(__root.int_regs.len());
                let __saved_int_regs: Vec<Option<majit_ir::OpRef>> =
                    __root.int_regs[..__n].to_vec();
                let __saved_int_values: Vec<Option<i64>> =
                    __root.int_values[..__n].to_vec();
                // `populate_frame_int_regs` also seeds the ref-scalar
                // identity slots (`ref_regs[ref_identity_base..]`,
                // `codegen_state.rs` populate_ref_scalar_parts), so the ref
                // bank needs the same transient save/restore the int bank
                // gets — mirrors `record_state_guard`
                // (`pyjitpl/dispatch.rs:959-1019`).  Without it the ref
                // scalars stay clobbered in the live frame after the
                // jitdriver-level GuardAlwaysFails snapshot is built.
                let __rn = sym.ref_identity_slots_end().min(__root.ref_regs.len());
                let __saved_ref_regs: Vec<Option<majit_ir::OpRef>> =
                    __root.ref_regs[..__rn].to_vec();
                let __saved_ref_values: Vec<Option<i64>> =
                    __root.ref_values[..__rn].to_vec();
                sym.populate_frame_int_regs(__root);
                // pyjitpl.py:2586-2610 `capture_resumedata(framestack,
                // virtualizable_boxes, virtualref_boxes,
                // last_snapshot)` — the snapshot must carry the live
                // vable + vref box lists or the resume reader sees
                // empty arrays on guard failure.
                let __snapshot = majit_metainterp::build_state_field_snapshot(
                    frames,
                    __op_live,
                    __all_liveness,
                    false,
                    __virtualizable_boxes,
                    __virtualref_boxes,
                    __identity_const,
                    Some((sym.int_identity_slots_base(), sym.int_identity_slots_end())),
                );
                let __root = &mut frames.frames[0];
                __root.int_regs[..__n].copy_from_slice(&__saved_int_regs);
                __root.int_values[..__n].copy_from_slice(&__saved_int_values);
                __root.ref_regs[..__rn].copy_from_slice(&__saved_ref_regs);
                __root.ref_values[..__rn].copy_from_slice(&__saved_ref_values);
                Some(__snapshot)
            }

            #build_vinfo_override
        }
    }
}
