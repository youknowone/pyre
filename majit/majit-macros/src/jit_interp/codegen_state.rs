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
    // Every module-level item this macro emits is suffixed with the annotated
    // function's name, because the expansion lands in the CALLER's module and two
    // machines in one module would otherwise collide (E0428).  These five were
    // fixed names while the three above were already suffixed; the split was not
    // deliberate.  Note the ceiling: unique names let two machines share a module
    // only when their `state` types DIFFER -- two machines over the same state
    // type still conflict on `impl JitState for #state_type` (E0119), which no
    // naming scheme can fix.
    let meta_ty = format_ident!("__JitMeta_{}", func.sig.ident);
    let sym_ty = format_ident!("__JitSym_{}", func.sig.ident);
    // `_name` because `loop_carried_boxes_fn` is already taken further down by the
    // TokenStream holding the whole function definition; this is only its ident.
    let loop_carried_boxes_fn_name = format_ident!("__jit_loop_carried_boxes_{}", func.sig.ident);
    let fresh_alloc_fn = format_ident!("__majit_recursive_fresh_alloc_{}", func.sig.ident);
    let fresh_free_fn = format_ident!("__majit_recursive_fresh_free_{}", func.sig.ident);
    let sf = config.state_fields.as_ref().unwrap();

    let unsupported_fields: Vec<String> = sf
        .fields
        .iter()
        .filter_map(|f| match &f.kind {
            StateFieldKind::Scalar { ir_type, .. } => {
                let ty = ir_type.to_string();
                if ty == "int" || ty == "float" {
                    None
                } else {
                    Some(format!("{}: {}", f.name, ty))
                }
            }
            StateFieldKind::Array(tp) => {
                let ty = tp.to_string();
                if ty == "int" {
                    None
                } else {
                    Some(format!("{}: {}", f.name, ty))
                }
            }
            StateFieldKind::VirtArray(tp) => {
                let ty = tp.to_string();
                if ty == "int" || ty == "float" {
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
            "state_fields supports int, float, [int], [int; virt], [float; virt], and opaque(T); unsupported: {}",
            unsupported_fields.join(", ")
        );
        return quote! {
            compile_error!(#message);
        };
    }

    // Separate int scalars, flattened arrays, virtualizable arrays, float scalars,
    // and ref scalars.
    let scalars: Vec<_> = sf
        .fields
        .iter()
        .enumerate()
        .filter(
            |(_, f)| matches!(&f.kind, StateFieldKind::Scalar { ir_type, .. } if ir_type == "int"),
        )
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
            StateFieldKind::Scalar { ir_type, .. } if ir_type == "float" => quote! { f64 },
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
    let float_scalars: Vec<_> = sf
        .fields
        .iter()
        .enumerate()
        .filter(|(_, f)| {
            matches!(&f.kind, StateFieldKind::Scalar { ir_type, .. } if ir_type == "float")
        })
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
    // `pyjitpl.py:2984-2989 reached_loop_header` carries the virtualizable
    // exactly ONCE: it is a single red (`warmspot.py:529-538
    // jd.index_of_virtualizable = jitdriver.reds.index(vname)`), and
    // `virtualizable.py:150-153` reads every `[.. ; virt]` array's length off
    // the live object instead of boxing it.  So a state contributes one
    // identity slot however many virt arrays it declares — a presence flag,
    // not a per-array count.
    let has_vable_identity = num_virt_arrays > 0;
    let num_vable_identity_slots = usize::from(has_vable_identity);
    // Int-bank slots a sub-JitCode actually reserves, i.e. the ones the
    // inline-frame snapshot trim may blank. Only `split_dispatch` pushes a
    // sub-JitCode's register allocation past the identity range
    // (`split_identity_floor`), so the reservation — and the trim — is empty
    // without it. See `int_identity_reserved_end` below.
    let num_reserved_identity_slots = if config.split_dispatch {
        num_scalars + num_vable_identity_slots
    } else {
        0
    };
    let num_ref_scalars = ref_scalars.len();
    let num_float_scalars = float_scalars.len();
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
    let float_identity_base: usize = config
        .green_type_tags
        .iter()
        .filter(|tag| {
            matches!(
                tag,
                Some(crate::jit_interp::green_type_tag::GreenTypeTag::Float)
            )
        })
        .count();
    let float_identity_end: usize = float_identity_base + num_float_scalars;

    let recover_body: TokenStream = if let Some(ref recover_path) = config.recover {
        quote! { self.#recover_path(); }
    } else {
        quote! {}
    };

    // `__JitMeta_<fn>` fields: one `{name}_len: usize` per flattened array
    // Virt arrays do NOT store length in meta: `virtualizable.py:150-153` reads
    // each one off the live object, so it is neither a meta field nor a box.
    let meta_fields: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let len_name = quote::format_ident!("{}_len", f.name);
            quote! { #len_name: usize, }
        })
        .collect();

    // `__JitSym_<fn>` fields
    // scalar → OpRef
    // flattened array → Vec<OpRef>
    // virt array → an `i64` length mirror, and nothing else (see
    //   `sym_virt_array_fields` below). This line used to read
    //   `(OpRef, OpRef) for (data_ptr, len)`; no such pair is built here or
    //   anywhere else. A virt array's base address is NEVER materialised as an
    //   SSA value: the vable-relative `getarrayitem_vable_*` /
    //   `setarrayitem_vable_*` ops carry `(fdescr, adescr)` and resolve the base
    //   inside the op, off the live virtualizable. The only OpRef a
    //   virtualizable gets is the single `__vable_identity` below.
    //   Worth stating rather than deleting: read as a promise that the base is
    //   already available, the old wording collapses the design of anything that
    //   needs one.
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
    // Per virt array only a plain `i64` length mirror survives: it feeds the
    // fresh-callee capacity and the debug dumps, and is NEVER an inputarg
    // (`virtualizable.py:150-153` reads the length off the live object).
    let sym_virt_array_fields: Vec<TokenStream> = virt_arrays
        .iter()
        .map(|(_, f)| {
            let len_value_name = quote::format_ident!("{}_len_value", f.name);
            quote! { #len_value_name: i64, }
        })
        .collect();
    // `virtualizable.py:139-144` names the virtualizable once, so one OpRef
    // slot serves every `[.. ; virt]` array on the state.
    let sym_vable_identity_fields: TokenStream = if has_vable_identity {
        quote! {
            __vable_identity: majit_ir::OpRef,
            __vable_identity_value: i64,
        }
    } else {
        quote! {}
    };

    // ── JitCodeSym: total_slots ──
    // num_scalars + sum(flattened array lengths) + num_vable_identity_slots
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

    // ── collect_jump_args: scalars, then flattened arrays, then the vable identity ──
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
    let collect_vable_identity_part: TokenStream = if has_vable_identity {
        quote! { args.push(sym.__vable_identity); }
    } else {
        quote! {}
    };
    // ── populate_frame_int_regs: scalars + flattened arrays ──
    // Matches `live_slots_for_state_field_jit` slot order so a
    // `MIFrame::get_list_of_active_boxes` walk against the canonical
    // liveness entry decodes back the same OpRefs / values that
    // `__JitSym_<fn>` and the macro-emitted `live/<offset>` placeholder
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
    // Virtualizable populate: ONE slot holding the `&state` identity, past the
    // scalars and fixed-array elements, matching
    // `live_slots_for_state_field_jit` and `StateFieldLayout::total_slots`.
    // It occupies an INT slot even though it carries a Ref: the resume reader
    // decodes this bank by the int `live_i` index, and writing the identity to
    // the ref bank instead would desync it, leaving `int_regs[slot]` unset when
    // the guard snapshot is collected.
    let populate_vable_identity_part: TokenStream = if has_vable_identity {
        quote! {
            if __slot < frame.int_regs.len() {
                frame.int_regs[__slot] = Some(self.__vable_identity);
                frame.int_values[__slot] = Some(self.__vable_identity_value);
            }
            __slot += 1;
        }
    } else {
        quote! {}
    };

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
    // The fresh callee's identity is not known at the call site (it allocates
    // its own state), so seed the one identity slot as const 0; the array
    // capacities it re-allocates at come from the caller's `<arr>_len_value`
    // mirrors in `recursive_fresh_entry_reds`, not from a boxed length.
    let seed_vable_identity_part: TokenStream = if has_vable_identity {
        quote! {
            if __slot < frame.int_regs.len() {
                frame.int_regs[__slot] = Some(majit_ir::OpRef::const_int(0));
                frame.int_values[__slot] = Some(0);
            }
            __slot += 1;
        }
    } else {
        quote! {}
    };

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
    let fail_vable_identity_part: TokenStream = if has_vable_identity {
        quote! { args.push(self.__vable_identity); }
    } else {
        quote! {}
    };

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
    // captured in `__JitMeta_<fn>::<arr>_len` (one per flattened array).
    let canonical_liveness_array_len_refs: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let len_name = quote::format_ident!("{}_len", f.name);
            quote! { self.#len_name }
        })
        .collect();

    // ── extract_live: scalars, then flattened array elements, then the vable identity ──
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
    // One value: the virtualizable identity (`&state` ==
    // `virtualizable_heap_ptr`), NOT any array's data pointer —
    // `vable_getarrayitem_*` reaches every element from this base through the
    // storage each field registered, and all `[.. ; virt]` arrays share it.
    // Emitting it once
    // is `virtualizable.py:139-144`; the lengths stay off the red vector
    // (`virtualizable.py:150-153` reads them off the live object).
    let extract_vable_identity_part: TokenStream = if has_vable_identity {
        quote! { values.push(self as *const Self as i64); }
    } else {
        quote! {}
    };
    let debug_scalar_state_parts: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let label = f.name.to_string();
            quote! {
                let _ = ::std::fmt::Write::write_fmt(
                    &mut out,
                    format_args!("  {} = {}\n", #label, self.#fname as i64),
                );
            }
        })
        .collect();
    let debug_array_state_parts: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let label = f.name.to_string();
            quote! {
                let _ = ::std::fmt::Write::write_fmt(
                    &mut out,
                    format_args!("  {} len={}\n", #label, self.#fname.len()),
                );
                for (__i, __v) in self.#fname.iter().enumerate() {
                    let _ = ::std::fmt::Write::write_fmt(
                        &mut out,
                        format_args!("    {}[{}] = {}\n", #label, __i, *__v as i64),
                    );
                }
            }
        })
        .collect();
    let debug_virt_array_state_parts: Vec<TokenStream> = virt_arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let label = f.name.to_string();
            quote! {
                let _ = ::std::fmt::Write::write_fmt(
                    &mut out,
                    format_args!(
                        "  {} len={} vable={:#x}\n",
                        #label,
                        self.#fname.len(),
                        self as *const Self as usize,
                    ),
                );
                for (__i, __v) in self.#fname.iter().enumerate() {
                    let _ = ::std::fmt::Write::write_fmt(
                        &mut out,
                        format_args!("    {}[{}] = {}\n", #label, __i, *__v as i64),
                    );
                }
            }
        })
        .collect();
    let debug_ref_scalar_state_parts: Vec<TokenStream> = ref_scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let label = f.name.to_string();
            quote! {
                let _ = ::std::fmt::Write::write_fmt(
                    &mut out,
                    format_args!("  {} = {:#x}\n", #label, self.#fname as usize),
                );
            }
        })
        .collect();
    let debug_scalar_label_parts: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let label = f.name.to_string();
            quote! { labels.push(::std::string::String::from(#label)); }
        })
        .collect();
    let debug_array_label_parts: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let label = f.name.to_string();
            quote! {
                for __i in 0..self.#fname.len() {
                    labels.push(::std::format!("{}[{}]", #label, __i));
                }
            }
        })
        .collect();
    let debug_vable_identity_label_part: TokenStream = if has_vable_identity {
        quote! { labels.push(::std::string::String::from("<vable>")); }
    } else {
        quote! {}
    };
    let debug_ref_scalar_label_parts: Vec<TokenStream> = ref_scalars
        .iter()
        .map(|(_, f)| {
            let label = f.name.to_string();
            quote! { labels.push(::std::string::String::from(#label)); }
        })
        .collect();

    // Recursive CALL_ASSEMBLER portal entry on the JitCodeSym side
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
            let zero = match &f.kind {
                StateFieldKind::VirtArray(tp) if tp == "float" => quote! { 0.0f64 },
                _ => quote! { 0i64 },
            };
            // Constructed through the backing trait rather than as a `vec![]`,
            // so the field keeps whatever container it was declared with. The
            // target type comes from the field this initializer fills.
            quote! {
                #fname: majit_metainterp::virt_array::VirtArrayBacking::filled(
                    #zero,
                    self.#len_value_name as usize,
                ),
            }
        })
        .collect();
    // Reds in `extract_live` order: int scalars, then flattened fixed-array
    // elements, then the single `&state` identity (Ref) when the state has a
    // virtualizable.  Mirrors `extract_scalar_parts` / `extract_array_parts` /
    // `extract_vable_identity_part` so the fresh reds match the callee loop's
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
    let fresh_entry_vable_identity_push: TokenStream = if has_vable_identity {
        quote! {
            __values.push(majit_ir::Value::Ref(majit_ir::GcRef(__base as usize)));
        }
    } else {
        quote! {}
    };
    // The freshly-boxed `&state` identity feeds the one vable Ref slot; only
    // bound when there is at least one virt array (fixed-array-only reds carry
    // no pointer).
    let fresh_entry_base_let: TokenStream = if has_vable_identity {
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
        if num_ref_scalars == 0 && num_float_scalars == 0 && opaque_fields.is_empty() {
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
                    #fresh_entry_vable_identity_push
                    Some((__values, __fresh as Box<dyn ::core::any::Any>))
                }
            }
        } else {
            quote! {}
        };

    // Recursive CALL_ASSEMBLER portal entry for host allocation and release
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
    if supports_fresh_alloc && num_float_scalars > 0 {
        return quote! {
            compile_error!(
                "state_fields float scalars are not supported with recursive portal fresh allocation yet"
            );
        };
    }
    let recursive_fresh_alloc_free_fns: TokenStream = if supports_fresh_alloc {
        let virt_name = &virt_arrays[0].1.name;
        let virt_zero = match &virt_arrays[0].1.kind {
            StateFieldKind::VirtArray(tp) if tp == "float" => quote! { 0.0f64 },
            _ => quote! { 0i64 },
        };
        quote! {
            #[doc(hidden)]
            #[allow(non_snake_case)]
            extern "C" fn #fresh_alloc_fn(__cap: i64) -> i64 {
                let __fresh: ::std::boxed::Box<#state_type> = ::std::boxed::Box::new(#state_type {
                    #(#fresh_entry_scalar_inits)*
                    // Same backing-trait construction the fresh-reds path uses:
                    // the field keeps whatever container it was declared with,
                    // and the target type comes from the field being filled.
                    #virt_name: majit_metainterp::virt_array::VirtArrayBacking::filled(
                        #virt_zero,
                        __cap as usize,
                    ),
                });
                ::std::boxed::Box::into_raw(__fresh) as i64
            }
            #[doc(hidden)]
            #[allow(non_snake_case)]
            extern "C" fn #fresh_free_fn(__ptr: i64) {
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
                    #fresh_alloc_fn as usize as *const (),
                    #fresh_free_fn as usize as *const (),
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
            let len_value_name = quote::format_ident!("{}_len_value", f.name);
            quote! { let #len_value_name = 0i64; }
        })
        .collect();
    // One inputarg for the whole virtualizable, in the same flat offset space
    // as every other inputarg.  The array elements that follow are carried by
    // `virtualizable_boxes`, so this is the last header slot — which makes the
    // loop's entry contract the same shape as a guard's vable section
    // (`resume.py:1404` + `virtualizable.py:139-144`) and therefore the same
    // shape a bridge entering that loop presents.
    let create_sym_vable_identity_init: TokenStream = if has_vable_identity {
        quote! {
            let __vable_identity = majit_ir::OpRef::input_arg_ref(__offset as u32);
            __offset += 1;
            let __vable_identity_value = 0i64;
        }
    } else {
        quote! {}
    };
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
    let create_sym_vable_identity_field_names: TokenStream = if has_vable_identity {
        quote! {
            __vable_identity,
            __vable_identity_value,
        }
    } else {
        quote! {}
    };
    let create_sym_virt_array_len_value_names: Vec<syn::Ident> = virt_arrays
        .iter()
        .map(|(_, f)| quote::format_ident!("{}_len_value", f.name))
        .collect();

    // ── is_compatible: check flattened array lengths match meta ──
    // Virt arrays always compatible (their length is read off the live object).
    let compat_checks: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let len_name = quote::format_ident!("{}_len", f.name);
            quote! { && self.#fname.len() == meta.#len_name }
        })
        .collect();

    // ── restore: write values back to state fields ──
    // The virtualizable identity slot is skipped (the Vec owns its storage and
    // compiled code already mutated the elements in place).
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
    // caches, so only scalars recover cannot re-derive (a `selected`-style
    // storage index, say) meaningfully carry through.
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
    let writeback_live_scalar_arms: Vec<TokenStream> = scalars
        .iter()
        .enumerate()
        .map(|(idx, (_, f))| {
            let fname = &f.name;
            let rust_ty = scalar_rust_type(&f.kind);
            quote! { #idx => { self.#fname = value as #rust_ty; } }
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
    // Skip the one identity slot — the virt-array data lives on the heap and
    // was already modified in place by compiled code.
    let restore_vable_identity_part: TokenStream = if has_vable_identity {
        quote! { __offset += 1; }
    } else {
        quote! {}
    };
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
            let assign = match &f.kind {
                StateFieldKind::VirtArray(tp) if tp == "float" => {
                    quote! { self.#fname[i] = f64::from_bits(values[__offset + i] as u64); }
                }
                _ => quote! { self.#fname[i] = values[__offset + i]; },
            };
            quote! {
                let __len = self.#fname.len();
                debug_assert!(
                    values.len() >= __offset + __len,
                    "writeback_virt_array: fewer element values than array length",
                );
                for i in 0..__len {
                    #assign
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
    // `<varr>_len_value` mirrors the current `<state>.<varr>` length for the
    // fresh-callee capacity, and `__vable_identity_value` the `&state` identity
    // so `populate_frame_int_regs` can fill the corresponding
    // `MIFrame.int_values` slot without re-reading the live state at guard time
    // TODO:
    // accurate iff the user's varray Vec does not reallocate during
    // tracing — true for the 6 macro examples
    // (`vec![0i64; program.len()]` is fixed-capacity).  Dynamic
    // varrays would need per-mutation refresh hooks.
    let initialize_sym_virt_array_parts: Vec<TokenStream> = virt_arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let len_value_name = quote::format_ident!("{}_len_value", f.name);
            quote! { sym.#len_value_name = self.#fname.len() as i64; }
        })
        .collect();
    // Vable base identity (`&state`), matching `extract_live` and
    // `standard_virtualizable_concrete` so the traced vable box is recognized
    // as the standard virtualizable (no force).
    let initialize_sym_vable_identity_part: TokenStream = if has_vable_identity {
        quote! { sym.__vable_identity_value = self as *const Self as i64; }
    } else {
        quote! {}
    };

    // ── validate_close: flattened array lengths in sym match meta ──
    // Virt arrays always validate (their length is read off the live object).
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
            // `live_value_types` routes the single vable identity `Ref` into
            // the ref bank ahead of the ref scalars, so ref scalar `j` lives at
            // `ref_values[num_vable_identity_slots + j]`, not `ref_values[j]`.
            // Mirrors `populate_ref_scalar_parts`, which skips the same prefix
            // in the register bank via `ref_identity_base`.
            let slot = num_vable_identity_slots + ref_idx;
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
    // Ref-bank mirror of `collect_scalar_values_parts`: the concrete pointer
    // bits the walk carries for each ref state field, in ref-scalar index
    // order. Consumed by the tracing-abort blackhole conversion, which seeds
    // `registers_r[ref_scalar_slot(j)]` before running the aborted framestack.
    let collect_ref_scalar_values_parts: Vec<TokenStream> = ref_scalars
        .iter()
        .map(|(_, f)| {
            let value_name = quote::format_ident!("{}_value", f.name);
            quote! { values.push(sym.#value_name); }
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
    // pyjitpl.py:2981-2989 `live_arg_boxes` — THE loop-carried box list, built
    // in exactly one place because it has two consumers that must agree:
    // `JitState::collect_jump_args_with_boxes` (the closing JUMP) and
    // `JitCodeSym::loop_carried_boxes` (the merge-point registration, i.e. a
    // later cross-loop cut's LABEL). RPython gets that agreement for free —
    // `reached_loop_header` builds one list and passes it to both — and
    // asserts it at pyjitpl.py:3020. Two independent constructions here is
    // what made every purely-virt-array interpreter's nested loop decline on
    // `jump.numargs() != label.numargs()` (compile.py:334).
    //
    // The list must be slot-for-slot identical to the trace-entry Label, whose
    // inputargs are minted by `create_sym` advancing `__offset` in declaration
    // order — int scalars (Int), each fixed `[int]` array's cells (Int), the
    // one `__vable_identity` (Ref), then ref scalars (Ref), then float scalars
    // (Float) — after which `JitDriver::extend_compiled_live_values` appends
    // the element block with `live_values.extend(extra_values)`.
    //
    // So the element block is a strict SUFFIX of the reds, exactly as upstream
    // builds it: `live_arg_boxes = greenboxes + redboxes` and then
    // `live_arg_boxes += self.virtualizable_boxes; live_arg_boxes.pop()`
    // (pyjitpl.py:2981-2989) — `+=` appends, so no element can precede a red.
    // Splicing the elements before the ref/float scalars (as this did until
    // the order was unified) leaves the JUMP and the Label at the SAME arity
    // with different slot meanings, so `jump.numargs() == label.numargs()`
    // (compile.py:334) still passes and nothing downstream catches it.
    //
    // `__boxes` = `TraceCtx::collect_virtualizable_typed_boxes()` =
    // `[arr0_elem0.., arr1_elem0.., .., identity]` (`num_static_extra_boxes==0`
    // for state-field; `initialize_virtualizable` concatenates the arrays in
    // declaration order; identity LAST), so one contiguous splice of
    // `__boxes[..len-1]` reproduces the Label's element tail for any number of
    // arrays. Pushing it once-per-array would interleave the arrays and shift
    // every later slot.
    //
    // With 0 virt arrays (tlr/tinyframe) there is no element shadow to splice,
    // so the body degenerates to `collect_jump_args` and `__boxes` is unused —
    // which is why the `collect_jump_args_with_boxes` override below stays
    // gated on `num_virt_arrays >= 1` (the trait default already delegates to
    // `collect_jump_args` there).
    let typed_scalar_parts: Vec<TokenStream> = scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { args.push((sym.#fname, majit_ir::Type::Int)); }
        })
        .collect();
    let typed_array_parts: Vec<TokenStream> = arrays
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! {
                for &__cell in sym.#fname.iter() {
                    args.push((__cell, majit_ir::Type::Int));
                }
            }
        })
        .collect();
    let typed_vable_identity_part: TokenStream = if has_vable_identity {
        quote! { args.push((sym.__vable_identity, majit_ir::Type::Ref)); }
    } else {
        quote! {}
    };
    let typed_ref_scalar_parts: Vec<TokenStream> = ref_scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { args.push((sym.#fname, majit_ir::Type::Ref)); }
        })
        .collect();
    let typed_float_scalar_parts: Vec<TokenStream> = float_scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { args.push((sym.#fname, majit_ir::Type::Float)); }
        })
        .collect();
    let typed_element_splice: TokenStream = if num_virt_arrays >= 1 {
        quote! {
            let __elem_count = __boxes.len().saturating_sub(1);
            args.extend_from_slice(&__boxes[..__elem_count]);
        }
    } else {
        quote! {}
    };
    let loop_carried_boxes_fn: TokenStream = quote! {
        /// pyjitpl.py:2981-2989 `live_arg_boxes`, typed. See the emit-site
        /// comment in `majit-macros/src/jit_interp/codegen_state.rs`.
        #[allow(unused_variables)]
        fn #loop_carried_boxes_fn_name(
            sym: &#sym_ty,
            __boxes: &[(majit_ir::OpRef, majit_ir::Type)],
        ) -> Vec<(majit_ir::OpRef, majit_ir::Type)> {
            let mut args: Vec<(majit_ir::OpRef, majit_ir::Type)> = Vec::new();
            // The reds, in `extract_live` / `live_value_types` order …
            #(#typed_scalar_parts)*
            #(#typed_array_parts)*
            #typed_vable_identity_part
            #(#typed_ref_scalar_parts)*
            #(#typed_float_scalar_parts)*
            // … then the element block as a strict SUFFIX, which is what
            // `live_arg_boxes += self.virtualizable_boxes; live_arg_boxes.pop()`
            // (pyjitpl.py:2988-2989) makes it, and what the entry contract
            // builds with `live_values.extend(extra_values)`
            // (`JitDriver::extend_compiled_live_values`).
            #typed_element_splice
            args
        }
    };
    let collect_jump_args_with_boxes_method: TokenStream = if num_virt_arrays >= 1 {
        quote! {
            fn collect_jump_args_with_boxes(
                sym: &#sym_ty,
                __boxes: &[(majit_ir::OpRef, majit_ir::Type)],
            ) -> Vec<majit_ir::OpRef> {
                #loop_carried_boxes_fn_name(sym, __boxes)
                    .into_iter()
                    .map(|(__opref, _)| __opref)
                    .collect()
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
    let writeback_live_ref_scalar_arms: Vec<TokenStream> = ref_scalars
        .iter()
        .enumerate()
        .map(|(ref_idx, (_, f))| {
            let fname = &f.name;
            quote! { #ref_idx => { self.#fname = value as usize; } }
        })
        .collect();
    // ── float scalars ──
    //
    // Float scalar concrete shadows are raw f64 bits (`i64`) because
    // `MIFrame.float_values` and `BlackholeInterpreter.registers_f` carry the
    // same bit pattern. Native state stores f64 by default.
    let sym_float_scalar_fields: Vec<TokenStream> = float_scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { #fname: majit_ir::OpRef, }
        })
        .collect();
    let sym_float_scalar_value_fields: Vec<TokenStream> = float_scalars
        .iter()
        .map(|(_, f)| {
            let value_name = quote::format_ident!("{}_value", f.name);
            quote! { #value_name: i64, }
        })
        .collect();
    let create_sym_float_scalar_inits: Vec<TokenStream> = float_scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let value_name = quote::format_ident!("{}_value", f.name);
            quote! {
                let #fname = majit_ir::OpRef::input_arg_float(__offset as u32);
                __offset += 1;
                let #value_name = 0i64;
            }
        })
        .collect();
    let create_sym_float_scalar_names: Vec<&syn::Ident> =
        float_scalars.iter().map(|(_, f)| &f.name).collect();
    let create_sym_float_scalar_value_names: Vec<syn::Ident> = float_scalars
        .iter()
        .map(|(_, f)| quote::format_ident!("{}_value", f.name))
        .collect();
    // `clear_sym_inputarg_bindings`: drop every `OpRef` that `create_sym`
    // minted off `__offset`, and only those — the `_value` / `_len_value`
    // mirrors are concrete runtime data seeded by `initialize_sym`, not
    // positions, so they stay.  One statement per field the `#sym_ty`
    // constructor above lists as `OpRef` / `Vec<OpRef>`; virt arrays carry
    // only an `i64` length mirror and so contribute nothing here.
    // Counterpart of `clear_sym_binding_parts`, over the identical five
    // sources so the two cannot drift: if a kind is added to the clearing and
    // not to the count, the assertion at the call site stops covering it.
    let count_bound_sym_parts: Vec<TokenStream> = create_sym_scalar_names
        .iter()
        .map(|fname| quote! { if !sym.#fname.is_none() { __bound += 1; } })
        .chain(create_sym_array_names.iter().map(|fname| {
            quote! {
                for __cell in sym.#fname.iter() {
                    if !__cell.is_none() { __bound += 1; }
                }
            }
        }))
        .chain(has_vable_identity.then(|| {
            quote! { if !sym.__vable_identity.is_none() { __bound += 1; } }
        }))
        .chain(
            create_sym_ref_scalar_names
                .iter()
                .map(|fname| quote! { if !sym.#fname.is_none() { __bound += 1; } }),
        )
        .chain(
            create_sym_float_scalar_names
                .iter()
                .map(|fname| quote! { if !sym.#fname.is_none() { __bound += 1; } }),
        )
        .collect();
    let clear_sym_binding_parts: Vec<TokenStream> = create_sym_scalar_names
        .iter()
        .map(|fname| quote! { sym.#fname = majit_ir::OpRef::NONE; })
        .chain(create_sym_array_names.iter().map(|fname| {
            quote! {
                for __cell in sym.#fname.iter_mut() {
                    *__cell = majit_ir::OpRef::NONE;
                }
            }
        }))
        .chain(has_vable_identity.then(|| {
            quote! { sym.__vable_identity = majit_ir::OpRef::NONE; }
        }))
        .chain(
            create_sym_ref_scalar_names
                .iter()
                .map(|fname| quote! { sym.#fname = majit_ir::OpRef::NONE; }),
        )
        .chain(
            create_sym_float_scalar_names
                .iter()
                .map(|fname| quote! { sym.#fname = majit_ir::OpRef::NONE; }),
        )
        .collect();
    let extract_float_scalar_parts: Vec<TokenStream> = float_scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            // Widen to f64 before taking bits so the encoding is the 64-bit
            // representation the restore path reads back via
            // `f64::from_bits(_ as u64) as #rust_ty`. A `float(f32)` field's
            // own `to_bits()` yields 32-bit bits, which would round-trip
            // through `f64::from_bits` as a bogus value.
            // Widen to f64 before taking bits so the encoding is the 64-bit
            // representation the restore path reads back via
            // `f64::from_bits(_ as u64) as #rust_ty`. A `float(f32)` field's
            // own `to_bits()` yields 32-bit bits, which would round-trip
            // through `f64::from_bits` as a bogus value.
            quote! { values.push((self.#fname as f64).to_bits() as i64); }
        })
        .collect();
    let restore_float_scalar_parts: Vec<TokenStream> = float_scalars
        .iter()
        .enumerate()
        .map(|(float_idx, (_, f))| {
            let fname = &f.name;
            let rust_ty = scalar_rust_type(&f.kind);
            quote! {
                self.#fname = f64::from_bits(float_values[#float_idx] as u64) as #rust_ty;
            }
        })
        .collect();
    let initialize_sym_float_scalar_parts: Vec<TokenStream> = float_scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let value_name = quote::format_ident!("{}_value", f.name);
            quote! { sym.#value_name = (self.#fname as f64).to_bits() as i64; }
        })
        .collect();
    let collect_float_scalar_parts: Vec<TokenStream> = float_scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { args.push(sym.#fname); }
        })
        .collect();
    let fail_float_scalar_parts: Vec<TokenStream> = float_scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            quote! { args.push(self.#fname); }
        })
        .collect();
    let collect_float_scalar_values_parts: Vec<TokenStream> = float_scalars
        .iter()
        .map(|(_, f)| {
            let value_name = quote::format_ident!("{}_value", f.name);
            quote! { values.push(sym.#value_name); }
        })
        .collect();
    let writeback_float_from_values_parts: Vec<TokenStream> = float_scalars
        .iter()
        .enumerate()
        .map(|(float_idx, (_, f))| {
            let fname = &f.name;
            let rust_ty = scalar_rust_type(&f.kind);
            let idx_lit = num_scalars + float_idx;
            quote! {
                self.#fname = f64::from_bits(values[#idx_lit] as u64) as #rust_ty;
            }
        })
        .collect();
    let writeback_live_float_scalar_arms: Vec<TokenStream> = float_scalars
        .iter()
        .enumerate()
        .map(|(float_idx, (_, f))| {
            let fname = &f.name;
            let rust_ty = scalar_rust_type(&f.kind);
            quote! {
                #float_idx => {
                    self.#fname = f64::from_bits(value as u64) as #rust_ty;
                }
            }
        })
        .collect();
    let state_float_field_ref_arms: Vec<TokenStream> = float_scalars
        .iter()
        .enumerate()
        .map(|(float_idx, (_, f))| {
            let fname = &f.name;
            quote! { #float_idx => Some(self.#fname), }
        })
        .collect();
    let set_state_float_field_ref_arms: Vec<TokenStream> = float_scalars
        .iter()
        .enumerate()
        .map(|(float_idx, (_, f))| {
            let fname = &f.name;
            quote! { #float_idx => { self.#fname = value; } }
        })
        .collect();
    let state_float_field_value_arms: Vec<TokenStream> = float_scalars
        .iter()
        .enumerate()
        .map(|(float_idx, (_, f))| {
            let value_name = quote::format_ident!("{}_value", f.name);
            quote! { #float_idx => Some(self.#value_name), }
        })
        .collect();
    let set_state_float_field_value_arms: Vec<TokenStream> = float_scalars
        .iter()
        .enumerate()
        .map(|(float_idx, (_, f))| {
            let value_name = quote::format_ident!("{}_value", f.name);
            quote! { #float_idx => { self.#value_name = value; } }
        })
        .collect();
    let populate_float_scalar_parts: Vec<TokenStream> = float_scalars
        .iter()
        .enumerate()
        .map(|(float_idx, (_, f))| {
            let fname = &f.name;
            let value_name = quote::format_ident!("{}_value", f.name);
            let slot = float_identity_base + float_idx;
            quote! {
                if #slot < frame.float_regs.len() {
                    frame.float_regs[#slot] = Some(self.#fname);
                    frame.float_values[#slot] = Some(self.#value_name);
                }
            }
        })
        .collect();
    let debug_float_scalar_state_parts: Vec<TokenStream> = float_scalars
        .iter()
        .map(|(_, f)| {
            let fname = &f.name;
            let label = f.name.to_string();
            quote! {
                let _ = ::std::fmt::Write::write_fmt(
                    &mut out,
                    format_args!("  {} = {}\n", #label, self.#fname as f64),
                );
            }
        })
        .collect();
    let debug_float_scalar_label_parts: Vec<TokenStream> = float_scalars
        .iter()
        .map(|(_, f)| {
            let label = f.name.to_string();
            quote! { labels.push(::std::string::String::from(#label)); }
        })
        .collect();

    // Optional method overrides — emitted ONLY when ref scalars exist, so
    // interps with none generate a byte-identical token stream (the trait
    // defaults from `JitState` / `JitCodeSym` apply).
    // The virtualizable's value-routing type: one Ref for the identity,
    // whatever the number of `[.. ; virt]` arrays.
    let vable_identity_type_part: TokenStream = if has_vable_identity {
        quote! { types.push(majit_ir::Type::Ref); }
    } else {
        quote! {}
    };
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
    // Pairs the two halves above straight into the driver's entry buffer.
    // Emitted under the same condition as `live_value_types_into` — without
    // that override the type half falls back to an allocating default, which
    // would leave this one allocating anyway.
    let extract_live_values_into_override: TokenStream =
        if num_ref_scalars > 0 || num_virt_arrays > 0 || num_float_scalars > 0 {
            quote! {
                fn extract_live_values_into(
                    &self,
                    _meta: &#meta_ty,
                    out: &mut ::std::vec::Vec<majit_ir::Value>,
                    raw: &mut ::std::vec::Vec<i64>,
                    types: &mut ::std::vec::Vec<majit_ir::Type>,
                ) {
                    self.extract_live_into(_meta, raw);
                    self.live_value_types_into(_meta, types);
                    out.extend(raw.iter().zip(types.iter()).map(|(&__v, &__t)| match __t {
                        majit_ir::Type::Float => {
                            majit_ir::Value::Float(f64::from_bits(__v as u64))
                        }
                        majit_ir::Type::Ref => {
                            majit_ir::Value::Ref(majit_ir::GcRef(__v as usize))
                        }
                        _ => majit_ir::Value::Int(__v),
                    }));
                }
            }
        } else {
            quote! {}
        };
    let live_value_types_override: TokenStream =
        if num_ref_scalars > 0 || num_virt_arrays > 0 || num_float_scalars > 0 {
            quote! {
                // A bank with no scalars emits `0..0`, which is data, not the
                // `for i in 10..0` typo `reversed_empty_ranges` is aimed at —
                // and that lint is deny-by-default in every consumer of this
                // macro, so a state of arrays only would not compile there.
                #[allow(clippy::reversed_empty_ranges)]
                fn live_value_types_into(
                    &self,
                    _meta: &#meta_ty,
                    types: &mut ::std::vec::Vec<majit_ir::Type>,
                ) {
                    // Value-routing types in `extract_live` order: int scalars,
                    // int array elements, then the ONE virtualizable identity
                    // (Ref), then appended ref scalars (Ref), then appended float
                    // scalars (Float).
                    // The identity slot MUST be Ref so the live `&state` is a Ref
                    // failarg (TAGBOX), which the resume reader decodes through
                    // `decode_ref` in both the vable section and the frame
                    // ref-liveness.
                    for _ in 0..#num_scalars {
                        types.push(majit_ir::Type::Int);
                    }
                    #(#array_type_parts)*
                    #vable_identity_type_part
                    for _ in 0..#num_ref_scalars {
                        types.push(majit_ir::Type::Ref);
                    }
                    for _ in 0..#num_float_scalars {
                        types.push(majit_ir::Type::Float);
                    }
                }

                fn live_value_types(&self, _meta: &#meta_ty) -> Vec<majit_ir::Type> {
                    let mut types: Vec<majit_ir::Type> = Vec::new();
                    self.live_value_types_into(_meta, &mut types);
                    types
                }
            }
        } else {
            quote! {}
        };
    let restore_banked_override: TokenStream = if num_ref_scalars > 0 || num_float_scalars > 0 {
        quote! {
            fn restore_banked3(
                &mut self,
                meta: &#meta_ty,
                int_values: &[i64],
                ref_values: &[i64],
                float_values: &[i64],
            ) {
                // Int scalars/arrays/virt restore from the int bank exactly
                // as `restore`; ref/float scalars restore from their bank by
                // 0-based bank-local index. Float values are raw f64 bits.
                self.restore(meta, int_values);
                #(#restore_ref_scalar_parts)*
                #(#restore_float_scalar_parts)*
            }
        }
    } else {
        quote! {}
    };
    // Emit the virt-array write-back override only for consumers that declare a
    // `[.. ; virt]` array; 0-array interps (tlr, tinyframe, i64env) keep
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
    let float_field_accessor_overrides: TokenStream = if num_float_scalars > 0 {
        quote! {
            fn state_float_field_ref(&self, field_idx: usize) -> Option<majit_ir::OpRef> {
                match field_idx {
                    #(#state_float_field_ref_arms)*
                    _ => None,
                }
            }
            fn set_state_float_field_ref(&mut self, field_idx: usize, value: majit_ir::OpRef) {
                match field_idx {
                    #(#set_state_float_field_ref_arms)*
                    _ => {}
                }
            }
            fn state_float_field_value(&self, field_idx: usize) -> Option<i64> {
                match field_idx {
                    #(#state_float_field_value_arms)*
                    _ => None,
                }
            }
            fn set_state_float_field_value(&mut self, field_idx: usize, value: i64) {
                match field_idx {
                    #(#set_state_float_field_value_arms)*
                    _ => {}
                }
            }
            fn float_identity_slots_base(&self) -> usize {
                #float_identity_base
            }
            fn float_identity_slots_end(&self) -> usize {
                #float_identity_end
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
                #num_vable_identity_slots,
                #num_ref_scalars,
                #ref_identity_base,
                #int_identity_base,
            ).with_float_scalars(#num_float_scalars, #float_identity_base)
        }
    } else {
        quote! {
            majit_metainterp::blackhole::StateFieldLayout::new(
                #num_scalars,
                ::std::vec![#(self.#create_sym_array_names.len()),*],
                #num_vable_identity_slots,
                #int_identity_base,
            ).with_float_scalars(#num_float_scalars, #float_identity_base)
        }
    };

    // Naming the virtualizable on the jitdriver static data
    // (`warmspot.py:520-545 make_virtualizable_infos`) is what makes
    // `compile.py:508-511`'s field-reload preamble run, which retires the
    // per-entry re-export of the virtualizable's array elements: the compiled
    // entry reloads them from the virtualizable pointer instead of being handed
    // one entry argument per element.
    //
    // The name arrives through `JitDriver::declare_flat_entry_contract` together
    // with the width of the entry it applies to, because the two halves are only
    // meaningful together: the entry is the flat live-value prefix, not the red
    // count (the whole state is a single red in the merge-point payload), and an
    // index into one picks out a different value than the same index into the
    // other. That prefix is `num_scalars + num_vable_identity_slots +
    // num_ref_scalars + num_float_scalars` with the identity at flat index
    // `num_scalars` — the order `extract_live` emits — valid only while the
    // state declares no fixed arrays, the same restriction `identity_live_index`
    // below carries and for the same reason: a fixed array contributes its
    // runtime length to the prefix, and no constant here can name a length that
    // is read off a state instance.
    //
    // Whether the contract CAN be declared at all is the declared field type's
    // answer, not this expansion's: `compile.py:441-457`'s reconstruction
    // reaches each array's data pointer with a load the trace IR has to be able
    // to express, and a `Vec` embedded by value defeats that — its data pointer
    // is not at a specified offset within it, so no field load portably finds
    // it. `JitDriver::arm_flat_entry_contract` answers that off the vinfo this
    // expansion already installed, and declines rather than declaring, so the
    // width below is stated unconditionally and the structural gate stays where
    // the storage is known. Left unarmed, a state keeps the per-entry re-export
    // and behaves exactly as it did before.
    //
    // SOUNDNESS. Arming puts this driver on
    // `patch_new_loop_to_load_virtualizable_fields`, which BAKES each array's
    // trace-start length into the prologue as a fixed count of `GETARRAYITEM`
    // ops (`compile.py:443`), and nothing re-reads it afterwards. The invariant
    // stated on that helper is that the lengths are a function of the trace's
    // GREENS, so a virtualizable with other lengths keys to a different trace
    // and never reaches this entry. It is the interpreter author's to satisfy —
    // a `[.. ; virt]` array whose length can vary while the greens stay fixed
    // must not be block-backed — and it cannot be checked here: the lengths live
    // on state instances that do not exist at install time.
    let arm_flat_entry_contract: TokenStream = if num_virt_arrays > 0 && arrays.is_empty() {
        let entry_len =
            num_scalars + num_vable_identity_slots + num_ref_scalars + num_float_scalars;
        let index_of_virtualizable = num_scalars;
        quote! {
            driver.arm_flat_entry_contract(
                majit_metainterp::FlatEntryContract {
                    len: #entry_len,
                    index_of_virtualizable: #index_of_virtualizable,
                },
            );
        }
    } else {
        quote! {}
    };

    // pyjitpl.py:3443-3444 `rebuild_state_after_failure`:
    //     if vinfo is not None:
    //         self.virtualizable_boxes = virtualizable_boxes
    // The guard's vable section is decoded by `rebuild_from_resumedata` above;
    // rebuild the shadow from it so the bridge traces with the same
    // virtualizable the parent loop had.  Without it `virtualizable_boxes`
    // stays None for the whole bridge and `__trace_*` aborts at its first
    // statement (`standard_virtualizable_jitcode_argbox` has nothing to
    // resolve), so a state with `[.. ; virt]` arrays never forms a bridge at
    // all — every guard exit deopts through the blackhole instead.
    // Rebind the vable identity the bridge actually entered with.
    //
    // `clear_sym_inputarg_bindings` retires this field along with the other
    // `create_sym` mints, and the frame-register seeding below has no arm for
    // it: `reg_indices` addresses the int/ref/float scalar banks and this
    // field is in none of them. Left retired it travels straight into the
    // jump args as `OpRef::NONE` (`collect_jump_args` and
    // `collect_jump_args_with_boxes` both push it), where the optimizer has
    // neither an operand nor a type for it and `not_virtual` faults.
    //
    // `standard_virtualizable_jitcode_argbox` is the right source rather than
    // a constant built from the mirror: it prefers the exact trace-entry red
    // inputarg named by `index_of_virtualizable`, falling back to
    // `virtualizable_boxes[-1]`, which `#seed_bridge_vable` has just
    // populated. Note that seeding does not do this itself —
    // `seed_bridge_virtualizable_boxes` takes no `Sym` and writes only the
    // ctx-side boxes.
    let rebind_bridge_vable_identity: TokenStream = if has_vable_identity {
        quote! {
            if let Some((_, __vable_op, __vable_val)) =
                ctx.standard_virtualizable_jitcode_argbox()
            {
                sym.__vable_identity = __vable_op;
                sym.__vable_identity_value = __vable_val;
                if std::env::var("MAJIT_BRIDGE_DEBUG").is_ok() {
                    eprintln!("  vable identity REBOUND to {:?}", __vable_op);
                }
            } else if std::env::var("MAJIT_BRIDGE_DEBUG").is_ok() {
                eprintln!("  vable identity NOT REBOUND — no standard argbox");
            }
        }
    } else {
        quote! {}
    };
    let seed_bridge_vable: TokenStream = if num_virt_arrays > 0 {
        quote! {
            if let Some(__vinfo) = Self::__build_virtualizable_info() {
                let __seeded = majit_metainterp::seed_bridge_virtualizable_boxes(
                    ctx,
                    &__vinfo,
                    rd_virtuals,
                    resume_data,
                    &mut __bridge_cache,
                    fail_values,
                );
                if std::env::var("MAJIT_BRIDGE_DEBUG").is_ok() {
                    eprintln!(
                        "[bridgeB] vable seed={} stream={}",
                        __seeded,
                        resume_data.virtualizable_values.len(),
                    );
                }
                // A declined seed is not recoverable here and must not pass
                // silently: `virtualizable_boxes` stays unset, so the first
                // vable-shaped op in `__trace_*` has nothing to resolve and the
                // bridge trace aborts — the guard keeps deopting through the
                // blackhole, which is `compile.giveup()`'s outcome
                // (compile.py:27) reached one step later. Surface it on the
                // ordinary log channel so a bridge that never forms is
                // attributable to the seed rather than looking like an
                // unexplained abort.
                if !__seeded && majit_metainterp::majit_log_enabled() {
                    eprintln!(
                        "[jit][bridgeB] virtualizable seed declined \
                         (vable stream {} entries) — bridge will abort and \
                         the guard deopts through the blackhole",
                        resume_data.virtualizable_values.len(),
                    );
                }
            }
        }
    } else {
        quote! {}
    };

    // ── VirtualizableInfo / heap-ptr overrides for `[int; virt]` arrays ──
    // Each virt array becomes a standard-virtualizable array field on a
    // zero-static-field vinfo, so `state.<arr>[i]` lowers through the
    // `virtualizable_boxes` devirt path. Scalars stay in the state-field
    // scalar resume mechanism (disjoint from the array restore).
    //
    // Which storage the field registers as is the declared field type's to say,
    // not this expansion's: `register_virt_array_field` resolves it from the
    // container the interpreter author wrote. That is the difference the
    // compiled entry sees — only a field holding a pointer to a block with a
    // fixed payload offset can be reloaded by `compile.py:441-457`, and a `Vec`
    // embedded by value is not one.
    let build_vinfo_override: TokenStream = if num_virt_arrays > 0 {
        // Per virt array: nested data-ptr/len extractor fns + a registration
        // keyed on the field byte offset.
        let virt_array_field_parts: Vec<TokenStream> = virt_arrays
            .iter()
            .map(|(_, f)| {
                let fname = &f.name;
                let data_ptr_fn = quote::format_ident!("__vinfo_{}_data_ptr", f.name);
                let len_fn = quote::format_ident!("__vinfo_{}_len", f.name);
                let fname_str = f.name.to_string();
                let (item_size, item_type) = match &f.kind {
                    StateFieldKind::VirtArray(tp) if tp == "float" => (
                        quote! { ::std::mem::size_of::<f64>() },
                        quote! { majit_ir::Type::Float },
                    ),
                    _ => (
                        quote! { ::std::mem::size_of::<i64>() },
                        quote! { majit_ir::Type::Int },
                    ),
                };
                quote! {
                    fn #data_ptr_fn(__p: *mut u8) -> *mut i64 {
                        unsafe { (*(__p as *mut #state_type)).#fname.as_mut_ptr() as *mut i64 }
                    }
                    fn #len_fn(__p: *const u8) -> usize {
                        unsafe { (*(__p as *const #state_type)).#fname.len() }
                    }
                    majit_metainterp::virt_array::register_virt_array_field(
                        &mut __info,
                        #fname_str,
                        #item_type,
                        #item_size,
                        ::std::mem::offset_of!(#state_type, #fname),
                        #data_ptr_fn,
                        #len_fn,
                        |__s: &#state_type| &__s.#fname,
                    );
                }
            })
            .collect();
        // Field idents in `virt_arrays` order — the same order
        // `add_rust_vec_array_field` registers them, which is the order
        // `flatten_virtualizable_values` (jitdriver.rs) reads them back.
        let virt_array_export_parts: Vec<TokenStream> = virt_arrays
            .iter()
            .map(|(_, f)| {
                let fname = &f.name;
                match &f.kind {
                    StateFieldKind::VirtArray(tp) if tp == "float" => {
                        quote! { self.#fname.iter().map(|&__x| __x.to_bits() as i64).collect() }
                    }
                    _ => quote! { self.#fname.iter().map(|&__x| __x as i64).collect() },
                }
            })
            .collect();
        // The same reads appended into a caller-owned slot rather than
        // collected into a fresh `Vec`. Indexed by position so the two forms
        // fill the outer list in the identical order.
        let virt_array_export_into_parts: Vec<TokenStream> = virt_arrays
            .iter()
            .enumerate()
            .map(|(i, (_, f))| {
                let fname = &f.name;
                let source = match &f.kind {
                    StateFieldKind::VirtArray(tp) if tp == "float" => {
                        quote! { self.#fname.iter().map(|&__x| __x.to_bits() as i64) }
                    }
                    _ => quote! { self.#fname.iter().map(|&__x| __x as i64) },
                };
                quote! { arrays[#i].extend(#source); }
            })
            .collect();
        // `extract_live` pushes int scalars, then every fixed array's items,
        // then the one identity slot. With no fixed arrays that position is a
        // constant — `num_scalars` — so the identity can be DECLARED the way
        // `warmspot.py:529-538` declares `index_of_virtualizable`, and
        // `initialize_virtualizable` can look it up instead of searching the
        // reds for a matching pointer.
        //
        // A fixed `[int]` array alongside a `[int; virt]` one — which
        // `majit-metainterp/tests/jit_interp_float_state_field.rs`
        // `virt_array_with_float_scalar` declares — makes the position
        // `num_scalars + sum(array lengths)`, and those lengths are the runtime
        // `Vec` lengths this expansion reads back through `meta.<arr>_len`
        // (`create_sym_array_inits`), so no constant can be emitted here — the
        // vinfo itself is built once per driver by the `&self`-less
        // `__build_virtualizable_info`, before any state instance exists.
        // Emitting nothing is therefore correct, but it is NOT harmless on its
        // own: it leaves the identity's position unstated, and both consumers
        // must resolve it instead of assuming one.
        // `MetaInterp::identity_live_position` does resolve it for the runtime
        // path — it pointer-matches `vable_ptr` against the reds, so a wrong or
        // absent declaration is survivable there. The optimizer has no pointer
        // to match against, so with nothing declared it DECLINES to track the
        // virtualizable (`VirtualizableConfig::identity_input_index` is `None`)
        // rather than probing flat slot 0 — an int scalar on this layout, which
        // made every trace abort with VirtualStatesCantMatch. Declining was
        // measured to cost nothing here; see
        // `tests/jit_interp_fixed_array_identity_slot.rs`.
        let identity_live_index_stmt: TokenStream = if arrays.is_empty() {
            quote! { __info.identity_live_index = Some(#num_scalars); }
        } else {
            quote! {}
        };
        quote! {
            #[allow(non_snake_case)]
            fn __build_virtualizable_info()
            -> Option<::std::sync::Arc<majit_metainterp::virtualizable::VirtualizableInfo>> {
                use majit_metainterp::virtualizable::VirtualizableInfo;
                // No `vable_token` field: the stack-local state struct is
                // non-GC and never moved, so the token protocol is inert — the
                // identity value (a `&state` pointer) is recovered straight
                // from the resume snapshot, not via a heap token. The struct's
                // offset 0 is a live user field, so every token read/write must
                // no-op rather than land there.
                let mut __info = VirtualizableInfo::without_vable_token();
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
                #identity_live_index_stmt
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

            fn blackhole_virtualizable_identity(
                &self,
                _meta: &Self::Meta,
                _virtualizable: &str,
                _info: &majit_metainterp::virtualizable::VirtualizableInfo,
            ) -> Option<*mut u8> {
                // This is intentionally distinct from the PyFrame path:
                // `state` is the current host-stack object, not a movable GC
                // object whose trace-time address may be baked into resume
                // data. At deopt, re-derive its identity from this call.
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
                    #( #virt_array_export_parts ),*
                ];
                Some((__static_boxes, __array_boxes))
            }

            // Same export into buffers the driver reuses. Statics stays empty
            // for the reason above, so only the arrays are written; the outer
            // `Vec` is resized rather than rebuilt so the inner element
            // storage survives across entries.
            fn export_virtualizable_boxes_into(
                &self,
                _meta: &Self::Meta,
                _virtualizable: &str,
                _info: &majit_metainterp::virtualizable::VirtualizableInfo,
                _statics: &mut ::std::vec::Vec<i64>,
                arrays: &mut ::std::vec::Vec<::std::vec::Vec<i64>>,
            ) -> bool {
                arrays.resize_with(#num_virt_arrays, ::std::vec::Vec::new);
                #( #virt_array_export_into_parts )*
                true
            }
        }
    } else {
        quote! {}
    };

    quote! {
        /// Compiled loop metadata for state_fields mode: flattened array lengths at trace start.
        #[derive(Clone)]
        #[allow(non_camel_case_types)]
        struct #meta_ty {
            #(#meta_fields)*
        }

        impl #meta_ty {
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
                    #num_vable_identity_slots,
                    #num_ref_scalars,
                    #ref_identity_base,
                    #num_float_scalars,
                    #float_identity_base,
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
                // `warmspot.py:520-545 make_virtualizable_infos` names the
                // virtualizable on the jitdriver static data during setup, i.e.
                // before the driver is registered. Order matters here for the
                // same reason: `ensure_descriptor_registered` MOVES the
                // descriptor into the `MetaInterpStaticData` table, so a
                // contract declared after it would land on a descriptor no
                // consumer reads.
                #arm_flat_entry_contract
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
        struct #sym_ty {
            #(#sym_scalar_fields)*
            #(#sym_scalar_value_fields)*
            #(#sym_array_fields)*
            #(#sym_array_value_fields)*
            #(#sym_virt_array_fields)*
            #sym_vable_identity_fields
            #(#sym_ref_scalar_fields)*
            #(#sym_ref_scalar_value_fields)*
            #(#sym_float_scalar_fields)*
            #(#sym_float_scalar_value_fields)*
            loop_header_pc: usize,
            trace_started: bool,
        }

        #loop_carried_boxes_fn

        impl majit_metainterp::JitCodeSym for #sym_ty {
            fn total_slots(&self) -> usize {
                #num_scalars #(#total_slots_array_parts)* + #num_vable_identity_slots
            }

            fn loop_carried_boxes(
                &self,
                __boxes: &[(majit_ir::OpRef, majit_ir::Type)],
            ) -> Option<Vec<(majit_ir::OpRef, majit_ir::Type)>> {
                Some(#loop_carried_boxes_fn_name(self, __boxes))
            }

            fn int_identity_slots_end(&self) -> usize {
                #int_identity_base + self.total_slots()
            }

            fn int_identity_slots_base(&self) -> usize {
                #int_identity_base
            }

            // Mirrors `split_identity_reg_ends`' int end in
            // `jitcode_lower/mod.rs` exactly: the working-register floor stops
            // after the scalars plus the single vable-identity slot, because a
            // virt array's element count is only known from the live object.
            //
            // It must also mirror the CONDITION under which that end is
            // applied. `split_identity_floor` (`jitcode_lower/api.rs`) raises a
            // sub-JitCode's `alloc_reg()` floor past the range only when
            // `split_dispatch` is on; with it off, no sub-JitCode reserves
            // anything and `[base, end)` holds ordinary working registers. The
            // inline-frame snapshot trim keyed on this end blanks the range
            // unconditionally, so reporting a non-empty range here would drop
            // live data from a sub-frame's snapshot. Report an empty range
            // instead, so the trim is inert exactly where the reservation is.
            fn int_identity_reserved_end(&self) -> usize {
                #int_identity_base + #num_reserved_identity_slots
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
            #float_field_accessor_overrides

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
                #fail_vable_identity_part
                #(#fail_ref_scalar_parts)*
                #(#fail_float_scalar_parts)*
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
                #populate_vable_identity_part
                let _ = __slot;
                #(#populate_ref_scalar_parts)*
                #(#populate_float_scalar_parts)*
            }

            #[allow(unused_assignments, unused_variables)]
            fn seed_recursive_fresh_frame(
                &self,
                frame: &mut majit_metainterp::MIFrame,
            ) {
                let mut __slot: usize = #int_identity_base;
                #(#seed_scalar_parts)*
                #(#seed_array_parts)*
                #seed_vable_identity_part
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
            type Meta = #meta_ty;
            type Sym = #sym_ty;
            type Env = #env_type;

            fn can_trace(&self) -> bool {
                true
            }

            fn build_meta(&self, _header_pc: usize, _program: &#env_type) -> #meta_ty {
                #meta_ty {
                    #(#build_meta_fields)*
                }
            }

            // The buffer-filling form is the primary one and the owning form
            // wraps it: `warmstate.py:503-511 maybe_compile_and_run` hands
            // `execute_assembler` reds that are already unboxed locals, so the
            // entry path allocates nothing to describe them. The driver keeps
            // these buffers across calls, so a warm entry refills them.
            fn extract_live_into(&self, _meta: &#meta_ty, values: &mut ::std::vec::Vec<i64>) {
                #(#extract_scalar_parts)*
                #(#extract_array_parts)*
                #extract_vable_identity_part
                #(#extract_ref_scalar_parts)*
                #(#extract_float_scalar_parts)*
            }

            fn extract_live(&self, _meta: &#meta_ty) -> Vec<i64> {
                let mut values = Vec::new();
                self.extract_live_into(_meta, &mut values);
                values
            }

            #live_value_types_override

            #extract_live_values_into_override

            fn create_sym(meta: &#meta_ty, header_pc: usize) -> #sym_ty {
                let mut __offset: usize = 0;
                #(#create_sym_scalar_inits)*
                #(#create_sym_array_inits)*
                #(#create_sym_virt_array_inits)*
                #create_sym_vable_identity_init
                #(#create_sym_ref_scalar_inits)*
                #(#create_sym_float_scalar_inits)*
                #sym_ty {
                    #(#create_sym_scalar_names,)*
                    #(#create_sym_scalar_value_names,)*
                    #(#create_sym_array_names,)*
                    #(#create_sym_array_value_names,)*
                    #(#create_sym_virt_array_len_value_names,)*
                    #create_sym_vable_identity_field_names
                    #(#create_sym_ref_scalar_names,)*
                    #(#create_sym_ref_scalar_value_names,)*
                    #(#create_sym_float_scalar_names,)*
                    #(#create_sym_float_scalar_value_names,)*
                    loop_header_pc: header_pc,
                    trace_started: false,
                }
            }

            fn initialize_sym(&self, sym: &mut #sym_ty, _meta: &#meta_ty) {
                #(#initialize_sym_scalar_parts)*
                #(#initialize_sym_array_parts)*
                #(#initialize_sym_virt_array_parts)*
                #initialize_sym_vable_identity_part
                #(#initialize_sym_ref_scalar_parts)*
                #(#initialize_sym_float_scalar_parts)*
            }

            // Retires `create_sym`'s `__offset` numbering for callers whose
            // trace does not number its inputargs the same way — see the trait
            // declaration for why a bridge is such a caller and why a stale
            // mint resolves instead of missing.  Mirrors the `#sym_ty`
            // constructor field-for-field: every `OpRef` it fills from
            // `__offset` is cleared here, and nothing else is touched.
            fn count_bound_sym_inputargs(sym: &#sym_ty) -> Option<usize> {
                let mut __bound = 0usize;
                #(#count_bound_sym_parts)*
                Some(__bound)
            }

            fn clear_sym_inputarg_bindings(sym: &mut #sym_ty) {
                #(#clear_sym_binding_parts)*
            }

            // ── Part A (bridge resume-decode). ──
            //
            // resume.py:1042-1057 rebuild_from_resumedata parity for the
            // JitDriver state.  Without this the trait default returns None and
            // `start_bridge_tracing` aborts (jitdriver.rs:3789) so no guard-exit
            // bridge ever forms — a failing loop guard re-enters via
            // ContinueRunningNormally instead of forming a bridge.  Adding it
            // flips `start_bridge_tracing` ok=false→ok=true and bridges form;
            // existing consumers stay byte-identical.
            //
            // NOTE: this is general guard-exit bridge-formation infrastructure.
            // A trace that spins is not by itself evidence of a missing or
            // unseeded bridge: one such spin was instead a peeled loop that
            // constant-folded a red and dropped the matching exit guard,
            // leaving a state-mutating residual to loop forever.
            fn rebuild_from_resumedata(
                _meta: &mut #meta_ty,
                fail_arg_types: &[majit_ir::Type],
                storage: Option<&std::sync::Arc<majit_metainterp::resume::ResumeStorage>>,
            ) -> Option<majit_metainterp::ResumeDataResult> {
                // resume.py:1049-1055 rebuild_from_resumedata:
                //     while not resumereader.done_reading():
                //         jitcode_pos, pc = resumereader.read_jitcode_pos_pc()
                //         jitcode = metainterp.staticdata.jitcodes[jitcode_pos]
                //         f = metainterp.newframe(jitcode); f.setup_resume_at_op(pc)
                //         resumereader.consume_boxes(f.get_current_position_info(), ..)
                //
                // Every section is delimited by ITS OWN jitcode's `-live-`
                // liveness at ITS OWN pc (jitcode.py:147 `enumerate_vars` ->
                // length_i + length_r + length_f); RPython never consumes "the
                // rest of the stream" as one frame.  The writer already honours
                // that contract — `build_state_field_snapshot`
                // (pyjitpl/dispatch.rs) emits one section per MIFrame, outermost
                // to innermost, each stamping its own absolute jitcode index —
                // so a `#[jit_inline]` callee on the frame stack at guard time
                // publishes a multi-frame stream.  Decoding that with the `None`
                // fallback folds the next section's [jitcode_index, pc, py_pc]
                // header and values into frame 0, so frame 0 stops matching its
                // own liveness and the per-bank register -> sym-slot map in
                // `setup_bridge_sym` is meaningless.
                //
                // `register_dispatch_jitcode` installs the liveness splitter
                // (`install_state_field_fvc`); decode through it exactly like
                // the other compile-time decoders of the same `rd_numb` already
                // do.  Deliberately not `.expect()`ed: a state whose
                // `lower_dispatch_body` failed never calls
                // `register_dispatch_jitcode`, so an absent callback is
                // legitimate and keeps the previous fallback behaviour.
                let storage = storage?;
                let rd_numb = storage.rd_numb.as_slice();
                let rd_consts = storage.rd_consts();
                let __fvc = majit_ir::resumedata::get_frame_value_count_fn();
                let __fvc_ref: ::std::option::Option<&dyn Fn(i32, i32) -> usize> =
                    __fvc.as_ref().map(|f| f as &dyn Fn(i32, i32) -> usize);
                let (num_failargs, vable_values, vref_values, frames) =
                    majit_ir::resumedata::rebuild_from_numbering(
                        rd_numb,
                        rd_consts,
                        fail_arg_types,
                        __fvc_ref,
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
            // STATUS: this seeds int/ref SCALAR slots only.  No available
            // workload has been observed to reach this path (setup_bridge_sym
            // emits no MAJIT_BRIDGE_DEBUG lines for any of them), so part B is
            // present-but-unexercised; treat the
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
            //     values.  Both bite only for macro states that declare
            //     arrays or inline sub-frames.
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
            #[allow(clippy::reversed_empty_ranges)]
            fn setup_bridge_sym(
                sym: &mut #sym_ty,
                ctx: &mut majit_metainterp::TraceCtx,
                resume_data: &majit_metainterp::ResumeDataResult,
                rd_virtuals: Option<&[std::rc::Rc<majit_ir::RdVirtualInfo>]>,
                fail_values: &[i64],
                fail_types: &[majit_ir::Type],
            ) {
                use majit_ir::resumedata::RebuiltValue;
                use majit_metainterp::JitCodeSym as _;
                if std::env::var_os("MAJIT_BRIDGE_DIAG").is_some() {
                    eprintln!(
                        "[setup_bridge_sym] CALLED frames={} rd_virtuals={}",
                        resume_data.frames.len(),
                        rd_virtuals.map_or(0, |v| v.len()),
                    );
                }
                // resume.py:993-1007 — materialize the guard's virtuals + replay
                // its deferred heap writes as bridge-entry NEW/SETFIELD_GC ops so
                // the compiled bridge observes the heap state the blackhole deopt
                // would rebuild. A push/dup whose node is virtualized-and-elided
                // defers its head-store to rd_pendingfields while the size store
                // commits inline; without this replay the bridge reads size>chain
                // and dereferences a NULL node head. Runs independent of the
                // frame-register seeding below (which may decline).
                let mut __bridge_cache = majit_metainterp::BridgeVirtualCache::new(
                    rd_virtuals.map_or(0, |v| v.len()),
                    majit_metainterp::default_bridge_array_descr,
                );
                majit_metainterp::replay_pending_fields(
                    ctx,
                    resume_data,
                    rd_virtuals,
                    &mut __bridge_cache,
                );
                #seed_bridge_vable
                #rebind_bridge_vable_identity
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
                        "[bridgeB] reg_indices int={:?} ref={:?} float={:?} values.len={} int_base={} ref_base={} float_base={}",
                        reg_indices.int,
                        reg_indices.ref_,
                        reg_indices.float,
                        frame.values.len(),
                        #int_identity_base,
                        #ref_identity_base,
                        #float_identity_base
                    );
                }
                if reg_indices.total_len() != frame.values.len() {
                    if __dbg {
                        eprintln!("[bridgeB] reg_indices/frame length mismatch — declining");
                    }
                    return;
                }
                let __ref_off = reg_indices.int.len();
                let __float_off = reg_indices.int.len() + reg_indices.ref_.len();
                // int scalars: identity register = int_identity_base + k.
                for __k in 0..#num_scalars {
                    let __target = #int_identity_base + __k;
                    let __pos = reg_indices.int.iter().position(|&r| r as usize == __target);
                    let __pos = match __pos {
                        Some(p) => p,
                        None => {
                            // The guard's resume frame does not carry this
                            // field's identity register, so there is no red to
                            // decode — measured on dualtape, where the frame's
                            // live int set is `[0]` while the two int scalars
                            // sit at identity registers 1 and 2.
                            //
                            // Its concrete value is still known: `initialize_sym`
                            // read it off the live state at bridge entry and
                            // `clear_sym_inputarg_bindings` preserves the value
                            // mirrors. Bind it as a constant, which is exactly
                            // what the `RebuiltValue::Const` arm below does for a
                            // field the frame carries already folded — the two
                            // cases differ in who folded it, not in what the
                            // bridge should observe.
                            //
                            // Leaving it unbound instead hands the optimizer an
                            // `OpRef::NONE`, which has no operand and no type;
                            // `materialize_operand_at` and `not_virtual` both
                            // reject it.
                            //
                            // The `.expect` is load-bearing and it has been
                            // exercised: on `examples/dualtape` this arm is
                            // reached 24 times in a passing run (counted off
                            // the `MAJIT_BRIDGE_DEBUG` line below) and fires 0
                            // times. So the mirror is populated at every reach
                            // — the value is recovered from the state, never
                            // fabricated, and `const_int` never mints a
                            // stand-in zero. That separation is measured here
                            // and nowhere else: the dualtape fixture greens on
                            // no-panic plus correct output, so a fabricated
                            // value would pass it. Same for the ref and float
                            // arms below, which share this shape.
                            let __bits = sym.state_field_value(__k).expect("state field concrete value not initialized");
                            let __op = ctx.const_int(__bits);
                            sym.set_state_field_ref(__k, __op);
                            sym.set_state_field_value(__k, __bits);
                            if __dbg {
                                eprintln!("  int scalar {} <- reg {} ABSENT from frame, bound Const {}", __k, __target, __bits);
                            }
                            continue;
                        }
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
                        RebuiltValue::Virtual(__vidx) => {
                            // resume.py:945-956 getvirtual — materialize the
                            // virtual as bridge NEW/SETFIELD_GC ops and bind the
                            // state field to its OpRef. Symbolic only; the concrete
                            // int shadow needs backend/callinfocollection.
                            let __op = majit_metainterp::materialize_bridge_virtual(
                                ctx, *__vidx, rd_virtuals, resume_data, &mut __bridge_cache,
                            );
                            sym.set_state_field_ref(__k, __op);
                            if __dbg {
                                eprintln!("  int scalar {} <- reg {} Virtual {}", __k, __target, __vidx);
                            }
                        }
                        __other => {
                            if __dbg {
                                eprintln!("  int scalar {} <- reg {} UNSEEDED variant={:?}", __k, __target, std::mem::discriminant(__other));
                            }
                        }
                    }
                }
                // ref scalars: identity register = ref_identity_base + j.
                for __j in 0..#num_ref_scalars {
                    let __target = #ref_identity_base + __j;
                    let __pos = reg_indices.ref_.iter().position(|&r| r as usize == __target);
                    let __pos = match __pos {
                        Some(p) => p,
                        None => {
                            // Ref twin of the int arm above — same reason, same
                            // remedy, `const_ref` instead of `const_int`. Kept
                            // symmetric deliberately: a bank left on the bare
                            // `continue` reintroduces the unbound-field hazard
                            // for any state that declares one.
                            let __bits = sym.state_ref_field_value(__j).expect("ref state field concrete value not initialized");
                            let __op = ctx.const_ref(__bits);
                            sym.set_state_ref_field_ref(__j, __op);
                            sym.set_state_ref_field_value(__j, __bits);
                            if __dbg {
                                eprintln!("  ref scalar {} <- reg {} ABSENT from frame, bound Const {:#x}", __j, __target, __bits);
                            }
                            continue;
                        }
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
                        RebuiltValue::Virtual(__vidx) => {
                            // resume.py:945-956 getvirtual — materialize the
                            // virtual and bind the ref state field to its OpRef.
                            let __op = majit_metainterp::materialize_bridge_virtual(
                                ctx, *__vidx, rd_virtuals, resume_data, &mut __bridge_cache,
                            );
                            sym.set_state_ref_field_ref(__j, __op);
                            if __dbg {
                                eprintln!("  ref scalar {} <- reg {} Virtual {}", __j, __target, __vidx);
                            }
                        }
                        __other => {
                            if __dbg {
                                eprintln!("  ref scalar {} <- reg {} UNSEEDED variant={:?}", __j, __target, std::mem::discriminant(__other));
                            }
                        }
                    }
                }
                // float scalars: identity register = float_identity_base + k.
                for __k in 0..#num_float_scalars {
                    let __target = #float_identity_base + __k;
                    let __pos = reg_indices.float.iter().position(|&r| r as usize == __target);
                    let __pos = match __pos {
                        Some(p) => p,
                        None => {
                            // Float twin of the int/ref arms above. The mirror
                            // stores the float's raw bits, so the constant is
                            // minted from the bit pattern, matching how
                            // `initialize_sym` and `restore_banked3` carry it.
                            let __bits = sym.state_float_field_value(__k).expect("float state field concrete value not initialized");
                            let __op = ctx.const_float(__bits);
                            sym.set_state_float_field_ref(__k, __op);
                            sym.set_state_float_field_value(__k, __bits);
                            if __dbg {
                                eprintln!("  float scalar {} <- reg {} ABSENT from frame, bound Const {}", __k, __target, f64::from_bits(__bits as u64));
                            }
                            continue;
                        }
                    };
                    match &frame.values[__float_off + __pos] {
                        RebuiltValue::Box(n, kind) if matches!(kind, majit_ir::Type::Float) => {
                            let (__op, __shadow) = majit_metainterp::bridge_decode_red(
                                *n, *kind, fail_values, fail_types,
                            );
                            sym.set_state_float_field_ref(__k, __op);
                            sym.set_state_float_field_value(__k, __shadow);
                            if __dbg {
                                eprintln!("  float scalar {} <- reg {} Box {} = {:#x}", __k, __target, n, __shadow);
                            }
                        }
                        RebuiltValue::Const(c) => {
                            let __bits = c.as_raw_i64();
                            let __op = ctx.const_float(__bits);
                            sym.set_state_float_field_ref(__k, __op);
                            sym.set_state_float_field_value(__k, __bits);
                            if __dbg {
                                eprintln!("  float scalar {} <- reg {} Const {:#x}", __k, __target, __bits);
                            }
                        }
                        _ => {}
                    }
                }
            }

            fn is_compatible(&self, meta: &#meta_ty) -> bool {
                true #(#compat_checks)*
            }

            fn restore(&mut self, _meta: &#meta_ty, values: &[i64]) {
                let mut __offset: usize = 0;
                #(#restore_scalar_parts)*
                #(#restore_array_parts)*
                #restore_vable_identity_part
            }

            #restore_banked_override

            fn recover_after_compiled_run(&mut self) {
                #recover_body
            }

            fn debug_state_fields(&self, _meta: &#meta_ty) -> Option<::std::string::String> {
                let mut out = ::std::string::String::new();
                #(#debug_scalar_state_parts)*
                #(#debug_array_state_parts)*
                #(#debug_virt_array_state_parts)*
                #(#debug_ref_scalar_state_parts)*
                #(#debug_float_scalar_state_parts)*
                Some(out)
            }

            fn debug_state_live_labels(&self, _meta: &#meta_ty) -> Option<::std::vec::Vec<::std::string::String>> {
                let mut labels: ::std::vec::Vec<::std::string::String> = ::std::vec::Vec::new();
                #(#debug_scalar_label_parts)*
                #(#debug_array_label_parts)*
                #debug_vable_identity_label_part
                #(#debug_ref_scalar_label_parts)*
                #(#debug_float_scalar_label_parts)*
                Some(labels)
            }

            fn collect_scalar_state_field_values(sym: &Self::Sym) -> Vec<i64> {
                let mut values = Vec::new();
                #(#collect_scalar_values_parts)*
                #(#collect_float_scalar_values_parts)*
                values
            }

            fn collect_ref_scalar_state_field_values(sym: &Self::Sym) -> Vec<i64> {
                let mut values = Vec::new();
                #(#collect_ref_scalar_values_parts)*
                values
            }

            fn writeback_scalar_state_fields_from_values(&mut self, values: &[i64]) {
                #(#writeback_from_values_parts)*
                #(#writeback_float_from_values_parts)*
            }

            fn writeback_live_scalar_state_field(&mut self, field_idx: usize, value: i64) {
                match field_idx {
                    #(#writeback_live_scalar_arms)*
                    _ => {}
                }
            }

            fn writeback_live_ref_scalar_state_field(&mut self, field_idx: usize, value: i64) {
                match field_idx {
                    #(#writeback_live_ref_scalar_arms)*
                    _ => {}
                }
            }

            fn writeback_live_float_scalar_state_field(&mut self, field_idx: usize, value: i64) {
                match field_idx {
                    #(#writeback_live_float_scalar_arms)*
                    _ => {}
                }
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

            fn collect_jump_args(sym: &#sym_ty) -> Vec<majit_ir::OpRef> {
                let mut args = Vec::new();
                #(#collect_scalar_parts)*
                #(#collect_array_parts)*
                #collect_vable_identity_part
                #(#collect_ref_scalar_parts)*
                #(#collect_float_scalar_parts)*
                args
            }

            #collect_jump_args_with_boxes_method

            fn validate_close(sym: &#sym_ty, meta: &#meta_ty) -> bool {
                true #(#validate_array_checks)*
            }

            // State-field JIT
            // override of `JitState::populate_frame_for_guard` so
            // jitdriver-level guard sites (e.g. `force_finish_trace`'s
            // GuardAlwaysFails fallback) get the same snapshot wire-up
            // as the dispatch-level `record_state_guard`
            // (`pyjitpl/dispatch.rs:284`).  Calls the macro-emitted
            // `JitCodeSym::populate_frame_int_regs` to bridge
            // `__JitSym_<fn>` slots onto `MIFrame.int_regs`, then builds a
            // single-frame snapshot via the canonical helper.
            fn populate_frame_for_guard(
                sym: &#sym_ty,
                frames: &mut majit_metainterp::MIFrameStack,
                __op_live: u8,
                __all_liveness: &[u8],
                __virtualizable_boxes: &[majit_ir::OpRef],
                __virtualref_boxes: &[(majit_ir::OpRef, usize)],
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
                let __fn = sym.float_identity_slots_end().min(__root.float_regs.len());
                let __saved_float_regs: Vec<Option<majit_ir::OpRef>> =
                    __root.float_regs[..__fn].to_vec();
                let __saved_float_values: Vec<Option<i64>> =
                    __root.float_values[..__fn].to_vec();
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
                    Some((sym.int_identity_slots_base(), sym.int_identity_reserved_end())),
                );
                let __root = &mut frames.frames[0];
                __root.int_regs[..__n].copy_from_slice(&__saved_int_regs);
                __root.int_values[..__n].copy_from_slice(&__saved_int_values);
                __root.ref_regs[..__rn].copy_from_slice(&__saved_ref_regs);
                __root.ref_values[..__rn].copy_from_slice(&__saved_ref_values);
                __root.float_regs[..__fn].copy_from_slice(&__saved_float_regs);
                __root.float_values[..__fn].copy_from_slice(&__saved_float_values);
                Some(__snapshot)
            }

            #build_vinfo_override
        }
    }
}
