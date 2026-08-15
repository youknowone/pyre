use super::lower_value::struct_type_id_tokens;
use super::*;

/// The `(field_size, is_signed)` a struct-layout registration reports for one
/// field, plus a compile-time check that the declaration matches the Rust
/// struct.  `descr.py:218-239 get_field_descr` derives both from `FIELDTYPE`,
/// but the macro sees only the field's name at the access site, so a sub-word
/// integer field has to be named in `int_fields` to be registered as one.
/// Anything undeclared is registered as an `i64` — and has to be that wide to
/// stay so, which the returned witness enforces. `i64`, not `usize`: the two
/// part company on a 32-bit target such as wasm32, and it is the eight-byte
/// word that is claimed here.
pub(super) fn field_scalar_tokens(
    config: &LowererConfig,
    key: &str,
    struct_path: &syn::Path,
    member: &syn::Member,
) -> (TokenStream, TokenStream, TokenStream) {
    // Every consultation THIS lowering makes reaches the maps through here, so
    // one line records which declared keys an access site asked about.
    // Recorded whether or not the key is declared: the report is over the
    // DECLARED keys, and an undeclared one describes nothing to begin with.
    //
    // Not the only reader of `ref_fields`, and the difference bounds what the
    // report can claim.  `RefFieldRewriter` consults it too, to rewrite the
    // CONCRETE body's `x.field` into a deref of the raw-pointer carrier.  Both
    // walk the same source, so they normally agree — they diverge exactly where
    // this lowering stopped early and the rewriter did not, which is a degraded
    // arm.  That is why an unconsulted key is reported rather than rejected,
    // and why the gate prints the degraded arms beside it.
    config
        .consulted_field_keys
        .borrow_mut()
        .insert(key.to_string());
    match config.int_fields.get(key) {
        Some((ty, signed)) => (
            quote! { ::core::mem::size_of::<#ty>() },
            quote! { #signed },
            // Fails to compile unless the field really has the declared type,
            // so the declaration cannot drift from the struct it describes.
            quote! {
                const _: fn(&#struct_path) -> #ty = |__s| __s.#member;
            },
        ),
        // `scalar_size`'s own default for a non-`Ref` field: the `i64` storage,
        // which is 8 on every target and NOT the machine word (a `usize` here
        // would report 4 on wasm32).
        //
        // That default is a CLAIM about the field — eight bytes — and it used to
        // be the one claim here that carried no witness.  The declared arm above
        // cannot drift from the struct; the undeclared arm could say eight bytes
        // about a `u32` and register a descr four bytes too wide, with the
        // sub-word range a declaration exists to buy silently gone.  Omission is
        // the dangerous direction precisely because it is spelled as nothing at
        // all, so the default witnesses itself too.
        //
        // WIDTH, not type identity, because width is what the claim is about.
        // The storage this registers is reached as a raw eight-byte word, and a
        // `#[repr(transparent)]` newtype over `i64` — a tagged value word is the
        // usual one — is exactly that word.  Demanding the field spell `i64`
        // would reject those for no defect.  What stays uncheckable either way
        // is the sign, which no Rust type-level test can recover from a field
        // the macro only knows by name.
        None => (
            quote! { ::core::mem::size_of::<i64>() },
            quote! { true },
            match ref_field_witness_tokens(&config.ref_fields, key, struct_path, member) {
                // A ref field is a pointer word and carries its own witness.
                witness if !witness.is_empty() => witness,
                _ => {
                    let message = format!(
                        "`{key}` is not the eight-byte scalar an undeclared field is \
                         registered as. Name its Rust integer type in `int_fields`, or its \
                         pointee in `ref_fields` / its element type in `array_fields`, so \
                         the emitted descr reports the field's own width instead of this \
                         default.",
                    );
                    quote! {
                        const _: () = {
                            // Names the field's type without spelling it, which
                            // is the only handle available here: the macro sees
                            // the member, not its declaration.
                            //
                            // Through a reference, not by value. `|s| s.field`
                            // returns `T` and so moves out of a shared borrow,
                            // which only compiles for a `Copy` field — an
                            // eight-byte field that is not `Copy` would fail
                            // expansion with a borrow error naming generated
                            // code, for a struct with nothing wrong with it.
                            const fn __field_width<T>(_: fn(&#struct_path) -> &T) -> usize {
                                ::core::mem::size_of::<T>()
                            }
                            assert!(
                                __field_width(|__s: &#struct_path| &__s.#member)
                                    == ::core::mem::size_of::<i64>(),
                                #message,
                            );
                        };
                    }
                }
            },
        ),
    }
}

/// A compile-time check that a `ref_fields` entry describes the field it names.
///
/// The lowering trusts the declaration twice and verifies it nowhere: the read
/// becomes `getfield_gc_r` into the ref bank, and the binding's `struct_type`
/// is what the NEXT hop resolves `offset_of!` against.  So a field that is not
/// a pointer to the declared pointee produces either a ref-bank read of
/// something that is not a reference, or an offset computed in the wrong
/// struct, and nothing between the declaration and the emitted descr says so.
///
/// `usize` is admitted alongside the two raw-pointer spellings because it is
/// the sanctioned carrier for a pointer whose declaring crate would rather not
/// name the pointee's type.  A carrier's pointee is not checkable, which is the
/// price of the carrier — what survives for it is that the field is
/// pointer-width and pointer-kind, not an `i64` or a `u32` that drifted into
/// the ref map.
///
/// The witness names the pointee rather than a whole pointer type so that
/// `*mut T` and `*const T` both satisfy it, mirroring `emit_array_field_base`.
///
/// Where this adds coverage, measured rather than assumed.  On the
/// `#[jit_inline]` path a drifted pointee on a raw-pointer field already fails
/// the build: the concrete rewriter types the loaded value against the
/// declaration and reports E0308.  On the `#[jit_interp]` state-field path it
/// does not — a machine declaring `Holder::link => Wrong` over a
/// `link: *mut Holder` compiles clean and faults at run time.  That path is
/// what this witness closes; on the inline path it is a second, earlier line.
///
/// Empty for a key `ref_fields` does not declare, and that is a decision, not
/// an omission: an undeclared field reads into the Int bank, and stable Rust
/// has no way to assert that a type is *not* a pointer.  Catching that half
/// takes a mechanism that works by disagreement instead of by declaration —
/// `Assembler::register_struct_layout`'s conflict check, which reports one word
/// registered as a pointer at one emit site and as a scalar at another.
///
/// It is not a general second line for this one, and the reason is worth
/// stating: each jitcode gets its OWN layout map, so two *declarations* never
/// meet there.  What meets is two emit sites within one jitcode — the same
/// member reached by two lowering paths.  A machine whose sole access to an
/// undeclared pointer field goes through one path stays undetected by both
/// mechanisms.
fn ref_field_witness_tokens(
    ref_fields: &HashMap<String, (syn::Path, Ident, syn::Path)>,
    key: &str,
    struct_path: &syn::Path,
    member: &syn::Member,
) -> TokenStream {
    let Some((_, _, pointee)) = ref_fields.get(key) else {
        return quote! {};
    };
    quote! {
        const _: () = {
            trait __MajitRefField {}
            impl __MajitRefField for *mut #pointee {}
            impl __MajitRefField for *const #pointee {}
            impl __MajitRefField for usize {}
            #[allow(dead_code)]
            fn __majit_ref_field_witness(__s: &#struct_path) {
                fn __accept<T: __MajitRefField>(_: T) {}
                __accept(__s.#member);
            }
        };
    }
}

/// The `(base_size, len_offset, witness)` a `pool_arrays` declaration reports
/// for its array, and the compile-time checks that the declaration describes
/// the struct it names.
///
/// Both numbers are `offset_of!` on the declaration's own field names, so a
/// field reordered ahead of the items moves the read with it instead of leaving
/// it behind.  The witnesses cover what an offset cannot: that `items` really
/// is an array of pointer-width elements, and that `len` really is a `usize` —
/// the width the descr's lendescr reads it at.  Without the second one, a `u32`
/// length word would be read as a machine word with its top half taken from
/// whatever follows.
///
/// The element witness needs the declared pointee, so it is emitted only for a
/// declaration that names one.  `offset_of!` is unconditional.
fn pool_array_layout_tokens(
    entry: &super::PoolArrayLowering,
) -> (TokenStream, TokenStream, TokenStream) {
    let struct_path = &entry.struct_path;
    let items = &entry.items_field;
    let base_size = quote! { ::core::mem::offset_of!(#struct_path, #items) };
    let (len_offset, len_witness) = match &entry.len_field {
        Some(len) => (
            quote! { ::core::option::Option::Some(::core::mem::offset_of!(#struct_path, #len)) },
            quote! {
                const _: fn(&#struct_path) -> usize = |__s| __s.#len;
            },
        ),
        None => (quote! { ::core::option::Option::None }, quote! {}),
    };
    let element_witness = match &entry.element_type {
        Some(element) => quote! {
            const _: fn(&#struct_path) -> *mut #element = |__s| __s.#items[0];
        },
        None => quote! {},
    };
    (
        base_size,
        len_offset,
        quote! { #len_witness #element_witness },
    )
}

/// A `<local ref binding>.<field>` access that `array_fields` declares, resolved
/// but not yet emitted.  See [`JitCodeLowerer::match_array_field_base`].
struct ArrayFieldBase {
    base_reg: u16,
    struct_path: syn::Path,
    member: syn::Member,
    element_type: syn::Path,
}

impl<'c> Lowerer<'c> {
    /// Read an immediate operand byte from the green bytecode array:
    /// `program[<index>]`.  Mirrors the dispatch-top opcode fetch
    /// (`try_lower_opcode_fetch_stmt`) so the same `getarrayitem_gc_i` is
    /// emitted for operand reads inside opcode arm bodies.  RPython
    /// `pyopcode.py:171 ord(co_code[next_instr])`: the bytecode array and
    /// `pc` are green, so the optimizer constant-folds the load.
    ///
    /// The env parameter name is the macro convention `program`, matching
    /// `try_lower_opcode_fetch_stmt`'s hard-coded recognition.
    pub(super) fn lower_env_array_read(&mut self, expr: &Expr) -> Option<Binding> {
        // The constant-fold this read relies on ("array + index are green",
        // `pyopcode.py:171 ord(co_code[next_instr])`) only holds when `pc` is
        // a declared green — PyPy `tl.py greens=['pc','code']` makes both the
        // bytecode array and the instruction pointer green, so the load folds
        // away.  With a red `pc` the `getarrayitem_gc_i` does not fold and the
        // surrounding state-field dispatch loop cannot close on it; leave the
        // operand read unlowered so the arm aborts and the interpreter handles
        // that opcode (the pre-green-pc behaviour).
        if !self.config.map(pc_is_green).unwrap_or(false) {
            return None;
        }
        let Expr::Index(index_expr) = expr else {
            return None;
        };
        if !expr_matches_local_name(&index_expr.expr, "program") {
            return None;
        }
        let program_binding = self.bindings.get("program").cloned()?;
        if !matches!(program_binding.kind, BindingKind::Ref) {
            return None;
        }
        let program_reg = program_binding.reg;
        let idx_binding = self.lower_value_expr(&index_expr.index)?;
        if !matches!(idx_binding.kind, BindingKind::Int) {
            return None;
        }
        let index_reg = idx_binding.reg;
        let result_reg = self.alloc_reg();
        // The descr scales the index by the env element size (see
        // `env_array_descr_expr`): `&[u8]` reads a raw byte, a wider env
        // (`&[i64]`) reads the element at byte offset `size_of::<elem>() *
        // index`. Shared with the opcode-fetch sites so operands and opcodes
        // read at the same stride.
        let descr_tok = env_array_descr_expr(self.config);
        self.emit_op(
            OpMeta::linear(
                OpKind::Vable,
                vec![Register::ref_(program_reg), Register::int(index_reg)],
                vec![Register::int(result_reg)],
            ),
            quote! {
                let __descr_idx = #descr_tok;
                __builder.getarrayitem_gc_i(
                    #result_reg as u16,
                    #program_reg as u16,
                    #index_reg as u16,
                    __descr_idx,
                );
            },
        );
        Some(Binding {
            reg: result_reg,
            kind: BindingKind::Int,
            depends_on_stack: false,
            struct_type: None,
        })
    }

    pub(super) fn lower_vable_field_write(&mut self, expr: &Expr) -> Option<()> {
        let config = self.config?;
        let vable_var = config.vable_var.as_ref()?;

        let assign = match expr {
            Expr::Assign(a) => a,
            _ => return None,
        };
        let field = match &*assign.left {
            Expr::Field(f) => f,
            _ => return None,
        };
        if !expr_matches_local_name(&field.base, vable_var) {
            return None;
        }
        let member_name = named_member(&field.member)?;
        let &(field_index, field_type) = config.vable_fields.get(&member_name)?;
        let vable_reg = self.vable_base_reg()?;
        let fi = field_index as u16;
        let binding = self.lower_value_expr(&assign.right)?;
        let src = binding.reg;
        // vable_reg is always Ref (the virtualizable input register); src bank
        // follows `field_type` per `assembler.py:217` argcode mapping.
        // jtransform.py:926 — `-live-` precedes `setfield_vable_*`.
        self.emit_op(
            OpMeta::live_marker(),
            quote! { let _ = __builder.live_placeholder(); },
        );
        let vable_r = Register::ref_(vable_reg);
        match field_type {
            ValueKind::Ref => self.emit_op(
                OpMeta::linear(OpKind::Vable, vec![vable_r, Register::ref_(src)], vec![]),
                quote! { __builder.vable_setfield_ref_with_base(#vable_reg, #fi, #src); },
            ),
            ValueKind::Float => self.emit_op(
                OpMeta::linear(OpKind::Vable, vec![vable_r, Register::float(src)], vec![]),
                quote! { __builder.vable_setfield_float_with_base(#vable_reg, #fi, #src); },
            ),
            ValueKind::Int => self.emit_op(
                OpMeta::linear(OpKind::Vable, vec![vable_r, Register::int(src)], vec![]),
                quote! { __builder.vable_setfield_int_with_base(#vable_reg, #fi, #src); },
            ),
        }
        Some(())
    }

    /// RPython jtransform.py:794 `setarrayitem_vable_*`.
    ///
    /// Recognizes `frame.locals_w[i] = val` and emits vable_setarrayitem.
    pub(super) fn lower_vable_array_write(&mut self, expr: &Expr) -> Option<()> {
        let config = self.config?;
        let vable_var = config.vable_var.as_ref()?;

        let assign = match expr {
            Expr::Assign(a) => a,
            _ => return None,
        };
        // LHS: frame.array_field[index]
        let index_expr = match &*assign.left {
            Expr::Index(idx) => idx,
            _ => return None,
        };
        let field = match &*index_expr.expr {
            Expr::Field(f) => f,
            _ => return None,
        };
        if !expr_matches_local_name(&field.base, vable_var) {
            return None;
        }
        let member_name = named_member(&field.member)?;
        let &(array_index, item_type) = config.vable_arrays.get(&member_name)?;
        let vable_reg = self.vable_base_reg()?;
        let ai = array_index as u16;

        // Lower index and value
        let idx_binding = self.lower_value_expr(&index_expr.index)?;
        let idx_reg = idx_binding.reg;
        let val_binding = self.lower_value_expr(&assign.right)?;
        let val_reg = val_binding.reg;

        // vable_reg: Ref. idx_reg: Int (array index). val_reg: bank by item_type.
        // jtransform.py:798 — `-live-` precedes `setarrayitem_vable_*`.
        self.emit_op(
            OpMeta::live_marker(),
            quote! { let _ = __builder.live_placeholder(); },
        );
        let vable_r = Register::ref_(vable_reg);
        let idx_r = Register::int(idx_reg);
        match item_type {
            ValueKind::Ref => self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![vable_r, idx_r, Register::ref_(val_reg)],
                    vec![],
                ),
                quote! { __builder.vable_setarrayitem_ref_with_base(#vable_reg, #ai, #idx_reg, #val_reg); },
            ),
            ValueKind::Float => self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![vable_r, idx_r, Register::float(val_reg)],
                    vec![],
                ),
                quote! { __builder.vable_setarrayitem_float_with_base(#vable_reg, #ai, #idx_reg, #val_reg); },
            ),
            ValueKind::Int => self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![vable_r, idx_r, Register::int(val_reg)],
                    vec![],
                ),
                quote! { __builder.vable_setarrayitem_int_with_base(#vable_reg, #ai, #idx_reg, #val_reg); },
            ),
        }
        Some(())
    }

    /// Recognizes `frame.array_field[index] += value` and the other numeric
    /// compound assignments. Rust evaluates `index` once, so the lowering
    /// shares one index register between the vable read and write.
    pub(super) fn lower_vable_array_update(&mut self, expr: &Expr) -> Option<()> {
        let config = self.config?;
        let vable_var = config.vable_var.as_ref()?;
        let binary = match expr {
            Expr::Binary(binary) => binary,
            _ => return None,
        };
        let index_expr = match &*binary.left {
            Expr::Index(index) => index,
            _ => return None,
        };
        let field = match &*index_expr.expr {
            Expr::Field(field) => field,
            _ => return None,
        };
        if !expr_matches_local_name(&field.base, vable_var) {
            return None;
        }
        let member_name = named_member(&field.member)?;
        let &(array_index, item_type) = config.vable_arrays.get(&member_name)?;
        let opcode = match item_type {
            ValueKind::Float => opcode_for_assign_binop_f(&binary.op)?,
            ValueKind::Int => opcode_for_assign_binop(&binary.op)?,
            ValueKind::Ref => return None,
        };
        let vable_reg = self.vable_base_reg()?;
        let idx_binding = self.lower_value_expr(&index_expr.index)?;
        if !matches!(idx_binding.kind, BindingKind::Int) {
            return None;
        }
        let idx_reg = idx_binding.reg;
        let rhs = self.lower_value_expr(&binary.right)?;
        match item_type {
            ValueKind::Float if !matches!(rhs.kind, BindingKind::Float) => return None,
            ValueKind::Int if !matches!(rhs.kind, BindingKind::Int) => return None,
            ValueKind::Ref => unreachable!(),
            _ => {}
        }
        let ai = array_index as u16;
        let lhs_reg = self.alloc_reg();

        self.emit_op(
            OpMeta::live_marker(),
            quote! { let _ = __builder.live_placeholder(); },
        );
        match item_type {
            ValueKind::Float => self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![Register::ref_(vable_reg), Register::int(idx_reg)],
                    vec![Register::float(lhs_reg)],
                ),
                quote! { __builder.vable_getarrayitem_float_with_base(#lhs_reg, #vable_reg, #ai, #idx_reg); },
            ),
            ValueKind::Int => self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![Register::ref_(vable_reg), Register::int(idx_reg)],
                    vec![Register::int(lhs_reg)],
                ),
                quote! { __builder.vable_getarrayitem_int_with_base(#lhs_reg, #vable_reg, #ai, #idx_reg); },
            ),
            ValueKind::Ref => unreachable!(),
        }

        let dst = self.alloc_reg();
        match item_type {
            ValueKind::Float => self.emit_op(
                OpMeta::linear(
                    OpKind::BinopF,
                    vec![Register::float(lhs_reg), Register::float(rhs.reg)],
                    vec![Register::float(dst)],
                ),
                binop_f_emit_tokens(dst, &opcode, lhs_reg, rhs.reg),
            ),
            ValueKind::Int => self.emit_op(
                OpMeta::linear(
                    OpKind::BinopI,
                    Register::ints(&[lhs_reg, rhs.reg]),
                    vec![Register::int(dst)],
                ),
                binop_i_emit_tokens(dst, &opcode, lhs_reg, rhs.reg),
            ),
            ValueKind::Ref => unreachable!(),
        }

        self.emit_op(
            OpMeta::live_marker(),
            quote! { let _ = __builder.live_placeholder(); },
        );
        match item_type {
            ValueKind::Float => self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![
                        Register::ref_(vable_reg),
                        Register::int(idx_reg),
                        Register::float(dst),
                    ],
                    vec![],
                ),
                quote! { __builder.vable_setarrayitem_float_with_base(#vable_reg, #ai, #idx_reg, #dst); },
            ),
            ValueKind::Int => self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![
                        Register::ref_(vable_reg),
                        Register::int(idx_reg),
                        Register::int(dst),
                    ],
                    vec![],
                ),
                quote! { __builder.vable_setarrayitem_int_with_base(#vable_reg, #ai, #idx_reg, #dst); },
            ),
            ValueKind::Ref => unreachable!(),
        }
        Some(())
    }

    /// Recognizes `state.field = expr` for scalar state fields.
    pub(super) fn lower_state_field_write(&mut self, expr: &Expr) -> Option<()> {
        let config = self.config?;
        let assign = match expr {
            Expr::Assign(a) => a,
            _ => return None,
        };
        let field = match &*assign.left {
            Expr::Field(f) => f,
            _ => return None,
        };
        let base = &field.base;
        if !expr_matches_local_name(base, "state") {
            return None;
        }
        let member_name = named_member(&field.member)?;
        if let Some(&field_index) = config.state_scalars.get(&member_name) {
            let fi = field_index as u16;
            let binding = self.lower_value_expr(&assign.right)?;
            let src = binding.reg;
            // store_state_field/di — `src` is Int per assembler.py:217 'i' argcode.
            self.emit_op(
                OpMeta::linear(OpKind::StateField, vec![Register::int(src)], vec![]),
                quote! { __builder.store_state_field(#fi, #src); },
            );
            return Some(());
        }
        // ref(T) scalar: the RHS must lower to a ref binding (another ref
        // state read or a residual ref-returning call).
        if let Some((field_index, _)) = config.state_ref_scalars.get(&member_name) {
            let fi = *field_index as u16;
            let binding = self.lower_value_expr(&assign.right)?;
            if !matches!(binding.kind, BindingKind::Ref) {
                return None;
            }
            let src = binding.reg;
            self.emit_op(
                OpMeta::linear(OpKind::StateField, vec![Register::ref_(src)], vec![]),
                quote! { __builder.store_state_field_ref(#fi, #src); },
            );
            return Some(());
        }
        if let Some(&field_index) = config.state_float_scalars.get(&member_name) {
            let fi = field_index as u16;
            let binding = self.lower_value_expr(&assign.right)?;
            if !matches!(binding.kind, BindingKind::Float) {
                return None;
            }
            let src = binding.reg;
            self.emit_op(
                OpMeta::linear(OpKind::StateField, vec![Register::float(src)], vec![]),
                quote! { __builder.store_state_field_float(#fi, #src); },
            );
            return Some(());
        }
        None
    }

    /// Recognizes `state.field += expr` for scalar state fields.
    pub(super) fn lower_state_field_update(&mut self, expr: &Expr) -> Option<()> {
        let config = self.config?;
        let binary = match expr {
            Expr::Binary(binary) => binary,
            _ => return None,
        };
        let field = match &*binary.left {
            Expr::Field(f) => f,
            _ => return None,
        };
        if !expr_matches_local_name(&field.base, "state") {
            return None;
        }
        let member_name = named_member(&field.member)?;
        if let Some(&field_index) = config.state_scalars.get(&member_name) {
            let opcode = opcode_for_assign_binop(&binary.op)?;
            let lhs = self.lower_state_field_read(&binary.left)?;
            let rhs = self.lower_value_expr(&binary.right)?;
            if !matches!(lhs.kind, BindingKind::Int) || !matches!(rhs.kind, BindingKind::Int) {
                return None;
            }
            let dst = self.alloc_reg();
            let lhs_reg = lhs.reg;
            let rhs_reg = rhs.reg;
            self.emit_op(
                OpMeta::linear(
                    OpKind::BinopI,
                    Register::ints(&[lhs_reg, rhs_reg]),
                    vec![Register::int(dst)],
                ),
                binop_i_emit_tokens(dst, &opcode, lhs_reg, rhs_reg),
            );
            let fi = field_index as u16;
            self.emit_op(
                OpMeta::linear(OpKind::StateField, vec![Register::int(dst)], vec![]),
                quote! { __builder.store_state_field(#fi, #dst); },
            );
            return Some(());
        }

        let &field_index = config.state_float_scalars.get(&member_name)?;
        let opcode = opcode_for_assign_binop_f(&binary.op)?;
        let lhs = self.lower_state_field_read(&binary.left)?;
        let rhs = self.lower_value_expr(&binary.right)?;
        if !matches!(lhs.kind, BindingKind::Float) || !matches!(rhs.kind, BindingKind::Float) {
            return None;
        }
        let dst = self.alloc_reg();
        let lhs_reg = lhs.reg;
        let rhs_reg = rhs.reg;
        self.emit_op(
            OpMeta::linear(
                OpKind::BinopF,
                vec![Register::float(lhs_reg), Register::float(rhs_reg)],
                vec![Register::float(dst)],
            ),
            binop_f_emit_tokens(dst, &opcode, lhs_reg, rhs_reg),
        );
        let fi = field_index as u16;
        self.emit_op(
            OpMeta::linear(OpKind::StateField, vec![Register::float(dst)], vec![]),
            quote! { __builder.store_state_field_float(#fi, #dst); },
        );
        Some(())
    }

    /// Recognizes `state.array[index] = expr` for array state fields.
    /// Virtualizable arrays bail to the standard vable path; flattened arrays
    /// emit `store_state_array`.
    pub(super) fn lower_state_array_write(&mut self, expr: &Expr) -> Option<()> {
        let config = self.config?;
        let assign = match expr {
            Expr::Assign(a) => a,
            _ => return None,
        };
        let index_expr = match &*assign.left {
            Expr::Index(idx) => idx,
            _ => return None,
        };
        let field = match &*index_expr.expr {
            Expr::Field(f) => f,
            _ => return None,
        };
        let base = &field.base;
        if !expr_matches_local_name(base, "state") {
            return None;
        }
        let member_name = named_member(&field.member)?;
        // `[int; virt]` arrays are NOT handled here — they lower through the
        // standard virtualizable path (`lower_vable_array_write`), reached
        // because `LowererConfig::new` registers the state binding as the
        // vable identity var. Bail before lowering the index/value so they
        // are not emitted twice when the dispatch falls through.
        if config.state_virt_arrays.contains_key(&member_name) {
            return None;
        }
        let idx_binding = self.lower_value_expr(&index_expr.index)?;
        let idx_reg = idx_binding.reg;
        let val_binding = self.lower_value_expr(&assign.right)?;
        let val_reg = val_binding.reg;

        // store_state_array/dii — both reg args are Int per
        // assembler.py:217 'i' argcode.
        let idx_r = Register::int(idx_reg);
        let val_r = Register::int(val_reg);
        let &array_index = config.state_arrays.get(&member_name)?;
        let ai = array_index as u16;
        self.emit_op(
            OpMeta::linear(OpKind::StateField, vec![idx_r, val_r], vec![]),
            quote! { __builder.store_state_array(#ai, #idx_reg, #val_reg); },
        );
        Some(())
    }

    /// RPython jtransform.py:650 `hint_force_virtualizable`.
    ///
    /// Recognizes `hint_force_virtualizable!(frame)` macro invocation.
    pub(super) fn lower_vable_force(&mut self, expr: &Expr) -> Option<()> {
        let config = self.config?;
        let _vable_var = config.vable_var.as_ref()?;

        let mac = match expr {
            Expr::Macro(m) => m,
            _ => return None,
        };
        let hint = classify_virtualizable_hint_syn_path(&mac.mac.path)?;
        if hint != VirtualizableHintKind::ForceVirtualizable {
            return None;
        }
        let arg: Expr = syn::parse2(mac.mac.tokens.clone()).ok()?;
        let binding = self.lower_value_expr(&arg)?;
        let vable_reg = binding.reg;
        // vable_force/r — vable_reg is Ref per assembler.py:217 'r' argcode.
        self.emit_op(
            OpMeta::linear(OpKind::Vable, vec![Register::ref_(vable_reg)], vec![]),
            quote! { __builder.vable_force_with_base(#vable_reg); },
        );
        Some(())
    }

    /// RPython jtransform.py:655 — suppress identity hint function calls.
    ///
    /// `hint_access_directly(frame)` and `hint_fresh_virtualizable(frame)`
    /// are identity functions that return their argument unchanged.
    /// The Lowerer recognizes these calls and lowers the argument directly,
    /// effectively eliminating the hint call.
    pub(super) fn lower_vable_hint_identity_call(&mut self, expr: &Expr) -> Option<Binding> {
        let call = match expr {
            Expr::Call(c) => c,
            _ => return None,
        };
        let func_name = match &*call.func {
            Expr::Path(p) => classify_virtualizable_hint_syn_path(&p.path),
            _ => return None,
        };
        match func_name {
            Some(
                VirtualizableHintKind::AccessDirectly | VirtualizableHintKind::FreshVirtualizable,
            ) => {
                let arg = call.args.first()?;
                self.lower_value_expr(arg)
            }
            _ => None,
        }
    }

    /// RPython jtransform.py:655 `hint(access_directly=True)` /
    /// `hint(fresh_virtualizable=True)`.
    ///
    /// These hints are consumed by the translator — jtransform suppresses
    /// them (returns None = no opcode generated). The codewriter has already
    /// rewritten field accesses to use vable_getfield/setfield, so the
    /// access_directly hint is redundant at this point.
    ///
    /// In majit, the Lowerer recognizes these macro calls and emits nothing,
    /// which matches RPython's behavior exactly.
    pub(super) fn lower_vable_hint_suppress(&self, expr: &Expr) -> Option<()> {
        let _config = self.config?;
        let mac = match expr {
            Expr::Macro(m) => m,
            _ => return None,
        };
        match classify_virtualizable_hint_syn_path(&mac.mac.path) {
            Some(
                VirtualizableHintKind::AccessDirectly | VirtualizableHintKind::FreshVirtualizable,
            ) => Some(()),
            _ => None,
        }
    }

    // ── conditional_call / record_known_result JIT op emission ──────

    /// RPython jtransform.py:832 `getfield_vable_*`.
    ///
    /// Recognizes `frame.field` where `frame` is the virtualizable variable
    /// and `field` is a declared virtualizable scalar field.
    pub(super) fn lower_vable_field_read(&mut self, expr: &Expr) -> Option<Binding> {
        let config = self.config?;
        let vable_var = config.vable_var.as_ref()?;

        if let Expr::Field(field) = expr {
            if !expr_matches_local_name(&field.base, vable_var) {
                return None;
            }
            let member_name = named_member(&field.member)?;

            if let Some(&(field_index, field_type)) = config.vable_fields.get(&member_name) {
                let vable_reg = self.vable_base_reg()?;
                let reg = self.alloc_reg();
                let fi = field_index as u16;
                // vable_reg is Ref; result `reg` bank follows field_type.
                // jtransform.py:845 — `-live-` precedes `getfield_vable_*`.
                self.emit_op(
                    OpMeta::live_marker(),
                    quote! { let _ = __builder.live_placeholder(); },
                );
                let vable_r = Register::ref_(vable_reg);
                let kind = match field_type {
                    ValueKind::Ref => {
                        self.emit_op(
                            OpMeta::linear(
                                OpKind::Vable,
                                vec![vable_r],
                                vec![Register::ref_(reg)],
                            ),
                            quote! { __builder.vable_getfield_ref_with_base(#reg, #vable_reg, #fi); },
                        );
                        BindingKind::Ref
                    }
                    ValueKind::Float => {
                        self.emit_op(
                            OpMeta::linear(
                                OpKind::Vable,
                                vec![vable_r],
                                vec![Register::float(reg)],
                            ),
                            quote! { __builder.vable_getfield_float_with_base(#reg, #vable_reg, #fi); },
                        );
                        BindingKind::Float
                    }
                    ValueKind::Int => {
                        self.emit_op(
                            OpMeta::linear(
                                OpKind::Vable,
                                vec![vable_r],
                                vec![Register::int(reg)],
                            ),
                            quote! { __builder.vable_getfield_int_with_base(#reg, #vable_reg, #fi); },
                        );
                        BindingKind::Int
                    }
                };
                return Some(Binding {
                    reg,
                    kind,
                    depends_on_stack: false,
                    struct_type: None,
                });
            }
        }
        None
    }

    /// RPython jtransform.py:760 `getarrayitem_vable_*`.
    ///
    /// Recognizes `frame.locals_w[i]` where `frame` is the virtualizable
    /// variable and `locals_w` is a declared virtualizable array field.
    pub(super) fn lower_vable_array_read(&mut self, expr: &Expr) -> Option<Binding> {
        let config = self.config?;
        let vable_var = config.vable_var.as_ref()?;

        // Pattern: Expr::Index where base is Expr::Field on vable_var
        let index_expr = match expr {
            Expr::Index(idx) => idx,
            _ => return None,
        };
        let field = match &*index_expr.expr {
            Expr::Field(f) => f,
            _ => return None,
        };
        if !expr_matches_local_name(&field.base, vable_var) {
            return None;
        }
        let member_name = named_member(&field.member)?;
        let &(array_index, item_type) = config.vable_arrays.get(&member_name)?;
        let vable_reg = self.vable_base_reg()?;

        // Lower the index expression to a register
        let idx_binding = self.lower_value_expr(&index_expr.index)?;
        let idx_reg = idx_binding.reg;

        let reg = self.alloc_reg();
        let ai = array_index as u16;
        // vable_reg: Ref. idx_reg: Int. result `reg` bank by item_type.
        // jtransform.py:764 — `-live-` precedes `getarrayitem_vable_*`.
        self.emit_op(
            OpMeta::live_marker(),
            quote! { let _ = __builder.live_placeholder(); },
        );
        let vable_r = Register::ref_(vable_reg);
        let idx_r = Register::int(idx_reg);
        let kind = match item_type {
            ValueKind::Ref => {
                self.emit_op(
                    OpMeta::linear(
                        OpKind::Vable,
                        vec![vable_r, idx_r],
                        vec![Register::ref_(reg)],
                    ),
                    quote! { __builder.vable_getarrayitem_ref_with_base(#reg, #vable_reg, #ai, #idx_reg); },
                );
                BindingKind::Ref
            }
            ValueKind::Float => {
                self.emit_op(
                    OpMeta::linear(
                        OpKind::Vable,
                        vec![vable_r, idx_r],
                        vec![Register::float(reg)],
                    ),
                    quote! { __builder.vable_getarrayitem_float_with_base(#reg, #vable_reg, #ai, #idx_reg); },
                );
                BindingKind::Float
            }
            ValueKind::Int => {
                self.emit_op(
                    OpMeta::linear(
                        OpKind::Vable,
                        vec![vable_r, idx_r],
                        vec![Register::int(reg)],
                    ),
                    quote! { __builder.vable_getarrayitem_int_with_base(#reg, #vable_reg, #ai, #idx_reg); },
                );
                BindingKind::Int
            }
        };
        Some(Binding {
            reg,
            kind,
            depends_on_stack: false,
            struct_type: None,
        })
    }

    /// RPython jtransform.py:815 `arraylen_vable`.
    ///
    /// Recognizes `frame.locals_w.len()` for declared virtualizable arrays.
    pub(super) fn lower_vable_array_len(&mut self, expr: &Expr) -> Option<Binding> {
        let config = self.config?;
        let vable_var = config.vable_var.as_ref()?;
        let call = match expr {
            Expr::MethodCall(call) => call,
            _ => return None,
        };
        if call.method != "len" || !call.args.is_empty() {
            return None;
        }
        let field = match &*call.receiver {
            Expr::Field(field) => field,
            _ => return None,
        };
        if !expr_matches_local_name(&field.base, vable_var) {
            return None;
        }
        let member_name = named_member(&field.member)?;
        let &array_index = config.vable_arrays.get(&member_name).map(|(idx, _)| idx)?;
        let vable_reg = self.vable_base_reg()?;
        let reg = self.alloc_reg();
        let ai = array_index as u16;
        // jtransform.py:814 — `-live-` precedes `arraylen_vable`.
        self.emit_op(
            OpMeta::live_marker(),
            quote! { let _ = __builder.live_placeholder(); },
        );
        // vable_arraylen reads vable_reg (Ref) and writes the length to an int reg.
        self.emit_op(
            OpMeta::linear(
                OpKind::Vable,
                vec![Register::ref_(vable_reg)],
                vec![Register::int(reg)],
            ),
            quote! { __builder.vable_arraylen_with_base(#reg, #vable_reg, #ai); },
        );
        Some(Binding {
            reg,
            kind: BindingKind::Int,
            depends_on_stack: false,
            struct_type: None,
        })
    }

    /// Recognizes `state.field` for scalar state fields.
    pub(super) fn lower_state_field_read(&mut self, expr: &Expr) -> Option<Binding> {
        let config = self.config?;
        let field = match expr {
            Expr::Field(f) => f,
            _ => return None,
        };
        let base = &field.base;
        if !expr_matches_local_name(base, "state") {
            return None;
        }
        let member_name = named_member(&field.member)?;
        if let Some(&field_index) = config.state_scalars.get(&member_name) {
            let fi = field_index as u16;
            let reg = self.alloc_reg();
            // load_state_field reads the field at its int identity slot into
            // int `reg`.  Declare the identity slot as a read (liveness.py:67
            // adds every Register arg to the alive set) so the backward
            // liveness walk keeps the slot live across every downstream
            // `-live-` marker — without it the marker omits the slot, the
            // blackhole resume seeder never restores it, and forward
            // re-execution reads garbage.  Mirrors `store_state_field`
            // declaring its source and `vable_field_read` declaring `vable_r`.
            // The slot sits at `int_identity_base() + fi`, past the dispatch
            // JitCode's int argument (`pc` at i0).
            let slot = config.int_identity_base() + fi;
            self.emit_op(
                OpMeta::linear(
                    OpKind::StateField,
                    vec![Register::int(slot)],
                    vec![Register::int(reg)],
                ),
                quote! { __builder.load_state_field(#fi, #reg); },
            );
            return Some(Binding {
                reg,
                kind: BindingKind::Int,
                depends_on_stack: false,
                struct_type: None,
            });
        }
        // ref(T) scalar: read into the ref register bank so a subsequent
        // getfield_gc reads its struct base from a real ref value.
        if let Some((field_index, _)) = config.state_ref_scalars.get(&member_name) {
            let fi = *field_index as u16;
            let reg = self.alloc_reg();
            // Declare the ref identity slot as a read so the backward
            // liveness walk keeps it live across guards (mirrors the int
            // load_state_field). Without it the slot drops from interior
            // -live- markers and the blackhole resume seeder leaves
            // registers_r[ref_scalar_slot(fi)] uninitialized. The slot
            // sits at `ref_identity_base() + fi`, past the dispatch
            // JitCode's ref-bank arguments.
            let slot = config.ref_identity_base() + fi;
            self.emit_op(
                OpMeta::linear(
                    OpKind::StateField,
                    vec![Register::ref_(slot)],
                    vec![Register::ref_(reg)],
                ),
                quote! { __builder.load_state_field_ref(#fi, #reg); },
            );
            return Some(Binding {
                reg,
                kind: BindingKind::Ref,
                depends_on_stack: false,
                struct_type: None,
            });
        }
        if let Some(&field_index) = config.state_float_scalars.get(&member_name) {
            let fi = field_index as u16;
            let reg = self.alloc_reg();
            let slot = config.float_identity_base() + fi;
            self.emit_op(
                OpMeta::linear(
                    OpKind::StateField,
                    vec![Register::float(slot)],
                    vec![Register::float(reg)],
                ),
                quote! { __builder.load_state_field_float(#fi, #reg); },
            );
            return Some(Binding {
                reg,
                kind: BindingKind::Float,
                depends_on_stack: false,
                struct_type: None,
            });
        }
        None
    }

    /// Recognizes a field READ through a `ref(T)` state scalar:
    /// `state.<ref_scalar>.<member>` → `getfield_gc_i` on the heap object the
    /// ref points at.  Unlike the residual-call form (an opaque CALL_I whose
    /// result the optimizer can neither re-produce in the short preamble nor
    /// invalidate on a write), a `getfield_gc` on a non-immutable field is
    /// re-readable each loop entry and is invalidated by a matching
    /// `setfield_gc` on the same `(struct_type_id, field)` — mirroring an
    /// RPython `len(obj)`/`obj.field` getfield_gc_i on a mutable field.
    ///
    /// Only int fields are lowered here (the length read); a ref field
    /// read would route through `getfield_gc_r` and is left unimplemented
    /// until a caller needs it.
    pub(super) fn lower_state_ref_field_getfield(&mut self, expr: &Expr) -> Option<Binding> {
        let config = self.config?;
        let Expr::Field(field) = expr else {
            return None;
        };
        // The base must be `state.<ref_scalar>` (a ref(T) state field).
        let Expr::Field(base_field) = &*field.base else {
            return None;
        };
        if !expr_matches_local_name(&base_field.base, "state") {
            return None;
        }
        let base_name = named_member(&base_field.member)?;
        let (_, struct_path) = config.state_ref_scalars.get(&base_name).cloned()?;
        let member = field.member.clone();
        let member_name = named_member(&field.member)?;
        // Check if this field is declared ref-kind in `ref_fields`.
        let struct_last = struct_path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_default();
        let ref_field_key = format!("{}::{}", struct_last, member_name);
        let (__fsize, __fsigned, __fcheck) =
            field_scalar_tokens(config, &ref_field_key, &struct_path, &member);
        let ref_field_entry = config.ref_fields.get(&ref_field_key);
        let is_ref_field = ref_field_entry.is_some();
        // Raw (headerless) ref-scalar pointee → `is_gc_managed = false`, a
        // distinct descriptor id from any GC `new_struct` of the same type.
        let tid = struct_type_id_tokens(&struct_path, false);
        // Lower the `state.<ref_scalar>` base to a ref binding (its
        // load_state_field_ref already declares the ref identity slot live for
        // resume), then read the field off that concrete ref.
        let base = self.lower_state_field_read(&field.base)?;
        if !matches!(base.kind, BindingKind::Ref) {
            return None;
        }
        let base_reg = base.reg;
        let result_reg = self.alloc_reg();
        if is_ref_field {
            // Ref-kind field → getfield_gc_r, result is a ref binding.
            self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![Register::ref_(base_reg)],
                    vec![Register::ref_(result_reg)],
                ),
                quote! {
                    #__fcheck
                    __builder.register_struct_layout(
                        ::core::mem::size_of::<#struct_path>(),
                        #tid,
                        false,
                        false,
                        &[(
                            ::core::mem::offset_of!(#struct_path, #member),
                            true,
                            stringify!(#member),
                            #__fsize,
                            #__fsigned,
                        )],
                    );
                    __builder.getfield_gc_r(
                        #result_reg,
                        #base_reg,
                        ::core::mem::offset_of!(#struct_path, #member),
                        #tid,
                        stringify!(#member),
                    );
                },
            );
            Some(Binding {
                reg: result_reg,
                kind: BindingKind::Ref,
                depends_on_stack: false,
                struct_type: ref_field_entry.map(|(_, _, pointee_path)| pointee_path.clone()),
            })
        } else {
            // Int-kind field (default) → getfield_gc_i.
            self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![Register::ref_(base_reg)],
                    vec![Register::int(result_reg)],
                ),
                quote! {
                    #__fcheck
                    __builder.register_struct_layout(
                        ::core::mem::size_of::<#struct_path>(),
                        #tid,
                        false,
                        false,
                        &[(
                            ::core::mem::offset_of!(#struct_path, #member),
                            false,
                            stringify!(#member),
                            #__fsize,
                            #__fsigned,
                        )],
                    );
                    __builder.getfield_gc_i(
                        #result_reg,
                        #base_reg,
                        ::core::mem::offset_of!(#struct_path, #member),
                        #tid,
                        stringify!(#member),
                    );
                },
            );
            Some(Binding {
                reg: result_reg,
                kind: BindingKind::Int,
                depends_on_stack: false,
                struct_type: None,
            })
        }
    }

    /// Field read on an arbitrary local ref binding with known struct type:
    /// `<ident>.<field>` where `<ident>` was bound as `BindingKind::Ref` with
    /// a non-None `struct_type`.  Uses the same `ref_fields` config as
    /// `lower_state_ref_field_getfield` to determine whether the field is
    /// ref-kind or int-kind.
    pub(super) fn lower_ref_binding_getfield(&mut self, expr: &Expr) -> Option<Binding> {
        let config = self.config?;
        let Expr::Field(field) = expr else {
            return None;
        };
        // Base must be a simple ident (a local variable), not `state.x.y`.
        let Expr::Path(path) = &*field.base else {
            return None;
        };
        let ident = path.path.get_ident()?;
        let binding = self.bindings.get(&ident.to_string())?.clone();
        if !matches!(binding.kind, BindingKind::Ref) {
            return None;
        }
        let struct_path = binding.struct_type.as_ref()?.clone();
        let member = field.member.clone();
        let member_name = named_member(&field.member)?;
        let struct_last = struct_path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_default();
        let ref_field_key = format!("{}::{}", struct_last, member_name);
        let (__fsize, __fsigned, __fcheck) =
            field_scalar_tokens(config, &ref_field_key, &struct_path, &member);
        let ref_field_entry = config.ref_fields.get(&ref_field_key);
        let is_ref_field = ref_field_entry.is_some();
        let tid = struct_type_id_tokens(&struct_path, false);
        let base_reg = binding.reg;
        let result_reg = self.alloc_reg();
        if is_ref_field {
            self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![Register::ref_(base_reg)],
                    vec![Register::ref_(result_reg)],
                ),
                quote! {
                    #__fcheck
                    __builder.register_struct_layout(
                        ::core::mem::size_of::<#struct_path>(),
                        #tid,
                        false,
                        false,
                        &[(
                            ::core::mem::offset_of!(#struct_path, #member),
                            true,
                            stringify!(#member),
                            #__fsize,
                            #__fsigned,
                        )],
                    );
                    __builder.getfield_gc_r(
                        #result_reg,
                        #base_reg,
                        ::core::mem::offset_of!(#struct_path, #member),
                        #tid,
                        stringify!(#member),
                    );
                },
            );
            Some(Binding {
                reg: result_reg,
                kind: BindingKind::Ref,
                depends_on_stack: false,
                struct_type: ref_field_entry.map(|(_, _, pointee_path)| pointee_path.clone()),
            })
        } else {
            self.emit_op(
                OpMeta::linear(
                    OpKind::Vable,
                    vec![Register::ref_(base_reg)],
                    vec![Register::int(result_reg)],
                ),
                quote! {
                    #__fcheck
                    __builder.register_struct_layout(
                        ::core::mem::size_of::<#struct_path>(),
                        #tid,
                        false,
                        false,
                        &[(
                            ::core::mem::offset_of!(#struct_path, #member),
                            false,
                            stringify!(#member),
                            #__fsize,
                            #__fsigned,
                        )],
                    );
                    __builder.getfield_gc_i(
                        #result_reg,
                        #base_reg,
                        ::core::mem::offset_of!(#struct_path, #member),
                        #tid,
                        stringify!(#member),
                    );
                },
            );
            Some(Binding {
                reg: result_reg,
                kind: BindingKind::Int,
                depends_on_stack: false,
                struct_type: None,
            })
        }
    }

    /// Match `<local ref binding>.<field>` against `array_fields` WITHOUT
    /// emitting anything, so a caller can finish rejecting before it commits
    /// any op to the trace.
    ///
    /// Split from [`Self::emit_array_field_base`] because the element write
    /// has to emit its right-hand side first: Rust evaluates the assigned
    /// value before the assignee place, and a lowering that emitted the base
    /// load first would run the two in the opposite order from the
    /// interpreter.
    fn match_array_field_base(&self, field: &syn::ExprField) -> Option<ArrayFieldBase> {
        let config = self.config?;
        let Expr::Path(path) = &*field.base else {
            return None;
        };
        let ident = path.path.get_ident()?;
        let binding = self.bindings.get(&ident.to_string())?.clone();
        if !matches!(binding.kind, BindingKind::Ref) {
            return None;
        }
        let struct_path = binding.struct_type.as_ref()?.clone();
        let member = field.member.clone();
        let member_name = named_member(&field.member)?;
        let struct_last = struct_path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_default();
        let key = format!("{}::{}", struct_last, member_name);
        let element_type = config.array_fields.get(&key)?.2.clone();
        Some(ArrayFieldBase {
            base_reg: binding.reg,
            struct_path,
            member,
            element_type,
        })
    }

    /// Emit the `getfield_gc_r` that loads the array's base pointer for a
    /// shape [`Self::match_array_field_base`] already accepted.
    ///
    /// Shared by the element read and the element write so both take the
    /// buffer pointer through the SAME interned field descr — a store to the
    /// base (a realloc on growth) then invalidates the load the heapcache
    /// served, instead of leaving a stale buffer pointer live in the trace.
    fn emit_array_field_base(&mut self, shape: &ArrayFieldBase) -> u16 {
        let ArrayFieldBase {
            base_reg,
            struct_path,
            member,
            element_type,
        } = shape;
        let (base_reg, struct_path, member, element_type) = (
            *base_reg,
            struct_path.clone(),
            member.clone(),
            element_type.clone(),
        );
        let tid = struct_type_id_tokens(&struct_path, false);
        let buffer_reg = self.alloc_reg();
        self.emit_op(
            OpMeta::linear(
                OpKind::Vable,
                vec![Register::ref_(base_reg)],
                vec![Register::ref_(buffer_reg)],
            ),
            quote! {
                // The `int_fields` twin of this check lives in
                // `field_scalar_tokens`.  Nothing else ties the declared
                // element type to the field: the concrete rewrite indexes the
                // real pointer while the emitted descr takes its stride and
                // signedness from the declaration, so a declaration that
                // drifted from the struct (`Stack::data => u16` over a
                // `*mut u8`) would compile and read past the buffer.  Naming
                // the pointee rather than the pointer accepts `*const T` as
                // well as `*mut T`; the closure body is type-checked but never
                // evaluated.
                const _: fn(&#struct_path) -> #element_type =
                    |__s| unsafe { *__s.#member };
                __builder.register_struct_layout(
                    ::core::mem::size_of::<#struct_path>(),
                    #tid,
                    false,
                    false,
                    &[(
                        ::core::mem::offset_of!(#struct_path, #member),
                        true,
                        stringify!(#member),
                        ::core::mem::size_of::<usize>(),
                        false,
                    )],
                );
                __builder.getfield_gc_r(
                    #buffer_reg,
                    #base_reg,
                    ::core::mem::offset_of!(#struct_path, #member),
                    #tid,
                    stringify!(#member),
                );
            },
        );
        buffer_reg
    }

    /// Recognizes an array element READ on a local ref binding:
    /// `<binding>.<array_field>[<int expr>]` → `getfield_gc_r` for the buffer
    /// base followed by `getarrayitem_gc_i`.
    ///
    /// This is the non-virtualizable counterpart of the `[..; virt]` element
    /// read: the loaded value is an ordinary trace value, so it does NOT ride
    /// the guard's `vable_array` resume section and the snapshot stays O(1) in
    /// the array's length.
    ///
    /// No `arraylen_gc` is emitted anywhere on this path — the buffer has no
    /// object header to read a length from. A bound belongs on a sibling
    /// `int_fields` length field.
    pub(super) fn lower_ref_binding_array_read(&mut self, expr: &Expr) -> Option<Binding> {
        let Expr::Index(index_expr) = expr else {
            return None;
        };
        let Expr::Field(field) = &*index_expr.expr else {
            return None;
        };
        // Match before emitting anything, so a shape this does not handle
        // leaves the trace untouched.  A place expression evaluates its base
        // before its index, which is the order emitted here.
        let shape = self.match_array_field_base(field)?;
        let element_type = shape.element_type.clone();
        let buffer_reg = self.emit_array_field_base(&shape);
        let index = self.lower_value_expr(&index_expr.index)?;
        if !matches!(index.kind, BindingKind::Int) {
            return None;
        }
        let index_reg = index.reg;
        let result_reg = self.alloc_reg();
        self.emit_op(
            OpMeta::linear(
                OpKind::Vable,
                vec![Register::ref_(buffer_reg), Register::int(index_reg)],
                vec![Register::int(result_reg)],
            ),
            quote! {
                let __descr_idx = __builder.add_raw_int_array_descr_signed(
                    ::core::mem::size_of::<#element_type>(),
                    // descr.py:240-254 get_type_flag reads signedness off the
                    // declared element type.  Hard-coding signed would
                    // sign-extend a `u8` element of 0x80 to -128 where the
                    // concrete Rust read yields 128.  `MIN` resolves through a
                    // type alias, and a non-integer element type fails to
                    // compile here rather than loading as a signed word.
                    (<#element_type>::MIN as i128) < 0,
                );
                __builder.getarrayitem_gc_i(
                    #result_reg as u16,
                    #buffer_reg as u16,
                    #index_reg as u16,
                    __descr_idx,
                );
            },
        );
        Some(Binding {
            reg: result_reg,
            kind: BindingKind::Int,
            depends_on_stack: index.depends_on_stack,
            struct_type: None,
        })
    }

    /// Recognizes an array element WRITE on a local ref binding:
    /// `<binding>.<array_field>[<int expr>] = <int expr>` → `getfield_gc_r`
    /// for the buffer base followed by `setarrayitem_gc_i`.
    ///
    /// Store counterpart of [`Self::lower_ref_binding_array_read`]; both take
    /// the element descr from `add_raw_int_array_descr_signed`, so the store
    /// invalidates the heapcache entry the matching load populated.
    pub(super) fn lower_ref_binding_array_write(&mut self, expr: &Expr) -> Option<()> {
        let Expr::Assign(assign) = expr else {
            return None;
        };
        let Expr::Index(index_expr) = &*assign.left else {
            return None;
        };
        let Expr::Field(field) = &*index_expr.expr else {
            return None;
        };
        // Match before emitting, then emit in Rust's evaluation order for an
        // assignment: the assigned value first, then the assignee place (base,
        // then index).  Emitting the base load first would run the array's
        // side effects ahead of the right-hand side's.
        let shape = self.match_array_field_base(field)?;
        let element_type = shape.element_type.clone();
        let value = self.lower_value_expr(&assign.right)?;
        if !matches!(value.kind, BindingKind::Int) {
            return None;
        }
        let buffer_reg = self.emit_array_field_base(&shape);
        let index = self.lower_value_expr(&index_expr.index)?;
        if !matches!(index.kind, BindingKind::Int) {
            return None;
        }
        let index_reg = index.reg;
        let value_reg = value.reg;
        self.emit_op(
            OpMeta::linear(
                OpKind::Vable,
                vec![
                    Register::ref_(buffer_reg),
                    Register::int(index_reg),
                    Register::int(value_reg),
                ],
                vec![],
            ),
            quote! {
                let __descr_idx = __builder.add_raw_int_array_descr_signed(
                    ::core::mem::size_of::<#element_type>(),
                    // descr.py:240-254 get_type_flag reads signedness off the
                    // declared element type.  Hard-coding signed would
                    // sign-extend a `u8` element of 0x80 to -128 where the
                    // concrete Rust read yields 128.  `MIN` resolves through a
                    // type alias, and a non-integer element type fails to
                    // compile here rather than loading as a signed word.
                    (<#element_type>::MIN as i128) < 0,
                );
                __builder.setarrayitem_gc_i(
                    #buffer_reg as u16,
                    #index_reg as u16,
                    #value_reg as u16,
                    __descr_idx,
                );
            },
        );
        Some(())
    }

    /// Recognizes a pool-array element read through the registered getter call
    /// `<getter>(state.<pool_base_ref>, <int index>)` → `getarrayitem_gc_r` on
    /// the raw-pointer array (`[*mut U; N]` at the declared `items` offset) the
    /// ref-scalar points at — the `pools[selected]` read.  Unlike the
    /// residual-call form (an
    /// opaque CALL_R the optimizer can neither re-produce in the short preamble
    /// nor invalidate), the getarrayitem on the immutable `pools` array
    /// re-derives the element each loop entry from the consistent `selected`
    /// index, so the loaded ref can no longer be carried as an independent
    /// loop-red that diverges from the promoted index.
    ///
    /// Selection is keyed on OPERATION IDENTITY: the call's function path must
    /// match the `getter` registered for this `base` in `pool_arrays`, and arg0
    /// must be `state.<base>`.  An unrelated helper that happens to share the
    /// `(state.<base>, int)` arg shape does NOT match, so it is not miscompiled
    /// into a pool read — it falls through to its own residual body (which is
    /// also the getter's concrete fallback when no `pool_arrays` is configured).
    /// Elements are pointer-width; where they start, and whether a length word
    /// precedes them, come from the declaration (`pool_array_layout_tokens`).
    pub(super) fn lower_pool_array_get_call(&mut self, call: &syn::ExprCall) -> Option<Binding> {
        let config = self.config?;
        if call.args.len() != 2 {
            return None;
        }
        // arg0 must be `state.<base>` where <base> is a declared pool-array base.
        let Expr::Field(base_field) = &call.args[0] else {
            return None;
        };
        if !expr_matches_local_name(&base_field.base, "state") {
            return None;
        }
        let base_name = named_member(&base_field.member)?;
        // Operation identity: the call's function must be the registered getter
        // for this base, not merely any call sharing the `(state.<base>, int)`
        // arg shape.  A function mismatch falls through to the residual CALL_R
        // fallback (the marker function's own body) rather than miscompiling an
        // unrelated helper into a `getarrayitem_gc_r`.
        let func_segments = canonical_expr_segments(&call.func)?;
        let entry = config
            .pool_arrays
            .iter()
            .find(|entry| entry.base == base_name && entry.getter == func_segments)?;
        let element_type = entry.element_type.clone();
        let (base_size, len_offset, layout_witness) = pool_array_layout_tokens(entry);
        // Lower the `state.<base>` ref-scalar (declares its ref identity slot
        // live for resume) and the index, then read the pointer element.
        let base = self.lower_state_field_read(&call.args[0])?;
        if !matches!(base.kind, BindingKind::Ref) {
            return None;
        }
        let base_reg = base.reg;
        let index = self.lower_value_expr(&call.args[1])?;
        if !matches!(index.kind, BindingKind::Int) {
            return None;
        }
        let index_reg = index.reg;
        let result_reg = self.alloc_reg();
        self.emit_op(
            OpMeta::linear(
                OpKind::Vable,
                vec![Register::ref_(base_reg), Register::int(index_reg)],
                vec![Register::ref_(result_reg)],
            ),
            quote! {
                #layout_witness
                let __descr_idx = __builder.add_ptr_array_descr(#base_size, #len_offset);
                __builder.getarrayitem_gc_r(
                    #result_reg as u16,
                    #base_reg as u16,
                    #index_reg as u16,
                    __descr_idx,
                );
            },
        );
        Some(Binding {
            reg: result_reg,
            kind: BindingKind::Ref,
            depends_on_stack: base.depends_on_stack || index.depends_on_stack,
            struct_type: element_type,
        })
    }

    /// Recognizes a field WRITE through a `ref(T)` state scalar:
    /// `state.<ref_scalar>.<member> = <int expr>` → `setfield_gc_i`.  The
    /// store shares the same `(struct_type_id, field)` interned `Field` descr
    /// as [`Self::lower_state_ref_field_getfield`], so the heapcache
    /// invalidates the cached getfield on every write — the in-trace
    /// counterpart of an RPython inlined `self.field = ...` store that keeps a
    /// length getfield from freezing to a loop-invariant constant.
    pub(super) fn lower_state_ref_field_setfield(&mut self, expr: &Expr) -> Option<()> {
        let config = self.config?;
        let Expr::Assign(assign) = expr else {
            return None;
        };
        let Expr::Field(field) = &*assign.left else {
            return None;
        };
        let Expr::Field(base_field) = &*field.base else {
            return None;
        };
        if !expr_matches_local_name(&base_field.base, "state") {
            return None;
        }
        let base_name = named_member(&base_field.member)?;
        let (_, struct_path) = config.state_ref_scalars.get(&base_name).cloned()?;
        let member = field.member.clone();
        let member_name = named_member(&field.member)?;
        // Check if this field is declared ref-kind in `ref_fields`.
        let struct_last = struct_path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_default();
        let ref_field_key = format!("{}::{}", struct_last, member_name);
        let (__fsize, __fsigned, __fcheck) =
            field_scalar_tokens(config, &ref_field_key, &struct_path, &member);
        let is_ref_field = config.ref_fields.contains_key(&ref_field_key);
        // Raw (headerless) ref-scalar pointee → `is_gc_managed = false`, the
        // same id the matching getfield uses so this setfield invalidates it.
        let tid = struct_type_id_tokens(&struct_path, false);
        let base = self.lower_state_field_read(&field.base)?;
        if !matches!(base.kind, BindingKind::Ref) {
            return None;
        }
        let base_reg = base.reg;
        let rhs = self.lower_value_expr(&assign.right)?;
        if is_ref_field {
            // Ref-kind field → setfield_gc_r.
            if !matches!(rhs.kind, BindingKind::Ref) {
                return None;
            }
            let src = rhs.reg;
            self.emit_op(
                OpMeta::linear(
                    OpKind::SetfieldGc,
                    vec![Register::ref_(base_reg), Register::ref_(src)],
                    vec![],
                ),
                quote! {
                    #__fcheck
                    __builder.register_struct_layout(
                        ::core::mem::size_of::<#struct_path>(),
                        #tid,
                        false,
                        false,
                        &[(
                            ::core::mem::offset_of!(#struct_path, #member),
                            true,
                            stringify!(#member),
                            #__fsize,
                            #__fsigned,
                        )],
                    );
                    __builder.setfield_gc_r(
                        #base_reg,
                        #src,
                        ::core::mem::offset_of!(#struct_path, #member),
                        #tid,
                        stringify!(#member),
                    );
                },
            );
        } else {
            // Int-kind field (default) → setfield_gc_i.
            if !matches!(rhs.kind, BindingKind::Int) {
                return None;
            }
            let src = rhs.reg;
            self.emit_op(
                OpMeta::linear(
                    OpKind::SetfieldGc,
                    vec![Register::ref_(base_reg), Register::int(src)],
                    vec![],
                ),
                quote! {
                    #__fcheck
                    __builder.register_struct_layout(
                        ::core::mem::size_of::<#struct_path>(),
                        #tid,
                        false,
                        false,
                        &[(
                            ::core::mem::offset_of!(#struct_path, #member),
                            false,
                            stringify!(#member),
                            #__fsize,
                            #__fsigned,
                        )],
                    );
                    __builder.setfield_gc_i(
                        #base_reg,
                        #src,
                        ::core::mem::offset_of!(#struct_path, #member),
                        #tid,
                        stringify!(#member),
                    );
                },
            );
        }
        Some(())
    }

    /// Field WRITE on an arbitrary local ref binding with known struct type:
    /// `<ident>.<field> = <expr>` where `<ident>` was bound as
    /// `BindingKind::Ref` with a non-None `struct_type`.
    pub(super) fn lower_ref_binding_setfield(&mut self, expr: &Expr) -> Option<()> {
        let config = self.config?;
        let Expr::Assign(assign) = expr else {
            return None;
        };
        let Expr::Field(field) = &*assign.left else {
            return None;
        };
        // Base must be a simple ident (a local variable), not `state.x.y`.
        let Expr::Path(path) = &*field.base else {
            return None;
        };
        let ident = path.path.get_ident()?;
        let binding = self.bindings.get(&ident.to_string())?.clone();
        if !matches!(binding.kind, BindingKind::Ref) {
            return None;
        }
        let struct_path = binding.struct_type.as_ref()?.clone();
        let member = field.member.clone();
        let member_name = named_member(&field.member)?;
        let struct_last = struct_path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_default();
        let ref_field_key = format!("{}::{}", struct_last, member_name);
        let (__fsize, __fsigned, __fcheck) =
            field_scalar_tokens(config, &ref_field_key, &struct_path, &member);
        let is_ref_field = config.ref_fields.contains_key(&ref_field_key);
        let tid = struct_type_id_tokens(&struct_path, false);
        let base_reg = binding.reg;
        let rhs = self.lower_value_expr(&assign.right)?;
        if is_ref_field {
            if !matches!(rhs.kind, BindingKind::Ref) {
                return None;
            }
            let src = rhs.reg;
            self.emit_op(
                OpMeta::linear(
                    OpKind::SetfieldGc,
                    vec![Register::ref_(base_reg), Register::ref_(src)],
                    vec![],
                ),
                quote! {
                    #__fcheck
                    __builder.register_struct_layout(
                        ::core::mem::size_of::<#struct_path>(),
                        #tid,
                        false,
                        false,
                        &[(
                            ::core::mem::offset_of!(#struct_path, #member),
                            true,
                            stringify!(#member),
                            #__fsize,
                            #__fsigned,
                        )],
                    );
                    __builder.setfield_gc_r(
                        #base_reg,
                        #src,
                        ::core::mem::offset_of!(#struct_path, #member),
                        #tid,
                        stringify!(#member),
                    );
                },
            );
        } else {
            if !matches!(rhs.kind, BindingKind::Int) {
                return None;
            }
            let src = rhs.reg;
            self.emit_op(
                OpMeta::linear(
                    OpKind::SetfieldGc,
                    vec![Register::ref_(base_reg), Register::int(src)],
                    vec![],
                ),
                quote! {
                    #__fcheck
                    __builder.register_struct_layout(
                        ::core::mem::size_of::<#struct_path>(),
                        #tid,
                        false,
                        false,
                        &[(
                            ::core::mem::offset_of!(#struct_path, #member),
                            false,
                            stringify!(#member),
                            #__fsize,
                            #__fsigned,
                        )],
                    );
                    __builder.setfield_gc_i(
                        #base_reg,
                        #src,
                        ::core::mem::offset_of!(#struct_path, #member),
                        #tid,
                        stringify!(#member),
                    );
                },
            );
        }
        Some(())
    }

    /// Recognizes `state.array[index]` for array state fields.
    /// Virtualizable arrays bail to the standard vable path; flattened arrays
    /// emit `load_state_array`.
    pub(super) fn lower_state_array_read(&mut self, expr: &Expr) -> Option<Binding> {
        let config = self.config?;
        let index_expr = match expr {
            Expr::Index(idx) => idx,
            _ => return None,
        };
        let field = match &*index_expr.expr {
            Expr::Field(f) => f,
            _ => return None,
        };
        let base = &field.base;
        if !expr_matches_local_name(base, "state") {
            return None;
        }
        let member_name = named_member(&field.member)?;
        // `[int; virt]` arrays are NOT handled here — they lower through the
        // standard virtualizable path (`lower_vable_array_read`). Bail before
        // lowering the index expression so it is not emitted twice when the
        // dispatch falls through to the vable read.
        if config.state_virt_arrays.contains_key(&member_name) {
            return None;
        }
        let idx_binding = self.lower_value_expr(&index_expr.index)?;
        let idx_reg = idx_binding.reg;
        let reg = self.alloc_reg();

        let &array_index = config.state_arrays.get(&member_name)?;
        let ai = array_index as u16;
        self.emit_op(
            OpMeta::linear(
                OpKind::StateField,
                vec![Register::int(idx_reg)],
                vec![Register::int(reg)],
            ),
            quote! { __builder.load_state_array(#ai, #idx_reg, #reg); },
        );
        Some(Binding {
            reg,
            kind: BindingKind::Int,
            depends_on_stack: false,
            struct_type: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_fields_map(
        entries: &[(&str, &str, &str, &str)],
    ) -> HashMap<String, (syn::Path, Ident, syn::Path)> {
        entries
            .iter()
            .map(|(key, struct_path, field, pointee)| {
                (
                    (*key).to_string(),
                    (
                        syn::parse_str::<syn::Path>(struct_path).expect("struct path"),
                        syn::parse_str::<Ident>(field).expect("field ident"),
                        syn::parse_str::<syn::Path>(pointee).expect("pointee path"),
                    ),
                )
            })
            .collect()
    }

    fn witness_for(key: &str) -> String {
        let map = ref_fields_map(&[("Stack::head", "Stack", "head", "Node")]);
        let struct_path: syn::Path = syn::parse_str("Stack").expect("struct path");
        let member: syn::Member = syn::parse_str("head").expect("member");
        ref_field_witness_tokens(&map, key, &struct_path, &member).to_string()
    }

    /// A declared ref field admits the three spellings the lowering can
    /// actually receive, and no others. The pointee is named once per pointer
    /// spelling, which is what turns a drifted declaration into a type error.
    #[test]
    fn a_declared_ref_field_witnesses_its_pointee() {
        let tokens = witness_for("Stack::head");
        for expected in [
            "impl __MajitRefField for * mut Node",
            "impl __MajitRefField for * const Node",
            "impl __MajitRefField for usize",
        ] {
            assert!(
                tokens.contains(expected),
                "the witness must admit `{expected}`; tokens={tokens}"
            );
        }
        assert!(
            tokens.contains("__accept (__s . head)"),
            "the witness must touch the field it names, or it witnesses \
             nothing about the struct; tokens={tokens}"
        );
    }

    /// The other half of the same predicate, stated so the gap is a decision
    /// rather than an oversight: a field named in no map reads into the Int
    /// bank, and stable Rust cannot assert that a type is *not* a pointer, so
    /// no witness is emitted for it.
    #[test]
    fn an_undeclared_field_gets_no_ref_witness() {
        assert_eq!(
            witness_for("Stack::size"),
            "",
            "a key `ref_fields` does not declare must emit nothing; emitting a \
             witness there would reject every legitimate integer field"
        );
    }
}
