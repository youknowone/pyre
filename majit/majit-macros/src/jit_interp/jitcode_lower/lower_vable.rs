use super::lower_value::struct_type_id;
use super::*;

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
        let &field_index = config.state_scalars.get(&member_name)?;
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
    /// Only int fields are lowered here (the aheui length read); a ref field
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
        let ref_field_entry = config.ref_fields.get(&ref_field_key);
        let is_ref_field = ref_field_entry.is_some();
        // Raw (headerless) ref-scalar pointee → `is_gc_managed = false`, a
        // distinct descriptor id from any GC `new_struct` of the same type.
        let tid = struct_type_id(&struct_path, false);
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
                    __builder.register_struct_layout(
                        ::core::mem::size_of::<#struct_path>(),
                        #tid,
                        false,
                        &[(
                            ::core::mem::offset_of!(#struct_path, #member),
                            true,
                            stringify!(#member),
                        )],
                    );
                    __builder.getfield_gc_r(
                        #result_reg,
                        #base_reg,
                        ::core::mem::offset_of!(#struct_path, #member),
                        #tid,
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
                    __builder.register_struct_layout(
                        ::core::mem::size_of::<#struct_path>(),
                        #tid,
                        false,
                        &[(
                            ::core::mem::offset_of!(#struct_path, #member),
                            false,
                            stringify!(#member),
                        )],
                    );
                    __builder.getfield_gc_i(
                        #result_reg,
                        #base_reg,
                        ::core::mem::offset_of!(#struct_path, #member),
                        #tid,
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
        let ref_field_entry = config.ref_fields.get(&ref_field_key);
        let is_ref_field = ref_field_entry.is_some();
        let tid = struct_type_id(&struct_path, false);
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
                    __builder.register_struct_layout(
                        ::core::mem::size_of::<#struct_path>(),
                        #tid,
                        false,
                        &[(
                            ::core::mem::offset_of!(#struct_path, #member),
                            true,
                            stringify!(#member),
                        )],
                    );
                    __builder.getfield_gc_r(
                        #result_reg,
                        #base_reg,
                        ::core::mem::offset_of!(#struct_path, #member),
                        #tid,
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
                    __builder.register_struct_layout(
                        ::core::mem::size_of::<#struct_path>(),
                        #tid,
                        false,
                        &[(
                            ::core::mem::offset_of!(#struct_path, #member),
                            false,
                            stringify!(#member),
                        )],
                    );
                    __builder.getfield_gc_i(
                        #result_reg,
                        #base_reg,
                        ::core::mem::offset_of!(#struct_path, #member),
                        #tid,
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

    /// Recognizes a pool-array element read through the registered getter call
    /// `<getter>(state.<pool_base_ref>, <int index>)` → `getarrayitem_gc_r` on
    /// the raw-pointer array (`[*mut U; N]` at offset 0) the ref-scalar points
    /// at — the aheui `pools[selected]` read.  Unlike the residual-call form (an
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
    /// Pointer elements are 8 bytes at array offset 0 (`add_ptr_array_descr`).
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
        if !config
            .pool_arrays
            .iter()
            .any(|(base, getter)| base == &base_name && getter == &func_segments)
        {
            return None;
        }
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
                let __descr_idx = __builder.add_ptr_array_descr();
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
            struct_type: None,
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
        let is_ref_field = config.ref_fields.contains_key(&ref_field_key);
        // Raw (headerless) ref-scalar pointee → `is_gc_managed = false`, the
        // same id the matching getfield uses so this setfield invalidates it.
        let tid = struct_type_id(&struct_path, false);
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
                    __builder.register_struct_layout(
                        ::core::mem::size_of::<#struct_path>(),
                        #tid,
                        false,
                        &[(
                            ::core::mem::offset_of!(#struct_path, #member),
                            true,
                            stringify!(#member),
                        )],
                    );
                    __builder.setfield_gc_r(
                        #base_reg,
                        #src,
                        ::core::mem::offset_of!(#struct_path, #member),
                        #tid,
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
                    __builder.register_struct_layout(
                        ::core::mem::size_of::<#struct_path>(),
                        #tid,
                        false,
                        &[(
                            ::core::mem::offset_of!(#struct_path, #member),
                            false,
                            stringify!(#member),
                        )],
                    );
                    __builder.setfield_gc_i(
                        #base_reg,
                        #src,
                        ::core::mem::offset_of!(#struct_path, #member),
                        #tid,
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
        let is_ref_field = config.ref_fields.contains_key(&ref_field_key);
        let tid = struct_type_id(&struct_path, false);
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
                    __builder.register_struct_layout(
                        ::core::mem::size_of::<#struct_path>(),
                        #tid,
                        false,
                        &[(
                            ::core::mem::offset_of!(#struct_path, #member),
                            true,
                            stringify!(#member),
                        )],
                    );
                    __builder.setfield_gc_r(
                        #base_reg,
                        #src,
                        ::core::mem::offset_of!(#struct_path, #member),
                        #tid,
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
                    __builder.register_struct_layout(
                        ::core::mem::size_of::<#struct_path>(),
                        #tid,
                        false,
                        &[(
                            ::core::mem::offset_of!(#struct_path, #member),
                            false,
                            stringify!(#member),
                        )],
                    );
                    __builder.setfield_gc_i(
                        #base_reg,
                        #src,
                        ::core::mem::offset_of!(#struct_path, #member),
                        #tid,
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
