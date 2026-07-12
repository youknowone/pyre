use super::*;

/// Deterministic per-struct type id for a size descr.  Hashes the struct path
/// tokens; the same id keys the builder's `struct_size_specs` cache so each
/// `setfield_gc_*` resolves its field's parent SizeDescr + `index_in_parent`
/// (`descr.py:238`).  Distinct struct paths collide only at `DefaultHasher`'s
/// 64-bit range, matching the runtime `LLType::Struct(type_id)` cache-key
/// identity.
///
/// `descr.py:105 get_size_descr` keys the cache by the actual lltype STRUCT, so
/// a `GcStruct` and a raw `Struct` with identical fields are DISTINCT lltypes
/// with distinct descrs.  A GC `new_struct(T)` and a raw `ref(T)` state scalar
/// share a path and would otherwise share an id; the first-registered
/// `is_gc_managed` flag would then pin both, so a raw getfield could inherit a
/// GC descr's `GUARD_GC_TYPE` and read a non-existent type-id word at `ptr - 8`
/// (or a GC alloc could lose its type guard).  Fold `is_gc_managed` into the id
/// so the two layouts never alias: GC keeps the bare hash (existing
/// `new_struct` ids unchanged), raw appends a discriminator.
pub(super) fn struct_type_id(path: &syn::Path, is_gc_managed: bool) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    quote!(#path).to_string().hash(&mut hasher);
    if !is_gc_managed {
        "raw".hash(&mut hasher);
    }
    hasher.finish()
}

impl<'c> Lowerer<'c> {
    pub(super) fn lower_value_expr(&mut self, expr: &Expr) -> Option<Binding> {
        // State field read (register/tape machines).
        if let Some(binding) = self.lower_state_field_read(expr) {
            return Some(binding);
        }
        // Field read through a `ref(T)` state scalar: `state.<ref>.<member>`
        // → getfield_gc_i. Placed after lower_state_field_read (which only
        // matches `state.<scalar>` whose base is `state`) so the plain scalar
        // read wins and this handles the `Field`-of-`Field` shape.
        if let Some(binding) = self.lower_state_ref_field_getfield(expr) {
            return Some(binding);
        }
        // Field read on a local ref binding with known struct type:
        // `<binding>.<field>` → getfield_gc_i/getfield_gc_r.
        if let Some(binding) = self.lower_ref_binding_getfield(expr) {
            return Some(binding);
        }
        if let Some(binding) = self.lower_state_array_read(expr) {
            return Some(binding);
        }
        // RPython pyopcode.py:171 `ord(co_code[next_instr])` — an indexed
        // read of the green bytecode array fetches an immediate operand.
        // The dispatch-top opcode fetch is lowered by the stmt-level
        // `try_lower_opcode_fetch_stmt`, which arm-body lowering does not
        // reach; this handler lowers the identical `program[<idx>]` shape
        // in value position so operand reads inside opcode arm bodies emit
        // the same `getarrayitem_gc_i` on the green array (folded by the
        // optimizer since array + index are green).
        if let Some(binding) = self.lower_env_array_read(expr) {
            return Some(binding);
        }
        // RPython jtransform.py:832 — virtualizable field read rewrite.
        if let Some(binding) = self.lower_vable_field_read(expr) {
            return Some(binding);
        }
        // RPython jtransform.py:760 — virtualizable array read rewrite.
        if let Some(binding) = self.lower_vable_array_read(expr) {
            return Some(binding);
        }
        if let Some(binding) = self.lower_vable_array_len(expr) {
            return Some(binding);
        }
        // RPython jtransform.py:655 — suppress hint_access_directly(frame) /
        // hint_fresh_virtualizable(frame) function calls as identity.
        // These return the frame unchanged, so lower the argument instead.
        if let Some(binding) = self.lower_vable_hint_identity_call(expr) {
            return Some(binding);
        }
        // RPython jtransform.py:1687 — conditional_call_elidable!(value, func, args...)
        if let Some(binding) = self.lower_conditional_call_elidable(expr) {
            return Some(binding);
        }
        // RPython jtransform.py:522 — recursive_portal_call!(driver, greens...)
        if let Some(binding) = self.lower_recursive_portal_call(expr) {
            return Some(binding);
        }

        match expr {
            Expr::Lit(ExprLit {
                lit: Lit::Int(int_lit),
                ..
            }) => {
                let value = int_lit.base10_parse::<i64>().ok()?;
                let reg = self.alloc_reg();
                self.emit_op(
                    OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(reg)]),
                    quote! { __builder.load_const_i_value(#reg, #value); },
                );
                Some(Binding {
                    reg,
                    kind: BindingKind::Int,
                    depends_on_stack: false,
                    struct_type: None,
                })
            }
            // Bool comparisons (`stackok == false`) ride the int channel:
            // bool bindings are IntLt/IntLe/… results carrying 0/1.
            Expr::Lit(ExprLit {
                lit: Lit::Bool(bool_lit),
                ..
            }) => {
                let value = i64::from(bool_lit.value);
                let reg = self.alloc_reg();
                self.emit_op(
                    OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(reg)]),
                    quote! { __builder.load_const_i_value(#reg, #value); },
                );
                Some(Binding {
                    reg,
                    kind: BindingKind::Int,
                    depends_on_stack: false,
                    struct_type: None,
                })
            }
            Expr::Path(ExprPath { path, .. }) => {
                let ident = path.get_ident()?;
                self.bindings.get(&ident.to_string()).cloned()
            }
            Expr::Cast(ExprCast { expr, ty, .. }) if is_supported_int_cast(ty) => {
                let binding = self.lower_value_expr(expr)?;
                if !matches!(binding.kind, BindingKind::Int) {
                    return None;
                }
                self.lower_int_cast(binding, ty)
            }
            Expr::Paren(ExprParen { expr, .. }) => self.lower_value_expr(expr),
            Expr::If(expr_if) => self.lower_if_value(expr_if),
            Expr::Match(expr_match) => self.lower_match_value(expr_match),
            Expr::Unary(ExprUnary { op, expr, .. }) => self.lower_unary(op, expr),
            Expr::Binary(binary) => self.lower_binary(binary),
            Expr::Call(call) => {
                // jtransform.py:596 rewrite_op_hint: promote → int_guard_value
                if let Some(binding) = self.lower_promote_call(call) {
                    return Some(binding);
                }
                // pyjitpl.py:385-391 opimpl_assert_not_none — emit
                // BC_ASSERT_NOT_NONE and return the unwrapped binding.
                if let Some(binding) = self.lower_assert_not_none_call(call) {
                    return Some(binding);
                }
                // A `<fn>(state.<pool_base_ref>, <idx>)` marker call reading a
                // raw-pointer pool array lowers to getarrayitem_gc_r instead of
                // a residual CALL_R (re-producible heap read, no diverging red).
                if let Some(binding) = self.lower_pool_array_get_call(call) {
                    return Some(binding);
                }
                // jtransform.py:2030 _handle_int_special — pure int binop
                // aliases bypass the call-policy machinery and emit a native
                // IR op (IntAdd, IntSub, etc.) directly.
                if let Some(binding) = self.lower_native_int_binop_call(call) {
                    return Some(binding);
                }
                // Raw native-memory load intrinsic `majit_raw_load_i64(base, ea)`
                // → raw_load_i (jtransform.py:1165-1171 rewrite_op_raw_load).
                if let Some(binding) = self.lower_raw_load_call(call) {
                    return Some(binding);
                }
                self.lower_call_value(call)
            }
            Expr::MethodCall(call) => self.lower_method_call_value(call),
            Expr::Struct(s) => self.lower_struct_value(s),
            _ => None,
        }
    }

    /// Lower a raw native-memory load intrinsic
    /// `majit_raw_load_i64(base, ea)` to a `raw_load_i` op (the read-side
    /// analogue of `lower_raw_store_stmt`).  RPython parity:
    /// `jtransform.py:1165-1171 rewrite_op_raw_load` lowers a
    /// `rffi.raw_storage_getitem` to `raw_load_i(base, offset,
    /// arraydescrof(CArray(T)))`.  Both operands are int-kind (raw address,
    /// byte offset); the op writes an int result.
    fn lower_raw_load_call(&mut self, call: &syn::ExprCall) -> Option<Binding> {
        let segments = canonical_expr_segments(&call.func)?;
        if segments.last().map(String::as_str) != Some("majit_raw_load_i64") {
            return None;
        }
        if call.args.len() != 2 {
            return None;
        }
        let base = self.lower_value_expr(&call.args[0])?;
        let ea = self.lower_value_expr(&call.args[1])?;
        let (base_reg, ea_reg) = (base.reg, ea.reg);
        let dst = self.alloc_reg();
        self.emit_op(
            OpMeta::linear(
                OpKind::RawLoad,
                vec![Register::int(base_reg), Register::int(ea_reg)],
                vec![Register::int(dst)],
            ),
            quote! {
                let __raw_descr = __builder.add_raw_int_array_descr(8);
                __builder.raw_load_i(
                    #dst as u16,
                    #base_reg as u16,
                    #ea_reg as u16,
                    __raw_descr,
                );
            },
        );
        Some(Binding {
            reg: dst,
            kind: BindingKind::Int,
            depends_on_stack: base.depends_on_stack || ea.depends_on_stack,
            struct_type: None,
        })
    }

    /// Lower a struct literal `Path { f0: v0, f1: v1, .. }` to a JIT
    /// allocation plus per-field stores: `new` (size from `size_of`) then
    /// `setfield_gc_<kind>` at each field's `offset_of`.  Mirrors
    /// `jtransform.py` malloc + setfield rewrite; the optimizer's
    /// virtualize pass folds the New away when the struct does not escape.
    /// Returns the result ref binding, or `None` (helper falls back to a
    /// residual call) for struct-update base syntax or float fields, which
    /// the bytecode field-op path does not express yet.
    fn lower_struct_value(&mut self, s: &syn::ExprStruct) -> Option<Binding> {
        if s.rest.is_some() {
            return None;
        }
        let struct_path = &s.path;
        // Lower every field value (rejecting unsupported kinds) before
        // emitting the allocation, so a failed field never leaves a
        // dangling New in the op stream.
        let mut fields = Vec::new();
        let mut depends_on_stack = false;
        for field in &s.fields {
            let value = self.lower_value_expr(&field.expr)?;
            if matches!(value.kind, BindingKind::Float) {
                return None;
            }
            depends_on_stack |= value.depends_on_stack;
            fields.push((field.member.clone(), value));
        }
        // GC-allocated struct (`new_struct`): keep the bare-hash id.
        let type_id = struct_type_id(struct_path, true);
        let result_reg = self.alloc_reg();
        // descr.py:122-126 init_size_descr: the SizeDescr carries the
        // struct's full `(offset, is_ref, name)` layout so the optimizer can
        // virtualize the New and resolve each field's parent SizeDescr +
        // index. A Rust struct literal may list fields in any order, so the
        // builder canonicalizes `index_in_parent` by byte offset; we only
        // need to hand it each field's `(offset, is_ref, name)`.
        let field_layout: Vec<TokenStream> = fields
            .iter()
            .map(|(member, value)| {
                let is_ref = matches!(value.kind, BindingKind::Ref);
                quote! {
                    (
                        ::core::mem::offset_of!(#struct_path, #member),
                        #is_ref,
                        stringify!(#member),
                    )
                }
            })
            .collect();
        self.emit_op(
            OpMeta::linear(OpKind::New, vec![], vec![Register::ref_(result_reg)]),
            quote! {
                __builder.new_struct(
                    #result_reg,
                    ::core::mem::size_of::<#struct_path>(),
                    #type_id,
                    &[ #(#field_layout),* ],
                );
            },
        );
        for (member, value) in fields.iter() {
            let value_reg = value.reg;
            let (reads, tokens) = match value.kind {
                BindingKind::Int => (
                    vec![Register::ref_(result_reg), Register::int(value_reg)],
                    quote! {
                        __builder.setfield_gc_i(
                            #result_reg,
                            #value_reg,
                            ::core::mem::offset_of!(#struct_path, #member),
                            #type_id,
                        );
                    },
                ),
                BindingKind::Ref => (
                    vec![Register::ref_(result_reg), Register::ref_(value_reg)],
                    quote! {
                        __builder.setfield_gc_r(
                            #result_reg,
                            #value_reg,
                            ::core::mem::offset_of!(#struct_path, #member),
                            #type_id,
                        );
                    },
                ),
                BindingKind::Float => unreachable!("float fields rejected above"),
            };
            self.emit_op(OpMeta::linear(OpKind::SetfieldGc, reads, vec![]), tokens);
        }
        Some(Binding {
            reg: result_reg,
            kind: BindingKind::Ref,
            depends_on_stack,
            struct_type: Some(struct_path.clone()),
        })
    }

    /// Statement-context lowering for `hint(x, promote=True)`:
    ///
    /// - `x = promote(arg);` (plain local re-assignment — `lower_state_
    ///   field_write` falls through because LHS isn't a `state.foo` field;
    ///   without this site `stmt_modifies_jit_state` returns false and
    ///   `lower_stmt`'s config-aware fallback silently consumes the stmt
    ///   without emitting the guard).
    /// - `promote(x);` (bare statement-form — same fall-through path as
    ///   plain assign).
    ///
    /// In both forms the actual guard emission is delegated to
    /// `lower_promote_call` via `lower_value_expr` so the resulting op
    /// shape (`-live-` + `<kind>_guard_value`) is identical to the
    /// value-context lowering.  For assignment form, mirror
    /// jtransform.py:613-615: the `None` sentinel means the result is
    /// considered equal to arg0, so the LHS aliases the RHS binding
    /// (`x = promote(y)` makes `x` read from y's register).
    /// Reassign a local variable that already has a binding: `pc = expr`.
    /// Lowers the RHS via `lower_value_expr` and rebinds the LHS name to
    /// the new register. RPython parity: the flat dispatch's
    /// `pc = target; continue` updates the JitCode's pc register in-place
    /// (`flatten.py` emits a goto to the bytecode label, not an SSA rename,
    /// but majit's binding model is SSA-like — rebinding the name achieves
    /// the same effect because subsequent reads of `pc` pick up the new
    /// register).
    pub(super) fn lower_local_reassign(&mut self, expr: &Expr) -> Option<()> {
        let Expr::Assign(assign) = expr else {
            return None;
        };
        let Expr::Path(lhs_path) = &*assign.left else {
            return None;
        };
        let lhs_ident = lhs_path.path.get_ident()?.to_string();
        if !self.bindings.contains_key(&lhs_ident) {
            return None;
        }
        let binding = self.lower_value_expr(&assign.right)?;
        self.bindings.insert(lhs_ident, binding);
        Some(())
    }

    pub(super) fn lower_promote_stmt(&mut self, expr: &Expr) -> Option<()> {
        match expr {
            Expr::Assign(assign) => {
                let Expr::Path(lhs_path) = &*assign.left else {
                    return None;
                };
                let lhs_ident = lhs_path.path.get_ident()?.to_string();
                let Expr::Call(call) = &*assign.right else {
                    return None;
                };
                if !is_promote_call_path(&call.func) {
                    return None;
                }
                let binding = self.lower_value_expr(&assign.right)?;
                self.bindings.insert(lhs_ident, binding);
                Some(())
            }
            Expr::Call(call) => {
                if !is_promote_call_path(&call.func) {
                    return None;
                }
                self.lower_value_expr(expr)?;
                Some(())
            }
            _ => None,
        }
    }

    /// Statement-context lowering for `jit::assert_not_none(x);`.
    ///
    /// pyjitpl.py:385-391 `opimpl_assert_not_none`. Bare statement form
    /// discards the unwrapped value but still emits the
    /// `BC_ASSERT_NOT_NONE` op so the trace records the nullity hint.
    /// The `let x = jit::assert_not_none(y);` form goes through the
    /// value-context lowerer instead (lower_value_expr →
    /// lower_assert_not_none_call); this stmt site catches only the
    /// bare-call shape.
    pub(super) fn lower_assert_not_none_stmt(&mut self, expr: &Expr) -> Option<()> {
        let Expr::Call(call) = expr else {
            return None;
        };
        if !is_assert_not_none_call_path(&call.func) {
            return None;
        }
        self.lower_value_expr(expr)?;
        Some(())
    }

    /// Statement-context lowering for `jit::record_exact_class(value, cls);`.
    ///
    /// pyjitpl.py:393-410 `opimpl_record_exact_class`. RPython
    /// `rlib/jit.py:1181 record_exact_class` is a void hint, so this is
    /// the only emission shape — there is no value-form (the runtime
    /// stub at `jit.rs:735` returns `()`).
    ///
    /// `value` must be Ref-kind and `cls` must be Int-kind, matching
    /// `blackhole.py:616 @arguments("r", "i")`.  Non-matching kinds
    /// silently skip per `pyjitpl.py:399 if isinstance(clsbox, Const):` —
    /// dispatch-time `trace_record_exact_class` also gates on
    /// `cls_const.is_constant()`.
    pub(super) fn lower_record_exact_class_stmt(&mut self, expr: &Expr) -> Option<()> {
        let Expr::Call(call) = expr else {
            return None;
        };
        if !is_record_exact_class_call_path(&call.func) {
            return None;
        }
        if call.args.len() != 2 {
            return None;
        }
        let value_binding = self.lower_value_expr(&call.args[0])?;
        let cls_binding = self.lower_value_expr(&call.args[1])?;
        if !matches!(value_binding.kind, BindingKind::Ref) {
            return None;
        }
        if !matches!(cls_binding.kind, BindingKind::Int) {
            return None;
        }
        let value_reg = value_binding.reg;
        let cls_reg = cls_binding.reg;
        // jtransform.py:289 emits a single SpaceOperation with no
        // preceding `-live-` annotation (-live- is only attached to
        // promote, jtransform.py:611).
        self.emit_op(
            OpMeta::linear(
                OpKind::RecordExactClass,
                vec![Register::ref_(value_reg), Register::int(cls_reg)],
                vec![],
            ),
            quote! { __builder.record_exact_class(#value_reg, #cls_reg); },
        );
        Some(())
    }

    /// Lower `promote(x)` → emit `-live-` + `<kind>_guard_value(x_reg)`,
    /// return x binding.
    ///
    /// RPython: `hint(x, promote=True)` rewrites to a `-live-` marker
    /// (jtransform.py:611) immediately followed by `int_guard_value(x)`
    /// (jtransform.py:612).  The leading `-live-` pins the per-marker
    /// liveness triple at the source position so the resume protocol can
    /// rebuild the live frame state if the guard fails.  Without this
    /// pair, the snapshot path falls back to the canonical "everything-
    /// alive" entry — correct for blackhole resume but a per-marker
    /// liveness parity loss vs RPython.
    ///
    /// Blackhole: no-op (the live marker is metadata, the guard a no-op
    /// at non-trace time).  Tracing: emits GUARD_VALUE to specialize on
    /// current value with the per-pc live set saved into all_liveness.
    ///
    /// Recognizes: `assert_not_none(x)`, `jit::assert_not_none(x)`.
    ///
    /// pyjitpl.py:385-391 `opimpl_assert_not_none(box)`. Emits
    /// `BC_ASSERT_NOT_NONE`; trace-time dispatcher routes through
    /// `TraceCtx::trace_assert_not_none` which gates on
    /// `heap_cache.is_nullity_known` + bumps `HEAPCACHED_OPS` on cache
    /// hit. Only fires for ref-typed bindings — `jit::assert_not_none<T>`
    /// is documented as the `Option<T>::expect` analog at the runtime
    /// stub (`jit.rs:756`), so non-ref bindings (int/float) cannot
    /// reach this site through `Option<T>` unwrap.
    fn lower_assert_not_none_call(&mut self, call: &ExprCall) -> Option<Binding> {
        if !is_assert_not_none_call_path(&call.func) {
            return None;
        }
        if call.args.len() != 1 {
            return None;
        }
        let binding = self.lower_value_expr(&call.args[0])?;
        if !matches!(binding.kind, BindingKind::Ref) {
            return None;
        }
        let reg = binding.reg;
        // jtransform.py:324-328 emits a single SpaceOperation with no
        // preceding `-live-` annotation (-live- is only attached to
        // promote, jtransform.py:611).
        self.emit_op(
            OpMeta::linear(OpKind::AssertNotNone, vec![Register::ref_(reg)], vec![]),
            quote! { __builder.assert_not_none(#reg); },
        );
        Some(binding)
    }

    /// Recognizes: `promote(x)`, `hint_promote(x)`, `jit::promote(x)`.
    fn lower_promote_call(&mut self, call: &ExprCall) -> Option<Binding> {
        if !is_promote_call_path(&call.func) {
            return None;
        }
        if call.args.len() != 1 {
            return None;
        }
        let binding = self.lower_value_expr(&call.args[0])?;
        let reg = binding.reg;
        // jtransform.py:611 — emit `-live-` before the guard op so the
        // codewriter's per-marker liveness analysis records the alive
        // set at this CFG position.  `live_placeholder_with_triple` will
        // patch the BC_LIVE 2-byte slot to the dedup'd offset at
        // `finalize_liveness` time.
        self.emit_op(
            OpMeta::live_marker(),
            quote! { __builder.live_placeholder(); },
        );
        match binding.kind {
            BindingKind::Int => {
                self.emit_op(
                    OpMeta::linear(OpKind::GuardValue, vec![Register::int(reg)], vec![]),
                    quote! { __builder.int_guard_value(#reg); },
                );
            }
            BindingKind::Ref => {
                self.emit_op(
                    OpMeta::linear(OpKind::GuardValue, vec![Register::ref_(reg)], vec![]),
                    quote! { __builder.ref_guard_value(#reg); },
                );
            }
            BindingKind::Float => {
                self.emit_op(
                    OpMeta::linear(OpKind::GuardValue, vec![Register::float(reg)], vec![]),
                    quote! { __builder.float_guard_value(#reg); },
                );
            }
        }
        Some(binding)
    }

    pub(super) fn lower_call_value(&mut self, call: &ExprCall) -> Option<Binding> {
        let policy = self.resolve_call_policy(&call.func)?;
        if call.args.len() > MAX_HELPER_CALL_ARITY {
            return None;
        }

        let mut arg_bindings = Vec::with_capacity(call.args.len());
        let mut depends_on_stack = false;
        for arg in &call.args {
            let binding = self.lower_value_expr(arg)?;
            arg_bindings.push(binding.clone());
            depends_on_stack |= binding.depends_on_stack;
        }

        let reg = self.alloc_reg();
        let func = &call.func;
        let mut result_kind = BindingKind::Int;
        // jtransform.py:467-471 / 480-482: `-live-` follows the call, it does
        // not precede it.  Decide here whether the explicit arm below needs a
        // trailing marker; the inline arms emit their own (`post_live`) and
        // the inferred arm emits a runtime-conditional one, so both report
        // `false` and are excluded.
        let post_live_after_call = match &policy {
            CallPolicySpec::Explicit(kind) => explicit_call_emits_post_live(*kind),
            CallPolicySpec::Infer => false,
        };
        match policy {
            CallPolicySpec::Explicit(kind) => match kind {
                crate::jit_interp::CallPolicyKind::ResidualInt
                | crate::jit_interp::CallPolicyKind::MayForceInt
                | crate::jit_interp::CallPolicyKind::ReleaseGilInt
                | crate::jit_interp::CallPolicyKind::LoopInvariantInt => {
                    let canonical_call = match kind {
                        crate::jit_interp::CallPolicyKind::ResidualInt => {
                            quote! { residual_call_int_canonical_via_target }
                        }
                        crate::jit_interp::CallPolicyKind::MayForceInt => {
                            quote! { call_may_force_int_canonical_via_target }
                        }
                        crate::jit_interp::CallPolicyKind::ReleaseGilInt => {
                            quote! { call_release_gil_int_canonical_via_target }
                        }
                        crate::jit_interp::CallPolicyKind::LoopInvariantInt => {
                            quote! { call_loopinvariant_int_canonical_via_target }
                        }
                        _ => unreachable!(),
                    };
                    // A declared `residual_writes` mutator that RETURNS a
                    // used value routes through this value-call path; it must
                    // still carry the field write-set `EffectInfo` so the
                    // optimizer invalidates the cached `getfield_gc_i` after
                    // the call — exactly as the statement form does.  Without
                    // it a stack-mutating residual (e.g. `jit_pop_is_zero`
                    // feeding a branch) leaves the cached `selected.size`
                    // stale and loop-peel const-folds it.  Only the residual
                    // policy qualifies (may-force / release-gil /
                    // loop-invariant carry their own effects).
                    let write_ei = match kind {
                        crate::jit_interp::CallPolicyKind::ResidualInt => {
                            self.residual_write_effect_info_tokens(func, true)
                        }
                        _ => None,
                    };
                    if let Some(arg_regs) = int_arg_regs(&arg_bindings) {
                        let call_invocation = if let Some(write_ei) = &write_ei {
                            quote! {
                                __builder.residual_call_int_canonical_via_target_with_effect_info(
                                    __fn_idx,
                                    &[#(majit_metainterp::JitCallArg::int(#arg_regs)),*],
                                    #reg,
                                    #write_ei,
                                );
                            }
                        } else {
                            quote! {
                                __builder.#canonical_call(
                                    __fn_idx,
                                    &[#(majit_metainterp::JitCallArg::int(#arg_regs)),*],
                                    #reg,
                                );
                            }
                        };
                        self.emit_op(
                            OpMeta::linear(
                                OpKind::Call,
                                Register::ints(&arg_regs),
                                vec![Register::new(result_kind, reg)],
                            ),
                            quote! {
                                let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                                #call_invocation
                            },
                        );
                    } else {
                        let typed_args = typed_call_arg_tokens(&arg_bindings);
                        let __arg_regs: Vec<Register> =
                            arg_bindings.iter().map(Register::from_binding).collect();
                        let call_invocation = if let Some(write_ei) = &write_ei {
                            quote! {
                                __builder.residual_call_int_canonical_via_target_with_effect_info(
                                    __fn_idx,
                                    #typed_args,
                                    #reg,
                                    #write_ei,
                                );
                            }
                        } else {
                            quote! {
                                __builder.#canonical_call(__fn_idx, #typed_args, #reg);
                            }
                        };
                        self.emit_op(
                            OpMeta::linear(
                                OpKind::Call,
                                __arg_regs,
                                vec![Register::new(result_kind, reg)],
                            ),
                            quote! {
                                let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                                #call_invocation
                            },
                        );
                    }
                }
                // `call.py:303` non-elidable EF_CANNOT_RAISE int — explicit policy.
                crate::jit_interp::CallPolicyKind::ResidualIntCannotRaise => {
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::Call,
                            __arg_regs,
                            vec![Register::new(result_kind, reg)],
                        ),
                        quote! {
                            let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                            __builder.residual_call_int_canonical_via_target_with_effect_info(
                                __fn_idx,
                                #typed_args,
                                #reg,
                                majit_metainterp::cannot_raise_effect_info(),
                            );
                        },
                    );
                }
                crate::jit_interp::CallPolicyKind::ElidableInt
                | crate::jit_interp::CallPolicyKind::ElidableIntCannotRaise
                | crate::jit_interp::CallPolicyKind::ElidableIntOrMemerror => {
                    // Parity #14 Slice C.4 + Parity #20: see the stmt-form
                    // ElidableInt arm earlier in this file for the
                    // canonical migration rationale and the 3-way
                    // `_canraise(op)` pick from `call.py:292-299`.
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    let call_stmt = match kind {
                        crate::jit_interp::CallPolicyKind::ElidableInt => quote! {
                            __builder.call_pure_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                        },
                        crate::jit_interp::CallPolicyKind::ElidableIntCannotRaise => quote! {
                            __builder.call_pure_int_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #reg);
                        },
                        crate::jit_interp::CallPolicyKind::ElidableIntOrMemerror => quote! {
                            __builder.call_pure_int_canonical_via_target_or_memerror(__fn_idx, #typed_args, #reg);
                        },
                        _ => unreachable!(),
                    };
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::Call,
                            __arg_regs,
                            vec![Register::new(result_kind, reg)],
                        ),
                        quote! {
                            let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                            #call_stmt
                        },
                    );
                }
                // Non-wrapped ref-returning call (value position).
                crate::jit_interp::CallPolicyKind::ResidualRef => {
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let reg = self.alloc_reg();
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    self.emit_op(
                        OpMeta::linear(OpKind::Call, __arg_regs, vec![Register::ref_(reg)]),
                        quote! {
                            let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                            let __typed_args = #typed_args;
                            __builder.residual_call_ref_canonical_via_target(__fn_idx, __typed_args, #reg);
                        },
                    );
                    // jtransform.py:467-470 — a residual ref call can raise, so
                    // the trailing `-live-` follows it exactly as the shared
                    // path below emits for the other explicit residual arms.
                    // This arm returns early (to attach `struct_type`), so it
                    // cannot fall through to that shared emission and must emit
                    // the marker itself.
                    if post_live_after_call {
                        self.emit_op(
                            OpMeta::live_marker(),
                            quote! { let _ = __builder.live_placeholder(); },
                        );
                    }
                    // Check `call_returns` config for a declared return
                    // struct type, enabling subsequent `result.field`
                    // access to resolve through `ref_fields`.
                    let func_segments = canonical_expr_segments(func);
                    let struct_type = func_segments.and_then(|segs| {
                        self.config.and_then(|c| c.call_returns.get(&segs).cloned())
                    });
                    return Some(Binding {
                        reg,
                        kind: BindingKind::Ref,
                        depends_on_stack: false,
                        struct_type,
                    });
                }
                crate::jit_interp::CallPolicyKind::NurseryAllocRef => {
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let reg = self.alloc_reg();
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    self.emit_op(
                        OpMeta::linear(OpKind::Call, __arg_regs, vec![Register::ref_(reg)]),
                        quote! {
                            let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                            let __typed_args = #typed_args;
                            __builder.residual_call_ref_canonical_via_target_with_effect_info(__fn_idx, __typed_args, #reg, majit_metainterp::nursery_alloc_effect_info());
                        },
                    );
                    // jtransform.py:467-470 — a residual ref call can raise, so
                    // the trailing `-live-` follows it exactly as the shared
                    // path below emits for the other explicit residual arms.
                    // This arm returns early (to attach `struct_type`), so it
                    // cannot fall through to that shared emission and must emit
                    // the marker itself.
                    if post_live_after_call {
                        self.emit_op(
                            OpMeta::live_marker(),
                            quote! { let _ = __builder.live_placeholder(); },
                        );
                    }
                    // Check `call_returns` config for a declared return
                    // struct type, enabling subsequent `result.field`
                    // access to resolve through `ref_fields`.
                    let func_segments = canonical_expr_segments(func);
                    let struct_type = func_segments.and_then(|segs| {
                        self.config.and_then(|c| c.call_returns.get(&segs).cloned())
                    });
                    return Some(Binding {
                        reg,
                        kind: BindingKind::Ref,
                        depends_on_stack: false,
                        struct_type,
                    });
                }
                crate::jit_interp::CallPolicyKind::ResidualIntWrapped
                | crate::jit_interp::CallPolicyKind::ResidualIntCannotRaiseWrapped
                | crate::jit_interp::CallPolicyKind::MayForceIntWrapped
                | crate::jit_interp::CallPolicyKind::ReleaseGilIntWrapped
                | crate::jit_interp::CallPolicyKind::LoopInvariantIntWrapped
                | crate::jit_interp::CallPolicyKind::ElidableIntWrapped
                | crate::jit_interp::CallPolicyKind::ElidableIntCannotRaiseWrapped
                | crate::jit_interp::CallPolicyKind::ElidableIntOrMemerrorWrapped
                | crate::jit_interp::CallPolicyKind::ResidualRefWrapped
                | crate::jit_interp::CallPolicyKind::ResidualRefCannotRaiseWrapped
                | crate::jit_interp::CallPolicyKind::MayForceRefWrapped
                | crate::jit_interp::CallPolicyKind::LoopInvariantRefWrapped
                | crate::jit_interp::CallPolicyKind::ElidableRefWrapped
                | crate::jit_interp::CallPolicyKind::ElidableRefCannotRaiseWrapped
                | crate::jit_interp::CallPolicyKind::ElidableRefOrMemerrorWrapped
                | crate::jit_interp::CallPolicyKind::ResidualFloatWrapped
                | crate::jit_interp::CallPolicyKind::ResidualFloatCannotRaiseWrapped
                | crate::jit_interp::CallPolicyKind::MayForceFloatWrapped
                | crate::jit_interp::CallPolicyKind::ReleaseGilFloatWrapped
                | crate::jit_interp::CallPolicyKind::LoopInvariantFloatWrapped
                | crate::jit_interp::CallPolicyKind::ElidableFloatWrapped
                | crate::jit_interp::CallPolicyKind::ElidableFloatCannotRaiseWrapped
                | crate::jit_interp::CallPolicyKind::ElidableFloatOrMemerrorWrapped => {
                    let policy_path = helper_policy_path(&call.func)?;
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let call_stmt = match kind {
                        crate::jit_interp::CallPolicyKind::ResidualIntWrapped => {
                            quote! { __builder.residual_call_int_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        // `call.py:303` non-elidable EF_CANNOT_RAISE int — wrapped value-form.
                        crate::jit_interp::CallPolicyKind::ResidualIntCannotRaiseWrapped => {
                            quote! {
                                __builder.residual_call_int_canonical_via_target_with_effect_info(
                                    __fn_idx,
                                    #typed_args,
                                    #reg,
                                    majit_metainterp::cannot_raise_effect_info(),
                                );
                            }
                        }
                        crate::jit_interp::CallPolicyKind::MayForceIntWrapped => {
                            quote! { __builder.call_may_force_int_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ReleaseGilIntWrapped => {
                            quote! { __builder.call_release_gil_int_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::LoopInvariantIntWrapped => {
                            quote! { __builder.call_loopinvariant_int_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableIntWrapped => {
                            quote! { __builder.call_pure_int_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableIntCannotRaiseWrapped => {
                            quote! { __builder.call_pure_int_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableIntOrMemerrorWrapped => {
                            quote! { __builder.call_pure_int_canonical_via_target_or_memerror(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ResidualRefWrapped => {
                            result_kind = BindingKind::Ref;
                            quote! { __builder.residual_call_ref_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        // `call.py:303` non-elidable EF_CANNOT_RAISE ref — wrapped value-form.
                        crate::jit_interp::CallPolicyKind::ResidualRefCannotRaiseWrapped => {
                            result_kind = BindingKind::Ref;
                            quote! {
                                __builder.residual_call_ref_canonical_via_target_with_effect_info(
                                    __fn_idx,
                                    #typed_args,
                                    #reg,
                                    majit_metainterp::cannot_raise_effect_info(),
                                );
                            }
                        }
                        crate::jit_interp::CallPolicyKind::MayForceRefWrapped => {
                            result_kind = BindingKind::Ref;
                            quote! { __builder.call_may_force_ref_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::LoopInvariantRefWrapped => {
                            result_kind = BindingKind::Ref;
                            quote! { __builder.call_loopinvariant_ref_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableRefWrapped => {
                            result_kind = BindingKind::Ref;
                            quote! { __builder.call_pure_ref_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableRefCannotRaiseWrapped => {
                            result_kind = BindingKind::Ref;
                            quote! { __builder.call_pure_ref_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableRefOrMemerrorWrapped => {
                            result_kind = BindingKind::Ref;
                            quote! { __builder.call_pure_ref_canonical_via_target_or_memerror(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ResidualFloatWrapped => {
                            result_kind = BindingKind::Float;
                            quote! { __builder.residual_call_float_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        // `call.py:303` non-elidable EF_CANNOT_RAISE float — wrapped value-form.
                        crate::jit_interp::CallPolicyKind::ResidualFloatCannotRaiseWrapped => {
                            result_kind = BindingKind::Float;
                            quote! {
                                __builder.residual_call_float_canonical_via_target_with_effect_info(
                                    __fn_idx,
                                    #typed_args,
                                    #reg,
                                    majit_metainterp::cannot_raise_effect_info(),
                                );
                            }
                        }
                        crate::jit_interp::CallPolicyKind::MayForceFloatWrapped => {
                            result_kind = BindingKind::Float;
                            quote! { __builder.call_may_force_float_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ReleaseGilFloatWrapped => {
                            result_kind = BindingKind::Float;
                            quote! { __builder.call_release_gil_float_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::LoopInvariantFloatWrapped => {
                            result_kind = BindingKind::Float;
                            quote! { __builder.call_loopinvariant_float_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableFloatWrapped => {
                            result_kind = BindingKind::Float;
                            quote! { __builder.call_pure_float_canonical_via_target(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableFloatCannotRaiseWrapped => {
                            result_kind = BindingKind::Float;
                            quote! { __builder.call_pure_float_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableFloatOrMemerrorWrapped => {
                            result_kind = BindingKind::Float;
                            quote! { __builder.call_pure_float_canonical_via_target_or_memerror(__fn_idx, #typed_args, #reg); }
                        }
                        _ => unreachable!(),
                    };
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    let __slot_tokens = CondCallEffectSlot::for_wrapped_kind(kind);
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::Call,
                            __arg_regs,
                            vec![Register::new(result_kind, reg)],
                        ),
                        quote! {
                            let (__policy, _inline_builder, __trace_target, __concrete_target, _prebuild, __save_err) = #policy_path();
                            if __trace_target.is_null() && __concrete_target.is_null() {
                                panic!("wrapped helper policy requires generated call-target wrappers");
                            }
                            let __trace_target = if __trace_target.is_null() {
                                __concrete_target
                            } else {
                                __trace_target
                            };
                            let __concrete_target = if __concrete_target.is_null() {
                                __trace_target
                            } else {
                                __concrete_target
                            };
                            let __fn_idx = __builder.add_call_target_with_save_err(
                                __trace_target,
                                __concrete_target,
                                #__slot_tokens,
                                __save_err,
                            );
                            #call_stmt
                        },
                    );
                }
                crate::jit_interp::CallPolicyKind::InlineInt
                | crate::jit_interp::CallPolicyKind::InlineRef
                | crate::jit_interp::CallPolicyKind::InlineFloat => {
                    result_kind = binding_kind_for_inline_policy(kind).unwrap();
                    let builder_path = inline_builder_path(&call.func)?;
                    let prebuild_path = inline_prebuild_path(&call.func)?;
                    let (inline_call, post_live) = inline_call_tokens(&arg_bindings, reg);
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    // RPython `pyjitpl.py:2255 finish_setup` order: the
                    // helper's per-marker `-live-` triples must land in
                    // `asm.all_liveness` before the parent's
                    // `JitDriver::install_canonical_liveness` snapshot.
                    // Queue the helper's `__majit_inline_jitcode_<name>_
                    // prebuild(__asm)` call into the parent's prebuild
                    // accumulator; `liveness_prebuild_tokens` will splice
                    // it ahead of the parent arm's own
                    // `__asm._register_liveness_offset` calls.
                    self.inline_liveness_prebuild.push(quote! {
                        #prebuild_path(__asm);
                    });
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::InlineCall,
                            __arg_regs,
                            vec![Register::new(result_kind, reg)],
                        ),
                        quote! {
                            let __sub_jitcode = #builder_path(__asm);
                            let (__sub_return_kind, _) = __sub_jitcode
                                .trailing_return_info()
                                .expect("inline helper jitcode must end in a typed return opcode");
                            let __sub_idx = __builder.add_sub_jitcode(__sub_jitcode);
                            #inline_call
                        },
                    );
                    self.emit_op(OpMeta::live_marker(), post_live);
                }
                crate::jit_interp::CallPolicyKind::InlinePipelineInt
                | crate::jit_interp::CallPolicyKind::InlinePipelineRef
                | crate::jit_interp::CallPolicyKind::InlinePipelineFloat => {
                    result_kind = binding_kind_for_inline_policy(kind).unwrap();
                    // Resolve the callee by name through the host's
                    // `__majit_pipeline_jitcode`, which returns the
                    // pipeline-built (`make_jitcodes()`, codewriter.py:89)
                    // sub-jitcode shell. Unlike the macro-helper `Inline*`
                    // path there is no `_prebuild` symbol: the pipeline
                    // jitcode carries its own `-live-` markers from the build,
                    // installed when the host registers the descr pool.
                    let pipeline_name = match &*call.func {
                        syn::Expr::Path(ep) => ep.path.segments.last().map(|s| s.ident.to_string()),
                        _ => None,
                    }?;
                    let (inline_call, post_live) = inline_call_tokens(&arg_bindings, reg);
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::InlineCall,
                            __arg_regs,
                            vec![Register::new(result_kind, reg)],
                        ),
                        quote! {
                            use majit_metainterp::jitcode::JitCodeRuntimeExt as _;
                            let __sub_jitcode = __majit_pipeline_jitcode(#pipeline_name);
                            let (__sub_return_kind, _) = __sub_jitcode
                                .trailing_return_info()
                                .expect("pipeline jitcode must end in a typed return opcode");
                            let __sub_idx = __builder.add_sub_jitcode_arc(__sub_jitcode);
                            #inline_call
                        },
                    );
                    self.emit_op(OpMeta::live_marker(), post_live);
                }
                _ => return None,
            },
            CallPolicySpec::Infer => {
                let policy_path = helper_policy_path(&call.func)?;
                let typed_args = typed_call_arg_tokens(&arg_bindings);
                let (inline_call, post_live) = inline_call_tokens(&arg_bindings, reg);
                let int_arg_regs = int_arg_regs(&arg_bindings);
                let unsupported = self.inference_failure_tokens(
                    "inferred helper policy only supports int-return value calls here; use an explicit inline_ref/inline_float or *_ref_wrapped/*_float_wrapped policy",
                );
                // RPython `codewriter.py:55` precomputes per-helper
                // `-live-` triples and `pyjitpl.py:2255 finish_setup`
                // snapshots `metainterp_sd.liveness_info` after every
                // helper has had its triples registered.  When the
                // inferred path's runtime policy resolves to `4u8`
                // (`call.py` `EF_INLINE_HELPER`), the inline-helper
                // builder at line 4u8's runtime arm below executes
                // `__inline_builder(__asm)` which materialises a sub-
                // JitCode containing the helper's `-live-` markers.
                // Without an install-time prebuild call queuing into
                // `inline_liveness_prebuild`, the helper's per-marker
                // triples never enter `asm.all_liveness` ahead of the
                // parent driver's `install_canonical_liveness`
                // snapshot — diverging from the
                // `codewriter.py:55 → finish_setup` order.
                //
                // Queue the prebuild path when the helper exposes one
                // (only inline-able helpers do).  At runtime, the
                // prebuild call is idempotent — it registers triples
                // even when the runtime-resolved policy is non-inline,
                // matching the explicit `InlineInt/InlineRef/InlineFloat`
                // arms above which queue unconditionally.  Non-inline
                // helpers don't expose a `inline_prebuild_path` symbol,
                // so the `.ok()` swallow here is the parity-correct
                // "no triples to register" signal.
                if let Some(prebuild_path) = inline_prebuild_path(&call.func) {
                    self.inline_liveness_prebuild.push(quote! {
                        #prebuild_path(__asm);
                    });
                }
                let __arg_regs: Vec<Register> =
                    arg_bindings.iter().map(Register::from_binding).collect();
                let __slot_tokens = CondCallEffectSlot::slot_from_policy_tokens();
                if let Some(_arg_regs) = int_arg_regs {
                    // Inferred path: result is Int (the explicit-Inline case is
                    // handled at line 4u8 of the runtime match below, but the
                    // emit-time OpMeta destination still tracks the call's
                    // post-condition register slot — Int matches the typed-call
                    // helper that produces it).
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::Call,
                            __arg_regs.clone(),
                            vec![Register::int(reg)],
                        ),
                        quote! {
                            let (__policy, __inline_builder, __trace_target, __concrete_target, _prebuild, __save_err) = #policy_path();
                            let __trace_target = if __trace_target.is_null() {
                                #func as *const ()
                            } else {
                                __trace_target
                            };
                            let __concrete_target = if __concrete_target.is_null() {
                                __trace_target
                            } else {
                                __concrete_target
                            };
                            let __fn_idx = __builder.add_call_target_with_save_err(
                                __trace_target,
                                __concrete_target,
                                #__slot_tokens,
                                __save_err,
                            );
                            match __policy {
                                #INT_DONT_LOOK_INSIDE => {
                                    __builder.residual_call_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                                }
                                // `call.py:303` non-elidable EF_CANNOT_RAISE for int.
                                #INT_DONT_LOOK_INSIDE_CANNOT_RAISE => {
                                    __builder.residual_call_int_canonical_via_target_with_effect_info(
                                        __fn_idx,
                                        #typed_args,
                                        #reg,
                                        majit_metainterp::cannot_raise_effect_info(),
                                    );
                                }
                                #INT_ELIDABLE => {
                                    __builder.call_pure_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                                }
                                // call.py:299 _canraise == False — EF_ELIDABLE_CANNOT_RAISE.
                                #INT_ELIDABLE_CANNOT_RAISE => {
                                    __builder.call_pure_int_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #reg);
                                }
                                // call.py:295 _canraise == "mem" — EF_ELIDABLE_OR_MEMORYERROR.
                                #INT_ELIDABLE_OR_MEMERROR => {
                                    __builder.call_pure_int_canonical_via_target_or_memerror(__fn_idx, #typed_args, #reg);
                                }
                                #INT_INLINE => {
                                    let __builder_fn: fn(&mut majit_metainterp::Assembler) -> majit_metainterp::JitCode =
                                        unsafe { std::mem::transmute(__inline_builder) };
                                    let __sub_jitcode = __builder_fn(__asm);
                                    let (__sub_return_kind, _) =
                                        <majit_metainterp::JitCode as majit_metainterp::jitcode::JitCodeRuntimeExt>::trailing_return_info(&__sub_jitcode)
                                        .expect("inline helper jitcode must end in a typed return opcode");
                                    let __sub_idx = __builder.add_sub_jitcode(__sub_jitcode);
                                    #inline_call
                                }
                                #INT_MAY_FORCE => {
                                    __builder.call_may_force_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                                }
                                #INT_RELEASE_GIL => {
                                    __builder.call_release_gil_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                                }
                                #INT_LOOP_INVARIANT => {
                                    __builder.call_loopinvariant_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                                }
                                _ => {
                                    #unsupported
                                }
                            }
                        },
                    );
                    // jtransform.py:467/480-482 — `inline_call_*` is always
                    // followed by `-live-`; a residual call (`call_*`)
                    // appends `-live-` only when `calldescr_canraise(calldescr)`
                    // (`call.py:295-300`).  In inferred mode the policy
                    // byte selects the calldescr at runtime, so emit the
                    // marker conditional on the can-raise codes plus the
                    // inline byte (4u8) which forces emit per
                    // jtransform.py:480-482.  LoopInvariantInt (18u8) and
                    // ElidableCannotRaiseInt (19u8) skip the marker
                    // because their calldescrs are statically cannot-raise.
                    self.emit_op(
                        OpMeta::live_marker_if(inferred_policy_live_condition(
                            func,
                            &[
                                INT_DONT_LOOK_INSIDE,
                                INT_ELIDABLE,
                                INT_INLINE,
                                INT_MAY_FORCE,
                                INT_RELEASE_GIL,
                                INT_ELIDABLE_OR_MEMERROR,
                            ],
                        )),
                        post_live.clone(),
                    );
                } else {
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::Call,
                            __arg_regs,
                            vec![Register::int(reg)],
                        ),
                        quote! {
                            let (__policy, __inline_builder, __trace_target, __concrete_target, _prebuild, __save_err) = #policy_path();
                            let __trace_target = if __trace_target.is_null() {
                                #func as *const ()
                            } else {
                                __trace_target
                            };
                            let __concrete_target = if __concrete_target.is_null() {
                                __trace_target
                            } else {
                                __concrete_target
                            };
                            let __fn_idx = __builder.add_call_target_with_save_err(
                                __trace_target,
                                __concrete_target,
                                #__slot_tokens,
                                __save_err,
                            );
                            match __policy {
                                #INT_DONT_LOOK_INSIDE => {
                                    __builder.residual_call_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                                }
                                // `call.py:303` non-elidable EF_CANNOT_RAISE for int.
                                #INT_DONT_LOOK_INSIDE_CANNOT_RAISE => {
                                    __builder.residual_call_int_canonical_via_target_with_effect_info(
                                        __fn_idx,
                                        #typed_args,
                                        #reg,
                                        majit_metainterp::cannot_raise_effect_info(),
                                    );
                                }
                                #INT_ELIDABLE => {
                                    __builder.call_pure_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                                }
                                // call.py:299 _canraise == False — EF_ELIDABLE_CANNOT_RAISE.
                                #INT_ELIDABLE_CANNOT_RAISE => {
                                    __builder.call_pure_int_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #reg);
                                }
                                // call.py:295 _canraise == "mem" — EF_ELIDABLE_OR_MEMORYERROR.
                                #INT_ELIDABLE_OR_MEMERROR => {
                                    __builder.call_pure_int_canonical_via_target_or_memerror(__fn_idx, #typed_args, #reg);
                                }
                                #INT_INLINE => {
                                let __builder_fn: fn(&mut majit_metainterp::Assembler) -> majit_metainterp::JitCode =
                                    unsafe { std::mem::transmute(__inline_builder) };
                                let __sub_jitcode = __builder_fn(__asm);
                                let (__sub_return_kind, _) =
                                    <majit_metainterp::JitCode as majit_metainterp::jitcode::JitCodeRuntimeExt>::trailing_return_info(&__sub_jitcode)
                                    .expect("inline helper jitcode must end in a typed return opcode");
                                let __sub_idx = __builder.add_sub_jitcode(__sub_jitcode);
                                #inline_call
                            }
                            #INT_MAY_FORCE => {
                                __builder.call_may_force_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                            }
                            #INT_RELEASE_GIL => {
                                __builder.call_release_gil_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                            }
                            #INT_LOOP_INVARIANT => {
                                __builder.call_loopinvariant_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                            }
                            _ => {
                                #unsupported
                            }
                        }
                    });
                    // jtransform.py:467/480-482 — see int_arg_regs branch above.
                    self.emit_op(
                        OpMeta::live_marker_if(inferred_policy_live_condition(
                            func,
                            &[
                                INT_DONT_LOOK_INSIDE,
                                INT_ELIDABLE,
                                INT_INLINE,
                                INT_MAY_FORCE,
                                INT_RELEASE_GIL,
                                INT_ELIDABLE_OR_MEMERROR,
                            ],
                        )),
                        post_live,
                    );
                }
            }
        }

        // jtransform.py:467-471 / 480-482 — trailing `-live-` for the explicit
        // residual / elidable / may-force / release-gil arms (computed above).
        if post_live_after_call {
            self.emit_op(
                OpMeta::live_marker(),
                quote! { let _ = __builder.live_placeholder(); },
            );
        }

        Some(Binding {
            reg,
            kind: result_kind,
            depends_on_stack,
            struct_type: None,
        })
    }

    /// Lower an `Expr::MethodCall` by mapping the receiver ident to its
    /// canonical owning type (via `LowererConfig.state_type_name` /
    /// `env_type_name`) and dispatching through the existing call-policy
    /// table keyed on `<type>::<method>` segments. The receiver is lowered
    /// as the first call argument so the owning type's `&self` parameter
    /// gets a real register binding.
    ///
    /// Receiver-resolution policy: only the env parameter (`program`) and
    /// the state parameter (`state`) are accepted; any other receiver
    /// returns `None`. RPython `call.py:282-324 getcalldescr` keys on
    /// graph identity (no naming collision possible); pyre keys on
    /// canonical path so the `<state_type|env_type>::<method>` lookup
    /// preserves that fidelity. Arbitrary receivers cannot be resolved
    /// without the owning type identity, so they fall through.
    ///
    /// Currently only the `Elidable*` Int policy family is supported
    /// (the consumer set required for `Program::get_req_size`-shaped
    /// helpers). Wrapped / inline / non-int return policies fall through;
    /// extending them mirrors the corresponding `lower_call_value` arms
    /// when needed.
    ///
    /// RPython parity: `jtransform.py:456-470 rewrite_op` (graph-identity
    /// lookup) + `call.py:282-324 getcalldescr`.
    fn lower_method_call_value(&mut self, call: &ExprMethodCall) -> Option<Binding> {
        let receiver_ident = match &*call.receiver {
            Expr::Path(ExprPath { path, .. }) => path.get_ident()?,
            _ => return None,
        };
        let receiver_name = receiver_ident.to_string();
        let config = self.config?;
        let type_name = match receiver_name.as_str() {
            "program" => config.env_type_name.clone(),
            "state" => config.state_type_name.clone(),
            _ => return None,
        };

        let method_segments = vec![type_name.clone(), call.method.to_string()];
        let policy = self
            .call_policies
            .iter()
            .find(|(p, _)| *p == method_segments)
            .map(|(_, spec)| spec.clone())?;
        let kind = match policy {
            CallPolicySpec::Explicit(kind) => kind,
            // Method-call inference is not supported; the policy table
            // must declare the method explicitly.
            CallPolicySpec::Infer => return None,
        };

        // Receiver counts as the first call argument; RPython
        // `jtransform.py:456 rewrite_op` similarly threads `op.args[0]`
        // (the receiver / first positional) ahead of the rest.
        if call.args.len() + 1 > MAX_HELPER_CALL_ARITY {
            return None;
        }

        let receiver_binding = self.lower_value_expr(&call.receiver)?;
        let mut arg_bindings = Vec::with_capacity(call.args.len() + 1);
        let mut depends_on_stack = receiver_binding.depends_on_stack;
        arg_bindings.push(receiver_binding);
        for arg in &call.args {
            let binding = self.lower_value_expr(arg)?;
            depends_on_stack |= binding.depends_on_stack;
            arg_bindings.push(binding);
        }

        // Construct the `<Type>::<method>` path tokens for `add_fn_ptr`.
        let type_ident = format_ident!("{}", type_name);
        let method_ident = &call.method;
        let func_path = quote! { <#type_ident>::#method_ident };

        let reg = self.alloc_reg();
        let result_kind = BindingKind::Int;
        match kind {
            crate::jit_interp::CallPolicyKind::ElidableInt
            | crate::jit_interp::CallPolicyKind::ElidableIntCannotRaise
            | crate::jit_interp::CallPolicyKind::ElidableIntOrMemerror => {
                let typed_args = typed_call_arg_tokens(&arg_bindings);
                let __arg_regs: Vec<Register> =
                    arg_bindings.iter().map(Register::from_binding).collect();
                let call_stmt = match kind {
                    crate::jit_interp::CallPolicyKind::ElidableInt => quote! {
                        __builder.call_pure_int_canonical_via_target(__fn_idx, #typed_args, #reg);
                    },
                    crate::jit_interp::CallPolicyKind::ElidableIntCannotRaise => quote! {
                        __builder.call_pure_int_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #reg);
                    },
                    crate::jit_interp::CallPolicyKind::ElidableIntOrMemerror => quote! {
                        __builder.call_pure_int_canonical_via_target_or_memerror(__fn_idx, #typed_args, #reg);
                    },
                    _ => unreachable!(),
                };
                self.emit_op(
                    OpMeta::linear(
                        OpKind::Call,
                        __arg_regs,
                        vec![Register::new(result_kind, reg)],
                    ),
                    quote! {
                        let __fn_idx = __builder.add_fn_ptr(#func_path as *const ());
                        #call_stmt
                    },
                );
            }
            // Other policy kinds are not yet wired for method-call RHS.
            // Consumers needing residual / may_force / wrapped / inline
            // method-call lowering must add the corresponding arm here
            // mirroring `lower_call_value`'s shape.
            _ => return None,
        }

        Some(Binding {
            reg,
            kind: result_kind,
            depends_on_stack,
            struct_type: None,
        })
    }

    fn lower_if_value(&mut self, expr_if: &ExprIf) -> Option<Binding> {
        if let Some(binding) = self.lower_bool_if(expr_if) {
            return Some(binding);
        }

        let cond = self.lower_value_expr(&expr_if.cond)?;
        if !matches!(cond.kind, BindingKind::Int) {
            return None;
        }
        let (_, else_expr) = expr_if.else_branch.as_ref()?;
        // A branch that fails to lower fails the whole if-expression as
        // unsupported (`?`): the codewriter lowers expressible ops exactly
        // or rejects the construct (`jtransform.py`), never replacing one
        // branch with a runtime abort stub. The caller degrades cleanly
        // (sub-JitCode entry returns `None`, arm runs in the interpreter).
        let (then_seq, then_binding) =
            self.lower_branch_value_expr(&Expr::Block(syn::ExprBlock {
                attrs: Vec::new(),
                label: None,
                block: expr_if.then_branch.clone(),
            }))?;
        let (else_seq, else_binding) = self.lower_branch_value_expr(else_expr)?;
        if !matches!(then_binding.kind, BindingKind::Int)
            || !matches!(else_binding.kind, BindingKind::Int)
        {
            return None;
        }
        let then_reg = then_binding.reg;
        let else_reg = else_binding.reg;

        let else_label = self.alloc_label();
        let end_label = self.alloc_label();
        let result_reg = self.alloc_reg();
        let cond_reg = cond.reg;

        self.emit_aux(quote! { let #else_label = __builder.new_label(); });
        self.emit_aux(quote! { let #end_label = __builder.new_label(); });
        self.emit_op(
            OpMeta::live_marker(),
            quote! { let _ = __builder.live_placeholder(); },
        );
        self.emit_conditional_guard(cond_reg, &else_label);
        self.append_lowered_sequence(then_seq);
        self.emit_op(
            OpMeta::linear(
                OpKind::MoveI,
                vec![Register::int(then_reg)],
                vec![Register::int(result_reg)],
            ),
            quote! { __builder.move_i(#result_reg, #then_reg); },
        );
        self.emit_jump(&end_label);
        self.emit_label_def(&else_label);
        self.append_lowered_sequence(else_seq);
        self.emit_op(
            OpMeta::linear(
                OpKind::MoveI,
                vec![Register::int(else_reg)],
                vec![Register::int(result_reg)],
            ),
            quote! { __builder.move_i(#result_reg, #else_reg); },
        );
        self.emit_label_def(&end_label);

        Some(Binding {
            reg: result_reg,
            kind: BindingKind::Int,
            depends_on_stack: cond.depends_on_stack
                || then_binding.depends_on_stack
                || else_binding.depends_on_stack,
            struct_type: None,
        })
    }

    fn lower_bool_if(&mut self, expr_if: &ExprIf) -> Option<Binding> {
        let (then_value, else_value) = extract_bool_branch_values(expr_if)?;
        let cond = self.lower_value_expr(&expr_if.cond)?;
        if !matches!(cond.kind, BindingKind::Int) {
            return None;
        }
        match (then_value, else_value) {
            (1, 0) => Some(cond),
            (0, 1) => {
                let zero_reg = self.alloc_reg();
                self.emit_op(
                    OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(zero_reg)]),
                    quote! { __builder.load_const_i_value(#zero_reg, 0); },
                );
                let reg = self.alloc_reg();
                let cond_reg = cond.reg;
                self.emit_op(
                    OpMeta::linear(
                        OpKind::BinopI,
                        Register::ints(&[cond_reg, zero_reg]),
                        vec![Register::int(reg)],
                    ),
                    quote! { __builder.record_binop_i(#reg, majit_ir::OpCode::IntEq, #cond_reg, #zero_reg); },
                );
                Some(Binding {
                    reg,
                    kind: BindingKind::Int,
                    depends_on_stack: cond.depends_on_stack,
                    struct_type: None,
                })
            }
            _ => None,
        }
    }

    fn lower_unary(&mut self, op: &UnOp, expr: &Expr) -> Option<Binding> {
        match op {
            UnOp::Neg(_) => {
                let inner = self.lower_value_expr(expr)?;
                if !matches!(inner.kind, BindingKind::Int) {
                    return None;
                }
                let reg = self.alloc_reg();
                let src_reg = inner.reg;
                self.emit_op(
                    OpMeta::linear(
                        OpKind::UnaryI,
                        vec![Register::int(src_reg)],
                        vec![Register::int(reg)],
                    ),
                    quote! { __builder.record_unary_i(#reg, majit_ir::OpCode::IntNeg, #src_reg); },
                );
                Some(Binding {
                    reg,
                    kind: BindingKind::Int,
                    depends_on_stack: inner.depends_on_stack,
                    struct_type: None,
                })
            }
            _ => None,
        }
    }

    /// Recognize a function call registered as a native int binop alias.
    /// Emits the corresponding IR op (IntAdd, IntSub, etc.) directly,
    /// bypassing the call-policy machinery.  The concrete path calls the
    /// original function normally — no observer replay needed (pure op).
    ///
    /// jtransform.py:2030 `_handle_int_special()` parity.
    fn lower_native_int_binop_call(&mut self, call: &ExprCall) -> Option<Binding> {
        let config = self.config?;
        let func_segments = canonical_expr_segments(&call.func)?;
        let opcode_name = config
            .native_int_binops
            .iter()
            .find(|(path, _)| *path == func_segments)
            .map(|(_, op)| op.clone())?;

        if call.args.len() != 2 {
            return None;
        }

        let lhs = self.lower_value_expr(&call.args[0])?;
        let rhs = self.lower_value_expr(&call.args[1])?;
        if !matches!(lhs.kind, BindingKind::Int) || !matches!(rhs.kind, BindingKind::Int) {
            return None;
        }

        let reg = self.alloc_reg();
        let lhs_reg = lhs.reg;
        let rhs_reg = rhs.reg;
        let opcode = Ident::new(&opcode_name, proc_macro2::Span::call_site());
        self.emit_op(
            OpMeta::linear(
                OpKind::BinopI,
                Register::ints(&[lhs_reg, rhs_reg]),
                vec![Register::int(reg)],
            ),
            binop_i_emit_tokens(reg, &opcode, lhs_reg, rhs_reg),
        );
        Some(Binding {
            reg,
            kind: BindingKind::Int,
            depends_on_stack: lhs.depends_on_stack || rhs.depends_on_stack,
            struct_type: None,
        })
    }

    fn lower_binary(&mut self, expr: &ExprBinary) -> Option<Binding> {
        let lhs = self.lower_value_expr(&expr.left)?;
        let rhs = self.lower_value_expr(&expr.right)?;
        if !matches!(lhs.kind, BindingKind::Int) || !matches!(rhs.kind, BindingKind::Int) {
            return None;
        }
        let opcode = opcode_for_binop(&expr.op)?;
        let reg = self.alloc_reg();
        let lhs_reg = lhs.reg;
        let rhs_reg = rhs.reg;
        self.emit_op(
            OpMeta::linear(
                OpKind::BinopI,
                Register::ints(&[lhs_reg, rhs_reg]),
                vec![Register::int(reg)],
            ),
            binop_i_emit_tokens(reg, &opcode, lhs_reg, rhs_reg),
        );
        Some(Binding {
            reg,
            kind: BindingKind::Int,
            depends_on_stack: lhs.depends_on_stack || rhs.depends_on_stack,
            struct_type: None,
        })
    }

    /// Lower an integer `as <ty>` cast to the truncate + (sign|zero)-extend
    /// the Rust `as` operator performs (RPython's rtyper inserts the same
    /// `int_signext` / mask when an interpreter narrows a wider value).
    /// A cast to a 64-bit or pointer-width target is a no-op on the
    /// already-machine-word binding.  A narrowing cast to a signed type
    /// sign-extends from the type's high bit (`(x << s) >> s`, arithmetic
    /// shift); to an unsigned type it masks the low bits (`x & ((1<<n)-1)`).
    /// Previously every int cast was a no-op, dropping the sign extension a
    /// `program[pc] as i8 as i64` operand fetch relies on (an 8-bit `0xEE`
    /// stayed `238` instead of `-18`, so backward branch targets computed
    /// out of range).
    fn lower_int_cast(&mut self, binding: Binding, ty: &Type) -> Option<Binding> {
        let ident = match ty {
            Type::Path(tp) => tp.path.get_ident()?,
            _ => return None,
        };
        let (signed, bits): (bool, u32) = match ident.to_string().as_str() {
            "i8" => (true, 8),
            "i16" => (true, 16),
            "i32" => (true, 32),
            "u8" => (false, 8),
            "u16" => (false, 16),
            "u32" => (false, 32),
            // 64-bit / pointer-width targets keep the full machine word.
            "i64" | "u64" | "isize" | "usize" => return Some(binding),
            _ => return None,
        };
        let depends_on_stack = binding.depends_on_stack;
        let x_reg = binding.reg;
        let result_reg = if signed {
            let shift = (64 - bits) as i64;
            let shift_reg = self.alloc_reg();
            self.emit_op(
                OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(shift_reg)]),
                quote! { __builder.load_const_i_value(#shift_reg, #shift); },
            );
            let shl_reg = self.alloc_reg();
            let lshift = syn::Ident::new("IntLshift", proc_macro2::Span::call_site());
            self.emit_op(
                OpMeta::linear(
                    OpKind::BinopI,
                    Register::ints(&[x_reg, shift_reg]),
                    vec![Register::int(shl_reg)],
                ),
                binop_i_emit_tokens(shl_reg, &lshift, x_reg, shift_reg),
            );
            let res_reg = self.alloc_reg();
            let rshift = syn::Ident::new("IntRshift", proc_macro2::Span::call_site());
            self.emit_op(
                OpMeta::linear(
                    OpKind::BinopI,
                    Register::ints(&[shl_reg, shift_reg]),
                    vec![Register::int(res_reg)],
                ),
                binop_i_emit_tokens(res_reg, &rshift, shl_reg, shift_reg),
            );
            res_reg
        } else {
            let mask = ((1u64 << bits) - 1) as i64;
            let mask_reg = self.alloc_reg();
            self.emit_op(
                OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(mask_reg)]),
                quote! { __builder.load_const_i_value(#mask_reg, #mask); },
            );
            let res_reg = self.alloc_reg();
            let and = syn::Ident::new("IntAnd", proc_macro2::Span::call_site());
            self.emit_op(
                OpMeta::linear(
                    OpKind::BinopI,
                    Register::ints(&[x_reg, mask_reg]),
                    vec![Register::int(res_reg)],
                ),
                binop_i_emit_tokens(res_reg, &and, x_reg, mask_reg),
            );
            res_reg
        };
        Some(Binding {
            reg: result_reg,
            kind: BindingKind::Int,
            depends_on_stack,
            struct_type: None,
        })
    }

    pub(super) fn lower_branch_expr(&mut self, expr: &Expr) -> Option<LoweredSequence> {
        let stmts = extract_stmts(expr);
        let mut nested = Lowerer {
            bindings: self.bindings.clone(),
            statements: Vec::new(),
            op_metadata: Vec::new(),
            next_reg: self.next_reg,
            next_label: self.next_label,
            config: self.config,
            call_policies: self.call_policies.clone(),
            inference_failure_mode: self.inference_failure_mode,
            auto_calls: self.auto_calls,
            inline_liveness_prebuild: Vec::new(),
            dispatch_tainted_reason: None,
            opcode_var_name: self.opcode_var_name.clone(),
            in_dispatch_arm_body: self.in_dispatch_arm_body,
            dispatch_loop_label: self.dispatch_loop_label.clone(),
            pc_pinned: self.pc_pinned,
        };

        for stmt in &stmts {
            nested.lower_stmt(stmt)?;
        }

        self.next_reg = self.next_reg.max(nested.next_reg);
        self.next_label = self.next_label.max(nested.next_label);
        Some(LoweredSequence::new(nested.statements, nested.op_metadata))
    }

    pub(super) fn lower_branch_value_expr(
        &mut self,
        expr: &Expr,
    ) -> Option<(LoweredSequence, Binding)> {
        let mut nested = Lowerer {
            bindings: self.bindings.clone(),
            statements: Vec::new(),
            op_metadata: Vec::new(),
            next_reg: self.next_reg,
            next_label: self.next_label,
            config: self.config,
            call_policies: self.call_policies.clone(),
            inference_failure_mode: self.inference_failure_mode,
            auto_calls: self.auto_calls,
            inline_liveness_prebuild: Vec::new(),
            dispatch_tainted_reason: None,
            opcode_var_name: self.opcode_var_name.clone(),
            in_dispatch_arm_body: self.in_dispatch_arm_body,
            dispatch_loop_label: self.dispatch_loop_label.clone(),
            pc_pinned: self.pc_pinned,
        };

        let binding = nested.lower_scoped_value_expr(expr)?;
        self.next_reg = self.next_reg.max(nested.next_reg);
        self.next_label = self.next_label.max(nested.next_label);
        Some((
            LoweredSequence::new(nested.statements, nested.op_metadata),
            binding,
        ))
    }

    fn lower_scoped_value_expr(&mut self, expr: &Expr) -> Option<Binding> {
        match expr {
            Expr::Block(block) => self.lower_block_value(&block.block),
            _ => self.lower_value_expr(expr),
        }
    }

    pub(super) fn lower_block_value(&mut self, block: &Block) -> Option<Binding> {
        let (tail, prefix) = block.stmts.split_last()?;

        for stmt in prefix {
            self.lower_stmt(stmt)?;
        }

        match tail {
            Stmt::Expr(expr, None) => self.lower_value_expr(expr),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(reg: u16, kind: BindingKind) -> Binding {
        Binding {
            reg,
            kind,
            depends_on_stack: false,
            struct_type: None,
        }
    }

    #[test]
    fn struct_literal_lowers_to_new_plus_setfield_gc() {
        // `Node { value: v, next: n }` with v:Int, n:Ref lowers to a New
        // allocation followed by a per-field setfield_gc_{i,r}.
        let mut lowerer = Lowerer::new(None);
        lowerer
            .bindings
            .insert("v".to_string(), binding(1, BindingKind::Int));
        lowerer
            .bindings
            .insert("n".to_string(), binding(2, BindingKind::Ref));
        let expr: Expr =
            syn::parse_str("Node { value: v, next: n }").expect("parse struct literal");

        let result = lowerer.lower_value_expr(&expr).expect("struct lowers");
        assert!(matches!(result.kind, BindingKind::Ref));

        let kinds: Vec<_> = lowerer.op_metadata.iter().map(|m| m.kind).collect();
        assert_eq!(
            kinds,
            vec![OpKind::New, OpKind::SetfieldGc, OpKind::SetfieldGc]
        );
        // The int field reads the int bank, the ref field the ref bank.
        assert_eq!(
            lowerer.op_metadata[1].reads,
            vec![Register::ref_(result.reg), Register::int(1)]
        );
        assert_eq!(
            lowerer.op_metadata[2].reads,
            vec![Register::ref_(result.reg), Register::ref_(2)]
        );
        let emitted = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(emitted.contains("new_struct"));
        assert!(emitted.contains("setfield_gc_i"));
        assert!(emitted.contains("setfield_gc_r"));
        assert!(emitted.contains("offset_of"));
        assert!(emitted.contains("size_of"));
    }

    #[test]
    fn raw_store_intrinsic_lowers_to_raw_store_i() {
        // `majit_raw_store_i64(base, ea, val)` lowers to a single RawStore
        // op reading the three int operands (base, ea, val) and emitting a
        // `raw_store_i` with an 8-byte raw-int array descr.
        let mut lowerer = Lowerer::new(None);
        lowerer
            .bindings
            .insert("base".to_string(), binding(3, BindingKind::Int));
        lowerer
            .bindings
            .insert("ea".to_string(), binding(4, BindingKind::Int));
        lowerer
            .bindings
            .insert("val".to_string(), binding(5, BindingKind::Int));
        let stmt: syn::Stmt =
            syn::parse_str("majit_raw_store_i64(base, ea, val);").expect("parse raw store");

        lowerer.lower_stmt(&stmt).expect("raw store lowers");

        let kinds: Vec<_> = lowerer.op_metadata.iter().map(|m| m.kind).collect();
        assert_eq!(kinds, vec![OpKind::RawStore]);
        assert_eq!(
            lowerer.op_metadata[0].reads,
            vec![Register::int(3), Register::int(4), Register::int(5)]
        );
        assert!(lowerer.op_metadata[0].writes.is_empty());
        let emitted = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(emitted.contains("raw_store_i"));
        assert!(emitted.contains("add_raw_int_array_descr"));
    }

    #[test]
    fn record_exact_class_statement_uses_ref_int_argcodes() {
        let mut lowerer = Lowerer::new(None);
        lowerer
            .bindings
            .insert("value".to_string(), binding(1, BindingKind::Ref));
        lowerer
            .bindings
            .insert("cls".to_string(), binding(2, BindingKind::Int));
        let expr: Expr =
            syn::parse_str("jit::record_exact_class(value, cls)").expect("parse hint call");

        assert_eq!(lowerer.lower_record_exact_class_stmt(&expr), Some(()));
        assert_eq!(lowerer.op_metadata.len(), 1);
        assert_eq!(lowerer.op_metadata[0].kind, OpKind::RecordExactClass);
        assert_eq!(
            lowerer.op_metadata[0].reads,
            vec![Register::ref_(1), Register::int(2)]
        );
        let emitted = lowerer
            .statements
            .iter()
            .map(ToString::to_string)
            .collect::<String>();
        assert!(emitted.contains("record_exact_class"));
    }

    #[test]
    fn record_exact_class_statement_rejects_ref_class_operand() {
        let mut lowerer = Lowerer::new(None);
        lowerer
            .bindings
            .insert("value".to_string(), binding(1, BindingKind::Ref));
        lowerer
            .bindings
            .insert("cls".to_string(), binding(2, BindingKind::Ref));
        let expr: Expr =
            syn::parse_str("jit::record_exact_class(value, cls)").expect("parse hint call");

        assert_eq!(lowerer.lower_record_exact_class_stmt(&expr), None);
        assert!(lowerer.op_metadata.is_empty());
        assert!(lowerer.statements.is_empty());
    }
}
