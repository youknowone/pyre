use super::lower_value::struct_type_id;
use super::*;

impl<'c> Lowerer<'c> {
    /// If `func` is a registered `residual_writes` mutator, return the
    /// `struct_field_write_effect_info(...)` expression naming the written
    /// field, so the residual call records a write-set `EffectInfo` that
    /// invalidates the cached `getfield_gc_i` on that field.  `None` for a
    /// plain residual call (empty write-set).  `can_raise` carries the
    /// call policy's extra-effect into the write-set `EffectInfo` so a
    /// `ResidualVoidCannotRaise` mutator keeps `CannotRaise` instead of
    /// silently widening to `CanRaise`.
    pub(super) fn residual_write_effect_info_tokens(
        &self,
        func: &Expr,
        can_raise: bool,
    ) -> Option<TokenStream> {
        let config = self.config?;
        let func_segments = canonical_expr_segments(func)?;
        let (_, struct_path, field) = config
            .residual_writes
            .iter()
            .find(|(segments, _, _)| *segments == func_segments)?;
        // Raw host-owned struct (the ref-scalar's pointee, no GC header) →
        // `is_gc_managed = false`, the same id the getfield/setfield lowering
        // uses so this write-EI rebuilds the SAME parent SizeDescr identity.
        let tid = struct_type_id(struct_path, false);
        Some(quote! {
            // The residual mutates a host-owned native struct field (no
            // GC header) → `is_gc_managed = false`, matching the
            // getfield/setfield lowering so the write-EI rebuilds the
            // SAME parent SizeDescr identity the getfield reads back.
            majit_metainterp::struct_field_write_effect_info(
                ::core::mem::size_of::<#struct_path>(),
                #tid,
                false,
                &[(
                    ::core::mem::offset_of!(#struct_path, #field),
                    false,
                    stringify!(#field),
                )],
                stringify!(#field),
                #can_raise,
            )
        })
    }

    pub(super) fn lower_stmt(&mut self, stmt: &Stmt) -> Option<()> {
        match stmt {
            Stmt::Local(local) => {
                if let Some(()) = self.lower_local(local) {
                    return Some(());
                }
                self.lower_stmt_fallback(stmt, "local")
            }
            Stmt::Expr(expr, _) => {
                if matches!(expr, Expr::Continue(_)) {
                    if let Some(label) = self.dispatch_loop_label.clone() {
                        self.emit_jump(&label);
                    }
                    return Some(());
                }
                if let Some(()) = self.lower_expr_stmt(expr) {
                    return Some(());
                }
                self.lower_stmt_fallback(stmt, "expr")
            }
            Stmt::Macro(stmt_macro) => {
                // jtransform.py:1714-1723 handle_jit_marker__loop_header —
                // a `can_enter_jit!()` call at the user's source-level
                // back-edge (interp_jit.py:118 inside `jump_absolute`'s
                // backward-jump branch) lowers to `loop_header(jd.index)`
                // at the SAME source position.  Per-arm emission at the
                // dispatch JitCode level (post-INLINE_CALL) would over-
                // emit on every arm execution including forward-jump
                // path; emitting at the call site inside the arm body
                // sub-JitCode makes the LH op execute only when control
                // reaches the conditional that contains can_enter_jit!.
                //
                // Only fire when this Lowerer is producing the dispatch
                // arm body sub-JitCode (where the surrounding
                // `__dispatch_jitcode_<fn>` provides `__jdindex` in
                // scope).  For the per-arm trace JitCode path (whose
                // surrounding fn has no `__jdindex`) the recognition
                // falls through to `None` and the body lowering aborts
                // — pyre's per-arm trace JitCode is a
                // TODO: not present in RPython, so
                // omitting `loop_header` there is consistent with
                // upstream's single-JitCode model.
                // Allow can_enter_jit! in the dispatch JitCode body
                // (both at the dispatch level and inside arm sub-JitCodes).
                // __jdindex is in scope in both contexts.
                let path_str = stmt_macro
                    .mac
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect::<Vec<_>>()
                    .join("::");
                if path_str == "can_enter_jit"
                    || path_str.ends_with("::can_enter_jit")
                    || path_str == "jit_loop_header"
                    || path_str.ends_with("::jit_loop_header")
                {
                    self.emit_op(
                        OpMeta::linear(OpKind::LoopHeader, vec![], vec![]),
                        quote! {
                            // jtransform.py:1716 c_index = Constant(jd.index, ...);
                            // __jdindex is the runtime parameter of the
                            // enclosing `__dispatch_jitcode_<fn>(__asm,
                            // __jdindex: i64)` and remains in scope of
                            // the arm body sub-builder block.
                            __builder.loop_header(__jdindex);
                        },
                    );
                    return Some(());
                }
                None
            }
            Stmt::Item(_) => None,
        }
    }

    /// Last resort for a statement no lowering arm accepted, in the
    /// state-field dispatch body (`config` present).
    ///
    /// A statement with no observable effect — it neither writes jit
    /// state nor touches storage/heap/user locals — is dropped from the
    /// jitcode (e.g. `pc += 1`: the dispatch loop manages the pc
    /// register itself).
    ///
    /// Anything else (state-field writes, residual side effects,
    /// bindings later statements may observe) cannot be expressed and
    /// fails as unsupported (`None`). The codewriter lowers a graph op
    /// exactly or rejects it — `jtransform.py` `rewrite_operation` raises
    /// for operations it cannot transform — rather than emitting a
    /// runtime abort for part of a body. Returning `None` lets the caller
    /// degrade cleanly: `try_inline_dispatch_arm` rolls back the partial
    /// emission and the sub-JitCode entry returns `None` too, so the arm
    /// runs in the interpreter instead of compiling to a trace that
    /// aborts mid-record.
    fn lower_stmt_fallback(&mut self, stmt: &Stmt, what: &str) -> Option<()> {
        if self.config.is_none() {
            return None;
        }
        let inert = !self.stmt_modifies_jit_state(stmt) && !self.stmt_touches_storage(stmt);
        if inert {
            if std::env::var_os("MAJIT_MACRO_DEBUG").is_some() {
                eprintln!(
                    "[majit-macro] lower_stmt silent-skip ({what}): {}",
                    quote!(#stmt)
                );
            }
            return Some(());
        }
        if std::env::var_os("MAJIT_MACRO_DEBUG").is_some() {
            eprintln!(
                "[majit-macro] lower_stmt unsupported ({what}): {}",
                quote!(#stmt)
            );
        }
        None
    }

    pub(super) fn lower_local(&mut self, local: &Local) -> Option<()> {
        let Pat::Ident(pat_ident) = &local.pat else {
            return None;
        };
        let init = local.init.as_ref()?;

        // Try normal lowering
        if let Some(binding) = self.lower_value_expr(&init.expr) {
            // When a stack pop is lowered to a JitCode register, also emit a
            // Rust `let` binding so that subsequent un-lowered code (e.g.,
            // complex expressions referencing the variable) can still compile.
            // The value is 0 — only the JitCode register carries the real
            // runtime value, but this prevents "cannot find value" errors.
            if binding.depends_on_stack {
                let ident = &pat_ident.ident;
                self.emit_aux(quote! { let #ident: i64 = 0; });
            }
            self.bindings.insert(pat_ident.ident.to_string(), binding);
            return Some(());
        }

        // Config-aware: runtime constant (expression not touching storage).
        //
        // Slice ε.3 fail-closed: ALSO refuse this fallback when the init
        // expression references any name already bound in `self.bindings`.
        // The fallback emits the original `let X = <init_expr>;` line as
        // verbatim Rust into the surrounding `__builder` block scope, then
        // a `__builder.load_const_i_value(reg, X as i64)`.  That contract
        // assumes `init_expr` is a true compile-time constant whose
        // identifiers (if any) are Rust types / `const` items / module
        // paths — NOT JIT-level bindings (`program` Ref / `pc` Int /
        // arm-pattern bound names) which are not in scope at the
        // surrounding Rust scope.  Without this guard, dispatch arm
        // sub-JitCode bodies that contain unrecognised method calls on
        // a parent binding (e.g. aheui-jit's `program.get_operand(pc - 1)`
        // when no `Program::get_operand` call policy is registered) would
        // emit verbatim Rust referencing `program`/`pc` in the
        // `__sub_builder` block — failing to compile.  Returning `None`
        // here triggers the dispatch arm's `None` branch which substitutes
        // an `abort_permanent()` sub-JitCode (see `lower_dispatch_chain`).
        if self.config.is_some()
            && !self.expr_touches_storage(&init.expr)
            && !self.expr_references_any_binding(&init.expr)
        {
            let reg = self.alloc_reg();
            let ident = &pat_ident.ident;
            let init_expr = &init.expr;
            self.emit_op(
                OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(reg)]),
                quote! {
                    let #ident = #init_expr;
                    __builder.load_const_i_value(#reg, #ident as i64);
                },
            );
            self.bindings.insert(
                ident.to_string(),
                Binding {
                    reg,
                    kind: BindingKind::Int,
                    depends_on_stack: false,
                    struct_type: None,
                },
            );
            return Some(());
        }

        None
    }

    /// Walk `expr` and return `true` if any single-segment `Expr::Path`
    /// references a name bound in `self.bindings`, **excluding** names
    /// shadowed by an inner `let` in the same expression.  Mirrors the
    /// recognition core of `collect_arm_caller_locals` but stops at the
    /// first match — used as a fail-closed gate inside `lower_local`'s
    /// runtime-constant fallback.
    ///
    /// Scope tracking: PyPy `flowspace` produces distinct flowgraph
    /// variables per lexical scope.  Pyre's probe approximates this by
    /// pushing a fresh scope frame on entering an `ExprBlock` and
    /// popping on exit; `let X = ...` inside the block adds X to the
    /// innermost frame.  An ident is "locally bound" if any frame in
    /// the stack contains it, so the inner `let pc = 42; pc + 1` shape
    /// correctly suppresses the outer `pc` parent-binding match.
    fn expr_references_any_binding(&self, expr: &Expr) -> bool {
        use syn::visit::Visit;
        struct BindingProbe<'a> {
            bindings: &'a HashMap<String, Binding>,
            hit: bool,
            /// Stack of per-block local-binding sets (innermost on top).
            scope_stack: Vec<HashSet<String>>,
        }
        impl BindingProbe<'_> {
            fn is_locally_bound(&self, name: &str) -> bool {
                self.scope_stack.iter().any(|s| s.contains(name))
            }
        }
        impl<'ast> Visit<'ast> for BindingProbe<'_> {
            fn visit_expr_path(&mut self, p: &'ast ExprPath) {
                if self.hit || p.qself.is_some() || p.path.segments.len() != 1 {
                    return;
                }
                let seg = &p.path.segments[0];
                if !seg.arguments.is_none() {
                    return;
                }
                let name = seg.ident.to_string();
                if self.is_locally_bound(&name) {
                    return;
                }
                if self.bindings.contains_key(&name) {
                    self.hit = true;
                }
            }
            fn visit_expr_field(&mut self, ef: &'ast syn::ExprField) {
                self.visit_expr(&ef.base);
            }
            fn visit_expr_method_call(&mut self, mc: &'ast ExprMethodCall) {
                self.visit_expr(&mc.receiver);
                for arg in &mc.args {
                    self.visit_expr(arg);
                }
            }
            fn visit_block(&mut self, b: &'ast Block) {
                // Cover every `Block` traversal — explicit `{ ... }`
                // (`ExprBlock`'s default impl forwards here), if/else
                // branches (`ExprIf::then_branch` / `else_branch`),
                // while / loop / for bodies — not just the explicit
                // block expression form.  Each lexical block pushes a
                // fresh scope frame so inner `let X = ...` shadows the
                // parent binding inside that block only.
                self.scope_stack.push(HashSet::new());
                for stmt in &b.stmts {
                    self.visit_stmt(stmt);
                    if self.hit {
                        break;
                    }
                }
                self.scope_stack.pop();
            }
            fn visit_expr_match(&mut self, em: &'ast ExprMatch) {
                self.visit_expr(&em.expr);
                for arm in &em.arms {
                    if self.hit {
                        break;
                    }
                    // Each match arm introduces a scope: pattern-bound
                    // names shadow outer bindings inside the arm body.
                    // Mirrors `flowspace`'s SpaceOperation scope per
                    // match arm.
                    let mut arm_scope = HashSet::new();
                    collect_pat_bound_idents(&arm.pat, &mut arm_scope);
                    self.scope_stack.push(arm_scope);
                    if let Some((_, guard)) = &arm.guard {
                        self.visit_expr(guard);
                    }
                    self.visit_expr(&arm.body);
                    self.scope_stack.pop();
                }
            }
            fn visit_local(&mut self, local: &'ast Local) {
                // Visit init RHS BEFORE adding the bound name so the
                // init expression's references are still probed against
                // outer scope (`let X = X + 1` at scope entry uses the
                // outer X for the RHS).
                if let Some(init) = &local.init {
                    self.visit_expr(&init.expr);
                    if let Some((_, diverge)) = &init.diverge {
                        self.visit_expr(diverge);
                    }
                }
                // All bindings produced by the pattern enter the
                // innermost scope frame.  `let (a, b) = ...;`,
                // `let Foo { x } = ...;`, `let A(y) | B(y) = ...;`
                // — each pattern shape contributes its bound names.
                // Mirrors `flowspace`'s SpaceOperation per
                // pattern-extraction step.
                if let Some(top) = self.scope_stack.last_mut() {
                    collect_pat_bound_idents(&local.pat, top);
                }
            }
        }
        let mut probe = BindingProbe {
            bindings: &self.bindings,
            hit: false,
            // Seed with one root frame so `visit_local` inside the
            // top-level expression (no enclosing block) can still
            // record bindings.
            scope_stack: vec![HashSet::new()],
        };
        probe.visit_expr(expr);
        probe.hit
    }

    /// RPython jtransform.py:923 `_rewrite_op_setfield` for virtualizable.
    ///
    /// Recognizes `frame.field_name = value` and emits vable_setfield JitCode.
    pub(super) fn lower_conditional_call(&mut self, expr: &Expr) -> Option<()> {
        let mac = match expr {
            Expr::Macro(m) => m,
            _ => return None,
        };
        let name = mac.mac.path.segments.last()?.ident.to_string();
        if name != "conditional_call" {
            return None;
        }
        let args: syn::punctuated::Punctuated<Expr, syn::Token![,]> = mac
            .mac
            .parse_body_with(syn::punctuated::Punctuated::parse_terminated)
            .ok()?;
        let args: Vec<&Expr> = args.iter().collect();
        if args.len() < 2 {
            return None;
        }
        // args[0] = condition, args[1] = func path, args[2..] = function arguments
        let func_args = &args[2..];
        // jtransform.py:1666-1672: no floats, no more than 4 function args
        if func_args.len() > 4 {
            panic!("conditional_call does not support more than 4 arguments");
        }
        let cond_binding = self.lower_value_expr(args[0])?;
        let cond_reg = cond_binding.reg;
        // RPython make_three_lists: tag each arg with its kind (int/ref).
        let mut typed_arg_tokens = Vec::new();
        // cond_reg is Int per the conditional_call argcode prefix.
        let mut arg_regs: Vec<Register> = vec![Register::int(cond_reg)];
        for arg in func_args {
            let b = self.lower_value_expr(arg)?;
            let reg = b.reg;
            arg_regs.push(Register::from_binding(&b));
            let token = match b.kind {
                // jtransform.py:1668: float → raise Exception
                BindingKind::Float => {
                    panic!("Conditional call does not support floats");
                }
                BindingKind::Ref => {
                    quote! { majit_metainterp::jitcode::JitCallArg::reference(#reg) }
                }
                BindingKind::Int => quote! { majit_metainterp::jitcode::JitCallArg::int(#reg) },
            };
            typed_arg_tokens.push(token);
        }
        let func_path = args[1];
        // `conditional_call!` always lowers to a void residual_call.
        // Default to `ResidualVoidWrapped` for `Infer` so the
        // analyzer-absent CanRaise slot is the lowering's static slot;
        // the runtime helper-policy lookup overrides this for callees
        // whose flavor turns out otherwise.
        let (policy, is_inferred) = self.cond_call_policy_or_inferred_default(
            func_path,
            "conditional_call!",
            crate::jit_interp::CallPolicyKind::ResidualVoidWrapped,
        );
        let Some(result_kind) = call_policy_result_kind(policy) else {
            panic!("conditional_call! helper policy {policy:?} has no direct-call result kind");
        };
        if result_kind != CallResultKind::Void {
            panic!("conditional_call! requires a void-return helper policy, got {policy:?}");
        }
        let slot = self.cond_call_slot_for_policy(policy, "conditional_call!");
        // `call.py:249-251 getcalldescr`:
        //   if loopinvariant:
        //       assert not NON_VOID_ARGS, ("arguments not supported for "
        //                                  "loop-invariant function!")
        // The canonical `call_loopinvariant_*_canonical_via_target`
        // builders enforce the same invariant via `arg_regs.is_empty()`
        // (`jitcode/assembler.rs:1849`), but the cond_call helper
        // dispatch routes through `conditional_call_ir_v_typed_args`
        // which doesn't share that assert. Mirror the check here so
        // a `conditional_call!(cond, loop_invariant_helper, arg)`
        // panics at expansion time instead of silently registering a
        // bytecode shape RPython would reject at calldescr build.  In
        // `Infer` mode the slot is decided at runtime from `__policy`,
        // so the static check only fires when the macro-time default
        // resolves to LoopInvariant — explicit policy paths preserve
        // the original eager assert.
        if !is_inferred
            && matches!(slot, CondCallEffectSlot::LoopInvariant)
            && !func_args.is_empty()
        {
            panic!(
                "conditional_call!: arguments not supported for loop-invariant function (policy {policy:?})",
            );
        }
        let inferred_policy_check = if is_inferred {
            Some(inferred_conditional_call_policy_check(func_args.is_empty()))
        } else {
            None
        };
        let register_target = self.call_target_registration_tokens(
            func_path,
            policy,
            slot,
            is_inferred,
            inferred_policy_check,
        );
        self.emit_op(
            OpMeta::linear(OpKind::Call, arg_regs, vec![]),
            quote! {
                #register_target
                __builder.conditional_call_ir_v_typed_args(__fn_idx, #cond_reg, &[#(#typed_arg_tokens),*]);
            },
        );
        // `jtransform.py:1681-1683`: append `-live-` exactly when
        // `calldescr_canraise(calldescr)` for the selected calldescr.
        // In inferred mode the physical BC_LIVE is guarded by the same
        // helper-policy byte that selects the calldescr slot, preserving
        // PyPy's cannot-raise / loop-invariant no-marker shape.
        if is_inferred {
            let condition = inferred_policy_live_condition(func_path, &[1]);
            self.emit_op(
                OpMeta::live_marker_if(condition),
                quote! { let _ = __builder.live_placeholder(); },
            );
        } else if slot.can_raise() {
            self.emit_op(
                OpMeta::live_marker(),
                quote! { let _ = __builder.live_placeholder(); },
            );
        }
        Some(())
    }

    /// RPython jtransform.py:1687 — `rewrite_op_jit_conditional_call_value`.
    ///
    /// Recognizes `conditional_call_elidable!(value, func, args...)` and emits
    /// the canonical `conditional_call_value_ir_{i,r}` builder entrypoint.
    pub(super) fn lower_conditional_call_elidable(&mut self, expr: &Expr) -> Option<Binding> {
        let mac = match expr {
            Expr::Macro(m) => m,
            _ => return None,
        };
        let name = mac.mac.path.segments.last()?.ident.to_string();
        if name != "conditional_call_elidable" {
            return None;
        }
        let args: syn::punctuated::Punctuated<Expr, syn::Token![,]> = mac
            .mac
            .parse_body_with(syn::punctuated::Punctuated::parse_terminated)
            .ok()?;
        let args: Vec<&Expr> = args.iter().collect();
        if args.len() < 2 {
            return None;
        }
        let func_args = &args[2..];
        // jtransform.py:1666-1672: no floats, no more than 4 function args
        if func_args.len() > 4 {
            panic!("Conditional call does not support more than 4 arguments");
        }
        let value_binding = self.lower_value_expr(args[0])?;
        let value_reg = value_binding.reg;
        // jtransform.py:1668: value itself must not be float
        if matches!(value_binding.kind, BindingKind::Float) {
            panic!("Conditional call does not support floats");
        }
        // RPython make_three_lists: tag each arg with its kind.
        let mut typed_arg_tokens = Vec::new();
        // value_reg is Int or Ref per the conditional_call_value_ir_{i|r} arm.
        let value_kind = value_binding.kind;
        let mut arg_regs: Vec<Register> = vec![Register::new(value_kind, value_reg)];
        for arg in func_args {
            let b = self.lower_value_expr(arg)?;
            let reg = b.reg;
            arg_regs.push(Register::from_binding(&b));
            let token = match b.kind {
                BindingKind::Float => {
                    panic!("Conditional call does not support floats");
                }
                BindingKind::Ref => {
                    quote! { majit_metainterp::jitcode::JitCallArg::reference(#reg) }
                }
                BindingKind::Int => quote! { majit_metainterp::jitcode::JitCallArg::int(#reg) },
            };
            typed_arg_tokens.push(token);
        }
        let func_path = args[1];
        let result_reg = self.alloc_reg();
        // RPython jtransform.py:1687 — conditional_call_value_ir_{i|r}
        let builder_call = match value_kind {
            BindingKind::Ref => quote! {
                __builder.conditional_call_value_ir_r_typed_args(__fn_idx, #value_reg, &[#(#typed_arg_tokens),*], #result_reg);
            },
            _ => quote! {
                __builder.conditional_call_value_ir_i_typed_args(__fn_idx, #value_reg, &[#(#typed_arg_tokens),*], #result_reg);
            },
        };
        // `conditional_call_elidable!` is the elidable cache helper; per
        // `rlib/jit.py:1334-1336` the callee need not be `@elidable` but
        // the cond_call_value op itself caches the result.  Default to
        // `Elidable*Wrapped` based on the leading value-kind so an
        // inferred policy still classifies as elidable.
        let inferred_default = match value_kind {
            BindingKind::Ref => crate::jit_interp::CallPolicyKind::ElidableRefWrapped,
            BindingKind::Float => crate::jit_interp::CallPolicyKind::ElidableFloatWrapped,
            BindingKind::Int => crate::jit_interp::CallPolicyKind::ElidableIntWrapped,
        };
        let (policy, is_inferred) = self.cond_call_policy_or_inferred_default(
            func_path,
            "conditional_call_elidable!",
            inferred_default,
        );
        let Some(result_kind) = call_policy_result_kind(policy) else {
            panic!(
                "conditional_call_elidable! helper policy {policy:?} has no direct-call result kind"
            );
        };
        if !call_result_matches_binding(result_kind, value_kind) {
            panic!(
                "conditional_call_elidable! value/result kind mismatch for helper policy {policy:?}"
            );
        }
        let slot = self.cond_call_slot_for_policy(policy, "conditional_call_elidable!");
        // `call.py:249-251 getcalldescr`'s loop-invariant non-void-args
        // assert (see plain `conditional_call!` lowerer for the citation).
        // `conditional_call_elidable!` accepts non-elidable cache-computing
        // helpers per `rlib/jit.py:1334-1336`, so a `LoopInvariant` slot is
        // legal in principle and must enforce the same args-empty rule.
        // Static check applies only to explicit-policy paths; `Infer`
        // resolves slot at runtime from the `__policy` byte.
        if !is_inferred
            && matches!(slot, CondCallEffectSlot::LoopInvariant)
            && !func_args.is_empty()
        {
            panic!(
                "conditional_call_elidable!: arguments not supported for loop-invariant function (policy {policy:?})",
            );
        }
        let inferred_policy_check = if is_inferred {
            Some(inferred_conditional_call_value_policy_check(
                value_kind,
                func_args.is_empty(),
            ))
        } else {
            None
        };
        let register_target = self.call_target_registration_tokens(
            func_path,
            policy,
            slot,
            is_inferred,
            inferred_policy_check,
        );
        self.emit_op(
            OpMeta::linear(
                OpKind::Call,
                arg_regs,
                vec![Register::new(value_kind, result_reg)],
            ),
            quote! {
                #register_target
                #builder_call
            },
        );
        // `jtransform.py:1681-1683`: append `-live-` exactly when
        // `calldescr_canraise(calldescr)`.  `conditional_call_elidable`
        // still accepts non-elidable cache-computing helpers per
        // `rlib/jit.py:1334-1336`; their explicit policy maps to
        // `EffectInfoSlot::CanRaise` and therefore keeps the marker.
        // `Infer` resolves slot at runtime; guard the physical marker with
        // the same can-raise policy cases instead of emitting a redundant
        // PyPy-invisible marker.
        if is_inferred {
            let can_raise_codes: &[u8] = match value_kind {
                BindingKind::Int => &[INT_DONT_LOOK_INSIDE, INT_ELIDABLE, INT_ELIDABLE_OR_MEMERROR],
                BindingKind::Ref => &[REF_ELIDABLE, REF_ELIDABLE_OR_MEMERROR, REF_DONT_LOOK_INSIDE],
                BindingKind::Float => &[],
            };
            self.emit_op(
                OpMeta::live_marker_if(inferred_policy_live_condition(func_path, can_raise_codes)),
                quote! { let _ = __builder.live_placeholder(); },
            );
        } else if slot.can_raise() {
            self.emit_op(
                OpMeta::live_marker(),
                quote! { let _ = __builder.live_placeholder(); },
            );
        }
        Some(Binding {
            reg: result_reg,
            kind: value_kind,
            depends_on_stack: false,
            struct_type: None,
        })
    }

    /// RPython jtransform.py:522 `handle_recursive_call` — recognises
    /// `recursive_portal_call!(driver, green0, green1, ...)` and emits the
    /// `recursive_call_int` opcode (pyjitpl.py:1376 `opimpl_recursive_call`
    /// → BC_RECURSIVE_CALL_INT).
    ///
    /// The greens are lowered in jitdriver declaration order and each is
    /// tagged with its register-bank kind, so the dispatcher reads it from
    /// the matching bank (an int `pc` green from the int bank, a ref
    /// `program` green from the ref bank) and hashes the key against
    /// `green_args_spec`.  The callee's fresh reds are built by the
    /// dispatcher from `recursive_fresh_entry_reds`, so no caller→callee
    /// argument moves are recorded here (`args = &[]`).  `jd_index` is the
    /// runtime `__jdindex` parameter of `__dispatch_jitcode_*`
    /// (jtransform.py:1704 `portal_jd.index`, threaded exactly like
    /// `jit_merge_point` / `loop_header`), not a literal 0: when a consumer
    /// also installs the propagate-descr placeholder driver via
    /// `ensure_default_driver_sd`, the real portal driver registers at slot
    /// 1+, so a hardcoded 0 would read the placeholder's empty
    /// `green_args_spec`.
    pub(super) fn lower_recursive_portal_call(&mut self, expr: &Expr) -> Option<Binding> {
        let mac = match expr {
            Expr::Macro(m) => m,
            _ => return None,
        };
        let name = mac.mac.path.segments.last()?.ident.to_string();
        if name != "recursive_portal_call" {
            return None;
        }
        let args: syn::punctuated::Punctuated<Expr, syn::Token![,]> = mac
            .mac
            .parse_body_with(syn::punctuated::Punctuated::parse_terminated)
            .ok()?;
        let args: Vec<&Expr> = args.iter().collect();
        // `driver`, then one expression per green in declaration order.
        if args.len() < 2 {
            panic!("recursive_portal_call! requires a driver and at least one green");
        }
        let green_exprs = &args[1..];
        let mut green_bindings = Vec::with_capacity(green_exprs.len());
        for g in green_exprs {
            green_bindings.push(self.lower_value_expr(g)?);
        }
        let green_reads: Vec<Register> =
            green_bindings.iter().map(Register::from_binding).collect();
        let green_tokens: Vec<proc_macro2::TokenStream> = green_bindings
            .iter()
            .map(|b| {
                let reg = b.reg;
                let kind = match b.kind {
                    BindingKind::Int => quote! { majit_metainterp::jitcode::JitArgKind::Int },
                    BindingKind::Ref => quote! { majit_metainterp::jitcode::JitArgKind::Ref },
                    BindingKind::Float => quote! { majit_metainterp::jitcode::JitArgKind::Float },
                };
                quote! { (#kind, #reg) }
            })
            .collect();
        let result_reg = self.alloc_reg();
        self.emit_op(
            OpMeta::linear(OpKind::Call, green_reads, vec![Register::int(result_reg)]),
            quote! {
                __builder.recursive_call_int(__jdindex as u16, #result_reg, &[#(#green_tokens),*], &[]);
            },
        );
        // jtransform.py:533 — `recursive_call_*` is always followed by `-live-`.
        self.emit_op(
            OpMeta::live_marker(),
            quote! { let _ = __builder.live_placeholder(); },
        );
        Some(Binding {
            reg: result_reg,
            kind: BindingKind::Int,
            depends_on_stack: false,
            struct_type: None,
        })
    }

    /// RPython jtransform.py:292-313 — `rewrite_op_jit_record_known_result`.
    ///
    /// Recognizes `record_known_result!(result, func, args...)` and emits
    /// the canonical `record_known_result_{i,r}_ir_v` builder entrypoint.
    pub(super) fn lower_record_known_result(&mut self, expr: &Expr) -> Option<()> {
        let mac = match expr {
            Expr::Macro(m) => m,
            _ => return None,
        };
        let name = mac.mac.path.segments.last()?.ident.to_string();
        if name != "record_known_result" {
            return None;
        }
        let args: syn::punctuated::Punctuated<Expr, syn::Token![,]> = mac
            .mac
            .parse_body_with(syn::punctuated::Punctuated::parse_terminated)
            .ok()?;
        let args: Vec<&Expr> = args.iter().collect();
        if args.len() < 2 {
            return None;
        }
        // args[0] = known result, args[1] = func path, args[2..] = function arguments
        let result_binding = self.lower_value_expr(args[0])?;
        let result_reg = result_binding.reg;
        // jtransform.py:293-295: float → raise Exception
        if matches!(result_binding.kind, BindingKind::Float) {
            panic!("record_known_result does not support floats");
        }
        // RPython make_three_lists: tag each arg with its kind.
        let mut typed_arg_tokens = Vec::new();
        let mut arg_regs: Vec<Register> = Vec::new();
        for arg in &args[2..] {
            let b = self.lower_value_expr(arg)?;
            let reg = b.reg;
            arg_regs.push(Register::from_binding(&b));
            let token = match b.kind {
                BindingKind::Float => {
                    panic!("record_known_result does not support floats");
                }
                BindingKind::Ref => {
                    quote! { majit_metainterp::jitcode::JitCallArg::reference(#reg) }
                }
                BindingKind::Int => quote! { majit_metainterp::jitcode::JitCallArg::int(#reg) },
            };
            typed_arg_tokens.push(token);
        }
        let func_path = args[1];
        // RPython jtransform.py:302-307 — record_known_result_{i|r}
        let builder_call = match result_binding.kind {
            BindingKind::Ref => quote! {
                __builder.record_known_result_r_ir_v_typed_args(__fn_idx, #result_reg, &[#(#typed_arg_tokens),*]);
            },
            _ => quote! {
                __builder.record_known_result_i_ir_v_typed_args(__fn_idx, #result_reg, &[#(#typed_arg_tokens),*]);
            },
        };
        // RPython pyjitpl.py:413-419 passes the known result box as
        // `prepend_box=resbox`; record_known_result reads that box and
        // produces no result (`_v` suffix).
        // `record_known_result!` requires an elidable callee — the
        // `slot.is_elidable()` assert below catches non-elidable
        // policies.  Default `Infer` to `Elidable*Wrapped` so the
        // assert succeeds when the helper is registered through the
        // wrapped policy path.
        let inferred_default = match result_binding.kind {
            BindingKind::Ref => crate::jit_interp::CallPolicyKind::ElidableRefWrapped,
            BindingKind::Float => crate::jit_interp::CallPolicyKind::ElidableFloatWrapped,
            BindingKind::Int => crate::jit_interp::CallPolicyKind::ElidableIntWrapped,
        };
        let (policy, is_inferred) = self.cond_call_policy_or_inferred_default(
            func_path,
            "record_known_result!",
            inferred_default,
        );
        let Some(result_kind) = call_policy_result_kind(policy) else {
            panic!("record_known_result! helper policy {policy:?} has no direct-call result kind");
        };
        if !call_result_matches_binding(result_kind, result_binding.kind) {
            panic!("record_known_result! result kind mismatch for helper policy {policy:?}");
        }
        let slot = self.cond_call_slot_for_policy(policy, "record_known_result!");
        if !slot.is_elidable() {
            panic!("record_known_result! requires an elidable helper policy, got {policy:?}");
        }
        let inferred_policy_check = if is_inferred {
            Some(inferred_record_known_result_policy_check(
                result_binding.kind,
            ))
        } else {
            None
        };
        let register_target = self.call_target_registration_tokens(
            func_path,
            policy,
            slot,
            is_inferred,
            inferred_policy_check,
        );
        let result_typed = Register::new(result_binding.kind, result_reg);
        let mut reads = Vec::with_capacity(arg_regs.len() + 1);
        reads.push(result_typed);
        reads.extend(arg_regs);
        self.emit_op(
            OpMeta::linear(OpKind::RecordKnownResult, reads, Vec::new()),
            quote! {
                #register_target
                #builder_call
            },
        );
        // `jtransform.py:311-312`: append `-live-` exactly when the
        // elidable calldescr can raise.  In inferred mode, guard the
        // physical marker on the elidable-can-raise / memoryerror policy
        // bytes instead of emitting one for elidable_cannot_raise.
        if is_inferred {
            let can_raise_codes: &[u8] = match result_binding.kind {
                BindingKind::Int => &[INT_ELIDABLE, INT_ELIDABLE_OR_MEMERROR],
                BindingKind::Ref => &[REF_ELIDABLE, REF_ELIDABLE_OR_MEMERROR],
                BindingKind::Float => &[],
            };
            self.emit_op(
                OpMeta::live_marker_if(inferred_policy_live_condition(func_path, can_raise_codes)),
                quote! { let _ = __builder.live_placeholder(); },
            );
        } else if slot.can_raise() {
            self.emit_op(
                OpMeta::live_marker(),
                quote! { let _ = __builder.live_placeholder(); },
            );
        }
        Some(())
    }

    fn lower_expr_stmt(&mut self, expr: &Expr) -> Option<()> {
        // Green-pc inline dispatch: a `pc += N` / `pc = target` write inside
        // an inlined arm body must land in pc's pinned register (reg0), not
        // SSA-rebind the `pc` name (`lower_local_reassign`) nor be dropped as
        // non-jit-state.  Runs first so it intercepts the pc-write before the
        // generic reassign / drop path; a no-op when `!self.pc_pinned`.
        if let Some(()) = self.lower_pc_pinned_write(expr) {
            return Some(());
        }
        // jtransform.py:596 rewrite_op_hint — `hint(x, promote=True)` in
        // statement context.  Routes both `x = promote(arg)` (plain local
        // re-assignment, no state-write to trigger
        // `lower_state_field_write`'s RHS recursion) and bare
        // `promote(x);` through `lower_promote_call`, which emits the
        // `-live-` + `<kind>_guard_value` pair.  Without this site the
        // statement-form promote would silently no-op when the
        // config-aware fall-through later observes `stmt_modifies_jit_
        // state(stmt) == false`.
        if let Some(()) = self.lower_promote_stmt(expr) {
            return Some(());
        }
        // pyjitpl.py:385-391 opimpl_assert_not_none — statement-form
        // `jit::assert_not_none(x);` (discard return value).
        if let Some(()) = self.lower_assert_not_none_stmt(expr) {
            return Some(());
        }
        // pyjitpl.py:393-410 opimpl_record_exact_class — statement-form
        // `jit::record_exact_class(value, cls);` (no return value).
        if let Some(()) = self.lower_record_exact_class_stmt(expr) {
            return Some(());
        }
        // State field writes (register/tape machines).
        if let Some(()) = self.lower_state_field_update(expr) {
            return Some(());
        }
        if let Some(()) = self.lower_state_field_write(expr) {
            return Some(());
        }
        // Field write-through a `ref(T)` state scalar:
        // `state.<ref>.<member> = <int expr>` → setfield_gc_i (invalidates the
        // matching cached getfield). After lower_state_field_write (which only
        // matches `state.<scalar> = ...` whose LHS base is `state`).
        if let Some(()) = self.lower_state_ref_field_setfield(expr) {
            return Some(());
        }
        // Field write on a local ref binding with known struct type:
        // `<binding>.<field> = <expr>` → setfield_gc_i/setfield_gc_r.
        if let Some(()) = self.lower_ref_binding_setfield(expr) {
            return Some(());
        }
        // Raw native-memory store: `majit_raw_store_i64(base, ea, val);` →
        // raw_store_i (jtransform.py:1156-1163 rewrite_op_raw_store).  Runs
        // before the residual-call path so the store lowers to an inline
        // side-effecting IR op instead of an opaque helper call.
        if let Some(()) = self.lower_raw_store_stmt(expr) {
            return Some(());
        }
        if let Some(()) = self.lower_state_array_write(expr) {
            return Some(());
        }
        // RPython jtransform.py:923 — virtualizable field write rewrite.
        if let Some(()) = self.lower_vable_field_write(expr) {
            return Some(());
        }
        // RPython jtransform.py:794 — virtualizable array write rewrite.
        if let Some(()) = self.lower_vable_array_write(expr) {
            return Some(());
        }
        // RPython jtransform.py:650 — hint_force_virtualizable rewrite.
        if let Some(()) = self.lower_vable_force(expr) {
            return Some(());
        }
        // RPython jtransform.py:655 — access_directly/fresh_virtualizable suppression.
        if let Some(()) = self.lower_vable_hint_suppress(expr) {
            return Some(());
        }
        // RPython jtransform.py:1685 — conditional_call!(condition, func, args...)
        if let Some(()) = self.lower_conditional_call(expr) {
            return Some(());
        }
        // RPython jtransform.py:292 — record_known_result!(result, func, args...)
        if let Some(()) = self.lower_record_known_result(expr) {
            return Some(());
        }
        // Local variable reassignment: `pc = expr` or `stackok = expr`.
        // Rebinds an already-known local to a freshly-lowered RHS value.
        if let Some(()) = self.lower_local_reassign(expr) {
            return Some(());
        }

        if let Expr::If(expr_if) = expr {
            return self.lower_if_stmt(expr_if);
        }

        if let Expr::Match(expr_match) = expr {
            return self.lower_match_stmt(expr_match);
        }

        if let Expr::While(expr_while) = expr {
            return self.lower_while_loop(expr_while);
        }

        if let Expr::Loop(expr_loop) = expr {
            return self.lower_loop_expr(expr_loop);
        }

        if let Expr::ForLoop(expr_for) = expr {
            return self.lower_for_loop(expr_for);
        }

        if let Some(()) = self.lower_config_call_stmt(expr) {
            return Some(());
        }

        // Config-aware patterns
        if self.config.is_some() {
            if let Some(()) = self.lower_io_call_stmt(expr) {
                return Some(());
            }
        }

        None
    }

    // ── Config-aware lowering methods ────────────────────────────────

    pub(super) fn lower_config_call_stmt(&mut self, expr: &Expr) -> Option<()> {
        let Expr::Call(call) = expr else {
            return None;
        };
        let policy = self.resolve_call_policy(&call.func)?;
        if call.args.len() > MAX_HELPER_CALL_ARITY {
            return None;
        }

        let mut arg_bindings = Vec::with_capacity(call.args.len());
        for arg in &call.args {
            let binding = self.lower_value_expr(arg)?;
            arg_bindings.push(binding);
        }
        let func = &call.func;
        // jtransform.py:467-471 / 480-482: `-live-` follows the call, it does
        // not precede it.  Decide here whether the explicit arm below needs a
        // trailing marker; the inferred arm emits its own runtime-conditional
        // one, so it reports `false` and is excluded.
        let post_live_after_call = match &policy {
            CallPolicySpec::Explicit(kind) => explicit_call_emits_post_live(*kind),
            CallPolicySpec::Infer => false,
        };
        match policy {
            CallPolicySpec::Explicit(kind) => match kind {
                crate::jit_interp::CallPolicyKind::ResidualVoid
                | crate::jit_interp::CallPolicyKind::ResidualVoidCannotRaise => {
                    let cannot_raise = matches!(
                        kind,
                        crate::jit_interp::CallPolicyKind::ResidualVoidCannotRaise,
                    );
                    let write_ei = self.residual_write_effect_info_tokens(func, !cannot_raise);
                    let call_stmt = if let Some(write_ei) = write_ei {
                        // Declared field mutator: residual with a write-set
                        // naming the mutated field so the optimizer invalidates
                        // its cached `getfield_gc_i`, preserving the policy's
                        // can-raise / cannot-raise extra-effect.
                        quote! {
                            __builder.residual_call_void_canonical_via_target_with_effect_info(
                                __fn_idx,
                                __typed_args,
                                #write_ei,
                            );
                        }
                    } else if cannot_raise {
                        quote! {
                            __builder.residual_call_void_canonical_via_target_with_effect_info(
                                __fn_idx,
                                __typed_args,
                                majit_metainterp::cannot_raise_effect_info(),
                            );
                        }
                    } else {
                        quote! {
                            __builder.residual_call_void_canonical_via_target(__fn_idx, __typed_args);
                        }
                    };
                    if let Some(arg_regs) = int_arg_regs(&arg_bindings) {
                        let typed_args = quote! {
                            &[#(majit_metainterp::JitCallArg::int(#arg_regs)),*]
                        };
                        self.emit_op(
                            OpMeta::linear(OpKind::Call, Register::ints(&arg_regs), vec![]),
                            quote! {
                                let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                                let __typed_args = #typed_args;
                                #call_stmt
                            },
                        );
                    } else {
                        let typed_args = typed_call_arg_tokens(&arg_bindings);
                        let __arg_regs: Vec<Register> =
                            arg_bindings.iter().map(Register::from_binding).collect();
                        self.emit_op(
                            OpMeta::linear(OpKind::Call, __arg_regs, vec![]),
                            quote! {
                                let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                                let __typed_args = #typed_args;
                                #call_stmt
                            },
                        );
                    }
                }
                crate::jit_interp::CallPolicyKind::MayForceVoid => {
                    if let Some(arg_regs) = int_arg_regs(&arg_bindings) {
                        let typed_args = quote! {
                            &[#(majit_metainterp::JitCallArg::int(#arg_regs)),*]
                        };
                        self.emit_op(
                            OpMeta::linear(OpKind::Call, Register::ints(&arg_regs), vec![]),
                            quote! {
                                let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                                __builder.call_may_force_void_canonical_via_target(__fn_idx, #typed_args);
                            },
                        );
                    } else {
                        let typed_args = typed_call_arg_tokens(&arg_bindings);
                        let __arg_regs: Vec<Register> =
                            arg_bindings.iter().map(Register::from_binding).collect();
                        self.emit_op(
                            OpMeta::linear(OpKind::Call, __arg_regs, vec![]),
                            quote! {
                                let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                                __builder.call_may_force_void_canonical_via_target(__fn_idx, #typed_args);
                            },
                        );
                    }
                }
                crate::jit_interp::CallPolicyKind::ReleaseGilVoid => {
                    if let Some(arg_regs) = int_arg_regs(&arg_bindings) {
                        let typed_args = quote! {
                            &[#(majit_metainterp::JitCallArg::int(#arg_regs)),*]
                        };
                        self.emit_op(
                            OpMeta::linear(OpKind::Call, Register::ints(&arg_regs), vec![]),
                            quote! {
                                let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                                __builder.call_release_gil_void_canonical_via_target(__fn_idx, #typed_args);
                            },
                        );
                    } else {
                        let typed_args = typed_call_arg_tokens(&arg_bindings);
                        let __arg_regs: Vec<Register> =
                            arg_bindings.iter().map(Register::from_binding).collect();
                        self.emit_op(
                            OpMeta::linear(OpKind::Call, __arg_regs, vec![]),
                            quote! {
                                let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                                __builder.call_release_gil_void_canonical_via_target(__fn_idx, #typed_args);
                            },
                        );
                    }
                }
                crate::jit_interp::CallPolicyKind::LoopInvariantVoid => {
                    if let Some(arg_regs) = int_arg_regs(&arg_bindings) {
                        let typed_args = quote! {
                            &[#(majit_metainterp::JitCallArg::int(#arg_regs)),*]
                        };
                        self.emit_op(
                            OpMeta::linear(OpKind::Call, Register::ints(&arg_regs), vec![]),
                            quote! {
                                let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                                __builder.call_loopinvariant_void_canonical_via_target(__fn_idx, #typed_args);
                            },
                        );
                    } else {
                        let typed_args = typed_call_arg_tokens(&arg_bindings);
                        let __arg_regs: Vec<Register> =
                            arg_bindings.iter().map(Register::from_binding).collect();
                        self.emit_op(
                            OpMeta::linear(OpKind::Call, __arg_regs, vec![]),
                            quote! {
                                let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                                __builder.call_loopinvariant_void_canonical_via_target(__fn_idx, #typed_args);
                            },
                        );
                    }
                }
                // Stmt-form variants of result-returning policies discard
                // the value but still need the IR call op recorded so the
                // compiled trace runs the side effect (e.g. aheui OP_POP's
                // `lj::stack_pop(state.selected_ref);` discards the popped
                // value but the pop side effect must reach compiled code).
                // Allocate a throwaway destination register; never read it.
                //
                // RPython jtransform.py:456 `handle_residual_call` lowers
                // every direct_call to a residual_call regardless of result
                // usage; majit's CallPolicyKind enum captures the effect
                // distinction (Residual / MayForce / ReleaseGil /
                // LoopInvariant / Elidable) so the dispatched bytecode
                // varies per policy here.  Wrapped variants stay deferred
                // — wrapper closure plumbing is shared with the void path
                // and not exercised by current `#[jit_interp]` users.
                crate::jit_interp::CallPolicyKind::ResidualInt
                | crate::jit_interp::CallPolicyKind::MayForceInt
                | crate::jit_interp::CallPolicyKind::ReleaseGilInt
                | crate::jit_interp::CallPolicyKind::LoopInvariantInt => {
                    let throwaway_reg = self.alloc_reg();
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
                    // A declared `residual_writes` mutator routes through the
                    // `_with_effect_info` int variant carrying the field
                    // write-set; only the residual policy qualifies (may-force /
                    // release-gil / loop-invariant carry their own effects).
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
                                    #throwaway_reg,
                                    #write_ei,
                                );
                            }
                        } else {
                            quote! {
                                __builder.#canonical_call(
                                    __fn_idx,
                                    &[#(majit_metainterp::JitCallArg::int(#arg_regs)),*],
                                    #throwaway_reg,
                                );
                            }
                        };
                        self.emit_op(
                            OpMeta::linear(
                                OpKind::Call,
                                Register::ints(&arg_regs),
                                vec![Register::int(throwaway_reg)],
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
                                    #throwaway_reg,
                                    #write_ei,
                                );
                            }
                        } else {
                            quote! {
                                __builder.#canonical_call(__fn_idx, #typed_args, #throwaway_reg);
                            }
                        };
                        self.emit_op(
                            OpMeta::linear(
                                OpKind::Call,
                                __arg_regs,
                                vec![Register::int(throwaway_reg)],
                            ),
                            quote! {
                                let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                                #call_invocation
                            },
                        );
                    }
                }
                // `call.py:303 getcalldescr` non-elidable EF_CANNOT_RAISE
                // for int residuals.  Dispatches via the
                // `_with_effect_info(cannot_raise_effect_info())` builder
                // method so the recorded calldescr's `EffectInfo`
                // matches PyPy's `cannot_raise_effect_info()`.
                crate::jit_interp::CallPolicyKind::ResidualIntCannotRaise => {
                    let throwaway_reg = self.alloc_reg();
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::Call,
                            __arg_regs,
                            vec![Register::int(throwaway_reg)],
                        ),
                        quote! {
                            let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                            __builder.residual_call_int_canonical_via_target_with_effect_info(
                                __fn_idx,
                                #typed_args,
                                #throwaway_reg,
                                majit_metainterp::cannot_raise_effect_info(),
                            );
                        },
                    );
                }
                crate::jit_interp::CallPolicyKind::ElidableInt
                | crate::jit_interp::CallPolicyKind::ElidableIntCannotRaise
                | crate::jit_interp::CallPolicyKind::ElidableIntOrMemerror => {
                    // Parity #14 Slice C.4 + Parity #20: Pure flows through
                    // the canonical `BC_RESIDUAL_CALL_*_I` family with the
                    // calldescr's `extra_info` set per `call.py:292-299
                    // _canraise(op)`'s 3-way pick.  The walker
                    // (`pyjitpl/dispatch.rs` Slice C.1) reads
                    // `effectinfo.check_is_elidable()` and routes through
                    // `record_result_of_call_pure` mirroring
                    // `pyjitpl.py:2111-2115`; the trailing
                    // `GUARD_NO_EXCEPTION` is gated on
                    // `effectinfo.check_can_raise(False)` so cannot-raise
                    // elidable callees skip it.
                    let throwaway_reg = self.alloc_reg();
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    let call_stmt = match kind {
                        crate::jit_interp::CallPolicyKind::ElidableInt => quote! {
                            __builder.call_pure_int_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg);
                        },
                        crate::jit_interp::CallPolicyKind::ElidableIntCannotRaise => quote! {
                            __builder.call_pure_int_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #throwaway_reg);
                        },
                        crate::jit_interp::CallPolicyKind::ElidableIntOrMemerror => quote! {
                            __builder.call_pure_int_canonical_via_target_or_memerror(__fn_idx, #typed_args, #throwaway_reg);
                        },
                        _ => unreachable!(),
                    };
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::Call,
                            __arg_regs,
                            vec![Register::int(throwaway_reg)],
                        ),
                        quote! {
                            let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                            #call_stmt
                        },
                    );
                }
                crate::jit_interp::CallPolicyKind::ResidualVoidWrapped
                | crate::jit_interp::CallPolicyKind::ResidualVoidCannotRaiseWrapped => {
                    let policy_path = helper_policy_path(&call.func)?;
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    let __slot_tokens = CondCallEffectSlot::for_wrapped_kind(kind);
                    // `call.py:301-303 getcalldescr`: descr's `EffectInfo`
                    // differs by the analyzer's `_canraise` result, but the
                    // residual_call dispatch family is the same.
                    let call_stmt = if matches!(
                        kind,
                        crate::jit_interp::CallPolicyKind::ResidualVoidCannotRaiseWrapped,
                    ) {
                        quote! {
                            __builder.residual_call_void_canonical_via_target_with_effect_info(
                                __fn_idx,
                                #typed_args,
                                majit_metainterp::cannot_raise_effect_info(),
                            );
                        }
                    } else {
                        quote! { __builder.residual_call_void_canonical_via_target(__fn_idx, #typed_args); }
                    };
                    self.emit_op(
                        OpMeta::linear(OpKind::Call, __arg_regs, vec![]),
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
                crate::jit_interp::CallPolicyKind::MayForceVoidWrapped
                | crate::jit_interp::CallPolicyKind::ReleaseGilVoidWrapped
                | crate::jit_interp::CallPolicyKind::LoopInvariantVoidWrapped => {
                    let policy_path = helper_policy_path(&call.func)?;
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    let __slot_tokens = CondCallEffectSlot::for_wrapped_kind(kind);
                    let call_stmt = match kind {
                        crate::jit_interp::CallPolicyKind::MayForceVoidWrapped => {
                            quote! { __builder.call_may_force_void_canonical_via_target(__fn_idx, #typed_args); }
                        }
                        crate::jit_interp::CallPolicyKind::ReleaseGilVoidWrapped => {
                            quote! { __builder.call_release_gil_void_canonical_via_target(__fn_idx, #typed_args); }
                        }
                        crate::jit_interp::CallPolicyKind::LoopInvariantVoidWrapped => {
                            quote! { __builder.call_loopinvariant_void_canonical_via_target(__fn_idx, #typed_args); }
                        }
                        _ => unreachable!(),
                    };
                    self.emit_op(
                        OpMeta::linear(OpKind::Call, __arg_regs, vec![]),
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
                // Non-wrapped Ref statement-form (result discarded).
                // Like ResidualVoid but emits a ref-return residual call
                // whose result is discarded.
                crate::jit_interp::CallPolicyKind::ResidualRef => {
                    let typed_args = typed_call_arg_tokens(&arg_bindings);
                    let throwaway_reg = self.alloc_reg();
                    let __arg_regs: Vec<Register> =
                        arg_bindings.iter().map(Register::from_binding).collect();
                    self.emit_op(
                        OpMeta::linear(OpKind::Call, __arg_regs, vec![Register::ref_(throwaway_reg)]),
                        quote! {
                            let __fn_idx = __builder.add_fn_ptr(#func as *const ());
                            let __typed_args = #typed_args;
                            __builder.residual_call_ref_canonical_via_target(__fn_idx, __typed_args, #throwaway_reg);
                        },
                    );
                }
                // Wrapped Int / Ref / Float statement-form: result discarded,
                // but the residual_call must still execute the side effect on
                // the compiled trace.  RPython jtransform.py:456
                // handle_residual_call lowers every direct_call regardless of
                // result usage; the wrapped policy adds the trace_target /
                // concrete_target tuple resolution shared with the void
                // wrapped variants above.  Throwaway destination register is
                // allocated (per-bank slot picked by JitCodeBuilder when the
                // typed call dispatches) and never read.
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
                    let throwaway_reg = self.alloc_reg();
                    // Result bank — pick from the wrapped policy variant family.
                    let result_kind = match kind {
                        crate::jit_interp::CallPolicyKind::ResidualIntWrapped
                        | crate::jit_interp::CallPolicyKind::ResidualIntCannotRaiseWrapped
                        | crate::jit_interp::CallPolicyKind::MayForceIntWrapped
                        | crate::jit_interp::CallPolicyKind::ReleaseGilIntWrapped
                        | crate::jit_interp::CallPolicyKind::LoopInvariantIntWrapped
                        | crate::jit_interp::CallPolicyKind::ElidableIntWrapped
                        | crate::jit_interp::CallPolicyKind::ElidableIntCannotRaiseWrapped
                        | crate::jit_interp::CallPolicyKind::ElidableIntOrMemerrorWrapped => {
                            BindingKind::Int
                        }
                        crate::jit_interp::CallPolicyKind::ResidualRefWrapped
                        | crate::jit_interp::CallPolicyKind::ResidualRefCannotRaiseWrapped
                        | crate::jit_interp::CallPolicyKind::MayForceRefWrapped
                        | crate::jit_interp::CallPolicyKind::LoopInvariantRefWrapped
                        | crate::jit_interp::CallPolicyKind::ElidableRefWrapped
                        | crate::jit_interp::CallPolicyKind::ElidableRefCannotRaiseWrapped
                        | crate::jit_interp::CallPolicyKind::ElidableRefOrMemerrorWrapped => {
                            BindingKind::Ref
                        }
                        _ => BindingKind::Float,
                    };
                    let call_stmt = match kind {
                        crate::jit_interp::CallPolicyKind::ResidualIntWrapped => {
                            quote! { __builder.residual_call_int_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        // `call.py:303` non-elidable EF_CANNOT_RAISE int — wrapped.
                        crate::jit_interp::CallPolicyKind::ResidualIntCannotRaiseWrapped => {
                            quote! {
                                __builder.residual_call_int_canonical_via_target_with_effect_info(
                                    __fn_idx,
                                    #typed_args,
                                    #throwaway_reg,
                                    majit_metainterp::cannot_raise_effect_info(),
                                );
                            }
                        }
                        crate::jit_interp::CallPolicyKind::MayForceIntWrapped => {
                            quote! { __builder.call_may_force_int_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ReleaseGilIntWrapped => {
                            quote! { __builder.call_release_gil_int_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::LoopInvariantIntWrapped => {
                            quote! { __builder.call_loopinvariant_int_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableIntWrapped => {
                            quote! { __builder.call_pure_int_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableIntCannotRaiseWrapped => {
                            quote! { __builder.call_pure_int_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableIntOrMemerrorWrapped => {
                            quote! { __builder.call_pure_int_canonical_via_target_or_memerror(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ResidualRefWrapped => {
                            quote! { __builder.residual_call_ref_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        // `call.py:303` non-elidable EF_CANNOT_RAISE ref — wrapped.
                        crate::jit_interp::CallPolicyKind::ResidualRefCannotRaiseWrapped => {
                            quote! {
                                __builder.residual_call_ref_canonical_via_target_with_effect_info(
                                    __fn_idx,
                                    #typed_args,
                                    #throwaway_reg,
                                    majit_metainterp::cannot_raise_effect_info(),
                                );
                            }
                        }
                        crate::jit_interp::CallPolicyKind::MayForceRefWrapped => {
                            quote! { __builder.call_may_force_ref_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::LoopInvariantRefWrapped => {
                            quote! { __builder.call_loopinvariant_ref_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableRefWrapped => {
                            quote! { __builder.call_pure_ref_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableRefCannotRaiseWrapped => {
                            quote! { __builder.call_pure_ref_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableRefOrMemerrorWrapped => {
                            quote! { __builder.call_pure_ref_canonical_via_target_or_memerror(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ResidualFloatWrapped => {
                            quote! { __builder.residual_call_float_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        // `call.py:303` non-elidable EF_CANNOT_RAISE float — wrapped.
                        crate::jit_interp::CallPolicyKind::ResidualFloatCannotRaiseWrapped => {
                            quote! {
                                __builder.residual_call_float_canonical_via_target_with_effect_info(
                                    __fn_idx,
                                    #typed_args,
                                    #throwaway_reg,
                                    majit_metainterp::cannot_raise_effect_info(),
                                );
                            }
                        }
                        crate::jit_interp::CallPolicyKind::MayForceFloatWrapped => {
                            quote! { __builder.call_may_force_float_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ReleaseGilFloatWrapped => {
                            quote! { __builder.call_release_gil_float_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::LoopInvariantFloatWrapped => {
                            quote! { __builder.call_loopinvariant_float_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableFloatWrapped => {
                            quote! { __builder.call_pure_float_canonical_via_target(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableFloatCannotRaiseWrapped => {
                            quote! { __builder.call_pure_float_canonical_via_target_cannot_raise(__fn_idx, #typed_args, #throwaway_reg); }
                        }
                        crate::jit_interp::CallPolicyKind::ElidableFloatOrMemerrorWrapped => {
                            quote! { __builder.call_pure_float_canonical_via_target_or_memerror(__fn_idx, #typed_args, #throwaway_reg); }
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
                            vec![Register::new(result_kind, throwaway_reg)],
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
                crate::jit_interp::CallPolicyKind::ConcreteOnlyVoid => {
                    // No IR ops emitted on the JIT path. The concrete
                    // RefFieldRewriter path calls the function normally.
                }
                _ => return None,
            },
            CallPolicySpec::Infer => {
                let policy_path = helper_policy_path(&call.func)?;
                let typed_args = typed_call_arg_tokens(&arg_bindings);
                let __arg_regs: Vec<Register> =
                    arg_bindings.iter().map(Register::from_binding).collect();
                let __slot_tokens = CondCallEffectSlot::slot_from_policy_tokens();
                self.emit_op(
                    OpMeta::linear(OpKind::Call, __arg_regs, vec![]),
                    quote! {
                        let (__policy, _inline_builder, __trace_target, __concrete_target, _prebuild, __save_err) = #policy_path();
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
                            #VOID_DONT_LOOK_INSIDE => {
                                __builder.residual_call_void_canonical_via_target(__fn_idx, #typed_args);
                            }
                            // `call.py:303` non-elidable EF_CANNOT_RAISE for void.
                            #VOID_DONT_LOOK_INSIDE_CANNOT_RAISE => {
                                __builder.residual_call_void_canonical_via_target_with_effect_info(
                                    __fn_idx,
                                    #typed_args,
                                    majit_metainterp::cannot_raise_effect_info(),
                                );
                            }
                            #VOID_MAY_FORCE => {
                                __builder.call_may_force_void_canonical_via_target(__fn_idx, #typed_args);
                            }
                            #VOID_RELEASE_GIL => {
                                __builder.call_release_gil_void_canonical_via_target(__fn_idx, #typed_args);
                            }
                            #VOID_LOOP_INVARIANT => {
                                __builder.call_loopinvariant_void_canonical_via_target(__fn_idx, #typed_args);
                            }
                            // The `_ =>` arm is a runtime invariant violation
                            // (helper policy companion fn returned an
                            // unrecognized byte), NOT a recoverable lower-time
                            // inference failure, so it panics regardless of the
                            // outer Lowerer's `InferenceFailureMode`. Earlier
                            // versions routed this through
                            // `inference_failure_tokens` which emits
                            // `return None;` in `ReturnNone` mode — wrong for
                            // dispatch-body wrappers that return `JitCode`,
                            // not `Option<_>`, surfaced as a type-check error
                            // when a `dont_look_inside` helper is called from
                            // a dispatch JitCode body (A.2.5).
                            other => panic!(
                                "inferred void-call policy returned unrecognized byte {other}; \
                                 expected one of 1 (residual), 9 (may_force), 13 (release_gil), \
                                 17 (loopinvariant)"
                            ),
                        }
                    },
                );
                // jtransform.py:467-471 — trailing `-live-` gated on the
                // runtime policy byte's can-raise codes (void residual /
                // may-force / release-gil); LOOP_INVARIANT and the
                // CANNOT_RAISE void surface skip it.
                self.emit_op(
                    OpMeta::live_marker_if(inferred_policy_live_condition(
                        func,
                        &[VOID_DONT_LOOK_INSIDE, VOID_MAY_FORCE, VOID_RELEASE_GIL],
                    )),
                    quote! { let _ = __builder.live_placeholder(); },
                );
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
        Some(())
    }

    /// Lower a raw native-memory store intrinsic
    /// `majit_raw_store_i64(base, ea, val)` to a `raw_store_i` op.
    ///
    /// RPython parity: an `rffi.raw_storage_setitem(base, offset, value)`
    /// in the interpreter source is rewritten by
    /// `jtransform.py:1156-1163 rewrite_op_raw_store` to
    /// `raw_store_i(base, offset, value, arraydescrof(CArray(T)))`.  Here
    /// the macro recognizes the kernel's `unsafe` intrinsic by name and
    /// emits the same op with an 8-byte raw-int array descr
    /// (`add_raw_int_array_descr`).  The three operands are all int-kind
    /// (raw address, byte offset, stored value); the op has no result.
    fn lower_raw_store_stmt(&mut self, expr: &Expr) -> Option<()> {
        let Expr::Call(call) = expr else {
            return None;
        };
        let segments = canonical_expr_segments(&call.func)?;
        if segments.last().map(String::as_str) != Some("majit_raw_store_i64") {
            return None;
        }
        if call.args.len() != 3 {
            return None;
        }
        let base = self.lower_value_expr(&call.args[0])?;
        let ea = self.lower_value_expr(&call.args[1])?;
        let val = self.lower_value_expr(&call.args[2])?;
        let (base_reg, ea_reg, val_reg) = (base.reg, ea.reg, val.reg);
        self.emit_op(
            OpMeta::linear(
                OpKind::RawStore,
                vec![
                    Register::int(base_reg),
                    Register::int(ea_reg),
                    Register::int(val_reg),
                ],
                vec![],
            ),
            quote! {
                let __raw_descr = __builder.add_raw_int_array_descr(8);
                __builder.raw_store_i(
                    #base_reg as u16,
                    #ea_reg as u16,
                    #val_reg as u16,
                    __raw_descr,
                );
            },
        );
        Some(())
    }

    /// Lower I/O call: aheui_io::write_number(r, writer) → residual_call_void(shim, r)
    fn lower_io_call_stmt(&mut self, expr: &Expr) -> Option<()> {
        let Expr::Call(call) = expr else {
            return None;
        };
        let config = self.config?;
        let func_segments = canonical_expr_segments(&call.func)?;

        for (io_path, shim) in &config.io_shims {
            if func_segments == *io_path {
                let arg = unwrap_ref_expr(call.args.first()?);
                let binding = self.lower_value_expr(arg)?;
                let reg = binding.reg;
                self.emit_op(
                    OpMeta::linear(OpKind::Call, vec![Register::int(reg)], vec![]),
                    quote! {
                        let __fn_idx = __builder.add_fn_ptr(#shim as *const ());
                        __builder.residual_call_void_canonical_via_target(
                            __fn_idx,
                            &[majit_metainterp::JitCallArg::int(#reg)],
                        );
                    },
                );
                // jtransform.py:467-471 — the void shim is a may-raise
                // residual call, so `-live-` follows it.
                self.emit_op(
                    OpMeta::live_marker(),
                    quote! { let _ = __builder.live_placeholder(); },
                );
                return Some(());
            }
        }

        None
    }
}
