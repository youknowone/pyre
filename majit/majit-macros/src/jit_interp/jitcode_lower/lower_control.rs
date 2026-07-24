use super::*;

fn literal_nonnegative_i64(expr: &Expr) -> Option<i64> {
    let Expr::Lit(ExprLit {
        lit: Lit::Int(lit), ..
    }) = expr
    else {
        return None;
    };
    let value = lit.base10_parse::<i64>().ok()?;
    (value >= 0).then_some(value)
}

impl<'c> Lowerer<'c> {
    /// True when any op recorded since index `since` is a float
    /// comparison — an `OpKind::BinopF` whose result register is int
    /// banked.  Float arithmetic writes a float result; only the value-
    /// form `float_lt/le/eq/ne/gt/ge` (`ff>i`) writes an int.  A float
    /// comparison feeding a conditional guard grows a bridge that hangs
    /// the compiled trace (a4e191f71b5), so the branch lowerers roll back
    /// and bail to interpreter fallback when the condition lowered through
    /// one.
    fn ops_since_contain_float_compare(&self, since: usize) -> bool {
        self.op_metadata[since..].iter().any(|op| {
            matches!(op.kind, OpKind::BinopF)
                && op.writes.iter().any(|w| matches!(w.kind, BindingKind::Int))
        })
    }

    pub(super) fn lower_if_stmt(&mut self, expr_if: &ExprIf) -> Option<()> {
        let snap_stmts = self.statements.len();
        let snap_meta = self.op_metadata.len();
        let snap_reg = self.next_reg;
        let snap_bindings = self.bindings.clone();
        let cond = self.lower_value_expr(&expr_if.cond)?;
        if self.ops_since_contain_float_compare(snap_meta) {
            // Guard over a float comparison hangs the compiled trace; roll
            // back the condition ops and bail so the arm runs interpreted.
            self.statements.truncate(snap_stmts);
            self.op_metadata.truncate(snap_meta);
            self.next_reg = snap_reg;
            self.bindings = snap_bindings;
            return None;
        }
        let else_label = self.alloc_label();
        let end_label = self.alloc_label();
        let cond_reg = cond.reg;
        let then_seq = self.lower_branch_expr(&Expr::Block(syn::ExprBlock {
            attrs: Vec::new(),
            label: None,
            block: expr_if.then_branch.clone(),
        }))?;
        let else_seq = match expr_if.else_branch.as_ref() {
            Some((_, else_expr)) => self.lower_branch_expr(else_expr)?,
            None => LoweredSequence::default(),
        };

        self.emit_aux(quote! { let #else_label = __builder.new_label(); });
        self.emit_aux(quote! { let #end_label = __builder.new_label(); });
        // RPython `flatten.py:259` `-live-` convention: every guard-bearing
        // instruction is *preceded* by a `live` marker (byte order:
        // `BC_LIVE+offset` then the guard op). The recorded `orgpc` (=
        // RPython `pyjitpl.py:3713 orgpc = position`, copied to the guard's
        // `resumepc` via `record_state_guard`) is the byte position of the
        // guard op itself, so the BC_LIVE marker sits at `orgpc - SIZE_LIVE_OP`
        // and blackhole's `get_current_position_info` reads liveness from
        // there.  Without this preceding marker, blackhole panics with
        // `missing liveness[N] in JitCode`.
        self.emit_op(
            OpMeta::live_marker(),
            quote! { let _ = __builder.live_placeholder(); },
        );
        self.emit_conditional_guard(cond_reg, &else_label);
        self.append_lowered_sequence(then_seq);
        self.emit_jump(&end_label);
        self.emit_label_def(&else_label);
        self.append_lowered_sequence(else_seq);
        self.emit_label_def(&end_label);
        Some(())
    }

    /// Lower a standalone match expression to a chained if-else guard sequence.
    ///
    /// ```text
    /// match x { 1 => body1, 2 => body2, _ => default }
    /// ```
    /// becomes:
    /// ```text
    /// eq_1 = (x == 1); brz eq_1, next1; body1; jmp end; next1:
    /// eq_2 = (x == 2); brz eq_2, next2; body2; jmp end; next2:
    /// default; end:
    /// ```
    pub(super) fn lower_match_stmt(&mut self, expr_match: &syn::ExprMatch) -> Option<()> {
        let discriminant = self.lower_value_expr(&expr_match.expr)?;
        if !matches!(discriminant.kind, BindingKind::Int) {
            return None;
        }

        let end_label = self.alloc_label();
        self.emit_aux(quote! { let #end_label = __builder.new_label(); });

        // Separate literal/path arms from the wildcard/default arm.
        // Uses extract_pat_value_tokens (not extract_pat_literals) so
        // symbolic constants like OP_JMP are accepted alongside literals.
        let mut guarded_arms: Vec<(Vec<TokenStream>, &Box<Expr>)> = Vec::new();
        let mut default_arm = None;

        for arm in &expr_match.arms {
            match &arm.pat {
                Pat::Wild(_) => {
                    default_arm = Some(&arm.body);
                }
                Pat::Ident(pat_ident) if pat_ident.subpat.is_none() => {
                    let name = pat_ident.ident.to_string();
                    if name.starts_with(|c: char| c.is_lowercase()) {
                        default_arm = Some(&arm.body);
                    } else {
                        let tokens = extract_pat_value_tokens(&arm.pat)?;
                        guarded_arms.push((tokens, &arm.body));
                    }
                }
                _ => {
                    let tokens = extract_pat_value_tokens(&arm.pat)?;
                    guarded_arms.push((tokens, &arm.body));
                }
            }
        }

        let disc_reg = discriminant.reg;

        for (value_tokens, body) in &guarded_arms {
            let next_label = self.alloc_label();
            self.emit_aux(quote! { let #next_label = __builder.new_label(); });

            if value_tokens.len() == 1 {
                let value_tok = &value_tokens[0];
                let const_reg = self.alloc_reg();
                let eq_reg = self.alloc_reg();
                self.emit_op(
                    OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(const_reg)]),
                    quote! { __builder.load_const_i_value(#const_reg, #value_tok); },
                );
                self.emit_op(
                    OpMeta::linear(
                        OpKind::BinopI,
                        Register::ints(&[disc_reg, const_reg]),
                        vec![Register::int(eq_reg)],
                    ),
                    quote! { __builder.record_binop_i(#eq_reg, majit_ir::OpCode::IntEq, #disc_reg, #const_reg); },
                );
                self.emit_op(
                    OpMeta::live_marker(),
                    quote! { let _ = __builder.live_placeholder(); },
                );
                self.emit_conditional_guard(eq_reg, &next_label);
            } else {
                let first_tok = &value_tokens[0];
                let first_const_reg = self.alloc_reg();
                let mut or_reg = self.alloc_reg();
                self.emit_op(
                    OpMeta::linear(
                        OpKind::LoadConstI,
                        vec![],
                        vec![Register::int(first_const_reg)],
                    ),
                    quote! { __builder.load_const_i_value(#first_const_reg, #first_tok); },
                );
                self.emit_op(
                    OpMeta::linear(
                        OpKind::BinopI,
                        Register::ints(&[disc_reg, first_const_reg]),
                        vec![Register::int(or_reg)],
                    ),
                    quote! { __builder.record_binop_i(#or_reg, majit_ir::OpCode::IntEq, #disc_reg, #first_const_reg); },
                );
                for tok in &value_tokens[1..] {
                    let const_reg = self.alloc_reg();
                    let eq_reg = self.alloc_reg();
                    let new_or_reg = self.alloc_reg();
                    self.emit_op(
                        OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(const_reg)]),
                        quote! { __builder.load_const_i_value(#const_reg, #tok); },
                    );
                    self.emit_op(
                    OpMeta::linear(
                        OpKind::BinopI,
                        Register::ints(&[disc_reg, const_reg]),
                        vec![Register::int(eq_reg)],
                    ),
                    quote! { __builder.record_binop_i(#eq_reg, majit_ir::OpCode::IntEq, #disc_reg, #const_reg); },
                );
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::BinopI,
                            Register::ints(&[or_reg, eq_reg]),
                            vec![Register::int(new_or_reg)],
                        ),
                        quote! { __builder.record_binop_i(#new_or_reg, majit_ir::OpCode::IntOr, #or_reg, #eq_reg); },
                    );
                    or_reg = new_or_reg;
                }
                self.emit_op(
                    OpMeta::live_marker(),
                    quote! { let _ = __builder.live_placeholder(); },
                );
                self.emit_conditional_guard(or_reg, &next_label);
            }

            let body_seq = self.lower_branch_expr(body)?;
            self.append_lowered_sequence(body_seq);
            self.emit_jump(&end_label);
            self.emit_label_def(&next_label);
        }

        // Default arm
        if let Some(default_body) = default_arm {
            let default_seq = self.lower_branch_expr(default_body)?;
            self.append_lowered_sequence(default_seq);
        }

        self.emit_label_def(&end_label);
        Some(())
    }

    // ── Loop lowering ────────────────────────────────────────────────

    /// Lower `while cond { body }` to a JitCode branch sequence:
    /// ```text
    /// loop_start:
    ///   eval cond
    ///   goto_if_not_int_is_true(cond, loop_end)
    ///   eval body
    ///   jump(loop_start)
    /// loop_end:
    /// ```
    pub(super) fn lower_while_loop(&mut self, expr_while: &syn::ExprWhile) -> Option<()> {
        let snap_stmts = self.statements.len();
        let snap_meta = self.op_metadata.len();
        let snap_reg = self.next_reg;
        let snap_label = self.next_label;
        let snap_bindings = self.bindings.clone();

        let loop_start = self.alloc_label();
        let loop_end = self.alloc_label();

        self.emit_aux(quote! { let #loop_start = __builder.new_label(); });
        self.emit_aux(quote! { let #loop_end = __builder.new_label(); });
        self.emit_label_def(&loop_start);

        // Evaluate the condition
        let cond = self.lower_value_expr(&expr_while.cond)?;
        if self.ops_since_contain_float_compare(snap_meta) {
            // Guard over a float comparison hangs the compiled trace; roll
            // back everything emitted for this loop and bail so the arm
            // runs interpreted instead.
            self.statements.truncate(snap_stmts);
            self.op_metadata.truncate(snap_meta);
            self.next_reg = snap_reg;
            self.next_label = snap_label;
            self.bindings = snap_bindings;
            return None;
        }
        let cond_reg = cond.reg;
        self.emit_op(
            OpMeta::live_marker(),
            quote! { let _ = __builder.live_placeholder(); },
        );
        self.emit_conditional_guard(cond_reg, &loop_end);

        // Lower the body, with break targets pointing to loop_end
        let body_seq = self.lower_loop_body(&expr_while.body, &loop_end, &loop_start)?;
        self.append_lowered_sequence(body_seq);

        // Back-edge jump
        self.emit_jump(&loop_start);
        self.emit_label_def(&loop_end);
        Some(())
    }

    /// Lower `loop { body }` to a JitCode branch sequence:
    /// ```text
    /// loop_start:
    ///   eval body (break → jump loop_end, continue → jump loop_start)
    ///   jump(loop_start)
    /// loop_end:
    /// ```
    pub(super) fn lower_loop_expr(&mut self, expr_loop: &syn::ExprLoop) -> Option<()> {
        let loop_start = self.alloc_label();
        let loop_end = self.alloc_label();

        self.emit_aux(quote! { let #loop_start = __builder.new_label(); });
        self.emit_aux(quote! { let #loop_end = __builder.new_label(); });
        self.emit_label_def(&loop_start);

        let body_seq = self.lower_loop_body(&expr_loop.body, &loop_end, &loop_start)?;
        self.append_lowered_sequence(body_seq);

        self.emit_jump(&loop_start);
        self.emit_label_def(&loop_end);
        Some(())
    }

    /// Lower a small literal-range `for` loop by proc-macro-time unrolling.
    ///
    /// Only `for ident in START..END { ... }`, `for ident in START..=END { ... }`,
    /// and wildcard variants are accepted.  Other iterator protocol shapes
    /// still return `None` so the containing arm falls back unchanged.
    pub(super) fn lower_for_loop(&mut self, expr_for: &syn::ExprForLoop) -> Option<()> {
        let loop_var = match &*expr_for.pat {
            Pat::Ident(pat_ident) if pat_ident.subpat.is_none() => Some(pat_ident.ident.clone()),
            Pat::Wild(_) => None,
            _ => return None,
        };

        let Expr::Range(range) = &*expr_for.expr else {
            return None;
        };
        let start = literal_nonnegative_i64(range.start.as_deref()?)?;
        let end = literal_nonnegative_i64(range.end.as_deref()?)?;
        // Compute the iteration count with checked arithmetic BEFORE
        // materializing the range.  `(start..end).collect()` on a huge
        // literal range (`for _ in 0..i64::MAX`) would exhaust memory
        // during macro expansion; bail as soon as the count is known to
        // exceed the unroll cap (or overflows i64 for a closed range).
        let count: Option<i64> = match range.limits {
            // `end > start >= 0` and both `<= i64::MAX`, so the difference
            // never overflows.
            syn::RangeLimits::HalfOpen(_) => Some((start < end).then(|| end - start).unwrap_or(0)),
            // Closed range spans `end - start + 1` values; the `+ 1` can
            // overflow when `end == i64::MAX`, so guard it.
            syn::RangeLimits::Closed(_) => {
                if start > end {
                    Some(0)
                } else {
                    (end - start).checked_add(1)
                }
            }
        };
        let Some(count) = count else {
            return None;
        };
        if count > 64 || block_has_loop_control(&expr_for.body) {
            return None;
        }
        if count == 0 {
            return Some(());
        }
        let values: Vec<i64> = match range.limits {
            syn::RangeLimits::HalfOpen(_) => (start..end).collect(),
            syn::RangeLimits::Closed(_) => (start..=end).collect(),
        };

        let snap_stmts = self.statements.len();
        let snap_meta = self.op_metadata.len();
        let snap_reg = self.next_reg;
        let snap_bindings = self.bindings.clone();

        for value in values {
            if let Some(loop_var) = &loop_var {
                let loop_let: Stmt = syn::parse_quote! {
                    let #loop_var = #value;
                };
                if self.lower_stmt(&loop_let).is_none() {
                    self.statements.truncate(snap_stmts);
                    self.op_metadata.truncate(snap_meta);
                    self.next_reg = snap_reg;
                    self.bindings = snap_bindings;
                    return None;
                }
            }
            for stmt in &expr_for.body.stmts {
                if self.lower_stmt(stmt).is_none() {
                    self.statements.truncate(snap_stmts);
                    self.op_metadata.truncate(snap_meta);
                    self.next_reg = snap_reg;
                    self.bindings = snap_bindings;
                    return None;
                }
            }
        }

        // Names `let`-bound at the top level of the loop body are scoped to
        // the body. Dropping the ones that did not exist before the loop is
        // handled by the `retain` below, but a `let` that SHADOWS an outer
        // binding must revert to the outer value rather than escaping with the
        // last iteration's inner binding. Assignments to existing locals are
        // not `let`s, so they are absent here and correctly persist.
        let body_let_names: Vec<String> = expr_for
            .body
            .stmts
            .iter()
            .filter_map(|stmt| match stmt {
                Stmt::Local(local) => match &local.pat {
                    Pat::Ident(pat_ident) => Some(pat_ident.ident.to_string()),
                    _ => None,
                },
                _ => None,
            })
            .collect();

        self.bindings
            .retain(|name, _| snap_bindings.contains_key(name));
        for name in &body_let_names {
            if let Some(outer_binding) = snap_bindings.get(name) {
                self.bindings.insert(name.clone(), outer_binding.clone());
            }
        }
        if let Some(loop_var) = &loop_var {
            let name = loop_var.to_string();
            if let Some(old_binding) = snap_bindings.get(&name) {
                self.bindings.insert(name, old_binding.clone());
            } else {
                self.bindings.remove(&name);
            }
        }
        Some(())
    }

    /// Lower a loop body block, translating `break` → jump to `break_label`
    /// and `continue` → jump to `continue_label`.
    fn lower_loop_body(
        &mut self,
        block: &syn::Block,
        break_label: &syn::Ident,
        continue_label: &syn::Ident,
    ) -> Option<LoweredSequence> {
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

        for stmt in &block.stmts {
            if nested
                .lower_loop_stmt(stmt, break_label, continue_label)
                .is_none()
            {
                // Fall back: try normal lowering
                nested.lower_stmt(stmt)?;
            }
        }

        self.next_reg = self.next_reg.max(nested.next_reg);
        self.next_label = self.next_label.max(nested.next_label);
        Some(LoweredSequence::new(nested.statements, nested.op_metadata))
    }

    /// Lower a statement inside a loop body, handling break/continue specially.
    fn lower_loop_stmt(
        &mut self,
        stmt: &Stmt,
        break_label: &syn::Ident,
        continue_label: &syn::Ident,
    ) -> Option<()> {
        match stmt {
            Stmt::Expr(Expr::Break(_), _) => {
                self.emit_jump(&break_label);
                Some(())
            }
            Stmt::Expr(Expr::Continue(_), _) => {
                self.emit_jump(&continue_label);
                Some(())
            }
            Stmt::Expr(Expr::If(expr_if), _) => {
                self.lower_loop_if(expr_if, break_label, continue_label)
            }
            _ => None,
        }
    }

    /// Lower an if-expression inside a loop body, where branches may
    /// contain break/continue.
    fn lower_loop_if(
        &mut self,
        expr_if: &ExprIf,
        break_label: &syn::Ident,
        continue_label: &syn::Ident,
    ) -> Option<()> {
        // Check if any branch contains break or continue
        let then_has_loop_ctrl = block_has_loop_control(&expr_if.then_branch);
        let else_has_loop_ctrl = expr_if
            .else_branch
            .as_ref()
            .is_some_and(|(_, e)| expr_has_loop_control(e));

        if !then_has_loop_ctrl && !else_has_loop_ctrl {
            return None; // no break/continue, fall back to normal lowering
        }

        let snap_stmts = self.statements.len();
        let snap_meta = self.op_metadata.len();
        let snap_reg = self.next_reg;
        let snap_label = self.next_label;
        let snap_bindings = self.bindings.clone();
        let cond = self.lower_value_expr(&expr_if.cond)?;
        if self.ops_since_contain_float_compare(snap_meta) {
            // Guard over a float comparison hangs the compiled trace; roll
            // back and bail so the arm runs interpreted instead.
            self.statements.truncate(snap_stmts);
            self.op_metadata.truncate(snap_meta);
            self.next_reg = snap_reg;
            self.next_label = snap_label;
            self.bindings = snap_bindings;
            return None;
        }
        let else_label = self.alloc_label();
        let end_label = self.alloc_label();
        let cond_reg = cond.reg;

        self.emit_aux(quote! { let #else_label = __builder.new_label(); });
        self.emit_aux(quote! { let #end_label = __builder.new_label(); });
        self.emit_op(
            OpMeta::live_marker(),
            quote! { let _ = __builder.live_placeholder(); },
        );
        self.emit_conditional_guard(cond_reg, &else_label);

        // Lower then-branch with loop control
        let then_seq = self.lower_loop_body(&expr_if.then_branch, break_label, continue_label)?;
        self.append_lowered_sequence(then_seq);
        self.emit_jump(&end_label);
        self.emit_label_def(&else_label);

        // Lower else-branch with loop control
        if let Some((_, else_expr)) = &expr_if.else_branch {
            let else_block = match &**else_expr {
                Expr::Block(block) => &block.block,
                _ => return None,
            };
            let else_seq = self.lower_loop_body(else_block, break_label, continue_label)?;
            self.append_lowered_sequence(else_seq);
        }

        self.emit_label_def(&end_label);
        Some(())
    }

    /// Lower a match expression in value position to chained if-else guards
    /// that produce a value.
    pub(super) fn lower_match_value(&mut self, expr_match: &syn::ExprMatch) -> Option<Binding> {
        let discriminant = self.lower_value_expr(&expr_match.expr)?;
        if !matches!(discriminant.kind, BindingKind::Int) {
            return None;
        }

        let end_label = self.alloc_label();
        let result_reg = self.alloc_reg();
        self.emit_aux(quote! { let #end_label = __builder.new_label(); });

        let mut guarded_arms = Vec::new();
        let mut default_arm = None;
        let mut depends_on_stack = discriminant.depends_on_stack;

        for arm in &expr_match.arms {
            match &arm.pat {
                Pat::Wild(_) => {
                    default_arm = Some(&arm.body);
                }
                Pat::Ident(pat_ident) if pat_ident.subpat.is_none() => {
                    default_arm = Some(&arm.body);
                }
                _ => {
                    let literals = extract_pat_literals(&arm.pat)?;
                    guarded_arms.push((literals, &arm.body));
                }
            }
        }

        let disc_reg = discriminant.reg;

        for (literals, body) in &guarded_arms {
            let next_label = self.alloc_label();
            self.emit_aux(quote! { let #next_label = __builder.new_label(); });

            if literals.len() == 1 {
                let value = literals[0];
                let const_reg = self.alloc_reg();
                let eq_reg = self.alloc_reg();
                self.emit_op(
                    OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(const_reg)]),
                    quote! { __builder.load_const_i_value(#const_reg, #value); },
                );
                self.emit_op(
                    OpMeta::linear(
                        OpKind::BinopI,
                        Register::ints(&[disc_reg, const_reg]),
                        vec![Register::int(eq_reg)],
                    ),
                    quote! { __builder.record_binop_i(#eq_reg, majit_ir::OpCode::IntEq, #disc_reg, #const_reg); },
                );
                self.emit_op(
                    OpMeta::live_marker(),
                    quote! { let _ = __builder.live_placeholder(); },
                );
                self.emit_conditional_guard(eq_reg, &next_label);
            } else {
                let first_val = literals[0];
                let first_const_reg = self.alloc_reg();
                let mut or_reg = self.alloc_reg();
                self.emit_op(
                    OpMeta::linear(
                        OpKind::LoadConstI,
                        vec![],
                        vec![Register::int(first_const_reg)],
                    ),
                    quote! { __builder.load_const_i_value(#first_const_reg, #first_val); },
                );
                self.emit_op(
                    OpMeta::linear(
                        OpKind::BinopI,
                        Register::ints(&[disc_reg, first_const_reg]),
                        vec![Register::int(or_reg)],
                    ),
                    quote! { __builder.record_binop_i(#or_reg, majit_ir::OpCode::IntEq, #disc_reg, #first_const_reg); },
                );
                for &lit_val in &literals[1..] {
                    let const_reg = self.alloc_reg();
                    let eq_reg = self.alloc_reg();
                    let new_or_reg = self.alloc_reg();
                    self.emit_op(
                        OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(const_reg)]),
                        quote! { __builder.load_const_i_value(#const_reg, #lit_val); },
                    );
                    self.emit_op(
                    OpMeta::linear(
                        OpKind::BinopI,
                        Register::ints(&[disc_reg, const_reg]),
                        vec![Register::int(eq_reg)],
                    ),
                    quote! { __builder.record_binop_i(#eq_reg, majit_ir::OpCode::IntEq, #disc_reg, #const_reg); },
                );
                    self.emit_op(
                        OpMeta::linear(
                            OpKind::BinopI,
                            Register::ints(&[or_reg, eq_reg]),
                            vec![Register::int(new_or_reg)],
                        ),
                        quote! { __builder.record_binop_i(#new_or_reg, majit_ir::OpCode::IntOr, #or_reg, #eq_reg); },
                    );
                    or_reg = new_or_reg;
                }
                self.emit_op(
                    OpMeta::live_marker(),
                    quote! { let _ = __builder.live_placeholder(); },
                );
                self.emit_conditional_guard(or_reg, &next_label);
            }

            let (body_seq, binding) = self.lower_branch_value_expr(body)?;
            if !matches!(binding.kind, BindingKind::Int) {
                return None;
            }
            depends_on_stack |= binding.depends_on_stack;
            let arm_reg = binding.reg;
            self.append_lowered_sequence(body_seq);
            self.emit_op(
                OpMeta::linear(
                    OpKind::MoveI,
                    vec![Register::int(arm_reg)],
                    vec![Register::int(result_reg)],
                ),
                quote! { __builder.move_i(#result_reg, #arm_reg); },
            );
            self.emit_jump(&end_label);
            self.emit_label_def(&next_label);
        }

        // Default arm
        if let Some(default_body) = default_arm {
            let (default_seq, default_binding) = self.lower_branch_value_expr(default_body)?;
            if !matches!(default_binding.kind, BindingKind::Int) {
                return None;
            }
            depends_on_stack |= default_binding.depends_on_stack;
            let default_reg = default_binding.reg;
            self.append_lowered_sequence(default_seq);
            self.emit_op(
                OpMeta::linear(
                    OpKind::MoveI,
                    vec![Register::int(default_reg)],
                    vec![Register::int(result_reg)],
                ),
                quote! { __builder.move_i(#result_reg, #default_reg); },
            );
        }

        self.emit_label_def(&end_label);

        Some(Binding {
            reg: result_reg,
            kind: BindingKind::Int,
            depends_on_stack,
            struct_type: None,
        })
    }

    /// Overflow-checked arithmetic value match — the orthodox `ovfcheck`
    /// idiom (`match a.checked_add(b) { Some(v) => v, None => <handler> }`).
    /// Returns `None` when `expr_match` is not this idiom (the caller then
    /// tries the generic value-match path); returns `Some(result)` once
    /// committed, where `result` is `None` only on a hard lowering failure.
    pub(super) fn lower_checked_ovf_match(
        &mut self,
        expr_match: &syn::ExprMatch,
    ) -> Option<Option<Binding>> {
        let (builder_method, recv, arg, none_body) = parse_checked_ovf_match(expr_match)?;
        Some(self.emit_checked_ovf_match(builder_method, recv, arg, none_body))
    }

    fn emit_checked_ovf_match(
        &mut self,
        builder_method: &str,
        recv: &Expr,
        arg: &Expr,
        none_body: &Expr,
    ) -> Option<Binding> {
        let lhs = self.lower_value_expr(recv)?;
        let rhs = self.lower_value_expr(arg)?;
        if !matches!(lhs.kind, BindingKind::Int) || !matches!(rhs.kind, BindingKind::Int) {
            return None;
        }
        // Lower the overflow (None) arm into a deferred sequence up front so its
        // result register is known for the convergence move.
        let (none_seq, none_binding) = self.lower_branch_value_expr(none_body)?;
        if !matches!(none_binding.kind, BindingKind::Int) {
            return None;
        }

        let dst = self.alloc_reg();
        let ovf_label = self.alloc_label();
        let end_label = self.alloc_label();
        let builder_ident = format_ident!("{builder_method}");
        let lhs_reg = lhs.reg;
        let rhs_reg = rhs.reg;
        let none_reg = none_binding.reg;

        self.emit_aux(quote! { let #ovf_label = __builder.new_label(); });
        self.emit_aux(quote! { let #end_label = __builder.new_label(); });
        // `-live-` precedes the fused guard so blackhole liveness decodes at the
        // op's resume position (see `lower_if_stmt`).
        self.emit_op(
            OpMeta::live_marker(),
            quote! { let _ = __builder.live_placeholder(); },
        );
        self.emit_op(
            OpMeta::int_binop_jump_if_ovf(
                Register::int(lhs_reg),
                Register::int(rhs_reg),
                Register::int(dst),
                ovf_label.clone(),
            ),
            quote! { __builder.#builder_ident(#dst, #lhs_reg, #rhs_reg, #ovf_label); },
        );
        // No-overflow path: `dst` holds the result; skip the overflow arm.
        self.emit_jump(&end_label);
        // Overflow path: run the None arm and converge its result into `dst`.
        self.emit_label_def(&ovf_label);
        self.append_lowered_sequence(none_seq);
        self.emit_op(
            OpMeta::linear(
                OpKind::MoveI,
                vec![Register::int(none_reg)],
                vec![Register::int(dst)],
            ),
            quote! { __builder.move_i(#dst, #none_reg); },
        );
        self.emit_label_def(&end_label);

        Some(Binding {
            reg: dst,
            kind: BindingKind::Int,
            depends_on_stack: lhs.depends_on_stack
                || rhs.depends_on_stack
                || none_binding.depends_on_stack,
            struct_type: None,
        })
    }
}

/// Overflow (`None`) vs no-overflow (`Some`) variant of an option pattern.
enum OptionPatVariant {
    Some,
    None,
}

fn option_variant_of_pat(pat: &Pat) -> Option<OptionPatVariant> {
    let ident = match pat {
        Pat::TupleStruct(ts) => ts.path.segments.last()?.ident.to_string(),
        Pat::Path(p) => p.path.segments.last()?.ident.to_string(),
        Pat::Ident(pi) if pi.subpat.is_none() => pi.ident.to_string(),
        _ => return None,
    };
    match ident.as_str() {
        "Some" => Some(OptionPatVariant::Some),
        "None" => Some(OptionPatVariant::None),
        _ => None,
    }
}

/// Recognize the orthodox `ovfcheck` idiom
/// `match recv.checked_{add,sub,mul}(arg) { Some(_) => _, None => <handler> }`.
/// Returns the fused builder method name, the two operand expressions, and the
/// overflow (`None`) arm body; `None` when `expr_match` is not this shape.
fn parse_checked_ovf_match(
    expr_match: &syn::ExprMatch,
) -> Option<(&'static str, &Expr, &Expr, &Expr)> {
    let call = match &*expr_match.expr {
        Expr::MethodCall(call) => call,
        _ => return None,
    };
    let builder_method = match call.method.to_string().as_str() {
        "checked_add" => "int_add_jump_if_ovf",
        "checked_sub" => "int_sub_jump_if_ovf",
        "checked_mul" => "int_mul_jump_if_ovf",
        _ => return None,
    };
    if call.args.len() != 1 {
        return None;
    }
    let recv = &*call.receiver;
    let arg = call.args.first()?;
    // Exactly two guard-less arms: `Some(_) => _` and `None => <handler>`.
    if expr_match.arms.len() != 2 {
        return None;
    }
    let mut some_seen = false;
    let mut none_body: Option<&Expr> = None;
    for arm in &expr_match.arms {
        if arm.guard.is_some() {
            return None;
        }
        match option_variant_of_pat(&arm.pat)? {
            OptionPatVariant::Some => some_seen = true,
            OptionPatVariant::None => none_body = Some(&*arm.body),
        }
    }
    if !some_seen {
        return None;
    }
    Some((builder_method, recv, arg, none_body?))
}

#[cfg(test)]
mod unroll_binding_tests {
    use super::*;

    fn int_binding(reg: u16) -> Binding {
        Binding {
            reg,
            kind: BindingKind::Int,
            depends_on_stack: false,
            struct_type: None,
        }
    }

    #[test]
    fn unrolled_body_let_shadow_reverts_to_outer_binding() {
        let mut lowerer = Lowerer::new(None);
        lowerer.bindings.insert("x".to_string(), int_binding(42));
        let expr_for: syn::ExprForLoop = syn::parse_quote! {
            for _ in 0..2 {
                let x = 7;
            }
        };
        assert!(
            lowerer.lower_for_loop(&expr_for).is_some(),
            "literal-range unroll must succeed"
        );
        // The body `let x` shadows the outer `x`; its binding is scoped to the
        // loop body, so after the loop `x` must be the outer binding (reg 42),
        // not the last iteration's inner `let`.
        assert_eq!(
            lowerer.bindings.get("x").map(|b| b.reg),
            Some(42),
            "a shadowed outer binding must be restored after unrolling"
        );
    }

    #[test]
    fn unrolled_body_let_without_outer_is_removed() {
        let mut lowerer = Lowerer::new(None);
        let expr_for: syn::ExprForLoop = syn::parse_quote! {
            for _ in 0..2 {
                let y = 7;
            }
        };
        assert!(
            lowerer.lower_for_loop(&expr_for).is_some(),
            "literal-range unroll must succeed"
        );
        // `y` did not exist before the loop, so its body-local binding must not
        // escape.
        assert!(
            lowerer.bindings.get("y").is_none(),
            "a body-local let must not escape the loop"
        );
    }
}
