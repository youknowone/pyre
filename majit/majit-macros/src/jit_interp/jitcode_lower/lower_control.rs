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
    /// Lower an `if` / `while` condition, peeling a leading `!` into the
    /// branch instead of computing it.
    ///
    /// `jtransform.py` renames `bool_not` to `int_is_zero`, and
    /// `optimize_goto_if_not` then folds that operation into the block's
    /// exitswitch, so `if not x` reaches flatten as one
    /// `goto_if_not_int_is_zero` and no separate negation op.
    ///
    /// The peel belongs to condition position and nowhere else. `BindingKind`
    /// carries no `bool`, and `!` is logical on a Rust `bool` but bitwise on
    /// an integer, so a general `UnOp::Not` arm could not tell the two apart
    /// and does not exist. A condition is `bool` by the language's own typing
    /// rule, which is the guarantee `optimize_goto_if_not` spells out as
    /// `v.concretetype != lltype.Bool: return False`.
    pub(super) fn lower_condition(&mut self, cond: &Expr) -> Option<LoweredCondition> {
        let mut negated = false;
        let mut expr = cond;
        loop {
            match expr {
                Expr::Paren(paren) => expr = &paren.expr,
                Expr::Unary(ExprUnary {
                    op: UnOp::Not(_),
                    expr: inner,
                    ..
                }) => {
                    negated = !negated;
                    expr = inner;
                }
                _ => break,
            }
        }

        // RPython `jtransform.py` `optimize_goto_if_not`: when the boolean
        // comparison result is used only as this block's exitswitch, remove
        // the value-producing op and carry its operands in the exitswitch.
        // Flatten then emits `goto_if_not_<opname>/{ii,ff}L`.
        //
        // Do not fuse through `!`: `!(a < b)` is not `a >= b` for NaNs.
        if !negated {
            let fused = self.transactional(|s| {
                let Expr::Binary(binary) = expr else {
                    return None;
                };
                let suffix = match &binary.op {
                    BinOp::Lt(_) => "lt",
                    BinOp::Le(_) => "le",
                    BinOp::Eq(_) => "eq",
                    BinOp::Ne(_) => "ne",
                    BinOp::Gt(_) => "gt",
                    BinOp::Ge(_) => "ge",
                    _ => return None,
                };
                let lhs = s.lower_value_expr(&binary.left)?;
                let rhs = s.lower_value_expr(&binary.right)?;
                if lhs.kind != rhs.kind
                    || !matches!(lhs.kind, BindingKind::Int | BindingKind::Float)
                {
                    return None;
                }
                // `_rewrite_equality` then `optimize_goto_if_not`:
                // `int_eq(x, 0)` → `int_is_zero` → `goto_if_not_int_is_zero`.
                if matches!(lhs.kind, BindingKind::Int)
                    && matches!(binary.op, BinOp::Eq(_) | BinOp::Ne(_))
                {
                    let left_zero = int_literal_value(&binary.left) == Some(0);
                    let right_zero = int_literal_value(&binary.right) == Some(0);
                    if left_zero || right_zero {
                        let value = if right_zero { lhs } else { rhs };
                        let negated = matches!(binary.op, BinOp::Eq(_));
                        return Some(LoweredCondition::Value {
                            binding: value,
                            negated,
                            int_is_true: true,
                        });
                    }
                }
                let prefix = match lhs.kind {
                    BindingKind::Int => "goto_if_not_int_",
                    BindingKind::Float => "goto_if_not_float_",
                    BindingKind::Ref => unreachable!(),
                };
                Some(LoweredCondition::Compare {
                    lhs,
                    rhs,
                    branch: format_ident!("{prefix}{suffix}"),
                })
            });
            if fused.is_some() {
                return fused;
            }
        }

        let binding = self.lower_value_expr(expr)?;
        // The int bank is what `goto_if_not_int_is_zero` reads. An unnegated
        // condition keeps whatever the caller already accepted.
        if negated && !matches!(binding.kind, BindingKind::Int) {
            return None;
        }
        Some(LoweredCondition::Value {
            binding,
            negated,
            int_is_true: false,
        })
    }

    pub(super) fn lower_if_stmt(&mut self, expr_if: &ExprIf) -> Option<()> {
        let cond = self.lower_condition(&expr_if.cond)?;
        let then_seq = self.lower_branch_expr(&Expr::Block(syn::ExprBlock {
            attrs: Vec::new(),
            label: None,
            block: expr_if.then_branch.clone(),
        }))?;
        let else_seq = match expr_if.else_branch.as_ref() {
            Some((_, else_expr)) => self.lower_branch_expr(else_expr)?,
            None => LoweredSequence::default(),
        };

        // Allocated below the branch lowerings, not above them: both labels are
        // forward targets, so where they are defined carries no meaning, and
        // either `?` above would otherwise return with `next_label` advanced.
        let else_label = self.alloc_label();
        let end_label = self.alloc_label();
        self.emit_aux(quote! { let #else_label = __builder.new_label(); });
        self.emit_aux(quote! { let #end_label = __builder.new_label(); });
        // RPython `flatten.py:259` `-live-` convention: every guard-bearing
        // instruction is *preceded* by a `live` marker (byte order:
        // `BC_LIVE+offset` then the guard op). The recorded `orgpc` (=
        // RPython `pyjitpl.py orgpc = position`, copied to the guard's
        // `resumepc` via `record_state_guard`) is the byte position of the
        // guard op itself, so the BC_LIVE marker sits at `orgpc - SIZE_LIVE_OP`
        // and blackhole's `get_current_position_info` reads liveness from
        // there.  Without this preceding marker, blackhole panics with
        // `missing liveness[N] in JitCode`.
        self.emit_op(
            OpMeta::live_marker(),
            quote! { let _ = __builder.live_placeholder(); },
        );
        self.emit_lowered_condition_guard(&cond, &else_label);
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
        self.transactional(|s| s.lower_match_stmt_inner(expr_match))
    }

    fn lower_match_stmt_inner(&mut self, expr_match: &syn::ExprMatch) -> Option<()> {
        let discriminant = self.lower_value_expr(&expr_match.expr)?;
        if !matches!(discriminant.kind, BindingKind::Int) {
            return None;
        }

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
                _ if is_lowercase_binding_pat(&arm.pat) => {
                    default_arm = Some(&arm.body);
                }
                _ => {
                    let tokens = extract_pat_value_tokens(&arm.pat)?;
                    guarded_arms.push((tokens, &arm.body));
                }
            }
        }

        // Allocated below the arm classification, not above it: `end_label` is
        // a forward target, so where it is defined carries no meaning, and the
        // `?`s above would otherwise return with `next_label` advanced.  The
        // classification emits no ops, so the statement stream is unchanged.
        let end_label = self.alloc_label();
        self.emit_aux(quote! { let #end_label = __builder.new_label(); });

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
        self.transactional(|s| s.lower_while_loop_inner(expr_while))
    }

    fn lower_while_loop_inner(&mut self, expr_while: &syn::ExprWhile) -> Option<()> {
        // `loop_start` has to be marked *before* the condition lowers:
        // `mark_label` records the builder's current bytecode position, the
        // back edge re-enters there, and the condition must be re-tested on
        // every iteration.  So these allocations cannot be deferred past the
        // first fallible call the way `lower_if_with_loop_control` defers its
        // own — both of that function's labels are forward targets. Every exit
        // below leaves two labels and three statements emitted, which is what
        // the `transactional` wrapper puts back.
        let loop_start = self.alloc_label();
        let loop_end = self.alloc_label();

        self.emit_aux(quote! { let #loop_start = __builder.new_label(); });
        self.emit_aux(quote! { let #loop_end = __builder.new_label(); });
        self.emit_label_def(&loop_start);

        // Evaluate the condition
        let cond = self.lower_condition(&expr_while.cond)?;
        self.emit_op(
            OpMeta::live_marker(),
            quote! { let _ = __builder.live_placeholder(); },
        );
        self.emit_lowered_condition_guard(&cond, &loop_end);

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
        self.transactional(|s| s.lower_loop_expr_inner(expr_loop))
    }

    fn lower_loop_expr_inner(&mut self, expr_loop: &syn::ExprLoop) -> Option<()> {
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
        self.transactional(|s| s.lower_for_loop_inner(expr_for))
    }

    fn lower_for_loop_inner(&mut self, expr_for: &syn::ExprForLoop) -> Option<()> {
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
            syn::RangeLimits::HalfOpen(_) => Some(if start < end { end - start } else { 0 }),
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
        let count = count?;
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
            body_failure_reason: None,
            nested_failure_reasons: Vec::new(),
            opcode_var_name: self.opcode_var_name.clone(),
            in_dispatch_arm_body: self.in_dispatch_arm_body,
            // Never inherited: inside a loop body an unlabelled `continue`
            // binds to that loop, not to the dispatch back-edge.  The
            // spellings this block lowers itself -- a bare `continue` and one
            // in an `if` branch -- are taken by `lower_loop_stmt` /
            // `lower_loop_if` with `continue_label`; any other spelling falls
            // back to `lower_stmt`, whose `Expr::Continue` arm emits a jump to
            // `dispatch_loop_label`.  Carrying the dispatch label in would aim
            // that jump at the dispatch head instead of this loop.  Cleared, it
            // refuses there.
            dispatch_loop_label: None,
            pc_pinned: self.pc_pinned,
            // Never inherited: a loop body statement is not the arm body's
            // tail, so a `return` inside it must be rejected, not lowered.
            inline_arm_tail_stmt: false,
        };

        for stmt in &block.stmts {
            if nested
                .lower_loop_stmt(stmt, break_label, continue_label)
                .is_none()
            {
                // Fall back: try normal lowering
                if nested.lower_stmt(stmt).is_none() {
                    // Carry the diagnosis out before the child is dropped; a
                    // bare `?` here propagates the failure and loses the reason.
                    self.absorb_nested_failure(&mut nested);
                    return None;
                }
            }
        }

        self.next_reg = self.next_reg.max(nested.next_reg);
        self.next_label = self.next_label.max(nested.next_label);
        self.inline_liveness_prebuild
            .extend(nested.inline_liveness_prebuild);
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
                self.emit_jump(break_label);
                Some(())
            }
            Stmt::Expr(Expr::Continue(cont), _) => {
                if cont.label.is_some() {
                    return None;
                }
                self.emit_jump(continue_label);
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
        self.transactional(|s| s.lower_loop_if_inner(expr_if, break_label, continue_label))
    }

    fn lower_loop_if_inner(
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

        let cond = self.lower_condition(&expr_if.cond)?;
        let else_label = self.alloc_label();
        let end_label = self.alloc_label();

        self.emit_aux(quote! { let #else_label = __builder.new_label(); });
        self.emit_aux(quote! { let #end_label = __builder.new_label(); });
        self.emit_op(
            OpMeta::live_marker(),
            quote! { let _ = __builder.live_placeholder(); },
        );
        self.emit_lowered_condition_guard(&cond, &else_label);

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
        self.transactional(|s| s.lower_match_value_inner(expr_match))
    }

    fn lower_match_value_inner(&mut self, expr_match: &syn::ExprMatch) -> Option<Binding> {
        let discriminant = self.lower_value_expr(&expr_match.expr)?;
        if !matches!(discriminant.kind, BindingKind::Int) {
            return None;
        }

        let result_reg = self.alloc_reg();

        let mut guarded_arms = Vec::new();
        let mut default_arm = None;
        let mut depends_on_stack = discriminant.depends_on_stack;

        for arm in &expr_match.arms {
            match &arm.pat {
                Pat::Wild(_) => {
                    default_arm = Some(&arm.body);
                }
                // syn parses `OP_NOP => ..` and `other => ..` identically, so
                // the name is all there is to go on, and `is_lowercase_binding_pat`
                // is the one place that decides it.
                _ if is_lowercase_binding_pat(&arm.pat) => {
                    default_arm = Some(&arm.body);
                }
                _ => {
                    let values = extract_pat_value_tokens(&arm.pat)?;
                    guarded_arms.push((values, &arm.body));
                }
            }
        }

        // Allocated below the arm classification, not above it: `end_label` is
        // a forward target, so where it is defined carries no meaning, and the
        // `?` above would otherwise return with `next_label` advanced.  The
        // classification emits no ops, so the statement stream is unchanged.
        let end_label = self.alloc_label();
        self.emit_aux(quote! { let #end_label = __builder.new_label(); });

        let disc_reg = discriminant.reg;

        for (values, body) in &guarded_arms {
            let next_label = self.alloc_label();
            self.emit_aux(quote! { let #next_label = __builder.new_label(); });

            if values.len() == 1 {
                let value = &values[0];
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
                let first_val = &values[0];
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
                for lit_val in &values[1..] {
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
        let parsed = parse_checked_ovf_match(expr_match)?;
        Some(self.emit_checked_ovf_match(parsed))
    }

    fn emit_checked_ovf_match(&mut self, parsed: CheckedOvfMatch<'_>) -> Option<Binding> {
        let lhs = self.lower_value_expr(parsed.recv)?;
        let rhs = self.lower_value_expr(parsed.arg)?;
        if !matches!(lhs.kind, BindingKind::Int) || !matches!(rhs.kind, BindingKind::Int) {
            return None;
        }
        let (none_seq, none_binding) = self.lower_branch_value_expr(parsed.none_body)?;
        if !matches!(none_binding.kind, BindingKind::Int) {
            return None;
        }

        let dst = self.alloc_reg();
        let ovf_label = self.alloc_label();
        let end_label = self.alloc_label();
        let builder_ident = format_ident!("{}", parsed.builder_method);
        let lhs_reg = lhs.reg;
        let rhs_reg = rhs.reg;
        let none_reg = none_binding.reg;

        self.emit_aux(quote! { let #ovf_label = __builder.new_label(); });
        self.emit_aux(quote! { let #end_label = __builder.new_label(); });
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
        if !some_arm_is_identity(parsed.some_pat, parsed.some_body) {
            let saved = self.bindings.clone();
            if let Some(name) = some_pat_bound_name(parsed.some_pat) {
                self.bindings.insert(
                    name,
                    Binding {
                        reg: dst,
                        kind: BindingKind::Int,
                        depends_on_stack: false,
                        struct_type: None,
                    },
                );
            }
            let lowered = self.lower_branch_value_expr(parsed.some_body);
            self.bindings = saved;
            let (seq, binding) = lowered?;
            if !matches!(binding.kind, BindingKind::Int) {
                return None;
            }
            self.append_lowered_sequence(seq);
            if binding.reg != dst {
                let some_reg = binding.reg;
                self.emit_op(
                    OpMeta::linear(
                        OpKind::MoveI,
                        vec![Register::int(some_reg)],
                        vec![Register::int(dst)],
                    ),
                    quote! { __builder.move_i(#dst, #some_reg); },
                );
            }
        }
        self.emit_jump(&end_label);
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

struct CheckedOvfMatch<'a> {
    builder_method: &'static str,
    recv: &'a Expr,
    arg: &'a Expr,
    none_body: &'a Expr,
    some_pat: &'a Pat,
    some_body: &'a Expr,
}

fn some_pat_bound_name(pat: &Pat) -> Option<String> {
    let Pat::TupleStruct(ts) = pat else {
        return None;
    };
    if ts.elems.len() != 1 {
        return None;
    }
    match &ts.elems[0] {
        Pat::Ident(pi) if pi.subpat.is_none() => Some(pi.ident.to_string()),
        _ => None,
    }
}

/// `Some(v) => v` is RPython `ovfcheck`'s success edge: the residual is
/// the overflow-checked add itself. Any other success body must be
/// lowered; treating it as identity miscompiles.
fn some_arm_is_identity(pat: &Pat, body: &Expr) -> bool {
    let Some(name) = some_pat_bound_name(pat) else {
        return false;
    };
    match body {
        Expr::Path(p) => p
            .path
            .get_ident()
            .is_some_and(|ident| ident == name.as_str()),
        Expr::Block(block) if block.block.stmts.len() == 1 => match &block.block.stmts[0] {
            Stmt::Expr(Expr::Path(p), None) => p
                .path
                .get_ident()
                .is_some_and(|ident| ident == name.as_str()),
            _ => false,
        },
        _ => false,
    }
}

/// Recognize `match recv.checked_{add,sub,mul}(arg) { Some(..) => .., None => .. }`.
fn parse_checked_ovf_match(expr_match: &syn::ExprMatch) -> Option<CheckedOvfMatch<'_>> {
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
    if expr_match.arms.len() != 2 {
        return None;
    }
    let mut some_arm = None;
    let mut none_body = None;
    for arm in &expr_match.arms {
        if arm.guard.is_some() {
            return None;
        }
        match option_variant_of_pat(&arm.pat)? {
            OptionPatVariant::Some => some_arm = Some((&arm.pat, &*arm.body)),
            OptionPatVariant::None => none_body = Some(&*arm.body),
        }
    }
    let (some_pat, some_body) = some_arm?;
    Some(CheckedOvfMatch {
        builder_method,
        recv,
        arg,
        none_body: none_body?,
        some_pat,
        some_body,
    })
}

#[cfg(test)]
mod unroll_binding_tests {
    use super::*;
    use crate::jit_interp::jitcode_lower::dispatch::InlineArmOutcome;

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
            !lowerer.bindings.contains_key("y"),
            "a body-local let must not escape the loop"
        );
    }

    #[test]
    fn unlowerable_while_condition_restores_the_label_snapshot() {
        let mut lowerer = Lowerer::new(None);
        lowerer.next_label = 17;
        let expr: syn::ExprWhile = syn::parse_quote! {
            while unsupported_condition() {}
        };

        assert!(lowerer.lower_while_loop(&expr).is_none());
        assert_eq!(lowerer.next_label, 17);
        assert!(lowerer.statements.is_empty());
        assert!(lowerer.op_metadata.is_empty());
    }

    #[test]
    fn failed_if_and_match_classification_do_not_consume_forward_labels() {
        let mut lowerer = Lowerer::new(None);
        lowerer.next_label = 23;

        let expr_if: syn::ExprIf = syn::parse_quote! {
            if 1 { unsupported_body() }
        };
        assert!(lowerer.lower_if_stmt(&expr_if).is_none());
        assert_eq!(lowerer.next_label, 23);

        let expr_match: syn::ExprMatch = syn::parse_quote! {
            match 0 {
                (1, 2) => {},
                _ => {},
            }
        };
        assert!(lowerer.lower_match_stmt(&expr_match).is_none());
        assert_eq!(lowerer.next_label, 23);
    }

    #[test]
    fn failed_inline_dispatch_arm_restores_allocated_labels() {
        let mut lowerer = Lowerer::new(None);
        lowerer.next_label = 31;
        let body: Expr = syn::parse_quote! {{
            match 0 {
                0 => unsupported_body(),
                _ => 1,
            };
        }};

        assert_eq!(
            lowerer.try_inline_dispatch_arm(&body),
            InlineArmOutcome::Rejected,
        );
        assert_eq!(lowerer.next_label, 31);
        assert!(lowerer.statements.is_empty());
        assert!(lowerer.op_metadata.is_empty());
    }

    /// The classification test above covers a match that refuses before it
    /// emits. This is the other half: the discriminant lowers, the arm guard
    /// and its two `new_label()`s are already in the stream, and then the arm
    /// body refuses. What the stream must not keep is a label created without
    /// a `mark_label` — the jitcode assembler cannot see that until it patches
    /// a `goto` naming a position no block ever took.
    #[test]
    fn failed_match_arm_body_leaves_no_label_behind() {
        let mut lowerer = Lowerer::new(None);
        lowerer.next_label = 41;

        let expr_match: syn::ExprMatch = syn::parse_quote! {
            match 0 {
                1 => unsupported_body(),
                _ => {},
            }
        };
        assert!(lowerer.lower_match_stmt(&expr_match).is_none());
        assert_eq!(lowerer.next_label, 41);
        assert!(lowerer.statements.is_empty());
        assert!(lowerer.op_metadata.is_empty());
    }

    /// `not cond` is a different branch opname, not an operation plus a
    /// branch: `jtransform.py` renames `bool_not` to `int_is_zero` and
    /// `optimize_goto_if_not` folds it into the exitswitch.
    #[test]
    fn a_negated_condition_selects_the_is_zero_branch() {
        let mut lowerer = Lowerer::new(None);
        let stmt: Stmt = syn::parse_quote! { let flag = 1; };
        let Stmt::Local(local) = stmt else {
            unreachable!("parse_quote produced the requested let statement")
        };
        assert!(lowerer.lower_local(&local).is_some());

        let expr_if: syn::ExprIf = syn::parse_quote! { if !flag { } };
        assert!(lowerer.lower_if_stmt(&expr_if).is_some());

        let emitted = lowerer
            .statements
            .iter()
            .map(|tokens| tokens.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            emitted.contains("goto_if_not_int_is_zero"),
            "`if !flag` must branch on int_is_zero, got:\n{emitted}"
        );
        assert!(
            !emitted.contains("goto_if_not_int_is_true"),
            "the negation must replace the branch, not add one:\n{emitted}"
        );
    }

    /// A `continue` inside a `match` in a loop body binds to that loop, not
    /// to the dispatch back-edge.  `lower_loop_stmt` / `lower_loop_if` take
    /// the direct and `if`-branch spellings with the loop's own labels; a
    /// `match` arm falls back to `lower_stmt`, whose `Expr::Continue` arm
    /// answered it with a jump to `dispatch_loop_label`.  On the pre-fix
    /// lowerer this `while` lowers and its stream names the dispatch head —
    /// one loop out from the loop the source wrote.
    #[test]
    fn a_continue_in_a_match_inside_a_loop_never_targets_the_dispatch_head() {
        let mut lowerer = Lowerer::new(None);
        lowerer.dispatch_loop_label = Some(syn::Ident::new(
            "__l_dispatch",
            proc_macro2::Span::call_site(),
        ));
        let stmt: Stmt = syn::parse_quote! { let flag = 1; };
        let Stmt::Local(local) = stmt else {
            unreachable!("parse_quote produced the requested let statement")
        };
        assert!(lowerer.lower_local(&local).is_some());

        let expr: syn::ExprWhile = syn::parse_quote! {
            while flag {
                match flag {
                    1 => continue,
                    _ => {},
                }
            }
        };
        let lowered = lowerer.lower_while_loop(&expr);

        let emitted = lowerer
            .statements
            .iter()
            .map(|tokens| tokens.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            !emitted.contains("__l_dispatch"),
            "the inner loop's `continue` must not jump to the dispatch head:\n{emitted}"
        );
        assert!(
            lowered.is_none(),
            "a `continue` this loop cannot spell must refuse the loop, not \
             retarget it:\n{emitted}"
        );
    }

    #[test]
    fn typed_local_binds_like_the_unannotated_spelling() {
        let mut lowerer = Lowerer::new(None);
        let stmt: Stmt = syn::parse_quote! { let value: i64 = 7; };
        let Stmt::Local(local) = stmt else {
            unreachable!("parse_quote produced the requested let statement")
        };

        assert!(lowerer.lower_local(&local).is_some());
        let binding = lowerer
            .bindings
            .get("value")
            .expect("the typed local must create its identifier binding");
        assert!(matches!(binding.kind, BindingKind::Int));
    }

    #[test]
    fn a_labelled_continue_in_a_loop_refuses() {
        let mut lowerer = Lowerer::new(None);
        lowerer.bindings.insert(
            "flag".to_string(),
            Binding {
                reg: 0,
                kind: BindingKind::Int,
                depends_on_stack: false,
                struct_type: None,
            },
        );
        let expr: syn::ExprWhile = syn::parse_quote! {
            'outer: while flag {
                continue 'outer;
            }
        };
        assert!(
            lowerer.lower_while_loop(&expr).is_none(),
            "a labelled continue must not retarget the innermost header"
        );
    }

    #[test]
    fn if_eq_zero_fuses_to_goto_if_not_int_is_zero() {
        let mut lowerer = Lowerer::new(None);
        lowerer.bindings.insert(
            "n".to_string(),
            Binding {
                reg: 4,
                kind: BindingKind::Int,
                depends_on_stack: false,
                struct_type: None,
            },
        );
        let expr_if: syn::ExprIf = syn::parse_quote! { if n == 0 { } };
        assert!(lowerer.lower_if_stmt(&expr_if).is_some());
        let emitted = lowerer
            .statements
            .iter()
            .map(|tokens| tokens.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            emitted.contains("goto_if_not_int_is_zero"),
            "`if n == 0` must fuse to int_is_zero, got:\n{emitted}"
        );
        assert!(
            !emitted.contains("goto_if_not_int_eq"),
            "must not keep the binary compare:\n{emitted}"
        );
    }

    #[test]
    fn checked_add_some_body_is_lowered() {
        let mut lowerer = Lowerer::new(None);
        lowerer.bindings.insert(
            "a".to_string(),
            Binding {
                reg: 1,
                kind: BindingKind::Int,
                depends_on_stack: false,
                struct_type: None,
            },
        );
        lowerer.bindings.insert(
            "b".to_string(),
            Binding {
                reg: 2,
                kind: BindingKind::Int,
                depends_on_stack: false,
                struct_type: None,
            },
        );
        let expr: syn::ExprMatch = syn::parse_quote! {
            match a.checked_add(b) {
                Some(v) => v + 1,
                None => 0,
            }
        };
        let binding = lowerer
            .lower_checked_ovf_match(&expr)
            .expect("recognized")
            .expect("lowered");
        assert!(matches!(binding.kind, BindingKind::Int));
        let emitted = lowerer
            .statements
            .iter()
            .map(|tokens| tokens.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            emitted.contains("int_add_jump_if_ovf"),
            "expected ovf jump, got:\n{emitted}"
        );
        assert!(
            emitted.contains("IntAdd") || emitted.contains("int_add"),
            "Some(v) => v + 1 must emit the increment, got:\n{emitted}"
        );
    }
}
