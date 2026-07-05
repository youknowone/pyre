use super::*;

pub struct GeneratedJitCodeBody {
    pub body: TokenStream,
    pub liveness_prebuild: TokenStream,
    /// Slice (audit Issue #5) — green schema in declaration order,
    /// each pair is `(name, green_type_token)` where the token
    /// resolves to a `majit_ir::GreenType` variant at the install
    /// site (Int / Ref / Float / Void / Str / Unicode).  The
    /// dispatch path passes this to
    /// `JitDriver::declare_schema_typed` so
    /// `JitDriverStaticData::green_args_spec` reports STR/UNICODE
    /// where the user tagged `: str` / `: unicode` instead of
    /// collapsing to `GreenType::Ref`.  Per-arm bodies leave it
    /// empty.
    pub green_schema: Vec<(String, TokenStream)>,
    /// Red schema — `(name, ir_type_token)` resolving to a
    /// `majit_ir::Type` (Int / Ref / Float / Void).  Reds carry no
    /// upstream lltype subtype distinction (no `equal_whatever`
    /// dispatch on runtime args).
    pub red_schema: Vec<(String, TokenStream)>,
}

#[allow(dead_code)]
pub fn try_generate_jitcode_body(body: &Expr) -> Option<TokenStream> {
    try_generate_jitcode_body_inner(body, None).map(|p| p.body)
}

#[allow(dead_code)]
pub fn try_generate_jitcode_body_with_config(
    config: &LowererConfig,
    body: &Expr,
) -> Option<TokenStream> {
    try_generate_jitcode_body_inner(body, Some(config)).map(|p| p.body)
}

/// Per-caller-local layout descriptor produced by
/// [`try_generate_jitcode_body_parts_with_caller_bindings`].
///
/// The dispatch arm parent emit needs three things to construct the
/// `inline_call_<types>_v(__sub_idx, args_i, args_r, args_f)` call:
/// - the parent's reg (read from caller's `Binding`),
/// - the callee's portal-input reg (assigned per-bank by the sub-Lowerer
///   pre-bind pass),
/// - the bank (Int / Ref / Float) so the (parent, callee) pair lands in
///   the matching `args_<kind>` vector.
#[allow(private_interfaces)]
#[derive(Clone, Debug)]
pub(crate) struct CallerLocalLayout {
    #[allow(dead_code)]
    pub name: String,
    pub parent_reg: u16,
    pub callee_reg: u16,
    pub kind: BindingKind,
}

/// Dispatch arm caller-local plumbing.
///
/// Dispatch-arm lowering entry that pre-binds a list of caller-locals as
/// portal-input bindings on the sub-Lowerer before lowering the body.
/// The caller (slice 1.3 — dispatch arm emit at `lower_dispatch_chain`)
/// collects them via
/// [`collect_arm_caller_locals`] and threads the same list through
/// `inline_call_<types>_v` as `(parent_reg, callee_reg)` pairs.
///
/// Layout convention (mirrors the existing portal-input pre-bind in
/// `lower_dispatch_body` at `:7440-7457`):
/// - per-bank packed regs starting at 0 (first Int → int_reg 0, second
///   Int → int_reg 1, …; same for Ref and Float independently);
/// - flat `next_reg` counter advanced past the highest per-bank slot
///   so subsequent `alloc_reg()` calls inside the body lowering do not
///   collide with the pre-bound caller-locals.
///
/// `pyopcode.py:179` and `jtransform.py:480` parity: PyPy's dispatch
/// inline_call passes `(opcode, oparg, ...)` as call args; the callee
/// jitcode receives them via portal-input binding slots indexed
/// per-bank.  Pyre's sub-frame uses the same `int_regs[]` / `ref_regs[]`
/// arrays per kind, so per-bank packing is the orthodox layout.
/// Layout-side helper extracted from
/// [`try_generate_jitcode_body_parts_with_caller_bindings`] so the
/// per-bank packing rule is testable without needing a body that
/// lowers cleanly.  Returns the layout descriptors plus the
/// worst-case `next_reg` advance that the caller must apply so
/// subsequent `alloc_reg()` cannot collide.
#[allow(private_interfaces)]
pub(crate) fn assign_caller_local_layout(
    caller_locals: &[(String, Binding)],
) -> (Vec<CallerLocalLayout>, u16) {
    let mut next_int = 0u16;
    let mut next_ref = 0u16;
    let mut next_float = 0u16;
    let mut layout = Vec::with_capacity(caller_locals.len());
    for (name, parent_binding) in caller_locals {
        let callee_reg = match parent_binding.kind {
            BindingKind::Int => {
                let r = next_int;
                next_int = next_int.saturating_add(1);
                r
            }
            BindingKind::Ref => {
                let r = next_ref;
                next_ref = next_ref.saturating_add(1);
                r
            }
            BindingKind::Float => {
                let r = next_float;
                next_float = next_float.saturating_add(1);
                r
            }
        };
        layout.push(CallerLocalLayout {
            name: name.clone(),
            parent_reg: parent_binding.reg,
            callee_reg,
            kind: parent_binding.kind,
        });
    }
    let max_pre_bound = next_int.max(next_ref).max(next_float);
    (layout, max_pre_bound)
}

/// Floor for a split sub-JitCode body's flat `next_reg`: the max of the int
/// and ref identity-slot ends, so body-side `alloc_reg()` never reuses a
/// reserved identity slot. Inert (0) when `split_dispatch` is off or the
/// kernel has no identity slots.
fn split_identity_floor(config: Option<&LowererConfig>) -> u16 {
    config
        .filter(|c| c.split_dispatch)
        .map(|c| {
            let (int_end, ref_end) = c.split_identity_reg_ends();
            int_end.max(ref_end)
        })
        .unwrap_or(0)
}

/// Compile-time guard for `split_dispatch`: no caller-local may be pre-bound
/// into its bank's reserved identity range `[identity_base, identity_end)`.
/// Those registers hold the virtualizable identity (array base/len, state
/// scalars) that the arm body's `load/store_state_field` ops address and the
/// resume path re-derives at deopt; a caller-local packed onto one would share
/// the physical register and be read as the identity slot (or vice-versa) — a
/// silent miscompile that the `frame.rs` trim's `is_none()` gate cannot catch,
/// since the arg writes the slot `Some`. Caller-locals pack densely from reg 0,
/// so the first int local lands at reg 0 (below `int_identity_base` = 1) and is
/// safe; this only fires if an arm captures a SECOND identity-range local.
/// Inert unless `split_dispatch` is on.
fn assert_no_split_identity_alias(layout: &[CallerLocalLayout], config: Option<&LowererConfig>) {
    let Some(config) = config.filter(|c| c.split_dispatch) else {
        return;
    };
    let (int_end, ref_end) = config.split_identity_reg_ends();
    for entry in layout {
        let (base, end, bank) = match entry.kind {
            BindingKind::Int => (config.int_identity_base(), int_end, "int"),
            BindingKind::Ref => (config.ref_identity_base(), ref_end, "ref"),
            BindingKind::Float => continue,
        };
        assert!(
            !(base <= entry.callee_reg && entry.callee_reg < end),
            "split_dispatch: caller-local `{}` binds {bank}-reg {} inside the \
             reserved identity range [{base}, {end}). A split arm may capture at \
             most {base} {bank} caller-local(s); offset the arg packing past the \
             identity range or reduce the arm's captured locals.",
            entry.name,
            entry.callee_reg,
        );
    }
}

#[allow(private_interfaces)]
pub(crate) fn try_generate_jitcode_body_parts_with_caller_bindings(
    body: &Expr,
    config: Option<&LowererConfig>,
    caller_locals: &[(String, Binding)],
) -> Option<(GeneratedJitCodeBody, Vec<CallerLocalLayout>)> {
    let stmts = extract_stmts(body);
    if stmts.is_empty() {
        return None;
    }

    let mut lowerer = Lowerer::new(config);
    // Sole dispatch-arm-body lowerer entry — `lower_stmt`'s `Stmt::Macro`
    // recognition for `can_enter_jit!()` emits
    // `__builder.loop_header(__jdindex);` which references the
    // `__jdindex: i64` parameter of the enclosing `__dispatch_jitcode_<fn>`.
    // The remaining public `try_generate_jitcode_body*` helpers are legacy
    // test/API shims and do not set this flag, so they still reject
    // `can_enter_jit!()` bodies instead of emitting a loop header without
    // a driver index.
    lowerer.in_dispatch_arm_body = true;

    let (layout, max_pre_bound) = assign_caller_local_layout(caller_locals);
    assert_no_split_identity_alias(&layout, config);
    for entry in &layout {
        lowerer.bindings.insert(
            entry.name.clone(),
            Binding {
                reg: entry.callee_reg,
                kind: entry.kind,
                depends_on_stack: false,
                struct_type: None,
            },
        );
    }
    // Advance the flat `next_reg` past the worst-case per-bank max so
    // body-side `alloc_reg()` cannot reuse any pre-bound slot in any
    // bank.  Mirrors the `next_reg.max(1)` advance after the
    // pc=Int(0)+program=Ref(0) pre-bind in `lower_dispatch_body`.
    lowerer.next_reg = lowerer
        .next_reg
        .max(max_pre_bound)
        .max(split_identity_floor(config));

    for stmt in &stmts {
        lowerer.lower_stmt(stmt)?;
    }

    annotate_live_markers_with_liveness(&mut lowerer.op_metadata);
    remove_repeated_live(&mut lowerer.op_metadata, &mut lowerer.statements);
    rewrite_live_marker_statements_with_triples(&lowerer.op_metadata, &mut lowerer.statements);
    maybe_dump_liveness("jitcode_body_with_caller_bindings", &lowerer.op_metadata);
    let liveness_prebuild =
        liveness_prebuild_tokens(&lowerer.op_metadata, &lowerer.inline_liveness_prebuild);
    let statements = lowerer.statements;
    Some((
        GeneratedJitCodeBody {
            body: quote! {
                #(#statements)*
            },
            liveness_prebuild,
            green_schema: Vec::new(),
            red_schema: Vec::new(),
        },
        layout,
    ))
}

/// pc-returning variant of
/// [`try_generate_jitcode_body_parts_with_caller_bindings`] for `split_dispatch`
/// pure forward-advancing arms.  The body's straight-line `work` statements are
/// lowered as usual; the trailing `pc += increment` is NOT lowered (on the
/// non-pinned sub-JitCode path it is inert and dropped — the dispatch loop owns
/// the pc register).  Instead, an explicit `BC_INT_RETURN(pc + increment)` is
/// emitted so the paired `inline_call_<types>_i` writes the advanced pc back
/// into the caller's green pc register (`next_instr = self.OPCODE(...)`).
#[allow(private_interfaces)]
pub(crate) fn try_generate_jitcode_pc_return_body_with_caller_bindings(
    body: &Expr,
    config: Option<&LowererConfig>,
    caller_locals: &[(String, Binding)],
    increment: i64,
) -> Option<(GeneratedJitCodeBody, Vec<CallerLocalLayout>)> {
    let stmts = extract_stmts(body);
    // The predicate (`arm_is_pure_pc_advance`) guarantees a trailing `pc += N`,
    // which is replaced by the explicit pc-return below.  Lower only the work.
    let (_pc_advance, work) = stmts.split_last()?;

    let mut lowerer = Lowerer::new(config);
    lowerer.in_dispatch_arm_body = true;

    let (layout, max_pre_bound) = assign_caller_local_layout(caller_locals);
    assert_no_split_identity_alias(&layout, config);
    for entry in &layout {
        lowerer.bindings.insert(
            entry.name.clone(),
            Binding {
                reg: entry.callee_reg,
                kind: entry.kind,
                depends_on_stack: false,
                struct_type: None,
            },
        );
    }
    lowerer.next_reg = lowerer
        .next_reg
        .max(max_pre_bound)
        .max(split_identity_floor(config));

    for stmt in work {
        lowerer.lower_stmt(stmt)?;
    }

    // `pc` is collected as a caller-local (the trailing `pc += N` references it),
    // so it is pre-bound at its callee reg; the work statements only read it, so
    // the binding still holds the incoming pc.  Return pc + increment.
    let pc_reg = lowerer.bindings.get("pc")?.reg;
    let tmp_reg = lowerer.alloc_reg();
    lowerer.emit_op(
        OpMeta::linear(OpKind::LoadConstI, vec![], vec![Register::int(tmp_reg)]),
        quote! {
            __builder.load_const_i_value(#tmp_reg as u16, #increment as i64);
        },
    );
    let ret_reg = lowerer.alloc_reg();
    lowerer.emit_op(
        OpMeta::linear(
            OpKind::BinopI,
            vec![Register::int(pc_reg), Register::int(tmp_reg)],
            vec![Register::int(ret_reg)],
        ),
        quote! {
            __builder.record_binop_i(
                #ret_reg as u16,
                majit_ir::OpCode::IntAdd,
                #pc_reg as u16,
                #tmp_reg as u16,
            );
        },
    );
    lowerer.emit_op(
        OpMeta::terminal(vec![Register::int(ret_reg)]),
        quote! { __builder.int_return(#ret_reg as u16); },
    );

    annotate_live_markers_with_liveness(&mut lowerer.op_metadata);
    remove_repeated_live(&mut lowerer.op_metadata, &mut lowerer.statements);
    rewrite_live_marker_statements_with_triples(&lowerer.op_metadata, &mut lowerer.statements);
    maybe_dump_liveness("jitcode_pc_return_body", &lowerer.op_metadata);
    let liveness_prebuild =
        liveness_prebuild_tokens(&lowerer.op_metadata, &lowerer.inline_liveness_prebuild);
    let statements = lowerer.statements;
    Some((
        GeneratedJitCodeBody {
            body: quote! {
                #(#statements)*
            },
            liveness_prebuild,
            green_schema: Vec::new(),
            red_schema: Vec::new(),
        },
        layout,
    ))
}

pub(crate) fn generate_inline_helper_jitcode_with_calls(
    func: &ItemFn,
    calls: &[crate::jit_interp::CallEntry],
) -> syn::Result<Option<InlineHelperJitCode>> {
    if !func.sig.generics.params.is_empty() {
        return Err(syn::Error::new_spanned(
            &func.sig.generics,
            "#[jit_inline] does not support generic helper functions yet",
        ));
    }

    let ReturnType::Type(_, return_ty) = &func.sig.output else {
        return Err(syn::Error::new_spanned(
            &func.sig.output,
            "#[jit_inline] requires a return type",
        ));
    };
    let return_kind = classify_param_type(return_ty).ok_or_else(|| {
        syn::Error::new_spanned(
            return_ty,
            "#[jit_inline] supports i64/isize (Int), usize/pointer (Ref), or f64 (Float) return types",
        )
    })?;

    let call_policies = calls
        .iter()
        .map(|entry| {
            let spec = match entry.policy {
                Some(kind) => CallPolicySpec::Explicit(kind),
                None => CallPolicySpec::Infer,
            };
            (canonical_path_segments(&entry.path), spec)
        })
        .collect();
    let mut lowerer =
        Lowerer::new_with_call_policies(None, call_policies, InferenceFailureMode::Panic);
    let param_layout = inline_helper_param_layout(func)?;
    let mut max_reg = 0u16;
    for (arg, (param_kind, reg)) in func.sig.inputs.iter().zip(param_layout.into_iter()) {
        let FnArg::Typed(pat_type) = arg else {
            return Err(syn::Error::new_spanned(
                arg,
                "#[jit_inline] does not support methods or self receivers",
            ));
        };
        let Pat::Ident(pat_ident) = &*pat_type.pat else {
            return Err(syn::Error::new_spanned(
                &pat_type.pat,
                "#[jit_inline] parameters must be simple identifiers",
            ));
        };
        let binding_kind = match param_kind {
            InlineReturnKind::Int => BindingKind::Int,
            InlineReturnKind::Ref => BindingKind::Ref,
            InlineReturnKind::Float => BindingKind::Float,
        };
        max_reg = max_reg.max(reg.saturating_add(1));
        lowerer.bindings.insert(
            pat_ident.ident.to_string(),
            Binding {
                reg,
                kind: binding_kind,
                depends_on_stack: false,
                struct_type: None,
            },
        );
    }
    lowerer.next_reg = max_reg;

    let Some(binding) = lowerer.lower_block_value(&func.block) else {
        return Ok(None);
    };

    let helper_name = func.sig.ident.to_string();
    annotate_live_markers_with_liveness(&mut lowerer.op_metadata);
    remove_repeated_live(&mut lowerer.op_metadata, &mut lowerer.statements);
    rewrite_live_marker_statements_with_triples(&lowerer.op_metadata, &mut lowerer.statements);
    maybe_dump_liveness(&helper_name, &lowerer.op_metadata);
    let liveness_prebuild =
        liveness_prebuild_tokens(&lowerer.op_metadata, &lowerer.inline_liveness_prebuild);
    let statements = lowerer.statements;
    Ok(Some(InlineHelperJitCode {
        body: quote! {
            #(#statements)*
        },
        return_reg: binding.reg,
        return_kind,
        liveness_prebuild,
    }))
}

pub(crate) fn inline_helper_param_layout(
    func: &ItemFn,
) -> syn::Result<Vec<(InlineReturnKind, u16)>> {
    let mut next_i = 0u16;
    let mut next_r = 0u16;
    let mut next_f = 0u16;
    let mut layout = Vec::with_capacity(func.sig.inputs.len());
    for arg in &func.sig.inputs {
        let FnArg::Typed(pat_type) = arg else {
            return Err(syn::Error::new_spanned(
                arg,
                "#[jit_inline] does not support methods or self receivers",
            ));
        };
        let param_kind = classify_param_type(&pat_type.ty).ok_or_else(|| {
            syn::Error::new_spanned(
                &pat_type.ty,
                "#[jit_inline] parameters must use i64/isize (Int), usize/pointer (Ref), or f64 (Float)",
            )
        })?;
        let reg = match param_kind {
            InlineReturnKind::Int => {
                let reg = next_i;
                next_i = next_i.saturating_add(1);
                reg
            }
            InlineReturnKind::Ref => {
                let reg = next_r;
                next_r = next_r.saturating_add(1);
                reg
            }
            InlineReturnKind::Float => {
                let reg = next_f;
                next_f = next_f.saturating_add(1);
                reg
            }
        };
        layout.push((param_kind, reg));
    }
    Ok(layout)
}

pub(crate) fn inline_helper_param_counts(func: &ItemFn) -> syn::Result<(u16, u16, u16)> {
    let layout = inline_helper_param_layout(func)?;
    let mut count_i = 0u16;
    let mut count_r = 0u16;
    let mut count_f = 0u16;
    for (kind, _) in layout {
        match kind {
            InlineReturnKind::Int => count_i = count_i.saturating_add(1),
            InlineReturnKind::Ref => count_r = count_r.saturating_add(1),
            InlineReturnKind::Float => count_f = count_f.saturating_add(1),
        }
    }
    Ok((count_i, count_r, count_f))
}

fn try_generate_jitcode_body_inner(
    body: &Expr,
    config: Option<&LowererConfig>,
) -> Option<GeneratedJitCodeBody> {
    let stmts = extract_stmts(body);
    if stmts.is_empty() {
        return None;
    }

    let mut lowerer = Lowerer::new(config);
    for stmt in &stmts {
        lowerer.lower_stmt(stmt)?;
    }

    // RPython `compute_liveness(ssarepr) → remove_repeated_live(ssarepr)
    // → assemble()` (codewriter.py call order). `annotate_live_markers_
    // with_liveness` materialises the per-marker fixed-point alive set
    // onto each `LiveMarker.reads` so the repeated-live pass and the
    // emit-time triple rewrite both consume the same ssarepr-mutated
    // shape `liveness.py:33-79` produces.
    annotate_live_markers_with_liveness(&mut lowerer.op_metadata);
    remove_repeated_live(&mut lowerer.op_metadata, &mut lowerer.statements);
    rewrite_live_marker_statements_with_triples(&lowerer.op_metadata, &mut lowerer.statements);
    maybe_dump_liveness("jitcode_body", &lowerer.op_metadata);
    let liveness_prebuild =
        liveness_prebuild_tokens(&lowerer.op_metadata, &lowerer.inline_liveness_prebuild);
    let statements = lowerer.statements;
    Some(GeneratedJitCodeBody {
        body: quote! {
            #(#statements)*
        },
        liveness_prebuild,
        green_schema: Vec::new(),
        red_schema: Vec::new(),
    })
}

/// A.3.6.1 (jtransform.py:1693-1714): bind body-local `let` stmts that
/// appear in the dispatch while-body BEFORE the `jit_merge_point!()`
/// macro stmt, so that consumer-declared
/// `#[jit_interp(greens = [<body-local>])]` (e.g. aheui-jit's
/// `greens = [stackok]` with `let stackok = program.get_req_size(pc) <= ...`)
/// flow through `resolve_greens` / `emit_promote_greens` without panic.
///
/// TODO: RPython has no equivalent two-pass walker.
/// Its annotator-driven flowgraph SpaceOperation-lowers every stmt before
/// `jtransform.handle_jit_marker__jit_merge_point` fires, so `Variable`
/// records exist for body-locals at merge-point rewrite time. Pyre's
/// proc-macro lowerer walks `syn::Block` AST directly with limited
/// expression-lowering coverage, requiring this explicit pre-pass.
///
/// Returns `Some(())` once the `jit_merge_point!()` macro stmt is reached
/// (or the while-body ends without one). Eligibility for binding is
/// delegated to `Lowerer::lower_local`, which in turn delegates to
/// `lower_value_expr`; non-`let` stmts and `let`s with unsupported RHS
/// shapes are skipped silently here (the existing `lower_pre_dispatch_stmts`
/// post-merge-point walker / dispatch-body emit is responsible for
/// diagnostics on its own pass).
pub(super) fn bind_pre_merge_point_stmts(
    lowerer: &mut Lowerer,
    func_block: &syn::Block,
) -> Option<()> {
    let dispatch_match = find_dispatch_match(func_block)?;
    let loop_body = find_dispatch_loop_body(func_block, dispatch_match)?;
    for stmt in &loop_body.stmts {
        if is_jit_merge_point_macro(stmt) {
            break;
        }
        if let syn::Stmt::Local(local) = stmt {
            // lower_local delegates to lower_value_expr; failure here is
            // intentionally silent — emit_promote_greens will produce the
            // diagnostic if the green's binding is still missing.
            let _ = lowerer.lower_local(local);
        }
    }
    Some(())
}
